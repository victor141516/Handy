use crate::audio_toolkit::{
    apply_custom_words, detect_output_language, normalize_transcription_output,
    remove_filler_words, OutputLanguageEvidence,
};
use crate::chinese_script::{convert_chinese_script, ChineseVariety};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::model::{EngineType, ModelManager};
use crate::memory_diagnostics::log_memory_snapshot;
use crate::settings::{
    get_settings, AppSettings, ChineseScript, ModelUnloadTimeout, OrtAcceleratorSetting,
    TranscribeAcceleratorSetting,
};
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use tauri::{AppHandle, Emitter, Manager};
use tauri_specta::Event;
use transcribe_cpp::{
    Backend, Feature, Model, ModelOptions, RunExtension, RunOptions, Session, StreamOptions, Task,
    TimestampKind, WhisperRunOptions,
};
use transcribe_rs::{
    onnx::{
        canary::CanaryModel,
        cohere::CohereModel,
        gigaam::GigaAMModel,
        moonshine::{MoonshineModel, MoonshineVariant, StreamingModel},
        parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity},
        sense_voice::{SenseVoiceModel, SenseVoiceParams},
        Quantization,
    },
    SpeechModel, TranscribeOptions,
};

const STREAM_PERF_LOG_INTERVAL: Duration = Duration::from_secs(5);
const STREAM_MEMORY_LOG_INTERVAL: Duration = Duration::from_secs(60);
const STREAM_FINALIZE_REPLY_TIMEOUT: Duration = Duration::from_secs(30);
const STREAM_QUEUE_CAPACITY_FRAMES: usize = 512;
const LONG_AUDIO_BATCH_THRESHOLD_SECONDS: usize = 120;
const LONG_AUDIO_CHUNK_SECONDS: usize = 30;
const TRANSCRIPTION_SAMPLE_RATE_HZ: usize = 16_000;
const LONG_AUDIO_BATCH_THRESHOLD_SAMPLES: usize =
    TRANSCRIPTION_SAMPLE_RATE_HZ * LONG_AUDIO_BATCH_THRESHOLD_SECONDS;
const LONG_AUDIO_CHUNK_SAMPLES: NonZeroUsize =
    match NonZeroUsize::new(TRANSCRIPTION_SAMPLE_RATE_HZ * LONG_AUDIO_CHUNK_SECONDS) {
        Some(samples) => samples,
        None => panic!("long audio chunk duration and sample rate must be non-zero"),
    };

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

/// Live transcription snapshot emitted to the overlay during a streaming run.
/// `committed` is the append-only, flicker-free prefix; `tentative` is the
/// volatile suffix the model may still rewrite.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct StreamTextEvent {
    pub committed: String,
    pub tentative: String,
}

/// Phase of the streaming overlay card, emitted to drive its UI state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum StreamPhase {
    /// Receiving audio / live text (or waiting for the stream to begin). Rust
    /// does not emit this today; the frontend starts in this phase and Rust only
    /// emits transitions away from it.
    Listening,
    /// Finalizing or post-processing — show a spinner.
    Working,
}

/// Semantic kind of "working" phase, used to localize the spinner label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum StreamWorkKind {
    Transcribing,
    Polishing,
}

/// Emitted to switch the streaming overlay to a working spinner.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct StreamPhaseEvent {
    pub phase: StreamPhase,
    /// Present only when `phase` is `Working`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<StreamWorkKind>,
}

#[derive(Clone, Debug)]
pub struct TranscriptionSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct TranscriptionOutput {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
}

impl TranscriptionOutput {
    pub fn text_only(text: String, audio_len: usize) -> Self {
        let trimmed = text.trim();
        let segments = if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![TranscriptionSegment {
                start_ms: 0,
                end_ms: audio_len_to_ms(audio_len),
                text: text.clone(),
            }]
        };

        Self { text, segments }
    }
}

#[derive(Clone, Copy)]
enum ModelRetention {
    ApplyUnloadPolicy,
    Retain,
}

#[derive(Clone, Copy)]
struct ChunkPosition {
    index: usize,
    count: usize,
}

/// Commands sent to the streaming worker thread. Audio frames and the finalize
/// request travel the same channel so FIFO ordering guarantees every fed frame
/// is processed before finalize runs.
enum StreamCmd {
    Feed(Vec<f32>),
    /// Flush the stream and reply with the final text, or `None` if no stream
    /// was ever active (caller should fall back to batch transcription).
    Finalize(mpsc::Sender<Option<FinalizedStreamText>>),
    Cancel,
}

struct FinalizedStreamText {
    output: TranscriptionOutput,
    output_language: OutputLanguageEvidence,
    /// The streaming model's supported languages, for text-based detection.
    supported_languages: Vec<String>,
}

/// Routes real-time audio frames to the active streaming worker. Shared between
/// the [`TranscriptionManager`] (opens/closes the route) and the audio recorder's
/// per-frame callback (feeds frames). The recorder holds an `Arc<StreamRouter>`
/// directly, so a frame with no stream pending costs a single relaxed atomic
/// load — no Tauri state lookup, no mutex lock.
pub struct StreamRouter {
    /// Command channel to the active streaming worker, present from
    /// `start_stream` until `finalize_stream`/`cancel_stream`.
    tx: Mutex<Option<mpsc::SyncSender<StreamCmd>>>,
    /// True while a stream is pending or active (channel is open). The audio
    /// callback checks this first to avoid the mutex lock when no stream runs.
    open: Arc<AtomicBool>,
    queued_frames: AtomicU64,
    queue_high_water: AtomicU64,
    dropped_frames: AtomicU64,
}

impl StreamRouter {
    fn new() -> Self {
        Self {
            tx: Mutex::new(None),
            open: Arc::new(AtomicBool::new(false)),
            queued_frames: AtomicU64::new(0),
            queue_high_water: AtomicU64::new(0),
            dropped_frames: AtomicU64::new(0),
        }
    }

    /// Open a fresh command channel for a new streaming session, returning the
    /// receiver the worker should drain. Caller must ensure no prior channel is
    /// still open.
    fn open(&self) -> mpsc::Receiver<StreamCmd> {
        let (tx, rx) = mpsc::sync_channel::<StreamCmd>(STREAM_QUEUE_CAPACITY_FRAMES);
        *self.tx.lock().unwrap() = Some(tx);
        self.queued_frames.store(0, Ordering::Release);
        self.queue_high_water.store(0, Ordering::Release);
        self.dropped_frames.store(0, Ordering::Release);
        self.open.store(true, Ordering::Relaxed);
        rx
    }

    /// Take the sender out (closing the channel to new feeds). Returns the
    /// sender so the caller can send the final `Finalize`/`Cancel` command.
    fn take(&self) -> Option<mpsc::SyncSender<StreamCmd>> {
        self.open.store(false, Ordering::Relaxed);
        self.tx.lock().unwrap().take()
    }

    /// Drop the channel and mark closed without sending a final command (used
    /// when the worker exits without a finalize/cancel handshake).
    fn clear(&self) {
        self.open.store(false, Ordering::Relaxed);
        *self.tx.lock().unwrap() = None;
    }

    /// Forward a 16 kHz frame to the active streaming worker. Cheap no-op (a
    /// single relaxed atomic load) when no stream is pending.
    pub fn feed(&self, frame: &[f32]) {
        if !self.open.load(Ordering::Relaxed) {
            return;
        }
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let queued = self.queued_frames.fetch_add(1, Ordering::AcqRel) + 1;
            match tx.try_send(StreamCmd::Feed(frame.to_vec())) {
                Ok(()) => {
                    self.queue_high_water.fetch_max(queued, Ordering::AcqRel);
                }
                Err(mpsc::TrySendError::Full(_)) => {
                    self.queued_frames.fetch_sub(1, Ordering::AcqRel);
                    let dropped = self.dropped_frames.fetch_add(1, Ordering::AcqRel) + 1;
                    if dropped == 1 {
                        warn!(
                            "Live preview queue reached its {}-frame limit; dropping preview frames and marking the stream incomplete",
                            STREAM_QUEUE_CAPACITY_FRAMES
                        );
                        log_memory_snapshot("live transcription queue overflow");
                    }
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    self.queued_frames.fetch_sub(1, Ordering::AcqRel);
                    self.open.store(false, Ordering::Release);
                }
            }
        }
    }

    fn record_feed_dequeued(&self) {
        let _ = self
            .queued_frames
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                Some(queued.saturating_sub(1))
            });
    }

    fn was_incomplete(&self) -> bool {
        self.dropped_frames.load(Ordering::Acquire) > 0
    }

    fn log_queue_summary(&self) {
        info!(
            "Live preview queue summary: capacity_frames={}, queued_frames={}, high_water_frames={}, dropped_frames={}",
            STREAM_QUEUE_CAPACITY_FRAMES,
            self.queued_frames.load(Ordering::Acquire),
            self.queue_high_water.load(Ordering::Acquire),
            self.dropped_frames.load(Ordering::Acquire),
        );
    }

    /// Whether a stream is pending or active.
    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Relaxed)
    }
}

enum LoadedEngine {
    /// Whisper-family models (whisper, breeze-asr, custom .bin/.gguf) via
    /// transcribe-cpp. Holds the live `Session`, which keeps its `Model` alive
    /// internally, so repeated dictation reuses the session without reloading.
    TranscribeCpp(Session),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
    MoonshineStreaming(StreamingModel),
    SenseVoice(SenseVoiceModel),
    GigaAM(GigaAMModel),
    Canary(CanaryModel),
    Cohere(CohereModel),
}

/// RAII guard that clears the `is_loading` flag and notifies waiters on drop.
/// Ensures the loading flag is always reset, even on early returns or panics.
pub struct LoadingGuard {
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        // Recover from a poisoned mutex instead of panicking —
        // a panic inside Drop calls abort().
        let mut is_loading = match self.is_loading.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Recovered poisoned is_loading mutex during LoadingGuard drop — a panic occurred earlier this session");
                e.into_inner()
            }
        };
        *is_loading = false;
        self.loading_condvar.notify_all();
    }
}

/// RAII guard that clears the streaming worker/lease flags on any worker exit -
/// normal return, early return, or a panic in an engine call that unwinds the
/// detached worker thread. Tokens prevent an older worker from clearing a newer
/// worker's state if a start/finalize race ever slips through.
struct StreamWorkerGuard {
    worker_id: u64,
    active_stream_worker: Arc<AtomicU64>,
    active_engine_lease: Arc<AtomicU64>,
    stream_active: Arc<AtomicBool>,
}

impl Drop for StreamWorkerGuard {
    fn drop(&mut self) {
        if self.active_stream_worker.load(Ordering::Acquire) == self.worker_id {
            self.stream_active.store(false, Ordering::Release);
        }
        let _ = self.active_engine_lease.compare_exchange(
            self.worker_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = self.active_stream_worker.compare_exchange(
            self.worker_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

#[derive(Clone)]
pub struct TranscriptionManager {
    engine: Arc<Mutex<Option<LoadedEngine>>>,
    model_manager: Arc<ModelManager>,
    app_handle: AppHandle,
    current_model_id: Arc<Mutex<Option<String>>>,
    last_activity: Arc<AtomicU64>,
    shutdown_signal: Arc<AtomicBool>,
    watcher_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
    reload_model_on_next_use: Arc<AtomicBool>,
    /// Routes real-time audio frames to the active streaming worker; see
    /// [`StreamRouter`]. Shared with the audio recorder so per-frame feeds skip
    /// Tauri state and the manager lock.
    router: Arc<StreamRouter>,
    /// True only while a transcribe-cpp `Stream` is actually in flight (set by
    /// the worker once `stream()` succeeds). Used for overlay/UI decisions.
    stream_active: Arc<AtomicBool>,
    /// Streaming uses four independent flags: router open = frames should route,
    /// worker active = no second worker may start, engine lease = engine is out
    /// of the mutex, stream active = UI should show a live session.
    ///
    /// Monotonic id source for stream workers; zero means "no worker".
    next_stream_worker_id: Arc<AtomicU64>,
    /// Nonzero while a stream worker exists, even if it has not leased the engine
    /// yet. This prevents a second worker from starting after finalize/cancel
    /// closes the router but before the first worker has fully exited.
    active_stream_worker: Arc<AtomicU64>,
    /// Nonzero while the streaming worker has taken the engine out of `engine`.
    /// `is_model_loaded()` consults this so the model still reports "loaded"
    /// while the worker holds it.
    active_engine_lease: Arc<AtomicU64>,
}

impl TranscriptionManager {
    pub fn new(app_handle: &AppHandle, model_manager: Arc<ModelManager>) -> Result<Self> {
        let manager = Self {
            engine: Arc::new(Mutex::new(None)),
            model_manager,
            app_handle: app_handle.clone(),
            current_model_id: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(AtomicU64::new(Self::now_ms())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            watcher_handle: Arc::new(Mutex::new(None)),
            is_loading: Arc::new(Mutex::new(false)),
            loading_condvar: Arc::new(Condvar::new()),
            reload_model_on_next_use: Arc::new(AtomicBool::new(false)),
            router: Arc::new(StreamRouter::new()),
            stream_active: Arc::new(AtomicBool::new(false)),
            next_stream_worker_id: Arc::new(AtomicU64::new(1)),
            active_stream_worker: Arc::new(AtomicU64::new(0)),
            active_engine_lease: Arc::new(AtomicU64::new(0)),
        };

        // Start the idle watcher
        {
            let app_handle_cloned = app_handle.clone();
            let manager_cloned = manager.clone();
            let shutdown_signal = manager.shutdown_signal.clone();
            let handle = thread::spawn(move || {
                debug!("Idle watcher thread started");
                while !shutdown_signal.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(10)); // Check every 10 seconds

                    // Check shutdown signal again after sleep
                    if shutdown_signal.load(Ordering::Relaxed) {
                        break;
                    }

                    let settings = get_settings(&app_handle_cloned);
                    let timeout = settings.model_unload_timeout;

                    // Skip Immediately — that variant is handled by
                    // maybe_unload_immediately() after each transcription.
                    // Treating it as 0s here would unload the model mid-recording.
                    if timeout == ModelUnloadTimeout::Immediately {
                        continue;
                    }

                    // While recording, keep the idle timer fresh so the
                    // model is never unloaded mid-session.
                    let is_recording = app_handle_cloned
                        .try_state::<Arc<AudioRecordingManager>>()
                        .is_some_and(|a| a.is_recording());
                    if is_recording {
                        manager_cloned.touch_activity();
                        continue;
                    }

                    if let Some(limit_seconds) = timeout.to_seconds() {
                        let last = manager_cloned.last_activity.load(Ordering::Relaxed);
                        let now_ms = TranscriptionManager::now_ms();
                        let idle_ms = now_ms.saturating_sub(last);
                        let limit_ms = limit_seconds * 1000;

                        if idle_ms > limit_ms {
                            // idle -> unload
                            if manager_cloned.is_model_loaded() {
                                let unload_start = std::time::Instant::now();
                                info!(
                                    "Model idle for {}s (limit: {}s), unloading",
                                    idle_ms / 1000,
                                    limit_seconds
                                );
                                match manager_cloned.unload_model() {
                                    Ok(()) => {
                                        let unload_duration = unload_start.elapsed();
                                        info!(
                                            "Model unloaded due to inactivity (took {}ms)",
                                            unload_duration.as_millis()
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to unload idle model: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                debug!("Idle watcher thread shutting down gracefully");
            });
            *manager.watcher_handle.lock().unwrap() = Some(handle);
        }

        Ok(manager)
    }

    /// Lock the engine mutex, recovering from poison if a previous transcription panicked.
    fn lock_engine(&self) -> MutexGuard<'_, Option<LoadedEngine>> {
        self.engine.lock().unwrap_or_else(|poisoned| {
            warn!("Engine mutex was poisoned by a previous panic, recovering");
            poisoned.into_inner()
        })
    }

    pub fn is_model_loaded(&self) -> bool {
        // The engine may be leased out to the streaming worker (taken out of
        // the mutex). It's still loaded, just in use, so report true.
        self.lock_engine().is_some() || self.active_engine_lease.load(Ordering::Acquire) != 0
    }

    /// Accelerator changes should not disturb the current transcription. Mark
    /// the cached engine stale; the next model-use path reloads it with the
    /// latest settings.
    pub fn reload_model_on_next_use(&self) {
        self.reload_model_on_next_use.store(true, Ordering::Release);
    }

    /// Atomically check whether a model load is in progress and, if not, mark
    /// one as starting. Returns a [`LoadingGuard`] whose [`Drop`] impl will
    /// clear the flag and wake waiters. Returns `None` if a load is already in
    /// progress.
    pub fn try_start_loading(&self) -> Option<LoadingGuard> {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return None;
        }
        *is_loading = true;
        Some(LoadingGuard {
            is_loading: self.is_loading.clone(),
            loading_condvar: self.loading_condvar.clone(),
        })
    }

    pub fn unload_model(&self) -> Result<()> {
        let unload_start = std::time::Instant::now();
        debug!("Starting to unload model");

        {
            let mut engine = self.lock_engine();
            // Dropping the engine frees all resources
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Emit unloaded event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: None,
            },
        );

        let unload_duration = unload_start.elapsed();
        debug!(
            "Model unloaded manually (took {}ms)",
            unload_duration.as_millis()
        );
        Ok(())
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Reset the idle timer to now.
    fn touch_activity(&self) {
        self.last_activity.store(Self::now_ms(), Ordering::Relaxed);
    }

    /// Unloads the model immediately if the setting is enabled and the model is loaded
    pub fn maybe_unload_immediately(&self, context: &str) {
        let settings = get_settings(&self.app_handle);
        if settings.model_unload_timeout == ModelUnloadTimeout::Immediately
            && self.is_model_loaded()
        {
            info!("Immediately unloading model after {}", context);
            if let Err(e) = self.unload_model() {
                warn!("Failed to immediately unload model: {}", e);
            }
        }
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        self.load_model_with_device(model_id, None)
    }

    /// Like [`load_model`](Self::load_model), but lets a caller hard-select the
    /// compute device for this one load by its `transcribe_cpp::devices()`
    /// registry index (the index shown by `--list-devices`). `None` keeps the
    /// persisted accelerator setting (which may be Auto). Only affects
    /// transcribe-cpp (whisper-family) models; the selection is not persisted.
    pub fn load_model_with_device(
        &self,
        model_id: &str,
        device_index: Option<usize>,
    ) -> Result<()> {
        apply_accelerator_settings(&self.app_handle);

        let load_start = std::time::Instant::now();
        debug!("Starting to load model: {}", model_id);

        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = match self.model_manager.get_model_info(model_id) {
            Some(model_info) => model_info,
            None => {
                let error_msg = format!("Model not found: {}", model_id);
                let _ = self.app_handle.emit(
                    "model-state-changed",
                    ModelStateEvent {
                        event_type: "loading_failed".to_string(),
                        model_id: Some(model_id.to_string()),
                        model_name: None,
                        error: Some(error_msg.clone()),
                    },
                );
                return Err(anyhow::anyhow!(error_msg));
            }
        };

        // Every failure after loading starts must emit a terminal event so the
        // frontend can never remain in its loading state.
        let emit_loading_failed = |error_msg: &str| {
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
        };

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            emit_loading_failed(error_msg);
            return Err(anyhow::anyhow!(error_msg));
        }

        let model_path = self
            .model_manager
            .get_model_path(model_id)
            .inspect_err(|error| emit_loading_failed(&error.to_string()))?;

        // Drop the current engine BEFORE building the new one so transcribe-cpp
        // frees the previous native context first — avoids holding two models at
        // once (peak memory on large GGUFs). Clear the id too: if the new load
        // fails, status should read "no loaded model", not the dropped engine.
        {
            let mut engine = self.lock_engine();
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Create appropriate engine based on model type

        let loaded_engine = match model_info.engine_type {
            EngineType::TranscribeCpp => {
                // The whisper backend is chosen at load time (transcribe-cpp has
                // no runtime global). With an explicit `device_index` (the
                // --device-index flag) hard-select that registered device;
                // otherwise re-read the persisted accelerator preference (so an
                // accelerator change marked for reload takes effect here).
                let (backend, device) = match device_index {
                    Some(index) => resolve_device_index(index).inspect_err(|e| {
                        emit_loading_failed(&e.to_string());
                    })?,
                    None => {
                        let settings = get_settings(&self.app_handle);
                        let accelerator = settings.transcribe_accelerator;
                        let device = resolve_gpu_device(
                            accelerator,
                            settings.transcribe_gpu_device.as_deref(),
                        );
                        // Backend::Auto accepts an exact GPU device. Without a
                        // valid exact device, backend selection handles the
                        // retired generic GPU state and host CPU guard.
                        let backend = if device.is_some() {
                            Backend::Auto
                        } else {
                            select_transcribe_backend(accelerator)
                        };
                        (backend, device)
                    }
                };
                let requested_device = device
                    .as_ref()
                    .map(transcribe_device_label)
                    .unwrap_or_else(|| "automatic".to_string());
                let model_options = ModelOptions { backend, device };
                let model = Model::load_with(&model_path, &model_options).map_err(|e| {
                    let error_msg = format!("Failed to load whisper model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                // The bound backend may differ from the request (e.g. CPU
                // fallback under Auto); log what actually loaded.
                let bound_backend = model.backend();
                let session = model.session().map_err(|e| {
                    let error_msg = format!(
                        "Failed to create session for whisper model {}: {}",
                        model_id, e
                    );
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                // Reconcile the registry's advertised capabilities with the
                // loaded model's real ones (GGUF metadata) so badges/gating
                // reflect runtime truth, not the pre-download probe. The
                // load-completed event below triggers the frontend refresh.
                let caps = session.model().capabilities();
                self.model_manager.set_runtime_capabilities(
                    model_id,
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.supports_language_detect,
                    caps.languages.clone(),
                );
                let bound_device = model
                    .device()
                    .map(|device| transcribe_device_label(&device))
                    .unwrap_or_else(|_| "unknown".to_string());
                info!(
                    "Loaded whisper model '{}' (requested {:?}, requested device '{}', \
                     bound backend '{}', bound device '{}', supports_streaming={}, \
                     supports_translate={}, supports_language_detect={})",
                    model_id,
                    backend,
                    requested_device,
                    bound_backend,
                    bound_device,
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.supports_language_detect
                );
                LoadedEngine::TranscribeCpp(session)
            }
            EngineType::Parakeet => {
                let engine =
                    ParakeetModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load parakeet model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::Parakeet(engine)
            }
            EngineType::Moonshine => {
                let engine = MoonshineModel::load(
                    &model_path,
                    MoonshineVariant::Base,
                    &Quantization::default(),
                )
                .map_err(|e| {
                    let error_msg = format!("Failed to load moonshine model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Moonshine(engine)
            }
            EngineType::MoonshineStreaming => {
                let engine = StreamingModel::load(&model_path, 0, &Quantization::default())
                    .map_err(|e| {
                        let error_msg = format!(
                            "Failed to load moonshine streaming model {}: {}",
                            model_id, e
                        );
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::MoonshineStreaming(engine)
            }
            EngineType::SenseVoice => {
                let engine =
                    SenseVoiceModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load SenseVoice model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::SenseVoice(engine)
            }
            EngineType::GigaAM => {
                let engine = GigaAMModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load gigaam model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::GigaAM(engine)
            }
            EngineType::Canary => {
                let engine = CanaryModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load canary model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Canary(engine)
            }
            EngineType::Cohere => {
                let engine = CohereModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load cohere model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Cohere(engine)
            }
        };

        // Update the current engine and model ID
        {
            let mut engine = self.lock_engine();
            *engine = Some(loaded_engine);
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = Some(model_id.to_string());
        }

        // Reset idle timer so the watcher doesn't immediately unload a just-loaded model
        self.touch_activity();

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        let load_duration = load_start.elapsed();
        debug!(
            "Successfully loaded transcription model: {} (took {}ms)",
            model_id,
            load_duration.as_millis()
        );
        Ok(())
    }

    /// Kicks off the model loading in a background thread if it's not already loaded
    pub fn initiate_model_load(&self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return;
        }

        let reload_pending = self.reload_model_on_next_use.load(Ordering::Acquire);
        if !reload_pending && self.is_model_loaded() {
            return;
        }

        *is_loading = true;
        let self_clone = self.clone();
        thread::spawn(move || {
            if reload_pending {
                self_clone
                    .reload_model_on_next_use
                    .store(false, Ordering::Release);
            }
            let settings = get_settings(&self_clone.app_handle);
            if let Err(e) = self_clone.load_model(&settings.selected_model) {
                error!("Failed to load model: {}", e);
            }
            let mut is_loading = self_clone.is_loading.lock().unwrap();
            *is_loading = false;
            self_clone.loading_condvar.notify_all();
        });
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self.current_model_id.lock().unwrap();
        current_model.clone()
    }

    /// The compute backend the currently-loaded engine is bound to, for
    /// diagnostics (e.g. confirming `--device-index` actually bound a GPU rather
    /// than falling back to CPU/auto). transcribe-cpp (whisper-family) reports
    /// its real backend string; ONNX engines report "onnx"; `None` when no
    /// model is loaded.
    pub fn current_backend(&self) -> Option<String> {
        match self.lock_engine().as_ref() {
            Some(LoadedEngine::TranscribeCpp(session)) => {
                Some(session.model().backend().to_string())
            }
            Some(_) => Some("onnx".to_string()),
            None => None,
        }
    }

    /// Whether a live streaming run is currently in flight.
    pub fn is_streaming(&self) -> bool {
        self.stream_active.load(Ordering::Acquire)
    }

    /// Shared handle to the stream router, used by the audio recorder to feed
    /// real-time frames without going through Tauri state on every frame.
    pub fn stream_router(&self) -> Arc<StreamRouter> {
        Arc::clone(&self.router)
    }

    /// Begin a live streaming transcription on the held engine's session.
    /// Audio frames pushed via [`StreamRouter::feed`] (captured directly by the
    /// audio recorder) are decoded incrementally and emitted to the overlay as
    /// [`StreamTextEvent`].
    ///
    /// Non-blocking: spawns a worker that waits for any in-progress model load,
    /// verifies the model supports streaming, then begins the stream. If the
    /// model can't stream, the worker idles until finalize/cancel and reports
    /// `None` so the caller falls back to batch transcription. Frames sent
    /// before the stream begins queue on the channel and are not lost.
    pub fn start_stream(&self) {
        if self.router.is_open() || self.active_stream_worker.load(Ordering::Acquire) != 0 {
            warn!("start_stream called while a stream worker is already active");
            return;
        }
        let worker_id = self.next_stream_worker_id.fetch_add(1, Ordering::Relaxed);
        if self
            .active_stream_worker
            .compare_exchange(0, worker_id, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            warn!("start_stream lost a race with another stream worker");
            return;
        }
        let rx = self.router.open();
        self.stream_active.store(false, Ordering::Release);

        let manager = self.clone();
        thread::spawn(move || manager.run_stream_worker(rx, worker_id));
    }

    fn run_stream_worker(&self, rx: mpsc::Receiver<StreamCmd>, worker_id: u64) {
        let _worker = StreamWorkerGuard {
            worker_id,
            active_stream_worker: Arc::clone(&self.active_stream_worker),
            active_engine_lease: Arc::clone(&self.active_engine_lease),
            stream_active: Arc::clone(&self.stream_active),
        };

        // Wait for any in-progress model load to finish (start_stream races the
        // background load kicked off when recording starts).
        {
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }
        }

        let model_id = self.get_current_model().unwrap_or_default();

        // Take the engine out of the mutex so we own it during streaming,
        // structurally excluding any concurrent batch transcription (which
        // transcribe-cpp's compute_lock would refuse anyway). Returned when the
        // worker exits, or dropped if the model was switched/unloaded mid-stream.
        if self
            .active_engine_lease
            .compare_exchange(0, worker_id, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            warn!("Live preview: another worker already holds the transcription engine");
            self.router.clear();
            drain_until_finalize(rx);
            return;
        }
        let mut engine = match self.lock_engine().take() {
            Some(e) => e,
            None => {
                info!(
                    "Live preview: model '{}' was unloaded before streaming could begin; \
                     falling back to batch transcription",
                    model_id
                );
                let _ = self.active_engine_lease.compare_exchange(
                    worker_id,
                    0,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                self.router.clear();
                drain_until_finalize(rx);
                return;
            }
        };

        // Only transcribe-cpp models expose streaming; ONNX engines fall back to
        // batch. The loaded session (not the ModelManager copy) is the source of
        // truth for run-path capabilities.
        let (supports_streaming, supports_translate, languages) = match &engine {
            LoadedEngine::TranscribeCpp(session) => {
                let model = session.model();
                let caps = model.capabilities();
                info!(
                    "Live preview: model '{}' arch='{}' variant='{}' supports_streaming={} \
                     supports_translate={} languages={:?}",
                    model_id,
                    model.arch(),
                    model.variant(),
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.languages,
                );
                (
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.languages,
                )
            }
            _ => {
                info!(
                    "Live preview: model '{}' is not a transcribe-cpp model; \
                     streaming is unavailable, using batch transcription",
                    model_id
                );
                (false, false, Vec::new())
            }
        };

        if !supports_streaming {
            self.return_engine(engine, &model_id);
            self.router.clear();
            drain_until_finalize(rx);
            return;
        }

        // Build run options mirroring the offline transcribe-cpp path: task +
        // language gated against what the model actually advertises.
        let settings = get_settings(&self.app_handle);
        let effective_language =
            effective_language_for_model(&settings, self.model_manager.as_ref(), &model_id);
        let run_plan = transcribe_cpp_run_plan(
            settings.translate_to_english,
            &effective_language,
            &languages,
            supports_translate,
        );
        let output_language = resolve_output_language_evidence(
            &settings,
            run_plan.language.as_deref(),
            &languages,
            run_plan.target_language.as_deref() == Some("en"),
        );
        let run_options = RunOptions {
            task: run_plan.task,
            language: run_plan.language,
            target_language: run_plan.target_language,
            ..Default::default()
        };

        // Run the stream on the held session. The Stream borrows the session
        // (and thus the engine) for its lifetime, so the feed/finalize loop
        // lives in a labeled block — when it exits, the borrow is released and
        // the engine can be moved into return_engine().
        let mut preview_script = PreviewScript::new(settings.chinese_script, &output_language);
        let mut finalize_reply: Option<mpsc::Sender<Option<FinalizedStreamText>>> = None;
        let mut finalize_result: Option<Option<FinalizedStreamText>> = None;
        let stream_started = 'stream: {
            let session = match &mut engine {
                LoadedEngine::TranscribeCpp(s) => s,
                _ => break 'stream false,
            };

            // Read the backend string before beginning the stream — the
            // `Stream` borrows `session` mutably for its lifetime, so we can't
            // call `session.model()` once it exists.
            let backend = session.model().backend();

            // StreamOptions::default() uses CommitPolicy::Auto and lets the
            // family pick its own streaming strategy (no family-specific ext).
            let mut stream = match session.stream(&run_options, &StreamOptions::default()) {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to begin stream: {}", e);
                    break 'stream false;
                }
            };

            self.stream_active.store(true, Ordering::Release);
            self.touch_activity();
            info!(
                "Live streaming transcription started (model '{}', backend '{}')",
                model_id, backend
            );

            let mut perf = StreamPerf::new();
            let mut committed_text = String::new();
            let mut committed_segments = Vec::new();
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    StreamCmd::Feed(pcm) => {
                        self.router.record_feed_dequeued();
                        self.touch_activity();
                        perf.record_feed(pcm.len());
                        let feed_start = Instant::now();
                        match stream.feed(&pcm) {
                            Ok(update) => {
                                perf.record_compute(feed_start.elapsed());
                                perf.record_update(
                                    update.revision,
                                    update.input_received_ms,
                                    update.audio_committed_ms,
                                    update.buffered_ms,
                                );
                                if update.committed_changed || update.tentative_changed {
                                    let text = stream.text();
                                    if update.committed_changed {
                                        append_stream_text_segment(
                                            &mut committed_segments,
                                            &mut committed_text,
                                            &text.committed,
                                            update.audio_committed_ms,
                                        );
                                    }
                                    perf.record_emit();
                                    let (committed, tentative) = preview_script.convert(
                                        &text.committed,
                                        &text.tentative,
                                        &languages,
                                    );
                                    self.emit_stream_text(&committed, &tentative);
                                }
                                perf.maybe_log();
                            }
                            Err(e) => {
                                perf.record_compute(feed_start.elapsed());
                                warn!("stream feed failed: {}", e);
                            }
                        }
                    }
                    StreamCmd::Finalize(reply) => {
                        let finalize_start = Instant::now();
                        let result = match stream.finalize() {
                            // After finalize the committed prefix holds the full
                            // text; display() = committed + tentative is the safe read.
                            Ok(update) => {
                                perf.record_compute(finalize_start.elapsed());
                                perf.record_update(
                                    update.revision,
                                    update.input_received_ms,
                                    update.audio_committed_ms,
                                    update.buffered_ms,
                                );
                                // In auto mode the model's own LID is the best
                                // remaining evidence; the snapshot is only
                                // materialized when it can change the outcome.
                                let output_language = match &output_language {
                                    OutputLanguageEvidence::Unknown => {
                                        with_model_detected_language(
                                            OutputLanguageEvidence::Unknown,
                                            stream.snapshot().language,
                                        )
                                    }
                                    resolved => resolved.clone(),
                                };
                                let text = stream.text();
                                append_stream_text_segment(
                                    &mut committed_segments,
                                    &mut committed_text,
                                    &text.committed,
                                    update.audio_committed_ms,
                                );
                                let display = text.display();
                                append_stream_text_segment(
                                    &mut committed_segments,
                                    &mut committed_text,
                                    &display,
                                    update.input_received_ms.max(update.audio_committed_ms),
                                );
                                if committed_segments.is_empty() {
                                    Some(FinalizedStreamText {
                                        output: TranscriptionOutput::text_only(
                                            display,
                                            perf.streamed_samples as usize,
                                        ),
                                        output_language: output_language.clone(),
                                        supported_languages: languages.clone(),
                                    })
                                } else {
                                    Some(FinalizedStreamText {
                                        output: TranscriptionOutput {
                                            text: display,
                                            segments: std::mem::take(&mut committed_segments),
                                        },
                                        output_language,
                                        supported_languages: languages.clone(),
                                    })
                                }
                            }
                            Err(e) => {
                                perf.record_compute(finalize_start.elapsed());
                                error!(
                                    "stream finalize failed: {}; falling back to batch transcription",
                                    e
                                );
                                None
                            }
                        };
                        let chars = match &result {
                            Some(finalized) => finalized.output.text.len(),
                            _ => 0,
                        };
                        perf.log_finalized(chars);
                        finalize_reply = Some(reply);
                        finalize_result = Some(result);
                        break;
                    }
                    StreamCmd::Cancel => {
                        stream.reset();
                        break;
                    }
                }
            }

            true
        };
        // `stream` + the `&mut engine` borrow are released here.

        if !stream_started {
            // Stream never began (model doesn't support streaming or begin
            // failed); drain so the finalize handshake still completes and the
            // caller falls back to batch transcription. Return the engine first
            // so the fallback can immediately use it.
            self.return_engine(engine, &model_id);
            drain_until_finalize(rx);
            return;
        }

        self.return_engine(engine, &model_id);
        if let (Some(reply), Some(result)) = (finalize_reply, finalize_result) {
            let _ = reply.send(result);
        }
        // `_worker` drops here, clearing this worker's active/lease flags after
        // the engine has been returned to the pool.
    }

    /// Return the leased engine to the mutex, unless the model was switched or
    /// unloaded during transcription (in which case the stale engine is dropped).
    fn return_engine(&self, engine: LoadedEngine, expected_model_id: &str) {
        let still_current =
            self.current_model_id.lock().unwrap().as_deref() == Some(expected_model_id);
        if still_current {
            *self.lock_engine() = Some(engine);
        } else {
            info!(
                "Model changed/unloaded during transcription; dropping stale engine (was '{}')",
                expected_model_id
            );
            // `engine` drops here, freeing its resources.
        }
    }

    /// Flush the active stream and return final, post-filtered text plus
    /// timestamped segments accumulated from its append-only committed prefix.
    ///
    /// `Ok(None)` means no usable stream was active and the caller may fall back
    /// to batch transcription. `Err` means finalize itself failed or timed out.
    /// A timeout may still leave the worker holding the engine, so callers
    /// should surface it instead of immediately starting a batch fallback.
    /// Finalize a stream while retaining its model for an immediate batch
    /// fallback. The eventual stream result or batch owns applying unload policy.
    pub fn finalize_stream_retaining_model(&self) -> Result<Option<TranscriptionOutput>> {
        let Some(tx) = self.router.take() else {
            return Ok(None);
        };
        let stream_was_incomplete = self.router.was_incomplete();
        let (reply_tx, reply_rx) = mpsc::channel();
        let finalize_send_started = Instant::now();
        let mut finalize_cmd = StreamCmd::Finalize(reply_tx);
        loop {
            match tx.try_send(finalize_cmd) {
                Ok(()) => break,
                Err(mpsc::TrySendError::Full(cmd)) => {
                    if finalize_send_started.elapsed() >= STREAM_FINALIZE_REPLY_TIMEOUT {
                        self.stream_active.store(false, Ordering::Release);
                        return Err(anyhow::anyhow!(
                            "Timed out waiting {:?} to enqueue live transcription finalization (queue remained full)",
                            STREAM_FINALIZE_REPLY_TIMEOUT
                        ));
                    }
                    finalize_cmd = cmd;
                    thread::sleep(Duration::from_millis(10));
                }
                Err(mpsc::TrySendError::Disconnected(_)) => return Ok(None),
            }
        }
        let finalized = match reply_rx.recv_timeout(STREAM_FINALIZE_REPLY_TIMEOUT) {
            Ok(Some(finalized)) => finalized,
            Ok(None) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.stream_active.store(false, Ordering::Release);
                return Err(anyhow::anyhow!(
                    "Timed out waiting {:?} for live transcription to finalize",
                    STREAM_FINALIZE_REPLY_TIMEOUT
                ));
            }
        };

        let settings = get_settings(&self.app_handle);
        self.router.log_queue_summary();
        if stream_was_incomplete {
            warn!(
                "Live preview result is incomplete because input frames were dropped; falling back to batch transcription"
            );
            return Ok(None);
        }
        // Streaming models do not receive a decode prompt, so custom words
        // always go through the shared fuzzy post-correction path.
        let mut output = finalized.output;
        let filtered = post_process_transcription_text(
            output.text.clone(),
            &settings,
            false,
            &finalized.output_language,
            &finalized.supported_languages,
        );
        if filtered != output.text {
            let audio_len = output
                .segments
                .last()
                .map_or(0, |segment| segment.end_ms.saturating_mul(16) as usize);
            output = post_process_transcription_output_segments(
                output,
                filtered,
                &settings,
                false,
                audio_len,
                &finalized.output_language,
                &finalized.supported_languages,
            );
        }

        info!(
            "Live preview output ready: chars={}, segments={}",
            output.text.len(),
            output.segments.len(),
        );
        Ok(Some(output))
    }

    /// Abandon any active stream without producing text (e.g. on cancel).
    pub fn cancel_stream(&self) {
        if let Some(tx) = self.router.take() {
            let _ = tx.send(StreamCmd::Cancel);
        }
        self.stream_active.store(false, Ordering::Release);
    }

    /// Emit a working-phase event to the streaming overlay (spinner + label).
    pub fn emit_stream_working(&self, kind: StreamWorkKind) {
        let _ = StreamPhaseEvent {
            phase: StreamPhase::Working,
            kind: Some(kind),
        }
        .emit(&self.app_handle);
    }

    fn emit_stream_text(&self, committed: &str, tentative: &str) {
        let _ = StreamTextEvent {
            committed: committed.to_string(),
            tentative: tentative.to_string(),
        }
        .emit(&self.app_handle);
    }

    pub fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
        self.transcribe_shared(Arc::new(audio))
    }

    /// Transcribe one part of a larger operation without applying the user's
    /// immediate-unload policy between parts.
    pub fn transcribe_retaining_model(&self, audio: Vec<f32>) -> Result<String> {
        self.transcribe_detailed_with_options(Arc::new(audio), false, ModelRetention::Retain)
            .map(|output| output.text)
    }

    pub fn transcribe_shared(&self, audio: Arc<Vec<f32>>) -> Result<String> {
        self.transcribe_detailed_shared(audio)
            .map(|output| output.text)
    }

    pub fn transcribe_detailed_shared(&self, audio: Arc<Vec<f32>>) -> Result<TranscriptionOutput> {
        self.transcribe_bounded_batch(audio, false)
    }

    pub fn transcribe_detailed_for_context_shared(
        &self,
        audio: Arc<Vec<f32>>,
    ) -> Result<TranscriptionOutput> {
        self.transcribe_bounded_batch(audio, true)
    }

    fn transcribe_bounded_batch(
        &self,
        audio: Arc<Vec<f32>>,
        prefer_word_timestamps: bool,
    ) -> Result<TranscriptionOutput> {
        // A finalized stream may have applied the immediate-unload policy
        // before asking for a batch fallback. This is a no-op when still loaded.
        self.initiate_model_load();

        if !should_chunk_long_audio(audio.len()) {
            return self.transcribe_detailed_with_options(
                audio,
                prefer_word_timestamps,
                ModelRetention::ApplyUnloadPolicy,
            );
        }

        let chunk_count = audio.len().div_ceil(LONG_AUDIO_CHUNK_SAMPLES.get());
        info!(
            "Long recording batch transcription started: total_samples={}, duration_secs={:.2}, threshold_secs={}, chunk_seconds={}, chunks={}, detailed_timestamps={}",
            audio.len(),
            audio.len() as f64 / TRANSCRIPTION_SAMPLE_RATE_HZ as f64,
            LONG_AUDIO_BATCH_THRESHOLD_SECONDS,
            LONG_AUDIO_CHUNK_SECONDS,
            chunk_count,
            prefer_word_timestamps,
        );

        let started = Instant::now();
        let mut combined = run_chunked_transcription(
            &audio,
            LONG_AUDIO_CHUNK_SAMPLES,
            |chunk, position| {
                self.transcribe_long_audio_chunk(chunk, position, prefer_word_timestamps)
            },
            |position, error| {
                error!(
                    "Long recording chunk failed: chunk={}/{}, error={}",
                    position.index + 1,
                    position.count,
                    error,
                );
                self.maybe_unload_immediately("failed long recording batch transcription");
            },
        )?;

        self.maybe_unload_immediately("long recording batch transcription");
        info!(
            "Long recording batch transcription completed: chunks={}, elapsed={:?}, chars={}, segments={}",
            chunk_count,
            started.elapsed(),
            combined.text.len(),
            combined.segments.len(),
        );
        combined.text.shrink_to_fit();
        combined.segments.shrink_to_fit();
        Ok(combined)
    }

    fn transcribe_long_audio_chunk(
        &self,
        chunk: Vec<f32>,
        position: ChunkPosition,
        prefer_word_timestamps: bool,
    ) -> Result<TranscriptionOutput> {
        let chunk_started = Instant::now();
        let chunk_start_samples = position.index * LONG_AUDIO_CHUNK_SAMPLES.get();
        info!(
            "Long recording chunk started: chunk={}/{}, start_secs={:.2}, samples={}, pcm_bytes={}",
            position.index + 1,
            position.count,
            chunk_start_samples as f64 / TRANSCRIPTION_SAMPLE_RATE_HZ as f64,
            chunk.len(),
            chunk.len().saturating_mul(std::mem::size_of::<f32>()),
        );

        let output = self.transcribe_detailed_with_options(
            Arc::new(chunk),
            prefer_word_timestamps,
            ModelRetention::Retain,
        )?;
        info!(
            "Long recording chunk finished: chunk={}/{}, elapsed={:?}, chars={}, segments={}",
            position.index + 1,
            position.count,
            chunk_started.elapsed(),
            output.text.len(),
            output.segments.len(),
        );
        Ok(output)
    }

    fn transcribe_detailed_with_options(
        &self,
        audio: Arc<Vec<f32>>,
        prefer_word_timestamps: bool,
        model_retention: ModelRetention,
    ) -> Result<TranscriptionOutput> {
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        // Update last activity timestamp
        self.touch_activity();

        let st = std::time::Instant::now();
        let audio_len = audio.len();

        info!(
            "Transcription input: samples={}, duration={:.2}s, pcm_bytes={}, shared_owners={}, detailed_timestamps={}",
            audio_len,
            audio_len as f64 / 16_000.0,
            audio_len.saturating_mul(std::mem::size_of::<f32>()),
            Arc::strong_count(&audio),
            prefer_word_timestamps,
        );
        log_memory_snapshot("transcription manager entry");

        if audio.is_empty() {
            debug!("Empty audio vector");
            if matches!(model_retention, ModelRetention::ApplyUnloadPolicy) {
                self.maybe_unload_immediately("empty audio");
            }
            return Ok(TranscriptionOutput::text_only(String::new(), 0));
        }

        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        // Get current settings for configuration
        let settings = get_settings(&self.app_handle);

        // Validate selected language against the model's supported languages.
        // If the language isn't supported, fall back to "auto" to prevent errors.
        // Validate against the model that's actually loaded (which can differ
        // from settings.selected_model when a caller loaded a specific model —
        // e.g. the --transcribe-file path's --model), not the persisted
        // selection.
        let active_model = self
            .get_current_model()
            .unwrap_or_else(|| settings.selected_model.clone());
        // Resolve the persisted language *intent* into the language this model
        // will actually use. The coercion is capability-aware (a must-pick model
        // never receives "auto") and computed fresh here — it is never written
        // back to settings, so the intent survives switching models and back.
        let validated_language =
            effective_language_for_model(&settings, self.model_manager.as_ref(), &active_model);
        if validated_language != settings.selected_language {
            debug!(
                "Language intent '{}' resolved to '{}' for model '{}'",
                settings.selected_language, validated_language, active_model
            );
        }

        // Whether the loaded model is actually whisper-family (arch string).
        // Non-whisper archs (e.g. Voxtral Small) can advertise
        // Feature::InitialPrompt yet reject the whisper-kind run extension
        // with INVALID_ARG, so the whisper extension must be gated on the
        // arch, not on the feature (see #1601).
        let mut model_is_whisper = false;

        // Perform transcription with the appropriate engine.
        // We use catch_unwind to prevent engine panics from poisoning the mutex,
        // which would make the app hang indefinitely on subsequent operations.
        let (result, output_language, model_languages) = {
            let mut engine_guard = self.lock_engine();

            // Take the engine out so we own it during transcription.
            // If the engine panics, we simply don't put it back (effectively unloading it)
            // instead of poisoning the mutex.
            let mut engine = match engine_guard.take() {
                Some(e) => e,
                None => {
                    return Err(anyhow::anyhow!(
                        "Model failed to load after auto-load attempt. Please check your model settings."
                    ));
                }
            };

            // Release the lock before transcribing — no mutex held during the engine call
            drop(engine_guard);

            // Probe live transcribe-cpp capabilities once (cheap GGUF-metadata
            // reads); the loaded session is the source of truth, not the
            // ModelManager copy. The whisper run extension is kind-tagged, so
            // non-whisper archs (parakeet, voxtral, …) reject it with
            // INVALID_ARG; attach it — and translate — only where supported.
            let mut model_supports_translate = false;
            let mut model_languages = self
                .model_manager
                .get_model_info(&active_model)
                .map(|info| info.supported_languages)
                .unwrap_or_default();
            let mut output_was_translated = false;
            let mut applied_language_hint: Option<String> = None;
            let mut model_detected_language: Option<String> = None;
            let mut model_max_timestamp_kind = TimestampKind::None;
            if let LoadedEngine::TranscribeCpp(session) = &engine {
                let model = session.model();
                let caps = model.capabilities();
                let model_takes_initial_prompt = model.supports(Feature::InitialPrompt);
                model_is_whisper = model.arch() == "whisper";
                model_supports_translate = caps.supports_translate;
                model_languages = caps.languages;
                model_max_timestamp_kind = caps.max_timestamp_kind;
                debug!(
                    "transcribe-cpp model '{}' on '{}': initial_prompt={}, translate={}, max_timestamps={:?}, languages={:?}",
                    settings.selected_model,
                    model.backend(),
                    model_takes_initial_prompt,
                    model_supports_translate,
                    model_max_timestamp_kind,
                    model_languages
                );
            }

            let transcribe_result = catch_unwind(AssertUnwindSafe(
                || -> Result<TranscriptionOutput> {
                    log_memory_snapshot("before native transcription");
                    match &mut engine {
                        LoadedEngine::TranscribeCpp(session) => {
                            // Custom words become the initial prompt ONLY for models
                            // that accept one (whisper family). Attaching the
                            // whisper run extension to a non-whisper arch is rejected
                            // with INVALID_ARG, so skip it there and let the fuzzy
                            // post-correction handle custom words instead.
                            let family = if settings.custom_words.is_empty() || !model_is_whisper {
                                None
                            } else {
                                Some(RunExtension::Whisper(WhisperRunOptions {
                                    initial_prompt: Some(settings.custom_words.join(", ")),
                                    ..Default::default()
                                }))
                            };

                            let run_plan = transcribe_cpp_run_plan(
                                settings.translate_to_english,
                                &validated_language,
                                &model_languages,
                                model_supports_translate,
                            );

                            output_was_translated =
                                run_plan.target_language.as_deref() == Some("en");
                            applied_language_hint = run_plan.language.clone();
                            let run_options = RunOptions {
                                task: run_plan.task,
                                timestamps: detailed_timestamp_kind(
                                    model_max_timestamp_kind,
                                    prefer_word_timestamps,
                                ),
                                language: run_plan.language,
                                target_language: run_plan.target_language,
                                family,
                                ..Default::default()
                            };

                            debug!(
                                "transcribe-cpp run: task={:?}, language={:?}, timestamps={:?}, initial_prompt={}",
                                run_options.task,
                                run_options.language,
                                run_options.timestamps,
                                run_options.family.is_some()
                            );

                            session
                                .run(&audio, &run_options)
                                .map(|t| {
                                    model_detected_language = t.language.clone();
                                    transcription_output_from_cpp(t, audio_len)
                                })
                                .map_err(|e| {
                                    anyhow::anyhow!("transcribe-cpp transcription failed: {}", e)
                                })
                        }
                        LoadedEngine::Parakeet(parakeet_engine) => {
                            let params = ParakeetParams {
                                timestamp_granularity: Some(TimestampGranularity::Segment),
                                ..Default::default()
                            };
                            parakeet_engine
                                .transcribe_with(&audio, &params)
                                .map(|r| transcription_output_from_rs(r, audio_len))
                                .map_err(|e| {
                                    anyhow::anyhow!("Parakeet transcription failed: {}", e)
                                })
                        }
                        LoadedEngine::Moonshine(moonshine_engine) => moonshine_engine
                            .transcribe(&audio, &TranscribeOptions::default())
                            .map(|r| transcription_output_from_rs(r, audio_len))
                            .map_err(|e| anyhow::anyhow!("Moonshine transcription failed: {}", e)),
                        LoadedEngine::MoonshineStreaming(streaming_engine) => streaming_engine
                            .transcribe(&audio, &TranscribeOptions::default())
                            .map(|r| transcription_output_from_rs(r, audio_len))
                            .map_err(|e| {
                                anyhow::anyhow!("Moonshine streaming transcription failed: {}", e)
                            }),
                        LoadedEngine::SenseVoice(sense_voice_engine) => {
                            let language = match validated_language.as_str() {
                                "zh" => Some("zh".to_string()),
                                "en" => Some("en".to_string()),
                                "ja" => Some("ja".to_string()),
                                "ko" => Some("ko".to_string()),
                                "yue" => Some("yue".to_string()),
                                _ => None,
                            };
                            applied_language_hint = language.clone();
                            let params = SenseVoiceParams {
                                language,
                                use_itn: Some(true),
                            };
                            sense_voice_engine
                                .transcribe_with(&audio, &params)
                                .map(|r| transcription_output_from_rs(r, audio_len))
                                .map_err(|e| {
                                    anyhow::anyhow!("SenseVoice transcription failed: {}", e)
                                })
                        }
                        LoadedEngine::GigaAM(gigaam_engine) => gigaam_engine
                            .transcribe(&audio, &TranscribeOptions::default())
                            .map(|r| transcription_output_from_rs(r, audio_len))
                            .map_err(|e| anyhow::anyhow!("GigaAM transcription failed: {}", e)),
                        LoadedEngine::Canary(canary_engine) => {
                            output_was_translated = settings.translate_to_english;
                            let lang = if validated_language == "auto" {
                                None
                            } else {
                                Some(validated_language.clone())
                            };
                            applied_language_hint = lang.clone();
                            let options = TranscribeOptions {
                                language: lang,
                                translate: settings.translate_to_english,
                                ..Default::default()
                            };
                            canary_engine
                                .transcribe(&audio, &options)
                                .map(|r| transcription_output_from_rs(r, audio_len))
                                .map_err(|e| anyhow::anyhow!("Canary transcription failed: {}", e))
                        }
                        LoadedEngine::Cohere(cohere_engine) => {
                            let lang = if validated_language == "auto" {
                                None
                            } else {
                                Some(validated_language.clone())
                            };
                            applied_language_hint = lang.clone();
                            let options = TranscribeOptions {
                                language: lang,
                                ..Default::default()
                            };
                            cohere_engine
                                .transcribe(&audio, &options)
                                .map(|r| transcription_output_from_rs(r, audio_len))
                                .map_err(|e| anyhow::anyhow!("Cohere transcription failed: {}", e))
                        }
                    }
                },
            ));

            let text = match transcribe_result {
                Ok(inner_result) => {
                    log_memory_snapshot("after native transcription");
                    // Success or normal error: return the engine unless a model
                    // switch/unload invalidated it while it was in use.
                    self.return_engine(engine, &active_model);
                    inner_result?
                }
                Err(panic_payload) => {
                    // Engine panicked — do NOT put it back (it's in an unknown state).
                    // The engine is dropped here, effectively unloading it.
                    let panic_msg = panic_payload_message(panic_payload.as_ref());
                    error!(
                        "Transcription engine panicked: {}. Model has been unloaded.",
                        panic_msg
                    );

                    // Clear the model ID so it will be reloaded on next attempt
                    {
                        let mut current_model = self
                            .current_model_id
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *current_model = None;
                    }

                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "unloaded".to_string(),
                            model_id: None,
                            model_name: None,
                            error: Some(format!("Engine panicked: {}", panic_msg)),
                        },
                    );

                    return Err(anyhow::anyhow!(
                        "Transcription engine panicked: {}. The model has been unloaded and will reload on next attempt.",
                        panic_msg
                    ));
                }
            };

            let output_language = with_model_detected_language(
                resolve_output_language_evidence(
                    &settings,
                    applied_language_hint.as_deref(),
                    &model_languages,
                    output_was_translated,
                ),
                model_detected_language,
            );
            debug!("Output language evidence: {:?}", output_language);

            (text, output_language, model_languages)
        };

        // Apply fuzzy word correction if custom words are configured — UNLESS the
        // words were already handed to the model as an initial prompt (whisper
        // family). We don't pass a prompt to non-whisper models (it requires the
        // whisper-kind run extension), so they still get fuzzy correction here,
        // same as the ONNX engines.
        let mut output = result;
        let filtered_result = post_process_transcription_text(
            output.text.clone(),
            &settings,
            model_is_whisper,
            &output_language,
            &model_languages,
        );
        if filtered_result != output.text {
            output = post_process_transcription_output_segments(
                output,
                filtered_result,
                &settings,
                model_is_whisper,
                audio_len,
                &output_language,
                &model_languages,
            );
        } else {
            output.text = filtered_result;
        }

        let et = std::time::Instant::now();
        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        // Real-time factor. Input PCM is 16 kHz mono, so audio length in seconds
        // is samples / 16000. `speedup` is audio_secs / elapsed_secs — e.g. 4.00x
        // means transcribed 4x faster than real time
        let elapsed_secs = (et - st).as_secs_f64();
        let audio_secs = audio_len as f64 / 16_000.0;
        let speedup = real_time_factor(audio_secs, elapsed_secs);
        info!(
            "Transcription completed in {:.2}s for {:.2}s of audio ({:.2}x real-time){}",
            elapsed_secs, audio_secs, speedup, translation_note
        );

        if output.text.is_empty() {
            info!("Transcription result is empty");
        } else {
            info!(
                "Transcription result: {}",
                crate::utils::redact_text(&output.text)
            );
        }

        if matches!(model_retention, ModelRetention::ApplyUnloadPolicy) {
            self.maybe_unload_immediately("transcription");
        }
        log_memory_snapshot("transcription manager exit");

        Ok(output)
    }
}

fn audio_len_to_ms(audio_len: usize) -> u64 {
    ((audio_len as f64 / TRANSCRIPTION_SAMPLE_RATE_HZ as f64) * 1000.0)
        .round()
        .max(0.0) as u64
}

fn should_chunk_long_audio(sample_count: usize) -> bool {
    sample_count > LONG_AUDIO_BATCH_THRESHOLD_SAMPLES
}

fn run_chunked_transcription<E, Transcribe, OnFailure>(
    audio: &[f32],
    chunk_samples: NonZeroUsize,
    mut transcribe: Transcribe,
    mut on_failure: OnFailure,
) -> std::result::Result<TranscriptionOutput, E>
where
    Transcribe: FnMut(Vec<f32>, ChunkPosition) -> std::result::Result<TranscriptionOutput, E>,
    OnFailure: FnMut(ChunkPosition, &E),
{
    let chunk_samples = chunk_samples.get();
    let chunk_count = audio.len().div_ceil(chunk_samples);
    let mut combined = TranscriptionOutput {
        text: String::new(),
        segments: Vec::new(),
    };

    for (chunk_index, chunk) in audio.chunks(chunk_samples).enumerate() {
        let position = ChunkPosition {
            index: chunk_index,
            count: chunk_count,
        };
        match transcribe(chunk.to_vec(), position) {
            Ok(output) => {
                append_chunk_output(&mut combined, output, chunk_index * chunk_samples);
            }
            Err(error) => {
                on_failure(position, &error);
                return Err(error);
            }
        }
    }

    Ok(combined)
}

fn append_chunk_output(
    combined: &mut TranscriptionOutput,
    mut chunk: TranscriptionOutput,
    chunk_start_samples: usize,
) {
    let offset_ms = audio_len_to_ms(chunk_start_samples);
    for segment in &mut chunk.segments {
        segment.start_ms = segment.start_ms.saturating_add(offset_ms);
        segment.end_ms = segment.end_ms.saturating_add(offset_ms);
    }

    let chunk_text = chunk.text.trim();
    if !chunk_text.is_empty() {
        if should_separate_chunk_text(&combined.text, chunk_text) {
            combined.text.push(' ');
        }
        combined.text.push_str(chunk_text);
    }
    combined.segments.append(&mut chunk.segments);
}

fn should_separate_chunk_text(combined: &str, next: &str) -> bool {
    let Some(previous_char) = combined.chars().next_back() else {
        return false;
    };
    let Some(next_char) = next.chars().next() else {
        return false;
    };

    if previous_char.is_whitespace() || next_char.is_whitespace() {
        return false;
    }
    if is_opening_punctuation(previous_char) || is_closing_punctuation(next_char) {
        return false;
    }
    !(is_cjk_character(previous_char) && is_cjk_character(next_char))
}

fn is_opening_punctuation(character: char) -> bool {
    matches!(
        character,
        '(' | '['
            | '{'
            | '¿'
            | '¡'
            | '«'
            | '（'
            | '「'
            | '『'
            | '【'
            | '《'
            | '〈'
            | '〔'
            | '〖'
            | '〘'
            | '〚'
    )
}

fn is_closing_punctuation(character: char) -> bool {
    matches!(
        character,
        '.' | ','
            | '!'
            | '?'
            | ';'
            | ':'
            | ')'
            | ']'
            | '}'
            | '»'
            | '…'
            | '，'
            | '。'
            | '！'
            | '？'
            | '；'
            | '：'
            | '、'
            | '）'
            | '」'
            | '』'
            | '】'
            | '》'
            | '〉'
            | '〕'
            | '〗'
            | '〙'
            | '〛'
    )
}

fn is_cjk_character(character: char) -> bool {
    matches!(
        character as u32,
        0x3040..=0x30ff
            | 0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0xf900..=0xfaff
            | 0x20000..=0x2fa1f
    )
}

fn append_stream_text_segment(
    segments: &mut Vec<TranscriptionSegment>,
    previous_text: &mut String,
    current_text: &str,
    end_ms: i64,
) {
    if current_text == previous_text {
        return;
    }

    let (start_ms, appended) = match current_text.strip_prefix(previous_text.as_str()) {
        Some(appended) => (
            segments.last().map_or(0, |segment| segment.end_ms),
            appended,
        ),
        None => {
            // The streaming API documents `committed` as append-only. Recover
            // conservatively if a backend violates that contract: replace the
            // stale timing history with one segment covering the latest text.
            warn!("Live preview committed text rewrote its prefix; rebuilding stream segments");
            segments.clear();
            (0, current_text)
        }
    };

    let appended = appended.trim();
    if !appended.is_empty() {
        segments.push(TranscriptionSegment {
            start_ms,
            end_ms: (end_ms.max(0) as u64).max(start_ms),
            text: appended.to_string(),
        });
    }
    previous_text.clear();
    previous_text.push_str(current_text);
}

fn seconds_to_ms(seconds: f32) -> u64 {
    (seconds.max(0.0) * 1000.0).round() as u64
}

fn detailed_timestamp_kind(
    max_timestamp_kind: TimestampKind,
    prefer_word_timestamps: bool,
) -> TimestampKind {
    if prefer_word_timestamps {
        match max_timestamp_kind {
            TimestampKind::Word | TimestampKind::Token => TimestampKind::Word,
            _ => TimestampKind::Segment,
        }
    } else {
        TimestampKind::Segment
    }
}

fn transcription_output_from_rs(
    result: transcribe_rs::TranscriptionResult,
    audio_len: usize,
) -> TranscriptionOutput {
    let segments = result
        .segments
        .unwrap_or_default()
        .into_iter()
        .filter_map(|segment| {
            let text = segment.text.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some(TranscriptionSegment {
                start_ms: seconds_to_ms(segment.start),
                end_ms: seconds_to_ms(segment.end),
                text,
            })
        })
        .collect::<Vec<_>>();

    if segments.is_empty() {
        TranscriptionOutput::text_only(result.text, audio_len)
    } else {
        TranscriptionOutput {
            text: result.text,
            segments,
        }
    }
}

fn transcription_output_from_cpp(
    transcript: transcribe_cpp::Transcript,
    audio_len: usize,
) -> TranscriptionOutput {
    let segments = if transcript.words.is_empty() {
        transcript
            .segments
            .into_iter()
            .filter_map(|segment| {
                let text = segment.text.trim().to_string();
                if text.is_empty() {
                    return None;
                }
                Some(TranscriptionSegment {
                    start_ms: segment.t0_ms.max(0) as u64,
                    end_ms: segment.t1_ms.max(0) as u64,
                    text,
                })
            })
            .collect::<Vec<_>>()
    } else {
        transcription_segments_from_cpp_words(&transcript.words)
    };

    if segments.is_empty() {
        TranscriptionOutput::text_only(transcript.text, audio_len)
    } else {
        TranscriptionOutput {
            text: transcript.text,
            segments,
        }
    }
}

fn transcription_segments_from_cpp_words(
    words: &[transcribe_cpp::Word],
) -> Vec<TranscriptionSegment> {
    words
        .iter()
        .filter_map(|word| {
            let text = word.text.trim().to_string();
            if text.is_empty() {
                return None;
            }

            Some(TranscriptionSegment {
                start_ms: word.t0_ms.max(0) as u64,
                end_ms: word.t1_ms.max(word.t0_ms).max(0) as u64,
                text,
            })
        })
        .collect()
}

fn post_process_transcription_output_segments(
    mut output: TranscriptionOutput,
    filtered_text: String,
    settings: &AppSettings,
    custom_words_already_prompted: bool,
    audio_len: usize,
    output_language: &OutputLanguageEvidence,
    supported_languages: &[String],
) -> TranscriptionOutput {
    if output.segments.is_empty() {
        return TranscriptionOutput::text_only(filtered_text, audio_len);
    }

    let segments = std::mem::take(&mut output.segments)
        .into_iter()
        .filter_map(|segment| {
            let TranscriptionSegment {
                start_ms,
                end_ms,
                text,
            } = segment;
            let text = post_process_transcription_text(
                text,
                settings,
                custom_words_already_prompted,
                output_language,
                supported_languages,
            );
            if text.trim().is_empty() {
                None
            } else {
                Some(TranscriptionSegment {
                    start_ms,
                    end_ms,
                    text,
                })
            }
        })
        .collect::<Vec<_>>();

    output.text = filtered_text;
    output.segments = if segments.is_empty() {
        TranscriptionOutput::text_only(output.text.clone(), audio_len).segments
    } else {
        segments
    };
    output
}

struct StreamPerf {
    feed_count: u64,
    emit_count: u64,
    streamed_samples: u64,
    stream_compute_elapsed: Duration,
    last_log: Instant,
    last_memory_log: Instant,
    latest_revision: i32,
    latest_input_received_ms: i64,
    latest_audio_committed_ms: i64,
    latest_buffered_ms: i64,
}

impl StreamPerf {
    fn new() -> Self {
        Self {
            feed_count: 0,
            emit_count: 0,
            streamed_samples: 0,
            stream_compute_elapsed: Duration::ZERO,
            last_log: Instant::now(),
            last_memory_log: Instant::now(),
            latest_revision: 0,
            latest_input_received_ms: 0,
            latest_audio_committed_ms: 0,
            latest_buffered_ms: 0,
        }
    }

    fn record_feed(&mut self, samples: usize) {
        self.feed_count += 1;
        self.streamed_samples += samples as u64;
    }

    fn record_compute(&mut self, elapsed: Duration) {
        self.stream_compute_elapsed += elapsed;
    }

    fn record_update(
        &mut self,
        revision: i32,
        input_received_ms: i64,
        audio_committed_ms: i64,
        buffered_ms: i64,
    ) {
        self.latest_revision = revision;
        self.latest_input_received_ms = input_received_ms;
        self.latest_audio_committed_ms = audio_committed_ms;
        self.latest_buffered_ms = buffered_ms;
    }

    fn record_emit(&mut self) {
        self.emit_count += 1;
    }

    fn maybe_log(&mut self) {
        if self.last_log.elapsed() < STREAM_PERF_LOG_INTERVAL {
            return;
        }

        let audio_secs = self.audio_secs();
        let compute_secs = self.compute_secs();
        debug!(
            "Live preview perf: {:.2}s streamed audio, {:.2}s model compute ({:.2}x real-time), \
             input_received={:.2}s, committed_audio={:.2}s, buffered={}ms, revision={}, \
             {} frames fed, {} updates emitted",
            audio_secs,
            compute_secs,
            real_time_factor(audio_secs, compute_secs),
            self.latest_input_received_ms as f64 / 1000.0,
            self.latest_audio_committed_ms as f64 / 1000.0,
            self.latest_buffered_ms,
            self.latest_revision,
            self.feed_count,
            self.emit_count,
        );
        if self.last_memory_log.elapsed() >= STREAM_MEMORY_LOG_INTERVAL {
            log_memory_snapshot("live transcription stream progress");
            self.last_memory_log = Instant::now();
        }
        self.last_log = Instant::now();
    }

    fn log_finalized(&self, chars: usize) {
        let audio_secs = self.audio_secs();
        let compute_secs = self.compute_secs();
        info!(
            "Live preview finalized in {:.2}s model compute for {:.2}s streamed audio ({:.2}x real-time): \
             input_received={:.2}s, committed_audio={:.2}s, buffered={}ms, revision={}, \
             {} frames fed, {} updates emitted, {} chars",
            compute_secs,
            audio_secs,
            real_time_factor(audio_secs, compute_secs),
            self.latest_input_received_ms as f64 / 1000.0,
            self.latest_audio_committed_ms as f64 / 1000.0,
            self.latest_buffered_ms,
            self.latest_revision,
            self.feed_count,
            self.emit_count,
            chars
        );
        log_memory_snapshot("live transcription stream finalized");
    }

    fn audio_secs(&self) -> f64 {
        self.streamed_samples as f64 / 16_000.0
    }

    fn compute_secs(&self) -> f64 {
        self.stream_compute_elapsed.as_secs_f64()
    }
}

fn real_time_factor(audio_secs: f64, compute_secs: f64) -> f64 {
    if compute_secs > 0.0 {
        audio_secs / compute_secs
    } else {
        0.0
    }
}

/// Resolve the persisted language intent into the language a specific model can
/// use without writing the coerced value back to settings.
fn effective_language_for_model(
    settings: &AppSettings,
    model_manager: &ModelManager,
    model_id: &str,
) -> String {
    match model_manager.get_model_info(model_id) {
        Some(info) => crate::managers::model::effective_language(
            &settings.selected_language,
            &info.supported_languages,
            info.supports_language_detection,
        ),
        None => settings.selected_language.clone(),
    }
}

/// Resolve how confidently Handy knows the language of the text produced by a
/// transcription run. The UI language is deliberately not part of this
/// decision.
fn resolve_output_language_evidence(
    settings: &AppSettings,
    applied_language_hint: Option<&str>,
    supported_languages: &[String],
    translated_to_english: bool,
) -> OutputLanguageEvidence {
    if translated_to_english {
        return OutputLanguageEvidence::TranslatedToEnglish;
    }

    // Stored language intent is only evidence when this specific engine run
    // actually received the hint. Some multilingual engines (notably Parakeet
    // V3) always auto-detect and ignore Handy's selection; transcribe-cpp also
    // drops a requested hint when the loaded model does not advertise it.
    if let Some(language) = applied_language_hint.filter(|lang| !lang.is_empty() && *lang != "auto")
    {
        if settings.selected_language != "auto"
            && crate::managers::model::canonical_language_code(&settings.selected_language)
                == crate::managers::model::canonical_language_code(language)
        {
            return OutputLanguageEvidence::UserSelected(language.to_string());
        }

        // The engine may have required a concrete fallback even though the
        // user's persisted language was auto or unsupported.
        return OutputLanguageEvidence::ModelConstrained(language.to_string());
    }

    // A single-language model has a known output language without needing a
    // selectable language hint.
    if let [language] = supported_languages {
        return OutputLanguageEvidence::ModelConstrained(language.clone());
    }

    OutputLanguageEvidence::Unknown
}

/// Upgrade [`OutputLanguageEvidence::Unknown`] with the language the model
/// itself detected during the run (audio-based LID, e.g. Whisper in auto
/// mode). Stronger evidence resolved before the run is never overridden.
fn with_model_detected_language(
    evidence: OutputLanguageEvidence,
    detected: Option<String>,
) -> OutputLanguageEvidence {
    match (evidence, detected) {
        (OutputLanguageEvidence::Unknown, Some(language))
            if !language.is_empty() && language != "auto" =>
        {
            OutputLanguageEvidence::ModelDetected(language)
        }
        (evidence, _) => evidence,
    }
}

struct TranscribeCppRunPlan {
    task: Task,
    language: Option<String>,
    target_language: Option<String>,
}

/// Build the transcribe-cpp language/task options shared by batch and live
/// streaming paths.
fn transcribe_cpp_run_plan(
    translate_to_english: bool,
    effective_language: &str,
    model_languages: &[String],
    model_supports_translate: bool,
) -> TranscribeCppRunPlan {
    let requested_language = match effective_language {
        "auto" => None,
        other => Some(other.to_string()),
    };
    // Only pass a language the loaded model actually advertises (per
    // capabilities().languages); otherwise auto-detect rather than failing with
    // UNSUPPORTED_LANGUAGE. Language-agnostic models report an empty list, so
    // they always stay on auto.
    let language = requested_language.filter(|lang| model_languages.iter().any(|l| l == lang));
    let (task, target_language) = cpp_translation_task(
        translate_to_english,
        model_supports_translate,
        language.as_deref(),
    );

    TranscribeCppRunPlan {
        task,
        language,
        target_language,
    }
}

fn post_process_transcription_text(
    raw: String,
    settings: &AppSettings,
    custom_words_already_prompted: bool,
    output_language: &OutputLanguageEvidence,
    supported_languages: &[String],
) -> String {
    let converts_script = settings.chinese_script != ChineseScript::AsTranscribed;
    fail_open_text_transform(raw, |raw| {
        // Last-resort language evidence: confidence-gated detection from the
        // transcribed text itself, constrained to the model's languages. Only
        // consulted when it can change the outcome (built-in gated fillers or
        // Chinese script conversion).
        let output_language = match output_language {
            OutputLanguageEvidence::Unknown
                if converts_script
                    || (settings.filler_word_removal_enabled
                        && settings.custom_filler_words.is_none()) =>
            {
                match detect_output_language(&raw, supported_languages) {
                    Some(language) => {
                        debug!("Text-based language detection resolved '{}'", language);
                        OutputLanguageEvidence::TextDetected(language)
                    }
                    None => OutputLanguageEvidence::Unknown,
                }
            }
            other => other.clone(),
        };

        // Convert the script before custom words so they match in the script
        // the user writes in. Only output known to be Chinese is touched, so
        // e.g. Japanese kanji are never rewritten.
        let variety = output_language
            .language()
            .and_then(ChineseVariety::from_language);
        let raw = match variety {
            Some(variety) if converts_script => {
                convert_chinese_script(&raw, variety, settings.chinese_script)
            }
            _ => raw,
        };

        let corrected = if !settings.custom_words.is_empty() && !custom_words_already_prompted {
            apply_custom_words(
                &raw,
                &settings.custom_words,
                settings.word_correction_threshold,
            )
        } else {
            raw
        };

        let without_fillers = remove_filler_words(
            &corrected,
            &output_language,
            &settings.custom_filler_words,
            settings.filler_word_removal_enabled,
        );

        normalize_transcription_output(&without_fillers)
    })
}

/// Characters to wait for before detecting the preview's language. A lone
/// Chinese character reads as Mandarin, but Japanese often opens with kanji
/// before any kana; waiting a few characters keeps those from locking in.
const PREVIEW_DETECTION_MIN_CHARS: usize = 6;

/// Converts live-preview text into the configured Chinese script as it streams.
///
/// The preview is cosmetic: the final paste re-resolves the language and
/// converts on its own, so this only has to look right while speaking.
struct PreviewScript {
    script: ChineseScript,
    variety: Option<ChineseVariety>,
    /// The language wasn't known when the stream started (auto-detect), so it
    /// is detected from the streamed text until it turns out to be Chinese.
    detect: bool,
}

impl PreviewScript {
    fn new(script: ChineseScript, output_language: &OutputLanguageEvidence) -> Self {
        let enabled = script != ChineseScript::AsTranscribed;
        Self {
            script,
            variety: output_language
                .language()
                .and_then(ChineseVariety::from_language)
                .filter(|_| enabled),
            detect: enabled && *output_language == OutputLanguageEvidence::Unknown,
        }
    }

    /// Converts the model's raw committed/tentative text. Always fed the raw
    /// text, never a previous conversion, so it stays consistent with the
    /// final conversion. Once Chinese is detected it sticks for the rest of
    /// the stream so the preview doesn't flip back and forth.
    fn convert(
        &mut self,
        committed: &str,
        tentative: &str,
        supported_languages: &[String],
    ) -> (String, String) {
        if self.variety.is_none() && self.detect {
            let text = format!("{committed}{tentative}");
            if text.chars().count() >= PREVIEW_DETECTION_MIN_CHARS {
                self.variety = detect_output_language(&text, supported_languages)
                    .as_deref()
                    .and_then(ChineseVariety::from_language);
            }
        }

        match self.variety {
            Some(variety) => (
                convert_chinese_script(committed, variety, self.script),
                convert_chinese_script(tentative, variety, self.script),
            ),
            None => (committed.to_string(), tentative.to_string()),
        }
    }
}

/// Optional text cleanup must never discard a successful model result. The
/// transform is pure and owns its input, so recovering the untouched text is
/// safe even if a bug in custom-word or filler filtering unwinds.
fn fail_open_text_transform<F>(raw: String, transform: F) -> String
where
    F: FnOnce(String) -> String,
{
    let fallback = raw.clone();
    match catch_unwind(AssertUnwindSafe(|| transform(raw))) {
        Ok(processed) => processed,
        Err(payload) => {
            error!(
                "Optional transcription text post-processing panicked: {}; using the raw transcription",
                panic_payload_message(payload.as_ref())
            );
            fallback
        }
    }
}

/// Decide a transcribe-cpp run's task + translation target from settings.
///
/// "Translate to English" only fires where the model advertises translation.
/// Unlike transcribe-rs (which forces the target to English itself when its
/// `translate` flag is set), transcribe-cpp requires an explicit
/// `target_language`: a null target defaults to the *source*, so a non-English
/// source silently becomes e.g. es→es and Canary rejects the unadvertised pair.
/// An English source is skipped entirely — en→en is not a real translation, and
/// it's reachable by default since auto-detect-less models coerce intent to "en".
///
/// Returns `(task, target_language)` ready to drop into `RunOptions`.
fn cpp_translation_task(
    translate_to_english: bool,
    model_supports_translate: bool,
    source_language: Option<&str>,
) -> (Task, Option<String>) {
    let translate_to_en =
        translate_to_english && model_supports_translate && source_language != Some("en");
    if translate_to_en {
        (Task::Translate, Some("en".to_string()))
    } else {
        (Task::Transcribe, None)
    }
}

/// Drain a stream command channel, ignoring fed audio, until the caller
/// finalizes or cancels. Used when streaming can't actually run (model not
/// loaded / not streaming-capable) so the finalize handshake still completes
/// and the caller falls back to batch transcription.
fn drain_until_finalize(rx: mpsc::Receiver<StreamCmd>) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            StreamCmd::Feed(_) => {}
            StreamCmd::Finalize(reply) => {
                let _ = reply.send(None);
                break;
            }
            StreamCmd::Cancel => break,
        }
    }
}

/// Initialize the transcribe-cpp native backend once at startup: route native +
/// ggml diagnostics into the `log` facade and register compute backend modules.
/// In a static build (macOS Metal) `init_backends_default` is a harmless no-op;
/// in a `dynamic-backends` build it loads the per-ISA CPU / GPU modules. Must run
/// before the first model load.
pub fn init_transcribe_backend() {
    transcribe_cpp::init_logging();
    match transcribe_cpp::init_backends_default() {
        Ok(()) => {
            if transcribe_gpu_disabled_for_host() {
                warn!(
                    "Windows x64 build is running under emulation on an ARM64 host; \
                     disabling transcribe.cpp GPU acceleration and using CPU"
                );
            }
        }
        Err(e) => warn!("Failed to initialize transcribe-cpp backends: {}", e),
    }
}

/// Log the compute devices [`init_transcribe_backend`] registered.
///
/// Listing devices is what first opens the GPU. On macOS that loads ggml's
/// Metal library, which is compiled from source whenever the system's shader
/// cache does not hold it yet (the first launch after an install or update),
/// so the app calls this from a background thread instead of its startup
/// path. A model load that comes first waits on the same one-time compile.
pub fn report_compute_devices() {
    let devices = transcribe_compute_devices();
    info!(
        "transcribe-cpp initialized with {} compute device(s): [{}]",
        devices.len(),
        devices
            .iter()
            .map(|d| format!("{} ({})", d.name, d.kind))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Human-readable list of the transcribe-cpp compute devices registered at
/// startup, for the `--list-devices` flag. The reported `index` is the
/// value to pass to `--device-index`. Backends must be initialized first
/// (see [`init_transcribe_backend`]).
pub fn describe_compute_devices() -> Vec<String> {
    transcribe_compute_devices()
        .into_iter()
        .map(|d| {
            let idx = d
                .index
                .map(|i| i.to_string())
                .unwrap_or_else(|| "-".to_string());
            let name = if d.description.is_empty() {
                d.name
            } else {
                d.description
            };
            let vram_mb = d.memory_total / (1024 * 1024);
            format!(
                "index={} kind={} name={} vram={}MB",
                idx, d.kind, name, vram_mb
            )
        })
        .collect()
}

/// Resolve a `--list-devices` registry index to an exact opaque device handle
/// for a transcribe-cpp model load (the `--device-index` flag). In 0.2 index 0
/// is an exact selection too; only an omitted index requests automatic device
/// selection. Errors if the index isn't a registered, loadable primary device.
fn resolve_device_index(index: usize) -> Result<(Backend, Option<transcribe_cpp::Device>)> {
    let device = transcribe_compute_devices()
        .into_iter()
        .find(|d| d.index == Some(index))
        .ok_or_else(|| {
            anyhow::anyhow!("No compute device with index {index} (see --list-devices)")
        })?;
    if matches!(
        device.device_type,
        transcribe_cpp::DeviceType::Accel | transcribe_cpp::DeviceType::Unknown
    ) {
        return Err(anyhow::anyhow!(
            "Device index {index} ({}) cannot host a model",
            device.kind
        ));
    }

    // 0.2's opaque handle makes every index, including zero, an exact
    // selection. Backend::Auto accepts any primary device and cannot conflict
    // with the selected device's vendor backend.
    Ok((Backend::Auto, Some(device)))
}

/// Map Handy's whisper accelerator setting to a transcribe-cpp [`Backend`].
///
/// `Auto` lets the library pick the best device (with CPU fallback), while
/// `Cpu` forces strict CPU. `Gpu` only remains as the companion setting for an
/// exact device; without a valid exact device it has the retired generic GPU
/// state's new Auto semantics. An emulated x64 process on Windows ARM64 forces
/// strict CPU for every setting.
fn select_transcribe_backend(setting: TranscribeAcceleratorSetting) -> Backend {
    select_transcribe_backend_for_host(setting, transcribe_gpu_disabled_for_host())
}

fn select_transcribe_backend_for_host(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
) -> Backend {
    match effective_transcribe_accelerator(setting, gpu_disabled) {
        TranscribeAcceleratorSetting::Cpu => Backend::Cpu,
        TranscribeAcceleratorSetting::Auto | TranscribeAcceleratorSetting::Gpu => Backend::Auto,
    }
}

/// Resolve the user's persisted GPU identity to a fresh opaque 0.2 device
/// handle. Registry indices and handles are process-local, so settings store a
/// key based on the backend's stable `device_id` (falling back to name for
/// backends such as Metal that do not report one).
fn resolve_gpu_device(
    setting: TranscribeAcceleratorSetting,
    gpu_device: Option<&str>,
) -> Option<transcribe_cpp::Device> {
    if transcribe_gpu_disabled_for_host() || setting != TranscribeAcceleratorSetting::Gpu {
        return None;
    }
    let gpu_device = gpu_device?;
    let resolved = transcribe_compute_devices().into_iter().find(|device| {
        is_transcribe_gpu_device(device) && transcribe_device_key(device) == gpu_device
    });
    if resolved.is_none() {
        warn!(
            "Stored transcribe GPU device '{}' is no longer available; using automatic device selection",
            gpu_device
        );
    }
    resolved
}

fn transcribe_device_key(device: &transcribe_cpp::Device) -> String {
    let (identity_kind, identity) = match device.device_id.as_deref() {
        Some(device_id) => ("id", device_id),
        None => ("name", device.name.as_str()),
    };
    serde_json::to_string(&(device.kind.as_str(), identity_kind, identity))
        .expect("transcribe device identity is always JSON serializable")
}

fn transcribe_device_label(device: &transcribe_cpp::Device) -> String {
    if device.description.is_empty() {
        device.name.clone()
    } else {
        device.description.clone()
    }
}

/// Apply the user's ORT accelerator preference to the transcribe-rs global.
/// Called on startup and before loading a model.
///
/// The transcribe.cpp (whisper-family) backend is no longer set here: it is
/// chosen at model-load time from [`select_transcribe_backend`], so changing the
/// accelerator only needs a model reload (see `reload_model_on_next_use`).
pub fn apply_accelerator_settings(app: &tauri::AppHandle) {
    use transcribe_rs::accel;

    let settings = get_settings(app);

    info!(
        "transcribe.cpp accelerator preference: {:?} (applied on next model load)",
        settings.transcribe_accelerator
    );

    let ort_pref = match settings.ort_accelerator {
        OrtAcceleratorSetting::Auto => accel::OrtAccelerator::Auto,
        OrtAcceleratorSetting::Cpu => accel::OrtAccelerator::CpuOnly,
        OrtAcceleratorSetting::Cuda => accel::OrtAccelerator::Cuda,
        OrtAcceleratorSetting::DirectMl => accel::OrtAccelerator::DirectMl,
        OrtAcceleratorSetting::Rocm => accel::OrtAccelerator::Rocm,
    };
    accel::set_ort_accelerator(ort_pref);
    info!("ORT accelerator set to: {}", ort_pref);
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct GpuDeviceOption {
    pub id: String,
    pub name: String,
    pub total_vram_mb: usize,
}

static GPU_DEVICES: OnceLock<Vec<GpuDeviceOption>> = OnceLock::new();

fn transcribe_gpu_disabled_for_host() -> bool {
    crate::utils::is_windows_x64_emulated_on_arm64()
}

fn effective_transcribe_accelerator(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
) -> TranscribeAcceleratorSetting {
    if gpu_disabled {
        TranscribeAcceleratorSetting::Cpu
    } else {
        setting
    }
}

fn is_transcribe_gpu_device(device: &transcribe_cpp::Device) -> bool {
    matches!(
        device.device_type,
        transcribe_cpp::DeviceType::Gpu | transcribe_cpp::DeviceType::Igpu
    )
}

fn transcribe_device_allowed(kind: &str, gpu_disabled: bool) -> bool {
    !gpu_disabled || matches!(kind, "cpu" | "accel")
}

fn transcribe_compute_devices() -> Vec<transcribe_cpp::Device> {
    let devices = transcribe_cpp::devices();
    let gpu_disabled = transcribe_gpu_disabled_for_host();
    if !gpu_disabled {
        return devices;
    }

    devices
        .into_iter()
        .filter(|device| transcribe_device_allowed(&device.kind, gpu_disabled))
        .collect()
}

fn available_transcribe_accelerators(gpu_disabled: bool) -> Vec<String> {
    if gpu_disabled {
        vec!["cpu".to_string()]
    } else {
        vec!["auto".to_string(), "cpu".to_string(), "gpu".to_string()]
    }
}

fn cached_gpu_devices() -> &'static [GpuDeviceOption] {
    // GPU compute devices transcribe-cpp registered at startup. `id` is a
    // persistent identity key, never the process-local registry index. It uses
    // the backend's device_id where available and its name otherwise (Metal).
    // `total_vram_mb` is 0 when the backend does not report capacity.
    GPU_DEVICES.get_or_init(|| {
        transcribe_compute_devices()
            .into_iter()
            .filter(is_transcribe_gpu_device)
            .map(|d| GpuDeviceOption {
                id: transcribe_device_key(&d),
                name: transcribe_device_label(&d),
                total_vram_mb: (d.memory_total / (1024 * 1024)) as usize,
            })
            .collect()
    })
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct AvailableAccelerators {
    pub transcribe: Vec<String>,
    pub ort: Vec<String>,
    pub gpu_devices: Vec<GpuDeviceOption>,
}

/// Return the accelerators available to this process on its current host.
pub fn get_available_accelerators() -> AvailableAccelerators {
    use transcribe_rs::accel::OrtAccelerator;

    let ort_options: Vec<String> = OrtAccelerator::available()
        .into_iter()
        .map(|a| a.to_string())
        .collect();

    let transcribe_options = available_transcribe_accelerators(transcribe_gpu_disabled_for_host());

    AvailableAccelerators {
        transcribe: transcribe_options,
        ort: ort_options,
        gpu_devices: cached_gpu_devices().to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcribe_cpp::{Transcript, Word};

    fn languages(codes: &[&str]) -> Vec<String> {
        codes.iter().map(|code| (*code).to_string()).collect()
    }

    fn word(start_ms: i64, end_ms: i64, text: &str) -> Word {
        Word {
            t0_ms: start_ms,
            t1_ms: end_ms,
            text: text.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn segment_filtering_preserves_timestamps_and_output_language() {
        let settings = AppSettings {
            chinese_script: ChineseScript::Traditional,
            ..Default::default()
        };
        let evidence = OutputLanguageEvidence::UserSelected("zh".into());
        let supported = languages(&["zh"]);
        let original = "这是一个测试".to_string();
        let filtered = post_process_transcription_text(
            original.clone(),
            &settings,
            false,
            &evidence,
            &supported,
        );
        let output = post_process_transcription_output_segments(
            TranscriptionOutput {
                text: original.clone(),
                segments: vec![TranscriptionSegment {
                    start_ms: 500,
                    end_ms: 1_500,
                    text: original,
                }],
            },
            filtered.clone(),
            &settings,
            false,
            32_000,
            &evidence,
            &supported,
        );
        assert_eq!(output.text, filtered);
        assert_eq!(output.segments[0].text, filtered);
        assert_eq!(output.segments[0].start_ms, 500);
        assert_eq!(output.segments[0].end_ms, 1_500);
    }

    #[test]
    fn normal_hosts_preserve_every_transcribe_accelerator_setting() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(effective_transcribe_accelerator(setting, false), setting);
        }
        assert_eq!(
            available_transcribe_accelerators(false),
            ["auto", "cpu", "gpu"]
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Auto, false),
            Backend::Auto
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Cpu, false),
            Backend::Cpu
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Gpu, false),
            Backend::Auto
        );
        for kind in ["cpu", "accel", "metal", "cuda", "vulkan", "gpu"] {
            assert!(transcribe_device_allowed(kind, false));
        }
    }

    #[test]
    fn long_audio_batching_starts_only_above_the_threshold() {
        assert!(!should_chunk_long_audio(LONG_AUDIO_BATCH_THRESHOLD_SAMPLES));
        assert!(should_chunk_long_audio(
            LONG_AUDIO_BATCH_THRESHOLD_SAMPLES + 1
        ));
        assert!(LONG_AUDIO_BATCH_THRESHOLD_SAMPLES > LONG_AUDIO_CHUNK_SAMPLES.get());
    }

    #[test]
    fn chunk_outputs_are_joined_and_timestamps_are_offset() {
        let mut combined = TranscriptionOutput {
            text: String::new(),
            segments: Vec::new(),
        };
        append_chunk_output(
            &mut combined,
            TranscriptionOutput {
                text: " first ".to_string(),
                segments: vec![TranscriptionSegment {
                    start_ms: 100,
                    end_ms: 500,
                    text: "first".to_string(),
                }],
            },
            0,
        );
        append_chunk_output(
            &mut combined,
            TranscriptionOutput {
                text: "second".to_string(),
                segments: vec![TranscriptionSegment {
                    start_ms: 0,
                    end_ms: 400,
                    text: "second".to_string(),
                }],
            },
            LONG_AUDIO_CHUNK_SAMPLES.get(),
        );

        assert_eq!(combined.text, "first second");
        assert_eq!(combined.segments.len(), 2);
        assert_eq!(
            (combined.segments[1].start_ms, combined.segments[1].end_ms),
            (30_000, 30_400)
        );
    }

    #[test]
    fn chunk_text_joining_preserves_punctuation_and_cjk_spacing() {
        assert!(should_separate_chunk_text("hello", "world"));
        assert!(!should_separate_chunk_text("hello", ", world"));
        assert!(!should_separate_chunk_text("¿", "Qué tal?"));
        assert!(!should_separate_chunk_text("你好", "世界"));
        assert!(!should_separate_chunk_text("これは", "テスト"));
        assert!(should_separate_chunk_text("He said \"yes\"", "then left"));
        assert!(!should_separate_chunk_text("「", "你好"));
        assert!(!should_separate_chunk_text("你好", "」"));
    }

    #[test]
    fn chunk_failure_stops_the_batch_and_runs_cleanup_hook() {
        let completed_chunks = std::cell::Cell::new(0);
        let cleaned_up_chunk = std::cell::Cell::new(None);
        let audio = vec![0.25; 6];

        let result = run_chunked_transcription(
            &audio,
            NonZeroUsize::new(2).unwrap(),
            |_, position| {
                if position.index == 1 {
                    return Err("simulated failure");
                }
                completed_chunks.set(completed_chunks.get() + 1);
                Ok(TranscriptionOutput::text_only(
                    format!("part {}", position.index),
                    2,
                ))
            },
            |position, error| {
                assert_eq!(position.count, 3);
                assert_eq!(*error, "simulated failure");
                cleaned_up_chunk.set(Some(position.index));
            },
        );

        assert!(matches!(result, Err("simulated failure")));
        assert_eq!(completed_chunks.get(), 1);
        assert_eq!(cleaned_up_chunk.get(), Some(1));
    }

    #[test]
    fn emulated_x64_on_arm64_forces_every_transcribe_setting_to_cpu() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(
                effective_transcribe_accelerator(setting, true),
                TranscribeAcceleratorSetting::Cpu
            );
            assert_eq!(
                select_transcribe_backend_for_host(setting, true),
                Backend::Cpu
            );
        }
        assert_eq!(available_transcribe_accelerators(true), ["cpu"]);
        assert!(transcribe_device_allowed("cpu", true));
        assert!(transcribe_device_allowed("accel", true));
        for kind in ["metal", "cuda", "vulkan", "gpu", "unknown"] {
            assert!(!transcribe_device_allowed(kind, true));
        }
    }

    #[test]
    fn optional_text_transform_falls_back_to_raw_text_after_panic() {
        let raw = "原始轉錄。".to_string();
        let result = fail_open_text_transform(raw.clone(), |_| {
            panic!("simulated optional cleanup failure")
        });

        assert_eq!(result, raw);
    }

    #[test]
    fn non_chinese_output_is_never_converted() {
        let settings = AppSettings {
            chinese_script: ChineseScript::Traditional,
            ..Default::default()
        };
        for evidence in [
            OutputLanguageEvidence::ModelDetected("ja".to_string()),
            OutputLanguageEvidence::UserSelected("en".to_string()),
            OutputLanguageEvidence::TranslatedToEnglish,
        ] {
            let result = post_process_transcription_text(
                "学校に行きます".to_string(),
                &settings,
                false,
                &evidence,
                &languages(&["zh", "ja", "en"]),
            );

            assert_eq!(result, "学校に行きます", "{evidence:?}");
        }
    }

    #[test]
    fn auto_detected_chinese_text_is_converted() {
        let settings = AppSettings {
            chinese_script: ChineseScript::Simplified,
            filler_word_removal_enabled: false,
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "我們今天下午一起去學校圖書館看書，然後再去吃晚飯。".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::Unknown,
            &languages(&["zh", "en", "ja"]),
        );

        assert_eq!(result, "我们今天下午一起去学校图书馆看书，然后再去吃晚饭。");
    }

    #[test]
    fn portuguese_transcription_does_not_use_english_ui_filler_words() {
        let settings = AppSettings {
            app_language: "en".to_string(),
            selected_language: "pt-BR".to_string(),
            ..Default::default()
        };
        let supported = languages(&["en", "pt"]);
        let evidence = resolve_output_language_evidence(&settings, Some("pt"), &supported, false);

        let result = post_process_transcription_text(
            "eu vi um carro".to_string(),
            &settings,
            false,
            &evidence,
            &supported,
        );

        assert_eq!(
            evidence,
            OutputLanguageEvidence::UserSelected("pt".to_string())
        );
        assert_eq!(result, "eu vi um carro");
    }

    #[test]
    fn norwegian_alias_is_recorded_as_user_selected_evidence() {
        let settings = AppSettings {
            selected_language: "no".to_string(),
            ..Default::default()
        };

        let evidence =
            resolve_output_language_evidence(&settings, Some("nb"), &languages(&["nb"]), false);

        assert_eq!(
            evidence,
            OutputLanguageEvidence::UserSelected("nb".to_string())
        );
    }

    #[test]
    fn auto_language_without_detection_skips_gated_filler_removal() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };
        let evidence =
            resolve_output_language_evidence(&settings, None, &languages(&["en", "pt"]), false);

        // Too short for a reliable text detection, so the gated "um" must
        // survive; the universal "uhm" is removed regardless.
        let result = post_process_transcription_text(
            "um uhm ok".to_string(),
            &settings,
            false,
            &evidence,
            &languages(&["en", "pt"]),
        );

        assert_eq!(evidence, OutputLanguageEvidence::Unknown);
        assert_eq!(result, "um ok");
    }

    #[test]
    fn unknown_evidence_with_confident_text_detection_removes_gated_fillers() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let result = post_process_transcription_text(
            "um so the weather forecast said it would probably rain throughout the whole weekend"
                .to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::Unknown,
            &languages(&["en", "pt", "es", "de"]),
        );

        assert_eq!(
            result,
            "so the weather forecast said it would probably rain throughout the whole weekend"
        );
    }

    #[test]
    fn unknown_evidence_with_portuguese_text_preserves_um() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let result = post_process_transcription_text(
            "eu vi um carro na rua ontem de manhã quando fui ao mercado".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::Unknown,
            &languages(&["en", "pt", "es", "de"]),
        );

        assert_eq!(
            result,
            "eu vi um carro na rua ontem de manhã quando fui ao mercado"
        );
    }

    #[test]
    fn model_detected_language_upgrades_unknown_evidence_only() {
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, Some("en".to_string())),
            OutputLanguageEvidence::ModelDetected("en".to_string())
        );
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, Some("auto".to_string())),
            OutputLanguageEvidence::Unknown
        );
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, None),
            OutputLanguageEvidence::Unknown
        );
        assert_eq!(
            with_model_detected_language(
                OutputLanguageEvidence::UserSelected("pt".to_string()),
                Some("en".to_string())
            ),
            OutputLanguageEvidence::UserSelected("pt".to_string())
        );
    }

    #[test]
    fn auto_language_uses_single_language_model_as_evidence() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let evidence =
            resolve_output_language_evidence(&settings, None, &languages(&["en"]), false);

        assert_eq!(
            evidence,
            OutputLanguageEvidence::ModelConstrained("en".to_string())
        );
    }

    #[test]
    fn unsupported_explicit_language_uses_model_fallback_as_evidence() {
        let settings = AppSettings {
            selected_language: "pt".to_string(),
            ..Default::default()
        };

        let evidence = resolve_output_language_evidence(
            &settings,
            Some("en"),
            &languages(&["en", "de"]),
            false,
        );

        assert_eq!(
            evidence,
            OutputLanguageEvidence::ModelConstrained("en".to_string())
        );
    }

    #[test]
    fn ignored_user_language_is_not_output_evidence() {
        let settings = AppSettings {
            // Parakeet V3 ignores language hints and auto-detects even when a
            // selection from the previously active model remains persisted.
            selected_language: "en".to_string(),
            ..Default::default()
        };
        let supported = languages(&["en", "de", "pt"]);

        let evidence = resolve_output_language_evidence(&settings, None, &supported, false);
        assert_eq!(evidence, OutputLanguageEvidence::Unknown);

        let result = post_process_transcription_text(
            "eu vi um carro".to_string(),
            &settings,
            false,
            &evidence,
            &supported,
        );
        assert_eq!(result, "eu vi um carro");
    }

    #[test]
    fn unapplied_transcribe_cpp_language_is_not_output_evidence() {
        let settings = AppSettings {
            selected_language: "en".to_string(),
            ..Default::default()
        };
        let supported = languages(&[]);
        let plan = transcribe_cpp_run_plan(false, "en", &supported, false);

        assert_eq!(plan.language, None);
        assert_eq!(
            resolve_output_language_evidence(
                &settings,
                plan.language.as_deref(),
                &supported,
                false,
            ),
            OutputLanguageEvidence::Unknown
        );
    }

    #[test]
    fn translated_output_is_treated_as_english() {
        let settings = AppSettings {
            selected_language: "pt".to_string(),
            ..Default::default()
        };

        let evidence = resolve_output_language_evidence(
            &settings,
            Some("pt"),
            &languages(&["en", "pt"]),
            true,
        );

        assert_eq!(evidence, OutputLanguageEvidence::TranslatedToEnglish);
    }

    #[test]
    fn transcribe_cpp_run_plan_skips_english_translation() {
        let plan = transcribe_cpp_run_plan(true, "en", &languages(&["en", "es"]), true);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("en"));
        assert_eq!(plan.target_language, None);
    }

    #[test]
    fn transcribe_cpp_run_plan_translates_supported_non_english() {
        let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), true);

        assert!(matches!(plan.task, Task::Translate));
        assert_eq!(plan.language.as_deref(), Some("es"));
        assert_eq!(plan.target_language.as_deref(), Some("en"));
    }

    #[test]
    fn transcribe_cpp_run_plan_requires_model_translation_support() {
        let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), false);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("es"));
        assert_eq!(plan.target_language, None);
    }

    #[test]
    fn detailed_context_prefers_word_timestamps_when_supported() {
        assert_eq!(
            detailed_timestamp_kind(TimestampKind::Word, true),
            TimestampKind::Word
        );
        assert_eq!(
            detailed_timestamp_kind(TimestampKind::Token, true),
            TimestampKind::Word
        );
        assert_eq!(
            detailed_timestamp_kind(TimestampKind::Segment, true),
            TimestampKind::Segment
        );
        assert_eq!(
            detailed_timestamp_kind(TimestampKind::Word, false),
            TimestampKind::Segment
        );
    }

    #[test]
    fn cpp_word_timestamps_preserve_word_timed_units() {
        let transcript = Transcript {
            text: "first phrase second phrase".to_string(),
            words: vec![
                word(100, 250, "first"),
                word(260, 500, "phrase"),
                word(1_500, 1_700, "second"),
                word(1_720, 1_950, "phrase"),
            ],
            ..Default::default()
        };

        let output = transcription_output_from_cpp(transcript, 32_000);

        assert_eq!(output.segments.len(), 4);
        assert_eq!(output.segments[0].start_ms, 100);
        assert_eq!(output.segments[0].end_ms, 250);
        assert_eq!(output.segments[0].text, "first");
        assert_eq!(output.segments[1].start_ms, 260);
        assert_eq!(output.segments[1].end_ms, 500);
        assert_eq!(output.segments[1].text, "phrase");
        assert_eq!(output.segments[2].start_ms, 1_500);
        assert_eq!(output.segments[2].end_ms, 1_700);
        assert_eq!(output.segments[2].text, "second");
        assert_eq!(output.segments[3].start_ms, 1_720);
        assert_eq!(output.segments[3].end_ms, 1_950);
        assert_eq!(output.segments[3].text, "phrase");
    }

    #[test]
    fn cpp_word_timestamps_are_not_merged_by_speech_gaps() {
        let transcript = Transcript {
            text: "one continuous thought".to_string(),
            words: vec![
                word(100, 250, "one"),
                word(300, 550, "continuous"),
                word(600, 800, "thought"),
            ],
            ..Default::default()
        };

        let output = transcription_output_from_cpp(transcript, 16_000);

        assert_eq!(output.segments.len(), 3);
        assert_eq!(output.segments[0].start_ms, 100);
        assert_eq!(output.segments[0].end_ms, 250);
        assert_eq!(output.segments[0].text, "one");
        assert_eq!(output.segments[1].start_ms, 300);
        assert_eq!(output.segments[1].end_ms, 550);
        assert_eq!(output.segments[1].text, "continuous");
        assert_eq!(output.segments[2].start_ms, 600);
        assert_eq!(output.segments[2].end_ms, 800);
        assert_eq!(output.segments[2].text, "thought");
    }

    #[test]
    fn streaming_commits_become_incremental_timestamped_segments() {
        let mut segments = Vec::new();
        let mut previous = String::new();

        append_stream_text_segment(&mut segments, &mut previous, "hello", 500);
        append_stream_text_segment(&mut segments, &mut previous, "hello brave world", 1_200);
        append_stream_text_segment(
            &mut segments,
            &mut previous,
            "hello brave world again",
            1_700,
        );

        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].text, "hello");
        assert_eq!((segments[0].start_ms, segments[0].end_ms), (0, 500));
        assert_eq!(segments[1].text, "brave world");
        assert_eq!((segments[1].start_ms, segments[1].end_ms), (500, 1_200));
        assert_eq!(segments[2].text, "again");
        assert_eq!((segments[2].start_ms, segments[2].end_ms), (1_200, 1_700));
    }

    #[test]
    fn streaming_prefix_rewrite_rebuilds_stale_segments() {
        let mut segments = Vec::new();
        let mut previous = String::new();
        append_stream_text_segment(&mut segments, &mut previous, "old prefix", 500);

        append_stream_text_segment(&mut segments, &mut previous, "new result", 900);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "new result");
        assert_eq!((segments[0].start_ms, segments[0].end_ms), (0, 900));
    }

    #[test]
    fn streaming_input_queue_is_bounded() {
        let router = StreamRouter::new();
        let _receiver = router.open();
        let frame = vec![0.0; 480];

        for _ in 0..=STREAM_QUEUE_CAPACITY_FRAMES {
            router.feed(&frame);
        }

        assert_eq!(
            router.queued_frames.load(Ordering::Acquire),
            STREAM_QUEUE_CAPACITY_FRAMES as u64
        );
        assert_eq!(
            router.queue_high_water.load(Ordering::Acquire),
            STREAM_QUEUE_CAPACITY_FRAMES as u64
        );
        assert_eq!(router.dropped_frames.load(Ordering::Acquire), 1);
        assert!(router.was_incomplete());
    }
}

impl Drop for TranscriptionManager {
    fn drop(&mut self) {
        // Skip shutdown unless this is the very last clone. TranscriptionManager
        // is cloned by initiate_model_load() and the watcher thread — those
        // clones dropping must not kill the watcher. The watcher thread holds
        // its own clone, so engine's strong_count is always >= 2 while the
        // watcher is alive. When it reaches 1, only this instance remains
        // and we can safely shut down.
        if Arc::strong_count(&self.engine) > 1 {
            return;
        }

        // Signal the watcher thread to shutdown
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for the thread to finish gracefully.
        // Use match instead of unwrap to avoid panicking if the mutex is
        // poisoned — a panic inside Drop calls abort().
        let mut guard = match self.watcher_handle.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Recovered poisoned watcher_handle mutex during TranscriptionManager drop — a panic occurred earlier this session");
                e.into_inner()
            }
        };
        if let Some(handle) = guard.take() {
            if let Err(e) = handle.join() {
                warn!("Failed to join idle watcher thread: {:?}", e);
            } else {
                debug!("Idle watcher thread joined successfully");
            }
        }
    }
}
