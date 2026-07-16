use crate::accessibility_context::{deduplicate_accessibility_events, AccessibilityContextEvent};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error, VadPolicy};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::model::ModelManager;
use crate::managers::transcription::TranscriptionManager;
use crate::managers::transcription::{StreamWorkKind, TranscriptionOutput};
use crate::settings::{get_settings, AppSettings, OverlayStyle, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::tray::{set_tray_state, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::Manager;
use tauri::{AppHandle, Emitter};

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that finishes the transcription pipeline, including immediate
/// model unloading on early exits.
struct FinishGuard(AppHandle, Arc<TranscriptionManager>);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        self.1.maybe_unload_immediately("transcription session");
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
        // The pipeline just freed its large transient buffers (captured PCM,
        // WAV copy, engine scratch); hand the cached pages back to the OS so
        // they don't sit in malloc arenas until they get swapped out (#1792).
        crate::memory::trim_freed_memory();
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Strip a leading `<think>...</think>` block. Some endpoints can't disable
/// reasoning, and some local servers put the reasoning text into `content`
/// instead of a separate field — without this the user would get the model's
/// chain of thought pasted along with the cleaned transcription.
fn strip_think_block(s: &str) -> &str {
    if let Some(rest) = s.trim_start().strip_prefix("<think>") {
        if let Some(end) = rest.find("</think>") {
            return rest[end + "</think>".len()..].trim_start();
        }
    }
    s
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

/// Returns `true` when a transcription has no meaningful content to
/// post-process (empty or whitespace-only). Used to skip the post-processing
/// LLM call when nothing was actually transcribed, which would otherwise make
/// the model reply with an error message such as "you need to provide the
/// transcription".
fn is_blank_transcription(transcription: &str) -> bool {
    transcription.trim().is_empty()
}

enum TimelineEntry {
    Transcription {
        start_ms: u64,
        end_ms: u64,
        text: String,
    },
    Accessibility {
        elapsed_ms: u64,
        app_name: String,
        window_title: String,
        text: String,
    },
}

fn compact_timeline_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn accessibility_hint_location(app_name: &str, window_title: &str) -> String {
    let app_name = compact_timeline_text(app_name);
    let window_title = compact_timeline_text(window_title);

    match (app_name.is_empty(), window_title.is_empty()) {
        (false, true) => app_name,
        (true, false) => window_title,
        (true, true) => "Unknown".to_string(),
        (false, false) if app_name == window_title => app_name,
        (false, false) => format!("{} ({})", app_name, window_title),
    }
}

fn accessibility_hint_text(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = normalized
        .lines()
        .map(compact_timeline_text)
        .collect::<Vec<_>>();

    while lines.first().is_some_and(|line| line.is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }

    match lines.as_slice() {
        [] => String::new(),
        [line] => line.clone(),
        _ => lines
            .into_iter()
            .map(|line| {
                if line.is_empty() {
                    ">".to_string()
                } else {
                    format!("> {}", line)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn format_timestamp(ms: u64) -> String {
    let minutes = ms / 60_000;
    let seconds = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{:02}:{:02}.{:03}", minutes, seconds, millis)
}

fn build_accessibility_timeline(
    transcription: &TranscriptionOutput,
    accessibility_events: &[AccessibilityContextEvent],
) -> Option<String> {
    if accessibility_events.is_empty() {
        return None;
    }

    let mut transcription_entries = Vec::new();
    for segment in &transcription.segments {
        let text = compact_timeline_text(&segment.text);
        if !text.is_empty() {
            transcription_entries.push(TimelineEntry::Transcription {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text,
            });
        }
    }

    if transcription_entries.is_empty() {
        let text = compact_timeline_text(&transcription.text);
        if !text.is_empty() {
            transcription_entries.push(TimelineEntry::Transcription {
                start_ms: 0,
                end_ms: 0,
                text,
            });
        }
    }

    transcription_entries.sort_by_key(|entry| match entry {
        TimelineEntry::Transcription { start_ms, .. } => *start_ms,
        TimelineEntry::Accessibility { .. } => unreachable!("only transcription entries here"),
    });

    let mut accessibility_entries = Vec::new();
    for event in deduplicate_accessibility_events(accessibility_events) {
        let text = accessibility_hint_text(&event.text);
        if !text.is_empty() {
            accessibility_entries.push(TimelineEntry::Accessibility {
                elapsed_ms: event.elapsed_ms,
                app_name: event.app_name.clone(),
                window_title: event.window_title.clone(),
                text,
            });
        }
    }

    accessibility_entries.sort_by_key(|entry| match entry {
        TimelineEntry::Accessibility { elapsed_ms, .. } => *elapsed_ms,
        TimelineEntry::Transcription { .. } => unreachable!("only accessibility entries here"),
    });

    if transcription_entries.is_empty() && accessibility_entries.is_empty() {
        return None;
    }

    let mut output = String::new();
    let mut transcription_index = 0;

    for entry in accessibility_entries {
        let TimelineEntry::Accessibility {
            elapsed_ms,
            app_name,
            window_title,
            text,
        } = entry
        else {
            unreachable!("only accessibility entries here");
        };

        append_transcription_until(
            &mut output,
            &transcription_entries,
            &mut transcription_index,
            elapsed_ms,
        );
        append_accessibility_entry(&mut output, elapsed_ms, &app_name, &window_title, &text);
    }

    append_remaining_transcription(&mut output, &transcription_entries, transcription_index);

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

fn append_transcription_until(
    output: &mut String,
    entries: &[TimelineEntry],
    index: &mut usize,
    boundary_ms: u64,
) {
    let start_index = *index;
    while *index < entries.len() {
        match &entries[*index] {
            TimelineEntry::Transcription { start_ms, .. } if *start_ms < boundary_ms => {
                *index += 1;
            }
            TimelineEntry::Transcription { .. } => break,
            TimelineEntry::Accessibility { .. } => unreachable!("only transcription entries here"),
        }
    }

    append_transcription_group(output, &entries[start_index..*index]);
}

fn append_remaining_transcription(
    output: &mut String,
    entries: &[TimelineEntry],
    start_index: usize,
) {
    append_transcription_group(output, &entries[start_index..]);
}

fn append_transcription_group(output: &mut String, entries: &[TimelineEntry]) {
    let mut start_ms = None;
    let mut end_ms = None;
    let mut text_parts = Vec::new();

    for entry in entries {
        let TimelineEntry::Transcription {
            start_ms: entry_start_ms,
            end_ms: entry_end_ms,
            text,
        } = entry
        else {
            unreachable!("only transcription entries here");
        };

        if text.trim().is_empty() {
            continue;
        }

        start_ms.get_or_insert(*entry_start_ms);
        end_ms = Some(*entry_end_ms);
        text_parts.push(text.as_str());
    }

    let Some(start_ms) = start_ms else {
        return;
    };
    let end_ms = end_ms.unwrap_or(start_ms);
    let text = compact_timeline_text(&text_parts.join(" "));

    if end_ms > start_ms {
        output.push_str(&format!(
            "[{}-{}] Transcription: {}\n",
            format_timestamp(start_ms),
            format_timestamp(end_ms),
            text
        ));
    } else {
        output.push_str(&format!(
            "[{}] Transcription: {}\n",
            format_timestamp(start_ms),
            text
        ));
    }
}

fn append_accessibility_entry(
    output: &mut String,
    elapsed_ms: u64,
    app_name: &str,
    window_title: &str,
    text: &str,
) {
    let location = accessibility_hint_location(app_name, window_title);
    if text.contains('\n') {
        output.push_str(&format!(
            "[{}] Hint: {} - selected element:\n{}\n",
            format_timestamp(elapsed_ms),
            location,
            text
        ));
    } else {
        output.push_str(&format!(
            "[{}] Hint: {} - selected element: {}\n",
            format_timestamp(elapsed_ms),
            location,
            text
        ));
    }
}

async fn complete_unless_cancelled<F, C>(operation: F, is_cancelled: C) -> Option<F::Output>
where
    F: Future,
    C: Fn() -> bool,
{
    tokio::pin!(operation);

    loop {
        if is_cancelled() {
            return None;
        }

        if let Ok(result) =
            tokio::time::timeout(CANCELLATION_POLL_INTERVAL, operation.as_mut()).await
        {
            return Some(result);
        }
    }
}

fn should_use_streaming_overlay(style: OverlayStyle, is_streaming: bool) -> bool {
    style == OverlayStyle::Live && is_streaming
}

async fn post_process_transcription(settings: &AppSettings, transcription: &str) -> Option<String> {
    if is_blank_transcription(transcription) {
        debug!("Post-processing skipped because the transcription is empty");
        return None;
    }

    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return None;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return None;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return None;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return None;
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Ask these providers to skip reasoning/thinking — post-processing rarely
    // benefits from it and it adds seconds of latency. llm_client picks the
    // field the endpoint understands and retries without it if rejected.
    let disable_reasoning = matches!(provider.id.as_str(), "custom" | "openrouter");

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let system_prompt = build_system_prompt(&prompt);
        let user_content = transcription.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                crate::llm_client::log_debug_llm_payload(
                    "Apple Intelligence post-processing request",
                    &serde_json::json!({
                        "provider": &provider.id,
                        "model": &model,
                        "system_prompt": &system_prompt,
                        "user_content": &user_content,
                        "max_tokens": token_limit,
                    }),
                );

                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            None
                        } else {
                            let result = strip_invisible_chars(&result);
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            Some(result)
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
            disable_reasoning,
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                let content = strip_think_block(&content);
                match serde_json::from_str::<serde_json::Value>(content) {
                    Ok(json) => {
                        if let Some(transcription_value) =
                            json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())
                        {
                            let result = strip_invisible_chars(transcription_value);
                            debug!(
                                "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                                provider.id,
                                result.len()
                            );
                            return Some(result);
                        } else {
                            error!("Structured output response missing 'transcription' field");
                            return Some(strip_invisible_chars(content));
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        return Some(strip_invisible_chars(content));
                    }
                }
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return None;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Replace ${output} variable in the prompt with the actual text
    let processed_prompt = prompt.replace("${output}", transcription);
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt,
        disable_reasoning,
    )
    .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(strip_think_block(&content));
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            Some(content)
        }
        Ok(None) => {
            error!("LLM API response has no content");
            None
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
    post_process_input: Option<&str>,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;

    if post_process {
        let llm_input = post_process_input.unwrap_or(&final_text);
        if let Some(processed_text) = post_process_transcription(&settings, llm_input).await {
            post_processed_text = Some(processed_text.clone());
            final_text = processed_text;

            if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                if let Some(prompt) = settings
                    .post_process_prompts
                    .iter()
                    .find(|prompt| &prompt.id == prompt_id)
                {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    }

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Load ASR model and VAD model in parallel
        let kickoff_started = Instant::now();
        tm.initiate_model_load();
        let rm_clone = Arc::clone(&rm);
        std::thread::spawn(move || {
            if let Err(e) = rm_clone.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });
        let kickoff_elapsed = kickoff_started.elapsed();

        // Don't open the mic if nothing can transcribe the recording; the load
        // kicked off above fails and reports why.
        if !tm.is_model_loaded() {
            let selected_model = get_settings(app).selected_model;
            if let Err(e) = app
                .state::<Arc<ModelManager>>()
                .get_model_path(&selected_model)
            {
                warn!("Not starting recording: no model can transcribe it ({})", e);
                return;
            }
        }

        let binding_id = binding_id.to_string();
        let tray_started = Instant::now();
        set_tray_state(app, TrayIconState::Recording);
        let tray_elapsed = tray_started.elapsed();

        // Get the microphone mode to determine audio feedback timing
        let plan_started = Instant::now();
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;

        let selected_model_info = app
            .state::<Arc<ModelManager>>()
            .get_model_info(&settings.selected_model);

        // Use the app-facing model capability as the single pre-recording source
        // for live streaming decisions. Unknown support is represented as false
        // until the model registry is updated by discovery or runtime load.
        let model_supports_streaming = selected_model_info
            .as_ref()
            .map(|m| m.supports_streaming)
            .unwrap_or(false);
        let vad_policy = if !settings.vad_enabled {
            VadPolicy::Disabled
        } else if model_supports_streaming {
            VadPolicy::Streaming
        } else {
            VadPolicy::Offline
        };
        if model_supports_streaming {
            tm.start_stream();
        }
        let plan_elapsed = plan_started.elapsed();

        // Sizing the overlay follows the same advertised capability. A model that
        // doesn't stream (or whose capability is not known yet) gets the compact
        // pill instead of an oversized transparent live window.
        let overlay_started = Instant::now();
        match settings.overlay_style {
            OverlayStyle::Live if model_supports_streaming => utils::show_streaming_overlay(app),
            OverlayStyle::Live | OverlayStyle::Minimal => show_recording_overlay(app),
            OverlayStyle::None => {} // show_overlay_state no-ops on None anyway
        }
        // Everything above runs before capture can begin, so each span here is
        // added keypress->capture latency.
        debug!(
            "start-path pre-recording steps: model_kickoff={:?} tray={:?} settings+stream_plan={:?} overlay={:?}",
            kickoff_elapsed,
            tray_elapsed,
            plan_elapsed,
            overlay_started.elapsed()
        );
        debug!("Microphone mode - always_on: {}", is_always_on);

        let mut recording_error: Option<String> = None;
        let recording_start_time = Instant::now();
        match rm.try_start_recording(&binding_id, vad_policy) {
            Ok(readiness) => {
                debug!(
                    "Recording request accepted in {:?}; waiting for first microphone samples",
                    recording_start_time.elapsed()
                );
                let generation = readiness.generation();
                let app_clone = app.clone();
                let rm_clone = Arc::clone(&rm);
                std::thread::spawn(move || {
                    if !readiness.wait() {
                        debug!("Microphone readiness wait ended without receiving samples");
                        return;
                    }

                    // Development-only preview hook for evaluating the brief
                    // arming animation on hardware that normally starts too fast
                    // to make it visible.
                    #[cfg(debug_assertions)]
                    if let Ok(delay_ms) = std::env::var("HANDY_DEBUG_MIC_READY_DELAY_MS")
                        .unwrap_or_default()
                        .parse::<u64>()
                    {
                        let delay_ms = delay_ms.min(10_000);
                        if delay_ms > 0 {
                            debug!("Delaying microphone-ready cue by {delay_ms}ms for UI preview");
                            std::thread::sleep(Duration::from_millis(delay_ms));
                        }
                    }

                    if !rm_clone.is_recording_readiness_current(generation) {
                        debug!("Microphone became ready for an inactive recording");
                        return;
                    }

                    debug!("Microphone is receiving samples; recording is ready");
                    utils::emit_recording_ready(&app_clone);

                    // The start chime is a readiness cue, so it must follow the
                    // first real input callback rather than Stream::play() or a
                    // fixed delay. The helper returns immediately when feedback
                    // is disabled; mute still follows the same readiness point.
                    if rm_clone.is_recording_readiness_current(generation) {
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                    }
                    if rm_clone.is_recording_readiness_current(generation) {
                        rm_clone.apply_mute();
                    }
                });
            }
            Err(e) => {
                debug!("Failed to start recording: {}", e);
                recording_error = Some(e);
            }
        }

        if recording_error.is_none() {
            if self.post_process && settings.post_process_accessibility_context_enabled {
                crate::accessibility_context::start_recording_context();
            }
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);
        } else {
            let _ = crate::accessibility_context::stop_recording_context();
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            tm.cancel_stream();
            utils::hide_recording_overlay(app);
            set_tray_state(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let error_type = if is_microphone_access_denied(&err) {
                    "microphone_permission_denied"
                } else if is_no_input_device_error(&err) {
                    "no_input_device"
                } else {
                    "unknown"
                };
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: error_type.to_string(),
                        detail: Some(err),
                    },
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Prevent a slow microphone from emitting a ready event or start chime
        // after the user has already requested stop.
        app.state::<Arc<AudioRecordingManager>>()
            .invalidate_recording_readiness();

        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        set_tray_state(app, TrayIconState::Transcribing);
        // Stop should give immediate visual feedback. Live streaming can keep
        // the larger panel, but it still switches from listening to a working
        // spinner while the stream finalizes. Non-streaming paths use the
        // compact transcribing pill (None no-ops in show_*).
        let settings = get_settings(app);
        let style = settings.overlay_style;
        // Capture this before finalizing the stream so every later working state
        // targets the same overlay that was shown for this transcription.
        let use_streaming_overlay = should_use_streaming_overlay(style, tm.is_streaming());
        if use_streaming_overlay {
            tm.emit_stream_working(StreamWorkKind::Transcribing);
        } else {
            show_transcribing_overlay(app);
        }

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;
        let use_accessibility_context =
            post_process && settings.post_process_accessibility_context_enabled;
        let accessibility_events = crate::accessibility_context::stop_recording_context();
        let cancel_generation = rm.cancel_generation();

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone(), Arc::clone(&tm));
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id, cancel_generation) {
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if rm.was_cancelled_since(cancel_generation) {
                    debug!("Transcription operation cancelled after recording stop");
                    tm.cancel_stream();
                    utils::hide_recording_overlay(&ah);
                    set_tray_state(&ah, TrayIconState::Idle);
                    return;
                }

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    // Tear down any streaming worker so its channel doesn't leak
                    // and block the next start_stream.
                    tm.cancel_stream();
                    utils::hide_recording_overlay(&ah);
                    set_tray_state(&ah, TrayIconState::Idle);
                } else {
                    // Save WAV concurrently with transcription
                    let sample_count = samples.len();
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // Transcribe concurrently with WAV save. If a live stream was
                    // running, finalize it and use its text (all audio was already
                    // fed to the stream); otherwise batch-transcribe the samples.
                    let transcription_time = Instant::now();
                    let transcription_result = if use_accessibility_context {
                        // Accessibility timelines need real segment timestamps. The
                        // current live-streaming path only returns final text, so close
                        // the stream worker first, then batch-transcribe with segments.
                        match tm.finalize_stream() {
                            Ok(_) => tm.transcribe_detailed_for_context(samples),
                            Err(err) => Err(err),
                        }
                    } else {
                        match tm.finalize_stream() {
                            // A finalized stream with usable text wins. An empty result
                            // (no active stream, produced nothing, or a finalize error
                            // after the engine was returned) falls back to a full batch
                            // transcription of the same audio. A finalize timeout is
                            // surfaced instead — the worker may still hold the engine,
                            // so a batch fallback would contend with it.
                            Ok(Some(text)) if !text.trim().is_empty() => {
                                Ok(TranscriptionOutput::text_only(text, sample_count))
                            }
                            Ok(_) => tm.transcribe_detailed(samples),
                            Err(err) => Err(err),
                        }
                    };

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    if rm.was_cancelled_since(cancel_generation) {
                        debug!("Transcription operation cancelled before output handling");
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                        return;
                    }

                    match transcription_result {
                        Ok(transcription) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                utils::redact_text(&transcription.text)
                            );

                            if post_process {
                                if use_streaming_overlay {
                                    tm.emit_stream_working(StreamWorkKind::Polishing);
                                } else {
                                    show_processing_overlay(&ah);
                                }
                            }
                            let post_process_input = if use_accessibility_context {
                                build_accessibility_timeline(&transcription, &accessibility_events)
                            } else {
                                None
                            };
                            let Some(processed) = complete_unless_cancelled(
                                process_transcription_output(
                                    &ah,
                                    &transcription.text,
                                    post_process,
                                    post_process_input.as_deref(),
                                ),
                                || rm.was_cancelled_since(cancel_generation),
                            )
                            .await
                            else {
                                debug!("Transcription operation cancelled during output handling");
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            };

                            if rm.was_cancelled_since(cancel_generation) {
                                debug!("Transcription operation cancelled before paste");
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            }

                            // Save to history if WAV was saved
                            if wav_saved {
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription.text.clone(),
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            if processed.final_text.is_empty() {
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let final_text = processed.final_text;
                                let rm_for_paste = Arc::clone(&rm);
                                ah.run_on_main_thread(move || {
                                    if rm_for_paste.was_cancelled_since(cancel_generation) {
                                        debug!("Transcription operation cancelled before paste");
                                        utils::hide_recording_overlay(&ah_clone);
                                        set_tray_state(&ah_clone, TrayIconState::Idle);
                                        return;
                                    }

                                    match utils::paste(final_text, ah_clone.clone()) {
                                        Ok(()) => debug!(
                                            "Text pasted successfully in {:?}",
                                            paste_time.elapsed()
                                        ),
                                        Err(e) => {
                                            error!("Failed to paste transcription: {}", e);
                                            let _ = ah_clone.emit("paste-error", ());
                                        }
                                    }
                                    utils::hide_recording_overlay(&ah_clone);
                                    set_tray_state(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    utils::hide_recording_overlay(&ah);
                                    set_tray_state(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            if rm.was_cancelled_since(cancel_generation) {
                                debug!(
                                    "Transcription operation cancelled after transcription error"
                                );
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            }

                            error!("Transcription failed: {}", err);
                            // Surface the failure to the UI (toast). The full
                            // message is also in handy.log via the line above.
                            let _ = ah.emit("transcription-error", err.to_string());
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            utils::hide_recording_overlay(&ah);
                            set_tray_state(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                // Tear down any streaming worker so its channel doesn't leak.
                tm.cancel_stream();
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod tests {
    use super::{
        build_accessibility_timeline, complete_unless_cancelled, is_blank_transcription,
        should_use_streaming_overlay, strip_think_block,
    };
    use crate::accessibility_context::AccessibilityContextEvent;
    use crate::managers::transcription::{TranscriptionOutput, TranscriptionSegment};
    use crate::settings::OverlayStyle;
    use std::future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn blank_transcription_is_detected() {
        assert!(is_blank_transcription(""));
        assert!(is_blank_transcription("   "));
        assert!(is_blank_transcription("\t\n  \r\n"));
    }

    #[test]
    fn non_blank_transcription_is_kept() {
        assert!(!is_blank_transcription("hello"));
        assert!(!is_blank_transcription("  hello  "));
    }

    #[test]
    fn accessibility_timeline_is_absent_without_accessibility_events() {
        let transcription = TranscriptionOutput {
            text: "hello there".to_string(),
            segments: vec![TranscriptionSegment {
                start_ms: 0,
                end_ms: 1_000,
                text: "hello there".to_string(),
            }],
        };

        assert!(build_accessibility_timeline(&transcription, &[]).is_none());
    }

    #[test]
    fn accessibility_timeline_interleaves_transcription_and_hints() {
        let transcription = TranscriptionOutput {
            text: "open this button".to_string(),
            segments: vec![
                TranscriptionSegment {
                    start_ms: 0,
                    end_ms: 900,
                    text: "open this".to_string(),
                },
                TranscriptionSegment {
                    start_ms: 1_500,
                    end_ms: 2_000,
                    text: "button".to_string(),
                },
            ],
        };
        let accessibility_events = vec![AccessibilityContextEvent {
            elapsed_ms: 1_000,
            app_name: "Google Chrome".to_string(),
            window_title: "Example".to_string(),
            element: "AXButton".to_string(),
            text: "Submit".to_string(),
        }];

        let timeline = build_accessibility_timeline(&transcription, &accessibility_events).unwrap();

        assert!(!timeline.contains("Ordered timeline"));
        assert!(timeline.contains("[00:00.000-00:00.900] Transcription: open this"));
        assert!(timeline
            .contains("[00:01.000] Hint: Google Chrome (Example) - selected element: Submit"));
        assert!(timeline.contains("[00:01.500-00:02.000] Transcription: button"));
        assert!(
            timeline.find("Transcription: open this").unwrap()
                < timeline.find("Hint: Google Chrome").unwrap()
        );
        assert!(
            timeline.find("Hint: Google Chrome").unwrap()
                < timeline.find("Transcription: button").unwrap()
        );
    }

    #[test]
    fn accessibility_timeline_uses_only_app_name_when_window_matches() {
        let transcription = TranscriptionOutput {
            text: "look here".to_string(),
            segments: vec![],
        };
        let accessibility_events = vec![AccessibilityContextEvent {
            elapsed_ms: 250,
            app_name: "Ghosty".to_string(),
            window_title: "Ghosty".to_string(),
            element: "AXGroup".to_string(),
            text: "Terminal content area".to_string(),
        }];

        let timeline = build_accessibility_timeline(&transcription, &accessibility_events).unwrap();

        assert!(
            timeline.contains("[00:00.250] Hint: Ghosty - selected element: Terminal content area")
        );
        assert!(!timeline.contains("Hint: Ghosty (Ghosty)"));
        assert!(!timeline.contains("Application:"));
        assert!(!timeline.contains("Window:"));
        assert!(!timeline.contains("Element:"));
    }

    #[test]
    fn accessibility_timeline_quotes_multiline_selected_text() {
        let transcription = TranscriptionOutput {
            text: "summarize this".to_string(),
            segments: vec![],
        };
        let accessibility_events = vec![AccessibilityContextEvent {
            elapsed_ms: 500,
            app_name: "Notes".to_string(),
            window_title: "Meeting Notes".to_string(),
            element: "AXTextArea".to_string(),
            text: "first line\nsecond line\n\nfourth line".to_string(),
        }];

        let timeline = build_accessibility_timeline(&transcription, &accessibility_events).unwrap();

        assert!(timeline.contains(
            "[00:00.500] Hint: Notes (Meeting Notes) - selected element:\n> first line\n> second line\n>\n> fourth line"
        ));
    }

    #[test]
    fn accessibility_timeline_keeps_only_latest_expanded_selection_hint() {
        let transcription = TranscriptionOutput {
            text: "use this context".to_string(),
            segments: vec![],
        };
        let accessibility_events = vec![
            AccessibilityContextEvent {
                elapsed_ms: 100,
                app_name: "Notes".to_string(),
                window_title: "Draft".to_string(),
                element: "Selected text".to_string(),
                text: "alpha".to_string(),
            },
            AccessibilityContextEvent {
                elapsed_ms: 200,
                app_name: "Notes".to_string(),
                window_title: "Draft".to_string(),
                element: "Selected text".to_string(),
                text: "alpha beta".to_string(),
            },
            AccessibilityContextEvent {
                elapsed_ms: 300,
                app_name: "Notes".to_string(),
                window_title: "Draft".to_string(),
                element: "Selected text".to_string(),
                text: "alpha beta gamma".to_string(),
            },
        ];

        let timeline = build_accessibility_timeline(&transcription, &accessibility_events).unwrap();

        assert!(!timeline.contains("[00:00.100] Hint:"));
        assert!(!timeline.contains("[00:00.200] Hint:"));
        assert!(timeline
            .contains("[00:00.300] Hint: Notes (Draft) - selected element: alpha beta gamma"));
    }

    #[test]
    fn accessibility_timeline_splits_transcription_at_hint_boundaries() {
        let transcription = TranscriptionOutput {
            text: "Vale y sigo hablando despues final".to_string(),
            segments: vec![
                TranscriptionSegment {
                    start_ms: 0,
                    end_ms: 500,
                    text: "Vale".to_string(),
                },
                TranscriptionSegment {
                    start_ms: 700,
                    end_ms: 900,
                    text: "y".to_string(),
                },
                TranscriptionSegment {
                    start_ms: 1_200,
                    end_ms: 1_500,
                    text: "sigo".to_string(),
                },
                TranscriptionSegment {
                    start_ms: 1_600,
                    end_ms: 2_000,
                    text: "hablando".to_string(),
                },
                TranscriptionSegment {
                    start_ms: 4_200,
                    end_ms: 4_700,
                    text: "despues".to_string(),
                },
                TranscriptionSegment {
                    start_ms: 5_000,
                    end_ms: 5_500,
                    text: "final".to_string(),
                },
            ],
        };
        let accessibility_events = vec![
            AccessibilityContextEvent {
                elapsed_ms: 1_000,
                app_name: "Helium".to_string(),
                window_title: "".to_string(),
                element: "Selected text".to_string(),
                text: "automation".to_string(),
            },
            AccessibilityContextEvent {
                elapsed_ms: 4_000,
                app_name: "Helium".to_string(),
                window_title: "".to_string(),
                element: "Selected text".to_string(),
                text: "script".to_string(),
            },
        ];

        let timeline = build_accessibility_timeline(&transcription, &accessibility_events).unwrap();

        assert!(timeline.contains("[00:00.000-00:00.900] Transcription: Vale y"));
        assert!(timeline.contains("[00:01.000] Hint: Helium - selected element: automation"));
        assert!(timeline.contains("[00:01.200-00:02.000] Transcription: sigo hablando"));
        assert!(timeline.contains("[00:04.000] Hint: Helium - selected element: script"));
        assert!(timeline.contains("[00:04.200-00:05.500] Transcription: despues final"));
        assert!(
            timeline.find("Transcription: Vale y").unwrap()
                < timeline.find("selected element: automation").unwrap()
        );
        assert!(
            timeline.find("selected element: automation").unwrap()
                < timeline.find("Transcription: sigo hablando").unwrap()
        );
        assert!(
            timeline.find("Transcription: sigo hablando").unwrap()
                < timeline.find("selected element: script").unwrap()
        );
        assert!(
            timeline.find("selected element: script").unwrap()
                < timeline.find("Transcription: despues final").unwrap()
        );
    }

    #[test]
    fn completed_operation_returns_its_output() {
        let result = tauri::async_runtime::block_on(complete_unless_cancelled(
            future::ready("done"),
            || false,
        ));

        assert_eq!(result, Some("done"));
    }

    #[test]
    fn pending_operation_stops_after_cancellation() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_for_thread = Arc::clone(&cancelled);
        let cancel_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            cancelled_for_thread.store(true, Ordering::Release);
        });

        let result = tauri::async_runtime::block_on(complete_unless_cancelled(
            future::pending::<()>(),
            || cancelled.load(Ordering::Acquire),
        ));

        cancel_thread.join().unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn leading_think_block_is_stripped() {
        assert_eq!(
            strip_think_block("<think>pondering...</think>Cleaned text."),
            "Cleaned text."
        );
        assert_eq!(
            strip_think_block("  \n<think>multi\nline</think>\n  Cleaned text."),
            "Cleaned text."
        );
    }

    #[test]
    fn content_without_think_block_is_unchanged() {
        assert_eq!(strip_think_block("Cleaned text."), "Cleaned text.");
        assert_eq!(
            strip_think_block("Mentions <think> mid-sentence."),
            "Mentions <think> mid-sentence."
        );
        // Unclosed block: leave untouched rather than guess
        assert_eq!(
            strip_think_block("<think>never closed"),
            "<think>never closed"
        );
    }

    #[test]
    fn live_overlay_uses_streaming_states_only_for_streaming_models() {
        assert!(should_use_streaming_overlay(OverlayStyle::Live, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::Live, false));
        assert!(!should_use_streaming_overlay(OverlayStyle::Minimal, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::None, true));
    }
}
