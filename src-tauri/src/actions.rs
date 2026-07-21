#[cfg(test)]
use crate::accessibility_context::deduplicate_accessibility_events;
use crate::accessibility_context::{
    deduplicate_owned_accessibility_events, AccessibilityContextEvent,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error, VadPolicy};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::model::ModelManager;
use crate::managers::transcription::TranscriptionManager;
use crate::managers::transcription::{StreamWorkKind, TranscriptionOutput};
use crate::memory_diagnostics::log_memory_snapshot;
use crate::settings::{get_settings, AppSettings, OverlayStyle, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::tray::{change_tray_icon, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, info, warn};
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

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
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

#[cfg(test)]
fn build_accessibility_timeline(
    transcription: &TranscriptionOutput,
    accessibility_events: &[AccessibilityContextEvent],
) -> Option<String> {
    if accessibility_events.is_empty() {
        return None;
    }

    build_accessibility_timeline_from_deduplicated(
        transcription,
        deduplicate_accessibility_events(accessibility_events),
    )
}

fn build_accessibility_timeline_owned(
    transcription: &TranscriptionOutput,
    accessibility_events: Vec<AccessibilityContextEvent>,
) -> Option<String> {
    if accessibility_events.is_empty() {
        return None;
    }

    build_accessibility_timeline_from_deduplicated(
        transcription,
        deduplicate_owned_accessibility_events(accessibility_events),
    )
}

fn build_accessibility_timeline_from_deduplicated(
    transcription: &TranscriptionOutput,
    accessibility_events: Vec<AccessibilityContextEvent>,
) -> Option<String> {
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
    for event in accessibility_events {
        let text = accessibility_hint_text(&event.text);
        if !text.is_empty() {
            accessibility_entries.push(TimelineEntry::Accessibility {
                elapsed_ms: event.elapsed_ms,
                app_name: event.app_name,
                window_title: event.window_title,
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
    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.as_str(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return None;
        }
    };

    post_process_transcription_with_prompt(settings, transcription, selected_prompt_id).await
}

pub(crate) async fn post_process_transcription_with_prompt(
    settings: &AppSettings,
    transcription: &str,
    prompt_id: &str,
) -> Option<String> {
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

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                prompt_id
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
    info!(
        "Post-processing input: provider={}, model={}, prompt_bytes={}, transcription_bytes={}, transcription_chars={}",
        provider.id,
        model,
        prompt.len(),
        transcription.len(),
        transcription.chars().count(),
    );
    log_memory_snapshot("post-processing input prepared");

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Disable reasoning for providers where post-processing rarely benefits from it.
    // - custom: top-level reasoning_effort (works for local OpenAI-compat servers)
    // - openrouter: nested reasoning object; exclude:true also keeps reasoning text
    //   out of the response so it can't pollute structured-output JSON parsing
    let (reasoning_effort, reasoning) = match provider.id.as_str() {
        "custom" => (Some("none".to_string()), None),
        "openrouter" => (
            None,
            Some(crate::llm_client::ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
        ),
        _ => (None, None),
    };

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
            reasoning_effort.clone(),
            reasoning.clone(),
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                match serde_json::from_str::<serde_json::Value>(&content) {
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
                            return Some(strip_invisible_chars(&content));
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        return Some(strip_invisible_chars(&content));
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
        reasoning_effort,
        reasoning,
    )
    .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(&content);
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

async fn maybe_convert_chinese_variant(
    effective_language: &str,
    transcription: &str,
) -> Option<String> {
    // Gate on the language the model actually transcribed in (the effective
    // language), not the persisted intent. A leftover zh-Hans/zh-Hant intent
    // from a previously selected model must not run OpenCC S2T/T2S over output a
    // non-Chinese model produced — that would silently rewrite any shared CJK
    // characters (e.g. Japanese kanji) in the result.
    let is_simplified = effective_language == "zh-Hans";
    let is_traditional = effective_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("effective language is not Simplified or Traditional Chinese; skipping conversion");
        return None;
    }

    debug!(
        "Starting Chinese variant conversion using OpenCC for language: {}",
        effective_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2tw
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
}

/// Resolve the persisted language *intent* into the language the currently-loaded
/// model will actually use — the same capability-aware coercion the transcription
/// paths apply (see [`crate::managers::model::effective_language`]). Post-processing
/// resolves it independently so it agrees with the language the transcription ran
/// in, without threading a value through the pipeline.
fn resolve_effective_language(app: &AppHandle, settings: &AppSettings) -> String {
    let tm = app.state::<Arc<TranscriptionManager>>();
    let model_manager = app.state::<Arc<ModelManager>>();
    let active_model = tm
        .get_current_model()
        .unwrap_or_else(|| settings.selected_model.clone());
    match model_manager.get_model_info(&active_model) {
        Some(info) => crate::managers::model::effective_language(
            &settings.selected_language,
            &info.supported_languages,
            info.supports_language_detection,
        ),
        None => settings.selected_language.clone(),
    }
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
    post_process_prompt_id: Option<&str>,
    post_process_input: Option<&str>,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;

    // Resolve the language the transcription actually ran in (the persisted
    // intent coerced against the loaded model's capabilities) so OpenCC keys off
    // the effective language rather than a possibly-stale intent.
    let effective_language = resolve_effective_language(app, &settings);
    if let Some(converted_text) =
        maybe_convert_chinese_variant(&effective_language, transcription).await
    {
        final_text = converted_text;
    }

    if post_process {
        let llm_input = post_process_input.unwrap_or(&final_text);
        let resolved_prompt_id = post_process_prompt_id
            .map(str::to_string)
            .or_else(|| settings.post_process_selected_prompt_id.clone());
        let processed_text = match resolved_prompt_id {
            Some(ref prompt_id) => {
                post_process_transcription_with_prompt(&settings, llm_input, prompt_id).await
            }
            None => post_process_transcription(&settings, llm_input).await,
        };

        if let Some(processed_text) = processed_text {
            post_processed_text = Some(processed_text.clone());
            final_text = processed_text;

            if let Some(prompt_id) = resolved_prompt_id {
                if let Some(prompt) = settings
                    .post_process_prompts
                    .iter()
                    .find(|prompt| prompt.id == prompt_id)
                {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    } else if final_text != transcription {
        post_processed_text = Some(final_text.clone());
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

        let binding_id = binding_id.to_string();
        let tray_started = Instant::now();
        change_tray_icon(app, TrayIconState::Recording);
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
        if is_always_on {
            // Always-on mode: Play audio feedback immediately, then apply mute after sound finishes
            debug!("Always-on mode: Playing audio feedback immediately");
            let rm_clone = Arc::clone(&rm);
            let app_clone = app.clone();
            // The blocking helper exits immediately if audio feedback is disabled,
            // so we can always reuse this thread to ensure mute happens right after playback.
            std::thread::spawn(move || {
                play_feedback_sound_blocking(&app_clone, SoundType::Start);
                rm_clone.apply_mute();
            });

            if let Err(e) = rm.try_start_recording(&binding_id, vad_policy) {
                debug!("Recording failed: {}", e);
                recording_error = Some(e);
            }
        } else {
            // On-demand mode: Start recording first, then play audio feedback, then apply mute
            // This allows the microphone to be activated before playing the sound
            debug!("On-demand mode: Starting recording first, then audio feedback");
            let recording_start_time = Instant::now();
            match rm.try_start_recording(&binding_id, vad_policy) {
                Ok(()) => {
                    debug!("Recording started in {:?}", recording_start_time.elapsed());
                    // Small delay to ensure microphone stream is active
                    let app_clone = app.clone();
                    let rm_clone = Arc::clone(&rm);
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        debug!("Handling delayed audio feedback/mute sequence");
                        // Helper handles disabled audio feedback by returning early, so we reuse it
                        // to keep mute sequencing consistent in every mode.
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                        rm_clone.apply_mute();
                    });
                }
                Err(e) => {
                    debug!("Failed to start recording: {}", e);
                    recording_error = Some(e);
                }
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
            change_tray_icon(app, TrayIconState::Idle);
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
        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        change_tray_icon(app, TrayIconState::Transcribing);
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
        let post_process_prompt_id =
            shortcut::post_process_prompt_id_from_binding_id(&binding_id).map(str::to_string);
        let use_accessibility_context =
            post_process && settings.post_process_accessibility_context_enabled;
        let accessibility_events = crate::accessibility_context::stop_recording_context();
        let accessibility_payload_bytes = accessibility_events
            .iter()
            .map(|event| {
                event.app_name.len()
                    + event.window_title.len()
                    + event.element.len()
                    + event.text.len()
            })
            .sum::<usize>();
        info!(
            "Transcription pipeline stop: binding={}, post_process={}, accessibility_context={}, accessibility_events={}, accessibility_payload_bytes={}",
            binding_id,
            post_process,
            use_accessibility_context,
            accessibility_events.len(),
            accessibility_payload_bytes,
        );
        log_memory_snapshot("after stopping accessibility recorder");
        let cancel_generation = rm.cancel_generation();

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );
            log_memory_snapshot("transcription task started");

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id, cancel_generation) {
                info!(
                    "Recording stopped: elapsed={:?}, samples={}, capacity={}, duration={:.2}s, len_bytes={}, capacity_bytes={}",
                    stop_recording_time.elapsed(),
                    samples.len(),
                    samples.capacity(),
                    samples.len() as f64 / 16_000.0,
                    samples.len().saturating_mul(std::mem::size_of::<f32>()),
                    samples.capacity().saturating_mul(std::mem::size_of::<f32>()),
                );
                log_memory_snapshot("recording samples retrieved");

                if rm.was_cancelled_since(cancel_generation) {
                    debug!("Transcription operation cancelled after recording stop");
                    tm.cancel_stream();
                    utils::hide_recording_overlay(&ah);
                    change_tray_icon(&ah, TrayIconState::Idle);
                    return;
                }

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    // Tear down any streaming worker so its channel doesn't leak
                    // and block the next start_stream.
                    tm.cancel_stream();
                    utils::hide_recording_overlay(&ah);
                    change_tray_icon(&ah, TrayIconState::Idle);
                } else {
                    // Share the original Vec allocation between WAV persistence and
                    // transcription. Cloning a 20-minute PCM Vec here used to add a
                    // second ~73 MiB allocation exactly when native inference was
                    // also reserving its largest buffers.
                    let sample_count = samples.len();
                    let samples = Arc::new(samples);
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = Arc::clone(&samples);
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });
                    info!(
                        "Started WAV save with shared PCM buffer: samples={}, bytes={}, owners={}",
                        sample_count,
                        sample_count.saturating_mul(std::mem::size_of::<f32>()),
                        Arc::strong_count(&samples),
                    );
                    log_memory_snapshot("WAV save and transcription about to run");

                    // Transcribe concurrently with WAV save. If a live stream was
                    // running, finalize it and use its text (all audio was already
                    // fed to the stream); otherwise batch-transcribe the samples.
                    let transcription_time = Instant::now();
                    let transcription_result = if use_accessibility_context {
                        // Streaming returns incrementally committed timestamped
                        // segments. Reuse them instead of re-running the entire
                        // recording in batch solely to build the timeline.
                        match tm.finalize_stream_retaining_model() {
                            Ok(Some(output)) if !output.text.trim().is_empty() => {
                                info!(
                                    "Reusing finalized stream for accessibility timeline: chars={}, segments={}; batch retranscription avoided",
                                    output.text.len(),
                                    output.segments.len(),
                                );
                                tm.maybe_unload_immediately("streaming transcription");
                                Ok(output)
                            }
                            Ok(_) => {
                                warn!(
                                    "No complete streaming result available for accessibility timeline; falling back to detailed batch transcription"
                                );
                                log_memory_snapshot(
                                    "before accessibility batch transcription fallback",
                                );
                                tm.transcribe_detailed_for_context_shared(Arc::clone(&samples))
                            }
                            Err(err) => Err(err),
                        }
                    } else {
                        match tm.finalize_stream_retaining_model() {
                            // A finalized stream with usable text wins. An empty result
                            // (no active stream, produced nothing, or a finalize error
                            // after the engine was returned) falls back to a full batch
                            // transcription of the same audio. A finalize timeout is
                            // surfaced instead — the worker may still hold the engine,
                            // so a batch fallback would contend with it.
                            Ok(Some(output)) if !output.text.trim().is_empty() => {
                                info!(
                                    "Using finalized stream output: chars={}, segments={}",
                                    output.text.len(),
                                    output.segments.len(),
                                );
                                tm.maybe_unload_immediately("streaming transcription");
                                Ok(output)
                            }
                            Ok(_) => {
                                info!("No usable streaming result; using batch transcription");
                                log_memory_snapshot("before batch transcription fallback");
                                tm.transcribe_detailed_shared(Arc::clone(&samples))
                            }
                            Err(err) => Err(err),
                        }
                    };
                    drop(samples);
                    log_memory_snapshot("transcription call returned");

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
                    info!("WAV persistence finished: saved={wav_saved}");
                    log_memory_snapshot("WAV persistence finished");

                    if rm.was_cancelled_since(cancel_generation) {
                        debug!("Transcription operation cancelled before output handling");
                        utils::hide_recording_overlay(&ah);
                        change_tray_icon(&ah, TrayIconState::Idle);
                        return;
                    }

                    match transcription_result {
                        Ok(transcription) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                transcription.text
                            );

                            if post_process {
                                if use_streaming_overlay {
                                    tm.emit_stream_working(StreamWorkKind::Polishing);
                                } else {
                                    show_processing_overlay(&ah);
                                }
                            }
                            let post_process_input = if use_accessibility_context {
                                let timeline_started = Instant::now();
                                let input = build_accessibility_timeline_owned(
                                    &transcription,
                                    accessibility_events,
                                );
                                info!(
                                    "Accessibility timeline built: elapsed={:?}, chars={}, bytes={}, transcription_segments={}",
                                    timeline_started.elapsed(),
                                    input.as_ref().map_or(0, |value| value.chars().count()),
                                    input.as_ref().map_or(0, String::len),
                                    transcription.segments.len(),
                                );
                                log_memory_snapshot("accessibility timeline built");
                                input
                            } else {
                                None
                            };
                            let Some(processed) = complete_unless_cancelled(
                                process_transcription_output(
                                    &ah,
                                    &transcription.text,
                                    post_process,
                                    post_process_prompt_id.as_deref(),
                                    post_process_input.as_deref(),
                                ),
                                || rm.was_cancelled_since(cancel_generation),
                            )
                            .await
                            else {
                                debug!("Transcription operation cancelled during output handling");
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
                                return;
                            };
                            log_memory_snapshot("post-processing and output handling completed");

                            if rm.was_cancelled_since(cancel_generation) {
                                debug!("Transcription operation cancelled before paste");
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
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
                                change_tray_icon(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let final_text = processed.final_text;
                                let rm_for_paste = Arc::clone(&rm);
                                ah.run_on_main_thread(move || {
                                    if rm_for_paste.was_cancelled_since(cancel_generation) {
                                        debug!("Transcription operation cancelled before paste");
                                        utils::hide_recording_overlay(&ah_clone);
                                        change_tray_icon(&ah_clone, TrayIconState::Idle);
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
                                    change_tray_icon(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    utils::hide_recording_overlay(&ah);
                                    change_tray_icon(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            if rm.was_cancelled_since(cancel_generation) {
                                debug!(
                                    "Transcription operation cancelled after transcription error"
                                );
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
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
                            change_tray_icon(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                // Tear down any streaming worker so its channel doesn't leak.
                tm.cancel_stream();
                utils::hide_recording_overlay(&ah);
                change_tray_icon(&ah, TrayIconState::Idle);
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
        build_accessibility_timeline, build_accessibility_timeline_owned,
        complete_unless_cancelled, is_blank_transcription, should_use_streaming_overlay,
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
    fn long_recording_timeline_stays_bounded_with_max_sized_context() {
        const RECORDING_SECONDS: u64 = 955;
        let segments = (0..RECORDING_SECONDS / 5)
            .map(|index| TranscriptionSegment {
                start_ms: index * 5_000,
                end_ms: (index + 1) * 5_000,
                text: format!("spoken segment {index}"),
            })
            .collect::<Vec<_>>();
        let transcription = TranscriptionOutput {
            text: segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            segments,
        };
        // 60 distinct ~8 KiB events approximate the recorder's 512 KiB
        // retained-context ceiling over a 15:55 recording.
        let accessibility_events = (0..60_u64)
            .map(|index| AccessibilityContextEvent {
                elapsed_ms: index * (RECORDING_SECONDS * 1_000 / 60),
                app_name: "Synthetic Editor".to_string(),
                window_title: format!("Synthetic document {index}"),
                element: "Selected text".to_string(),
                text: format!("context-{index} {}", "x".repeat(7_900)),
            })
            .collect::<Vec<_>>();

        let timeline =
            build_accessibility_timeline_owned(&transcription, accessibility_events).unwrap();

        assert!(timeline.len() < 512 * 1024);
        assert!(timeline.contains("spoken segment 0"));
        assert!(timeline.contains("spoken segment 190"));
        assert!(timeline.contains("context-59"));
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
    fn live_overlay_uses_streaming_states_only_for_streaming_models() {
        assert!(should_use_streaming_overlay(OverlayStyle::Live, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::Live, false));
        assert!(!should_use_streaming_overlay(OverlayStyle::Minimal, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::None, true));
    }
}
