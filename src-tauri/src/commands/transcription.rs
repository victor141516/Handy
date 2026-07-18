use crate::managers::{
    audio::AudioRecordingManager,
    history::{HistoryEntry, HistoryManager},
    transcription::TranscriptionManager,
};
use crate::settings::{get_settings, write_settings, ModelUnloadTimeout};
use crate::TranscriptionCoordinator;
use chrono::Utc;
use log::{error, info, warn};
use serde::Serialize;
use specta::Type;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Manager, State};

enum ImportedTranscriptionOutcome {
    Completed(String),
    NoSpeech,
    Failed(String),
    Panicked(String),
}

#[derive(Serialize, Type)]
pub struct ModelLoadStatus {
    is_loaded: bool,
    current_model: Option<String>,
}

#[tauri::command]
#[specta::specta]
pub fn set_model_unload_timeout(app: AppHandle, timeout: ModelUnloadTimeout) {
    let mut settings = get_settings(&app);
    settings.model_unload_timeout = timeout;
    write_settings(&app, settings);
}

#[tauri::command]
#[specta::specta]
pub fn get_model_load_status(
    transcription_manager: State<TranscriptionManager>,
) -> Result<ModelLoadStatus, String> {
    Ok(ModelLoadStatus {
        is_loaded: transcription_manager.is_model_loaded(),
        current_model: transcription_manager.get_current_model(),
    })
}

#[tauri::command]
#[specta::specta]
pub fn unload_model_manually(
    transcription_manager: State<TranscriptionManager>,
) -> Result<(), String> {
    transcription_manager
        .unload_model()
        .map_err(|e| format!("Failed to unload model: {}", e))
}

fn validate_audio_source(path: String) -> Result<(PathBuf, u64), String> {
    if path.is_empty() {
        return Err("No audio file was selected".to_string());
    }

    let source_path = PathBuf::from(path);
    if !source_path.is_file() {
        return Err("The selected audio file does not exist".to_string());
    }

    let source_bytes = source_path
        .metadata()
        .map_err(|e| format!("Failed to inspect the selected audio file: {}", e))?
        .len();
    Ok((source_path, source_bytes))
}

async fn decode_imported_audio(source_path: &std::path::Path) -> Result<Vec<f32>, String> {
    let decode_path = source_path.to_path_buf();
    let samples = tauri::async_runtime::spawn_blocking(move || {
        crate::audio_toolkit::decode_audio_file(&decode_path)
    })
    .await
    .map_err(|e| format!("Audio decoding task panicked: {}", e))?
    .map_err(|e| format!("Failed to decode the selected audio file: {}", e))?;

    if samples.is_empty() {
        return Err("The selected audio file contains no audio samples".to_string());
    }
    Ok(samples)
}

async fn save_and_transcribe_imported_audio(
    wav_path: PathBuf,
    samples: Vec<f32>,
    transcription_manager: Arc<TranscriptionManager>,
) -> Result<ImportedTranscriptionOutcome, String> {
    let task_path = wav_path.clone();
    let task_result = tauri::async_runtime::spawn_blocking(move || {
        crate::audio_toolkit::save_wav_file(&task_path, &samples)
            .map_err(|e| format!("Failed to save imported audio: {}", e))?;
        Ok::<_, String>(transcription_manager.transcribe(samples))
    })
    .await;

    let transcription_result = match task_result {
        Ok(Ok(result)) => result,
        Ok(Err(save_error)) => {
            remove_imported_wav(&wav_path, "partial imported WAV");
            return Err(save_error);
        }
        Err(join_error) => {
            return Ok(ImportedTranscriptionOutcome::Panicked(
                join_error.to_string(),
            ));
        }
    };

    Ok(match transcription_result {
        Ok(text) if !text.trim().is_empty() => ImportedTranscriptionOutcome::Completed(text),
        Ok(_) => ImportedTranscriptionOutcome::NoSpeech,
        Err(transcription_error) => {
            ImportedTranscriptionOutcome::Failed(transcription_error.to_string())
        }
    })
}

fn save_failed_import(history_manager: &HistoryManager, file_name: &str) -> Result<(), String> {
    history_manager
        .save_entry(file_name.to_string(), String::new(), false, None, None)
        .map(|_| ())
        .map_err(|e| format!("Failed to save failed import to history: {}", e))
}

fn remove_imported_wav(wav_path: &std::path::Path, description: &str) {
    if let Err(remove_error) = std::fs::remove_file(wav_path) {
        if remove_error.kind() != std::io::ErrorKind::NotFound {
            warn!(
                "Failed to remove {}: path={}, error={}",
                description,
                wav_path.display(),
                remove_error,
            );
        }
    }
}

/// Import an audio file, convert it to Handy's canonical WAV format, transcribe
/// it with the selected model, and add it to the regular transcription history.
///
/// Imported files deliberately never run post-processing or paste text into the
/// focused application. Post-processing remains an explicit action in History.
#[tauri::command]
#[specta::specta]
pub async fn transcribe_audio_file(
    app: AppHandle,
    history_manager: State<'_, Arc<HistoryManager>>,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    path: String,
) -> Result<HistoryEntry, String> {
    let started = Instant::now();
    let (source_path, source_bytes) = validate_audio_source(path)?;
    let audio_manager = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
    let cancel_generation = audio_manager.cancel_generation();
    let _processing_guard = app
        .state::<TranscriptionCoordinator>()
        .try_begin_processing()
        .ok_or_else(|| "Another transcription is already in progress".to_string())?;

    info!(
        "Audio file import started: path={}, source_bytes={}",
        source_path.display(),
        source_bytes,
    );

    let samples = decode_imported_audio(&source_path).await?;
    if audio_manager.was_cancelled_since(cancel_generation) {
        return Err("Audio file transcription was cancelled".to_string());
    }

    info!(
        "Audio file decoded: elapsed={:?}, samples={}, duration_secs={:.2}, pcm_bytes={}",
        started.elapsed(),
        samples.len(),
        samples.len() as f64 / 16_000.0,
        samples.len().saturating_mul(std::mem::size_of::<f32>()),
    );

    let file_name = format!(
        "handy-import-{}.wav",
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    );
    let wav_path = history_manager.recordings_dir().join(&file_name);
    transcription_manager.initiate_model_load();
    let transcription_outcome = save_and_transcribe_imported_audio(
        wav_path.clone(),
        samples,
        Arc::clone(&transcription_manager),
    )
    .await?;
    info!(
        "Imported audio saved: path={}, elapsed={:?}",
        wav_path.display(),
        started.elapsed(),
    );

    if audio_manager.was_cancelled_since(cancel_generation) {
        remove_imported_wav(&wav_path, "cancelled imported WAV");
        return Err("Audio file transcription was cancelled".to_string());
    }

    let transcription = match transcription_outcome {
        ImportedTranscriptionOutcome::Completed(text) => text,
        ImportedTranscriptionOutcome::NoSpeech => {
            warn!(
                "Imported audio contained no recognized speech: path={}",
                source_path.display()
            );
            save_failed_import(&history_manager, &file_name)?;
            return Err("The selected audio file contains no recognized speech".to_string());
        }
        ImportedTranscriptionOutcome::Failed(transcription_error) => {
            error!(
                "Imported audio transcription failed: path={}, error={}",
                source_path.display(),
                transcription_error
            );
            save_failed_import(&history_manager, &file_name)?;
            return Err(format!(
                "Failed to transcribe the selected audio file: {}",
                transcription_error
            ));
        }
        ImportedTranscriptionOutcome::Panicked(join_error) => {
            error!(
                "Imported audio transcription task panicked: path={}, error={}",
                source_path.display(),
                join_error
            );
            save_failed_import(&history_manager, &file_name)?;
            return Err(format!("Audio transcription task panicked: {}", join_error));
        }
    };

    let entry = match history_manager.save_entry(file_name, transcription, false, None, None) {
        Ok(entry) => entry,
        Err(history_error) => {
            remove_imported_wav(&wav_path, "orphaned imported WAV after history error");
            return Err(format!(
                "Failed to add imported transcription to history: {}",
                history_error
            ));
        }
    };

    info!(
        "Audio file import completed: history_id={}, transcription_chars={}, elapsed={:?}",
        entry.id,
        entry.transcription_text.chars().count(),
        started.elapsed(),
    );

    Ok(entry)
}
