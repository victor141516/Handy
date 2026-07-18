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

const IMPORT_AUDIO_SAMPLE_RATE_HZ: usize = 16_000;
const IMPORT_TRANSCRIPTION_CHUNK_SECONDS: usize = 30;
const IMPORT_TRANSCRIPTION_CHUNK_SAMPLES: usize =
    IMPORT_AUDIO_SAMPLE_RATE_HZ * IMPORT_TRANSCRIPTION_CHUNK_SECONDS;

enum ImportedTranscriptionOutcome {
    Completed(String),
    Cancelled,
    NoSpeech,
    Failed(String),
    Panicked(String),
}

enum ChunkedTranscription {
    Completed(String),
    Cancelled,
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
    audio_manager: Arc<AudioRecordingManager>,
    cancel_generation: u64,
) -> Result<ImportedTranscriptionOutcome, String> {
    let task_path = wav_path.clone();
    let task_result = tauri::async_runtime::spawn_blocking(move || {
        crate::audio_toolkit::save_wav_file(&task_path, &samples)
            .map_err(|e| format!("Failed to save imported audio: {}", e))?;
        Ok::<_, String>(transcribe_imported_audio_chunks(
            &samples,
            &transcription_manager,
            &audio_manager,
            cancel_generation,
        ))
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
        Ok(ChunkedTranscription::Completed(text)) if !text.trim().is_empty() => {
            ImportedTranscriptionOutcome::Completed(text)
        }
        Ok(ChunkedTranscription::Completed(_)) => ImportedTranscriptionOutcome::NoSpeech,
        Ok(ChunkedTranscription::Cancelled) => ImportedTranscriptionOutcome::Cancelled,
        Err(transcription_error) => {
            ImportedTranscriptionOutcome::Failed(transcription_error.to_string())
        }
    })
}

fn transcribe_imported_audio_chunks(
    samples: &[f32],
    transcription_manager: &TranscriptionManager,
    audio_manager: &AudioRecordingManager,
    cancel_generation: u64,
) -> Result<ChunkedTranscription, String> {
    let total_chunks = samples.len().div_ceil(IMPORT_TRANSCRIPTION_CHUNK_SAMPLES);
    info!(
        "Imported audio transcription chunking: total_samples={}, chunk_samples={}, chunk_seconds={}, total_chunks={}",
        samples.len(),
        IMPORT_TRANSCRIPTION_CHUNK_SAMPLES,
        IMPORT_TRANSCRIPTION_CHUNK_SECONDS,
        total_chunks,
    );

    let transcription_result = transcribe_audio_chunks(
        samples,
        IMPORT_TRANSCRIPTION_CHUNK_SAMPLES,
        || audio_manager.was_cancelled_since(cancel_generation),
        |chunk, chunk_index, chunk_count| {
            transcribe_imported_chunk(transcription_manager, chunk, chunk_index, chunk_count)
        },
    );

    // Native GPU backends may retain large scratch buffers after an interrupted run.
    if matches!(
        &transcription_result,
        Err(_) | Ok(ChunkedTranscription::Cancelled)
    ) {
        info!("Unloading transcription model after failed or cancelled audio import");
        if let Err(unload_error) = transcription_manager.unload_model() {
            warn!(
                "Failed to unload transcription model after audio import stopped: {}",
                unload_error
            );
        }
    } else {
        transcription_manager.maybe_unload_immediately("imported audio transcription");
    }

    transcription_result
}

fn transcribe_imported_chunk(
    transcription_manager: &TranscriptionManager,
    chunk: Vec<f32>,
    chunk_index: usize,
    chunk_count: usize,
) -> Result<String, String> {
    let chunk_started = Instant::now();
    info!(
        "Imported audio chunk started: chunk={}/{}, samples={}, duration_secs={:.2}, pcm_bytes={}",
        chunk_index + 1,
        chunk_count,
        chunk.len(),
        chunk.len() as f64 / IMPORT_AUDIO_SAMPLE_RATE_HZ as f64,
        chunk.len().saturating_mul(std::mem::size_of::<f32>()),
    );

    transcription_manager.initiate_model_load();
    let result = transcription_manager
        .transcribe_retaining_model(chunk)
        .map_err(|error| error.to_string());

    if let Err(transcription_error) = &result {
        error!(
            "Imported audio chunk failed: chunk={}/{}, error={}",
            chunk_index + 1,
            chunk_count,
            transcription_error,
        );
    }
    info!(
        "Imported audio chunk finished: chunk={}/{}, elapsed={:?}, success={}",
        chunk_index + 1,
        chunk_count,
        chunk_started.elapsed(),
        result.is_ok(),
    );
    result
}

fn transcribe_audio_chunks<Transcribe, IsCancelled>(
    samples: &[f32],
    chunk_samples: usize,
    mut is_cancelled: IsCancelled,
    mut transcribe: Transcribe,
) -> Result<ChunkedTranscription, String>
where
    Transcribe: FnMut(Vec<f32>, usize, usize) -> Result<String, String>,
    IsCancelled: FnMut() -> bool,
{
    if chunk_samples == 0 {
        return Err("Audio transcription chunk size must be greater than zero".to_string());
    }

    let chunk_count = samples.len().div_ceil(chunk_samples);
    let mut transcription_parts = Vec::with_capacity(chunk_count);

    for (chunk_index, chunk) in samples.chunks(chunk_samples).enumerate() {
        // Poll both sides of inference so cancellation never starts another chunk
        // and discards a just-finished result when the user cancelled mid-run.
        if is_cancelled() {
            return Ok(ChunkedTranscription::Cancelled);
        }

        let text = transcribe(chunk.to_vec(), chunk_index, chunk_count)?;

        if is_cancelled() {
            return Ok(ChunkedTranscription::Cancelled);
        }

        let text = text.trim();
        if !text.is_empty() {
            transcription_parts.push(text.to_string());
        }
    }

    Ok(ChunkedTranscription::Completed(
        transcription_parts.join(" "),
    ))
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
        samples.len() as f64 / IMPORT_AUDIO_SAMPLE_RATE_HZ as f64,
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
        Arc::clone(&audio_manager),
        cancel_generation,
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
        ImportedTranscriptionOutcome::Cancelled => {
            remove_imported_wav(&wav_path, "cancelled imported WAV");
            return Err("Audio file transcription was cancelled".to_string());
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imported_audio_is_transcribed_in_bounded_chunks_and_joined() {
        let samples = vec![0.25; 10];
        let mut observed_chunks = Vec::new();

        let result = transcribe_audio_chunks(
            &samples,
            4,
            || false,
            |chunk, chunk_index, chunk_count| {
                observed_chunks.push((chunk.len(), chunk_index, chunk_count));
                Ok(format!("part {}", chunk_index + 1))
            },
        )
        .unwrap();

        assert_eq!(observed_chunks, vec![(4, 0, 3), (4, 1, 3), (2, 2, 3)]);
        assert!(matches!(
            result,
            ChunkedTranscription::Completed(text) if text == "part 1 part 2 part 3"
        ));
    }

    #[test]
    fn imported_audio_skips_blank_chunk_results() {
        let samples = vec![0.25; 6];

        let result = transcribe_audio_chunks(
            &samples,
            2,
            || false,
            |_, chunk_index, _| match chunk_index {
                0 => Ok(" first ".to_string()),
                1 => Ok("  ".to_string()),
                _ => Ok("last".to_string()),
            },
        )
        .unwrap();

        assert!(matches!(
            result,
            ChunkedTranscription::Completed(text) if text == "first last"
        ));
    }

    #[test]
    fn imported_audio_cancellation_stops_before_the_next_chunk() {
        let samples = vec![0.25; 10];
        let completed_chunks = std::cell::Cell::new(0);

        let result = transcribe_audio_chunks(
            &samples,
            2,
            || completed_chunks.get() == 2,
            |_, _, _| {
                completed_chunks.set(completed_chunks.get() + 1);
                Ok("part".to_string())
            },
        )
        .unwrap();

        assert_eq!(completed_chunks.get(), 2);
        assert!(matches!(result, ChunkedTranscription::Cancelled));
    }

    #[test]
    fn imported_audio_rejects_a_zero_chunk_size() {
        let result = transcribe_audio_chunks(&[0.25], 0, || false, |_, _, _| Ok(String::new()));

        assert!(matches!(
            result,
            Err(error) if error.contains("greater than zero")
        ));
    }
}
