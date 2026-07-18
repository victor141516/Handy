use anyhow::Result;
use hound::{WavReader, WavSpec, WavWriter};
use log::{debug, info};
use rodio::{Decoder, Source};
use std::fs::File;
use std::path::Path;
use std::time::Duration;

use super::resampler::FrameResampler;

const TRANSCRIPTION_SAMPLE_RATE: usize = 16_000;
const DECODE_CHUNK_SAMPLES: usize = 4_096;
const MAX_INITIAL_OUTPUT_CAPACITY: usize = TRANSCRIPTION_SAMPLE_RATE * 60;

/// Read a WAV file and return normalised f32 samples.
pub fn read_wav_samples<P: AsRef<Path>>(file_path: P) -> Result<Vec<f32>> {
    let reader = WavReader::open(file_path.as_ref())?;
    let samples = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<f32>, _>>()?;
    Ok(samples)
}

/// Decode a supported audio file, downmix it to mono, and resample it to the
/// 16 kHz input expected by transcription models.
///
/// Decoding and resampling are performed incrementally so compressed source
/// audio is never expanded into a second full-size intermediate buffer.
pub fn decode_audio_file<P: AsRef<Path>>(file_path: P) -> Result<Vec<f32>> {
    let path = file_path.as_ref();
    let file = File::open(path)?;
    let decoder = Decoder::try_from(file)?;
    let channels = decoder.channels() as usize;
    let sample_rate = decoder.sample_rate() as usize;
    let total_duration = decoder.total_duration();

    if channels == 0 {
        anyhow::bail!("Audio file reports zero channels");
    }
    if sample_rate == 0 {
        anyhow::bail!("Audio file reports a zero sample rate");
    }

    // Duration metadata is untrusted and can be wildly incorrect. Preallocate
    // at most one minute, then let the vector grow with audio actually decoded.
    let initial_capacity = total_duration
        .map(|duration| (duration.as_secs_f64() * TRANSCRIPTION_SAMPLE_RATE as f64) as usize)
        .unwrap_or_default()
        .min(MAX_INITIAL_OUTPUT_CAPACITY);
    let mut output = Vec::with_capacity(initial_capacity);
    let mut mono_chunk = Vec::with_capacity(DECODE_CHUNK_SAMPLES);
    let mut resampler = FrameResampler::new(
        sample_rate,
        TRANSCRIPTION_SAMPLE_RATE,
        Duration::from_millis(30),
    );
    let mut channel_sum = 0.0_f32;
    let mut channel_index = 0_usize;

    for sample in decoder {
        channel_sum += sample;
        channel_index += 1;

        if channel_index == channels {
            mono_chunk.push(channel_sum / channels as f32);
            channel_sum = 0.0;
            channel_index = 0;

            if mono_chunk.len() == DECODE_CHUNK_SAMPLES {
                resampler.push(&mono_chunk, |frame| output.extend_from_slice(frame));
                mono_chunk.clear();
            }
        }
    }

    if channel_index > 0 {
        mono_chunk.push(channel_sum / channel_index as f32);
    }
    if !mono_chunk.is_empty() {
        resampler.push(&mono_chunk, |frame| output.extend_from_slice(frame));
    }
    resampler.finish(|frame| output.extend_from_slice(frame));

    info!(
        "Decoded imported audio: path={}, source_rate={}, source_channels={}, source_duration_secs={:.2}, output_samples={}, output_duration_secs={:.2}, output_bytes={}",
        path.display(),
        sample_rate,
        channels,
        total_duration.map_or(0.0, |duration| duration.as_secs_f64()),
        output.len(),
        output.len() as f64 / TRANSCRIPTION_SAMPLE_RATE as f64,
        output.len().saturating_mul(std::mem::size_of::<f32>()),
    );

    Ok(output)
}

/// Verify a WAV file by reading it back and checking the sample count.
pub fn verify_wav_file<P: AsRef<Path>>(file_path: P, expected_samples: usize) -> Result<()> {
    let reader = WavReader::open(file_path.as_ref())?;
    let actual_samples = reader.len() as usize;
    if actual_samples != expected_samples {
        anyhow::bail!(
            "WAV sample count mismatch: expected {}, got {}",
            expected_samples,
            actual_samples
        );
    }
    Ok(())
}

/// Save audio samples as a WAV file
pub fn save_wav_file<P: AsRef<Path>>(file_path: P, samples: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(file_path.as_ref(), spec)?;

    // Convert f32 samples to i16 for WAV
    for sample in samples {
        let sample_i16 = (sample * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    debug!("Saved WAV file: {:?}", file_path.as_ref());
    Ok(())
}
