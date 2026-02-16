// ---------------------------------------------------------------------------
// audio — Audio file I/O and sample manipulation
//
// This is the only module that reads or writes audio files.
// It decodes input audio (any format symphonia supports), converts to mono,
// resamples to the target sample rate, and writes 24-bit WAV output.
// It also applies time offsets (pad/trim) so tracks line up with the master.
//
// The other modules (sync, vad, mfcc) only work with in-memory sample buffers.
// ---------------------------------------------------------------------------

use std::fmt;
use std::fs;
use std::path::Path;

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Standard podcast sample rate (44.1 kHz).
/// This matches common digital audio workstation (DAW) defaults and podcast
/// distribution expectations.
/// All internal processing uses this rate.
pub const TARGET_SAMPLE_RATE: u32 = 44_100;

/// Largest positive 24-bit signed PCM value (2^23 - 1).
/// Used when converting float samples in [-1.0, 1.0] to 24-bit integer samples
/// for WAV output.
const PCM_24_MAX: f32 = 8_388_607.0;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// All errors that can occur in audio loading, writing, or manipulation.
///
/// Each variant includes enough context to produce a useful error message
/// without needing to inspect the call site.
#[derive(Debug)]
pub enum PodsyncError {
    /// The audio file was not found on disk.
    FileNotFound(String),

    /// symphonia could not decode the audio file (corrupt file or unsupported format).
    DecodeFailed(String),

    /// The resampler failed (invalid ratio or internal error).
    ResampleFailed(String),

    /// The WAV writer failed (for example: permissions error, full disk).
    WriteFailed(String),

    /// An offset operation was invalid (for example: trimming more samples than exist).
    InvalidOffset(String),
}

impl fmt::Display for PodsyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PodsyncError::FileNotFound(msg) => write!(f, "File not found: {}", msg),
            PodsyncError::DecodeFailed(msg) => write!(f, "Decode failed: {}", msg),
            PodsyncError::ResampleFailed(msg) => write!(f, "Resample failed: {}", msg),
            PodsyncError::WriteFailed(msg) => write!(f, "Write failed: {}", msg),
            PodsyncError::InvalidOffset(msg) => write!(f, "Invalid offset: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Convert a duration in seconds to a sample count.
///
/// Uses rounding rather than truncation so that a value like 1.9999999 seconds
/// (common with floating-point arithmetic) doesn't lose a sample.
///
/// Returns a signed integer because offsets can be negative (track started
/// before master). Convert to usize only after explicit bounds checking.
pub fn seconds_to_samples(seconds: f64, sr: u32) -> i64 {
    (seconds * sr as f64).round() as i64
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load an audio file, resample to `target_sr`, and convert to mono `f32` samples.
///
/// Supports WAV, MP3, FLAC, OGG/Vorbis, AIFF, and other formats symphonia can decode.
///
/// The loading pipeline is:
/// 1. Open the file and probe the container format.
/// 2. Decode all packets into interleaved `f32` samples.
/// 3. Downmix to mono by averaging channels.
/// 4. Resample to `target_sr` if the file's native rate differs.
///
/// Returns `(samples, sample_rate_hz)`:
/// - `samples`: mono float samples (typically in [-1.0, 1.0])
/// - `sample_rate_hz`: the sample rate in Hz (this will be `target_sr` on success)
pub fn load_audio(path: &Path, target_sr: u32) -> Result<(Vec<f32>, u32), PodsyncError> {
    // --- Validate the file exists before attempting decode -------------------
    if !path.exists() {
        return Err(PodsyncError::FileNotFound(path.display().to_string()));
    }

    // --- Open the file and create a media source stream ---------------------
    let file = fs::File::open(path)
        .map_err(|e| PodsyncError::DecodeFailed(format!("{}: {}", path.display(), e)))?;
    let media_source = MediaSourceStream::new(Box::new(file), Default::default());

    // --- Probe the container format ----------------------------------------
    // The Hint lets symphonia use the file extension to narrow down formats,
    // but it will still probe the actual bytes if the hint is wrong.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            hint.with_extension(ext_str);
        } else {
            // Extension is not valid UTF-8. Symphonia can still probe without
            // a hint, so this is not an error.
        }
    } else {
        // No file extension. Symphonia will probe the raw bytes.
    }

    let probe_result = symphonia::default::get_probe()
        .format(&hint, media_source, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| PodsyncError::DecodeFailed(format!("{}: {}", path.display(), e)))?;

    let mut format_reader = probe_result.format;

    // --- Find the first audio track and its sample rate ---------------------
    let track = format_reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| {
            PodsyncError::DecodeFailed(format!("{}: no audio track found", path.display()))
        })?;

    let track_id = track.id;
    let n_channels = track.codec_params.channels.map_or(1, |ch| ch.count());
    let native_sr = track
        .codec_params
        .sample_rate
        .ok_or_else(|| {
            PodsyncError::DecodeFailed(format!("{}: unknown sample rate", path.display()))
        })?;

    // --- Create a decoder for this track ------------------------------------
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| PodsyncError::DecodeFailed(format!("{}: {}", path.display(), e)))?;

    // --- Decode all packets into interleaved f32 samples --------------------
    // Symphonia decodes packet-by-packet. We accumulate all samples into a
    // flat Vec<f32>. The samples are interleaved: [L0, R0, L1, R1, ...] for
    // stereo, or just [S0, S1, ...] for mono.
    let mut interleaved_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format_reader.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                // End of stream — all packets decoded.
                break;
            }
            Err(e) => {
                return Err(PodsyncError::DecodeFailed(format!(
                    "{}: {}",
                    path.display(),
                    e
                )));
            }
        };

        // Skip packets that belong to other tracks (e.g. video, subtitles).
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(|e| PodsyncError::DecodeFailed(format!("{}: {}", path.display(), e)))?;

        // Convert the decoded audio buffer to interleaved f32 samples.
        let spec = *decoded.spec();
        let num_frames = decoded.frames();
        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        interleaved_samples.extend_from_slice(sample_buf.samples());
    }

    // --- Downmix to mono by averaging channels ------------------------------
    // Match librosa's mono conversion (average channels) so results line up
    // with the reference pipeline.
    let mono_samples = downmix_to_mono(&interleaved_samples, n_channels);

    // --- Resample to target_sr if needed ------------------------------------
    if native_sr == target_sr {
        // No resampling needed — the file is already at the target rate.
        Ok((mono_samples, target_sr))
    } else {
        // Resample from native_sr to target_sr.
        let resampled = resample(&mono_samples, native_sr, target_sr)?;
        Ok((resampled, target_sr))
    }
}

/// Downmix interleaved multi-channel audio to mono by averaging channels.
///
/// For stereo input [L0, R0, L1, R1, ...], output is [(L0+R0)/2, (L1+R1)/2, ...].
/// For mono input, this returns a copy of the input.
fn downmix_to_mono(interleaved: &[f32], n_channels: usize) -> Vec<f32> {
    if n_channels == 1 {
        // Already mono — return a copy.
        return interleaved.to_vec();
    }

    let n_frames = interleaved.len() / n_channels;
    let mut mono = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        // Sum all channels for this frame, then divide by channel count.
        let mut sum: f32 = 0.0;
        for ch in 0..n_channels {
            sum += interleaved[frame_idx * n_channels + ch];
        }
        mono.push(sum / n_channels as f32);
    }

    mono
}

/// Resample mono audio from one sample rate to another using sinc interpolation.
///
/// Uses rubato's async sinc resampler with parameters chosen for quality over
/// speed (sinc_len=256, BlackmanHarris2 window). This matches the quality level
/// of librosa's resampling (which uses scipy's polyphase resampler).
///
/// The `process_all_into_buffer` method handles chunking and flushing internally,
/// so we just pass in the entire input and get back the resampled output.
fn resample(audio: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>, PodsyncError> {
    let ratio = to_sr as f64 / from_sr as f64;
    let n_channels = 1; // Mono — we downmixed before resampling.
    let n_input_frames = audio.len();

    // Sinc interpolation parameters. These control the quality/speed tradeoff:
    // - sinc_len=256: length of the sinc interpolation filter. Longer = better
    //   quality, slower. 256 is high quality, suitable for offline processing.
    // - f_cutoff=0.95: anti-aliasing filter cutoff as a fraction of Nyquist
    //   (half the sample rate).
    //   0.95 preserves most of the bandwidth while preventing aliasing.
    // - oversampling_factor=256: internal upsampling for sinc table lookup.
    //   Higher = more accurate interpolation between sinc table entries.
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // Chunk size for internal processing. 1024 frames per chunk is a reasonable
    // balance between memory usage and processing overhead.
    let chunk_size = 1024;

    let mut resampler = Async::<f64>::new_sinc(
        ratio,
        1.1,        // max_resample_ratio_relative — headroom for ratio adjustments.
                     // We don't adjust dynamically, but rubato requires > 1.0.
        &params,
        chunk_size,
        n_channels,
        FixedAsync::Input,
    )
    .map_err(|e| PodsyncError::ResampleFailed(format!("failed to create resampler: {}", e)))?;

    // Rubato works with f64 internally, so we convert f32 -> f64 for input
    // and f64 -> f32 for output.
    let input_f64: Vec<f64> = audio.iter().map(|&s| s as f64).collect();

    // Allocate output buffer. The expected output length is approximately
    // input_length * ratio, but we allocate 2x to be safe (the resampler
    // may overshoot slightly due to filter ramp-up).
    let output_capacity = (n_input_frames as f64 * ratio * 2.0).ceil() as usize;
    let mut output_f64 = vec![0.0f64; output_capacity];

    // Wrap input/output in audioadapter slices for rubato 1.0's API.
    // InterleavedSlice is the simplest adapter — it treats a flat &[f64] as
    // interleaved multi-channel data (trivial for mono: 1 channel).
    let input_adapter =
        InterleavedSlice::new(&input_f64, n_channels, n_input_frames)
            .map_err(|e| PodsyncError::ResampleFailed(format!("input adapter: {:?}", e)))?;
    let mut output_adapter =
        InterleavedSlice::new_mut(&mut output_f64, n_channels, output_capacity)
            .map_err(|e| PodsyncError::ResampleFailed(format!("output adapter: {:?}", e)))?;

    // Process the entire input in one call. This handles chunking, partial
    // chunks, and flushing the resampler's internal delay buffer.
    let (_frames_in, frames_out) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, n_input_frames, None)
        .map_err(|e| PodsyncError::ResampleFailed(format!("{}", e)))?;

    // Convert back to f32, taking only the valid output frames.
    let output_f32: Vec<f32> = output_f64[..frames_out]
        .iter()
        .map(|&s| s as f32)
        .collect();

    Ok(output_f32)
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write a mono audio buffer to a 24-bit PCM WAV file.
///
/// Creates parent directories if they don't exist.
///
/// The output format is fixed:
/// - 1 channel (mono)
/// - 24-bit signed PCM samples
/// - WAV container (RIFF)
///
/// The format is explicit (not inferred from the file extension) to prevent
/// silent production of non-WAV files.
pub fn write_audio(path: &Path, audio: &[f32], sr: u32) -> Result<(), PodsyncError> {
    // Create parent directories if they don't exist.
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| {
                PodsyncError::WriteFailed(format!("cannot create directory {}: {}", parent.display(), e))
            })?;
        } else {
            // Parent directory already exists — nothing to do.
        }
    } else {
        // No parent directory (writing to root or current dir) — nothing to do.
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| PodsyncError::WriteFailed(format!("{}: {}", path.display(), e)))?;

    for &sample in audio {
        // Convert float32 [-1.0, 1.0] to 24-bit signed integer range.
        // Clamp first to prevent overflow if the audio exceeds [-1.0, 1.0].
        let clamped = sample.clamp(-1.0, 1.0);
        let scaled = (clamped * PCM_24_MAX) as i32;
        writer
            .write_sample(scaled)
            .map_err(|e| PodsyncError::WriteFailed(format!("{}: {}", path.display(), e)))?;
    }

    // finalize() writes the WAV header with the correct data size. We call it
    // explicitly (rather than relying on Drop) so write errors are not silently
    // swallowed.
    writer
        .finalize()
        .map_err(|e| PodsyncError::WriteFailed(format!("{}: {}", path.display(), e)))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Sample manipulation
// ---------------------------------------------------------------------------

/// Shift audio by `offset_samples` and fit into a buffer of `target_length`.
///
/// - **Positive offset**: pad the beginning with silence (track starts after master).
/// - **Negative offset**: trim the beginning (track starts before master).
/// - The result is always exactly `target_length` samples, zero-padded at the
///   end if the shifted audio is shorter.
///
/// Returns an error if the negative offset is larger than the audio length
/// (nothing left after trimming).
pub fn apply_offset(
    audio: &[f32],
    offset_samples: i64,
    target_length: usize,
) -> Result<Vec<f32>, PodsyncError> {
    let mut output = vec![0.0f32; target_length];

    if offset_samples >= 0 {
        let offset = offset_samples as usize;

        if offset >= target_length {
            // Offset is beyond the target length — entire output is silence.
            // This is unusual but not an error; the track simply doesn't
            // overlap with the master's time range.
            return Ok(output);
        } else {
            // Normal case: pad beginning with silence, copy audio into the
            // remaining space.
            let available = target_length - offset;
            let copy_len = audio.len().min(available);
            output[offset..offset + copy_len].copy_from_slice(&audio[..copy_len]);
        }
    } else {
        // Negative offset — trim the beginning of the audio.
        let trim = (-offset_samples) as usize;

        if trim >= audio.len() {
            return Err(PodsyncError::InvalidOffset(format!(
                "negative offset ({} samples) is larger than audio length ({} samples) \
                 — nothing left after trim",
                offset_samples,
                audio.len()
            )));
        } else {
            let remaining = audio.len() - trim;
            let copy_len = remaining.min(target_length);
            output[..copy_len].copy_from_slice(&audio[trim..trim + copy_len]);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate a sine wave at the given frequency, sample rate, and duration.
    /// Returns a Vec<f32> of mono samples in [-1.0, 1.0].
    fn make_sine(frequency_hz: f32, sr: u32, duration_s: f32) -> Vec<f32> {
        let n_samples = (sr as f32 * duration_s) as usize;
        let mut samples = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f32 / sr as f32;
            samples.push((2.0 * PI * frequency_hz * t).sin());
        }
        samples
    }

    // --- seconds_to_samples ------------------------------------------------

    #[test]
    fn test_seconds_to_samples_basic() {
        // 1 second at 44100 Hz = 44100 samples.
        assert_eq!(seconds_to_samples(1.0, 44100), 44100);
    }

    #[test]
    fn test_seconds_to_samples_fractional() {
        // 0.5 seconds at 44100 Hz = 22050 samples.
        assert_eq!(seconds_to_samples(0.5, 44100), 22050);
    }

    #[test]
    fn test_seconds_to_samples_rounds_not_truncates() {
        // 1.9999999 seconds at 44100 Hz should round to 88200, not truncate to 88199.
        // This tests the same edge case the Python code documents.
        let result = seconds_to_samples(1.9999999, 44100);
        assert_eq!(result, 88200);
    }

    #[test]
    fn test_seconds_to_samples_negative() {
        // Negative seconds (track started early) should produce negative samples.
        assert_eq!(seconds_to_samples(-2.0, 44100), -88200);
    }

    // --- write_audio + read back (roundtrip) --------------------------------

    #[test]
    fn test_write_audio_roundtrip_wav() {
        // Write a sine wave to WAV, read it back, and verify the samples match
        // within 24-bit quantization tolerance.
        let sr = 44100;
        let audio = make_sine(440.0, sr, 0.5);

        let temp_dir = tempfile::tempdir().unwrap();
        let wav_path = temp_dir.path().join("test_roundtrip.wav");

        // Write.
        write_audio(&wav_path, &audio, sr).unwrap();

        // Read back with hound.
        let mut reader = hound::WavReader::open(&wav_path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, sr);
        assert_eq!(spec.bits_per_sample, 24);

        // Convert 24-bit integers back to f32 for comparison.
        let read_samples: Vec<f32> = reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / PCM_24_MAX)
            .collect();

        assert_eq!(read_samples.len(), audio.len());

        // 24-bit quantization introduces a maximum error of 1 / (2^23) ≈ 1.2e-7.
        // We use a tolerance of 1e-4 (matching the Python test) which is very
        // conservative — the actual error should be much smaller.
        let tolerance = 1e-4;
        for (i, (&original, &roundtripped)) in audio.iter().zip(read_samples.iter()).enumerate() {
            let diff = (original - roundtripped).abs();
            assert!(
                diff < tolerance,
                "sample {} differs by {} (original={}, roundtripped={})",
                i,
                diff,
                original,
                roundtripped,
            );
        }
    }

    // --- load_audio ---------------------------------------------------------

    #[test]
    fn test_load_audio_resamples_and_downmixes() {
        // Create a 48kHz stereo WAV, load it with target_sr=44100, and verify
        // it comes back as 44.1kHz mono with approximately the same duration.
        let sr_original: u32 = 48000;
        let duration_s: f32 = 1.0;
        let n_samples = (sr_original as f32 * duration_s) as usize;

        // Create stereo interleaved samples: [L0, R0, L1, R1, ...]
        // Both channels are the same 440Hz sine.
        let mono = make_sine(440.0, sr_original, duration_s);
        let mut stereo_interleaved: Vec<f32> = Vec::with_capacity(n_samples * 2);
        for &s in &mono {
            stereo_interleaved.push(s); // left
            stereo_interleaved.push(s); // right
        }

        // Write as a 48kHz stereo WAV using hound.
        let temp_dir = tempfile::tempdir().unwrap();
        let wav_path = temp_dir.path().join("test_48k_stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: sr_original,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for &s in &stereo_interleaved {
            let scaled = (s * 32767.0) as i16;
            writer.write_sample(scaled).unwrap();
        }
        writer.finalize().unwrap();

        // Load with target_sr=44100.
        let (audio, sr) = load_audio(&wav_path, TARGET_SAMPLE_RATE).unwrap();

        // Should be resampled to 44.1kHz.
        assert_eq!(sr, TARGET_SAMPLE_RATE);

        // Duration should be preserved (approximately).
        let loaded_duration = audio.len() as f64 / sr as f64;
        let duration_diff = (loaded_duration - duration_s as f64).abs();
        assert!(
            duration_diff < 0.05,
            "duration differs by {}s (loaded {}s, expected {}s)",
            duration_diff,
            loaded_duration,
            duration_s,
        );
    }

    #[test]
    fn test_load_audio_file_not_found() {
        let result = load_audio(Path::new("/nonexistent/file.mp3"), TARGET_SAMPLE_RATE);
        assert!(result.is_err());
        match result.unwrap_err() {
            PodsyncError::FileNotFound(_) => {} // Expected.
            other => panic!("expected FileNotFound, got: {}", other),
        }
    }

    // --- apply_offset -------------------------------------------------------

    #[test]
    fn test_apply_offset_positive() {
        // Positive offset: 3 samples of silence, then audio.
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = apply_offset(&audio, 3, 10).unwrap();

        assert_eq!(result.len(), 10);
        // First 3 samples should be silence.
        assert_eq!(&result[0..3], &[0.0, 0.0, 0.0]);
        // Then the audio.
        assert_eq!(&result[3..8], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Remaining should be silence.
        assert_eq!(&result[8..10], &[0.0, 0.0]);
    }

    #[test]
    fn test_apply_offset_negative() {
        // Negative offset: trim 2 samples from the beginning.
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = apply_offset(&audio, -2, 10).unwrap();

        assert_eq!(result.len(), 10);
        // Trimmed audio starts at sample index 2.
        assert_eq!(&result[0..3], &[3.0, 4.0, 5.0]);
        // Rest is silence.
        assert_eq!(&result[3..10], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_apply_offset_positive_beyond_target_returns_silence() {
        // Offset is larger than target_length — entire output is silence.
        let audio = vec![1.0, 2.0, 3.0];
        let result = apply_offset(&audio, 100, 10).unwrap();

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_apply_offset_negative_overflow_returns_error() {
        // Trimming more than the audio length is an error.
        let audio = vec![1.0, 2.0, 3.0];
        let result = apply_offset(&audio, -5, 10);

        assert!(result.is_err());
        match result.unwrap_err() {
            PodsyncError::InvalidOffset(_) => {} // Expected.
            other => panic!("expected InvalidOffset, got: {}", other),
        }
    }

    #[test]
    fn test_apply_offset_zero() {
        // Zero offset: audio copied from the start, padded at end.
        let audio = vec![1.0, 2.0, 3.0];
        let result = apply_offset(&audio, 0, 5).unwrap();

        assert_eq!(result, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }
}
