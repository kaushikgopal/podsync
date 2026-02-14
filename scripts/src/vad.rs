// ---------------------------------------------------------------------------
// vad — Voice Activity Detection and speech segment selection
//
// Uses Google's WebRTC VAD (via the webrtc-vad crate) to detect speech regions
// in audio, then selects the best segment for cross-correlation using a
// three-tier fallback strategy:
//
//   Tier 1: Find a single region >= min_duration (ideal: clean, unambiguous)
//   Tier 2: Use the longest single region >= 10s (contiguous preferred)
//   Tier 3: Accumulate consecutive regions within 2s gaps until reaching
//           min_duration (fragmented speech, still usable)
//
// All timestamps are in seconds on the original audio timeline, even when
// the audio is resampled internally for VAD processing.
// ---------------------------------------------------------------------------

use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use audioadapter_buffers::direct::InterleavedSlice;
use webrtc_vad::{Vad, SampleRate, VadMode};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Merge speech regions separated by gaps shorter than this. Typical inter-word
/// pause is ~200ms; 300ms captures brief pauses within sentences while still
/// splitting genuinely separate utterances.
const SPEECH_MERGE_GAP_S: f64 = 0.3;

/// Minimum contiguous speech for a single region to be used directly for
/// correlation. Below this threshold, a single region is too short to produce
/// a reliable cross-correlation peak, so we fall back to accumulating nearby
/// shorter regions.
const MIN_SINGLE_REGION_DURATION_S: f64 = 10.0;

/// Maximum gap between speech regions that still counts as "nearby" for the
/// accumulation fallback. Larger than SPEECH_MERGE_GAP_S because we're more
/// tolerant when assembling a usable segment from fragments.
const ACCUMULATION_GAP_LIMIT_S: f64 = 2.0;

/// How far into the audio to search for speech. 600s = 10 minutes — covers
/// cases where a host joins late or there's a long intro before they speak.
pub const VAD_SEARCH_LIMIT_S: f64 = 600.0;

/// Preferred minimum duration of speech needed for reliable correlation.
/// The cross-correlation needs enough material to produce a clear peak.
/// 30s of speech is sufficient for MFCC-based matching.
pub const PREFERRED_SPEECH_DURATION_S: f64 = 30.0;

/// WebRTC VAD only works with specific sample rates. If the input doesn't
/// match, we resample to 16kHz (good balance of accuracy and speed for VAD).
const WEBRTC_RESAMPLE_TARGET: u32 = 16_000;

/// WebRTC VAD aggressiveness level (0-3). Higher = more aggressive filtering
/// of non-speech. 2 is a good default: filters most background noise without
/// clipping soft speech.
const WEBRTC_AGGRESSIVENESS: VadMode = VadMode::Aggressive; // = 2

/// Frame duration for WebRTC VAD. Must be 10, 20, or 30 ms.
/// 30ms gives the best accuracy per the WebRTC documentation.
const WEBRTC_FRAME_DURATION_MS: u32 = 30;

/// Valid sample rates for WebRTC VAD. If the audio is not one of these rates,
/// we resample to WEBRTC_RESAMPLE_TARGET before running VAD.
const WEBRTC_VALID_RATES: [u32; 4] = [8000, 16000, 32000, 48000];

// ---------------------------------------------------------------------------
// Speech region detection
// ---------------------------------------------------------------------------

/// A speech region: (start_time_seconds, end_time_seconds).
type Region = (f64, f64);

/// Detect speech regions in audio using WebRTC VAD.
///
/// Input: mono audio as f32 samples in [-1.0, 1.0].
/// Output: list of (start_time, end_time) tuples in seconds, sorted by start
/// time. Times are relative to the *original* audio's timeline regardless of
/// any internal resampling.
pub fn detect_speech_regions(audio: &[f32], sr: u32) -> Vec<Region> {
    // --- Resample if needed ------------------------------------------------
    // WebRTC VAD requires one of a few specific sample rates. If the input
    // doesn't match, we resample to 16kHz for VAD processing only.
    let (vad_samples, vad_sr) = if WEBRTC_VALID_RATES.contains(&sr) {
        // Input rate is VAD-compatible — use as-is.
        (audio.to_vec(), sr)
    } else {
        // Resample to 16kHz for VAD. The returned timestamps will be converted
        // back to the original audio's timeline.
        let resampled = resample_for_vad(audio, sr, WEBRTC_RESAMPLE_TARGET);
        (resampled, WEBRTC_RESAMPLE_TARGET)
    };

    // --- Convert to 16-bit PCM --------------------------------------------
    // WebRTC VAD expects raw 16-bit PCM. Clip first to prevent i16 overflow —
    // audio processing can produce values outside [-1, 1].
    let audio_int16: Vec<i16> = vad_samples
        .iter()
        .map(|&s| {
            let clamped = s.clamp(-1.0, 1.0);
            (clamped * i16::MAX as f32) as i16
        })
        .collect();

    let mut vad = Vad::new_with_rate_and_mode(
        sample_rate_to_enum(vad_sr),
        WEBRTC_AGGRESSIVENESS,
    );

    let frame_size = (vad_sr * WEBRTC_FRAME_DURATION_MS / 1000) as usize;

    // --- Frame-by-frame VAD ------------------------------------------------
    let mut regions: Vec<Region> = Vec::new();
    let mut is_speech = false;
    let mut speech_start = 0.0;
    let mut position = 0;

    while position + frame_size <= audio_int16.len() {
        let frame = &audio_int16[position..position + frame_size];

        let frame_is_speech = vad.is_voice_segment(frame).unwrap_or_default();

        // Convert frame position to time in the VAD sample rate's timeline.
        let current_time = position as f64 / vad_sr as f64;

        if frame_is_speech && !is_speech {
            // Transition: silence → speech.
            speech_start = current_time;
            is_speech = true;
        } else if !frame_is_speech && is_speech {
            // Transition: speech → silence.
            regions.push((speech_start, current_time));
            is_speech = false;
        } else {
            // No transition — either still in speech or still in silence.
        }

        position += frame_size;
    }

    // Handle speech that extends to the end of the audio.
    if is_speech {
        regions.push((speech_start, audio_int16.len() as f64 / vad_sr as f64));
    } else {
        // Audio ended during silence — nothing to append.
    }

    // --- Merge nearby regions ----------------------------------------------
    merge_regions(&regions, SPEECH_MERGE_GAP_S)
}

/// Merge speech regions that are separated by less than `gap_threshold` seconds.
///
/// Walks regions chronologically. If the gap between two consecutive regions
/// is small enough, they are merged into one. Otherwise, a new region starts.
fn merge_regions(regions: &[Region], gap_threshold: f64) -> Vec<Region> {
    if regions.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<Region> = vec![regions[0]];

    for &(start, end) in &regions[1..] {
        let (prev_start, prev_end) = *merged.last().unwrap();

        if start - prev_end <= gap_threshold {
            // Gap is small enough — extend the previous region.
            *merged.last_mut().unwrap() = (prev_start, end);
        } else {
            // Gap is too large — start a new region.
            merged.push((start, end));
        }
    }

    merged
}

// ---------------------------------------------------------------------------
// Speech segment selection
// ---------------------------------------------------------------------------

/// Find a speech segment suitable for cross-correlation.
///
/// Uses a three-tier strategy, from most to least reliable:
///
/// 1. **Single region >= min_duration**: A long contiguous speech region is
///    ideal for correlation — it gives a clean, unambiguous peak.
///
/// 2. **Longest single region >= MIN_SINGLE_REGION_DURATION_S**: If no single
///    region meets min_duration but one is at least 10s, use it. A shorter
///    but contiguous region is more reliable than stitching fragments.
///
/// 3. **Accumulated nearby regions**: As a last resort, accumulate consecutive
///    regions within ACCUMULATION_GAP_LIMIT_S of each other until we reach
///    min_duration.
///
/// Returns (start_time, end_time), or None if no speech was found at all.
pub fn find_first_speech_segment(
    audio: &[f32],
    sr: u32,
    min_duration: f64,
    search_limit: f64,
) -> Option<Region> {
    // Limit search scope.
    let max_samples = (search_limit * sr as f64) as usize;
    let limited_len = audio.len().min(max_samples);
    let audio_limited = &audio[..limited_len];

    let regions = detect_speech_regions(audio_limited, sr);

    if regions.is_empty() {
        return None;
    }

    // ------------------------------------------------------------------
    // Tier 1: Find a single region that meets the preferred duration.
    // Sort by duration (longest first) so we pick the best candidate.
    // ------------------------------------------------------------------
    let mut regions_by_length = regions.clone();
    regions_by_length.sort_by(|a, b| {
        let dur_a = a.1 - a.0;
        let dur_b = b.1 - b.0;
        dur_b.partial_cmp(&dur_a).unwrap()
    });

    for &(start, end) in &regions_by_length {
        let duration = end - start;
        if duration >= min_duration {
            return Some((start, end));
        } else {
            // This region is too short — try the next one.
        }
    }

    // ------------------------------------------------------------------
    // Tier 2: Use the longest single region if it's at least 10s.
    // A shorter but contiguous region produces a cleaner correlation peak
    // than a sparse accumulation of tiny fragments with gaps.
    // ------------------------------------------------------------------
    let (longest_start, longest_end) = regions_by_length[0];
    let longest_duration = longest_end - longest_start;

    if longest_duration >= MIN_SINGLE_REGION_DURATION_S {
        return Some((longest_start, longest_end));
    } else {
        // Longest region is very short — fall through to accumulation.
    }

    // ------------------------------------------------------------------
    // Tier 3: Accumulate consecutive regions that are close together.
    // Walk regions in chronological order. Reset accumulation when a gap
    // exceeds ACCUMULATION_GAP_LIMIT_S.
    // ------------------------------------------------------------------
    let mut total_speech = 0.0;
    let mut segment_start: Option<f64> = None;
    let mut prev_end: Option<f64> = None;

    for &(start, end) in &regions {
        if let Some(pe) = prev_end {
            if (start - pe) > ACCUMULATION_GAP_LIMIT_S {
                // Gap too large — the previous cluster wasn't enough. Reset.
                total_speech = 0.0;
                segment_start = None;
            } else {
                // Gap is small enough — keep accumulating.
            }
        } else {
            // First region — start accumulating.
        }

        if segment_start.is_none() {
            segment_start = Some(start);
        } else {
            // Continuing accumulation from an earlier region.
        }

        total_speech += end - start;
        prev_end = Some(end);

        if total_speech >= min_duration {
            return Some((segment_start.unwrap(), end));
        } else {
            // Haven't accumulated enough yet — continue to next region.
        }
    }

    // ------------------------------------------------------------------
    // Fallback: Return whatever we accumulated, even if it's shorter than
    // min_duration. Some speech is better than none — the caller can check
    // correlation confidence to decide if the result is trustworthy.
    // ------------------------------------------------------------------
    if let (Some(seg_start), Some(pe)) = (segment_start, prev_end) {
        if total_speech > 0.0 {
            return Some((seg_start, pe));
        } else {
            // No speech accumulated at all.
        }
    } else {
        // No regions were processed.
    }

    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a u32 sample rate to the webrtc_vad SampleRate enum.
fn sample_rate_to_enum(sr: u32) -> SampleRate {
    match sr {
        8000 => SampleRate::Rate8kHz,
        16000 => SampleRate::Rate16kHz,
        32000 => SampleRate::Rate32kHz,
        48000 => SampleRate::Rate48kHz,
        _ => panic!("Invalid VAD sample rate: {}Hz (must be 8k, 16k, 32k, or 48k)", sr),
    }
}

/// Resample audio for VAD processing. Uses rubato with moderate quality
/// settings — VAD doesn't need audiophile-grade resampling.
fn resample_for_vad(audio: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    let ratio = to_sr as f64 / from_sr as f64;
    let n_input_frames = audio.len();

    // Lower quality than the main audio resampler — VAD just needs speech/non-speech
    // discrimination, not sample-perfect reconstruction.
    let params = SincInterpolationParameters {
        sinc_len: 64,
        f_cutoff: 0.90,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = Async::<f64>::new_sinc(
        ratio,
        1.1,
        &params,
        1024,
        1, // mono
        FixedAsync::Input,
    )
    .expect("failed to create VAD resampler");

    let input_f64: Vec<f64> = audio.iter().map(|&s| s as f64).collect();
    let output_capacity = (n_input_frames as f64 * ratio * 2.0).ceil() as usize;
    let mut output_f64 = vec![0.0f64; output_capacity];

    let input_adapter = InterleavedSlice::new(&input_f64, 1, n_input_frames).unwrap();
    let mut output_adapter = InterleavedSlice::new_mut(&mut output_f64, 1, output_capacity).unwrap();

    let (_frames_in, frames_out) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, n_input_frames, None)
        .expect("VAD resampling failed");

    output_f64[..frames_out].iter().map(|&s| s as f32).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate pseudo-random noise at the given sample rate and duration.
    /// Uses a deterministic LCG so tests are reproducible.
    fn make_noise(sr: u32, duration_s: f32, seed: u64) -> Vec<f32> {
        let n_samples = (sr as f32 * duration_s) as usize;
        let mut samples = Vec::with_capacity(n_samples);
        let mut state = seed;
        for _ in 0..n_samples {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let value = ((state >> 33) as f32 / u32::MAX as f32 - 0.5) * 1.0;
            samples.push(value);
        }
        samples
    }

    #[test]
    fn test_detects_speech_in_simple_signal() {
        // Signal: 1s silence, 2s "speech" (noise), 1s silence, 1s "speech"
        let sr: u32 = 16000;
        let silence: Vec<f32> = vec![0.0; sr as usize];
        let speech = make_noise(sr, 1.0, 42);

        let mut audio = Vec::new();
        audio.extend_from_slice(&silence);       // 0-1s: silence
        audio.extend_from_slice(&speech);         // 1-2s: speech
        audio.extend_from_slice(&speech);         // 2-3s: speech
        audio.extend_from_slice(&silence);        // 3-4s: silence
        audio.extend_from_slice(&speech);         // 4-5s: speech

        let regions = detect_speech_regions(&audio, sr);

        // Should detect at least 1 speech region.
        assert!(
            !regions.is_empty(),
            "should detect at least one speech region"
        );
        // First region should start around 1s.
        assert!(
            regions[0].0 >= 0.8 && regions[0].0 <= 1.5,
            "first region should start around 1s, got {}",
            regions[0].0
        );
    }

    #[test]
    fn test_returns_empty_for_silence() {
        let sr: u32 = 16000;
        let silence: Vec<f32> = vec![0.0; sr as usize * 5];

        let regions = detect_speech_regions(&silence, sr);

        assert!(regions.is_empty(), "silence should produce no regions");
    }

    #[test]
    fn test_finds_segment_with_minimum_duration() {
        let sr: u32 = 16000;

        let silence: Vec<f32> = vec![0.0; sr as usize];
        let short_speech = make_noise(sr, 0.5, 42);
        let long_speech = make_noise(sr, 3.0, 42);
        let half_silence: Vec<f32> = vec![0.0; sr as usize / 2];

        let mut audio = Vec::new();
        audio.extend_from_slice(&silence);        // 0-1s: silence
        audio.extend_from_slice(&short_speech);   // 1-1.5s: short speech
        audio.extend_from_slice(&half_silence);   // 1.5-2s: silence
        audio.extend_from_slice(&long_speech);    // 2-5s: long speech

        let result = find_first_speech_segment(&audio, sr, 2.0, 600.0);

        assert!(result.is_some(), "should find a speech segment");
        let (start, end) = result.unwrap();
        assert!(start >= 1.5, "segment should start after the short speech, got {}", start);
        assert!(end - start >= 2.0, "segment should be >= 2s, got {}s", end - start);
    }

    #[test]
    fn test_returns_none_for_pure_silence() {
        let sr: u32 = 16000;
        let audio: Vec<f32> = vec![0.0; sr as usize * 2];

        let result = find_first_speech_segment(&audio, sr, 30.0, 600.0);

        assert!(result.is_none(), "pure silence should return None");
    }

    #[test]
    fn test_merge_regions_basic() {
        // Two regions separated by 0.2s (below threshold) should merge.
        let regions = vec![(1.0, 2.0), (2.2, 3.0)];
        let merged = merge_regions(&regions, 0.3);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0], (1.0, 3.0));

        // Two regions separated by 1.0s (above threshold) should not merge.
        let regions = vec![(1.0, 2.0), (3.0, 4.0)];
        let merged = merge_regions(&regions, 0.3);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_regions_empty() {
        let merged = merge_regions(&[], 0.3);
        assert!(merged.is_empty());
    }
}
