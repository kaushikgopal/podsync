// ---------------------------------------------------------------------------
// sync — MFCC cross-correlation for offset detection and drift measurement
//
// Finds the time offset between a track and a master recording by:
//   1. Extracting MFCC features from both
//   2. Cross-correlating each MFCC coefficient independently
//   3. Summing the correlations and finding the peak
//   4. Converting the peak position to a time offset in seconds
//
// Also measures clock drift by comparing alignment at the end of the
// recording versus the start.
// ---------------------------------------------------------------------------

use realfft::RealFftPlanner;

use crate::audio::seconds_to_samples;
use crate::mfcc::extract_mfcc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of MFCC coefficients to extract. 20 captures the vocal tract shape
/// (formants) without introducing high-frequency noise. Standard in speech
/// processing — Kaldi uses 13, librosa defaults to 20.
const N_MFCC_COEFFICIENTS: usize = 20;

/// STFT hop length in samples. At 44.1kHz this is ~11.6ms per frame — the
/// standard resolution for speech analysis. Smaller values increase precision
/// but also computation time quadratically (via cross-correlation).
pub const HOP_LENGTH: usize = 512;

/// Small epsilon to prevent division by zero in normalization and confidence
/// calculations. 1e-6 is appropriate for f32 audio data (the Python code uses
/// 1e-8 for float64, and explicitly notes to use 1e-6 when porting to f32).
const EPSILON: f64 = 1e-6;

/// When searching for the second-best correlation peak (to compute confidence),
/// exclude this many frames around the best peak. At HOP_LENGTH=512 and
/// sr=44100, 50 frames ≈ 0.58 seconds. This prevents sidelobes of the main
/// peak from being counted as an independent match.
const PEAK_EXCLUSION_RADIUS: usize = 50;

/// Confidence threshold: below this, the correlation peak isn't distinct enough
/// to trust. A ratio of 1.0 means the best and second-best peaks are equal
/// (ambiguous). The formula maps ratio 1.0 → confidence 0.0, ratio 1.5+ →
/// confidence 1.0, linearly between. The 0.5 divisor controls sensitivity.
const CONFIDENCE_SENSITIVITY: f64 = 0.5;

/// Below this confidence, warn the user that the sync may be inaccurate.
pub const LOW_CONFIDENCE_THRESHOLD: f64 = 0.5;

// ---------------------------------------------------------------------------
// Cross-correlation (FFT-based)
// ---------------------------------------------------------------------------

/// Cross-correlate two real signals (equivalent to scipy.signal.correlate mode='full').
///
/// Output length is a.len() + b.len() - 1.
///
/// The zero-lag point (where the signals are fully overlapping) is at index
/// b.len() - 1. A peak to the right of zero-lag means signal `a` contains the
/// content of `b` at a later position.
///
/// Implementation: FFT-based correlation via the convolution theorem.
///   correlate(a, b) = IFFT(FFT(a) * conj(FFT(b)))
/// padded to the next power of two for FFT efficiency.
fn correlate_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    let output_len = a.len() + b.len() - 1;

    // Pad to next power of two for FFT efficiency.
    let fft_len = output_len.next_power_of_two();

    let mut planner = RealFftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(fft_len);
    let inverse = planner.plan_fft_inverse(fft_len);

    // --- Zero-pad inputs to fft_len ----------------------------------------
    let mut a_padded = forward.make_input_vec();
    for (i, &val) in a.iter().enumerate() {
        a_padded[i] = val;
    }

    let mut b_padded = forward.make_input_vec();
    for (i, &val) in b.iter().enumerate() {
        b_padded[i] = val;
    }

    // --- Forward FFT both signals ------------------------------------------
    let mut a_spectrum = forward.make_output_vec();
    let mut b_spectrum = forward.make_output_vec();
    forward.process(&mut a_padded, &mut a_spectrum).unwrap();
    forward.process(&mut b_padded, &mut b_spectrum).unwrap();

    // --- Multiply A * conj(B) in frequency domain --------------------------
    // This implements cross-correlation via the convolution theorem.
    for (a_val, b_val) in a_spectrum.iter_mut().zip(b_spectrum.iter()) {
        let re = a_val.re * b_val.re + a_val.im * b_val.im;
        let im = a_val.im * b_val.re - a_val.re * b_val.im;
        a_val.re = re;
        a_val.im = im;
    }

    // --- Inverse FFT -------------------------------------------------------
    let mut result = inverse.make_output_vec();
    inverse.process(&mut a_spectrum, &mut result).unwrap();

    // realfft does NOT normalize the inverse FFT — we must divide by fft_len.
    let scale = 1.0 / fft_len as f64;
    for val in &mut result {
        *val *= scale;
    }

    // --- Rearrange to match scipy correlate mode='full' --------------------
    // The IFFT produces circular cross-correlation with zero-lag at index 0:
    //   result[0]            = zero lag
    //   result[1..a_len]     = positive lags (a leads b)
    //   result[N-b_len+1..]  = negative lags (b leads a)
    //
    // scipy's 'full' mode arranges output as:
    //   output[0..b_len-1]          = negative lags (lag -(b_len-1) to -1)
    //   output[b_len-1]             = zero lag
    //   output[b_len..output_len]   = positive lags (lag 1 to a_len-1)
    //
    // We rearrange by pulling negative lags from the end of the circular
    // buffer and placing them before the zero-lag and positive-lag values.
    let a_len = a.len();
    let b_len = b.len();
    let mut output = Vec::with_capacity(output_len);

    // Negative lags: these wrap around to the end of the circular buffer.
    for i in 0..(b_len - 1) {
        output.push(result[fft_len - b_len + 1 + i]);
    }

    // Zero lag and positive lags: these sit at the start of the circular buffer.
    for &val in result.iter().take(a_len) {
        output.push(val);
    }

    output
}

// ---------------------------------------------------------------------------
// Offset detection
// ---------------------------------------------------------------------------

/// Find the time offset of *track* relative to *master* using MFCC cross-correlation.
///
/// Returns (offset_seconds, confidence) where:
/// - Positive offset: track content appears later in master (track started late).
/// - Negative offset: track content appears earlier (track started early).
/// - Confidence: 0.0 (ambiguous) to 1.0 (clear match).
pub fn find_offset(
    master: &[f32],
    track: &[f32],
    sr: u32,
    search_window: f64,
    correlation_window: f64,
) -> (f64, f64) {
    let hop_length = HOP_LENGTH;

    // --- Limit inputs to relevant windows ----------------------------------
    let master_samples = seconds_to_samples(search_window, sr) as usize;
    let master_limited = if master.len() > master_samples {
        &master[..master_samples]
    } else {
        master
    };

    let track_samples = seconds_to_samples(correlation_window, sr) as usize;
    let track_limited = if track.len() > track_samples {
        &track[..track_samples]
    } else {
        track
    };

    // --- Extract MFCCs -----------------------------------------------------
    let mfcc_master = extract_mfcc(master_limited, sr, N_MFCC_COEFFICIENTS, hop_length);
    let mfcc_track = extract_mfcc(track_limited, sr, N_MFCC_COEFFICIENTS, hop_length);

    let n_coeffs = mfcc_master.len();

    // --- Cross-correlate each coefficient independently, then sum ----------
    // Correlating per-coefficient preserves spectral discrimination. If we
    // flattened first, high-energy coefficients would dominate and subtle
    // spectral differences (like distinguishing two speakers) would be lost.
    let mut correlation: Option<Vec<f64>> = None;

    for i in 0..n_coeffs {
        let m = &mfcc_master[i];
        let t = &mfcc_track[i];

        // Normalize each coefficient's time series to zero mean, unit variance.
        // This ensures all coefficients contribute equally to the sum.
        // Use f64 accumulators for numerical stability.
        let m_mean: f64 = m.iter().sum::<f64>() / m.len() as f64;
        let m_std: f64 = {
            let variance = m.iter().map(|&v| (v - m_mean) * (v - m_mean)).sum::<f64>() / m.len() as f64;
            variance.sqrt()
        };
        let m_norm: Vec<f64> = m.iter().map(|&v| (v - m_mean) / (m_std + EPSILON)).collect();

        let t_mean: f64 = t.iter().sum::<f64>() / t.len() as f64;
        let t_std: f64 = {
            let variance = t.iter().map(|&v| (v - t_mean) * (v - t_mean)).sum::<f64>() / t.len() as f64;
            variance.sqrt()
        };
        let t_norm: Vec<f64> = t.iter().map(|&v| (v - t_mean) / (t_std + EPSILON)).collect();

        let c = correlate_full(&m_norm, &t_norm);

        match &mut correlation {
            None => {
                correlation = Some(c);
            }
            Some(existing) => {
                for (j, val) in c.iter().enumerate() {
                    existing[j] += val;
                }
            }
        }
    }

    let correlation = correlation.expect("at least one MFCC coefficient should exist");

    // --- Find primary peak -------------------------------------------------
    let (peak_idx, peak_value) = correlation
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();

    // --- Compute confidence ------------------------------------------------
    // Find the second-best peak (excluding a neighborhood around the best)
    // to determine how distinct our match is.
    let exclude_start = peak_idx.saturating_sub(PEAK_EXCLUSION_RADIUS);
    let exclude_end = (peak_idx + PEAK_EXCLUSION_RADIUS + 1).min(correlation.len());

    let second_peak_value = correlation
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx < exclude_start || *idx >= exclude_end)
        .map(|(_, &val)| val)
        .fold(f64::NEG_INFINITY, f64::max);

    let peak_ratio = if second_peak_value > 0.0 {
        peak_value / (second_peak_value + EPSILON)
    } else if *peak_value > 0.0 {
        // Primary peak is positive but there's no meaningful second peak.
        10.0
    } else {
        // Both peaks are zero or negative — no meaningful correlation.
        1.0
    };

    // Map ratio to [0, 1]: ratio 1.0 → 0.0 (peaks are equal, ambiguous),
    // ratio 1.0 + CONFIDENCE_SENSITIVITY → 1.0 (clear winner).
    let confidence = ((peak_ratio - 1.0) / CONFIDENCE_SENSITIVITY).clamp(0.0, 1.0);

    // --- Convert peak index to time offset ---------------------------------
    // In correlate mode='full', the zero-lag point is at
    // len(shorter_signal) - 1. A peak to the right of zero-lag means
    // the track content appears later in the master.
    let zero_lag_idx = mfcc_track[0].len() - 1;
    let lag_frames = peak_idx as i64 - zero_lag_idx as i64;
    let offset_seconds = lag_frames as f64 * hop_length as f64 / sr as f64;

    (offset_seconds, confidence)
}

// ---------------------------------------------------------------------------
// Drift measurement
// ---------------------------------------------------------------------------

/// Compute clock drift by comparing alignment at end vs start.
///
/// If the track's recording device has a slightly different clock rate than
/// the master's, the offset at the end of the recording will differ from the
/// offset at the start. This difference is drift.
///
/// Returns drift in seconds, or None if the audio is too short to measure.
/// - Positive drift: track ran faster (audio compressed over time).
/// - Negative drift: track ran slower (audio stretched over time).
pub fn compute_drift(
    master: &[f32],
    track: &[f32],
    sr: u32,
    initial_offset: f64,
    end_window: f64,
) -> Option<f64> {
    let end_samples = seconds_to_samples(end_window, sr) as usize;

    // Need at least 2x the end window — otherwise the "end" overlaps with
    // the "start" and drift measurement is meaningless.
    if master.len() < end_samples * 2 {
        return None;
    }

    let master_end = &master[master.len() - end_samples..];

    // Calculate where in the track corresponds to the master's end region.
    // master[M] corresponds to track[M - initial_offset_in_samples].
    let track_end_start = (master.len() as f64 - end_samples as f64 - initial_offset * sr as f64) as i64;

    if track_end_start < 0 {
        // Track doesn't extend far enough back to cover master's end.
        return None;
    } else if (track_end_start as usize) + end_samples > track.len() {
        // Track is shorter than master at the end.
        return None;
    } else {
        // Track covers the expected region — proceed.
    }

    let track_end_start = track_end_start as usize;
    let track_end = &track[track_end_start..track_end_start + end_samples];

    // Use half the end window (capped at 60s, floored at 1s) for drift
    // correlation — we already know approximately where the track should be.
    let (end_offset, _confidence) = find_offset(
        master_end,
        track_end,
        sr,
        end_window,
        (end_window.min(60.0) / 2.0).max(1.0),
    );

    // end_offset should be ~0 if no drift (since we positioned track_end
    // based on initial_offset). Any deviation is drift.
    let drift = end_offset;

    Some(drift)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Create a spectrally rich synthetic signal suitable for MFCC correlation.
    ///
    /// A simple sine wave produces flat MFCCs (no spectral variation over time),
    /// which makes cross-correlation unreliable. This combines frequency sweeps
    /// and noise to give the MFCCs meaningful temporal structure.
    ///
    /// Mirrors the Python `_make_rich_signal` test helper.
    fn make_rich_signal(duration: f32, sr: u32, seed: u64) -> Vec<f32> {
        let n_samples = (sr as f32 * duration) as usize;
        let mut samples = Vec::with_capacity(n_samples);
        let mut rng_state = seed;
        for i in 0..n_samples {
            let t = i as f32 / sr as f32;
            let sweep1 = (2.0 * PI * (200.0 + 50.0 * t) * t).sin();
            let sweep2 = 0.5 * (2.0 * PI * (800.0 + 30.0 * t) * t).sin();
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((rng_state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2;
            samples.push(sweep1 + sweep2 + noise);
        }
        samples
    }

    /// Naive O(n²) cross-correlation for testing the FFT-based implementation.
    /// correlate(a, b, mode='full') computes: out[k] = sum_n a[n] * b[n - k + (len(b)-1)]
    fn naive_correlate_full(a: &[f64], b: &[f64]) -> Vec<f64> {
        let output_len = a.len() + b.len() - 1;
        let mut result = vec![0.0; output_len];
        for k in 0..output_len {
            let mut sum = 0.0;
            for n in 0..a.len() {
                let b_idx = n as i64 - k as i64 + b.len() as i64 - 1;
                if b_idx >= 0 && (b_idx as usize) < b.len() {
                    sum += a[n] * b[b_idx as usize];
                } else {
                    // b index is out of range — this element contributes zero.
                }
            }
            result[k] = sum;
        }
        result
    }

    #[test]
    fn test_correlate_matches_naive_small() {
        // Verify that the FFT-based correlate matches the naive implementation.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 1.0];

        let fft_result = correlate_full(&a, &b);
        let naive_result = naive_correlate_full(&a, &b);

        assert_eq!(fft_result.len(), naive_result.len());
        for (i, (&fft_val, &naive_val)) in fft_result.iter().zip(naive_result.iter()).enumerate() {
            let diff = (fft_val - naive_val).abs();
            assert!(
                diff < 1e-8,
                "index {} differs: fft={} naive={} diff={}",
                i, fft_val, naive_val, diff
            );
        }
    }

    #[test]
    fn test_finds_positive_offset() {
        // When track content appears later in master, offset is positive.
        //
        // Master: [silence...][content...]
        // Track:  [content...]
        //
        // find_offset should report that the content appears at +2s in master.
        let sr: u32 = 44100;
        let content_duration = 10.0;
        let master_duration = 15.0;
        let offset_seconds: f64 = 2.0;

        let content = make_rich_signal(content_duration, sr, 42);

        // Build master: silence then content.
        let offset_samples = (offset_seconds * sr as f64) as usize;
        let master_len = (master_duration * sr as f64) as usize;
        let mut master = vec![0.0f32; master_len];
        let copy_len = content.len().min(master_len - offset_samples);
        master[offset_samples..offset_samples + copy_len].copy_from_slice(&content[..copy_len]);

        // Track is just the content (like after VAD extraction).
        let track = content;

        let (found_offset, confidence) = find_offset(
            &master,
            &track,
            sr,
            master_duration,
            content_duration as f64,
        );

        assert!(
            (found_offset - offset_seconds).abs() < 0.2,
            "offset should be ~{}s, got {}s",
            offset_seconds, found_offset
        );
        assert!(confidence > 0.3, "confidence should be > 0.3, got {}", confidence);
    }

    #[test]
    fn test_finds_near_zero_offset() {
        // When track and master start with the same content, offset is ~0.
        let sr: u32 = 44100;
        let duration = 10.0;

        let signal = make_rich_signal(duration, sr, 42);

        let (found_offset, confidence) = find_offset(
            &signal,
            &signal,
            sr,
            duration as f64,
            5.0,
        );

        assert!(
            found_offset.abs() < 0.2,
            "offset should be ~0s, got {}s",
            found_offset
        );
        assert!(confidence > 0.3, "confidence should be > 0.3, got {}", confidence);
    }

    #[test]
    fn test_drift_returns_none_for_short_audio() {
        // Drift measurement requires audio longer than 2x the end window.
        let sr: u32 = 44100;
        let duration = 10.0;
        let n_samples = (sr as f64 * duration) as usize;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let drift = compute_drift(&audio, &audio, sr, 0.0, 120.0);

        assert!(drift.is_none(), "short audio should return None for drift");
    }

    #[test]
    fn test_drift_near_zero_for_same_signal() {
        // Same signal correlated against itself should have near-zero drift.
        let sr: u32 = 44100;
        let duration = 300.0; // 5 minutes — long enough for drift measurement

        let audio = make_rich_signal(duration, sr, 42);

        let drift = compute_drift(&audio, &audio, sr, 0.0, 120.0);

        assert!(drift.is_some(), "should be able to measure drift");
        assert!(
            drift.unwrap().abs() < 0.5,
            "drift should be ~0s, got {}s",
            drift.unwrap()
        );
    }
}
