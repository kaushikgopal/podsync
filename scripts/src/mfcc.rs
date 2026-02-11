// ---------------------------------------------------------------------------
// mfcc — Mel-Frequency Cepstral Coefficient extraction
//
// Reimplements the MFCC pipeline from librosa 0.11.0 for parity with the
// Python version. The pipeline is:
//
//   audio samples
//     -> center-pad with zeros (n_fft / 2 on each side)
//     -> frame into overlapping windows (n_fft=2048, hop_length=512)
//     -> apply Hann window to each frame
//     -> FFT each frame -> power spectrum |X(f)|^2
//     -> apply mel filterbank (128 triangular filters, Slaney normalization)
//     -> convert power to decibels (ref=1.0, amin=1e-10, top_db=80.0)
//     -> DCT type II (orthonormal) -> keep first n_mfcc coefficients
//
// Each step is a separate function so the pipeline is easy to follow and test.
// ---------------------------------------------------------------------------

use std::f64::consts::PI;

use realfft::RealFftPlanner;

// ---------------------------------------------------------------------------
// Constants — librosa 0.11.0 defaults
// ---------------------------------------------------------------------------

/// FFT window size. librosa default for `melspectrogram` and `stft`.
/// 2048 samples at 44.1kHz ≈ 46ms, which captures enough low-frequency detail
/// for speech while keeping temporal resolution reasonable.
const DEFAULT_N_FFT: usize = 2048;

/// Number of mel filterbank bands. librosa default.
/// 128 gives good spectral resolution across the full frequency range.
const DEFAULT_N_MELS: usize = 128;

/// Minimum floor value for power-to-dB conversion. Prevents log(0).
/// Matches librosa's `power_to_db(amin=1e-10)` default.
const POWER_TO_DB_AMIN: f64 = 1e-10;

/// Maximum dynamic range in decibels. Values more than this many dB below the
/// peak are clipped to (peak - top_db). Matches librosa default.
const POWER_TO_DB_TOP_DB: f64 = 80.0;

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------

/// Create a periodic Hann window of the given size.
///
/// The periodic variant (as opposed to symmetric) is standard for STFT because
/// it has better frequency-domain properties for overlapping frames. This
/// matches `scipy.signal.get_window('hann', n_fft)` and librosa's default.
///
/// Formula: w[n] = 0.5 * (1 - cos(2*pi*n / size))
fn create_hann_window(size: usize) -> Vec<f64> {
    let mut window = Vec::with_capacity(size);
    for n in 0..size {
        let value = 0.5 * (1.0 - (2.0 * PI * n as f64 / size as f64).cos());
        window.push(value);
    }
    window
}

// ---------------------------------------------------------------------------
// STFT power spectrum
// ---------------------------------------------------------------------------

/// Compute the power spectrum of audio using the Short-Time Fourier Transform.
///
/// Matches librosa's `np.abs(librosa.stft(y, ...))**2` with these defaults:
///   - center=True: pad audio with n_fft/2 zeros on each side before framing
///   - window='hann': Hann window applied to each frame
///   - power=2.0: return squared magnitude (power spectrum)
///
/// Returns a Vec of frames, where each frame is a Vec<f64> of length
/// (n_fft/2 + 1) — the positive-frequency bins of the FFT.
fn stft_power(audio: &[f32], n_fft: usize, hop_length: usize) -> Vec<Vec<f64>> {
    // --- Center-pad the audio -----------------------------------------------
    // librosa pads with n_fft/2 zeros on each side so that the first frame is
    // centered on the first sample. This affects the number of output frames
    // and must be matched for parity.
    let pad_length = n_fft / 2;
    let padded_len = pad_length + audio.len() + pad_length;
    let mut padded = vec![0.0f64; padded_len];
    for (i, &sample) in audio.iter().enumerate() {
        padded[pad_length + i] = sample as f64;
    }

    // --- Create Hann window and FFT planner ---------------------------------
    let window = create_hann_window(n_fft);
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Number of frequency bins in the output (positive frequencies only).
    let n_bins = n_fft / 2 + 1;

    // --- Frame and transform ------------------------------------------------
    // Slide a window of n_fft samples with hop_length stride.
    let n_frames = (padded_len - n_fft) / hop_length + 1;
    let mut power_frames = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;

        // Extract frame and apply Hann window.
        let mut windowed_frame = fft.make_input_vec();
        for i in 0..n_fft {
            windowed_frame[i] = padded[start + i] * window[i];
        }

        // Forward FFT.
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut windowed_frame, &mut spectrum).unwrap();

        // Compute power spectrum: |X(f)|^2 = re^2 + im^2
        let mut power = Vec::with_capacity(n_bins);
        for complex_val in &spectrum {
            let magnitude_squared = complex_val.re * complex_val.re + complex_val.im * complex_val.im;
            power.push(magnitude_squared);
        }

        power_frames.push(power);
    }

    power_frames
}

// ---------------------------------------------------------------------------
// Mel filterbank
// ---------------------------------------------------------------------------

/// Convert frequency in Hz to the mel scale.
///
/// Uses the Slaney formula (linear below 1000 Hz, logarithmic above), which
/// matches librosa's default (htk=False).
///
/// The mel scale approximates human pitch perception: equal distances in mel
/// correspond to equal perceived pitch differences.
fn hz_to_mel(hz: f64) -> f64 {
    // Slaney's mel scale has two regions:
    //   Linear: below 1000 Hz, where f_sp = 200/3 Hz per mel
    //   Logarithmic: above 1000 Hz, where logstep = ln(6.4) / 27 per mel
    //
    // The break point is 1000 Hz = 15.0 mel.
    let f_sp = 200.0 / 3.0;           // Hz per mel in the linear region
    let min_log_hz = 1000.0;           // start of the log region
    let min_log_mel = min_log_hz / f_sp; // = 15.0 mel
    let logstep = 6.4_f64.ln() / 27.0;  // log Hz step per mel above 1000 Hz

    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

/// Convert mel scale value back to Hz.
///
/// Inverse of hz_to_mel. Uses the Slaney formula (htk=False).
fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;             // Hz per mel in the linear region
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // = 15.0 mel
    let logstep = 6.4_f64.ln() / 27.0;

    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    }
}

/// Create a mel filterbank matrix with Slaney normalization.
///
/// Produces `n_mels` triangular filters spanning from `fmin` to `fmax` Hz.
/// Each filter is a triangle centered on its mel-spaced frequency, with
/// vertices at the centers of the adjacent filters.
///
/// Slaney normalization divides each filter by the width of its mel band
/// (in Hz), so that each filter has equal energy. This prevents low-frequency
/// filters (which are narrow in Hz) from having disproportionately high peaks
/// compared to high-frequency filters (which are wide in Hz).
///
/// Returns a Vec of n_mels filters, where each filter is a Vec<f64> of length
/// n_fft/2 + 1 (one weight per frequency bin).
fn create_mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;

    // Create n_mels + 2 equally-spaced points on the mel scale.
    // The +2 gives us the left edge of the first filter and the right edge
    // of the last filter.
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let n_points = n_mels + 2;
    let mut mel_points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64;
        mel_points.push(mel);
    }

    // Convert mel points back to Hz, then to FFT bin indices.
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_indices: Vec<f64> = hz_points
        .iter()
        .map(|&hz| hz * n_fft as f64 / sr as f64)
        .collect();

    // Build triangular filters.
    let mut filterbank = Vec::with_capacity(n_mels);

    for m in 0..n_mels {
        let left = bin_indices[m];
        let center = bin_indices[m + 1];
        let right = bin_indices[m + 2];

        let mut filter = vec![0.0f64; n_bins];

        for k in 0..n_bins {
            let k_f64 = k as f64;

            if k_f64 >= left && k_f64 <= center && center > left {
                // Rising slope: left edge to center.
                filter[k] = (k_f64 - left) / (center - left);
            } else if k_f64 > center && k_f64 <= right && right > center {
                // Falling slope: center to right edge.
                filter[k] = (right - k_f64) / (right - center);
            } else {
                // Outside this filter's range — weight is zero.
            }
        }

        // Slaney normalization: divide by the width of the mel band in Hz.
        // This makes each filter contribute equal energy regardless of its
        // bandwidth, which is important because mel filters get wider (in Hz)
        // at higher frequencies.
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);
        for weight in &mut filter {
            *weight *= enorm;
        }

        filterbank.push(filter);
    }

    filterbank
}

// ---------------------------------------------------------------------------
// Power to decibels
// ---------------------------------------------------------------------------

/// Convert a power spectrum to decibels.
///
/// Matches librosa's `power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0)`:
///   dB = 10 * log10(max(power, amin) / ref)
///   dB = max(dB, max(dB) - top_db)
///
/// The amin floor prevents log(0). The top_db clipping limits dynamic range,
/// which improves numerical stability in downstream processing.
fn power_to_db(power: &[f64], amin: f64, top_db: f64) -> Vec<f64> {
    let reference = 1.0; // librosa default ref=1.0 (no scaling)

    // Convert to dB.
    let mut db: Vec<f64> = power
        .iter()
        .map(|&p| 10.0 * (p.max(amin) / reference).log10())
        .collect();

    // Clip to top_db below the peak.
    let max_db = db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let threshold = max_db - top_db;
    for val in &mut db {
        if *val < threshold {
            *val = threshold;
        } else {
            // Above threshold — no clipping needed.
        }
    }

    db
}

// ---------------------------------------------------------------------------
// DCT type II (orthonormal)
// ---------------------------------------------------------------------------

/// Compute the DCT type II with orthonormal normalization.
///
/// Takes `input` of length N and returns the first `n_output` DCT coefficients.
///
/// The DCT-II formula with ortho normalization is:
///   X[k] = scale * sum_{n=0}^{N-1} x[n] * cos(pi * (n + 0.5) * k / N)
///
/// where scale = sqrt(1/N) for k=0, sqrt(2/N) for k>0 (orthonormal).
///
/// This matches `scipy.fft.dct(type=2, norm='ortho')` which librosa uses.
fn dct_ii_ortho(input: &[f64], n_output: usize) -> Vec<f64> {
    let n = input.len();
    let mut output = Vec::with_capacity(n_output);

    for k in 0..n_output {
        let mut sum = 0.0;
        for (i, &x) in input.iter().enumerate() {
            // cos(pi * (i + 0.5) * k / N)
            sum += x * (PI * (i as f64 + 0.5) * k as f64 / n as f64).cos();
        }

        // Orthonormal scaling.
        let scale = if k == 0 {
            (1.0 / n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
        };

        output.push(sum * scale);
    }

    output
}

// ---------------------------------------------------------------------------
// MFCC extraction (full pipeline)
// ---------------------------------------------------------------------------

/// Extract MFCC features from audio.
///
/// This is the Rust equivalent of:
///   librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
///
/// with all librosa 0.11.0 defaults:
///   n_fft=2048, n_mels=128, fmin=0, fmax=sr/2, center=True, power=2.0,
///   norm='slaney', ref=1.0, amin=1e-10, top_db=80.0, dct type=2 norm='ortho'
///
/// Returns a Vec of n_mfcc coefficient vectors, where each inner Vec has
/// `time_frames` elements. Layout: result[coefficient_index][time_frame_index].
///
/// This layout matches the Python code's `mfcc_master[i]` indexing in sync.py.
pub fn extract_mfcc(
    audio: &[f32],
    sr: u32,
    n_mfcc: usize,
    hop_length: usize,
) -> Vec<Vec<f64>> {
    let n_fft = DEFAULT_N_FFT;
    let n_mels = DEFAULT_N_MELS;
    let fmin = 0.0;
    let fmax = sr as f64 / 2.0;

    // Step 1: STFT -> power spectrum.
    // Each frame is a Vec<f64> of length n_fft/2 + 1.
    let power_frames = stft_power(audio, n_fft, hop_length);
    let n_frames = power_frames.len();

    // Step 2: Create mel filterbank.
    let filterbank = create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax);

    // Step 3: Apply mel filterbank to each frame, convert to dB, then DCT.
    // Build the output as [n_mfcc][n_frames].
    let mut mfcc_matrix: Vec<Vec<f64>> = vec![Vec::with_capacity(n_frames); n_mfcc];

    for frame_power in &power_frames {
        // Apply mel filterbank: dot product of each filter with the power spectrum.
        let mut mel_energies = Vec::with_capacity(n_mels);
        for filter in &filterbank {
            let mut energy = 0.0;
            for (k, &weight) in filter.iter().enumerate() {
                energy += weight * frame_power[k];
            }
            mel_energies.push(energy);
        }

        // Convert mel energies to decibels.
        let mel_db = power_to_db(&mel_energies, POWER_TO_DB_AMIN, POWER_TO_DB_TOP_DB);

        // DCT type II (orthonormal) — keep first n_mfcc coefficients.
        let mfcc_coeffs = dct_ii_ortho(&mel_db, n_mfcc);

        // Distribute coefficients into the output matrix.
        for (coeff_idx, &value) in mfcc_coeffs.iter().enumerate() {
            mfcc_matrix[coeff_idx].push(value);
        }
    }

    mfcc_matrix
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;

    /// Generate a spectrally rich signal for MFCC testing.
    /// A pure sine wave produces flat MFCCs (no spectral variation), so we
    /// combine sweeps and noise for meaningful temporal structure.
    fn make_rich_signal(duration: f32, sr: u32) -> Vec<f32> {
        let n_samples = (sr as f32 * duration) as usize;
        let mut samples = Vec::with_capacity(n_samples);
        // Simple deterministic pseudo-random noise using a linear congruential
        // generator. Avoids depending on a rand crate for tests.
        let mut rng_state: u64 = 42;
        for i in 0..n_samples {
            let t = i as f32 / sr as f32;
            // Frequency sweep from 200 Hz ramping up over time.
            let sweep1 = (2.0 * PI_F32 * (200.0 + 50.0 * t) * t).sin();
            // Second sweep at a different base frequency.
            let sweep2 = 0.5 * (2.0 * PI_F32 * (800.0 + 30.0 * t) * t).sin();
            // Deterministic pseudo-random noise.
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((rng_state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2;
            samples.push(sweep1 + sweep2 + noise);
        }
        samples
    }

    #[test]
    fn test_hann_window_properties() {
        let size = 1024;
        let window = create_hann_window(size);

        assert_eq!(window.len(), size);

        // Periodic Hann window: w[0] = 0.0 (starts at zero).
        assert!((window[0]).abs() < 1e-10, "window should start at 0.0");

        // Peak is at the center of the window.
        let mid = size / 2;
        assert!(window[mid] > 0.99, "window peak should be near 1.0");
    }

    #[test]
    fn test_mel_hz_roundtrip() {
        // hz_to_mel and mel_to_hz should be inverses of each other.
        let test_frequencies = [0.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0];
        for &hz in &test_frequencies {
            let mel = hz_to_mel(hz);
            let recovered_hz = mel_to_hz(mel);
            let diff = (hz - recovered_hz).abs();
            assert!(
                diff < 0.01,
                "roundtrip failed for {}Hz: got {}Hz (diff {})",
                hz,
                recovered_hz,
                diff
            );
        }
    }

    #[test]
    fn test_extract_mfcc_shape() {
        // Verify the output dimensions match what librosa would produce.
        let sr: u32 = 44100;
        let duration = 2.0;
        let n_samples = (sr as f32 * duration) as usize;
        let hop_length: usize = 512;
        let n_mfcc: usize = 20;

        // Generate a sine wave.
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI_F32 * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        let mfcc = extract_mfcc(&audio, sr, n_mfcc, hop_length);

        // Should have n_mfcc coefficient vectors.
        assert_eq!(mfcc.len(), n_mfcc);

        // Each coefficient vector should have the expected number of time frames.
        // librosa formula: 1 + floor((n_samples + 2*pad - n_fft) / hop_length)
        // where pad = n_fft/2 for center=True.
        // = 1 + floor(n_samples / hop_length)
        // For n_samples=88200, hop_length=512: 1 + 172 = 173
        let expected_frames = 1 + n_samples / hop_length;
        assert_eq!(
            mfcc[0].len(),
            expected_frames,
            "expected {} frames, got {}",
            expected_frames,
            mfcc[0].len()
        );

        // All coefficient vectors should be the same length.
        for (i, coeffs) in mfcc.iter().enumerate() {
            assert_eq!(
                coeffs.len(),
                expected_frames,
                "coefficient {} has wrong length: {} vs {}",
                i,
                coeffs.len(),
                expected_frames
            );
        }
    }

    #[test]
    fn test_mfcc_nonzero_for_rich_signal() {
        // A spectrally rich signal should produce non-zero MFCCs with
        // meaningful variation across time.
        let sr: u32 = 44100;
        let audio = make_rich_signal(2.0, sr);
        let mfcc = extract_mfcc(&audio, sr, 20, 512);

        // Check that MFCCs are not all zero.
        let total_energy: f64 = mfcc.iter().flat_map(|c| c.iter()).map(|&v| v * v).sum();
        assert!(total_energy > 0.0, "MFCCs should not be all zero");

        // Check that there's temporal variation (not constant across time).
        for (i, coeffs) in mfcc.iter().enumerate() {
            let mean: f64 = coeffs.iter().sum::<f64>() / coeffs.len() as f64;
            let variance: f64 = coeffs.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
                / coeffs.len() as f64;
            // At least some coefficients should have non-trivial variance.
            if i < 5 {
                assert!(
                    variance > 1e-6,
                    "coefficient {} has near-zero variance ({})",
                    i,
                    variance
                );
            } else {
                // Higher coefficients may have less variation — that's fine.
            }
        }
    }

    #[test]
    fn test_dct_ii_ortho_known_values() {
        // DCT-II of a constant signal should concentrate all energy in the
        // first (DC) coefficient.
        let input = vec![1.0; 8];
        let output = dct_ii_ortho(&input, 4);

        // DC coefficient should be sqrt(N) * value = sqrt(8) * 1.0 = 2.828...
        // With ortho scaling: sqrt(1/N) * N * value = sqrt(N) * value
        let expected_dc = (8.0_f64).sqrt();
        assert!(
            (output[0] - expected_dc).abs() < 1e-10,
            "DC coefficient should be {}, got {}",
            expected_dc,
            output[0]
        );

        // All other coefficients should be zero for a constant input.
        for (i, &val) in output.iter().enumerate().skip(1) {
            assert!(
                val.abs() < 1e-10,
                "coefficient {} should be 0.0, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_power_to_db_basic() {
        // A power of 1.0 with ref=1.0 should give 0 dB.
        let result = power_to_db(&[1.0], POWER_TO_DB_AMIN, POWER_TO_DB_TOP_DB);
        assert!((result[0]).abs() < 1e-10, "power=1.0 should be 0 dB");

        // A power of 0.0 should be clamped to amin, giving 10*log10(amin).
        let result = power_to_db(&[0.0], POWER_TO_DB_AMIN, POWER_TO_DB_TOP_DB);
        let expected = 10.0 * POWER_TO_DB_AMIN.log10(); // -100 dB
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "power=0.0 should be {} dB, got {}",
            expected,
            result[0]
        );
    }
}
