"""Audio synchronization via MFCC cross-correlation."""

from typing import Optional, Tuple

import librosa
import numpy as np
from scipy import signal

from .audio import seconds_to_samples


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of MFCC coefficients to extract. 20 captures the vocal tract shape
# (formants) without introducing high-frequency noise. Standard in speech
# processing — Kaldi uses 13, librosa defaults to 20.
N_MFCC_COEFFICIENTS = 20

# STFT hop length in samples. At 44.1kHz this is ~11.6ms per frame — the
# standard resolution for speech analysis. Smaller values increase precision
# but also computation time quadratically (via cross-correlation).
HOP_LENGTH = 512

# How far into the master track to search for a match. 600s = 10 minutes.
# A host could join late, so we search well beyond the expected start time.
CORRELATION_SEARCH_WINDOW_S = 600.0

# Small epsilon to prevent division by zero in normalization and confidence
# calculations. 1e-8 is appropriate for float64 (numpy's default). If porting
# to float32, use 1e-6 instead.
EPSILON = 1e-8

# When searching for the second-best correlation peak (to compute confidence),
# exclude this many frames around the best peak. At HOP_LENGTH=512 and
# sr=44100, 50 frames ≈ 0.58 seconds. This prevents sidelobes of the main
# peak from being counted as an independent match.
PEAK_EXCLUSION_RADIUS = 50

# Confidence threshold: below this, the correlation peak isn't distinct enough
# to trust. A ratio of 1.0 means the best and second-best peaks are equal
# (ambiguous). The formula maps ratio 1.0 → confidence 0.0, ratio 1.5+ →
# confidence 1.0, linearly between. The 0.5 divisor controls sensitivity.
CONFIDENCE_SENSITIVITY = 0.5

# Below this confidence, warn the user that the sync may be inaccurate.
LOW_CONFIDENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# MFCC extraction
# ---------------------------------------------------------------------------

def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC_COEFFICIENTS,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extract MFCC features from audio.

    Returns a matrix of shape (n_mfcc, time_frames).
    """
    return librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )


# ---------------------------------------------------------------------------
# Offset detection
# ---------------------------------------------------------------------------

def find_offset(
    master: np.ndarray,
    track: np.ndarray,
    sr: int,
    search_window: float = CORRELATION_SEARCH_WINDOW_S,
    correlation_window: float = 120.0,
    hop_length: int = HOP_LENGTH,
) -> Tuple[float, float]:
    """Find the time offset of *track* relative to *master* using MFCC cross-correlation.

    Returns:
        (offset_seconds, confidence) where:
        - Positive offset: track content appears later in master (track started late).
        - Negative offset: track content appears earlier (track started early).
        - Confidence: 0.0 (ambiguous) to 1.0 (clear match).
    """
    # --- Limit inputs to relevant windows ----------------------------------
    master_samples = seconds_to_samples(search_window, sr)
    master_limited = master[:master_samples]

    track_samples = seconds_to_samples(correlation_window, sr)
    track_limited = track[:track_samples]

    # --- Extract MFCCs -----------------------------------------------------
    mfcc_master = extract_mfcc(master_limited, sr, hop_length=hop_length)
    mfcc_track = extract_mfcc(track_limited, sr, hop_length=hop_length)

    # --- Cross-correlate each coefficient independently, then sum ----------
    # Correlating per-coefficient preserves spectral discrimination. If we
    # flattened first, high-energy coefficients would dominate and subtle
    # spectral differences (like distinguishing two speakers) would be lost.
    n_coeffs = mfcc_master.shape[0]
    correlation = None

    for i in range(n_coeffs):
        m = mfcc_master[i]
        t = mfcc_track[i]

        # Normalize each coefficient's time series to zero mean, unit variance.
        # This ensures all coefficients contribute equally to the sum.
        m = (m - np.mean(m)) / (np.std(m) + EPSILON)
        t = (t - np.mean(t)) / (np.std(t) + EPSILON)

        c = signal.correlate(m, t, mode='full')
        if correlation is None:
            correlation = c
        else:
            correlation += c

    # --- Find primary peak -------------------------------------------------
    peak_idx = np.argmax(correlation)
    peak_value = correlation[peak_idx]

    # --- Compute confidence ------------------------------------------------
    # Find the second-best peak (excluding a neighborhood around the best)
    # to determine how distinct our match is.
    corr_copy = correlation.copy()
    exclude_start = max(0, peak_idx - PEAK_EXCLUSION_RADIUS)
    exclude_end = min(len(corr_copy), peak_idx + PEAK_EXCLUSION_RADIUS + 1)
    corr_copy[exclude_start:exclude_end] = -np.inf
    second_peak_value = np.max(corr_copy)

    if second_peak_value > 0:
        peak_ratio = peak_value / (second_peak_value + EPSILON)
    else:
        # Second peak is zero or negative — the primary peak is the only
        # meaningful one. This is a strong match.
        peak_ratio = 10.0

    # Map ratio to [0, 1]: ratio 1.0 → 0.0 (peaks are equal, ambiguous),
    # ratio 1.0 + CONFIDENCE_SENSITIVITY → 1.0 (clear winner).
    confidence = min(1.0, max(0.0, (peak_ratio - 1.0) / CONFIDENCE_SENSITIVITY))

    # --- Convert peak index to time offset ---------------------------------
    # In scipy correlate mode='full', the zero-lag point is at
    # len(shorter_signal) - 1. A peak to the right of zero-lag means
    # the track content appears later in the master.
    zero_lag_idx = mfcc_track.shape[1] - 1
    lag_frames = peak_idx - zero_lag_idx
    offset_seconds = lag_frames * hop_length / sr

    return offset_seconds, confidence


# ---------------------------------------------------------------------------
# Drift measurement
# ---------------------------------------------------------------------------

def compute_drift(
    master: np.ndarray,
    track: np.ndarray,
    sr: int,
    initial_offset: float,
    end_window: float = 120.0,
) -> Optional[float]:
    """Compute clock drift by comparing alignment at end vs start.

    If the track's recording device has a slightly different clock rate than
    the master's, the offset at the end of the recording will differ from the
    offset at the start. This difference is drift.

    Returns:
        Drift in seconds, or None if the audio is too short to measure.
        - Positive drift: track ran faster (audio compressed over time).
        - Negative drift: track ran slower (audio stretched over time).

        How this works: we take a window from the end of the master and the
        corresponding expected position in the track (based on initial_offset).
        We correlate them to find the *actual* offset at the end. If there's
        no drift, this offset should be ~0 (since we positioned the track
        window based on the initial offset). Any non-zero value is drift.
    """
    end_samples = seconds_to_samples(end_window, sr)

    # Need at least 2x the end window — otherwise the "end" overlaps with
    # the "start" and drift measurement is meaningless.
    if len(master) < end_samples * 2:
        return None

    master_end = master[-end_samples:]

    # Calculate where in the track corresponds to the master's end region.
    # master[M] corresponds to track[M - initial_offset_in_samples].
    track_end_start = int(len(master) - end_samples - initial_offset * sr)

    if track_end_start < 0:
        # Track doesn't extend far enough back to cover master's end.
        return None
    elif track_end_start + end_samples > len(track):
        # Track is shorter than master at the end.
        return None
    else:
        # Track covers the expected region — proceed.
        pass

    track_end = track[track_end_start:track_end_start + end_samples]

    # Correlate the end regions. Use a smaller correlation window (half the
    # search window, capped at 60s) because we're searching a narrower range
    # — we already know approximately where the track should be.
    end_offset, _ = find_offset(
        master_end, track_end, sr,
        search_window=end_window,
        correlation_window=min(60.0, end_window / 2),
    )

    # end_offset should be ~0 if no drift (since we positioned track_end
    # based on initial_offset). Any deviation is drift.
    drift = end_offset

    return drift
