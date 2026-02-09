"""Audio synchronization via MFCC cross-correlation."""

from typing import Tuple

import librosa
import numpy as np
from scipy import signal


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extract MFCC features from audio.

    Args:
        audio: Mono audio array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for STFT

    Returns:
        MFCC matrix (n_mfcc, time_frames)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )
    return mfcc


def find_offset(
    master: np.ndarray,
    track: np.ndarray,
    sr: int,
    search_window: float = 600.0,
    correlation_window: float = 120.0,
    hop_length: int = 512,
) -> Tuple[float, float]:
    """
    Find the time offset of track relative to master using MFCC cross-correlation.

    Args:
        master: Master audio array
        track: Track to align
        sr: Sample rate
        search_window: How far into master to search (seconds)
        correlation_window: How much of track to use for correlation (seconds)
        hop_length: Hop length for MFCC extraction

    Returns:
        Tuple of (offset_seconds, confidence)
        - Positive offset means track starts later than master
        - Negative offset means track starts earlier than master
    """
    # Limit master to search window
    master_samples = int(search_window * sr)
    master_limited = master[:master_samples]

    # Limit track to correlation window
    track_samples = int(correlation_window * sr)
    track_limited = track[:track_samples]

    # Extract MFCCs
    mfcc_master = extract_mfcc(master_limited, sr, hop_length=hop_length)
    mfcc_track = extract_mfcc(track_limited, sr, hop_length=hop_length)

    # Cross-correlate each MFCC coefficient independently, then sum
    # This preserves spectral discrimination instead of collapsing to energy
    n_coeffs = mfcc_master.shape[0]
    correlation = None

    for i in range(n_coeffs):
        m = mfcc_master[i]
        t = mfcc_track[i]

        # Normalize each coefficient's time series
        m = (m - np.mean(m)) / (np.std(m) + 1e-8)
        t = (t - np.mean(t)) / (np.std(t) + 1e-8)

        c = signal.correlate(m, t, mode='full')
        if correlation is None:
            correlation = c
        else:
            correlation += c

    # Find peak
    peak_idx = np.argmax(correlation)
    peak_value = correlation[peak_idx]

    # Peak validation: check if top peak is clearly above second-best
    # Exclude a neighborhood around the peak (+/- 50 frames) to find the next independent peak
    corr_copy = correlation.copy()
    exclude_radius = 50
    exclude_start = max(0, peak_idx - exclude_radius)
    exclude_end = min(len(corr_copy), peak_idx + exclude_radius + 1)
    corr_copy[exclude_start:exclude_end] = -np.inf
    second_peak_value = np.max(corr_copy)

    # Confidence based on how much the peak stands out from the second-best
    peak_ratio = peak_value / (second_peak_value + 1e-8) if second_peak_value > 0 else 10.0
    confidence = min(1.0, max(0.0, (peak_ratio - 1.0) / 0.5))  # 1.0 ratio → 0, 1.5+ ratio → 1.0

    # Convert peak index to time offset
    # In 'full' mode, the zero-lag point is at len(track_mfcc_frames) - 1
    zero_lag_idx = mfcc_track.shape[1] - 1
    lag_frames = peak_idx - zero_lag_idx

    # Convert frames to seconds
    # Positive lag = track content found later in master = positive offset
    offset_seconds = lag_frames * hop_length / sr

    return offset_seconds, confidence


def compute_drift(
    master: np.ndarray,
    track: np.ndarray,
    sr: int,
    initial_offset: float,
    end_window: float = 120.0,
) -> float:
    """
    Compute drift by comparing offset at end vs start.

    Args:
        master: Master audio array
        track: Track audio array (already know its start offset)
        sr: Sample rate
        initial_offset: The offset found at the start
        end_window: Seconds from end to use for drift measurement

    Returns:
        Drift in seconds (positive = track ran faster, negative = track ran slower)
    """
    # Get the last end_window seconds of master
    end_samples = int(end_window * sr)

    if len(master) < end_samples * 2:
        # Audio too short for drift measurement
        return 0.0

    master_end = master[-end_samples:]

    # Calculate where in track this should correspond to
    # master[M] corresponds to track[M - initial_offset]
    track_end_start = int(len(master) - end_samples - initial_offset * sr)

    if track_end_start < 0 or track_end_start + end_samples > len(track):
        # Can't measure drift - track doesn't cover master's end
        return 0.0

    track_end = track[track_end_start:track_end_start + end_samples]

    # Find offset at end
    end_offset, _ = find_offset(
        master_end, track_end, sr,
        search_window=end_window,
        correlation_window=min(60.0, end_window / 2),
    )

    # Drift = difference between end offset and initial offset
    # (end_offset should be ~0 if no drift, since we positioned track_end correctly)
    drift = end_offset

    return drift
