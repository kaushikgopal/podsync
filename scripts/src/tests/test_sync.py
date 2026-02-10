"""Tests for audio synchronization via cross-correlation."""

import numpy as np

from podsync.sync import extract_mfcc, find_offset, compute_drift


def _make_rich_signal(duration: float, sr: int, seed: int = 42) -> np.ndarray:
    """Create a spectrally rich synthetic signal suitable for MFCC correlation.

    A simple sine wave produces flat MFCCs (no spectral variation over time),
    which makes cross-correlation unreliable. This combines frequency sweeps
    and noise to give the MFCCs meaningful temporal structure.
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (
        np.sin(2 * np.pi * (200 + 50 * t) * t)
        + 0.5 * np.sin(2 * np.pi * (800 + 30 * t) * t)
        + 0.1 * np.random.randn(len(t))
    ).astype(np.float32)


class TestExtractMfcc:
    def test_extracts_mfcc_features(self):
        """Extract MFCC features from audio."""
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        mfcc = extract_mfcc(audio, sr)

        assert mfcc.ndim == 2
        assert mfcc.shape[0] >= 13  # n_mfcc
        assert mfcc.shape[1] > 0    # time frames


class TestFindOffset:
    def test_finds_positive_offset(self):
        """When track content appears later in master, offset is positive.

        Simulates production usage: the track passed to find_offset is the
        speech content (no silence prefix). The master contains the same
        content starting at a later position.

        Master: [silence...][content...]
        Track:  [content...]  (just the content portion)

        find_offset should report that the content appears at +2s in master.
        """
        sr = 44100
        content_duration = 10.0
        master_duration = 15.0
        offset_seconds = 2.0

        content = _make_rich_signal(content_duration, sr)

        # Build master: silence then content
        offset_samples = int(offset_seconds * sr)
        master = np.zeros(int(master_duration * sr), dtype=np.float32)
        master[offset_samples:offset_samples + len(content)] = content

        # Track is just the content (like after VAD extraction)
        track = content

        found_offset, confidence = find_offset(
            master, track, sr,
            search_window=master_duration,
            correlation_window=content_duration,
        )

        assert abs(found_offset - offset_seconds) < 0.2
        assert confidence > 0.3

    def test_finds_near_zero_offset(self):
        """When track and master start with the same content, offset is ~0."""
        sr = 44100
        duration = 10.0

        signal = _make_rich_signal(duration, sr)

        found_offset, confidence = find_offset(
            signal, signal, sr,
            search_window=duration,
            correlation_window=5.0,
        )

        assert abs(found_offset) < 0.2
        assert confidence > 0.3


class TestComputeDrift:
    def test_returns_none_for_short_audio(self):
        """Drift measurement requires audio longer than 2x the end window.

        For a 10s signal with the default 120s end_window, the audio is too
        short to measure drift. compute_drift returns None (not 0.0) to
        distinguish 'unmeasurable' from 'measured and zero'.
        """
        sr = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        drift = compute_drift(audio, audio, sr, initial_offset=0.0)

        assert drift is None

    def test_measures_near_zero_drift_for_same_signal(self):
        """Same signal correlated against itself should have near-zero drift."""
        sr = 44100
        duration = 300.0  # 5 minutes — long enough for drift measurement

        audio = _make_rich_signal(duration, sr)

        drift = compute_drift(audio, audio, sr, initial_offset=0.0)

        assert drift is not None
        assert abs(drift) < 0.5
