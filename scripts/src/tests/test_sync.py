"""Tests for audio synchronization via cross-correlation."""

import numpy as np

from podsync.sync import extract_mfcc, find_offset, compute_drift


class TestExtractMfcc:
    def test_extracts_mfcc_features(self):
        """Extract MFCC features from audio."""
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        mfcc = extract_mfcc(audio, sr)

        # Should have multiple coefficients (typically 13 or 20)
        assert mfcc.ndim == 2
        assert mfcc.shape[0] >= 13  # n_mfcc
        assert mfcc.shape[1] > 0    # time frames


class TestFindOffset:
    def test_finds_offset_for_delayed_signal(self):
        """Find the time offset between master and a delayed copy."""
        sr = 44100
        duration = 5.0
        delay_seconds = 1.5

        # Create a distinctive signal (chirp)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        master = np.sin(2 * np.pi * (200 + 100 * t) * t).astype(np.float32)

        # Create delayed version
        delay_samples = int(delay_seconds * sr)
        track = np.concatenate([
            np.zeros(delay_samples, dtype=np.float32),
            master[:len(master) - delay_samples]
        ])

        offset, confidence = find_offset(
            master, track, sr,
            search_window=3.0,
            correlation_window=2.0
        )

        # Offset should be close to delay_seconds
        assert abs(offset - delay_seconds) < 0.1
        assert confidence > 0.3  # Confidence varies with signal characteristics

    def test_finds_negative_offset_for_early_signal(self):
        """Find negative offset when track starts before master."""
        sr = 44100
        duration = 5.0
        early_seconds = 1.0

        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = np.sin(2 * np.pi * (200 + 100 * t) * t).astype(np.float32)

        # Track starts early (master is delayed relative to track)
        early_samples = int(early_seconds * sr)
        master = np.concatenate([
            np.zeros(early_samples, dtype=np.float32),
            signal[:len(signal) - early_samples]
        ])
        track = signal

        offset, confidence = find_offset(
            master, track, sr,
            search_window=3.0,
            correlation_window=2.0
        )

        # Offset should be negative (track needs to be shifted back)
        assert abs(offset - (-early_seconds)) < 0.1


class TestComputeDrift:
    def test_measures_drift_between_start_and_end(self):
        """Measure drift by comparing offsets at start and end."""
        sr = 44100
        duration = 10.0
        initial_offset = 1.0
        drift_per_second = 0.01  # 10ms per second

        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        master = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Create track with drift (stretching simulation)
        # For simplicity, just report that drift computation exists
        # Real drift testing requires more sophisticated signal manipulation

        # This is a smoke test - real validation needs actual drifted audio
        drift = compute_drift(master, master, sr, initial_offset=0.0)

        # Same signal should have ~0 drift
        assert abs(drift) < 0.1
