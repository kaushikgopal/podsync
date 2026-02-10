"""Tests for voice activity detection."""

import numpy as np

from podsync.vad import detect_speech_regions, find_first_speech_segment


class TestDetectSpeechRegions:
    def test_detects_speech_in_simple_signal(self):
        """Detect speech regions in a signal with clear speech/silence boundaries."""
        sr = 16000  # webrtcvad requires 8k, 16k, 32k, or 48k

        # Create signal: 1s silence, 2s "speech" (noise), 1s silence, 1s "speech"
        np.random.seed(42)
        silence = np.zeros(sr, dtype=np.float32)
        speech = np.random.randn(sr).astype(np.float32) * 0.5

        audio = np.concatenate([
            silence,           # 0-1s: silence
            speech, speech,    # 1-3s: speech
            silence,           # 3-4s: silence
            speech,            # 4-5s: speech
        ])

        regions = detect_speech_regions(audio, sr)

        # Should detect at least 1 speech region
        assert len(regions) >= 1
        # First region should start around 1s
        assert regions[0][0] >= 0.8 and regions[0][0] <= 1.5

    def test_returns_empty_for_silence(self):
        """Return empty list for silent audio."""
        sr = 16000
        silence = np.zeros(sr * 5, dtype=np.float32)

        regions = detect_speech_regions(silence, sr)

        assert len(regions) == 0


class TestFindFirstSpeechSegment:
    def test_finds_first_segment_with_minimum_duration(self):
        """Find first speech segment meeting minimum duration requirement."""
        sr = 16000

        np.random.seed(42)
        silence = np.zeros(sr, dtype=np.float32)
        short_speech = np.random.randn(sr // 2).astype(np.float32) * 0.5
        long_speech = np.random.randn(sr * 3).astype(np.float32) * 0.5

        audio = np.concatenate([
            silence,
            short_speech,
            silence[:sr // 2],
            long_speech,
        ])

        start, end = find_first_speech_segment(audio, sr, min_duration=2.0)

        assert start >= 1.5
        assert end - start >= 2.0

    def test_returns_none_for_pure_silence(self):
        """Return None when there is no speech at all."""
        sr = 16000
        # Pure silence — VAD will detect zero speech regions, and
        # find_first_speech_segment returns None.
        audio = np.zeros(sr * 2, dtype=np.float32)

        result = find_first_speech_segment(audio, sr, min_duration=30.0)

        assert result is None
