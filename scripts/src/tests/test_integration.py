"""Integration tests using real podcast files.

These tests are skipped by default. Run with:
    PODSYNC_TEST_DIR=/path/to/episode RUN_INTEGRATION=1 pytest tests/test_integration.py
"""

import os
from pathlib import Path

import pytest

from podsync.audio import load_audio
from podsync.vad import find_first_speech_segment
from podsync.sync import find_offset


# Skip all tests in this module unless --run-integration is passed
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="Integration tests skipped. Set RUN_INTEGRATION=1 to run."
)


# Path to real test files — set PODSYNC_TEST_DIR to your episode folder
TEST_EPISODE_DIR = Path(os.environ.get("PODSYNC_TEST_DIR", "."))

# File names within the test episode dir — override via env vars
MASTER_FILE = os.environ.get("PODSYNC_TEST_MASTER", "master.mp3")
TRACK_FILE = os.environ.get("PODSYNC_TEST_TRACK", "track.wav")


class TestRealFiles:
    def test_loads_master_mp3(self):
        """Load the real master MP3 file."""
        master_path = TEST_EPISODE_DIR / MASTER_FILE
        if not master_path.exists():
            pytest.skip(f"Test file not found: {master_path}")

        audio, sr = load_audio(master_path)

        assert sr == 44100
        assert len(audio) > sr * 60  # At least 1 minute

    def test_detects_speech_in_real_track(self):
        """Detect speech in a real participant track."""
        track_path = TEST_EPISODE_DIR / TRACK_FILE
        if not track_path.exists():
            pytest.skip(f"Test file not found: {track_path}")

        audio, sr = load_audio(track_path)
        segment = find_first_speech_segment(audio, sr, min_duration=30.0)

        assert segment is not None
        start, end = segment
        assert end - start >= 30.0

    def test_finds_offset_between_real_tracks(self):
        """Find offset between master and real track."""
        master_path = TEST_EPISODE_DIR / MASTER_FILE
        track_path = TEST_EPISODE_DIR / TRACK_FILE

        if not master_path.exists() or not track_path.exists():
            pytest.skip("Test files not found")

        master, sr = load_audio(master_path)
        track, _ = load_audio(track_path, target_sr=sr)

        offset, confidence = find_offset(master, track, sr)

        # Offset should be reasonable (within 10 minutes)
        assert abs(offset) < 600
        # Confidence should be decent
        assert confidence > 0.3
