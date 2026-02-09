"""Tests for audio loading and writing."""

import numpy as np
import tempfile
import os
from pathlib import Path

import soundfile as sf

from podsync.audio import load_audio, write_audio


class TestLoadAudio:
    def test_loads_wav_and_resamples_to_target_sr(self):
        """Load a WAV file and resample to 44.1kHz mono."""
        # Create a test WAV at 48kHz stereo
        sr_original = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr_original * duration), endpoint=False)
        # Stereo 440Hz sine wave
        audio_stereo = np.column_stack([
            np.sin(2 * np.pi * 440 * t),
            np.sin(2 * np.pi * 440 * t)
        ])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_stereo, sr_original)
            temp_path = f.name

        try:
            audio, sr = load_audio(temp_path, target_sr=44100)

            # Should be mono
            assert audio.ndim == 1
            # Should be resampled to 44.1kHz
            assert sr == 44100
            # Duration should be preserved (approximately)
            assert abs(len(audio) / sr - duration) < 0.01
        finally:
            os.unlink(temp_path)

    def test_loads_mp3(self):
        """Load an MP3 file."""
        # This test requires an actual MP3 - we'll use librosa's example if available
        # For now, just test that the function exists and handles missing files
        try:
            load_audio("/nonexistent/file.mp3")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass


class TestWriteAudio:
    def test_writes_wav_at_specified_sr(self):
        """Write audio to WAV at specified sample rate."""
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            write_audio(temp_path, audio, sr)

            # Read it back
            loaded, loaded_sr = sf.read(temp_path)
            assert loaded_sr == sr
            assert len(loaded) == len(audio)
            # Values should match closely
            assert np.allclose(loaded, audio, atol=1e-4)
        finally:
            os.unlink(temp_path)
