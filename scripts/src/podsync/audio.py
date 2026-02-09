"""Audio loading and writing utilities."""

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate as mono.

    Args:
        path: Path to audio file (supports wav, mp3, aiff, etc.)
        target_sr: Target sample rate (default 44100)

    Returns:
        Tuple of (audio array as float32, sample rate)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # librosa.load handles format detection, resampling, and mono conversion
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)

    return audio.astype(np.float32), sr


def write_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """
    Write audio to a WAV file.

    Args:
        path: Output path (will be written as WAV regardless of extension)
        audio: Audio array (mono, float32)
        sr: Sample rate
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype='PCM_24')
