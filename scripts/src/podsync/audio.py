"""Audio loading, writing, and sample manipulation utilities."""

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard podcast sample rate. Matches most DAW defaults and is the de facto
# standard for podcast distribution. All internal processing uses this rate.
TARGET_SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seconds_to_samples(seconds: float, sr: int) -> int:
    """Convert a duration in seconds to a sample count.

    Uses round() rather than int() truncation so that a value like
    1.9999999 seconds (common with float arithmetic) doesn't lose a sample.
    """
    return int(round(seconds * sr))


# ---------------------------------------------------------------------------
# Loading / Writing
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file, resample to *target_sr*, and convert to mono float32.

    Supports wav, mp3, aiff, flac, ogg — anything librosa can decode.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *target_sr* is not a positive integer.
    """
    if target_sr <= 0:
        raise ValueError(f"target_sr must be positive, got {target_sr}")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # librosa.load handles format detection, resampling, and mono conversion.
    # mono=True downmixes multi-channel files by averaging channels.
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)

    return audio.astype(np.float32), sr


def write_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Write a mono audio array to a 24-bit WAV file.

    Raises:
        ValueError: If *audio* is not a 1-D array (i.e. not mono).
    """
    if audio.ndim != 1:
        raise ValueError(
            f"write_audio expects mono (1-D) audio, got shape {audio.shape}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # format='WAV' is explicit — soundfile would otherwise infer from the
    # file extension, which could silently produce a non-WAV file.
    sf.write(str(path), audio, sr, subtype='PCM_24', format='WAV')


# ---------------------------------------------------------------------------
# Sample manipulation
# ---------------------------------------------------------------------------

def apply_offset(
    audio: np.ndarray,
    offset_samples: int,
    target_length: int,
) -> np.ndarray:
    """Shift *audio* by *offset_samples* and fit into a buffer of *target_length*.

    - Positive offset: pad the beginning with silence (track starts after master).
    - Negative offset: trim the beginning (track starts before master).
    - Result is always exactly *target_length* samples, zero-padded at the end
      if the shifted audio is shorter.

    Raises:
        ValueError: If the negative offset is larger than the audio length
                    (nothing left after trimming).
    """
    output = np.zeros(target_length, dtype=np.float32)

    if offset_samples >= 0:
        # Track starts after master — pad beginning with silence, then copy
        # as much audio as fits within target_length.
        available = target_length - offset_samples
        if available <= 0:
            # Offset is beyond the target length — entire output is silence.
            # This is unusual but not an error; the track simply doesn't
            # overlap with the master's time range.
            return output
        else:
            copy_len = min(len(audio), available)
            output[offset_samples:offset_samples + copy_len] = audio[:copy_len]
    else:
        # Track starts before master — trim the beginning.
        trim = abs(offset_samples)
        if trim >= len(audio):
            raise ValueError(
                f"Negative offset ({offset_samples} samples) is larger than "
                f"audio length ({len(audio)} samples) — nothing left after trim"
            )
        else:
            remaining = len(audio) - trim
            copy_len = min(remaining, target_length)
            output[:copy_len] = audio[trim:trim + copy_len]

    return output
