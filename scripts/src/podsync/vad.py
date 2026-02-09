"""Voice activity detection for finding speech regions."""

from typing import List, Tuple, Optional

import numpy as np
import webrtcvad


def detect_speech_regions(
    audio: np.ndarray,
    sr: int,
    frame_duration_ms: int = 30,
    aggressiveness: int = 2,
) -> List[Tuple[float, float]]:
    """
    Detect speech regions in audio using WebRTC VAD.

    Args:
        audio: Mono audio array (float32)
        sr: Sample rate (must be 8000, 16000, 32000, or 48000)
        frame_duration_ms: Frame size in ms (10, 20, or 30)
        aggressiveness: VAD aggressiveness (0-3, higher = more aggressive filtering)

    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    # WebRTC VAD requires specific sample rates
    valid_rates = [8000, 16000, 32000, 48000]
    if sr not in valid_rates:
        # Resample to 16kHz for VAD
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    vad = webrtcvad.Vad(aggressiveness)

    frame_size = int(sr * frame_duration_ms / 1000)
    frame_bytes = frame_size * 2  # 16-bit = 2 bytes per sample

    regions = []
    is_speech = False
    speech_start = 0.0

    for i in range(0, len(audio_int16) - frame_size, frame_size):
        frame = audio_int16[i:i + frame_size].tobytes()

        try:
            frame_is_speech = vad.is_speech(frame, sr)
        except Exception:
            frame_is_speech = False

        current_time = i / sr

        if frame_is_speech and not is_speech:
            # Speech started
            speech_start = current_time
            is_speech = True
        elif not frame_is_speech and is_speech:
            # Speech ended
            regions.append((speech_start, current_time))
            is_speech = False

    # Handle case where speech extends to end
    if is_speech:
        regions.append((speech_start, len(audio_int16) / sr))

    # Merge nearby regions (within 0.3s)
    merged = _merge_regions(regions, gap_threshold=0.3)

    return merged


def _merge_regions(
    regions: List[Tuple[float, float]],
    gap_threshold: float = 0.3
) -> List[Tuple[float, float]]:
    """Merge speech regions that are close together."""
    if not regions:
        return []

    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_threshold:
            # Merge with previous
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def find_first_speech_segment(
    audio: np.ndarray,
    sr: int,
    min_duration: float = 30.0,
    search_limit: float = 600.0,
) -> Optional[Tuple[float, float]]:
    """
    Find the first speech segment with at least min_duration seconds.

    Args:
        audio: Mono audio array
        sr: Sample rate
        min_duration: Minimum duration in seconds for a valid segment
        search_limit: Maximum seconds to search into the audio

    Returns:
        (start_time, end_time) tuple or None if not found
    """
    # Limit search to first search_limit seconds
    max_samples = int(search_limit * sr)
    audio_limited = audio[:max_samples]

    regions = detect_speech_regions(audio_limited, sr)

    if not regions:
        return None

    # Sort regions by duration (longest first)
    regions_by_length = sorted(regions, key=lambda r: r[1] - r[0], reverse=True)

    # First: find a single region meeting minimum duration
    for start, end in regions_by_length:
        duration = end - start
        if duration >= min_duration:
            return (start, end)

    # Second: use the longest single region even if under min_duration
    # A shorter but contiguous speech region is more reliable for correlation
    # than a sparse accumulation of tiny regions with gaps
    longest_start, longest_end = regions_by_length[0]
    longest_duration = longest_end - longest_start

    # Only fall back to accumulation if the longest region is very short
    if longest_duration >= 10.0:
        return (longest_start, longest_end)

    # Last resort: accumulate consecutive regions but only nearby ones (within 2s gap)
    total_speech = 0.0
    segment_start = None
    prev_end = None

    for start, end in regions:
        if prev_end is not None and (start - prev_end) > 2.0:
            # Gap too large, restart accumulation
            total_speech = 0.0
            segment_start = None

        if segment_start is None:
            segment_start = start

        total_speech += (end - start)
        prev_end = end

        if total_speech >= min_duration:
            return (segment_start, end)

    # Return whatever we have if any speech was found
    if segment_start is not None and total_speech > 0:
        return (segment_start, prev_end)

    return None
