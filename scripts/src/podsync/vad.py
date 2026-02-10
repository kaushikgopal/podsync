"""Voice activity detection for finding speech regions."""

from typing import List, Optional, Tuple

import librosa
import numpy as np
import webrtcvad


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Merge speech regions separated by gaps shorter than this. Typical inter-word
# pause is ~200ms; 300ms captures brief pauses within sentences while still
# splitting genuinely separate utterances.
SPEECH_MERGE_GAP_S = 0.3

# Minimum contiguous speech for a single region to be used directly for
# correlation. Below this threshold, a single region is too short to produce
# a reliable cross-correlation peak, so we fall back to accumulating nearby
# shorter regions.
MIN_SINGLE_REGION_DURATION_S = 10.0

# Maximum gap between speech regions that still counts as "nearby" for the
# accumulation fallback. Larger than SPEECH_MERGE_GAP_S because we're more
# tolerant when assembling a usable segment from fragments.
ACCUMULATION_GAP_LIMIT_S = 2.0

# How far into the audio to search for speech. 600s = 10 minutes — covers
# cases where a host joins late or there's a long intro before they speak.
VAD_SEARCH_LIMIT_S = 600.0

# Preferred minimum duration of speech needed for reliable correlation.
# The cross-correlation needs enough material to produce a clear peak.
# 30s of speech is sufficient for MFCC-based matching.
PREFERRED_SPEECH_DURATION_S = 30.0

# WebRTC VAD only works with these sample rates. If the input doesn't match,
# we resample to 16kHz (good balance of accuracy and speed for VAD).
_WEBRTC_VALID_RATES = (8000, 16000, 32000, 48000)
_WEBRTC_RESAMPLE_TARGET = 16000

# WebRTC VAD aggressiveness level (0-3). Higher = more aggressive filtering
# of non-speech. 2 is a good default: filters most background noise without
# clipping soft speech.
_WEBRTC_AGGRESSIVENESS = 2

# Frame duration for WebRTC VAD. Must be 10, 20, or 30 ms.
# 30ms gives the best accuracy per the WebRTC documentation.
_WEBRTC_FRAME_DURATION_MS = 30


# ---------------------------------------------------------------------------
# Speech region detection
# ---------------------------------------------------------------------------

def detect_speech_regions(
    audio: np.ndarray,
    sr: int,
) -> List[Tuple[float, float]]:
    """Detect speech regions in audio using WebRTC VAD.

    Args:
        audio: Mono audio array (float32, values in [-1.0, 1.0]).
        sr: Sample rate of *audio*.

    Returns:
        List of (start_time, end_time) tuples in seconds, sorted by start time.
        Times are relative to the *original* audio's timeline regardless of
        any internal resampling.
    """
    # --- Resample if needed ------------------------------------------------
    # WebRTC VAD requires one of a few specific sample rates. If the input
    # doesn't match, we resample to 16kHz for VAD processing only.
    # The returned timestamps are still in terms of the original audio.
    if sr in _WEBRTC_VALID_RATES:
        vad_audio = audio
        vad_sr = sr
    else:
        vad_audio = librosa.resample(audio, orig_sr=sr, target_sr=_WEBRTC_RESAMPLE_TARGET)
        vad_sr = _WEBRTC_RESAMPLE_TARGET

    # --- Convert to 16-bit PCM --------------------------------------------
    # WebRTC VAD expects raw 16-bit PCM bytes. Clip first to prevent int16
    # overflow — librosa can produce values outside [-1, 1] after processing.
    clipped = np.clip(vad_audio, -1.0, 1.0)
    audio_int16 = (clipped * np.iinfo(np.int16).max).astype(np.int16)

    vad = webrtcvad.Vad(_WEBRTC_AGGRESSIVENESS)

    frame_size = int(vad_sr * _WEBRTC_FRAME_DURATION_MS / 1000)

    # --- Frame-by-frame VAD ------------------------------------------------
    regions: List[Tuple[float, float]] = []
    is_speech = False
    speech_start = 0.0

    for i in range(0, len(audio_int16) - frame_size, frame_size):
        frame = audio_int16[i:i + frame_size].tobytes()

        try:
            frame_is_speech = vad.is_speech(frame, vad_sr)
        except webrtcvad.Error:
            # webrtcvad.Error is raised for invalid frame sizes or sample rates.
            # If the VAD rejects a frame, treat it as non-speech rather than
            # crashing — but this shouldn't happen with correct frame sizing.
            frame_is_speech = False

        current_time = i / vad_sr

        if frame_is_speech and not is_speech:
            # Transition: silence → speech
            speech_start = current_time
            is_speech = True
        elif not frame_is_speech and is_speech:
            # Transition: speech → silence
            regions.append((speech_start, current_time))
            is_speech = False
        else:
            # No transition — either still in speech or still in silence.
            pass

    # Handle speech that extends to the end of the audio
    if is_speech:
        regions.append((speech_start, len(audio_int16) / vad_sr))
    else:
        # Audio ended during silence — nothing to append.
        pass

    # --- Merge nearby regions ----------------------------------------------
    merged = _merge_regions(regions, gap_threshold=SPEECH_MERGE_GAP_S)

    return merged


def _merge_regions(
    regions: List[Tuple[float, float]],
    gap_threshold: float,
) -> List[Tuple[float, float]]:
    """Merge speech regions that are separated by less than *gap_threshold* seconds."""
    if not regions:
        return []

    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_threshold:
            # Gap is small enough — extend the previous region.
            merged[-1] = (prev_start, end)
        else:
            # Gap is too large — start a new region.
            merged.append((start, end))

    return merged


# ---------------------------------------------------------------------------
# Speech segment selection
# ---------------------------------------------------------------------------

def find_first_speech_segment(
    audio: np.ndarray,
    sr: int,
    min_duration: float = PREFERRED_SPEECH_DURATION_S,
    search_limit: float = VAD_SEARCH_LIMIT_S,
) -> Optional[Tuple[float, float]]:
    """Find a speech segment suitable for cross-correlation.

    Uses a three-tier strategy, from most to least reliable:

    1. **Single region >= min_duration**: A long contiguous speech region is
       ideal for correlation — it gives a clean, unambiguous peak.

    2. **Longest single region >= MIN_SINGLE_REGION_DURATION_S**: If no single
       region meets min_duration but one is at least 10s, use it. A shorter
       but contiguous region is more reliable than stitching fragments.

    3. **Accumulated nearby regions**: As a last resort, accumulate consecutive
       regions within ACCUMULATION_GAP_LIMIT_S of each other until we reach
       min_duration. This handles cases where speech is fragmented (e.g.,
       short responses in a conversation).

    Args:
        audio: Mono audio array.
        sr: Sample rate.
        min_duration: Preferred minimum speech duration in seconds. The function
            tries to meet this but may return a shorter segment if that's all
            the audio contains. This is a *preference*, not a hard constraint.
        search_limit: Maximum seconds into the audio to search.

    Returns:
        (start_time, end_time) tuple, or None if no speech was found at all.
    """
    # Limit search scope
    max_samples = int(search_limit * sr)
    audio_limited = audio[:max_samples]

    regions = detect_speech_regions(audio_limited, sr)

    if not regions:
        return None

    # ------------------------------------------------------------------
    # Tier 1: Find a single region that meets the preferred duration.
    # Sort by duration (longest first) so we pick the best candidate.
    # ------------------------------------------------------------------
    regions_by_length = sorted(regions, key=lambda r: r[1] - r[0], reverse=True)

    for start, end in regions_by_length:
        duration = end - start
        if duration >= min_duration:
            return (start, end)

    # ------------------------------------------------------------------
    # Tier 2: Use the longest single region if it's at least 10s.
    # A shorter but contiguous region produces a cleaner correlation peak
    # than a sparse accumulation of tiny fragments with gaps.
    # ------------------------------------------------------------------
    longest_start, longest_end = regions_by_length[0]
    longest_duration = longest_end - longest_start

    if longest_duration >= MIN_SINGLE_REGION_DURATION_S:
        return (longest_start, longest_end)
    else:
        # Longest region is very short — fall through to accumulation.
        pass

    # ------------------------------------------------------------------
    # Tier 3: Accumulate consecutive regions that are close together.
    # Walk regions in chronological order. Reset accumulation when a gap
    # exceeds ACCUMULATION_GAP_LIMIT_S.
    # ------------------------------------------------------------------
    total_speech = 0.0
    segment_start: Optional[float] = None
    prev_end: Optional[float] = None

    for start, end in regions:
        if prev_end is not None and (start - prev_end) > ACCUMULATION_GAP_LIMIT_S:
            # Gap too large — the previous cluster wasn't enough. Reset.
            total_speech = 0.0
            segment_start = None
        else:
            # Either first region or gap is small enough — keep accumulating.
            pass

        if segment_start is None:
            segment_start = start
        else:
            # Continuing accumulation from an earlier region.
            pass

        total_speech += (end - start)
        prev_end = end

        if total_speech >= min_duration:
            return (segment_start, end)
        else:
            # Haven't accumulated enough yet — continue to next region.
            pass

    # ------------------------------------------------------------------
    # Fallback: Return whatever we accumulated, even if it's shorter than
    # min_duration. Some speech is better than none — the caller can check
    # correlation confidence to decide if the result is trustworthy.
    # ------------------------------------------------------------------
    if segment_start is not None and total_speech > 0:
        return (segment_start, prev_end)
    else:
        # No speech found at all.
        return None
