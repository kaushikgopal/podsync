"""CLI entry point for podsync."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click
import numpy as np

from .audio import (
    TARGET_SAMPLE_RATE,
    apply_offset,
    load_audio,
    seconds_to_samples,
    write_audio,
)
from .sync import LOW_CONFIDENCE_THRESHOLD, compute_drift, find_offset
from .vad import (
    PREFERRED_SPEECH_DURATION_S,
    VAD_SEARCH_LIMIT_S,
    find_first_speech_segment,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TrackResult:
    """Result of processing a single track.

    On success: audio, offset, drift, confidence, and output_path are populated.
    On failure: error is populated, everything else is None.
    """
    path: Path
    audio: Optional[np.ndarray] = None
    offset: Optional[float] = None
    drift: Optional[float] = None
    confidence: Optional[float] = None
    output_path: Optional[Path] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.output_path is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format a signed offset as a human-readable string."""
    if seconds > 0:
        return f"+{seconds:.2f}s"
    elif seconds < 0:
        return f"{seconds:.2f}s"
    else:
        # Exactly zero — no sign prefix.
        return "0.00s"


def format_duration(seconds: float) -> str:
    """Format a duration as NmSSs."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs:02d}s"


# ---------------------------------------------------------------------------
# Per-track processing
# ---------------------------------------------------------------------------

def process_track(
    master_audio: np.ndarray,
    master_sr: int,
    track_path: Path,
    sync_window: float,
) -> TrackResult:
    """Process a single track: load, detect speech, correlate, measure drift.

    Returns a TrackResult containing the loaded audio and all computed metadata.
    The caller is responsible for writing the output file.
    """
    result = TrackResult(path=track_path)

    click.echo(f"\nProcessing {track_path.name}...")

    # --- Load track --------------------------------------------------------
    try:
        track_audio, track_sr = load_audio(track_path, target_sr=master_sr)
    except Exception as e:
        result.error = f"Failed to load: {e}"
        click.echo(f"  ERROR: {result.error}")
        return result

    # --- Detect speech region ----------------------------------------------
    click.echo("  Detecting speech regions...", nl=False)
    speech_segment = find_first_speech_segment(
        track_audio, track_sr,
        min_duration=PREFERRED_SPEECH_DURATION_S,
        search_limit=VAD_SEARCH_LIMIT_S,
    )

    if speech_segment is None:
        result.error = (
            f"Insufficient speech detected "
            f"(< {PREFERRED_SPEECH_DURATION_S:.0f}s in first "
            f"{VAD_SEARCH_LIMIT_S / 60:.0f} min)"
        )
        click.echo(" FAILED")
        click.echo(f"  ERROR: {result.error}")
        return result
    else:
        speech_start, speech_end = speech_segment
        speech_duration = speech_end - speech_start
        click.echo(f" found {format_duration(speech_duration)} of speech")

    # --- Extract speech portion for correlation ----------------------------
    speech_start_samples = seconds_to_samples(speech_start, track_sr)
    speech_samples = seconds_to_samples(sync_window, track_sr)
    track_speech = track_audio[speech_start_samples:speech_start_samples + speech_samples]

    # --- Find offset -------------------------------------------------------
    click.echo("  Correlating against master...", nl=False)
    offset, confidence = find_offset(
        master_audio, track_speech, track_sr,
        search_window=VAD_SEARCH_LIMIT_S,
        correlation_window=sync_window,
    )

    # Adjust offset to account for where speech starts within the track.
    # The correlation found where track_speech appears in the master, but
    # track_speech starts at speech_start within the track. So the full
    # track's offset is: (where speech was found in master) - speech_start.
    total_offset = offset - speech_start

    result.audio = track_audio
    result.offset = total_offset
    result.confidence = confidence
    click.echo(f" offset: {format_time(total_offset)} (confidence: {confidence:.2f})")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        click.echo("  WARNING: Low confidence — sync may be inaccurate")
    else:
        # Confidence is acceptable — no additional action needed.
        pass

    # --- Measure drift -----------------------------------------------------
    click.echo("  Measuring drift...", nl=False)
    drift = compute_drift(master_audio, track_audio, track_sr, total_offset)

    if drift is not None:
        result.drift = drift
        click.echo(f" {abs(drift):.2f}s at master end")
    else:
        result.drift = None
        click.echo(" N/A (audio too short to measure)")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    '--master',
    required=True,
    type=click.Path(exists=True),
    help='Master/sync reference track',
)
@click.option(
    '--tracks',
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help='Individual tracks to sync (can specify multiple)',
)
@click.option(
    '--sync-window',
    default=120.0,
    type=float,
    help='Seconds of speech to use for correlation (default: 120)',
)
@click.option(
    '--output-suffix',
    default='synced',
    help='Suffix for output files (default: synced)',
)
def main(master: str, tracks: tuple, sync_window: float, output_suffix: str):
    """Synchronize podcast tracks to a master recording.

    Aligns individual participant tracks to a master track using
    MFCC-based cross-correlation, then outputs synced WAV files
    that can be dropped into a DAW at position 0:00.
    """
    master_path = Path(master)
    track_paths = [Path(t) for t in tracks]

    # --- Load master -------------------------------------------------------
    click.echo(f"Loading master: {master_path.name}")

    try:
        master_audio, master_sr = load_audio(master_path, target_sr=TARGET_SAMPLE_RATE)
    except Exception as e:
        click.echo(f"ERROR: Failed to load master: {e}", err=True)
        sys.exit(1)

    master_duration = len(master_audio) / master_sr
    click.echo(f"  Duration: {format_duration(master_duration)} at {master_sr}Hz")

    # --- Process each track ------------------------------------------------
    results: List[TrackResult] = []

    for track_path in track_paths:
        result = process_track(master_audio, master_sr, track_path, sync_window)
        results.append(result)

    # --- Write output files ------------------------------------------------
    # All output files match the master track's length so they can be dropped
    # into a DAW at position 0:00.
    max_length = len(master_audio)

    click.echo("\nWriting output files...")

    for result in results:
        if result.error is not None:
            # Track failed during processing — skip.
            continue
        else:
            # Track processed successfully — write aligned output.
            pass

        offset_samples = seconds_to_samples(result.offset, master_sr)

        try:
            padded = apply_offset(result.audio, offset_samples, max_length)
        except ValueError as e:
            click.echo(f"  ERROR writing {result.path.name}: {e}")
            result.error = str(e)
            result.output_path = None
            continue

        output_path = result.path.parent / f"{result.path.stem}-{output_suffix}.wav"
        write_audio(output_path, padded, master_sr)
        result.output_path = output_path
        click.echo(f"  Writing {output_path.name}")

    # --- Summary -----------------------------------------------------------
    click.echo("\n" + "=" * 60)
    click.echo("Summary:")

    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count

    for result in results:
        if result.success:
            drift_str = (
                f"drift: {result.drift:.2f}s"
                if result.drift is not None
                else "drift: N/A"
            )
            click.echo(
                f"  {result.output_path.name:<40} "
                f"offset: {format_time(result.offset)}   "
                f"{drift_str}   ✓"
            )
        else:
            click.echo(f"  {result.path.name:<40} FAILED: {result.error}")

    click.echo("=" * 60)

    if fail_count > 0:
        click.echo(f"\n{success_count} succeeded, {fail_count} failed")
        sys.exit(1)
    else:
        click.echo(f"\n{success_count} tracks synchronized successfully")


if __name__ == '__main__':
    main()
