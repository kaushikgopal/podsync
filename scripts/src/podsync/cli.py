"""CLI entry point for podsync."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click
import numpy as np

from .audio import load_audio, write_audio
from .vad import find_first_speech_segment
from .sync import find_offset, compute_drift


@dataclass
class TrackResult:
    """Result of processing a single track."""
    path: Path
    output_path: Optional[Path] = None
    offset: Optional[float] = None
    drift: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.output_path is not None


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds >= 0:
        return f"+{seconds:.2f}s"
    return f"{seconds:.2f}s"


def process_track(
    master_audio: np.ndarray,
    master_sr: int,
    track_path: Path,
    output_suffix: str,
    sync_window: float,
    all_track_audios: List[np.ndarray],
) -> TrackResult:
    """
    Process a single track: load, find offset, measure drift, write output.

    Args:
        master_audio: Loaded master audio
        master_sr: Master sample rate
        track_path: Path to track file
        output_suffix: Suffix for output file
        sync_window: Seconds of speech to use for correlation
        all_track_audios: List to append loaded audio for length calculation

    Returns:
        TrackResult with processing outcome
    """
    result = TrackResult(path=track_path)

    click.echo(f"\nProcessing {track_path.name}...")

    # Load track
    try:
        track_audio, track_sr = load_audio(track_path, target_sr=master_sr)
    except Exception as e:
        result.error = f"Failed to load: {e}"
        click.echo(f"  ERROR: {result.error}")
        return result

    # Find speech region
    click.echo("  Detecting speech regions...", nl=False)
    speech_segment = find_first_speech_segment(
        track_audio, track_sr,
        min_duration=30.0,
        search_limit=600.0
    )

    if speech_segment is None:
        result.error = "Insufficient speech detected (< 30s in first 10 min)"
        click.echo(f" FAILED")
        click.echo(f"  ERROR: {result.error}")
        return result

    speech_start, speech_end = speech_segment
    speech_duration = speech_end - speech_start
    minutes = int(speech_duration // 60)
    seconds = int(speech_duration % 60)
    click.echo(f" found {minutes}m{seconds:02d}s of speech")

    # Extract speech portion for correlation
    speech_start_samples = int(speech_start * track_sr)
    speech_samples = int(sync_window * track_sr)
    track_speech = track_audio[speech_start_samples:speech_start_samples + speech_samples]

    # Find offset
    click.echo("  Correlating against master...", nl=False)
    offset, confidence = find_offset(
        master_audio, track_speech, track_sr,
        search_window=600.0,
        correlation_window=sync_window,
    )

    # Adjust offset to account for speech_start
    total_offset = offset - speech_start

    result.offset = total_offset
    result.confidence = confidence
    click.echo(f" offset: {format_time(total_offset)} (confidence: {confidence:.2f})")

    if confidence < 0.5:
        click.echo("  WARNING: Low confidence - sync may be inaccurate")

    # Measure drift
    click.echo("  Measuring drift...", nl=False)
    drift = compute_drift(master_audio, track_audio, track_sr, total_offset)
    result.drift = drift
    click.echo(f" {abs(drift):.2f}s at master end")

    # Store audio for length calculation
    all_track_audios.append((track_audio, total_offset))

    return result


@click.command()
@click.option(
    '--master',
    required=True,
    type=click.Path(exists=True),
    help='Master/sync reference track'
)
@click.option(
    '--tracks',
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help='Individual tracks to sync (can specify multiple)'
)
@click.option(
    '--sync-window',
    default=120,
    type=int,
    help='Seconds of speech to use for correlation (default: 120)'
)
@click.option(
    '--output-suffix',
    default='synced',
    help='Suffix for output files (default: synced)'
)
def main(master: str, tracks: tuple, sync_window: int, output_suffix: str):
    """
    Synchronize podcast tracks to a master recording.

    Aligns individual participant tracks to a master track using
    MFCC-based cross-correlation, then outputs synced WAV files
    that can be dropped into a DAW at position 0:00.
    """
    master_path = Path(master)
    track_paths = [Path(t) for t in tracks]

    click.echo(f"Loading master: {master_path.name}")

    try:
        master_audio, master_sr = load_audio(master_path, target_sr=44100)
    except Exception as e:
        click.echo(f"ERROR: Failed to load master: {e}", err=True)
        sys.exit(1)

    master_duration = len(master_audio) / master_sr
    minutes = int(master_duration // 60)
    seconds = int(master_duration % 60)
    click.echo(f"  Duration: {minutes}m{seconds:02d}s at {master_sr}Hz")

    # Process each track
    results: List[TrackResult] = []
    all_track_audios: List[tuple] = []  # (audio, offset) pairs

    for track_path in track_paths:
        result = process_track(
            master_audio, master_sr,
            track_path, output_suffix, sync_window,
            all_track_audios
        )
        results.append(result)

    # Output length matches master length
    max_length = len(master_audio)

    # Write output files
    click.echo("\nWriting output files...")

    for result, (audio, offset) in zip(results, all_track_audios):
        if result.error:
            continue

        # Apply offset and pad to max_length
        offset_samples = int(offset * master_sr)

        if offset_samples >= 0:
            # Pad start with silence
            padded = np.zeros(max_length, dtype=np.float32)
            end_idx = min(offset_samples + len(audio), max_length)
            copy_len = end_idx - offset_samples
            padded[offset_samples:end_idx] = audio[:copy_len]
        else:
            # Trim start
            trim = abs(offset_samples)
            padded = np.zeros(max_length, dtype=np.float32)
            copy_len = min(len(audio) - trim, max_length)
            padded[:copy_len] = audio[trim:trim + copy_len]

        # Generate output path
        output_path = result.path.parent / f"{result.path.stem}-{output_suffix}.wav"
        write_audio(output_path, padded, master_sr)
        result.output_path = output_path
        click.echo(f"  Writing {output_path.name}")

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("Summary:")

    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count

    for result in results:
        if result.success:
            click.echo(
                f"  {result.output_path.name:<40} "
                f"offset: {format_time(result.offset)}   "
                f"drift: {result.drift:.2f}s   ✓"
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
