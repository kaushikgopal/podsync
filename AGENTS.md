# Podsync — Agent Instructions

CLI tool that aligns multi-track podcast recordings to a master track using
MFCC cross-correlation. Written in Python, managed with `uv`.

## Project layout

```
scripts/src/
  pyproject.toml          # dependencies, build config (hatchling)
  podsync/
    cli.py                # click-based CLI entry point (main)
    sync.py               # MFCC extraction + cross-correlation offset/drift
    vad.py                # WebRTC VAD speech region detection
    audio.py              # librosa load / soundfile write wrappers
  tests/
    test_cli.py           # CLI argument validation
    test_sync.py          # offset/drift unit tests
    test_vad.py           # VAD unit tests
    test_audio.py         # load/write round-trip tests
    test_integration.py   # real-file tests (env-gated)
references/
  DESIGN.md               # architecture + algorithm overview
  ALGORITHM.md            # MFCC cross-correlation deep dive
  DEPENDENCIES.md         # library rationale
  TROUBLESHOOTING.md      # common failure modes
SKILL.md                  # AI skill orchestration instructions
```

## Tech stack

- **Language:** Python 3.11+
- **Package manager:** uv (Astral) — no pip, no virtualenv
- **Build backend:** hatchling
- **CLI framework:** click
- **Audio:** librosa (load/MFCC), scipy (cross-correlation), soundfile (WAV write), webrtcvad (VAD)
- **Tests:** pytest

## How to run

```sh
# install globally
uv tool install scripts/src

# or run without installing
cd scripts/src && uv run podsync --help

# run tests
cd scripts/src && uv run pytest

# integration tests (requires real audio files)
cd scripts/src && PODSYNC_TEST_DIR=/path/to/episode RUN_INTEGRATION=1 uv run pytest tests/test_integration.py
```

## Architecture

Single-process pipeline, no services, no network:

```
master audio + track audio(s)
  → load & resample to 44.1kHz mono (librosa)
  → VAD on each track (webrtcvad) → find first speech segment
  → MFCC extraction (librosa) on speech window
  → cross-correlate MFCCs (scipy) → time offset + confidence
  → drift measurement (correlate near end vs start)
  → pad/trim each track to master length
  → write synced WAV files (soundfile, 24-bit PCM)
```

Entry point: `podsync.cli:main` (registered in `pyproject.toml` `[project.scripts]`).

Core modules: `sync.py` (offset/drift), `vad.py` (speech detection), `audio.py` (I/O).

## Key parameters

| Parameter | Default | Where |
|-----------|---------|-------|
| `--sync-window` | 120s | seconds of speech for correlation |
| `--output-suffix` | `synced` | appended to output filenames |
| search_window | 600s (10 min) | how far into master to search (`find_offset`) |
| min_duration (VAD) | 30s | minimum speech for valid segment |
| n_mfcc | 20 | MFCC coefficients |
| hop_length | 512 | STFT hop (~11.6ms at 44.1kHz) |
| VAD aggressiveness | 2 | WebRTC VAD level (0–3) |

## Editing guidelines

- All source lives under `scripts/src/podsync/`. Tests under `scripts/src/tests/`.
- The four modules are intentionally small and single-purpose. Keep them that way.
- `cli.py` owns all user-facing output (`click.echo`) and orchestration. The other
  modules are pure computation — no I/O side effects except `audio.py:write_audio`.
- Test with synthetic signals (sine waves, chirps, random noise). Integration tests
  are gated behind `RUN_INTEGRATION=1` and require real audio files.
- Output format is always 44.1kHz 24-bit WAV. Don't add format options.

## Failure modes to know about

- **"Insufficient speech detected"** — VAD couldn't find 30s of continuous speech
  in the first 10 minutes of a track.
- **Low confidence (<0.5)** — cross-correlation peak wasn't distinctive. File still
  written, but offset may be wrong.
- **Drift > 1s** — clock rate mismatch between devices. Reported but not corrected.
- WebRTC VAD only accepts 8/16/32/48kHz — `vad.py` resamples to 16kHz internally.
