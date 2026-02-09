# podsync

Automatically align multi-track podcast recordings to a master track.

## The problem

Many podcasts are recorded as "double-enders" or "triple-enders" — each participant records their own audio locally while a live session captures everyone together. The result:

- A **master track** (the merged live recording with all voices)
- **Individual tracks** per participant (higher quality, isolated audio)

The individual tracks are better for editing (cleaner audio, per-person volume control, noise removal), but they don't start at the same time as the master. One host might join a few seconds late, recording devices have different start times, and clock drift means tracks slowly desync over the course of an hour.

Podsync fixes this. Give it the master track and the individual tracks, and it outputs new WAV files that are time-aligned to the master. Drop them all into your DAW at position 0:00 and they line up.

## How it works

1. **Voice Activity Detection** — Uses WebRTC VAD to find where speech actually occurs in each track. This handles cases where a participant is silent at the start.
2. **MFCC Cross-Correlation** — Extracts Mel-frequency cepstral coefficients (spectral features used in speech recognition) from both the master and each track, then cross-correlates to find the precise time offset. MFCCs are robust to volume differences, EQ differences, and the fact that a single voice needs to match against a mixed master.
3. **Drift Measurement** — Correlates again near the end of the recording and compares the offset to the start. The difference is drift caused by different clock rates between recording devices. Podsync reports the drift so you know if manual adjustment is needed.
4. **Output** — Pads or trims each track to match the master's length and writes synced WAV files.

See [references/ALGORITHM.md](references/ALGORITHM.md) for the full technical deep dive.

## Install

Requires [uv](https://docs.astral.sh/uv/) (Python package manager).

```sh
git clone https://github.com/kaushikgopal/podsync.git
cd podsync

uv tool install scripts/src
```

This puts `podsync` on your PATH (typically at `~/.local/bin/podsync`).

To update after pulling new changes:

```sh
uv tool install --reinstall scripts/src
```

### Run without installing

```sh
cd scripts/src
uv run podsync --help
```

## Usage

```sh
podsync \
  --master /path/to/episode-master.mp3 \
  --tracks /path/to/host1-clean.wav /path/to/host2-clean.wav
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--master` | Yes | — | Master/sync reference track (mp3, wav, aiff, flac, ogg) |
| `--tracks` | Yes | — | Individual tracks to sync (specify multiple) |
| `--sync-window` | No | `120` | Seconds of speech to use for cross-correlation |
| `--output-suffix` | No | `synced` | Suffix appended to output filenames |

### Output

Synced files are written to the same directory as the input tracks:

- Format: `{original_name}-{suffix}.wav`
- Sample rate: 44.1kHz, 24-bit WAV
- Length: matches master track exactly

### Example

```
$ podsync --master ep-master.mp3 --tracks ep-host1-clean.wav ep-host2-clean.wav

Loading master: ep-master.mp3
  Duration: 58m32s at 44100Hz

Processing ep-host1-clean.wav...
  Detecting speech regions... found 47m32s of speech
  Correlating against master... offset: +1.23s (confidence: 0.94)
  Measuring drift... 0.15s at master end

Processing ep-host2-clean.wav...
  Detecting speech regions... found 38m15s of speech
  Correlating against master... offset: +0.87s (confidence: 0.91)
  Measuring drift... 0.08s at master end

============================================================
Summary:
  ep-host1-clean-synced.wav        offset: +1.23s   drift: 0.15s   ✓
  ep-host2-clean-synced.wav        offset: +0.87s   drift: 0.08s   ✓
============================================================

2 tracks synchronized successfully
```

### Verifying results

After syncing, verify in your DAW:

1. Import all `-synced.wav` files
2. Place all at position 0:00
3. Solo each track to confirm voices align
4. Spot-check the middle and end for drift

## AI Skill

This repo doubles as an AI coding agent skill. `SKILL.md` contains orchestration instructions for AI agents (e.g. Claude Code) — scanning episode folders for files, confirming selections with the user, invoking the CLI, and reporting results.

To use as a skill, symlink or copy this repo into your project's skill directory.

## Running tests

```sh
cd scripts/src
uv run pytest
```

Integration tests with real audio files:

```sh
PODSYNC_TEST_DIR=/path/to/episode RUN_INTEGRATION=1 uv run pytest tests/test_integration.py
```

## Dependencies

| Library | Purpose |
|---------|---------|
| [librosa](https://librosa.org/) | Audio loading, MFCC extraction, resampling |
| [scipy](https://scipy.org/) | Cross-correlation via `signal.correlate` |
| [soundfile](https://pysoundfile.readthedocs.io/) | WAV file writing |
| [webrtcvad](https://github.com/wiseman/py-webrtcvad) | Voice activity detection |
| [click](https://click.palletsprojects.com/) | CLI framework |

See [references/DEPENDENCIES.md](references/DEPENDENCIES.md) for detailed rationale.

## License

MIT
