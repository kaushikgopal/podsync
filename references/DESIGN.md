# Podsync Design Document

> Purpose: Automatically synchronize multi-track podcast recordings to a master track

## Problem Statement

When recording podcasts as "double/triple-enders" (each participant records locally), the editor receives:
- A **master track** with all voices merged (from the live session)
- **Individual tracks** for each participant (higher quality, isolated)

The individual tracks need to be aligned to the master so the editor can drop them in at 0:00 and have them synced. Without automation, this requires manual alignment by ear.

**Additional challenges:**
- Audio files may be in different formats (mp3, wav, aiff)
- Volume levels differ between tracks
- Minor clock drift between recording devices
- Some participants may be silent at the start

## Solution Overview

Two components:

```
┌─────────────────────────────────────────────────────────┐
│  AI Skill (/podsync)                                    │
│  - Prompts for episode folder                           │
│  - Auto-detects files using naming conventions          │
│  - Confirms selections with user                        │
│  - Invokes CLI, presents results                        │
└─────────────────────┬───────────────────────────────────┘
                      │ calls
                      ▼
┌─────────────────────────────────────────────────────────┐
│  CLI Tool (podsync)                                     │
│  - Python script (managed by uv)                        │
│  - Explicit parameters only                             │
│  - Audio analysis + sync logic                          │
│  - Outputs synced WAV files + report                    │
└─────────────────────────────────────────────────────────┘
```

## File Naming Conventions

Episode folder structure (e.g., `episodes/42/`):

| Pattern | Description | Example |
|---------|-------------|---------|
| `{ep}-src-ap.mp3` | Master/sync track (Adobe Podcast) | `42-src-ap.mp3` |
| `{ep}-{name}-src-{source}.wav` | Raw individual track | `42-alice-src-logic.wav` |
| `{ep}-{name}-src-clean*.wav` | Cleaned individual track | `42-alice-src-clean.wav` |
| `{ep}-{name}-*-synced.wav` | Output synced track | `42-alice-src-clean-synced.wav` |

## CLI Interface

```bash
podsync \
  --master /path/to/ep-master.mp3 \
  --tracks /path/to/host1-clean.wav /path/to/host2-clean.wav \
  --sync-window 120 \
  --output-suffix synced
```

### Parameters

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--master` | Yes | - | Master/sync reference track |
| `--tracks` | Yes | - | Individual tracks to sync (space-separated) |
| `--sync-window` | No | 120 | Seconds of speech to use for correlation |
| `--output-suffix` | No | `synced` | Suffix for output files |

### Output

- Files written to same directory as input with `-{suffix}.wav` appended
- All outputs: WAV format, 44.1kHz sample rate
- Console report with offset, drift, and any failures

```
Processing host1-src-clean.wav...
  Detecting speech regions... found 47m32s of speech
  Correlating against master... offset: +1.23s (confidence: 0.94)
  Measuring drift... 0.15s at master end
  Writing host1-src-clean-synced.wav

Processing host2-src-clean.wav...
  Detecting speech regions... found 38m15s of speech
  Correlating against master... offset: +0.87s (confidence: 0.91)
  Measuring drift... 0.08s at master end
  Writing host2-src-clean-synced.wav

Summary:
  host1-src-clean-synced.wav       offset: +1.23s   drift: 0.15s   ✓
  host2-src-clean-synced.wav       offset: +0.87s   drift: 0.08s   ✓
```

## Sync Algorithm

### Step-by-Step Flow

```
1. LOAD & NORMALIZE
   ├── Load master track (any format) → resample to 44.1kHz mono
   └── Load individual track → resample to 44.1kHz mono

2. VOICE ACTIVITY DETECTION (on individual track)
   ├── Run VAD to find speech regions
   ├── Find first segment with ≥30s continuous speech
   └── If <30s total speech found → FAIL track with message

3. EXTRACT FEATURES
   ├── Extract MFCCs from individual track's speech region (first 2 min of speech)
   └── Extract MFCCs from corresponding search window in master
       (search window = first 10 min of master, to allow for late starts)

4. CROSS-CORRELATE
   ├── Compute cross-correlation between MFCC sequences
   ├── Find peak → this is the time offset
   └── Confidence = peak height relative to noise floor

5. MEASURE DRIFT
   ├── Correlate again near master's end (last 2 min)
   ├── Compare end-offset vs start-offset
   └── Difference = drift

6. GENERATE OUTPUT
   ├── Apply offset (pad start with silence or trim)
   ├── Trim/pad to match master track length
   └── Write as WAV 44.1kHz
```

### Why MFCCs?

MFCCs (Mel-frequency cepstral coefficients) are used instead of raw waveform comparison because:
- Robust to volume differences between tracks
- Standard for audio fingerprinting and speech recognition
- Better at matching a single voice against a mixed master track
- Less affected by EQ differences between recording setups

### Voice Activity Detection

VAD ensures we correlate speech-to-speech, not silence-to-speech. This handles the case where one host is silent for the first few minutes while another does the intro.

The algorithm finds the first region with at least 30 seconds of continuous speech, then uses that for correlation.

### Drift Measurement

Drift occurs due to different clock rates between recording devices. For a 1-hour podcast with minor drift:
- Start is aligned precisely via cross-correlation
- End may drift by 100-300ms (acceptable for editing)
- The tool reports drift so the editor knows if manual adjustment is needed

**Decision:** Align at start only, report drift. Do not auto-correct drift (adds complexity, usually unnecessary).

### Output Length

All synced tracks are trimmed/padded to match the master track length. This ensures:
- All files are the same length as the master
- Editor can drop all files at 0:00 and they align
- Output matches the expected final episode length

## Failure Handling

| Condition | Behavior |
|-----------|----------|
| Insufficient speech (<30s in first 10 min) | Skip track, continue others, report failure with reason |
| Low correlation confidence (<0.5) | Warn but still output the file |
| Unsupported audio format | Fail with clear error message |

## AI Skill Flow

1. **User invokes** `/podsync`
2. **Skill asks** for episode folder path
3. **Skill scans** folder for:
   - Master: `*-src-ap.mp3` pattern
   - Tracks: `*-src-clean*.wav` pattern
4. **Skill confirms** detected files with user
5. **Skill executes** CLI with detected files
6. **Skill reports** results (offsets, drift, any failures)

### Edge Cases

| Situation | Skill Behavior |
|-----------|----------------|
| No master found | Ask user to specify master file path |
| No cleaned tracks found | Ask if raw tracks should be used instead |
| Multiple potential masters | Ask user to pick |

## Technology Stack

### Python Libraries

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading, MFCC extraction, resampling |
| `scipy` | Cross-correlation via `scipy.signal.correlate` |
| `webrtcvad` or `silero-vad` | Voice activity detection |
| `soundfile` | WAV output |

### Package Management

Using `uv` (Astral) instead of pip for:
- Isolated environments without virtualenv headaches
- Fast, reliable dependency resolution
- Single binary, no pip hell
- Run with `uvx podsync` or install with `uv tool install`

## Directory Structure

```
podsync/
├── SKILL.md                    # Main skill instructions
├── scripts/
│   ├── podsync                 # Wrapper script (runs uv, handles build)
│   └── src/
│       ├── pyproject.toml      # Python project config (uv compatible)
│       ├── podsync/
│       │   ├── __init__.py
│       │   ├── cli.py          # CLI entry point
│       │   ├── sync.py         # Cross-correlation logic
│       │   ├── vad.py          # Voice activity detection
│       │   └── audio.py        # Audio loading/writing
│       └── tests/
│           └── test_sync.py
├── references/
│   ├── DESIGN.md               # This document
│   ├── ALGORITHM.md            # Deep dive on sync algorithm
│   ├── TROUBLESHOOTING.md      # Common issues and solutions
│   └── DEPENDENCIES.md         # Python libraries and why
└── assets/
    └── example-output.txt      # Sample CLI output
```

## Future Considerations (Not Implemented)

These were discussed but explicitly deferred:

1. **Auto-drift correction** - Split at silence, realign segments, rejoin. Adds complexity, unnecessary for minor drift.
2. **Glob patterns for tracks** - `--tracks "ep-*-clean*.wav"`. Kept simple with explicit file list.
3. **Multiple output formats** - Always output WAV for simplicity.
