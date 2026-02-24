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
│  - Rust binary (built via cargo)                        │
│  - Explicit parameters only                             │
│  - Audio analysis + sync logic                          │
│  - Outputs synced WAV files + log                       │
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
  --tracks /path/to/host1-clean.wav --tracks /path/to/host2-clean.wav \
  --sync-window 120 \
  --output-suffix synced
```

### Parameters

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--master` | Yes | - | Master/sync reference track |
| `--tracks` | Yes | - | Individual tracks to sync (repeat for multiple) |
| `--sync-window` | No | 120 | Seconds of speech to use for correlation |
| `--output-suffix` | No | `synced` | Suffix for output files |

**Internal parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DRIFT_END_WINDOW_S` | 120s | Window at recording end used for drift measurement |

### Output

- Files written to same directory as input with `-{suffix}.wav` appended
- All outputs: WAV format, 44.1kHz sample rate, 24-bit PCM
- Console report with offset, drift, and any failures
- Timestamped log file next to master

```
Processing host1-src-clean.wav...
  Detecting speech regions... found 47m32s of speech
  Correlating against master... offset: +1.23s (confidence: 0.94)
  Measuring drift... 0.15s at master end

Processing host2-src-clean.wav...
  Detecting speech regions... found 38m15s of speech
  Correlating against master... offset: +0.87s (confidence: 0.91)
  Measuring drift... 0.08s at master end

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
   ├── Select best segment via three-tier fallback:
   │   ├── Tier 1: single region ≥30s (preferred)
   │   ├── Tier 2: longest single region ≥10s
   │   └── Tier 3: accumulated nearby regions ≥30s
   └── If no speech detected at all → FAIL track with message

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

The algorithm uses a three-tier fallback: first looks for a single long region (≥30s), then the longest region ≥10s, then accumulates nearby shorter regions.

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
| No speech detected (in first 10 min) | Skip track, continue others, report failure with reason |
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

### Rust Crates

| Crate | Purpose |
|-------|---------|
| `symphonia` | Audio decoding (MP3, WAV, FLAC, OGG, AIFF) |
| `rubato` | Sample rate conversion (sinc interpolation) |
| `hound` | 24-bit WAV output |
| `webrtc-vad` | Voice activity detection (Google WebRTC C lib via FFI) |
| `realfft` | FFT for MFCC extraction and cross-correlation |
| `clap` | CLI argument parsing (derive macros) |

### Build

Standard cargo project in `scripts/`. `Makefile` at repo root runs
`cargo build --release` and copies the binary to `scripts/podsync`.

## Directory Structure

```
podsync/
├── Makefile                    # Build + copy binary
├── scripts/
│   ├── Cargo.toml              # Rust project root
│   ├── Cargo.lock
│   └── src/
│       ├── main.rs             # CLI entry point
│       ├── audio.rs            # Audio I/O
│       ├── mfcc.rs             # MFCC feature extraction
│       ├── sync.rs             # Cross-correlation logic
│       └── vad.rs              # Voice activity detection
└── .agents/
    └── skills/
        └── podsync/
            ├── SKILL.md            # AI skill orchestration instructions
            ├── references/
            │   ├── DESIGN.md       # This document
            │   ├── ALGORITHM.md    # Deep dive on sync algorithm
            │   ├── TROUBLESHOOTING.md
            │   └── DEPENDENCIES.md # Rust crates and why
            └── assets/
                └── example-output.txt
```

## Future Considerations (Not Implemented)

These were discussed but explicitly deferred:

1. **Auto-drift correction** - Split at silence, realign segments, rejoin. Adds complexity, unnecessary for minor drift.
2. **Glob patterns for tracks** - `--tracks "ep-*-clean*.wav"`. Kept simple with explicit file list.
3. **Multiple output formats** - Always output WAV for simplicity.
