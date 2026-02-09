---
name: podsync
description: >-
  Synchronize multi-track podcast recordings to a master track. Use when user says
  "sync podcast", "align podcast tracks", "podsync", or needs to prepare individual
  voice recordings for editing by aligning them to a master recording.
---

# Podsync: Podcast Track Synchronizer

Automatically align individual participant audio tracks to a master recording so they can be dropped into a DAW at position 0:00 and be perfectly synchronized.

## Reference Files

- **references/DESIGN.md** — Full design document with architecture decisions
- **references/ALGORITHM.md** — Deep dive on MFCC cross-correlation sync algorithm
- **references/TROUBLESHOOTING.md** — Common issues and solutions
- **references/DEPENDENCIES.md** — Python libraries and rationale

## Workflow

### Step 1: Get Episode Folder

Ask the user for the episode folder path:

```
Which episode folder should I sync?
```

### Step 2: Scan for Files

Scan the folder for:
- **Master track**: Files matching `*-src-ap.mp3` or `*-src-*.mp3`
- **Individual tracks**: Files matching `*-src-clean*.wav`

### Step 3: Confirm with User

Present findings and confirm:

```
Found in {folder}:
  Master: {master_file}
  Tracks to sync:
    - {track1}
    - {track2}

Proceed with sync?
```

**Edge cases:**
- No master found → Ask user to specify the master file path
- No cleaned tracks → Ask if raw tracks (`*-src-*.wav` without "clean") should be used
- Multiple potential masters → Ask user to select one

### Step 4: Run Sync

Execute the CLI:

```bash
podsync \
  --master "{master_path}" \
  --tracks "{track1}" "{track2}" \
  --sync-window 120 \
  --output-suffix synced
```

### Step 5: Report Results

Display the CLI output showing:
- Offset applied to each track
- Drift measured for each track
- Any failures with reasons

## CLI Reference

```bash
podsync \
  --master <path>           # Master/sync reference track (required)
  --tracks <path> [<path>]  # Individual tracks to sync (required, multiple)
  --sync-window <seconds>   # Seconds of speech for correlation (default: 120)
  --output-suffix <suffix>  # Output file suffix (default: synced)
```

## Output

- Synced files written to same directory as input
- Filename format: `{original}-{suffix}.wav`
- All outputs: WAV format at 44.1kHz
- All outputs match master track length

## Prerequisites

- **podsync** CLI installed globally via `uv tool install`
- See the repo README for install instructions

## File Naming Conventions

For auto-detection, the skill expects:

| Pattern | Description |
|---------|-------------|
| `{ep}-src-ap.mp3` | Master track (Adobe Podcast) |
| `{ep}-{name}-src-clean*.wav` | Cleaned individual tracks |
| `{ep}-{name}-src-*.wav` | Raw individual tracks (fallback) |
