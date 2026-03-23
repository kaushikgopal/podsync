---
name: podsync-voice-tracks
description: >-
  Synchronize multi-track podcast recordings to a master track. Use when user says
  "sync podcast", "align podcast tracks", "podsync", or needs to prepare individual
  voice recordings for editing by aligning them to a master recording.
---

# Podsync: Podcast Track Synchronizer

Automatically align individual participant audio tracks to a master recording so they can be dropped into a DAW at position 0:00 and be perfectly synchronized.

## Reference Files

- **[references/DESIGN.md](references/DESIGN.md)** — Full design document with architecture decisions
- **[references/ALGORITHM.md](references/ALGORITHM.md)** — Deep dive on MFCC cross-correlation sync algorithm
- **[references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md)** — Common issues and solutions
- **[references/DEPENDENCIES.md](references/DEPENDENCIES.md)** — Rust crates and rationale
- **[assets/example-output.txt](assets/example-output.txt)** — Sample CLI output

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

### Step 3: Convert Non-WAV Tracks

Before syncing, automatically convert any individual tracks that are not `.wav` to WAV using ffmpeg. Do this silently without asking the user.

```bash
ffmpeg -i "{track.aifc}" -c:a pcm_s24le "{track}.wav"
```

- Convert in-place: output file sits next to the original with the same base name and `.wav` extension
- Supported input formats: `.aifc`, `.aiff`, `.m4a`, `.flac`, `.ogg`, `.mp3`
- Use `pcm_s24le` (24-bit PCM) to preserve quality
- After conversion, use the `.wav` path for all subsequent steps

### Step 4: Confirm with User

Present findings and confirm:

```
Found in {folder}:
  Master: {master_file}
  Tracks to sync:
    - {track1}  [converted from .aifc]
    - {track2}

Proceed with sync?
```

**Edge cases:**
- No master found → Ask user to specify the master file path
- No cleaned tracks → Use raw tracks (`*-src-*.wav` without "clean") automatically
- Multiple potential masters → Ask user to select one

### Step 5: Run Sync

Execute the repo-local CLI:

```bash
scripts/podsync \
  --master "{master_path}" \
  --tracks "{track1}" --tracks "{track2}" \
  --sync-window 120 \
  --output-suffix synced
```

### Step 6: Report Results

Display the CLI output showing:
- Offset applied to each track
- Drift measured for each track
- Any failures with reasons
- Log file location

## CLI Reference

```bash
scripts/podsync \
  --master <path>           # Master/sync reference track (required)
  --tracks <path>           # Individual tracks to sync (required, repeat for multiple)
  --sync-window <seconds>   # Seconds of speech for correlation (default: 120)
  --output-suffix <suffix>  # Output file suffix (default: synced)
```

## Output

- Synced files written to same directory as input
- Filename format: `{original}-{suffix}.wav`
- All outputs: 44.1kHz 24-bit WAV
- All outputs match master track length
- Timestamped log file written next to master

## Prerequisites

- **Repo-local binary** built via `make` in the repo root at `scripts/podsync`
- See the repo README for build instructions

## File Naming Conventions

For auto-detection, the skill expects:

| Pattern | Description |
|---------|-------------|
| `{ep}-src-ap.mp3` | Master track (Adobe Podcast) |
| `{ep}-{name}-src-clean*.wav` | Cleaned individual tracks |
| `{ep}-{name}-src-*.wav` | Raw individual tracks (fallback) |
