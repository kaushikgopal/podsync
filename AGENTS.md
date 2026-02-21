# Podsync — Agent Instructions

CLI tool that aligns multi-track podcast recordings to a master track using
MFCC cross-correlation. Written in Rust.

## Project layout

```
Makefile                     # cargo build + copy binary to scripts/podsync
scripts/
  Cargo.toml                 # Rust project root
  Cargo.lock
  src/
    main.rs                  # clap CLI, orchestration, all user output
    audio.rs                 # decode (symphonia), resample (rubato), write WAV (hound)
    mfcc.rs                  # MFCC pipeline (librosa 0.11.0 parity)
    sync.rs                  # FFT cross-correlation, offset detection, drift
    vad.rs                   # WebRTC VAD speech detection, segment selection
  podsync                    # compiled binary (gitignored)
references/
  DESIGN.md                  # architecture + algorithm overview
  ALGORITHM.md               # MFCC cross-correlation deep dive
  DEPENDENCIES.md            # library rationale
  TROUBLESHOOTING.md         # common failure modes
.agents/
  skills/
    podsync/
      SKILL.md               # AI skill orchestration instructions
      references/             # skill-bundled reference docs
```

## Tech stack

- **Language:** Rust (edition 2021)
- **Build:** cargo + Makefile
- **CLI framework:** clap (derive)
- **Audio decoding:** symphonia (MP3, WAV, FLAC, OGG, AIFF)
- **Resampling:** rubato (sinc interpolation)
- **WAV writing:** hound (24-bit PCM)
- **VAD:** webrtc-vad (Google's C library via FFI)
- **FFT:** realfft (MFCC extraction + cross-correlation)
- **Tests:** built-in `#[cfg(test)]` modules

## How to run

```sh
# build release binary
make

# run
scripts/podsync --help
scripts/podsync --master master.mp3 --tracks track.wav

# run tests
make test

# or directly
cd scripts && cargo test
```

## Architecture

Single-process pipeline, no services, no network:

```
master audio + track audio(s)
  → decode & resample to 44.1kHz mono (symphonia + rubato)
  → VAD on each track (webrtc-vad) → find first speech segment
  → MFCC extraction (manual pipeline matching librosa 0.11.0)
  → FFT cross-correlate MFCCs (realfft) → time offset + confidence
  → drift measurement (correlate near end vs start)
  → pad/trim each track to master length
  → write synced WAV files (hound, 24-bit PCM)
  → write timestamped log file (podsync-<epoch>.log)
```

Entry point: `main()` in `scripts/src/main.rs`.

Core modules: `sync.rs` (offset/drift), `vad.rs` (speech detection),
`audio.rs` (I/O), `mfcc.rs` (feature extraction).

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
| DRIFT_END_WINDOW_S | 120s | window at recording end for drift measurement |

## Editing guidelines

- All source lives under `scripts/src/`. No separate test directory — tests are
  inline `#[cfg(test)]` modules in each file.
- The five modules are intentionally small and single-purpose. Keep them that way.
- `main.rs` owns all user-facing output (`eprintln!`) and orchestration. The other
  modules are pure computation — no I/O side effects except `audio.rs:write_audio`.
- Test with synthetic signals (sine waves, chirps, random noise). 38 tests total.
- Output format is always 44.1kHz 24-bit WAV. Don't add format options.
- Space shuttle style throughout: every `if` has an `else`, every constant is named
  with a rationale comment, every function has doc comments.

## Failure modes to know about

- **"No speech detected"** — VAD found no speech at all in the first 10 minutes.
  The VAD uses a three-tier fallback (single region >=30s, longest region >=10s,
  accumulated nearby regions), so this only triggers when there is zero detected speech.
- **Low confidence (<0.5)** — cross-correlation peak wasn't distinctive. File still
  written, but offset may be wrong.
- **Drift > 1s** — clock rate mismatch between devices. Reported but not corrected.
- WebRTC VAD only accepts 8/16/32/48kHz — `vad.rs` resamples to 16kHz internally.
