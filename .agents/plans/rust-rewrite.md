# Plan: Port podsync from Python to Rust

## Context

podsync is a CLI tool that aligns multi-track podcast recordings to a master track using MFCC cross-correlation. The Python implementation is complete and tested (15 tests passing). The goal is a Rust port that is hyper-legible — space shuttle style programming throughout. Every branch explicit, every constant named and documented, every decision commented.

The Rust port will eventually replace the Python code entirely.

---

## Definition of Done (parity)

Rust is "done" when all of the following are true:

- **CLI parity:** flags and semantics match the Python CLI (`--master`, `--tracks` (repeatable), `--sync-window`, `--output-suffix`).
- **Exit codes:** exit 0 only if all tracks succeed; exit 1 if the master fails to load or if any track fails.
- **Outputs:** mono 44.1kHz 24-bit WAV, one output per input track, **exactly** master length, named `<stem>-<output-suffix>.wav` in the same directory as the input track.
- **Offset semantics:** positive offset means track content appears later in master (track started late). Negative means track started early.
- **Drift semantics:** drift is the measured end-of-master alignment delta (same meaning as Python `compute_drift`).
- **Numeric tolerances:** on synthetic tests, offset within ~0.2s, drift within ~0.5s (matching the current Python tests' expectations).
- **Real-file sanity:** on at least one real episode (when available), Rust and Python offsets agree within ~0.1–0.2s and the resulting WAVs line up in a DAW.

Reference behavior is pinned to the current Python dependency set (as of 2026-02-10):
- `numpy 2.3.5`, `scipy 1.17.0`, `librosa 0.11.0`, `soundfile 0.13.1`, `webrtcvad 2.0.10`

---

## Project layout

```
Cargo.toml                   # Rust project root
Makefile                     # cargo build + copy binary to scripts/podsync-rs
src/
  main.rs                    # CLI (clap), orchestration, all user output
  audio.rs                   # load (symphonia+rubato), write (hound), apply_offset
  vad.rs                     # speech detection (webrtc-vad), segment selection
  sync.rs                    # cross-correlation, offset detection, drift
  mfcc.rs                    # MFCC pipeline (librosa 0.11.0 parity)
scripts/
  podsync                    # existing bash wrapper that runs Python via uv (do not overwrite)
  podsync-rs                 # compiled Rust binary (gitignored)
  src/                       # Python (existing, untouched during port)
```

Add to `.gitignore`:
- `scripts/podsync-rs`
- `target/`

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `clap` (derive) | 4.5 | CLI argument parsing |
| `symphonia` | 0.5 | Decode MP3/WAV/FLAC/OGG/AIFF |
| `rubato` | 1.0 | Resample to 44.1kHz |
| `hound` | 3.5 | Read/write 24-bit WAV |
| `webrtc-vad` | 0.4 | WebRTC VAD (FFI to C lib) |
| `realfft` | 3.5 | FFT primitives for MFCC + correlation |

Dev-dependencies (keep minimal):
- `tempfile` (or equivalent) for file-based WAV roundtrips
- `approx` (or hand-rolled helpers) for float comparisons

---

## Key porting decisions

- **Audio buffers are `f32`**, but use **`f64` accumulators** for sums/means/stddevs where it improves parity and stability.
- **Signed sample counts:** use a signed `seconds_to_samples(seconds: f64, sr: u32) -> i64` (Python returns `int`, and offsets are signed). Convert to `usize` only after explicit bounds checks.
- **MFCC parity is explicit:** the Rust MFCC implementation must match `librosa.feature.mfcc` default behavior for `librosa 0.11.0` (details in Step 3).
- **Correlation parity is tested:** `correlate(a, b)` must match `scipy.signal.correlate(a, b, mode='full')` (lag indexing included). Keep a tiny O(n²) reference implementation for tests.
- **Tests live inline** in `#[cfg(test)]` modules unless integration tests become unavoidable.
- **No `anyhow`/`thiserror`** unless error handling becomes unwieldy. Start with a small `PodsyncError` enum and `Display`.

---

## Implementation order

### Step 0: Snapshot Python reference behavior

- Record dependency versions (above) and keep them in the Rust repo docs (e.g. `references/PORTING_NOTES.md`).
- Run `cd scripts/src && uv run pytest` to confirm the Python baseline stays green.

### Step 1: Scaffold

- Create branch `kg/rust-port`
- Create `Cargo.toml` with dependencies
- Create `Makefile` (`cargo build --release`, copy binary to `scripts/podsync-rs`)
- Create empty module files (`main.rs`, `audio.rs`, `vad.rs`, `sync.rs`, `mfcc.rs`)
- Update `.gitignore`
- Verify `make` compiles an empty binary

### Step 2: `audio.rs` — Audio I/O and sample manipulation

Port behavior from: `scripts/src/podsync/audio.py`

Functions:
- `seconds_to_samples(seconds: f64, sr: u32) -> i64` — **round-based** conversion (signed)
- `load_audio(path: &Path, target_sr: u32) -> Result<(Vec<f32>, u32), PodsyncError>`
  - symphonia decode
  - convert to `f32`
  - downmix to mono **before** resampling (match `librosa.load(..., mono=True)`)
  - resample with rubato **only if needed** (`decoded_sr != target_sr`)
- `write_audio(path: &Path, audio: &[f32], sr: u32) -> Result<(), PodsyncError>` — 24-bit WAV via hound
- `apply_offset(audio: &[f32], offset_samples: i64, target_length: usize) -> Result<Vec<f32>, PodsyncError>`
  - positive offset beyond `target_length` returns all-silence (match Python behavior)
  - negative offset that trims away the entire signal returns error

Constants:
- `TARGET_SAMPLE_RATE: u32 = 44_100`

Tests:
- `test_load_audio_resamples_and_downmixes` — 48k stereo → 44.1k mono duration preserved (mirrors Python intent)
- `test_write_audio_roundtrip_wav` — write then read back, compare within quantization tolerance
- `test_apply_offset_positive_and_negative` — match Python edge cases

### Step 3: `mfcc.rs` — MFCC feature extraction (librosa 0.11.0 parity)

Python reference is `scripts/src/podsync/sync.py:extract_mfcc`, which calls:
- `librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=..., hop_length=...)`

For `librosa 0.11.0`, MFCC is:
1. `melspectrogram(y=..., sr=..., norm='slaney', **kwargs)` with defaults:
   - `n_fft=2048`, `hop_length=512`, `win_length=n_fft`
   - `window='hann'`, `center=True`, `pad_mode='constant'` (zero padding)
   - `power=2.0`
   - mel filterbank: `n_mels=128`, `fmin=0.0`, `fmax=sr/2`, `htk=False`, `norm='slaney'`
2. `power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0)`
3. `dct(type=2, norm='ortho', axis=mel)` and keep first `n_mfcc`

Implementation notes:
- Treat this as a **parity problem**, not just "an MFCC implementation".
- Centering/padding and the exact `power_to_db` behavior are the common divergence points.

Functions:
- `stft_power(y: &[f32], n_fft: usize, hop_length: usize, center: bool, pad_mode_zero: bool) -> Vec<Vec<f32>>`
- `create_hann_window(size: usize) -> Vec<f32>`
- `create_mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<Vec<f32>>` (Slaney normalization)
- `power_to_db(power: &[f32], amin: f32, top_db: f32) -> Vec<f32>` (ref=1.0)
- `dct_ii_ortho(input: &[f32], n_output: usize) -> Vec<f32>`
- `extract_mfcc(audio: &[f32], sr: u32, n_mfcc: usize, hop_length: usize) -> Vec<Vec<f32>>` — returns `[n_mfcc][time_frames]`

Tests:
- `test_extract_mfcc_shape` — dimensions match expectation
- `test_mfcc_nonzero_for_rich_signal` — sanity check
- **Golden parity test:** generate MFCC output using Python/librosa for a fixed deterministic signal, store it as a small fixture, and assert Rust matches within a tight tolerance.

### Step 4: `vad.rs` — Voice activity detection

Port from: `scripts/src/podsync/vad.py`

Behavioral requirements:
- Resample to 16kHz for VAD only when needed.
- Clip to `[-1, 1]`, scale to int16 using `i16::MAX`, and run 30ms frames.
- Returned timestamps are in seconds on the **original** audio timeline (even if resampled internally).

Constants (mirror Python):
- `SPEECH_MERGE_GAP_S: f64 = 0.3`
- `MIN_SINGLE_REGION_DURATION_S: f64 = 10.0`
- `ACCUMULATION_GAP_LIMIT_S: f64 = 2.0`
- `VAD_SEARCH_LIMIT_S: f64 = 600.0`
- `PREFERRED_SPEECH_DURATION_S: f64 = 30.0`
- `WEBRTC_RESAMPLE_TARGET: u32 = 16_000`
- `WEBRTC_AGGRESSIVENESS: i32 = 2`
- `WEBRTC_FRAME_DURATION_MS: u32 = 30`

Tests (mirror `scripts/src/tests/test_vad.py`):
- detects speech in noise bursts
- empty for silence
- finds segment meeting min duration

### Step 5: `sync.rs` — Cross-correlation and offset/drift (SciPy parity)

Port from: `scripts/src/podsync/sync.py`

Core parity points:
- Per-coefficient normalization: subtract mean, divide by (std + EPSILON).
- Cross-correlation: must match `scipy.signal.correlate(m, t, mode='full')`.
  - For real 1-D signals, this is equivalent to `convolve(m, reverse(t))`.
  - Zero-lag index is `len(t) - 1` (matches Python logic).
- Confidence: peak exclusion radius and ratio mapping match Python.

Functions:
- `correlate_full(a: &[f32], b: &[f32]) -> Vec<f32>` — FFT implementation
- `naive_correlate_full(a: &[f32], b: &[f32]) -> Vec<f32>` — tests only
- `find_offset(...) -> (f64, f64)` — returns (offset_seconds, confidence)
- `compute_drift(...) -> Option<f64>`

Constants (mirror Python, with float32 adjustment):
- `N_MFCC_COEFFICIENTS: usize = 20`
- `HOP_LENGTH: usize = 512`
- `CORRELATION_SEARCH_WINDOW_S: f64 = 600.0`
- `EPSILON: f32 = 1e-6`  (Python note: 1e-8 for float64; 1e-6 for float32)
- `PEAK_EXCLUSION_RADIUS: usize = 50`
- `CONFIDENCE_SENSITIVITY: f64 = 0.5`
- `LOW_CONFIDENCE_THRESHOLD: f64 = 0.5`

Tests (mirror `scripts/src/tests/test_sync.py` + add parity guards):
- `test_correlate_matches_naive_small`
- `test_finds_positive_offset`
- `test_finds_near_zero_offset`
- `test_drift_returns_none_for_short_audio`
- `test_drift_near_zero_for_same_signal`

### Step 6: `main.rs` — CLI and orchestration

Port from: `scripts/src/podsync/cli.py`

Must-match behaviors:
- Speech-start adjustment: `total_offset = offset - speech_start`.
- Output length equals master length.
- Any failed track causes overall exit code 1 (but still write successful outputs).

Structures:
- `Cli` (clap derive)
- `TrackResult` (mirrors Python dataclass semantics)

Tests:
- clap validation via `try_parse_from`
- formatting helpers parity (`format_time`, `format_duration`)

### Step 7: Integration verification

- `make` produces `scripts/podsync-rs`
- `scripts/podsync-rs --help` shows the expected options
- `cargo test` passes
- Compare behavior on a real audio file (if available):
  - `scripts/podsync --master ... --tracks ...`  (Python wrapper)
  - `scripts/podsync-rs --master ... --tracks ...` (Rust)
  - offsets within ~0.1–0.2s, WAV outputs align in a DAW

---

## Space shuttle style conventions (applied throughout)

1. **Every `if` has a matching `else`** — even if the else is `// No action needed: <reason>`
2. **Every constant is named** with a comment explaining *why* that value, not just what it is
3. **Every function has a doc comment** explaining what it does, its inputs, outputs, and edge cases
4. **No implicit behavior** — explicit type conversions, explicit error handling, explicit branches
5. **Comments explain "why"** at every non-obvious decision point
6. **Match arms are exhaustive** with comments on each arm
7. **No cleverness** — prefer 5 clear lines over 1 clever line

---

## Verification

```sh
# Build
make

# Run tests
cargo test

# Verify CLI
scripts/podsync-rs --help

# Compare with Python (if real audio available)
scripts/podsync --master /path/to/master.mp3 --tracks /path/to/track.wav
scripts/podsync-rs --master /path/to/master.mp3 --tracks /path/to/track.wav
# Compare offsets — should match within ~0.1–0.2s tolerance
```
