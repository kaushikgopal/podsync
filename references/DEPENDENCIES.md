# Dependencies

## Core Crates

### symphonia (0.5)
**Purpose:** Audio decoding

**Why this crate:**
- Pure Rust audio decoder — no system dependencies
- Supports MP3, WAV, FLAC, OGG/Vorbis, AIFF
- Well-maintained, actively developed
- Feature-gated codecs keep binary size reasonable

**Key usage:**
- Decode any supported audio format into raw PCM samples
- Read sample rate, channel count, bit depth metadata

### rubato (1.0)
**Purpose:** Sample rate conversion

**Why this crate:**
- High-quality sinc interpolation resampler
- Supports arbitrary ratios (e.g. 48kHz → 44.1kHz)
- Pure Rust, no system dependencies
- Configurable quality/speed tradeoff

**Key usage:**
- Resample decoded audio to target 44.1kHz
- Lower-quality resample to 16kHz for VAD processing

### hound (3.5)
**Purpose:** WAV file writing

**Why this crate:**
- Simple, focused WAV reader/writer
- Supports 24-bit PCM output
- Minimal dependencies
- Well-tested, stable

**Key usage:**
- Write synced output as 24-bit 44.1kHz mono WAV

### webrtc-vad (0.4)
**Purpose:** Voice activity detection

**Why this crate:**
- Wraps Google's production WebRTC VAD C library
- Extremely fast (real-time capable)
- Good accuracy for speech/non-speech classification
- No ML model loading required

**Alternative considered:** silero-vad (requires ONNX runtime, more accurate but heavier)

**Key usage:**
- Frame-level speech detection (30ms frames)
- Requires specific sample rates (8/16/32/48kHz)

### realfft (3.5)
**Purpose:** FFT computation

**Why this crate:**
- Optimized for real-valued inputs (audio data)
- Built on rustfft — well-tested FFT implementation
- Returns half-spectrum (exploiting conjugate symmetry)

**Key usage:**
- STFT computation in the MFCC pipeline
- FFT-based cross-correlation (convolution theorem)

### clap (4.5, derive)
**Purpose:** CLI argument parsing

**Why this crate:**
- De facto standard for Rust CLIs
- Derive macros for declarative argument definitions
- Generates help text automatically
- Handles validation, defaults, and error messages

**Key usage:**
- Parse `--master`, `--tracks`, `--sync-window`, `--output-suffix`

## Dev Dependencies

### tempfile (3)
**Purpose:** Temporary files for tests

**Key usage:**
- WAV roundtrip tests (write to temp file, read back, compare)

## Build

Standard cargo build. No system dependencies beyond a C compiler (needed by
webrtc-vad to compile the Google WebRTC VAD C source).

```sh
make        # builds release binary at scripts/podsync
make test   # runs cargo test
```
