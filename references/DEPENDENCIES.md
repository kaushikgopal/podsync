# Dependencies

## Core Libraries

### librosa (>=0.10.0)
**Purpose:** Audio loading, MFCC extraction, resampling

**Why this library:**
- De facto standard for Python audio analysis
- Handles all common audio formats via soundfile/audioread
- Optimized MFCC implementation
- Well-documented, actively maintained

**Key functions used:**
- `librosa.load()` — Load and resample audio
- `librosa.feature.mfcc()` — Extract MFCC features
- `librosa.resample()` — Sample rate conversion

### scipy (>=1.11.0)
**Purpose:** Cross-correlation

**Why this library:**
- Standard scientific computing library
- Optimized signal processing routines
- `signal.correlate()` is fast and reliable

**Key functions used:**
- `scipy.signal.correlate()` — Cross-correlation for offset detection

### soundfile (>=0.12.0)
**Purpose:** WAV file writing

**Why this library:**
- Direct libsndfile bindings
- Supports 24-bit WAV output
- Fast and reliable
- Used by librosa under the hood

**Key functions used:**
- `sf.write()` — Write audio to WAV

### webrtcvad (>=2.0.10)
**Purpose:** Voice activity detection

**Why this library:**
- Google's production VAD from WebRTC
- Extremely fast (real-time capable)
- Good accuracy for speech/non-speech classification
- No ML model loading required

**Alternative considered:** silero-vad (PyTorch-based, more accurate but heavier)

**Key functions used:**
- `Vad.is_speech()` — Frame-level speech detection

### numpy (>=1.24.0)
**Purpose:** Array operations

**Why this library:**
- Required by all audio libraries
- Efficient numerical operations
- Standard for scientific Python

### click (>=8.1.0)
**Purpose:** CLI framework

**Why this library:**
- Clean, decorator-based API
- Good help generation
- Handles argument parsing edge cases

## Package Management

### uv
**Purpose:** Python package manager

**Why uv over pip:**
- Isolated environments without manual virtualenv
- Fast, reliable dependency resolution
- Single binary installation
- No "pip hell" or dependency conflicts
- Rust-based, from Astral (ruff maintainers)

**Usage:**
- `uv run podsync` — Run without explicit install
- `uv sync` — Install dependencies
- `uv tool install` — Global install
