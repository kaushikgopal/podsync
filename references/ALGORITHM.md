# Sync Algorithm Deep Dive

## Overview

Podsync uses MFCC-based cross-correlation to find the time offset between a master track and individual participant tracks.

## Why MFCCs?

MFCCs (Mel-frequency cepstral coefficients) are preferred over raw waveform comparison because:

1. **Volume invariant** — Different recording levels don't affect correlation
2. **EQ invariant** — Different microphones/processing chains still match
3. **Perceptually motivated** — Mel scale matches human hearing
4. **Dimensionality reduction** — Faster computation than full spectrum

## Algorithm Steps

### 1. Voice Activity Detection (VAD)

Before correlating, we find where speech actually occurs in the track:

```
Track audio → WebRTC VAD → List of (start, end) speech regions
```

This handles the case where one participant is silent at the start. We correlate speech-to-speech, not silence-to-speech.

**Parameters:**
- Frame size: 30ms
- Aggressiveness: 2 (medium)
- Minimum segment: 30 seconds of continuous speech

### 2. MFCC Extraction

Extract spectral features from both master and track:

```
Audio → STFT → Mel filterbank → Log → DCT → MFCCs
```

**Parameters:**
- n_mfcc: 20 coefficients
- hop_length: 512 samples (~11.6ms at 44.1kHz)
- Window: Hann

### 3. Cross-Correlation

Find the lag that maximizes correlation between MFCC sequences:

```python
correlation = scipy.signal.correlate(master_mfcc, track_mfcc, mode='full')
peak_idx = np.argmax(correlation)
lag = peak_idx - (len(track_mfcc) - 1)
offset_seconds = lag * hop_length / sr
```

**Confidence calculation:**
- Peak value divided by standard deviation of correlation
- Normalized to 0-1 range
- Values < 0.5 indicate potential misalignment

### 4. Drift Measurement

Drift occurs when recording devices have different clock rates. We measure it by:

1. Find offset at start (done in step 3)
2. Find offset at end (last 2 minutes)
3. Drift = end_offset - start_offset

For a 1-hour podcast:
- 0.1s drift = ~0.003% clock difference (negligible)
- 1.0s drift = ~0.03% clock difference (noticeable but usually acceptable)

## Search Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| search_window | 600s (10 min) | How far into master to search for match |
| correlation_window | 120s (2 min) | How much track audio to use for correlation |

These defaults handle cases where:
- One host joins late (up to 10 minutes)
- Short clips still correlate reliably (2 minutes is plenty)

## Output Generation

After finding offsets:

1. Use master track length as output length
2. For each track:
   - Pad start with silence (positive offset) or trim (negative offset)
   - Trim/pad end to match master length
3. Write as 24-bit WAV at 44.1kHz
