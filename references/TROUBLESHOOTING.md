# Troubleshooting Guide

## Common Issues

### "Insufficient speech detected"

**Cause:** VAD couldn't find 30+ seconds of continuous speech in the first 10 minutes.

**Solutions:**
1. Check if the track is mostly silence
2. Check if audio is very quiet (normalize first)
3. Try a different track from the same participant
4. Manually specify a different sync point

### Low confidence warning

**Cause:** Cross-correlation peak wasn't significantly higher than noise.

**Possible reasons:**
1. Track audio quality is poor
2. Participant has very different voice from master mix
3. Wrong track selected (not from this episode)

**Solutions:**
1. Try increasing `--sync-window` to use more audio
2. Manually verify the track is from the correct episode
3. Check if track needs preprocessing (noise reduction)

### Drift > 1 second

**Cause:** Recording devices had different clock rates.

**Solutions:**
1. For most editing, 1s drift over 1 hour is acceptable
2. If critical, manually adjust in DAW
3. Consider time-stretching in post (not handled by this tool)

### "Failed to load" error

**Cause:** Audio file format not supported or file corrupted.

**Supported formats (via symphonia):**
- WAV (all sample rates, PCM)
- MP3
- AIFF
- FLAC
- OGG/Vorbis

**Solutions:**
1. Convert to WAV using ffmpeg: `ffmpeg -i input.xyz output.wav`
2. Check file isn't corrupted: `ffprobe input.xyz`

## Verifying Sync

After running podsync, verify in your DAW:

1. Import all `-synced.wav` files
2. Place all at position 0:00
3. Solo each track briefly to confirm voices align
4. Check a point in the middle and near the end for drift

Check the log file (`podsync-<epoch>.log` next to the master) for a quick summary
of offsets, confidence scores, and drift measurements.

## Performance

For 1-hour episodes, expect:
- ~5 seconds for loading/resampling per track
- ~3 seconds for VAD per track
- ~30 seconds for MFCC extraction and correlation per track
- ~5 seconds for writing per track

Total: ~1-2 minutes for a 2-person podcast.

The Rust implementation is single-threaded. The correlation step dominates
runtime for longer recordings.
