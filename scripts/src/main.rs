// ---------------------------------------------------------------------------
// podsync — CLI entry point
//
// Synchronize multi-track podcast recordings to a master track using MFCC
// cross-correlation. This module owns all user-facing output and orchestration.
// The other modules (audio, vad, sync, mfcc) are pure computation.
// ---------------------------------------------------------------------------

mod audio;
mod mfcc;
mod sync;
mod vad;

use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::SystemTime;

use clap::Parser;

use audio::{
    TARGET_SAMPLE_RATE,
    apply_offset, load_audio, seconds_to_samples, write_audio,
};
use sync::{LOW_CONFIDENCE_THRESHOLD, compute_drift, extract_master_mfcc, find_offset_with_mfcc};
use vad::{PREFERRED_SPEECH_DURATION_S, VAD_SEARCH_LIMIT_S, find_first_speech_segment};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of seconds of speech to extract for cross-correlation.
/// 120s gives enough spectral material for a reliable MFCC correlation peak
/// while keeping runtime manageable (~3–5s per track on typical hardware).
const DEFAULT_SYNC_WINDOW_S: f64 = 120.0;

/// Default suffix appended to output filenames. The output for "track.wav"
/// becomes "track-synced.wav". This avoids overwriting the original files.
const DEFAULT_OUTPUT_SUFFIX: &str = "synced";

/// Window length at the end of the recording used for drift measurement.
/// 120s gives enough material for a second correlation at the tail end.
const DRIFT_END_WINDOW_S: f64 = 120.0;

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

/// Synchronize podcast tracks to a master recording.
///
/// Aligns individual participant tracks to a master track using MFCC-based
/// cross-correlation, then outputs synced WAV files that can be dropped into
/// a DAW at position 0:00.
#[derive(Parser)]
#[command(name = "podsync")]
struct Cli {
    /// Master/sync reference track.
    #[arg(long, required = true)]
    master: PathBuf,

    /// Individual tracks to sync (can specify multiple: --tracks A --tracks B).
    #[arg(long, required = true)]
    tracks: Vec<PathBuf>,

    /// Seconds of speech to use for correlation.
    #[arg(long, default_value_t = DEFAULT_SYNC_WINDOW_S)]
    sync_window: f64,

    /// Suffix for output files (default: synced).
    #[arg(long, default_value = DEFAULT_OUTPUT_SUFFIX)]
    output_suffix: String,
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Result of processing a single track.
///
/// On success: audio, offset, drift, and confidence are populated.
/// On failure: error is populated, everything else is None.
struct TrackResult {
    /// Original path to this track's audio file.
    path: PathBuf,

    /// Loaded and resampled audio data (populated on successful load).
    audio: Option<Vec<f32>>,

    /// Time offset in seconds: positive = track started late, negative = early.
    offset: Option<f64>,

    /// Clock drift in seconds at the end of the recording.
    drift: Option<f64>,

    /// Confidence of the cross-correlation match (0.0 = ambiguous, 1.0 = clear).
    confidence: Option<f64>,

    /// Error message if processing failed at any stage.
    error: Option<String>,
}

impl TrackResult {
    /// Create a new TrackResult with all fields initialized to None/empty.
    fn new(path: PathBuf) -> Self {
        TrackResult {
            path,
            audio: None,
            offset: None,
            drift: None,
            confidence: None,
            error: None,
        }
    }
}

/// Lightweight summary of a track's sync result, without the audio buffer.
///
/// Created after writing the output file so the audio buffer can be freed.
/// Used for the summary table and log file.
struct TrackSummary {
    /// Original path to this track's audio file.
    path: PathBuf,

    /// Time offset in seconds.
    offset: Option<f64>,

    /// Confidence of the cross-correlation match.
    confidence: Option<f64>,

    /// Clock drift in seconds.
    drift: Option<f64>,

    /// Path where the synced output was written.
    output_path: Option<PathBuf>,

    /// Error message if processing or writing failed.
    error: Option<String>,
}

impl TrackSummary {
    /// A track is successful if it completed without error and produced output.
    fn success(&self) -> bool {
        self.error.is_none() && self.output_path.is_some()
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a signed time offset as a human-readable string.
///
/// Positive values get a "+" prefix for clarity ("+2.00s").
/// Negative values already have the "-" from formatting ("-1.50s").
/// Exactly zero shows no sign prefix ("0.00s").
fn format_time(seconds: f64) -> String {
    if seconds > 0.0 {
        format!("+{:.2}s", seconds)
    } else if seconds < 0.0 {
        format!("{:.2}s", seconds)
    } else {
        // Exactly zero — no sign prefix.
        "0.00s".to_string()
    }
}

/// Format a duration as NmSSs (e.g. "2m05s").
///
/// Used for displaying audio lengths in the CLI output. Always shows
/// minutes and two-digit seconds for consistent alignment.
fn format_duration(seconds: f64) -> String {
    let total_secs = seconds as u64;
    let minutes = total_secs / 60;
    let secs = total_secs % 60;
    format!("{}m{:02}s", minutes, secs)
}

// ---------------------------------------------------------------------------
// Log file
// ---------------------------------------------------------------------------

/// Write a mini log file summarizing the sync run.
///
/// Placed next to the master file as `podsync-<timestamp>.log`. Each run
/// produces a unique file so previous logs are preserved.
fn write_log_file(
    master_path: &Path,
    master_duration: f64,
    master_sr: u32,
    results: &[TrackSummary],
) {
    // --- Build timestamp for filename and header ---------------------------
    // Uses Unix epoch seconds — no chrono dependency needed for a log name.
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let log_filename = format!("podsync-{}.log", timestamp);
    let log_path = master_path.parent().unwrap_or(Path::new(".")).join(log_filename);

    let mut log = String::new();

    // --- Header ------------------------------------------------------------
    writeln!(log, "podsync run ({})", timestamp).unwrap();
    writeln!(log, "master: {} ({}, {}Hz)",
        master_path.file_name().unwrap_or_default().to_string_lossy(),
        format_duration(master_duration),
        master_sr,
    ).unwrap();
    writeln!(log).unwrap();

    // --- Per-track results -------------------------------------------------
    writeln!(log, "tracks:").unwrap();

    for result in results {
        let input_name = result.path.file_name().unwrap_or_default().to_string_lossy();

        if result.success() {
            let output_name = result.output_path.as_ref().unwrap()
                .file_name().unwrap_or_default().to_string_lossy();
            let offset_str = format_time(result.offset.unwrap());
            let confidence = result.confidence.unwrap_or(0.0);
            let drift_str = match result.drift {
                Some(d) => format!("{:.2}s", d),
                None => "N/A".to_string(),
            };
            writeln!(log, "  {} → {}", input_name, output_name).unwrap();
            writeln!(log, "    offset: {}  confidence: {:.2}  drift: {}", offset_str, confidence, drift_str).unwrap();
        } else {
            writeln!(log, "  {} — FAILED: {}", input_name, result.error.as_deref().unwrap_or("unknown")).unwrap();
        }
    }

    // --- Summary line ------------------------------------------------------
    let success_count = results.iter().filter(|r| r.success()).count();
    let fail_count = results.len() - success_count;

    writeln!(log).unwrap();
    if fail_count > 0 {
        writeln!(log, "result: {} succeeded, {} failed", success_count, fail_count).unwrap();
    } else {
        writeln!(log, "result: {} tracks synchronized", success_count).unwrap();
    }

    // --- Write to disk -----------------------------------------------------
    match fs::write(&log_path, &log) {
        Ok(()) => {
            eprintln!("  Log written to {}", log_path.display());
        }
        Err(e) => {
            // Log file is best-effort — don't fail the run over it.
            eprintln!("  WARNING: could not write log file: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-track processing
// ---------------------------------------------------------------------------

/// Process a single track: load, detect speech, correlate, measure drift.
///
/// Returns a TrackResult containing the loaded audio and all computed metadata.
/// The caller is responsible for writing the output file.
///
/// This function prints progress to stderr as it goes.
fn process_track(
    master_mfcc: &[Vec<f64>],
    master_audio: &[f32],
    master_sr: u32,
    track_path: &Path,
    sync_window: f64,
) -> TrackResult {
    let mut result = TrackResult::new(track_path.to_path_buf());

    eprintln!("\nProcessing {}...", track_path.file_name().unwrap_or_default().to_string_lossy());

    // --- Load track --------------------------------------------------------
    let (track_audio, track_sr) = match load_audio(track_path, master_sr) {
        Ok((audio, sr)) => (audio, sr),
        Err(e) => {
            result.error = Some(format!("Failed to load: {}", e));
            eprintln!("  ERROR: {}", result.error.as_ref().unwrap());
            return result;
        }
    };

    // --- Detect speech region ----------------------------------------------
    eprint!("  Detecting speech regions...");
    let speech_segment = find_first_speech_segment(
        &track_audio,
        track_sr,
        PREFERRED_SPEECH_DURATION_S,
        VAD_SEARCH_LIMIT_S,
    );

    let (speech_start, _speech_end) = match speech_segment {
        Some((start, end)) => {
            let speech_duration = end - start;
            eprintln!(" found {} of speech", format_duration(speech_duration));
            (start, end)
        }
        None => {
            result.error = Some(format!(
                "Insufficient speech detected (< {:.0}s in first {:.0} min)",
                PREFERRED_SPEECH_DURATION_S,
                VAD_SEARCH_LIMIT_S / 60.0,
            ));
            eprintln!(" FAILED");
            eprintln!("  ERROR: {}", result.error.as_ref().unwrap());
            return result;
        }
    };

    // --- Extract speech portion for correlation ----------------------------
    // We extract up to sync_window seconds of speech starting at the detected
    // speech onset. This focused window gives the cross-correlation the best
    // signal-to-noise ratio.
    let speech_start_samples = seconds_to_samples(speech_start, track_sr) as usize;
    let speech_sample_count = seconds_to_samples(sync_window, track_sr) as usize;
    let speech_end_sample = (speech_start_samples + speech_sample_count).min(track_audio.len());
    let track_speech = &track_audio[speech_start_samples..speech_end_sample];

    // --- Find offset -------------------------------------------------------
    eprint!("  Correlating against master...");
    let (offset, confidence) = find_offset_with_mfcc(
        master_mfcc,
        track_speech,
        track_sr,
        sync_window,
    );

    // Adjust offset to account for where speech starts within the track.
    // The correlation found where track_speech appears in the master, but
    // track_speech starts at speech_start within the track. So the full
    // track's offset is: (where speech was found in master) - speech_start.
    let total_offset = offset - speech_start;

    result.audio = Some(track_audio);
    result.offset = Some(total_offset);
    result.confidence = Some(confidence);
    eprintln!(" offset: {} (confidence: {:.2})", format_time(total_offset), confidence);

    if confidence < LOW_CONFIDENCE_THRESHOLD {
        eprintln!("  WARNING: Low confidence — sync may be inaccurate");
    } else {
        // Confidence is acceptable — no additional warning needed.
    }

    // --- Measure drift -----------------------------------------------------
    eprint!("  Measuring drift...");
    let drift = compute_drift(
        master_audio,
        result.audio.as_ref().unwrap(),
        master_sr,
        total_offset,
        DRIFT_END_WINDOW_S,
    );

    match drift {
        Some(d) => {
            result.drift = Some(d);
            eprintln!(" {:.2}s at master end", d.abs());
        }
        None => {
            result.drift = None;
            eprintln!(" N/A (audio too short to measure)");
        }
    }

    result
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let master_path = &cli.master;
    let track_paths = &cli.tracks;

    // --- Load master -------------------------------------------------------
    eprintln!(
        "Loading master: {}",
        master_path.file_name().unwrap_or_default().to_string_lossy()
    );

    let (master_audio, master_sr) = match load_audio(master_path, TARGET_SAMPLE_RATE) {
        Ok((audio, sr)) => (audio, sr),
        Err(e) => {
            eprintln!("ERROR: Failed to load master: {}", e);
            process::exit(1);
        }
    };

    let master_duration = master_audio.len() as f64 / master_sr as f64;
    eprintln!("  Duration: {} at {}Hz", format_duration(master_duration), master_sr);

    // --- Pre-extract master MFCCs ------------------------------------------
    // Extract once before the track loop so every track reuses the same
    // master features instead of re-computing them.
    eprint!("  Extracting master MFCCs...");
    let master_mfcc = extract_master_mfcc(&master_audio, master_sr, VAD_SEARCH_LIMIT_S);
    eprintln!(" done ({} coefficients × {} frames)", master_mfcc.len(), master_mfcc[0].len());

    // --- Process and write each track --------------------------------------
    // Process each track, write its output immediately, then drop the audio
    // buffer. This keeps peak memory at O(1) audio buffers instead of O(N).
    let max_length = master_audio.len();
    let mut summaries: Vec<TrackSummary> = Vec::with_capacity(track_paths.len());

    for track_path in track_paths {
        let result = process_track(&master_mfcc, &master_audio, master_sr, track_path, cli.sync_window);

        let summary = if result.error.is_some() {
            // Track failed during processing — nothing to write.
            TrackSummary {
                path: result.path,
                offset: None,
                confidence: None,
                drift: None,
                output_path: None,
                error: result.error,
            }
        } else {
            let offset_samples = seconds_to_samples(result.offset.unwrap(), master_sr);

            let (output_path, write_error) = match apply_offset(result.audio.as_ref().unwrap(), offset_samples, max_length) {
                Ok(padded) => {
                    let path = result.path.parent().unwrap_or(Path::new(".")).join(
                        format!("{}-{}.wav",
                            result.path.file_stem().unwrap_or_default().to_string_lossy(),
                            cli.output_suffix)
                    );
                    match write_audio(&path, &padded, master_sr) {
                        Ok(()) => {
                            eprintln!("  Writing {}", path.file_name().unwrap_or_default().to_string_lossy());
                            (Some(path), None)
                        }
                        Err(e) => {
                            eprintln!("  ERROR writing {}: {}", path.file_name().unwrap_or_default().to_string_lossy(), e);
                            (None, Some(e.to_string()))
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  ERROR writing {}: {}", result.path.file_name().unwrap_or_default().to_string_lossy(), e);
                    (None, Some(e.to_string()))
                }
            };

            TrackSummary {
                path: result.path,
                offset: result.offset,
                confidence: result.confidence,
                drift: result.drift,
                output_path,
                error: write_error,
            }
        };

        // Audio buffer from `result` is dropped here.
        summaries.push(summary);
    }

    // --- Summary -----------------------------------------------------------
    eprintln!();
    eprintln!("{}", "=".repeat(60));
    eprintln!("Summary:");

    let success_count = summaries.iter().filter(|r| r.success()).count();
    let fail_count = summaries.len() - success_count;

    for summary in &summaries {
        if summary.success() {
            let drift_str = match summary.drift {
                Some(d) => format!("drift: {:.2}s", d),
                None => "drift: N/A".to_string(),
            };
            eprintln!(
                "  {:<40} offset: {}   {}   ✓",
                summary.output_path.as_ref().unwrap().file_name().unwrap_or_default().to_string_lossy(),
                format_time(summary.offset.unwrap()),
                drift_str,
            );
        } else {
            eprintln!(
                "  {:<40} FAILED: {}",
                summary.path.file_name().unwrap_or_default().to_string_lossy(),
                summary.error.as_deref().unwrap_or("unknown error"),
            );
        }
    }

    eprintln!("{}", "=".repeat(60));

    // --- Write log file ----------------------------------------------------
    write_log_file(master_path, master_duration, master_sr, &summaries);

    if fail_count > 0 {
        eprintln!("\n{} succeeded, {} failed", success_count, fail_count);
        process::exit(1);
    } else {
        eprintln!("\n{} tracks synchronized successfully", success_count);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Format helpers ----------------------------------------------------

    #[test]
    fn test_format_time_positive() {
        assert_eq!(format_time(2.0), "+2.00s");
        assert_eq!(format_time(0.5), "+0.50s");
    }

    #[test]
    fn test_format_time_negative() {
        assert_eq!(format_time(-1.5), "-1.50s");
        assert_eq!(format_time(-0.01), "-0.01s");
    }

    #[test]
    fn test_format_time_zero() {
        assert_eq!(format_time(0.0), "0.00s");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(125.0), "2m05s");
        assert_eq!(format_duration(60.0), "1m00s");
        assert_eq!(format_duration(5.0), "0m05s");
        assert_eq!(format_duration(3661.0), "61m01s");
    }

    // --- CLI argument parsing ----------------------------------------------

    #[test]
    fn test_cli_requires_master() {
        // Parsing with no arguments should fail.
        let result = Cli::try_parse_from(["podsync"]);
        assert!(result.is_err(), "CLI should require --master");
    }

    #[test]
    fn test_cli_requires_tracks() {
        // Parsing with only --master should fail.
        let result = Cli::try_parse_from(["podsync", "--master", "master.wav"]);
        assert!(result.is_err(), "CLI should require --tracks");
    }

    #[test]
    fn test_cli_accepts_all_options() {
        let result = Cli::try_parse_from([
            "podsync",
            "--master", "master.wav",
            "--tracks", "track1.wav",
            "--tracks", "track2.wav",
            "--sync-window", "60",
            "--output-suffix", "aligned",
        ]);
        assert!(result.is_ok(), "CLI should accept all documented options");

        let cli = result.unwrap();
        assert_eq!(cli.master, PathBuf::from("master.wav"));
        assert_eq!(cli.tracks, vec![PathBuf::from("track1.wav"), PathBuf::from("track2.wav")]);
        assert_eq!(cli.sync_window, 60.0);
        assert_eq!(cli.output_suffix, "aligned");
    }

    #[test]
    fn test_cli_defaults() {
        let result = Cli::try_parse_from([
            "podsync",
            "--master", "master.wav",
            "--tracks", "track.wav",
        ]);
        assert!(result.is_ok());

        let cli = result.unwrap();
        assert_eq!(cli.sync_window, DEFAULT_SYNC_WINDOW_S);
        assert_eq!(cli.output_suffix, DEFAULT_OUTPUT_SUFFIX);
    }

    // --- TrackSummary ------------------------------------------------------

    #[test]
    fn test_track_summary_success() {
        let summary = TrackSummary {
            path: PathBuf::from("track.wav"),
            offset: Some(1.0),
            confidence: Some(0.9),
            drift: None,
            output_path: Some(PathBuf::from("track-synced.wav")),
            error: None,
        };
        assert!(summary.success());

        // Without output_path, not successful.
        let summary_no_output = TrackSummary {
            path: PathBuf::from("track.wav"),
            offset: Some(1.0),
            confidence: Some(0.9),
            drift: None,
            output_path: None,
            error: None,
        };
        assert!(!summary_no_output.success());
    }

    #[test]
    fn test_track_summary_failure() {
        let summary = TrackSummary {
            path: PathBuf::from("track.wav"),
            offset: None,
            confidence: None,
            drift: None,
            output_path: None,
            error: Some("something went wrong".to_string()),
        };
        assert!(!summary.success());
    }
}
