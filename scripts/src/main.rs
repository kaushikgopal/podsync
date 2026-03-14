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

use audio::{apply_offset, load_audio, seconds_to_samples, write_audio, TARGET_SAMPLE_RATE};
use sync::{compute_drift, extract_master_mfcc, find_offset_with_mfcc, LOW_CONFIDENCE_THRESHOLD};
use vad::{find_speech_candidates, PREFERRED_SPEECH_DURATION_S, VAD_SEARCH_LIMIT_S};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of seconds of speech to extract for cross-correlation.
/// 120s is usually enough speech for a reliable MFCC match while keeping
/// runtime manageable (~3–5s per track on typical hardware).
const DEFAULT_SYNC_WINDOW_S: f64 = 120.0;

/// Default suffix appended to output filenames. The output for "track.wav"
/// becomes "track-synced.wav". This avoids overwriting the original files.
const DEFAULT_OUTPUT_SUFFIX: &str = "synced";

/// Window length at the end of the recording used for drift measurement.
/// 120s gives enough material for a second correlation at the tail end.
const DRIFT_END_WINDOW_S: f64 = 120.0;

/// Maximum number of speech candidates to try per track.
/// Three gives the matcher more than one shot without blowing up runtime.
const MAX_SPEECH_CANDIDATES: usize = 3;

/// Number of subwindows to try inside a long speech region.
/// Start, middle, and end cover the obvious alternatives without over-searching.
const WINDOWS_PER_CANDIDATE: usize = 3;

/// Minimum spacing between subwindow starts inside one speech region.
/// Windows closer than this are effectively the same clip.
const MIN_WINDOW_START_SEPARATION_S: f64 = 10.0;

/// Offset agreement tolerance between candidate windows.
/// 250ms is wide enough to absorb MFCC frame quantization but narrow enough to
/// distinguish meaningfully different alignments.
const OFFSET_AGREEMENT_TOLERANCE_S: f64 = 0.25;

/// Minimum confidence required for a second candidate region to count as
/// corroboration. Below this, the peak is too weak to treat as support.
const MIN_CORROBORATING_CONFIDENCE: f64 = 0.35;

/// Confidence gap small enough to treat two matches as operationally tied.
/// This lets corroboration decide between near-equal peaks without letting a
/// cluster of weak matches overrule a clearly stronger one.
const CORROBORATION_CONFIDENCE_SLACK: f64 = 0.03;

/// Small epsilon for comparing floating-point match scores during selection.
const MATCH_EPSILON: f64 = 1e-9;

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

/// Synchronize podcast tracks to a master recording.
///
/// Aligns individual participant tracks to a master track using MFCC-based
/// cross-correlation, then outputs synced WAV files that can be dropped into
/// a digital audio workstation (DAW) at position 0:00.
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

/// A single offset estimate produced from one candidate speech window.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MatchAttempt {
    /// Which ranked speech candidate produced this attempt.
    candidate_index: usize,

    /// Where the tested subwindow starts in the original track.
    window_start: f64,

    /// Track offset relative to the master after accounting for `window_start`.
    total_offset: f64,

    /// Distinctiveness of the correlation peak for this window.
    confidence: f64,
}

/// The selected match plus how many candidate regions agreed with it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MatchSelection {
    /// The chosen attempt.
    attempt: MatchAttempt,

    /// Number of candidate regions that corroborate the chosen offset,
    /// including the winner's own region.
    agreeing_candidates: usize,
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

/// Build the candidate window starts to try within one speech region.
///
/// Regions shorter than `sync_window` produce one window at the region start.
/// Longer regions produce three windows: start, middle, and end.
///
/// This is intentionally track-side only. The master is already a broad search
/// space, so the extra robustness comes from giving the participant track more
/// than one shot at presenting a distinctive speech chunk.
fn candidate_window_starts(
    speech_region_start: f64,
    speech_region_end: f64,
    sync_window: f64,
) -> Vec<f64> {
    let region_duration = speech_region_end - speech_region_start;

    if region_duration <= sync_window {
        vec![speech_region_start]
    } else {
        let latest_start = speech_region_end - sync_window;
        let middle_start = speech_region_start + (latest_start - speech_region_start) / 2.0;
        let raw_starts = [speech_region_start, middle_start, latest_start];
        let mut starts: Vec<f64> = Vec::with_capacity(WINDOWS_PER_CANDIDATE);

        for &start in &raw_starts {
            let far_enough = starts.last().map_or(true, |prev_start| {
                (start - *prev_start).abs() >= MIN_WINDOW_START_SEPARATION_S
            });

            if far_enough {
                starts.push(start);
            } else {
                // This start is too close to the previous one to be distinct.
            }
        }

        starts
    }
}

/// Count how many distinct candidate regions corroborate a chosen attempt.
///
/// The winner's own candidate always counts. Additional support must come from
/// other candidate regions whose confidence is at least
/// `MIN_CORROBORATING_CONFIDENCE`.
///
/// We deliberately count *regions*, not raw attempts. Multiple windows from
/// one long region are useful for exploration, but they are not independent
/// evidence that the offset is correct.
fn count_corroborating_candidates(attempts: &[MatchAttempt], anchor: MatchAttempt) -> usize {
    let mut corroborating_candidates = vec![anchor.candidate_index];

    for &other in attempts {
        let same_candidate = other.candidate_index == anchor.candidate_index;
        let close_offset =
            (other.total_offset - anchor.total_offset).abs() <= OFFSET_AGREEMENT_TOLERANCE_S;
        let strong_enough = other.confidence >= MIN_CORROBORATING_CONFIDENCE;
        let already_counted = corroborating_candidates.contains(&other.candidate_index);

        if same_candidate {
            // Multiple windows from the same candidate region do not count as
            // independent corroboration.
        } else if !close_offset {
            // Offset disagreement is too large to count as support.
        } else if !strong_enough {
            // Peak is too weak to count as corroboration.
        } else if already_counted {
            // We already counted this candidate region.
        } else {
            corroborating_candidates.push(other.candidate_index);
        }
    }

    corroborating_candidates.len()
}

/// Select the best match attempt.
///
/// The highest-confidence attempt wins unless two attempts are within
/// `CORROBORATION_CONFIDENCE_SLACK`, in which case corroboration breaks the
/// near-tie before raw confidence does. The earlier window start remains the
/// stable final tie-breaker.
///
/// This ordering is intentional: one sharp, distinctive peak should beat a
/// crowd of weak guesses. Corroboration helps when peaks are effectively neck
/// and neck, but it should not let several mediocre windows outweigh one
/// clearly better match.
fn select_best_match(attempts: &[MatchAttempt]) -> Option<MatchSelection> {
    if attempts.is_empty() {
        return None;
    }

    let mut best_attempt = attempts[0];
    let mut best_corroboration = count_corroborating_candidates(attempts, best_attempt);

    for &attempt in attempts {
        let confidence_delta = attempt.confidence - best_attempt.confidence;
        let materially_stronger_confidence =
            confidence_delta > CORROBORATION_CONFIDENCE_SLACK;
        let near_tie_confidence =
            confidence_delta.abs() <= CORROBORATION_CONFIDENCE_SLACK;
        let higher_confidence_in_near_tie = confidence_delta > MATCH_EPSILON;
        let exact_confidence_tie = confidence_delta.abs() <= MATCH_EPSILON;
        let attempt_corroboration = count_corroborating_candidates(attempts, attempt);
        let stronger_corroboration = attempt_corroboration > best_corroboration;
        let earlier_window = attempt.window_start < best_attempt.window_start;

        if materially_stronger_confidence
            || (near_tie_confidence && stronger_corroboration)
            || (near_tie_confidence
                && attempt_corroboration == best_corroboration
                && higher_confidence_in_near_tie)
            || (exact_confidence_tie
                && attempt_corroboration == best_corroboration
                && earlier_window)
        {
            best_attempt = attempt;
            best_corroboration = attempt_corroboration;
        } else {
            // Existing best selection still wins.
        }
    }

    Some(MatchSelection {
        attempt: best_attempt,
        agreeing_candidates: best_corroboration,
    })
}

// ---------------------------------------------------------------------------
// Log file
// ---------------------------------------------------------------------------

/// Write a log file summarizing the sync run.
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
    let log_path = master_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(log_filename);

    let mut log = String::new();

    // --- Header ------------------------------------------------------------
    writeln!(log, "podsync run ({})", timestamp).unwrap();
    writeln!(
        log,
        "master: {} ({}, {}Hz)",
        master_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        format_duration(master_duration),
        master_sr,
    )
    .unwrap();
    writeln!(log).unwrap();

    // --- Per-track results -------------------------------------------------
    writeln!(log, "tracks:").unwrap();

    for result in results {
        let input_name = result
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();

        if result.success() {
            let output_name = result
                .output_path
                .as_ref()
                .unwrap()
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            let offset_str = format_time(result.offset.unwrap());
            let confidence = result.confidence.unwrap_or(0.0);
            let drift_str = match result.drift {
                Some(d) => format!("{:.2}s", d),
                None => "N/A".to_string(),
            };
            writeln!(log, "  {} → {}", input_name, output_name).unwrap();
            writeln!(
                log,
                "    offset: {}  confidence: {:.2}  drift: {}",
                offset_str, confidence, drift_str
            )
            .unwrap();
        } else {
            writeln!(
                log,
                "  {} — FAILED: {}",
                input_name,
                result.error.as_deref().unwrap_or("unknown")
            )
            .unwrap();
        }
    }

    // --- Summary line ------------------------------------------------------
    let success_count = results.iter().filter(|r| r.success()).count();
    let fail_count = results.len() - success_count;

    writeln!(log).unwrap();
    if fail_count > 0 {
        writeln!(
            log,
            "result: {} succeeded, {} failed",
            success_count, fail_count
        )
        .unwrap();
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

    eprintln!(
        "\nProcessing {}...",
        track_path.file_name().unwrap_or_default().to_string_lossy()
    );

    // --- Load track --------------------------------------------------------
    let (track_audio, track_sr) = match load_audio(track_path, master_sr) {
        Ok((audio, sr)) => (audio, sr),
        Err(e) => {
            result.error = Some(format!("Failed to load: {}", e));
            eprintln!("  ERROR: {}", result.error.as_ref().unwrap());
            return result;
        }
    };

    // --- Detect speech regions ---------------------------------------------
    // The old matcher made one bet on one speech segment. The newer matcher
    // still starts with VAD, but keeps several promising regions so the track
    // gets more than one chance to match cleanly against the master.
    eprint!("  Detecting speech regions...");
    let speech_candidates = find_speech_candidates(
        &track_audio,
        track_sr,
        PREFERRED_SPEECH_DURATION_S,
        VAD_SEARCH_LIMIT_S,
        MAX_SPEECH_CANDIDATES,
    );

    if speech_candidates.is_empty() {
        result.error = Some(format!(
            "No speech detected in first {:.0} min",
            VAD_SEARCH_LIMIT_S / 60.0,
        ));
        eprintln!(" FAILED");
        eprintln!("  ERROR: {}", result.error.as_ref().unwrap());
        return result;
    } else {
        let best_duration = speech_candidates
            .iter()
            .map(|&(start, end)| end - start)
            .fold(0.0, f64::max);
        let plural = if speech_candidates.len() == 1 {
            ""
        } else {
            "s"
        };
        eprintln!(
            " found {} candidate region{} (best: {})",
            speech_candidates.len(),
            plural,
            format_duration(best_duration),
        );
    }

    // --- Correlate candidate windows --------------------------------------
    // Each attempt answers the same question: "if this speech chunk is the
    // right one, where does it land in the master timeline?"
    eprint!("  Correlating against master...");
    let speech_sample_count = seconds_to_samples(sync_window, track_sr) as usize;
    let mut attempts: Vec<MatchAttempt> = Vec::new();

    for (candidate_index, &(speech_start, speech_end)) in speech_candidates.iter().enumerate() {
        let window_starts = candidate_window_starts(speech_start, speech_end, sync_window);

        for &window_start in &window_starts {
            let speech_start_samples = seconds_to_samples(window_start, track_sr) as usize;
            let speech_end_sample =
                (speech_start_samples + speech_sample_count).min(track_audio.len());
            let track_speech = &track_audio[speech_start_samples..speech_end_sample];

            let (offset, confidence) =
                find_offset_with_mfcc(master_mfcc, track_speech, track_sr, sync_window);

            // Convert the window-local offset back into the full-track offset
            // so every attempt can be compared on the same timeline.
            attempts.push(MatchAttempt {
                candidate_index,
                window_start,
                total_offset: offset - window_start,
                confidence,
            });
        }
    }

    // Selection happens after all attempts are on the table so we can prefer
    // the strongest peak first and only use agreement as secondary evidence.
    let selection = match select_best_match(&attempts) {
        Some(selection) => selection,
        None => {
            result.error = Some(format!(
                "No usable speech windows found in first {:.0} min",
                VAD_SEARCH_LIMIT_S / 60.0,
            ));
            eprintln!(" FAILED");
            eprintln!("  ERROR: {}", result.error.as_ref().unwrap());
            return result;
        }
    };

    let total_offset = selection.attempt.total_offset;
    let confidence = selection.attempt.confidence;
    let corroborating_regions = selection.agreeing_candidates.saturating_sub(1);
    let candidate_plural = if corroborating_regions == 1 { "" } else { "s" };

    result.audio = Some(track_audio);
    result.offset = Some(total_offset);
    result.confidence = Some(confidence);
    if corroborating_regions > 0 {
        eprintln!(
            " offset: {} (confidence: {:.2}, corroborated by {} other candidate region{})",
            format_time(total_offset),
            confidence,
            corroborating_regions,
            candidate_plural,
        );
    } else {
        eprintln!(
            " offset: {} (confidence: {:.2})",
            format_time(total_offset),
            confidence
        );
    }

    if confidence < LOW_CONFIDENCE_THRESHOLD {
        if corroborating_regions > 0 {
            eprintln!(
                "  Note: low peak confidence, but {} other candidate region{} agreed within {:.2}s",
                corroborating_regions, candidate_plural, OFFSET_AGREEMENT_TOLERANCE_S,
            );
        } else {
            eprintln!("  WARNING: Low confidence — sync may be inaccurate");
        }
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

/// CLI entry point.
///
/// The master path is intentionally prepared once up front so each participant
/// track can reuse the same master MFCC search window without recomputing it.
fn main() {
    let cli = Cli::parse();

    let master_path = &cli.master;
    let track_paths = &cli.tracks;

    // --- Load master -------------------------------------------------------
    eprintln!(
        "Loading master: {}",
        master_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );

    let (master_audio, master_sr) = match load_audio(master_path, TARGET_SAMPLE_RATE) {
        Ok((audio, sr)) => (audio, sr),
        Err(e) => {
            eprintln!("ERROR: Failed to load master: {}", e);
            process::exit(1);
        }
    };

    let master_duration = master_audio.len() as f64 / master_sr as f64;
    eprintln!(
        "  Duration: {} at {}Hz",
        format_duration(master_duration),
        master_sr
    );

    // --- Pre-extract master MFCCs ------------------------------------------
    // Extract once before the track loop so every track reuses the same
    // master features instead of re-computing them.
    eprint!("  Extracting master MFCCs...");
    let master_mfcc = extract_master_mfcc(&master_audio, master_sr, VAD_SEARCH_LIMIT_S);
    eprintln!(
        " done ({} coefficients × {} frames)",
        master_mfcc.len(),
        master_mfcc[0].len()
    );

    // --- Process and write each track --------------------------------------
    // Process each track, write its output immediately, then drop the audio buffer.
    // This keeps peak memory to one track buffer at a time.
    let max_length = master_audio.len();
    let mut summaries: Vec<TrackSummary> = Vec::with_capacity(track_paths.len());

    for track_path in track_paths {
        let result = process_track(
            &master_mfcc,
            &master_audio,
            master_sr,
            track_path,
            cli.sync_window,
        );

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

            let (output_path, write_error) =
                match apply_offset(result.audio.as_ref().unwrap(), offset_samples, max_length) {
                    Ok(padded) => {
                        let path = result.path.parent().unwrap_or(Path::new(".")).join(format!(
                            "{}-{}.wav",
                            result
                                .path
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy(),
                            cli.output_suffix
                        ));
                        match write_audio(&path, &padded, master_sr) {
                            Ok(()) => {
                                eprintln!(
                                    "  Writing {}",
                                    path.file_name().unwrap_or_default().to_string_lossy()
                                );
                                (Some(path), None)
                            }
                            Err(e) => {
                                eprintln!(
                                    "  ERROR writing {}: {}",
                                    path.file_name().unwrap_or_default().to_string_lossy(),
                                    e
                                );
                                (None, Some(e.to_string()))
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "  ERROR writing {}: {}",
                            result
                                .path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy(),
                            e
                        );
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

        // Drop the per-track audio buffer here to free memory.
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
                summary
                    .output_path
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                format_time(summary.offset.unwrap()),
                drift_str,
            );
        } else {
            eprintln!(
                "  {:<40} FAILED: {}",
                summary
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
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

    // --- Match candidate helpers -----------------------------------------

    #[test]
    fn test_candidate_window_starts_short_region_returns_one_start() {
        let starts = candidate_window_starts(12.0, 30.0, 20.0);
        assert_eq!(starts, vec![12.0]);
    }

    #[test]
    fn test_candidate_window_starts_long_region_returns_start_middle_end() {
        let starts = candidate_window_starts(10.0, 70.0, 20.0);
        assert_eq!(starts, vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_candidate_window_starts_skips_near_duplicate_windows() {
        let starts = candidate_window_starts(10.0, 131.0, 120.0);
        assert_eq!(starts, vec![10.0]);
    }

    #[test]
    fn test_select_best_match_uses_agreement_to_break_confidence_ties() {
        let attempts = vec![
            MatchAttempt {
                candidate_index: 0,
                window_start: 5.0,
                total_offset: 12.0,
                confidence: 0.62,
            },
            MatchAttempt {
                candidate_index: 1,
                window_start: 45.0,
                total_offset: 20.0,
                confidence: 0.62,
            },
            MatchAttempt {
                candidate_index: 2,
                window_start: 95.0,
                total_offset: 20.1,
                confidence: 0.47,
            },
        ];

        let selection = select_best_match(&attempts).expect("selection should exist");

        assert!(
            (selection.attempt.total_offset - 20.0).abs() < 0.2,
            "corroborated tie should win, got {}",
            selection.attempt.total_offset
        );
        assert_eq!(selection.agreeing_candidates, 2);
    }

    #[test]
    fn test_select_best_match_uses_agreement_for_near_ties() {
        let attempts = vec![
            MatchAttempt {
                candidate_index: 0,
                window_start: 5.0,
                total_offset: 12.0,
                confidence: 0.62,
            },
            MatchAttempt {
                candidate_index: 1,
                window_start: 45.0,
                total_offset: 20.0,
                confidence: 0.60,
            },
            MatchAttempt {
                candidate_index: 2,
                window_start: 95.0,
                total_offset: 20.1,
                confidence: 0.48,
            },
        ];

        let selection = select_best_match(&attempts).expect("selection should exist");

        assert!(
            (selection.attempt.total_offset - 20.0).abs() < 0.2,
            "corroborated near-tie should win, got {}",
            selection.attempt.total_offset
        );
        assert_eq!(selection.agreeing_candidates, 2);
    }

    #[test]
    fn test_select_best_match_keeps_stronger_single_match_over_weaker_group() {
        let attempts = vec![
            MatchAttempt {
                candidate_index: 0,
                window_start: 10.0,
                total_offset: 15.0,
                confidence: 0.75,
            },
            MatchAttempt {
                candidate_index: 1,
                window_start: 30.0,
                total_offset: 25.0,
                confidence: 0.26,
            },
            MatchAttempt {
                candidate_index: 2,
                window_start: 60.0,
                total_offset: 25.1,
                confidence: 0.26,
            },
            MatchAttempt {
                candidate_index: 3,
                window_start: 90.0,
                total_offset: 24.9,
                confidence: 0.26,
            },
        ];

        let selection = select_best_match(&attempts).expect("selection should exist");

        assert_eq!(selection.attempt.total_offset, 15.0);
        assert_eq!(selection.agreeing_candidates, 1);
    }

    #[test]
    fn test_select_best_match_keeps_materially_higher_confidence() {
        let attempts = vec![
            MatchAttempt {
                candidate_index: 0,
                window_start: 10.0,
                total_offset: 15.0,
                confidence: 0.64,
            },
            MatchAttempt {
                candidate_index: 1,
                window_start: 30.0,
                total_offset: 25.0,
                confidence: 0.60,
            },
            MatchAttempt {
                candidate_index: 2,
                window_start: 60.0,
                total_offset: 25.1,
                confidence: 0.49,
            },
        ];

        let selection = select_best_match(&attempts).expect("selection should exist");

        assert_eq!(selection.attempt.total_offset, 15.0);
        assert_eq!(selection.agreeing_candidates, 1);
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
            "--master",
            "master.wav",
            "--tracks",
            "track1.wav",
            "--tracks",
            "track2.wav",
            "--sync-window",
            "60",
            "--output-suffix",
            "aligned",
        ]);
        assert!(result.is_ok(), "CLI should accept all documented options");

        let cli = result.unwrap();
        assert_eq!(cli.master, PathBuf::from("master.wav"));
        assert_eq!(
            cli.tracks,
            vec![PathBuf::from("track1.wav"), PathBuf::from("track2.wav")]
        );
        assert_eq!(cli.sync_window, 60.0);
        assert_eq!(cli.output_suffix, "aligned");
    }

    #[test]
    fn test_cli_defaults() {
        let result =
            Cli::try_parse_from(["podsync", "--master", "master.wav", "--tracks", "track.wav"]);
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
