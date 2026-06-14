//! Web Platform Tests aggregating runner CLI.
//!
//! Discovers testharness.js-based `.html` WPT tests in a local corpus and runs
//! each through the engine via `conclave --type wpt-one <file>` (process per
//! test, so a panic/hang is isolated), then aggregates the per-subtest results
//! into a JSON report with a per-area breakdown and frequency-ranked failure
//! clusters.
//!
//! Usage:
//!   wpt_runner [CORPUS_ROOT] [--exe PATH] [--area NAME] [--sample N]
//!              [--limit N] [--jobs N] [--timeout-ms N] [--out PATH]
//!
//! Defaults: CORPUS_ROOT=conformance/wpt, exe=<sibling conclave>, jobs=4,
//! timeout-ms=20000, out=conformance/wpt_report.json.
//!
//! MACHINE SAFETY: bounded worker pool + per-test wall-clock kill + sample modes.
//! Run ONE driver at a time; never concurrent with a build.

use std::process::ExitCode;

use cv_base::cli::Cli;
use wpt_runner::driver;

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.has("help") || cli.has("h") {
        eprintln!(
            "wpt_runner [CORPUS_ROOT] [--exe PATH] [--area NAME] [--sample N] \
             [--limit N] [--jobs N] [--timeout-ms N] [--out PATH]\n\
             \n\
             Runs testharness.js-based WPT .html tests through conclave and \
             writes a JSON conformance report. Areas: dom, css, html (per corpus \
             layout). Use --sample N to run every Nth test for a fast baseline."
        );
        return ExitCode::SUCCESS;
    }
    match driver::run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("wpt_runner failed: {e}");
            ExitCode::from(1)
        }
    }
}
