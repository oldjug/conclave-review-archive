//! test262 conformance AGGREGATING driver.
//!
//! Runs the local test262 corpus by spawning the existing per-test runner
//! (`conclave --type test262-one <file>`) as a separate process per test so
//! a panic (`panic=abort`) or an infinite loop cannot kill the aggregate run.
//! Produces total/pass/fail/skip counts, a pass rate (pass / (pass+fail),
//! excluding honest skips), and a frequency-ranked failure-cluster report.
//!
//! MACHINE SAFETY (hard requirement): the run is BOUNDED + THROTTLED.
//!   * A small worker pool (`--jobs N`, default 4) caps concurrency — never
//!     all 53k spawns at once.
//!   * A per-test wall-clock timeout (`--timeout-ms`, default 10000) kills a
//!     hung child (its `panic=abort` + `CV_JS_TIME_BUDGET_MS` are the inner
//!     guards; this is the outer backstop).
//!   * `--sample N` runs every Nth file and `--per-dir N` caps per directory
//!     for a fast representative baseline before any full run.
//!
//! Output: a machine-readable JSON report (counts + ranked clusters + a few
//! representative failing files per cluster) to `conformance/test262_report.json`
//! (override with `--out <path>`).
//!
//! This is PRODUCTION measurement: skips are counted honestly (they come from
//! the per-test runner's own `RESULT: SKIP:<reason>` for genuinely-unsupported
//! features/flags), and pass rate excludes skips so it cannot be inflated by
//! declaring more features unsupported.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cv_base::cli::Cli;

/// The outcome of a single test, as classified by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail,
    Skip,
    Timeout,
    /// Child produced no `RESULT:` line (crash / spawn error / non-abort exit).
    NoResult,
}

#[derive(Debug, Clone)]
struct TestRecord {
    rel_path: String,
    outcome: Outcome,
    /// FAIL/SKIP/timeout reason (raw, untruncated as emitted by the child).
    reason: String,
}

/// Driver configuration parsed from CLI flags.
#[derive(Debug, Clone)]
struct DriverCfg {
    corpus_root: PathBuf,
    out_path: PathBuf,
    jobs: usize,
    timeout_ms: u64,
    /// Run every Nth discovered test (1 = all). Representative sample mode.
    sample_every: usize,
    /// Cap of tests per leaf directory (0 = unlimited).
    per_dir_cap: usize,
    /// Hard cap on total tests run (0 = unlimited).
    limit: usize,
    /// Restrict to a single top-level chapter (e.g. "language", "built-ins").
    chapter: Option<String>,
    /// Skip the slow Intl chapter unless explicitly requested.
    include_intl: bool,
    /// Skip the staging chapter (proposals, churny) unless requested.
    include_staging: bool,
    /// Milliseconds to sleep before each child spawn (THROTTLE). The full ~48k
    /// corpus froze the machine because the RATE of process create/reap
    /// overwhelmed the OS even at low --jobs. A small per-spawn delay paces the
    /// churn so it stays sane while keeping perfect crash isolation. Default 0
    /// (no throttle) for small/sampled runs; set e.g. --spawn-delay-ms 4 for the
    /// full corpus. 0 = original behavior.
    spawn_delay_ms: u64,
}

impl DriverCfg {
    fn from_cli(cli: &Cli) -> Result<Self, String> {
        let pos = cli.positional();
        // The corpus root can be given positionally; otherwise default to the
        // known local checkout. We accept either the repo-relative path or an
        // absolute one.
        let corpus_root = pos
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(default_corpus_root);
        let out_path = cli
            .flag("out")
            .map(PathBuf::from)
            .unwrap_or_else(default_report_path);
        let jobs = parse_usize(cli.flag("jobs"), 4).max(1);
        let timeout_ms = cli
            .flag("timeout-ms")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(10_000)
            .max(250);
        let sample_every = parse_usize(cli.flag("sample"), 1).max(1);
        let per_dir_cap = parse_usize(cli.flag("per-dir"), 0);
        let limit = parse_usize(cli.flag("limit"), 0);
        let chapter = cli.flag("chapter").map(|s| s.to_string());
        let include_intl = cli.has("include-intl") || chapter.as_deref() == Some("intl402");
        let include_staging = cli.has("include-staging") || chapter.as_deref() == Some("staging");
        let test_root = corpus_root.join("test");
        if !test_root.is_dir() {
            return Err(format!(
                "corpus test dir not found at {} (pass the test262 checkout root positionally)",
                test_root.display()
            ));
        }
        Ok(Self {
            corpus_root,
            out_path,
            jobs,
            timeout_ms,
            sample_every,
            per_dir_cap,
            limit,
            chapter,
            include_intl,
            include_staging,
            spawn_delay_ms: cli
                .flag("spawn-delay-ms")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0),
        })
    }
}

fn parse_usize(s: Option<&str>, default: usize) -> usize {
    s.and_then(|s| s.parse::<usize>().ok()).unwrap_or(default)
}

fn default_corpus_root() -> PathBuf {
    // Resolved relative to the current working dir (the repo root when invoked
    // by the workflow). Falls back gracefully — `from_cli` validates existence.
    PathBuf::from("conformance/tmp/test262")
}

fn default_report_path() -> PathBuf {
    PathBuf::from("conformance/test262_report.json")
}

/// Recursively collect `.js` test files under `dir`, honoring the test262
/// `_FIXTURE.js` exclusion (those are includes, never standalone tests) and
/// the driver's per-directory cap / chapter filters. Results are returned in
/// a deterministic sorted order so sampling is reproducible.
fn collect_tests(cfg: &DriverCfg) -> Result<Vec<PathBuf>, String> {
    let test_root = cfg.corpus_root.join("test");
    let mut chapters: Vec<PathBuf> = Vec::new();
    if let Some(ch) = &cfg.chapter {
        chapters.push(test_root.join(ch));
    } else {
        // Default chapter set: the core language + built-ins + annexB. Intl and
        // staging are opt-in (slow / churny / proposal-heavy). `harness/` holds
        // self-tests for the harness — included (they exercise real engine
        // behavior and are cheap).
        for ch in ["language", "built-ins", "annexB", "harness"] {
            chapters.push(test_root.join(ch));
        }
        if cfg.include_intl {
            chapters.push(test_root.join("intl402"));
        }
        if cfg.include_staging {
            chapters.push(test_root.join("staging"));
        }
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for ch in chapters {
        if !ch.is_dir() {
            continue;
        }
        walk_dir(&ch, cfg.per_dir_cap, &mut files)?;
    }
    files.sort();
    // Apply the every-Nth sample.
    if cfg.sample_every > 1 {
        files = files
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % cfg.sample_every == 0)
            .map(|(_, p)| p)
            .collect();
    }
    if cfg.limit > 0 && files.len() > cfg.limit {
        files.truncate(cfg.limit);
    }
    Ok(files)
}

/// Depth-first directory walk. `per_dir_cap` (when > 0) limits how many test
/// files are taken from each directory that directly contains tests — a cheap
/// stratified sample that keeps coverage broad across the deep test262 tree.
fn walk_dir(dir: &Path, per_dir_cap: usize, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    let mut subdirs: Vec<PathBuf> = Vec::new();
    let mut here: Vec<PathBuf> = Vec::new();
    for e in entries {
        let e = match e {
            Ok(e) => e,
            Err(_) => continue,
        };
        let p = e.path();
        let ft = match e.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            subdirs.push(p);
        } else if ft.is_file() {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // test262 convention: `_FIXTURE.js` files are includes, NOT tests.
            if name.ends_with(".js") && !name.contains("_FIXTURE") {
                here.push(p);
            }
        }
    }
    here.sort();
    if per_dir_cap > 0 && here.len() > per_dir_cap {
        // Even stride across the directory so the cap samples the whole dir,
        // not just the alphabetical head.
        let stride = here.len().div_ceil(per_dir_cap);
        here = here.into_iter().step_by(stride).take(per_dir_cap).collect();
    }
    out.extend(here);
    subdirs.sort();
    for d in subdirs {
        walk_dir(&d, per_dir_cap, out)?;
    }
    Ok(())
}

/// Run a single test in a child process with a wall-clock timeout. Returns the
/// classified outcome + reason. On timeout the child is killed (its tree is
/// `panic=abort` single-process so a kill is clean). No-deps timeout: a watcher
/// thread kills the child at the deadline; the main thread blocks on `wait`.
fn run_one(exe: &Path, file: &Path, timeout_ms: u64) -> (Outcome, String) {
    let mut cmd = Command::new(exe);
    cmd.args(["--type", "test262-one"])
        .arg(file)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // Inner per-task budget: keep the engine's own watchdog well under our
    // process-level timeout so the child self-terminates pathological regex /
    // loops before we have to kill it (cleaner, keeps a RESULT line possible).
    let inner_budget = (timeout_ms.saturating_sub(1500)).max(1000);
    cmd.env("CV_JS_TIME_BUDGET_MS", inner_budget.to_string());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (Outcome::NoResult, format!("spawn-error: {e}")),
    };

    // CONCURRENT DRAIN: read the child's stdout on a dedicated thread so the
    // child can never block on a full OS pipe buffer (~64KB). The OLD code read
    // stdout only AFTER the wait loop, so a test that floods stdout (a runaway
    // console.log, or a panic backtrace) filled the pipe → child blocked on
    // write → never exited → the parent wedged. This is the deadlock that hung
    // the full-corpus run. Draining concurrently makes the timeout authoritative.
    let stdout = child.stdout.take();
    let reader = stdout.map(|mut s| {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = s.read_to_string(&mut buf);
            buf
        })
    });

    let deadline = Duration::from_millis(timeout_ms);
    let start = Instant::now();
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if start.elapsed() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break,
        }
    }

    // The child has exited (or been killed) → its stdout is closed → the reader
    // thread's read_to_string returns. Join with a short grace so we never block
    // forever even if the handle lingers.
    let buf = match reader {
        Some(h) => h.join().unwrap_or_default(),
        None => String::new(),
    };

    if timed_out {
        return (Outcome::Timeout, format!("timeout>{timeout_ms}ms"));
    }
    classify_output(&buf)
}

/// Parse the child's stdout for the `RESULT:` line emitted by `run_test262_one`.
fn classify_output(out: &str) -> (Outcome, String) {
    for line in out.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("RESULT:") {
            let rest = rest.trim();
            if rest == "PASS" {
                return (Outcome::Pass, String::new());
            }
            if let Some(reason) = rest.strip_prefix("SKIP:") {
                return (Outcome::Skip, reason.trim().to_string());
            }
            if let Some(reason) = rest.strip_prefix("FAIL") {
                // Format: `FAIL | <reason>` — trim whitespace first, then the
                // separator pipe, then whitespace again.
                let reason = reason.trim().trim_start_matches('|').trim().to_string();
                return (Outcome::Fail, reason);
            }
            // Unknown RESULT shape — treat as no-result so it is visible.
            return (Outcome::NoResult, format!("unknown-result: {rest}"));
        }
    }
    (Outcome::NoResult, "no-result-line".to_string())
}

/// Normalize a FAIL reason into a stable CLUSTER key so the report ranks the
/// biggest classes of failure (the actionable wins) rather than 1000 unique
/// strings. This is the heart of "fix by frequency". The mapping is
/// conservative: it strips identifiers/values so that
/// "x is not a function" and "y is not a function" collapse, while keeping
/// the structural error class.
fn cluster_for(outcome: Outcome, reason: &str) -> String {
    match outcome {
        Outcome::Timeout => return "Timeout (wall-clock)".to_string(),
        Outcome::NoResult => {
            if reason.starts_with("spawn-error") {
                return "Driver: spawn error".to_string();
            }
            // No RESULT line = the child aborted (panic) or exited abnormally
            // before printing. This is an engine crash class — high value.
            return "Crash / abort (no RESULT line)".to_string();
        }
        Outcome::Skip => {
            // Skips are clustered separately (they are not failures) but we
            // still want the frequency breakdown.
            return format!("SKIP:{}", reason);
        }
        Outcome::Pass => return "PASS".to_string(),
        Outcome::Fail => {}
    }

    let r = reason;

    // Negative-test mismatches (expected a throw, got none).
    if r.starts_with("negative:") {
        return "Negative test: expected throw missing".to_string();
    }
    if r == "ok-but-failed" {
        return "Positive test reported failure".to_string();
    }

    // Internal engine errors surfaced by describe_js_throw as Internal(...).
    if let Some(rest) = r.strip_prefix("Internal(") {
        let head = rest.split([':', '(']).next().unwrap_or(rest).trim();
        return format!("Internal: {}", truncate(head, 48));
    }

    // Thrown native errors. describe_js_throw renders Object throws as
    // `Throw[Name: message] keys=... stack=...`. Cluster on Name + a
    // normalized message shape.
    if let Some(rest) = r.strip_prefix("Throw[") {
        let inner = rest.split(']').next().unwrap_or(rest);
        let (name, msg) = match inner.split_once(':') {
            Some((n, m)) => (n.trim(), m.trim()),
            None => (inner.trim(), ""),
        };
        return classify_thrown(name, msg);
    }
    // Non-object throws render as `Throw(<value>)`.
    if let Some(rest) = r.strip_prefix("Throw(") {
        let v = rest.trim_end_matches(')');
        return format!("Threw value: {}", truncate(normalize_message(v).as_str(), 48));
    }

    // assert.js failures come through as plain messages (sta.js's Test262Error
    // is an Object → handled above as Throw[Test262Error: ...]). Anything else:
    format!("Other: {}", truncate(normalize_message(r).as_str(), 56))
}

/// Cluster a thrown error by its constructor name + a canonicalized message
/// pattern, so the report surfaces the dominant *kinds* of failure.
fn classify_thrown(name: &str, msg: &str) -> String {
    let nm = normalize_message(msg);
    // Common, high-signal patterns get their own buckets.
    if nm.contains("is not a function") {
        return format!("{name}: <x> is not a function");
    }
    if nm.contains("is not defined") {
        return format!("{name}: <x> is not defined");
    }
    if nm.contains("is not a constructor") {
        return format!("{name}: <x> is not a constructor");
    }
    if nm.contains("cannot read") || nm.contains("of undefined") || nm.contains("of null") {
        return format!("{name}: property access on undefined/null");
    }
    // assert.js / sta.js assertion families. These come through as either a
    // `Test262Error` or a plain `Error` (sta.js's Test262Error message is
    // surfaced; the constructor name varies). Cluster on the assertion SHAPE so
    // the dominant assertion classes (the real signal) rank together.
    if name == "Test262Error" || name == "Error" {
        if nm.contains("expected a") && nm.contains("to be thrown but no exception") {
            return "assert.throws: expected throw, none thrown".to_string();
        }
        if nm.contains("expected a") && nm.contains("to be thrown") {
            return "assert.throws: wrong error type thrown".to_string();
        }
        if nm.contains("expected samevalue") || nm.contains("expected truthy") || nm.contains("to be true") {
            return "assert: value mismatch (SameValue/truthy)".to_string();
        }
        if nm.contains("descriptor") && (nm.contains("enumerable") || nm.contains("writable") || nm.contains("configurable")) {
            return "assert: property descriptor mismatch".to_string();
        }
        if nm.contains("should be an own") || nm.contains("own property") {
            return "assert: missing/own-property mismatch".to_string();
        }
        return format!("{name} (assert): {}", truncate(nm.as_str(), 48));
    }
    if nm.is_empty() {
        return format!("{name} (no message)");
    }
    format!("{name}: {}", truncate(nm.as_str(), 56))
}

/// Lowercase + strip quoted identifiers/numbers so "'foo' is not a function"
/// and "bar is not a function" map to one shape.
fn normalize_message(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    let mut chars = msg.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' | '`' => {
                // Drop the quoted run.
                while let Some(&n) = chars.peek() {
                    chars.next();
                    if n == c {
                        break;
                    }
                }
                out.push_str("<x>");
            }
            d if d.is_ascii_digit() => {
                while let Some(&n) = chars.peek() {
                    if n.is_ascii_digit() || n == '.' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push_str("<n>");
            }
            _ => out.push(c.to_ascii_lowercase()),
        }
    }
    // collapse internal whitespace
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}…")
    }
}

/// Aggregated counts + clusters.
#[derive(Default)]
struct Aggregate {
    total: usize,
    pass: usize,
    fail: usize,
    skip: usize,
    timeout: usize,
    no_result: usize,
    /// cluster -> (count, up to 30 representative rel paths)
    fail_clusters: HashMap<String, (usize, Vec<String>)>,
    skip_clusters: HashMap<String, usize>,
    /// per-chapter (top-level dir) pass/fail/skip
    by_chapter: HashMap<String, (usize, usize, usize)>,
}

impl Aggregate {
    fn record(&mut self, rec: &TestRecord) {
        self.total += 1;
        let chapter = rec
            .rel_path
            .split(['/', '\\'])
            .next()
            .unwrap_or("?")
            .to_string();
        let ch = self.by_chapter.entry(chapter).or_insert((0, 0, 0));
        match rec.outcome {
            Outcome::Pass => {
                self.pass += 1;
                ch.0 += 1;
            }
            Outcome::Skip => {
                self.skip += 1;
                ch.2 += 1;
                let key = cluster_for(rec.outcome, &rec.reason);
                *self.skip_clusters.entry(key).or_insert(0) += 1;
            }
            Outcome::Fail | Outcome::Timeout | Outcome::NoResult => {
                self.fail += 1;
                ch.1 += 1;
                if rec.outcome == Outcome::Timeout {
                    self.timeout += 1;
                }
                if rec.outcome == Outcome::NoResult {
                    self.no_result += 1;
                }
                let key = cluster_for(rec.outcome, &rec.reason);
                let e = self.fail_clusters.entry(key).or_insert((0, Vec::new()));
                e.0 += 1;
                // Keep up to 30 representative paths per cluster (was 5): a
                // larger sample makes the report actionable — the failing-file
                // spread within a cluster reveals the real sub-pattern to fix.
                if e.1.len() < 30 {
                    e.1.push(rec.rel_path.clone());
                }
            }
        }
    }

    /// pass / (pass + fail). Excludes skips. Returns percent string.
    fn pass_rate_pct(&self) -> f64 {
        let denom = self.pass + self.fail;
        if denom == 0 {
            0.0
        } else {
            (self.pass as f64) * 100.0 / (denom as f64)
        }
    }
}

/// Entry point for `--type test262-run`.
pub fn run(cli: &Cli) -> Result<(), String> {
    let cfg = DriverCfg::from_cli(cli)?;
    let exe = std::env::current_exe().map_err(|e| format!("current exe: {e}"))?;

    eprintln!("test262 driver: discovering tests under {} ...", cfg.corpus_root.join("test").display());
    let files = collect_tests(&cfg)?;
    let total = files.len();
    if total == 0 {
        return Err("no tests matched the selection".to_string());
    }
    eprintln!(
        "test262 driver: {} tests selected | jobs={} timeout={}ms sample-every={} per-dir-cap={} limit={} chapter={:?}",
        total, cfg.jobs, cfg.timeout_ms, cfg.sample_every, cfg.per_dir_cap, cfg.limit, cfg.chapter
    );

    let corpus_root = cfg.corpus_root.clone();
    let files = Arc::new(files);
    let next = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<TestRecord>>> = Arc::new(Mutex::new(Vec::with_capacity(total)));
    let start = Instant::now();

    let mut handles = Vec::new();
    for _ in 0..cfg.jobs {
        let files = Arc::clone(&files);
        let next = Arc::clone(&next);
        let done = Arc::clone(&done);
        let results = Arc::clone(&results);
        let exe = exe.clone();
        let corpus_root = corpus_root.clone();
        let timeout_ms = cfg.timeout_ms;
        let spawn_delay_ms = cfg.spawn_delay_ms;
        let handle = std::thread::spawn(move || {
            loop {
                let idx = next.fetch_add(1, Ordering::SeqCst);
                if idx >= files.len() {
                    break;
                }
                // THROTTLE: pace child spawns so the OS process create/reap rate
                // can't overwhelm the machine on the full corpus (the freeze).
                if spawn_delay_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(spawn_delay_ms));
                }
                let file = &files[idx];
                let (outcome, reason) = run_one(&exe, file, timeout_ms);
                let rel = rel_path(&corpus_root, file);
                let rec = TestRecord {
                    rel_path: rel,
                    outcome,
                    reason,
                };
                results.lock().unwrap().push(rec);
                let d = done.fetch_add(1, Ordering::SeqCst) + 1;
                if d % 250 == 0 || d == files.len() {
                    eprintln!(
                        "  progress: {}/{} ({:.0}%) elapsed {:.1}s",
                        d,
                        files.len(),
                        d as f64 * 100.0 / files.len() as f64,
                        start.elapsed().as_secs_f64()
                    );
                }
            }
        });
        handles.push(handle);
    }
    for h in handles {
        let _ = h.join();
    }

    let records = Arc::try_unwrap(results)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_default();
    let elapsed = start.elapsed();

    let mut agg = Aggregate::default();
    for rec in &records {
        agg.record(rec);
    }

    // Console summary.
    eprintln!("================ test262 BASELINE ================");
    eprintln!("total run : {}", agg.total);
    eprintln!("pass      : {}", agg.pass);
    eprintln!("fail      : {} (incl {} timeout, {} crash/no-result)", agg.fail, agg.timeout, agg.no_result);
    eprintln!("skip      : {} (honest; excluded from pass rate)", agg.skip);
    eprintln!("PASS RATE : {:.2}% (pass / (pass+fail))", agg.pass_rate_pct());
    eprintln!("wall time : {:.1}s", elapsed.as_secs_f64());
    eprintln!("------------ top failure clusters ----------------");
    let ranked = ranked_failclusters(&agg);
    for (i, (cluster, count, _)) in ranked.iter().take(20).enumerate() {
        eprintln!("  {:>2}. [{:>5}] {}", i + 1, count, cluster);
    }

    let report = build_json(&cfg, &agg, &ranked, elapsed);
    if let Some(parent) = cfg.out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&cfg.out_path, report)
        .map_err(|e| format!("write report {}: {e}", cfg.out_path.display()))?;
    eprintln!("report written: {}", cfg.out_path.display());
    Ok(())
}

/// IN-PROCESS test262 runner (`--type test262-inproc`). Runs every selected
/// test inside THIS process — no child spawn per test — to eliminate the
/// process-spawn-storm that froze the machine when running the full ~48k corpus
/// process-per-test (the OS chokes on the create/reap RATE, regardless of
/// concurrency). Single-threaded by design: the engine + thread-locals (watchdog
/// deadline, native-this stack) are per-thread, and in-process execution is fast
/// enough (~no per-test process overhead) that one thread runs the full corpus in
/// well under the spawn runner's wall time.
///
/// Isolation trade-off vs the spawn runner: a genuine engine PANIC can't be
/// contained by a separate process, so each test runs under `catch_unwind`
/// (counted as a crash/NoResult, like the spawn runner's missing-RESULT). An
/// infinite loop is bounded by the per-task wall-clock watchdog
/// (CV_JS_TIME_BUDGET_MS, default-on). The verdict for each test comes from the
/// SAME `crate::run_test262_source` the spawn runner uses, so pass/fail/skip are
/// identical. Writes the identical JSON report.
pub fn run_inproc(cli: &Cli) -> Result<(), String> {
    let cfg = DriverCfg::from_cli(cli)?;
    eprintln!(
        "test262 IN-PROCESS runner: discovering tests under {} ...",
        cfg.corpus_root.join("test").display()
    );
    let files = collect_tests(&cfg)?;
    let total = files.len();
    if total == 0 {
        return Err("no tests matched the selection".to_string());
    }
    eprintln!(
        "test262 in-process: {} tests selected | sample-every={} per-dir-cap={} limit={} chapter={:?}",
        total, cfg.sample_every, cfg.per_dir_cap, cfg.limit, cfg.chapter
    );

    // Silence the default panic hook so a caught per-test panic doesn't spam
    // stderr with backtraces (we record it as a crash verdict and move on).
    std::panic::set_hook(Box::new(|_| {}));

    let start = Instant::now();
    let mut records: Vec<TestRecord> = Vec::with_capacity(total);
    for (i, file) in files.iter().enumerate() {
        let rel = rel_path(&cfg.corpus_root, file);
        let (outcome, reason) = match std::fs::read_to_string(file) {
            Ok(src) => {
                let path_str = file.to_string_lossy().to_string();
                // Run EACH test on its own bounded-stack worker thread. This is
                // the crash-isolation boundary that makes the in-process runner
                // robust: a STACK OVERFLOW (uncatchable by catch_unwind — there's
                // no stack left to unwind) kills only this sub-thread, not the
                // whole process, and `join()` returns Err so we count it as a
                // crash and continue. A normal unwindable panic is caught by the
                // inner catch_unwind. Threads are ~100x cheaper than processes
                // and don't churn the OS process table — so this avoids the
                // 48k-process spawn storm that froze the machine. 64 MB stack is
                // ample for legitimate deep recursion while keeping an overflow
                // contained. Infinite loops are bounded by the per-task wall-clock
                // watchdog (CV_JS_TIME_BUDGET_MS, default-on).
                let builder = std::thread::Builder::new().stack_size(64 * 1024 * 1024);
                match builder.spawn(move || {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        crate::run_test262_source(&path_str, &src)
                    }))
                }) {
                    Ok(handle) => match handle.join() {
                        Ok(Ok(crate::T262Verdict::Pass)) => (Outcome::Pass, String::new()),
                        Ok(Ok(crate::T262Verdict::Fail(r))) => (Outcome::Fail, r),
                        Ok(Ok(crate::T262Verdict::Skip(r))) => (Outcome::Skip, r),
                        Ok(Err(_)) => {
                            (Outcome::NoResult, "panic (caught)".to_string())
                        }
                        Err(_) => {
                            (Outcome::NoResult, "crash (stack-overflow/abort)".to_string())
                        }
                    },
                    Err(e) => (Outcome::NoResult, format!("thread-spawn-error: {e}")),
                }
            }
            Err(e) => (Outcome::NoResult, format!("read-error: {e}")),
        };
        records.push(TestRecord {
            rel_path: rel,
            outcome,
            reason,
        });
        let d = i + 1;
        if d % 1000 == 0 || d == total {
            eprintln!(
                "  progress: {}/{} ({:.0}%) elapsed {:.1}s",
                d,
                total,
                d as f64 * 100.0 / total as f64,
                start.elapsed().as_secs_f64()
            );
        }
    }
    let elapsed = start.elapsed();

    let mut agg = Aggregate::default();
    for rec in &records {
        agg.record(rec);
    }
    eprintln!("============ test262 IN-PROCESS RESULT ============");
    eprintln!("total run : {}", agg.total);
    eprintln!("pass      : {}", agg.pass);
    eprintln!(
        "fail      : {} (incl {} timeout, {} crash/no-result)",
        agg.fail, agg.timeout, agg.no_result
    );
    eprintln!("skip      : {} (honest; excluded from pass rate)", agg.skip);
    eprintln!("PASS RATE : {:.2}% (pass / (pass+fail))", agg.pass_rate_pct());
    eprintln!("wall time : {:.1}s", elapsed.as_secs_f64());
    let ranked = ranked_failclusters(&agg);
    eprintln!("------------ top failure clusters ----------------");
    for (i, (cluster, count, _)) in ranked.iter().take(20).enumerate() {
        eprintln!("  {:>2}. [{:>5}] {}", i + 1, count, cluster);
    }
    let report = build_json(&cfg, &agg, &ranked, elapsed);
    if let Some(parent) = cfg.out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&cfg.out_path, report)
        .map_err(|e| format!("write report {}: {e}", cfg.out_path.display()))?;
    eprintln!("report written: {}", cfg.out_path.display());
    Ok(())
}

fn rel_path(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("test/")
        .to_string()
}

fn ranked_failclusters(agg: &Aggregate) -> Vec<(String, usize, Vec<String>)> {
    let mut v: Vec<(String, usize, Vec<String>)> = agg
        .fail_clusters
        .iter()
        .map(|(k, (c, ex))| (k.clone(), *c, ex.clone()))
        .collect();
    // Sort by count desc, then name for determinism.
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

/// Hand-rolled JSON writer (no serde — strict no-deps policy).
fn build_json(
    cfg: &DriverCfg,
    agg: &Aggregate,
    ranked: &[(String, usize, Vec<String>)],
    elapsed: Duration,
) -> String {
    let mut s = String::with_capacity(8192);
    s.push_str("{\n");
    s.push_str(&format!("  \"schema\": \"test262-report/v1\",\n"));
    s.push_str(&format!("  \"corpus_root\": {},\n", jstr(&cfg.corpus_root.to_string_lossy())));
    s.push_str("  \"config\": {\n");
    s.push_str(&format!("    \"jobs\": {},\n", cfg.jobs));
    s.push_str(&format!("    \"timeout_ms\": {},\n", cfg.timeout_ms));
    s.push_str(&format!("    \"sample_every\": {},\n", cfg.sample_every));
    s.push_str(&format!("    \"per_dir_cap\": {},\n", cfg.per_dir_cap));
    s.push_str(&format!("    \"limit\": {},\n", cfg.limit));
    s.push_str(&format!("    \"chapter\": {},\n", opt_jstr(cfg.chapter.as_deref())));
    s.push_str(&format!("    \"include_intl\": {},\n", cfg.include_intl));
    s.push_str(&format!("    \"include_staging\": {}\n", cfg.include_staging));
    s.push_str("  },\n");
    s.push_str("  \"summary\": {\n");
    s.push_str(&format!("    \"total\": {},\n", agg.total));
    s.push_str(&format!("    \"pass\": {},\n", agg.pass));
    s.push_str(&format!("    \"fail\": {},\n", agg.fail));
    s.push_str(&format!("    \"skip\": {},\n", agg.skip));
    s.push_str(&format!("    \"timeout\": {},\n", agg.timeout));
    s.push_str(&format!("    \"no_result_crash\": {},\n", agg.no_result));
    s.push_str(&format!("    \"pass_rate_pct\": {:.2},\n", agg.pass_rate_pct()));
    s.push_str(&format!("    \"wall_seconds\": {:.1}\n", elapsed.as_secs_f64()));
    s.push_str("  },\n");

    // Per-chapter breakdown.
    s.push_str("  \"by_chapter\": {\n");
    let mut chapters: Vec<(&String, &(usize, usize, usize))> = agg.by_chapter.iter().collect();
    chapters.sort_by(|a, b| a.0.cmp(b.0));
    for (i, (name, (p, f, sk))) in chapters.iter().enumerate() {
        let denom = p + f;
        let rate = if denom == 0 { 0.0 } else { *p as f64 * 100.0 / denom as f64 };
        let comma = if i + 1 == chapters.len() { "" } else { "," };
        s.push_str(&format!(
            "    {}: {{ \"pass\": {}, \"fail\": {}, \"skip\": {}, \"pass_rate_pct\": {:.2} }}{}\n",
            jstr(name), p, f, sk, rate, comma
        ));
    }
    s.push_str("  },\n");

    // Ranked failure clusters with representative files.
    s.push_str("  \"failure_clusters\": [\n");
    for (i, (cluster, count, examples)) in ranked.iter().enumerate() {
        let comma = if i + 1 == ranked.len() { "" } else { "," };
        s.push_str("    {\n");
        s.push_str(&format!("      \"cluster\": {},\n", jstr(cluster)));
        s.push_str(&format!("      \"count\": {},\n", count));
        s.push_str("      \"examples\": [");
        for (j, ex) in examples.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            s.push_str(&jstr(ex));
        }
        s.push_str("]\n");
        s.push_str(&format!("    }}{}\n", comma));
    }
    s.push_str("  ],\n");

    // Skip clusters (honest accounting of what we declined and why).
    s.push_str("  \"skip_clusters\": [\n");
    let mut skips: Vec<(&String, &usize)> = agg.skip_clusters.iter().collect();
    skips.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (i, (cluster, count)) in skips.iter().enumerate() {
        let comma = if i + 1 == skips.len() { "" } else { "," };
        s.push_str(&format!(
            "    {{ \"cluster\": {}, \"count\": {} }}{}\n",
            jstr(cluster), count, comma
        ));
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

fn opt_jstr(s: Option<&str>) -> String {
    match s {
        Some(v) => jstr(v),
        None => "null".to_string(),
    }
}

/// JSON-escape a string into a quoted literal.
fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_pass_fail_skip() {
        assert_eq!(classify_output("RESULT: PASS").0, Outcome::Pass);
        let (o, r) = classify_output("RESULT: FAIL | Throw[TypeError: x is not a function]");
        assert_eq!(o, Outcome::Fail);
        assert!(r.contains("not a function"));
        let (o, r) = classify_output("RESULT: SKIP:feature");
        assert_eq!(o, Outcome::Skip);
        assert_eq!(r, "feature");
        assert_eq!(classify_output("garbage with no result").0, Outcome::NoResult);
    }

    #[test]
    fn cluster_collapses_identifiers() {
        let a = cluster_for(Outcome::Fail, "Throw[TypeError: foo is not a function] keys=[] stack=");
        let b = cluster_for(Outcome::Fail, "Throw[TypeError: bar is not a function] keys=[] stack=");
        assert_eq!(a, b, "different identifiers must cluster together");
        assert_eq!(a, "TypeError: <x> is not a function");
    }

    #[test]
    fn cluster_negative_and_crash() {
        assert_eq!(
            cluster_for(Outcome::Fail, "negative:expected-throw-missing"),
            "Negative test: expected throw missing"
        );
        assert_eq!(
            cluster_for(Outcome::Timeout, "timeout>10000ms"),
            "Timeout (wall-clock)"
        );
        assert_eq!(
            cluster_for(Outcome::NoResult, "no-result-line"),
            "Crash / abort (no RESULT line)"
        );
    }

    #[test]
    fn cluster_test262error_normalizes() {
        let a = cluster_for(
            Outcome::Fail,
            "Throw[Test262Error: Expected SameValue(«1», «2») to be true] keys=[] stack=",
        );
        let b = cluster_for(
            Outcome::Fail,
            "Throw[Test262Error: Expected SameValue(«5», «9») to be true] keys=[] stack=",
        );
        assert_eq!(a, b, "Test262Error values must collapse");
        assert_eq!(a, "assert: value mismatch (SameValue/truthy)");
    }

    #[test]
    fn cluster_assert_throws_families() {
        // Both plain Error and Test262Error name the assert.throws family.
        let a = cluster_for(
            Outcome::Fail,
            "Throw[Error: 1n + 1 throws TypeError Expected a TypeError to be thrown but no exception was thrown at all] keys=[\"message\"] stack=",
        );
        let b = cluster_for(
            Outcome::Fail,
            "Throw[Test262Error: Expected a SyntaxError to be thrown but no exception was thrown at all] keys=[] stack=",
        );
        assert_eq!(a, b);
        assert_eq!(a, "assert.throws: expected throw, none thrown");
    }

    #[test]
    fn fail_reason_strips_pipe_separator() {
        // The child prints `RESULT: FAIL | <reason>`; the leading pipe must not
        // leak into the clustered reason.
        let (o, r) = classify_output("RESULT: FAIL | Throw[Error: boom] keys=[] stack=");
        assert_eq!(o, Outcome::Fail);
        assert!(r.starts_with("Throw["), "pipe leaked: {r:?}");
    }

    #[test]
    fn pass_rate_excludes_skips() {
        let mut agg = Aggregate::default();
        agg.record(&TestRecord { rel_path: "language/a.js".into(), outcome: Outcome::Pass, reason: String::new() });
        agg.record(&TestRecord { rel_path: "language/b.js".into(), outcome: Outcome::Fail, reason: "x".into() });
        agg.record(&TestRecord { rel_path: "language/c.js".into(), outcome: Outcome::Skip, reason: "feature".into() });
        // 1 pass, 1 fail, 1 skip => 50%, not 33%.
        assert!((agg.pass_rate_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn fixture_files_excluded_by_walk() {
        // Build a tiny temp tree.
        let base = std::env::temp_dir().join(format!("t262walk_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&base);
        std::fs::write(base.join("real.js"), "x").unwrap();
        std::fs::write(base.join("dep_FIXTURE.js"), "x").unwrap();
        let mut out = Vec::new();
        walk_dir(&base, 0, &mut out).unwrap();
        let names: Vec<String> = out
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"real.js".to_string()));
        assert!(!names.iter().any(|n| n.contains("_FIXTURE")));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn json_escapes_quotes_and_newlines() {
        assert_eq!(jstr("a\"b\nc"), "\"a\\\"b\\nc\"");
    }
}
