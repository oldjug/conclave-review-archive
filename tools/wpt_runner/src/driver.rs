//! The aggregating WPT driver: discover tests → run each in a child process →
//! parse `WPT-RESULT:`/`WPT-SUMMARY:` lines → aggregate + cluster → JSON report.
//!
//! Mirrors the test262 driver's machine-safety design: a bounded worker pool, a
//! per-test wall-clock kill, and `--sample`/`--limit` sample modes so a fast
//! representative baseline runs before any full sweep. One bounded driver run at
//! a time; never concurrent with a build.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cv_base::cli::Cli;

/// One subtest result harvested from a child's `WPT-RESULT:` line.
#[derive(Debug, Clone)]
pub struct SubResult {
    pub area: String,
    pub file: String,
    pub name: String,
    pub status: SubStatus,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubStatus {
    Pass,
    Fail,
    Timeout,
    NotRun,
    PreconditionFailed,
}

/// File-level outcome (distinct from per-subtest outcome): did the doc even load
/// the harness, did the child crash / time out, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileOutcome {
    /// Ran and produced subtest results.
    Ok,
    /// Child timed out (killed at the deadline).
    Timeout,
    /// Child produced no `WPT-SUMMARY:` line (crash/abort/spawn error) or printed
    /// a `WPT-HARNESS-ERR:` line (the doc never loaded our shim).
    HarnessError,
}

#[derive(Debug, Clone)]
pub struct FileRecord {
    pub area: String,
    pub rel_path: String,
    pub outcome: FileOutcome,
    pub harness_error: String,
    pub subs: Vec<SubResult>,
}

/// Driver configuration parsed from CLI flags.
#[derive(Debug, Clone)]
pub struct DriverCfg {
    pub corpus_root: PathBuf,
    pub exe: PathBuf,
    pub out_path: PathBuf,
    pub jobs: usize,
    pub timeout_ms: u64,
    /// Run every Nth discovered test (1 = all). Representative sample mode.
    pub sample_every: usize,
    /// Hard cap on total tests run (0 = unlimited).
    pub limit: usize,
    /// Restrict to a single top-level area directory (e.g. "dom", "css").
    pub area: Option<String>,
}

impl DriverCfg {
    pub fn from_cli(cli: &Cli) -> Result<Self, String> {
        let pos = cli.positional();
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
            .unwrap_or(20_000)
            .max(1000);
        let sample_every = parse_usize(cli.flag("sample"), 1).max(1);
        let limit = parse_usize(cli.flag("limit"), 0);
        let area = cli.flag("area").map(|s| s.to_string());

        // Resolve the conclave executable: explicit --exe, else the sibling of
        // this binary in the same target dir.
        let exe = if let Some(e) = cli.flag("exe") {
            PathBuf::from(e)
        } else {
            resolve_sibling_tb_browser()?
        };
        if !exe.is_file() {
            return Err(format!(
                "conclave executable not found at {} (pass --exe <path>)",
                exe.display()
            ));
        }
        if !corpus_root.is_dir() {
            return Err(format!(
                "corpus dir not found at {} (pass the WPT test root positionally)",
                corpus_root.display()
            ));
        }
        Ok(Self {
            corpus_root,
            exe,
            out_path,
            jobs,
            timeout_ms,
            sample_every,
            limit,
            area,
        })
    }
}

fn parse_usize(s: Option<&str>, default: usize) -> usize {
    s.and_then(|s| s.parse::<usize>().ok()).unwrap_or(default)
}

fn default_corpus_root() -> PathBuf {
    PathBuf::from("conformance/wpt")
}

fn default_report_path() -> PathBuf {
    PathBuf::from("conformance/wpt_report.json")
}

/// conclave.exe lives next to this binary in the same target dir.
fn resolve_sibling_tb_browser() -> Result<PathBuf, String> {
    let me = std::env::current_exe().map_err(|e| format!("current exe: {e}"))?;
    let dir = me
        .parent()
        .ok_or("current exe has no parent dir")?
        .to_path_buf();
    let name = if cfg!(windows) {
        "conclave.exe"
    } else {
        "conclave"
    };
    Ok(dir.join(name))
}

/// Recursively collect `.html`/`.htm` test files under the corpus root, skipping
/// support/helper dirs (`_support`, `support`, `resources`) and helper files
/// (anything that is referenced by `src=` rather than run directly). Results are
/// sorted so sampling is reproducible.
pub fn collect_tests(cfg: &DriverCfg) -> Result<Vec<PathBuf>, String> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(a) = &cfg.area {
        // Area can name a subdir under any of the corpus sets (authored/upstream).
        for set in ["", "authored", "upstream"] {
            let p = if set.is_empty() {
                cfg.corpus_root.join(a)
            } else {
                cfg.corpus_root.join(set).join(a)
            };
            if p.is_dir() {
                roots.push(p);
            }
        }
        if roots.is_empty() {
            roots.push(cfg.corpus_root.join(a));
        }
    } else {
        roots.push(cfg.corpus_root.clone());
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for r in roots {
        walk_dir(&r, &mut files)?;
    }
    files.sort();
    files.dedup();
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

fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // missing dir is not fatal (area may not exist)
    };
    let mut subdirs: Vec<PathBuf> = Vec::new();
    let mut here: Vec<PathBuf> = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        let ft = match e.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if ft.is_dir() {
            // Skip helper/support dirs — they hold included scripts, not tests.
            if matches!(name.as_str(), "_support" | "support" | "resources" | "common") {
                continue;
            }
            subdirs.push(p);
        } else if ft.is_file() {
            let lower = name.to_ascii_lowercase();
            if (lower.ends_with(".html") || lower.ends_with(".htm"))
                // WPT non-test conventions: `.tentative` is fine, but `-ref`/`-manual`
                // tests aren't testharness-runnable.
                && !lower.contains("-manual")
                && !lower.ends_with("-ref.html")
                && !lower.ends_with("-ref.htm")
            {
                here.push(p);
            }
        }
    }
    here.sort();
    out.extend(here);
    subdirs.sort();
    for d in subdirs {
        walk_dir(&d, out)?;
    }
    Ok(())
}

/// Run one test file in a child process with a wall-clock timeout. Returns the
/// file outcome + harvested subtests. No-deps timeout: poll `try_wait` in a short
/// loop and kill at the deadline.
pub fn run_one(cfg: &DriverCfg, file: &Path) -> (FileOutcome, String, Vec<RawSub>) {
    let mut cmd = Command::new(&cfg.exe);
    cmd.args(["--type", "wpt-one"])
        .arg(file)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // Inner per-task budget: keep the engine's own JS watchdog under our
    // process-level timeout so a runaway script self-terminates before we kill it.
    let inner_budget = (cfg.timeout_ms.saturating_sub(3000)).max(2000);
    cmd.env("CV_JS_TIME_BUDGET_MS", inner_budget.to_string());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (FileOutcome::HarnessError, format!("spawn-error: {e}"), Vec::new()),
    };
    // CRITICAL (pipe-buffer deadlock): a WPT test can emit MANY `WPT-RESULT:`
    // lines (e.g. Element-classlist has ~1400 subtests). If we wait for exit
    // BEFORE reading stdout, the child blocks writing once the OS pipe buffer
    // (a few tens of KB) fills, and we block waiting for exit → mutual deadlock
    // → spurious timeout. So we drain stdout on a DEDICATED thread concurrently
    // with the wait loop. The reader thread owns the pipe and exits at EOF
    // (which happens when the child exits or is killed).
    let stdout = child.stdout.take();
    let reader = stdout.map(|mut s| {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = s.read_to_string(&mut buf);
            buf
        })
    });

    let killed = Arc::new(AtomicBool::new(false));
    let deadline = Duration::from_millis(cfg.timeout_ms);
    let start = Instant::now();
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    killed.store(true, Ordering::SeqCst);
                    timed_out = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break,
        }
    }
    // Join the reader (it returns once the pipe hits EOF — guaranteed after the
    // child exits or is killed above).
    let buf = reader.and_then(|h| h.join().ok()).unwrap_or_default();
    if timed_out {
        // Even on timeout we may have captured partial results before the hang;
        // but a genuine timeout means the file didn't finish, so report it as a
        // file timeout (its subtests are not credited).
        return (
            FileOutcome::Timeout,
            format!("timeout>{}ms", cfg.timeout_ms),
            Vec::new(),
        );
    }
    parse_child_output(&buf)
}

/// A raw subtest line before area/file are attached.
#[derive(Debug, Clone)]
pub struct RawSub {
    pub status: SubStatus,
    pub name: String,
    pub message: String,
}

/// Parse a child's stdout for `WPT-RESULT:`/`WPT-SUMMARY:`/`WPT-HARNESS-ERR:`.
pub fn parse_child_output(out: &str) -> (FileOutcome, String, Vec<RawSub>) {
    let mut subs = Vec::new();
    let mut saw_summary = false;
    let mut harness_err = String::new();
    for line in out.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("WPT-RESULT:") {
            if let Some(s) = parse_result_line(rest.trim()) {
                subs.push(s);
            }
        } else if line.starts_with("WPT-SUMMARY:") {
            saw_summary = true;
        } else if let Some(rest) = line.strip_prefix("WPT-HARNESS-ERR:") {
            harness_err = rest.trim().to_string();
        }
    }
    if !harness_err.is_empty() {
        return (FileOutcome::HarnessError, harness_err, subs);
    }
    if !saw_summary && subs.is_empty() {
        return (
            FileOutcome::HarnessError,
            "no WPT-SUMMARY line (crash/abort/no output)".to_string(),
            subs,
        );
    }
    (FileOutcome::Ok, String::new(), subs)
}

/// Parse a `STATUS | name | message` result line.
fn parse_result_line(rest: &str) -> Option<RawSub> {
    let mut parts = rest.splitn(3, '|');
    let status_s = parts.next()?.trim();
    let name = parts.next().unwrap_or("").trim().to_string();
    let message = parts.next().unwrap_or("").trim().to_string();
    let status = match status_s {
        "PASS" => SubStatus::Pass,
        "FAIL" => SubStatus::Fail,
        "TIMEOUT" => SubStatus::Timeout,
        "NOTRUN" => SubStatus::NotRun,
        "PRECONDITION_FAILED" => SubStatus::PreconditionFailed,
        _ => SubStatus::Fail,
    };
    Some(RawSub { status, name, message })
}

/// Derive the test area ("dom", "css", "html", ...) from a corpus-relative path.
/// We look at the path components and return the first that is a known area, else
/// the first component after any `authored`/`upstream` set dir.
pub fn area_for(rel: &str) -> String {
    let comps: Vec<&str> = rel.split(['/', '\\']).filter(|c| !c.is_empty()).collect();
    let mut i = 0;
    while i < comps.len() {
        let c = comps[i];
        if c == "authored" || c == "upstream" {
            i += 1;
            continue;
        }
        return c.to_string();
    }
    "?".to_string()
}

/// Normalize a subtest failure message into a stable cluster key (fix-by-frequency).
/// Strips quoted identifiers / numbers so different instances of the same failure
/// shape collapse together.
pub fn cluster_for(status: SubStatus, message: &str) -> String {
    match status {
        SubStatus::Timeout => return "Subtest timeout".to_string(),
        SubStatus::NotRun => return "Subtest did not run (async never completed)".to_string(),
        SubStatus::PreconditionFailed => {
            return "Precondition failed (optional feature unsupported)".to_string();
        }
        SubStatus::Pass => return "PASS".to_string(),
        SubStatus::Fail => {}
    }
    // Check engine-specific structured-error shapes on the RAW (un-normalized)
    // message FIRST: the meaningful keyword often sits inside quotes that the
    // normalizer collapses to `<x>`. The engine surfaces a missing/undefined
    // method call as `... callee is not callable: undefined (property `x`) ...` —
    // a distinct, very actionable class (a DOM/JS API our engine doesn't have).
    let raw_lower = message.to_ascii_lowercase();
    if raw_lower.contains("not callable") || (raw_lower.contains("callee") && raw_lower.contains("undefined")) {
        return "TypeError: method not callable (missing API)".to_string();
    }

    let n = normalize_message(message);
    if n.is_empty() {
        return "assert: (no message)".to_string();
    }
    // High-signal buckets.
    if n.contains("is not defined") {
        return "ReferenceError: <x> is not defined".to_string();
    }
    if n.contains("is not a function") {
        return "TypeError: <x> is not a function".to_string();
    }
    if n.contains("cannot read") || n.contains("of undefined") || n.contains("of null") {
        return "TypeError: property access on undefined/null".to_string();
    }
    // Any other thrown TypeError (e.g. constructor misuse) — keep it grouped.
    if n.contains("typeerror") {
        return "TypeError: other".to_string();
    }
    if n.starts_with("assert_equals:") {
        return "assert_equals: value mismatch".to_string();
    }
    if n.starts_with("assert_true:") {
        return "assert_true: expected true".to_string();
    }
    if n.starts_with("assert_false:") {
        return "assert_false: expected false".to_string();
    }
    if n.starts_with("assert_throws_js:") {
        return "assert_throws_js: wrong/no throw".to_string();
    }
    if n.starts_with("assert_throws_dom:") {
        return "assert_throws_dom: wrong/no DOMException".to_string();
    }
    if n.starts_with("assert_array_equals:") {
        return "assert_array_equals: mismatch".to_string();
    }
    if n.starts_with("assert_own_property:") {
        return "assert_own_property: missing property".to_string();
    }
    if n.starts_with("assert_idl_attribute:") || n.starts_with("assert_inherits:") {
        return "assert: missing IDL/inherited attribute".to_string();
    }
    // The first token (assert name or error name) as the cluster, else a prefix.
    let head = n.split([':', ' ']).next().unwrap_or(&n);
    format!("Other: {}", truncate(head, 48))
}

fn normalize_message(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    let mut chars = msg.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' | '`' => {
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

/// Aggregated counts + clusters across all run files.
#[derive(Default)]
pub struct Aggregate {
    pub files: usize,
    pub files_ok: usize,
    pub files_timeout: usize,
    pub files_harness_error: usize,
    pub sub_pass: usize,
    pub sub_fail: usize,
    pub sub_timeout: usize,
    pub sub_notrun: usize,
    pub sub_precondition: usize,
    /// cluster -> (count, up to 5 representative "area/file: name")
    pub fail_clusters: HashMap<String, (usize, Vec<String>)>,
    /// area -> (sub_pass, sub_fail, files_with_harness_error)
    pub by_area: HashMap<String, (usize, usize, usize)>,
    /// harness-error reason -> count
    pub harness_error_clusters: HashMap<String, usize>,
}

impl Aggregate {
    pub fn record(&mut self, rec: &FileRecord) {
        self.files += 1;
        let area = self.by_area.entry(rec.area.clone()).or_insert((0, 0, 0));
        match rec.outcome {
            FileOutcome::Ok => self.files_ok += 1,
            FileOutcome::Timeout => {
                self.files_timeout += 1;
                area.2 += 1;
                *self
                    .harness_error_clusters
                    .entry("File timeout".to_string())
                    .or_insert(0) += 1;
            }
            FileOutcome::HarnessError => {
                self.files_harness_error += 1;
                area.2 += 1;
                let key = harness_error_cluster(&rec.harness_error);
                *self.harness_error_clusters.entry(key).or_insert(0) += 1;
            }
        }
        for s in &rec.subs {
            match s.status {
                SubStatus::Pass => {
                    self.sub_pass += 1;
                    self.by_area.entry(rec.area.clone()).or_insert((0, 0, 0)).0 += 1;
                }
                SubStatus::PreconditionFailed => self.sub_precondition += 1,
                other => {
                    self.sub_fail += 1;
                    self.by_area.entry(rec.area.clone()).or_insert((0, 0, 0)).1 += 1;
                    if other == SubStatus::Timeout {
                        self.sub_timeout += 1;
                    }
                    if other == SubStatus::NotRun {
                        self.sub_notrun += 1;
                    }
                    let key = cluster_for(s.status, &s.message);
                    let e = self.fail_clusters.entry(key).or_insert((0, Vec::new()));
                    e.0 += 1;
                    if e.1.len() < 5 {
                        e.1.push(format!("{}: {}", s.file, truncate(&s.name, 60)));
                    }
                }
            }
        }
    }

    /// sub_pass / (sub_pass + sub_fail). Excludes precondition-failed (honest
    /// "feature unsupported" skips). Percent.
    pub fn pass_rate_pct(&self) -> f64 {
        let denom = self.sub_pass + self.sub_fail;
        if denom == 0 {
            0.0
        } else {
            (self.sub_pass as f64) * 100.0 / (denom as f64)
        }
    }
}

fn harness_error_cluster(reason: &str) -> String {
    let r = reason.to_ascii_lowercase();
    if r.contains("did not load the harness") || r.contains("no __wpt") {
        "Harness did not load (doc/script error before shim)".to_string()
    } else if r.starts_with("spawn-error") {
        "Driver: spawn error".to_string()
    } else if r.contains("timeout") {
        "File timeout".to_string()
    } else if r.contains("no wpt-summary") {
        "Crash / abort (no summary line)".to_string()
    } else if r.starts_with("build:") {
        "Page build failed".to_string()
    } else if r.starts_with("harvest:") {
        "Result harvest threw".to_string()
    } else {
        format!("Harness: {}", truncate(reason, 60))
    }
}

/// Entry point.
pub fn run(cli: &Cli) -> Result<(), String> {
    let cfg = DriverCfg::from_cli(cli)?;
    eprintln!(
        "wpt driver: discovering tests under {} ...",
        cfg.corpus_root.display()
    );
    let files = collect_tests(&cfg)?;
    let total = files.len();
    if total == 0 {
        return Err("no tests matched the selection".to_string());
    }
    eprintln!(
        "wpt driver: {} test files selected | exe={} jobs={} timeout={}ms sample-every={} limit={} area={:?}",
        total,
        cfg.exe.display(),
        cfg.jobs,
        cfg.timeout_ms,
        cfg.sample_every,
        cfg.limit,
        cfg.area
    );

    let corpus_root = cfg.corpus_root.clone();
    let files = Arc::new(files);
    let next = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<FileRecord>>> = Arc::new(Mutex::new(Vec::with_capacity(total)));
    let start = Instant::now();

    let mut handles = Vec::new();
    for _ in 0..cfg.jobs {
        let files = Arc::clone(&files);
        let next = Arc::clone(&next);
        let done = Arc::clone(&done);
        let results = Arc::clone(&results);
        let cfg = cfg.clone();
        let corpus_root = corpus_root.clone();
        let handle = std::thread::spawn(move || loop {
            let idx = next.fetch_add(1, Ordering::SeqCst);
            if idx >= files.len() {
                break;
            }
            let file = &files[idx];
            let (outcome, harness_error, raw) = run_one(&cfg, file);
            let rel = rel_path(&corpus_root, file);
            let area = area_for(&rel);
            let subs: Vec<SubResult> = raw
                .into_iter()
                .map(|r| SubResult {
                    area: area.clone(),
                    file: rel.clone(),
                    name: r.name,
                    status: r.status,
                    message: r.message,
                })
                .collect();
            let rec = FileRecord {
                area,
                rel_path: rel,
                outcome,
                harness_error,
                subs,
            };
            results.lock().unwrap().push(rec);
            let d = done.fetch_add(1, Ordering::SeqCst) + 1;
            if d % 25 == 0 || d == files.len() {
                eprintln!(
                    "  progress: {}/{} ({:.0}%) elapsed {:.1}s",
                    d,
                    files.len(),
                    d as f64 * 100.0 / files.len() as f64,
                    start.elapsed().as_secs_f64()
                );
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

    eprintln!("================ WPT BASELINE ================");
    eprintln!("test files run     : {}", agg.files);
    eprintln!(
        "  loaded harness   : {} | timeout {} | harness-error {}",
        agg.files_ok, agg.files_timeout, agg.files_harness_error
    );
    eprintln!("subtests pass      : {}", agg.sub_pass);
    eprintln!(
        "subtests fail      : {} (incl {} timeout, {} notrun)",
        agg.sub_fail, agg.sub_timeout, agg.sub_notrun
    );
    eprintln!(
        "subtests precond.  : {} (optional-feature, excluded from pass rate)",
        agg.sub_precondition
    );
    eprintln!("PASS RATE          : {:.2}% (pass / (pass+fail))", agg.pass_rate_pct());
    eprintln!("wall time          : {:.1}s", elapsed.as_secs_f64());
    eprintln!("------------ per-area breakdown -------------");
    let mut areas: Vec<(&String, &(usize, usize, usize))> = agg.by_area.iter().collect();
    areas.sort_by(|a, b| a.0.cmp(b.0));
    for (name, (p, f, he)) in &areas {
        let denom = p + f;
        let rate = if denom == 0 {
            0.0
        } else {
            *p as f64 * 100.0 / denom as f64
        };
        eprintln!(
            "  {:<10} pass {:>5} fail {:>5} ({:.1}%)  harness-err-files {}",
            name, p, f, rate, he
        );
    }
    eprintln!("------------ top failure clusters ----------");
    let ranked = ranked_failclusters(&agg);
    for (i, (cluster, count, _)) in ranked.iter().take(20).enumerate() {
        eprintln!("  {:>2}. [{:>5}] {}", i + 1, count, cluster);
    }

    let report = build_json(&cfg, &agg, &ranked, elapsed, total);
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
}

pub fn ranked_failclusters(agg: &Aggregate) -> Vec<(String, usize, Vec<String>)> {
    let mut v: Vec<(String, usize, Vec<String>)> = agg
        .fail_clusters
        .iter()
        .map(|(k, (c, ex))| (k.clone(), *c, ex.clone()))
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

/// Hand-rolled JSON (no serde — strict no-deps policy).
fn build_json(
    cfg: &DriverCfg,
    agg: &Aggregate,
    ranked: &[(String, usize, Vec<String>)],
    elapsed: Duration,
    total_files: usize,
) -> String {
    let mut s = String::with_capacity(8192);
    s.push_str("{\n");
    s.push_str("  \"schema\": \"wpt-report/v1\",\n");
    s.push_str(&format!(
        "  \"corpus_root\": {},\n",
        jstr(&cfg.corpus_root.to_string_lossy())
    ));
    s.push_str("  \"config\": {\n");
    s.push_str(&format!("    \"jobs\": {},\n", cfg.jobs));
    s.push_str(&format!("    \"timeout_ms\": {},\n", cfg.timeout_ms));
    s.push_str(&format!("    \"sample_every\": {},\n", cfg.sample_every));
    s.push_str(&format!("    \"limit\": {},\n", cfg.limit));
    s.push_str(&format!("    \"area\": {}\n", opt_jstr(cfg.area.as_deref())));
    s.push_str("  },\n");
    s.push_str("  \"summary\": {\n");
    s.push_str(&format!("    \"test_files\": {},\n", total_files));
    s.push_str(&format!("    \"files_loaded_harness\": {},\n", agg.files_ok));
    s.push_str(&format!("    \"files_timeout\": {},\n", agg.files_timeout));
    s.push_str(&format!(
        "    \"files_harness_error\": {},\n",
        agg.files_harness_error
    ));
    s.push_str(&format!("    \"subtests_pass\": {},\n", agg.sub_pass));
    s.push_str(&format!("    \"subtests_fail\": {},\n", agg.sub_fail));
    s.push_str(&format!("    \"subtests_timeout\": {},\n", agg.sub_timeout));
    s.push_str(&format!("    \"subtests_notrun\": {},\n", agg.sub_notrun));
    s.push_str(&format!(
        "    \"subtests_precondition_failed\": {},\n",
        agg.sub_precondition
    ));
    s.push_str(&format!(
        "    \"pass_rate_pct\": {:.2},\n",
        agg.pass_rate_pct()
    ));
    s.push_str(&format!(
        "    \"wall_seconds\": {:.1}\n",
        elapsed.as_secs_f64()
    ));
    s.push_str("  },\n");

    s.push_str("  \"by_area\": {\n");
    let mut areas: Vec<(&String, &(usize, usize, usize))> = agg.by_area.iter().collect();
    areas.sort_by(|a, b| a.0.cmp(b.0));
    for (i, (name, (p, f, he))) in areas.iter().enumerate() {
        let denom = *p + *f;
        let rate = if denom == 0 {
            0.0
        } else {
            *p as f64 * 100.0 / denom as f64
        };
        let comma = if i + 1 == areas.len() { "" } else { "," };
        s.push_str(&format!(
            "    {}: {{ \"subtests_pass\": {}, \"subtests_fail\": {}, \"pass_rate_pct\": {:.2}, \"harness_error_files\": {} }}{}\n",
            jstr(name), p, f, rate, he, comma
        ));
    }
    s.push_str("  },\n");

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
        s.push_str("]\n    }");
        s.push_str(comma);
        s.push('\n');
    }
    s.push_str("  ],\n");

    s.push_str("  \"harness_error_clusters\": [\n");
    let mut hes: Vec<(&String, &usize)> = agg.harness_error_clusters.iter().collect();
    hes.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (i, (cluster, count)) in hes.iter().enumerate() {
        let comma = if i + 1 == hes.len() { "" } else { "," };
        s.push_str(&format!(
            "    {{ \"cluster\": {}, \"count\": {} }}{}\n",
            jstr(cluster),
            count,
            comma
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
    fn parse_result_and_summary() {
        let out = "WPT-RESULT: PASS | first test | \n\
                   WPT-RESULT: FAIL | second | assert_equals: x expected 2 but got 1\n\
                   WPT-SUMMARY: harness_status=0 pass=1 fail=1\n";
        let (outcome, _e, subs) = parse_child_output(out);
        assert_eq!(outcome, FileOutcome::Ok);
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].status, SubStatus::Pass);
        assert_eq!(subs[1].status, SubStatus::Fail);
        assert_eq!(subs[0].name, "first test");
    }

    #[test]
    fn harness_error_detected() {
        let out = "WPT-HARNESS-ERR: test did not load the harness (no __wpt)\n";
        let (outcome, e, _subs) = parse_child_output(out);
        assert_eq!(outcome, FileOutcome::HarnessError);
        assert!(e.contains("did not load"));
    }

    #[test]
    fn no_output_is_harness_error() {
        let (outcome, _e, _s) = parse_child_output("garbage with nothing useful");
        assert_eq!(outcome, FileOutcome::HarnessError);
    }

    #[test]
    fn cluster_collapses_reference_errors() {
        let a = cluster_for(SubStatus::Fail, "ReferenceError: root is not defined");
        let b = cluster_for(SubStatus::Fail, "ReferenceError: foo is not defined");
        assert_eq!(a, b);
        assert_eq!(a, "ReferenceError: <x> is not defined");
    }

    #[test]
    fn cluster_assert_equals() {
        let a = cluster_for(SubStatus::Fail, "assert_equals: width expected \"100px\" but got \"0px\"");
        let b = cluster_for(SubStatus::Fail, "assert_equals: color expected \"red\" but got \"blue\"");
        assert_eq!(a, b);
        assert_eq!(a, "assert_equals: value mismatch");
    }

    #[test]
    fn cluster_missing_api_method() {
        let a = cluster_for(
            SubStatus::Fail,
            "TypeError: bc: TypeError(\"callee is not callable: undefined (property `hasAttributeNS`)\")",
        );
        let b = cluster_for(
            SubStatus::Fail,
            "TypeError: bc: TypeError(\"callee is not callable: undefined (property `getAttributeNS`)\")",
        );
        assert_eq!(a, b);
        assert_eq!(a, "TypeError: method not callable (missing API)");
    }

    #[test]
    fn cluster_precondition_is_separate() {
        let c = cluster_for(SubStatus::PreconditionFailed, "feature x");
        assert_eq!(c, "Precondition failed (optional feature unsupported)");
    }

    #[test]
    fn area_extraction() {
        assert_eq!(area_for("authored/dom/node-basics.html"), "dom");
        assert_eq!(area_for("upstream/css/color-computed.html"), "css");
        assert_eq!(area_for("dom/nodes/x.html"), "dom");
    }

    #[test]
    fn pass_rate_excludes_precondition() {
        let mut agg = Aggregate::default();
        let rec = FileRecord {
            area: "dom".into(),
            rel_path: "dom/a.html".into(),
            outcome: FileOutcome::Ok,
            harness_error: String::new(),
            subs: vec![
                SubResult { area: "dom".into(), file: "dom/a.html".into(), name: "p".into(), status: SubStatus::Pass, message: String::new() },
                SubResult { area: "dom".into(), file: "dom/a.html".into(), name: "f".into(), status: SubStatus::Fail, message: "assert_true: x".into() },
                SubResult { area: "dom".into(), file: "dom/a.html".into(), name: "pre".into(), status: SubStatus::PreconditionFailed, message: "feat".into() },
            ],
        };
        agg.record(&rec);
        // 1 pass, 1 fail, 1 precondition => 50%, not 33%.
        assert!((agg.pass_rate_pct() - 50.0).abs() < 1e-9);
        assert_eq!(agg.sub_precondition, 1);
    }

    #[test]
    fn json_escapes() {
        assert_eq!(jstr("a\"b\nc"), "\"a\\\"b\\nc\"");
    }

    #[test]
    fn harness_error_cluster_buckets() {
        assert_eq!(
            harness_error_cluster("test did not load the harness (no __wpt)"),
            "Harness did not load (doc/script error before shim)"
        );
        assert_eq!(harness_error_cluster("spawn-error: foo"), "Driver: spawn error");
    }
}
