//! Web Platform Tests harness — aggregating runner + result model.
//!
//! The runner discovers `.html`/`.htm` WPT tests in a local corpus and runs each
//! one through the engine by spawning `conclave --type wpt-one <file>` as a
//! SEPARATE process per test (so a panic / infinite loop / OOM in one test
//! cannot kill the aggregate run — `conclave` is built `panic=abort`). The
//! child prints one `WPT-RESULT:` line per subtest and a `WPT-SUMMARY:` line,
//! which this crate parses, aggregates, clusters by failure shape, and writes to
//! a JSON report.
//!
//! MACHINE SAFETY (hard requirement): the run is BOUNDED + THROTTLED — a small
//! worker pool (`--jobs`, default 4), a per-test wall-clock timeout
//! (`--timeout-ms`, default 20000) that kills a hung child, and `--sample`/
//! `--limit` selectors for a fast representative baseline before any full run.
//!
//! This is PRODUCTION measurement: subtests the engine genuinely cannot run are
//! counted honestly; the pass rate is pass / (pass + fail) over subtests and is
//! never inflated.

pub mod driver;

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestStatus {
    Pass,
    Fail,
    Timeout,
    Crash,
    Skip,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestResult {
    pub name: String,
    pub status: TestStatus,
    pub duration_ms: u32,
    pub message: Option<String>,
}

#[derive(Debug, Default)]
pub struct WptReport {
    by_module: HashMap<String, Vec<TestResult>>,
}

impl WptReport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn record(&mut self, module: impl Into<String>, result: TestResult) {
        self.by_module
            .entry(module.into())
            .or_default()
            .push(result);
    }
    pub fn pass_rate(&self, module: &str) -> f32 {
        let v = match self.by_module.get(module) {
            Some(v) if !v.is_empty() => v,
            _ => return 0.0,
        };
        let passed = v.iter().filter(|r| r.status == TestStatus::Pass).count();
        (passed as f32) / (v.len() as f32)
    }
    pub fn module_count(&self) -> usize {
        self.by_module.len()
    }
    pub fn modules(&self) -> Vec<&String> {
        self.by_module.keys().collect()
    }
    /// Per-milestone gate: returns true if every module's pass rate
    /// meets `threshold`.
    pub fn passes_gate(&self, threshold: f32) -> bool {
        self.modules()
            .into_iter()
            .all(|m| self.pass_rate(m) >= threshold)
    }
}

/// Sandbox-escape fixture: a renderer-side test that tries every
/// privileged syscall and asserts each one fails. V1 ships the test
/// shape; the actual syscall attempts plug in when the fixture
/// binary runs inside the renderer process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxProbe {
    pub name: String,
    /// True if the privileged call was correctly denied.
    pub denied: bool,
    /// Win32 error code if any.
    pub last_error: u32,
}

#[derive(Debug, Default)]
pub struct SandboxEscapeReport {
    pub probes: Vec<SandboxProbe>,
}

impl SandboxEscapeReport {
    pub fn record(&mut self, probe: SandboxProbe) {
        self.probes.push(probe);
    }
    pub fn all_denied(&self) -> bool {
        !self.probes.is_empty() && self.probes.iter().all(|p| p.denied)
    }
    pub fn first_failure(&self) -> Option<&SandboxProbe> {
        self.probes.iter().find(|p| !p.denied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(name: &str) -> TestResult {
        TestResult {
            name: name.into(),
            status: TestStatus::Pass,
            duration_ms: 10,
            message: None,
        }
    }
    fn fail(name: &str) -> TestResult {
        TestResult {
            name: name.into(),
            status: TestStatus::Fail,
            duration_ms: 10,
            message: Some("boom".into()),
        }
    }

    #[test]
    fn pass_rate_computed_per_module() {
        let mut r = WptReport::new();
        r.record("dom", pass("a"));
        r.record("dom", fail("b"));
        r.record("css", pass("c"));
        assert!((r.pass_rate("dom") - 0.5).abs() < 1e-6);
        assert!((r.pass_rate("css") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_module_has_zero_pass_rate() {
        let r = WptReport::new();
        assert_eq!(r.pass_rate("missing"), 0.0);
    }

    #[test]
    fn gate_requires_all_modules_to_meet_threshold() {
        let mut r = WptReport::new();
        r.record("dom", pass("a"));
        r.record("dom", pass("b"));
        r.record("css", pass("c"));
        r.record("css", fail("d"));
        assert!(!r.passes_gate(0.95));
        assert!(r.passes_gate(0.5));
    }

    #[test]
    fn sandbox_all_denied_only_when_no_probe_failed() {
        let mut r = SandboxEscapeReport::default();
        r.record(SandboxProbe {
            name: "CreateFileW".into(),
            denied: true,
            last_error: 5,
        });
        r.record(SandboxProbe {
            name: "RegOpenKeyExW".into(),
            denied: true,
            last_error: 5,
        });
        assert!(r.all_denied());
        r.record(SandboxProbe {
            name: "GetEnvironmentVariable".into(),
            denied: false,
            last_error: 0,
        });
        assert!(!r.all_denied());
        assert_eq!(
            r.first_failure().map(|p| p.name.as_str()),
            Some("GetEnvironmentVariable")
        );
    }
}
