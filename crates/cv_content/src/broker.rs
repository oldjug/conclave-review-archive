//! Browser broker — process manager that spawns the renderer, GPU,
//! network, and storage processes at startup and tracks their
//! lifecycle.
//!
//! The actual `CreateProcessAsUserW` call lives in `cv_ipc::sandbox`
//! / `cv_sandbox::appcontainer`; this layer is the orchestration
//! authority: which child types exist, what command lines they get,
//! and the bookkeeping the browser process uses to mark them
//! alive/dead.

use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChildKind {
    Renderer,
    Gpu,
    Network,
    Storage,
    Utility,
}

impl ChildKind {
    pub fn cmdline_arg(self) -> &'static str {
        match self {
            Self::Renderer => "--type=renderer",
            Self::Gpu => "--type=gpu",
            Self::Network => "--type=network",
            Self::Storage => "--type=storage",
            Self::Utility => "--type=utility",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChildDeathReason {
    CleanExit,
    Crash,
    BadMessage,
}

#[derive(Debug, Clone)]
pub struct ChildRecord {
    pub kind: ChildKind,
    pub pid: u32,
    pub alive: bool,
    pub origin: Option<String>,
    pub death_reason: Option<ChildDeathReason>,
}

#[derive(Default)]
pub struct Broker {
    pub children: Mutex<HashMap<u32, ChildRecord>>,
}

impl Broker {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&self, rec: ChildRecord) {
        if let Ok(mut m) = self.children.lock() {
            m.insert(rec.pid, rec);
        }
    }
    pub fn mark_dead(&self, pid: u32) {
        self.mark_dead_with_reason(pid, ChildDeathReason::Crash);
    }
    pub fn mark_dead_with_reason(&self, pid: u32, reason: ChildDeathReason) {
        if let Ok(mut m) = self.children.lock() {
            if let Some(r) = m.get_mut(&pid) {
                r.alive = false;
                r.death_reason = Some(reason);
            }
        }
    }
    pub fn mark_bad_message(&self, pid: u32) {
        self.mark_dead_with_reason(pid, ChildDeathReason::BadMessage);
    }
    pub fn count_alive(&self, kind: ChildKind) -> usize {
        self.children
            .lock()
            .map(|m| m.values().filter(|r| r.alive && r.kind == kind).count())
            .unwrap_or(0)
    }

    pub fn death_reason(&self, pid: u32) -> Option<ChildDeathReason> {
        self.children
            .lock()
            .ok()
            .and_then(|m| m.get(&pid).and_then(|r| r.death_reason.clone()))
    }

    /// Bring up the standard set of helper processes the M4 design
    /// expects. The actual spawn syscall lives downstream — this
    /// returns the cmdlines the broker should hand to
    /// `cv_sandbox::AppContainerSandbox::create` + `CreateProcessAsUserW`.
    pub fn startup_plan(exe_path: &str) -> Vec<(ChildKind, String)> {
        vec![
            (ChildKind::Network, format!("{exe_path} --type=network")),
            (ChildKind::Storage, format!("{exe_path} --type=storage")),
            (ChildKind::Gpu, format!("{exe_path} --type=gpu")),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_and_alive_count() {
        let b = Broker::new();
        b.register(ChildRecord {
            kind: ChildKind::Renderer,
            pid: 100,
            alive: true,
            origin: Some("https://a.com".into()),
            death_reason: None,
        });
        b.register(ChildRecord {
            kind: ChildKind::Renderer,
            pid: 101,
            alive: true,
            origin: Some("https://b.com".into()),
            death_reason: None,
        });
        assert_eq!(b.count_alive(ChildKind::Renderer), 2);
        b.mark_dead(100);
        assert_eq!(b.count_alive(ChildKind::Renderer), 1);
        assert_eq!(b.death_reason(100), Some(ChildDeathReason::Crash));
    }

    #[test]
    fn bad_message_is_recorded_explicitly() {
        let b = Broker::new();
        b.register(ChildRecord {
            kind: ChildKind::Renderer,
            pid: 200,
            alive: true,
            origin: Some("https://example.com".into()),
            death_reason: None,
        });
        b.mark_bad_message(200);
        assert_eq!(b.count_alive(ChildKind::Renderer), 0);
        assert_eq!(b.death_reason(200), Some(ChildDeathReason::BadMessage));
    }

    #[test]
    fn startup_plan_includes_helper_processes() {
        let plan = Broker::startup_plan("conclave.exe");
        let kinds: Vec<_> = plan.iter().map(|(k, _)| *k).collect();
        assert!(kinds.contains(&ChildKind::Network));
        assert!(kinds.contains(&ChildKind::Storage));
        assert!(kinds.contains(&ChildKind::Gpu));
    }
}
