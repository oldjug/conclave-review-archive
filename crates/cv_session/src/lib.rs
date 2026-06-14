//! `cv_session` — Tab-state persistence for crash recovery /
//! "Reopen last session" / "Restore on next start".
//!
//! Mirrors Chrome's `Last Session` / `Current Session` SNSS files,
//! simplified to a single newline-delimited record format we read
//! back at startup. No JSON parser dependency — keys are flat strings.
//!
//! Format (`session.tab`, one record per line):
//!   `TAB <tab-id> <active-bool> <scroll-y> <url>\t<title>`
//!   `HIST <tab-id> <pos> <url>\t<title>`
//!   `ACTIVE <tab-id>`
//!
//! Wire-up: conclave writes a session record on every navigation +
//! tab create/close/switch, and on a flush every 5s. At startup it
//! reads the file, hands `Vec<TabRecord>` to the session manager,
//! and the manager spawns tabs in-place.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct TabRecord {
    pub id: u32,
    pub url: String,
    pub title: String,
    pub scroll_y: i32,
    pub active: bool,
    pub history: Vec<HistoryEntry>,
}

#[derive(Debug, Clone, Default)]
pub struct HistoryEntry {
    pub pos: u32,
    pub url: String,
    pub title: String,
}

#[derive(Debug, Clone, Default)]
pub struct Session {
    pub tabs: Vec<TabRecord>,
    pub active_id: Option<u32>,
}

impl Session {
    /// Read a session file. Returns an empty session if the file is
    /// missing or empty — caller treats that as "fresh start".
    pub fn load(path: &Path) -> Self {
        let text = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => return Self::default(),
        };
        let mut by_id: HashMap<u32, TabRecord> = HashMap::new();
        let mut active_id: Option<u32> = None;
        for line in text.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(5, ' ');
            let kind = parts.next().unwrap_or("");
            match kind {
                "TAB" => {
                    let id = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let active = parts.next() == Some("1");
                    let scroll = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let rest = parts.next().unwrap_or("");
                    let (url, title) = split_url_title(rest);
                    let rec = by_id.entry(id).or_insert(TabRecord {
                        id,
                        ..Default::default()
                    });
                    rec.url = url;
                    rec.title = title;
                    rec.scroll_y = scroll;
                    rec.active = active;
                }
                "HIST" => {
                    let id = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let pos = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let rest = parts.collect::<Vec<_>>().join(" ");
                    let (url, title) = split_url_title(&rest);
                    let rec = by_id.entry(id).or_insert(TabRecord {
                        id,
                        ..Default::default()
                    });
                    rec.history.push(HistoryEntry { pos, url, title });
                }
                "ACTIVE" => {
                    active_id = parts.next().and_then(|s| s.parse().ok());
                }
                _ => {}
            }
        }
        // Stable order: by id.
        let mut tabs: Vec<TabRecord> = by_id.into_values().collect();
        tabs.sort_by_key(|t| t.id);
        for t in &mut tabs {
            t.history.sort_by_key(|h| h.pos);
        }
        Self { tabs, active_id }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let tmp = with_suffix(path, ".tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            for t in &self.tabs {
                writeln!(
                    f,
                    "TAB {} {} {} {}\t{}",
                    t.id,
                    if t.active { 1 } else { 0 },
                    t.scroll_y,
                    sanitize(&t.url),
                    sanitize(&t.title),
                )?;
                for h in &t.history {
                    writeln!(
                        f,
                        "HIST {} {} {}\t{}",
                        t.id,
                        h.pos,
                        sanitize(&h.url),
                        sanitize(&h.title),
                    )?;
                }
            }
            if let Some(id) = self.active_id {
                writeln!(f, "ACTIVE {id}")?;
            }
        }
        // Atomic-ish replace.
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Convenience for `conclave` callers: rotate `current_session` →
    /// `last_session` at startup, then re-open `current_session` for
    /// the new run.
    pub fn rotate_at_startup(dir: &Path) -> std::io::Result<Self> {
        let cur = dir.join("current_session");
        let last = dir.join("last_session");
        if cur.exists() {
            let _ = fs::remove_file(&last);
            let _ = fs::rename(&cur, &last);
        }
        Ok(Self::load(&last))
    }
}

fn split_url_title(s: &str) -> (String, String) {
    match s.split_once('\t') {
        Some((u, t)) => (u.to_string(), t.to_string()),
        None => (s.to_string(), String::new()),
    }
}

fn sanitize(s: &str) -> String {
    s.replace(['\n', '\t', '\r'], " ")
}

fn with_suffix(p: &Path, suf: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suf);
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmpdir() -> PathBuf {
        let mut p = env::temp_dir();
        let suffix = format!(
            "tb_session_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );
        p.push(suffix);
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[test]
    fn roundtrip_one_tab_one_history() {
        let dir = tmpdir();
        let path = dir.join("session");
        let s = Session {
            tabs: vec![TabRecord {
                id: 7,
                url: "https://example.com/article".into(),
                title: "Example Article".into(),
                scroll_y: 220,
                active: true,
                history: vec![HistoryEntry {
                    pos: 0,
                    url: "https://example.com/".into(),
                    title: "Example".into(),
                }],
            }],
            active_id: Some(7),
        };
        s.save(&path).unwrap();
        let r = Session::load(&path);
        assert_eq!(r.tabs.len(), 1);
        assert_eq!(r.tabs[0].id, 7);
        assert_eq!(r.tabs[0].title, "Example Article");
        assert_eq!(r.tabs[0].scroll_y, 220);
        assert!(r.tabs[0].active);
        assert_eq!(r.tabs[0].history.len(), 1);
        assert_eq!(r.tabs[0].history[0].url, "https://example.com/");
        assert_eq!(r.active_id, Some(7));
    }

    #[test]
    fn empty_load_is_default() {
        let s = Session::load(Path::new("nonexistent-session-file"));
        assert!(s.tabs.is_empty());
        assert!(s.active_id.is_none());
    }

    #[test]
    fn rotate_moves_current_to_last() {
        let dir = tmpdir();
        let s = Session {
            tabs: vec![TabRecord {
                id: 1,
                url: "u".into(),
                title: "t".into(),
                ..Default::default()
            }],
            active_id: Some(1),
        };
        s.save(&dir.join("current_session")).unwrap();
        let restored = Session::rotate_at_startup(&dir).unwrap();
        assert_eq!(restored.tabs.len(), 1);
        assert!(dir.join("last_session").exists());
        assert!(!dir.join("current_session").exists());
    }
}
