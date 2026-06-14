//! Tab strip — multi-tab browsing surface.
//!
//! Each tab owns its own URL, history stack, and (in the engine) its
//! own JS interp + DOM root. The shell maintains a `TabStrip` of
//! `Tab`s, an active index, and lifecycle hooks.

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Tab {
    pub id: u64,
    pub title: String,
    pub url: String,
    pub favicon: Option<Vec<u8>>,
    pub back: Vec<String>,
    pub forward: Vec<String>,
    pub muted: bool,
    pub pinned: bool,
    pub incognito: bool,
    pub created_at_ms: u64,
}

impl Tab {
    pub fn new(id: u64, url: String, incognito: bool, now_ms: u64) -> Self {
        Self {
            id,
            title: url.clone(),
            url,
            favicon: None,
            back: Vec::new(),
            forward: Vec::new(),
            muted: false,
            pinned: false,
            incognito,
            created_at_ms: now_ms,
        }
    }

    pub fn navigate(&mut self, url: String) {
        if !self.url.is_empty() {
            self.back.push(std::mem::take(&mut self.url));
        }
        self.url = url;
        self.forward.clear();
    }

    pub fn go_back(&mut self) -> bool {
        if let Some(prev) = self.back.pop() {
            self.forward.push(std::mem::take(&mut self.url));
            self.url = prev;
            true
        } else {
            false
        }
    }

    pub fn go_forward(&mut self) -> bool {
        if let Some(next) = self.forward.pop() {
            self.back.push(std::mem::take(&mut self.url));
            self.url = next;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Default)]
pub struct TabStrip {
    pub tabs: Vec<Tab>,
    pub active: usize,
    pub closed_recent: VecDeque<Tab>,
    pub next_id: u64,
}

impl TabStrip {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&mut self, url: String, incognito: bool, now_ms: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let tab = Tab::new(id, url, incognito, now_ms);
        self.tabs.push(tab);
        self.active = self.tabs.len() - 1;
        id
    }

    pub fn close(&mut self, id: u64) {
        if let Some(pos) = self.tabs.iter().position(|t| t.id == id) {
            let removed = self.tabs.remove(pos);
            self.closed_recent.push_front(removed);
            if self.closed_recent.len() > 16 {
                self.closed_recent.pop_back();
            }
            if self.active >= self.tabs.len() && !self.tabs.is_empty() {
                self.active = self.tabs.len() - 1;
            }
        }
    }

    pub fn reopen_last(&mut self) -> Option<u64> {
        let tab = self.closed_recent.pop_front()?;
        let id = tab.id;
        self.tabs.push(tab);
        self.active = self.tabs.len() - 1;
        Some(id)
    }

    pub fn activate(&mut self, id: u64) -> bool {
        if let Some(pos) = self.tabs.iter().position(|t| t.id == id) {
            self.active = pos;
            true
        } else {
            false
        }
    }

    pub fn active_tab(&self) -> Option<&Tab> {
        self.tabs.get(self.active)
    }

    pub fn active_tab_mut(&mut self) -> Option<&mut Tab> {
        self.tabs.get_mut(self.active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_close_reopen() {
        let mut s = TabStrip::new();
        let a = s.open("about:blank".into(), false, 0);
        let b = s.open("https://example.com".into(), false, 0);
        assert_eq!(s.tabs.len(), 2);
        s.close(b);
        assert_eq!(s.tabs.len(), 1);
        assert_eq!(s.active_tab().unwrap().id, a);
        let reopened = s.reopen_last().unwrap();
        assert_eq!(reopened, b);
        assert_eq!(s.tabs.len(), 2);
    }

    #[test]
    fn navigation_history() {
        let mut t = Tab::new(0, "a".into(), false, 0);
        t.navigate("b".into());
        t.navigate("c".into());
        assert_eq!(t.url, "c");
        assert!(t.go_back());
        assert_eq!(t.url, "b");
        assert!(t.go_forward());
        assert_eq!(t.url, "c");
    }
}
