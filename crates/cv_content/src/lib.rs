//! `cv_content` — frame tree + navigation history + process mgr.

#![allow(missing_debug_implementations)]

pub mod broker;
pub mod m4_verify;

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    Main,
    Iframe,
    Worker,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub id: FrameId,
    pub parent: Option<FrameId>,
    pub kind: FrameKind,
    pub url: String,
    pub process: ProcessId,
    pub origin: String,
}

#[derive(Debug, Default)]
pub struct FrameTree {
    frames: HashMap<FrameId, Frame>,
    next_id: u32,
    root: Option<FrameId>,
}

impl FrameTree {
    pub fn new() -> Self {
        Self::default()
    }
    fn alloc(&mut self) -> FrameId {
        self.next_id += 1;
        FrameId(self.next_id)
    }
    pub fn create_main(
        &mut self,
        url: impl Into<String>,
        origin: impl Into<String>,
        process: ProcessId,
    ) -> FrameId {
        let id = self.alloc();
        self.frames.insert(
            id,
            Frame {
                id,
                parent: None,
                kind: FrameKind::Main,
                url: url.into(),
                process,
                origin: origin.into(),
            },
        );
        self.root = Some(id);
        id
    }
    pub fn add_iframe(
        &mut self,
        parent: FrameId,
        url: impl Into<String>,
        origin: impl Into<String>,
        process: ProcessId,
    ) -> FrameId {
        let id = self.alloc();
        self.frames.insert(
            id,
            Frame {
                id,
                parent: Some(parent),
                kind: FrameKind::Iframe,
                url: url.into(),
                process,
                origin: origin.into(),
            },
        );
        id
    }
    pub fn get(&self, id: FrameId) -> Option<&Frame> {
        self.frames.get(&id)
    }
    pub fn root(&self) -> Option<FrameId> {
        self.root
    }
    pub fn children(&self, parent: FrameId) -> Vec<FrameId> {
        self.frames
            .values()
            .filter(|f| f.parent == Some(parent))
            .map(|f| f.id)
            .collect()
    }
    pub fn remove(&mut self, id: FrameId) {
        for k in self.children(id) {
            self.remove(k);
        }
        self.frames.remove(&id);
        if self.root == Some(id) {
            self.root = None;
        }
    }
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

#[derive(Debug, Default)]
pub struct History {
    entries: Vec<HistoryEntry>,
    cursor: usize,
}

#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub url: String,
    pub title: String,
    pub scroll_y: i32,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, url: impl Into<String>, title: impl Into<String>) {
        if !self.entries.is_empty() {
            self.entries.truncate(self.cursor + 1);
        }
        self.entries.push(HistoryEntry {
            url: url.into(),
            title: title.into(),
            scroll_y: 0,
        });
        self.cursor = self.entries.len() - 1;
    }
    pub fn can_back(&self) -> bool {
        self.cursor > 0 && !self.entries.is_empty()
    }
    pub fn can_forward(&self) -> bool {
        self.cursor + 1 < self.entries.len()
    }
    pub fn back(&mut self) -> Option<&HistoryEntry> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        self.entries.get(self.cursor)
    }
    pub fn forward(&mut self) -> Option<&HistoryEntry> {
        if self.cursor + 1 >= self.entries.len() {
            return None;
        }
        self.cursor += 1;
        self.entries.get(self.cursor)
    }
    pub fn current(&self) -> Option<&HistoryEntry> {
        self.entries.get(self.cursor)
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Default)]
pub struct ProcessManager {
    next_pid: u32,
    by_origin: HashMap<String, ProcessId>,
}

impl ProcessManager {
    pub fn new() -> Self {
        Self::default()
    }
    fn alloc(&mut self) -> ProcessId {
        self.next_pid += 1;
        ProcessId(self.next_pid)
    }
    pub fn process_for_origin(&mut self, origin: &str) -> ProcessId {
        if let Some(&p) = self.by_origin.get(origin) {
            return p;
        }
        let p = self.alloc();
        self.by_origin.insert(origin.to_string(), p);
        p
    }
    pub fn process_count(&self) -> usize {
        self.by_origin.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_tree_grows() {
        let mut t = FrameTree::new();
        let mut pm = ProcessManager::new();
        let p = pm.process_for_origin("https://example.com");
        let m = t.create_main("u", "https://example.com", p);
        let i = t.add_iframe(
            m,
            "u",
            "https://embed.com",
            pm.process_for_origin("https://embed.com"),
        );
        assert_eq!(t.children(m), vec![i]);
    }

    #[test]
    fn pm_reuses_origin_process() {
        let mut pm = ProcessManager::new();
        let a = pm.process_for_origin("https://x.com");
        let b = pm.process_for_origin("https://x.com");
        let c = pm.process_for_origin("https://y.com");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn history_back_forward() {
        let mut h = History::new();
        h.push("a", "A");
        h.push("b", "B");
        h.push("c", "C");
        assert_eq!(h.back().unwrap().url, "b");
        assert_eq!(h.back().unwrap().url, "a");
        assert!(!h.can_back());
        assert_eq!(h.forward().unwrap().url, "b");
    }

    #[test]
    fn history_push_drops_forward() {
        let mut h = History::new();
        h.push("a", "");
        h.push("b", "");
        h.push("c", "");
        h.back();
        h.back();
        h.push("d", "");
        assert_eq!(h.len(), 2);
        assert!(!h.can_forward());
    }

    #[test]
    fn remove_drops_subtree() {
        let mut t = FrameTree::new();
        let mut pm = ProcessManager::new();
        let p = pm.process_for_origin("o");
        let m = t.create_main("u", "o", p);
        let a = t.add_iframe(m, "u", "o", p);
        t.add_iframe(a, "u", "o", p);
        t.remove(a);
        assert_eq!(t.len(), 1);
    }
}
