//! Browser shell UI — settings panel, bookmarks/history/downloads
//! lists, and find-in-page state. Each surface is a pure data model
//! the renderer paints; no Win32 control objects are created here so
//! the layer remains platform-agnostic.

use std::collections::VecDeque;

#[derive(Debug, Clone, Default)]
pub struct SettingsState {
    pub homepage: String,
    pub search_engine: String,
    pub default_proxy: Option<String>,
    pub clear_on_exit: bool,
    pub block_third_party_cookies: bool,
    pub do_not_track: bool,
    pub theme: ThemePref,
    pub font_size_base: u16,
    pub safe_browsing_enabled: bool,
    pub auto_update_enabled: bool,
    pub active_profile: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ThemePref {
    #[default]
    System,
    Light,
    Dark,
}

#[derive(Debug, Clone)]
pub struct BookmarksList {
    pub items: Vec<BookmarkItem>,
}

#[derive(Debug, Clone)]
pub struct BookmarkItem {
    pub url: String,
    pub title: String,
    pub folder: String,
    pub added_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HistoryList {
    pub entries: VecDeque<HistoryEntry>,
}

#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub url: String,
    pub title: String,
    pub visit_time_ms: u64,
    pub visit_count: u32,
}

#[derive(Debug, Clone, Default)]
pub struct DownloadsList {
    pub items: Vec<DownloadItem>,
}

#[derive(Debug, Clone)]
pub struct DownloadItem {
    pub url: String,
    pub local_path: String,
    pub size_bytes: u64,
    pub bytes_done: u64,
    pub state: DownloadState,
    pub started_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    Pending,
    InProgress,
    Done,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Default)]
pub struct FindInPage {
    pub query: String,
    pub case_sensitive: bool,
    pub current_match: usize,
    pub total_matches: usize,
    pub matches: Vec<FindMatch>,
}

#[derive(Debug, Clone, Copy)]
pub struct FindMatch {
    pub start_byte: usize,
    pub end_byte: usize,
}

impl FindInPage {
    pub fn search(&mut self, haystack: &str, needle: &str) {
        self.query = needle.to_string();
        self.matches.clear();
        if needle.is_empty() {
            self.total_matches = 0;
            self.current_match = 0;
            return;
        }
        let h = if self.case_sensitive {
            haystack.to_string()
        } else {
            haystack.to_ascii_lowercase()
        };
        let n = if self.case_sensitive {
            needle.to_string()
        } else {
            needle.to_ascii_lowercase()
        };
        let mut start = 0;
        while let Some(at) = h[start..].find(&n) {
            let s = start + at;
            self.matches.push(FindMatch {
                start_byte: s,
                end_byte: s + n.len(),
            });
            start = s + n.len();
        }
        self.total_matches = self.matches.len();
        self.current_match = if self.matches.is_empty() { 0 } else { 1 };
    }

    pub fn next(&mut self) {
        if self.matches.is_empty() {
            return;
        }
        self.current_match = self.current_match % self.matches.len() + 1;
    }

    pub fn prev(&mut self) {
        if self.matches.is_empty() {
            return;
        }
        if self.current_match <= 1 {
            self.current_match = self.matches.len();
        } else {
            self.current_match -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_counts_and_cycles() {
        let mut f = FindInPage::default();
        f.search("the quick brown fox jumps over the lazy dog", "the");
        assert_eq!(f.total_matches, 2);
        assert_eq!(f.current_match, 1);
        f.next();
        assert_eq!(f.current_match, 2);
        f.next();
        assert_eq!(f.current_match, 1);
        f.prev();
        assert_eq!(f.current_match, 2);
    }

    #[test]
    fn case_insensitive_by_default() {
        let mut f = FindInPage::default();
        f.search("HELLO", "hello");
        assert_eq!(f.total_matches, 1);
    }
}
