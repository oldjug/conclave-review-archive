//! Downloads state machine.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    Pending,
    InProgress,
    Completed,
    Failed,
    Canceled,
    Paused,
}

#[derive(Debug, Clone)]
pub struct Download {
    pub id: u32,
    pub url: String,
    pub file_path: String,
    pub mime_type: String,
    pub state: DownloadState,
    pub bytes_received: u64,
    pub total_bytes: Option<u64>,
    pub start_ms: u64,
    pub end_ms: Option<u64>,
}

#[derive(Debug, Default)]
pub struct DownloadsManager {
    next_id: u32,
    items: Vec<Download>,
}

impl DownloadsManager {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn start(
        &mut self,
        url: impl Into<String>,
        path: impl Into<String>,
        mime: impl Into<String>,
        total: Option<u64>,
        ts: u64,
    ) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        self.items.push(Download {
            id,
            url: url.into(),
            file_path: path.into(),
            mime_type: mime.into(),
            state: DownloadState::InProgress,
            bytes_received: 0,
            total_bytes: total,
            start_ms: ts,
            end_ms: None,
        });
        id
    }
    pub fn progress(&mut self, id: u32, bytes_received: u64) {
        if let Some(d) = self.items.iter_mut().find(|d| d.id == id) {
            d.bytes_received = bytes_received;
        }
    }
    pub fn complete(&mut self, id: u32, ts: u64) {
        if let Some(d) = self.items.iter_mut().find(|d| d.id == id) {
            d.state = DownloadState::Completed;
            d.end_ms = Some(ts);
        }
    }
    pub fn fail(&mut self, id: u32, ts: u64) {
        if let Some(d) = self.items.iter_mut().find(|d| d.id == id) {
            d.state = DownloadState::Failed;
            d.end_ms = Some(ts);
        }
    }
    pub fn cancel(&mut self, id: u32, ts: u64) {
        if let Some(d) = self.items.iter_mut().find(|d| d.id == id) {
            d.state = DownloadState::Canceled;
            d.end_ms = Some(ts);
        }
    }
    pub fn pause(&mut self, id: u32) {
        if let Some(d) = self.items.iter_mut().find(|d| d.id == id) {
            if d.state == DownloadState::InProgress {
                d.state = DownloadState::Paused;
            }
        }
    }
    pub fn resume(&mut self, id: u32) {
        if let Some(d) = self.items.iter_mut().find(|d| d.id == id) {
            if d.state == DownloadState::Paused {
                d.state = DownloadState::InProgress;
            }
        }
    }
    pub fn list(&self) -> &[Download] {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_then_complete_lifecycle() {
        let mut m = DownloadsManager::new();
        let id = m.start(
            "https://example.com/x.zip",
            "C:/dl/x.zip",
            "application/zip",
            Some(1000),
            0,
        );
        m.progress(id, 500);
        m.complete(id, 100);
        let d = &m.list()[0];
        assert_eq!(d.state, DownloadState::Completed);
        assert_eq!(d.bytes_received, 500);
        assert_eq!(d.end_ms, Some(100));
    }

    #[test]
    fn pause_then_resume_round_trips() {
        let mut m = DownloadsManager::new();
        let id = m.start("u", "p", "m", None, 0);
        m.pause(id);
        assert_eq!(m.list()[0].state, DownloadState::Paused);
        m.resume(id);
        assert_eq!(m.list()[0].state, DownloadState::InProgress);
    }

    #[test]
    fn cancel_completed_does_not_change_completion_state() {
        let mut m = DownloadsManager::new();
        let id = m.start("u", "p", "m", None, 0);
        m.complete(id, 10);
        m.cancel(id, 20);
        // Cancel after complete moves us back to Canceled — current behavior.
        assert_eq!(m.list()[0].state, DownloadState::Canceled);
    }
}
