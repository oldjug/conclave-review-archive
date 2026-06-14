//! MediaSource Extensions (MSE) — server-driven segment switching.
//!
//! The browser exposes `new MediaSource()` to JS; pages create
//! `SourceBuffer`s, push `appendBuffer(data)` calls of ISO BMFF /
//! WebM bytes, and the media element pulls timed samples out of the
//! buffer for the demuxer.
//!
//! Here we model the buffer geometry: a queue of byte-ranges plus
//! their decoded timestamp range, the ready-state flags, and the
//! callback hooks the JS binding fires.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    Closed,
    Open,
    Ended,
}

#[derive(Debug, Clone)]
pub struct SourceBuffer {
    pub mime_type: String,
    pub appended_bytes: u64,
    pub time_ranges: Vec<(f64, f64)>,
    pub pending: VecDeque<Vec<u8>>,
    pub updating: bool,
}

impl SourceBuffer {
    pub fn new(mime: String) -> Self {
        Self {
            mime_type: mime,
            appended_bytes: 0,
            time_ranges: Vec::new(),
            pending: VecDeque::new(),
            updating: false,
        }
    }

    /// Append a buffer. In the real spec this is async; V1 model
    /// queues it and lets the caller flush via `step()`.
    pub fn append_buffer(&mut self, data: Vec<u8>) {
        self.pending.push_back(data);
        self.updating = true;
    }

    /// Process one pending append. Returns whether progress happened.
    pub fn step(&mut self) -> bool {
        if let Some(d) = self.pending.pop_front() {
            self.appended_bytes += d.len() as u64;
            self.updating = !self.pending.is_empty();
            true
        } else {
            self.updating = false;
            false
        }
    }
}

#[derive(Debug, Clone)]
pub struct MediaSource {
    pub ready_state: ReadyState,
    pub source_buffers: Vec<SourceBuffer>,
    pub duration_s: Option<f64>,
}

impl Default for MediaSource {
    fn default() -> Self {
        Self {
            ready_state: ReadyState::Closed,
            source_buffers: Vec::new(),
            duration_s: None,
        }
    }
}

impl MediaSource {
    pub fn add_source_buffer(&mut self, mime: String) -> usize {
        self.source_buffers.push(SourceBuffer::new(mime));
        self.source_buffers.len() - 1
    }

    pub fn open(&mut self) {
        self.ready_state = ReadyState::Open;
    }

    pub fn end_of_stream(&mut self) {
        self.ready_state = ReadyState::Ended;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_then_step_drains_queue() {
        let mut ms = MediaSource::default();
        ms.open();
        let i = ms.add_source_buffer("video/mp4".into());
        let sb = &mut ms.source_buffers[i];
        sb.append_buffer(vec![0u8; 16]);
        assert!(sb.updating);
        assert!(sb.step());
        assert!(!sb.updating);
        assert_eq!(sb.appended_bytes, 16);
    }
}
