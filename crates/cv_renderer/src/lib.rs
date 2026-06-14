//! `cv_renderer` — renderer-process lifecycle + paint pipeline.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererState {
    /// Process spawned, waiting for handshake.
    Booting,
    /// Handshake done; idle.
    Ready,
    /// Currently servicing a paint request.
    Painting,
    /// Pipe closed or shutdown requested.
    Closing,
}

#[derive(Debug, Clone)]
pub struct PaintRequest {
    pub url: String,
    pub html: Vec<u8>,
    pub viewport_w: u32,
    pub viewport_h: u32,
    pub seq: u64,
}

#[derive(Debug, Clone)]
pub struct PaintResponse {
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
}

/// Renderer-side lifecycle + paint queue. The browser process sends
/// `PaintRequest`s; the renderer pumps them via `tick`.
#[derive(Debug)]
pub struct Renderer {
    state: RendererState,
    pending: VecDeque<PaintRequest>,
    completed: Vec<PaintResponse>,
    paint_count: u64,
    next_seq: u64,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            state: RendererState::Booting,
            pending: VecDeque::new(),
            completed: Vec::new(),
            paint_count: 0,
            next_seq: 1,
        }
    }

    pub fn state(&self) -> RendererState {
        self.state
    }
    pub fn complete_handshake(&mut self) {
        if self.state == RendererState::Booting {
            self.state = RendererState::Ready;
        }
    }
    pub fn enqueue(&mut self, mut req: PaintRequest) -> u64 {
        if req.seq == 0 {
            req.seq = self.next_seq;
            self.next_seq += 1;
        }
        let seq = req.seq;
        self.pending.push_back(req);
        seq
    }

    /// One iteration of the renderer pump.
    /// Returns Some response if a paint completed this tick.
    pub fn tick<F>(&mut self, mut paint: F) -> Option<PaintResponse>
    where
        F: FnMut(&PaintRequest) -> PaintResponse,
    {
        if self.state != RendererState::Ready {
            return None;
        }
        let req = self.pending.pop_front()?;
        self.state = RendererState::Painting;
        let resp = paint(&req);
        self.completed.push(resp.clone());
        self.paint_count += 1;
        self.state = RendererState::Ready;
        Some(resp)
    }

    pub fn shutdown(&mut self) {
        self.state = RendererState::Closing;
        self.pending.clear();
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }
    pub fn paint_count(&self) -> u64 {
        self.paint_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_paint(req: &PaintRequest) -> PaintResponse {
        let n = (req.viewport_w * req.viewport_h * 4) as usize;
        PaintResponse {
            seq: req.seq,
            width: req.viewport_w,
            height: req.viewport_h,
            bgra: vec![255; n],
        }
    }

    #[test]
    fn state_starts_booting() {
        let r = Renderer::new();
        assert_eq!(r.state(), RendererState::Booting);
    }

    #[test]
    fn handshake_transitions_to_ready() {
        let mut r = Renderer::new();
        r.complete_handshake();
        assert_eq!(r.state(), RendererState::Ready);
    }

    #[test]
    fn tick_drains_paint_queue() {
        let mut r = Renderer::new();
        r.complete_handshake();
        r.enqueue(PaintRequest {
            url: "https://example.com".into(),
            html: b"<html></html>".to_vec(),
            viewport_w: 8,
            viewport_h: 8,
            seq: 0,
        });
        assert_eq!(r.pending_count(), 1);
        let resp = r.tick(solid_paint).unwrap();
        assert_eq!(resp.width, 8);
        assert_eq!(r.pending_count(), 0);
        assert_eq!(r.paint_count(), 1);
    }

    #[test]
    fn tick_returns_none_when_not_ready() {
        let mut r = Renderer::new();
        // Booting — not ready, even with queued work.
        r.enqueue(PaintRequest {
            url: "x".into(),
            html: vec![],
            viewport_w: 1,
            viewport_h: 1,
            seq: 0,
        });
        assert!(r.tick(solid_paint).is_none());
    }

    #[test]
    fn shutdown_clears_queue() {
        let mut r = Renderer::new();
        r.complete_handshake();
        for _ in 0..3 {
            r.enqueue(PaintRequest {
                url: "u".into(),
                html: vec![],
                viewport_w: 1,
                viewport_h: 1,
                seq: 0,
            });
        }
        r.shutdown();
        assert_eq!(r.state(), RendererState::Closing);
        assert_eq!(r.pending_count(), 0);
    }

    #[test]
    fn enqueue_assigns_monotonic_seq() {
        let mut r = Renderer::new();
        let s1 = r.enqueue(PaintRequest {
            url: "u".into(),
            html: vec![],
            viewport_w: 1,
            viewport_h: 1,
            seq: 0,
        });
        let s2 = r.enqueue(PaintRequest {
            url: "u".into(),
            html: vec![],
            viewport_w: 1,
            viewport_h: 1,
            seq: 0,
        });
        assert!(s2 > s1);
    }
}
