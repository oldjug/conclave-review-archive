//! `HTMLMediaElement` — shared state machine for `<video>` and
//! `<audio>` per HTML §4.8.12.
//!
//! Models the spec's `readyState`, `networkState`, `paused`,
//! `currentTime`, and the event-firing order. The DOM binding in
//! `conclave` drives the state machine; the codec pipeline
//! callbacks supply decode progress.
//!
//! Out of scope here: actual decoding, time-update tick scheduling,
//! TextTrack/captions — those land at the wiring sites.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReadyState {
    /// HAVE_NOTHING — no info about resource.
    HaveNothing = 0,
    /// HAVE_METADATA — duration / dimensions known.
    HaveMetadata = 1,
    /// HAVE_CURRENT_DATA — playback up to currentTime decoded.
    HaveCurrentData = 2,
    /// HAVE_FUTURE_DATA — at least one future frame ready.
    HaveFutureData = 3,
    /// HAVE_ENOUGH_DATA — enough buffered to play without underrun.
    HaveEnoughData = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkState {
    Empty = 0,
    Idle = 1,
    Loading = 2,
    NoSource = 3,
}

/// Events the state machine emits — the DOM binding queues these on
/// the element's task queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaEvent {
    LoadStart,
    LoadedMetadata,
    LoadedData,
    CanPlay,
    CanPlayThrough,
    Play,
    Playing,
    Pause,
    TimeUpdate,
    Ended,
    Error(String),
    Stalled,
    Waiting,
    Seeking,
    Seeked,
}

#[derive(Debug)]
pub struct MediaElement {
    pub ready_state: ReadyState,
    pub network_state: NetworkState,
    pub current_time_s: f64,
    pub duration_s: f64,
    pub paused: bool,
    pub muted: bool,
    pub volume: f32,
    pub playback_rate: f32,
    pub loop_: bool,
    pub ended: bool,
    pub error: Option<String>,
    pending: Vec<MediaEvent>,
}

impl Default for MediaElement {
    fn default() -> Self {
        Self::new()
    }
}

impl MediaElement {
    pub fn new() -> Self {
        Self {
            ready_state: ReadyState::HaveNothing,
            network_state: NetworkState::Empty,
            current_time_s: 0.0,
            duration_s: f64::NAN,
            paused: true,
            muted: false,
            volume: 1.0,
            playback_rate: 1.0,
            loop_: false,
            ended: false,
            error: None,
            pending: Vec::new(),
        }
    }

    /// `load()` — kick off resource selection.
    pub fn load(&mut self) {
        self.network_state = NetworkState::Loading;
        self.ready_state = ReadyState::HaveNothing;
        self.current_time_s = 0.0;
        self.ended = false;
        self.error = None;
        self.pending.push(MediaEvent::LoadStart);
    }

    /// Codec pipeline reports that metadata (duration / dimensions)
    /// is decoded.
    pub fn on_metadata(&mut self, duration_s: f64) {
        self.duration_s = duration_s;
        self.ready_state = ReadyState::HaveMetadata;
        self.pending.push(MediaEvent::LoadedMetadata);
    }

    /// Codec pipeline reports the first decoded frame is ready.
    pub fn on_first_frame(&mut self) {
        if self.ready_state < ReadyState::HaveCurrentData {
            self.ready_state = ReadyState::HaveCurrentData;
            self.pending.push(MediaEvent::LoadedData);
        }
    }

    /// Codec pipeline reports enough buffer to keep playing.
    pub fn on_can_play_through(&mut self) {
        if self.ready_state < ReadyState::HaveEnoughData {
            self.ready_state = ReadyState::HaveEnoughData;
            self.pending.push(MediaEvent::CanPlay);
            self.pending.push(MediaEvent::CanPlayThrough);
        }
        self.network_state = NetworkState::Idle;
    }

    pub fn play(&mut self) {
        if self.paused {
            self.paused = false;
            self.ended = false;
            self.pending.push(MediaEvent::Play);
            if self.ready_state >= ReadyState::HaveFutureData {
                self.pending.push(MediaEvent::Playing);
            } else {
                self.pending.push(MediaEvent::Waiting);
            }
        }
    }

    pub fn pause(&mut self) {
        if !self.paused {
            self.paused = true;
            self.pending.push(MediaEvent::Pause);
        }
    }

    /// Advance currentTime by `delta_s` seconds. Caller drives this
    /// from the compositor's per-frame tick. Fires `timeupdate` and
    /// `ended` as appropriate.
    pub fn tick(&mut self, delta_s: f64) {
        if self.paused || self.ended {
            return;
        }
        self.current_time_s += delta_s * (self.playback_rate as f64);
        self.pending.push(MediaEvent::TimeUpdate);
        if !self.duration_s.is_nan() && self.current_time_s >= self.duration_s {
            if self.loop_ {
                self.current_time_s = 0.0;
                self.pending.push(MediaEvent::Seeked);
            } else {
                self.current_time_s = self.duration_s;
                self.ended = true;
                self.paused = true;
                self.pending.push(MediaEvent::Ended);
            }
        }
    }

    pub fn seek(&mut self, time_s: f64) {
        self.pending.push(MediaEvent::Seeking);
        self.current_time_s = time_s.max(0.0);
        if !self.duration_s.is_nan() {
            self.current_time_s = self.current_time_s.min(self.duration_s);
        }
        self.ended = false;
        self.pending.push(MediaEvent::Seeked);
    }

    pub fn raise_error(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        self.error = Some(msg.clone());
        self.network_state = NetworkState::NoSource;
        self.pending.push(MediaEvent::Error(msg));
    }

    /// Drain queued events. The DOM binding posts these to the
    /// element's task queue.
    pub fn drain_events(&mut self) -> Vec<MediaEvent> {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_element_is_paused_havenothing() {
        let e = MediaElement::new();
        assert!(e.paused);
        assert_eq!(e.ready_state, ReadyState::HaveNothing);
        assert_eq!(e.network_state, NetworkState::Empty);
    }

    #[test]
    fn load_then_metadata_then_first_frame_progresses_ready_state() {
        let mut e = MediaElement::new();
        e.load();
        e.on_metadata(10.0);
        e.on_first_frame();
        assert_eq!(e.ready_state, ReadyState::HaveCurrentData);
        let events = e.drain_events();
        assert!(events.contains(&MediaEvent::LoadStart));
        assert!(events.contains(&MediaEvent::LoadedMetadata));
        assert!(events.contains(&MediaEvent::LoadedData));
    }

    #[test]
    fn play_fires_play_event_and_unpauses() {
        let mut e = MediaElement::new();
        e.play();
        assert!(!e.paused);
        let events = e.drain_events();
        assert!(events.contains(&MediaEvent::Play));
    }

    #[test]
    fn tick_advances_current_time_and_fires_timeupdate() {
        let mut e = MediaElement::new();
        e.on_metadata(5.0);
        e.play();
        e.drain_events();
        e.tick(1.5);
        assert!((e.current_time_s - 1.5).abs() < 1e-9);
        let events = e.drain_events();
        assert!(events.contains(&MediaEvent::TimeUpdate));
    }

    #[test]
    fn tick_past_duration_fires_ended_and_pauses() {
        let mut e = MediaElement::new();
        e.on_metadata(2.0);
        e.play();
        e.drain_events();
        e.tick(3.0);
        assert!(e.ended);
        assert!(e.paused);
        let events = e.drain_events();
        assert!(events.contains(&MediaEvent::Ended));
    }

    #[test]
    fn loop_resets_current_time_instead_of_ending() {
        let mut e = MediaElement::new();
        e.on_metadata(2.0);
        e.loop_ = true;
        e.play();
        e.drain_events();
        e.tick(3.0);
        assert!(!e.ended);
        assert_eq!(e.current_time_s, 0.0);
    }

    #[test]
    fn seek_clamps_to_duration() {
        let mut e = MediaElement::new();
        e.on_metadata(5.0);
        e.seek(10.0);
        assert_eq!(e.current_time_s, 5.0);
    }

    #[test]
    fn error_transitions_network_state_to_nosource() {
        let mut e = MediaElement::new();
        e.raise_error("decode failed");
        assert_eq!(e.network_state, NetworkState::NoSource);
        assert_eq!(e.error.as_deref(), Some("decode failed"));
    }
}
