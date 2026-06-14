//! Fullscreen / Pointer Lock / Touch / Gamepad / WebMIDI dispatcher.
//!
//! Each Web API gets its own data model + state machine. The window
//! integration calls into these from its message pump.

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct FullscreenStack {
    elements: Vec<u32>,
}

impl FullscreenStack {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn current(&self) -> Option<u32> {
        self.elements.last().copied()
    }
    pub fn request(&mut self, element_id: u32) {
        if self.elements.last() != Some(&element_id) {
            self.elements.push(element_id);
        }
    }
    pub fn exit(&mut self) -> Option<u32> {
        self.elements.pop()
    }
    pub fn depth(&self) -> usize {
        self.elements.len()
    }
}

#[derive(Debug)]
pub struct PointerLock {
    locked_element: Option<u32>,
}

impl Default for PointerLock {
    fn default() -> Self {
        Self::new()
    }
}

impl PointerLock {
    pub fn new() -> Self {
        Self {
            locked_element: None,
        }
    }
    pub fn is_locked(&self) -> bool {
        self.locked_element.is_some()
    }
    pub fn request(&mut self, element_id: u32) -> Result<(), &'static str> {
        if self.locked_element.is_some() {
            return Err("already locked");
        }
        self.locked_element = Some(element_id);
        Ok(())
    }
    pub fn release(&mut self) {
        self.locked_element = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct TouchPoint {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub phase: TouchPhase,
}

#[derive(Debug, Default)]
pub struct TouchTracker {
    active: HashMap<u32, TouchPoint>,
}

impl TouchTracker {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn ingest(&mut self, p: TouchPoint) {
        match p.phase {
            TouchPhase::Started | TouchPhase::Moved => {
                self.active.insert(p.id, p);
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.active.remove(&p.id);
            }
        }
    }
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
    pub fn get(&self, id: u32) -> Option<&TouchPoint> {
        self.active.get(&id)
    }
}

#[derive(Debug, Clone)]
pub struct GamepadState {
    pub id: String,
    pub buttons: Vec<bool>,
    pub axes: Vec<f32>,
    pub timestamp_ms: u64,
}

#[derive(Debug, Default)]
pub struct GamepadRegistry {
    pads: Vec<GamepadState>,
}

impl GamepadRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, g: GamepadState) -> usize {
        self.pads.push(g);
        self.pads.len() - 1
    }
    pub fn update(&mut self, idx: usize, buttons: Vec<bool>, axes: Vec<f32>, ts: u64) {
        if let Some(p) = self.pads.get_mut(idx) {
            p.buttons = buttons;
            p.axes = axes;
            p.timestamp_ms = ts;
        }
    }
    pub fn list(&self) -> &[GamepadState] {
        &self.pads
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MidiMessage {
    pub status: u8,
    pub data: Vec<u8>,
    pub timestamp_ms: u64,
}

impl MidiMessage {
    pub fn is_note_on(&self) -> bool {
        (self.status & 0xF0) == 0x90 && self.data.get(1).copied().unwrap_or(0) > 0
    }
    pub fn is_note_off(&self) -> bool {
        (self.status & 0xF0) == 0x80
            || ((self.status & 0xF0) == 0x90 && self.data.get(1).copied().unwrap_or(0) == 0)
    }
    pub fn note(&self) -> Option<u8> {
        if matches!(self.status & 0xF0, 0x80 | 0x90) {
            self.data.first().copied()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullscreen_stack_lifo() {
        let mut s = FullscreenStack::new();
        assert!(s.current().is_none());
        s.request(1);
        s.request(2);
        assert_eq!(s.depth(), 2);
        assert_eq!(s.current(), Some(2));
        s.exit();
        assert_eq!(s.current(), Some(1));
    }

    #[test]
    fn pointer_lock_rejects_second_lock() {
        let mut p = PointerLock::new();
        assert!(p.request(1).is_ok());
        assert!(p.request(2).is_err());
        p.release();
        assert!(p.request(2).is_ok());
    }

    #[test]
    fn touch_tracker_keeps_active_set() {
        let mut t = TouchTracker::new();
        t.ingest(TouchPoint {
            id: 1,
            x: 0.0,
            y: 0.0,
            phase: TouchPhase::Started,
        });
        t.ingest(TouchPoint {
            id: 2,
            x: 5.0,
            y: 5.0,
            phase: TouchPhase::Started,
        });
        assert_eq!(t.active_count(), 2);
        t.ingest(TouchPoint {
            id: 1,
            x: 0.0,
            y: 0.0,
            phase: TouchPhase::Ended,
        });
        assert_eq!(t.active_count(), 1);
        assert!(t.get(1).is_none());
    }

    #[test]
    fn gamepad_update_replaces_state() {
        let mut g = GamepadRegistry::new();
        let idx = g.add(GamepadState {
            id: "xbox".into(),
            buttons: vec![false; 17],
            axes: vec![0.0; 4],
            timestamp_ms: 0,
        });
        g.update(idx, vec![true; 17], vec![1.0; 4], 100);
        let pads = g.list();
        assert!(pads[idx].buttons.iter().all(|&b| b));
        assert_eq!(pads[idx].timestamp_ms, 100);
    }

    #[test]
    fn midi_note_on_off_detect() {
        let on = MidiMessage {
            status: 0x90,
            data: vec![60, 127],
            timestamp_ms: 0,
        };
        let off = MidiMessage {
            status: 0x80,
            data: vec![60, 0],
            timestamp_ms: 0,
        };
        assert!(on.is_note_on());
        assert!(off.is_note_off());
        assert_eq!(on.note(), Some(60));
    }

    #[test]
    fn midi_velocity_zero_note_on_is_note_off() {
        let m = MidiMessage {
            status: 0x90,
            data: vec![60, 0],
            timestamp_ms: 0,
        };
        assert!(m.is_note_off());
        assert!(!m.is_note_on());
    }
}
