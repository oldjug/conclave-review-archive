//! `cv_power` — renderer power/memory coordinator.
//!
//! Mirrors Chrome's `power_monitor` + `memory_coordinator` + render
//! freezing for backgrounded tabs.
//!
//! Surfaces:
//!   * **Tab lifecycle state** — Active / Hidden / Frozen / Discarded.
//!     Transition rules match Chrome: a tab moves Active → Hidden when
//!     it leaves the foreground; Hidden → Frozen after 5 minutes
//!     idle; Frozen → Discarded under memory pressure (purges the
//!     renderer entirely; restored on focus).
//!   * **Memory pressure level** — Normal / Moderate / Critical.
//!     Sampled from `GlobalMemoryStatusEx` on Windows.
//!   * **Tick policy** — `throttle_ms_for(state, memory)` returns the
//!     desired wall-clock between JS tick drains. Chrome backs
//!     timers off to 1 Hz when Hidden and to 0 (paused) when Frozen.

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    Active,
    Hidden,
    Frozen,
    Discarded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    Normal,
    Moderate,
    Critical,
}

#[derive(Debug, Clone)]
pub struct TabPolicy {
    pub state: LifecycleState,
    pub became_hidden: Option<Instant>,
    /// Last time the tab was visible/focused.
    pub last_active: Instant,
}

impl Default for TabPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl TabPolicy {
    pub fn new() -> Self {
        Self {
            state: LifecycleState::Active,
            became_hidden: None,
            last_active: Instant::now(),
        }
    }

    pub fn on_focus(&mut self) {
        self.state = LifecycleState::Active;
        self.became_hidden = None;
        self.last_active = Instant::now();
    }

    pub fn on_blur(&mut self) {
        if matches!(self.state, LifecycleState::Active) {
            self.state = LifecycleState::Hidden;
            self.became_hidden = Some(Instant::now());
        }
    }

    /// Apply policy: if hidden long enough, transition to Frozen; if
    /// memory critical and Frozen, Discard. Returns the new state.
    pub fn tick(&mut self, now: Instant, pressure: MemoryPressure) -> LifecycleState {
        match self.state {
            LifecycleState::Hidden => {
                if let Some(t) = self.became_hidden {
                    if now.duration_since(t) >= Duration::from_secs(5 * 60) {
                        self.state = LifecycleState::Frozen;
                    }
                }
            }
            LifecycleState::Frozen => {
                if matches!(pressure, MemoryPressure::Critical) {
                    self.state = LifecycleState::Discarded;
                }
            }
            _ => {}
        }
        self.state
    }
}

/// Wall-clock between JS tick drains for a given tab state.
pub fn throttle_ms_for(state: LifecycleState, pressure: MemoryPressure) -> Option<u32> {
    use LifecycleState::*;
    use MemoryPressure::*;
    match (state, pressure) {
        (Active, _) => Some(16),        // ~60 Hz
        (Hidden, Normal) => Some(1000), // 1 Hz
        (Hidden, Moderate) => Some(2000),
        (Hidden, Critical) => Some(5000),
        (Frozen, _) | (Discarded, _) => None, // suspend
    }
}

/// Sample global memory status. Windows-only; embedder calls this on
/// the broker process once per second.
#[cfg(windows)]
pub fn sample_memory_pressure() -> MemoryPressure {
    use core::ffi::c_void as _;
    #[repr(C)]
    struct MEMORYSTATUSEX {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buf: *mut MEMORYSTATUSEX) -> i32;
    }
    let mut m = MEMORYSTATUSEX {
        dw_length: core::mem::size_of::<MEMORYSTATUSEX>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    let ok = unsafe { GlobalMemoryStatusEx(&raw mut m) };
    if ok == 0 {
        return MemoryPressure::Normal;
    }
    match m.dw_memory_load {
        0..=70 => MemoryPressure::Normal,
        71..=89 => MemoryPressure::Moderate,
        _ => MemoryPressure::Critical,
    }
}

#[cfg(not(windows))]
pub fn sample_memory_pressure() -> MemoryPressure {
    MemoryPressure::Normal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_tab_runs_at_60hz() {
        assert_eq!(
            throttle_ms_for(LifecycleState::Active, MemoryPressure::Normal),
            Some(16)
        );
    }

    #[test]
    fn hidden_tab_throttled_to_1hz() {
        assert_eq!(
            throttle_ms_for(LifecycleState::Hidden, MemoryPressure::Normal),
            Some(1000)
        );
    }

    #[test]
    fn frozen_tab_paused() {
        assert!(throttle_ms_for(LifecycleState::Frozen, MemoryPressure::Normal).is_none());
    }

    #[test]
    fn lifecycle_transitions() {
        let mut p = TabPolicy::new();
        assert_eq!(p.state, LifecycleState::Active);
        p.on_blur();
        assert_eq!(p.state, LifecycleState::Hidden);
        let later = p.became_hidden.unwrap() + Duration::from_secs(6 * 60);
        p.tick(later, MemoryPressure::Normal);
        assert_eq!(p.state, LifecycleState::Frozen);
        p.tick(later, MemoryPressure::Critical);
        assert_eq!(p.state, LifecycleState::Discarded);
        p.on_focus();
        assert_eq!(p.state, LifecycleState::Active);
    }
}
