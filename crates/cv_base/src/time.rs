//! Monotonic clock and wall-clock time. Backed by QPC / FILETIME.

use crate::sys;
use core::sync::atomic::{AtomicI64, Ordering};

static QPC_FREQ: AtomicI64 = AtomicI64::new(0);

fn qpc_freq() -> i64 {
    let cached = QPC_FREQ.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let mut f: i64 = 0;
    let ok = unsafe { sys::QueryPerformanceFrequency(&raw mut f) };
    assert!(ok != 0 && f > 0, "QueryPerformanceFrequency failed");
    QPC_FREQ.store(f, Ordering::Relaxed);
    f
}

fn qpc_now() -> i64 {
    let mut c: i64 = 0;
    let ok = unsafe { sys::QueryPerformanceCounter(&raw mut c) };
    assert!(ok != 0, "QueryPerformanceCounter failed");
    c
}

/// Monotonic timestamp. Anchored at process start; never goes backward.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant(i64);

impl Instant {
    pub fn now() -> Self {
        Self(qpc_now())
    }

    pub fn duration_since(self, earlier: Self) -> Duration {
        let ticks = self.0.saturating_sub(earlier.0);
        Duration::from_ticks(ticks)
    }

    pub fn elapsed(self) -> Duration {
        Self::now().duration_since(self)
    }
}

/// Difference between two `Instant`s. Stored as a count of QPC ticks so
/// conversion is exact for the platform; converters round on the way out.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(i64);

impl Duration {
    pub const ZERO: Self = Self(0);

    pub const fn from_ticks(ticks: i64) -> Self {
        Self(ticks)
    }

    pub fn from_millis(ms: i64) -> Self {
        let f = qpc_freq();
        Self(ms.saturating_mul(f) / 1000)
    }

    pub fn from_micros(us: i64) -> Self {
        let f = qpc_freq();
        Self(us.saturating_mul(f) / 1_000_000)
    }

    pub fn as_millis(self) -> i64 {
        self.0 * 1000 / qpc_freq()
    }

    pub fn as_micros(self) -> i64 {
        self.0 * 1_000_000 / qpc_freq()
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / qpc_freq() as f64
    }

    pub fn raw_ticks(self) -> i64 {
        self.0
    }
}

impl core::ops::Add<Duration> for Instant {
    type Output = Instant;
    fn add(self, rhs: Duration) -> Instant {
        Instant(self.0.saturating_add(rhs.0))
    }
}

impl core::ops::Sub<Duration> for Instant {
    type Output = Instant;
    fn sub(self, rhs: Duration) -> Instant {
        Instant(self.0.saturating_sub(rhs.0))
    }
}

impl core::ops::Sub<Instant> for Instant {
    type Output = Duration;
    fn sub(self, rhs: Instant) -> Duration {
        Duration(self.0.saturating_sub(rhs.0))
    }
}

/// 100-ns ticks since Windows epoch (1601-01-01 UTC).
pub fn wall_clock_ticks_100ns() -> u64 {
    let mut ft = sys::FILETIME::default();
    unsafe { sys::GetSystemTimePreciseAsFileTime(&raw mut ft) };
    (u64::from(ft.high) << 32) | u64::from(ft.low)
}

/// Unix epoch milliseconds. Used by the wall-clock side of the timer wheel.
pub fn unix_epoch_millis() -> i64 {
    const WIN_EPOCH_TO_UNIX_100NS: u64 = 116_444_736_000_000_000;
    let t = wall_clock_ticks_100ns();
    (t.saturating_sub(WIN_EPOCH_TO_UNIX_100NS) / 10_000) as i64
}
