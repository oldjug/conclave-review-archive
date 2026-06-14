//! `cv_base` — foundation primitives for the Conclave browser.
//!
//! Equivalent of Chromium's `//base`. Every other crate in the workspace
//! depends on this. Per workspace policy, no external crates: only `std`
//! and raw Win32 syscalls declared inline.

pub mod cli;
pub mod log;
pub mod string16;
pub mod sys;
pub mod task;
pub mod time;

pub use log::{LogLevel, log_at};
pub use string16::{Str16, String16};
pub use time::{Duration, Instant};
