//! Tiny logger. Writes UTF-16 to the Windows console (or UTF-8 to stderr if
//! the handle isn't a console). Lock-free for the common case.

use crate::sys;
use core::fmt;
use core::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Fatal = 5,
}

impl LogLevel {
    fn tag(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO ",
            Self::Warn => "WARN ",
            Self::Error => "ERROR",
            Self::Fatal => "FATAL",
        }
    }
}

static MIN_LEVEL: AtomicU32 = AtomicU32::new(LogLevel::Info as u32);

pub fn set_min_level(level: LogLevel) {
    MIN_LEVEL.store(level as u32, Ordering::Relaxed);
}

pub fn enabled(level: LogLevel) -> bool {
    (level as u32) >= MIN_LEVEL.load(Ordering::Relaxed)
}

static OUT_LOCK: Mutex<()> = Mutex::new(());

pub fn log_at(level: LogLevel, args: fmt::Arguments<'_>) {
    if !enabled(level) {
        return;
    }
    let pid = unsafe { sys::GetCurrentProcessId() };
    let tid = unsafe { sys::GetCurrentThreadId() };
    let ms = crate::time::unix_epoch_millis();
    let line = format!("[{ms} {pid}/{tid} {}] {args}\n", level.tag());
    let _g = OUT_LOCK.lock().ok();
    write_line(&line);
    if level == LogLevel::Fatal {
        std::process::abort();
    }
}

fn write_line(s: &str) {
    let h = unsafe { sys::GetStdHandle(sys::STD_ERROR_HANDLE) };
    if h.is_null() || h == sys::INVALID_HANDLE_VALUE {
        return;
    }
    let utf16: Vec<u16> = s.encode_utf16().collect();
    let mut written: u32 = 0;
    let ok = unsafe {
        sys::WriteConsoleW(
            h,
            utf16.as_ptr(),
            utf16.len() as u32,
            &raw mut written,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        // Not a console — fall back to byte writes.
        let bytes = s.as_bytes();
        let mut w2: u32 = 0;
        unsafe {
            sys::WriteFile(
                h,
                bytes.as_ptr().cast(),
                bytes.len() as u32,
                &raw mut w2,
                std::ptr::null_mut(),
            );
        }
    }
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => { $crate::log::log_at($crate::log::LogLevel::Info, format_args!($($arg)*)) };
}

#[macro_export]
macro_rules! warn_ {
    ($($arg:tt)*) => { $crate::log::log_at($crate::log::LogLevel::Warn, format_args!($($arg)*)) };
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => { $crate::log::log_at($crate::log::LogLevel::Error, format_args!($($arg)*)) };
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => { $crate::log::log_at($crate::log::LogLevel::Debug, format_args!($($arg)*)) };
}

#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => { $crate::log::log_at($crate::log::LogLevel::Trace, format_args!($($arg)*)) };
}
