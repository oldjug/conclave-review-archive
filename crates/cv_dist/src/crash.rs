//! Crash reporter — Win32 SetUnhandledExceptionFilter + minidump shape.

#![allow(non_snake_case, non_camel_case_types)]

use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrashReport {
    pub timestamp_ms: u64,
    pub thread_id: u32,
    pub exception_code: u32,
    pub exception_address: u64,
    pub stack: Vec<u64>,
    pub module_list: Vec<String>,
    pub product: String,
    pub version: String,
    pub channel: String,
    pub user_consent: bool,
}

#[derive(Debug, Default)]
pub struct CrashSpooler {
    pending: Vec<CrashReport>,
}

impl CrashSpooler {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn enqueue(&mut self, report: CrashReport) -> bool {
        if !report.user_consent {
            return false;
        }
        self.pending.push(report);
        true
    }
    pub fn drain(&mut self) -> Vec<CrashReport> {
        std::mem::take(&mut self.pending)
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

// ----------------------------------------------------------------------
// Win32 SEH integration
// ----------------------------------------------------------------------

// EXCEPTION_RECORD per WinNT.h.
#[repr(C)]
struct EXCEPTION_RECORD {
    ExceptionCode: u32,
    ExceptionFlags: u32,
    ExceptionRecord: *mut EXCEPTION_RECORD,
    ExceptionAddress: *mut std::ffi::c_void,
    NumberParameters: u32,
    ExceptionInformation: [usize; 15],
}

#[repr(C)]
struct CONTEXT {
    _opaque: [u8; 1232], // x64 CONTEXT is 1232 bytes — we don't deref it
}

#[repr(C)]
struct EXCEPTION_POINTERS {
    ExceptionRecord: *mut EXCEPTION_RECORD,
    ContextRecord: *mut CONTEXT,
}

type LPTOP_LEVEL_EXCEPTION_FILTER =
    Option<unsafe extern "system" fn(*mut EXCEPTION_POINTERS) -> i32>;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetUnhandledExceptionFilter(
        new: LPTOP_LEVEL_EXCEPTION_FILTER,
    ) -> LPTOP_LEVEL_EXCEPTION_FILTER;
    fn GetCurrentThreadId() -> u32;
    fn RtlCaptureStackBackTrace(
        FramesToSkip: u32,
        FramesToCapture: u32,
        BackTrace: *mut *mut std::ffi::c_void,
        BackTraceHash: *mut u32,
    ) -> u16;
    fn GetSystemTimeAsFileTime(lpSystemTimeAsFileTime: *mut FILETIME);
}

#[repr(C)]
struct FILETIME {
    dwLowDateTime: u32,
    dwHighDateTime: u32,
}

const EXCEPTION_EXECUTE_HANDLER: i32 = 1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// Crash-reporter configuration shared with the SEH callback.
#[derive(Debug, Clone)]
pub struct CrashConfig {
    pub product: String,
    pub version: String,
    pub channel: String,
    pub user_consent: bool,
}

static CRASH_CONFIG: OnceLock<Mutex<Option<CrashConfig>>> = OnceLock::new();
static CRASH_REPORT: OnceLock<Mutex<Option<CrashReport>>> = OnceLock::new();

unsafe extern "system" fn unhandled_filter(info: *mut EXCEPTION_POINTERS) -> i32 {
    let cfg = match CRASH_CONFIG
        .get()
        .and_then(|m| m.lock().ok())
        .and_then(|g| g.clone())
    {
        Some(c) => c,
        None => return EXCEPTION_CONTINUE_SEARCH,
    };
    if info.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Read ExceptionRecord safely.
    let rec = unsafe { (*info).ExceptionRecord };
    let (code, addr) = if rec.is_null() {
        (0u32, 0u64)
    } else {
        unsafe { ((*rec).ExceptionCode, (*rec).ExceptionAddress as u64) }
    };
    // Capture up to 64 frames of stack.
    let mut frames: [*mut std::ffi::c_void; 64] = [std::ptr::null_mut(); 64];
    let n = unsafe { RtlCaptureStackBackTrace(0, 64, frames.as_mut_ptr(), std::ptr::null_mut()) };
    let stack: Vec<u64> = (0..n as usize).map(|i| frames[i] as u64).collect();
    let tid = unsafe { GetCurrentThreadId() };
    let mut ft = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    unsafe { GetSystemTimeAsFileTime(&mut ft) };
    let ft_100ns = ((ft.dwHighDateTime as u64) << 32) | ft.dwLowDateTime as u64;
    // FILETIME is 100-ns since 1601; convert to ms since UNIX epoch.
    let unix_ms = (ft_100ns / 10_000).saturating_sub(11_644_473_600_000);
    let report = CrashReport {
        timestamp_ms: unix_ms,
        thread_id: tid,
        exception_code: code,
        exception_address: addr,
        stack,
        module_list: Vec::new(),
        product: cfg.product,
        version: cfg.version,
        channel: cfg.channel,
        user_consent: cfg.user_consent,
    };
    if let Some(slot) = CRASH_REPORT.get() {
        if let Ok(mut g) = slot.lock() {
            *g = Some(report);
        }
    }
    EXCEPTION_EXECUTE_HANDLER
}

/// Install the unhandled-exception filter. Call once at startup.
pub fn install_seh_hook(config: CrashConfig) {
    let cfg_slot = CRASH_CONFIG.get_or_init(|| Mutex::new(None));
    *cfg_slot.lock().unwrap() = Some(config);
    let _ = CRASH_REPORT.get_or_init(|| Mutex::new(None));
    unsafe {
        SetUnhandledExceptionFilter(Some(unhandled_filter));
    }
}

/// Take any captured crash report (the SEH callback only stores one;
/// callers read it after fork-resurrection).
pub fn take_captured_report() -> Option<CrashReport> {
    CRASH_REPORT.get()?.lock().ok()?.take()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(consent: bool) -> CrashReport {
        CrashReport {
            timestamp_ms: 1,
            thread_id: 100,
            exception_code: 0xC0000005,
            exception_address: 0xDEAD_BEEF,
            stack: vec![1, 2, 3],
            module_list: vec!["conclave.exe".into()],
            product: "Conclave".into(),
            version: "0.0.1".into(),
            channel: "dev".into(),
            user_consent: consent,
        }
    }

    #[test]
    fn enqueue_requires_consent() {
        let mut s = CrashSpooler::new();
        assert!(!s.enqueue(report(false)));
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn drain_empties() {
        let mut s = CrashSpooler::new();
        s.enqueue(report(true));
        s.enqueue(report(true));
        assert_eq!(s.drain().len(), 2);
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn install_seh_hook_stores_config() {
        // Install — calls Win32 SetUnhandledExceptionFilter.
        // Verified by being able to retrieve the config.
        install_seh_hook(CrashConfig {
            product: "Test".into(),
            version: "1.0".into(),
            channel: "test".into(),
            user_consent: true,
        });
        let cfg = CRASH_CONFIG.get().unwrap().lock().unwrap().clone();
        assert!(cfg.is_some());
        assert_eq!(cfg.unwrap().product, "Test");
    }
}
