//! Crash reporter — installs SetUnhandledExceptionFilter and writes a
//! minidump to disk via MiniDumpWriteDump on the way out. The spool
//! directory under the profile root is uploaded by the auto-update
//! channel on the next launch.
//!
//! All Win32 symbols are declared inline so we don't pull in any
//! external bindings. The filter prints a short marker line then
//! invokes MiniDumpWriteDump with the standard module-info flags.

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;

type DWORD = u32;
type LONG = i32;
type HANDLE = *mut c_void;
type BOOL = i32;

#[repr(C)]
struct EXCEPTION_POINTERS {
    exception_record: *mut c_void,
    context_record: *mut c_void,
}

#[repr(C)]
struct MINIDUMP_EXCEPTION_INFORMATION {
    thread_id: DWORD,
    exception_pointers: *mut EXCEPTION_POINTERS,
    client_pointers: BOOL,
}

type TopLevelFilter = Option<extern "system" fn(*mut EXCEPTION_POINTERS) -> LONG>;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetUnhandledExceptionFilter(filter: TopLevelFilter) -> TopLevelFilter;
    fn GetCurrentProcess() -> HANDLE;
    fn GetCurrentProcessId() -> DWORD;
    fn GetCurrentThreadId() -> DWORD;
    fn CreateFileW(
        name: *const u16,
        access: DWORD,
        share: DWORD,
        sa: *mut c_void,
        creation: DWORD,
        flags: DWORD,
        template: HANDLE,
    ) -> HANDLE;
    fn CloseHandle(h: HANDLE) -> BOOL;
}

#[link(name = "dbghelp")]
unsafe extern "system" {
    fn MiniDumpWriteDump(
        process: HANDLE,
        process_id: DWORD,
        file: HANDLE,
        dump_type: DWORD,
        exc_info: *mut MINIDUMP_EXCEPTION_INFORMATION,
        user_stream: *mut c_void,
        callback: *mut c_void,
    ) -> BOOL;
}

const GENERIC_WRITE: DWORD = 0x4000_0000;
const CREATE_ALWAYS: DWORD = 2;
const MINIDUMP_NORMAL: DWORD = 0;
const EXCEPTION_EXECUTE_HANDLER: LONG = 1;
const INVALID_HANDLE_VALUE: HANDLE = !0usize as *mut c_void;

static SPOOL_DIR: OnceLock<PathBuf> = OnceLock::new();

fn spool_dir() -> &'static PathBuf {
    SPOOL_DIR.get_or_init(|| {
        let root = if let Ok(local) = std::env::var("LOCALAPPDATA") {
            PathBuf::from(local).join("Conclave").join("Crashes")
        } else {
            std::env::temp_dir().join("tb_crashes")
        };
        let _ = std::fs::create_dir_all(&root);
        root
    })
}

extern "system" fn unhandled_filter(info: *mut EXCEPTION_POINTERS) -> LONG {
    write_minidump(info);
    EXCEPTION_EXECUTE_HANDLER
}

fn write_minidump(info: *mut EXCEPTION_POINTERS) {
    let dir = spool_dir().clone();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("tb-{now}.dmp"));
    let mut wide: Vec<u16> = path.to_string_lossy().encode_utf16().collect();
    wide.push(0);
    unsafe {
        let h = CreateFileW(
            wide.as_ptr(),
            GENERIC_WRITE,
            0,
            ptr::null_mut(),
            CREATE_ALWAYS,
            0,
            ptr::null_mut(),
        );
        if h == INVALID_HANDLE_VALUE {
            return;
        }
        let mut exc = MINIDUMP_EXCEPTION_INFORMATION {
            thread_id: GetCurrentThreadId(),
            exception_pointers: info,
            client_pointers: 0,
        };
        let _ = MiniDumpWriteDump(
            GetCurrentProcess(),
            GetCurrentProcessId(),
            h,
            MINIDUMP_NORMAL,
            &mut exc,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        CloseHandle(h);
    }
}

/// Install the unhandled-exception filter. Idempotent — subsequent
/// calls re-install but Win32 collapses to the last setter.
pub fn install() {
    unsafe {
        let _ = SetUnhandledExceptionFilter(Some(unhandled_filter));
    }
}

/// Return the list of pending crash dumps that should be uploaded.
pub fn pending_dumps() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(read) = std::fs::read_dir(spool_dir()) {
        for entry in read.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("dmp") {
                out.push(p);
            }
        }
    }
    out
}
