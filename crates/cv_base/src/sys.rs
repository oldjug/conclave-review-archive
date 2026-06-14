//! Raw Win32 FFI declarations used by `cv_base` and re-exported for the rest
//! of the workspace. We declare what we need ourselves rather than pulling a
//! bindings crate, per the strict third-party policy.

#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]

use core::ffi::c_void;

pub type BOOL = i32;
pub type DWORD = u32;
pub type LARGE_INTEGER = i64;
pub type HANDLE = *mut c_void;
pub type HRESULT = i32;
pub type LPCWSTR = *const u16;
pub type LPVOID = *mut c_void;
pub type LPCVOID = *const c_void;
pub type ULONG_PTR = usize;

pub const INVALID_HANDLE_VALUE: HANDLE = !0_usize as HANDLE;
pub const STD_OUTPUT_HANDLE: DWORD = (-11_i32) as DWORD;
pub const STD_ERROR_HANDLE: DWORD = (-12_i32) as DWORD;

unsafe extern "system" {
    pub fn QueryPerformanceCounter(lp: *mut LARGE_INTEGER) -> BOOL;
    pub fn QueryPerformanceFrequency(lp: *mut LARGE_INTEGER) -> BOOL;
    pub fn GetStdHandle(nStdHandle: DWORD) -> HANDLE;
    pub fn WriteConsoleW(
        h: HANDLE,
        buf: *const u16,
        n: DWORD,
        written: *mut DWORD,
        reserved: LPVOID,
    ) -> BOOL;
    pub fn WriteFile(
        h: HANDLE,
        buf: LPCVOID,
        n: DWORD,
        written: *mut DWORD,
        overlapped: LPVOID,
    ) -> BOOL;
    pub fn GetSystemTimePreciseAsFileTime(out: *mut FILETIME);
    pub fn GetCurrentThreadId() -> DWORD;
    pub fn GetCurrentProcessId() -> DWORD;
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct FILETIME {
    pub low: DWORD,
    pub high: DWORD,
}
