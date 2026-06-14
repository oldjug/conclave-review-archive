//! `cv_ime` — IME (Input Method Editor) integration for CJK and other
//! complex-script input.
//!
//! Two surfaces:
//!   * **Win32 IMM glue**: declared `extern "system"` bindings to
//!     `ImmGetContext` / `ImmReleaseContext` / `ImmGetCompositionStringW`
//!     so cv_ui's WndProc can deliver `WM_IME_COMPOSITION` /
//!     `WM_IME_STARTCOMPOSITION` / `WM_IME_ENDCOMPOSITION` events to
//!     editable elements with the right composition + cursor state.
//!   * **Composition model**: a `Composition` struct the editable input
//!     code holds while the user types so we can render the composing
//!     run with a dotted underline (Chrome's "composition underline").
//!
//! The Win32 IMM path is the same one Chrome uses (`//ui/base/ime/win`)
//! — Microsoft IME → IMM32 → WM_IME_*.
//!
//! V1 scope: handle the Windows-native IME flow. TSF (Text Services
//! Framework) integration — Chrome's newer code path — is a V2 slice.

#![allow(non_snake_case, non_camel_case_types, clippy::missing_safety_doc)]

use core::ffi::c_void;

pub type HWND = *mut c_void;
pub type HIMC = *mut c_void;
pub type LPARAM = isize;
pub type WPARAM = usize;
pub type DWORD = u32;

/// `GCS_COMPSTR` — the composing run we draw with an underline.
pub const GCS_COMPSTR: DWORD = 0x0008;
/// `GCS_RESULTSTR` — the run committed when IME ends; we splice into the edit buffer.
pub const GCS_RESULTSTR: DWORD = 0x0800;
/// `GCS_CURSORPOS` — cursor position WITHIN the composition (UTF-16 chars).
pub const GCS_CURSORPOS: DWORD = 0x0080;

#[link(name = "imm32")]
unsafe extern "system" {
    pub fn ImmGetContext(h: HWND) -> HIMC;
    pub fn ImmReleaseContext(h: HWND, himc: HIMC) -> i32;
    pub fn ImmGetCompositionStringW(himc: HIMC, index: DWORD, buf: *mut u16, buf_len: DWORD)
    -> i32;
}

/// In-flight composition state held by an editable element.
#[derive(Debug, Default, Clone)]
pub struct Composition {
    /// Composing text (UTF-16 → String). Drawn with a dotted underline.
    pub composing: String,
    /// Cursor offset inside the composing run, in characters.
    pub cursor: usize,
    /// `true` between WM_IME_STARTCOMPOSITION and ENDCOMPOSITION.
    pub active: bool,
}

impl Composition {
    pub fn start(&mut self) {
        self.active = true;
        self.composing.clear();
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.active = false;
        self.composing.clear();
        self.cursor = 0;
    }

    /// Refresh the composing string + cursor from the current IMM
    /// context. Returns `Some(commit)` if the message also carried
    /// `GCS_RESULTSTR` — the caller splices `commit` into the buffer.
    ///
    /// Safety: `hwnd` must be a live window handle.
    pub unsafe fn update_from_hwnd(&mut self, hwnd: HWND, lparam: LPARAM) -> Option<String> {
        let himc = unsafe { ImmGetContext(hwnd) };
        if himc.is_null() {
            return None;
        }
        let mut commit: Option<String> = None;

        // GCS_COMPSTR: refresh composing text.
        if (lparam as DWORD) & GCS_COMPSTR != 0 {
            self.composing = unsafe { read_imm_string(himc, GCS_COMPSTR) }.unwrap_or_default();
            self.active = true;
        }
        if (lparam as DWORD) & GCS_CURSORPOS != 0 {
            // Cursor pos is returned in the high word via a separate
            // call with a zero-length buffer.
            let raw =
                unsafe { ImmGetCompositionStringW(himc, GCS_CURSORPOS, core::ptr::null_mut(), 0) };
            if raw >= 0 {
                self.cursor = raw as usize;
            }
        }
        if (lparam as DWORD) & GCS_RESULTSTR != 0 {
            commit = unsafe { read_imm_string(himc, GCS_RESULTSTR) };
            self.composing.clear();
            self.cursor = 0;
        }
        unsafe { ImmReleaseContext(hwnd, himc) };
        commit
    }
}

/// SAFETY: caller holds a live HIMC and is on the UI thread.
unsafe fn read_imm_string(himc: HIMC, index: DWORD) -> Option<String> {
    let byte_len = unsafe { ImmGetCompositionStringW(himc, index, core::ptr::null_mut(), 0) };
    if byte_len <= 0 {
        return None;
    }
    let utf16_len = (byte_len as usize) / 2;
    let mut buf: Vec<u16> = vec![0u16; utf16_len];
    let got = unsafe { ImmGetCompositionStringW(himc, index, buf.as_mut_ptr(), byte_len as DWORD) };
    if got <= 0 {
        return None;
    }
    Some(String::from_utf16_lossy(&buf[..utf16_len]))
}

/// Test helper: drive composition state directly without IMM, so the
/// rest of the system (paint, input event dispatch) can be exercised
/// in unit tests on machines without an IME installed.
pub fn drive_composition_for_test(
    c: &mut Composition,
    composing: &str,
    cursor: usize,
    commit: Option<&str>,
) -> Option<String> {
    if !composing.is_empty() {
        c.composing = composing.to_string();
        c.cursor = cursor;
        c.active = true;
    }
    if let Some(s) = commit {
        c.composing.clear();
        c.cursor = 0;
        return Some(s.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composition_lifecycle() {
        let mut c = Composition::default();
        c.start();
        assert!(c.active);
        drive_composition_for_test(&mut c, "你好", 1, None);
        assert_eq!(c.composing, "你好");
        assert_eq!(c.cursor, 1);
        let commit = drive_composition_for_test(&mut c, "", 0, Some("你好"));
        assert_eq!(commit.as_deref(), Some("你好"));
        assert!(c.composing.is_empty());
        c.end();
        assert!(!c.active);
    }

    #[test]
    fn end_clears_state() {
        let mut c = Composition::default();
        c.composing = "abc".into();
        c.cursor = 3;
        c.active = true;
        c.end();
        assert!(c.composing.is_empty());
        assert_eq!(c.cursor, 0);
        assert!(!c.active);
    }
}
