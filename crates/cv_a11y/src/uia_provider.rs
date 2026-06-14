//! UIA WM_GETOBJECT response + IRawElementProviderSimple COM vtable
//! (vtable layout only — Win32 calls UiaReturnRawElementProvider with
//! the pointer the broker hands back).

#![allow(non_snake_case, non_camel_case_types)]

use core::ffi::c_void;

const IID_IRAW_ELEMENT_PROVIDER_SIMPLE: [u32; 4] = [0xD6DD68D1, 0x86FD4E0C, 0xA8E70D45, 0xBA9D40FF];

#[repr(C)]
pub struct IRawElementProviderSimpleVtbl {
    pub QueryInterface: extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> i32,
    pub AddRef: extern "system" fn(*mut c_void) -> u32,
    pub Release: extern "system" fn(*mut c_void) -> u32,
    pub get_ProviderOptions: extern "system" fn(*mut c_void, *mut u32) -> i32,
    pub GetPatternProvider: extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> i32,
    pub GetPropertyValue: extern "system" fn(*mut c_void, i32, *mut Variant) -> i32,
    pub get_HostRawElementProvider: extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
}

#[repr(C)]
pub struct Variant {
    pub vt: u16,
    pub _pad: [u16; 3],
    pub data: u64,
}

/// Reply value for `SendMessage(WM_GETOBJECT)` when lParam ==
/// `UiaRootObjectId`. The accessibility runtime expects an HRESULT
/// produced by `UiaReturnRawElementProvider`.
#[link(name = "uiautomationcore")]
unsafe extern "system" {
    pub fn UiaReturnRawElementProvider(
        hwnd: *mut c_void,
        wparam: usize,
        lparam: isize,
        el: *mut c_void,
    ) -> isize;
}

pub const UIA_ROOT_OBJECT_ID: isize = -25;

/// Handle WM_GETOBJECT. Returns the LRESULT that
/// SendMessage/DefWindowProc should propagate. Caller supplies the
/// IRawElementProviderSimple pointer for the window's root.
pub fn handle_wm_getobject(
    hwnd: *mut c_void,
    wparam: usize,
    lparam: isize,
    root_provider: *mut c_void,
) -> isize {
    if lparam == UIA_ROOT_OBJECT_ID && !root_provider.is_null() {
        unsafe { UiaReturnRawElementProvider(hwnd, wparam, lparam, root_provider) }
    } else {
        0
    }
}
