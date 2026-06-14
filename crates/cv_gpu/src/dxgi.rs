//! DXGI (DirectX Graphics Infrastructure) — Win32 FFI.
//!
//! Declares the COM interface vtables for IDXGIFactory2, IDXGIAdapter1,
//! IDXGIDevice, IDXGISwapChain1, all the SwapChain descriptor structs,
//! and the formats/usages we use for browser rendering.

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use std::ffi::c_void;

pub type HRESULT = i32;
pub type DWORD = u32;
pub type UINT = u32;
pub type LPVOID = *mut c_void;
pub type HANDLE = *mut c_void;
pub type HWND = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GUID {
    pub Data1: u32,
    pub Data2: u16,
    pub Data3: u16,
    pub Data4: [u8; 8],
}

// IID_IDXGIFactory2: 50C83A1C-E072-4C48-87B0-3630FA36A6D0
pub const IID_IDXGI_FACTORY2: GUID = GUID {
    Data1: 0x50C83A1C,
    Data2: 0xE072,
    Data3: 0x4C48,
    Data4: [0x87, 0xB0, 0x36, 0x30, 0xFA, 0x36, 0xA6, 0xD0],
};
// IID_IDXGIDevice: 54EC77FA-1377-44E6-8C32-88FD5F44C84C
pub const IID_IDXGI_DEVICE: GUID = GUID {
    Data1: 0x54EC77FA,
    Data2: 0x1377,
    Data3: 0x44E6,
    Data4: [0x8C, 0x32, 0x88, 0xFD, 0x5F, 0x44, 0xC8, 0x4C],
};

// DXGI formats.
pub const DXGI_FORMAT_B8G8R8A8_UNORM: u32 = 87;
pub const DXGI_FORMAT_R8G8B8A8_UNORM: u32 = 28;
pub const DXGI_FORMAT_R16G16B16A16_FLOAT: u32 = 10;
pub const DXGI_FORMAT_R10G10B10A2_UNORM: u32 = 24;
pub const DXGI_FORMAT_D24_UNORM_S8_UINT: u32 = 45;

// SwapChain usage flags.
pub const DXGI_USAGE_RENDER_TARGET_OUTPUT: u32 = 0x00000020;
pub const DXGI_USAGE_BACK_BUFFER: u32 = 0x00000040;
pub const DXGI_USAGE_SHADER_INPUT: u32 = 0x00000010;

// Swap effects.
pub const DXGI_SWAP_EFFECT_DISCARD: u32 = 0;
pub const DXGI_SWAP_EFFECT_SEQUENTIAL: u32 = 1;
pub const DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL: u32 = 3;
pub const DXGI_SWAP_EFFECT_FLIP_DISCARD: u32 = 4;

// Scaling modes (DXGI_SCALING). A FLIP_* swap chain bound to a DComp visual
// must use NONE — STRETCH is invalid/ill-behaved for flip-model chains and
// produces offset/stretch artifacts when the back-buffer size and the target
// client area momentarily disagree (e.g. during maximize/resize). With NONE
// the back buffer maps 1:1 to the visual's top-left with no implicit scaling.
pub const DXGI_SCALING_STRETCH: u32 = 0;
pub const DXGI_SCALING_NONE: u32 = 1;
pub const DXGI_SCALING_ASPECT_RATIO_STRETCH: u32 = 2;

// Device-loss HRESULTs (from winerror.h). These are stored as the raw u32
// fourcc-style codes; compare against an `i32` HRESULT by `as i32`
// (sign-extension makes both negative — a positive-u32 comparison never
// matches a negative HRESULT, which is the classic TDR-detection bug).
pub const DXGI_ERROR_DEVICE_REMOVED: u32 = 0x887A_0005;
pub const DXGI_ERROR_DEVICE_RESET: u32 = 0x887A_0007;
pub const DXGI_ERROR_DEVICE_HUNG: u32 = 0x887A_0006;

// Alpha mode.
pub const DXGI_ALPHA_MODE_UNSPECIFIED: u32 = 0;
pub const DXGI_ALPHA_MODE_PREMULTIPLIED: u32 = 1;
pub const DXGI_ALPHA_MODE_STRAIGHT: u32 = 2;
pub const DXGI_ALPHA_MODE_IGNORE: u32 = 3;

#[repr(C)]
pub struct DXGI_SAMPLE_DESC {
    pub Count: u32,
    pub Quality: u32,
}

#[repr(C)]
pub struct DXGI_SWAP_CHAIN_DESC1 {
    pub Width: u32,
    pub Height: u32,
    pub Format: u32,
    pub Stereo: i32,
    pub SampleDesc: DXGI_SAMPLE_DESC,
    pub BufferUsage: u32,
    pub BufferCount: u32,
    pub Scaling: u32,
    pub SwapEffect: u32,
    pub AlphaMode: u32,
    pub Flags: u32,
}

#[repr(C)]
pub struct DXGI_ADAPTER_DESC1 {
    pub Description: [u16; 128],
    pub VendorId: u32,
    pub DeviceId: u32,
    pub SubSysId: u32,
    pub Revision: u32,
    pub DedicatedVideoMemory: usize,
    pub DedicatedSystemMemory: usize,
    pub SharedSystemMemory: usize,
    pub AdapterLuid: i64,
    pub Flags: u32,
}

// IUnknown vtable head.
#[repr(C)]
pub struct IUnknownVtbl {
    pub QueryInterface: unsafe extern "system" fn(*mut c_void, *const GUID, *mut LPVOID) -> HRESULT,
    pub AddRef: unsafe extern "system" fn(*mut c_void) -> u32,
    pub Release: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[repr(C)]
pub struct IDXGIFactory2Vtbl {
    pub iunknown: IUnknownVtbl,
    pub SetPrivateData: unsafe extern "system" fn() -> HRESULT,
    pub SetPrivateDataInterface: unsafe extern "system" fn() -> HRESULT,
    pub GetPrivateData: unsafe extern "system" fn() -> HRESULT,
    pub GetParent: unsafe extern "system" fn() -> HRESULT,
    pub EnumAdapters: unsafe extern "system" fn() -> HRESULT,
    pub MakeWindowAssociation: unsafe extern "system" fn() -> HRESULT,
    pub GetWindowAssociation: unsafe extern "system" fn() -> HRESULT,
    pub CreateSwapChain: unsafe extern "system" fn() -> HRESULT,
    pub CreateSoftwareAdapter: unsafe extern "system" fn() -> HRESULT,
    pub EnumAdapters1: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> HRESULT,
    pub IsCurrent: unsafe extern "system" fn() -> i32,
    pub IsWindowedStereoEnabled: unsafe extern "system" fn() -> i32,
    pub CreateSwapChainForHwnd: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void, // pDevice
        HWND,
        *const DXGI_SWAP_CHAIN_DESC1,
        *const c_void,    // pFullscreenDesc
        *mut c_void,      // pRestrictToOutput
        *mut *mut c_void, // ppSwapChain
    ) -> HRESULT,
    pub CreateSwapChainForCoreWindow: unsafe extern "system" fn() -> HRESULT,
    pub GetSharedResourceAdapterLuid: unsafe extern "system" fn() -> HRESULT,
    pub RegisterStereoStatusWindow: unsafe extern "system" fn() -> HRESULT,
    pub RegisterStereoStatusEvent: unsafe extern "system" fn() -> HRESULT,
    pub UnregisterStereoStatus: unsafe extern "system" fn() -> HRESULT,
    pub RegisterOcclusionStatusWindow: unsafe extern "system" fn() -> HRESULT,
    pub RegisterOcclusionStatusEvent: unsafe extern "system" fn() -> HRESULT,
    pub UnregisterOcclusionStatus: unsafe extern "system" fn() -> HRESULT,
    pub CreateSwapChainForComposition: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        *const DXGI_SWAP_CHAIN_DESC1,
        *mut c_void,
        *mut *mut c_void,
    ) -> HRESULT,
}

#[repr(C)]
pub struct IDXGIFactory2 {
    pub vtbl: *mut IDXGIFactory2Vtbl,
}

#[link(name = "dxgi")]
unsafe extern "system" {
    pub fn CreateDXGIFactory2(Flags: u32, riid: *const GUID, ppFactory: *mut LPVOID) -> HRESULT;
}

/// Create an IDXGIFactory2 with default flags. Returns a raw pointer
/// the caller is responsible for Releasing.
pub fn create_factory() -> Result<*mut IDXGIFactory2, HRESULT> {
    unsafe {
        let mut p: LPVOID = std::ptr::null_mut();
        let hr = CreateDXGIFactory2(0, &IID_IDXGI_FACTORY2, &mut p);
        if hr < 0 || p.is_null() {
            return Err(hr);
        }
        Ok(p as *mut IDXGIFactory2)
    }
}

/// Release a COM pointer.
pub unsafe fn release<T>(p: *mut T) {
    if p.is_null() {
        return;
    }
    let iunk = p as *mut c_void;
    // Safety: every DXGI/D3D object's vtable starts with the IUnknown
    // QueryInterface/AddRef/Release triple.
    let vtbl = unsafe { *(iunk as *mut *mut IUnknownVtbl) };
    unsafe { ((*vtbl).Release)(iunk) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iid_constants_match_dxgi_h() {
        // Spot-check the GUID byte pattern.
        assert_eq!(IID_IDXGI_FACTORY2.Data1, 0x50C83A1C);
        assert_eq!(IID_IDXGI_FACTORY2.Data4[7], 0xD0);
    }

    #[test]
    fn dxgi_format_constants_match_dxgi_h() {
        assert_eq!(DXGI_FORMAT_B8G8R8A8_UNORM, 87);
        assert_eq!(DXGI_FORMAT_R8G8B8A8_UNORM, 28);
        assert_eq!(DXGI_FORMAT_R16G16B16A16_FLOAT, 10);
    }

    #[test]
    fn device_loss_hresults_are_negative_as_i32() {
        // The TDR-detection path compares an `i32` HRESULT against these.
        // Both must sign-extend to NEGATIVE i32 or the comparison silently
        // never matches (the classic "compare against the u32 literal" bug).
        assert_eq!(DXGI_ERROR_DEVICE_REMOVED, 0x887A_0005);
        assert_eq!(DXGI_ERROR_DEVICE_RESET, 0x887A_0007);
        assert!((DXGI_ERROR_DEVICE_REMOVED as i32) < 0);
        assert!((DXGI_ERROR_DEVICE_RESET as i32) < 0);
        assert_eq!(DXGI_ERROR_DEVICE_REMOVED as i32, -2005270523);
    }

    #[test]
    fn create_factory_succeeds() {
        // Real call into dxgi.dll. On any modern Windows this must
        // return a non-null pointer.
        let f = create_factory().expect("CreateDXGIFactory2");
        assert!(!f.is_null());
        unsafe { release(f) };
    }

    #[test]
    fn swap_chain_desc_buffer_usage_offset() {
        // Layout invariant: BufferUsage sits at offset 24 (Width u32 +
        // Height u32 + Format u32 + Stereo i32 + SampleDesc {u32,u32} = 24).
        let d = DXGI_SWAP_CHAIN_DESC1 {
            Width: 0,
            Height: 0,
            Format: 0,
            Stereo: 0,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 0,
                Quality: 0,
            },
            BufferUsage: 0,
            BufferCount: 0,
            Scaling: 0,
            SwapEffect: 0,
            AlphaMode: 0,
            Flags: 0,
        };
        let base = &d as *const _ as usize;
        let usage_off = &d.BufferUsage as *const _ as usize - base;
        assert_eq!(usage_off, 24);
    }
}
