//! Direct3D 11 — Win32 FFI for D3D11CreateDevice + device/context.

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use crate::dxgi::{HRESULT, LPVOID};
use std::ffi::c_void;

// D3D_DRIVER_TYPE.
pub const D3D_DRIVER_TYPE_UNKNOWN: u32 = 0;
pub const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
pub const D3D_DRIVER_TYPE_REFERENCE: u32 = 2;
pub const D3D_DRIVER_TYPE_NULL: u32 = 3;
pub const D3D_DRIVER_TYPE_SOFTWARE: u32 = 4;
pub const D3D_DRIVER_TYPE_WARP: u32 = 5;

// D3D11 device creation flags.
pub const D3D11_CREATE_DEVICE_SINGLETHREADED: u32 = 0x1;
pub const D3D11_CREATE_DEVICE_DEBUG: u32 = 0x2;
pub const D3D11_CREATE_DEVICE_BGRA_SUPPORT: u32 = 0x20;

// Feature levels.
pub const D3D_FEATURE_LEVEL_11_1: u32 = 0xb100;
pub const D3D_FEATURE_LEVEL_11_0: u32 = 0xb000;
pub const D3D_FEATURE_LEVEL_10_1: u32 = 0xa100;
pub const D3D_FEATURE_LEVEL_10_0: u32 = 0xa000;
pub const D3D_FEATURE_LEVEL_9_3: u32 = 0x9300;
pub const D3D_FEATURE_LEVEL_9_2: u32 = 0x9200;
pub const D3D_FEATURE_LEVEL_9_1: u32 = 0x9100;

// D3D11 SDK version (from D3D11.h).
pub const D3D11_SDK_VERSION: u32 = 7;

#[link(name = "d3d11")]
unsafe extern "system" {
    pub fn D3D11CreateDevice(
        pAdapter: *mut c_void,
        DriverType: u32,
        Software: *mut c_void,
        Flags: u32,
        pFeatureLevels: *const u32,
        FeatureLevels: u32,
        SDKVersion: u32,
        ppDevice: *mut LPVOID,
        pFeatureLevel: *mut u32,
        ppImmediateContext: *mut LPVOID,
    ) -> HRESULT;
}

/// Create a D3D11 device on the default hardware adapter.
/// Returns (device, feature_level, immediate_context). Caller is
/// responsible for releasing the COM pointers.
pub fn create_device() -> Result<(LPVOID, u32, LPVOID), HRESULT> {
    let levels = [
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
    ];
    let mut device: LPVOID = std::ptr::null_mut();
    let mut context: LPVOID = std::ptr::null_mut();
    let mut fl: u32 = 0;
    unsafe {
        let hr = D3D11CreateDevice(
            std::ptr::null_mut(),
            D3D_DRIVER_TYPE_HARDWARE,
            std::ptr::null_mut(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, // required for DComp
            levels.as_ptr(),
            levels.len() as u32,
            D3D11_SDK_VERSION,
            &mut device,
            &mut fl,
            &mut context,
        );
        if hr < 0 {
            // Fall back to WARP if hardware isn't available (CI etc).
            let hr2 = D3D11CreateDevice(
                std::ptr::null_mut(),
                D3D_DRIVER_TYPE_WARP,
                std::ptr::null_mut(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                levels.as_ptr(),
                levels.len() as u32,
                D3D11_SDK_VERSION,
                &mut device,
                &mut fl,
                &mut context,
            );
            if hr2 < 0 {
                return Err(hr2);
            }
        }
    }
    Ok((device, fl, context))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_type_constants() {
        assert_eq!(D3D_DRIVER_TYPE_HARDWARE, 1);
        assert_eq!(D3D_DRIVER_TYPE_WARP, 5);
    }

    #[test]
    fn feature_level_constants() {
        assert_eq!(D3D_FEATURE_LEVEL_11_0, 0xb000);
        assert_eq!(D3D_FEATURE_LEVEL_11_1, 0xb100);
    }

    #[test]
    fn create_device_hardware_or_warp() {
        // Real call into d3d11.dll. Hardware on real machines, WARP on
        // CI. Either way must succeed.
        let (dev, fl, ctx) = create_device().expect("D3D11CreateDevice");
        assert!(!dev.is_null());
        assert!(!ctx.is_null());
        // Feature level must be one of the requested ones.
        assert!(matches!(
            fl,
            D3D_FEATURE_LEVEL_11_1
                | D3D_FEATURE_LEVEL_11_0
                | D3D_FEATURE_LEVEL_10_1
                | D3D_FEATURE_LEVEL_10_0
                | D3D_FEATURE_LEVEL_9_3
        ));
        unsafe {
            crate::dxgi::release(ctx as *mut std::ffi::c_void);
            crate::dxgi::release(dev as *mut std::ffi::c_void);
        }
    }
}
