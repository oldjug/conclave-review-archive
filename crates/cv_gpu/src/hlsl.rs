//! HLSL → DXBC bytecode compiler — Win32 D3DCompile FFI.
//!
//! Verbatim copy of the `hlsl` module from `cv_gfx/src/webgl.rs` so that
//! `cv_gpu` gains a usable HLSL→DXBC compiler WITHOUT taking a `cv_gfx`
//! dependency edge (`cv_gpu/Cargo.toml` stays dep-free). This adds a
//! `#[link(name = "d3dcompiler")]` reference — i.e. a RUNTIME dependency on
//! `d3dcompiler_47.dll`. That DLL ships with Windows and the developer SDK,
//! so for M5.1 bring-up the dependency is acceptable; it lets the shader
//! pipeline iterate on HLSL without a precompile toolchain.
//!
//! RECOMMENDATION for the `CV_GPU_PIPELINE` default-on flip: replace this
//! runtime compile with EMBEDDED precompiled DXBC (`const VS_DXBC: [u8; N]`
//! / `const PS_DXBC: [u8; N]`) fed straight to `CreateVertexShader`/
//! `CreatePixelShader`, eliminating the `d3dcompiler_47.dll` runtime
//! dependency entirely. A compile failure here is treated by the pipeline
//! init as a degrade-to-CopyResource trigger (no-stub policy), so a missing
//! `d3dcompiler_47.dll` never crashes — it just keeps the CopyResource
//! backend.

#![allow(non_snake_case, non_camel_case_types, dead_code)]
use std::ffi::{CString, c_void};

type HRESULT = i32;
type LPCSTR = *const u8;
type LPCVOID = *const c_void;
type SIZE_T = usize;

#[repr(C)]
struct ID3DBlobVtbl {
    QueryInterface: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> HRESULT,
    AddRef: unsafe extern "system" fn(*mut c_void) -> u32,
    Release: unsafe extern "system" fn(*mut c_void) -> u32,
    GetBufferPointer: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
    GetBufferSize: unsafe extern "system" fn(*mut c_void) -> SIZE_T,
}
#[repr(C)]
struct ID3DBlob {
    vtbl: *mut ID3DBlobVtbl,
}

#[link(name = "d3dcompiler")]
unsafe extern "system" {
    fn D3DCompile(
        pSrcData: LPCVOID,
        SrcDataSize: SIZE_T,
        pSourceName: LPCSTR,
        pDefines: LPCVOID,
        pInclude: LPCVOID,
        pEntrypoint: LPCSTR,
        pTarget: LPCSTR,
        Flags1: u32,
        Flags2: u32,
        ppCode: *mut *mut c_void,
        ppErrorMsgs: *mut *mut c_void,
    ) -> HRESULT;
}

/// Compile an HLSL source to DXBC. Returns the bytecode bytes on success;
/// an error string from `ID3DBlob::GetBufferPointer` on failure.
pub(crate) fn compile(hlsl: &str, entry: &str, target: &str) -> Result<Vec<u8>, String> {
    let entry_c = CString::new(entry).map_err(|e| e.to_string())?;
    let target_c = CString::new(target).map_err(|e| e.to_string())?;
    unsafe {
        let mut code: *mut c_void = std::ptr::null_mut();
        let mut err: *mut c_void = std::ptr::null_mut();
        let hr = D3DCompile(
            hlsl.as_ptr() as LPCVOID,
            hlsl.len(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            entry_c.as_ptr() as LPCSTR,
            target_c.as_ptr() as LPCSTR,
            0,
            0,
            &mut code,
            &mut err,
        );
        if hr < 0 || code.is_null() {
            let msg = if err.is_null() {
                format!("D3DCompile failed: 0x{:08X}", hr as u32)
            } else {
                let blob = err as *mut ID3DBlob;
                let ptr = ((*(*blob).vtbl).GetBufferPointer)(err);
                let len = ((*(*blob).vtbl).GetBufferSize)(err);
                let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
                let s = String::from_utf8_lossy(bytes).to_string();
                ((*(*blob).vtbl).Release)(err);
                s
            };
            return Err(msg);
        }
        let blob = code as *mut ID3DBlob;
        let ptr = ((*(*blob).vtbl).GetBufferPointer)(code);
        let len = ((*(*blob).vtbl).GetBufferSize)(code);
        let bytes = std::slice::from_raw_parts(ptr as *const u8, len).to_vec();
        ((*(*blob).vtbl).Release)(code);
        if !err.is_null() {
            let blob = err as *mut ID3DBlob;
            ((*(*blob).vtbl).Release)(err);
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed M5.1 VS/PS must compile to non-empty DXBC at vs_4_0/ps_4_0.
    /// If `d3dcompiler_47.dll` is missing this would fail to LINK rather than
    /// run, so reaching this test at all proves the dep is present; the
    /// pipeline still degrades gracefully on a compile error at runtime.
    #[test]
    fn fullscreen_shaders_compile() {
        let vs = crate::hw_present::VS_HLSL;
        let ps = crate::hw_present::PS_HLSL;
        let vs_dxbc = compile(vs, "VSMain", "vs_4_0").expect("VS compile");
        let ps_dxbc = compile(ps, "PSMain", "ps_4_0").expect("PS compile");
        assert!(!vs_dxbc.is_empty());
        assert!(!ps_dxbc.is_empty());
        // DXBC blobs begin with the 'DXBC' fourcc.
        assert_eq!(&vs_dxbc[0..4], b"DXBC");
        assert_eq!(&ps_dxbc[0..4], b"DXBC");
    }
}
