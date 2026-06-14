//! Hardware-accelerated present path — D3D11 + DXGI swap chain + DComp.
//!
//! Replaces GDI `StretchDIBits` with a DXGI flip-model swap chain for
//! tear-free, GPU-composited presentation.  The compositor's BGRA
//! bitmap is uploaded to a D3D11 staging texture, copied to the swap
//! chain back buffer, and presented via `IDXGISwapChain1::Present`.
//!
//! DirectComposition is layered on top: the swap chain is bound as
//! content of a DComp visual tree rooted at the HWND target, giving
//! us the same compositor model Chrome uses (per-layer visuals with
//! independent transform/opacity — ready for future per-layer GPU
//! compositing when we promote layers).
//!
//! The `HwPresenter` is created once per HWND and reused across
//! frames.  Fallback: if D3D11/DComp init fails (e.g. headless CI,
//! remote desktop without GPU), the caller falls back to the existing
//! StretchDIBits path — no panic.

#![allow(non_snake_case, non_camel_case_types, dead_code, unsafe_op_in_unsafe_fn)]

use crate::d3d11;
use crate::dcomp;
use crate::dxgi;
use std::ffi::c_void;

// ── Thread-affinity enforcement (M5.5 off-main compositor) ───────────
//
// The D3D11 IMMEDIATE context + the DComp device/target are thread-affine:
// they MUST be created on, and only ever called from, ONE thread. The
// off-main compositor moves all present/resize/construct work onto a
// dedicated `tb-compositor` thread. To make any accidental cross-thread
// call a loud panic in dev (and an optional hard panic under
// `CV_OFFMAIN_COMPOSITOR_AUDIT` in release), `HwPresenter` records the OS
// thread id of its creating thread and asserts it on every COM path.
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentThreadId() -> u32;
}

/// Returns true when the `CV_OFFMAIN_COMPOSITOR_AUDIT` env var is set to a
/// value-affirmative string. When on, the creator-thread affinity check in
/// `present_bgra`/`resize` is promoted from a `debug_assert` to a hard
/// `assert` so it fires in release soak builds too.
fn affinity_audit_enabled() -> bool {
    matches!(
        std::env::var("CV_OFFMAIN_COMPOSITOR_AUDIT").as_deref(),
        Ok("1") | Ok("on") | Ok("true") | Ok("yes")
    )
}

/// The pure affinity check shared by every COM funnel. Panics (debug-assert,
/// or hard assert under the audit env) when `cur != creator`. Extracted as a
/// free function so it is unit-testable WITHOUT a real device / HWND: a
/// `#[should_panic]` test can drive a deliberate tid mismatch directly.
#[inline]
fn check_affinity(cur: u32, creator: u32) {
    if affinity_audit_enabled() {
        assert_eq!(cur, creator, "HwPresenter called off its creating thread (audit)");
    } else {
        debug_assert_eq!(cur, creator, "HwPresenter called off its creating thread");
    }
}

// ── COM vtable slot helpers ──────────────────────────────────────────
//
// COM objects are `*mut *mut [fn_ptr; N]` — the first indirection
// reaches the vtable pointer, the second indexes the method.  We call
// specific slots without defining full vtable structs for every
// interface, which keeps this module self-contained.

/// Call a COM method with 0 extra args (besides `this`).
unsafe fn com0(obj: *mut c_void, slot: usize) -> i32 {
    let vtbl = *(obj as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void) -> i32 =
        core::mem::transmute(*vtbl.add(slot));
    fp(obj)
}

/// Release a COM object (IUnknown::Release = slot 2).
unsafe fn com_release(obj: *mut c_void) {
    if obj.is_null() { return; }
    let vtbl = *(obj as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void) -> u32 =
        core::mem::transmute(*vtbl.add(2));
    fp(obj);
}

/// IUnknown::QueryInterface = slot 0.
unsafe fn com_qi(obj: *mut c_void, iid: &dxgi::GUID, out: &mut *mut c_void) -> i32 {
    let vtbl = *(obj as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, *const dxgi::GUID, *mut *mut c_void) -> i32 =
        core::mem::transmute(*vtbl.add(0));
    fp(obj, iid as *const _, out as *mut _)
}

// ── D3D11 texture descriptor ─────────────────────────────────────────

#[repr(C)]
struct D3D11_TEXTURE2D_DESC {
    Width: u32,
    Height: u32,
    MipLevels: u32,
    ArraySize: u32,
    Format: u32,
    SampleDesc: dxgi::DXGI_SAMPLE_DESC,
    Usage: u32,
    BindFlags: u32,
    CPUAccessFlags: u32,
    MiscFlags: u32,
}

#[repr(C)]
struct D3D11_MAPPED_SUBRESOURCE {
    pData: *mut c_void,
    RowPitch: u32,
    DepthPitch: u32,
}

const D3D11_USAGE_STAGING: u32 = 3;
const D3D11_CPU_ACCESS_WRITE: u32 = 0x10000;
const D3D11_MAP_WRITE: u32 = 3;

// ── D3D11 usages / bind flags / map / sampler / blend constants ──────
//
// Added for the M5.1 ShaderQuad pipeline. Values transcribed from d3d11.h;
// they are validated against the documented numeric values in the tests.
const D3D11_BIND_VERTEX_BUFFER: u32 = 0x1;
const D3D11_BIND_INDEX_BUFFER: u32 = 0x2;
const D3D11_BIND_CONSTANT_BUFFER: u32 = 0x4;
const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
const D3D11_BIND_RENDER_TARGET: u32 = 0x20;

const D3D11_USAGE_DEFAULT: u32 = 0;
const D3D11_USAGE_IMMUTABLE: u32 = 1;
const D3D11_USAGE_DYNAMIC: u32 = 2;

const D3D11_CPU_ACCESS_READ: u32 = 0x20000;

const D3D11_MAP_READ: u32 = 1;
const D3D11_MAP_WRITE_DISCARD: u32 = 4;

const D3D11_FILTER_MIN_MAG_MIP_POINT: u32 = 0;
const D3D11_FILTER_MIN_MAG_MIP_LINEAR: u32 = 0x15;
const D3D11_TEXTURE_ADDRESS_CLAMP: u32 = 3;
const D3D11_COMPARISON_NEVER: u32 = 1;
const D3D11_FLOAT32_MAX: f32 = 3.402823466e+38_f32;

const D3D11_BLEND_ZERO: u32 = 1;
const D3D11_BLEND_ONE: u32 = 2;
const D3D11_BLEND_OP_ADD: u32 = 1;
const D3D11_COLOR_WRITE_ENABLE_ALL: u8 = 0x0F;

const D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: u32 = 5;

// ── M5.1 fixed full-screen-quad shaders (compiled at runtime via hlsl) ──
//
// Full-screen quad via SV_VertexID — NO vertex/index buffer. Triangle-strip
// of 4 verts covering clip space; uv in [0,1] with the V axis arranged so uv
// (0,0) maps to the TOP-LEFT texel (matches the top-left-origin BGRA row
// layout used by present_bgra / Bitmap::pixels). DO NOT "fix" the V flip —
// it is correct for top-left-origin rows.
pub(crate) const VS_HLSL: &str = "\
struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };\n\
VSOut VSMain(uint vid : SV_VertexID) {\n\
    VSOut o;\n\
    float2 uv = float2((vid & 1) ? 1.0 : 0.0, (vid & 2) ? 1.0 : 0.0);\n\
    o.uv = uv;\n\
    o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);\n\
    return o;\n\
}";

// Sample the composited-viewport texture and output the color UNCHANGED.
// NO sRGB conversion (format is _UNORM not _SRGB), NO premultiply, NO gamma.
// For a 1:1 full-screen quad with point sampling this returns each source
// texel exactly, byte-matching blend_bgra's opaque-copy case. PSMain_solid
// is STEP-1 only (proves draw wiring with no texture).
pub(crate) const PS_HLSL: &str = "\
Texture2D tex0 : register(t0);\n\
SamplerState samp0 : register(s0);\n\
struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };\n\
float4 PSMain(VSOut i) : SV_TARGET {\n\
    return tex0.Sample(samp0, i.uv);\n\
}\n\
float4 PSMain_solid(VSOut i) : SV_TARGET { return float4(1.0, 0.0, 0.0, 1.0); }";

// ── M5.1 EMBEDDED precompiled DXBC (the production happy path) ─────────
//
// `init_shader_pipeline` creates the VS/PS from these embedded bytes by
// DEFAULT, so the GPU render path needs NO `d3dcompiler_47.dll` at runtime
// (the runtime `crate::hlsl::compile` D3DCompile FFI is kept ONLY as a
// defensive fallback if shader-create-from-embedded ever fails).
//
// The bytes were produced OFFLINE from `src/shaders/quad_vs.hlsl` /
// `src/shaders/quad_ps.hlsl` (byte-identical to VS_HLSL / PS_HLSL above)
// with the Windows SDK `fxc.exe`. They are functionally identical to the
// runtime-D3DCompile output (same source, entry, target), so the M5.1
// golden-diff stays at max diff 0.
//
// REGENERATE (run from `crates/cv_gpu/src/shaders/`, SDK 10.0.26100.0):
//   fxc /nologo /T vs_4_0 /E VSMain /Fo quad_vs.dxbc quad_vs.hlsl
//   fxc /nologo /T ps_4_0 /E PSMain /Fo quad_ps.dxbc quad_ps.hlsl
// (fxc.exe lives under
//  "C:/Program Files (x86)/Windows Kits/10/bin/<ver>/x64/fxc.exe")
pub(crate) const QUAD_VS_DXBC: &[u8] = include_bytes!("shaders/quad_vs.dxbc");
pub(crate) const QUAD_PS_DXBC: &[u8] = include_bytes!("shaders/quad_ps.dxbc");

// ── repr(C) descriptor structs for the ShaderQuad pipeline ───────────

#[repr(C)]
struct D3D11_VIEWPORT {
    TopLeftX: f32,
    TopLeftY: f32,
    Width: f32,
    Height: f32,
    MinDepth: f32,
    MaxDepth: f32,
}

#[repr(C)]
struct D3D11_SAMPLER_DESC {
    Filter: u32,
    AddressU: u32,
    AddressV: u32,
    AddressW: u32,
    MipLODBias: f32,
    MaxAnisotropy: u32,
    ComparisonFunc: u32,
    BorderColor: [f32; 4],
    MinLOD: f32,
    MaxLOD: f32,
}

// RenderTargetWriteMask is a UINT8; the C struct pads to 4-byte alignment
// (7 u32 + 1 u8 + 3 pad = 32 bytes per element). #[repr(C)] reproduces this
// exactly because the largest field (u32) drives 4-byte alignment.
#[repr(C)]
#[derive(Clone, Copy)]
struct D3D11_RENDER_TARGET_BLEND_DESC {
    BlendEnable: i32,
    SrcBlend: u32,
    DestBlend: u32,
    BlendOp: u32,
    SrcBlendAlpha: u32,
    DestBlendAlpha: u32,
    BlendOpAlpha: u32,
    RenderTargetWriteMask: u8,
}

#[repr(C)]
struct D3D11_BLEND_DESC {
    AlphaToCoverageEnable: i32,
    IndependentBlendEnable: i32,
    RenderTarget: [D3D11_RENDER_TARGET_BLEND_DESC; 8],
}

// ── IDXGISwapChain vtable slot indices ───────────────────────────────
//
// Inheritance chain:
//   IUnknown (0..2) → IDXGIObject (3..6) → IDXGIDeviceSubObject (7)
//   → IDXGISwapChain (8..17) → IDXGISwapChain1 (18..)
const SWAP_PRESENT: usize       = 8;   // IDXGISwapChain::Present
const SWAP_GET_BUFFER: usize    = 9;   // IDXGISwapChain::GetBuffer
const SWAP_RESIZE_BUFFERS: usize = 13; // IDXGISwapChain::ResizeBuffers

// ── ID3D11Device vtable slot indices ─────────────────────────────────
//   IUnknown (0..2): QueryInterface, AddRef, Release.
//   ID3D11Device (3..): methods follow in d3d11.h declaration order:
//     3  CreateBuffer
//     4  CreateTexture1D
//     5  CreateTexture2D
//     6  CreateTexture3D
//     7  CreateShaderResourceView
//     8  CreateUnorderedAccessView
//     9  CreateRenderTargetView
//     10 CreateDepthStencilView
//     11 CreateInputLayout
//     12 CreateVertexShader
//     13 CreateGeometryShader
//     14 CreateGeometryShaderWithStreamOutput
//     15 CreatePixelShader
//     16 CreateHullShader
//     17 CreateDomainShader
//     18 CreateComputeShader
//     19 CreateClassLinkage
//     20 CreateBlendState
//     21 CreateDepthStencilState
//     22 CreateRasterizerState
//     23 CreateSamplerState
const DEV_CREATE_BUFFER: usize               = 3;  // ID3D11Device::CreateBuffer
const DEV_CREATE_TEXTURE2D: usize            = 5;  // ID3D11Device::CreateTexture2D
const DEV_CREATE_SHADER_RESOURCE_VIEW: usize = 7;  // ID3D11Device::CreateShaderResourceView
const DEV_CREATE_RENDER_TARGET_VIEW: usize   = 9;  // ID3D11Device::CreateRenderTargetView
const DEV_CREATE_INPUT_LAYOUT: usize         = 11; // ID3D11Device::CreateInputLayout
const DEV_CREATE_VERTEX_SHADER: usize        = 12; // ID3D11Device::CreateVertexShader
const DEV_CREATE_PIXEL_SHADER: usize         = 15; // ID3D11Device::CreatePixelShader
const DEV_CREATE_BLEND_STATE: usize          = 20; // ID3D11Device::CreateBlendState
const DEV_CREATE_SAMPLER_STATE: usize        = 23; // ID3D11Device::CreateSamplerState

// ── ID3D11DeviceContext vtable slot indices ───────────────────────────
//   IUnknown (0..2): QueryInterface, AddRef, Release.
//   ID3D11DeviceChild (3..6): GetDevice, GetPrivateData, SetPrivateData,
//     SetPrivateDataInterface.
//   ID3D11DeviceContext (7..): methods follow in d3d11.h declaration order:
//     7  VSSetConstantBuffers
//     8  PSSetShaderResources
//     9  PSSetShader
//     10 PSSetSamplers
//     11 VSSetShader
//     12 DrawIndexed
//     13 Draw
//     14 Map
//     15 Unmap
//     16 PSSetConstantBuffers
//     17 IASetInputLayout
//     18 IASetVertexBuffers
//     19 IASetIndexBuffer
//     20 DrawIndexedInstanced
//     21 DrawInstanced
//     22 GSSetConstantBuffers
//     23 GSSetShader
//     24 IASetPrimitiveTopology
//     25 VSSetShaderResources
//     26 VSSetSamplers
//     27 Begin
//     28 End
//     29 GetData
//     30 SetPredication
//     31 GSSetShaderResources
//     32 GSSetSamplers
//     33 OMSetRenderTargets
//     34 OMSetRenderTargetsAndUnorderedAccessViews
//     35 OMSetBlendState
//     36 OMSetDepthStencilState
//     37 SOSetTargets
//     38 DrawAuto
//     39 DrawIndexedInstancedIndirect
//     40 DrawInstancedIndirect
//     41 Dispatch
//     42 DispatchIndirect
//     43 RSSetState
//     44 RSSetViewports
//     45 RSSetScissorRects
//     46 CopySubresourceRegion
//     47 CopyResource
//     48 UpdateSubresource
//     49 CopyStructureCount
//     50 ClearRenderTargetView
const CTX_IA_SET_INPUT_LAYOUT: usize     = 17; // IASetInputLayout
const CTX_IA_SET_VERTEX_BUFFERS: usize   = 18; // IASetVertexBuffers
const CTX_IA_SET_INDEX_BUFFER: usize     = 19; // IASetIndexBuffer
const CTX_IA_SET_PRIMITIVE_TOPOLOGY: usize = 24; // IASetPrimitiveTopology
const CTX_VS_SET_SHADER: usize           = 11; // VSSetShader
const CTX_PS_SET_SHADER: usize           = 9;  // PSSetShader
const CTX_PS_SET_SHADER_RESOURCES: usize = 8;  // PSSetShaderResources
const CTX_PS_SET_SAMPLERS: usize         = 10; // PSSetSamplers
const CTX_OM_SET_RENDER_TARGETS: usize   = 33; // OMSetRenderTargets
const CTX_OM_SET_BLEND_STATE: usize      = 35; // OMSetBlendState
const CTX_RS_SET_VIEWPORTS: usize        = 44; // RSSetViewports
const CTX_CLEAR_RENDER_TARGET_VIEW: usize = 50; // ClearRenderTargetView
const CTX_DRAW_INDEXED_INSTANCED: usize  = 20; // DrawIndexedInstanced
const CTX_DRAW: usize           = 13;  // Draw (4-vertex SV_VertexID strip)
const CTX_MAP: usize            = 14;  // Map
const CTX_UNMAP: usize          = 15;  // Unmap
const CTX_COPY_RESOURCE: usize  = 47;  // CopyResource
const CTX_UPDATE_SUBRESOURCE: usize = 48; // UpdateSubresource

// ── IDCompositionDevice vtable slot indices ───────────────────────────
//   IUnknown (0..2) → IDCompositionDevice (3..)
const DCOMP_COMMIT: usize               = 3;
const DCOMP_CREATE_TARGET_FOR_HWND: usize = 6;
const DCOMP_CREATE_VISUAL: usize         = 7;

// ── IDCompositionTarget vtable slot indices ───────────────────────────
const TARGET_SET_ROOT: usize = 3;

// ── IDCompositionVisual vtable slot indices ───────────────────────────
//   IUnknown (0..2) → IDCompositionVisual:
//     3 SetOffsetX(IDCompositionAnimation*)   4 SetOffsetX(float)
//     5 SetOffsetY(IDCompositionAnimation*)   6 SetOffsetY(float)
//     7 SetTransform(animation) … 15 SetContent  (confirmed below)
const VISUAL_SET_OFFSET_X: usize = 4; // SetOffsetX(float)
const VISUAL_SET_OFFSET_Y: usize = 6; // SetOffsetY(float)
const VISUAL_SET_CONTENT: usize = 15;

// IID_ID3D11Texture2D: 6F15AAF2-D208-4E89-9AB4-489535D34F9C
const IID_ID3D11_TEXTURE2D: dxgi::GUID = dxgi::GUID {
    Data1: 0x6F15AAF2,
    Data2: 0xD208,
    Data3: 0x4E89,
    Data4: [0x9A, 0xB4, 0x48, 0x95, 0x35, 0xD3, 0x4F, 0x9C],
};

// ── Present backend selection ────────────────────────────────────────
//
// The presenter has a single named notion of "how the composited frame
// reaches the swap chain back buffer". Today there is exactly ONE backend:
// `CopyResource` — the staging-texture → `CopyResource` scanout blit driven
// by `present_bgra`/`present_u32`. M5.1 will add a second variant,
// `ShaderQuad`, that draws a full-screen textured quad through a real
// VS/PS/RTV pipeline; that variant is intentionally NOT present in Phase 0
// (Phase 0 ships no GPU render/draw path).

/// Which path uploads the CPU-composited frame to the swap chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentBackend {
    /// Staging texture → `ID3D11DeviceContext::CopyResource` → back buffer.
    /// The default whenever `CV_GPU_PIPELINE` is unset/`"0"`, and the
    /// degrade target whenever the ShaderQuad pipeline fails to initialize
    /// or the device is lost and cannot be rebuilt.
    CopyResource,
    /// M5.1: textured full-screen-quad draw via a real VS/PS/RTV pipeline.
    /// Selected when `CV_GPU_PIPELINE=1` AND the pipeline objects
    /// initialized successfully. Per frame: `UpdateSubresource` the
    /// composited viewport into a sampled DEFAULT texture, then draw one
    /// opaque full-screen quad (SV_VertexID, no vertex buffer) that samples
    /// it 1:1 into the swap back buffer. For the single opaque 1:1 quad the
    /// output is byte-identical to the CopyResource scanout.
    ShaderQuad,
}

/// Read the `CV_GPU_PIPELINE` env flag once. **Default ON** (flipped 2026-06-13
/// after a live-window soak on example.com / Hacker News / Wikipedia confirmed
/// the GPU shader render path + off-main compositor render real content correctly
/// with no crash; the golden-diff vs the CPU oracle is delta=0 and the path
/// degrades to CopyResource→StretchDIBits on any init failure). Escape hatch:
/// `CV_GPU_PIPELINE=0` forces it OFF (CPU staging-blit present); unset or any
/// other value → ON. Mirrors the `!= Ok("0")` default-on idiom used by
/// `CV_DAMAGE_RASTER` / `CV_PAINT_CACHE`.
fn gpu_pipeline_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_GPU_PIPELINE").as_deref() != Ok("0"))
}

/// Choose the present backend from the `CV_GPU_PIPELINE` flag. When the flag
/// is ON this returns `ShaderQuad`; the actual pipeline objects are built
/// lazily on the first present, and if that init fails the backend is
/// degraded back to `CopyResource` (the flag being ON is never worse than
/// today). With the flag OFF this always returns `CopyResource`.
fn select_present_backend() -> PresentBackend {
    if gpu_pipeline_enabled() {
        PresentBackend::ShaderQuad
    } else {
        PresentBackend::CopyResource
    }
}

// ── Public API ───────────────────────────────────────────────────────

/// Error from the hardware presenter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HwPresentError {
    D3D11DeviceFailed(i32),
    DxgiFactoryFailed(i32),
    SwapChainFailed(i32),
    DCompDeviceFailed(i32),
    DCompTargetFailed(i32),
    DCompVisualFailed(i32),
    StagingTextureFailed(i32),
    MapFailed(i32),
    PresentFailed(i32),
    ResizeFailed(i32),
    GetBufferFailed(i32),
    /// ShaderQuad pipeline init failed (shader compile, Create* call, or a
    /// missing `d3dcompiler_47.dll`). Always handled by degrading to the
    /// CopyResource backend — never propagated out of `present_bgra`.
    PipelineInitFailed,
}

/// GPU-backed presenter bound to an HWND.  Created once, reused per
/// frame.  Owns:
/// - D3D11 device + immediate context
/// - DXGI swap chain (FLIP_DISCARD, B8G8R8A8_UNORM)
/// - DComp device + target + root visual (swap chain as content)
/// - Staging texture for CPU→GPU BGRA upload
pub struct HwPresenter {
    device: *mut c_void,
    context: *mut c_void,
    swap_chain: *mut c_void,    // IDXGISwapChain1
    staging: *mut c_void,       // ID3D11Texture2D (STAGING, CPU_WRITE)
    dcomp_device: *mut c_void,  // IDCompositionDevice
    dcomp_target: *mut c_void,  // IDCompositionTarget
    dcomp_visual: *mut c_void,  // IDCompositionVisual
    width: u32,
    height: u32,
    /// The active present backend (see `PresentBackend`). Resolved from
    /// `CV_GPU_PIPELINE` at construction; may degrade to `CopyResource` at
    /// runtime if the ShaderQuad pipeline fails to init or the device is lost.
    backend: PresentBackend,

    // ── ShaderQuad pipeline objects (null/false under CopyResource) ──
    //
    // The HWND is captured for self-contained TDR rebuilds (present_bgra
    // does not otherwise receive it). Unused under CopyResource, so it
    // cannot regress flag-OFF behavior.
    hwnd: *mut c_void,
    /// SAMPLED source texture (DEFAULT, BIND_SHADER_RESOURCE, _UNORM) — the
    /// draw input. Size-dependent: recreated in `resize` and on TDR. The
    /// CPU-write STAGING texture is KEPT for the CopyResource fallback.
    source_tex: *mut c_void,    // ID3D11Texture2D
    source_srv: *mut c_void,    // ID3D11ShaderResourceView (size-dependent)
    vs: *mut c_void,            // ID3D11VertexShader (size-independent)
    ps_tex: *mut c_void,        // ID3D11PixelShader (PSMain; size-independent)
    sampler: *mut c_void,       // ID3D11SamplerState (size-independent)
    blend_state: *mut c_void,   // ID3D11BlendState (size-independent)
    /// True once all ShaderQuad pipeline objects are built. False forces a
    /// lazy init on the next ShaderQuad present.
    pipeline_ready: bool,
    /// OS thread id of the thread that called `new()`. The D3D11 immediate
    /// context + DComp objects are thread-affine; `present_bgra`/`resize`
    /// assert the calling thread matches this so an accidental cross-thread
    /// COM call (e.g. the off-main compositor presenter being touched from
    /// the UI thread) panics in dev instead of silently corrupting GPU
    /// state. See `affinity_audit_enabled` for the release-soak promotion.
    creator_tid: u32,
}

impl HwPresenter {
    /// Create a hardware presenter for the given HWND.
    ///
    /// On success, the swap chain is bound to a DComp visual tree
    /// targeting `hwnd`.  Returns `Err` if any D3D11/DXGI/DComp call
    /// fails — the caller should fall back to StretchDIBits.
    pub fn new(hwnd: *mut c_void, width: u32, height: u32) -> Result<Self, HwPresentError> {
        // 1. D3D11 device + immediate context.
        let (device, _fl, context) = d3d11::create_device()
            .map_err(HwPresentError::D3D11DeviceFailed)?;

        // 2. DXGI factory.
        let factory = dxgi::create_factory()
            .map_err(HwPresentError::DxgiFactoryFailed)?;

        // 3. Swap chain for HWND (FLIP_DISCARD).
        let swap_chain = unsafe {
            create_swap_chain_for_hwnd(factory, device, hwnd, width, height)
        };
        // Release factory — we don't need it after swap chain creation.
        unsafe { dxgi::release(factory) };
        let swap_chain = swap_chain.map_err(HwPresentError::SwapChainFailed)?;

        // 4. DComp device from the D3D11 device's IDXGIDevice.
        let dcomp_device = unsafe {
            create_dcomp_device(device)
        }.map_err(HwPresentError::DCompDeviceFailed)?;

        // 5. DComp target for HWND.
        let dcomp_target = unsafe {
            create_dcomp_target(dcomp_device, hwnd)
        }.map_err(HwPresentError::DCompTargetFailed)?;

        // 6. DComp visual with swap chain as content.
        let dcomp_visual = unsafe {
            create_dcomp_visual(dcomp_device, dcomp_target, swap_chain)
        }.map_err(HwPresentError::DCompVisualFailed)?;

        // 7. Commit the DComp tree so the swap chain is bound.
        unsafe { dcomp_commit(dcomp_device) };

        // 8. Staging texture for CPU→GPU upload.
        let staging = unsafe {
            create_staging_texture(device, width, height)
        }.map_err(HwPresentError::StagingTextureFailed)?;

        // Resolve the present backend once from `CV_GPU_PIPELINE`. Under the
        // flag this is `ShaderQuad`; the pipeline objects are built lazily on
        // the first present (and degrade to `CopyResource` if init fails).
        // `present_bgra`/`present_u32` read this to pick the path.
        let backend = select_present_backend();

        Ok(Self {
            device,
            context,
            swap_chain,
            staging,
            dcomp_device,
            dcomp_target,
            dcomp_visual,
            width,
            height,
            backend,
            hwnd,
            source_tex: std::ptr::null_mut(),
            source_srv: std::ptr::null_mut(),
            vs: std::ptr::null_mut(),
            ps_tex: std::ptr::null_mut(),
            sampler: std::ptr::null_mut(),
            blend_state: std::ptr::null_mut(),
            pipeline_ready: false,
            creator_tid: unsafe { GetCurrentThreadId() },
        })
    }

    /// The OS thread id this presenter was constructed on. All COM calls
    /// (`present_bgra`/`present_u32`/`resize`) are valid ONLY on this thread.
    pub fn creator_thread_id(&self) -> u32 {
        self.creator_tid
    }

    /// Assert (debug, or hard under the audit env) that the current thread is
    /// the presenter's creating thread. Called at the top of every COM funnel.
    #[inline]
    fn assert_affinity(&self) {
        let cur = unsafe { GetCurrentThreadId() };
        check_affinity(cur, self.creator_tid);
    }

    /// Present a BGRA pixel buffer to the swap chain.
    ///
    /// `bgra` must be exactly `width * height * 4` bytes, laid out as
    /// rows of B,G,R,A quads from top-left to bottom-right (matching
    /// `cv_gfx::Bitmap::pixels` after packing into bytes).
    ///
    /// The staging texture is mapped, pixels are copied in, unmapped,
    /// then CopyResource'd to the swap chain back buffer, and
    /// `Present(1, 0)` is called for vsync-aligned display.
    pub fn present_bgra(&mut self, bgra: &[u8], width: u32, height: u32) -> Result<(), HwPresentError> {
        // Affinity gate: present uses the thread-affine immediate context.
        self.assert_affinity();
        if width != self.width || height != self.height {
            self.resize(width, height)?;
        }

        // Dispatch on the active present backend. With the flag OFF the
        // backend is `CopyResource` and the staging → `CopyResource` scanout
        // blit below runs byte-for-byte as before. With the flag ON the
        // backend is `ShaderQuad`: lazily build the pipeline (degrading to
        // CopyResource on failure) and draw the textured full-screen quad.
        if self.backend == PresentBackend::ShaderQuad {
            if !self.pipeline_ready {
                if self.init_shader_pipeline().is_err() {
                    // Degrade permanently: pipeline init failed (no
                    // d3dcompiler, shader compile error, Create* failed).
                    // The flag being ON must never be worse than today.
                    self.backend = PresentBackend::CopyResource;
                }
            }
            if self.backend == PresentBackend::ShaderQuad {
                match self.present_shader_quad(bgra, width, height) {
                    Ok(()) => return Ok(()),
                    Err(PresentBackend::CopyResource) => {
                        // Device-loss rebuild failed mid-frame -> degraded.
                        // Skip this frame's present and let the next frame
                        // retry on the CopyResource path. Never error here.
                        return Ok(());
                    }
                    Err(_) => {
                        // Unreachable sentinel; treated as degrade.
                        self.backend = PresentBackend::CopyResource;
                    }
                }
            }
        }

        unsafe {
            // Map staging texture.
            let mut mapped = D3D11_MAPPED_SUBRESOURCE {
                pData: std::ptr::null_mut(),
                RowPitch: 0,
                DepthPitch: 0,
            };
            let hr = self.ctx_map(self.staging, &mut mapped);
            if hr < 0 {
                return Err(HwPresentError::MapFailed(hr));
            }

            // Copy BGRA data row by row (staging row pitch may differ
            // from width*4 due to GPU alignment).
            let src_stride = (width * 4) as usize;
            let dst_stride = mapped.RowPitch as usize;
            let src = bgra.as_ptr();
            let dst = mapped.pData as *mut u8;
            if src_stride == dst_stride {
                // Fast path: memcpy the whole thing.
                std::ptr::copy_nonoverlapping(src, dst, src_stride * height as usize);
            } else {
                for row in 0..height as usize {
                    std::ptr::copy_nonoverlapping(
                        src.add(row * src_stride),
                        dst.add(row * dst_stride),
                        src_stride,
                    );
                }
            }

            self.ctx_unmap(self.staging);

            // Get back buffer and copy staging → back buffer.
            let back_buffer = self.swap_get_buffer()?;
            self.ctx_copy_resource(back_buffer, self.staging);
            com_release(back_buffer);

            // Present with vsync (sync interval = 1).
            let hr = self.swap_present(1, 0);
            if hr < 0 {
                return Err(HwPresentError::PresentFailed(hr));
            }

            Ok(())
        }
    }

    /// Present a u32 pixel buffer (packed BGRA as u32 per pixel, same
    /// layout as `cv_gfx::Bitmap::pixels`).
    pub fn present_u32(&mut self, pixels: &[u32], width: u32, height: u32) -> Result<(), HwPresentError> {
        let bgra = unsafe {
            std::slice::from_raw_parts(
                pixels.as_ptr() as *const u8,
                pixels.len() * 4,
            )
        };
        self.present_bgra(bgra, width, height)
    }

    /// Resize the swap chain + staging texture (+ the size-dependent
    /// ShaderQuad sampled texture and its SRV, if the pipeline is active).
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), HwPresentError> {
        // Affinity gate: resize rebuilds the thread-affine swap chain + staging.
        self.assert_affinity();
        if width == 0 || height == 0 {
            return Ok(());
        }

        // Release old staging texture.
        unsafe { com_release(self.staging) };
        self.staging = std::ptr::null_mut();

        // Release the size-dependent ShaderQuad objects (SRV before its
        // texture). The size-INDEPENDENT vs/ps/sampler/blend_state survive a
        // resize. If we leaked these we'd slowly bleed GPU memory per resize.
        unsafe {
            com_release(self.source_srv);
            com_release(self.source_tex);
        }
        self.source_srv = std::ptr::null_mut();
        self.source_tex = std::ptr::null_mut();

        // Resize swap chain buffers.
        let hr = unsafe { self.swap_resize(width, height) };
        if hr < 0 {
            return Err(HwPresentError::ResizeFailed(hr));
        }

        // Re-fit the DComp visual to the client area, anchored at origin (0,0),
        // and commit. With DXGI_SCALING_NONE the swap chain maps 1:1 into the
        // visual at this offset — pinning it to the top-left prevents the
        // maximize/restore rightward-shift + left desktop gap.
        unsafe { self.refit_visual_to_origin() };

        // Recreate staging texture at new size.
        let staging = unsafe { create_staging_texture(self.device, width, height) }
            .map_err(HwPresentError::StagingTextureFailed)?;
        self.staging = staging;
        self.width = width;
        self.height = height;

        // Recreate the size-dependent sampled texture + SRV if the ShaderQuad
        // pipeline is active. On failure, degrade to CopyResource (never
        // error out of resize for a GPU-pipeline-only object).
        if self.backend == PresentBackend::ShaderQuad && self.pipeline_ready {
            match unsafe { self.create_source_tex_and_srv(width, height) } {
                Ok((tex, srv)) => {
                    self.source_tex = tex;
                    self.source_srv = srv;
                }
                Err(_) => {
                    self.backend = PresentBackend::CopyResource;
                    self.pipeline_ready = false;
                }
            }
        }

        Ok(())
    }

    /// Current swap chain dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The active present backend, resolved once at construction from
    /// `CV_GPU_PIPELINE`. Phase 0 always reports `CopyResource`.
    pub fn backend(&self) -> PresentBackend {
        self.backend
    }

    // ── COM method wrappers ──────────────────────────────────────────

    unsafe fn ctx_map(&self, texture: *mut c_void, mapped: &mut D3D11_MAPPED_SUBRESOURCE) -> i32 {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *mut c_void, u32, u32, u32, *mut D3D11_MAPPED_SUBRESOURCE
        ) -> i32 = core::mem::transmute(*vtbl.add(CTX_MAP));
        fp(self.context, texture, 0, D3D11_MAP_WRITE, 0, mapped)
    }

    unsafe fn ctx_unmap(&self, texture: *mut c_void) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) =
            core::mem::transmute(*vtbl.add(CTX_UNMAP));
        fp(self.context, texture, 0);
    }

    unsafe fn ctx_copy_resource(&self, dst: *mut c_void, src: *mut c_void) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) =
            core::mem::transmute(*vtbl.add(CTX_COPY_RESOURCE));
        fp(self.context, dst, src);
    }

    unsafe fn swap_present(&self, sync_interval: u32, flags: u32) -> i32 {
        let vtbl = *(self.swap_chain as *const *const usize);
        let fp: unsafe extern "system" fn(*mut c_void, u32, u32) -> i32 =
            core::mem::transmute(*vtbl.add(SWAP_PRESENT));
        fp(self.swap_chain, sync_interval, flags)
    }

    unsafe fn swap_get_buffer(&self) -> Result<*mut c_void, HwPresentError> {
        let vtbl = *(self.swap_chain as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, u32, *const dxgi::GUID, *mut *mut c_void
        ) -> i32 = core::mem::transmute(*vtbl.add(SWAP_GET_BUFFER));
        let mut buf: *mut c_void = std::ptr::null_mut();
        let hr = fp(self.swap_chain, 0, &IID_ID3D11_TEXTURE2D, &mut buf);
        if hr < 0 || buf.is_null() {
            return Err(HwPresentError::GetBufferFailed(hr));
        }
        Ok(buf)
    }

    unsafe fn swap_resize(&self, w: u32, h: u32) -> i32 {
        let vtbl = *(self.swap_chain as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, u32, u32, u32, u32, u32
        ) -> i32 = core::mem::transmute(*vtbl.add(SWAP_RESIZE_BUFFERS));
        fp(self.swap_chain, 2, w, h, dxgi::DXGI_FORMAT_B8G8R8A8_UNORM, 0)
    }

    /// Re-fit the DComp visual to the client area after a swap-chain resize.
    ///
    /// The visual hosts the swap chain via SetContent and (with the swap chain
    /// at DXGI_SCALING_NONE) maps it 1:1 at the visual's offset. We pin that
    /// offset to (0,0) so the content always anchors to the window's top-left —
    /// guarding against any stray offset/transform that could otherwise survive
    /// a maximize/restore and shift the page right with a desktop gap on the
    /// left. After ResizeBuffers the swap chain reports the new size; resetting
    /// the offset + committing makes DComp recompose the visual at the new
    /// extent anchored at origin. No-op (returns) if the DComp tree is absent
    /// (e.g. headless/offscreen harness).
    unsafe fn refit_visual_to_origin(&self) {
        if self.dcomp_visual.is_null() || self.dcomp_device.is_null() {
            return;
        }
        let vvtbl = *(self.dcomp_visual as *const *const usize);
        let set_off_x: unsafe extern "system" fn(*mut c_void, f32) -> i32 =
            core::mem::transmute(*vvtbl.add(VISUAL_SET_OFFSET_X));
        let set_off_y: unsafe extern "system" fn(*mut c_void, f32) -> i32 =
            core::mem::transmute(*vvtbl.add(VISUAL_SET_OFFSET_Y));
        let _ = set_off_x(self.dcomp_visual, 0.0);
        let _ = set_off_y(self.dcomp_visual, 0.0);
        dcomp_commit(self.dcomp_device);
    }

    // ── ShaderQuad COM method wrappers (device) ──────────────────────

    /// ID3D11Device::CreateRenderTargetView(pResource, pDesc=null, &out).
    unsafe fn dev_create_render_target_view(&self, resource: *mut c_void) -> Result<*mut c_void, i32> {
        let vtbl = *(self.device as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *mut c_void, *const c_void, *mut *mut c_void
        ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_RENDER_TARGET_VIEW));
        let mut out: *mut c_void = std::ptr::null_mut();
        let hr = fp(self.device, resource, std::ptr::null(), &mut out);
        if hr < 0 || out.is_null() { return Err(hr); }
        Ok(out)
    }

    /// ID3D11Device::CreateShaderResourceView(pResource, pDesc=null, &out).
    unsafe fn dev_create_shader_resource_view(&self, resource: *mut c_void) -> Result<*mut c_void, i32> {
        let vtbl = *(self.device as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *mut c_void, *const c_void, *mut *mut c_void
        ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_SHADER_RESOURCE_VIEW));
        let mut out: *mut c_void = std::ptr::null_mut();
        let hr = fp(self.device, resource, std::ptr::null(), &mut out);
        if hr < 0 || out.is_null() { return Err(hr); }
        Ok(out)
    }

    /// ID3D11Device::CreateVertexShader(pBytecode, len, pClassLinkage=null, &out).
    unsafe fn dev_create_vertex_shader(&self, dxbc: &[u8]) -> Result<*mut c_void, i32> {
        let vtbl = *(self.device as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *const c_void, usize, *mut c_void, *mut *mut c_void
        ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_VERTEX_SHADER));
        let mut out: *mut c_void = std::ptr::null_mut();
        let hr = fp(self.device, dxbc.as_ptr() as *const c_void, dxbc.len(), std::ptr::null_mut(), &mut out);
        if hr < 0 || out.is_null() { return Err(hr); }
        Ok(out)
    }

    /// ID3D11Device::CreatePixelShader(pBytecode, len, pClassLinkage=null, &out).
    unsafe fn dev_create_pixel_shader(&self, dxbc: &[u8]) -> Result<*mut c_void, i32> {
        let vtbl = *(self.device as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *const c_void, usize, *mut c_void, *mut *mut c_void
        ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_PIXEL_SHADER));
        let mut out: *mut c_void = std::ptr::null_mut();
        let hr = fp(self.device, dxbc.as_ptr() as *const c_void, dxbc.len(), std::ptr::null_mut(), &mut out);
        if hr < 0 || out.is_null() { return Err(hr); }
        Ok(out)
    }

    /// ID3D11Device::CreateSamplerState(pDesc, &out).
    unsafe fn dev_create_sampler_state(&self, desc: &D3D11_SAMPLER_DESC) -> Result<*mut c_void, i32> {
        let vtbl = *(self.device as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *const D3D11_SAMPLER_DESC, *mut *mut c_void
        ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_SAMPLER_STATE));
        let mut out: *mut c_void = std::ptr::null_mut();
        let hr = fp(self.device, desc as *const _, &mut out);
        if hr < 0 || out.is_null() { return Err(hr); }
        Ok(out)
    }

    /// ID3D11Device::CreateBlendState(pDesc, &out).
    unsafe fn dev_create_blend_state(&self, desc: &D3D11_BLEND_DESC) -> Result<*mut c_void, i32> {
        let vtbl = *(self.device as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *const D3D11_BLEND_DESC, *mut *mut c_void
        ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_BLEND_STATE));
        let mut out: *mut c_void = std::ptr::null_mut();
        let hr = fp(self.device, desc as *const _, &mut out);
        if hr < 0 || out.is_null() { return Err(hr); }
        Ok(out)
    }

    // ── ShaderQuad COM method wrappers (context) ─────────────────────

    /// ID3D11DeviceContext::UpdateSubresource(dst, 0, null box, src, rowpitch, 0).
    unsafe fn ctx_update_subresource(&self, dst: *mut c_void, src: *const c_void, row_pitch: u32) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *mut c_void, u32, *const c_void, *const c_void, u32, u32
        ) = core::mem::transmute(*vtbl.add(CTX_UPDATE_SUBRESOURCE));
        fp(self.context, dst, 0, std::ptr::null(), src, row_pitch, 0);
    }

    /// ID3D11DeviceContext::OMSetRenderTargets(1, &rtv, null DSV).
    unsafe fn ctx_om_set_render_targets(&self, rtv: *mut c_void) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, u32, *const *mut c_void, *mut c_void
        ) = core::mem::transmute(*vtbl.add(CTX_OM_SET_RENDER_TARGETS));
        let rtvs = [rtv];
        fp(self.context, 1, rtvs.as_ptr(), std::ptr::null_mut());
    }

    /// ID3D11DeviceContext::RSSetViewports(1, &vp).
    unsafe fn ctx_rs_set_viewports(&self, vp: &D3D11_VIEWPORT) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, u32, *const D3D11_VIEWPORT
        ) = core::mem::transmute(*vtbl.add(CTX_RS_SET_VIEWPORTS));
        fp(self.context, 1, vp as *const _);
    }

    /// ID3D11DeviceContext::ClearRenderTargetView(rtv, &rgba).
    unsafe fn ctx_clear_render_target_view(&self, rtv: *mut c_void, rgba: &[f32; 4]) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *mut c_void, *const f32
        ) = core::mem::transmute(*vtbl.add(CTX_CLEAR_RENDER_TARGET_VIEW));
        fp(self.context, rtv, rgba.as_ptr());
    }

    /// ID3D11DeviceContext::IASetPrimitiveTopology(topology).
    unsafe fn ctx_ia_set_primitive_topology(&self, topology: u32) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(*mut c_void, u32) =
            core::mem::transmute(*vtbl.add(CTX_IA_SET_PRIMITIVE_TOPOLOGY));
        fp(self.context, topology);
    }

    /// ID3D11DeviceContext::IASetInputLayout(null) — SV_VertexID path.
    unsafe fn ctx_ia_set_input_layout_null(&self) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(*mut c_void, *mut c_void) =
            core::mem::transmute(*vtbl.add(CTX_IA_SET_INPUT_LAYOUT));
        fp(self.context, std::ptr::null_mut());
    }

    /// ID3D11DeviceContext::VSSetShader(vs, null, 0).
    unsafe fn ctx_vs_set_shader(&self, vs: *mut c_void) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *mut c_void, *const *mut c_void, u32
        ) = core::mem::transmute(*vtbl.add(CTX_VS_SET_SHADER));
        fp(self.context, vs, std::ptr::null(), 0);
    }

    /// ID3D11DeviceContext::PSSetShader(ps, null, 0).
    unsafe fn ctx_ps_set_shader(&self, ps: *mut c_void) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *mut c_void, *const *mut c_void, u32
        ) = core::mem::transmute(*vtbl.add(CTX_PS_SET_SHADER));
        fp(self.context, ps, std::ptr::null(), 0);
    }

    /// ID3D11DeviceContext::PSSetShaderResources(0, 1, &srv).
    /// NOTE: takes an ARRAY even for count=1 — pass `&srv` as `*const *mut`.
    unsafe fn ctx_ps_set_shader_resources(&self, srv: *mut c_void) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, u32, u32, *const *mut c_void
        ) = core::mem::transmute(*vtbl.add(CTX_PS_SET_SHADER_RESOURCES));
        let srvs = [srv];
        fp(self.context, 0, 1, srvs.as_ptr());
    }

    /// ID3D11DeviceContext::PSSetSamplers(0, 1, &sampler).
    unsafe fn ctx_ps_set_samplers(&self, sampler: *mut c_void) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, u32, u32, *const *mut c_void
        ) = core::mem::transmute(*vtbl.add(CTX_PS_SET_SAMPLERS));
        let samps = [sampler];
        fp(self.context, 0, 1, samps.as_ptr());
    }

    /// ID3D11DeviceContext::OMSetBlendState(state, null factor, 0xFFFFFFFF mask).
    /// SampleMask MUST be 0xFFFFFFFF — passing 0 disables every sample => black.
    unsafe fn ctx_om_set_blend_state(&self, state: *mut c_void) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(
            *mut c_void, *mut c_void, *const f32, u32
        ) = core::mem::transmute(*vtbl.add(CTX_OM_SET_BLEND_STATE));
        fp(self.context, state, std::ptr::null(), 0xFFFF_FFFF);
    }

    /// ID3D11DeviceContext::Draw(VertexCount, StartVertexLocation).
    unsafe fn ctx_draw(&self, vertex_count: u32, start: u32) {
        let vtbl = *(self.context as *const *const usize);
        let fp: unsafe extern "system" fn(*mut c_void, u32, u32) =
            core::mem::transmute(*vtbl.add(CTX_DRAW));
        fp(self.context, vertex_count, start);
    }

    // ── ShaderQuad pipeline build / per-frame draw / TDR rebuild ─────

    /// Create the SAMPLED source texture (DEFAULT + BIND_SHADER_RESOURCE,
    /// B8G8R8A8_UNORM, no CPU access) and its full-resource SRV. These are
    /// size-dependent (rebuilt on resize/TDR). Returns `(tex, srv)`; on any
    /// failure releases whatever it created and returns the HRESULT.
    unsafe fn create_source_tex_and_srv(&self, width: u32, height: u32) -> Result<(*mut c_void, *mut c_void), i32> {
        let tex = create_sampled_texture(self.device, width, height)?;
        match self.dev_create_shader_resource_view(tex) {
            Ok(srv) => Ok((tex, srv)),
            Err(hr) => {
                com_release(tex);
                Err(hr)
            }
        }
    }

    /// Build ALL ShaderQuad pipeline objects. On ANY failure, releases every
    /// partially-built object, leaves all fields null, and returns Err so the
    /// caller degrades to CopyResource (no half-built bound state).
    ///
    /// The VS/PS are created from EMBEDDED precompiled DXBC (`QUAD_VS_DXBC` /
    /// `QUAD_PS_DXBC`) on the happy path — NO runtime D3DCompile, so NO
    /// `d3dcompiler_47.dll` dependency at runtime. The runtime
    /// `crate::hlsl::compile` FFI is invoked ONLY as a defensive fallback if
    /// creating a shader from the embedded bytes fails (e.g. a corrupt embed
    /// or a future HLSL edit that wasn't re-fxc'd); if that fallback also
    /// fails (or `d3dcompiler_47.dll` is missing), this degrades to
    /// CopyResource like any other init failure.
    fn init_shader_pipeline(&mut self) -> Result<(), HwPresentError> {
        unsafe {
            // Size-dependent: sampled source texture + SRV.
            let (source_tex, source_srv) = match self.create_source_tex_and_srv(self.width, self.height) {
                Ok(v) => v,
                Err(_) => return Err(HwPresentError::PipelineInitFailed),
            };

            // Size-independent: shaders, sampler, blend state.
            // Create the VS from embedded DXBC first; only on failure fall
            // back to runtime-compiling VS_HLSL (vs_4_0/VSMain).
            let vs = match self.dev_create_vertex_shader(QUAD_VS_DXBC) {
                Ok(v) => v,
                Err(_) => match crate::hlsl::compile(VS_HLSL, "VSMain", "vs_4_0")
                    .ok()
                    .and_then(|dxbc| self.dev_create_vertex_shader(&dxbc).ok())
                {
                    Some(v) => v,
                    None => {
                        com_release(source_srv); com_release(source_tex);
                        return Err(HwPresentError::PipelineInitFailed);
                    }
                },
            };
            // Same policy for the PS (ps_4_0/PSMain).
            let ps_tex = match self.dev_create_pixel_shader(QUAD_PS_DXBC) {
                Ok(v) => v,
                Err(_) => match crate::hlsl::compile(PS_HLSL, "PSMain", "ps_4_0")
                    .ok()
                    .and_then(|dxbc| self.dev_create_pixel_shader(&dxbc).ok())
                {
                    Some(v) => v,
                    None => {
                        com_release(vs); com_release(source_srv); com_release(source_tex);
                        return Err(HwPresentError::PipelineInitFailed);
                    }
                },
            };
            let sampler_desc = D3D11_SAMPLER_DESC {
                // POINT filter so the 1:1 quad is provably bit-exact (zero
                // epsilon golden-diff). LINEAR could introduce +/-1 LSB on
                // WARP at edges.
                Filter: D3D11_FILTER_MIN_MAG_MIP_POINT,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_NEVER,
                BorderColor: [0.0, 0.0, 0.0, 0.0],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let sampler = match self.dev_create_sampler_state(&sampler_desc) {
                Ok(v) => v,
                Err(_) => {
                    com_release(ps_tex); com_release(vs);
                    com_release(source_srv); com_release(source_tex);
                    return Err(HwPresentError::PipelineInitFailed);
                }
            };
            let blend_desc = opaque_blend_desc();
            let blend_state = match self.dev_create_blend_state(&blend_desc) {
                Ok(v) => v,
                Err(_) => {
                    com_release(sampler); com_release(ps_tex); com_release(vs);
                    com_release(source_srv); com_release(source_tex);
                    return Err(HwPresentError::PipelineInitFailed);
                }
            };

            self.source_tex = source_tex;
            self.source_srv = source_srv;
            self.vs = vs;
            self.ps_tex = ps_tex;
            self.sampler = sampler;
            self.blend_state = blend_state;
            self.pipeline_ready = true;
        }
        Ok(())
    }

    /// Draw one opaque full-screen textured quad sampling the composited
    /// `bgra` viewport into the swap back buffer. Returns:
    ///   - `Ok(())`            present succeeded;
    ///   - `Err(CopyResource)` device lost AND the single rebuild failed —
    ///     caller skips this frame and the next falls through to CopyResource.
    /// On a recoverable device loss it rebuilds once and retries the present.
    fn present_shader_quad(&mut self, bgra: &[u8], width: u32, height: u32) -> Result<(), PresentBackend> {
        unsafe {
            // Push the composited CPU viewport into the DEFAULT sampled
            // texture. SrcRowPitch is the TIGHT width*4 (NOT a staging pitch).
            self.ctx_update_subresource(
                self.source_tex,
                bgra.as_ptr() as *const c_void,
                width * 4,
            );

            // Per-frame back buffer + RTV (FLIP_DISCARD rotates the back
            // buffer every Present — caching an RTV across frames is UB).
            let back_buffer = match self.swap_get_buffer() {
                Ok(b) => b,
                Err(_) => return self.handle_device_loss_then_retry(bgra, width, height),
            };
            let rtv = match self.dev_create_render_target_view(back_buffer) {
                Ok(r) => r,
                Err(_) => {
                    com_release(back_buffer);
                    return self.handle_device_loss_then_retry(bgra, width, height);
                }
            };

            self.ctx_om_set_render_targets(rtv);
            let vp = D3D11_VIEWPORT {
                TopLeftX: 0.0, TopLeftY: 0.0,
                Width: width as f32, Height: height as f32,
                MinDepth: 0.0, MaxDepth: 1.0,
            };
            self.ctx_rs_set_viewports(&vp);
            // Belt-and-suspenders; the opaque quad overwrites every pixel.
            self.ctx_clear_render_target_view(rtv, &[0.0, 0.0, 0.0, 1.0]);
            self.ctx_ia_set_primitive_topology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            self.ctx_ia_set_input_layout_null();
            self.ctx_vs_set_shader(self.vs);
            self.ctx_ps_set_shader(self.ps_tex);
            self.ctx_ps_set_shader_resources(self.source_srv);
            self.ctx_ps_set_samplers(self.sampler);
            self.ctx_om_set_blend_state(self.blend_state);
            self.ctx_draw(4, 0);

            com_release(rtv);
            com_release(back_buffer);

            let hr = self.swap_present(1, 0);
            if is_device_loss(hr) {
                return self.handle_device_loss_then_retry(bgra, width, height);
            }
            if hr < 0 {
                // Non-device-loss present failure: degrade to CopyResource
                // for subsequent frames; skip this frame (never error).
                self.backend = PresentBackend::CopyResource;
                return Err(PresentBackend::CopyResource);
            }
            Ok(())
        }
    }

    /// On a detected device loss, attempt ONE full rebuild. On success retry
    /// the present once; on rebuild failure degrade to CopyResource and
    /// signal the caller to skip this frame.
    fn handle_device_loss_then_retry(&mut self, bgra: &[u8], width: u32, height: u32) -> Result<(), PresentBackend> {
        if self.rebuild_device().is_err() {
            self.backend = PresentBackend::CopyResource;
            self.pipeline_ready = false;
            return Err(PresentBackend::CopyResource);
        }
        // Retry exactly once. If the retry itself reports loss again, do NOT
        // recurse into another rebuild — degrade instead (one attempt per loss).
        unsafe {
            self.ctx_update_subresource(self.source_tex, bgra.as_ptr() as *const c_void, width * 4);
            let back_buffer = match self.swap_get_buffer() {
                Ok(b) => b,
                Err(_) => { self.backend = PresentBackend::CopyResource; return Err(PresentBackend::CopyResource); }
            };
            let rtv = match self.dev_create_render_target_view(back_buffer) {
                Ok(r) => r,
                Err(_) => {
                    com_release(back_buffer);
                    self.backend = PresentBackend::CopyResource;
                    return Err(PresentBackend::CopyResource);
                }
            };
            self.ctx_om_set_render_targets(rtv);
            let vp = D3D11_VIEWPORT {
                TopLeftX: 0.0, TopLeftY: 0.0,
                Width: width as f32, Height: height as f32,
                MinDepth: 0.0, MaxDepth: 1.0,
            };
            self.ctx_rs_set_viewports(&vp);
            self.ctx_clear_render_target_view(rtv, &[0.0, 0.0, 0.0, 1.0]);
            self.ctx_ia_set_primitive_topology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            self.ctx_ia_set_input_layout_null();
            self.ctx_vs_set_shader(self.vs);
            self.ctx_ps_set_shader(self.ps_tex);
            self.ctx_ps_set_shader_resources(self.source_srv);
            self.ctx_ps_set_samplers(self.sampler);
            self.ctx_om_set_blend_state(self.blend_state);
            self.ctx_draw(4, 0);
            com_release(rtv);
            com_release(back_buffer);
            let hr = self.swap_present(1, 0);
            if hr < 0 {
                self.backend = PresentBackend::CopyResource;
                return Err(PresentBackend::CopyResource);
            }
            Ok(())
        }
    }

    /// Full device rebuild after a TDR / device-loss. Releases every owned
    /// COM object (reverse create order), recreates device+context, swap
    /// chain (BEFORE the DComp visual, which references it as content),
    /// DComp tree, staging, and the ShaderQuad pipeline. Any step failing
    /// returns Err so the caller degrades to CopyResource. Never panics.
    fn rebuild_device(&mut self) -> Result<(), HwPresentError> {
        unsafe {
            // (a) Release ALL objects, reverse create order, null every field.
            com_release(self.blend_state); self.blend_state = std::ptr::null_mut();
            com_release(self.sampler);     self.sampler = std::ptr::null_mut();
            com_release(self.ps_tex);      self.ps_tex = std::ptr::null_mut();
            com_release(self.vs);          self.vs = std::ptr::null_mut();
            com_release(self.source_srv);  self.source_srv = std::ptr::null_mut();
            com_release(self.source_tex);  self.source_tex = std::ptr::null_mut();
            com_release(self.staging);     self.staging = std::ptr::null_mut();
            com_release(self.dcomp_visual);  self.dcomp_visual = std::ptr::null_mut();
            com_release(self.dcomp_target);  self.dcomp_target = std::ptr::null_mut();
            com_release(self.dcomp_device);  self.dcomp_device = std::ptr::null_mut();
            com_release(self.swap_chain);    self.swap_chain = std::ptr::null_mut();
            com_release(self.context);       self.context = std::ptr::null_mut();
            com_release(self.device);        self.device = std::ptr::null_mut();
            self.pipeline_ready = false;
        }

        // (b) device + context.
        let (device, _fl, context) = d3d11::create_device()
            .map_err(HwPresentError::D3D11DeviceFailed)?;
        self.device = device;
        self.context = context;

        // (c) factory + swap chain (BEFORE the DComp visual).
        let factory = dxgi::create_factory().map_err(HwPresentError::DxgiFactoryFailed)?;
        let swap_chain = unsafe {
            create_swap_chain_for_hwnd(factory, device, self.hwnd, self.width, self.height)
        };
        unsafe { dxgi::release(factory) };
        self.swap_chain = swap_chain.map_err(HwPresentError::SwapChainFailed)?;

        // (d) DComp device + target + visual(SetContent=swap_chain) + Commit.
        let dcomp_device = unsafe { create_dcomp_device(device) }
            .map_err(HwPresentError::DCompDeviceFailed)?;
        self.dcomp_device = dcomp_device;
        let dcomp_target = unsafe { create_dcomp_target(dcomp_device, self.hwnd) }
            .map_err(HwPresentError::DCompTargetFailed)?;
        self.dcomp_target = dcomp_target;
        let dcomp_visual = unsafe { create_dcomp_visual(dcomp_device, dcomp_target, self.swap_chain) }
            .map_err(HwPresentError::DCompVisualFailed)?;
        self.dcomp_visual = dcomp_visual;
        unsafe { dcomp_commit(dcomp_device) };

        // (e) staging.
        let staging = unsafe { create_staging_texture(self.device, self.width, self.height) }
            .map_err(HwPresentError::StagingTextureFailed)?;
        self.staging = staging;

        // (f) ShaderQuad pipeline objects.
        self.init_shader_pipeline()?;

        Ok(())
    }
}

impl Drop for HwPresenter {
    fn drop(&mut self) {
        unsafe {
            // ShaderQuad pipeline objects first (reverse create order). These
            // are null under CopyResource, and com_release tolerates null.
            com_release(self.blend_state);
            com_release(self.sampler);
            com_release(self.ps_tex);
            com_release(self.vs);
            com_release(self.source_srv);
            com_release(self.source_tex);
            // Core objects.
            com_release(self.staging);
            com_release(self.dcomp_visual);
            com_release(self.dcomp_target);
            com_release(self.dcomp_device);
            com_release(self.swap_chain);
            com_release(self.context);
            com_release(self.device);
        }
    }
}

// ── Init helpers ─────────────────────────────────────────────────────

unsafe fn create_swap_chain_for_hwnd(
    factory: *mut dxgi::IDXGIFactory2,
    device: *mut c_void,
    hwnd: *mut c_void,
    width: u32,
    height: u32,
) -> Result<*mut c_void, i32> {
    let desc = dxgi::DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: dxgi::DXGI_FORMAT_B8G8R8A8_UNORM,
        Stereo: 0,
        SampleDesc: dxgi::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        BufferUsage: dxgi::DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: 2,
        // FLIP_DISCARD swap chains bound to a DComp visual MUST use
        // DXGI_SCALING_NONE (=1). The previous literal `1` was mislabeled
        // "DXGI_SCALING_STRETCH" in a comment, but STRETCH is actually 0 and
        // is invalid for the flip model — it causes offset/stretch artifacts on
        // maximize/resize when the back-buffer size and client area transiently
        // disagree. NONE maps the back buffer 1:1 to the visual top-left.
        Scaling: dxgi::DXGI_SCALING_NONE,
        SwapEffect: dxgi::DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: dxgi::DXGI_ALPHA_MODE_IGNORE,
        Flags: 0,
    };
    let mut sc: *mut c_void = std::ptr::null_mut();
    let vt = &*(*factory).vtbl;
    let hr = (vt.CreateSwapChainForHwnd)(
        factory as *mut c_void,
        device,
        hwnd,
        &desc,
        std::ptr::null(),
        std::ptr::null_mut(),
        &mut sc,
    );
    if hr < 0 || sc.is_null() {
        return Err(hr);
    }
    Ok(sc)
}

unsafe fn create_dcomp_device(d3d11_device: *mut c_void) -> Result<*mut c_void, i32> {
    // QueryInterface the D3D11 device for IDXGIDevice.
    let mut dxgi_device: *mut c_void = std::ptr::null_mut();
    let hr = com_qi(d3d11_device, &dxgi::IID_IDXGI_DEVICE, &mut dxgi_device);
    if hr < 0 || dxgi_device.is_null() {
        return Err(hr);
    }
    let result = dcomp::create_device(dxgi_device);
    com_release(dxgi_device);
    result.map_err(|hr| hr)
}

unsafe fn create_dcomp_target(dcomp_device: *mut c_void, hwnd: *mut c_void) -> Result<*mut c_void, i32> {
    let vtbl = *(dcomp_device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void, *mut c_void, i32, *mut *mut c_void
    ) -> i32 = core::mem::transmute(*vtbl.add(DCOMP_CREATE_TARGET_FOR_HWND));
    let mut target: *mut c_void = std::ptr::null_mut();
    // topmost = TRUE (1) — our visual tree should be on top.
    let hr = fp(dcomp_device, hwnd, 1, &mut target);
    if hr < 0 || target.is_null() {
        return Err(hr);
    }
    Ok(target)
}

unsafe fn create_dcomp_visual(
    dcomp_device: *mut c_void,
    target: *mut c_void,
    swap_chain: *mut c_void,
) -> Result<*mut c_void, i32> {
    // CreateVisual
    let vtbl = *(dcomp_device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void, *mut *mut c_void
    ) -> i32 = core::mem::transmute(*vtbl.add(DCOMP_CREATE_VISUAL));
    let mut visual: *mut c_void = std::ptr::null_mut();
    let hr = fp(dcomp_device, &mut visual);
    if hr < 0 || visual.is_null() {
        return Err(hr);
    }

    // SetContent(swap_chain) on the visual.
    let vvtbl = *(visual as *const *const usize);
    let set_content: unsafe extern "system" fn(
        *mut c_void, *mut c_void
    ) -> i32 = core::mem::transmute(*vvtbl.add(VISUAL_SET_CONTENT));
    let hr = set_content(visual, swap_chain);
    if hr < 0 {
        com_release(visual);
        return Err(hr);
    }

    // SetRoot(visual) on the target.
    let tvtbl = *(target as *const *const usize);
    let set_root: unsafe extern "system" fn(
        *mut c_void, *mut c_void
    ) -> i32 = core::mem::transmute(*tvtbl.add(TARGET_SET_ROOT));
    let hr = set_root(target, visual);
    if hr < 0 {
        com_release(visual);
        return Err(hr);
    }

    Ok(visual)
}

unsafe fn dcomp_commit(dcomp_device: *mut c_void) {
    let vtbl = *(dcomp_device as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void) -> i32 =
        core::mem::transmute(*vtbl.add(DCOMP_COMMIT));
    fp(dcomp_device);
}

unsafe fn create_staging_texture(device: *mut c_void, width: u32, height: u32) -> Result<*mut c_void, i32> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: dxgi::DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: dxgi::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE,
        MiscFlags: 0,
    };
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void, *const D3D11_TEXTURE2D_DESC, *const c_void, *mut *mut c_void
    ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_TEXTURE2D));
    let mut tex: *mut c_void = std::ptr::null_mut();
    let hr = fp(device, &desc, std::ptr::null(), &mut tex);
    if hr < 0 || tex.is_null() {
        return Err(hr);
    }
    Ok(tex)
}

/// Create a SAMPLED source texture: DEFAULT usage, BIND_SHADER_RESOURCE,
/// B8G8R8A8_UNORM, no CPU access, no mips. This is the ShaderQuad draw input
/// (the CPU viewport is pushed in via UpdateSubresource each frame).
unsafe fn create_sampled_texture(device: *mut c_void, width: u32, height: u32) -> Result<*mut c_void, i32> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: dxgi::DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: dxgi::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_SHADER_RESOURCE,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void, *const D3D11_TEXTURE2D_DESC, *const c_void, *mut *mut c_void
    ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_TEXTURE2D));
    let mut tex: *mut c_void = std::ptr::null_mut();
    let hr = fp(device, &desc, std::ptr::null(), &mut tex);
    if hr < 0 || tex.is_null() {
        return Err(hr);
    }
    Ok(tex)
}

/// Create a DEFAULT + BIND_RENDER_TARGET texture (B8G8R8A8_UNORM). Used by
/// the offscreen golden-diff harness as the render target.
unsafe fn create_render_target_texture(device: *mut c_void, width: u32, height: u32) -> Result<*mut c_void, i32> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: dxgi::DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: dxgi::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_RENDER_TARGET,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void, *const D3D11_TEXTURE2D_DESC, *const c_void, *mut *mut c_void
    ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_TEXTURE2D));
    let mut tex: *mut c_void = std::ptr::null_mut();
    let hr = fp(device, &desc, std::ptr::null(), &mut tex);
    if hr < 0 || tex.is_null() {
        return Err(hr);
    }
    Ok(tex)
}

/// Create a STAGING + CPU_ACCESS_READ texture (B8G8R8A8_UNORM, BindFlags=0).
/// Used by the offscreen golden-diff harness for CopyResource readback.
unsafe fn create_readback_texture(device: *mut c_void, width: u32, height: u32) -> Result<*mut c_void, i32> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: dxgi::DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: dxgi::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ,
        MiscFlags: 0,
    };
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void, *const D3D11_TEXTURE2D_DESC, *const c_void, *mut *mut c_void
    ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_TEXTURE2D));
    let mut tex: *mut c_void = std::ptr::null_mut();
    let hr = fp(device, &desc, std::ptr::null(), &mut tex);
    if hr < 0 || tex.is_null() {
        return Err(hr);
    }
    Ok(tex)
}

/// Build the opaque/disabled blend desc. BlendEnable=0 makes the OM write the
/// PS output verbatim with no blend math; WriteMask=ALL writes every channel.
/// For the single 1:1 opaque quad this byte-matches the CPU scanout.
fn opaque_blend_desc() -> D3D11_BLEND_DESC {
    let rt0 = D3D11_RENDER_TARGET_BLEND_DESC {
        BlendEnable: 0,
        SrcBlend: D3D11_BLEND_ONE,
        DestBlend: D3D11_BLEND_ZERO,
        BlendOp: D3D11_BLEND_OP_ADD,
        SrcBlendAlpha: D3D11_BLEND_ONE,
        DestBlendAlpha: D3D11_BLEND_ZERO,
        BlendOpAlpha: D3D11_BLEND_OP_ADD,
        RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL,
    };
    D3D11_BLEND_DESC {
        AlphaToCoverageEnable: 0,
        IndependentBlendEnable: 0,
        RenderTarget: [rt0; 8],
    }
}

/// True if an HRESULT signals device loss (TDR). Compares as `i32` so the
/// sign-extended negative DXGI codes match (the u32 literal never would).
fn is_device_loss(hr: i32) -> bool {
    hr == dxgi::DXGI_ERROR_DEVICE_REMOVED as i32
        || hr == dxgi::DXGI_ERROR_DEVICE_RESET as i32
        || hr == dxgi::DXGI_ERROR_DEVICE_HUNG as i32
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal user32 FFI for the affinity test's transient HWND. A
    // message-only window (HWND_MESSAGE parent) is enough for swap-chain +
    // DComp creation in the affinity guard test; it is never shown.
    #[link(name = "user32")]
    unsafe extern "system" {
        fn CreateWindowExW(
            ex_style: u32,
            class_name: *const u16,
            window_name: *const u16,
            style: u32,
            x: i32,
            y: i32,
            w: i32,
            h: i32,
            parent: *mut c_void,
            menu: *mut c_void,
            instance: *mut c_void,
            param: *mut c_void,
        ) -> *mut c_void;
        fn DestroyWindow(hwnd: *mut c_void) -> i32;
    }

    /// Create a transient top-level window (the "STATIC" system class always
    /// exists) for the affinity test. Returns null on failure (graceful skip).
    unsafe fn create_message_only_window() -> *mut c_void {
        let class: Vec<u16> = "STATIC\0".encode_utf16().collect();
        let name: Vec<u16> = "tb-affinity-test\0".encode_utf16().collect();
        // WS_OVERLAPPED (0) hidden window — never ShowWindow'd.
        CreateWindowExW(
            0,
            class.as_ptr(),
            name.as_ptr(),
            0,
            0,
            0,
            16,
            16,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    unsafe fn destroy_window(hwnd: *mut c_void) {
        if !hwnd.is_null() {
            DestroyWindow(hwnd);
        }
    }

    #[test]
    fn hw_presenter_error_variants() {
        // Ensure error variants are distinct.
        assert_ne!(
            HwPresentError::D3D11DeviceFailed(-1),
            HwPresentError::SwapChainFailed(-1),
        );
        assert_eq!(
            HwPresentError::MapFailed(-5),
            HwPresentError::MapFailed(-5),
        );
    }

    #[test]
    fn vtable_slot_constants() {
        // Sanity-check slot indices match the documented COM layout.
        // ── IDXGISwapChain ──
        assert_eq!(SWAP_PRESENT, 8);
        assert_eq!(SWAP_GET_BUFFER, 9);
        assert_eq!(SWAP_RESIZE_BUFFERS, 13);
        // ── ID3D11Device (IUnknown 0..2 → ID3D11Device 3..) ──
        assert_eq!(DEV_CREATE_BUFFER, 3);
        assert_eq!(DEV_CREATE_TEXTURE2D, 5);
        assert_eq!(DEV_CREATE_SHADER_RESOURCE_VIEW, 7);
        assert_eq!(DEV_CREATE_RENDER_TARGET_VIEW, 9);
        assert_eq!(DEV_CREATE_INPUT_LAYOUT, 11);
        assert_eq!(DEV_CREATE_VERTEX_SHADER, 12);
        assert_eq!(DEV_CREATE_PIXEL_SHADER, 15);
        assert_eq!(DEV_CREATE_BLEND_STATE, 20);
        assert_eq!(DEV_CREATE_SAMPLER_STATE, 23);
        // ── ID3D11DeviceContext (IUnknown 0..2 → ID3D11DeviceChild 3..6 →
        //    ID3D11DeviceContext 7..) ──
        assert_eq!(CTX_PS_SET_SHADER_RESOURCES, 8);
        assert_eq!(CTX_PS_SET_SHADER, 9);
        assert_eq!(CTX_PS_SET_SAMPLERS, 10);
        assert_eq!(CTX_VS_SET_SHADER, 11);
        assert_eq!(CTX_MAP, 14);
        assert_eq!(CTX_UNMAP, 15);
        assert_eq!(CTX_DRAW, 13);
        assert_eq!(CTX_IA_SET_INPUT_LAYOUT, 17);
        assert_eq!(CTX_IA_SET_VERTEX_BUFFERS, 18);
        assert_eq!(CTX_IA_SET_INDEX_BUFFER, 19);
        assert_eq!(CTX_DRAW_INDEXED_INSTANCED, 20);
        assert_eq!(CTX_IA_SET_PRIMITIVE_TOPOLOGY, 24);
        assert_eq!(CTX_OM_SET_RENDER_TARGETS, 33);
        assert_eq!(CTX_OM_SET_BLEND_STATE, 35);
        assert_eq!(CTX_RS_SET_VIEWPORTS, 44);
        assert_eq!(CTX_COPY_RESOURCE, 47);
        assert_eq!(CTX_UPDATE_SUBRESOURCE, 48);
        assert_eq!(CTX_CLEAR_RENDER_TARGET_VIEW, 50);
        // ── IDCompositionDevice / Target / Visual ──
        assert_eq!(DCOMP_COMMIT, 3);
        assert_eq!(DCOMP_CREATE_TARGET_FOR_HWND, 6);
        assert_eq!(DCOMP_CREATE_VISUAL, 7);
        assert_eq!(TARGET_SET_ROOT, 3);
        assert_eq!(VISUAL_SET_OFFSET_X, 4);
        assert_eq!(VISUAL_SET_OFFSET_Y, 6);
        assert_eq!(VISUAL_SET_CONTENT, 15);
    }

    #[test]
    fn present_backend_default_is_shaderquad_unless_flag_zero() {
        // Flipped 2026-06-13: CV_GPU_PIPELINE now DEFAULTS ON (`!= Ok("0")`).
        // Unset/any-non-"0" → ShaderQuad (the GPU shader present path, proven
        // correct by a live-window soak + delta=0 golden-diff, degrading to
        // CopyResource at RUNTIME on any init failure). Only CV_GPU_PIPELINE=0
        // is the escape hatch → CopyResource at SELECTION time.
        // gpu_pipeline_enabled() caches via OnceLock; the normal test process
        // has the env unset, so this is the default-ON path.
        if std::env::var("CV_GPU_PIPELINE").as_deref() == Ok("0") {
            assert_eq!(select_present_backend(), PresentBackend::CopyResource);
            assert!(!gpu_pipeline_enabled());
        } else {
            assert_eq!(select_present_backend(), PresentBackend::ShaderQuad);
            assert!(gpu_pipeline_enabled());
        }
    }

    #[test]
    fn staging_texture_desc_layout() {
        // Verify our repr(C) struct matches expected field offsets.
        let desc = D3D11_TEXTURE2D_DESC {
            Width: 1920,
            Height: 1080,
            MipLevels: 1,
            ArraySize: 1,
            Format: dxgi::DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: dxgi::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE,
            MiscFlags: 0,
        };
        assert_eq!(desc.Width, 1920);
        assert_eq!(desc.Height, 1080);
        assert_eq!(desc.Format, 87); // B8G8R8A8_UNORM
        assert_eq!(desc.Usage, 3);   // STAGING
        assert_eq!(desc.CPUAccessFlags, 0x10000); // CPU_ACCESS_WRITE
    }

    #[test]
    fn mapped_subresource_is_zeroed() {
        let m = D3D11_MAPPED_SUBRESOURCE {
            pData: std::ptr::null_mut(),
            RowPitch: 0,
            DepthPitch: 0,
        };
        assert!(m.pData.is_null());
        assert_eq!(m.RowPitch, 0);
    }

    #[test]
    fn iid_id3d11_texture2d() {
        assert_eq!(IID_ID3D11_TEXTURE2D.Data1, 0x6F15AAF2);
    }

    // Integration test that creates a real D3D11 device + staging
    // texture (no HWND needed — just device + texture).
    #[test]
    fn create_staging_texture_real() {
        let (device, _fl, context) = d3d11::create_device()
            .expect("D3D11CreateDevice");
        let tex = unsafe { create_staging_texture(device, 64, 64) }
            .expect("CreateTexture2D staging");
        assert!(!tex.is_null());
        unsafe {
            com_release(tex);
            com_release(context);
            com_release(device);
        }
    }

    // ── ShaderQuad descriptor layout / constant tests ────────────────

    #[test]
    fn shaderquad_constants_match_d3d11_h() {
        assert_eq!(D3D11_BIND_SHADER_RESOURCE, 0x8);
        assert_eq!(D3D11_BIND_RENDER_TARGET, 0x20);
        assert_eq!(D3D11_BIND_VERTEX_BUFFER, 0x1);
        assert_eq!(D3D11_USAGE_DEFAULT, 0);
        assert_eq!(D3D11_CPU_ACCESS_READ, 0x20000);
        assert_eq!(D3D11_MAP_READ, 1);
        assert_eq!(D3D11_FILTER_MIN_MAG_MIP_POINT, 0);
        assert_eq!(D3D11_FILTER_MIN_MAG_MIP_LINEAR, 0x15);
        assert_eq!(D3D11_TEXTURE_ADDRESS_CLAMP, 3);
        assert_eq!(D3D11_COMPARISON_NEVER, 1);
        assert_eq!(D3D11_BLEND_ZERO, 1);
        assert_eq!(D3D11_BLEND_ONE, 2);
        assert_eq!(D3D11_BLEND_OP_ADD, 1);
        assert_eq!(D3D11_COLOR_WRITE_ENABLE_ALL, 0x0F);
        assert_eq!(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, 5);
    }

    #[test]
    fn render_target_blend_desc_layout() {
        // CRITICAL ABI: 7 u32 fields + 1 u8 + 3 pad = 32 bytes per element,
        // 4-byte aligned. A wrong layout silently mis-sets BlendEnable /
        // WriteMask. Verify size, alignment, and the WriteMask byte offset.
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<D3D11_RENDER_TARGET_BLEND_DESC>(), 32);
        assert_eq!(align_of::<D3D11_RENDER_TARGET_BLEND_DESC>(), 4);
        let d = D3D11_RENDER_TARGET_BLEND_DESC {
            BlendEnable: 0,
            SrcBlend: D3D11_BLEND_ONE,
            DestBlend: D3D11_BLEND_ZERO,
            BlendOp: D3D11_BLEND_OP_ADD,
            SrcBlendAlpha: D3D11_BLEND_ONE,
            DestBlendAlpha: D3D11_BLEND_ZERO,
            BlendOpAlpha: D3D11_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL,
        };
        let base = &d as *const _ as usize;
        assert_eq!(&d.BlendEnable as *const _ as usize - base, 0);
        assert_eq!(&d.SrcBlend as *const _ as usize - base, 4);
        assert_eq!(&d.RenderTargetWriteMask as *const _ as usize - base, 28);
    }

    #[test]
    fn blend_desc_layout() {
        // D3D11_BLEND_DESC: i32 + i32 + [32-byte; 8] = 8 + 256 = 264 bytes.
        use std::mem::size_of;
        assert_eq!(size_of::<D3D11_BLEND_DESC>(), 8 + 256);
        let d = opaque_blend_desc();
        assert_eq!(d.AlphaToCoverageEnable, 0);
        assert_eq!(d.IndependentBlendEnable, 0);
        assert_eq!(d.RenderTarget[0].BlendEnable, 0);
        assert_eq!(d.RenderTarget[0].RenderTargetWriteMask, 0x0F);
        let base = &d as *const _ as usize;
        assert_eq!(&d.RenderTarget as *const _ as usize - base, 8);
    }

    #[test]
    fn sampler_desc_layout() {
        // D3D11_SAMPLER_DESC: Filter u32 + 3 address u32 + MipLODBias f32 +
        // MaxAnisotropy u32 + ComparisonFunc u32 + BorderColor [f32;4] +
        // MinLOD f32 + MaxLOD f32 = 13 * 4 = 52 bytes.
        use std::mem::size_of;
        assert_eq!(size_of::<D3D11_SAMPLER_DESC>(), 52);
        let d = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D11_COMPARISON_NEVER,
            BorderColor: [0.0; 4],
            MinLOD: 0.0,
            MaxLOD: D3D11_FLOAT32_MAX,
        };
        let base = &d as *const _ as usize;
        assert_eq!(&d.BorderColor as *const _ as usize - base, 28);
        assert_eq!(&d.MaxLOD as *const _ as usize - base, 48);
    }

    #[test]
    fn viewport_layout() {
        use std::mem::size_of;
        assert_eq!(size_of::<D3D11_VIEWPORT>(), 24); // 6 * f32
    }

    #[test]
    fn is_device_loss_matches_negative_hresults() {
        assert!(is_device_loss(dxgi::DXGI_ERROR_DEVICE_REMOVED as i32));
        assert!(is_device_loss(dxgi::DXGI_ERROR_DEVICE_RESET as i32));
        assert!(!is_device_loss(0));      // S_OK
        assert!(!is_device_loss(-1));     // E_FAIL but not device loss
    }

    // ── The headless offscreen-RTV golden-diff GATE (M5.1 deliverable) ──
    //
    // Runs the EXACT production draw sequence into an offscreen
    // DEFAULT+RENDER_TARGET texture (no swap chain, no HWND), then reads it
    // back via a STAGING+CPU_READ copy and asserts byte-for-byte equality
    // vs the uploaded source. Skips gracefully if no D3D device exists.

    /// Build a deterministic BGRA test pattern WxH with critical probe pixels
    /// (faint-gold, semi-white, opaque-red, transparent) plus a gradient.
    fn build_test_pattern(w: u32, h: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                // Default: a deterministic gradient.
                let b = (x & 0xFF) as u8;
                let g = (y & 0xFF) as u8;
                let r = ((x + y) & 0xFF) as u8;
                let a = 0xFF;
                let (bb, gg, rr, aa) = match (x, y) {
                    // BGRA byte order: [B, G, R, A].
                    (0, 0) => (0x00, 0x00, 0xFF, 0xFF), // opaque red
                    // faint-gold particle 0x1AFFD700 (ARGB) => RGB(255,215,0) a=0x1A.
                    (1, 0) => (0x00, 0xD7, 0xFF, 0x1A),
                    (2, 0) => (0xFF, 0xFF, 0xFF, 0x80), // semi white
                    (3, 0) => (0x00, 0x00, 0x00, 0x00), // transparent
                    _ => (b, g, r, a),
                };
                buf[i] = bb;
                buf[i + 1] = gg;
                buf[i + 2] = rr;
                buf[i + 3] = aa;
            }
        }
        buf
    }

    /// Shared offscreen-pipeline runner: builds the persistent pipeline
    /// objects, draws the uploaded `src` into an offscreen RT, reads it back,
    /// and returns the readback bytes (tightly packed WxH*4). Returns None if
    /// no device is available (graceful CI skip).
    unsafe fn run_offscreen_pipeline(w: u32, h: u32, src: &[u8]) -> Option<Vec<u8>> {
        let (device, _fl, context) = match d3d11::create_device() {
            Ok(v) => v,
            Err(_) => return None, // no D3D device — skip gracefully
        };

        // Use the EMBEDDED precompiled DXBC — the production happy path. This
        // makes the golden-diff tests (shader_quad_opaque_copy_is_bit_exact,
        // faint_gold_no_drift, present_offscreen_is_thread_independent) exercise
        // the embedded shaders, NOT the runtime-D3DCompile fallback, so they
        // prove the embedded bytes render bit-exact (max diff 0).
        let vs_dxbc: &[u8] = QUAD_VS_DXBC;
        let ps_dxbc: &[u8] = QUAD_PS_DXBC;

        // Helpers reuse the same vtable idiom as production by hand here
        // (no HwPresenter instance in the headless harness).
        let dev_create_vs = |dxbc: &[u8]| -> *mut c_void {
            let vtbl = *(device as *const *const usize);
            let fp: unsafe extern "system" fn(*mut c_void, *const c_void, usize, *mut c_void, *mut *mut c_void) -> i32 =
                core::mem::transmute(*vtbl.add(DEV_CREATE_VERTEX_SHADER));
            let mut out = std::ptr::null_mut();
            fp(device, dxbc.as_ptr() as *const c_void, dxbc.len(), std::ptr::null_mut(), &mut out);
            out
        };
        let dev_create_ps = |dxbc: &[u8]| -> *mut c_void {
            let vtbl = *(device as *const *const usize);
            let fp: unsafe extern "system" fn(*mut c_void, *const c_void, usize, *mut c_void, *mut *mut c_void) -> i32 =
                core::mem::transmute(*vtbl.add(DEV_CREATE_PIXEL_SHADER));
            let mut out = std::ptr::null_mut();
            fp(device, dxbc.as_ptr() as *const c_void, dxbc.len(), std::ptr::null_mut(), &mut out);
            out
        };
        let dev_create_srv = |res: *mut c_void| -> *mut c_void {
            let vtbl = *(device as *const *const usize);
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, *mut *mut c_void) -> i32 =
                core::mem::transmute(*vtbl.add(DEV_CREATE_SHADER_RESOURCE_VIEW));
            let mut out = std::ptr::null_mut();
            fp(device, res, std::ptr::null(), &mut out);
            out
        };
        let dev_create_rtv = |res: *mut c_void| -> *mut c_void {
            let vtbl = *(device as *const *const usize);
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, *mut *mut c_void) -> i32 =
                core::mem::transmute(*vtbl.add(DEV_CREATE_RENDER_TARGET_VIEW));
            let mut out = std::ptr::null_mut();
            fp(device, res, std::ptr::null(), &mut out);
            out
        };
        let dev_create_sampler = |desc: &D3D11_SAMPLER_DESC| -> *mut c_void {
            let vtbl = *(device as *const *const usize);
            let fp: unsafe extern "system" fn(*mut c_void, *const D3D11_SAMPLER_DESC, *mut *mut c_void) -> i32 =
                core::mem::transmute(*vtbl.add(DEV_CREATE_SAMPLER_STATE));
            let mut out = std::ptr::null_mut();
            fp(device, desc as *const _, &mut out);
            out
        };
        let dev_create_blend = |desc: &D3D11_BLEND_DESC| -> *mut c_void {
            let vtbl = *(device as *const *const usize);
            let fp: unsafe extern "system" fn(*mut c_void, *const D3D11_BLEND_DESC, *mut *mut c_void) -> i32 =
                core::mem::transmute(*vtbl.add(DEV_CREATE_BLEND_STATE));
            let mut out = std::ptr::null_mut();
            fp(device, desc as *const _, &mut out);
            out
        };

        let vs = dev_create_vs(vs_dxbc);
        let ps = dev_create_ps(ps_dxbc);
        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D11_COMPARISON_NEVER,
            BorderColor: [0.0; 4],
            MinLOD: 0.0,
            MaxLOD: D3D11_FLOAT32_MAX,
        };
        let sampler = dev_create_sampler(&sampler_desc);
        let blend_desc = opaque_blend_desc();
        let blend = dev_create_blend(&blend_desc);

        let source_tex = create_sampled_texture(device, w, h).expect("sampled tex");
        let srv = dev_create_srv(source_tex);
        let rt_tex = create_render_target_texture(device, w, h).expect("rt tex");
        let rtv = dev_create_rtv(rt_tex);
        let readback = create_readback_texture(device, w, h).expect("readback tex");

        assert!(!vs.is_null() && !ps.is_null() && !sampler.is_null()
            && !blend.is_null() && !srv.is_null() && !rtv.is_null());

        // UpdateSubresource the source pattern into the sampled texture.
        {
            let vtbl = *(context as *const *const usize);
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void, *const c_void, u32, u32) =
                core::mem::transmute(*vtbl.add(CTX_UPDATE_SUBRESOURCE));
            fp(context, source_tex, 0, std::ptr::null(), src.as_ptr() as *const c_void, w * 4, 0);
        }

        // ── EXACT production draw sequence into offscreen RT ──
        let ctx_call = |slot: usize| -> usize {
            let vtbl = *(context as *const *const usize);
            *vtbl.add(slot)
        };
        // OMSetRenderTargets(1, &rtv, null).
        {
            let fp: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *mut c_void) =
                core::mem::transmute(ctx_call(CTX_OM_SET_RENDER_TARGETS));
            let rtvs = [rtv];
            fp(context, 1, rtvs.as_ptr(), std::ptr::null_mut());
        }
        // RSSetViewports(1, &vp).
        {
            let vp = D3D11_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 };
            let fp: unsafe extern "system" fn(*mut c_void, u32, *const D3D11_VIEWPORT) =
                core::mem::transmute(ctx_call(CTX_RS_SET_VIEWPORTS));
            fp(context, 1, &vp as *const _);
        }
        // ClearRenderTargetView(rtv, green) — proves the quad overwrites it.
        {
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const f32) =
                core::mem::transmute(ctx_call(CTX_CLEAR_RENDER_TARGET_VIEW));
            let c = [0.0f32, 1.0, 0.0, 1.0];
            fp(context, rtv, c.as_ptr());
        }
        // IASetPrimitiveTopology(TRIANGLESTRIP).
        {
            let fp: unsafe extern "system" fn(*mut c_void, u32) =
                core::mem::transmute(ctx_call(CTX_IA_SET_PRIMITIVE_TOPOLOGY));
            fp(context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        }
        // IASetInputLayout(null).
        {
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void) =
                core::mem::transmute(ctx_call(CTX_IA_SET_INPUT_LAYOUT));
            fp(context, std::ptr::null_mut());
        }
        // VSSetShader / PSSetShader.
        {
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const *mut c_void, u32) =
                core::mem::transmute(ctx_call(CTX_VS_SET_SHADER));
            fp(context, vs, std::ptr::null(), 0);
        }
        {
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const *mut c_void, u32) =
                core::mem::transmute(ctx_call(CTX_PS_SET_SHADER));
            fp(context, ps, std::ptr::null(), 0);
        }
        // PSSetShaderResources(0,1,&srv) / PSSetSamplers(0,1,&sampler).
        {
            let fp: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void) =
                core::mem::transmute(ctx_call(CTX_PS_SET_SHADER_RESOURCES));
            let srvs = [srv];
            fp(context, 0, 1, srvs.as_ptr());
        }
        {
            let fp: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void) =
                core::mem::transmute(ctx_call(CTX_PS_SET_SAMPLERS));
            let samps = [sampler];
            fp(context, 0, 1, samps.as_ptr());
        }
        // OMSetBlendState(opaque, null, 0xFFFFFFFF).
        {
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const f32, u32) =
                core::mem::transmute(ctx_call(CTX_OM_SET_BLEND_STATE));
            fp(context, blend, std::ptr::null(), 0xFFFF_FFFF);
        }
        // Draw(4, 0).
        {
            let fp: unsafe extern "system" fn(*mut c_void, u32, u32) =
                core::mem::transmute(ctx_call(CTX_DRAW));
            fp(context, 4, 0);
        }

        // CopyResource(readback, rt_tex) then Map(READ) and read rows.
        {
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) =
                core::mem::transmute(ctx_call(CTX_COPY_RESOURCE));
            fp(context, readback, rt_tex);
        }
        let mut out = vec![0u8; (w * h * 4) as usize];
        {
            let mut mapped = D3D11_MAPPED_SUBRESOURCE { pData: std::ptr::null_mut(), RowPitch: 0, DepthPitch: 0 };
            let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut D3D11_MAPPED_SUBRESOURCE) -> i32 =
                core::mem::transmute(ctx_call(CTX_MAP));
            let hr = fp(context, readback, 0, D3D11_MAP_READ, 0, &mut mapped);
            assert!(hr >= 0 && !mapped.pData.is_null(), "Map(READ) failed hr=0x{:08X}", hr as u32);
            let src_pitch = mapped.RowPitch as usize;
            let dst_pitch = (w * 4) as usize;
            for row in 0..h as usize {
                std::ptr::copy_nonoverlapping(
                    (mapped.pData as *const u8).add(row * src_pitch),
                    out.as_mut_ptr().add(row * dst_pitch),
                    dst_pitch,
                );
            }
            let unmap: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) =
                core::mem::transmute(ctx_call(CTX_UNMAP));
            unmap(context, readback, 0);
        }

        // Cleanup every COM object.
        com_release(readback);
        com_release(rtv);
        com_release(rt_tex);
        com_release(srv);
        com_release(source_tex);
        com_release(blend);
        com_release(sampler);
        com_release(ps);
        com_release(vs);
        com_release(context);
        com_release(device);

        Some(out)
    }

    #[test]
    fn embedded_dxbc_is_well_formed() {
        // The committed .dxbc artifacts must be present, non-empty, and begin
        // with the 'DXBC' fourcc — guards against a corrupt/empty embed.
        assert!(!QUAD_VS_DXBC.is_empty(), "embedded VS DXBC is empty");
        assert!(!QUAD_PS_DXBC.is_empty(), "embedded PS DXBC is empty");
        assert_eq!(&QUAD_VS_DXBC[0..4], b"DXBC", "VS embed missing DXBC fourcc");
        assert_eq!(&QUAD_PS_DXBC[0..4], b"DXBC", "PS embed missing DXBC fourcc");
    }

    #[test]
    fn embedded_dxbc_creates_shaders_on_device() {
        // Prove the EMBEDDED bytes create real VS/PS objects on a live device
        // WITHOUT any runtime D3DCompile. If this passes, the happy path needs
        // no d3dcompiler_47.dll. (The golden-diff tests below also run through
        // the embedded path; this isolates "embed creates" from "embed renders".)
        let (device, _fl, _context) = match d3d11::create_device() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("embedded_dxbc_creates_shaders_on_device: no D3D device, skipping");
                return;
            }
        };
        unsafe {
            let vtbl = *(device as *const *const usize);
            let create_vs: unsafe extern "system" fn(*mut c_void, *const c_void, usize, *mut c_void, *mut *mut c_void) -> i32 =
                core::mem::transmute(*vtbl.add(DEV_CREATE_VERTEX_SHADER));
            let create_ps: unsafe extern "system" fn(*mut c_void, *const c_void, usize, *mut c_void, *mut *mut c_void) -> i32 =
                core::mem::transmute(*vtbl.add(DEV_CREATE_PIXEL_SHADER));
            let mut vs: *mut c_void = std::ptr::null_mut();
            let hr_vs = create_vs(device, QUAD_VS_DXBC.as_ptr() as *const c_void, QUAD_VS_DXBC.len(), std::ptr::null_mut(), &mut vs);
            assert!(hr_vs >= 0 && !vs.is_null(), "CreateVertexShader from embedded DXBC failed: 0x{:08X}", hr_vs as u32);
            let mut ps: *mut c_void = std::ptr::null_mut();
            let hr_ps = create_ps(device, QUAD_PS_DXBC.as_ptr() as *const c_void, QUAD_PS_DXBC.len(), std::ptr::null_mut(), &mut ps);
            assert!(hr_ps >= 0 && !ps.is_null(), "CreatePixelShader from embedded DXBC failed: 0x{:08X}", hr_ps as u32);
            com_release(ps);
            com_release(vs);
            com_release(device);
        }
    }

    #[test]
    fn shader_quad_opaque_copy_is_bit_exact() {
        let (w, h) = (64u32, 64u32);
        let src = build_test_pattern(w, h);
        let out = match unsafe { run_offscreen_pipeline(w, h, &src) } {
            Some(o) => o,
            None => {
                eprintln!("shader_quad_opaque_copy_is_bit_exact: no D3D device, skipping");
                return;
            }
        };
        // EXACT byte equality: opaque 1:1 quad, point sample, _UNORM both
        // sides, blend disabled -> every input texel reproduced verbatim.
        let mut max_delta = 0u8;
        let mut mismatches = 0usize;
        for i in 0..src.len() {
            let d = src[i].abs_diff(out[i]);
            if d > max_delta { max_delta = d; }
            if d != 0 { mismatches += 1; }
        }
        assert_eq!(max_delta, 0, "max per-channel delta {} ({} mismatched bytes); first byte src={:02X} out={:02X}",
            max_delta, mismatches, src[0], out[0]);
    }

    #[test]
    fn faint_gold_no_drift() {
        // Targeted regression: the faint-gold particle 0x1AFFD700 (RGB
        // 255,215,0 at a=0x1A) MUST survive byte-exact. A regression to a
        // _SRGB RTV, premultiply-on-upload, or alpha-aware blend would shift
        // these bytes and fail here.
        let (w, h) = (8u32, 8u32);
        let src = build_test_pattern(w, h);
        let out = match unsafe { run_offscreen_pipeline(w, h, &src) } {
            Some(o) => o,
            None => {
                eprintln!("faint_gold_no_drift: no D3D device, skipping");
                return;
            }
        };
        // Probe pixel (1,0) — BGRA [0x00, 0xD7, 0xFF, 0x1A].
        let i = ((0 * w + 1) * 4) as usize;
        assert_eq!(&out[i..i + 4], &[0x00, 0xD7, 0xFF, 0x1A],
            "faint-gold drifted: got {:02X?}", &out[i..i + 4]);
        // Opaque red (0,0) — BGRA [0x00,0x00,0xFF,0xFF].
        assert_eq!(&out[0..4], &[0x00, 0x00, 0xFF, 0xFF]);
        // Transparent (3,0) — BGRA all zero (straight-alpha, no premultiply).
        let t = ((0 * w + 3) * 4) as usize;
        assert_eq!(&out[t..t + 4], &[0x00, 0x00, 0x00, 0x00]);
    }

    // ── M5.5 oracle layer (2): present byte-identity regardless of the
    //     driving thread. The off-main compositor drives `run_offscreen_pipeline`
    //     equivalent (the GPU draw of a given buffer) from the compositor thread
    //     instead of the UI thread. This proves a buffer drawn on thread A is
    //     byte-for-byte identical to the same buffer drawn on thread B — so
    //     relocating present off the UI thread cannot change a single pixel.
    #[test]
    fn present_offscreen_is_thread_independent() {
        let (w, h) = (64u32, 64u32);
        let src = build_test_pattern(w, h);
        // Drive once on the test (UI-proxy) thread.
        let readback_a = match unsafe { run_offscreen_pipeline(w, h, &src) } {
            Some(o) => o,
            None => {
                eprintln!("present_offscreen_is_thread_independent: no D3D device, skipping");
                return;
            }
        };
        // Drive again inside a freshly-spawned thread (the compositor-thread proxy).
        let src_b = src.clone();
        let readback_b = std::thread::spawn(move || unsafe {
            run_offscreen_pipeline(w, h, &src_b)
        })
        .join()
        .expect("offscreen pipeline thread panicked");
        let readback_b = match readback_b {
            Some(o) => o,
            None => {
                // Device available on the main thread but not the spawned one is
                // not expected, but treat as a graceful skip rather than a fail.
                eprintln!("present_offscreen_is_thread_independent: spawned thread had no device, skipping");
                return;
            }
        };
        assert_eq!(
            readback_a.len(),
            readback_b.len(),
            "readback lengths differ across threads"
        );
        let mut max_delta = 0u8;
        for i in 0..readback_a.len() {
            let d = readback_a[i].abs_diff(readback_b[i]);
            if d > max_delta {
                max_delta = d;
            }
        }
        assert_eq!(
            max_delta, 0,
            "GPU present of the same buffer differed across threads (max delta {})",
            max_delta
        );
    }

    // ── M5.5 oracle layer (3b): the creator-thread affinity guard. The COM
    //     funnel (`present_bgra`/`resize`) calls `check_affinity(cur, creator)`.
    //     `HwPresenter` is correctly `!Send`, so we CANNOT move one across
    //     threads to drive a real cross-thread present (that would be a compile
    //     error — and is itself the first line of defense). Instead we drive the
    //     extracted pure check directly with a deliberate tid mismatch under the
    //     hard-assert (audit) mode, which is exactly the panic a stray
    //     cross-thread COM call would trigger.

    /// A matching (cur == creator) tid must NOT panic, even under the audit env.
    #[test]
    fn affinity_guard_same_thread_ok() {
        // SAFETY: single-threaded test setup.
        unsafe { std::env::set_var("CV_OFFMAIN_COMPOSITOR_AUDIT", "1") };
        let me = unsafe { GetCurrentThreadId() };
        check_affinity(me, me); // must not panic
        unsafe { std::env::remove_var("CV_OFFMAIN_COMPOSITOR_AUDIT") };
    }

    /// A tid mismatch (the stray cross-thread COM call) MUST panic under the
    /// hard-assert audit mode. This is the affinity gate firing.
    #[test]
    fn affinity_guard_cross_thread_panics() {
        // SAFETY: single-threaded test setup before the catch.
        unsafe { std::env::set_var("CV_OFFMAIN_COMPOSITOR_AUDIT", "1") };
        let result = std::panic::catch_unwind(|| {
            // creator=1, current=2 → mismatch → hard assert under audit.
            check_affinity(2, 1);
        });
        unsafe { std::env::remove_var("CV_OFFMAIN_COMPOSITOR_AUDIT") };
        assert!(
            result.is_err(),
            "affinity guard must panic on a creator/current thread-id mismatch"
        );
    }

    /// End-to-end (device-required, graceful skip): an HwPresenter built on this
    /// thread records THIS thread's id as creator and presents fine here. Proves
    /// the construct-on-thread path records the right id and the same-thread COM
    /// call passes the guard. (Cross-thread is impossible by `!Send` + covered
    /// by the pure-check test above.)
    #[test]
    fn presenter_same_thread_present_ok() {
        let hwnd = unsafe { create_message_only_window() };
        if hwnd.is_null() {
            eprintln!("presenter_same_thread_present_ok: no HWND, skipping");
            return;
        }
        let mut presenter = match HwPresenter::new(hwnd, 16, 16) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("presenter_same_thread_present_ok: no device, skipping");
                unsafe { destroy_window(hwnd) };
                return;
            }
        };
        assert_eq!(
            presenter.creator_thread_id(),
            unsafe { GetCurrentThreadId() },
            "creator_tid must be the constructing thread's id"
        );
        // Same-thread present must not trip the guard (result may Err if the
        // headless swap chain can't present, but it must not PANIC on affinity).
        let buf = vec![0xFF00FF00u32; 16 * 16];
        let _ = presenter.present_u32(&buf, 16, 16);
        unsafe { destroy_window(hwnd) };
    }
}
