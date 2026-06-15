//! Per-quad GPU rasterization — the Chrome cc/viz `DrawQuad` model.
//!
//! `HwPresenter`'s ShaderQuad path draws ONE full-screen textured quad to
//! scan out the already-composited CPU bitmap (a 1:1 present). This module is
//! the next layer: a real per-quad GPU DRAW pipeline that rasterizes the
//! individual draw primitives — solid-color rects, linear gradients, and
//! image (textured) quads — into a GPU render target via a vertex+pixel
//! shader, exactly as viz's renderer draws `SolidColorDrawQuad` /
//! `TextureDrawQuad` / gradient render passes (each quad's rect positioned in
//! target space by the VS, the color produced by the PS, composited with
//! `SkBlendMode::kSrcOver` straight-alpha source-over).
//!
//! ## Byte-exact golden gate
//!
//! The CPU rasterizer (`cv_gfx`) stays the oracle + fallback FOREVER. Every
//! quad drawn here is golden-diff gated against the CPU output:
//!   * solid opaque + image-copy quads are BYTE-IDENTICAL (max delta 0);
//!   * gradient + semi-transparent blends are within a tight 1-LSB tolerance
//!     (GPU `uv` interpolation can land a fraction off the CPU's integer
//!     pixel-center on a boundary; the blend math itself is reproduced
//!     exactly in `draw_quad_ps.hlsl`).
//!
//! ## Why the blend is IN-SHADER, not fixed-function
//!
//! `cv_gfx::blend_bgra` does straight-alpha (non-premultiplied) source-over,
//! UN-premultiplying by the output alpha and `round()`-ing each channel. The
//! D3D11 output-merger blend is premultiplied fixed-point and would drift by
//! +/-1 LSB. So the OM blend stays DISABLED (verbatim write) and the pixel
//! shader reads the backdrop from a second SRV and composites with the exact
//! same float math. This is viz's backdrop-readback technique.
//!
//! ## Flag
//!
//! Gated behind `CV_GPU_RASTER`, **default OFF** (a new GPU draw path must not
//! risk regressing the default present). `quad_raster_enabled()` reads it once.

#![allow(non_snake_case, non_camel_case_types, dead_code, unsafe_op_in_unsafe_fn)]

use crate::d3d11;
use std::ffi::c_void;

// ── Flag ─────────────────────────────────────────────────────────────

/// Read `CV_GPU_RASTER` once. **Default OFF**: only `=1`/`on`/`true`/`yes`
/// enables the per-quad GPU raster path. Any other value (incl. unset) → OFF,
/// so the default present path is untouched.
pub fn quad_raster_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("CV_GPU_RASTER").as_deref(),
            Ok("1") | Ok("on") | Ok("true") | Ok("yes")
        )
    })
}

// ── Public draw model (mirrors viz DrawQuad materials) ───────────────

/// Straight-alpha RGBA color, 0..255 per channel — matches `cv_gfx::Color`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
    /// Pack into a BGRA u32 (the `cv_gfx::Bitmap::pixels` layout): A<<24 |
    /// R<<16 | G<<8 | B.
    pub const fn to_bgra_u32(self) -> u32 {
        ((self.a as u32) << 24) | ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
    }
}

/// What a quad paints — mirrors viz's `DrawQuad::Material` (SolidColor,
/// TextureContent, and gradient render passes).
#[derive(Debug, Clone)]
pub enum QuadFill {
    /// Uniform straight-alpha color, source-over the backdrop.
    Solid(Rgba),
    /// Two-stop linear gradient along `angle_deg` (CSS convention: 0 = up,
    /// 90 = right), reproducing `cv_gfx::fill_rect_gradient` (per-pixel axis
    /// projection, truncating channel lerp, then source-over).
    LinearGradient {
        angle_deg: f32,
        from: Rgba,
        to: Rgba,
    },
    /// Textured quad: a tightly-packed BGRA (u32-per-pixel) source the size of
    /// the quad rect, sampled 1:1 and source-over. Mirrors `TextureDrawQuad`.
    Image { bgra: Vec<u32> },
}

/// One GPU draw quad: a device-pixel rect + a fill. `x`/`y` may be negative or
/// extend past the viewport; the rasterizer clips to the target (matching
/// `cv_gfx::fill_rect`'s clamp).
#[derive(Debug, Clone)]
pub struct GpuQuad {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub fill: QuadFill,
}

// ── Embedded precompiled DXBC (no runtime d3dcompiler dependency) ─────
//
// Produced offline from src/shaders/draw_quad_vs.hlsl / draw_quad_ps.hlsl
// with the Windows SDK fxc.exe (vs_4_0 / ps_4_0). Regenerate (from
// crates/cv_gpu/src/shaders/):
//   fxc /nologo /T vs_4_0 /E VSMain /Fo draw_quad_vs.dxbc draw_quad_vs.hlsl
//   fxc /nologo /T ps_4_0 /E PSMain /Fo draw_quad_ps.dxbc draw_quad_ps.hlsl
pub(crate) const DRAW_QUAD_VS_DXBC: &[u8] = include_bytes!("shaders/draw_quad_vs.dxbc");
pub(crate) const DRAW_QUAD_PS_DXBC: &[u8] = include_bytes!("shaders/draw_quad_ps.dxbc");

// ── D3D11 constants (transcribed from d3d11.h; asserted in tests) ─────

const D3D11_USAGE_DEFAULT: u32 = 0;
const D3D11_USAGE_DYNAMIC: u32 = 2;
const D3D11_BIND_CONSTANT_BUFFER: u32 = 0x4;
const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
const D3D11_BIND_RENDER_TARGET: u32 = 0x20;
const D3D11_CPU_ACCESS_WRITE: u32 = 0x10000;
const D3D11_CPU_ACCESS_READ: u32 = 0x20000;
const D3D11_USAGE_STAGING: u32 = 3;
const D3D11_MAP_WRITE_DISCARD: u32 = 4;
const D3D11_MAP_READ: u32 = 1;
const D3D11_FILTER_MIN_MAG_MIP_POINT: u32 = 0;
const D3D11_TEXTURE_ADDRESS_CLAMP: u32 = 3;
const D3D11_COMPARISON_NEVER: u32 = 1;
const D3D11_FLOAT32_MAX: f32 = 3.402823466e+38_f32;
const D3D11_BLEND_ZERO: u32 = 1;
const D3D11_BLEND_ONE: u32 = 2;
const D3D11_BLEND_OP_ADD: u32 = 1;
const D3D11_COLOR_WRITE_ENABLE_ALL: u8 = 0x0F;
const D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: u32 = 5;
const DXGI_FORMAT_B8G8R8A8_UNORM: u32 = 87;

// ── vtable slot indices (see hw_present.rs for the full annotated list) ──

const DEV_CREATE_BUFFER: usize = 3;
const DEV_CREATE_TEXTURE2D: usize = 5;
const DEV_CREATE_SHADER_RESOURCE_VIEW: usize = 7;
const DEV_CREATE_RENDER_TARGET_VIEW: usize = 9;
const DEV_CREATE_VERTEX_SHADER: usize = 12;
const DEV_CREATE_PIXEL_SHADER: usize = 15;
const DEV_CREATE_BLEND_STATE: usize = 20;
const DEV_CREATE_SAMPLER_STATE: usize = 23;

const CTX_VS_SET_CONSTANT_BUFFERS: usize = 7;
const CTX_PS_SET_SHADER_RESOURCES: usize = 8;
const CTX_PS_SET_SHADER: usize = 9;
const CTX_PS_SET_SAMPLERS: usize = 10;
const CTX_VS_SET_SHADER: usize = 11;
const CTX_DRAW: usize = 13;
const CTX_MAP: usize = 14;
const CTX_UNMAP: usize = 15;
const CTX_PS_SET_CONSTANT_BUFFERS: usize = 16;
const CTX_IA_SET_INPUT_LAYOUT: usize = 17;
const CTX_IA_SET_PRIMITIVE_TOPOLOGY: usize = 24;
const CTX_OM_SET_RENDER_TARGETS: usize = 33;
const CTX_OM_SET_BLEND_STATE: usize = 35;
const CTX_RS_SET_VIEWPORTS: usize = 44;
const CTX_COPY_RESOURCE: usize = 47;
const CTX_UPDATE_SUBRESOURCE: usize = 48;
const CTX_CLEAR_RENDER_TARGET_VIEW: usize = 50;

// ── repr(C) descriptors ──────────────────────────────────────────────

#[repr(C)]
struct D3D11_TEXTURE2D_DESC {
    Width: u32,
    Height: u32,
    MipLevels: u32,
    ArraySize: u32,
    Format: u32,
    SampleDescCount: u32,
    SampleDescQuality: u32,
    Usage: u32,
    BindFlags: u32,
    CPUAccessFlags: u32,
    MiscFlags: u32,
}

#[repr(C)]
struct D3D11_BUFFER_DESC {
    ByteWidth: u32,
    Usage: u32,
    BindFlags: u32,
    CPUAccessFlags: u32,
    MiscFlags: u32,
    StructureByteStride: u32,
}

#[repr(C)]
struct D3D11_SUBRESOURCE_DATA {
    pSysMem: *const c_void,
    SysMemPitch: u32,
    SysMemSlicePitch: u32,
}

#[repr(C)]
struct D3D11_MAPPED_SUBRESOURCE {
    pData: *mut c_void,
    RowPitch: u32,
    DepthPitch: u32,
}

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

// ── GPU-side constant-buffer layouts (16-byte float4 aligned) ─────────

/// Matches `cbuffer QuadCB` in draw_quad_vs.hlsl.
#[repr(C)]
#[derive(Clone, Copy)]
struct QuadVsCb {
    rect: [f32; 4],     // x, y, w, h
    viewport: [f32; 4], // vp_w, vp_h, 0, 0
}

/// Matches `cbuffer QuadPS` in draw_quad_ps.hlsl.
#[repr(C)]
#[derive(Clone, Copy)]
struct QuadPsCb {
    solid: [f32; 4],
    grad_from: [f32; 4],
    grad_to: [f32; 4],
    grad_axis: [f32; 4], // dx, dy, t_min, denom
    params: [f32; 4],    // kind, rect_w, rect_h, 0
    vp2: [f32; 4],       // vp_w, vp_h, 0, 0
}

// ── Errors ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuadDrawError {
    DeviceFailed(i32),
    CreateFailed(&'static str, i32),
    MapFailed(i32),
}

// ── The per-quad GPU drawer ──────────────────────────────────────────

/// Owns the persistent per-quad pipeline objects (shaders, sampler, blend
/// state, constant buffers) and renders `GpuQuad`s into a caller-provided
/// render target. Created once, reused across frames. All COM pointers are
/// released on `Drop`.
pub struct QuadDrawer {
    device: *mut c_void,
    context: *mut c_void,
    owns_device: bool,
    vs: *mut c_void,
    ps: *mut c_void,
    sampler: *mut c_void,
    blend_state: *mut c_void, // BlendEnable=0 verbatim write (blend is in-shader)
    vs_cb: *mut c_void,       // DYNAMIC QuadVsCb
    ps_cb: *mut c_void,       // DYNAMIC QuadPsCb
}

impl QuadDrawer {
    /// Build a drawer on a fresh D3D11 device (the standalone/offscreen path).
    /// Returns `Err` if no device is available or any Create* call fails.
    pub fn new() -> Result<Self, QuadDrawError> {
        let (device, _fl, context) =
            d3d11::create_device().map_err(QuadDrawError::DeviceFailed)?;
        // SAFETY: device/context just created and non-null on Ok.
        match unsafe { Self::build(device, context, true) } {
            Ok(s) => Ok(s),
            Err(e) => {
                unsafe {
                    com_release(context);
                    com_release(device);
                }
                Err(e)
            }
        }
    }

    /// Build a drawer on an EXISTING device + immediate context (shares the
    /// presenter's device). The caller retains ownership of device/context;
    /// this drawer will NOT release them on Drop.
    ///
    /// # Safety
    /// `device`/`context` must be a valid D3D11 device + its immediate context,
    /// alive for this drawer's lifetime, and only used on the device's thread.
    pub unsafe fn from_device(
        device: *mut c_void,
        context: *mut c_void,
    ) -> Result<Self, QuadDrawError> {
        Self::build(device, context, false)
    }

    unsafe fn build(
        device: *mut c_void,
        context: *mut c_void,
        owns_device: bool,
    ) -> Result<Self, QuadDrawError> {
        let vs = dev_create_vertex_shader(device, DRAW_QUAD_VS_DXBC)
            .map_err(|hr| QuadDrawError::CreateFailed("VertexShader", hr))?;
        let ps = dev_create_pixel_shader(device, DRAW_QUAD_PS_DXBC).map_err(|hr| {
            com_release(vs);
            QuadDrawError::CreateFailed("PixelShader", hr)
        })?;
        let sampler = dev_create_sampler_point_clamp(device).map_err(|hr| {
            com_release(ps);
            com_release(vs);
            QuadDrawError::CreateFailed("SamplerState", hr)
        })?;
        let blend_state = dev_create_blend_verbatim(device).map_err(|hr| {
            com_release(sampler);
            com_release(ps);
            com_release(vs);
            QuadDrawError::CreateFailed("BlendState", hr)
        })?;
        let vs_cb = dev_create_dynamic_cb(device, core::mem::size_of::<QuadVsCb>() as u32)
            .map_err(|hr| {
                com_release(blend_state);
                com_release(sampler);
                com_release(ps);
                com_release(vs);
                QuadDrawError::CreateFailed("VsConstantBuffer", hr)
            })?;
        let ps_cb = dev_create_dynamic_cb(device, core::mem::size_of::<QuadPsCb>() as u32)
            .map_err(|hr| {
                com_release(vs_cb);
                com_release(blend_state);
                com_release(sampler);
                com_release(ps);
                com_release(vs);
                QuadDrawError::CreateFailed("PsConstantBuffer", hr)
            })?;
        Ok(Self {
            device,
            context,
            owns_device,
            vs,
            ps,
            sampler,
            blend_state,
            vs_cb,
            ps_cb,
        })
    }

    /// Draw `quads` IN ORDER over `backdrop` (a tightly-packed BGRA u32 buffer
    /// of `vp_w * vp_h`), returning the composited frame as a fresh BGRA u32
    /// buffer (`vp_w * vp_h`). This is the offscreen / golden-diff entry: each
    /// quad is composited over the running result exactly as the CPU
    /// rasterizer would, but on the GPU via the VS/PS pipeline.
    ///
    /// Internally ping-pongs two RT/SRV textures: the PS samples the backdrop
    /// from the SRV of the previous state and writes the new state into the RT,
    /// so multiple overlapping quads compose correctly without reading a bound
    /// RTV (which is undefined in D3D11).
    pub fn draw_quads_offscreen(
        &self,
        vp_w: u32,
        vp_h: u32,
        backdrop: &[u32],
        quads: &[GpuQuad],
    ) -> Result<Vec<u32>, QuadDrawError> {
        assert_eq!(
            backdrop.len(),
            (vp_w * vp_h) as usize,
            "backdrop must be vp_w*vp_h pixels"
        );
        unsafe { self.draw_quads_impl(vp_w, vp_h, backdrop, quads) }
    }

    unsafe fn draw_quads_impl(
        &self,
        vp_w: u32,
        vp_h: u32,
        backdrop: &[u32],
        quads: &[GpuQuad],
    ) -> Result<Vec<u32>, QuadDrawError> {
        // Two ping-pong RT textures (DEFAULT, RENDER_TARGET|SHADER_RESOURCE),
        // each with an RTV + SRV, plus a readback staging texture.
        let tex_a = create_rt_srv_texture(self.device, vp_w, vp_h)
            .map_err(|hr| QuadDrawError::CreateFailed("PingTexA", hr))?;
        let tex_b = create_rt_srv_texture(self.device, vp_w, vp_h).map_err(|hr| {
            com_release(tex_a);
            QuadDrawError::CreateFailed("PingTexB", hr)
        })?;
        let rtv_a = dev_create_rtv(self.device, tex_a).map_err(|hr| {
            com_release(tex_b);
            com_release(tex_a);
            QuadDrawError::CreateFailed("RtvA", hr)
        })?;
        let rtv_b = dev_create_rtv(self.device, tex_b).map_err(|hr| {
            com_release(rtv_a);
            com_release(tex_b);
            com_release(tex_a);
            QuadDrawError::CreateFailed("RtvB", hr)
        })?;
        let srv_a = dev_create_srv(self.device, tex_a).map_err(|hr| {
            com_release(rtv_b);
            com_release(rtv_a);
            com_release(tex_b);
            com_release(tex_a);
            QuadDrawError::CreateFailed("SrvA", hr)
        })?;
        let srv_b = dev_create_srv(self.device, tex_b).map_err(|hr| {
            com_release(srv_a);
            com_release(rtv_b);
            com_release(rtv_a);
            com_release(tex_b);
            com_release(tex_a);
            QuadDrawError::CreateFailed("SrvB", hr)
        })?;
        let readback = create_readback_texture(self.device, vp_w, vp_h).map_err(|hr| {
            com_release(srv_b);
            com_release(srv_a);
            com_release(rtv_b);
            com_release(rtv_a);
            com_release(tex_b);
            com_release(tex_a);
            QuadDrawError::CreateFailed("Readback", hr)
        })?;

        // Seed tex_a with the backdrop.
        ctx_update_subresource(
            self.context,
            tex_a,
            backdrop.as_ptr() as *const c_void,
            vp_w * 4,
        );

        // Bind the size-independent pipeline state once.
        ctx_ia_set_primitive_topology(self.context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        ctx_ia_set_input_layout_null(self.context);
        ctx_vs_set_shader(self.context, self.vs);
        ctx_ps_set_shader(self.context, self.ps);
        ctx_ps_set_samplers(self.context, self.sampler);
        ctx_om_set_blend_state(self.context, self.blend_state);
        ctx_vs_set_constant_buffers(self.context, self.vs_cb);
        ctx_ps_set_constant_buffers(self.context, self.ps_cb);
        let vp = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: vp_w as f32,
            Height: vp_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        ctx_rs_set_viewports(self.context, &vp);

        // Ping-pong: `src_is_a` true => the current composite lives in tex_a
        // (read via srv_a), and we write the post-quad composite into tex_b
        // (rtv_b); false => the roles flip. The backdrop is seeded in tex_a, so
        // tex_a is the source for the first quad.
        let mut src_is_a = true;

        let mut result: Vec<u32> = Vec::new();
        let mut err: Option<QuadDrawError> = None;

        'frame: {
            for q in quads {
                let (src_tex, src_srv, dst_tex, dst_rtv) = if src_is_a {
                    (tex_a, srv_a, tex_b, rtv_b)
                } else {
                    (tex_b, srv_b, tex_a, rtv_a)
                };

                // Update the VS + PS constant buffers for this quad. For image
                // quads this also creates the source texture + SRV (released
                // below via `free_image`).
                let (ps_cb_val, image_tex, image_srv) =
                    match self.prepare_quad_cbs(q, vp_w, vp_h) {
                        Ok(v) => v,
                        Err(e) => {
                            err = Some(e);
                            break 'frame;
                        }
                    };
                let vs_cb_val = QuadVsCb {
                    rect: [q.x as f32, q.y as f32, q.w as f32, q.h as f32],
                    viewport: [vp_w as f32, vp_h as f32, 0.0, 0.0],
                };
                if let Err(e) = self.update_cb(self.vs_cb, &vs_cb_val) {
                    err = Some(e);
                    free_image(image_tex, image_srv);
                    break 'frame;
                }
                if let Err(e) = self.update_cb(self.ps_cb, &ps_cb_val) {
                    err = Some(e);
                    free_image(image_tex, image_srv);
                    break 'frame;
                }

                // PS SRVs: t0 = image source (a dummy = src_srv when not an
                // image; the shader ignores t0 unless kind==2), t1 = backdrop.
                let t0 = if image_srv.is_null() { src_srv } else { image_srv };
                ctx_ps_set_shader_resources2(self.context, t0, src_srv);

                // Copy the running composite into dst FIRST so pixels OUTSIDE
                // this quad's rect are preserved (the draw only touches the
                // quad's clip-space rect; the rest of dst must equal src).
                ctx_copy_resource(self.context, dst_tex, src_tex);

                // Target dst and draw the quad over it. The PS samples the
                // backdrop from src_srv (t1) so the composite is correct even
                // where this quad overlaps previously-drawn pixels.
                ctx_om_set_render_targets(self.context, dst_rtv);
                ctx_draw(self.context, 4, 0);

                free_image(image_tex, image_srv);

                // Unbind the dst RTV before it becomes next iteration's SRV
                // (a texture cannot be bound as RTV and SRV simultaneously).
                ctx_om_set_render_targets_null(self.context);

                src_is_a = !src_is_a;
            }

            // The final composite lives in whichever tex `src_is_a` now points
            // at (after the last swap, src is the most-recently-written dst).
            let final_tex = if src_is_a { tex_a } else { tex_b };
            result = read_back(self.context, final_tex, readback, vp_w, vp_h);
            break 'frame;
        }

        // Cleanup all per-call objects.
        com_release(readback);
        com_release(srv_b);
        com_release(srv_a);
        com_release(rtv_b);
        com_release(rtv_a);
        com_release(tex_b);
        com_release(tex_a);

        if let Some(e) = err {
            return Err(e);
        }
        Ok(result)
    }

    /// Build the PS constant buffer for a quad and, for image quads, create the
    /// source texture + SRV (returned so the caller releases them post-draw).
    unsafe fn prepare_quad_cbs(
        &self,
        q: &GpuQuad,
        vp_w: u32,
        vp_h: u32,
    ) -> Result<(QuadPsCb, *mut c_void, *mut c_void), QuadDrawError> {
        let mut cb = QuadPsCb {
            solid: [0.0; 4],
            grad_from: [0.0; 4],
            grad_to: [0.0; 4],
            grad_axis: [0.0; 4],
            params: [0.0, q.w as f32, q.h as f32, 0.0],
            vp2: [vp_w as f32, vp_h as f32, 0.0, 0.0],
        };
        match &q.fill {
            QuadFill::Solid(c) => {
                cb.params[0] = 0.0;
                cb.solid = [c.r as f32, c.g as f32, c.b as f32, c.a as f32];
                Ok((cb, std::ptr::null_mut(), std::ptr::null_mut()))
            }
            QuadFill::LinearGradient {
                angle_deg,
                from,
                to,
            } => {
                cb.params[0] = 1.0;
                cb.grad_from = [from.r as f32, from.g as f32, from.b as f32, from.a as f32];
                cb.grad_to = [to.r as f32, to.g as f32, to.b as f32, to.a as f32];
                let (dx, dy, t_min, denom) = gradient_axis(*angle_deg, q.w, q.h);
                cb.grad_axis = [dx, dy, t_min, denom];
                Ok((cb, std::ptr::null_mut(), std::ptr::null_mut()))
            }
            QuadFill::Image { bgra } => {
                cb.params[0] = 2.0;
                let n = (q.w.max(0) as usize) * (q.h.max(0) as usize);
                assert_eq!(
                    bgra.len(),
                    n,
                    "image fill must be w*h pixels (got {} want {})",
                    bgra.len(),
                    n
                );
                let tex = create_sampled_texture(self.device, q.w as u32, q.h as u32)
                    .map_err(|hr| QuadDrawError::CreateFailed("ImageTex", hr))?;
                ctx_update_subresource(
                    self.context,
                    tex,
                    bgra.as_ptr() as *const c_void,
                    (q.w as u32) * 4,
                );
                let srv = match dev_create_srv(self.device, tex) {
                    Ok(s) => s,
                    Err(hr) => {
                        com_release(tex);
                        return Err(QuadDrawError::CreateFailed("ImageSrv", hr));
                    }
                };
                Ok((cb, tex, srv))
            }
        }
    }

    unsafe fn update_cb<T: Copy>(&self, cb: *mut c_void, val: &T) -> Result<(), QuadDrawError> {
        let mut mapped = D3D11_MAPPED_SUBRESOURCE {
            pData: std::ptr::null_mut(),
            RowPitch: 0,
            DepthPitch: 0,
        };
        let hr = ctx_map_write_discard(self.context, cb, &mut mapped);
        if hr < 0 || mapped.pData.is_null() {
            return Err(QuadDrawError::MapFailed(hr));
        }
        std::ptr::copy_nonoverlapping(
            val as *const T as *const u8,
            mapped.pData as *mut u8,
            core::mem::size_of::<T>(),
        );
        ctx_unmap(self.context, cb);
        Ok(())
    }
}

impl Drop for QuadDrawer {
    fn drop(&mut self) {
        unsafe {
            com_release(self.ps_cb);
            com_release(self.vs_cb);
            com_release(self.blend_state);
            com_release(self.sampler);
            com_release(self.ps);
            com_release(self.vs);
            if self.owns_device {
                com_release(self.context);
                com_release(self.device);
            }
        }
    }
}

unsafe fn free_image(tex: *mut c_void, srv: *mut c_void) {
    com_release(srv);
    com_release(tex);
}

/// Read a texture back to a tightly-packed BGRA u32 buffer via a staging copy.
unsafe fn read_back(
    context: *mut c_void,
    tex: *mut c_void,
    readback: *mut c_void,
    w: u32,
    h: u32,
) -> Vec<u32> {
    ctx_copy_resource(context, readback, tex);
    let mut out = vec![0u32; (w * h) as usize];
    let mut mapped = D3D11_MAPPED_SUBRESOURCE {
        pData: std::ptr::null_mut(),
        RowPitch: 0,
        DepthPitch: 0,
    };
    let hr = ctx_map_read(context, readback, &mut mapped);
    if hr >= 0 && !mapped.pData.is_null() {
        let src_pitch = mapped.RowPitch as usize;
        let dst_pitch = (w * 4) as usize;
        for row in 0..h as usize {
            std::ptr::copy_nonoverlapping(
                (mapped.pData as *const u8).add(row * src_pitch),
                out.as_mut_ptr().cast::<u8>().add(row * dst_pitch),
                dst_pitch,
            );
        }
        ctx_unmap(context, readback);
    }
    out
}

/// Compute the gradient axis params for `cv_gfx::fill_rect_gradient`:
/// returns (dx, dy, t_min, denom) so a pixel's t = ((px*dx + py*dy) - t_min)
/// / denom. `px`/`py` are the box-local pixel centers.
fn gradient_axis(angle_deg: f32, w: i32, h: i32) -> (f32, f32, f32, f32) {
    let theta = (angle_deg - 90.0).to_radians();
    let dx = theta.cos();
    let dy = theta.sin();
    let corners = [
        (0.0_f32, 0.0_f32),
        (w as f32, 0.0_f32),
        (0.0_f32, h as f32),
        (w as f32, h as f32),
    ];
    let mut t_min = f32::INFINITY;
    let mut t_max = f32::NEG_INFINITY;
    for (cx, cy) in corners {
        let t = cx * dx + cy * dy;
        if t < t_min {
            t_min = t;
        }
        if t > t_max {
            t_max = t;
        }
    }
    let denom = (t_max - t_min).max(1e-6);
    (dx, dy, t_min, denom)
}

// ── COM wrappers (vtable-slot idiom, module-local) ───────────────────

unsafe fn com_release(obj: *mut c_void) {
    if obj.is_null() {
        return;
    }
    let vtbl = *(obj as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void) -> u32 = core::mem::transmute(*vtbl.add(2));
    fp(obj);
}

unsafe fn dev_call_create_shader(
    device: *mut c_void,
    slot: usize,
    dxbc: &[u8],
) -> Result<*mut c_void, i32> {
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *const c_void,
        usize,
        *mut c_void,
        *mut *mut c_void,
    ) -> i32 = core::mem::transmute(*vtbl.add(slot));
    let mut out = std::ptr::null_mut();
    let hr = fp(
        device,
        dxbc.as_ptr() as *const c_void,
        dxbc.len(),
        std::ptr::null_mut(),
        &mut out,
    );
    if hr < 0 || out.is_null() {
        return Err(hr);
    }
    Ok(out)
}

unsafe fn dev_create_vertex_shader(device: *mut c_void, dxbc: &[u8]) -> Result<*mut c_void, i32> {
    dev_call_create_shader(device, DEV_CREATE_VERTEX_SHADER, dxbc)
}
unsafe fn dev_create_pixel_shader(device: *mut c_void, dxbc: &[u8]) -> Result<*mut c_void, i32> {
    dev_call_create_shader(device, DEV_CREATE_PIXEL_SHADER, dxbc)
}

unsafe fn dev_create_view(
    device: *mut c_void,
    slot: usize,
    resource: *mut c_void,
) -> Result<*mut c_void, i32> {
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        *const c_void,
        *mut *mut c_void,
    ) -> i32 = core::mem::transmute(*vtbl.add(slot));
    let mut out = std::ptr::null_mut();
    let hr = fp(device, resource, std::ptr::null(), &mut out);
    if hr < 0 || out.is_null() {
        return Err(hr);
    }
    Ok(out)
}
unsafe fn dev_create_rtv(device: *mut c_void, res: *mut c_void) -> Result<*mut c_void, i32> {
    dev_create_view(device, DEV_CREATE_RENDER_TARGET_VIEW, res)
}
unsafe fn dev_create_srv(device: *mut c_void, res: *mut c_void) -> Result<*mut c_void, i32> {
    dev_create_view(device, DEV_CREATE_SHADER_RESOURCE_VIEW, res)
}

unsafe fn dev_create_sampler_point_clamp(device: *mut c_void) -> Result<*mut c_void, i32> {
    let desc = D3D11_SAMPLER_DESC {
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
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *const D3D11_SAMPLER_DESC,
        *mut *mut c_void,
    ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_SAMPLER_STATE));
    let mut out = std::ptr::null_mut();
    let hr = fp(device, &desc, &mut out);
    if hr < 0 || out.is_null() {
        return Err(hr);
    }
    Ok(out)
}

unsafe fn dev_create_blend_verbatim(device: *mut c_void) -> Result<*mut c_void, i32> {
    let rt0 = D3D11_RENDER_TARGET_BLEND_DESC {
        BlendEnable: 0, // verbatim PS output (the blend is in-shader)
        SrcBlend: D3D11_BLEND_ONE,
        DestBlend: D3D11_BLEND_ZERO,
        BlendOp: D3D11_BLEND_OP_ADD,
        SrcBlendAlpha: D3D11_BLEND_ONE,
        DestBlendAlpha: D3D11_BLEND_ZERO,
        BlendOpAlpha: D3D11_BLEND_OP_ADD,
        RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL,
    };
    let desc = D3D11_BLEND_DESC {
        AlphaToCoverageEnable: 0,
        IndependentBlendEnable: 0,
        RenderTarget: [rt0; 8],
    };
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *const D3D11_BLEND_DESC,
        *mut *mut c_void,
    ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_BLEND_STATE));
    let mut out = std::ptr::null_mut();
    let hr = fp(device, &desc, &mut out);
    if hr < 0 || out.is_null() {
        return Err(hr);
    }
    Ok(out)
}

unsafe fn dev_create_dynamic_cb(device: *mut c_void, byte_width: u32) -> Result<*mut c_void, i32> {
    // CB byte width must be a multiple of 16.
    let bw = (byte_width + 15) & !15;
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: bw,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE,
        MiscFlags: 0,
        StructureByteStride: 0,
    };
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *const D3D11_BUFFER_DESC,
        *const D3D11_SUBRESOURCE_DATA,
        *mut *mut c_void,
    ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_BUFFER));
    let mut out = std::ptr::null_mut();
    let hr = fp(device, &desc, std::ptr::null(), &mut out);
    if hr < 0 || out.is_null() {
        return Err(hr);
    }
    Ok(out)
}

unsafe fn create_tex2d(
    device: *mut c_void,
    w: u32,
    h: u32,
    usage: u32,
    bind: u32,
    cpu: u32,
) -> Result<*mut c_void, i32> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDescCount: 1,
        SampleDescQuality: 0,
        Usage: usage,
        BindFlags: bind,
        CPUAccessFlags: cpu,
        MiscFlags: 0,
    };
    let vtbl = *(device as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *const D3D11_TEXTURE2D_DESC,
        *const c_void,
        *mut *mut c_void,
    ) -> i32 = core::mem::transmute(*vtbl.add(DEV_CREATE_TEXTURE2D));
    let mut out = std::ptr::null_mut();
    let hr = fp(device, &desc, std::ptr::null(), &mut out);
    if hr < 0 || out.is_null() {
        return Err(hr);
    }
    Ok(out)
}

unsafe fn create_rt_srv_texture(device: *mut c_void, w: u32, h: u32) -> Result<*mut c_void, i32> {
    create_tex2d(
        device,
        w,
        h,
        D3D11_USAGE_DEFAULT,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        0,
    )
}
unsafe fn create_sampled_texture(device: *mut c_void, w: u32, h: u32) -> Result<*mut c_void, i32> {
    create_tex2d(
        device,
        w,
        h,
        D3D11_USAGE_DEFAULT,
        D3D11_BIND_SHADER_RESOURCE,
        0,
    )
}
unsafe fn create_readback_texture(device: *mut c_void, w: u32, h: u32) -> Result<*mut c_void, i32> {
    create_tex2d(device, w, h, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ)
}

// context-side wrappers

unsafe fn ctx_update_subresource(
    context: *mut c_void,
    dst: *mut c_void,
    src: *const c_void,
    row_pitch: u32,
) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        *const c_void,
        *const c_void,
        u32,
        u32,
    ) = core::mem::transmute(*vtbl.add(CTX_UPDATE_SUBRESOURCE));
    fp(context, dst, 0, std::ptr::null(), src, row_pitch, 0);
}

unsafe fn ctx_copy_resource(context: *mut c_void, dst: *mut c_void, src: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) =
        core::mem::transmute(*vtbl.add(CTX_COPY_RESOURCE));
    fp(context, dst, src);
}

unsafe fn ctx_ia_set_primitive_topology(context: *mut c_void, topo: u32) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32) =
        core::mem::transmute(*vtbl.add(CTX_IA_SET_PRIMITIVE_TOPOLOGY));
    fp(context, topo);
}
unsafe fn ctx_ia_set_input_layout_null(context: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, *mut c_void) =
        core::mem::transmute(*vtbl.add(CTX_IA_SET_INPUT_LAYOUT));
    fp(context, std::ptr::null_mut());
}
unsafe fn ctx_vs_set_shader(context: *mut c_void, vs: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const *mut c_void, u32) =
        core::mem::transmute(*vtbl.add(CTX_VS_SET_SHADER));
    fp(context, vs, std::ptr::null(), 0);
}
unsafe fn ctx_ps_set_shader(context: *mut c_void, ps: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const *mut c_void, u32) =
        core::mem::transmute(*vtbl.add(CTX_PS_SET_SHADER));
    fp(context, ps, std::ptr::null(), 0);
}
unsafe fn ctx_ps_set_samplers(context: *mut c_void, sampler: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void) =
        core::mem::transmute(*vtbl.add(CTX_PS_SET_SAMPLERS));
    let s = [sampler];
    fp(context, 0, 1, s.as_ptr());
}
/// PSSetShaderResources(0, 2, [t0, t1]).
unsafe fn ctx_ps_set_shader_resources2(context: *mut c_void, t0: *mut c_void, t1: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void) =
        core::mem::transmute(*vtbl.add(CTX_PS_SET_SHADER_RESOURCES));
    let srvs = [t0, t1];
    fp(context, 0, 2, srvs.as_ptr());
}
unsafe fn ctx_om_set_blend_state(context: *mut c_void, state: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, *const f32, u32) =
        core::mem::transmute(*vtbl.add(CTX_OM_SET_BLEND_STATE));
    fp(context, state, std::ptr::null(), 0xFFFF_FFFF);
}
unsafe fn ctx_om_set_render_targets(context: *mut c_void, rtv: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *mut c_void) =
        core::mem::transmute(*vtbl.add(CTX_OM_SET_RENDER_TARGETS));
    let rtvs = [rtv];
    fp(context, 1, rtvs.as_ptr(), std::ptr::null_mut());
}
unsafe fn ctx_om_set_render_targets_null(context: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *mut c_void) =
        core::mem::transmute(*vtbl.add(CTX_OM_SET_RENDER_TARGETS));
    let rtvs = [std::ptr::null_mut()];
    fp(context, 1, rtvs.as_ptr(), std::ptr::null_mut());
}
unsafe fn ctx_rs_set_viewports(context: *mut c_void, vp: &D3D11_VIEWPORT) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32, *const D3D11_VIEWPORT) =
        core::mem::transmute(*vtbl.add(CTX_RS_SET_VIEWPORTS));
    fp(context, 1, vp as *const _);
}
unsafe fn ctx_vs_set_constant_buffers(context: *mut c_void, cb: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void) =
        core::mem::transmute(*vtbl.add(CTX_VS_SET_CONSTANT_BUFFERS));
    let cbs = [cb];
    fp(context, 0, 1, cbs.as_ptr());
}
unsafe fn ctx_ps_set_constant_buffers(context: *mut c_void, cb: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void) =
        core::mem::transmute(*vtbl.add(CTX_PS_SET_CONSTANT_BUFFERS));
    let cbs = [cb];
    fp(context, 0, 1, cbs.as_ptr());
}
unsafe fn ctx_draw(context: *mut c_void, vertex_count: u32, start: u32) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, u32, u32) =
        core::mem::transmute(*vtbl.add(CTX_DRAW));
    fp(context, vertex_count, start);
}
unsafe fn ctx_map_write_discard(
    context: *mut c_void,
    res: *mut c_void,
    mapped: &mut D3D11_MAPPED_SUBRESOURCE,
) -> i32 {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        u32,
        u32,
        *mut D3D11_MAPPED_SUBRESOURCE,
    ) -> i32 = core::mem::transmute(*vtbl.add(CTX_MAP));
    fp(context, res, 0, D3D11_MAP_WRITE_DISCARD, 0, mapped)
}
unsafe fn ctx_map_read(
    context: *mut c_void,
    res: *mut c_void,
    mapped: &mut D3D11_MAPPED_SUBRESOURCE,
) -> i32 {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        u32,
        u32,
        *mut D3D11_MAPPED_SUBRESOURCE,
    ) -> i32 = core::mem::transmute(*vtbl.add(CTX_MAP));
    fp(context, res, 0, D3D11_MAP_READ, 0, mapped)
}
unsafe fn ctx_unmap(context: *mut c_void, res: *mut c_void) {
    let vtbl = *(context as *const *const usize);
    let fp: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) =
        core::mem::transmute(*vtbl.add(CTX_UNMAP));
    fp(context, res, 0);
}

// ── Tests ────────────────────────────────────────────────────────────
//
// The golden gate: each GPU-drawn quad is compared against a self-contained
// CPU oracle that reproduces cv_gfx's rasterizer EXACTLY (cv_gpu is dep-free,
// so the oracle is duplicated here rather than imported; the math is copied
// verbatim from crates/cv_gfx/src/lib.rs blend_bgra / fill_rect /
// fill_rect_gradient / blit_bgra). Device-required tests skip gracefully when
// no D3D11 device is available (headless CI without WARP).

#[cfg(test)]
mod tests {
    use super::*;

    // ── CPU oracle (verbatim cv_gfx math) ────────────────────────────

    /// cv_gfx::blend_bgra — straight-alpha source-over with un-premultiply +
    /// round. `dst` is packed BGRA u32; `src` is straight-alpha RGBA.
    fn oracle_blend_bgra(dst: u32, src: Rgba) -> u32 {
        let da = (dst >> 24) & 0xFF;
        let dr = (dst >> 16) & 0xFF;
        let dg = (dst >> 8) & 0xFF;
        let db = dst & 0xFF;
        let sa_f = src.a as f32 / 255.0;
        let da_f = da as f32 / 255.0;
        let inv = 1.0 - sa_f;
        let out_a = sa_f + da_f * inv;
        if out_a <= 0.0 {
            return 0;
        }
        let r = ((src.r as f32 * sa_f + dr as f32 * da_f * inv) / out_a).round() as u32;
        let g = ((src.g as f32 * sa_f + dg as f32 * da_f * inv) / out_a).round() as u32;
        let b = ((src.b as f32 * sa_f + db as f32 * da_f * inv) / out_a).round() as u32;
        let a = (out_a * 255.0).round() as u32;
        (a << 24) | (r << 16) | (g << 8) | b
    }

    /// CPU render of one quad over `fb` (BGRA u32, vp_w*vp_h), in place —
    /// reproduces cv_gfx::fill_rect / fill_rect_gradient / blit_bgra.
    fn oracle_draw_quad(fb: &mut [u32], vp_w: i32, vp_h: i32, q: &GpuQuad) {
        let x0 = q.x.max(0);
        let y0 = q.y.max(0);
        let x1 = (q.x + q.w).min(vp_w);
        let y1 = (q.y + q.h).min(vp_h);
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        match &q.fill {
            QuadFill::Solid(c) => {
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        let idx = (yy * vp_w + xx) as usize;
                        if c.a == 255 {
                            fb[idx] = c.to_bgra_u32();
                        } else if c.a > 0 {
                            fb[idx] = oracle_blend_bgra(fb[idx], *c);
                        }
                    }
                }
            }
            QuadFill::LinearGradient {
                angle_deg,
                from,
                to,
            } => {
                let (dx, dy, t_min, denom) = gradient_axis(*angle_deg, q.w, q.h);
                for yy in y0..y1 {
                    let py = (yy - q.y) as f32 + 0.5;
                    for xx in x0..x1 {
                        let px = (xx - q.x) as f32 + 0.5;
                        let t = (((px * dx + py * dy) - t_min) / denom).clamp(0.0, 1.0);
                        let r = (from.r as f32 * (1.0 - t) + to.r as f32 * t) as u8;
                        let g = (from.g as f32 * (1.0 - t) + to.g as f32 * t) as u8;
                        let b = (from.b as f32 * (1.0 - t) + to.b as f32 * t) as u8;
                        let a = (from.a as f32 * (1.0 - t) + to.a as f32 * t) as u8;
                        let c = Rgba { r, g, b, a };
                        let idx = (yy * vp_w + xx) as usize;
                        if a == 255 {
                            fb[idx] = c.to_bgra_u32();
                        } else if a > 0 {
                            fb[idx] = oracle_blend_bgra(fb[idx], c);
                        }
                    }
                }
            }
            QuadFill::Image { bgra } => {
                for yy in y0..y1 {
                    let sy = (yy - q.y) as usize;
                    for xx in x0..x1 {
                        let sx = (xx - q.x) as usize;
                        let s = bgra[sy * (q.w as usize) + sx];
                        let sa = ((s >> 24) & 0xFF) as u8;
                        if sa == 0 {
                            continue;
                        }
                        let idx = (yy * vp_w + xx) as usize;
                        if sa == 255 {
                            fb[idx] = s;
                        } else {
                            let c = Rgba {
                                r: ((s >> 16) & 0xFF) as u8,
                                g: ((s >> 8) & 0xFF) as u8,
                                b: (s & 0xFF) as u8,
                                a: sa,
                            };
                            fb[idx] = oracle_blend_bgra(fb[idx], c);
                        }
                    }
                }
            }
        }
    }

    fn oracle_render(vp_w: u32, vp_h: u32, backdrop: &[u32], quads: &[GpuQuad]) -> Vec<u32> {
        let mut fb = backdrop.to_vec();
        for q in quads {
            oracle_draw_quad(&mut fb, vp_w as i32, vp_h as i32, q);
        }
        fb
    }

    /// Max per-channel delta between two BGRA u32 buffers + the count of
    /// differing pixels.
    fn diff(a: &[u32], b: &[u32]) -> (u8, usize) {
        assert_eq!(a.len(), b.len());
        let mut max = 0u8;
        let mut n = 0usize;
        for i in 0..a.len() {
            let (pa, pb) = (a[i].to_le_bytes(), b[i].to_le_bytes());
            let mut differ = false;
            for c in 0..4 {
                let d = pa[c].abs_diff(pb[c]);
                if d > max {
                    max = d;
                }
                if d != 0 {
                    differ = true;
                }
            }
            if differ {
                n += 1;
            }
        }
        (max, n)
    }

    fn checkerboard(w: u32, h: u32) -> Vec<u32> {
        let mut v = vec![0u32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let on = ((x / 4 + y / 4) & 1) == 0;
                // BGRA u32: A<<24 | R<<16 | G<<8 | B.
                v[(y * w + x) as usize] = if on { 0xFF_30_60_90 } else { 0xFF_C0_A0_80 };
            }
        }
        v
    }

    /// Run quads on the GPU drawer, or None if no device.
    fn gpu_render(vp_w: u32, vp_h: u32, backdrop: &[u32], quads: &[GpuQuad]) -> Option<Vec<u32>> {
        let drawer = match QuadDrawer::new() {
            Ok(d) => d,
            Err(_) => return None,
        };
        drawer.draw_quads_offscreen(vp_w, vp_h, backdrop, quads).ok()
    }

    // ── Plumbing tests (no device) ───────────────────────────────────

    #[test]
    fn flag_default_off() {
        // CV_GPU_RASTER defaults OFF; only explicit affirmatives enable it.
        if std::env::var("CV_GPU_RASTER").is_err() {
            assert!(!quad_raster_enabled());
        }
    }

    #[test]
    fn rgba_packs_bgra_u32() {
        let c = Rgba::new(0x12, 0x34, 0x56, 0x78);
        assert_eq!(c.to_bgra_u32(), 0x78_12_34_56);
    }

    #[test]
    fn cb_layouts_are_16_aligned() {
        // Constant buffers must be a multiple of 16 bytes.
        assert_eq!(core::mem::size_of::<QuadVsCb>() % 16, 0);
        assert_eq!(core::mem::size_of::<QuadPsCb>() % 16, 0);
        assert_eq!(core::mem::size_of::<QuadVsCb>(), 32);
        assert_eq!(core::mem::size_of::<QuadPsCb>(), 96);
    }

    #[test]
    fn descriptor_constants_match_d3d11_h() {
        assert_eq!(D3D11_BIND_CONSTANT_BUFFER, 0x4);
        assert_eq!(D3D11_BIND_SHADER_RESOURCE, 0x8);
        assert_eq!(D3D11_BIND_RENDER_TARGET, 0x20);
        assert_eq!(D3D11_USAGE_DYNAMIC, 2);
        assert_eq!(D3D11_MAP_WRITE_DISCARD, 4);
        assert_eq!(DXGI_FORMAT_B8G8R8A8_UNORM, 87);
        assert_eq!(CTX_VS_SET_CONSTANT_BUFFERS, 7);
        assert_eq!(CTX_PS_SET_CONSTANT_BUFFERS, 16);
        assert_eq!(DEV_CREATE_BUFFER, 3);
    }

    #[test]
    fn embedded_draw_quad_dxbc_well_formed() {
        assert!(!DRAW_QUAD_VS_DXBC.is_empty());
        assert!(!DRAW_QUAD_PS_DXBC.is_empty());
        assert_eq!(&DRAW_QUAD_VS_DXBC[0..4], b"DXBC");
        assert_eq!(&DRAW_QUAD_PS_DXBC[0..4], b"DXBC");
    }

    #[test]
    fn oracle_blend_matches_cv_gfx_constants() {
        // Spot-check the oracle reproduces the documented faint-gold fix:
        // gold (255,215,0,a=26) over transparent stays gold (not dragged dark).
        let out = oracle_blend_bgra(0x0000_0000, Rgba::new(255, 215, 0, 26));
        let a = (out >> 24) & 0xFF;
        let r = (out >> 16) & 0xFF;
        // out_a = 26/255; un-premul recovers full-strength rgb.
        assert_eq!(a, 26);
        assert_eq!(r, 255);
    }

    // ── Device-required golden-diff GATE ─────────────────────────────

    #[test]
    fn gpu_solid_opaque_is_bit_exact() {
        let (w, h) = (32u32, 24u32);
        let backdrop = checkerboard(w, h);
        let quads = vec![GpuQuad {
            x: 5,
            y: 4,
            w: 18,
            h: 12,
            fill: QuadFill::Solid(Rgba::new(200, 60, 30, 255)),
        }];
        let gpu = match gpu_render(w, h, &backdrop, &quads) {
            Some(v) => v,
            None => {
                eprintln!("gpu_solid_opaque_is_bit_exact: no D3D device, skipping");
                return;
            }
        };
        let cpu = oracle_render(w, h, &backdrop, &quads);
        let (max, n) = diff(&gpu, &cpu);
        assert_eq!(max, 0, "solid opaque quad differs from CPU oracle (max {max}, {n} px)");
    }

    #[test]
    fn gpu_solid_semi_over_backdrop_matches_oracle() {
        // Semi-transparent solid over the checkerboard backdrop — exercises the
        // in-shader straight-alpha source-over. Must match cv_gfx::blend_bgra.
        let (w, h) = (32u32, 32u32);
        let backdrop = checkerboard(w, h);
        let quads = vec![GpuQuad {
            x: 2,
            y: 2,
            w: 28,
            h: 28,
            fill: QuadFill::Solid(Rgba::new(255, 215, 0, 96)), // gold @ a=96
        }];
        let gpu = match gpu_render(w, h, &backdrop, &quads) {
            Some(v) => v,
            None => {
                eprintln!("gpu_solid_semi_over_backdrop_matches_oracle: no device, skipping");
                return;
            }
        };
        let cpu = oracle_render(w, h, &backdrop, &quads);
        let (max, n) = diff(&gpu, &cpu);
        // Tight tolerance: in-shader float reproduces blend_bgra; allow <=1 LSB
        // for any GPU rounding-mode edge.
        assert!(max <= 1, "semi-solid blend drifted from oracle (max {max}, {n} px)");
    }

    #[test]
    fn gpu_image_quad_copy_is_bit_exact() {
        // An OPAQUE image quad placed at an offset over the backdrop must be a
        // byte-exact 1:1 copy of the source where it lands (sa==255 hard-write).
        let (w, h) = (32u32, 32u32);
        let backdrop = checkerboard(w, h);
        let (qw, qh) = (16i32, 16i32);
        let mut img = vec![0u32; (qw * qh) as usize];
        for y in 0..qh {
            for x in 0..qw {
                let r = (x * 16) as u32 & 0xFF;
                let g = (y * 16) as u32 & 0xFF;
                img[(y * qw + x) as usize] = 0xFF00_0000 | (r << 16) | (g << 8) | 0x40;
            }
        }
        let quads = vec![GpuQuad {
            x: 8,
            y: 6,
            w: qw,
            h: qh,
            fill: QuadFill::Image { bgra: img.clone() },
        }];
        let gpu = match gpu_render(w, h, &backdrop, &quads) {
            Some(v) => v,
            None => {
                eprintln!("gpu_image_quad_copy_is_bit_exact: no device, skipping");
                return;
            }
        };
        let cpu = oracle_render(w, h, &backdrop, &quads);
        let (max, n) = diff(&gpu, &cpu);
        assert_eq!(max, 0, "opaque image quad differs from CPU oracle (max {max}, {n} px)");
    }

    #[test]
    fn gpu_image_quad_semi_blends_match_oracle() {
        // A semi-transparent image quad (per-pixel alpha) source-over backdrop.
        let (w, h) = (24u32, 24u32);
        let backdrop = checkerboard(w, h);
        let (qw, qh) = (12i32, 12i32);
        let mut img = vec![0u32; (qw * qh) as usize];
        for y in 0..qh {
            for x in 0..qw {
                let a = ((x + y) * 8) as u32 & 0xFF; // ramp alpha incl. 0
                img[(y * qw + x) as usize] = (a << 24) | 0x00_FF_80_20;
            }
        }
        let quads = vec![GpuQuad {
            x: 4,
            y: 4,
            w: qw,
            h: qh,
            fill: QuadFill::Image { bgra: img },
        }];
        let gpu = match gpu_render(w, h, &backdrop, &quads) {
            Some(v) => v,
            None => {
                eprintln!("gpu_image_quad_semi_blends_match_oracle: no device, skipping");
                return;
            }
        };
        let cpu = oracle_render(w, h, &backdrop, &quads);
        let (max, n) = diff(&gpu, &cpu);
        assert!(max <= 1, "semi image blend drifted from oracle (max {max}, {n} px)");
    }

    #[test]
    fn gpu_linear_gradient_matches_oracle_within_tolerance() {
        // Opaque diagonal gradient — exercises the per-pixel axis projection +
        // truncating channel lerp reproduced in draw_quad_ps.hlsl.
        let (w, h) = (40u32, 28u32);
        let backdrop = checkerboard(w, h);
        let quads = vec![GpuQuad {
            x: 3,
            y: 3,
            w: 34,
            h: 22,
            fill: QuadFill::LinearGradient {
                angle_deg: 45.0,
                from: Rgba::new(255, 0, 0, 255),
                to: Rgba::new(0, 0, 255, 255),
            },
        }];
        let gpu = match gpu_render(w, h, &backdrop, &quads) {
            Some(v) => v,
            None => {
                eprintln!("gpu_linear_gradient_matches_oracle_within_tolerance: no device, skipping");
                return;
            }
        };
        let cpu = oracle_render(w, h, &backdrop, &quads);
        let (max, n) = diff(&gpu, &cpu);
        // Tight tolerance: the floor() truncation at an interpolation boundary
        // can flip a single channel by 1 between GPU uv-interp and CPU integer
        // pixel-center. The blend/lerp math is otherwise identical.
        assert!(max <= 1, "gradient drifted from oracle beyond tolerance (max {max}, {n}/{} px)", gpu.len());
    }

    #[test]
    fn gpu_overlapping_quads_compose_in_order() {
        // Two overlapping semi-transparent quads must compose in submission
        // order via the ping-pong backdrop — the multi-quad correctness gate.
        let (w, h) = (32u32, 32u32);
        let backdrop = checkerboard(w, h);
        let quads = vec![
            GpuQuad {
                x: 2,
                y: 2,
                w: 20,
                h: 20,
                fill: QuadFill::Solid(Rgba::new(255, 0, 0, 128)),
            },
            GpuQuad {
                x: 10,
                y: 10,
                w: 20,
                h: 20,
                fill: QuadFill::Solid(Rgba::new(0, 0, 255, 128)),
            },
        ];
        let gpu = match gpu_render(w, h, &backdrop, &quads) {
            Some(v) => v,
            None => {
                eprintln!("gpu_overlapping_quads_compose_in_order: no device, skipping");
                return;
            }
        };
        let cpu = oracle_render(w, h, &backdrop, &quads);
        let (max, n) = diff(&gpu, &cpu);
        assert!(max <= 1, "overlapping quads diverged from oracle (max {max}, {n} px)");
    }

    #[test]
    fn gpu_empty_quad_list_returns_backdrop() {
        let (w, h) = (16u32, 16u32);
        let backdrop = checkerboard(w, h);
        let gpu = match gpu_render(w, h, &backdrop, &[]) {
            Some(v) => v,
            None => {
                eprintln!("gpu_empty_quad_list_returns_backdrop: no device, skipping");
                return;
            }
        };
        let (max, _) = diff(&gpu, &backdrop);
        assert_eq!(max, 0, "no quads must return the backdrop unchanged");
    }
}

