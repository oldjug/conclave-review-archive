//! GPU rasterization of a `cv_paint::DisplayList` — the Chrome cc/viz +
//! Skia-Ganesh model: the recorded display list is rasterized on the GPU into a
//! render target, then composited/presented, with the CPU rasterizer kept as
//! the byte-exact oracle + fallback.
//!
//! ## How this maps to Chrome
//!
//! Chrome's `cc` records paint into a `cc::DisplayItemList` (an `SkPicture`),
//! then `GpuImageDecodeCache` / `RasterSource::PlaybackToCanvas` plays it back
//! onto a GPU-backed `SkSurface` via Skia-Ganesh. viz then draws the resulting
//! tiles as `TextureDrawQuad`s composited with `SkBlendMode::kSrcOver`. Our
//! `cv_paint::DisplayList` is the `DisplayItemList`; this module is the
//! Ganesh playback: it FLATTENS the display list's push/pop transform-clip-
//! opacity stack into screen-space draw quads (exactly as the CPU rasterizer
//! [`crate::display_list_rasterize::rasterize`] resolves the stack), then draws
//! those quads through [`cv_gpu::QuadDrawer`] (a VS/PS/RTV pipeline, viz's
//! per-quad draw) using the straight-alpha source-over blend that matches
//! `cv_gfx` — the live rasterizer.
//!
//! ## Byte-exact oracle + fallback
//!
//! The CPU twin of the GPU draw is [`cv_gpu::cpu_render_quads`] over the SAME
//! flattened quad list. The flattening is shared CPU code, so it cannot make
//! the GPU and CPU diverge; the ONLY divergence source is the GPU draw itself.
//! That isolates the property Chrome's pixel tests assert — "GPU raster ==
//! raster reference" — to the single thing under test. Solid + image quads are
//! BYTE-IDENTICAL (max-delta 0); linear gradients are within 1 LSB (GPU fp32
//! division at a `floor()` boundary; see `gpu_gradient_within_one_lsb`).
//!
//! When `CV_GPU_RASTER` is off, no device is available, or a draw fails, this
//! returns the CPU result — never a panic, never a divergent frame.

use cv_gpu::{GpuQuad, QuadDrawer, QuadFill, Rgba};
use cv_paint::{Bgra, DisplayList, PaintItem, PaintRect};

/// 2×3 affine (scale + translate only, the display-list subset), composed the
/// same way [`crate::display_list_rasterize`] composes the PushTransform stack.
#[derive(Clone, Copy)]
struct Affine {
    sx: f32,
    sy: f32,
    tx: f32,
    ty: f32,
}

impl Affine {
    fn identity() -> Self {
        Self { sx: 1.0, sy: 1.0, tx: 0.0, ty: 0.0 }
    }
    /// Compose `self ∘ local` (apply `local` first, then `self`) for the
    /// scale-translate subset: matches Mat23::compose for b=c=0.
    fn compose(self, sx: f32, sy: f32, tx: f32, ty: f32) -> Self {
        Self {
            sx: self.sx * sx,
            sy: self.sy * sy,
            tx: self.sx * tx + self.tx,
            ty: self.sy * ty + self.ty,
        }
    }
    fn apply_rect(self, r: PaintRect) -> PaintRect {
        // Scale-translate keeps rects axis-aligned; map the origin + extents.
        let x = (r.x as f32 * self.sx + self.tx) as i32;
        let y = (r.y as f32 * self.sy + self.ty) as i32;
        // Match Mat23::transform_rect: map both corners, take the span.
        let x1 = ((r.x as f32 + r.w as f32) * self.sx + self.tx) as i32;
        let y1 = ((r.y as f32 + r.h as f32) * self.sy + self.ty) as i32;
        PaintRect {
            x: x.min(x1),
            y: y.min(y1),
            w: (x1 - x).unsigned_abs(),
            h: (y1 - y).unsigned_abs(),
        }
    }
}

/// Intersect two rects (zero-area at the edge if disjoint), matching
/// `display_list_rasterize::intersect_rects` for clip stacking.
fn intersect(a: PaintRect, b: PaintRect) -> PaintRect {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w as i32).min(b.x + b.w as i32);
    let y1 = (a.y + a.h as i32).min(b.y + b.h as i32);
    if x1 > x0 && y1 > y0 {
        PaintRect { x: x0, y: y0, w: (x1 - x0) as u32, h: (y1 - y0) as u32 }
    } else {
        PaintRect { x: x0, y: y0, w: 0, h: 0 }
    }
}

/// Clip a rect to an optional active clip; returns the clipped rect (possibly
/// zero-area). Quads are clipped on the CPU before being handed to the GPU
/// (`GpuQuad` has no clip field; the rasterizer just clamps to the rect).
fn clip_rect(r: PaintRect, clip: Option<PaintRect>) -> PaintRect {
    match clip {
        None => r,
        Some(c) => intersect(r, c),
    }
}

fn bgra_to_rgba(c: Bgra) -> Rgba {
    Rgba::new(c.r(), c.g(), c.b(), c.a())
}

/// Fold a layer `opacity` (0..1) into a straight-alpha color by scaling alpha,
/// matching `display_list_rasterize::blend`'s `sa *= opacity` — except we keep
/// the cv_gfx straight-alpha convention (the GPU shader + `cpu_render_quads`
/// both un-premultiply, so scaling the straight alpha is exact).
fn apply_opacity(c: Rgba, opacity: f32) -> Rgba {
    if opacity >= 1.0 {
        return c;
    }
    let a = (c.a as f32 * opacity).round().clamp(0.0, 255.0) as u8;
    Rgba::new(c.r, c.g, c.b, a)
}

/// Convert the BGRA-byte `pixels` of a `PaintItem::Image` (RGBA byte order in
/// the buffer is B,G,R,A per `display_list_rasterize::blit_image`) into the
/// tightly-packed BGRA-u32 `q.w*q.h` source the `QuadFill::Image` expects,
/// resampling 1:1 over the destination rect (nearest, matching blit_image).
fn image_quad_pixels(r: PaintRect, pixels: &[u8], iw: u32, ih: u32) -> Vec<u32> {
    let w = r.w as usize;
    let h = r.h as usize;
    let mut out = vec![0u32; w * h];
    if w == 0 || h == 0 || iw == 0 || ih == 0 {
        return out;
    }
    for dy in 0..h {
        for dx in 0..w {
            let sx = ((dx as f32 / r.w as f32) * iw as f32) as u32;
            let sy = ((dy as f32 / r.h as f32) * ih as f32) as u32;
            if sx >= iw || sy >= ih {
                continue;
            }
            let p = ((sy * iw + sx) * 4) as usize;
            if p + 3 >= pixels.len() {
                continue;
            }
            // Buffer is B,G,R,A; pack to A<<24|R<<16|G<<8|B.
            out[dy * w + dx] = ((pixels[p + 3] as u32) << 24)
                | ((pixels[p + 2] as u32) << 16)
                | ((pixels[p + 1] as u32) << 8)
                | pixels[p] as u32;
        }
    }
    out
}

/// Whether a paint item is in the GPU-rasterizable, byte-exact subset. Text is
/// excluded (it lives in cv_gfx glyph raster, not the display list). BoxShadow
/// with blur is not yet modeled here. Gradients are included but carry the
/// documented 1-LSB caveat.
fn flatten(list: &DisplayList) -> Vec<GpuQuad> {
    let mut quads: Vec<GpuQuad> = Vec::new();
    let mut mat = Affine::identity();
    let mut mat_stack: Vec<Affine> = Vec::new();
    let mut opacity = 1.0f32;
    let mut op_stack: Vec<f32> = Vec::new();
    let mut clip: Option<PaintRect> = None;
    let mut clip_stack: Vec<Option<PaintRect>> = Vec::new();

    let push_solid = |quads: &mut Vec<GpuQuad>, r: PaintRect, color: Rgba, op: f32, clip: Option<PaintRect>| {
        let cr = clip_rect(r, clip);
        if cr.w == 0 || cr.h == 0 {
            return;
        }
        quads.push(GpuQuad {
            x: cr.x,
            y: cr.y,
            w: cr.w as i32,
            h: cr.h as i32,
            fill: QuadFill::Solid(apply_opacity(color, op)),
        });
    };

    for item in &list.items {
        match item {
            PaintItem::PushTransform { translate_x, translate_y, scale_x, scale_y } => {
                mat_stack.push(mat);
                mat = mat.compose(*scale_x, *scale_y, *translate_x, *translate_y);
            }
            PaintItem::PopTransform => {
                if let Some(p) = mat_stack.pop() {
                    mat = p;
                }
            }
            PaintItem::PushOpacity(v) => {
                op_stack.push(opacity);
                opacity *= *v;
            }
            PaintItem::PopOpacity => {
                if let Some(v) = op_stack.pop() {
                    opacity = v;
                }
            }
            PaintItem::PushClip(r) => {
                let screen = mat.apply_rect(*r);
                let new_clip = match clip {
                    Some(e) => intersect(e, screen),
                    None => screen,
                };
                clip_stack.push(clip);
                clip = Some(new_clip);
            }
            PaintItem::PopClip => {
                if let Some(c) = clip_stack.pop() {
                    clip = c;
                }
            }
            PaintItem::Fill { rect, color } => {
                push_solid(&mut quads, mat.apply_rect(*rect), bgra_to_rgba(*color), opacity, clip);
            }
            PaintItem::Border { rect, top, right, bottom, left, color } => {
                let r = mat.apply_rect(*rect);
                let c = bgra_to_rgba(*color);
                // Top.
                push_solid(&mut quads, PaintRect { x: r.x, y: r.y, w: r.w, h: *top }, c, opacity, clip);
                // Bottom.
                push_solid(
                    &mut quads,
                    PaintRect { x: r.x, y: r.y + r.h as i32 - *bottom as i32, w: r.w, h: *bottom },
                    c, opacity, clip,
                );
                // Left.
                push_solid(&mut quads, PaintRect { x: r.x, y: r.y, w: *left, h: r.h }, c, opacity, clip);
                // Right.
                push_solid(
                    &mut quads,
                    PaintRect { x: r.x + r.w as i32 - *right as i32, y: r.y, w: *right, h: r.h },
                    c, opacity, clip,
                );
            }
            PaintItem::Image { rect, pixels, width, height } => {
                let r = clip_rect(mat.apply_rect(*rect), clip);
                if r.w == 0 || r.h == 0 {
                    continue;
                }
                // Resample over the *clipped* rect so the image source aligns
                // with the clamped destination (the CPU `cpu_render_quads`
                // walks the same clipped rect 1:1).
                let src = image_quad_pixels(r, pixels, *width, *height);
                let src = if opacity < 1.0 {
                    src.into_iter()
                        .map(|p| {
                            let a = ((p >> 24) & 0xFF) as f32;
                            let na = (a * opacity).round().clamp(0.0, 255.0) as u32;
                            (na << 24) | (p & 0x00FF_FFFF)
                        })
                        .collect()
                } else {
                    src
                };
                quads.push(GpuQuad { x: r.x, y: r.y, w: r.w as i32, h: r.h as i32, fill: QuadFill::Image { bgra: src } });
            }
            PaintItem::LinearGradient { rect, from, to, angle_deg } => {
                let r = clip_rect(mat.apply_rect(*rect), clip);
                if r.w == 0 || r.h == 0 {
                    continue;
                }
                quads.push(GpuQuad {
                    x: r.x,
                    y: r.y,
                    w: r.w as i32,
                    h: r.h as i32,
                    fill: QuadFill::LinearGradient {
                        angle_deg: *angle_deg,
                        from: apply_opacity(bgra_to_rgba(*from), opacity),
                        to: apply_opacity(bgra_to_rgba(*to), opacity),
                    },
                });
            }
            // BoxShadow (blur) + Text are not in the GPU-exact subset; they are
            // left to the CPU rasterizer (callers that include them must use
            // the CPU path). Skipping them here would diverge from the CPU
            // oracle, so callers gate on `display_list_is_gpu_exact`.
            PaintItem::BoxShadow { .. } | PaintItem::Text { .. } => {}
        }
    }
    quads
}

/// True when every item in `list` is in the byte-exact GPU subset
/// (Fill / Border / Image / Clip / Transform / Opacity). Lists containing
/// `BoxShadow` or `Text` must use the CPU path — the GPU flattener does not
/// model them, so the GPU output would diverge from the full CPU rasterize.
pub fn display_list_is_gpu_exact(list: &DisplayList) -> bool {
    !list
        .items
        .iter()
        .any(|i| matches!(i, PaintItem::BoxShadow { .. } | PaintItem::Text { .. }))
}

/// Whether `list` contains a `LinearGradient` (the one primitive that is GPU-
/// rasterized within 1 LSB rather than byte-identically). A caller that demands
/// byte-identity must avoid the GPU path for gradient-bearing lists.
pub fn display_list_has_gradient(list: &DisplayList) -> bool {
    list.items.iter().any(|i| matches!(i, PaintItem::LinearGradient { .. }))
}

/// Flatten + CPU-rasterize `list` into a BGRA-u32 `width*height` buffer over
/// background `bg`, using the cv_gfx straight-alpha source-over math (the live
/// rasterizer's math, byte-identical to the GPU pixel shader). This is the
/// oracle + the fallback when no GPU device is available.
pub fn rasterize_cpu(list: &DisplayList, width: u32, height: u32, bg: Bgra) -> Vec<u32> {
    let quads = flatten(list);
    let backdrop = vec![bg.0; (width as usize) * (height as usize)];
    cv_gpu::cpu_render_quads(width, height, &backdrop, &quads)
}

/// GPU-rasterize `list` via [`cv_gpu::QuadDrawer`]. Returns the composited BGRA-
/// u32 `width*height` buffer, or `None` when no D3D11 device is available or a
/// draw fails — the caller then uses [`rasterize_cpu`] (the oracle/fallback).
pub fn rasterize_gpu(list: &DisplayList, width: u32, height: u32, bg: Bgra) -> Option<Vec<u32>> {
    let quads = flatten(list);
    let drawer = QuadDrawer::new().ok()?;
    let backdrop = vec![bg.0; (width as usize) * (height as usize)];
    drawer.draw_quads_offscreen(width, height, &backdrop, &quads).ok()
}

/// The production entry: rasterize `list` on the GPU when `CV_GPU_RASTER` is on
/// AND a device is available AND the list is in the GPU-exact subset; otherwise
/// the byte-exact CPU rasterizer. The CPU result is ALWAYS the spec — the GPU
/// path is taken only when its output is byte-identical (solids/images) to that
/// CPU result, so flipping the flag never changes a frame.
pub fn rasterize(list: &DisplayList, width: u32, height: u32, bg: Bgra) -> Vec<u32> {
    if cv_gpu::quad_raster_enabled() && display_list_is_gpu_exact(list) && !display_list_has_gradient(list) {
        if let Some(g) = rasterize_gpu(list, width, height, bg) {
            return g;
        }
    }
    rasterize_cpu(list, width, height, bg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_paint::{Bgra, DisplayList, PaintItem, PaintRect};

    fn checker_bg() -> Bgra {
        Bgra::new(0x90, 0x60, 0x30, 0xFF)
    }

    fn diff(a: &[u32], b: &[u32]) -> (u8, usize) {
        assert_eq!(a.len(), b.len());
        let mut max = 0u8;
        let mut n = 0usize;
        for i in 0..a.len() {
            let (pa, pb) = (a[i].to_le_bytes(), b[i].to_le_bytes());
            let mut d = false;
            for c in 0..4 {
                let dd = pa[c].abs_diff(pb[c]);
                if dd > max {
                    max = dd;
                }
                if dd != 0 {
                    d = true;
                }
            }
            if d {
                n += 1;
            }
        }
        (max, n)
    }

    fn gpu(list: &DisplayList, w: u32, h: u32) -> Option<Vec<u32>> {
        rasterize_gpu(list, w, h, checker_bg())
    }
    fn cpu(list: &DisplayList, w: u32, h: u32) -> Vec<u32> {
        rasterize_cpu(list, w, h, checker_bg())
    }

    // ── Flattening unit tests (no device) ────────────────────────────

    #[test]
    fn flatten_fill_clips_and_transforms() {
        // Transform offsets the fill; clip restricts it. Verify the FLATTENED
        // quad lands where the CPU rasterizer would paint it.
        let mut list = DisplayList::new();
        list.push(PaintItem::PushTransform { translate_x: 4.0, translate_y: 0.0, scale_x: 1.0, scale_y: 1.0 });
        list.push(PaintItem::Fill { rect: PaintRect { x: 0, y: 0, w: 2, h: 2 }, color: Bgra::new(255, 0, 0, 255) });
        list.push(PaintItem::PopTransform);
        let qs = flatten(&list);
        assert_eq!(qs.len(), 1);
        assert_eq!((qs[0].x, qs[0].y, qs[0].w, qs[0].h), (4, 0, 2, 2));
    }

    #[test]
    fn flatten_nested_clips_intersect() {
        let mut list = DisplayList::new();
        list.push(PaintItem::PushClip(PaintRect { x: 0, y: 0, w: 4, h: 8 }));
        list.push(PaintItem::PushClip(PaintRect { x: 2, y: 0, w: 4, h: 8 }));
        list.push(PaintItem::Fill { rect: PaintRect { x: 0, y: 0, w: 8, h: 8 }, color: Bgra::new(255, 0, 0, 255) });
        list.push(PaintItem::PopClip);
        list.push(PaintItem::PopClip);
        let qs = flatten(&list);
        assert_eq!(qs.len(), 1);
        // Intersection of [0,4) and [2,6) is [2,4).
        assert_eq!((qs[0].x, qs[0].w), (2, 2));
    }

    #[test]
    fn gpu_exact_subset_detection() {
        let mut ok = DisplayList::new();
        ok.push(PaintItem::Fill { rect: PaintRect { x: 0, y: 0, w: 1, h: 1 }, color: Bgra::new(1, 2, 3, 255) });
        assert!(display_list_is_gpu_exact(&ok));
        let mut has_text = ok.clone();
        has_text.push(PaintItem::Text { x: 0, y: 0, text: "x".into(), size_px: 12.0, color: Bgra::new(0, 0, 0, 255), bold: false, italic: false, underline: false });
        assert!(!display_list_is_gpu_exact(&has_text));
    }

    #[test]
    fn rasterize_falls_back_to_cpu_when_flag_off() {
        // With CV_GPU_RASTER unset, rasterize() must equal rasterize_cpu().
        if std::env::var("CV_GPU_RASTER").is_ok() {
            return; // env forces a value; skip the "off" assertion
        }
        let mut list = DisplayList::new();
        list.push(PaintItem::Fill { rect: PaintRect { x: 1, y: 1, w: 6, h: 6 }, color: Bgra::new(200, 50, 25, 255) });
        let a = rasterize(&list, 8, 8, checker_bg());
        let b = rasterize_cpu(&list, 8, 8, checker_bg());
        assert_eq!(a, b);
    }

    // ── Device-required byte-identity GOLDEN GATE ─────────────────────
    //
    // The Chrome cc/Ganesh property: GPU raster of the display list ==
    // the CPU raster reference. Solids/images are byte-identical (max 0);
    // gradients are within 1 LSB. Tests skip gracefully without a device.

    #[test]
    fn gpu_solid_page_is_byte_identical() {
        // A realistic "page": background fill, a card, a border, a clipped
        // inner fill, a translated badge. ALL solids/borders => byte-identical.
        let mut list = DisplayList::new();
        list.push(PaintItem::Fill { rect: PaintRect { x: 0, y: 0, w: 64, h: 48 }, color: Bgra::new(245, 245, 245, 255) });
        list.push(PaintItem::Fill { rect: PaintRect { x: 6, y: 6, w: 40, h: 28 }, color: Bgra::new(255, 255, 255, 255) });
        list.push(PaintItem::Border { rect: PaintRect { x: 6, y: 6, w: 40, h: 28 }, top: 2, right: 2, bottom: 2, left: 2, color: Bgra::new(40, 90, 200, 255) });
        list.push(PaintItem::PushClip(PaintRect { x: 10, y: 10, w: 20, h: 16 }));
        list.push(PaintItem::Fill { rect: PaintRect { x: 0, y: 0, w: 64, h: 48 }, color: Bgra::new(30, 160, 90, 180) }); // semi-transparent
        list.push(PaintItem::PopClip);
        list.push(PaintItem::PushTransform { translate_x: 48.0, translate_y: 4.0, scale_x: 1.0, scale_y: 1.0 });
        list.push(PaintItem::Fill { rect: PaintRect { x: 0, y: 0, w: 12, h: 12 }, color: Bgra::new(220, 40, 40, 255) });
        list.push(PaintItem::PopTransform);
        let (w, h) = (64u32, 48u32);
        let g = match gpu(&list, w, h) {
            Some(v) => v,
            None => {
                eprintln!("gpu_solid_page_is_byte_identical: no device, skipping");
                return;
            }
        };
        let c = cpu(&list, w, h);
        let (max, n) = diff(&g, &c);
        assert_eq!(max, 0, "solid page diverged from CPU oracle (max {max}, {n} px)");
    }

    #[test]
    fn gpu_image_in_list_is_byte_identical() {
        let (iw, ih) = (8u32, 8u32);
        let mut pix = vec![0u8; (iw * ih * 4) as usize];
        for y in 0..ih {
            for x in 0..iw {
                let p = ((y * iw + x) * 4) as usize;
                pix[p] = (x * 30) as u8; // B
                pix[p + 1] = (y * 30) as u8; // G
                pix[p + 2] = 0x80; // R
                pix[p + 3] = 0xFF; // A opaque => hard write, byte-exact
            }
        }
        let mut list = DisplayList::new();
        list.push(PaintItem::Fill { rect: PaintRect { x: 0, y: 0, w: 32, h: 32 }, color: Bgra::new(10, 20, 30, 255) });
        list.push(PaintItem::Image { rect: PaintRect { x: 8, y: 6, w: 16, h: 16 }, pixels: pix, width: iw, height: ih });
        let (w, h) = (32u32, 32u32);
        let g = match gpu(&list, w, h) {
            Some(v) => v,
            None => {
                eprintln!("gpu_image_in_list_is_byte_identical: no device, skipping");
                return;
            }
        };
        let c = cpu(&list, w, h);
        let (max, n) = diff(&g, &c);
        assert_eq!(max, 0, "image-in-list diverged from CPU oracle (max {max}, {n} px)");
    }

    #[test]
    fn gpu_gradient_within_one_lsb() {
        // Linear gradient: GPU fp32 division at floor() boundaries drifts by at
        // most 1 LSB vs the CPU oracle (the same tolerance Chrome's pixel tests
        // allow for GPU-rasterized gradients). This is the documented reason the
        // gradient is EXCLUDED from the byte-exact GPU path in `rasterize`.
        let mut list = DisplayList::new();
        list.push(PaintItem::LinearGradient {
            rect: PaintRect { x: 2, y: 2, w: 36, h: 24 },
            from: Bgra::new(255, 0, 0, 255),
            to: Bgra::new(0, 0, 255, 255),
            angle_deg: 45.0,
        });
        let (w, h) = (40u32, 28u32);
        let g = match gpu(&list, w, h) {
            Some(v) => v,
            None => {
                eprintln!("gpu_gradient_within_one_lsb: no device, skipping");
                return;
            }
        };
        let c = cpu(&list, w, h);
        let (max, _n) = diff(&g, &c);
        assert!(max <= 1, "gradient drifted beyond 1 LSB (max {max})");
    }

    #[test]
    fn rasterize_gpu_equals_cpu_on_solid_when_enabled() {
        // When CV_GPU_RASTER is on, the production `rasterize` of a solid-only
        // (no-gradient) GPU-exact list must equal the CPU rasterize exactly.
        if !cv_gpu::quad_raster_enabled() {
            return; // only meaningful with the flag on
        }
        let mut list = DisplayList::new();
        list.push(PaintItem::Fill { rect: PaintRect { x: 0, y: 0, w: 48, h: 32 }, color: Bgra::new(250, 250, 250, 255) });
        list.push(PaintItem::Fill { rect: PaintRect { x: 4, y: 4, w: 20, h: 20 }, color: Bgra::new(180, 30, 30, 200) });
        let (w, h) = (48u32, 32u32);
        let prod = rasterize(&list, w, h, checker_bg());
        let c = rasterize_cpu(&list, w, h, checker_bg());
        // If a device was present `rasterize` used the GPU; it must still be
        // byte-identical to the CPU spec. If no device, it already == CPU.
        let (max, n) = diff(&prod, &c);
        assert_eq!(max, 0, "production rasterize diverged from CPU spec (max {max}, {n} px)");
    }
}
