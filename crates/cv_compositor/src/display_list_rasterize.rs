//! Rasterize a `cv_paint::DisplayList` into a BGRA buffer, honoring
//! the PushClip/PushTransform/PushOpacity stack.

use cv_paint::{Bgra, DisplayList, PaintItem, PaintRect};

/// A 2-D affine transform stored as a 2×3 matrix:
///
/// ```text
/// [ a  b  tx ]   [ x ]   [ a*x + b*y + tx ]
/// [ c  d  ty ] × [ y ] = [ c*x + d*y + ty ]
///                [ 1 ]
/// ```
///
/// Identity: `a=1, b=0, tx=0, c=0, d=1, ty=0`.
#[derive(Clone, Copy, Debug)]
struct Mat23 {
    a: f32,
    b: f32,
    tx: f32,
    c: f32,
    d: f32,
    ty: f32,
}

impl Mat23 {
    fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            tx: 0.0,
            c: 0.0,
            d: 1.0,
            ty: 0.0,
        }
    }

    /// Compose: returns `self * rhs` (apply rhs first, then self).
    fn compose(self, rhs: Mat23) -> Mat23 {
        Mat23 {
            a: self.a * rhs.a + self.b * rhs.c,
            b: self.a * rhs.b + self.b * rhs.d,
            tx: self.a * rhs.tx + self.b * rhs.ty + self.tx,
            c: self.c * rhs.a + self.d * rhs.c,
            d: self.c * rhs.b + self.d * rhs.d,
            ty: self.c * rhs.tx + self.d * rhs.ty + self.ty,
        }
    }

    fn transform_point(self, x: f32, y: f32) -> (f32, f32) {
        (self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty)
    }

    fn transform_rect(self, r: PaintRect) -> PaintRect {
        // For axis-aligned scale+translate (the common case) map all four
        // corners and take the bounding box so rotated rects still work.
        let x0 = r.x as f32;
        let y0 = r.y as f32;
        let x1 = x0 + r.w as f32;
        let y1 = y0 + r.h as f32;
        let corners = [
            self.transform_point(x0, y0),
            self.transform_point(x1, y0),
            self.transform_point(x0, y1),
            self.transform_point(x1, y1),
        ];
        let min_x = corners.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let min_y = corners.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let max_x = corners.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
        let max_y = corners.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        PaintRect {
            x: min_x as i32,
            y: min_y as i32,
            w: (max_x - min_x).max(0.0) as u32,
            h: (max_y - min_y).max(0.0) as u32,
        }
    }
}

/// Intersect two `PaintRect`s.  Returns `None` if they don't overlap.
fn intersect_rects(a: PaintRect, b: PaintRect) -> Option<PaintRect> {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w as i32).min(b.x + b.w as i32);
    let y1 = (a.y + a.h as i32).min(b.y + b.h as i32);
    if x1 > x0 && y1 > y0 {
        Some(PaintRect {
            x: x0,
            y: y0,
            w: (x1 - x0) as u32,
            h: (y1 - y0) as u32,
        })
    } else {
        // No overlap — return a zero-area rect at the intersection edge so
        // subsequent clipping produces an empty span.
        Some(PaintRect {
            x: x0,
            y: y0,
            w: 0,
            h: 0,
        })
    }
}

pub fn rasterize(list: &DisplayList, width: u32, height: u32, bg: Bgra) -> Vec<u32> {
    let mut out = vec![bg.0; (width as usize) * (height as usize)];

    // Accumulated affine transform (identity = no transform).
    let mut mat: Mat23 = Mat23::identity();
    let mut mat_stack: Vec<Mat23> = Vec::new();

    let mut opacity = 1.0f32;
    let mut op_stack: Vec<f32> = Vec::new();

    // Active clip is the intersection of all pushed clips.
    let mut clip: Option<PaintRect> = None;
    let mut clip_stack: Vec<Option<PaintRect>> = Vec::new();

    for item in &list.items {
        match item {
            PaintItem::PushTransform {
                translate_x,
                translate_y,
                scale_x,
                scale_y,
            } => {
                mat_stack.push(mat);
                // Build a scale-then-translate matrix for the new layer and
                // compose it onto the current accumulated matrix.
                let local = Mat23 {
                    a: *scale_x,
                    b: 0.0,
                    tx: *translate_x,
                    c: 0.0,
                    d: *scale_y,
                    ty: *translate_y,
                };
                mat = mat.compose(local);
            }
            PaintItem::PopTransform => {
                if let Some(prev) = mat_stack.pop() {
                    mat = prev;
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
                // Transform the clip rect into screen space.
                let screen_clip = mat.transform_rect(*r);
                // Intersect with the current active clip (Bug 1 fix).
                let new_clip = if let Some(existing) = clip {
                    intersect_rects(existing, screen_clip)
                } else {
                    Some(screen_clip)
                };
                clip_stack.push(clip);
                clip = new_clip;
            }
            PaintItem::PopClip => {
                if let Some(c) = clip_stack.pop() {
                    clip = c;
                }
            }
            PaintItem::Fill { rect, color } => {
                let r = mat.transform_rect(*rect);
                blit_solid(&mut out, width, height, r, *color, opacity, clip);
            }
            PaintItem::LinearGradient {
                rect,
                from,
                to,
                angle_deg,
            } => {
                let r = mat.transform_rect(*rect);
                blit_linear_gradient(
                    &mut out, width, height, r, *from, *to, *angle_deg, opacity, clip,
                );
            }
            PaintItem::Image {
                rect,
                pixels,
                width: iw,
                height: ih,
            } => {
                let r = mat.transform_rect(*rect);
                blit_image(&mut out, width, height, r, pixels, *iw, *ih, opacity, clip);
            }
            PaintItem::BoxShadow {
                rect,
                color,
                offset_x,
                offset_y,
                spread,
                inset,
            } => {
                let base = mat.transform_rect(*rect);
                if *inset {
                    // Inset shadow: clip to the element box, paint 4 strips
                    // between the border box and the (spread-inset + offset)
                    // hole. No blur approximation here — just the filled shape.
                    let clip_r = if let Some(c) = clip {
                        // Intersect element box with active clip.
                        intersect_rects(base, c).unwrap_or(base)
                    } else {
                        base
                    };
                    let bx0 = base.x as i32;
                    let by0 = base.y as i32;
                    let bx1 = bx0 + base.w as i32;
                    let by1 = by0 + base.h as i32;
                    let hole_x0 = bx0 + *spread + *offset_x;
                    let hole_y0 = by0 + *spread + *offset_y;
                    let hole_x1 = bx1 - *spread + *offset_x;
                    let hole_y1 = by1 - *spread + *offset_y;
                    // Top strip
                    let top_strip = PaintRect {
                        x: bx0,
                        y: by0,
                        w: (bx1 - bx0).max(0) as u32,
                        h: (hole_y0 - by0).max(0) as u32,
                    };
                    blit_solid(&mut out, width, height, top_strip, *color, opacity, Some(clip_r));
                    // Bottom strip
                    let bot_strip = PaintRect {
                        x: bx0,
                        y: hole_y1,
                        w: (bx1 - bx0).max(0) as u32,
                        h: (by1 - hole_y1).max(0) as u32,
                    };
                    blit_solid(&mut out, width, height, bot_strip, *color, opacity, Some(clip_r));
                    // Left strip (between top/bottom)
                    let left_strip = PaintRect {
                        x: bx0,
                        y: hole_y0,
                        w: (hole_x0 - bx0).max(0) as u32,
                        h: (hole_y1 - hole_y0).max(0) as u32,
                    };
                    blit_solid(&mut out, width, height, left_strip, *color, opacity, Some(clip_r));
                    // Right strip
                    let right_strip = PaintRect {
                        x: hole_x1,
                        y: hole_y0,
                        w: (bx1 - hole_x1).max(0) as u32,
                        h: (hole_y1 - hole_y0).max(0) as u32,
                    };
                    blit_solid(&mut out, width, height, right_strip, *color, opacity, Some(clip_r));
                } else {
                    // Outset shadow: expand/contract by spread, then offset.
                    let r = PaintRect {
                        x: base.x - *spread,
                        y: base.y - *spread,
                        w: (base.w as i32 + 2 * *spread).max(0) as u32,
                        h: (base.h as i32 + 2 * *spread).max(0) as u32,
                    };
                    let mut r2 = r;
                    r2.x += *offset_x;
                    r2.y += *offset_y;
                    blit_solid(&mut out, width, height, r2, *color, opacity, clip);
                }
            }
            PaintItem::Border {
                rect,
                top,
                right,
                bottom,
                left,
                color,
            } => {
                let r = mat.transform_rect(*rect);
                // Top.
                blit_solid(
                    &mut out,
                    width,
                    height,
                    PaintRect {
                        x: r.x,
                        y: r.y,
                        w: r.w,
                        h: *top,
                    },
                    *color,
                    opacity,
                    clip,
                );
                // Bottom.
                blit_solid(
                    &mut out,
                    width,
                    height,
                    PaintRect {
                        x: r.x,
                        y: r.y + r.h as i32 - *bottom as i32,
                        w: r.w,
                        h: *bottom,
                    },
                    *color,
                    opacity,
                    clip,
                );
                // Left.
                blit_solid(
                    &mut out,
                    width,
                    height,
                    PaintRect {
                        x: r.x,
                        y: r.y,
                        w: *left,
                        h: r.h,
                    },
                    *color,
                    opacity,
                    clip,
                );
                // Right.
                blit_solid(
                    &mut out,
                    width,
                    height,
                    PaintRect {
                        x: r.x + r.w as i32 - *right as i32,
                        y: r.y,
                        w: *right,
                        h: r.h,
                    },
                    *color,
                    opacity,
                    clip,
                );
            }
            PaintItem::Text { .. } => {
                // Text rasterization stays in cv_gfx; the display
                // list serves text as already-rasterized images for
                // the compositor.
            }
        }
    }
    out
}

fn rect_clip(r: PaintRect, w: u32, h: u32, clip: Option<PaintRect>) -> (i32, i32, i32, i32) {
    let mut x0 = r.x.max(0);
    let mut y0 = r.y.max(0);
    let mut x1 = (r.x + r.w as i32).min(w as i32);
    let mut y1 = (r.y + r.h as i32).min(h as i32);
    if let Some(c) = clip {
        x0 = x0.max(c.x);
        y0 = y0.max(c.y);
        x1 = x1.min(c.x + c.w as i32);
        y1 = y1.min(c.y + c.h as i32);
    }
    (x0, y0, x1, y1)
}

fn blit_solid(
    out: &mut [u32],
    w: u32,
    h: u32,
    r: PaintRect,
    color: Bgra,
    opacity: f32,
    clip: Option<PaintRect>,
) {
    let (x0, y0, x1, y1) = rect_clip(r, w, h, clip);
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = (y as usize) * (w as usize) + x as usize;
            out[idx] = blend(out[idx], color.0, opacity);
        }
    }
}

/// Render a CSS linear gradient with the given `angle_deg`.
///
/// CSS angle convention: 0° = to top (gradient runs bottom→top),
/// 90° = to right, 180° = to bottom (default horizontal), 270° = to left.
///
/// For each pixel we project its position (relative to the rect centre) onto
/// the gradient axis to get `t ∈ [0, 1]`, then interpolate `from..to`.
fn blit_linear_gradient(
    out: &mut [u32],
    w: u32,
    h: u32,
    r: PaintRect,
    from: Bgra,
    to: Bgra,
    angle_deg: f32,
    opacity: f32,
    clip: Option<PaintRect>,
) {
    let (x0, y0, x1, y1) = rect_clip(r, w, h, clip);
    if x1 <= x0 || y1 <= y0 {
        return;
    }

    let angle_rad = angle_deg.to_radians();
    // CSS: the gradient axis direction unit vector (points toward the "to" end).
    let dx = angle_rad.sin(); // 90° → 1.0 = rightward
    let dy = -angle_rad.cos(); // 90° → 0.0; 0° → -1 = upward

    // Half-extents of the rect, used to compute gradient line length.
    let hw = r.w as f32 * 0.5;
    let hh = r.h as f32 * 0.5;

    // The gradient line length = the extent of the rect along the axis
    // (same formula Chrome uses for the "gradient line" length).
    let gradient_length = (dx * hw).abs() + (dy * hh).abs();
    // Avoid division by zero for degenerate rects.
    let inv_len = if gradient_length > 0.0 { 1.0 / gradient_length } else { 1.0 };

    // Centre of the (transformed) rect — the t=0.5 midpoint of the gradient.
    let cx = r.x as f32 + hw;
    let cy = r.y as f32 + hh;

    for y in y0..y1 {
        for x in x0..x1 {
            // Project (pixel − centre) onto the gradient axis.
            let px = x as f32 - cx;
            let py = y as f32 - cy;
            let proj = px * dx + py * dy;
            // proj ranges from −gradient_length to +gradient_length;
            // map to [0, 1].
            let t = (proj * inv_len * 0.5 + 0.5).clamp(0.0, 1.0);
            let lerp = lerp_bgra(from.0, to.0, t);
            let idx = (y as usize) * (w as usize) + x as usize;
            out[idx] = blend(out[idx], lerp, opacity);
        }
    }
}

fn blit_image(
    out: &mut [u32],
    w: u32,
    h: u32,
    r: PaintRect,
    pixels: &[u8],
    iw: u32,
    ih: u32,
    opacity: f32,
    clip: Option<PaintRect>,
) {
    let (x0, y0, x1, y1) = rect_clip(r, w, h, clip);
    for y in y0..y1 {
        for x in x0..x1 {
            let sx = (((x - r.x) as f32 / r.w as f32) * iw as f32) as u32;
            let sy = (((y - r.y) as f32 / r.h as f32) * ih as f32) as u32;
            if sx >= iw || sy >= ih {
                continue;
            }
            let p = ((sy * iw + sx) * 4) as usize;
            if p + 3 >= pixels.len() {
                continue;
            }
            let src = ((pixels[p + 3] as u32) << 24)
                | ((pixels[p + 2] as u32) << 16)
                | ((pixels[p + 1] as u32) << 8)
                | pixels[p] as u32;
            let idx = (y as usize) * (w as usize) + x as usize;
            out[idx] = blend(out[idx], src, opacity);
        }
    }
}

fn lerp_bgra(a: u32, b: u32, t: f32) -> u32 {
    let ar = ((a >> 16) & 0xFF) as f32;
    let ag = ((a >> 8) & 0xFF) as f32;
    let ab = (a & 0xFF) as f32;
    let aa = ((a >> 24) & 0xFF) as f32;
    let br = ((b >> 16) & 0xFF) as f32;
    let bg = ((b >> 8) & 0xFF) as f32;
    let bb = (b & 0xFF) as f32;
    let ba = ((b >> 24) & 0xFF) as f32;
    let r = (ar + (br - ar) * t) as u32;
    let g = (ag + (bg - ag) * t) as u32;
    let bb_ = (ab + (bb - ab) * t) as u32;
    let aaa = (aa + (ba - aa) * t) as u32;
    (aaa << 24) | (r << 16) | (g << 8) | bb_
}

fn blend(dst: u32, src: u32, opacity: f32) -> u32 {
    let sa = (((src >> 24) & 0xFF) as f32 / 255.0) * opacity;
    if sa <= 0.0 {
        return dst;
    }
    let sr = ((src >> 16) & 0xFF) as f32 / 255.0;
    let sg = ((src >> 8) & 0xFF) as f32 / 255.0;
    let sb = (src & 0xFF) as f32 / 255.0;
    let dr = ((dst >> 16) & 0xFF) as f32 / 255.0;
    let dg = ((dst >> 8) & 0xFF) as f32 / 255.0;
    let db = (dst & 0xFF) as f32 / 255.0;
    let da = ((dst >> 24) & 0xFF) as f32 / 255.0;
    let oa = sa + da * (1.0 - sa);
    let or = sr * sa + dr * da * (1.0 - sa);
    let og = sg * sa + dg * da * (1.0 - sa);
    let ob = sb * sa + db * da * (1.0 - sa);
    let q = |c: f32| -> u32 { (c.clamp(0.0, 1.0) * 255.0) as u32 };
    (q(oa) << 24) | (q(or) << 16) | (q(og) << 8) | q(ob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_paint::{Bgra, DisplayList, PaintItem, PaintRect};

    #[test]
    fn fill_paints_pixels() {
        let mut list = DisplayList::new();
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
            color: Bgra::new(255, 0, 0, 255),
        });
        let pixels = rasterize(&list, 4, 4, Bgra::new(0, 0, 0, 255));
        for p in pixels {
            assert_eq!(p, 0xFFFF0000);
        }
    }

    #[test]
    fn transform_offsets_fill() {
        let mut list = DisplayList::new();
        list.push(PaintItem::PushTransform {
            translate_x: 4.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
            },
            color: Bgra::new(255, 0, 0, 255),
        });
        list.push(PaintItem::PopTransform);
        let pixels = rasterize(&list, 8, 4, Bgra::new(0, 0, 0, 255));
        // (0,0) untouched, (4,0) painted red.
        assert_eq!(pixels[0], 0xFF000000);
        assert_eq!(pixels[4], 0xFFFF0000);
    }

    #[test]
    fn clip_bounds_fill() {
        let mut list = DisplayList::new();
        list.push(PaintItem::PushClip(PaintRect {
            x: 0,
            y: 0,
            w: 2,
            h: 2,
        }));
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            color: Bgra::new(255, 0, 0, 255),
        });
        list.push(PaintItem::PopClip);
        let pixels = rasterize(&list, 8, 8, Bgra::new(0, 0, 0, 255));
        // Only (0..2, 0..2) painted.
        assert_eq!(pixels[0], 0xFFFF0000);
        assert_eq!(pixels[2], 0xFF000000); // outside clip
        assert_eq!(pixels[8 * 2], 0xFF000000);
    }

    #[test]
    fn opacity_blends_with_background() {
        let mut list = DisplayList::new();
        list.push(PaintItem::PushOpacity(0.5));
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
            color: Bgra::new(255, 255, 255, 255),
        });
        list.push(PaintItem::PopOpacity);
        let pixels = rasterize(&list, 1, 1, Bgra::new(0, 0, 0, 255));
        let r = (pixels[0] >> 16) & 0xFF;
        assert!((100..=160).contains(&r));
    }

    #[test]
    fn border_paints_four_sides() {
        let mut list = DisplayList::new();
        list.push(PaintItem::Border {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
            top: 1,
            right: 1,
            bottom: 1,
            left: 1,
            color: Bgra::new(255, 0, 0, 255),
        });
        let pixels = rasterize(&list, 4, 4, Bgra::new(0, 0, 0, 255));
        // Corners painted (top + left).
        assert_eq!(pixels[0], 0xFFFF0000);
        // Center not painted.
        assert_eq!(pixels[5], 0xFF000000);
    }

    // ---- New regression tests for the three fixed bugs ----

    /// Bug 1: nested clips must intersect, not replace.
    /// Outer clip = columns 0..4, inner clip = columns 2..6.
    /// Intersection = columns 2..4.  A fill covering 0..8 should only land in 2..4.
    #[test]
    fn nested_clips_intersect() {
        let mut list = DisplayList::new();
        list.push(PaintItem::PushClip(PaintRect {
            x: 0,
            y: 0,
            w: 4,
            h: 8,
        }));
        list.push(PaintItem::PushClip(PaintRect {
            x: 2,
            y: 0,
            w: 4, // x: 2..6
            h: 8,
        }));
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            color: Bgra::new(255, 0, 0, 255),
        });
        list.push(PaintItem::PopClip);
        list.push(PaintItem::PopClip);
        let pixels = rasterize(&list, 8, 8, Bgra::new(0, 0, 0, 255));
        // x=0, x=1 — outside intersection (outer clip allows, inner does not cover 0..2 but
        // inner starts at 2, outer ends at 4 → intersection is [2,4)).
        assert_eq!(pixels[0], 0xFF000000, "x=0 must be unpainted");
        assert_eq!(pixels[1], 0xFF000000, "x=1 must be unpainted");
        assert_eq!(pixels[2], 0xFFFF0000, "x=2 must be painted");
        assert_eq!(pixels[3], 0xFFFF0000, "x=3 must be painted");
        // x=4 is outside the outer clip (outer width=4 means 0..4), so unpainted.
        assert_eq!(pixels[4], 0xFF000000, "x=4 must be unpainted (outside outer clip)");
    }

    /// Bug 2: nested transforms must compose — parent scale must multiply child translation.
    /// Parent: scale_x=2, translate_x=0.
    /// Child:  scale_x=1, translate_x=10.
    /// A rect at local (0,0,2,2) inside the child should appear at screen x=20 (not x=10).
    #[test]
    fn nested_transforms_compose_scale_times_translation() {
        let mut list = DisplayList::new();
        // Parent: scale by 2 (no translate).
        list.push(PaintItem::PushTransform {
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 2.0,
            scale_y: 1.0,
        });
        // Child: translate by 10.
        list.push(PaintItem::PushTransform {
            translate_x: 10.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
            },
            color: Bgra::new(255, 0, 0, 255),
        });
        list.push(PaintItem::PopTransform);
        list.push(PaintItem::PopTransform);
        // Canvas 40 wide so the rect fits either at x=10 (wrong) or x=20 (correct).
        let pixels = rasterize(&list, 40, 4, Bgra::new(0, 0, 0, 255));
        // x=10 should be black (wrong position, old bug).
        assert_eq!(pixels[10], 0xFF000000, "x=10 must be unpainted (parent scale not applied)");
        // x=20 and x=21 should be red (correct composition).
        assert_eq!(pixels[20], 0xFFFF0000, "x=20 must be painted (parent scale*child translate)");
        assert_eq!(pixels[21], 0xFFFF0000, "x=21 must be painted");
    }

    /// Bug 3: linear gradient with 90° (left→right) must vary horizontally.
    /// With angle=90°, `from` should appear on the left and `to` on the right.
    #[test]
    fn gradient_90deg_is_horizontal() {
        let mut list = DisplayList::new();
        list.push(PaintItem::LinearGradient {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 8,
                h: 1,
            },
            from: Bgra::new(255, 0, 0, 255), // red
            to: Bgra::new(0, 0, 255, 255),   // blue
            angle_deg: 90.0,
        });
        let pixels = rasterize(&list, 8, 1, Bgra::new(0, 0, 0, 255));
        let left_r = (pixels[0] >> 16) & 0xFF;
        let right_r = (pixels[7] >> 16) & 0xFF;
        let left_b = pixels[0] & 0xFF;
        let right_b = pixels[7] & 0xFF;
        assert!(left_r > 200, "left pixel should be mostly red, got r={left_r}");
        assert!(left_b < 55, "left pixel should have little blue, got b={left_b}");
        assert!(right_b > 200, "right pixel should be mostly blue, got b={right_b}");
        assert!(right_r < 55, "right pixel should have little red, got r={right_r}");
    }

    /// Bug 3 continued: with angle=0° the gradient runs bottom→top (CSS convention).
    /// The top row should show `from` colours; the bottom row should show `to` colours.
    #[test]
    fn gradient_0deg_is_bottom_to_top() {
        let mut list = DisplayList::new();
        list.push(PaintItem::LinearGradient {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 1,
                h: 8,
            },
            from: Bgra::new(255, 0, 0, 255), // red  (start = bottom)
            to: Bgra::new(0, 0, 255, 255),   // blue (end   = top)
            angle_deg: 0.0,
        });
        let pixels = rasterize(&list, 1, 8, Bgra::new(0, 0, 0, 255));
        // y=0 is the top row → should be near `to` (blue).
        let top_b = pixels[0] & 0xFF;
        let top_r = (pixels[0] >> 16) & 0xFF;
        // y=7 is the bottom row → should be near `from` (red).
        let bot_r = (pixels[7] >> 16) & 0xFF;
        let bot_b = pixels[7] & 0xFF;
        assert!(top_b > 200, "top pixel should be mostly blue (to), got b={top_b}");
        assert!(top_r < 55, "top pixel should have little red, got r={top_r}");
        assert!(bot_r > 200, "bottom pixel should be mostly red (from), got r={bot_r}");
        assert!(bot_b < 55, "bottom pixel should have little blue, got b={bot_b}");
    }

    /// Bug 3 continued: the old horizontal default (180°) still works.
    #[test]
    fn gradient_180deg_is_top_to_bottom() {
        let mut list = DisplayList::new();
        list.push(PaintItem::LinearGradient {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 1,
                h: 8,
            },
            from: Bgra::new(255, 0, 0, 255), // red
            to: Bgra::new(0, 0, 255, 255),   // blue
            angle_deg: 180.0,
        });
        let pixels = rasterize(&list, 1, 8, Bgra::new(0, 0, 0, 255));
        let top_r = (pixels[0] >> 16) & 0xFF;
        let top_b = pixels[0] & 0xFF;
        let bot_b = pixels[7] & 0xFF;
        let bot_r = (pixels[7] >> 16) & 0xFF;
        assert!(top_r > 200, "top pixel should be mostly red (from), got r={top_r}");
        assert!(top_b < 55, "top pixel should have little blue, got b={top_b}");
        assert!(bot_b > 200, "bottom pixel should be mostly blue (to), got b={bot_b}");
        assert!(bot_r < 55, "bottom pixel should have little red, got r={bot_r}");
    }
}
