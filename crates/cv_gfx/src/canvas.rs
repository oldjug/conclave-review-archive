//! `Canvas2D` drawing context.
//!
//! Mirrors the Web `CanvasRenderingContext2D` shape closely enough that
//! the JS bindings can pass calls through with minimal translation. The
//! state model (transform / fillStyle / strokeStyle / lineWidth) is
//! kept on a stack so `save()` / `restore()` Just Work. Path ops are
//! flattened into a Vec of segments and stroked / filled via the
//! existing `Bitmap` primitives.

use crate::font5x7;
use crate::{Bitmap, Color, blend_bgra};

/// A single color stop in a Canvas gradient (offset in [0.0, 1.0]).
#[derive(Debug, Clone)]
pub struct GradientStop {
    pub offset: f32,
    pub color: Color,
}

/// How a `createPattern(image, repetition)` source tiles when used as a
/// fill / stroke style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternRepeat {
    Repeat,
    RepeatX,
    RepeatY,
    NoRepeat,
}

impl PatternRepeat {
    /// Parse the `repetition` argument to `createPattern`. The empty string,
    /// `null`, and the literal `"repeat"` all map to `Repeat` (the default).
    pub fn from_str(s: &str) -> Self {
        match s {
            "repeat-x" => Self::RepeatX,
            "repeat-y" => Self::RepeatY,
            "no-repeat" => Self::NoRepeat,
            _ => Self::Repeat,
        }
    }
}

/// A tiled-image pattern source for `createPattern`. The image is stored as a
/// row-major BGRA u32 buffer of `width`×`height`. Sampling is in *destination
/// pixel space* (the same space the rasterizer iterates), which matches the
/// common no-transform pattern use; pattern-local transform is not applied
/// (documented divergence from the full spec's DOMMatrix-on-pattern feature).
#[derive(Debug, Clone)]
pub struct Pattern {
    pub width: u32,
    pub height: u32,
    pub pixels: std::rc::Rc<Vec<u32>>,
    pub repeat: PatternRepeat,
}

impl Pattern {
    /// Sample the pattern at destination pixel (px, py). Returns `None` when
    /// the point falls outside a non-repeating axis (so the rasterizer leaves
    /// the destination untouched there).
    fn sample(&self, px: i32, py: i32, global_alpha: f32) -> Option<Color> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let w = self.width as i32;
        let h = self.height as i32;
        let sx = match self.repeat {
            PatternRepeat::Repeat | PatternRepeat::RepeatX => px.rem_euclid(w),
            PatternRepeat::RepeatY | PatternRepeat::NoRepeat => {
                if px < 0 || px >= w {
                    return None;
                }
                px
            }
        };
        let sy = match self.repeat {
            PatternRepeat::Repeat | PatternRepeat::RepeatY => py.rem_euclid(h),
            PatternRepeat::RepeatX | PatternRepeat::NoRepeat => {
                if py < 0 || py >= h {
                    return None;
                }
                py
            }
        };
        let v = self.pixels[(sy as usize) * (self.width as usize) + sx as usize];
        let mut c = Color {
            r: ((v >> 16) & 0xFF) as u8,
            g: ((v >> 8) & 0xFF) as u8,
            b: (v & 0xFF) as u8,
            a: ((v >> 24) & 0xFF) as u8,
        };
        c.a = ((c.a as f32) * global_alpha) as u8;
        Some(c)
    }
}

/// The fill style for a Canvas 2D context — matches the JS `fillStyle` property.
#[derive(Debug, Clone)]
pub enum FillStyle {
    /// Solid RGBA color (the default).
    Color(Color),
    /// `createLinearGradient(x0, y0, x1, y1)` with N color stops.
    Linear {
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        stops: Vec<GradientStop>,
    },
    /// `createRadialGradient(x0, y0, r0, x1, y1, r1)` — the full two-circle
    /// form. The gradient is the cone swept between the start circle
    /// (x0, y0, r0) and the end circle (x1, y1, r1); each painted point's
    /// offset is the largest ω such that the point lies on the interpolated
    /// circle centered at ((1-ω)x0 + ω·x1, (1-ω)y0 + ω·y1) with radius
    /// (1-ω)r0 + ω·r1 (WHATWG Canvas §4.12.5.1.10). Concentric and focal
    /// (non-concentric) gradients are both handled exactly.
    Radial {
        x0: f32,
        y0: f32,
        r0: f32,
        x1: f32,
        y1: f32,
        r1: f32,
        stops: Vec<GradientStop>,
    },
    /// `createPattern(image, repetition)` — a tiled image source.
    Pattern(Pattern),
}

impl FillStyle {
    /// Sample the fill style at a given canvas point (px, py), returning the
    /// blended Color for that position.
    pub fn sample(&self, px: f32, py: f32, global_alpha: f32) -> Color {
        match self {
            FillStyle::Color(c) => {
                let mut out = *c;
                out.a = ((out.a as f32) * global_alpha) as u8;
                out
            }
            FillStyle::Linear { x0, y0, x1, y1, stops } => {
                if stops.is_empty() {
                    return Color::TRANSPARENT;
                }
                let dx = x1 - x0;
                let dy = y1 - y0;
                let len2 = dx * dx + dy * dy;
                let t = if len2 < 1e-10 {
                    0.0f32
                } else {
                    ((px - x0) * dx + (py - y0) * dy) / len2
                }
                .clamp(0.0, 1.0);
                sample_stops(stops, t, global_alpha)
            }
            FillStyle::Radial { x0, y0, r0, x1, y1, r1, stops } => {
                if stops.is_empty() {
                    return Color::TRANSPARENT;
                }
                match radial_offset(*x0, *y0, *r0, *x1, *y1, *r1, px, py) {
                    Some(omega) => sample_stops(stops, omega.clamp(0.0, 1.0), global_alpha),
                    // No interpolated circle with non-negative radius contains
                    // the point → the point is not painted (spec: transparent).
                    None => Color::TRANSPARENT,
                }
            }
            FillStyle::Pattern(p) => {
                // Sample at the pixel the destination loop is filling.
                p.sample(px.floor() as i32, py.floor() as i32, global_alpha)
                    .unwrap_or(Color::TRANSPARENT)
            }
        }
    }
}

/// Interpolate through `stops` at position `t` in [0, 1].
pub(crate) fn sample_stops(stops: &[GradientStop], t: f32, global_alpha: f32) -> Color {
    debug_assert!(!stops.is_empty());
    if stops.len() == 1 {
        let mut c = stops[0].color;
        c.a = ((c.a as f32) * global_alpha) as u8;
        return c;
    }
    // Find the two bracketing stops.
    let first = &stops[0];
    let last = &stops[stops.len() - 1];
    if t <= first.offset {
        let mut c = first.color;
        c.a = ((c.a as f32) * global_alpha) as u8;
        return c;
    }
    if t >= last.offset {
        let mut c = last.color;
        c.a = ((c.a as f32) * global_alpha) as u8;
        return c;
    }
    for w in stops.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if t >= a.offset && t <= b.offset {
            let span = (b.offset - a.offset).max(1e-9);
            let local_t = (t - a.offset) / span;
            let r = (a.color.r as f32 * (1.0 - local_t) + b.color.r as f32 * local_t) as u8;
            let g = (a.color.g as f32 * (1.0 - local_t) + b.color.g as f32 * local_t) as u8;
            let bl = (a.color.b as f32 * (1.0 - local_t) + b.color.b as f32 * local_t) as u8;
            let al = (a.color.a as f32 * (1.0 - local_t) + b.color.a as f32 * local_t) as u8;
            let raw_a = ((al as f32) * global_alpha) as u8;
            return Color { r, g, b: bl, a: raw_a };
        }
    }
    // Fallback (shouldn't happen if stops are sorted).
    let mut c = last.color;
    c.a = ((c.a as f32) * global_alpha) as u8;
    c
}

/// Compute the radial-gradient offset ω for a point (px, py) given the start
/// circle (x0, y0, r0) and end circle (x1, y1, r1).
///
/// WHATWG Canvas §4.12.5.1.10 "If radial: for all values of ω where r(ω) > 0,
/// starting with the value of ω nearest to positive infinity and ending with
/// the value of ω nearest to negative infinity, draw the circumference of the
/// circle with radius r(ω) at position (x(ω), y(ω)) [...]" — i.e. the painted
/// offset at a pixel is the **largest** ω for which the pixel lies on the
/// interpolated circle of non-negative radius. We solve the resulting quadratic
/// in ω directly:
///
///   x(ω) = (1-ω)·x0 + ω·x1,  y(ω) = (1-ω)·y0 + ω·y1,  r(ω) = (1-ω)·r0 + ω·r1
///
/// Point on circle ⇔ |P - C(ω)|² = r(ω)². With d = P1-P0, dr = r1-r0, f = P-P0:
///   a·ω² + b·ω + c = 0,
///   a = d·d - dr²,  b = -2(f·d + r0·dr),  c = f·f - r0²
///
/// Returns the largest root with r(ω) ≥ 0, or `None` when no such root exists
/// (the pixel is outside the painted cone and stays transparent). The returned
/// value is NOT clamped — callers clamp into [0,1] before sampling stops.
#[inline]
fn radial_offset(
    x0: f32,
    y0: f32,
    r0: f32,
    x1: f32,
    y1: f32,
    r1: f32,
    px: f32,
    py: f32,
) -> Option<f32> {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dr = r1 - r0;
    let fx = px - x0;
    let fy = py - y0;
    let a = dx * dx + dy * dy - dr * dr;
    let b = -2.0 * (fx * dx + fy * dy + r0 * dr);
    let c = fx * fx + fy * fy - r0 * r0;

    // r(ω) ≥ 0 guard.
    let radius_ok = |w: f32| r0 + w * dr >= 0.0;

    if a.abs() < 1e-7 {
        // Degenerate to linear in ω (the "spotlight" cone where |d| == |dr|, or
        // concentric circles where d == 0).
        if b.abs() < 1e-12 {
            // a == b == 0: either no solution (c != 0) or the point is the cone
            // apex / focal point (c == 0), satisfied by every ω. The apex is the
            // zero-radius circle, i.e. ω where r(ω) = 0 → ω = -r0/dr (offset 0
            // when r0 == 0). Falls back to 0 for fully concentric same-radius.
            if c.abs() < 1e-6 {
                if dr.abs() > 1e-9 {
                    let w = -r0 / dr;
                    return if radius_ok(w) { Some(w) } else { Some(0.0) };
                }
                return Some(0.0);
            }
            return None;
        }
        let w = -c / b;
        return if radius_ok(w) { Some(w) } else { None };
    }

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let w1 = (-b + sqrt_disc) / (2.0 * a);
    let w2 = (-b - sqrt_disc) / (2.0 * a);
    // Prefer the larger ω (spec walks ω from +∞ downward; the first circle that
    // contains the pixel — i.e. the largest valid ω — wins).
    let (hi, lo) = if w1 >= w2 { (w1, w2) } else { (w2, w1) };
    if radius_ok(hi) {
        Some(hi)
    } else if radius_ok(lo) {
        Some(lo)
    } else {
        None
    }
}

/// Approximate a Gaussian blur of standard deviation `sigma` over a single-
/// channel coverage buffer (`buf`, row-major `w`×`h`, values in [0,1]) using
/// three successive box blurs, per SVG 1.1 §15.17 "feGaussianBlur" (the same
/// algorithm Skia's `SkBlurMask` fast path uses):
///
/// > "if d is odd, use three box-blurs of size 'd', centered on the output
/// >  pixel. if d is even, use two box-blurs of size 'd' (the first one
/// >  centered on the pixel boundary between the output pixel and the one to the
/// >  left, the second one centered on the pixel boundary between the output
/// >  pixel and the one to the right) and one box blur of size 'd+1' centered on
/// >  the output pixel."
/// >  with d = floor(s * 3 * sqrt(2*PI)/4 + 0.5)
///
/// The three passes are applied independently in the horizontal and vertical
/// directions (the box blur is separable). No-ops for sigma below ~0.05.
fn box_blur_gaussian(buf: &mut [f32], w: usize, h: usize, sigma: f32) {
    if sigma < 0.05 || w == 0 || h == 0 {
        return;
    }
    // d per the SVG spec.
    let d = (sigma * 3.0 * (2.0 * std::f32::consts::PI).sqrt() / 4.0 + 0.5).floor() as i32;
    if d < 1 {
        return;
    }
    let mut tmp = vec![0.0f32; buf.len()];
    // The three (left-radius, right-radius) box passes.
    let passes: [(i32, i32); 3] = if d % 2 == 1 {
        let r = d / 2;
        [(r, r), (r, r), (r, r)]
    } else {
        let r = d / 2;
        // size d centered on left boundary → radii (r, r-1);
        // size d centered on right boundary → radii (r-1, r);
        // size d+1 centered → radii (r, r).
        [(r, r - 1), (r - 1, r), (r, r)]
    };
    for &(lr, rr) in &passes {
        // Horizontal pass: buf → tmp.
        box_blur_h(buf, &mut tmp, w, h, lr, rr);
        // Vertical pass: tmp → buf.
        box_blur_v(&tmp, buf, w, h, lr, rr);
    }
}

/// One horizontal box-blur pass with the given left/right radii (averaging
/// window width = lr + rr + 1), using a sliding-window running sum. Edges are
/// clamped to zero outside the buffer (the buffer is padded by the caller so
/// the shape never touches the border).
fn box_blur_h(src: &[f32], dst: &mut [f32], w: usize, h: usize, lr: i32, rr: i32) {
    let width = (lr + rr + 1) as f32;
    let iw = w as i32;
    for y in 0..h {
        let row = y * w;
        let mut sum = 0.0f32;
        // Prime the window for x = 0: indices [-lr, rr].
        for k in -lr..=rr {
            if k >= 0 && k < iw {
                sum += src[row + k as usize];
            }
        }
        for x in 0..w as i32 {
            dst[row + x as usize] = sum / width;
            // Slide: drop (x - lr), add (x + rr + 1).
            let out_idx = x - lr;
            let in_idx = x + rr + 1;
            if out_idx >= 0 && out_idx < iw {
                sum -= src[row + out_idx as usize];
            }
            if in_idx >= 0 && in_idx < iw {
                sum += src[row + in_idx as usize];
            }
        }
    }
}

/// One vertical box-blur pass with the given top/bottom radii.
fn box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, lr: i32, rr: i32) {
    let width = (lr + rr + 1) as f32;
    let ih = h as i32;
    for x in 0..w {
        let mut sum = 0.0f32;
        for k in -lr..=rr {
            if k >= 0 && k < ih {
                sum += src[(k as usize) * w + x];
            }
        }
        for y in 0..h as i32 {
            dst[(y as usize) * w + x] = sum / width;
            let out_idx = y - lr;
            let in_idx = y + rr + 1;
            if out_idx >= 0 && out_idx < ih {
                sum -= src[(out_idx as usize) * w + x];
            }
            if in_idx >= 0 && in_idx < ih {
                sum += src[(in_idx as usize) * w + x];
            }
        }
    }
}

/// An axis-aligned rectangle clip region stored on the state stack.
/// All drawing operations are clipped to this region when set.
#[derive(Debug, Clone, Copy)]
pub struct ClipRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

/// One drawing operation in the current path. Curves get flattened
/// into line segments at insertion time so the renderer below only
/// sees straight runs — keeps the raster path simple for V1.
#[derive(Debug, Clone)]
pub enum PathOp {
    MoveTo(f32, f32),
    LineTo(f32, f32),
    Close,
}

/// Fill rule for path filling — controls how the inside of a path is determined
/// when sub-paths overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRule {
    /// Nonzero winding rule (HTML Canvas 2D default): a point is inside if the
    /// net winding count from all edges around it is non-zero.
    Nonzero,
    /// Even-odd rule: a point is inside if the number of edges that cross a ray
    /// from it to infinity is odd.
    EvenOdd,
}

impl FillRule {
    /// Parse the string argument to `ctx.fill(fillRule)`.  Returns `Nonzero`
    /// for the empty string, `"nonzero"`, or any unrecognised value, and
    /// `EvenOdd` only for the literal `"evenodd"`.
    pub fn from_str(s: &str) -> Self {
        if s == "evenodd" { Self::EvenOdd } else { Self::Nonzero }
    }
}

/// `globalCompositeOperation` — how a source pixel combines with the
/// destination pixel already in the bitmap.
///
/// The default `SourceOver` MUST route to the existing `blend_bgra`
/// straight-alpha source-over so the common path stays byte-identical to the
/// pre-feature build. All other modes go through [`composite_pixel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeOp {
    // Porter-Duff modes.
    SourceOver,
    SourceIn,
    SourceOut,
    SourceAtop,
    DestinationOver,
    DestinationIn,
    DestinationOut,
    DestinationAtop,
    Copy,
    Xor,
    Lighter,
    // Separable blend modes (W3C Compositing & Blending Level 1).
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
}

impl CompositeOp {
    /// Parse the JS `globalCompositeOperation` string. Unknown values keep the
    /// current operation per spec — callers that want that behavior should use
    /// [`CompositeOp::try_from_str`] instead; this convenience returns
    /// `SourceOver` for unknown values.
    pub fn from_str(s: &str) -> Self {
        Self::try_from_str(s).unwrap_or(Self::SourceOver)
    }

    /// Parse the JS `globalCompositeOperation` string, returning `None` for an
    /// unrecognised value (so the property can be left unchanged per spec).
    pub fn try_from_str(s: &str) -> Option<Self> {
        Some(match s {
            "source-over" => Self::SourceOver,
            "source-in" => Self::SourceIn,
            "source-out" => Self::SourceOut,
            "source-atop" => Self::SourceAtop,
            "destination-over" => Self::DestinationOver,
            "destination-in" => Self::DestinationIn,
            "destination-out" => Self::DestinationOut,
            "destination-atop" => Self::DestinationAtop,
            "copy" => Self::Copy,
            "xor" => Self::Xor,
            "lighter" => Self::Lighter,
            "multiply" => Self::Multiply,
            "screen" => Self::Screen,
            "overlay" => Self::Overlay,
            "darken" => Self::Darken,
            "lighten" => Self::Lighten,
            "color-dodge" => Self::ColorDodge,
            "color-burn" => Self::ColorBurn,
            "hard-light" => Self::HardLight,
            "soft-light" => Self::SoftLight,
            "difference" => Self::Difference,
            "exclusion" => Self::Exclusion,
            _ => return None,
        })
    }
}

/// Composite a straight-alpha source `Color` over a packed BGRA destination
/// pixel using the given [`CompositeOp`], returning the new packed BGRA pixel.
///
/// `SourceOver` is intentionally delegated to `blend_bgra` so it is
/// byte-identical to the legacy path. Every other mode uses the general
/// Porter-Duff / separable-blend math from the W3C Compositing & Blending
/// spec, computed in straight (non-premultiplied) alpha then re-packed.
pub(crate) fn composite_pixel(dst: u32, src: Color, op: CompositeOp) -> u32 {
    // The hot, common path: byte-for-byte the same as before this feature.
    if matches!(op, CompositeOp::SourceOver) {
        if src.a == 0 {
            return dst;
        }
        if src.a == 255 {
            return src.to_bgra_u32();
        }
        return crate::blend_bgra(dst, src);
    }

    let da = ((dst >> 24) & 0xFF) as f32 / 255.0;
    let dr = ((dst >> 16) & 0xFF) as f32 / 255.0;
    let dg = ((dst >> 8) & 0xFF) as f32 / 255.0;
    let db = (dst & 0xFF) as f32 / 255.0;
    let sa = src.a as f32 / 255.0;
    let sr = src.r as f32 / 255.0;
    let sg = src.g as f32 / 255.0;
    let sb = src.b as f32 / 255.0;

    // Porter-Duff coverage coefficients (Fa, Fb): out = sa*Fa*Cs + da*Fb*Cb,
    // out_a = sa*Fa + da*Fb. The separable blend modes share source-over
    // coverage but replace Cs with B(Cb, Cs) blended by da.
    let (fa, fb): (f32, f32) = match op {
        CompositeOp::SourceOver => (1.0, 1.0 - sa),
        CompositeOp::SourceIn => (da, 0.0),
        CompositeOp::SourceOut => (1.0 - da, 0.0),
        CompositeOp::SourceAtop => (da, 1.0 - sa),
        CompositeOp::DestinationOver => (1.0 - da, 1.0),
        CompositeOp::DestinationIn => (0.0, sa),
        CompositeOp::DestinationOut => (0.0, 1.0 - sa),
        CompositeOp::DestinationAtop => (1.0 - da, sa),
        CompositeOp::Copy => (1.0, 0.0),
        CompositeOp::Xor => (1.0 - da, 1.0 - sa),
        CompositeOp::Lighter => (1.0, 1.0),
        // Separable blend modes: handled below.
        _ => (1.0, 1.0 - sa),
    };

    let separable = !matches!(
        op,
        CompositeOp::SourceOver
            | CompositeOp::SourceIn
            | CompositeOp::SourceOut
            | CompositeOp::SourceAtop
            | CompositeOp::DestinationOver
            | CompositeOp::DestinationIn
            | CompositeOp::DestinationOut
            | CompositeOp::DestinationAtop
            | CompositeOp::Copy
            | CompositeOp::Xor
            | CompositeOp::Lighter
    );

    let out_a = (sa * fa + da * fb).clamp(0.0, 1.0);
    if out_a <= 0.0 {
        return 0;
    }

    // For each channel produce a straight-alpha output value.
    let chan = |cs: f32, cb: f32| -> f32 {
        let cs_eff = if separable {
            // Per spec: Cs <- (1 - da)*Cs + da*B(Cb, Cs).
            let blended = blend_separable(op, cb, cs);
            (1.0 - da) * cs + da * blended
        } else {
            cs
        };
        // Composited premultiplied result, normalized by out_a to straight alpha.
        ((sa * fa * cs_eff) + (da * fb * cb)) / out_a
    };

    let r = (chan(sr, dr).clamp(0.0, 1.0) * 255.0).round() as u32;
    let g = (chan(sg, dg).clamp(0.0, 1.0) * 255.0).round() as u32;
    let b = (chan(sb, db).clamp(0.0, 1.0) * 255.0).round() as u32;
    let a = (out_a * 255.0).round() as u32;
    (a << 24) | (r << 16) | (g << 8) | b
}

/// The separable per-channel blend function B(Cb, Cs) from the W3C
/// Compositing & Blending spec. Inputs and output are in [0, 1].
fn blend_separable(op: CompositeOp, cb: f32, cs: f32) -> f32 {
    match op {
        CompositeOp::Multiply => cb * cs,
        CompositeOp::Screen => cb + cs - cb * cs,
        CompositeOp::Overlay => blend_separable(CompositeOp::HardLight, cs, cb),
        CompositeOp::Darken => cb.min(cs),
        CompositeOp::Lighten => cb.max(cs),
        CompositeOp::ColorDodge => {
            if cb == 0.0 {
                0.0
            } else if cs >= 1.0 {
                1.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
        CompositeOp::ColorBurn => {
            if cb >= 1.0 {
                1.0
            } else if cs <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - cb) / cs).min(1.0)
            }
        }
        CompositeOp::HardLight => {
            if cs <= 0.5 {
                cb * (2.0 * cs)
            } else {
                let s = 2.0 * cs - 1.0;
                cb + s - cb * s
            }
        }
        CompositeOp::SoftLight => {
            if cs <= 0.5 {
                cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
            } else {
                let d = if cb <= 0.25 {
                    ((16.0 * cb - 12.0) * cb + 4.0) * cb
                } else {
                    cb.sqrt()
                };
                cb + (2.0 * cs - 1.0) * (d - cb)
            }
        }
        CompositeOp::Difference => (cb - cs).abs(),
        CompositeOp::Exclusion => cb + cs - 2.0 * cb * cs,
        // Non-separable ops never reach here.
        _ => cs,
    }
}

/// Full text metrics returned by `CanvasContext2D::measure_text`.
///
/// All values are in CSS pixels at the current transform scale (untransformed
/// canvas pixel coordinates). Distances are measured from the *alphabetic baseline*
/// as the spec requires:
/// - positive `…Ascent` values are *above* the baseline (toward smaller y).
/// - positive `…Descent` values are *below* the baseline (toward larger y).
#[derive(Debug, Clone, Copy)]
pub struct TextMetrics {
    /// Advance width of the text — the only field returned pre-fix.
    pub width: f32,
    /// Distance from the baseline to the top of the highest glyph pixel
    /// (positive = above baseline). Derived from the font's declared ascender.
    pub actual_bounding_box_ascent: f32,
    /// Distance from the baseline to the bottom of the lowest glyph pixel
    /// (positive = below baseline). Derived from the font's declared descender.
    pub actual_bounding_box_descent: f32,
    /// Font-declared ascender above the baseline (same as `actual_bounding_box_ascent`
    /// for our bitmap font; real fonts distinguish the two).
    pub font_bounding_box_ascent: f32,
    /// Font-declared descender below the baseline (same as `actual_bounding_box_descent`
    /// for our bitmap font).
    pub font_bounding_box_descent: f32,
    /// Distance above the baseline to the top of the em box
    /// (≈ ascent for most purposes).
    pub em_height_ascent: f32,
    /// Distance below the baseline to the bottom of the em box
    /// (≈ descent for most purposes).
    pub em_height_descent: f32,
    /// Horizontal distance from the `x` argument to the left edge of the
    /// bounding box.  Positive when the box extends to the *left* of `x`
    /// (e.g. right-aligned text).
    pub actual_bounding_box_left: f32,
    /// Horizontal distance from the `x` argument to the right edge of the
    /// bounding box.  Positive when the box extends to the *right* of `x`.
    pub actual_bounding_box_right: f32,
    /// Baseline y distance — always 0 for the alphabetic baseline.
    pub alphabetic_baseline: f32,
    /// The hanging baseline (for Devanagari etc.) — approximated as `ascent * 0.8`.
    pub hanging_baseline: f32,
    /// The ideographic baseline — approximated as `−descent`.
    pub ideographic_baseline: f32,
}

/// The CSS `textBaseline` value, controlling how `fill_text`'s y coordinate
/// relates to the rendered text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextBaseline {
    /// Default: y is the alphabetic baseline. Glyphs are drawn above it.
    Alphabetic,
    /// y is the top of the em box.
    Top,
    /// y is the hanging baseline (roughly the top for ideographic scripts).
    Hanging,
    /// y is the midpoint of the em box.
    Middle,
    /// y is the ideographic baseline (bottom of the em box for CJK).
    Ideographic,
    /// y is the bottom of the descenders.
    Bottom,
}

impl TextBaseline {
    pub fn from_str(s: &str) -> Self {
        match s {
            "top" => Self::Top,
            "hanging" => Self::Hanging,
            "middle" => Self::Middle,
            "ideographic" => Self::Ideographic,
            "bottom" => Self::Bottom,
            _ => Self::Alphabetic, // "alphabetic" and any unknown → alphabetic
        }
    }
}

/// The CSS `textAlign` value, controlling how `fill_text`'s x coordinate
/// relates to the rendered text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    /// Default: x is the left edge of the text.
    Left,
    /// x is the right edge of the text.
    Right,
    /// x is the center of the text.
    Center,
    /// Same as Left in LTR contexts.
    Start,
    /// Same as Right in LTR contexts.
    End,
}

impl TextAlign {
    pub fn from_str(s: &str) -> Self {
        match s {
            "right" => Self::Right,
            "center" => Self::Center,
            "end" => Self::End,
            // "start" and "left" and any unknown → left
            _ => Self::Left,
        }
    }
}

/// A recorded, reusable path — the backing of the Web `Path2D` object.
///
/// Ops are stored in *user space* (no transform baked in). When a context
/// consumes a `Path2D` (e.g. `ctx.fill(path)`), the context applies its own
/// current transform as it replays the ops, exactly as the spec requires.
/// Curves are flattened into line segments at record time so the rasterizer
/// only sees straight runs — identical strategy to the inline path.
#[derive(Debug, Clone, Default)]
pub struct Path2D {
    ops: Vec<PathOp>,
    cursor: Option<(f32, f32)>,
    start: Option<(f32, f32)>,
}

impl Path2D {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from an SVG path-data string (the `new Path2D("M0 0 L10 0…")`
    /// constructor). Supports the common commands M/m L/l H/h V/v C/c Q/q Z/z.
    pub fn from_svg(d: &str) -> Self {
        let mut p = Self::new();
        p.add_svg(d);
        p
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        self.cursor = Some((x, y));
        self.start = Some((x, y));
        self.ops.push(PathOp::MoveTo(x, y));
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        if self.cursor.is_none() {
            self.move_to(x, y);
            return;
        }
        self.cursor = Some((x, y));
        self.ops.push(PathOp::LineTo(x, y));
    }

    pub fn close_path(&mut self) {
        self.ops.push(PathOp::Close);
        if let Some(s) = self.start {
            self.cursor = Some(s);
        }
    }

    pub fn quadratic_curve_to(&mut self, cpx: f32, cpy: f32, x: f32, y: f32) {
        let (sx, sy) = self.cursor.unwrap_or((cpx, cpy));
        if self.cursor.is_none() {
            self.move_to(sx, sy);
        }
        let segments = 16;
        for i in 1..=segments {
            let t = i as f32 / segments as f32;
            let omt = 1.0 - t;
            let px = omt * omt * sx + 2.0 * omt * t * cpx + t * t * x;
            let py = omt * omt * sy + 2.0 * omt * t * cpy + t * t * y;
            self.cursor = Some((px, py));
            self.ops.push(PathOp::LineTo(px, py));
        }
    }

    pub fn bezier_curve_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        let (sx, sy) = self.cursor.unwrap_or((c1x, c1y));
        if self.cursor.is_none() {
            self.move_to(sx, sy);
        }
        let segments = 24;
        for i in 1..=segments {
            let t = i as f32 / segments as f32;
            let omt = 1.0 - t;
            let omt2 = omt * omt;
            let omt3 = omt2 * omt;
            let t2 = t * t;
            let t3 = t2 * t;
            let px = omt3 * sx + 3.0 * omt2 * t * c1x + 3.0 * omt * t2 * c2x + t3 * x;
            let py = omt3 * sy + 3.0 * omt2 * t * c1y + 3.0 * omt * t2 * c2y + t3 * y;
            self.cursor = Some((px, py));
            self.ops.push(PathOp::LineTo(px, py));
        }
    }

    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.move_to(x, y);
        self.line_to(x + w, y);
        self.line_to(x + w, y + h);
        self.line_to(x, y + h);
        self.close_path();
        // Per spec rect() leaves the cursor at the start point (already set by close).
    }

    pub fn arc(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        anticlockwise: bool,
    ) {
        let r = radius.abs();
        if r == 0.0 {
            // Degenerate arc still moves the pen to the single point.
            self.line_to(cx, cy);
            return;
        }
        let two_pi = std::f32::consts::TAU;
        let sweep = if anticlockwise {
            let mut s = end_angle - start_angle;
            while s > 0.0 {
                s -= two_pi;
            }
            s
        } else {
            let mut s = end_angle - start_angle;
            while s < 0.0 {
                s += two_pi;
            }
            s
        };
        let arc_len = r * sweep.abs();
        let steps = ((arc_len / 2.0).ceil() as i32).clamp(8, 256);
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let theta = start_angle + sweep * t;
            let x = cx + r * theta.cos();
            let y = cy + r * theta.sin();
            if i == 0 {
                if self.cursor.is_some() {
                    self.line_to(x, y);
                } else {
                    self.move_to(x, y);
                }
            } else {
                self.line_to(x, y);
            }
        }
    }

    pub fn arc_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, radius: f32) {
        let (x0, y0) = match self.cursor {
            Some(p) => p,
            None => {
                self.move_to(x1, y1);
                return;
            }
        };
        if let Some(seg) = compute_arc_to(x0, y0, x1, y1, x2, y2, radius) {
            // Line to the first tangent point, then arc to the second.
            self.line_to(seg.t1.0, seg.t1.1);
            self.arc(
                seg.center.0,
                seg.center.1,
                seg.radius,
                seg.start_angle,
                seg.end_angle,
                seg.anticlockwise,
            );
        } else {
            // Degenerate (zero radius / collinear) — straight line to (x1,y1).
            self.line_to(x1, y1);
        }
    }

    pub fn ellipse(
        &mut self,
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        rotation: f32,
        start_angle: f32,
        end_angle: f32,
        anticlockwise: bool,
    ) {
        let (rx, ry) = (rx.abs(), ry.abs());
        if rx == 0.0 || ry == 0.0 {
            return;
        }
        let two_pi = std::f32::consts::TAU;
        let sweep = if anticlockwise {
            let mut s = end_angle - start_angle;
            while s > 0.0 {
                s -= two_pi;
            }
            s
        } else {
            let mut s = end_angle - start_angle;
            while s < 0.0 {
                s += two_pi;
            }
            s
        };
        let (rot_sin, rot_cos) = rotation.sin_cos();
        let steps = (((rx.max(ry)) * sweep.abs() / 2.0).ceil() as i32).clamp(8, 256);
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let theta = start_angle + sweep * t;
            let ex = rx * theta.cos();
            let ey = ry * theta.sin();
            let x = cx + ex * rot_cos - ey * rot_sin;
            let y = cy + ex * rot_sin + ey * rot_cos;
            if i == 0 {
                if self.cursor.is_some() {
                    self.line_to(x, y);
                } else {
                    self.move_to(x, y);
                }
            } else {
                self.line_to(x, y);
            }
        }
    }

    /// `addPath(path)` — append another path's ops. The optional DOMMatrix
    /// transform argument is applied if provided (a/b/c/d/e/f).
    pub fn add_path(&mut self, other: &Path2D, transform: Option<[f32; 6]>) {
        match transform {
            Some(m) => {
                for op in &other.ops {
                    let mapped = match op {
                        PathOp::MoveTo(x, y) => {
                            PathOp::MoveTo(m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
                        }
                        PathOp::LineTo(x, y) => {
                            PathOp::LineTo(m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
                        }
                        PathOp::Close => PathOp::Close,
                    };
                    if let PathOp::MoveTo(x, y) | PathOp::LineTo(x, y) = mapped {
                        self.cursor = Some((x, y));
                    }
                    self.ops.push(mapped);
                }
            }
            None => {
                self.ops.extend_from_slice(&other.ops);
                self.cursor = other.cursor.or(self.cursor);
            }
        }
    }

    /// Parse and append SVG path-data commands. Best-effort over the common
    /// subset; unrecognised commands are skipped.
    pub fn add_svg(&mut self, d: &str) {
        let nums = svg_tokenize(d);
        let mut i = 0usize;
        let mut idx = 0usize; // index into nums
        let mut last_cmd = ' ';
        let bytes: Vec<(char, usize)> = nums.cmds.clone();
        let take = |idx: &mut usize| -> f32 {
            let v = nums.values.get(*idx).copied().unwrap_or(0.0);
            *idx += 1;
            v
        };
        while i < bytes.len() {
            let (cmd, _) = bytes[i];
            i += 1;
            last_cmd = cmd;
            match cmd {
                'M' => {
                    let x = take(&mut idx);
                    let y = take(&mut idx);
                    self.move_to(x, y);
                }
                'm' => {
                    let (cx, cy) = self.cursor.unwrap_or((0.0, 0.0));
                    let x = cx + take(&mut idx);
                    let y = cy + take(&mut idx);
                    self.move_to(x, y);
                }
                'L' => {
                    let x = take(&mut idx);
                    let y = take(&mut idx);
                    self.line_to(x, y);
                }
                'l' => {
                    let (cx, cy) = self.cursor.unwrap_or((0.0, 0.0));
                    self.line_to(cx + take(&mut idx), cy + take(&mut idx));
                }
                'H' => {
                    let (_, cy) = self.cursor.unwrap_or((0.0, 0.0));
                    self.line_to(take(&mut idx), cy);
                }
                'h' => {
                    let (cx, cy) = self.cursor.unwrap_or((0.0, 0.0));
                    self.line_to(cx + take(&mut idx), cy);
                }
                'V' => {
                    let (cx, _) = self.cursor.unwrap_or((0.0, 0.0));
                    self.line_to(cx, take(&mut idx));
                }
                'v' => {
                    let (cx, cy) = self.cursor.unwrap_or((0.0, 0.0));
                    self.line_to(cx, cy + take(&mut idx));
                }
                'C' => {
                    let c1x = take(&mut idx);
                    let c1y = take(&mut idx);
                    let c2x = take(&mut idx);
                    let c2y = take(&mut idx);
                    let x = take(&mut idx);
                    let y = take(&mut idx);
                    self.bezier_curve_to(c1x, c1y, c2x, c2y, x, y);
                }
                'c' => {
                    let (cx, cy) = self.cursor.unwrap_or((0.0, 0.0));
                    let c1x = cx + take(&mut idx);
                    let c1y = cy + take(&mut idx);
                    let c2x = cx + take(&mut idx);
                    let c2y = cy + take(&mut idx);
                    let x = cx + take(&mut idx);
                    let y = cy + take(&mut idx);
                    self.bezier_curve_to(c1x, c1y, c2x, c2y, x, y);
                }
                'Q' => {
                    let cpx = take(&mut idx);
                    let cpy = take(&mut idx);
                    let x = take(&mut idx);
                    let y = take(&mut idx);
                    self.quadratic_curve_to(cpx, cpy, x, y);
                }
                'q' => {
                    let (cx, cy) = self.cursor.unwrap_or((0.0, 0.0));
                    let cpx = cx + take(&mut idx);
                    let cpy = cy + take(&mut idx);
                    let x = cx + take(&mut idx);
                    let y = cy + take(&mut idx);
                    self.quadratic_curve_to(cpx, cpy, x, y);
                }
                'Z' | 'z' => {
                    self.close_path();
                }
                _ => {}
            }
        }
        let _ = last_cmd;
    }

    /// The recorded ops transformed through `m` (device space), for the
    /// context to rasterize.
    fn ops_transformed(&self, m: [f32; 6]) -> Vec<PathOp> {
        self.ops
            .iter()
            .map(|op| match op {
                PathOp::MoveTo(x, y) => {
                    PathOp::MoveTo(m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
                }
                PathOp::LineTo(x, y) => {
                    PathOp::LineTo(m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
                }
                PathOp::Close => PathOp::Close,
            })
            .collect()
    }
}

/// Numbers + command letters extracted from an SVG path string.
struct SvgTokens {
    cmds: Vec<(char, usize)>,
    values: Vec<f32>,
}

/// Tokenize an SVG path-data string into a flat command/number stream. Each
/// command letter is recorded; implicit repeated commands (e.g. `L10,0 20,0`)
/// are expanded by emitting the command again for each coordinate group. The
/// expansion is driven by counting numbers consumed per command at parse time.
fn svg_tokenize(d: &str) -> SvgTokens {
    // First, split into (letter | number) tokens preserving order.
    #[derive(Clone)]
    enum Tok {
        Cmd(char),
        Num(f32),
    }
    let mut toks: Vec<Tok> = Vec::new();
    let chars: Vec<char> = d.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_alphabetic() {
            toks.push(Tok::Cmd(c));
            i += 1;
        } else if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() {
            // Parse a number (with optional exponent).
            let start = i;
            if c == '-' || c == '+' {
                i += 1;
            }
            let mut seen_dot = false;
            while i < chars.len() {
                let ch = chars[i];
                if ch.is_ascii_digit() {
                    i += 1;
                } else if ch == '.' && !seen_dot {
                    seen_dot = true;
                    i += 1;
                } else if (ch == 'e' || ch == 'E')
                    && i + 1 < chars.len()
                    && (chars[i + 1].is_ascii_digit()
                        || chars[i + 1] == '-'
                        || chars[i + 1] == '+')
                {
                    i += 2;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    break;
                } else {
                    break;
                }
            }
            let s: String = chars[start..i].iter().collect();
            if let Ok(v) = s.parse::<f32>() {
                toks.push(Tok::Num(v));
            }
        } else {
            // Whitespace or comma separator.
            i += 1;
        }
    }

    // Number of coordinates each command consumes per group.
    fn arity(c: char) -> usize {
        match c.to_ascii_uppercase() {
            'M' | 'L' | 'T' => 2,
            'H' | 'V' => 1,
            'C' => 6,
            'S' | 'Q' => 4,
            'A' => 7,
            'Z' => 0,
            _ => 0,
        }
    }

    let mut cmds: Vec<(char, usize)> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    let mut j = 0usize;
    let mut cur_cmd: Option<char> = None;
    while j < toks.len() {
        match &toks[j] {
            Tok::Cmd(c) => {
                cur_cmd = Some(*c);
                j += 1;
                if c.to_ascii_uppercase() == 'Z' {
                    cmds.push((*c, values.len()));
                }
            }
            Tok::Num(_) => {
                // Implicit repeat of the current command. After an explicit M,
                // subsequent coordinate pairs are implicit L (or l for m).
                let Some(mut c) = cur_cmd else {
                    j += 1;
                    continue;
                };
                if c == 'M' {
                    c = 'L';
                    cur_cmd = Some('L');
                } else if c == 'm' {
                    c = 'l';
                    cur_cmd = Some('l');
                }
                let n = arity(c);
                if n == 0 {
                    j += 1;
                    continue;
                }
                let mut group = Vec::with_capacity(n);
                let mut k = 0;
                while k < n && j < toks.len() {
                    if let Tok::Num(v) = toks[j] {
                        group.push(v);
                        j += 1;
                        k += 1;
                    } else {
                        break;
                    }
                }
                if group.len() == n {
                    cmds.push((c, values.len()));
                    values.extend_from_slice(&group);
                } else {
                    break;
                }
            }
        }
    }
    SvgTokens { cmds, values }
}

/// Result of the `arcTo` tangent-circle computation.
struct ArcToSeg {
    /// First tangent point (where the arc meets the P0->P1 line).
    t1: (f32, f32),
    center: (f32, f32),
    radius: f32,
    start_angle: f32,
    end_angle: f32,
    anticlockwise: bool,
}

/// Compute the arc of radius `radius` tangent to the line P0->P1 and the line
/// P1->P2 (the classic Canvas `arcTo` geometry). Returns `None` for the
/// degenerate cases (zero radius, coincident points, or collinear points).
fn compute_arc_to(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    radius: f32,
) -> Option<ArcToSeg> {
    if radius <= 0.0 {
        return None;
    }
    // Unit vectors from the corner P1 toward P0 and P2.
    let (v1x, v1y) = (x0 - x1, y0 - y1);
    let (v2x, v2y) = (x2 - x1, y2 - y1);
    let len1 = (v1x * v1x + v1y * v1y).sqrt();
    let len2 = (v2x * v2x + v2y * v2y).sqrt();
    if len1 < 1e-6 || len2 < 1e-6 {
        return None;
    }
    let (u1x, u1y) = (v1x / len1, v1y / len1);
    let (u2x, u2y) = (v2x / len2, v2y / len2);
    // Cross product detects collinearity.
    let cross = u1x * u2y - u1y * u2x;
    if cross.abs() < 1e-6 {
        return None; // collinear -> straight line
    }
    // Angle between the two vectors.
    let dot = (u1x * u2x + u1y * u2y).clamp(-1.0, 1.0);
    let angle = dot.acos();
    // Distance from P1 to each tangent point along the edges.
    let tan_dist = radius / (angle / 2.0).tan();
    // Tangent points.
    let t1 = (x1 + u1x * tan_dist, y1 + u1y * tan_dist);
    let t2 = (x1 + u2x * tan_dist, y1 + u2y * tan_dist);
    // The bisector direction (sum of the two unit edge vectors), normalized.
    let (bx, by) = (u1x + u2x, u1y + u2y);
    let blen = (bx * bx + by * by).sqrt();
    if blen < 1e-6 {
        return None;
    }
    let (bux, buy) = (bx / blen, by / blen);
    // Distance from P1 to the arc center along the bisector.
    let center_dist = radius / (angle / 2.0).sin();
    let center = (x1 + bux * center_dist, y1 + buy * center_dist);
    // Angles from the center to each tangent point.
    let start_angle = (t1.1 - center.1).atan2(t1.0 - center.0);
    let end_angle = (t2.1 - center.1).atan2(t2.0 - center.0);
    // Direction: the arc sweeps from t1 to t2 the SHORT way. cross>0 means the
    // corner turns left, so the arc is clockwise in canvas (y-down) space.
    let anticlockwise = cross > 0.0;
    Some(ArcToSeg {
        t1,
        center,
        radius,
        start_angle,
        end_angle,
        anticlockwise,
    })
}

#[derive(Debug, Clone)]
struct State {
    fill: Color,
    /// The canvas fill style — gradient or solid color. When set to a gradient
    /// this takes precedence over `fill` for path/rect fill operations.
    fill_style: Option<FillStyle>,
    stroke: Color,
    line_width: f32,
    /// 3x2 affine: [a, b, c, d, e, f]. Maps (x,y) to (a*x+c*y+e, b*x+d*y+f).
    transform: [f32; 6],
    global_alpha: f32,
    /// Parsed textBaseline — controls fill_text y placement.
    text_baseline: TextBaseline,
    /// Parsed textAlign — controls fill_text x placement.
    text_align: TextAlign,
    /// Font size in CSS px, parsed from ctx.font (e.g. "bold 16px Arial" → 16).
    font_size_px: f32,
    /// Current clip region (axis-aligned rectangle). When `Some`, all fill/stroke
    /// operations are restricted to pixels inside this region.
    clip: Option<ClipRect>,
    /// Current `globalCompositeOperation`. Defaults to `SourceOver`.
    composite_op: CompositeOp,
    /// `shadowBlur` (px) — the blur radius for the soft glow drawn under fills.
    shadow_blur: f32,
    /// `shadowColor` — the shadow/glow color (transparent disables the shadow).
    shadow_color: Color,
    /// `shadowOffsetX/Y` (px) — shadow displacement from the shape.
    shadow_offset_x: f32,
    shadow_offset_y: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            fill: Color::BLACK,
            fill_style: None,
            stroke: Color::BLACK,
            line_width: 1.0,
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            global_alpha: 1.0,
            text_baseline: TextBaseline::Alphabetic,
            text_align: TextAlign::Left,
            // Default CSS font is "10px sans-serif".
            font_size_px: 10.0,
            clip: None,
            composite_op: CompositeOp::SourceOver,
            shadow_blur: 0.0,
            shadow_color: Color::TRANSPARENT,
            shadow_offset_x: 0.0,
            shadow_offset_y: 0.0,
        }
    }
}

/// A Canvas drawing context. Owns its bitmap; render output flows
/// directly into `bitmap.pixels`. The context stays alive as long as
/// any JS reference to it does; the bitmap can be blitted into the
/// page's compositor at paint time.
#[derive(Debug)]
pub struct CanvasContext2D {
    pub bitmap: Bitmap,
    states: Vec<State>,
    path: Vec<PathOp>,
    cursor: Option<(f32, f32)>,
}

impl CanvasContext2D {
    pub fn new(width: u32, height: u32) -> Self {
        let mut bitmap = Bitmap::new(width, height);
        // Canvases start fully transparent so a page can layer them.
        bitmap.clear(Color::TRANSPARENT);
        Self {
            bitmap,
            states: vec![State::default()],
            path: Vec::new(),
            cursor: None,
        }
    }

    /// Resize the backing bitmap (e.g. when JS sets `canvas.width`/`height`
    /// after `getContext`). Per the HTML spec this also RESETS the canvas to
    /// transparent — particles.js sizes the canvas this way and then draws, so
    /// without this the bitmap stayed 300×150 and everything past it was clipped.
    /// No-op if the size is unchanged so it can be called cheaply before draws.
    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.bitmap.width == w && self.bitmap.height == h {
            return;
        }
        self.bitmap = Bitmap::new(w, h);
        self.bitmap.clear(Color::TRANSPARENT);
    }

    /// Current backing-bitmap dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.bitmap.width, self.bitmap.height)
    }

    fn state(&self) -> &State {
        self.states.last().unwrap()
    }

    fn state_mut(&mut self) -> &mut State {
        self.states.last_mut().unwrap()
    }

    pub fn set_fill_color(&mut self, c: Color) {
        self.state_mut().fill = c;
    }

    pub fn set_stroke_color(&mut self, c: Color) {
        self.state_mut().stroke = c;
    }

    pub fn set_line_width(&mut self, w: f32) {
        self.state_mut().line_width = w.max(0.0);
    }

    pub fn set_global_alpha(&mut self, a: f32) {
        self.state_mut().global_alpha = a.clamp(0.0, 1.0);
    }

    pub fn set_shadow_blur(&mut self, b: f32) {
        self.state_mut().shadow_blur = b.max(0.0);
    }

    pub fn set_shadow_color(&mut self, c: Color) {
        self.state_mut().shadow_color = c;
    }

    pub fn set_shadow_offset(&mut self, x: f32, y: f32) {
        let s = self.state_mut();
        s.shadow_offset_x = x;
        s.shadow_offset_y = y;
    }

    /// Set the text baseline from the CSS textBaseline string value.
    pub fn set_text_baseline(&mut self, baseline: &str) {
        self.state_mut().text_baseline = TextBaseline::from_str(baseline);
    }

    /// Set the text alignment from the CSS textAlign string value.
    pub fn set_text_align(&mut self, align: &str) {
        self.state_mut().text_align = TextAlign::from_str(align);
    }

    /// Set the font size (in px) parsed from ctx.font.
    /// Callers should parse the font shorthand (e.g. "bold 16px Arial") and
    /// extract the size before calling this.
    pub fn set_font_size(&mut self, px: f32) {
        if px > 0.0 {
            self.state_mut().font_size_px = px;
        }
    }

    /// Set the fill style to a linear gradient from (x0, y0) to (x1, y1) with the
    /// given color stops (already sorted by offset). Replaces any previous fill color.
    pub fn set_fill_linear_gradient(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        stops: Vec<GradientStop>,
    ) {
        self.state_mut().fill_style = Some(FillStyle::Linear { x0, y0, x1, y1, stops });
    }

    /// Set the fill style to a radial gradient defined by the start circle
    /// (x0, y0, r0) and the end circle (x1, y1, r1), with the given color stops
    /// (sorted by offset). Mirrors `createRadialGradient(x0,y0,r0,x1,y1,r1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_fill_radial_gradient(
        &mut self,
        x0: f32,
        y0: f32,
        r0: f32,
        x1: f32,
        y1: f32,
        r1: f32,
        stops: Vec<GradientStop>,
    ) {
        self.state_mut().fill_style =
            Some(FillStyle::Radial { x0, y0, r0, x1, y1, r1, stops });
    }

    /// Clear any gradient fill style, reverting to the solid color stored in `fill`.
    pub fn clear_fill_style(&mut self) {
        self.state_mut().fill_style = None;
    }

    /// `clip()` — intersect the current clip region with the axis-aligned bounding
    /// box of the current path. Subsequent fill/stroke operations are restricted to
    /// pixels inside the resulting clipped area.
    pub fn clip(&mut self) {
        // Compute the bounding box of the current path.
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for op in &self.path {
            let (x, y) = match op {
                PathOp::MoveTo(x, y) | PathOp::LineTo(x, y) => (*x, *y),
                PathOp::Close => continue,
            };
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        if !min_x.is_finite() {
            // Empty path — no change to clip.
            return;
        }
        let new_x0 = min_x.floor() as i32;
        let new_y0 = min_y.floor() as i32;
        let new_x1 = max_x.ceil() as i32;
        let new_y1 = max_y.ceil() as i32;
        // Intersect with any existing clip.
        let merged = match self.state().clip {
            Some(existing) => ClipRect {
                x0: existing.x0.max(new_x0),
                y0: existing.y0.max(new_y0),
                x1: existing.x1.min(new_x1),
                y1: existing.y1.min(new_y1),
            },
            None => ClipRect { x0: new_x0, y0: new_y0, x1: new_x1, y1: new_y1 },
        };
        self.state_mut().clip = Some(merged);
    }

    pub fn save(&mut self) {
        let copy = self.state().clone();
        self.states.push(copy);
    }

    pub fn restore(&mut self) {
        if self.states.len() > 1 {
            self.states.pop();
        }
    }

    pub fn translate(&mut self, tx: f32, ty: f32) {
        let m = self.state().transform;
        let e = m[4] + m[0] * tx + m[2] * ty;
        let f = m[5] + m[1] * tx + m[3] * ty;
        let s = self.state_mut();
        s.transform[4] = e;
        s.transform[5] = f;
    }

    pub fn scale(&mut self, sx: f32, sy: f32) {
        let m = self.state().transform;
        let s = self.state_mut();
        s.transform[0] = m[0] * sx;
        s.transform[1] = m[1] * sx;
        s.transform[2] = m[2] * sy;
        s.transform[3] = m[3] * sy;
    }

    pub fn rotate(&mut self, angle_rad: f32) {
        let (sin, cos) = angle_rad.sin_cos();
        let m = self.state().transform;
        let a = m[0] * cos + m[2] * sin;
        let b = m[1] * cos + m[3] * sin;
        let c = m[0] * -sin + m[2] * cos;
        let d = m[1] * -sin + m[3] * cos;
        let s = self.state_mut();
        s.transform[0] = a;
        s.transform[1] = b;
        s.transform[2] = c;
        s.transform[3] = d;
    }

    pub fn set_transform(&mut self, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) {
        self.state_mut().transform = [a, b, c, d, e, f];
    }

    /// Map a user-space point through the current transform.
    fn map(&self, x: f32, y: f32) -> (f32, f32) {
        let m = self.state().transform;
        (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
    }

    /// Canvas2D `drawImage`. `src` is row-major BGRA u32 (`src_w`×`src_h`); the
    /// `(sx,sy,sw,sh)` source rect is scaled into the `(dx,dy,dw,dh)` dest rect,
    /// mapped through the current transform (translate/scale). Rotation/skew is
    /// ignored for now (drawImage under rotation is rare). Clip is honored.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_image(
        &mut self,
        src: &[u32],
        src_w: u32,
        src_h: u32,
        sx: f32,
        sy: f32,
        sw: f32,
        sh: f32,
        dx: f32,
        dy: f32,
        dw: f32,
        dh: f32,
    ) {
        if src_w == 0 || src_h == 0 || sw <= 0.0 || sh <= 0.0 || dw <= 0.0 || dh <= 0.0 {
            return;
        }
        let (x0, y0) = self.map(dx, dy);
        let (x1, y1) = self.map(dx + dw, dy + dh);
        let blit_dx = x0.min(x1).floor() as i32;
        let blit_dy = y0.min(y1).floor() as i32;
        let blit_dw = (x1 - x0).abs().round() as u32;
        let blit_dh = (y1 - y0).abs().round() as u32;
        // Source-rect crop: use the buffer directly when it's the whole image.
        let full = sx == 0.0 && sy == 0.0 && sw as u32 >= src_w && sh as u32 >= src_h;
        if full {
            self.bitmap
                .blit_bgra_scaled(blit_dx, blit_dy, blit_dw, blit_dh, src_w, src_h, src);
        } else {
            let sxi = (sx.max(0.0) as u32).min(src_w);
            let syi = (sy.max(0.0) as u32).min(src_h);
            let swi = (sw as u32).min(src_w.saturating_sub(sxi));
            let shi = (sh as u32).min(src_h.saturating_sub(syi));
            if swi == 0 || shi == 0 {
                return;
            }
            let mut sub = Vec::with_capacity((swi * shi) as usize);
            for row in 0..shi {
                let start = ((syi + row) * src_w + sxi) as usize;
                sub.extend_from_slice(&src[start..start + swi as usize]);
            }
            self.bitmap
                .blit_bgra_scaled(blit_dx, blit_dy, blit_dw, blit_dh, swi, shi, &sub);
        }
    }

    pub fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        // Check whether the current transform has any rotation/skew (b or c non-zero).
        // If so, all four corners must be transformed individually to get the true
        // quadrilateral — using only the two diagonal corners produces the wrong
        // axis-aligned bounding box under rotation/skew.
        let m = self.state().transform;
        let has_rotation = m[1].abs() > 1e-6 || m[2].abs() > 1e-6;

        if has_rotation {
            // Delegate to the path-based fill so the scanline polygon filler
            // handles all four transformed corners correctly.  Save and restore
            // the path so fill_rect remains a standalone operation that never
            // disturbs the caller's current path.
            let saved_path = std::mem::take(&mut self.path);
            let saved_cursor = self.cursor.take();
            self.move_to(x, y);
            self.line_to(x + w, y);
            self.line_to(x + w, y + h);
            self.line_to(x, y + h);
            self.close_path();
            self.fill(FillRule::Nonzero);
            self.path = saved_path;
            self.cursor = saved_cursor;
            return;
        }

        // Fast path for axis-aligned transforms (pure translate/scale):
        // only two corners need to be mapped.
        let (x0, y0) = self.map(x, y);
        let (x1, y1) = self.map(x + w, y + h);
        let lx = x0.min(x1).floor() as i32;
        let ly = y0.min(y1).floor() as i32;
        let rx = x0.max(x1).ceil() as i32;
        let ry = y0.max(y1).ceil() as i32;

        // Apply clip.
        let cx0 = match self.state().clip { Some(cr) => lx.max(cr.x0), None => lx };
        let cy0 = match self.state().clip { Some(cr) => ly.max(cr.y0), None => ly };
        let cx1 = match self.state().clip { Some(cr) => rx.min(cr.x1), None => rx };
        let cy1 = match self.state().clip { Some(cr) => ry.min(cr.y1), None => ry };
        let cx0 = cx0.max(0);
        let cy0 = cy0.max(0);
        let cx1 = cx1.min(self.bitmap.width as i32);
        let cy1 = cy1.min(self.bitmap.height as i32);
        if cx1 <= cx0 || cy1 <= cy0 {
            return;
        }

        let global_alpha = self.state().global_alpha;
        let op = self.state().composite_op;

        // ── shadowBlur / shadowColor pre-pass ─────────────────────────────────
        // fillRect casts a shadow of the rectangle silhouette behind itself when
        // a shadow is set (Canvas spec: shadows apply to fill/stroke/image ops).
        // Use the DEVICE-space rect corners (axis-aligned here) as the polygon.
        {
            let sh = {
                let s = self.state();
                (s.shadow_color, s.shadow_blur, s.shadow_offset_x, s.shadow_offset_y, s.global_alpha)
            };
            let (scolor, blur, sox, soy, ga) = sh;
            if scolor.a > 0 && (blur > 0.0 || sox != 0.0 || soy != 0.0) {
                let rect = vec![vec![
                    (x0, y0),
                    (x1, y0),
                    (x1, y1),
                    (x0, y1),
                    (x0, y0),
                ]];
                let clip = self.state().clip;
                self.paint_shadow_glow(&rect, FillRule::Nonzero, scolor, blur, sox, soy, ga, clip);
            }
        }

        match &self.state().fill_style.clone() {
            Some(FillStyle::Pattern(p)) => {
                // Pattern fill: tile sampled per destination pixel.
                for yy in cy0..cy1 {
                    let row = (yy as usize) * (self.bitmap.width as usize);
                    for xx in cx0..cx1 {
                        let idx = row + xx as usize;
                        match p.sample(xx, yy, global_alpha) {
                            Some(c) => self.put_composite(idx, c, op),
                            None => self.put_composite(idx, Color::TRANSPARENT, op),
                        }
                    }
                }
            }
            Some(fs) => {
                // Gradient fill: sample per pixel.
                for yy in cy0..cy1 {
                    let row = (yy as usize) * (self.bitmap.width as usize);
                    for xx in cx0..cx1 {
                        let c = fs.sample(xx as f32 + 0.5, yy as f32 + 0.5, global_alpha);
                        let idx = row + xx as usize;
                        self.put_composite(idx, c, op);
                    }
                }
            }
            None => {
                let mut c = self.state().fill;
                c.a = ((c.a as f32) * global_alpha) as u8;
                if matches!(op, CompositeOp::SourceOver) {
                    // Byte-identical legacy path.
                    self.bitmap.fill_rect(cx0, cy0, cx1 - cx0, cy1 - cy0, c);
                } else {
                    for yy in cy0..cy1 {
                        let row = (yy as usize) * (self.bitmap.width as usize);
                        for xx in cx0..cx1 {
                            self.put_composite(row + xx as usize, c, op);
                        }
                    }
                }
            }
        }
    }

    /// Write a straight-alpha source color into `pixels[idx]` honoring the
    /// composite op. For `SourceOver` this is byte-identical to the legacy
    /// write (overwrite when fully opaque, `blend_bgra` when semi-transparent,
    /// skip when fully transparent). Other ops go through `composite_pixel`.
    #[inline]
    fn put_composite(&mut self, idx: usize, c: Color, op: CompositeOp) {
        if matches!(op, CompositeOp::SourceOver) {
            if c.a == 255 {
                self.bitmap.pixels[idx] = c.to_bgra_u32();
            } else if c.a > 0 {
                self.bitmap.pixels[idx] = blend_bgra(self.bitmap.pixels[idx], c);
            }
        } else {
            self.bitmap.pixels[idx] = composite_pixel(self.bitmap.pixels[idx], c, op);
        }
    }

    pub fn clear_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        // Replace pixels with fully transparent (no blend).
        // Under rotation/skew the four corners map to a non-axis-aligned quad;
        // we fill that quad with transparent using the scanline polygon filler
        // (override fill color to transparent, no-blend write via a helper).
        let m = self.state().transform;
        let has_rotation = m[1].abs() > 1e-6 || m[2].abs() > 1e-6;

        if has_rotation {
            // Compute all four pixel-space corners.
            let p00 = self.map(x, y);
            let p10 = self.map(x + w, y);
            let p11 = self.map(x + w, y + h);
            let p01 = self.map(x, y + h);
            let poly = vec![p00, p10, p11, p01, p00];

            // Scanline-fill the quad, overwriting each pixel with transparent.
            let (bw, bh) = (self.bitmap.width as i32, self.bitmap.height as i32);
            let v = Color::TRANSPARENT.to_bgra_u32();
            // Compute bounding box of the quad.
            let min_y = poly.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
            let max_y = poly.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
            let y0i = (min_y.floor() as i32).max(0);
            let y1i = (max_y.ceil() as i32).min(bh);
            let min_x = poly.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
            let max_x = poly.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
            let x0i = (min_x.floor() as i32).max(0);
            let x1i = (max_x.ceil() as i32).min(bw);
            for yy in y0i..y1i {
                let yc = yy as f32 + 0.5;
                // Collect x-intersections of the quad edges at this scanline.
                let mut xs: Vec<f32> = Vec::new();
                for w in poly.windows(2) {
                    let (ax, ay) = w[0];
                    let (bx, by) = w[1];
                    if (ay > yc) == (by > yc) {
                        continue;
                    }
                    let t = (yc - ay) / (by - ay);
                    xs.push(ax + t * (bx - ax));
                }
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if xs.len() < 2 {
                    continue;
                }
                let row = (yy as usize) * (self.bitmap.width as usize);
                for chunk in xs.chunks(2) {
                    if chunk.len() < 2 {
                        continue;
                    }
                    let lx = (chunk[0].floor() as i32).max(x0i);
                    let rx = (chunk[1].ceil() as i32).min(x1i);
                    for xx in lx..rx {
                        self.bitmap.pixels[row + xx as usize] = v;
                    }
                }
            }
            return;
        }

        // Fast path for axis-aligned transforms.
        let (x0, y0) = self.map(x, y);
        let (x1, y1) = self.map(x + w, y + h);
        let lx = x0.min(x1).floor() as i32;
        let ly = y0.min(y1).floor() as i32;
        let rx = x0.max(x1).ceil() as i32;
        let ry = y0.max(y1).ceil() as i32;
        let clipped_x0 = lx.max(0);
        let clipped_y0 = ly.max(0);
        let clipped_x1 = rx.min(self.bitmap.width as i32);
        let clipped_y1 = ry.min(self.bitmap.height as i32);
        let v = Color::TRANSPARENT.to_bgra_u32();
        for yy in clipped_y0..clipped_y1 {
            let row = (yy as usize) * (self.bitmap.width as usize);
            for xx in clipped_x0..clipped_x1 {
                self.bitmap.pixels[row + xx as usize] = v;
            }
        }
    }

    pub fn stroke_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        // Under rotation/skew, stroke all four edges of the transformed quad.
        // Save and restore the path so this is a standalone operation.
        let m = self.state().transform;
        let has_rotation = m[1].abs() > 1e-6 || m[2].abs() > 1e-6;

        if has_rotation {
            let saved_path = std::mem::take(&mut self.path);
            let saved_cursor = self.cursor.take();
            self.move_to(x, y);
            self.line_to(x + w, y);
            self.line_to(x + w, y + h);
            self.line_to(x, y + h);
            self.close_path();
            self.stroke();
            self.path = saved_path;
            self.cursor = saved_cursor;
            return;
        }

        // Fast path for axis-aligned transforms.
        let (x0, y0) = self.map(x, y);
        let (x1, y1) = self.map(x + w, y + h);
        let lx = x0.min(x1).floor() as i32;
        let ly = y0.min(y1).floor() as i32;
        let rx = x0.max(x1).ceil() as i32;
        let ry = y0.max(y1).ceil() as i32;
        let mut c = self.state().stroke;
        c.a = ((c.a as f32) * self.state().global_alpha) as u8;
        self.bitmap.stroke_rect(lx, ly, rx - lx, ry - ly, c);
    }

    // Path API. The path is rebuilt from scratch on `begin_path`;
    // `fill()` / `stroke()` consume it but the path stays for chained
    // calls (per spec — Canvas doesn't auto-reset after fill/stroke).

    pub fn begin_path(&mut self) {
        self.path.clear();
        self.cursor = None;
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        self.cursor = Some(p);
        self.path.push(PathOp::MoveTo(p.0, p.1));
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        self.cursor = Some(p);
        self.path.push(PathOp::LineTo(p.0, p.1));
    }

    pub fn close_path(&mut self) {
        self.path.push(PathOp::Close);
    }

    /// `quadraticCurveTo(cpx, cpy, x, y)` — flatten a quadratic Bezier
    /// from the current cursor to (x, y) via the control point
    /// (cpx, cpy). Subdivides into ~16 line segments.
    pub fn quadratic_curve_to(&mut self, cpx: f32, cpy: f32, x: f32, y: f32) {
        let (sx, sy) = self.cursor.unwrap_or((0.0, 0.0));
        let (tx, ty) = self.map(x, y);
        let (tcpx, tcpy) = self.map(cpx, cpy);
        let segments = 16;
        for i in 1..=segments {
            let t = i as f32 / segments as f32;
            let omt = 1.0 - t;
            let px = omt * omt * sx + 2.0 * omt * t * tcpx + t * t * tx;
            let py = omt * omt * sy + 2.0 * omt * t * tcpy + t * t * ty;
            self.cursor = Some((px, py));
            self.path.push(PathOp::LineTo(px, py));
        }
    }

    /// `bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y)` — cubic Bezier
    /// flattened into line segments.
    pub fn bezier_curve_to(&mut self, cp1x: f32, cp1y: f32, cp2x: f32, cp2y: f32, x: f32, y: f32) {
        let (sx, sy) = self.cursor.unwrap_or((0.0, 0.0));
        let (tx, ty) = self.map(x, y);
        let (tc1x, tc1y) = self.map(cp1x, cp1y);
        let (tc2x, tc2y) = self.map(cp2x, cp2y);
        let segments = 24;
        for i in 1..=segments {
            let t = i as f32 / segments as f32;
            let omt = 1.0 - t;
            let omt2 = omt * omt;
            let omt3 = omt2 * omt;
            let t2 = t * t;
            let t3 = t2 * t;
            let px = omt3 * sx + 3.0 * omt2 * t * tc1x + 3.0 * omt * t2 * tc2x + t3 * tx;
            let py = omt3 * sy + 3.0 * omt2 * t * tc1y + 3.0 * omt * t2 * tc2y + t3 * ty;
            self.cursor = Some((px, py));
            self.path.push(PathOp::LineTo(px, py));
        }
    }

    /// `rect(x, y, w, h)` — adds an unclosed rectangle to the current
    /// path. Spec semantics: it issues `moveTo(x,y)`, three `lineTo`s,
    /// then `closePath`. Useful as the polygon underneath
    /// `fill()` / `stroke()`.
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.move_to(x, y);
        self.line_to(x + w, y);
        self.line_to(x + w, y + h);
        self.line_to(x, y + h);
        self.close_path();
    }

    /// Draw `text` at `(x, y)` using the built-in 5×7 monospace font.
    ///
    /// The y coordinate is interpreted according to `ctx.textBaseline`:
    /// - `'alphabetic'` (default): y is the baseline; glyphs are drawn above it.
    ///   For the 5×7 font the ascent is 5px, so we subtract 5 from y.
    /// - `'top'` / `'hanging'`: y is the top of the em box; no y adjustment.
    /// - `'middle'`: y is the midpoint of the em height; subtract half (3.5px → 3px).
    /// - `'ideographic'` / `'bottom'`: y is the bottom of the em box; subtract
    ///   the full 7px glyph height.
    ///
    /// The x coordinate is interpreted according to `ctx.textAlign`:
    /// - `'left'` / `'start'` (default): x is the left edge of the text.
    /// - `'center'`: x is the center of the text.
    /// - `'right'` / `'end'`: x is the right edge of the text.
    ///
    /// When `ctx.font` sets a larger font size we scale the 5×7 bitmap font
    /// by an integer factor (clamped to [1, 8]) so rendering stays crisp.
    pub fn fill_text(&mut self, text: &str, x: f32, y: f32) {
        let mut c = self.state().fill;
        c.a = ((c.a as f32) * self.state().global_alpha) as u8;

        // Scale factor derived from ctx.font size vs the native 5×7 glyph height.
        const NATIVE_H: f32 = 7.0; // pixel height of the 5×7 glyph
        const NATIVE_ASCENT: f32 = 5.0; // distance from top of glyph to baseline
        let font_px = self.state().font_size_px;
        let scale = ((font_px / NATIVE_H).round() as i32).clamp(1, 8) as i32;

        // Compute the y-offset based on textBaseline and current scale.
        let ascent = NATIVE_ASCENT * scale as f32;
        let total_h = NATIVE_H * scale as f32;
        let y_offset = match self.state().text_baseline {
            TextBaseline::Top | TextBaseline::Hanging => 0.0,
            TextBaseline::Alphabetic => -ascent,
            TextBaseline::Middle => -(total_h / 2.0),
            TextBaseline::Ideographic | TextBaseline::Bottom => -total_h,
        };

        // Compute x-offset based on textAlign.
        let glyph_advance = (font5x7::GLYPH_W as i32 + 1) * scale;
        let text_width = text.chars().count() as f32 * glyph_advance as f32;
        let x_offset = match self.state().text_align {
            TextAlign::Left | TextAlign::Start => 0.0,
            TextAlign::Center => -(text_width / 2.0),
            TextAlign::Right | TextAlign::End => -text_width,
        };

        let (tx, ty) = self.map(x + x_offset, y + y_offset);
        let base_x = tx.round() as i32;
        let base_y = ty.round() as i32;
        let mut col = 0i32;
        for ch in text.chars() {
            let rows = font5x7::glyph(ch);
            for (row_i, row_mask) in rows.iter().enumerate() {
                for bit in 0..5i32 {
                    if (row_mask >> (4 - bit)) & 1 == 1 {
                        // Scale each source pixel to a scale×scale block.
                        let src_x = base_x + col + bit * scale;
                        let src_y = base_y + row_i as i32 * scale;
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let px = src_x + dx;
                                let py = src_y + dy;
                                if c.a == 255 {
                                    self.bitmap.put_pixel(px, py, c);
                                } else if c.a > 0 {
                                    if py >= 0
                                        && (py as u32) < self.bitmap.height
                                        && px >= 0
                                        && (px as u32) < self.bitmap.width
                                    {
                                        let idx = py as usize * self.bitmap.width as usize
                                            + px as usize;
                                        self.bitmap.pixels[idx] =
                                            blend_bgra(self.bitmap.pixels[idx], c);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            col += glyph_advance;
        }
    }

    /// Full `TextMetrics` for `text` rendered with the current font.
    ///
    /// The 5×7 bitmap font has the following metrics (at scale=1):
    ///   - total height:   7 px
    ///   - ascent (top→baseline):  5 px  (5 rows of ink, 2 below for descenders on 'g','y',…)
    ///   - descent (baseline→bottom): 2 px
    ///
    /// All distances are in CSS pixels, scaled by the current font scale factor.
    pub fn measure_text(&self, text: &str) -> TextMetrics {
        // Font constants for the built-in 5×7 bitmap font.
        const NATIVE_H: f32 = 7.0;
        const NATIVE_ASCENT: f32 = 5.0;  // pixels above the baseline
        const NATIVE_DESCENT: f32 = 2.0; // pixels below the baseline

        let font_px = self.state().font_size_px;
        let scale = ((font_px / NATIVE_H).round() as i32).clamp(1, 8) as f32;

        let width = font5x7::measure(text) as f32 * scale;
        let ascent  = NATIVE_ASCENT  * scale;
        let descent = NATIVE_DESCENT * scale;

        // `actual_bounding_box_left` / `_right` depend on textAlign, because
        // the spec defines them relative to the `x` argument passed to fillText.
        // At textAlign="left" x is the left edge, so left=0, right=width.
        // At textAlign="center" x is the center, so left=width/2, right=width/2.
        // At textAlign="right" x is the right edge, so left=width, right=0.
        let (bb_left, bb_right) = match self.state().text_align {
            TextAlign::Left | TextAlign::Start => (0.0, width),
            TextAlign::Center               => (width / 2.0, width / 2.0),
            TextAlign::Right | TextAlign::End  => (width, 0.0),
        };

        TextMetrics {
            width,
            actual_bounding_box_ascent:  ascent,
            actual_bounding_box_descent: descent,
            font_bounding_box_ascent:    ascent,
            font_bounding_box_descent:   descent,
            em_height_ascent:            ascent,
            em_height_descent:           descent,
            actual_bounding_box_left:    bb_left,
            actual_bounding_box_right:   bb_right,
            alphabetic_baseline:         0.0,
            hanging_baseline:            ascent * 0.8,
            ideographic_baseline:        -descent,
        }
    }

    /// `arc(cx, cy, r, startAngle, endAngle, anticlockwise)`. We
    /// flatten the arc into line segments — segment count scales
    /// with radius so big circles stay smooth without paying for it
    /// on tiny ones. If no `MoveTo` has been issued yet, the arc's
    /// first sampled point implicitly starts a new sub-path; if one
    /// has, we connect via `LineTo` (matches the Web spec's "if the
    /// path is non-empty, add a straight line from the last point to
    /// the first arc point").
    pub fn arc(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        anticlockwise: bool,
    ) {
        let r = radius.abs();
        if r == 0.0 {
            return;
        }
        // Pick a step count that gives ~1px chord error at the rim.
        let two_pi = std::f32::consts::TAU;
        let sweep = if anticlockwise {
            let mut s = end_angle - start_angle;
            // Anticlockwise sweep should be negative; wrap into [-2π, 0).
            while s > 0.0 {
                s -= two_pi;
            }
            s
        } else {
            let mut s = end_angle - start_angle;
            while s < 0.0 {
                s += two_pi;
            }
            s
        };
        let arc_len = r * sweep.abs();
        let steps = ((arc_len / 2.0).ceil() as i32).clamp(8, 256);
        let mut first_pt: Option<(f32, f32)> = None;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let theta = start_angle + sweep * t;
            let x = cx + r * theta.cos();
            let y = cy + r * theta.sin();
            if first_pt.is_none() {
                first_pt = Some((x, y));
                // Connect the previous subpath end to this start, or
                // begin a fresh subpath if there's nothing yet.
                if self.cursor.is_some() {
                    self.line_to(x, y);
                } else {
                    self.move_to(x, y);
                }
            } else {
                self.line_to(x, y);
            }
        }
    }

    /// `arcTo(x1, y1, x2, y2, radius)` — add an arc of the given radius tangent
    /// to the line from the current point to (x1,y1) and the line from (x1,y1)
    /// to (x2,y2). Adds a `lineTo` the first tangent point, then the arc to the
    /// second tangent point. Degenerate cases (no current point, zero radius, or
    /// collinear points) fall back to a straight `lineTo(x1,y1)`.
    ///
    /// Coordinates are in user space and mapped through the current transform,
    /// matching the inline `moveTo`/`lineTo`/`arc` convention.
    pub fn arc_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, radius: f32) {
        // The current point in USER space — invert the transform on the device
        // cursor so the tangent geometry is computed in user space, then mapped.
        let (x0, y0) = match self.cursor_user() {
            Some(p) => p,
            None => {
                self.move_to(x1, y1);
                return;
            }
        };
        if let Some(seg) = compute_arc_to(x0, y0, x1, y1, x2, y2, radius) {
            self.line_to(seg.t1.0, seg.t1.1);
            self.arc(
                seg.center.0,
                seg.center.1,
                seg.radius,
                seg.start_angle,
                seg.end_angle,
                seg.anticlockwise,
            );
        } else {
            self.line_to(x1, y1);
        }
    }

    /// The current sub-path point in user space (inverse of the current
    /// transform applied to the device cursor). `None` when no point yet.
    fn cursor_user(&self) -> Option<(f32, f32)> {
        let (cx, cy) = self.cursor?;
        let m = self.state().transform;
        let det = m[0] * m[3] - m[2] * m[1];
        if det.abs() < 1e-12 {
            return Some((cx, cy));
        }
        let inv = 1.0 / det;
        let rx = cx - m[4];
        let ry = cy - m[5];
        Some((
            (m[3] * rx - m[2] * ry) * inv,
            (-m[1] * rx + m[0] * ry) * inv,
        ))
    }

    /// `ellipse(cx, cy, rx, ry, rotation, start, end, anticlockwise)`.
    #[allow(clippy::too_many_arguments)]
    pub fn ellipse(
        &mut self,
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        rotation: f32,
        start_angle: f32,
        end_angle: f32,
        anticlockwise: bool,
    ) {
        let (rx, ry) = (rx.abs(), ry.abs());
        if rx == 0.0 || ry == 0.0 {
            return;
        }
        let two_pi = std::f32::consts::TAU;
        let sweep = if anticlockwise {
            let mut s = end_angle - start_angle;
            while s > 0.0 {
                s -= two_pi;
            }
            s
        } else {
            let mut s = end_angle - start_angle;
            while s < 0.0 {
                s += two_pi;
            }
            s
        };
        let (rot_sin, rot_cos) = rotation.sin_cos();
        let steps = (((rx.max(ry)) * sweep.abs() / 2.0).ceil() as i32).clamp(8, 256);
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let theta = start_angle + sweep * t;
            let ex = rx * theta.cos();
            let ey = ry * theta.sin();
            let x = cx + ex * rot_cos - ey * rot_sin;
            let y = cy + ex * rot_sin + ey * rot_cos;
            if i == 0 && self.cursor.is_none() {
                self.move_to(x, y);
            } else {
                self.line_to(x, y);
            }
        }
    }

    /// Set the `globalCompositeOperation`.
    pub fn set_composite_op(&mut self, op: CompositeOp) {
        self.state_mut().composite_op = op;
    }

    /// The current `globalCompositeOperation`.
    pub fn composite_op(&self) -> CompositeOp {
        self.state().composite_op
    }

    /// Set the fill style to a tiled image pattern.
    pub fn set_fill_pattern(
        &mut self,
        width: u32,
        height: u32,
        pixels: std::rc::Rc<Vec<u32>>,
        repeat: PatternRepeat,
    ) {
        self.state_mut().fill_style = Some(FillStyle::Pattern(Pattern {
            width,
            height,
            pixels,
            repeat,
        }));
    }

    /// Stroke the current path using the current stroke color and
    /// line width. V1 only honours line widths of ~1px for the
    /// software rasterizer — wider lines fall back to repeated
    /// 1px lines with a small offset.
    pub fn stroke(&mut self) {
        let ops = self.path.clone();
        self.stroke_device_ops(&ops);
    }

    /// Stroke a `Path2D` (recorded in user space) under the current transform.
    pub fn stroke_path(&mut self, path: &Path2D) {
        let ops = path.ops_transformed(self.state().transform);
        self.stroke_device_ops(&ops);
    }

    /// Stroke a sequence of DEVICE-space path ops with the current stroke color,
    /// width, and composite op. Shared by `stroke` and `stroke_path`.
    fn stroke_device_ops(&mut self, ops: &[PathOp]) {
        let mut c = self.state().stroke;
        c.a = ((c.a as f32) * self.state().global_alpha) as u8;
        let op = self.state().composite_op;
        let width = self.state().line_width.max(1.0).round() as i32;
        let half = width / 2;
        let mut sub_path_start: Option<(f32, f32)> = None;
        let mut last: Option<(f32, f32)> = None;
        for pop in ops {
            match *pop {
                PathOp::MoveTo(x, y) => {
                    sub_path_start = Some((x, y));
                    last = Some((x, y));
                }
                PathOp::LineTo(x, y) => {
                    if let Some((lx, ly)) = last {
                        for dy in -half..=half {
                            for dx in -half..=half {
                                self.draw_line_op(
                                    lx + dx as f32,
                                    ly + dy as f32,
                                    x + dx as f32,
                                    y + dy as f32,
                                    c,
                                    op,
                                );
                            }
                        }
                    }
                    last = Some((x, y));
                }
                PathOp::Close => {
                    if let (Some((lx, ly)), Some(start)) = (last, sub_path_start) {
                        for dy in -half..=half {
                            for dx in -half..=half {
                                self.draw_line_op(
                                    lx + dx as f32,
                                    ly + dy as f32,
                                    start.0 + dx as f32,
                                    start.1 + dy as f32,
                                    c,
                                    op,
                                );
                            }
                        }
                        last = Some(start);
                    }
                }
            }
        }
    }

    /// Fill the current path with the current fill color or gradient.
    ///
    /// `rule` controls whether the nonzero winding rule (default, `FillRule::Nonzero`)
    /// or the even-odd rule (`FillRule::EvenOdd`) is used to determine the inside
    /// of overlapping sub-paths.  Bounding box is computed from the path vertices.
    pub fn fill(&mut self, rule: FillRule) {
        let ops = self.path.clone();
        self.fill_device_ops(&ops, rule);
    }

    /// Fill a `Path2D` (recorded in user space) under the current transform and
    /// fill style.
    pub fn fill_path(&mut self, path: &Path2D, rule: FillRule) {
        let ops = path.ops_transformed(self.state().transform);
        self.fill_device_ops(&ops, rule);
    }

    /// Paint a real soft shadow for the given device-space polygons (used by the
    /// fill/stroke paths when shadowColor/shadowBlur are set).
    ///
    /// This is the Chrome/Skia shadow model: the shape's *silhouette* (its alpha
    /// coverage) is rendered in `shadowColor`, displaced by (shadowOffsetX,
    /// shadowOffsetY), Gaussian-blurred, and composited **behind** the shape
    /// (drawn here, before the fill itself paints over it).
    ///
    /// Blur amount: Blink converts `shadowBlur` → a Gaussian standard deviation
    /// via `ShadowData::BlurRadiusToStdDev(radius) = radius * 0.5`
    /// (`CanvasRenderingContext2DState::ShadowBlurAsSigma`), so
    /// `sigma = shadowBlur / 2`. The Gaussian is approximated by three
    /// successive box blurs per the SVG 1.1 §15.17 / CSS Filter Effects
    /// `feGaussianBlur` algorithm (box size `d = floor(sigma*3*sqrt(2π)/4 + 0.5)`),
    /// which is exactly Skia's `SkBlurMask` fast path.
    #[allow(clippy::too_many_arguments)]
    fn paint_shadow_glow(
        &mut self,
        polys: &[Vec<(f32, f32)>],
        rule: FillRule,
        scolor: Color,
        blur: f32,
        sox: f32,
        soy: f32,
        global_alpha: f32,
        clip: Option<ClipRect>,
    ) {
        // Shape bounds (unoffset shape space).
        let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for poly in polys {
            for &(x, y) in poly {
                min_x = min_x.min(x); min_y = min_y.min(y);
                max_x = max_x.max(x); max_y = max_y.max(y);
            }
        }
        if !min_x.is_finite() { return; }

        // sigma = shadowBlur / 2 (Blink BlurRadiusToStdDev).
        let sigma = (blur * 0.5).max(0.0);
        // The blur spreads ~3σ in each direction; pad the mask so the falloff
        // tail is fully captured. +2 for the supersampled-edge antialias.
        let pad = (sigma * 3.0).ceil() as i32 + 2;

        // Local coverage-mask buffer covering the (padded) shape bbox.
        let mlo_x = min_x.floor() as i32 - pad;
        let mlo_y = min_y.floor() as i32 - pad;
        let mhi_x = max_x.ceil() as i32 + pad;
        let mhi_y = max_y.ceil() as i32 + pad;
        let mw = (mhi_x - mlo_x).max(1) as usize;
        let mh = (mhi_y - mlo_y).max(1) as usize;
        // Guard against pathological allocation on huge shapes.
        if mw.saturating_mul(mh) > 64 * 1024 * 1024 {
            return;
        }
        let mut mask = vec![0.0f32; mw * mh];

        let inside = |w: i32| match rule {
            FillRule::Nonzero => w != 0,
            FillRule::EvenOdd => w % 2 != 0,
        };
        // Antialiased coverage via 4×4 supersampling per pixel — gives the
        // silhouette smooth edges (Chrome rasterizes the AA mask before blur).
        const SS: i32 = 4;
        let point_inside = |px: f32, py: f32| -> bool {
            let mut wind = 0i32;
            for poly in polys {
                for w in poly.windows(2) {
                    let (ax, ay) = w[0]; let (bx, by) = w[1];
                    if (ay > py) != (by > py) {
                        let t = (py - ay) / (by - ay);
                        if ax + t * (bx - ax) > px {
                            wind += if ay > by { 1 } else { -1 };
                        }
                    }
                }
            }
            inside(wind)
        };
        for my in 0..mh {
            let base_y = mlo_y as f32 + my as f32;
            for mx in 0..mw {
                let base_x = mlo_x as f32 + mx as f32;
                let mut hits = 0i32;
                for sj in 0..SS {
                    let py = base_y + (sj as f32 + 0.5) / SS as f32;
                    for si in 0..SS {
                        let px = base_x + (si as f32 + 0.5) / SS as f32;
                        if point_inside(px, py) { hits += 1; }
                    }
                }
                if hits != 0 {
                    mask[my * mw + mx] = hits as f32 / (SS * SS) as f32;
                }
            }
        }

        // Gaussian-blur the coverage mask in place (no-op when sigma≈0).
        box_blur_gaussian(&mut mask, mw, mh, sigma);

        // Composite the blurred coverage, tinted by shadowColor (× globalAlpha),
        // BEHIND the shape — displaced by the shadow offset. Straight-alpha
        // source-over so it layers under the subsequent fill.
        let bw = self.bitmap.width as i32;
        let bh = self.bitmap.height as i32;
        let base_a = (scolor.a as f32) * global_alpha;
        if base_a <= 0.0 { return; }
        for my in 0..mh {
            let cov_row = my * mw;
            // Destination row = mask row + shape-space origin + shadow offset.
            let dy = mlo_y + my as i32 + soy.round() as i32;
            if dy < 0 || dy >= bh { continue; }
            if let Some(c) = clip { if dy < c.y0 || dy >= c.y1 { continue; } }
            for mx in 0..mw {
                let cov = mask[cov_row + mx];
                if cov <= 0.0 { continue; }
                let dx = mlo_x + mx as i32 + sox.round() as i32;
                if dx < 0 || dx >= bw { continue; }
                if let Some(c) = clip { if dx < c.x0 || dx >= c.x1 { continue; } }
                let a = (base_a * cov).min(255.0);
                if a < 0.5 { continue; }
                let sc = Color { r: scolor.r, g: scolor.g, b: scolor.b, a: a as u8 };
                self.bitmap.blend_pixel(dx, dy, sc);
            }
        }
    }

    /// Fill a sequence of DEVICE-space path ops with the current fill style and
    /// composite op. The core scanline polygon filler shared by `fill`,
    /// `fill_path`, and `fill_rect`'s rotated path.
    fn fill_device_ops(&mut self, ops: &[PathOp], rule: FillRule) {
        // Snapshot fill style and clip so we don't borrow `self` inside the pixel loop.
        let global_alpha = self.state().global_alpha;
        let fill_style = self.state().fill_style.clone();
        let clip = self.state().clip;
        let op = self.state().composite_op;
        let solid_color = {
            let mut c = self.state().fill;
            c.a = ((c.a as f32) * global_alpha) as u8;
            c
        };
        // Collect sub-paths as closed polygons (auto-close each).
        let mut polys: Vec<Vec<(f32, f32)>> = Vec::new();
        let mut cur: Vec<(f32, f32)> = Vec::new();
        for op in ops {
            match op {
                PathOp::MoveTo(x, y) => {
                    if !cur.is_empty() {
                        polys.push(std::mem::take(&mut cur));
                    }
                    cur.push((*x, *y));
                }
                PathOp::LineTo(x, y) => {
                    cur.push((*x, *y));
                }
                PathOp::Close => {
                    if !cur.is_empty() {
                        let start = cur[0];
                        if cur.last() != Some(&start) {
                            cur.push(start);
                        }
                        polys.push(std::mem::take(&mut cur));
                    }
                }
            }
        }
        if !cur.is_empty() {
            polys.push(cur);
        }
        // ── shadowBlur / shadowColor glow pre-pass ────────────────────────────
        // Draw a soft glow of the shape UNDER the fill (Canvas shadows paint
        // behind the source) when a shadow is set. Falloff over `shadowBlur` px.
        {
            let sh = {
                let s = self.state();
                (s.shadow_color, s.shadow_blur, s.shadow_offset_x, s.shadow_offset_y, s.global_alpha)
            };
            let (scolor, blur, sox, soy, ga) = sh;
            if scolor.a > 0 && (blur > 0.0 || sox != 0.0 || soy != 0.0) && !polys.is_empty() {
                self.paint_shadow_glow(&polys, rule, scolor, blur, sox, soy, ga, clip);
            }
        }
        // Nonzero winding rule fill per scanline (HTML Canvas spec default).
        // For each scanline we collect (x_intersection, winding_delta) pairs:
        //   +1 when the edge crosses upward (ay > by, i.e. going toward smaller y)
        //   -1 when the edge crosses downward (ay < by)
        // We sort by x, then walk left-to-right accumulating a running winding
        // count. A pixel span is inside when the running count is non-zero.
        let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for poly in &polys {
            for &(x, y) in poly {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        if !min_x.is_finite() {
            return;
        }
        // Apply clip region to the scanline range.
        let y0 = {
            let raw = (min_y.floor() as i32).max(0);
            match clip { Some(cr) => raw.max(cr.y0), None => raw }
        };
        let y1 = {
            let raw = (max_y.ceil() as i32).min(self.bitmap.height as i32);
            match clip { Some(cr) => raw.min(cr.y1), None => raw }
        };
        let x0 = {
            let raw = (min_x.floor() as i32).max(0);
            match clip { Some(cr) => raw.max(cr.x0), None => raw }
        };
        let x1 = {
            let raw = (max_x.ceil() as i32).min(self.bitmap.width as i32);
            match clip { Some(cr) => raw.min(cr.x1), None => raw }
        };
        if y1 <= y0 || x1 <= x0 {
            return;
        }
        for y in y0..y1 {
            let yc = y as f32 + 0.5;
            // Collect (x_intersect, winding_delta) for all edges crossing yc.
            let mut crossings: Vec<(f32, i32)> = Vec::new();
            for poly in &polys {
                if poly.len() < 2 {
                    continue;
                }
                for w in poly.windows(2) {
                    let (ax, ay) = w[0];
                    let (bx, by) = w[1];
                    if (ay > yc) == (by > yc) {
                        continue; // edge doesn't cross this scanline
                    }
                    let t = (yc - ay) / (by - ay);
                    let xi = ax + t * (bx - ax);
                    // Winding: +1 for upward crossing (a below yc, b above),
                    // -1 for downward crossing (a above yc, b below).
                    let delta = if ay > by { 1i32 } else { -1i32 };
                    crossings.push((xi, delta));
                }
            }
            crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Walk left-to-right; fill spans according to the fill rule:
            //   Nonzero: inside when running winding count is non-zero.
            //   EvenOdd: inside when crossing count so far is odd.
            let is_inside = |w: i32| match rule {
                FillRule::Nonzero => w != 0,
                FillRule::EvenOdd => w % 2 != 0,
            };
            let row = (y as usize) * (self.bitmap.width as usize);
            let mut winding = 0i32;
            let mut span_start: Option<f32> = None;

            for (xi, delta) in &crossings {
                let was_inside = is_inside(winding);
                winding += delta;
                let now_inside = is_inside(winding);
                if !was_inside && now_inside {
                    // Entering a filled region.
                    span_start = Some(*xi);
                } else if was_inside && !now_inside {
                    // Leaving a filled region.
                    if let Some(sx) = span_start.take() {
                        let lx = (sx.floor() as i32).max(x0);
                        let rx = (xi.ceil() as i32).min(x1);
                        if rx > lx {
                            for xx in lx..rx {
                                let c = match &fill_style {
                                    Some(fs) => fs.sample(xx as f32 + 0.5, yc, global_alpha),
                                    None => solid_color,
                                };
                                self.put_composite(row + xx as usize, c, op);
                            }
                        }
                    }
                }
            }
            // If still inside at the end of all crossings, fill the trailing
            // span (can happen with malformed/open paths).
            if is_inside(winding) {
                if let Some(sx) = span_start.take() {
                    let lx = (sx.floor() as i32).max(x0);
                    let rx = x1;
                    if rx > lx {
                        for xx in lx..rx {
                            let c = match &fill_style {
                                Some(fs) => fs.sample(xx as f32 + 0.5, yc, global_alpha),
                                None => solid_color,
                            };
                            self.put_composite(row + xx as usize, c, op);
                        }
                    }
                }
            }
        }
    }

    /// Internal: Bresenham-style integer line draw. Bypasses the
    /// transform — caller is expected to have transformed already.
    fn draw_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, c: Color) {
        let mut x = x0.round() as i32;
        let mut y = y0.round() as i32;
        let xe = x1.round() as i32;
        let ye = y1.round() as i32;
        let dx = (xe - x).abs();
        let dy = -(ye - y).abs();
        let sx = if x < xe { 1 } else { -1 };
        let sy = if y < ye { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            // Alpha-blend, not overwrite — a semi-transparent stroke (e.g. the
            // 0.1-alpha particles.js connectors) must tint/accumulate, not
            // obliterate what's under it.
            self.bitmap.blend_pixel(x, y, c);
            if x == xe && y == ye {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Composite-aware line draw. For `SourceOver` it is byte-identical to
    /// `draw_line` (uses `Bitmap::blend_pixel`); other ops route per-pixel
    /// through `composite_pixel`.
    fn draw_line_op(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, c: Color, op: CompositeOp) {
        if matches!(op, CompositeOp::SourceOver) {
            self.draw_line(x0, y0, x1, y1, c);
            return;
        }
        let mut x = x0.round() as i32;
        let mut y = y0.round() as i32;
        let xe = x1.round() as i32;
        let ye = y1.round() as i32;
        let dx = (xe - x).abs();
        let dy = -(ye - y).abs();
        let sx = if x < xe { 1 } else { -1 };
        let sy = if y < ye { 1 } else { -1 };
        let mut err = dx + dy;
        let (bw, bh) = (self.bitmap.width as i32, self.bitmap.height as i32);
        loop {
            if x >= 0 && y >= 0 && x < bw && y < bh {
                let idx = (y as usize) * (self.bitmap.width as usize) + x as usize;
                self.bitmap.pixels[idx] = composite_pixel(self.bitmap.pixels[idx], c, op);
            }
            if x == xe && y == ye {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// `clip(path)` — intersect the clip with the AABB of a `Path2D` under the
    /// current transform (matching the inline `clip()` AABB behavior).
    pub fn clip_path(&mut self, path: &Path2D) {
        let m = self.state().transform;
        let ops = path.ops_transformed(m);
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for op in &ops {
            let (x, y) = match op {
                PathOp::MoveTo(x, y) | PathOp::LineTo(x, y) => (*x, *y),
                PathOp::Close => continue,
            };
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        if !min_x.is_finite() {
            return;
        }
        let new = ClipRect {
            x0: min_x.floor() as i32,
            y0: min_y.floor() as i32,
            x1: max_x.ceil() as i32,
            y1: max_y.ceil() as i32,
        };
        let merged = match self.state().clip {
            Some(e) => ClipRect {
                x0: e.x0.max(new.x0),
                y0: e.y0.max(new.y0),
                x1: e.x1.min(new.x1),
                y1: e.y1.min(new.y1),
            },
            None => new,
        };
        self.state_mut().clip = Some(merged);
    }

    /// `isPointInPath(x, y[, fillRule])` against the CURRENT path. The point is
    /// in user space and tested against the device-space path under the current
    /// transform.
    pub fn is_point_in_path(&self, x: f32, y: f32, rule: FillRule) -> bool {
        let (px, py) = self.map(x, y);
        point_in_device_ops(&self.path, px, py, rule)
    }

    /// `isPointInPath(path, x, y[, fillRule])` against a `Path2D`.
    pub fn is_point_in_path2d(&self, path: &Path2D, x: f32, y: f32, rule: FillRule) -> bool {
        let (px, py) = self.map(x, y);
        let ops = path.ops_transformed(self.state().transform);
        point_in_device_ops(&ops, px, py, rule)
    }

    /// `isPointInStroke(x, y)` against the CURRENT path — approximated by the
    /// minimum distance from the point to any path segment being within half
    /// the line width (documented band approximation).
    pub fn is_point_in_stroke(&self, x: f32, y: f32) -> bool {
        let (px, py) = self.map(x, y);
        let half = (self.state().line_width.max(1.0) / 2.0).max(0.5);
        point_near_device_ops(&self.path, px, py, half)
    }

    /// `isPointInStroke(path, x, y)` against a `Path2D`.
    pub fn is_point_in_stroke_path(&self, path: &Path2D, x: f32, y: f32) -> bool {
        let (px, py) = self.map(x, y);
        let half = (self.state().line_width.max(1.0) / 2.0).max(0.5);
        let ops = path.ops_transformed(self.state().transform);
        point_near_device_ops(&ops, px, py, half)
    }

    /// Snapshot of the backing pixels as a fresh BGRA buffer (used by
    /// `transferToImageBitmap` / pattern capture).
    pub fn snapshot_pixels(&self) -> Vec<u32> {
        self.bitmap.pixels.clone()
    }
}

/// Build closed polygons (device space) from a sequence of path ops, exactly
/// as the scanline filler does, then run the chosen fill-rule winding test at
/// (px, py).
fn point_in_device_ops(ops: &[PathOp], px: f32, py: f32, rule: FillRule) -> bool {
    let mut polys: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut cur: Vec<(f32, f32)> = Vec::new();
    for op in ops {
        match op {
            PathOp::MoveTo(x, y) => {
                if !cur.is_empty() {
                    polys.push(std::mem::take(&mut cur));
                }
                cur.push((*x, *y));
            }
            PathOp::LineTo(x, y) => cur.push((*x, *y)),
            PathOp::Close => {
                if !cur.is_empty() {
                    let start = cur[0];
                    if cur.last() != Some(&start) {
                        cur.push(start);
                    }
                    polys.push(std::mem::take(&mut cur));
                }
            }
        }
    }
    if !cur.is_empty() {
        // Auto-close the trailing sub-path for the inside test.
        let start = cur[0];
        if cur.last() != Some(&start) {
            cur.push(start);
        }
        polys.push(cur);
    }
    let mut winding = 0i32;
    let mut crossings = 0i32;
    for poly in &polys {
        if poly.len() < 2 {
            continue;
        }
        for w in poly.windows(2) {
            let (ax, ay) = w[0];
            let (bx, by) = w[1];
            if (ay > py) == (by > py) {
                continue;
            }
            let t = (py - ay) / (by - ay);
            let xi = ax + t * (bx - ax);
            if xi > px {
                crossings += 1;
                winding += if ay > by { 1 } else { -1 };
            }
        }
    }
    match rule {
        FillRule::Nonzero => winding != 0,
        FillRule::EvenOdd => crossings % 2 != 0,
    }
}

/// True when (px, py) is within `half` of any segment of the device-space path.
fn point_near_device_ops(ops: &[PathOp], px: f32, py: f32, half: f32) -> bool {
    let mut sub_start: Option<(f32, f32)> = None;
    let mut last: Option<(f32, f32)> = None;
    let half2 = half * half;
    for op in ops {
        match *op {
            PathOp::MoveTo(x, y) => {
                sub_start = Some((x, y));
                last = Some((x, y));
            }
            PathOp::LineTo(x, y) => {
                if let Some((lx, ly)) = last {
                    if dist2_point_seg(px, py, lx, ly, x, y) <= half2 {
                        return true;
                    }
                }
                last = Some((x, y));
            }
            PathOp::Close => {
                if let (Some((lx, ly)), Some(s)) = (last, sub_start) {
                    if dist2_point_seg(px, py, lx, ly, s.0, s.1) <= half2 {
                        return true;
                    }
                    last = Some(s);
                }
            }
        }
    }
    false
}

/// Squared distance from point (px,py) to segment (ax,ay)->(bx,by).
fn dist2_point_seg(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-9 {
        let ex = px - ax;
        let ey = py - ay;
        return ex * ex + ey * ey;
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
    let cx = ax + t * dx;
    let cy = ay + t * dy;
    let ex = px - cx;
    let ey = py - cy;
    ex * ex + ey * ey
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_blur_paints_a_falloff_glow_band() {
        // A solid disc with shadowBlur must paint a soft glow OUTSIDE the disc
        // (alpha falling off over the blur radius) and leave pixels well beyond
        // the blur untouched. `pixels` is packed BGRA u32; alpha is the high byte.
        let mut ctx = CanvasContext2D::new(100, 100);
        let gold = Color { r: 255, g: 215, b: 0, a: 255 };
        ctx.set_fill_color(gold);
        ctx.set_shadow_color(gold);
        ctx.set_shadow_blur(16.0);
        ctx.begin_path();
        ctx.arc(50.0, 50.0, 10.0, 0.0, std::f32::consts::PI * 2.0, false);
        ctx.fill(FillRule::Nonzero);
        let alpha = |x: usize, y: usize| ((ctx.bitmap.pixels[y * 100 + x] >> 24) & 0xFF) as u32;
        // Disc interior is fully filled (centre at 50,50; radius 10).
        assert!(alpha(50, 50) > 200, "disc centre should be filled, got {}", alpha(50, 50));
        // Just outside the disc edge (~6px out, well inside the 16px blur) there
        // must be a partially-lit glow pixel.
        let band = alpha(50, 34); // 16px above centre = 6px past the r=10 edge
        assert!(band > 0 && band < 255, "glow band should be partial, got {}", band);
        // Far beyond shape+blur stays fully transparent.
        assert_eq!(alpha(95, 95), 0, "corner far beyond shape+blur stays transparent");
    }

    #[test]
    fn fill_rect_paints_pixels() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        });
        ctx.fill_rect(5.0, 5.0, 10.0, 10.0);
        // Centre pixel must be red, corner pixel still transparent.
        assert_eq!(
            ctx.bitmap.pixels[10 * 20 + 10],
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }
            .to_bgra_u32()
        );
        assert_eq!(ctx.bitmap.pixels[0], Color::TRANSPARENT.to_bgra_u32());
    }

    #[test]
    fn clear_rect_resets_to_transparent() {
        let mut ctx = CanvasContext2D::new(10, 10);
        ctx.set_fill_color(Color {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        });
        ctx.fill_rect(0.0, 0.0, 10.0, 10.0);
        ctx.clear_rect(2.0, 2.0, 4.0, 4.0);
        // Inside the cleared rect: transparent.
        assert_eq!(
            ctx.bitmap.pixels[3 * 10 + 3],
            Color::TRANSPARENT.to_bgra_u32()
        );
        // Outside: still green.
        assert_eq!(
            ctx.bitmap.pixels[0],
            Color {
                r: 0,
                g: 255,
                b: 0,
                a: 255
            }
            .to_bgra_u32()
        );
    }

    #[test]
    fn draw_image_blits_scaled_into_context() {
        let mut ctx = CanvasContext2D::new(10, 10);
        let red = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        }
        .to_bgra_u32();
        let src = vec![red; 4]; // 2x2 solid opaque red
        // Scale the 2x2 source to a 4x4 dest at (1,1).
        ctx.draw_image(&src, 2, 2, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 4.0, 4.0);
        // A pixel inside the dest rect is red…
        assert_eq!(
            ctx.bitmap.pixels[3 * 10 + 3],
            red,
            "drawImage should blit the source into the dest rect"
        );
        // …and a pixel outside stays transparent.
        assert_eq!(
            ctx.bitmap.pixels[8 * 10 + 8],
            Color::TRANSPARENT.to_bgra_u32(),
            "outside the dest rect is untouched"
        );
    }

    #[test]
    fn save_restore_pops_state() {
        let mut ctx = CanvasContext2D::new(4, 4);
        ctx.set_fill_color(Color {
            r: 1,
            g: 2,
            b: 3,
            a: 255,
        });
        ctx.save();
        ctx.set_fill_color(Color {
            r: 9,
            g: 9,
            b: 9,
            a: 255,
        });
        assert_eq!(ctx.state().fill.r, 9);
        ctx.restore();
        assert_eq!(ctx.state().fill.r, 1);
    }

    #[test]
    fn translate_affects_fill_origin() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color {
            r: 0,
            g: 0,
            b: 255,
            a: 255,
        });
        ctx.translate(5.0, 5.0);
        ctx.fill_rect(0.0, 0.0, 3.0, 3.0);
        // Pixel (6,6) — well inside the translated rect — must be blue.
        assert_eq!(
            ctx.bitmap.pixels[6 * 20 + 6],
            Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }
            .to_bgra_u32()
        );
        // Pixel (0,0) — outside the translated rect — still transparent.
        assert_eq!(ctx.bitmap.pixels[0], Color::TRANSPARENT.to_bgra_u32());
    }

    #[test]
    fn fill_triangle_via_path() {
        // Build a 6×6 triangle anchored at (2,2)→(8,2)→(5,8) and
        // verify a centre pixel is filled.
        let mut ctx = CanvasContext2D::new(10, 10);
        ctx.set_fill_color(Color {
            r: 255,
            g: 255,
            b: 0,
            a: 255,
        });
        ctx.begin_path();
        ctx.move_to(2.0, 2.0);
        ctx.line_to(8.0, 2.0);
        ctx.line_to(5.0, 8.0);
        ctx.close_path();
        ctx.fill(FillRule::Nonzero);
        // (5, 4) is inside the triangle.
        assert_eq!(
            ctx.bitmap.pixels[4 * 10 + 5],
            Color {
                r: 255,
                g: 255,
                b: 0,
                a: 255
            }
            .to_bgra_u32()
        );
        // (0, 0) outside.
        assert_eq!(ctx.bitmap.pixels[0], Color::TRANSPARENT.to_bgra_u32());
    }

    #[test]
    fn arc_full_circle_fills_centre() {
        // Draw a filled full-circle of radius 5 at (10,10). Centre must
        // be filled; pixel far outside the radius must stay clear.
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color {
            r: 0,
            g: 200,
            b: 0,
            a: 255,
        });
        ctx.begin_path();
        ctx.arc(10.0, 10.0, 5.0, 0.0, std::f32::consts::TAU, false);
        ctx.close_path();
        ctx.fill(FillRule::Nonzero);
        // Centre painted.
        assert_eq!(
            ctx.bitmap.pixels[10 * 20 + 10],
            Color {
                r: 0,
                g: 200,
                b: 0,
                a: 255
            }
            .to_bgra_u32()
        );
        // Far corner untouched.
        assert_eq!(ctx.bitmap.pixels[0], Color::TRANSPARENT.to_bgra_u32());
    }

    #[test]
    fn fill_text_lights_pixels() {
        let mut ctx = CanvasContext2D::new(40, 16);
        ctx.set_fill_color(Color::BLACK);
        ctx.fill_text("HI", 0.0, 0.0);
        // 5x7 'H' has top-row mask 0b10001 so columns 0 and 4 of row 0 should be black.
        assert_eq!(ctx.bitmap.pixels[0], Color::BLACK.to_bgra_u32());
        assert_eq!(ctx.bitmap.pixels[4], Color::BLACK.to_bgra_u32());
        assert_eq!(ctx.bitmap.pixels[1], Color::TRANSPARENT.to_bgra_u32());
    }

    #[test]
    fn measure_text_returns_advance() {
        let ctx = CanvasContext2D::new(80, 16);
        // "HI" = 2 chars × 6 advance − 1 trailing = 11px wide.
        assert_eq!(ctx.measure_text("HI").width as u32, 11);
        assert_eq!(ctx.measure_text("").width as u32, 0);
    }

    #[test]
    fn quadratic_curve_to_appends_path() {
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_fill_color(Color {
            r: 80,
            g: 80,
            b: 80,
            a: 255,
        });
        ctx.begin_path();
        ctx.move_to(2.0, 30.0);
        ctx.quadratic_curve_to(20.0, 2.0, 38.0, 30.0);
        ctx.line_to(2.0, 30.0);
        ctx.close_path();
        ctx.fill(FillRule::Nonzero);
        // Parabola apex sits around (20, 16); pixel well inside the
        // filled region — (20, 24) — should be painted.
        let v = ctx.bitmap.pixels[24 * 40 + 20];
        assert_eq!(
            v,
            Color {
                r: 80,
                g: 80,
                b: 80,
                a: 255
            }
            .to_bgra_u32()
        );
    }

    #[test]
    fn rect_then_fill_paints_inside() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color {
            r: 200,
            g: 0,
            b: 0,
            a: 255,
        });
        ctx.begin_path();
        ctx.rect(2.0, 2.0, 10.0, 10.0);
        ctx.fill(FillRule::Nonzero);
        assert_eq!(
            ctx.bitmap.pixels[6 * 20 + 6],
            Color {
                r: 200,
                g: 0,
                b: 0,
                a: 255
            }
            .to_bgra_u32()
        );
    }

    #[test]
    fn stroke_horizontal_line() {
        let mut ctx = CanvasContext2D::new(10, 10);
        ctx.set_stroke_color(Color {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        });
        ctx.set_line_width(1.0);
        ctx.begin_path();
        ctx.move_to(1.0, 5.0);
        ctx.line_to(8.0, 5.0);
        ctx.stroke();
        // Middle of the line should be painted.
        assert_eq!(ctx.bitmap.pixels[5 * 10 + 4], Color::BLACK.to_bgra_u32());
    }

    // ---- Adversarial verification: faint particles.js stroke claim ----

    fn unpack(v: u32) -> (u8, u8, u8, u8) {
        let a = ((v >> 24) & 0xFF) as u8;
        let r = ((v >> 16) & 0xFF) as u8;
        let g = ((v >> 8) & 0xFF) as u8;
        let b = (v & 0xFF) as u8;
        (r, g, b, a)
    }

    // CONTROL: fill_rect with 0.5 alpha red over opaque white must BLEND
    // to ~ (255,128,128,255). This proves the fill path composites.
    #[test]
    fn fill_rect_blends_low_alpha_over_white() {
        let mut ctx = CanvasContext2D::new(8, 8);
        ctx.bitmap.clear(Color::WHITE);
        ctx.set_fill_color(Color {
            r: 255,
            g: 0,
            b: 0,
            a: 128,
        });
        ctx.fill_rect(0.0, 0.0, 8.0, 8.0);
        let (r, g, b, a) = unpack(ctx.bitmap.pixels[3 * 8 + 3]);
        // blend_bgra: (255*128 + 255*127)/255 = 255 ; (0*128 + 255*127)/255 = 127
        assert_eq!(a, 255, "fill over opaque stays opaque");
        assert!(r >= 250, "red channel ~255, got {r}");
        assert!(
            (120..=135).contains(&g),
            "green blended toward ~127, got {g}"
        );
        assert!(
            (120..=135).contains(&b),
            "blue blended toward ~127, got {b}"
        );
    }

    // A low-alpha stroked line must ALPHA-BLEND over the destination, not
    // overwrite it. Stroke gold rgba(255,215,0,0.1) (a=25) over opaque white:
    // the covered pixel should be ~ (255,251,230,255) — opaque white with a
    // faint gold tint — NOT the raw stroke color (255,215,0,25).
    #[test]
    fn stroke_low_alpha_line_blends_over_destination() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.bitmap.clear(Color::WHITE);
        // gold, 0.1 alpha -> a = round(0.1*255) = 25 (we pass 25 directly).
        ctx.set_stroke_color(Color {
            r: 255,
            g: 215,
            b: 0,
            a: 25,
        });
        ctx.set_line_width(1.0);
        ctx.begin_path();
        ctx.move_to(2.0, 10.0);
        ctx.line_to(17.0, 10.0); // horizontal line on row y=10
        ctx.stroke();
        let (r, _g, b, a) = unpack(ctx.bitmap.pixels[10 * 20 + 9]);
        // BLEND (correct): opaque dest stays opaque; white is only tinted.
        assert_eq!(a, 255, "blend over opaque white stays opaque, got a={a}");
        assert!(r >= 250, "red ~255, got {r}");
        assert!(
            (215..=250).contains(&b),
            "white's blue only faintly tinted (~230), not obliterated, got {b}"
        );
        assert!(b < 255, "but it IS tinted by the gold, got {b}");
    }

    // Two faint lines crossing must BUILD UP: Chrome composites ~0.1 over ~0.1
    // => ~0.19 at the crossing. With source-over blending the crossing alpha is
    // strictly greater than a single line's.
    #[test]
    fn crossing_low_alpha_lines_accumulate() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.bitmap.clear(Color::TRANSPARENT);
        ctx.set_stroke_color(Color {
            r: 255,
            g: 215,
            b: 0,
            a: 25,
        });
        ctx.set_line_width(1.0);
        // Two lines crossing at (10,10).
        ctx.begin_path();
        ctx.move_to(2.0, 10.0);
        ctx.line_to(18.0, 10.0);
        ctx.stroke();
        ctx.begin_path();
        ctx.move_to(10.0, 2.0);
        ctx.line_to(10.0, 18.0);
        ctx.stroke();
        let cross = unpack(ctx.bitmap.pixels[10 * 20 + 10]);
        let single = unpack(ctx.bitmap.pixels[10 * 20 + 5]); // only the horizontal line
        assert_eq!(single.3, 25, "a single faint line is ~0.1 alpha");
        assert!(
            cross.3 > single.3,
            "crossing accumulates: cross={cross:?} single={single:?}"
        );
        assert!(
            cross.3 >= 45,
            "crossing builds toward ~0.19 (a≈47), got {}",
            cross.3
        );
    }

    // SANITY: a small low-alpha fill (the particle DOTS) DOES blend, so
    // the dots are at least correct-color (faint mainly from no-AA hard
    // edges + low alpha), confirming the asymmetry is specific to stroke.
    #[test]
    fn small_low_alpha_arc_fill_blends() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.bitmap.clear(Color::WHITE);
        ctx.set_fill_color(Color {
            r: 255,
            g: 215,
            b: 0,
            a: 76,
        }); // 0.3 alpha
        ctx.begin_path();
        ctx.arc(10.0, 10.0, 3.0, 0.0, std::f32::consts::TAU, false);
        ctx.close_path();
        ctx.fill(FillRule::Nonzero);
        let (_r, _g, b, a) = unpack(ctx.bitmap.pixels[10 * 20 + 10]);
        // Blended over white: alpha stays 255, blue moved off 255 toward 0.
        assert_eq!(a, 255, "dot fill blends onto opaque white");
        assert!(b < 255, "blue channel moved by blend, got {b}");
    }

    // ADVERSARIAL VERIFICATION of the synthesis claim.
    // Reproduces the REAL particles.js pipeline:
    //   1) canvas starts/clears TRANSPARENT (clear_rect each frame),
    //   2) fill a 0.3-alpha gold disc onto the transparent canvas,
    //   3) composite that canvas over the WHITE page via blit_bgra.
    // Then compare the on-page pixel to what Chrome produces.
    #[test]
    fn particles_gold_dot_over_white_page_matches_chrome() {
        // Step 1+2: gold disc on a TRANSPARENT canvas (default ctor state).
        let mut canvas = CanvasContext2D::new(20, 20);
        canvas.set_fill_color(Color {
            r: 255,
            g: 215,
            b: 0,
            a: 76,
        }); // rgba(255,215,0,0.3)
        canvas.begin_path();
        canvas.arc(10.0, 10.0, 3.0, 0.0, std::f32::consts::TAU, false);
        canvas.close_path();
        canvas.fill(FillRule::Nonzero);

        let (cr, cg, cb, ca) = unpack(canvas.bitmap.pixels[10 * 20 + 10]);
        // The canvas pixel itself: src(255,215,0,76) over dst(0,0,0,0).
        // straight-alpha blend over transparent BLACK darkens the RGB.
        eprintln!("canvas pixel (before compositing): rgba({cr},{cg},{cb},{ca})");

        // Step 3: composite the canvas onto a WHITE page (z-index:1 over page).
        let mut page = Bitmap::new(20, 20);
        page.clear(Color::WHITE); // opaque white page background
        page.blit_bgra(0, 0, 20, 20, &canvas.bitmap.pixels);

        let (pr, pg, pb, _pa) = unpack(page.pixels[10 * 20 + 10]);
        eprintln!("on-page pixel (engine):  rgb({pr},{pg},{pb})");

        // What Chrome produces: a 0.3-alpha gold dot drawn straight over the
        // white page (Chrome composites the canvas with PREMULTIPLIED alpha,
        // so the transparent-black background never contaminates the color):
        //   r = 255*0.3 + 255*0.7 = 255
        //   g = 215*0.3 + 255*0.7 = 243
        //   b =   0*0.3 + 255*0.7 = 178 (≈179)
        let (chrome_r, chrome_g, chrome_b) = (255u8, 243u8, 178u8);
        eprintln!("on-page pixel (Chrome):  rgb({chrome_r},{chrome_g},{chrome_b})");

        let dr = (pr as i32 - chrome_r as i32).abs();
        let dg = (pg as i32 - chrome_g as i32).abs();
        let db = (pb as i32 - chrome_b as i32).abs();
        eprintln!("per-channel delta vs Chrome: dr={dr} dg={dg} db={db}");

        // After fixing blend_bgra to a correct straight-alpha source-over, the
        // gold dot drawn on the transparent canvas then composited over the page
        // matches Chrome's bright gold (within rounding) — no transparent-black
        // darkening, no double attenuation.
        assert!(
            dr <= 5 && dg <= 5 && db <= 5,
            "engine rgb({pr},{pg},{pb}) should match Chrome rgb(255,243,178); dr={dr} dg={dg} db={db}"
        );
    }

    // Counter-test the synthesis's anti-aliasing premise in isolation:
    // if we fill the gold disc DIRECTLY over white (no transparent-black
    // intermediate), the color is essentially correct — proving AA/octagon
    // shape is NOT the faintness cause; the transparent-black blend is.
    #[test]
    fn gold_dot_filled_directly_over_white_is_correct() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.bitmap.clear(Color::WHITE); // NO transparent intermediate
        ctx.set_fill_color(Color {
            r: 255,
            g: 215,
            b: 0,
            a: 76,
        });
        ctx.begin_path();
        ctx.arc(10.0, 10.0, 3.0, 0.0, std::f32::consts::TAU, false);
        ctx.close_path();
        ctx.fill(FillRule::Nonzero);
        let (r, g, b, _a) = unpack(ctx.bitmap.pixels[10 * 20 + 10]);
        eprintln!("direct-over-white center pixel: rgb({r},{g},{b}) (Chrome ~255,243,178)");
        // Center is fully covered (interior of disc), so this is the ideal
        // single-blend result: R≈255, close to Chrome. The octagon-vs-circle
        // AA gap only affects the 1px rim, not this dramatic faintness.
        assert!(r >= 250, "direct fill keeps gold warmth, R={r}");
    }

    // JS that sets canvas.width AFTER getContext (particles.js sizes the canvas
    // to the viewport this way) must GROW the backing bitmap — else draws past
    // the initial 300x150 are clipped, cropping a full-viewport canvas to a
    // corner.
    #[test]
    fn resize_grows_bitmap_so_draws_are_not_clipped() {
        let mut ctx = CanvasContext2D::new(300, 150);
        assert_eq!(ctx.size(), (300, 150));
        ctx.resize(800, 600);
        assert_eq!(ctx.size(), (800, 600), "bitmap must grow to the new size");
        // A pixel well past the old 300px width is now writable.
        ctx.set_fill_color(Color {
            r: 255,
            g: 215,
            b: 0,
            a: 255,
        });
        ctx.fill_rect(700.0, 500.0, 10.0, 10.0);
        let (r, _g, _b, a) = unpack(ctx.bitmap.pixels[505 * 800 + 705]);
        assert_eq!(a, 255, "draw at (705,505) lands on the resized bitmap");
        assert_eq!(r, 255);
        // resize resets the canvas to transparent (HTML spec).
        let mut ctx2 = CanvasContext2D::new(50, 50);
        ctx2.bitmap.clear(Color::WHITE);
        ctx2.resize(60, 60);
        let (_r2, _g2, _b2, a2) = unpack(ctx2.bitmap.pixels[0]);
        assert_eq!(a2, 0, "resize resets the canvas to transparent");
    }

    // ---- Bug-fix regression tests ----

    /// Bug 1: fill() now uses nonzero winding rule (HTML Canvas spec default),
    /// not the even-odd rule. Two overlapping same-direction rectangles must
    /// fill SOLIDLY, not cancel each other out at the overlap.
    #[test]
    fn fill_nonzero_winding_overlapping_rects_fill_solid() {
        // Draw two overlapping rectangles using separate move_to/line_to
        // subpaths so the overlap region has winding count 2 under nonzero,
        // but would be EMPTY under even-odd (2 crossings → outside).
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color {
            r: 0,
            g: 0,
            b: 200,
            a: 255,
        });
        ctx.begin_path();
        // Outer rect: (2,2) → (14,14)
        ctx.move_to(2.0, 2.0);
        ctx.line_to(14.0, 2.0);
        ctx.line_to(14.0, 14.0);
        ctx.line_to(2.0, 14.0);
        ctx.close_path();
        // Inner rect: (6,6) → (10,10), SAME winding direction as outer.
        ctx.move_to(6.0, 6.0);
        ctx.line_to(10.0, 6.0);
        ctx.line_to(10.0, 10.0);
        ctx.line_to(6.0, 10.0);
        ctx.close_path();
        ctx.fill(FillRule::Nonzero);
        // The inner rect's interior (8,8) has winding=2 → inside with nonzero.
        // Even-odd would leave it unfilled (winding even → outside).
        assert_eq!(
            ctx.bitmap.pixels[8 * 20 + 8],
            Color {
                r: 0,
                g: 0,
                b: 200,
                a: 255
            }
            .to_bgra_u32(),
            "inner overlap must be filled (nonzero winding, not even-odd)"
        );
        // Outer region (4,4) must also be filled.
        assert_eq!(
            ctx.bitmap.pixels[4 * 20 + 4],
            Color {
                r: 0,
                g: 0,
                b: 200,
                a: 255
            }
            .to_bgra_u32(),
            "outer region must be filled"
        );
    }

    /// Bug 1b: a counter-clockwise inner path (donut shape) creates winding=0
    /// at the hole, so the hole IS transparent — proving winding direction is tracked.
    #[test]
    fn fill_nonzero_winding_cw_outer_ccw_inner_leaves_hole() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color {
            r: 200,
            g: 0,
            b: 0,
            a: 255,
        });
        ctx.begin_path();
        // CW outer rect: (2,2) → (14,14), drawn clockwise (+1 winding inside)
        ctx.move_to(2.0, 2.0);
        ctx.line_to(14.0, 2.0);
        ctx.line_to(14.0, 14.0);
        ctx.line_to(2.0, 14.0);
        ctx.close_path();
        // CCW inner rect: (6,6) → (10,10), drawn counter-clockwise (-1 winding)
        // Net winding inside inner = +1 + (-1) = 0 → hole (transparent).
        ctx.move_to(6.0, 6.0);
        ctx.line_to(6.0, 10.0);  // go down first (CCW)
        ctx.line_to(10.0, 10.0);
        ctx.line_to(10.0, 6.0);
        ctx.close_path();
        ctx.fill(FillRule::Nonzero);
        // Outer region (4,4): winding=1 → filled.
        assert_eq!(
            ctx.bitmap.pixels[4 * 20 + 4],
            Color {
                r: 200,
                g: 0,
                b: 0,
                a: 255
            }
            .to_bgra_u32(),
            "outer region must be filled (winding=1)"
        );
        // Inner hole (8,8): winding=0 → transparent.
        assert_eq!(
            ctx.bitmap.pixels[8 * 20 + 8],
            Color::TRANSPARENT.to_bgra_u32(),
            "counter-wound inner rect must be a hole (winding=0)"
        );
    }

    /// Bug 2: fill_text with textBaseline='alphabetic' (default) places the
    /// glyph baseline at y, not the top. With y=5 and a 5×7 font (ascent=5),
    /// the glyph top is at y=0 and glyph rows land in rows 0..6.
    #[test]
    fn fill_text_alphabetic_baseline_y_places_glyph_above() {
        // Alphabetic baseline at y=5: glyph top = 5-5 = 0.
        // Row 0 of 'H' is 0b10001 → pixels (0,0) and (4,0) should be black.
        let mut ctx = CanvasContext2D::new(40, 16);
        ctx.set_fill_color(Color::BLACK);
        // Default textBaseline is Alphabetic, font_size_px=10.0, scale=1.
        ctx.fill_text("H", 0.0, 5.0);
        assert_eq!(
            ctx.bitmap.pixels[0 * 40 + 0],
            Color::BLACK.to_bgra_u32(),
            "alphabetic baseline: top-left of 'H' at (0,0) when y=5"
        );
        assert_eq!(
            ctx.bitmap.pixels[0 * 40 + 4],
            Color::BLACK.to_bgra_u32(),
            "alphabetic baseline: top-right of 'H' at (4,0) when y=5"
        );
        // Pixel above row 0 (i.e. y=-1) is out of bounds — row 0 must be the topmost.
        // We verify by ensuring pixel at row 6 (the last glyph row) is also set.
        // Row 6 of 'H' is 0b10001 too.
        assert_eq!(
            ctx.bitmap.pixels[6 * 40 + 0],
            Color::BLACK.to_bgra_u32(),
            "last glyph row (row 6) is also painted"
        );
    }

    /// Bug 2b: fill_text with textBaseline='top' places the top of the glyph at y.
    #[test]
    fn fill_text_top_baseline_places_top_at_y() {
        let mut ctx = CanvasContext2D::new(40, 16);
        ctx.set_fill_color(Color::BLACK);
        ctx.set_text_baseline("top");
        // With 'top', y=0 means glyph top at row 0 — same as old behavior.
        ctx.fill_text("H", 0.0, 0.0);
        assert_eq!(
            ctx.bitmap.pixels[0 * 40 + 0],
            Color::BLACK.to_bgra_u32(),
            "top baseline: 'H' top-left at (0,0) when y=0"
        );
    }

    /// Bug 2c: measure_text scales with font size.
    #[test]
    fn measure_text_scales_with_font_size() {
        let mut ctx = CanvasContext2D::new(200, 20);
        // Default 10px → scale=1 → "HI" = 11px
        assert_eq!(ctx.measure_text("HI").width as u32, 11);
        // 14px → scale=round(14/7)=2 → 11*2=22
        ctx.set_font_size(14.0);
        assert_eq!(ctx.measure_text("HI").width as u32, 22);
        // 21px → scale=round(21/7)=3 → 11*3=33
        ctx.set_font_size(21.0);
        assert_eq!(ctx.measure_text("HI").width as u32, 33);
    }

    /// Bug 1c: fill('evenodd') uses the even-odd rule.
    /// Two same-direction overlapping rects: overlap area has winding=2
    /// (inside under nonzero) but even count → outside under even-odd.
    #[test]
    fn fill_evenodd_leaves_overlap_transparent() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color { r: 0, g: 0, b: 200, a: 255 });
        ctx.begin_path();
        // Outer CW rect: (2,2) → (14,14)
        ctx.move_to(2.0, 2.0);
        ctx.line_to(14.0, 2.0);
        ctx.line_to(14.0, 14.0);
        ctx.line_to(2.0, 14.0);
        ctx.close_path();
        // Inner CW rect: (6,6) → (10,10) — same direction → winding=2 in overlap.
        ctx.move_to(6.0, 6.0);
        ctx.line_to(10.0, 6.0);
        ctx.line_to(10.0, 10.0);
        ctx.line_to(6.0, 10.0);
        ctx.close_path();
        ctx.fill(FillRule::EvenOdd);
        // Outer-only region (4,4): crossing count=1 (odd) → filled.
        assert_eq!(
            ctx.bitmap.pixels[4 * 20 + 4],
            Color { r: 0, g: 0, b: 200, a: 255 }.to_bgra_u32(),
            "evenodd: outer region (winding=1) must be filled"
        );
        // Overlap (8,8): crossing count=2 (even) → transparent.
        assert_eq!(
            ctx.bitmap.pixels[8 * 20 + 8],
            Color::TRANSPARENT.to_bgra_u32(),
            "evenodd: inner overlap (winding=2) must be transparent"
        );
    }

    /// FillRule::from_str recognises "evenodd" and defaults to nonzero.
    #[test]
    fn fill_rule_from_str() {
        assert_eq!(FillRule::from_str("evenodd"), FillRule::EvenOdd);
        assert_eq!(FillRule::from_str("nonzero"), FillRule::Nonzero);
        assert_eq!(FillRule::from_str(""), FillRule::Nonzero);
        assert_eq!(FillRule::from_str("garbage"), FillRule::Nonzero);
    }

    /// Bug 2d: fill_text with textAlign='center' centers on x.
    /// With a 1-char text "H" at scale=1, glyph_advance=6.
    /// center x_offset = -3, so fill_text("H", 9, 5) starts at x=6.
    #[test]
    fn fill_text_center_align_centers_on_x() {
        let mut ctx = CanvasContext2D::new(20, 16);
        ctx.set_fill_color(Color::BLACK);
        ctx.set_text_align("center");
        // Default alphabetic baseline: ascent=5; glyph top at y-5=0 for y=5.
        // "H" glyph_advance=6; center offset=-3; so x_start=9-3=6.
        // Row 0 bit 0 of 'H' → pixel (6, 0).
        ctx.fill_text("H", 9.0, 5.0);
        assert_eq!(
            ctx.bitmap.pixels[0 * 20 + 6],
            Color::BLACK.to_bgra_u32(),
            "center align: left bit of 'H' at x=6 when center x=9"
        );
        // Pixel at x=0 must be transparent (not left-aligned).
        assert_eq!(
            ctx.bitmap.pixels[0 * 20 + 0],
            Color::TRANSPARENT.to_bgra_u32(),
            "center align: x=0 must be transparent (not left-aligned)"
        );
    }

    /// Bug 2e: fill_text with textAlign='right' ends at x.
    /// "H" at scale=1, glyph_advance=6; right offset=-6; x_start=10-6=4.
    #[test]
    fn fill_text_right_align_ends_at_x() {
        let mut ctx = CanvasContext2D::new(20, 16);
        ctx.set_fill_color(Color::BLACK);
        ctx.set_text_align("right");
        // Alphabetic baseline y=5 → glyph top at 0.
        // x=10, right-align offset=-6, so glyph starts at x=4.
        ctx.fill_text("H", 10.0, 5.0);
        assert_eq!(
            ctx.bitmap.pixels[0 * 20 + 4],
            Color::BLACK.to_bgra_u32(),
            "right align: left bit of 'H' at x=4 when right x=10"
        );
        // x=0 must be transparent.
        assert_eq!(
            ctx.bitmap.pixels[0 * 20 + 0],
            Color::TRANSPARENT.to_bgra_u32(),
            "right align: x=0 must be transparent"
        );
    }

    // ---- Bug-fix regression tests: gradients + clip ----

    /// Bug fix 1: linear gradient fill via set_fill_linear_gradient paints
    /// a horizontally-interpolated rect. Left column ≈ red, right column ≈ blue.
    #[test]
    fn linear_gradient_fill_rect_interpolates() {
        let mut ctx = CanvasContext2D::new(10, 4);
        // Horizontal gradient: red at x=0, blue at x=10.
        ctx.set_fill_linear_gradient(
            0.0, 0.0, 10.0, 0.0,
            vec![
                GradientStop { offset: 0.0, color: Color { r: 255, g: 0, b: 0, a: 255 } },
                GradientStop { offset: 1.0, color: Color { r: 0, g: 0, b: 255, a: 255 } },
            ],
        );
        ctx.fill_rect(0.0, 0.0, 10.0, 4.0);
        // Left edge must be mostly red.
        let (_, _, _, _) = unpack(ctx.bitmap.pixels[0]);
        let (r0, _g0, b0, _a0) = unpack(ctx.bitmap.pixels[0]);
        assert!(r0 > 200, "left edge is red-dominant, got r={r0}");
        assert!(b0 < 100, "left edge is not blue, got b={b0}");
        // Right edge must be mostly blue.
        let (r9, _g9, b9, _a9) = unpack(ctx.bitmap.pixels[9]);
        assert!(b9 > 200, "right edge is blue-dominant, got b={b9}");
        assert!(r9 < 100, "right edge is not red, got r={r9}");
    }

    /// Bug fix 2: gradient fill via the path fill() respects the gradient.
    #[test]
    fn linear_gradient_fill_path_interpolates() {
        let mut ctx = CanvasContext2D::new(10, 4);
        ctx.set_fill_linear_gradient(
            0.0, 0.0, 10.0, 0.0,
            vec![
                GradientStop { offset: 0.0, color: Color { r: 255, g: 0, b: 0, a: 255 } },
                GradientStop { offset: 1.0, color: Color { r: 0, g: 0, b: 255, a: 255 } },
            ],
        );
        ctx.begin_path();
        ctx.rect(0.0, 0.0, 10.0, 4.0);
        ctx.fill(FillRule::Nonzero);
        // Midpoint (x=5) should be ~ equal red and blue.
        let (r5, _g5, b5, _a5) = unpack(ctx.bitmap.pixels[5]);
        assert!(r5 > 100 && r5 < 200, "midpoint red ~128, got {r5}");
        assert!(b5 > 100 && b5 < 200, "midpoint blue ~128, got {b5}");
    }

    /// Bug fix 3: clip() restricts subsequent fill_rect to the clipped region.
    /// A full-canvas fill after clip should only paint inside the clip area.
    #[test]
    fn clip_restricts_fill_rect() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color { r: 255, g: 0, b: 0, a: 255 });
        // Draw a clip rect at (5,5)→(15,15).
        ctx.begin_path();
        ctx.rect(5.0, 5.0, 10.0, 10.0);
        ctx.clip();
        // Now fill the full canvas — should only paint inside the clip.
        ctx.fill_rect(0.0, 0.0, 20.0, 20.0);
        // Inside the clip: pixel (10,10) must be red.
        assert_eq!(
            ctx.bitmap.pixels[10 * 20 + 10],
            Color { r: 255, g: 0, b: 0, a: 255 }.to_bgra_u32(),
            "inside clip must be filled"
        );
        // Outside the clip: pixel (0,0) must be transparent.
        assert_eq!(
            ctx.bitmap.pixels[0],
            Color::TRANSPARENT.to_bgra_u32(),
            "outside clip must be transparent"
        );
        // Edge: pixel (4,10) (just outside x0=5 boundary) must be transparent.
        assert_eq!(
            ctx.bitmap.pixels[10 * 20 + 4],
            Color::TRANSPARENT.to_bgra_u32(),
            "pixel just outside clip x0 must be transparent"
        );
    }

    /// Bug fix 3b: clip() restricts the path fill() operation.
    #[test]
    fn clip_restricts_path_fill() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color { r: 0, g: 255, b: 0, a: 255 });
        // Clip to the top-left quadrant.
        ctx.begin_path();
        ctx.rect(0.0, 0.0, 10.0, 10.0);
        ctx.clip();
        // Fill a rect covering the bottom-right quadrant — should stay transparent.
        ctx.begin_path();
        ctx.rect(0.0, 0.0, 20.0, 20.0);
        ctx.fill(FillRule::Nonzero);
        // Inside clip: (5,5) must be green.
        assert_eq!(
            ctx.bitmap.pixels[5 * 20 + 5],
            Color { r: 0, g: 255, b: 0, a: 255 }.to_bgra_u32(),
            "inside clip must be green"
        );
        // Outside clip: (15,15) must be transparent.
        assert_eq!(
            ctx.bitmap.pixels[15 * 20 + 15],
            Color::TRANSPARENT.to_bgra_u32(),
            "outside clip must be transparent"
        );
    }

    /// Bug fix: save/restore properly saves and restores the clip state.
    #[test]
    fn save_restore_preserves_clip() {
        let mut ctx = CanvasContext2D::new(20, 20);
        ctx.set_fill_color(Color { r: 255, g: 0, b: 0, a: 255 });
        // Set a clip in the inner state.
        ctx.save();
        ctx.begin_path();
        ctx.rect(5.0, 5.0, 5.0, 5.0);
        ctx.clip();
        ctx.fill_rect(0.0, 0.0, 20.0, 20.0);
        // Inside the clip: (7,7) must be red.
        assert_eq!(ctx.bitmap.pixels[7 * 20 + 7],
            Color { r: 255, g: 0, b: 0, a: 255 }.to_bgra_u32());
        // Outside: (1,1) must be transparent.
        assert_eq!(ctx.bitmap.pixels[1 * 20 + 1], Color::TRANSPARENT.to_bgra_u32());
        // Restore: clip should be gone, and a new fill should paint everywhere.
        ctx.restore();
        ctx.set_fill_color(Color { r: 0, g: 0, b: 255, a: 255 });
        ctx.fill_rect(0.0, 0.0, 20.0, 20.0);
        assert_eq!(ctx.bitmap.pixels[0],
            Color { r: 0, g: 0, b: 255, a: 255 }.to_bgra_u32(),
            "after restore, fill should paint full canvas");
    }

    // ---- Bug-fix regression: rotated fill_rect/stroke_rect/clear_rect ----

    /// fill_rect after a 45-degree rotation must paint a diamond, not an
    /// axis-aligned square.  The painted region straddles the diagonal through
    /// the centre of the canvas — pixels ON the diagonal must be filled, and
    /// corner pixels of the equivalent un-rotated square must NOT be filled.
    ///
    /// Setup: 40x40 canvas, translate to centre (20,20), rotate 45°, then
    /// fill_rect(-10,-10,20,20).  Without the fix the AABB of just the two
    /// diagonal corners ((20,20) and (20+0, 20+0) after 45° of a 20×20 rect
    /// in a degenerate case) collapses — use a clearer non-square rect:
    ///
    /// translate(20,20), rotate(PI/4), fill_rect(0, -5, 14, 10).
    /// The four user-space corners are:
    ///   (0,-5) → pixel (20 + 0*cos - (-5)*sin, 20 + 0*sin + (-5)*cos)
    ///           = (20 + 5*0.707, 20 - 5*0.707) ≈ (23.5, 16.5)
    ///   (14,-5) → (20 + 14*0.707 + 5*0.707, 20 + 14*0.707 - 5*0.707)
    ///           ≈ (26.4, 26.4)   — wait, let me recompute with the actual matrix.
    ///
    /// Actually use a simpler invariant: the centre pixel of the rotated rect
    /// always lands at the translate origin, and a pixel far from that origin
    /// but inside the un-rotated AABB must NOT be painted.
    #[test]
    fn fill_rect_rotated_paints_correct_quad() {
        let mut ctx = CanvasContext2D::new(60, 60);
        ctx.set_fill_color(Color { r: 255, g: 0, b: 0, a: 255 });
        // Translate to centre, rotate 45°, fill a 20×20 square at (-10,-10).
        ctx.translate(30.0, 30.0);
        ctx.rotate(std::f32::consts::FRAC_PI_4);
        ctx.fill_rect(-10.0, -10.0, 20.0, 20.0);

        // The centre of the rotated rect is at the translate origin = pixel (30,30).
        assert_eq!(
            ctx.bitmap.pixels[30 * 60 + 30],
            Color { r: 255, g: 0, b: 0, a: 255 }.to_bgra_u32(),
            "centre pixel of rotated rect must be filled"
        );

        // A 20x20 rect centred at (30,30) has corners at (20,20),(40,20),(40,40),(20,40).
        // After 45° rotation they map to (30,16),(44,30),(30,44),(16,30) approximately.
        // Pixel (20,20) is a corner of the AABB but FAR outside the actual diamond —
        // with the buggy code it would be filled; with the correct code it must be clear.
        assert_eq!(
            ctx.bitmap.pixels[20 * 60 + 20],
            Color::TRANSPARENT.to_bgra_u32(),
            "corner of AABB outside the rotated diamond must NOT be filled (was wrong with AABB-only fix)"
        );
    }

    /// stroke_rect after a 45-degree rotation strokes a diamond, not an
    /// axis-aligned rectangle outline.
    #[test]
    fn stroke_rect_rotated_strokes_correct_quad() {
        let mut ctx = CanvasContext2D::new(60, 60);
        ctx.set_stroke_color(Color { r: 0, g: 0, b: 200, a: 255 });
        ctx.set_line_width(2.0);
        ctx.translate(30.0, 30.0);
        ctx.rotate(std::f32::consts::FRAC_PI_4);
        ctx.stroke_rect(-10.0, -10.0, 20.0, 20.0);

        // The rightmost tip of the diamond lands at ≈ (30 + 10*sqrt(2)/2*2, 30) = (44,30).
        // A pixel near (44, 30) must be painted.
        let tip_row = 30usize;
        let tip_col = 43usize; // one pixel inward from exact tip for safety
        let tip_pixel = ctx.bitmap.pixels[tip_row * 60 + tip_col];
        assert_eq!(
            tip_pixel,
            Color { r: 0, g: 0, b: 200, a: 255 }.to_bgra_u32(),
            "tip of rotated stroke diamond must be painted"
        );

        // Pixel at (20,20) is a corner of the AABB but outside the diamond strokes.
        assert_eq!(
            ctx.bitmap.pixels[20 * 60 + 20],
            Color::TRANSPARENT.to_bgra_u32(),
            "AABB corner outside the rotated diamond stroke must be transparent"
        );
    }

    /// clear_rect after a 45-degree rotation clears only the diamond-shaped
    /// region, not the full AABB.
    #[test]
    fn clear_rect_rotated_clears_correct_quad() {
        let mut ctx = CanvasContext2D::new(60, 60);
        // Fill everything white first.
        ctx.set_fill_color(Color::WHITE);
        ctx.fill_rect(0.0, 0.0, 60.0, 60.0);

        // Now clear a rotated square.
        ctx.translate(30.0, 30.0);
        ctx.rotate(std::f32::consts::FRAC_PI_4);
        ctx.clear_rect(-10.0, -10.0, 20.0, 20.0);

        // Centre should be transparent.
        assert_eq!(
            ctx.bitmap.pixels[30 * 60 + 30],
            Color::TRANSPARENT.to_bgra_u32(),
            "centre of rotated clear_rect must be transparent"
        );

        // Corner of AABB is outside the diamond — must remain white.
        assert_eq!(
            ctx.bitmap.pixels[20 * 60 + 20],
            Color::WHITE.to_bgra_u32(),
            "AABB corner outside the rotated clear diamond must remain white"
        );
    }

    /// fill_rect / stroke_rect / clear_rect must NOT disturb the current path
    /// when called with rotation active.
    #[test]
    fn rotated_fill_rect_does_not_disturb_current_path() {
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_fill_color(Color::BLACK);
        // Build a path (triangle), then interleave a rotated fill_rect.
        ctx.begin_path();
        ctx.move_to(5.0, 5.0);
        ctx.line_to(20.0, 5.0);
        ctx.line_to(12.0, 20.0);
        ctx.close_path();
        // Rotate and call fill_rect — the path above must survive.
        ctx.rotate(std::f32::consts::FRAC_PI_4);
        ctx.fill_rect(0.0, 0.0, 5.0, 5.0);
        // Reset transform so we can fill the triangle unrotated.
        ctx.set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        // The path (triangle) must still be intact — fill it.
        ctx.set_fill_color(Color { r: 0, g: 0, b: 200, a: 255 });
        ctx.fill(FillRule::Nonzero);
        // Pixel inside the triangle must be blue.
        assert_eq!(
            ctx.bitmap.pixels[8 * 40 + 12],
            Color { r: 0, g: 0, b: 200, a: 255 }.to_bgra_u32(),
            "path must survive fill_rect call"
        );
    }

    // ============================================================
    //  NEW Canvas2D op-group tests (arcTo / blend modes / point-in-
    //  path / patterns / Path2D / OffscreenCanvas-style offscreen).
    // ============================================================

    /// arcTo on a 90-degree corner produces an arc tangent to both edges.
    /// Path: moveTo(10,90) -> arcTo(10,10, 90,10, r=40) -> the first tangent
    /// point lands at (10, 50) on the vertical edge, the arc curves toward the
    /// horizontal edge with tangent point (50, 10).
    #[test]
    fn arc_to_produces_tangent_arc() {
        let seg = compute_arc_to(10.0, 90.0, 10.0, 10.0, 90.0, 10.0, 40.0)
            .expect("non-degenerate corner");
        // First tangent point on the P0->P1 (vertical x=10) line, 40px up from corner.
        assert!((seg.t1.0 - 10.0).abs() < 0.5, "t1.x ~10, got {}", seg.t1.0);
        assert!((seg.t1.1 - 50.0).abs() < 0.5, "t1.y ~50, got {}", seg.t1.1);
        // Center sits at (50, 50) — radius 40 from each tangent line.
        assert!((seg.center.0 - 50.0).abs() < 0.5, "cx ~50, got {}", seg.center.0);
        assert!((seg.center.1 - 50.0).abs() < 0.5, "cy ~50, got {}", seg.center.1);
        assert!((seg.radius - 40.0).abs() < 0.01);

        // Drawing it: the path passes through the tangent points.
        let mut ctx = CanvasContext2D::new(120, 120);
        ctx.begin_path();
        ctx.move_to(10.0, 90.0);
        ctx.arc_to(10.0, 10.0, 90.0, 10.0, 40.0);
        // First op after the moveTo is the lineTo the first tangent point.
        let mapped: Vec<(f32, f32)> = ctx
            .path
            .iter()
            .filter_map(|o| match o {
                PathOp::MoveTo(x, y) | PathOp::LineTo(x, y) => Some((*x, *y)),
                _ => None,
            })
            .collect();
        // The path should reach near the second tangent point (50, 10).
        let reaches = mapped
            .iter()
            .any(|&(x, y)| (x - 50.0).abs() < 2.0 && (y - 10.0).abs() < 2.0);
        assert!(reaches, "arc should reach second tangent point ~ (50,10): {mapped:?}");
    }

    /// arcTo degenerate cases (zero radius / collinear) fall back to lineTo.
    #[test]
    fn arc_to_degenerate_is_line() {
        // Zero radius.
        assert!(compute_arc_to(0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 0.0).is_none());
        // Collinear (all on the x axis).
        assert!(compute_arc_to(0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 5.0).is_none());

        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.begin_path();
        ctx.move_to(0.0, 0.0);
        ctx.arc_to(10.0, 0.0, 20.0, 0.0, 5.0); // collinear -> lineTo(10,0)
        let pts: Vec<(f32, f32)> = ctx
            .path
            .iter()
            .filter_map(|o| match o {
                PathOp::LineTo(x, y) => Some((*x, *y)),
                _ => None,
            })
            .collect();
        assert_eq!(pts, vec![(10.0, 0.0)], "collinear arcTo collapses to lineTo");
    }

    // ---- globalCompositeOperation blend modes ----

    fn comp(op: CompositeOp, dst: Color, src: Color) -> (u8, u8, u8, u8) {
        let out = composite_pixel(dst.to_bgra_u32(), src, op);
        unpack(out)
    }

    /// source-over is byte-identical to blend_bgra / overwrite for the three
    /// alpha cases — the load-bearing "don't regress the common path" gate.
    #[test]
    fn source_over_is_byte_identical_to_legacy() {
        let dst = Color { r: 10, g: 20, b: 30, a: 200 };
        for src in [
            Color { r: 255, g: 0, b: 0, a: 255 },   // opaque -> overwrite
            Color { r: 100, g: 150, b: 200, a: 128 }, // semi -> blend_bgra
            Color { r: 9, g: 9, b: 9, a: 0 },        // transparent -> unchanged
        ] {
            let via_comp = composite_pixel(dst.to_bgra_u32(), src, CompositeOp::SourceOver);
            let legacy = if src.a == 0 {
                dst.to_bgra_u32()
            } else if src.a == 255 {
                src.to_bgra_u32()
            } else {
                crate::blend_bgra(dst.to_bgra_u32(), src)
            };
            assert_eq!(via_comp, legacy, "source-over must equal legacy for src={src:?}");
        }
    }

    /// multiply darkens; screen lightens; their known formulas hold for opaque
    /// src over opaque dst (alpha math collapses, channel = B(Cb,Cs)*255).
    #[test]
    fn blend_modes_known_colors() {
        let dst = Color { r: 200, g: 100, b: 50, a: 255 };
        let src = Color { r: 100, g: 200, b: 50, a: 255 };

        // multiply: Cb*Cs. r = 200/255 * 100/255 ≈ 0.3075 -> 78.
        let (r, g, b, a) = comp(CompositeOp::Multiply, dst, src);
        assert_eq!(a, 255);
        assert!((r as i32 - 78).abs() <= 2, "multiply r ~78 got {r}");
        assert!((g as i32 - 78).abs() <= 2, "multiply g ~78 got {g}");
        // multiply with b=50/50 -> 50*50/255 ≈ 9.8 -> 10.
        assert!((b as i32 - 10).abs() <= 2, "multiply b ~10 got {b}");
        // multiply always darkens (result <= min of inputs per channel).
        assert!(r <= 200 && r <= 100);

        // screen: Cb+Cs-Cb*Cs. r = 0.784+0.392-0.307 = 0.869 -> ~221.
        let (sr, sg, _sb, _) = comp(CompositeOp::Screen, dst, src);
        assert!((sr as i32 - 221).abs() <= 3, "screen r ~221 got {sr}");
        // screen always lightens.
        assert!(sr >= 200 && sg >= 200);

        // darken / lighten pick per-channel min / max.
        let (dr, dg, _, _) = comp(CompositeOp::Darken, dst, src);
        assert_eq!((dr, dg), (100, 100), "darken picks min");
        let (lr, lg, _, _) = comp(CompositeOp::Lighten, dst, src);
        assert_eq!((lr, lg), (200, 200), "lighten picks max");

        // difference: |Cb - Cs|. r = |200-100| = 100.
        let (fr, _, _, _) = comp(CompositeOp::Difference, dst, src);
        assert!((fr as i32 - 100).abs() <= 2, "difference r ~100 got {fr}");
    }

    /// MUTATION GUARD: a deliberately wrong multiply (screen instead) would NOT
    /// darken — this test pins the direction so breaking the formula fails.
    #[test]
    fn multiply_strictly_darkens() {
        let dst = Color { r: 180, g: 180, b: 180, a: 255 };
        let src = Color { r: 180, g: 180, b: 180, a: 255 };
        // multiply 180*180/255 ≈ 127 -> strictly darker than 180.
        let (r, _, _, _) = comp(CompositeOp::Multiply, dst, src);
        assert!(r < 180, "multiply must darken (r={r} < 180)");
        assert!((r as i32 - 127).abs() <= 2, "multiply r ~127 got {r}");
        // screen on the same inputs must be LIGHTER — proves they differ.
        let (sr, _, _, _) = comp(CompositeOp::Screen, dst, src);
        assert!(sr > 180, "screen must lighten (sr={sr} > 180)");
    }

    /// Porter-Duff: destination-out punches a transparent hole; copy replaces;
    /// lighter adds.
    #[test]
    fn porter_duff_modes() {
        let dst = Color { r: 0, g: 0, b: 255, a: 255 };
        let src = Color { r: 255, g: 0, b: 0, a: 255 };
        // copy: result is exactly src.
        assert_eq!(comp(CompositeOp::Copy, dst, src), (255, 0, 0, 255));
        // destination-out: opaque src removes dst -> fully transparent.
        let (_, _, _, a) = comp(CompositeOp::DestinationOut, dst, src);
        assert_eq!(a, 0, "destination-out with opaque src clears dst");
        // lighter: r adds (255+0), b adds (0+255) -> magenta, clamped.
        let (lr, _lg, lb, _) = comp(CompositeOp::Lighter, dst, src);
        assert_eq!(lr, 255);
        assert_eq!(lb, 255, "lighter sums channels");
    }

    /// The full pipeline through fill_rect honours globalCompositeOperation.
    #[test]
    fn fill_rect_multiply_darkens_destination() {
        let mut ctx = CanvasContext2D::new(8, 8);
        ctx.bitmap.clear(Color { r: 200, g: 200, b: 200, a: 255 });
        ctx.set_composite_op(CompositeOp::Multiply);
        ctx.set_fill_color(Color { r: 100, g: 100, b: 100, a: 255 });
        ctx.fill_rect(0.0, 0.0, 8.0, 8.0);
        let (r, _, _, _) = unpack(ctx.bitmap.pixels[3 * 8 + 3]);
        // 200*100/255 ≈ 78.
        assert!((r as i32 - 78).abs() <= 3, "fill_rect multiply r ~78 got {r}");
    }

    /// fill_rect with source-over (default) stays byte-identical to the legacy
    /// solid fast path.
    #[test]
    fn fill_rect_source_over_unchanged() {
        let mut a = CanvasContext2D::new(8, 8);
        a.bitmap.clear(Color { r: 10, g: 20, b: 30, a: 255 });
        a.set_fill_color(Color { r: 200, g: 100, b: 50, a: 128 });
        a.fill_rect(0.0, 0.0, 8.0, 8.0);

        // Reference: clear + manual blend_bgra over each pixel.
        let mut ref_bmp = Bitmap::new(8, 8);
        ref_bmp.clear(Color { r: 10, g: 20, b: 30, a: 255 });
        ref_bmp.fill_rect(0, 0, 8, 8, Color { r: 200, g: 100, b: 50, a: 128 });
        assert_eq!(a.bitmap.pixels, ref_bmp.pixels, "default GCO must be byte-identical");
    }

    // ---- isPointInPath / isPointInStroke ----

    #[test]
    fn is_point_in_path_rect() {
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.begin_path();
        ctx.rect(10.0, 10.0, 20.0, 20.0);
        assert!(ctx.is_point_in_path(20.0, 20.0, FillRule::Nonzero), "center inside");
        assert!(!ctx.is_point_in_path(5.0, 5.0, FillRule::Nonzero), "corner outside");
        assert!(!ctx.is_point_in_path(35.0, 20.0, FillRule::Nonzero), "right of rect outside");
    }

    #[test]
    fn is_point_in_path_triangle_and_evenodd() {
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.begin_path();
        ctx.move_to(5.0, 5.0);
        ctx.line_to(35.0, 5.0);
        ctx.line_to(20.0, 35.0);
        ctx.close_path();
        assert!(ctx.is_point_in_path(20.0, 12.0, FillRule::Nonzero), "inside triangle");
        assert!(!ctx.is_point_in_path(2.0, 2.0, FillRule::Nonzero), "outside triangle");

        // Even-odd donut: outer rect with a counter-wound inner rect leaves a hole.
        let mut d = CanvasContext2D::new(40, 40);
        d.begin_path();
        d.rect(2.0, 2.0, 30.0, 30.0);
        d.rect(10.0, 10.0, 10.0, 10.0);
        // Even-odd: inside the inner rect is OUTSIDE (2 crossings).
        assert!(!d.is_point_in_path(15.0, 15.0, FillRule::EvenOdd), "even-odd hole");
        // But the band between is inside.
        assert!(d.is_point_in_path(5.0, 15.0, FillRule::EvenOdd), "even-odd band inside");
    }

    #[test]
    fn is_point_in_stroke_band() {
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_line_width(4.0);
        ctx.begin_path();
        ctx.move_to(5.0, 20.0);
        ctx.line_to(35.0, 20.0);
        assert!(ctx.is_point_in_stroke(20.0, 20.0), "on the line");
        assert!(ctx.is_point_in_stroke(20.0, 21.0), "within half-width band");
        assert!(!ctx.is_point_in_stroke(20.0, 30.0), "far from the line");
    }

    // ---- createPattern (tiling) ----

    #[test]
    fn pattern_repeat_tiles_fill() {
        // 2x2 checker source: (0,0)=red opaque, others transparent.
        let red = Color { r: 255, g: 0, b: 0, a: 255 }.to_bgra_u32();
        let clear = 0u32;
        let src = std::rc::Rc::new(vec![red, clear, clear, red]); // 2x2
        let mut ctx = CanvasContext2D::new(8, 8);
        ctx.set_fill_pattern(2, 2, src, PatternRepeat::Repeat);
        ctx.fill_rect(0.0, 0.0, 8.0, 8.0);
        // (0,0) tiles to red; (1,0) clear; (4,0) (=0,0 of next tile) red again.
        assert_eq!(unpack(ctx.bitmap.pixels[0]).0, 255, "tile origin is red");
        assert_eq!(unpack(ctx.bitmap.pixels[4]).3, 255, "tile repeats at x=4 (red)");
        assert_eq!(unpack(ctx.bitmap.pixels[1]).3, 0, "checker hole stays transparent");
    }

    #[test]
    fn pattern_no_repeat_leaves_outside_untouched() {
        let red = Color { r: 255, g: 0, b: 0, a: 255 }.to_bgra_u32();
        let src = std::rc::Rc::new(vec![red, red, red, red]); // 2x2 solid red
        let mut ctx = CanvasContext2D::new(8, 8);
        ctx.bitmap.clear(Color::WHITE);
        ctx.set_fill_pattern(2, 2, src, PatternRepeat::NoRepeat);
        ctx.fill_rect(0.0, 0.0, 8.0, 8.0);
        // Inside the 2x2 tile -> red.
        assert_eq!(unpack(ctx.bitmap.pixels[0]).0, 255, "tile is red");
        // Outside the single tile (x=5) -> untouched white.
        assert_eq!(ctx.bitmap.pixels[5], Color::WHITE.to_bgra_u32(), "no-repeat leaves outside");
    }

    // ---- Path2D ----

    #[test]
    fn path2d_fill_renders_recorded_shape() {
        let mut p = Path2D::new();
        p.move_to(5.0, 5.0);
        p.line_to(35.0, 5.0);
        p.line_to(20.0, 35.0);
        p.close_path();
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_fill_color(Color { r: 0, g: 0, b: 200, a: 255 });
        ctx.fill_path(&p, FillRule::Nonzero);
        // Inside the triangle.
        assert_eq!(
            ctx.bitmap.pixels[12 * 40 + 20],
            Color { r: 0, g: 0, b: 200, a: 255 }.to_bgra_u32(),
            "Path2D triangle fills interior"
        );
        assert_eq!(ctx.bitmap.pixels[0], Color::TRANSPARENT.to_bgra_u32(), "corner clear");
        // is_point_in_path2d agrees.
        assert!(ctx.is_point_in_path2d(&p, 20.0, 12.0, FillRule::Nonzero));
        assert!(!ctx.is_point_in_path2d(&p, 2.0, 2.0, FillRule::Nonzero));
    }

    #[test]
    fn path2d_svg_string_constructor() {
        // A 20x20 square via SVG path data with H/V shorthands and Z.
        let p = Path2D::from_svg("M5 5 H25 V25 H5 Z");
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_fill_color(Color { r: 0, g: 180, b: 0, a: 255 });
        ctx.fill_path(&p, FillRule::Nonzero);
        assert_eq!(
            ctx.bitmap.pixels[15 * 40 + 15],
            Color { r: 0, g: 180, b: 0, a: 255 }.to_bgra_u32(),
            "SVG square fills interior"
        );
        assert_eq!(ctx.bitmap.pixels[0], Color::TRANSPARENT.to_bgra_u32());
    }

    #[test]
    fn path2d_add_path_and_transform_respected() {
        let mut base = Path2D::new();
        base.rect(0.0, 0.0, 10.0, 10.0);
        let mut combined = Path2D::new();
        combined.add_path(&base, Some([1.0, 0.0, 0.0, 1.0, 20.0, 20.0])); // translate (20,20)
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_fill_color(Color { r: 200, g: 0, b: 0, a: 255 });
        ctx.fill_path(&combined, FillRule::Nonzero);
        // The translated rect covers (20..30, 20..30).
        assert_eq!(
            ctx.bitmap.pixels[25 * 40 + 25],
            Color { r: 200, g: 0, b: 0, a: 255 }.to_bgra_u32(),
            "addPath translate places rect at (20,20)"
        );
        // Original location is empty.
        assert_eq!(ctx.bitmap.pixels[5 * 40 + 5], Color::TRANSPARENT.to_bgra_u32());
    }

    #[test]
    fn path2d_stroke_and_clip() {
        let mut p = Path2D::new();
        p.rect(5.0, 5.0, 20.0, 20.0);
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_stroke_color(Color::BLACK);
        ctx.set_line_width(1.0);
        ctx.stroke_path(&p);
        // Top edge of the rect should be painted.
        assert_eq!(ctx.bitmap.pixels[5 * 40 + 15], Color::BLACK.to_bgra_u32(), "stroke_path edge");

        // clip_path restricts a subsequent fill.
        let mut c2 = CanvasContext2D::new(40, 40);
        let mut clip = Path2D::new();
        clip.rect(0.0, 0.0, 10.0, 10.0);
        c2.set_fill_color(Color { r: 0, g: 0, b: 200, a: 255 });
        c2.clip_path(&clip);
        c2.fill_rect(0.0, 0.0, 40.0, 40.0);
        assert_eq!(
            c2.bitmap.pixels[5 * 40 + 5],
            Color { r: 0, g: 0, b: 200, a: 255 }.to_bgra_u32(),
            "inside clip filled"
        );
        assert_eq!(c2.bitmap.pixels[20 * 40 + 20], Color::TRANSPARENT.to_bgra_u32(), "outside clip clear");
    }

    /// An offscreen CanvasContext2D (the OffscreenCanvas backing) writes real
    /// pixels that read back through the same getImageData-style access.
    #[test]
    fn offscreen_context_writes_real_pixels() {
        // OffscreenCanvas.getContext("2d") is just a CanvasContext2D over a
        // freshly allocated bitmap — verify the same draw->readback contract.
        let mut off = CanvasContext2D::new(16, 16);
        off.set_fill_color(Color { r: 255, g: 204, b: 0, a: 255 });
        off.fill_rect(0.0, 0.0, 4.0, 4.0);
        // Read back a pixel inside the rect.
        let inside = off.bitmap.pixels[1 * 16 + 1];
        assert_eq!(inside, Color { r: 255, g: 204, b: 0, a: 255 }.to_bgra_u32());
        // And one outside.
        assert_eq!(off.bitmap.pixels[10 * 16 + 10], Color::TRANSPARENT.to_bgra_u32());
        // snapshot_pixels clones the buffer for transferToImageBitmap.
        let snap = off.snapshot_pixels();
        assert_eq!(snap.len(), 16 * 16);
        assert_eq!(snap[1 * 16 + 1], inside);
    }

    // ---- Radial gradient: full two-circle (cone) sampling ----

    /// Concentric radial gradient (r0=0 → r1=R) must sample a MID color partway
    /// out, not flatten to first/last only. red center → blue edge: a pixel at
    /// half the radius should be ~half-and-half (the real interpolation midpoint).
    #[test]
    fn radial_gradient_samples_mid_color() {
        let mut ctx = CanvasContext2D::new(40, 40);
        // Center (20,20), inner r0=0, outer r1=20; red→blue.
        ctx.set_fill_radial_gradient(
            20.0, 20.0, 0.0, 20.0, 20.0, 20.0,
            vec![
                GradientStop { offset: 0.0, color: Color { r: 255, g: 0, b: 0, a: 255 } },
                GradientStop { offset: 1.0, color: Color { r: 0, g: 0, b: 255, a: 255 } },
            ],
        );
        ctx.fill_rect(0.0, 0.0, 40.0, 40.0);
        // Center is red.
        let (rc, _gc, bc, _ac) = unpack(ctx.bitmap.pixels[20 * 40 + 20]);
        assert!(rc > 200 && bc < 60, "center should be red, got r={rc} b={bc}");
        // 10px right of center = halfway to the r=20 edge → ~50/50 red/blue.
        let (rm, _gm, bm, am) = unpack(ctx.bitmap.pixels[20 * 40 + 30]);
        assert!(am == 255, "midway pixel must be opaque, got a={am}");
        assert!(rm > 90 && rm < 170, "midway red ~128, got {rm}");
        assert!(bm > 90 && bm < 170, "midway blue ~128, got {bm}");
        // Edge (at radius 20) is blue.
        let (re, _ge, be, _ae) = unpack(ctx.bitmap.pixels[20 * 40 + 39]);
        assert!(be > 180 && re < 80, "edge should be blue, got r={re} b={be}");
    }

    /// Multi-stop radial gradient honors INTERIOR stops (not just first+last).
    /// red@0 → green@0.5 → blue@1: the pixel at half radius must be ~green.
    #[test]
    fn radial_gradient_honors_interior_stop() {
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_fill_radial_gradient(
            20.0, 20.0, 0.0, 20.0, 20.0, 20.0,
            vec![
                GradientStop { offset: 0.0, color: Color { r: 255, g: 0, b: 0, a: 255 } },
                GradientStop { offset: 0.5, color: Color { r: 0, g: 255, b: 0, a: 255 } },
                GradientStop { offset: 1.0, color: Color { r: 0, g: 0, b: 255, a: 255 } },
            ],
        );
        ctx.fill_rect(0.0, 0.0, 40.0, 40.0);
        // 10px from center (offset≈0.5) must be green-dominant.
        let (r, g, b, _a) = unpack(ctx.bitmap.pixels[20 * 40 + 30]);
        assert!(g > 180, "half-radius must be green-dominant, got g={g}");
        assert!(r < 80 && b < 80, "half-radius is not red/blue, got r={r} b={b}");
    }

    /// Pixels OUTSIDE the painted cone are transparent. With a NON-zero inner
    /// radius (r0>0) and r0==r1 (a tube), the spec only paints the annulus; far
    /// from both circle centers no valid ω exists → transparent.
    #[test]
    fn radial_gradient_outside_cone_is_transparent() {
        // Start circle (10,20,2), end circle (30,20,2): two small equal circles.
        // a = |d|^2 - dr^2 = 400 - 0 = 400 (well-posed). A point far above the
        // axis lies on no interpolated circle of radius 2 → not painted.
        let mut ctx = CanvasContext2D::new(40, 40);
        ctx.set_fill_radial_gradient(
            10.0, 20.0, 2.0, 30.0, 20.0, 2.0,
            vec![
                GradientStop { offset: 0.0, color: Color { r: 255, g: 0, b: 0, a: 255 } },
                GradientStop { offset: 1.0, color: Color { r: 0, g: 0, b: 255, a: 255 } },
            ],
        );
        ctx.fill_rect(0.0, 0.0, 40.0, 40.0);
        // On the axis between the circles is painted.
        let (_r, _g, _b, a_on) = unpack(ctx.bitmap.pixels[20 * 40 + 20]);
        assert!(a_on > 0, "on-axis between circles must be painted, got a={a_on}");
        // Far off-axis corner: no circle of radius 2 reaches it → transparent.
        let (_r2, _g2, _b2, a_off) = unpack(ctx.bitmap.pixels[2 * 40 + 2]);
        assert_eq!(a_off, 0, "far off-cone pixel must be transparent, got a={a_off}");
    }

    /// The raw cone solver returns the LARGEST ω with non-negative radius and
    /// interpolates focal (non-concentric) gradients. A point on the end circle
    /// (x1,y1,r1) maps to ω=1; on the start circle (x0,y0,r0) to ω=0.
    #[test]
    fn radial_offset_solver_endpoints() {
        // start (0,0,0), end (10,0,10): classic "spotlight" cone.
        // The end-circle rim point (20,0) lies on radius-10 circle at center
        // (10,0) → ω=1. The center (0,0) is the degenerate r=0 start → ω=0.
        let at = |px: f32, py: f32| super::radial_offset(0.0, 0.0, 0.0, 10.0, 0.0, 10.0, px, py);
        let w_center = at(0.0, 0.0).unwrap();
        assert!((w_center - 0.0).abs() < 1e-3, "start point ω≈0, got {w_center}");
        let w_rim = at(20.0, 0.0).unwrap();
        assert!((w_rim - 1.0).abs() < 1e-3, "end-circle rim ω≈1, got {w_rim}");
        // The TOP of the ω=0.5 circle (center (5,0), radius 5) is (5,5); it lies
        // on no larger-ω circle, so the solver must report ω=0.5 there. (The
        // on-axis point (5,0) is ω=0.25 — circle center (2.5,0), radius 2.5 —
        // which is the genuine cone geometry, not a bug.)
        let w_half = at(5.0, 5.0).unwrap();
        assert!((w_half - 0.5).abs() < 1e-3, "top-of-mid-circle ω≈0.5, got {w_half}");
        let w_quarter = at(5.0, 0.0).unwrap();
        assert!((w_quarter - 0.25).abs() < 1e-3, "on-axis (5,0) ω≈0.25, got {w_quarter}");
    }

    // ---- shadowBlur: real Gaussian blur of the silhouette ----

    /// The box-blur Gaussian approximation CONSERVES total mass (a box average
    /// neither creates nor destroys coverage, modulo the clamped border) and
    /// SPREADS a single point into a smooth, monotonically decaying kernel.
    #[test]
    fn box_blur_gaussian_conserves_mass_and_spreads() {
        let w = 41usize;
        let h = 41usize;
        let mut buf = vec![0.0f32; w * h];
        buf[20 * w + 20] = 1.0; // unit impulse at center
        let before: f32 = buf.iter().sum();
        super::box_blur_gaussian(&mut buf, w, h, 3.0);
        let after: f32 = buf.iter().sum();
        // Mass conserved (buffer is large enough that the 3σ tail stays inside).
        assert!((after - before).abs() < 0.02, "mass conserved: {before} -> {after}");
        // The impulse spread: the exact center decreased, neighbors increased.
        assert!(buf[20 * w + 20] < 1.0, "center spread out, got {}", buf[20 * w + 20]);
        assert!(buf[20 * w + 21] > 0.0, "neighbor lit, got {}", buf[20 * w + 21]);
        // Monotonic decay away from the center along a row.
        let c = buf[20 * w + 20];
        let n1 = buf[20 * w + 22];
        let n2 = buf[20 * w + 26];
        assert!(c > n1 && n1 > n2, "kernel decays: {c} > {n1} > {n2}");
    }

    /// A solid disc with shadowBlur paints a SMOOTH soft shadow OUTSIDE the disc
    /// (Gaussian falloff) and leaves pixels well beyond ~3σ untouched. The
    /// falloff must be gradual (monotonically decreasing), not a hard cutoff.
    #[test]
    fn shadow_blur_paints_gaussian_falloff() {
        let mut ctx = CanvasContext2D::new(120, 120);
        let gold = Color { r: 255, g: 215, b: 0, a: 255 };
        ctx.set_fill_color(gold);
        ctx.set_shadow_color(gold);
        ctx.set_shadow_blur(20.0); // sigma = 10
        ctx.begin_path();
        ctx.arc(60.0, 60.0, 12.0, 0.0, std::f32::consts::PI * 2.0, false);
        ctx.fill(FillRule::Nonzero);
        let alpha = |x: usize, y: usize| ((ctx.bitmap.pixels[y * 120 + x] >> 24) & 0xFF) as u32;
        // Disc interior fully filled.
        assert!(alpha(60, 60) > 200, "disc centre filled, got {}", alpha(60, 60));
        // The glow decays monotonically moving outward from the disc edge
        // (edge ~12px above center is y=48; sample progressively farther up).
        let a_near = alpha(60, 44); // ~4px past edge
        let a_mid = alpha(60, 36);  // ~12px past edge
        let a_far = alpha(60, 30);  // ~18px past edge (~near 3σ tail)
        assert!(a_near > 0, "near glow lit, got {a_near}");
        assert!(a_near > a_mid, "glow decays outward: near {a_near} > mid {a_mid}");
        assert!(a_mid >= a_far, "glow keeps decaying: mid {a_mid} >= far {a_far}");
        // Far corner (well beyond shape + 3σ) stays transparent.
        assert_eq!(alpha(118, 118), 0, "corner beyond shape+blur stays transparent");
    }

    /// shadowOffsetX/Y displace the shadow: with a positive offset the glow
    /// appears on the offset side and is ABSENT on the opposite side.
    #[test]
    fn shadow_offset_displaces_glow() {
        let mut ctx = CanvasContext2D::new(120, 120);
        let red = Color { r: 255, g: 0, b: 0, a: 255 };
        ctx.set_fill_color(Color { r: 0, g: 0, b: 255, a: 255 });
        ctx.set_shadow_color(red);
        ctx.set_shadow_blur(6.0);
        ctx.set_shadow_offset(20.0, 0.0); // shift shadow +20px in X
        ctx.fill_rect(40.0, 50.0, 20.0, 20.0); // rect [40..60]x[50..70]
        // The (blue) rect itself is at x=40..60. The red shadow is shifted to
        // x≈60..80. A pixel at x=70,y=60 (right of the rect) should be reddish.
        let (r_right, _g, b_right, a_right) = unpack(ctx.bitmap.pixels[60 * 120 + 70]);
        assert!(a_right > 0, "shadow present on offset (right) side, a={a_right}");
        assert!(r_right > b_right, "offset side is shadow-red, got r={r_right} b={b_right}");
        // The opposite (left) side, x≈25, has neither rect nor shadow → clear.
        let a_left = (ctx.bitmap.pixels[60 * 120 + 25] >> 24) & 0xFF;
        assert_eq!(a_left, 0, "no shadow on the non-offset (left) side, a={a_left}");
    }

    /// A transparent shadowColor disables the shadow entirely (spec: shadows are
    /// only drawn when shadowColor is non-transparent).
    #[test]
    fn transparent_shadow_color_disables_shadow() {
        let mut ctx = CanvasContext2D::new(80, 80);
        ctx.set_fill_color(Color { r: 0, g: 0, b: 255, a: 255 });
        ctx.set_shadow_color(Color::TRANSPARENT);
        ctx.set_shadow_blur(20.0);
        ctx.set_shadow_offset(10.0, 10.0);
        ctx.fill_rect(20.0, 20.0, 20.0, 20.0);
        // Outside the rect (where a shadow would have fallen) must be clear.
        let a_outside = (ctx.bitmap.pixels[55 * 80 + 55] >> 24) & 0xFF;
        assert_eq!(a_outside, 0, "transparent shadowColor → no shadow, a={a_outside}");
    }
}
