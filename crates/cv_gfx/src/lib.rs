//! `cv_gfx` — CPU 2D graphics primitives.
//!
//! Software-only `Bitmap` in DIB-friendly BGRA. Primitives: `fill_rect`,
//! single-pixel set, blit-to-other, plus a `CanvasContext2D` shaped to
//! the Web Canvas API (state stack, transforms, paths, fill, stroke).
//! GPU rasterization lands later.

#![allow(dead_code, missing_debug_implementations, unreachable_pub)]

mod canvas;
pub mod canvas_extra;
mod font5x7;
pub mod icc;
pub mod webgl;
pub mod webgl_compile;
pub mod webgpu;

pub use canvas::{CanvasContext2D, ClipRect, CompositeOp, FillRule, FillStyle, GradientStop, Path2D, Pattern, PatternRepeat, PathOp, TextAlign, TextBaseline, TextMetrics};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };

    /// Pack as BGRA in the Windows DIB layout (low byte = blue).
    pub fn to_bgra_u32(self) -> u32 {
        (u32::from(self.a) << 24)
            | (u32::from(self.r) << 16)
            | (u32::from(self.g) << 8)
            | u32::from(self.b)
    }

    /// Unpack a packed BGRA u32 (Windows DIB layout) into a `Color`.
    #[inline]
    pub fn from_bgra_u32(p: u32) -> Self {
        Self {
            b: (p & 0xFF) as u8,
            g: ((p >> 8) & 0xFF) as u8,
            r: ((p >> 16) & 0xFF) as u8,
            a: ((p >> 24) & 0xFF) as u8,
        }
    }
}

/// A CSS `mix-blend-mode` / `background-blend-mode` value (CSS Compositing &
/// Blending Level 1 §5/§6). `Normal` is plain source-over; every other mode
/// maps to the W3C separable / non-separable blend formula implemented in
/// [`canvas::composite_pixel`] via [`CompositeOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
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
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl BlendMode {
    /// Parse a CSS blend-mode keyword. Unknown / `normal` → `Normal`.
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
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
            "hue" => Self::Hue,
            "saturation" => Self::Saturation,
            "color" => Self::Color,
            "luminosity" => Self::Luminosity,
            _ => Self::Normal, // "normal" and any unknown
        }
    }

    pub fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }

    /// Map to the canvas [`CompositeOp`] that carries the matching blend
    /// formula. `Normal` maps to `SourceOver`.
    fn to_composite_op(self) -> CompositeOp {
        match self {
            Self::Normal => CompositeOp::SourceOver,
            Self::Multiply => CompositeOp::Multiply,
            Self::Screen => CompositeOp::Screen,
            Self::Overlay => CompositeOp::Overlay,
            Self::Darken => CompositeOp::Darken,
            Self::Lighten => CompositeOp::Lighten,
            Self::ColorDodge => CompositeOp::ColorDodge,
            Self::ColorBurn => CompositeOp::ColorBurn,
            Self::HardLight => CompositeOp::HardLight,
            Self::SoftLight => CompositeOp::SoftLight,
            Self::Difference => CompositeOp::Difference,
            Self::Exclusion => CompositeOp::Exclusion,
            Self::Hue => CompositeOp::Hue,
            Self::Saturation => CompositeOp::Saturation,
            Self::Color => CompositeOp::Color,
            Self::Luminosity => CompositeOp::Luminosity,
        }
    }
}

/// Composite a straight-alpha `src` over the packed BGRA backdrop pixel `dst`
/// with a CSS [`BlendMode`]. This is the single entry point the page painter
/// uses for `mix-blend-mode` / `background-blend-mode`. `Normal` is byte-
/// identical to the legacy source-over (`blend_bgra`).
#[inline]
pub fn blend_mode_pixel(dst: u32, src: Color, mode: BlendMode) -> u32 {
    canvas::composite_pixel(dst, src, mode.to_composite_op())
}

/// An abstract CSS gradient color stop, before position fix-up.
/// `pos_frac` is a fraction of the gradient extent (0..1) when the
/// author used a percentage / angle; `pos_px` is an absolute length that
/// resolves against the gradient extent (line length / radius) at paint
/// time. At most one is `Some`; both `None` means "unspecified" and the
/// position is distributed per CSS Images 3 §3.4.3.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AbstractStop {
    pub color: Color,
    pub pos_frac: Option<f32>,
    pub pos_px: Option<f32>,
}

/// CSS `radial-gradient` ending shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GfxRadialShape {
    Circle,
    Ellipse,
}

/// CSS `radial-gradient` ending-shape size. Lengths are pre-resolved to
/// px by the caller (percentages resolved against the box).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GfxRadialSize {
    ClosestSide,
    FarthestSide,
    ClosestCorner,
    FarthestCorner,
    /// Explicit radii in px (rx for circle == ry).
    ExplicitPx { rx: f32, ry: f32 },
}

/// Resolve abstract stops into concrete `GradientStop`s with offsets in
/// [0,1], applying the CSS Images 3 §3.4.3 color-stop "fix-up":
///   1. first unspecified → 0.0, last unspecified → 1.0
///   2. clamp any position below the running max to that running max
///      (non-monotonic positions become flat)
///   3. evenly distribute runs of still-unspecified positions between
///      their bracketing specified positions.
/// `extent_px` is the gradient line length (linear) or radius (radial)
/// or one turn in px-equivalent (conic uses 1.0 so pos_px is unused).
pub(crate) fn resolve_gradient_stops(stops: &[AbstractStop], extent_px: f32) -> Vec<GradientStop> {
    if stops.is_empty() {
        return Vec::new();
    }
    let n = stops.len();
    let ext = extent_px.max(1e-3);
    // Step A: convert each authored position to a fraction (or None).
    let mut pos: Vec<Option<f32>> = stops
        .iter()
        .map(|s| {
            if let Some(f) = s.pos_frac {
                Some(f)
            } else {
                s.pos_px.map(|px| px / ext)
            }
        })
        .collect();
    // Step 1: anchor first/last.
    if pos[0].is_none() {
        pos[0] = Some(0.0);
    }
    if pos[n - 1].is_none() {
        pos[n - 1] = Some(1.0);
    }
    // Step 2: enforce monotonicity (clamp to running max).
    let mut running = f32::NEG_INFINITY;
    for p in pos.iter_mut() {
        if let Some(v) = p {
            if *v < running {
                *v = running;
            }
            running = *v;
        }
    }
    // Step 3: evenly distribute unspecified runs.
    let mut i = 0;
    while i < n {
        if pos[i].is_some() {
            i += 1;
            continue;
        }
        // [i .. j) is a run of None, bracketed by pos[i-1] and pos[j].
        let start_val = pos[i - 1].unwrap();
        let mut j = i;
        while j < n && pos[j].is_none() {
            j += 1;
        }
        let end_val = pos[j].unwrap(); // j < n guaranteed (last anchored)
        let count = (j - i + 1) as f32; // number of gaps
        for (k, slot) in pos[i..j].iter_mut().enumerate() {
            let frac = (k as f32 + 1.0) / count;
            *slot = Some(start_val + (end_val - start_val) * frac);
        }
        i = j;
    }
    stops
        .iter()
        .zip(pos.iter())
        .map(|(s, p)| GradientStop {
            offset: p.unwrap().clamp(0.0, 1.0),
            color: s.color,
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Bitmap {
    pub width: u32,
    pub height: u32,
    /// One u32 per pixel; row-major; top-to-bottom; BGRA layout.
    pub pixels: Vec<u32>,
}

impl Bitmap {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0xFFFFFFFF; (width as usize) * (height as usize)],
        }
    }

    pub fn clear(&mut self, color: Color) {
        let v = color.to_bgra_u32();
        for px in &mut self.pixels {
            *px = v;
        }
    }

    pub fn put_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let i = (y as usize) * (self.width as usize) + (x as usize);
        self.pixels[i] = color.to_bgra_u32();
    }

    /// Alpha-composite `color` over the existing pixel (source-over), instead of
    /// overwriting. Used by stroked lines so a semi-transparent stroke (e.g. a
    /// 0.1-alpha particles.js connector) tints and accumulates instead of
    /// obliterating what's underneath. Opaque colors still hard-write.
    pub fn blend_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 || color.a == 0 {
            return;
        }
        let i = (y as usize) * (self.width as usize) + (x as usize);
        if color.a == 255 {
            self.pixels[i] = color.to_bgra_u32();
        } else {
            self.pixels[i] = blend_bgra(self.pixels[i], color);
        }
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Color) {
        let (x0, y0, x1, y1) = (
            x.max(0),
            y.max(0),
            (x + w).min(self.width as i32),
            (y + h).min(self.height as i32),
        );
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        if color.a == 255 {
            let v = color.to_bgra_u32();
            for yy in y0..y1 {
                let row = (yy as usize) * (self.width as usize);
                for xx in x0..x1 {
                    self.pixels[row + xx as usize] = v;
                }
            }
        } else if color.a > 0 {
            for yy in y0..y1 {
                let row = (yy as usize) * (self.width as usize);
                for xx in x0..x1 {
                    let idx = row + xx as usize;
                    self.pixels[idx] = blend_bgra(self.pixels[idx], color);
                }
            }
        }
    }

    /// Fill a rect with a 2-color linear gradient interpolated along
    /// `angle_deg` (CSS convention — 0 = up, 90 = right, 180 = down,
    /// 270 = left). Each pixel projects onto the gradient axis to
    /// derive a t in [0, 1]; the result is a linear interpolation
    /// between `from` and `to`. Not gamma-corrected (lerp in sRGB) but
    /// good enough for the hero / button gradients real sites use.
    pub fn fill_rect_gradient(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        angle_deg: f32,
        from: Color,
        to: Color,
    ) {
        let (x0, y0, x1, y1) = (
            x.max(0),
            y.max(0),
            (x + w).min(self.width as i32),
            (y + h).min(self.height as i32),
        );
        if x1 <= x0 || y1 <= y0 || w <= 0 || h <= 0 {
            return;
        }
        // Convert CSS angle (0 = up) to a vector pointing in the
        // gradient's growth direction. CSS positive angle = clockwise.
        let theta = (angle_deg - 90.0).to_radians();
        let dx = theta.cos();
        let dy = theta.sin();
        // Project box corners onto axis to find the t=0 / t=1 endpoints
        // — the gradient should span the entire box.
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
        for yy in y0..y1 {
            let py = (yy - y) as f32 + 0.5;
            let row = (yy as usize) * (self.width as usize);
            for xx in x0..x1 {
                let px = (xx - x) as f32 + 0.5;
                let t = ((px * dx + py * dy) - t_min) / denom;
                let t = t.clamp(0.0, 1.0);
                let r = (from.r as f32 * (1.0 - t) + to.r as f32 * t) as u8;
                let g = (from.g as f32 * (1.0 - t) + to.g as f32 * t) as u8;
                let b = (from.b as f32 * (1.0 - t) + to.b as f32 * t) as u8;
                let a = (from.a as f32 * (1.0 - t) + to.a as f32 * t) as u8;
                let c = Color { r, g, b, a };
                let idx = row + xx as usize;
                if a == 255 {
                    self.pixels[idx] = c.to_bgra_u32();
                } else if a > 0 {
                    self.pixels[idx] = blend_bgra(self.pixels[idx], c);
                }
            }
        }
    }

    /// Fill a rect with rounded corners of radius `r` (in pixels).
    /// V1: nearest-neighbour mask — for each row inside the corner
    /// band, compute the inset from the side via `sqrt(r^2 - (r-dy)^2)`
    /// and fill from there. Not anti-aliased, but visually reads as
    /// "rounded" which is what every modern card / button needs.
    /// Paint just the perimeter band of a rounded rectangle — the
    /// difference between an outer rounded rect and an inner one
    /// inset by `ring_width` pixels. Conclave equivalent of
    /// Chrome's `SkCanvas::drawDRRect` (used in
    /// `third_party/blink/renderer/core/paint/box_shadow_painter.cc`
    /// for rounded box-shadow rings).
    ///
    /// Why this exists: stacking N fill_rect_rounded calls (one per
    /// shadow ring) painted N fully-filled rounded rects on top of
    /// each other. The center pixels — which should have NO shadow
    /// because they're under the box's own background — accumulated
    /// N × low_alpha shadow color, ending up nearly opaque on dark
    /// themes. Visible symptom on the explorer.hyvechain.com cards:
    /// each rounded card had a gold tint inside its dark background.
    /// Painting just the ring's perimeter band keeps the shadow
    /// outside the box where it belongs.
    pub fn fill_rect_rounded_ring(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        outer_r: i32,
        ring_width: i32,
        color: Color,
    ) {
        if ring_width <= 0 || w <= 0 || h <= 0 || color.a == 0 {
            return;
        }
        let outer_r = outer_r.max(0).min(w / 2).min(h / 2);
        let inner_x = x + ring_width;
        let inner_y = y + ring_width;
        let inner_w = w - 2 * ring_width;
        let inner_h = h - 2 * ring_width;
        let inner_r = (outer_r - ring_width).max(0);
        // Pixel-in-rounded-rect test. Per CSS Backgrounds Level 3 §6.1
        // the rounded rect = rectangle minus the four corner discs.
        fn inside_rounded(px: i32, py: i32, bx: i32, by: i32, bw: i32, bh: i32, br: i32) -> bool {
            if px < bx || py < by || px >= bx + bw || py >= by + bh {
                return false;
            }
            if br <= 0 {
                return true;
            }
            // Top-left corner.
            let cx = bx + br;
            let cy = by + br;
            if px < cx && py < cy {
                let dx = px - cx;
                let dy = py - cy;
                return dx * dx + dy * dy <= br * br;
            }
            // Top-right.
            let cx = bx + bw - br - 1;
            let cy = by + br;
            if px > cx && py < cy {
                let dx = px - cx;
                let dy = py - cy;
                return dx * dx + dy * dy <= br * br;
            }
            // Bottom-left.
            let cx = bx + br;
            let cy = by + bh - br - 1;
            if px < cx && py > cy {
                let dx = px - cx;
                let dy = py - cy;
                return dx * dx + dy * dy <= br * br;
            }
            // Bottom-right.
            let cx = bx + bw - br - 1;
            let cy = by + bh - br - 1;
            if px > cx && py > cy {
                let dx = px - cx;
                let dy = py - cy;
                return dx * dx + dy * dy <= br * br;
            }
            // Inside the cross-shaped middle band.
            true
        }
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.width as i32);
        let y1 = (y + h).min(self.height as i32);
        for py in y0..y1 {
            for px in x0..x1 {
                if !inside_rounded(px, py, x, y, w, h, outer_r) {
                    continue;
                }
                if inner_w > 0
                    && inner_h > 0
                    && inside_rounded(px, py, inner_x, inner_y, inner_w, inner_h, inner_r)
                {
                    continue;
                }
                let i = (py as usize) * (self.width as usize) + (px as usize);
                if color.a == 255 {
                    self.pixels[i] = color.to_bgra_u32();
                } else {
                    self.pixels[i] = blend_bgra(self.pixels[i], color);
                }
            }
        }
    }

    /// Paint the top half of an elliptical border ring. This is a compact
    /// approximation of CSS rounded per-side border painting for the common
    /// `border-radius: 50%; border: ...; border-top: ...` pattern: the base
    /// ring is painted first, then this overlays the highlighted top side as an
    /// arc instead of a rectangular strip.
    pub fn fill_ellipse_ring_top(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        ring_width: i32,
        color: Color,
    ) {
        if ring_width <= 0 || w <= 0 || h <= 0 || color.a == 0 {
            return;
        }
        let rx = w as f32 * 0.5;
        let ry = h as f32 * 0.5;
        if rx <= 0.0 || ry <= 0.0 {
            return;
        }
        let inner_rx = (rx - ring_width as f32).max(0.0);
        let inner_ry = (ry - ring_width as f32).max(0.0);
        let cx = x as f32 + rx;
        let cy = y as f32 + ry;
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.width as i32);
        let y1 = (y + h).min(self.height as i32);
        for py in y0..y1 {
            let fy = py as f32 + 0.5;
            if fy > cy {
                continue;
            }
            for px in x0..x1 {
                let fx = px as f32 + 0.5;
                let nx = (fx - cx) / rx;
                let ny = (fy - cy) / ry;
                if nx * nx + ny * ny > 1.0 {
                    continue;
                }
                if inner_rx > 0.0 && inner_ry > 0.0 {
                    let inx = (fx - cx) / inner_rx;
                    let iny = (fy - cy) / inner_ry;
                    if inx * inx + iny * iny < 1.0 {
                        continue;
                    }
                }
                let i = (py as usize) * (self.width as usize) + (px as usize);
                if color.a == 255 {
                    self.pixels[i] = color.to_bgra_u32();
                } else {
                    self.pixels[i] = blend_bgra(self.pixels[i], color);
                }
            }
        }
    }

    pub fn fill_rect_rounded(&mut self, x: i32, y: i32, w: i32, h: i32, r: i32, color: Color) {
        if r <= 0 {
            self.fill_rect(x, y, w, h, color);
            return;
        }
        let r = r.min(w / 2).min(h / 2);
        // Middle band (full width) — between the top and bottom corner
        // bands.
        self.fill_rect(x, y + r, w, h - 2 * r, color);
        // Top and bottom bands — narrower toward the corners.
        for dy in 0..r {
            let row_top = y + dy;
            let row_bot = y + h - 1 - dy;
            // Inset from the corner: how many pixels in from the side
            // are still inside the circular cap?
            let v = ((r * r) - ((r - dy) * (r - dy))) as f32;
            let dx = r - v.sqrt() as i32;
            let xs = x + dx;
            let ws = w - 2 * dx;
            if ws > 0 {
                self.fill_rect(xs, row_top, ws, 1, color);
                self.fill_rect(xs, row_bot, ws, 1, color);
            }
        }
    }

    pub fn fill_rect_radial_gradient(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        radius_px: i32,
        inner: Color,
        outer: Color,
    ) {
        if w <= 0 || h <= 0 {
            return;
        }
        fn inside_rounded(px: i32, py: i32, bx: i32, by: i32, bw: i32, bh: i32, br: i32) -> bool {
            if px < bx || py < by || px >= bx + bw || py >= by + bh {
                return false;
            }
            if br <= 0 {
                return true;
            }
            let cx = if px < bx + br {
                bx + br
            } else if px >= bx + bw - br {
                bx + bw - br - 1
            } else {
                px
            };
            let cy = if py < by + br {
                by + br
            } else if py >= by + bh - br {
                by + bh - br - 1
            } else {
                py
            };
            let dx = px - cx;
            let dy = py - cy;
            dx * dx + dy * dy <= br * br
        }
        let cx = x as f32 + w as f32 * 0.5;
        let cy = y as f32 + h as f32 * 0.5;
        let max_r = (w.min(h) as f32 * 0.5).max(1.0);
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.width as i32);
        let y1 = (y + h).min(self.height as i32);
        let rr = radius_px.max(0).min(w / 2).min(h / 2);
        for py in y0..y1 {
            for px in x0..x1 {
                if !inside_rounded(px, py, x, y, w, h, rr) {
                    continue;
                }
                let dx = px as f32 + 0.5 - cx;
                let dy = py as f32 + 0.5 - cy;
                let t = ((dx * dx + dy * dy).sqrt() / max_r).clamp(0.0, 1.0);
                let c = mix_color(inner, outer, t);
                if c.a == 0 {
                    continue;
                }
                let i = (py as usize) * (self.width as usize) + (px as usize);
                if c.a == 255 {
                    self.pixels[i] = c.to_bgra_u32();
                } else {
                    self.pixels[i] = blend_bgra(self.pixels[i], c);
                }
            }
        }
    }

    /// Real N-stop CSS **linear-gradient** fill. CSS Images 3 §3.1.
    ///
    /// The gradient line passes through the box center at `angle_deg`
    /// (CSS convention: 0° points up, increasing clockwise). Its length
    /// is `abs(W·sin A) + abs(H·cos A)` (§3.1.1). Each pixel projects
    /// onto the line to get an offset; the offset is normalized to the
    /// line length and sampled through ALL stops. `repeating` tiles the
    /// stop band (the offset is taken modulo the band span).
    ///
    /// `stops` are abstract (positions may be unspecified or px); they
    /// are resolved via the §3.4.3 fix-up against the line length here.
    pub fn fill_rect_linear_gradient_stops(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        angle_deg: f32,
        stops: &[AbstractStop],
        repeating: bool,
        clip_radius: i32,
    ) {
        if w <= 0 || h <= 0 {
            return;
        }
        // Gradient line length per §3.1.1.
        let a = angle_deg.to_radians();
        let (wf, hf) = (w as f32, h as f32);
        let line_len = (wf * a.sin()).abs() + (hf * a.cos()).abs();
        let line_len = line_len.max(1e-3);
        let resolved = resolve_gradient_stops(stops, line_len);
        if resolved.is_empty() {
            return;
        }
        // Unit vector along the gradient line in screen space. CSS angle
        // 0° = up = (0,-1); +clockwise. Screen y grows downward, so
        // direction = (sin A, -cos A).
        let dx = a.sin();
        let dy = -a.cos();
        // Gradient start point = center - (line_len/2)·dir; the offset of
        // a pixel = projection onto dir from the start, / line_len.
        let cx = wf * 0.5;
        let cy = hf * 0.5;
        let start_x = cx - dir_scaled(dx, line_len);
        let start_y = cy - dir_scaled(dy, line_len);
        let (span, smin) = stop_band(&resolved);
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.width as i32);
        let y1 = (y + h).min(self.height as i32);
        let rr = clip_radius.max(0).min(w / 2).min(h / 2);
        for py in y0..y1 {
            for px in x0..x1 {
                if rr > 0 && !point_in_rounded(px, py, x, y, w, h, rr) {
                    continue;
                }
                let lx = (px - x) as f32 + 0.5 - start_x;
                let ly = (py - y) as f32 + 0.5 - start_y;
                let mut t = (lx * dx + ly * dy) / line_len;
                t = map_offset(t, repeating, span, smin);
                let c = sample_stops_offset(&resolved, t);
                self.blend_or_set(px, py, c);
            }
        }
    }

    /// Real N-stop CSS **radial-gradient** fill. CSS Images 3 §3.2.
    ///
    /// `center_*` is the gradient center in px relative to the box's
    /// top-left. `shape`/`size` give the ending shape; the per-pixel
    /// normalized distance to the ending shape (1.0 at the edge) drives
    /// the stop sampling. `repeating` tiles the band.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_rect_radial_gradient_stops(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        center_x: f32,
        center_y: f32,
        shape: GfxRadialShape,
        size: GfxRadialSize,
        stops: &[AbstractStop],
        repeating: bool,
        clip_radius: i32,
    ) {
        if w <= 0 || h <= 0 {
            return;
        }
        let (rx, ry) = radial_radii(w as f32, h as f32, center_x, center_y, shape, size);
        let (rx, ry) = (rx.max(1e-3), ry.max(1e-3));
        // Resolve stops against rx (the gradient extent along the major
        // axis; px stops resolve against rx per the spec — the gradient
        // ray length). For ellipses the normalized distance handles ry.
        let resolved = resolve_gradient_stops(stops, rx);
        if resolved.is_empty() {
            return;
        }
        let (span, smin) = stop_band(&resolved);
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.width as i32);
        let y1 = (y + h).min(self.height as i32);
        let rr = clip_radius.max(0).min(w / 2).min(h / 2);
        for py in y0..y1 {
            for px in x0..x1 {
                if rr > 0 && !point_in_rounded(px, py, x, y, w, h, rr) {
                    continue;
                }
                let ddx = ((px - x) as f32 + 0.5 - center_x) / rx;
                let ddy = ((py - y) as f32 + 0.5 - center_y) / ry;
                let mut t = (ddx * ddx + ddy * ddy).sqrt();
                t = map_offset(t, repeating, span, smin);
                let c = sample_stops_offset(&resolved, t);
                self.blend_or_set(px, py, c);
            }
        }
    }

    /// Real N-stop CSS **conic-gradient** fill. CSS Images 4 §3.3.
    ///
    /// The gradient sweeps clockwise around `center_*` starting at
    /// `from_deg` (0° = up/12-o'clock). A pixel's polar angle (measured
    /// clockwise from the start) normalized to one turn (0..1) drives the
    /// stop sampling. `repeating` tiles the band.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_rect_conic_gradient_stops(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        center_x: f32,
        center_y: f32,
        from_deg: f32,
        stops: &[AbstractStop],
        repeating: bool,
        clip_radius: i32,
    ) {
        if w <= 0 || h <= 0 {
            return;
        }
        // Conic positions are angles; resolve fix-up against 1 turn.
        let resolved = resolve_gradient_stops(stops, 1.0);
        if resolved.is_empty() {
            return;
        }
        let (span, smin) = stop_band(&resolved);
        let from_t = from_deg / 360.0;
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.width as i32);
        let y1 = (y + h).min(self.height as i32);
        let rr = clip_radius.max(0).min(w / 2).min(h / 2);
        for py in y0..y1 {
            for px in x0..x1 {
                if rr > 0 && !point_in_rounded(px, py, x, y, w, h, rr) {
                    continue;
                }
                let ddx = (px - x) as f32 + 0.5 - center_x;
                let ddy = (py - y) as f32 + 0.5 - center_y;
                // Angle clockwise from up (0° = +y-up = (0,-1)).
                // screen y grows down. atan2(dx, -dy) gives 0 at up,
                // increasing clockwise.
                let ang = ddx.atan2(-ddy); // (-pi, pi], 0 = up, cw positive
                let mut turn = ang / (2.0 * core::f32::consts::PI); // (-0.5, 0.5]
                turn -= from_t;
                // Normalize to [0,1).
                turn = turn.rem_euclid(1.0);
                let t = map_offset(turn, repeating, span, smin);
                let c = sample_stops_offset(&resolved, t);
                self.blend_or_set(px, py, c);
            }
        }
    }

    /// Paint a CSS **linear-gradient clipped to a text glyph mask** —
    /// the real `background-clip: text` / "gradient text" idiom (CSS
    /// Backgrounds 3 §3.2 `background-clip`, combined with
    /// `-webkit-text-fill-color: transparent`).
    ///
    /// Chrome/Skia render this by painting the element's background fill
    /// normally and then clipping it to the union of the text's glyph
    /// outlines (`SkCanvas::clipPath(textBlobPath)` →
    /// `drawPaint(gradientShader)`, see `TextPainter::Paint` →
    /// `GraphicsContext::ClipPath` in
    /// `third_party/blink/renderer/core/paint/text_painter.cc`). The box
    /// background paint itself is suppressed (the clip is the glyphs).
    ///
    /// We mirror that exactly: the gradient is rasterized in the run's
    /// box coordinate space (so the ramp runs across the whole word, not
    /// per-glyph), and each pixel's gradient-alpha is multiplied by the
    /// glyph coverage from `mask`. Where the glyph mask is 0 (between
    /// letters) nothing is written → transparent, never a solid fill
    /// block. `mask` is a row-major `mask_w × mask_h` alpha buffer whose
    /// top-left maps to `(box_x, box_y)`. `angle_deg` follows the CSS
    /// convention (0° = up, +clockwise). `global_alpha` (0..=255) scales
    /// the whole run (element `opacity`/inherited text alpha).
    #[allow(clippy::too_many_arguments)]
    pub fn blit_text_run_gradient(
        &mut self,
        box_x: i32,
        box_y: i32,
        box_w: i32,
        box_h: i32,
        mask: &[u8],
        mask_w: i32,
        mask_h: i32,
        angle_deg: f32,
        stops: &[AbstractStop],
        repeating: bool,
        global_alpha: u8,
        clip: Option<(i32, i32, i32, i32)>,
    ) {
        if box_w <= 0 || box_h <= 0 || mask_w <= 0 || mask_h <= 0 || global_alpha == 0 {
            return;
        }
        // Gradient line geometry over the RUN BOX — identical math to
        // `fill_rect_linear_gradient_stops` so the ramp matches a real
        // box gradient of the same size (single source of truth).
        let a = angle_deg.to_radians();
        let (wf, hf) = (box_w as f32, box_h as f32);
        let line_len = ((wf * a.sin()).abs() + (hf * a.cos()).abs()).max(1e-3);
        let resolved = resolve_gradient_stops(stops, line_len);
        if resolved.is_empty() {
            return;
        }
        let dx = a.sin();
        let dy = -a.cos();
        let cx = wf * 0.5;
        let cy = hf * 0.5;
        let start_x = cx - dir_scaled(dx, line_len);
        let start_y = cy - dir_scaled(dy, line_len);
        let (span, smin) = stop_band(&resolved);
        // Destination span = intersection of (mask footprint at box
        // origin) ∩ (bitmap) ∩ (optional damage clip R).
        let mut x0 = box_x.max(0);
        let mut y0 = box_y.max(0);
        let mut x1 = (box_x + mask_w).min(self.width as i32);
        let mut y1 = (box_y + mask_h).min(self.height as i32);
        if let Some((rx, ry, rw, rh)) = clip {
            x0 = x0.max(rx);
            y0 = y0.max(ry);
            x1 = x1.min(rx + rw);
            y1 = y1.min(ry + rh);
        }
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let ga = global_alpha as u32;
        for py in y0..y1 {
            let my = py - box_y; // mask row (>=0, < mask_h by clip above)
            let mrow = (my * mask_w) as usize;
            for px in x0..x1 {
                let mx = px - box_x;
                let cov = mask[mrow + mx as usize] as u32;
                if cov == 0 {
                    continue; // transparent between glyphs — no fill block
                }
                // Sample the gradient at this pixel's projection onto the
                // gradient line, in the run-box frame.
                let lx = (px - box_x) as f32 + 0.5 - start_x;
                let ly = (py - box_y) as f32 + 0.5 - start_y;
                let mut t = (lx * dx + ly * dy) / line_len;
                t = map_offset(t, repeating, span, smin);
                let mut c = sample_stops_offset(&resolved, t);
                // Multiply gradient alpha by glyph coverage and the run's
                // global alpha (straight-alpha): a = a_grad * cov * ga.
                let final_a = (c.a as u32 * cov * ga) / (255 * 255);
                if final_a == 0 {
                    continue;
                }
                c.a = final_a as u8;
                self.blend_or_set(px, py, c);
            }
        }
    }

    /// Source-over a color onto a pixel (opaque writes hard).
    #[inline]
    fn blend_or_set(&mut self, px: i32, py: i32, c: Color) {
        if c.a == 0 {
            return;
        }
        let i = (py as usize) * (self.width as usize) + (px as usize);
        if c.a == 255 {
            self.pixels[i] = c.to_bgra_u32();
        } else {
            self.pixels[i] = blend_bgra(self.pixels[i], c);
        }
    }

    /// Composite a sub-bitmap onto this one at `(x, y)` with a uniform
    /// group alpha. Each source pixel's existing alpha is multiplied by
    /// `group_alpha` and the result is alpha-blended over the destination.
    /// This is the Conclave equivalent of Chrome's
    /// `SkCanvas::saveLayerAlpha` → draw subtree → `restore()` pattern
    /// (from `cc/paint/skia_paint_canvas.cc`): everything painted into
    /// the source bitmap composes as a transparency group, then the
    /// whole group composites once onto the parent. Overlapping ops
    /// inside the source compose at full alpha first, then the
    /// composite-once step gives the correct final alpha — which the
    /// per-op `a * group_alpha` model could not match when ops within
    /// an element overlapped.
    pub fn blit_with_group_alpha(&mut self, x: i32, y: i32, src: &Bitmap, group_alpha: f32) {
        if group_alpha <= 0.0 {
            return;
        }
        let ga = group_alpha.clamp(0.0, 1.0);
        for sy in 0..src.height {
            let dy = y + sy as i32;
            if dy < 0 || dy >= self.height as i32 {
                continue;
            }
            for sx in 0..src.width {
                let dx = x + sx as i32;
                if dx < 0 || dx >= self.width as i32 {
                    continue;
                }
                let s = src.pixels[(sy as usize) * (src.width as usize) + sx as usize];
                let sa = ((s >> 24) & 0xFF) as f32;
                if sa == 0.0 {
                    continue;
                }
                let final_a = (sa * ga).round() as u8;
                if final_a == 0 {
                    continue;
                }
                let di = (dy as usize) * (self.width as usize) + dx as usize;
                self.pixels[di] = blend_bgra(
                    self.pixels[di],
                    Color {
                        b: (s & 0xFF) as u8,
                        g: ((s >> 8) & 0xFF) as u8,
                        r: ((s >> 16) & 0xFF) as u8,
                        a: final_a,
                    },
                );
            }
        }
    }

    /// Composite a transparent `src` layer over `self` at `(x, y)` using a CSS
    /// [`BlendMode`] and an additional `group_alpha`. This is the
    /// `mix-blend-mode` element-vs-backdrop composite: `src` holds the element's
    /// subtree painted into a transparent layer, and each non-transparent layer
    /// pixel blends with the page pixel beneath it per the W3C blend formula.
    /// `BlendMode::Normal` is byte-identical to `blit_with_group_alpha`.
    pub fn blit_layer_blend(&mut self, x: i32, y: i32, src: &Bitmap, mode: BlendMode, group_alpha: f32) {
        if group_alpha <= 0.0 {
            return;
        }
        let ga = group_alpha.clamp(0.0, 1.0);
        for sy in 0..src.height {
            let dy = y + sy as i32;
            if dy < 0 || dy >= self.height as i32 {
                continue;
            }
            for sx in 0..src.width {
                let dx = x + sx as i32;
                if dx < 0 || dx >= self.width as i32 {
                    continue;
                }
                let s = src.pixels[(sy as usize) * (src.width as usize) + sx as usize];
                let sa = ((s >> 24) & 0xFF) as f32;
                if sa == 0.0 {
                    continue;
                }
                let final_a = (sa * ga).round() as u8;
                if final_a == 0 {
                    continue;
                }
                let color = Color {
                    b: (s & 0xFF) as u8,
                    g: ((s >> 8) & 0xFF) as u8,
                    r: ((s >> 16) & 0xFF) as u8,
                    a: final_a,
                };
                let di = (dy as usize) * (self.width as usize) + dx as usize;
                self.pixels[di] = blend_mode_pixel(self.pixels[di], color, mode);
            }
        }
    }

    /// Bilinear sample of the BGRA buffer at fractional `(x, y)` (pixel
    /// centres at `i + 0.5`). Edges clamp. Returns a packed BGRA u32.
    fn sample_bilinear(&self, x: f32, y: f32) -> u32 {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let w = self.width as i32;
        let h = self.height as i32;
        let get = |xx: i32, yy: i32| -> (f32, f32, f32, f32) {
            let cx = xx.clamp(0, w - 1);
            let cy = yy.clamp(0, h - 1);
            let p = self.pixels[(cy as usize) * (self.width as usize) + cx as usize];
            (
                (p & 0xFF) as f32,
                ((p >> 8) & 0xFF) as f32,
                ((p >> 16) & 0xFF) as f32,
                ((p >> 24) & 0xFF) as f32,
            )
        };
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let mix = |c0: (f32, f32, f32, f32), c1: (f32, f32, f32, f32), t: f32| {
            (
                lerp(c0.0, c1.0, t),
                lerp(c0.1, c1.1, t),
                lerp(c0.2, c1.2, t),
                lerp(c0.3, c1.3, t),
            )
        };
        let top = mix(get(x0, y0), get(x0 + 1, y0), tx);
        let bot = mix(get(x0, y0 + 1), get(x0 + 1, y0 + 1), tx);
        let (b, g, r, a) = mix(top, bot, ty);
        ((a.round() as u32) << 24)
            | ((r.round() as u32) << 16)
            | ((g.round() as u32) << 8)
            | (b.round() as u32)
    }

    /// Composite `src` onto `self` through a 2×3 affine transform `m`
    /// that maps *source-local* pixel coordinates `(sx, sy)` to
    /// destination coordinates:
    ///   `dx = m[0]*sx + m[2]*sy + m[4]`,
    ///   `dy = m[1]*sx + m[3]*sy + m[5]`.
    /// The destination region is the bounding box of the four mapped
    /// source corners; each covered pixel is inverse-mapped, bilinearly
    /// sampled, and alpha-blended (fully-transparent samples skipped).
    /// This is the CSS `transform: rotate()/matrix()` layer composite —
    /// the rotated subtree is painted into `src` (a transparent layer)
    /// and stamped back through its transform matrix.
    pub fn blit_affine(&mut self, src: &Bitmap, m: [f32; 6], group_alpha: f32) {
        if group_alpha <= 0.0 || src.width == 0 || src.height == 0 {
            return;
        }
        let ga = group_alpha.clamp(0.0, 1.0);
        let (sw, sh) = (src.width as f32, src.height as f32);
        let mut minx = f32::MAX;
        let mut miny = f32::MAX;
        let mut maxx = f32::MIN;
        let mut maxy = f32::MIN;
        for (cx, cy) in [(0.0, 0.0), (sw, 0.0), (0.0, sh), (sw, sh)] {
            let dx = m[0] * cx + m[2] * cy + m[4];
            let dy = m[1] * cx + m[3] * cy + m[5];
            minx = minx.min(dx);
            maxx = maxx.max(dx);
            miny = miny.min(dy);
            maxy = maxy.max(dy);
        }
        let det = m[0] * m[3] - m[2] * m[1];
        if det.abs() < 1e-9 {
            return;
        }
        let inv = 1.0 / det;
        // Inverse of the 2×2 linear part [[m0, m2],[m1, m3]].
        let (ia, ic, ib, id) = (m[3] * inv, -m[2] * inv, -m[1] * inv, m[0] * inv);
        let x0 = (minx.floor() as i32).max(0);
        let y0 = (miny.floor() as i32).max(0);
        let x1 = (maxx.ceil() as i32).min(self.width as i32);
        let y1 = (maxy.ceil() as i32).min(self.height as i32);
        for dy in y0..y1 {
            for dx in x0..x1 {
                let rx = dx as f32 + 0.5 - m[4];
                let ry = dy as f32 + 0.5 - m[5];
                let sx = ia * rx + ic * ry;
                let sy = ib * rx + id * ry;
                if sx < 0.0 || sy < 0.0 || sx >= sw || sy >= sh {
                    continue;
                }
                let sample = src.sample_bilinear(sx, sy);
                let sa = ((sample >> 24) & 0xFF) as f32;
                if sa == 0.0 {
                    continue;
                }
                let final_a = (sa * ga).round() as u8;
                if final_a == 0 {
                    continue;
                }
                let di = (dy as usize) * (self.width as usize) + dx as usize;
                self.pixels[di] = blend_bgra(
                    self.pixels[di],
                    Color {
                        b: (sample & 0xFF) as u8,
                        g: ((sample >> 8) & 0xFF) as u8,
                        r: ((sample >> 16) & 0xFF) as u8,
                        a: final_a,
                    },
                );
            }
        }
    }

    /// Composite `src` onto `self` through a projective (perspective) quad
    /// mapping: the source rectangle's four corners — top-left `(0,0)`,
    /// top-right `(sw,0)`, bottom-right `(sw,sh)`, bottom-left `(0,sh)` —
    /// map to the four destination points `dst_corners` (same TL,TR,BR,BL
    /// order). Unlike `blit_affine` this supports a non-affine (perspective)
    /// warp, which is what CSS 3D transforms with perspective produce after
    /// the w-divide (CSS Transforms 2 §13.1) — e.g. rotateX foreshortening.
    ///
    /// Implementation: compute the inverse homography mapping destination
    /// pixels back to the unit square, scale to source pixels, bilinearly
    /// sample, alpha-blend. The destination region is the bounding box of
    /// the four corners. This is the Skia `SkMatrix`/`drawImageRect` model
    /// for a perspective texture map, done on the CPU.
    pub fn blit_quad_projective(
        &mut self,
        src: &Bitmap,
        dst_corners: [(f32, f32); 4],
        group_alpha: f32,
    ) {
        if group_alpha <= 0.0 || src.width == 0 || src.height == 0 {
            return;
        }
        let ga = group_alpha.clamp(0.0, 1.0);
        let (sw, sh) = (src.width as f32, src.height as f32);
        // Forward homography H mapping the UNIT square (u,v)∈[0,1]² to the
        // destination quad. Standard 4-point construction (Heckbert 1989,
        // "Fundamentals of Texture Mapping"). Corner order: (0,0),(1,0),
        // (1,1),(0,1) → dst[0],dst[1],dst[2],dst[3].
        let (x0, y0) = dst_corners[0];
        let (x1, y1) = dst_corners[1];
        let (x2, y2) = dst_corners[2];
        let (x3, y3) = dst_corners[3];
        let dx1 = x1 - x2;
        let dx2 = x3 - x2;
        let dx3 = x0 - x1 + x2 - x3;
        let dy1 = y1 - y2;
        let dy2 = y3 - y2;
        let dy3 = y0 - y1 + y2 - y3;
        let (a, b, c, d, e, f, g, h);
        if dx3.abs() < 1e-9 && dy3.abs() < 1e-9 {
            // Affine (parallelogram) case.
            a = x1 - x0;
            b = x2 - x1;
            c = x0;
            d = y1 - y0;
            e = y2 - y1;
            f = y0;
            g = 0.0;
            h = 0.0;
        } else {
            let den = dx1 * dy2 - dx2 * dy1;
            if den.abs() < 1e-12 {
                return;
            }
            g = (dx3 * dy2 - dx2 * dy3) / den;
            h = (dx1 * dy3 - dx3 * dy1) / den;
            a = x1 - x0 + g * x1;
            b = x3 - x0 + h * x3;
            c = x0;
            d = y1 - y0 + g * y1;
            e = y3 - y0 + h * y3;
            f = y0;
        }
        // Invert H (3×3) to map destination (X,Y,1) back to (u,v,w).
        let m = [a, b, c, d, e, f, g, h, 1.0];
        let det = m[0] * (m[4] * m[8] - m[5] * m[7])
            - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
        if det.abs() < 1e-12 {
            return;
        }
        let inv_det = 1.0 / det;
        let inv = [
            (m[4] * m[8] - m[5] * m[7]) * inv_det,
            (m[2] * m[7] - m[1] * m[8]) * inv_det,
            (m[1] * m[5] - m[2] * m[4]) * inv_det,
            (m[5] * m[6] - m[3] * m[8]) * inv_det,
            (m[0] * m[8] - m[2] * m[6]) * inv_det,
            (m[2] * m[3] - m[0] * m[5]) * inv_det,
            (m[3] * m[7] - m[4] * m[6]) * inv_det,
            (m[1] * m[6] - m[0] * m[7]) * inv_det,
            (m[0] * m[4] - m[1] * m[3]) * inv_det,
        ];
        // Destination bounding box.
        let minx = x0.min(x1).min(x2).min(x3);
        let maxx = x0.max(x1).max(x2).max(x3);
        let miny = y0.min(y1).min(y2).min(y3);
        let maxy = y0.max(y1).max(y2).max(y3);
        let px0 = (minx.floor() as i32).max(0);
        let py0 = (miny.floor() as i32).max(0);
        let px1 = (maxx.ceil() as i32).min(self.width as i32);
        let py1 = (maxy.ceil() as i32).min(self.height as i32);
        for dy in py0..py1 {
            for dx in px0..px1 {
                let fx = dx as f32 + 0.5;
                let fy = dy as f32 + 0.5;
                // (u', v', w) = inv · (fx, fy, 1)
                let up = inv[0] * fx + inv[1] * fy + inv[2];
                let vp = inv[3] * fx + inv[4] * fy + inv[5];
                let wp = inv[6] * fx + inv[7] * fy + inv[8];
                if wp.abs() < 1e-9 {
                    continue;
                }
                let u = up / wp;
                let v = vp / wp;
                if u < 0.0 || v < 0.0 || u > 1.0 || v > 1.0 {
                    continue;
                }
                let sx = u * sw;
                let sy = v * sh;
                // Clamp into the sampling range so edge pixels render.
                let sx = sx.clamp(0.0, sw - 1e-3);
                let sy = sy.clamp(0.0, sh - 1e-3);
                let sample = src.sample_bilinear(sx, sy);
                let sa = ((sample >> 24) & 0xFF) as f32;
                if sa == 0.0 {
                    continue;
                }
                let final_a = (sa * ga).round() as u8;
                if final_a == 0 {
                    continue;
                }
                let di = (dy as usize) * (self.width as usize) + dx as usize;
                self.pixels[di] = blend_bgra(
                    self.pixels[di],
                    Color {
                        b: (sample & 0xFF) as u8,
                        g: ((sample >> 8) & 0xFF) as u8,
                        r: ((sample >> 16) & 0xFF) as u8,
                        a: final_a,
                    },
                );
            }
        }
    }

    /// Blit a BGRA u32 image at `(x, y)`. Clips to the destination bounds.
    /// Fully transparent source pixels are skipped; fully opaque pixels
    /// overwrite; semi-transparent pixels alpha-blend over the current
    /// destination.
    pub fn blit_bgra(&mut self, x: i32, y: i32, src_w: u32, src_h: u32, src: &[u32]) {
        for sy in 0..src_h {
            let dy = y + sy as i32;
            if dy < 0 || dy >= self.height as i32 {
                continue;
            }
            for sx in 0..src_w {
                let dx = x + sx as i32;
                if dx < 0 || dx >= self.width as i32 {
                    continue;
                }
                let s = src[(sy as usize) * (src_w as usize) + sx as usize];
                let sa = ((s >> 24) & 0xFF) as u8;
                if sa == 0 {
                    continue;
                }
                let di = (dy as usize) * (self.width as usize) + dx as usize;
                if sa == 255 {
                    self.pixels[di] = s;
                } else {
                    self.pixels[di] = blend_bgra(self.pixels[di], unpack_bgra(s));
                }
            }
        }
    }

    /// Blit a BGRA image positioned by a (possibly negative) offset and
    /// CLIPPED to a box rect — the CSS `background-position` + `no-repeat`
    /// + `overflow:hidden` sprite case. The image's top-left is placed at
    /// `(box_x + off_x, box_y + off_y)`; only pixels falling inside the
    /// `[box_x, box_x+box_w) × [box_y, box_y+box_h)` window are painted,
    /// so a large sprite sheet shows only the sub-region the negative
    /// offset selects (e.g. Wikipedia's wordmark at `0px -304px`).
    #[allow(clippy::too_many_arguments)]
    pub fn blit_bgra_sprite(
        &mut self,
        box_x: i32,
        box_y: i32,
        box_w: i32,
        box_h: i32,
        off_x: i32,
        off_y: i32,
        src_w: u32,
        src_h: u32,
        src: &[u32],
    ) {
        if box_w <= 0 || box_h <= 0 || src_w == 0 || src_h == 0 {
            return;
        }
        // Clamp the paint window to the box AND the bitmap.
        let clip_x0 = box_x.max(0);
        let clip_y0 = box_y.max(0);
        let clip_x1 = (box_x + box_w).min(self.width as i32);
        let clip_y1 = (box_y + box_h).min(self.height as i32);
        let img_x = box_x + off_x;
        let img_y = box_y + off_y;
        for dy in clip_y0..clip_y1 {
            let sy = dy - img_y;
            if sy < 0 || sy >= src_h as i32 {
                continue;
            }
            for dx in clip_x0..clip_x1 {
                let sx = dx - img_x;
                if sx < 0 || sx >= src_w as i32 {
                    continue;
                }
                let s = src[(sy as usize) * (src_w as usize) + sx as usize];
                let sa = ((s >> 24) & 0xFF) as u8;
                if sa == 0 {
                    continue;
                }
                let di = (dy as usize) * (self.width as usize) + dx as usize;
                if sa == 255 {
                    self.pixels[di] = s;
                } else {
                    self.pixels[di] = blend_bgra(self.pixels[di], unpack_bgra(s));
                }
            }
        }
    }

    /// Scale-blit a BGRA u32 image into the destination rect using
    /// nearest-neighbour sampling. When dst dims equal src dims this
    /// is exactly `blit_bgra`. Used by the image painter when CSS
    /// `object-fit` resolves to a destination rect different from the
    /// source's intrinsic size — `contain` and `cover` both go through
    /// this path with the appropriate `dx`/`dy`/`dw`/`dh`.
    pub fn blit_bgra_scaled(
        &mut self,
        dx: i32,
        dy: i32,
        dw: u32,
        dh: u32,
        src_w: u32,
        src_h: u32,
        src: &[u32],
    ) {
        if dw == 0 || dh == 0 || src_w == 0 || src_h == 0 {
            return;
        }
        for yy in 0..dh {
            let dest_y = dy + yy as i32;
            if dest_y < 0 || dest_y >= self.height as i32 {
                continue;
            }
            for xx in 0..dw {
                let dest_x = dx + xx as i32;
                if dest_x < 0 || dest_x >= self.width as i32 {
                    continue;
                }
                // Sample at destination pixel centres and bilinearly
                // reconstruct from the surrounding source texels. This
                // preserves centred details when large source images are
                // shrunk into small UI slots, which nearest-neighbour
                // tends to obliterate.
                let s = if src_w > dw || src_h > dh {
                    let src_x0 = (xx as f32) * (src_w as f32) / (dw as f32);
                    let src_y0 = (yy as f32) * (src_h as f32) / (dh as f32);
                    let src_x1 = ((xx + 1) as f32) * (src_w as f32) / (dw as f32);
                    let src_y1 = ((yy + 1) as f32) * (src_h as f32) / (dh as f32);
                    sample_bgra_box(src_w, src_h, src, src_x0, src_y0, src_x1, src_y1)
                } else {
                    let src_fx = (((xx as f32) + 0.5) * (src_w as f32) / (dw as f32) - 0.5)
                        .clamp(0.0, (src_w - 1) as f32);
                    let src_fy = (((yy as f32) + 0.5) * (src_h as f32) / (dh as f32) - 0.5)
                        .clamp(0.0, (src_h - 1) as f32);
                    sample_bgra_bilinear(src_w, src_h, src, src_fx, src_fy)
                };
                let sa = s.a;
                if sa == 0 {
                    continue;
                }
                let di = (dest_y as usize) * (self.width as usize) + dest_x as usize;
                if sa == 255 {
                    self.pixels[di] = s.to_bgra_u32();
                } else {
                    self.pixels[di] = blend_bgra(self.pixels[di], s);
                }
            }
        }
    }

    /// Paint a solid color through the source image's alpha channel,
    /// scaling the mask into the destination rect with nearest-neighbour
    /// sampling. RGB from the source is ignored; only the source alpha
    /// matters. Useful for CSS `mask-image` / `-webkit-mask-image`
    /// tinted-icon rendering.
    pub fn blit_mask_tinted_scaled(
        &mut self,
        dx: i32,
        dy: i32,
        dw: u32,
        dh: u32,
        src_w: u32,
        src_h: u32,
        src: &[u32],
        color: Color,
    ) {
        if dw == 0 || dh == 0 || src_w == 0 || src_h == 0 || color.a == 0 {
            return;
        }
        for yy in 0..dh {
            let dest_y = dy + yy as i32;
            if dest_y < 0 || dest_y >= self.height as i32 {
                continue;
            }
            let sy = ((yy as u64 * src_h as u64) / dh as u64) as u32;
            let sy = sy.min(src_h - 1);
            for xx in 0..dw {
                let dest_x = dx + xx as i32;
                if dest_x < 0 || dest_x >= self.width as i32 {
                    continue;
                }
                let sx = ((xx as u64 * src_w as u64) / dw as u64) as u32;
                let sx = sx.min(src_w - 1);
                let s = src[(sy as usize) * (src_w as usize) + sx as usize];
                let mask_a = ((s >> 24) & 0xFF) as u8;
                if mask_a == 0 {
                    continue;
                }
                let di = (dest_y as usize) * (self.width as usize) + dest_x as usize;
                let src_color = Color {
                    r: color.r,
                    g: color.g,
                    b: color.b,
                    a: ((u16::from(color.a) * u16::from(mask_a)) / 255) as u8,
                };
                self.pixels[di] = blend_bgra(self.pixels[di], src_color);
            }
        }
    }

    /// Apply a single CSS filter function to the pixels inside
    /// `(x, y, w, h)`. `op` is a tagged enum of the operation; the
    /// numeric parameter is in CSS units already (blur is px, others
    /// are 0..1 or multipliers per CSS Filter Effects 1).
    pub fn apply_filter_rect(&mut self, x: i32, y: i32, w: i32, h: i32, op: FilterOp) {
        let (x0, y0, x1, y1) = (
            x.max(0),
            y.max(0),
            (x + w).min(self.width as i32),
            (y + h).min(self.height as i32),
        );
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        match op {
            FilterOp::Blur(radius) => {
                if radius <= 0.0 {
                    return;
                }
                // True Gaussian blur, approximated by THREE successive box
                // blurs per axis exactly as specified for `feGaussianBlur`
                // (SVG 1.1 §15.17 / Filter Effects 1 §9.14). For a standard
                // deviation `s` the box size is
                //     d = floor(s * 3 * sqrt(2*PI) / 4 + 0.5)
                // applied as:
                //   - if d is odd:  three box blurs of size d, centred.
                //   - if d is even: two box blurs of size d offset by ±½ px
                //                   (left/right boundary) and one of size d+1
                //                   centred.
                // `blur(radius)` maps `radius` directly to `stdDeviation`
                // (Filter Effects 1 §8.x: "The passed parameter defines the
                // value of the standard deviation to the Gaussian function").
                // This replaces the old single box blur, which produced a flat
                // (non-Gaussian) falloff that did not match Chrome/Skia.
                let s = radius.clamp(0.0, 200.0);
                let d = (s * 3.0 * (2.0 * std::f32::consts::PI).sqrt() / 4.0 + 0.5).floor() as i32;
                if d < 1 {
                    return;
                }
                gaussian_box3(self, x0, y0, x1, y1, d);
            }
            FilterOp::Brightness(a) => {
                color_per_pixel(self, x0, y0, x1, y1, |r, g, b| {
                    (
                        ((r as f32 * a).clamp(0.0, 255.0)) as u8,
                        ((g as f32 * a).clamp(0.0, 255.0)) as u8,
                        ((b as f32 * a).clamp(0.0, 255.0)) as u8,
                    )
                });
            }
            FilterOp::Contrast(a) => {
                color_per_pixel(self, x0, y0, x1, y1, |r, g, b| {
                    let f = |v: u8| -> u8 {
                        let n = (v as f32 / 255.0 - 0.5) * a + 0.5;
                        (n.clamp(0.0, 1.0) * 255.0) as u8
                    };
                    (f(r), f(g), f(b))
                });
            }
            FilterOp::Grayscale(amt) => {
                let amt = amt.clamp(0.0, 1.0);
                color_per_pixel(self, x0, y0, x1, y1, |r, g, b| {
                    // Rec. 709 luminance weights.
                    let l = 0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32;
                    let lerp = |c: u8| -> u8 { (c as f32 * (1.0 - amt) + l * amt) as u8 };
                    (lerp(r), lerp(g), lerp(b))
                });
            }
            FilterOp::Invert(amt) => {
                let amt = amt.clamp(0.0, 1.0);
                color_per_pixel(self, x0, y0, x1, y1, |r, g, b| {
                    let lerp = |c: u8| -> u8 {
                        let inv = 255 - c;
                        (c as f32 * (1.0 - amt) + inv as f32 * amt) as u8
                    };
                    (lerp(r), lerp(g), lerp(b))
                });
            }
            FilterOp::Sepia(amt) => {
                let amt = amt.clamp(0.0, 1.0);
                color_per_pixel(self, x0, y0, x1, y1, |r, g, b| {
                    let rf = r as f32;
                    let gf = g as f32;
                    let bf = b as f32;
                    // CSS Sepia matrix.
                    let nr = (0.393 * rf + 0.769 * gf + 0.189 * bf).min(255.0);
                    let ng = (0.349 * rf + 0.686 * gf + 0.168 * bf).min(255.0);
                    let nb = (0.272 * rf + 0.534 * gf + 0.131 * bf).min(255.0);
                    let lerp = |orig: f32, target: f32| -> u8 {
                        (orig * (1.0 - amt) + target * amt) as u8
                    };
                    (lerp(rf, nr), lerp(gf, ng), lerp(bf, nb))
                });
            }
            FilterOp::Saturate(amt) => {
                // CSS `saturate(s)` is exactly `feColorMatrix type="saturate"`
                // (Filter Effects 1 §8.4 / SVG 1.1 §15.10). The previous
                // per-pixel-luminance form used the wrong (Rec.709) weights and
                // is NOT what Chrome/Skia compute. The spec matrix:
                //   [0.213+0.787s  0.715-0.715s  0.072-0.072s]
                //   [0.213-0.213s  0.715+0.285s  0.072-0.072s]
                //   [0.213-0.213s  0.715-0.715s  0.072+0.928s]
                let s = amt;
                let m = [
                    [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
                    [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
                    [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
                ];
                color_per_pixel(self, x0, y0, x1, y1, |r, g, b| {
                    let rf = r as f32;
                    let gf = g as f32;
                    let bf = b as f32;
                    let nr = (m[0][0] * rf + m[0][1] * gf + m[0][2] * bf).clamp(0.0, 255.0);
                    let ng = (m[1][0] * rf + m[1][1] * gf + m[1][2] * bf).clamp(0.0, 255.0);
                    let nb = (m[2][0] * rf + m[2][1] * gf + m[2][2] * bf).clamp(0.0, 255.0);
                    (nr as u8, ng as u8, nb as u8)
                });
            }
            FilterOp::HueRotate(degrees) => {
                let theta = degrees.to_radians();
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                // CSS hue-rotate matrix per the spec.
                let m = [
                    [
                        0.213 + cos_t * 0.787 + sin_t * -0.213,
                        0.715 + cos_t * -0.715 + sin_t * -0.715,
                        0.072 + cos_t * -0.072 + sin_t * 0.928,
                    ],
                    [
                        0.213 + cos_t * -0.213 + sin_t * 0.143,
                        0.715 + cos_t * 0.285 + sin_t * 0.140,
                        0.072 + cos_t * -0.072 + sin_t * -0.283,
                    ],
                    [
                        0.213 + cos_t * -0.213 + sin_t * -0.787,
                        0.715 + cos_t * -0.715 + sin_t * 0.715,
                        0.072 + cos_t * 0.928 + sin_t * 0.072,
                    ],
                ];
                color_per_pixel(self, x0, y0, x1, y1, |r, g, b| {
                    let rf = r as f32;
                    let gf = g as f32;
                    let bf = b as f32;
                    let nr = (m[0][0] * rf + m[0][1] * gf + m[0][2] * bf).clamp(0.0, 255.0);
                    let ng = (m[1][0] * rf + m[1][1] * gf + m[1][2] * bf).clamp(0.0, 255.0);
                    let nb = (m[2][0] * rf + m[2][1] * gf + m[2][2] * bf).clamp(0.0, 255.0);
                    (nr as u8, ng as u8, nb as u8)
                });
            }
            FilterOp::Opacity(amt) => {
                let amt = amt.clamp(0.0, 1.0);
                for py in y0..y1 {
                    let row = (py as usize) * (self.width as usize);
                    for px_ in x0..x1 {
                        let i = row + px_ as usize;
                        let p = self.pixels[i];
                        let a = ((p >> 24) & 0xFF) as f32 * amt;
                        self.pixels[i] = (p & 0x00FFFFFF) | ((a as u32) << 24);
                    }
                }
            }
        }
    }

    /// Run an SVG `<filter>` primitive chain over the rect, in place.
    ///
    /// This is the `filter: url(#id)` paint path (Filter Effects 1 §4 / SVG
    /// 1.1 §15). The primitives are evaluated as a small named-result graph:
    /// `SourceGraphic` is the rect's current pixels, `SourceAlpha` is the
    /// rect's alpha as black, and each primitive writes a named result that
    /// later primitives can read. The final primitive's result is composited
    /// back over the rect's region. We support the primitives that make up
    /// every common filter (blur, colour matrix, offset, flood, merge,
    /// composite) — exactly the recipe Chrome/Skia build for the standard
    /// filter functions.
    pub fn apply_svg_filter_rect(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        prims: &[SvgFePrimitive],
    ) {
        if prims.is_empty() {
            return;
        }
        let (x0, y0, x1, y1) = (
            x.max(0),
            y.max(0),
            (x + w).min(self.width as i32),
            (y + h).min(self.height as i32),
        );
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let rw = (x1 - x0) as usize;
        let rh = (y1 - y0) as usize;
        // A named image is an RGBA8 buffer the size of the rect (0xAARRGGBB).
        let lift = |bmp: &Bitmap| -> Vec<u32> {
            let mut v = vec![0u32; rw * rh];
            for ry in 0..rh {
                let srow = (y0 as usize + ry) * bmp.width as usize + x0 as usize;
                v[ry * rw..ry * rw + rw].copy_from_slice(&bmp.pixels[srow..srow + rw]);
            }
            v
        };
        let source_graphic = lift(self);
        // SourceAlpha: keep alpha, force RGB to 0 (Filter Effects 1 §15.7.2).
        let source_alpha: Vec<u32> = source_graphic.iter().map(|p| p & 0xFF00_0000).collect();
        let mut results: Vec<(String, Vec<u32>)> = Vec::new();
        // Resolve a named filter input to its image buffer (Filter Effects 1
        // §15.7.2). An empty name = the previous primitive's result, or
        // SourceGraphic for the first primitive.
        let pick = |name: &str,
                    results: &[(String, Vec<u32>)],
                    last: Option<&Vec<u32>>|
         -> Vec<u32> {
            match name {
                "SourceGraphic" => source_graphic.clone(),
                "SourceAlpha" => source_alpha.clone(),
                "BackgroundImage" | "BackgroundAlpha" => vec![0u32; rw * rh],
                "" => last.cloned().unwrap_or_else(|| source_graphic.clone()),
                other => {
                    if let Some((_, buf)) = results.iter().rev().find(|(n, _)| n == other) {
                        buf.clone()
                    } else if let Some(l) = last {
                        l.clone()
                    } else {
                        source_graphic.clone()
                    }
                }
            }
        };
        let mut last_result: Option<Vec<u32>> = None;
        for prim in prims {
            let out = match prim {
                SvgFePrimitive::GaussianBlur { input, std_dev, .. } => {
                    let mut buf = if input.is_empty() {
                        last_result
                            .clone()
                            .unwrap_or_else(|| source_graphic.clone())
                    } else {
                        pick(input, &results, last_result.as_ref())
                    };
                    fe_gaussian_blur(&mut buf, rw, rh, *std_dev);
                    buf
                }
                SvgFePrimitive::ColorMatrix { input, matrix, .. } => {
                    let buf = if input.is_empty() {
                        last_result
                            .clone()
                            .unwrap_or_else(|| source_graphic.clone())
                    } else {
                        pick(input, &results, last_result.as_ref())
                    };
                    fe_color_matrix(&buf, matrix)
                }
                SvgFePrimitive::Offset { input, dx, dy, .. } => {
                    let buf = if input.is_empty() {
                        last_result
                            .clone()
                            .unwrap_or_else(|| source_graphic.clone())
                    } else {
                        pick(input, &results, last_result.as_ref())
                    };
                    fe_offset(&buf, rw, rh, *dx, *dy)
                }
                SvgFePrimitive::Flood { color, .. } => {
                    let packed = color.to_bgra_u32();
                    vec![packed; rw * rh]
                }
                SvgFePrimitive::Composite { input, in2, op, .. } => {
                    let a = pick(input, &results, last_result.as_ref());
                    let b = pick(in2, &results, last_result.as_ref());
                    fe_composite(&a, &b, *op)
                }
                SvgFePrimitive::Merge { inputs, .. } => {
                    // Stack each input source-over, bottom-to-top.
                    let mut acc = vec![0u32; rw * rh];
                    for name in inputs {
                        let layer = pick(name, &results, last_result.as_ref());
                        for i in 0..acc.len() {
                            acc[i] = src_over_u32(layer[i], acc[i]);
                        }
                    }
                    acc
                }
                SvgFePrimitive::Blend { input, in2, .. } => {
                    let a = pick(input, &results, last_result.as_ref());
                    let b = pick(in2, &results, last_result.as_ref());
                    let mut acc = vec![0u32; rw * rh];
                    for i in 0..acc.len() {
                        acc[i] = src_over_u32(a[i], b[i]);
                    }
                    acc
                }
            };
            if let Some(res_name) = prim.result_name() {
                results.push((res_name.to_string(), out.clone()));
            }
            last_result = Some(out);
        }
        // Composite the final primitive result back over the rect.
        if let Some(final_img) = last_result {
            for ry in 0..rh {
                let drow = (y0 as usize + ry) * self.width as usize + x0 as usize;
                for rx in 0..rw {
                    self.pixels[drow + rx] = final_img[ry * rw + rx];
                }
            }
        }
    }

    /// Pixel-mask everything outside `inset(top, right, bottom, left)`
    /// to fully transparent. Implemented by zeroing pixels in the
    /// margin strips (border-rect minus inset).
    pub fn clip_inset(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        top: i32,
        right: i32,
        bottom: i32,
        left: i32,
    ) {
        // Top strip.
        if top > 0 {
            self.fill_rect(x, y, w, top.min(h), Color::TRANSPARENT);
        }
        // Bottom strip.
        if bottom > 0 && h - bottom > 0 {
            self.fill_rect(x, y + h - bottom, w, bottom, Color::TRANSPARENT);
        }
        // Left strip.
        if left > 0 {
            self.fill_rect(
                x,
                y + top,
                left.min(w),
                (h - top - bottom).max(0),
                Color::TRANSPARENT,
            );
        }
        // Right strip.
        if right > 0 && w - right > 0 {
            self.fill_rect(
                x + w - right,
                y + top,
                right,
                (h - top - bottom).max(0),
                Color::TRANSPARENT,
            );
        }
    }

    /// Pixel-mask everything outside a circle. cx/cy/radius are in
    /// rect-local pixel coordinates.
    pub fn clip_circle(&mut self, x: i32, y: i32, w: i32, h: i32, cx: f32, cy: f32, radius: f32) {
        let r2 = radius * radius;
        let (x0, y0, x1, y1) = (
            x.max(0),
            y.max(0),
            (x + w).min(self.width as i32),
            (y + h).min(self.height as i32),
        );
        for py in y0..y1 {
            for px_ in x0..x1 {
                let dx = (px_ - x) as f32 - cx;
                let dy = (py - y) as f32 - cy;
                if dx * dx + dy * dy > r2 {
                    let i = (py as usize) * (self.width as usize) + px_ as usize;
                    self.pixels[i] = 0;
                }
            }
        }
    }

    /// Pixel-mask everything outside a polygon (vertex list in rect-
    /// local pixel coords). Even-odd fill rule.
    pub fn clip_polygon(&mut self, x: i32, y: i32, w: i32, h: i32, points: &[(f32, f32)]) {
        if points.len() < 3 {
            return;
        }
        let (x0, y0, x1, y1) = (
            x.max(0),
            y.max(0),
            (x + w).min(self.width as i32),
            (y + h).min(self.height as i32),
        );
        for py in y0..y1 {
            let pyf = (py - y) as f32 + 0.5;
            for px_ in x0..x1 {
                let pxf = (px_ - x) as f32 + 0.5;
                if !point_in_polygon(pxf, pyf, points) {
                    let i = (py as usize) * (self.width as usize) + px_ as usize;
                    self.pixels[i] = 0;
                }
            }
        }
    }

    pub fn stroke_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Color) {
        // 1-pixel border.
        self.fill_rect(x, y, w, 1, color);
        self.fill_rect(x, y + h - 1, w, 1, color);
        self.fill_rect(x, y, 1, h, color);
        self.fill_rect(x + w - 1, y, 1, h, color);
    }

    /// Zero out every pixel that falls outside a rounded rectangle.
    /// `r` is the corner radius in pixels (clamped to half the shortest side).
    /// Pixels outside the rounded shape are set to 0x00000000 (transparent),
    /// so this acts as an alpha-clip mask — call it after painting an image or
    /// background to get `border-radius` clipping on bitmap content.
    pub fn clip_rounded_rect(&mut self, x: i32, y: i32, w: i32, h: i32, r: i32) {
        if r <= 0 {
            return; // nothing to clip
        }
        let r = r.min(w / 2).min(h / 2);
        if r <= 0 {
            return;
        }
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.width as i32);
        let y1 = (y + h).min(self.height as i32);
        for py in y0..y1 {
            for px in x0..x1 {
                // Local coordinates relative to box origin.
                let lx = px - x;
                let ly = py - y;
                // Determine whether this pixel is inside the rounded rect.
                // Only the four corner quadrants need the circle test; the
                // middle band and the top/bottom centre band are always in.
                let in_left_col = lx < r;
                let in_right_col = lx >= w - r;
                let in_top_row = ly < r;
                let in_bot_row = ly >= h - r;
                let inside = if in_left_col && in_top_row {
                    let dx = r - 1 - lx;
                    let dy = r - 1 - ly;
                    dx * dx + dy * dy <= r * r
                } else if in_right_col && in_top_row {
                    let dx = lx - (w - r);
                    let dy = r - 1 - ly;
                    dx * dx + dy * dy <= r * r
                } else if in_left_col && in_bot_row {
                    let dx = r - 1 - lx;
                    let dy = ly - (h - r);
                    dx * dx + dy * dy <= r * r
                } else if in_right_col && in_bot_row {
                    let dx = lx - (w - r);
                    let dy = ly - (h - r);
                    dx * dx + dy * dy <= r * r
                } else {
                    true
                };
                if !inside {
                    let i = (py as usize) * (self.width as usize) + px as usize;
                    self.pixels[i] = 0;
                }
            }
        }
    }
}

/// Even-odd point-in-polygon — for clip-path polygon masking.
fn point_in_polygon(px: f32, py: f32, points: &[(f32, f32)]) -> bool {
    let n = points.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = points[i];
        let (xj, yj) = points[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi + f32::EPSILON) + xi)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// One CSS filter operation, plumbed into the rasterizer through
/// `Bitmap::apply_filter_rect`. Kept separate from `cv_css::FilterFn`
/// to avoid a dependency loop — conclave resolves CSS specs to these
/// ops at paint time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FilterOp {
    Blur(f32),
    Brightness(f32),
    Contrast(f32),
    Grayscale(f32),
    Invert(f32),
    Sepia(f32),
    Saturate(f32),
    HueRotate(f32),
    Opacity(f32),
}

/// `feComposite` operators (Filter Effects 1 §15.10 / Porter-Duff).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FeCompositeOp {
    Over,
    In,
    Out,
    Atop,
    Xor,
    Arithmetic { k1: f32, k2: f32, k3: f32, k4: f32 },
}

/// One SVG filter primitive (`feGaussianBlur`, `feColorMatrix`, …) as
/// consumed by [`Bitmap::apply_svg_filter_rect`]. `input`/`in2` name the
/// source image (`SourceGraphic`, `SourceAlpha`, a prior primitive's
/// `result`, or empty = previous primitive's output). `result` (when set)
/// publishes this primitive's output under a name for later primitives.
#[derive(Clone, Debug, PartialEq)]
pub enum SvgFePrimitive {
    /// `feGaussianBlur stdDeviation=...` — real Gaussian (3-box approx).
    GaussianBlur {
        input: String,
        std_dev: f32,
        result: Option<String>,
    },
    /// `feColorMatrix` — full 5×4 matrix (20 values, row-major, the last
    /// column is the +bias term ×255) on non-premultiplied RGBA.
    ColorMatrix {
        input: String,
        matrix: [f32; 20],
        result: Option<String>,
    },
    /// `feOffset dx dy` — translate the input image.
    Offset {
        input: String,
        dx: i32,
        dy: i32,
        result: Option<String>,
    },
    /// `feFlood flood-color` — fill the region with a solid colour.
    Flood {
        color: Color,
        result: Option<String>,
    },
    /// `feComposite` — Porter-Duff combine of `input` and `in2`.
    Composite {
        input: String,
        in2: String,
        op: FeCompositeOp,
        result: Option<String>,
    },
    /// `feBlend` — (normal-mode) source-over combine of `input`/`in2`.
    Blend {
        input: String,
        in2: String,
        result: Option<String>,
    },
    /// `feMerge` — stack a list of inputs source-over, bottom to top.
    Merge {
        inputs: Vec<String>,
        result: Option<String>,
    },
}

impl SvgFePrimitive {
    fn result_name(&self) -> Option<&str> {
        match self {
            SvgFePrimitive::GaussianBlur { result, .. }
            | SvgFePrimitive::ColorMatrix { result, .. }
            | SvgFePrimitive::Offset { result, .. }
            | SvgFePrimitive::Flood { result, .. }
            | SvgFePrimitive::Composite { result, .. }
            | SvgFePrimitive::Blend { result, .. }
            | SvgFePrimitive::Merge { result, .. } => result.as_deref(),
        }
    }
}

/// Build the canonical `feColorMatrix type="saturate"` 5×4 matrix for a
/// saturation value `s` (Filter Effects 1 §8.4). Helper for callers that
/// translate CSS filter functions into SVG primitives.
pub fn fe_saturate_matrix(s: f32) -> [f32; 20] {
    [
        0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s, 0.0, 0.0,
        0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s, 0.0, 0.0,
        0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0,
    ]
}

/// `feGaussianBlur` over an RGBA8 sub-image buffer (0xAARRGGBB), in place.
/// Uses the same spec 3-box approximation as `FilterOp::Blur`.
fn fe_gaussian_blur(buf: &mut [u32], w: usize, h: usize, std_dev: f32) {
    if std_dev <= 0.0 || w == 0 || h == 0 {
        return;
    }
    let s = std_dev.clamp(0.0, 200.0);
    let d = (s * 3.0 * (2.0 * std::f32::consts::PI).sqrt() / 4.0 + 0.5).floor() as i32;
    if d < 1 {
        return;
    }
    // Premultiplied f32 planes.
    let mut work = vec![0f32; w * h * 4];
    for i in 0..w * h {
        let p = buf[i];
        let a = ((p >> 24) & 0xFF) as f32;
        let af = a / 255.0;
        work[i * 4] = a;
        work[i * 4 + 1] = ((p >> 16) & 0xFF) as f32 * af;
        work[i * 4 + 2] = ((p >> 8) & 0xFF) as f32 * af;
        work[i * 4 + 3] = (p & 0xFF) as f32 * af;
    }
    let mut tmp = vec![0f32; w * h * 4];
    let r = d / 2;
    if d % 2 == 1 {
        box_blur_h(&mut work, &mut tmp, w, h, r, r);
        box_blur_h(&mut work, &mut tmp, w, h, r, r);
        box_blur_h(&mut work, &mut tmp, w, h, r, r);
        box_blur_v(&mut work, &mut tmp, w, h, r, r);
        box_blur_v(&mut work, &mut tmp, w, h, r, r);
        box_blur_v(&mut work, &mut tmp, w, h, r, r);
    } else {
        box_blur_h(&mut work, &mut tmp, w, h, r, r - 1);
        box_blur_h(&mut work, &mut tmp, w, h, r - 1, r);
        box_blur_h(&mut work, &mut tmp, w, h, r, r);
        box_blur_v(&mut work, &mut tmp, w, h, r, r - 1);
        box_blur_v(&mut work, &mut tmp, w, h, r - 1, r);
        box_blur_v(&mut work, &mut tmp, w, h, r, r);
    }
    for i in 0..w * h {
        let a = work[i * 4].clamp(0.0, 255.0);
        let (rr, gg, bb) = if a > 0.0 {
            let inv = 255.0 / a;
            (
                (work[i * 4 + 1] * inv).clamp(0.0, 255.0),
                (work[i * 4 + 2] * inv).clamp(0.0, 255.0),
                (work[i * 4 + 3] * inv).clamp(0.0, 255.0),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        buf[i] = ((a.round() as u32) << 24)
            | ((rr.round() as u32) << 16)
            | ((gg.round() as u32) << 8)
            | (bb.round() as u32);
    }
}

/// `feColorMatrix` — apply a 5×4 matrix to non-premultiplied RGBA
/// (Filter Effects 1 §15.10). Column 5 of each row is a bias added as a
/// fraction of full intensity (×255).
fn fe_color_matrix(buf: &[u32], m: &[f32; 20]) -> Vec<u32> {
    buf.iter()
        .map(|&p| {
            let a = ((p >> 24) & 0xFF) as f32 / 255.0;
            let r = ((p >> 16) & 0xFF) as f32 / 255.0;
            let g = ((p >> 8) & 0xFF) as f32 / 255.0;
            let b = (p & 0xFF) as f32 / 255.0;
            let nr = (m[0] * r + m[1] * g + m[2] * b + m[3] * a + m[4]).clamp(0.0, 1.0);
            let ng = (m[5] * r + m[6] * g + m[7] * b + m[8] * a + m[9]).clamp(0.0, 1.0);
            let nb = (m[10] * r + m[11] * g + m[12] * b + m[13] * a + m[14]).clamp(0.0, 1.0);
            let na = (m[15] * r + m[16] * g + m[17] * b + m[18] * a + m[19]).clamp(0.0, 1.0);
            ((((na * 255.0).round()) as u32) << 24)
                | ((((nr * 255.0).round()) as u32) << 16)
                | ((((ng * 255.0).round()) as u32) << 8)
                | (((nb * 255.0).round()) as u32)
        })
        .collect()
}

/// `feOffset` — translate the input image by (dx, dy); exposed area is
/// transparent.
fn fe_offset(buf: &[u32], w: usize, h: usize, dx: i32, dy: i32) -> Vec<u32> {
    let mut out = vec![0u32; w * h];
    for ry in 0..h as i32 {
        let sy = ry - dy;
        if sy < 0 || sy >= h as i32 {
            continue;
        }
        for rx in 0..w as i32 {
            let sx = rx - dx;
            if sx < 0 || sx >= w as i32 {
                continue;
            }
            out[ry as usize * w + rx as usize] = buf[sy as usize * w + sx as usize];
        }
    }
    out
}

/// `feComposite` — Porter-Duff / arithmetic combine of two RGBA8 images.
fn fe_composite(a: &[u32], b: &[u32], op: FeCompositeOp) -> Vec<u32> {
    a.iter()
        .zip(b.iter())
        .map(|(&pa, &pb)| {
            let (aa, ar, ag, ab) = unpack_f(pa);
            let (ba, br, bg, bb) = unpack_f(pb);
            let (fa, fb) = match op {
                FeCompositeOp::Over => (1.0, 1.0 - aa),
                FeCompositeOp::In => (ba, 0.0),
                FeCompositeOp::Out => (1.0 - ba, 0.0),
                FeCompositeOp::Atop => (ba, 1.0 - aa),
                FeCompositeOp::Xor => (1.0 - ba, 1.0 - aa),
                FeCompositeOp::Arithmetic { .. } => (0.0, 0.0),
            };
            if let FeCompositeOp::Arithmetic { k1, k2, k3, k4 } = op {
                // result = k1*i1*i2 + k2*i1 + k3*i2 + k4, channel-wise on
                // PREMULTIPLIED values (Filter Effects 1 §15.10).
                let comp = |i1: f32, i2: f32| (k1 * i1 * i2 + k2 * i1 + k3 * i2 + k4).clamp(0.0, 1.0);
                let ra = comp(aa, ba);
                let rr = comp(ar * aa, br * ba);
                let rg = comp(ag * aa, bg * ba);
                let rb = comp(ab * aa, bb * ba);
                return pack_premul(ra, rr, rg, rb);
            }
            // Premultiplied source-over family.
            let ra = aa * fa + ba * fb;
            let rr = (ar * aa) * fa + (br * ba) * fb;
            let rg = (ag * aa) * fa + (bg * ba) * fb;
            let rb = (ab * aa) * fa + (bb * ba) * fb;
            pack_premul(ra, rr, rg, rb)
        })
        .collect()
}

#[inline]
fn unpack_f(p: u32) -> (f32, f32, f32, f32) {
    (
        ((p >> 24) & 0xFF) as f32 / 255.0,
        ((p >> 16) & 0xFF) as f32 / 255.0,
        ((p >> 8) & 0xFF) as f32 / 255.0,
        (p & 0xFF) as f32 / 255.0,
    )
}

/// Pack a premultiplied (alpha, premul-r/g/b) tuple back to straight-alpha
/// 0xAARRGGBB.
#[inline]
fn pack_premul(a: f32, pr: f32, pg: f32, pb: f32) -> u32 {
    let a = a.clamp(0.0, 1.0);
    let (r, g, b) = if a > 0.0 {
        ((pr / a).clamp(0.0, 1.0), (pg / a).clamp(0.0, 1.0), (pb / a).clamp(0.0, 1.0))
    } else {
        (0.0, 0.0, 0.0)
    };
    (((a * 255.0).round() as u32) << 24)
        | (((r * 255.0).round() as u32) << 16)
        | (((g * 255.0).round() as u32) << 8)
        | ((b * 255.0).round() as u32)
}

/// Straight-alpha source-over of one 0xAARRGGBB pixel onto another.
#[inline]
fn src_over_u32(src: u32, dst: u32) -> u32 {
    let (sa, sr, sg, sb) = unpack_f(src);
    let (da, dr, dg, db) = unpack_f(dst);
    let oa = sa + da * (1.0 - sa);
    if oa <= 0.0 {
        return 0;
    }
    let or = (sr * sa + dr * da * (1.0 - sa)) / oa;
    let og = (sg * sa + dg * da * (1.0 - sa)) / oa;
    let ob = (sb * sa + db * da * (1.0 - sa)) / oa;
    (((oa * 255.0).round() as u32) << 24)
        | (((or * 255.0).round() as u32) << 16)
        | (((og * 255.0).round() as u32) << 8)
        | ((ob * 255.0).round() as u32)
}

/// Apply `f(r,g,b)` to every pixel in the rect, preserving alpha.
fn color_per_pixel(
    bmp: &mut Bitmap,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    f: impl Fn(u8, u8, u8) -> (u8, u8, u8),
) {
    for py in y0..y1 {
        let row = (py as usize) * (bmp.width as usize);
        for px_ in x0..x1 {
            let i = row + px_ as usize;
            let p = bmp.pixels[i];
            let a = (p >> 24) & 0xFF;
            let r = ((p >> 16) & 0xFF) as u8;
            let g = ((p >> 8) & 0xFF) as u8;
            let b = (p & 0xFF) as u8;
            let (nr, ng, nb) = f(r, g, b);
            bmp.pixels[i] = (a << 24) | ((nr as u32) << 16) | ((ng as u32) << 8) | (nb as u32);
        }
    }
}

/// True Gaussian blur over the rect, approximated by three successive
/// box blurs per axis (the `feGaussianBlur` reference algorithm,
/// SVG 1.1 §15.17 / Filter Effects 1 §9.14). For box size `d`:
///   - odd  d: three centred box blurs of width d.
///   - even d: a box blur of width d offset half a pixel left, one
///             offset half a pixel right, and one centred box of d+1.
///
/// All passes run on PREMULTIPLIED colour so transparent edges feather
/// to transparent without the dark fringe a straight-alpha average
/// produces — this is what Chrome/Skia do and is required for
/// drop-shadow and blurred soft edges to read correctly.
fn gaussian_box3(bmp: &mut Bitmap, x0: i32, y0: i32, x1: i32, y1: i32, d: i32) {
    let w = (x1 - x0) as usize;
    let h = (y1 - y0) as usize;
    if w == 0 || h == 0 {
        return;
    }
    // Lift the rect into a premultiplied f32 scratch buffer (RGBA planes
    // interleaved as [a, r, g, b] premultiplied: rgb already * a/255).
    let mut buf = vec![0f32; w * h * 4];
    for ry in 0..h {
        let srow = (y0 as usize + ry) * bmp.width as usize + x0 as usize;
        for rx in 0..w {
            let p = bmp.pixels[srow + rx];
            let a = ((p >> 24) & 0xFF) as f32;
            let af = a / 255.0;
            let i = (ry * w + rx) * 4;
            buf[i] = a;
            buf[i + 1] = ((p >> 16) & 0xFF) as f32 * af;
            buf[i + 2] = ((p >> 8) & 0xFF) as f32 * af;
            buf[i + 3] = (p & 0xFF) as f32 * af;
        }
    }
    let mut tmp = vec![0f32; w * h * 4];
    // Horizontal passes.
    if d % 2 == 1 {
        let r = d / 2;
        box_blur_h(&mut buf, &mut tmp, w, h, r, r);
        box_blur_h(&mut buf, &mut tmp, w, h, r, r);
        box_blur_h(&mut buf, &mut tmp, w, h, r, r);
    } else {
        let r = d / 2;
        // Two width-d boxes offset left/right, then a centred width-(d+1).
        box_blur_h(&mut buf, &mut tmp, w, h, r, r - 1);
        box_blur_h(&mut buf, &mut tmp, w, h, r - 1, r);
        box_blur_h(&mut buf, &mut tmp, w, h, r, r);
    }
    // Vertical passes.
    if d % 2 == 1 {
        let r = d / 2;
        box_blur_v(&mut buf, &mut tmp, w, h, r, r);
        box_blur_v(&mut buf, &mut tmp, w, h, r, r);
        box_blur_v(&mut buf, &mut tmp, w, h, r, r);
    } else {
        let r = d / 2;
        box_blur_v(&mut buf, &mut tmp, w, h, r, r - 1);
        box_blur_v(&mut buf, &mut tmp, w, h, r - 1, r);
        box_blur_v(&mut buf, &mut tmp, w, h, r, r);
    }
    // Un-premultiply and write back.
    for ry in 0..h {
        let drow = (y0 as usize + ry) * bmp.width as usize + x0 as usize;
        for rx in 0..w {
            let i = (ry * w + rx) * 4;
            let a = buf[i].clamp(0.0, 255.0);
            let (r, g, b) = if a > 0.0 {
                let inv = 255.0 / a;
                (
                    (buf[i + 1] * inv).clamp(0.0, 255.0),
                    (buf[i + 2] * inv).clamp(0.0, 255.0),
                    (buf[i + 3] * inv).clamp(0.0, 255.0),
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            bmp.pixels[drow + rx] = ((a.round() as u32) << 24)
                | ((r.round() as u32) << 16)
                | ((g.round() as u32) << 8)
                | (b.round() as u32);
        }
    }
}

/// One horizontal box-blur pass: each output pixel is the mean of the
/// window spanning `left` pixels to the left and `right` to the right
/// (inclusive of self). Edge samples are clamped to the rect edge
/// ("duplicate" border) so the rect interior darkens correctly toward a
/// transparent surround. Operates on the premultiplied [a,r,g,b] buffer.
fn box_blur_h(buf: &mut [f32], tmp: &mut [f32], w: usize, h: usize, left: i32, right: i32) {
    let n = (left + right + 1) as f32;
    for ry in 0..h {
        let base = ry * w * 4;
        for rx in 0..w as i32 {
            let (mut sa, mut sr, mut sg, mut sb) = (0.0, 0.0, 0.0, 0.0);
            for k in -left..=right {
                let sx = (rx + k).clamp(0, w as i32 - 1) as usize;
                let i = base + sx * 4;
                sa += buf[i];
                sr += buf[i + 1];
                sg += buf[i + 2];
                sb += buf[i + 3];
            }
            let o = base + rx as usize * 4;
            tmp[o] = sa / n;
            tmp[o + 1] = sr / n;
            tmp[o + 2] = sg / n;
            tmp[o + 3] = sb / n;
        }
    }
    buf.copy_from_slice(tmp);
}

/// One vertical box-blur pass — same as [`box_blur_h`] over columns.
fn box_blur_v(buf: &mut [f32], tmp: &mut [f32], w: usize, h: usize, up: i32, down: i32) {
    let n = (up + down + 1) as f32;
    for rx in 0..w {
        for ry in 0..h as i32 {
            let (mut sa, mut sr, mut sg, mut sb) = (0.0, 0.0, 0.0, 0.0);
            for k in -up..=down {
                let sy = (ry + k).clamp(0, h as i32 - 1) as usize;
                let i = (sy * w + rx) * 4;
                sa += buf[i];
                sr += buf[i + 1];
                sg += buf[i + 2];
                sb += buf[i + 3];
            }
            let o = (ry as usize * w + rx) * 4;
            tmp[o] = sa / n;
            tmp[o + 1] = sr / n;
            tmp[o + 2] = sg / n;
            tmp[o + 3] = sb / n;
        }
    }
    buf.copy_from_slice(tmp);
}

pub(crate) fn blend_bgra(dst: u32, src: Color) -> u32 {
    let da = (dst >> 24) & 0xFF;
    let dr = (dst >> 16) & 0xFF;
    let dg = (dst >> 8) & 0xFF;
    let db = dst & 0xFF;
    // Proper straight-alpha (non-premultiplied) Porter-Duff source-over. The old
    // form `src.rgb*sa + dst.rgb*(1-sa)` is only correct when the destination is
    // OPAQUE — it left a premultiplied RGB with a straight alpha. When the
    // destination is transparent (a canvas backing store cleared to TRANSPARENT),
    // it dragged the color toward the backdrop's zero RGB (gold 255→76), then the
    // page composite attenuated it again — making particles.js dots ~8.5x too
    // faint. Alpha-weight the destination and normalize RGB by the output alpha.
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

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    ((a as f32) * (1.0 - t) + (b as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn mix_color(a: Color, b: Color, t: f32) -> Color {
    let wa = a.a as f32 / 255.0;
    let wb = b.a as f32 / 255.0;
    let out_a = wa * (1.0 - t) + wb * t;
    if out_a <= 0.0 {
        return Color::TRANSPARENT;
    }
    let mix_premul = |ca: u8, cb: u8| -> u8 {
        (((ca as f32 * wa * (1.0 - t)) + (cb as f32 * wb * t)) / out_a)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color {
        r: mix_premul(a.r, b.r),
        g: mix_premul(a.g, b.g),
        b: mix_premul(a.b, b.b),
        a: lerp_u8(a.a, b.a, t),
    }
}

/// Half the gradient-line length projected onto a unit-axis component.
#[inline]
fn dir_scaled(component: f32, line_len: f32) -> f32 {
    component * line_len * 0.5
}

/// The [min, max] offsets covered by a resolved stop list and the span.
/// Returns (span, min). For non-repeating gradients these are unused
/// (offsets are clamped to [0,1]); for repeating they define the tile.
#[inline]
fn stop_band(stops: &[GradientStop]) -> (f32, f32) {
    if stops.is_empty() {
        return (1.0, 0.0);
    }
    let smin = stops[0].offset;
    let smax = stops[stops.len() - 1].offset;
    ((smax - smin).max(1e-4), smin)
}

/// Map a raw offset `t` into the sampling domain. Non-repeating clamps
/// to [0,1]; repeating tiles within the stop band [smin, smin+span).
/// CSS Images 3 §3.4 (repeating-*).
#[inline]
fn map_offset(t: f32, repeating: bool, span: f32, smin: f32) -> f32 {
    if !repeating {
        t
    } else {
        smin + (t - smin).rem_euclid(span)
    }
}

/// Sample resolved stops at offset `t` (clamps to ends; interpolates in
/// straight sRGB — same as the Canvas gradient path).
#[inline]
fn sample_stops_offset(stops: &[GradientStop], t: f32) -> Color {
    canvas::sample_stops(stops, t, 1.0)
}

/// Pixel-in-rounded-rect test (corner discs of radius `br`).
/// CSS Backgrounds 3 §6.1.
fn point_in_rounded(px: i32, py: i32, bx: i32, by: i32, bw: i32, bh: i32, br: i32) -> bool {
    if px < bx || py < by || px >= bx + bw || py >= by + bh {
        return false;
    }
    if br <= 0 {
        return true;
    }
    let cx = if px < bx + br {
        bx + br
    } else if px >= bx + bw - br {
        bx + bw - br - 1
    } else {
        px
    };
    let cy = if py < by + br {
        by + br
    } else if py >= by + bh - br {
        by + bh - br - 1
    } else {
        py
    };
    let dx = px - cx;
    let dy = py - cy;
    dx * dx + dy * dy <= br * br
}

/// Compute the radial-gradient ending-shape radii (rx, ry) in px.
/// CSS Images 3 §3.2.1: for `circle`, rx == ry; for `ellipse` the two
/// axes are sized independently. `closest/farthest-side` use the
/// per-axis distance from the center to the box edges;
/// `closest/farthest-corner` scale the corresponding *-side ellipse so
/// it passes through that corner (preserving the side ellipse's aspect
/// ratio); for a circle they are the straight-line distance to the
/// corner.
fn radial_radii(
    w: f32,
    h: f32,
    cx: f32,
    cy: f32,
    shape: GfxRadialShape,
    size: GfxRadialSize,
) -> (f32, f32) {
    // Distances from center to each side.
    let left = cx;
    let right = (w - cx).max(0.0);
    let top = cy;
    let bottom = (h - cy).max(0.0);
    let dx_near = left.min(right);
    let dx_far = left.max(right);
    let dy_near = top.min(bottom);
    let dy_far = top.max(bottom);
    match size {
        GfxRadialSize::ExplicitPx { rx, ry } => match shape {
            GfxRadialShape::Circle => (rx, rx),
            GfxRadialShape::Ellipse => (rx, ry),
        },
        GfxRadialSize::ClosestSide => match shape {
            // Circle radius = nearest side distance.
            GfxRadialShape::Circle => {
                let r = dx_near.min(dy_near);
                (r, r)
            }
            GfxRadialShape::Ellipse => (dx_near, dy_near),
        },
        GfxRadialSize::FarthestSide => match shape {
            GfxRadialShape::Circle => {
                let r = dx_far.max(dy_far);
                (r, r)
            }
            GfxRadialShape::Ellipse => (dx_far, dy_far),
        },
        GfxRadialSize::ClosestCorner => match shape {
            GfxRadialShape::Circle => {
                let r = (dx_near * dx_near + dy_near * dy_near).sqrt();
                (r, r)
            }
            GfxRadialShape::Ellipse => {
                // Scale the closest-side ellipse so it passes through the
                // closest corner: factor = sqrt((cx/sx)^2 + (cy/sy)^2)
                // with the closest-side axes. (CSS Images 3 §3.2.1.)
                let (sx, sy) = (dx_near.max(1e-3), dy_near.max(1e-3));
                let cdx = dx_near;
                let cdy = dy_near;
                let k = ((cdx / sx).powi(2) + (cdy / sy).powi(2)).sqrt().max(1e-3);
                (sx * k, sy * k)
            }
        },
        GfxRadialSize::FarthestCorner => match shape {
            GfxRadialShape::Circle => {
                let r = (dx_far * dx_far + dy_far * dy_far).sqrt();
                (r, r)
            }
            GfxRadialShape::Ellipse => {
                let (sx, sy) = (dx_far.max(1e-3), dy_far.max(1e-3));
                let cdx = dx_far;
                let cdy = dy_far;
                let k = ((cdx / sx).powi(2) + (cdy / sy).powi(2)).sqrt().max(1e-3);
                (sx * k, sy * k)
            }
        },
    }
}

fn sample_bgra_bilinear(src_w: u32, src_h: u32, src: &[u32], x: f32, y: f32) -> Color {
    let x0 = x.floor().clamp(0.0, (src_w - 1) as f32) as u32;
    let y0 = y.floor().clamp(0.0, (src_h - 1) as f32) as u32;
    let x1 = (x0 + 1).min(src_w - 1);
    let y1 = (y0 + 1).min(src_h - 1);
    let tx = (x - x0 as f32).clamp(0.0, 1.0);
    let ty = (y - y0 as f32).clamp(0.0, 1.0);
    let idx = |sx: u32, sy: u32| (sy as usize) * (src_w as usize) + sx as usize;
    let c00 = unpack_bgra(src[idx(x0, y0)]);
    let c10 = unpack_bgra(src[idx(x1, y0)]);
    let c01 = unpack_bgra(src[idx(x0, y1)]);
    let c11 = unpack_bgra(src[idx(x1, y1)]);
    let top = mix_color(c00, c10, tx);
    let bottom = mix_color(c01, c11, tx);
    mix_color(top, bottom, ty)
}

fn sample_bgra_box(
    src_w: u32,
    src_h: u32,
    src: &[u32],
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> Color {
    let taps_x = 4usize;
    let taps_y = 4usize;
    let mut acc_pr = 0.0f32;
    let mut acc_pg = 0.0f32;
    let mut acc_pb = 0.0f32;
    let mut acc_a = 0.0f32;
    let mut count = 0.0f32;
    for sy in 0..taps_y {
        let fy = y0 + ((sy as f32) + 0.5) * (y1 - y0) / taps_y as f32;
        for sx in 0..taps_x {
            let fx = x0 + ((sx as f32) + 0.5) * (x1 - x0) / taps_x as f32;
            let c = sample_bgra_bilinear(src_w, src_h, src, fx, fy);
            let a = c.a as f32 / 255.0;
            acc_pr += c.r as f32 * a;
            acc_pg += c.g as f32 * a;
            acc_pb += c.b as f32 * a;
            acc_a += a;
            count += 1.0;
        }
    }
    if acc_a <= 0.0 {
        return Color::TRANSPARENT;
    }
    Color {
        r: (acc_pr / acc_a).round().clamp(0.0, 255.0) as u8,
        g: (acc_pg / acc_a).round().clamp(0.0, 255.0) as u8,
        b: (acc_pb / acc_a).round().clamp(0.0, 255.0) as u8,
        a: ((acc_a / count) * 255.0).round().clamp(0.0, 255.0) as u8,
    }
}

fn unpack_bgra(px: u32) -> Color {
    Color {
        r: ((px >> 16) & 0xFF) as u8,
        g: ((px >> 8) & 0xFF) as u8,
        b: (px & 0xFF) as u8,
        a: ((px >> 24) & 0xFF) as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_text_clips_to_glyph_mask_and_ramps() {
        // background-clip:text idiom: a horizontal red→blue gradient
        // clipped to a glyph coverage mask. The mask is a 20-px-wide run
        // with TWO solid "glyph" columns (full coverage at x=2..6 and
        // x=14..18) and transparent gaps between them.
        let w = 20i32;
        let h = 6i32;
        let mut mask = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let on = (2..6).contains(&x) || (14..18).contains(&x);
                if on {
                    mask[(y * w + x) as usize] = 255;
                }
            }
        }
        let mut bmp = Bitmap::new(w as u32, h as u32);
        bmp.clear(Color { r: 0, g: 0, b: 0, a: 0 });
        // 90° = gradient runs left→right (CSS: 90deg points right).
        let stops = [
            AbstractStop { color: Color { r: 255, g: 0, b: 0, a: 255 }, pos_frac: Some(0.0), pos_px: None },
            AbstractStop { color: Color { r: 0, g: 0, b: 255, a: 255 }, pos_frac: Some(1.0), pos_px: None },
        ];
        bmp.blit_text_run_gradient(0, 0, w, h, &mask, w, h, 90.0, &stops, false, 255, None);

        let at = |x: i32, y: i32| Color::from_bgra_u32(bmp.pixels[(y * w + x) as usize]);
        // Pixel inside the LEFT glyph (x≈3) is mostly red (near start).
        let left = at(3, 3);
        assert!(left.a == 255, "left glyph pixel must be opaque, got a={}", left.a);
        assert!(left.r > 150 && left.b < 105, "left glyph near gradient start = red-ish, got {:?}", left);
        // Pixel inside the RIGHT glyph (x≈16) is mostly blue (near end).
        let right = at(16, 3);
        assert!(right.a == 255, "right glyph pixel must be opaque, got a={}", right.a);
        assert!(right.b > 150 && right.r < 105, "right glyph near gradient end = blue-ish, got {:?}", right);
        // BETWEEN the glyphs (x=10) the mask is 0 → pixel stays fully
        // transparent. This is the whole point: NOT a solid fill block.
        let gap = at(10, 3);
        assert_eq!(gap.a, 0, "between-glyph gap must be transparent (a=0), got {:?}", gap);
    }

    #[test]
    fn gradient_text_global_alpha_scales_coverage() {
        // Half global alpha halves the resulting pixel alpha (group opacity).
        let w = 4i32;
        let h = 2i32;
        let mask = vec![255u8; (w * h) as usize];
        let mut bmp = Bitmap::new(w as u32, h as u32);
        bmp.clear(Color { r: 0, g: 0, b: 0, a: 0 });
        let stops = [
            AbstractStop { color: Color { r: 255, g: 0, b: 0, a: 255 }, pos_frac: Some(0.0), pos_px: None },
            AbstractStop { color: Color { r: 255, g: 0, b: 0, a: 255 }, pos_frac: Some(1.0), pos_px: None },
        ];
        bmp.blit_text_run_gradient(0, 0, w, h, &mask, w, h, 90.0, &stops, false, 128, None);
        let p = Color::from_bgra_u32(bmp.pixels[0]);
        // 255 * 255 * 128 / (255*255) = 128.
        assert_eq!(p.a, 128, "global_alpha=128 over full coverage → a=128, got {}", p.a);
    }

    #[test]
    fn clear_sets_all_pixels() {
        let mut b = Bitmap::new(4, 4);
        b.clear(Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        });
        assert!(b.pixels.iter().all(|&p| p == 0xFFFF0000));
    }

    #[test]
    fn blit_affine_translates_opaque_pixels() {
        // A 4×4 opaque-red patch on a transparent source, translated by
        // (10, 5) via an identity-rotation matrix, lands at (10, 5).
        let mut src = Bitmap::new(8, 8);
        src.clear(Color {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        });
        src.fill_rect(
            0,
            0,
            4,
            4,
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        let mut dst = Bitmap::new(40, 40);
        dst.blit_affine(&src, [1.0, 0.0, 0.0, 1.0, 10.0, 5.0], 1.0);
        let p = dst.pixels[7 * 40 + 12]; // (x=12, y=7) — inside the patch
        assert_eq!((p >> 16) & 0xFF, 255, "translated patch should be red");
        // A pixel outside the patch stays the original white.
        assert_eq!(dst.pixels[30 * 40 + 30], 0xFFFFFFFF);
    }

    #[test]
    fn blit_affine_rotates_90_degrees() {
        // 90° clockwise rotation: m = [0,1,-1,0, e, f]. The src 4×4 red
        // square at (0..4, 0..4) maps to dest x∈[e-4,e], y∈[f,f+4].
        let mut src = Bitmap::new(8, 8);
        src.clear(Color {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        });
        src.fill_rect(
            0,
            0,
            4,
            4,
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        let mut dst = Bitmap::new(40, 40);
        dst.blit_affine(&src, [0.0, 1.0, -1.0, 0.0, 10.0, 10.0], 1.0);
        // (8,12) maps back to src (2.5, 1.5) — inside the red square.
        let p = dst.pixels[12 * 40 + 8];
        assert_eq!((p >> 16) & 0xFF, 255, "rotated square should cover (8,12)");
        // (12,12) maps back to src (2.5, 2.5)? dest x=12 -> rx=2.5 -> outside
        // x range [6,10], so it must be untouched white.
        assert_eq!(dst.pixels[12 * 40 + 14], 0xFFFFFFFF);
    }

    #[test]
    fn fill_rect_rounded_clears_corner_pixel() {
        // A 10×10 rect filled with a radius-3 corner should leave the
        // very corner pixel (0,0) untouched while filling the centre.
        let mut b = Bitmap::new(10, 10);
        b.clear(Color::WHITE);
        b.fill_rect_rounded(0, 0, 10, 10, 3, Color::BLACK);
        assert_eq!(
            b.pixels[0],
            Color::WHITE.to_bgra_u32(),
            "corner pixel should stay white"
        );
        // Centre pixel filled.
        let centre_idx = 5 * 10 + 5;
        assert_eq!(
            b.pixels[centre_idx],
            Color::BLACK.to_bgra_u32(),
            "centre should be black"
        );
    }

    #[test]
    fn fill_rect_clips() {
        let mut b = Bitmap::new(10, 10);
        b.clear(Color::WHITE);
        b.fill_rect(-5, -5, 8, 8, Color::BLACK); // partial overlap top-left
        assert_eq!(b.pixels[0], Color::BLACK.to_bgra_u32());
        assert_eq!(b.pixels[3], Color::WHITE.to_bgra_u32()); // x=3 outside the filled portion
    }

    #[test]
    fn blit_bgra_sprite_clips_to_box_and_offsets() {
        // A 4x4 source where row 2 is red, everything else green. Place it
        // in a 10x10 bitmap, cropped to a 4x1 box at (0,0) with a -2 y
        // offset — that selects source row 2 (red) into the box, and
        // nothing outside the box should change.
        let mut b = Bitmap::new(10, 10);
        b.clear(Color::WHITE);
        let green = Color {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        }
        .to_bgra_u32();
        let red = Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        }
        .to_bgra_u32();
        let mut src = vec![green; 16]; // 4x4
        for x in 0..4 {
            src[2 * 4 + x] = red; // row 2 red
        }
        // box at (0,0) size 4x1, image shifted up by 2 → box row shows src row 2.
        b.blit_bgra_sprite(0, 0, 4, 1, 0, -2, 4, 4, &src);
        // Box row (y=0) is red.
        for x in 0..4 {
            assert_eq!(b.pixels[x], red, "box pixel x={x} should be red");
        }
        // Outside the 4x1 box stays white (clipped) — e.g. (0,1) and (5,0).
        assert_eq!(b.pixels[10], Color::WHITE.to_bgra_u32());
        assert_eq!(b.pixels[5], Color::WHITE.to_bgra_u32());
    }

    #[test]
    fn stroke_rect_paints_edges() {
        let mut b = Bitmap::new(5, 5);
        b.clear(Color::WHITE);
        b.stroke_rect(0, 0, 5, 5, Color::BLACK);
        // Corners are black.
        assert_eq!(b.pixels[0], Color::BLACK.to_bgra_u32());
        assert_eq!(b.pixels[4], Color::BLACK.to_bgra_u32());
        // Center is white.
        assert_eq!(b.pixels[2 * 5 + 2], Color::WHITE.to_bgra_u32());
    }

    #[test]
    fn blit_mask_tinted_scaled_uses_source_alpha_as_stencil() {
        let mut b = Bitmap::new(2, 1);
        b.clear(Color::WHITE);
        let src = [0xFF000000u32, 0x00000000u32];
        b.blit_mask_tinted_scaled(
            0,
            0,
            2,
            1,
            2,
            1,
            &src,
            Color {
                r: 0,
                g: 200,
                b: 0,
                a: 255,
            },
        );
        let left = b.pixels[0];
        let right = b.pixels[1];
        assert_ne!(left, Color::WHITE.to_bgra_u32());
        assert_eq!(right, Color::WHITE.to_bgra_u32());
    }

    #[test]
    fn blit_bgra_blends_semitransparent_pixels() {
        let mut b = Bitmap::new(1, 1);
        b.clear(Color::WHITE);
        let src = [0x800000FFu32];
        b.blit_bgra(0, 0, 1, 1, &src);
        let px = b.pixels[0];
        let r = (px >> 16) & 0xFF;
        let g = (px >> 8) & 0xFF;
        let bch = px & 0xFF;
        assert!(r > 120 && r < 255, "red should blend, got {}", r);
        assert!(g > 120 && g < 255, "green should blend, got {}", g);
        assert_eq!(bch, 255, "blue should stay saturated, got {}", bch);
    }

    #[test]
    fn blit_bgra_scaled_blends_semitransparent_pixels() {
        let mut b = Bitmap::new(2, 2);
        b.clear(Color::WHITE);
        let src = [0x8000FF00u32];
        b.blit_bgra_scaled(0, 0, 2, 2, 1, 1, &src);
        let px = b.pixels[0];
        let r = (px >> 16) & 0xFF;
        let g = (px >> 8) & 0xFF;
        let bch = px & 0xFF;
        assert!(r > 120 && r < 255, "red should blend, got {}", r);
        assert_eq!(g, 255, "green should stay saturated, got {}", g);
        assert!(bch > 120 && bch < 255, "blue should blend, got {}", bch);
    }

    #[test]
    fn blit_bgra_scaled_down_does_not_bleed_transparent_rgb() {
        let mut b = Bitmap::new(1, 1);
        b.clear(Color::TRANSPARENT);
        let transparent_gold = Color {
            r: 255,
            g: 215,
            b: 0,
            a: 0,
        }
        .to_bgra_u32();
        let opaque_black = Color {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        }
        .to_bgra_u32();
        let src = [transparent_gold, opaque_black];

        b.blit_bgra_scaled(0, 0, 1, 1, 2, 1, &src);

        let c = unpack_bgra(b.pixels[0]);
        assert!(
            c.a > 0,
            "the opaque source pixel should contribute visible alpha"
        );
        assert_eq!(
            (c.r, c.g, c.b),
            (0, 0, 0),
            "transparent source RGB must not tint the downsampled pixel"
        );
    }

    #[test]
    fn blit_bgra_scaled_preserves_centered_detail_when_downscaling() {
        let mut b = Bitmap::new(4, 4);
        b.clear(Color::BLACK);
        let mut src = vec![Color::BLACK.to_bgra_u32(); 16 * 16];
        for y in 6..10 {
            for x in 6..10 {
                src[y * 16 + x] = Color {
                    r: 255,
                    g: 215,
                    b: 0,
                    a: 255,
                }
                .to_bgra_u32();
            }
        }
        b.blit_bgra_scaled(0, 0, 4, 4, 16, 16, &src);
        let brightest = b
            .pixels
            .iter()
            .copied()
            .map(unpack_bgra)
            .max_by_key(|c| c.r as u16 + c.g as u16 + c.b as u16)
            .expect("downscaled pixels");
        assert!(
            brightest.r > 90 && brightest.g > 75,
            "centered bright detail should survive aggressive downscaling, got {:?}",
            brightest
        );
    }

    #[test]
    fn fill_ellipse_ring_top_paints_arc_not_bottom_half() {
        let mut b = Bitmap::new(20, 20);
        b.clear(Color::TRANSPARENT);

        b.fill_ellipse_ring_top(2, 2, 16, 16, 2, Color::WHITE);

        let top = unpack_bgra(b.pixels[3 * 20 + 10]);
        let bottom = unpack_bgra(b.pixels[16 * 20 + 10]);
        assert_eq!(top.a, 255, "top arc should be painted");
        assert_eq!(bottom.a, 0, "bottom half should be untouched");
    }

    // ───────────────────────── CSS blend modes ─────────────────────────────

    /// `BlendMode::Normal` through `blend_mode_pixel` is byte-identical to the
    /// legacy `blend_bgra` source-over — the load-bearing "don't regress" gate.
    #[test]
    fn blend_mode_normal_is_source_over() {
        let dst = Color { r: 10, g: 20, b: 30, a: 200 }.to_bgra_u32();
        for src in [
            Color { r: 255, g: 0, b: 0, a: 255 },
            Color { r: 100, g: 150, b: 200, a: 128 },
        ] {
            let via = blend_mode_pixel(dst, src, BlendMode::Normal);
            let legacy = if src.a == 255 { src.to_bgra_u32() } else { blend_bgra(dst, src) };
            assert_eq!(via, legacy, "Normal must equal source-over for {src:?}");
        }
    }

    /// mix-blend-mode multiply: 0.5-gray element over 0.5-gray backdrop = 0.25.
    #[test]
    fn blend_mode_multiply_half_gray() {
        let dst = Color { r: 128, g: 128, b: 128, a: 255 }.to_bgra_u32();
        let src = Color { r: 128, g: 128, b: 128, a: 255 };
        let out = Color::from_bgra_u32(blend_mode_pixel(dst, src, BlendMode::Multiply));
        assert!((out.r as i32 - 64).abs() <= 1, "multiply 0.25 -> ~64 got {}", out.r);
    }

    /// `blit_layer_blend` composites an element layer over the backdrop with the
    /// blend formula. A multiply layer of mid gray over a mid-gray page must
    /// darken to ~0.25 where the layer is opaque, and leave transparent layer
    /// pixels untouched.
    #[test]
    fn blit_layer_blend_multiply_darkens_only_covered() {
        let mut page = Bitmap::new(4, 4);
        page.clear(Color { r: 128, g: 128, b: 128, a: 255 });
        // A 2x2 opaque mid-gray layer at (1,1), rest transparent.
        let mut layer = Bitmap::new(4, 4);
        layer.clear(Color::TRANSPARENT);
        for yy in 1..3 {
            for xx in 1..3 {
                layer.pixels[yy * 4 + xx] =
                    Color { r: 128, g: 128, b: 128, a: 255 }.to_bgra_u32();
            }
        }
        page.blit_layer_blend(0, 0, &layer, BlendMode::Multiply, 1.0);
        // Covered pixel darkened to ~64.
        let covered = Color::from_bgra_u32(page.pixels[1 * 4 + 1]);
        assert!((covered.r as i32 - 64).abs() <= 1, "covered multiply ~64 got {}", covered.r);
        // Uncovered pixel unchanged at 128.
        let uncovered = Color::from_bgra_u32(page.pixels[0]);
        assert_eq!(uncovered.r, 128, "transparent layer pixel leaves backdrop");
    }

    /// difference of an element over an equal backdrop is black (the classic
    /// "invert against identical content" check) through the layer path.
    #[test]
    fn blit_layer_blend_difference_equal_is_black() {
        let mut page = Bitmap::new(2, 2);
        page.clear(Color { r: 200, g: 100, b: 50, a: 255 });
        let mut layer = Bitmap::new(2, 2);
        layer.clear(Color { r: 200, g: 100, b: 50, a: 255 });
        page.blit_layer_blend(0, 0, &layer, BlendMode::Difference, 1.0);
        let out = Color::from_bgra_u32(page.pixels[0]);
        assert_eq!((out.r, out.g, out.b), (0, 0, 0), "difference of equal = black");
    }

    // ───────────────────────── gradient rasterizer ─────────────────────────

    fn s(r: u8, g: u8, b: u8, pos: Option<f32>) -> AbstractStop {
        AbstractStop {
            color: Color { r, g, b, a: 255 },
            pos_frac: pos,
            pos_px: None,
        }
    }
    fn at(b: &Bitmap, x: i32, y: i32) -> Color {
        unpack_bgra(b.pixels[(y as usize) * (b.width as usize) + x as usize])
    }

    /// CSS Images 3 §3.4.3 color-stop fix-up: first→0, last→1, an
    /// unspecified middle stop is evenly distributed.
    #[test]
    fn stop_fixup_distributes_unspecified_positions() {
        let stops = vec![
            s(255, 0, 0, None),
            s(0, 255, 0, None),
            s(0, 0, 255, None),
        ];
        let r = resolve_gradient_stops(&stops, 100.0);
        assert_eq!(r.len(), 3);
        assert!((r[0].offset - 0.0).abs() < 1e-4);
        assert!((r[1].offset - 0.5).abs() < 1e-4, "middle → 0.5, got {}", r[1].offset);
        assert!((r[2].offset - 1.0).abs() < 1e-4);
    }

    /// Non-monotonic positions clamp to the running max (CSS Images 3
    /// §3.4.3 step 3).
    #[test]
    fn stop_fixup_clamps_non_monotonic() {
        let stops = vec![
            s(0, 0, 0, Some(0.6)),
            s(0, 0, 0, Some(0.2)), // less than previous → clamps to 0.6
            s(0, 0, 0, Some(1.0)),
        ];
        let r = resolve_gradient_stops(&stops, 100.0);
        assert!((r[1].offset - 0.6).abs() < 1e-4, "clamped to 0.6, got {}", r[1].offset);
    }

    /// A 3-stop linear gradient (red 0%, green 50%, blue 100%) sampled
    /// at the geometric midpoint must be GREEN — proving N-stop
    /// interpolation, not a red→blue 2-stop blend (which would be gray).
    #[test]
    fn linear_three_stop_midpoint_is_green() {
        let mut b = Bitmap::new(101, 11);
        b.clear(Color::TRANSPARENT);
        let stops = vec![
            s(255, 0, 0, Some(0.0)),
            s(0, 255, 0, Some(0.5)),
            s(0, 0, 255, Some(1.0)),
        ];
        // `to right` == 90deg.
        b.fill_rect_linear_gradient_stops(0, 0, 101, 11, 90.0, &stops, false, 0);
        let mid = at(&b, 50, 5);
        assert!(
            mid.g > 230 && mid.r < 25 && mid.b < 25,
            "3-stop midpoint must be green, got {:?}",
            mid
        );
        // Endpoints.
        let left = at(&b, 0, 5);
        let right = at(&b, 100, 5);
        assert!(left.r > 230 && left.g < 25, "left = red, got {:?}", left);
        assert!(right.b > 230 && right.g < 25, "right = blue, got {:?}", right);
    }

    /// `to right` (side keyword) and `90deg` must produce the same
    /// gradient. CSS Images 3 §3.1.
    #[test]
    fn side_to_right_equals_90deg() {
        let stops = vec![s(255, 0, 0, Some(0.0)), s(0, 0, 255, Some(1.0))];
        let mut a = Bitmap::new(50, 10);
        let mut c = Bitmap::new(50, 10);
        // `to right` maps to 90deg in the direction parser; both call here
        // with 90deg, so verify a vertical scan is constant (axis is
        // horizontal) — i.e. the projection only depends on x.
        a.fill_rect_linear_gradient_stops(0, 0, 50, 10, 90.0, &stops, false, 0);
        c.fill_rect_linear_gradient_stops(0, 0, 50, 10, 90.0, &stops, false, 0);
        for y in 0..10 {
            assert_eq!(at(&a, 25, y).to_bgra_u32(), at(&a, 25, 0).to_bgra_u32(),
                "90deg gradient must be constant down a column");
        }
        assert_eq!(a.pixels, c.pixels);
    }

    /// A 3-stop radial gradient: the mid stop color appears at the mid
    /// radius (halfway from center to edge), not at the center or edge.
    #[test]
    fn radial_three_stop_mid_radius_is_green() {
        let mut b = Bitmap::new(101, 101);
        b.clear(Color::TRANSPARENT);
        let stops = vec![
            s(255, 0, 0, Some(0.0)),
            s(0, 255, 0, Some(0.5)),
            s(0, 0, 255, Some(1.0)),
        ];
        // Ellipse, farthest-side: rx=ry=50 (center 50,50). Mid radius
        // (t=0.5) is 25px from center → pixel (75, 50).
        b.fill_rect_radial_gradient_stops(
            0, 0, 101, 101, 50.0, 50.0,
            GfxRadialShape::Circle, GfxRadialSize::FarthestSide,
            &stops, false, 0,
        );
        let center = at(&b, 50, 50);
        let mid = at(&b, 75, 50);
        let edge = at(&b, 100, 50);
        assert!(center.r > 230 && center.g < 25, "center = red, got {:?}", center);
        assert!(mid.g > 220 && mid.r < 40 && mid.b < 40, "mid radius = green, got {:?}", mid);
        assert!(edge.b > 220, "edge = blue, got {:?}", edge);
    }

    /// A conic gradient with red at 0deg (top) and blue at 180deg
    /// (bottom): the pixel directly to the RIGHT of center is at 90deg
    /// and must be the halfway color (purple-ish: r≈b), proving angular
    /// sampling. CSS Images 4 §3.3.
    #[test]
    fn conic_samples_by_angle() {
        let mut b = Bitmap::new(101, 101);
        b.clear(Color::TRANSPARENT);
        // red at 0 (top), blue at 0.5turn (bottom), red again at 1turn.
        let stops = vec![
            s(255, 0, 0, Some(0.0)),
            s(0, 0, 255, Some(0.5)),
            s(255, 0, 0, Some(1.0)),
        ];
        b.fill_rect_conic_gradient_stops(0, 0, 101, 101, 50.0, 50.0, 0.0, &stops, false, 0);
        // Straight up (top) = 0deg → red.
        let top = at(&b, 50, 5);
        assert!(top.r > 200 && top.b < 60, "top (0deg) = red, got {:?}", top);
        // Straight down (bottom) = 180deg → blue.
        let bottom = at(&b, 50, 95);
        assert!(bottom.b > 200 && bottom.r < 60, "bottom (180deg) = blue, got {:?}", bottom);
        // Right (90deg) = halfway 0→0.5 → purple (r≈b, both ~128).
        let right = at(&b, 95, 50);
        assert!(
            right.r > 90 && right.b > 90 && (right.r as i32 - right.b as i32).abs() < 50,
            "right (90deg) = halfway purple, got {:?}",
            right
        );
    }

    /// `from <angle>` rotates the conic start: with from=90deg, the
    /// pixel to the RIGHT (90deg) becomes the 0deg stop color (red).
    #[test]
    fn conic_from_angle_rotates_start() {
        let mut b = Bitmap::new(101, 101);
        b.clear(Color::TRANSPARENT);
        let stops = vec![s(255, 0, 0, Some(0.0)), s(0, 0, 255, Some(1.0))];
        b.fill_rect_conic_gradient_stops(0, 0, 101, 101, 50.0, 50.0, 90.0, &stops, false, 0);
        let right = at(&b, 95, 50);
        assert!(right.r > 200 && right.b < 60, "right at from=90deg = red, got {:?}", right);
    }

    /// repeating-linear-gradient with a 0..0.25 band tiles: the same
    /// color recurs every quarter of the gradient line.
    #[test]
    fn repeating_linear_tiles_band() {
        let mut b = Bitmap::new(101, 5);
        b.clear(Color::TRANSPARENT);
        // band [0, 0.5]: red→blue, repeats twice across the box.
        let stops = vec![s(255, 0, 0, Some(0.0)), s(0, 0, 255, Some(0.5))];
        b.fill_rect_linear_gradient_stops(0, 0, 101, 5, 90.0, &stops, true, 0);
        // Start of each tile (t=0, t=0.5 of the line) is red.
        let tile0 = at(&b, 0, 2);
        let tile1 = at(&b, 50, 2);
        assert!(tile0.r > 220 && tile0.b < 40, "tile0 start = red, got {:?}", tile0);
        assert!(tile1.r > 200 && tile1.b < 70, "tile1 start (repeat) = red-ish, got {:?}", tile1);
        // Non-repeating with the SAME band would clamp to blue past 0.5,
        // so verify the repeat actually brought red back at x=50.
        let mut nb = Bitmap::new(101, 5);
        nb.fill_rect_linear_gradient_stops(0, 0, 101, 5, 90.0, &stops, false, 0);
        let nclamp = at(&nb, 50, 2);
        assert!(nclamp.b > 220, "non-repeating clamps to blue at x=50, got {:?}", nclamp);
    }

    /// Two-stop full-gradient path must still match the simple expected
    /// midpoint (no regression vs. the legacy 2-stop visual).
    #[test]
    fn two_stop_linear_midpoint_is_blend() {
        let mut b = Bitmap::new(101, 5);
        let stops = vec![s(0, 0, 0, Some(0.0)), s(255, 255, 255, Some(1.0))];
        b.fill_rect_linear_gradient_stops(0, 0, 101, 5, 90.0, &stops, false, 0);
        let mid = at(&b, 50, 2);
        assert!(
            (mid.r as i32 - 127).abs() < 12,
            "2-stop black→white midpoint ≈ gray 127, got {:?}",
            mid
        );
    }

    /// closest-side vs farthest-side sizing differ for an off-center
    /// radial: closest-side reaches the near edge sooner.
    #[test]
    fn radial_closest_side_smaller_than_farthest() {
        // Center near the left edge (cx=20) of a 100-wide box.
        let stops = vec![s(255, 255, 255, Some(0.0)), s(0, 0, 0, Some(1.0))];
        let (rx_c, _) = radial_radii(100.0, 100.0, 20.0, 50.0,
            GfxRadialShape::Circle, GfxRadialSize::ClosestSide);
        let (rx_f, _) = radial_radii(100.0, 100.0, 20.0, 50.0,
            GfxRadialShape::Circle, GfxRadialSize::FarthestSide);
        // closest side = min(20, 80, 50, 50) = 20; farthest = max(...) = 80.
        assert!((rx_c - 20.0).abs() < 1e-3, "closest-side circle r=20, got {}", rx_c);
        assert!((rx_f - 80.0).abs() < 1e-3, "farthest-side circle r=80, got {}", rx_f);
        let _ = stops;
    }

    /// Explicit-px radial radii pass through verbatim.
    #[test]
    fn radial_explicit_px_radii() {
        let (rx, ry) = radial_radii(200.0, 100.0, 100.0, 50.0,
            GfxRadialShape::Ellipse, GfxRadialSize::ExplicitPx { rx: 30.0, ry: 70.0 });
        assert!((rx - 30.0).abs() < 1e-3 && (ry - 70.0).abs() < 1e-3);
    }

    // ──────────────── projective (perspective) quad warp ────────────────

    /// Solid 16×16 red source on transparent — corners TL,TR,BR,BL.
    fn red_src() -> Bitmap {
        let mut src = Bitmap::new(16, 16);
        src.clear(Color { r: 0, g: 0, b: 0, a: 0 });
        src.fill_rect(0, 0, 16, 16, Color { r: 255, g: 0, b: 0, a: 255 });
        src
    }

    #[test]
    fn quad_identity_maps_source_in_place() {
        // Mapping the unit square to a rect at (10,10)-(26,26) reproduces
        // the source there; outside stays white. (Identity-style sanity.)
        let src = red_src();
        let mut dst = Bitmap::new(40, 40);
        dst.blit_quad_projective(
            &src,
            [(10.0, 10.0), (26.0, 10.0), (26.0, 26.0), (10.0, 26.0)],
            1.0,
        );
        let inside = dst.pixels[18 * 40 + 18];
        assert_eq!((inside >> 16) & 0xFF, 255, "centre of quad is red");
        assert_eq!((inside >> 24) & 0xFF, 255, "opaque");
        assert_eq!(dst.pixels[2 * 40 + 2], 0xFFFFFFFF, "outside stays white");
    }

    #[test]
    fn quad_foreshortening_collapses_height_toward_an_edge() {
        // A trapezoid where the TOP edge is full width but the BOTTOM edge
        // collapses to a point at the centre — like rotateX viewed under
        // perspective. Pixels near the top row are covered; pixels near the
        // (collapsed) bottom are sparse. Assert: top-centre is red, and the
        // bottom corners (outside the trapezoid) are NOT red.
        let src = red_src();
        let mut dst = Bitmap::new(60, 60);
        // TL(10,10) TR(50,10) BR(30,40) BL(30,40) — bottom collapsed to (30,40).
        dst.blit_quad_projective(
            &src,
            [(10.0, 10.0), (50.0, 10.0), (30.0, 40.0), (30.0, 40.0)],
            1.0,
        );
        // Top edge centre — inside.
        let top = dst.pixels[12 * 60 + 30];
        assert_eq!((top >> 16) & 0xFF, 255, "top of trapezoid is red");
        // Bottom-left CORNER of the bounding box (10,40) is OUTSIDE the
        // collapsed trapezoid → stays white (height foreshortened away).
        assert_eq!(dst.pixels[39 * 60 + 11], 0xFFFFFFFF, "collapsed corner not painted");
    }

    #[test]
    fn quad_half_width_samples_whole_source() {
        // Map the source onto a 8-wide × 16-tall rect: the FULL 16px-wide
        // source is squeezed into 8 dest px (downscale). The whole dest
        // rect is covered (red), and one column outside is untouched.
        let src = red_src();
        let mut dst = Bitmap::new(40, 40);
        dst.blit_quad_projective(
            &src,
            [(5.0, 5.0), (13.0, 5.0), (13.0, 21.0), (5.0, 21.0)],
            1.0,
        );
        // Sample several points across the squeezed width — all red.
        for x in [6, 9, 12] {
            let p = dst.pixels[12 * 40 + x];
            assert_eq!((p >> 16) & 0xFF, 255, "squeezed quad covers x={x}");
        }
        assert_eq!(dst.pixels[12 * 40 + 20], 0xFFFFFFFF, "outside the quad is white");
    }

    #[test]
    fn quad_group_alpha_scales_coverage() {
        // group_alpha=0.5 halves the source alpha at composite, so a red
        // source over white yields a pink (blend of red over white at 50%).
        let src = red_src();
        let mut dst = Bitmap::new(20, 20);
        dst.blit_quad_projective(
            &src,
            [(2.0, 2.0), (18.0, 2.0), (18.0, 18.0), (2.0, 18.0)],
            0.5,
        );
        let p = dst.pixels[10 * 20 + 10];
        let r = (p >> 16) & 0xFF;
        let g = (p >> 8) & 0xFF;
        // Red over white at 0.5: R stays ~255, G/B ~127 (pink), not 0 or 255.
        assert!(r > 200, "red channel high: {r}");
        assert!(g > 100 && g < 160, "green channel ~half (pink): {g}");
    }

    // ───────────────────────── CSS filter raster ─────────────────────────

    fn opaque(r: u8, g: u8, b: u8) -> u32 {
        Color { r, g, b, a: 255 }.to_bgra_u32()
    }

    /// `blur(r)` spreads a hard black/white edge into a falloff band: the
    /// pixel one step into the white side picks up some black (and vice
    /// versa), and the column right at the seam is a true mid-grey — proving
    /// a real Gaussian, not a no-op passthrough.
    #[test]
    fn blur_spreads_hard_edge_into_falloff_band() {
        let mut b = Bitmap::new(40, 8);
        b.clear(Color::WHITE);
        // Left half black, right half white.
        b.fill_rect(0, 0, 20, 8, Color::BLACK);
        // Sanity: hard edge before blur.
        assert_eq!(at(&b, 18, 4).r, 0, "left is black pre-blur");
        assert_eq!(at(&b, 21, 4).r, 255, "right is white pre-blur");
        b.apply_filter_rect(0, 0, 40, 8, FilterOp::Blur(3.0));
        // The seam column blends to a mid grey (strictly between 0 and 255).
        let seam = at(&b, 19, 4).r;
        assert!(seam > 30 && seam < 225, "seam is mid-grey, got {seam}");
        // Falloff reaches several px past the seam into the white side.
        let into_white = at(&b, 22, 4).r;
        assert!(into_white < 255 && into_white > seam, "white side darkened: {into_white}");
        // And the black side lightened.
        let into_black = at(&b, 17, 4).r;
        assert!(into_black > 0 && into_black < seam, "black side lightened: {into_black}");
    }

    /// `blur` of an opaque rect on a transparent canvas must NOT leave a dark
    /// fringe (the classic straight-alpha bug). The feathered edge of a white
    /// rect blurred over transparency stays neutral grey, not muddy/dark.
    #[test]
    fn blur_premultiplied_no_dark_fringe() {
        let mut b = Bitmap::new(40, 40);
        b.clear(Color::TRANSPARENT);
        b.fill_rect(10, 10, 20, 20, Color::WHITE);
        b.apply_filter_rect(0, 0, 40, 40, FilterOp::Blur(4.0));
        // Sample a pixel just OUTSIDE the original rect edge — it's now
        // partially covered by the feather. Its RGB (where alpha>0) must be
        // ~white, never dark, because we blur premultiplied.
        let p = at(&b, 8, 20);
        assert!(p.a > 0 && p.a < 255, "feather is partial alpha: {:?}", p);
        assert!(p.r > 200 && p.g > 200 && p.b > 200, "no dark fringe: {:?}", p);
    }

    /// `grayscale(1)` makes a coloured pixel grey: R==G==B at the luma value.
    #[test]
    fn grayscale_full_makes_gray_at_luma() {
        let mut b = Bitmap::new(4, 4);
        b.clear(Color { r: 200, g: 50, b: 50, a: 255 });
        b.apply_filter_rect(0, 0, 4, 4, FilterOp::Grayscale(1.0));
        let p = at(&b, 1, 1);
        assert_eq!(p.r, p.g, "R==G after grayscale: {:?}", p);
        assert_eq!(p.g, p.b, "G==B after grayscale: {:?}", p);
        // luma = 0.2126*200 + 0.7152*50 + 0.0722*50 ≈ 81.
        assert!((p.r as i32 - 81).abs() <= 2, "luma ≈81, got {}", p.r);
    }

    /// `brightness(2)` doubles RGB (clamped to 255).
    #[test]
    fn brightness_two_doubles_clamped() {
        let mut b = Bitmap::new(4, 4);
        b.clear(Color { r: 100, g: 200, b: 10, a: 255 });
        b.apply_filter_rect(0, 0, 4, 4, FilterOp::Brightness(2.0));
        let p = at(&b, 1, 1);
        assert_eq!(p.r, 200, "100*2=200");
        assert_eq!(p.g, 255, "200*2 clamps to 255");
        assert_eq!(p.b, 20, "10*2=20");
    }

    /// `invert(1)` flips each channel: 0→255, 255→0, 40→215.
    #[test]
    fn invert_full_flips_channels() {
        let mut b = Bitmap::new(4, 4);
        b.clear(Color { r: 40, g: 0, b: 255, a: 255 });
        b.apply_filter_rect(0, 0, 4, 4, FilterOp::Invert(1.0));
        let p = at(&b, 1, 1);
        assert_eq!((p.r, p.g, p.b), (215, 255, 0), "inverted: {:?}", p);
    }

    /// `saturate(0)` collapses a colour to its luma grey using the SPEC
    /// matrix (0.213/0.715/0.072 weights), not the old Rec.709 weights.
    #[test]
    fn saturate_zero_is_spec_luma_gray() {
        let mut b = Bitmap::new(4, 4);
        b.clear(Color { r: 0, g: 255, b: 0, a: 255 });
        b.apply_filter_rect(0, 0, 4, 4, FilterOp::Saturate(0.0));
        let p = at(&b, 1, 1);
        // s=0 → every output channel = 0.715*G = 182.
        assert_eq!(p.r, p.g);
        assert_eq!(p.g, p.b);
        assert!((p.r as i32 - 182).abs() <= 1, "spec luma 0.715*255≈182, got {}", p.r);
    }

    /// `hue-rotate(180deg)` on pure red shifts it toward cyan-ish (G and B
    /// rise, R drops) per the spec matrix.
    #[test]
    fn hue_rotate_180_shifts_red() {
        let mut b = Bitmap::new(4, 4);
        b.clear(Color { r: 255, g: 0, b: 0, a: 255 });
        b.apply_filter_rect(0, 0, 4, 4, FilterOp::HueRotate(180.0));
        let p = at(&b, 1, 1);
        assert!(p.g > p.r && p.b > p.r, "red rotated toward cyan: {:?}", p);
    }

    /// SVG `feGaussianBlur` via `apply_svg_filter_rect` blurs a hard edge —
    /// proving the `filter: url(#blur)` path runs a real blur primitive.
    #[test]
    fn svg_filter_fegaussianblur_blurs() {
        let mut b = Bitmap::new(40, 8);
        b.clear(Color::WHITE);
        b.fill_rect(0, 0, 20, 8, Color::BLACK);
        let prims = vec![SvgFePrimitive::GaussianBlur {
            input: "SourceGraphic".into(),
            std_dev: 3.0,
            result: None,
        }];
        b.apply_svg_filter_rect(0, 0, 40, 8, &prims);
        let seam = at(&b, 19, 4).r;
        assert!(seam > 30 && seam < 225, "feGaussianBlur seam mid-grey, got {seam}");
    }

    /// SVG `feColorMatrix type="saturate" 0` desaturates to spec luma —
    /// the colour-matrix primitive is real, not a passthrough.
    #[test]
    fn svg_filter_fecolormatrix_saturate_zero() {
        let mut b = Bitmap::new(4, 4);
        b.clear(Color { r: 0, g: 255, b: 0, a: 255 });
        let prims = vec![SvgFePrimitive::ColorMatrix {
            input: "SourceGraphic".into(),
            matrix: fe_saturate_matrix(0.0),
            result: None,
        }];
        b.apply_svg_filter_rect(0, 0, 4, 4, &prims);
        let p = at(&b, 1, 1);
        assert_eq!(p.r, p.g);
        assert_eq!(p.g, p.b);
        assert!((p.r as i32 - 182).abs() <= 1, "spec luma, got {}", p.r);
    }

    /// The canonical drop-shadow recipe expressed as SVG primitives:
    /// SourceAlpha → blur → offset → flood-tint → merge SourceGraphic on top.
    /// The shadow must appear OFFSET from the original opaque square, in the
    /// flood colour, while the original is preserved on top.
    #[test]
    fn svg_filter_drop_shadow_recipe() {
        let mut b = Bitmap::new(60, 60);
        b.clear(Color::TRANSPARENT);
        // Opaque red square at (10,10)-(30,30).
        b.fill_rect(10, 10, 20, 20, Color { r: 255, g: 0, b: 0, a: 255 });
        let prims = vec![
            SvgFePrimitive::GaussianBlur {
                input: "SourceAlpha".into(),
                std_dev: 1.5,
                result: Some("blur".into()),
            },
            SvgFePrimitive::Offset {
                input: "blur".into(),
                dx: 12,
                dy: 12,
                result: Some("off".into()),
            },
            // Tint the offset alpha to solid green using a color matrix that
            // forces RGB=green and keeps alpha (feFlood+feComposite-in equiv).
            SvgFePrimitive::ColorMatrix {
                input: "off".into(),
                matrix: [
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0,
                ],
                result: Some("shadow".into()),
            },
            SvgFePrimitive::Merge {
                inputs: vec!["shadow".into(), "SourceGraphic".into()],
                result: None,
            },
        ];
        b.apply_svg_filter_rect(0, 0, 60, 60, &prims);
        // Original red square preserved on top.
        let orig = at(&b, 20, 20);
        assert!(orig.r > 200 && orig.g < 40, "original red on top: {:?}", orig);
        // Shadow appears offset by +12,+12 in green.
        let shadow = at(&b, 32, 32);
        assert!(shadow.a > 0, "shadow has coverage: {:?}", shadow);
        assert!(shadow.g > 150 && shadow.r < 80, "shadow is green: {:?}", shadow);
        // Area with neither original nor shadow stays transparent.
        let empty = at(&b, 50, 5);
        assert_eq!(empty.a, 0, "untouched stays transparent: {:?}", empty);
    }

    /// `feOffset` translates the image and leaves transparency behind.
    #[test]
    fn svg_filter_feoffset_translates() {
        let mut b = Bitmap::new(40, 40);
        b.clear(Color::TRANSPARENT);
        b.fill_rect(0, 0, 10, 10, Color { r: 0, g: 0, b: 255, a: 255 });
        let prims = vec![SvgFePrimitive::Offset {
            input: "SourceGraphic".into(),
            dx: 15,
            dy: 15,
            result: None,
        }];
        b.apply_svg_filter_rect(0, 0, 40, 40, &prims);
        assert_eq!(at(&b, 3, 3).a, 0, "original location now empty");
        let moved = at(&b, 18, 18);
        assert!(moved.b > 200 && moved.a > 200, "moved to +15,+15: {:?}", moved);
    }

    // Keep the suppress-unused lint quiet for the opaque() helper if a future
    // refactor drops its last user.
    #[allow(dead_code)]
    fn _use_opaque() -> u32 {
        opaque(1, 2, 3)
    }
}
