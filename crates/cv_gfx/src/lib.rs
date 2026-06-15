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
                // Two-pass box blur — separable, O(W·H) per pass.
                // Radius is clamped to a sane upper bound to avoid
                // catastrophic O(W·H·r) blow-up on huge values.
                let r = (radius.round() as i32).clamp(1, 32);
                blur_h(self, x0, y0, x1, y1, r);
                blur_v(self, x0, y0, x1, y1, r);
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
                // Saturation around per-pixel luminance.
                color_per_pixel(self, x0, y0, x1, y1, |r, g, b| {
                    let l = 0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32;
                    let nudge = |c: u8| -> u8 {
                        let v = l + (c as f32 - l) * amt;
                        v.clamp(0.0, 255.0) as u8
                    };
                    (nudge(r), nudge(g), nudge(b))
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

/// Horizontal box blur over the rect. Mean of (2r+1) pixels in each
/// row, clipped at the rect edges (no edge wrap / mirror).
fn blur_h(bmp: &mut Bitmap, x0: i32, y0: i32, x1: i32, y1: i32, r: i32) {
    let w = (x1 - x0) as usize;
    let mut row = vec![0u32; w];
    for py in y0..y1 {
        for px_ in x0..x1 {
            row[(px_ - x0) as usize] =
                bmp.pixels[(py as usize) * (bmp.width as usize) + px_ as usize];
        }
        for ix in 0..w {
            let lo = ix.saturating_sub(r as usize);
            let hi = (ix + r as usize + 1).min(w);
            let n = (hi - lo) as u32;
            let (mut sa, mut sr, mut sg, mut sb) = (0u32, 0u32, 0u32, 0u32);
            for k in lo..hi {
                let p = row[k];
                sa += (p >> 24) & 0xFF;
                sr += (p >> 16) & 0xFF;
                sg += (p >> 8) & 0xFF;
                sb += p & 0xFF;
            }
            bmp.pixels[(py as usize) * (bmp.width as usize) + (x0 as usize + ix)] =
                ((sa / n) << 24) | ((sr / n) << 16) | ((sg / n) << 8) | (sb / n);
        }
    }
}

/// Vertical box blur — same idea over columns.
fn blur_v(bmp: &mut Bitmap, x0: i32, y0: i32, x1: i32, y1: i32, r: i32) {
    let h = (y1 - y0) as usize;
    let mut col = vec![0u32; h];
    for px_ in x0..x1 {
        for py in y0..y1 {
            col[(py - y0) as usize] =
                bmp.pixels[(py as usize) * (bmp.width as usize) + px_ as usize];
        }
        for iy in 0..h {
            let lo = iy.saturating_sub(r as usize);
            let hi = (iy + r as usize + 1).min(h);
            let n = (hi - lo) as u32;
            let (mut sa, mut sr, mut sg, mut sb) = (0u32, 0u32, 0u32, 0u32);
            for k in lo..hi {
                let p = col[k];
                sa += (p >> 24) & 0xFF;
                sr += (p >> 16) & 0xFF;
                sg += (p >> 8) & 0xFF;
                sb += p & 0xFF;
            }
            bmp.pixels[((y0 as usize + iy) * bmp.width as usize) + px_ as usize] =
                ((sa / n) << 24) | ((sr / n) << 16) | ((sg / n) << 8) | (sb / n);
        }
    }
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
}
