//! Minimum-viable SVG rasterizer.
//!
//! Parses the common SVG shape vocabulary — `<path>` (M, L, H, V, C,
//! S, Q, T, A, Z; absolute and relative variants), `<rect>`,
//! `<circle>`, `<ellipse>`, `<line>`, `<polyline>`, `<polygon>` — and
//! rasterizes them via a non-zero-winding scanline polygon fill into
//! a BGRA bitmap that conclave surfaces as the SVG element's
//! `embedded_image`. Curves subdivide adaptively into polylines.
//!
//! Not yet handled (each one a follow-up slice in its own right):
//! gradients, patterns, masks, clip-paths, filters, text-on-path,
//! `<use>`/`<defs>` references, stroke-linecap/linejoin/dasharray,
//! transforms beyond `viewBox` + per-element `transform="translate /
//! scale / rotate"`. Those silently degrade — the rest of the SVG
//! still renders, just without that effect.

#![allow(clippy::too_many_arguments)]

use crate::png::RgbaImage;

#[derive(Debug)]
pub enum SvgError {
    NoSize,
    EmptyDocument,
}

impl core::fmt::Display for SvgError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoSize => f.write_str("SVG: cannot determine output size"),
            Self::EmptyDocument => f.write_str("SVG: no shapes to render"),
        }
    }
}

impl std::error::Error for SvgError {}

/// Attributes the caller pulls off the `<svg>` element. Children
/// arrive as a flat list of (tag, attribute-map) tuples — conclave
/// builds this from the parsed HTML tree, which is the most natural
/// shape for our existing HTML tokeniser output.
#[derive(Debug, Clone, Default)]
pub struct SvgAttrs {
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub view_box: Option<[f32; 4]>,
    /// Inherited fill that descendants pick up unless they override
    /// it. Defaults to black per the SVG spec.
    pub fill: Option<[u8; 4]>,
    /// Resolved CSS `color` for the `<svg>` element. `fill: currentColor`
    /// and descendants that inherit `currentColor` should resolve against
    /// this instead of hard-coding black.
    pub current_color: Option<[u8; 4]>,
}

#[derive(Debug, Clone)]
pub struct SvgChild<'a> {
    pub tag: &'a str,
    pub attrs: &'a [(&'a str, &'a str)],
}

/// One color stop in a gradient. `offset` is normalised to 0.0..=1.0
/// at parse time; `rgba` already has stop-opacity multiplied into A.
#[derive(Debug, Clone, Copy)]
pub struct GradientStop {
    pub offset: f32,
    pub rgba: [u8; 4],
}

/// Parsed `<linearGradient>` definition. The four endpoint
/// coordinates are in *user space* (the same space the shape's
/// vertices live in), so the gradient picker just needs to map them
/// through the viewBox transform alongside the shape's vertices.
///
/// When `object_bounding_box` is true, x1/y1/x2/y2 are fractions
/// (0..1) of the filled shape's bounding box (the SVG default).
/// When false (`gradientUnits="userSpaceOnUse"`), they are raw
/// user-space coordinates.
#[derive(Debug, Clone)]
pub struct LinearGradientDef {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub stops: Vec<GradientStop>,
    /// True when `gradientUnits="objectBoundingBox"` (the spec default).
    pub object_bounding_box: bool,
}

/// What kind of paint a shape uses for its fill. Solid is the simple
/// case; LinearGradient samples per-pixel along the gradient axis.
#[derive(Debug, Clone)]
enum FillPaint {
    Solid([u8; 4]),
    LinearGradient(LinearGradientDef),
}

#[derive(Debug, Clone)]
enum StrokePaint {
    Solid([u8; 4]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillRule {
    NonZero,
    EvenOdd,
}

/// Parse, flatten, and rasterize the SVG element. Returns a square-
/// pixel `RgbaImage` sized from the SVG's `width`/`height` (or
/// `viewBox` dimensions if width/height are absent). The caller is
/// expected to clamp ridiculously large sizes before calling.
pub fn rasterize_svg_attrs(
    svg: &SvgAttrs,
    children: &[SvgChild<'_>],
) -> Result<RgbaImage, SvgError> {
    // Decide the rasterization grid. Prefer explicit width/height;
    // fall back to viewBox dimensions; bail if neither is known.
    //
    // For SVGs that ship only a viewBox (common for Material Symbols,
    // Tabler, Heroicons etc.) the viewBox may be in any coordinate
    // system — `0 0 24 24` for line-art icons, but `0 -960 960 960`
    // for Material Symbols on the modern Google web stack.  Using the
    // viewBox width directly as the rasterization size meant a 960px
    // icon embedded as an `<svg>` inside a tiny `<span>` would produce
    // a 960x960 bitmap and (since our layout takes its intrinsic
    // width from the embedded image) blow out the whole box.  Real
    // browsers cap default-sized replaced elements (HTML spec calls
    // for 300x150 fallback); we go further and cap to a tighter icon
    // size when there's no author-specified dimension, because
    // virtually every viewBox-only SVG in the wild is an icon meant
    // to flex to its CSS-sized container.
    const SVG_INTRINSIC_FALLBACK: f32 = 24.0;
    let view_box = svg.view_box.unwrap_or_else(|| {
        let w = svg.width.unwrap_or(SVG_INTRINSIC_FALLBACK);
        let h = svg.height.unwrap_or(SVG_INTRINSIC_FALLBACK);
        [0.0, 0.0, w, h]
    });
    // When neither attribute is present, scale the viewBox down to
    // `SVG_INTRINSIC_FALLBACK` along the longer axis, preserving the
    // viewBox aspect ratio so the path coords still rasterize
    // correctly — just into a small grid.
    let (default_w, default_h) = if svg.width.is_none() && svg.height.is_none() {
        let vbw = view_box[2].max(1.0);
        let vbh = view_box[3].max(1.0);
        let scale = SVG_INTRINSIC_FALLBACK / vbw.max(vbh);
        (vbw * scale, vbh * scale)
    } else {
        (view_box[2], view_box[3])
    };
    let out_w_f = svg.width.unwrap_or(default_w);
    let out_h_f = svg.height.unwrap_or(default_h);
    if out_w_f <= 0.0 || out_h_f <= 0.0 {
        return Err(SvgError::NoSize);
    }
    let out_w = out_w_f.round().max(1.0).min(4096.0) as u32;
    let out_h = out_h_f.round().max(1.0).min(4096.0) as u32;

    // viewBox → pixel mapping: scale x by out_w / vb_w, y by
    // out_h / vb_h, translate by (-vb_x, -vb_y).
    let scale_x = out_w as f32 / view_box[2];
    let scale_y = out_h as f32 / view_box[3];
    let map = |x: f32, y: f32| {
        let px = (x - view_box[0]) * scale_x;
        let py = (y - view_box[1]) * scale_y;
        (px, py)
    };

    // Working canvas: RGBA, initialised transparent.
    let mut pixels: Vec<u32> = vec![0u32; (out_w as usize) * (out_h as usize)];

    // Pre-pass: collect every `<linearGradient id="...">` (including
    // ones inside `<defs>` / `<g>`) so that `fill="url(#id)"` on a
    // shape can resolve to a real gradient rather than fall back to
    // black. Each gradient carries its endpoint coordinates (in user
    // space) plus a list of `(offset, RGBA)` stops.
    let gradients = collect_linear_gradients(children);

    let default_fill = svg.fill.or(svg.current_color).unwrap_or([0, 0, 0, 255]);

    let mut any_shape = false;
    // Track whether any non-`path` primitive (rect/circle/ellipse/
    // polygon/polyline/line) contributed pixels. The uniform-block
    // suppression below was originally a defence against `path d="..."`
    // commands we don't yet implement (cubic Béziers, arcs) flooding
    // the bounding box with the SVG's `fill` colour — but it was
    // false-positive-rejecting tiny legitimate rect/circle fills.
    // Only suppress when the rasterized content came from path-only.
    let mut primitive_shape_drawn = false;
    for c in children {
        let opacity = child_opacity(c, "opacity").unwrap_or(1.0).clamp(0.0, 1.0);
        let fill_opacity =
            (opacity * child_opacity(c, "fill-opacity").unwrap_or(1.0)).clamp(0.0, 1.0);
        let stroke_opacity =
            (opacity * child_opacity(c, "stroke-opacity").unwrap_or(1.0)).clamp(0.0, 1.0);
        // Per-child fill override (presentation attribute). A
        // `fill="url(#id)"` resolves to one of the collected
        // gradients; everything else falls through to solid colour
        // parsing.
        let raw_fill_attr = child_paint_attr(c, "fill");
        let gradient_ref = raw_fill_attr.and_then(|s| extract_url_id(s));
        let fill_paint: FillPaint = if let Some(id) = gradient_ref {
            match gradients.get(id) {
                Some(g) => FillPaint::LinearGradient(g.clone()),
                // Per SVG spec: if url(#id) can't be resolved, render as
                // `none` (transparent), not black/default. A missing gradient
                // should be invisible, not a black fill that masks the shape.
                None => FillPaint::Solid([0, 0, 0, 0]),
            }
        } else {
            FillPaint::Solid(apply_alpha(
                parse_fill(raw_fill_attr, svg.current_color).unwrap_or(default_fill),
                fill_opacity,
            ))
        };
        let stroke_paint: Option<StrokePaint> =
            parse_fill(child_paint_attr(c, "stroke"), svg.current_color)
                .map(|rgba| apply_alpha(rgba, stroke_opacity))
                .filter(|rgba| rgba[3] > 0)
                .map(StrokePaint::Solid);
        let stroke_width = child_paint_attr(c, "stroke-width")
            .and_then(parse_svg_number)
            .unwrap_or(1.0)
            .max(0.0);
        let fill_rule = child_fill_rule(c);
        let use_x = num_attr(c, "tb-use-x", 0.0);
        let use_y = num_attr(c, "tb-use-y", 0.0);
        // Optional nested-`<svg>` viewport clip. Coordinates are in this
        // SVG's user space (set by the HTML→SVG walker for top-level
        // nested viewports); map them to pixel space so each shape can be
        // confined to its sprite cell.
        let clip_rect: Option<(f32, f32, f32, f32)> = match (
            child_attr(c, "tb-clip-x0").and_then(parse_svg_number),
            child_attr(c, "tb-clip-y0").and_then(parse_svg_number),
            child_attr(c, "tb-clip-x1").and_then(parse_svg_number),
            child_attr(c, "tb-clip-y1").and_then(parse_svg_number),
        ) {
            (Some(cx0), Some(cy0), Some(cx1), Some(cy1)) => {
                let (px0, py0) = map(cx0, cy0);
                let (px1, py1) = map(cx1, cy1);
                Some((px0.min(px1), py0.min(py1), px0.max(px1), py0.max(py1)))
            }
            _ => None,
        };
        // `fill="none"` → skip drawing.
        if matches!(&fill_paint, FillPaint::Solid(c) if c[3] == 0) {
            if stroke_paint.is_none() || stroke_width <= 0.0 {
                continue;
            }
        }
        // Per-element `transform="..."` — parse into an affine 2x3.
        // None means identity; skip the per-vertex multiply if so.
        let xform = child_attr(c, "transform").and_then(parse_transform);
        any_shape = true;
        if !matches!(c.tag, "path") {
            primitive_shape_drawn = true;
        }

        // ViewBox dimensions for resolving percentage lengths.
        let vb_w = view_box[2];
        let vb_h = view_box[3];
        // Diagonal for radius-type attributes that use both dimensions.
        let vb_diag = ((vb_w * vb_w + vb_h * vb_h) / 2.0).sqrt();

        let polys = match c.tag {
            "path" => {
                let d = child_attr(c, "d").unwrap_or("");
                path_to_polygons(d)
            }
            "rect" => {
                // x/y/width/height can all be percentages of viewBox width/height.
                let x = length_attr(c, "x", 0.0, vb_w);
                let y = length_attr(c, "y", 0.0, vb_h);
                let w = length_attr(c, "width", 0.0, vb_w);
                let h = length_attr(c, "height", 0.0, vb_h);
                if w > 0.0 && h > 0.0 {
                    // SVG spec: if only one of rx/ry is specified the other
                    // takes the same value; clamp each to half the relevant side.
                    let rx_raw = length_attr(c, "rx", -1.0, vb_w);
                    let ry_raw = length_attr(c, "ry", -1.0, vb_h);
                    let (mut rx, mut ry) = match (rx_raw >= 0.0, ry_raw >= 0.0) {
                        (true, true)   => (rx_raw, ry_raw),
                        (true, false)  => (rx_raw, rx_raw),
                        (false, true)  => (ry_raw, ry_raw),
                        (false, false) => (0.0, 0.0),
                    };
                    rx = rx.min(w * 0.5);
                    ry = ry.min(h * 0.5);
                    if rx > 0.0 && ry > 0.0 {
                        vec![rounded_rect_polygon(x, y, w, h, rx, ry)]
                    } else {
                        vec![rect_polygon(x, y, w, h)]
                    }
                } else {
                    Vec::new()
                }
            }
            "circle" => {
                let cx = length_attr(c, "cx", 0.0, vb_w);
                let cy = length_attr(c, "cy", 0.0, vb_h);
                let r = length_attr(c, "r", 0.0, vb_diag);
                if r > 0.0 {
                    vec![ellipse_polygon(cx, cy, r, r)]
                } else {
                    Vec::new()
                }
            }
            "ellipse" => {
                let cx = length_attr(c, "cx", 0.0, vb_w);
                let cy = length_attr(c, "cy", 0.0, vb_h);
                let rx = length_attr(c, "rx", 0.0, vb_w);
                let ry = length_attr(c, "ry", 0.0, vb_h);
                if rx > 0.0 && ry > 0.0 {
                    vec![ellipse_polygon(cx, cy, rx, ry)]
                } else {
                    Vec::new()
                }
            }
            "polygon" | "polyline" => {
                let pts_s = child_attr(c, "points").unwrap_or("");
                let pts = parse_points(pts_s);
                if pts.len() >= 3 {
                    vec![pts]
                } else if c.tag == "polyline" && pts.len() == 2 {
                    // A 2-point polyline is a single line segment.  Thicken it
                    // to a 1-pixel-wide polygon so it can be stroked normally.
                    let (x1, y1) = pts[0];
                    let (x2, y2) = pts[1];
                    vec![thicken_line(x1, y1, x2, y2, 1.0)]
                } else {
                    Vec::new()
                }
            }
            "line" => {
                // No fillable area — render as a 1px polygon along
                // the segment (rough but useful for diagrams).
                let x1 = length_attr(c, "x1", 0.0, vb_w);
                let y1 = length_attr(c, "y1", 0.0, vb_h);
                let x2 = length_attr(c, "x2", 0.0, vb_w);
                let y2 = length_attr(c, "y2", 0.0, vb_h);
                vec![thicken_line(x1, y1, x2, y2, 1.0)]
            }
            _ => Vec::new(), // <g>, <defs>, <text>, etc. — TODO
        };

        let mut mapped_polys: Vec<Vec<(f32, f32)>> = Vec::new();
        for poly in &polys {
            if poly.len() < 3 {
                continue;
            }
            let mapped: Vec<(f32, f32)> = poly
                .iter()
                .map(|&(x, y)| {
                    // user-space coords: shift for `<use x= y=>`, then
                    // apply per-element transform (rotate/scale/skew/etc),
                    // then map through the viewBox into pixel space.
                    let (ux, uy) = (x + use_x, y + use_y);
                    let (tx, ty) = if let Some(m) = &xform {
                        apply_affine(m, ux, uy)
                    } else {
                        (ux, uy)
                    };
                    map(tx, ty)
                })
                .collect();
            // Confine to the nested-viewport cell if one was set.
            let mapped = match clip_rect {
                Some(cr) => clip_polygon_to_rect(&mapped, cr),
                None => mapped,
            };
            if mapped.len() < 3 {
                continue;
            }
            mapped_polys.push(mapped);
        }

        if !mapped_polys.is_empty() {
            match &fill_paint {
                FillPaint::Solid(c) => {
                    if c[3] > 0 {
                        match fill_rule {
                            FillRule::NonZero => fill_polygons_nonzero(
                                &mut pixels,
                                out_w as i32,
                                out_h as i32,
                                &mapped_polys,
                                *c,
                            ),
                            FillRule::EvenOdd => fill_polygons_evenodd(
                                &mut pixels,
                                out_w as i32,
                                out_h as i32,
                                &mapped_polys,
                                *c,
                            ),
                        }
                    }
                }
                FillPaint::LinearGradient(g) => {
                    // Apply fill-opacity to all stop alphas by creating a
                    // modified stop list. This respects `fill-opacity` and
                    // `opacity` on the element even for gradient fills.
                    let stops_with_opacity: Vec<GradientStop> = if fill_opacity < 1.0 {
                        g.stops
                            .iter()
                            .map(|s| GradientStop {
                                offset: s.offset,
                                rgba: apply_alpha(s.rgba, fill_opacity),
                            })
                            .collect()
                    } else {
                        g.stops.clone()
                    };

                    for mapped in &mapped_polys {
                        // Resolve gradient endpoints to pixel space.
                        // For objectBoundingBox mode, the x1/y1/x2/y2 are fractions
                        // of the shape's bounding box. Compute the bbox in pixel space
                        // and transform the gradient endpoints accordingly.
                        let (gx1, gy1, gx2, gy2) = if g.object_bounding_box {
                            // Compute the pixel-space bounding box of this polygon.
                            let mut bx0 = f32::INFINITY;
                            let mut by0 = f32::INFINITY;
                            let mut bx1 = f32::NEG_INFINITY;
                            let mut by1 = f32::NEG_INFINITY;
                            for &(px, py) in mapped.iter() {
                                if px < bx0 { bx0 = px; }
                                if py < by0 { by0 = py; }
                                if px > bx1 { bx1 = px; }
                                if py > by1 { by1 = py; }
                            }
                            let bw = bx1 - bx0;
                            let bh = by1 - by0;
                            // Transform fractional gradient coords by the bbox.
                            let gx1 = bx0 + g.x1 * bw;
                            let gy1 = by0 + g.y1 * bh;
                            let gx2 = bx0 + g.x2 * bw;
                            let gy2 = by0 + g.y2 * bh;
                            (gx1, gy1, gx2, gy2)
                        } else {
                            // userSpaceOnUse: map through use offset → element
                            // transform → viewBox, just like shape vertices.
                            let (gux1, guy1) = (g.x1 + use_x, g.y1 + use_y);
                            let (gux2, guy2) = (g.x2 + use_x, g.y2 + use_y);
                            let ((gtx1, gty1), (gtx2, gty2)) = if let Some(m) = &xform {
                                (apply_affine(m, gux1, guy1), apply_affine(m, gux2, guy2))
                            } else {
                                ((gux1, guy1), (gux2, guy2))
                            };
                            let (gx1, gy1) = map(gtx1, gty1);
                            let (gx2, gy2) = map(gtx2, gty2);
                            (gx1, gy1, gx2, gy2)
                        };

                        fill_polygon_nonzero_gradient(
                            &mut pixels,
                            out_w as i32,
                            out_h as i32,
                            mapped,
                            gx1,
                            gy1,
                            gx2,
                            gy2,
                            &stops_with_opacity,
                        );
                    }
                }
            }
            if let Some(stroke) = &stroke_paint {
                let pixel_stroke_width = match c.tag {
                    "line" => stroke_width * ((scale_x.abs() + scale_y.abs()) * 0.5),
                    _ => {
                        let avg = (scale_x.abs() + scale_y.abs()) * 0.5;
                        stroke_width * avg
                    }
                }
                .max(1.0);
                for mapped in &mapped_polys {
                    if mapped.len() < 2 {
                        continue;
                    }
                    for seg in mapped.windows(2) {
                        let [(x1, y1), (x2, y2)] = [seg[0], seg[1]];
                        let stroke_poly = thicken_line(x1, y1, x2, y2, pixel_stroke_width);
                        match stroke {
                            StrokePaint::Solid(rgba) => {
                                fill_polygon_nonzero(
                                    &mut pixels,
                                    out_w as i32,
                                    out_h as i32,
                                    &stroke_poly,
                                    *rgba,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    if !any_shape {
        return Err(SvgError::EmptyDocument);
    }

    // Reject SVGs that collapsed to a single solid-color block. This
    // catches the case where path commands we don't yet handle (cubic
    // Beziers, smooth-quadratic, arc-to) bail out and leave the whole
    // bounding box flooded with the SVG's `fill` colour. Google's
    // footer leaf emoji was hitting this — rendering as a bright
    // green square instead of a leaf. Smaller than 64×64 (~ a typical
    // inline icon) and uniform → treat as a render failure so the
    // caller can drop the image entirely.
    // Only apply the uniform-block suppression when every shape came
    // from a `<path>` (the original false-positive source). Explicit
    // `<rect>` / `<circle>` / `<ellipse>` / `<polygon>` etc. that
    // legitimately produce a uniform fill should pass through.
    if !primitive_shape_drawn && out_w <= 64 && out_h <= 64 && !pixels.is_empty() {
        let first = pixels[0];
        let opaque_first = (first >> 24) & 0xFF;
        if opaque_first > 32 && pixels.iter().all(|&px| px == first) {
            return Err(SvgError::EmptyDocument);
        }
    }

    Ok(RgbaImage {
        width: out_w,
        height: out_h,
        pixels,
    })
}

fn child_attr<'a>(c: &'a SvgChild<'_>, name: &str) -> Option<&'a str> {
    c.attrs
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| *v)
}

fn child_style_attr<'a>(c: &'a SvgChild<'_>, name: &str) -> Option<&'a str> {
    let style = child_attr(c, "style")?;
    for decl in style.split(';') {
        let decl = decl.trim();
        let Some((k, v)) = decl.split_once(':') else {
            continue;
        };
        if k.trim().eq_ignore_ascii_case(name) {
            return Some(v.trim());
        }
    }
    None
}

fn child_paint_attr<'a>(c: &'a SvgChild<'_>, name: &str) -> Option<&'a str> {
    child_attr(c, name).or_else(|| child_style_attr(c, name))
}

fn child_fill_rule(c: &SvgChild<'_>) -> FillRule {
    match child_paint_attr(c, "fill-rule")
        .unwrap_or("nonzero")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "evenodd" => FillRule::EvenOdd,
        _ => FillRule::NonZero,
    }
}

fn parse_svg_number(s: &str) -> Option<f32> {
    let s = s.trim();
    // Strip trailing unit suffixes that we don't need for numeric value.
    // Percentages stripped here return the raw number (e.g. "50%" → 50.0).
    // Call `parse_svg_length` instead when you need % resolved against a
    // reference dimension.
    s.trim_end_matches(&['p', 'x', '%', 'e', 'm', 't', 'r', 'c', 'h', 'v', 'w'][..])
        .parse::<f32>()
        .ok()
}

/// Parse an SVG length value, resolving percentage values against `reference`.
/// E.g. `parse_svg_length("50%", 200.0)` → 100.0.
/// Non-percentage values are returned as-is (same as `parse_svg_number`).
fn parse_svg_length(s: &str, reference: f32) -> Option<f32> {
    let s = s.trim();
    if let Some(pct_s) = s.strip_suffix('%') {
        let pct: f32 = pct_s.trim().parse().ok()?;
        return Some(pct / 100.0 * reference);
    }
    parse_svg_number(s)
}

/// Like `num_attr` but resolves percentage values against `reference`.
fn length_attr(c: &SvgChild<'_>, name: &str, default: f32, reference: f32) -> f32 {
    child_paint_attr(c, name)
        .and_then(|s| parse_svg_length(s, reference))
        .unwrap_or(default)
}

fn child_opacity(c: &SvgChild<'_>, name: &str) -> Option<f32> {
    child_paint_attr(c, name)
        .and_then(parse_svg_number)
        .map(|v| v.clamp(0.0, 1.0))
}

fn apply_alpha(mut rgba: [u8; 4], opacity: f32) -> [u8; 4] {
    rgba[3] = ((rgba[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
    rgba
}

fn num_attr(c: &SvgChild<'_>, name: &str, default: f32) -> f32 {
    child_paint_attr(c, name)
        .and_then(parse_svg_number)
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Affine transforms — `transform="translate(…) rotate(…) matrix(…)"`.
// We store a 2x3 row-major affine: [a, b, c, d, e, f] meaning
//   x' = a*x + c*y + e
//   y' = b*x + d*y + f
// Multiple primitives in one attribute compose left-to-right (the SVG
// rule: `transform="A B"` applied to a point p means A(B(p))).
// ---------------------------------------------------------------------------

/// 2D affine, row-major.
#[derive(Debug, Clone, Copy)]
pub struct Affine {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

impl Affine {
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }
}

fn apply_affine(m: &Affine, x: f32, y: f32) -> (f32, f32) {
    (m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f)
}

fn mul_affine(l: &Affine, r: &Affine) -> Affine {
    // (l * r) applied to p == l(r(p)). Standard 2x3 affine multiplication.
    Affine {
        a: l.a * r.a + l.c * r.b,
        b: l.b * r.a + l.d * r.b,
        c: l.a * r.c + l.c * r.d,
        d: l.b * r.c + l.d * r.d,
        e: l.a * r.e + l.c * r.f + l.e,
        f: l.b * r.e + l.d * r.f + l.f,
    }
}

/// Parse a `transform="…"` value. Recognises translate, scale, rotate,
/// skewX, skewY, matrix. Anything unrecognised or malformed is treated
/// as identity for that primitive and we keep parsing. Returns None if
/// the result is exactly identity (so callers can skip the multiply).
pub fn parse_transform(s: &str) -> Option<Affine> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut acc = Affine::identity();
    let mut any = false;
    while i < bytes.len() {
        // Skip whitespace and commas between primitives.
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // Read function name [a-zA-Z]+.
        let name_start = i;
        while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
            i += 1;
        }
        if name_start == i {
            // not a name — skip this byte and continue
            i += 1;
            continue;
        }
        let name = &s[name_start..i];
        // Skip whitespace then expect '('.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'(' {
            continue;
        }
        i += 1;
        // Read until matching ')'.
        let args_start = i;
        while i < bytes.len() && bytes[i] != b')' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let args_s = &s[args_start..i];
        i += 1; // consume ')'
        let nums: Vec<f32> = args_s
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter(|t| !t.is_empty())
            .filter_map(|t| t.parse::<f32>().ok())
            .collect();
        let m = match name.to_ascii_lowercase().as_str() {
            "translate" => {
                let tx = nums.get(0).copied().unwrap_or(0.0);
                let ty = nums.get(1).copied().unwrap_or(0.0);
                Affine {
                    a: 1.0,
                    b: 0.0,
                    c: 0.0,
                    d: 1.0,
                    e: tx,
                    f: ty,
                }
            }
            "scale" => {
                let sx = nums.get(0).copied().unwrap_or(1.0);
                let sy = nums.get(1).copied().unwrap_or(sx);
                Affine {
                    a: sx,
                    b: 0.0,
                    c: 0.0,
                    d: sy,
                    e: 0.0,
                    f: 0.0,
                }
            }
            "rotate" => {
                let deg = nums.get(0).copied().unwrap_or(0.0);
                let r = deg.to_radians();
                let (sn, cs) = (r.sin(), r.cos());
                let rot = Affine {
                    a: cs,
                    b: sn,
                    c: -sn,
                    d: cs,
                    e: 0.0,
                    f: 0.0,
                };
                if let (Some(&cx), Some(&cy)) = (nums.get(1), nums.get(2)) {
                    // rotate(deg, cx, cy) == T(cx,cy) * R * T(-cx,-cy)
                    let t1 = Affine {
                        a: 1.0,
                        b: 0.0,
                        c: 0.0,
                        d: 1.0,
                        e: cx,
                        f: cy,
                    };
                    let t2 = Affine {
                        a: 1.0,
                        b: 0.0,
                        c: 0.0,
                        d: 1.0,
                        e: -cx,
                        f: -cy,
                    };
                    mul_affine(&mul_affine(&t1, &rot), &t2)
                } else {
                    rot
                }
            }
            "skewx" => {
                let deg = nums.get(0).copied().unwrap_or(0.0);
                let t = deg.to_radians().tan();
                Affine {
                    a: 1.0,
                    b: 0.0,
                    c: t,
                    d: 1.0,
                    e: 0.0,
                    f: 0.0,
                }
            }
            "skewy" => {
                let deg = nums.get(0).copied().unwrap_or(0.0);
                let t = deg.to_radians().tan();
                Affine {
                    a: 1.0,
                    b: t,
                    c: 0.0,
                    d: 1.0,
                    e: 0.0,
                    f: 0.0,
                }
            }
            "matrix" => {
                if nums.len() < 6 {
                    continue;
                }
                Affine {
                    a: nums[0],
                    b: nums[1],
                    c: nums[2],
                    d: nums[3],
                    e: nums[4],
                    f: nums[5],
                }
            }
            _ => continue,
        };
        acc = mul_affine(&acc, &m);
        any = true;
    }
    if any { Some(acc) } else { None }
}

// ---------------------------------------------------------------------------
// Path parsing — d="M L H V C S Q T A Z" + relative variants.
// ---------------------------------------------------------------------------

/// Turn a path data string into a list of polygons (one per
/// subpath). Curves are flattened by adaptive subdivision.
pub fn path_to_polygons(d: &str) -> Vec<Vec<(f32, f32)>> {
    let mut tokens = PathTokens::new(d);
    let mut subpaths: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut current: Vec<(f32, f32)> = Vec::new();
    let mut cur = (0.0_f32, 0.0_f32);
    let mut start = (0.0_f32, 0.0_f32);
    // Previous control point for S/T smoothing.
    let mut last_cubic_ctrl: Option<(f32, f32)> = None;
    let mut last_quad_ctrl: Option<(f32, f32)> = None;

    while let Some(cmd) = tokens.command() {
        let is_relative = cmd.is_ascii_lowercase();
        let cmd_lower = cmd.to_ascii_lowercase();
        match cmd_lower {
            'm' => {
                // First M opens a subpath; subsequent implicit
                // coordinate pairs after M are treated as L's.
                let (x, y) = tokens.pair().unwrap_or((0.0, 0.0));
                if is_relative {
                    cur.0 += x;
                    cur.1 += y;
                } else {
                    cur = (x, y);
                }
                if !current.is_empty() {
                    subpaths.push(std::mem::take(&mut current));
                }
                current.push(cur);
                start = cur;
                while let Some((nx, ny)) = tokens.maybe_pair() {
                    if is_relative {
                        cur.0 += nx;
                        cur.1 += ny;
                    } else {
                        cur = (nx, ny);
                    }
                    current.push(cur);
                }
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
            }
            'l' => {
                while let Some((x, y)) = tokens.maybe_pair() {
                    if is_relative {
                        cur.0 += x;
                        cur.1 += y;
                    } else {
                        cur = (x, y);
                    }
                    current.push(cur);
                }
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
            }
            'h' => {
                while let Some(x) = tokens.maybe_num() {
                    if is_relative {
                        cur.0 += x;
                    } else {
                        cur.0 = x;
                    }
                    current.push(cur);
                }
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
            }
            'v' => {
                while let Some(y) = tokens.maybe_num() {
                    if is_relative {
                        cur.1 += y;
                    } else {
                        cur.1 = y;
                    }
                    current.push(cur);
                }
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
            }
            'c' => {
                while let Some((x1, y1)) = tokens.maybe_pair() {
                    let (x2, y2) = tokens.pair().unwrap_or((x1, y1));
                    let (x, y) = tokens.pair().unwrap_or((x2, y2));
                    let (c1x, c1y, c2x, c2y, ex, ey) = if is_relative {
                        (
                            cur.0 + x1,
                            cur.1 + y1,
                            cur.0 + x2,
                            cur.1 + y2,
                            cur.0 + x,
                            cur.1 + y,
                        )
                    } else {
                        (x1, y1, x2, y2, x, y)
                    };
                    flatten_cubic(cur, (c1x, c1y), (c2x, c2y), (ex, ey), &mut current);
                    last_cubic_ctrl = Some((c2x, c2y));
                    last_quad_ctrl = None;
                    cur = (ex, ey);
                }
            }
            's' => {
                while let Some((x2, y2)) = tokens.maybe_pair() {
                    let (x, y) = tokens.pair().unwrap_or((x2, y2));
                    // First control = reflection of previous cubic
                    // control around current point (or current itself
                    // if no previous cubic).
                    let (c1x, c1y) = match last_cubic_ctrl {
                        Some((px, py)) => (2.0 * cur.0 - px, 2.0 * cur.1 - py),
                        None => cur,
                    };
                    let (c2x, c2y, ex, ey) = if is_relative {
                        (cur.0 + x2, cur.1 + y2, cur.0 + x, cur.1 + y)
                    } else {
                        (x2, y2, x, y)
                    };
                    flatten_cubic(cur, (c1x, c1y), (c2x, c2y), (ex, ey), &mut current);
                    last_cubic_ctrl = Some((c2x, c2y));
                    last_quad_ctrl = None;
                    cur = (ex, ey);
                }
            }
            'q' => {
                while let Some((x1, y1)) = tokens.maybe_pair() {
                    let (x, y) = tokens.pair().unwrap_or((x1, y1));
                    let (cx, cy, ex, ey) = if is_relative {
                        (cur.0 + x1, cur.1 + y1, cur.0 + x, cur.1 + y)
                    } else {
                        (x1, y1, x, y)
                    };
                    flatten_quad(cur, (cx, cy), (ex, ey), &mut current);
                    last_quad_ctrl = Some((cx, cy));
                    last_cubic_ctrl = None;
                    cur = (ex, ey);
                }
            }
            't' => {
                while let Some((x, y)) = tokens.maybe_pair() {
                    let (cx, cy) = match last_quad_ctrl {
                        Some((px, py)) => (2.0 * cur.0 - px, 2.0 * cur.1 - py),
                        None => cur,
                    };
                    let (ex, ey) = if is_relative {
                        (cur.0 + x, cur.1 + y)
                    } else {
                        (x, y)
                    };
                    flatten_quad(cur, (cx, cy), (ex, ey), &mut current);
                    last_quad_ctrl = Some((cx, cy));
                    last_cubic_ctrl = None;
                    cur = (ex, ey);
                }
            }
            'a' => {
                // Elliptical arc — SVG 1.1 §F.6 center-parameterisation.
                // Decomposed into adaptive cubic Bézier segments via
                // the Riess approximation (same quality as browsers).
                while let Some(rx) = tokens.maybe_num() {
                    let ry = tokens.num().unwrap_or(rx);
                    let x_rot_deg = tokens.num().unwrap_or(0.0);
                    let large_arc = tokens.num().unwrap_or(0.0) != 0.0;
                    let sweep = tokens.num().unwrap_or(0.0) != 0.0;
                    let (x, y) = tokens.pair().unwrap_or(cur);
                    let (ex, ey) = if is_relative {
                        (cur.0 + x, cur.1 + y)
                    } else {
                        (x, y)
                    };
                    arc_to_polyline(cur, rx, ry, x_rot_deg, large_arc, sweep, (ex, ey), &mut current);
                    cur = (ex, ey);
                }
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
            }
            'z' => {
                if let Some(first) = current.first().copied() {
                    if cur != first {
                        current.push(first);
                    }
                }
                if !current.is_empty() {
                    subpaths.push(std::mem::take(&mut current));
                }
                cur = start;
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
            }
            _ => {
                // Unknown command — abort parsing the rest of this
                // path; whatever we collected so far still draws.
                break;
            }
        }
    }
    if !current.is_empty() {
        subpaths.push(current);
    }
    subpaths
}

// Adaptive subdivision tolerance: max acceptable distance from the
// flat polyline to the true curve, in viewBox units. 0.25 is a
// common default — visually smooth on raster output up to a few
// hundred pixels per em.
const FLATNESS_TOL_SQ: f32 = 0.0625;

fn flatten_quad(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), out: &mut Vec<(f32, f32)>) {
    // Midpoint distance from chord — a good flatness proxy for
    // quadratics: ||(p0 + 2 p1 + p2) - 2 (p0 + p2)|| / 4 = ||p1 - mid||
    let mid_x = (p0.0 + p2.0) * 0.5;
    let mid_y = (p0.1 + p2.1) * 0.5;
    let dx = p1.0 - mid_x;
    let dy = p1.1 - mid_y;
    if dx * dx + dy * dy < FLATNESS_TOL_SQ {
        out.push(p2);
        return;
    }
    let p01 = ((p0.0 + p1.0) * 0.5, (p0.1 + p1.1) * 0.5);
    let p12 = ((p1.0 + p2.0) * 0.5, (p1.1 + p2.1) * 0.5);
    let p012 = ((p01.0 + p12.0) * 0.5, (p01.1 + p12.1) * 0.5);
    flatten_quad(p0, p01, p012, out);
    flatten_quad(p012, p12, p2, out);
}

fn flatten_cubic(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    out: &mut Vec<(f32, f32)>,
) {
    // Distance of control points to the chord p0..p3.
    let dx = p3.0 - p0.0;
    let dy = p3.1 - p0.1;
    let d1 = ((p1.0 - p0.0) * dy - (p1.1 - p0.1) * dx).abs();
    let d2 = ((p2.0 - p0.0) * dy - (p2.1 - p0.1) * dx).abs();
    let len_sq = dx * dx + dy * dy;
    if (d1 + d2) * (d1 + d2) < FLATNESS_TOL_SQ * len_sq.max(1e-6) {
        out.push(p3);
        return;
    }
    let q0 = ((p0.0 + p1.0) * 0.5, (p0.1 + p1.1) * 0.5);
    let q1 = ((p1.0 + p2.0) * 0.5, (p1.1 + p2.1) * 0.5);
    let q2 = ((p2.0 + p3.0) * 0.5, (p2.1 + p3.1) * 0.5);
    let r0 = ((q0.0 + q1.0) * 0.5, (q0.1 + q1.1) * 0.5);
    let r1 = ((q1.0 + q2.0) * 0.5, (q1.1 + q2.1) * 0.5);
    let s = ((r0.0 + r1.0) * 0.5, (r0.1 + r1.1) * 0.5);
    flatten_cubic(p0, q0, r0, s, out);
    flatten_cubic(s, r1, q2, p3, out);
}

/// Signed angle from vector (ux,uy) to vector (vx,vy), in radians.
fn angle_between(ux: f32, uy: f32, vx: f32, vy: f32) -> f32 {
    let dot = ux * vx + uy * vy;
    let len = ((ux * ux + uy * uy) * (vx * vx + vy * vy)).sqrt();
    let cos_a = (dot / len).clamp(-1.0, 1.0);
    let sign = if ux * vy - uy * vx < 0.0 { -1.0_f32 } else { 1.0_f32 };
    sign * cos_a.acos()
}

/// Decompose one SVG elliptical arc command into adaptive cubic Bézier
/// polyline segments, appending each new point to `out`.
///
/// Follows SVG 1.1 Appendix F §F.6.5 (endpoint → center parameterisation)
/// and §F.6.6 (cubic approximation via the Riess formula).
fn arc_to_polyline(
    (x1, y1): (f32, f32),
    rx_in: f32,
    ry_in: f32,
    x_rot_deg: f32,
    large_arc: bool,
    sweep: bool,
    (x2, y2): (f32, f32),
    out: &mut Vec<(f32, f32)>,
) {
    // Degenerate: endpoints coincide → nothing to draw.
    if (x2 - x1).abs() < 1e-6 && (y2 - y1).abs() < 1e-6 {
        return;
    }
    // Degenerate radii → straight line to endpoint.
    let mut rx = rx_in.abs();
    let mut ry = ry_in.abs();
    if rx < 1e-6 || ry < 1e-6 {
        out.push((x2, y2));
        return;
    }

    let phi = x_rot_deg.to_radians();
    let (sin_phi, cos_phi) = phi.sin_cos();

    // §F.6.5.1 — midpoint in x-rotated coordinates.
    let dx2 = (x1 - x2) * 0.5;
    let dy2 = (y1 - y2) * 0.5;
    let x1p =  cos_phi * dx2 + sin_phi * dy2;
    let y1p = -sin_phi * dx2 + cos_phi * dy2;

    // §F.6.6.3 — scale up radii if they are too small.
    {
        let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
        if lambda > 1.0 {
            let s = lambda.sqrt();
            rx *= s;
            ry *= s;
        }
    }

    // §F.6.5.2 — center in rotated coordinates (cx′, cy′).
    let rx_sq = rx * rx;
    let ry_sq = ry * ry;
    let x1p_sq = x1p * x1p;
    let y1p_sq = y1p * y1p;
    let num = (rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq).max(0.0);
    let den = rx_sq * y1p_sq + ry_sq * x1p_sq;
    let sq = if den > 1e-12 { (num / den).sqrt() } else { 0.0 };
    let sign = if large_arc == sweep { -1.0_f32 } else { 1.0_f32 };
    let cxp =  sign * sq * rx * y1p / ry;
    let cyp = -sign * sq * ry * x1p / rx;

    // §F.6.5.3 — back to user-space center (cx, cy).
    let mx = (x1 + x2) * 0.5;
    let my = (y1 + y2) * 0.5;
    let cx = cos_phi * cxp - sin_phi * cyp + mx;
    let cy = sin_phi * cxp + cos_phi * cyp + my;

    // §F.6.5.5 — start angle θ₁ and angular extent Δθ.
    let ux = (x1p - cxp) / rx;
    let uy = (y1p - cyp) / ry;
    let vx = (-x1p - cxp) / rx;
    let vy = (-y1p - cyp) / ry;

    let theta1 = angle_between(1.0, 0.0, ux, uy);
    let mut d_theta = angle_between(ux, uy, vx, vy);
    if !sweep && d_theta > 0.0 {
        d_theta -= 2.0 * std::f32::consts::PI;
    } else if sweep && d_theta < 0.0 {
        d_theta += 2.0 * std::f32::consts::PI;
    }

    // Split into ≤ π/2 segments; convert each to a cubic Bézier.
    let n_segs = ((d_theta / (std::f32::consts::FRAC_PI_2)).abs().ceil() as u32).max(1);
    let seg = d_theta / (n_segs as f32);

    for i in 0..n_segs {
        let t1 = theta1 + (i as f32) * seg;
        let t2 = t1 + seg;

        // Riess approximation: α = sin(Δθ) · (√(4 + 3 tan²(Δθ/2)) − 1) / 3
        let alpha = {
            let th = seg * 0.5;
            let tan_h = th.tan();
            (seg.sin() * ((4.0 + 3.0 * tan_h * tan_h).sqrt() - 1.0)) / 3.0
        };

        // Parametric ellipse derivative at t (in user space):
        //   d/dt [cx + cos_phi·rx·cos(t) − sin_phi·ry·sin(t)]
        //       = −cos_phi·rx·sin(t) − sin_phi·ry·cos(t)
        //   d/dt [cy + sin_phi·rx·cos(t) + cos_phi·ry·sin(t)]
        //       = −sin_phi·rx·sin(t) + cos_phi·ry·cos(t)
        let (s1, c1) = t1.sin_cos();
        let (s2, c2) = t2.sin_cos();

        let p1x = cx + cos_phi * rx * c1 - sin_phi * ry * s1;
        let p1y = cy + sin_phi * rx * c1 + cos_phi * ry * s1;
        let p4x = cx + cos_phi * rx * c2 - sin_phi * ry * s2;
        let p4y = cy + sin_phi * rx * c2 + cos_phi * ry * s2;

        let dp1x = -cos_phi * rx * s1 - sin_phi * ry * c1;
        let dp1y = -sin_phi * rx * s1 + cos_phi * ry * c1;
        let dp2x = -cos_phi * rx * s2 - sin_phi * ry * c2;
        let dp2y = -sin_phi * rx * s2 + cos_phi * ry * c2;

        let p2x = p1x + alpha * dp1x;
        let p2y = p1y + alpha * dp1y;
        let p3x = p4x - alpha * dp2x;
        let p3y = p4y - alpha * dp2y;

        flatten_cubic((p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y), out);
    }
}

// ---------------------------------------------------------------------------
// Shape primitives.
// ---------------------------------------------------------------------------

fn rect_polygon(x: f32, y: f32, w: f32, h: f32) -> Vec<(f32, f32)> {
    vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
}

/// Rounded-rectangle polygon.  Each of the four corner arcs is approximated
/// by `STEPS` straight-line segments (10 → ≤0.5° error at typical icon sizes).
fn rounded_rect_polygon(x: f32, y: f32, w: f32, h: f32, rx: f32, ry: f32) -> Vec<(f32, f32)> {
    const STEPS: usize = 10;
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(STEPS * 4 + 1);

    // Corners: (cx, cy, start_angle_degrees)
    // SVG rounded-rect corners go:
    //   top-right    → quarter-arc from 270° to 0°   (cx = x+w-rx, cy = y+ry)
    //   bottom-right → quarter-arc from 0°   to 90°  (cx = x+w-rx, cy = y+h-ry)
    //   bottom-left  → quarter-arc from 90°  to 180° (cx = x+rx,   cy = y+h-ry)
    //   top-left     → quarter-arc from 180° to 270° (cx = x+rx,   cy = y+ry)
    let corners: [(f32, f32, f32, f32); 4] = [
        (x + w - rx, y + ry,     270.0, 360.0),
        (x + w - rx, y + h - ry,   0.0,  90.0),
        (x + rx,     y + h - ry,  90.0, 180.0),
        (x + rx,     y + ry,     180.0, 270.0),
    ];

    for &(cx, cy, a_start, a_end) in &corners {
        for i in 0..=STEPS {
            let t = i as f32 / STEPS as f32;
            let angle = (a_start + t * (a_end - a_start)).to_radians();
            pts.push((cx + rx * angle.cos(), cy + ry * angle.sin()));
        }
    }

    // Close the polygon.
    if let Some(&first) = pts.first() {
        pts.push(first);
    }
    pts
}

fn ellipse_polygon(cx: f32, cy: f32, rx: f32, ry: f32) -> Vec<(f32, f32)> {
    // 36 segments gives sub-pixel error at icon sizes.
    let n = 36usize;
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let theta = (i as f32) * std::f32::consts::TAU / (n as f32);
        pts.push((cx + rx * theta.cos(), cy + ry * theta.sin()));
    }
    pts
}

fn thicken_line(x1: f32, y1: f32, x2: f32, y2: f32, w: f32) -> Vec<(f32, f32)> {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt().max(1e-3);
    let nx = -dy / len * w * 0.5;
    let ny = dx / len * w * 0.5;
    vec![
        (x1 + nx, y1 + ny),
        (x2 + nx, y2 + ny),
        (x2 - nx, y2 - ny),
        (x1 - nx, y1 - ny),
        (x1 + nx, y1 + ny),
    ]
}

/// Walk the top-level `SvgChild` list for `<linearGradient>` entries
/// and build a registry keyed by element id. The caller (conclave
/// during the SVG-walk step) serialises the gradient's `<stop>`
/// children into a `tb-stops="offset:color[:alpha];..."` attribute
/// so the rasterizer can read them without recursing into nested
/// children (our `SvgChild` is intentionally flat).
fn collect_linear_gradients(
    children: &[SvgChild<'_>],
) -> std::collections::HashMap<String, LinearGradientDef> {
    let mut out = std::collections::HashMap::new();
    for c in children {
        if !c.tag.eq_ignore_ascii_case("lineargradient") {
            continue;
        }
        let id = match child_attr(c, "id") {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => continue,
        };
        // Determine coordinate system: "objectBoundingBox" (default per
        // SVG spec) means x1/y1/x2/y2 are fractions of the shape's bbox.
        // "userSpaceOnUse" means raw user-space coordinates.
        let units_s = child_attr(c, "gradientUnits").unwrap_or("objectBoundingBox");
        let object_bounding_box = !units_s.eq_ignore_ascii_case("userSpaceOnUse");

        // Endpoints. Spec defaults for objectBoundingBox: x1=0, y1=0, x2=1, y2=0
        // (a left-to-right gradient across the full bbox width). For
        // userSpaceOnUse the same defaults are 0% / 100% — i.e., (0,0)→(1,0)
        // fractions still make sense as fallback.
        let (x1, y1, x2, y2) = if object_bounding_box {
            // In objectBoundingBox mode the spec says bare numbers are fractions
            // (0..1) and percentages divide by 100. Parse them both ways.
            let parse_obb = |s: &str, fallback: f32| -> f32 {
                let s = s.trim();
                if let Some(pct_s) = s.strip_suffix('%') {
                    pct_s.trim().parse::<f32>().map(|v| v / 100.0).unwrap_or(fallback)
                } else {
                    s.parse::<f32>().unwrap_or(fallback)
                }
            };
            let x1 = child_attr(c, "x1").map(|s| parse_obb(s, 0.0)).unwrap_or(0.0);
            let y1 = child_attr(c, "y1").map(|s| parse_obb(s, 0.0)).unwrap_or(0.0);
            let x2 = child_attr(c, "x2").map(|s| parse_obb(s, 1.0)).unwrap_or(1.0);
            let y2 = child_attr(c, "y2").map(|s| parse_obb(s, 0.0)).unwrap_or(0.0);
            (x1, y1, x2, y2)
        } else {
            // userSpaceOnUse: plain numbers are user-space coordinates.
            let x1 = num_attr(c, "x1", 0.0);
            let y1 = num_attr(c, "y1", 0.0);
            let x2 = num_attr(c, "x2", 1.0);
            let y2 = num_attr(c, "y2", 0.0);
            (x1, y1, x2, y2)
        };

        let stops = parse_tb_stops(child_attr(c, "tb-stops").unwrap_or(""));
        if stops.is_empty() {
            continue;
        }
        out.insert(
            id,
            LinearGradientDef {
                x1,
                y1,
                x2,
                y2,
                stops,
                object_bounding_box,
            },
        );
    }
    out
}

/// Parse the `tb-stops="off1:col1[:a1];off2:col2[:a2];..."` form into
/// an ordered list of `GradientStop`. Stops are sorted ascending by
/// offset and any offsets outside `[0, 1]` are clamped.
fn parse_tb_stops(s: &str) -> Vec<GradientStop> {
    let mut stops: Vec<GradientStop> = Vec::new();
    for part in s.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let mut bits = part.split(':');
        let off_str = bits.next().unwrap_or("");
        let col_str = bits.next().unwrap_or("");
        let alpha_str = bits.next();
        let Ok(mut off) = off_str.parse::<f32>() else {
            continue;
        };
        off = off.clamp(0.0, 1.0);
        // Accept any SVG color syntax (hex, named, rgb(), etc.) for stop colors.
        let mut rgba = match parse_fill(Some(col_str), None) {
            Some(c) => c,
            None => continue,
        };
        if let Some(a_s) = alpha_str {
            if let Ok(a) = a_s.parse::<f32>() {
                rgba[3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        stops.push(GradientStop { offset: off, rgba });
    }
    stops.sort_by(|a, b| {
        a.offset
            .partial_cmp(&b.offset)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    stops
}

/// Extract the `id` from a `fill="url(#id)"` value (also accepts
/// `url('#id')` / `url("#id")`). Returns the id without the `#`.
fn extract_url_id(s: &str) -> Option<&str> {
    let s = s.trim();
    let inner = s.strip_prefix("url(")?.strip_suffix(')')?.trim();
    let inner = inner
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| inner.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(inner);
    inner.strip_prefix('#')
}

fn parse_points(s: &str) -> Vec<(f32, f32)> {
    let mut nums = Vec::new();
    let mut buf = String::new();
    for c in s.chars().chain(core::iter::once(' ')) {
        if c == '-' && !buf.is_empty() && !buf.ends_with('e') && !buf.ends_with('E') {
            if let Ok(n) = buf.parse::<f32>() {
                nums.push(n);
            }
            buf.clear();
            buf.push('-');
        } else if c == ',' || c.is_whitespace() {
            if !buf.is_empty() {
                if let Ok(n) = buf.parse::<f32>() {
                    nums.push(n);
                }
                buf.clear();
            }
        } else {
            buf.push(c);
        }
    }
    let mut out = Vec::with_capacity(nums.len() / 2);
    for chunk in nums.chunks_exact(2) {
        out.push((chunk[0], chunk[1]));
    }
    out
}

// ---------------------------------------------------------------------------
// Path command tokenizer.
// ---------------------------------------------------------------------------

struct PathTokens<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> PathTokens<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c == b' ' || c == b',' || c == b'\n' || c == b'\r' || c == b'\t' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
    fn command(&mut self) -> Option<char> {
        self.skip_ws();
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c.is_ascii_alphabetic() {
                self.pos += 1;
                return Some(c as char);
            }
            // A number after the previous command implies repetition
            // of that command — handled by `maybe_pair`/`maybe_num`
            // returning Some. The outer loop in `path_to_polygons`
            // treats any digit/sign/dot as continuation, so we only
            // get here when there really is no more data.
            return None;
        }
        None
    }
    fn maybe_num(&mut self) -> Option<f32> {
        self.skip_ws();
        let start = self.pos;
        let mut saw_digit = false;
        // Optional sign.
        if self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-')
        {
            self.pos += 1;
        }
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
            saw_digit = true;
        }
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
                saw_digit = true;
            }
        }
        // Optional exponent.
        if saw_digit
            && self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E')
        {
            self.pos += 1;
            if self.pos < self.bytes.len()
                && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-')
            {
                self.pos += 1;
            }
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if !saw_digit {
            self.pos = start;
            return None;
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
    }
    fn num(&mut self) -> Option<f32> {
        self.maybe_num()
    }
    fn pair(&mut self) -> Option<(f32, f32)> {
        let x = self.num()?;
        let y = self.num()?;
        Some((x, y))
    }
    fn maybe_pair(&mut self) -> Option<(f32, f32)> {
        let save = self.pos;
        let x = match self.maybe_num() {
            Some(v) => v,
            None => return None,
        };
        match self.maybe_num() {
            Some(y) => Some((x, y)),
            None => {
                self.pos = save;
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fill / colour parsing.
// ---------------------------------------------------------------------------

fn parse_fill(s: Option<&str>, current_color: Option<[u8; 4]>) -> Option<[u8; 4]> {
    let s = s?.trim();
    if s.eq_ignore_ascii_case("none") {
        return Some([0, 0, 0, 0]);
    }
    if s.eq_ignore_ascii_case("transparent") {
        return Some([0, 0, 0, 0]);
    }
    if let Some(rgb) = parse_hex_color(s) {
        return Some(rgb);
    }
    if let Some(rgb) = parse_rgb_color(s) {
        return Some(rgb);
    }
    if let Some(rgb) = parse_hsl_color(s) {
        return Some(rgb);
    }
    // Full CSS named-color table (148 named colors per CSS Color Level 4).
    let lc = s.to_ascii_lowercase();
    let rgb: [u8; 4] = match lc.as_str() {
        "currentcolor" => return Some(current_color.unwrap_or([0, 0, 0, 255])),
        // CSS basic colors
        "black"                => [0, 0, 0, 255],
        "silver"               => [192, 192, 192, 255],
        "gray" | "grey"        => [128, 128, 128, 255],
        "white"                => [255, 255, 255, 255],
        "maroon"               => [128, 0, 0, 255],
        "red"                  => [255, 0, 0, 255],
        "purple"               => [128, 0, 128, 255],
        "fuchsia" | "magenta"  => [255, 0, 255, 255],
        "green"                => [0, 128, 0, 255],
        "lime"                 => [0, 255, 0, 255],
        "olive"                => [128, 128, 0, 255],
        "yellow"               => [255, 255, 0, 255],
        "navy"                 => [0, 0, 128, 255],
        "blue"                 => [0, 0, 255, 255],
        "teal"                 => [0, 128, 128, 255],
        "aqua" | "cyan"        => [0, 255, 255, 255],
        // Extended named colors
        "aliceblue"            => [240, 248, 255, 255],
        "antiquewhite"         => [250, 235, 215, 255],
        "aquamarine"           => [127, 255, 212, 255],
        "azure"                => [240, 255, 255, 255],
        "beige"                => [245, 245, 220, 255],
        "bisque"               => [255, 228, 196, 255],
        "blanchedalmond"       => [255, 235, 205, 255],
        "blueviolet"           => [138, 43, 226, 255],
        "brown"                => [165, 42, 42, 255],
        "burlywood"            => [222, 184, 135, 255],
        "cadetblue"            => [95, 158, 160, 255],
        "chartreuse"           => [127, 255, 0, 255],
        "chocolate"            => [210, 105, 30, 255],
        "coral"                => [255, 127, 80, 255],
        "cornflowerblue"       => [100, 149, 237, 255],
        "cornsilk"             => [255, 248, 220, 255],
        "crimson"              => [220, 20, 60, 255],
        "darkblue"             => [0, 0, 139, 255],
        "darkcyan"             => [0, 139, 139, 255],
        "darkgoldenrod"        => [184, 134, 11, 255],
        "darkgray" | "darkgrey"=> [169, 169, 169, 255],
        "darkgreen"            => [0, 100, 0, 255],
        "darkkhaki"            => [189, 183, 107, 255],
        "darkmagenta"          => [139, 0, 139, 255],
        "darkolivegreen"       => [85, 107, 47, 255],
        "darkorange"           => [255, 140, 0, 255],
        "darkorchid"           => [153, 50, 204, 255],
        "darkred"              => [139, 0, 0, 255],
        "darksalmon"           => [233, 150, 122, 255],
        "darkseagreen"         => [143, 188, 143, 255],
        "darkslateblue"        => [72, 61, 139, 255],
        "darkslategray" | "darkslategrey" => [47, 79, 79, 255],
        "darkturquoise"        => [0, 206, 209, 255],
        "darkviolet"           => [148, 0, 211, 255],
        "deeppink"             => [255, 20, 147, 255],
        "deepskyblue"          => [0, 191, 255, 255],
        "dimgray" | "dimgrey"  => [105, 105, 105, 255],
        "dodgerblue"           => [30, 144, 255, 255],
        "firebrick"            => [178, 34, 34, 255],
        "floralwhite"          => [255, 250, 240, 255],
        "forestgreen"          => [34, 139, 34, 255],
        "gainsboro"            => [220, 220, 220, 255],
        "ghostwhite"           => [248, 248, 255, 255],
        "gold"                 => [255, 215, 0, 255],
        "goldenrod"            => [218, 165, 32, 255],
        "greenyellow"          => [173, 255, 47, 255],
        "honeydew"             => [240, 255, 240, 255],
        "hotpink"              => [255, 105, 180, 255],
        "indianred"            => [205, 92, 92, 255],
        "indigo"               => [75, 0, 130, 255],
        "ivory"                => [255, 255, 240, 255],
        "khaki"                => [240, 230, 140, 255],
        "lavender"             => [230, 230, 250, 255],
        "lavenderblush"        => [255, 240, 245, 255],
        "lawngreen"            => [124, 252, 0, 255],
        "lemonchiffon"         => [255, 250, 205, 255],
        "lightblue"            => [173, 216, 230, 255],
        "lightcoral"           => [240, 128, 128, 255],
        "lightcyan"            => [224, 255, 255, 255],
        "lightgoldenrodyellow" => [250, 250, 210, 255],
        "lightgray" | "lightgrey" => [211, 211, 211, 255],
        "lightgreen"           => [144, 238, 144, 255],
        "lightpink"            => [255, 182, 193, 255],
        "lightsalmon"          => [255, 160, 122, 255],
        "lightseagreen"        => [32, 178, 170, 255],
        "lightskyblue"         => [135, 206, 250, 255],
        "lightslategray" | "lightslategrey" => [119, 136, 153, 255],
        "lightsteelblue"       => [176, 196, 222, 255],
        "lightyellow"          => [255, 255, 224, 255],
        "limegreen"            => [50, 205, 50, 255],
        "linen"                => [250, 240, 230, 255],
        "mediumaquamarine"     => [102, 205, 170, 255],
        "mediumblue"           => [0, 0, 205, 255],
        "mediumorchid"         => [186, 85, 211, 255],
        "mediumpurple"         => [147, 112, 219, 255],
        "mediumseagreen"       => [60, 179, 113, 255],
        "mediumslateblue"      => [123, 104, 238, 255],
        "mediumspringgreen"    => [0, 250, 154, 255],
        "mediumturquoise"      => [72, 209, 204, 255],
        "mediumvioletred"      => [199, 21, 133, 255],
        "midnightblue"         => [25, 25, 112, 255],
        "mintcream"            => [245, 255, 250, 255],
        "mistyrose"            => [255, 228, 225, 255],
        "moccasin"             => [255, 228, 181, 255],
        "navajowhite"          => [255, 222, 173, 255],
        "oldlace"              => [253, 245, 230, 255],
        "olivedrab"            => [107, 142, 35, 255],
        "orange"               => [255, 165, 0, 255],
        "orangered"            => [255, 69, 0, 255],
        "orchid"               => [218, 112, 214, 255],
        "palegoldenrod"        => [238, 232, 170, 255],
        "palegreen"            => [152, 251, 152, 255],
        "paleturquoise"        => [175, 238, 238, 255],
        "palevioletred"        => [219, 112, 147, 255],
        "papayawhip"           => [255, 239, 213, 255],
        "peachpuff"            => [255, 218, 185, 255],
        "peru"                 => [205, 133, 63, 255],
        "pink"                 => [255, 192, 203, 255],
        "plum"                 => [221, 160, 221, 255],
        "powderblue"           => [176, 224, 230, 255],
        "rebeccapurple"        => [102, 51, 153, 255],
        "rosybrown"            => [188, 143, 143, 255],
        "royalblue"            => [65, 105, 225, 255],
        "saddlebrown"          => [139, 69, 19, 255],
        "salmon"               => [250, 128, 114, 255],
        "sandybrown"           => [244, 164, 96, 255],
        "seagreen"             => [46, 139, 87, 255],
        "seashell"             => [255, 245, 238, 255],
        "sienna"               => [160, 82, 45, 255],
        "skyblue"              => [135, 206, 235, 255],
        "slateblue"            => [106, 90, 205, 255],
        "slategray" | "slategrey" => [112, 128, 144, 255],
        "snow"                 => [255, 250, 250, 255],
        "springgreen"          => [0, 255, 127, 255],
        "steelblue"            => [70, 130, 180, 255],
        "tan"                  => [210, 180, 140, 255],
        "thistle"              => [216, 191, 216, 255],
        "tomato"               => [255, 99, 71, 255],
        "turquoise"            => [64, 224, 208, 255],
        "violet"               => [238, 130, 238, 255],
        "wheat"                => [245, 222, 179, 255],
        "whitesmoke"           => [245, 245, 245, 255],
        "yellowgreen"          => [154, 205, 50, 255],
        _ => return None,
    };
    Some(rgb)
}

/// Parse `rgb(r, g, b)` and `rgba(r, g, b, a)`.
/// Supports integer values (0-255) and percentage values (0%-100%).
fn parse_rgb_color(s: &str) -> Option<[u8; 4]> {
    let s = s.trim();
    let inner = if let Some(inner) = s.strip_prefix("rgba(").and_then(|i| i.strip_suffix(')')) {
        inner
    } else if let Some(inner) = s.strip_prefix("rgb(").and_then(|i| i.strip_suffix(')')) {
        inner
    } else {
        return None;
    };
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let parse_channel = |p: &str| -> Option<u8> {
        let p = p.trim();
        if let Some(pct_s) = p.strip_suffix('%') {
            let pct: f32 = pct_s.trim().parse().ok()?;
            Some((pct.clamp(0.0, 100.0) * 2.55).round() as u8)
        } else {
            let v: f32 = p.parse().ok()?;
            Some(v.clamp(0.0, 255.0).round() as u8)
        }
    };
    let r = parse_channel(parts[0])?;
    let g = parse_channel(parts[1])?;
    let b = parse_channel(parts[2])?;
    let a = if parts.len() >= 4 {
        let ap = parts[3].trim();
        let av: f32 = ap.strip_suffix('%')
            .and_then(|p| p.trim().parse::<f32>().ok())
            .map(|p| p / 100.0)
            .or_else(|| ap.parse::<f32>().ok())
            .unwrap_or(1.0);
        (av.clamp(0.0, 1.0) * 255.0).round() as u8
    } else {
        255
    };
    Some([r, g, b, a])
}

/// Parse `hsl(h, s%, l%)` and `hsla(h, s%, l%, a)` and convert to RGBA.
fn parse_hsl_color(s: &str) -> Option<[u8; 4]> {
    let s = s.trim();
    let inner = if let Some(inner) = s.strip_prefix("hsla(").and_then(|i| i.strip_suffix(')')) {
        inner
    } else if let Some(inner) = s.strip_prefix("hsl(").and_then(|i| i.strip_suffix(')')) {
        inner
    } else {
        return None;
    };
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let h: f32 = parts[0].trim().parse().ok()?;
    let s_pct: f32 = parts[1].trim().trim_end_matches('%').parse().ok()?;
    let l_pct: f32 = parts[2].trim().trim_end_matches('%').parse().ok()?;
    let a = if parts.len() >= 4 {
        let ap = parts[3].trim();
        let av: f32 = ap.strip_suffix('%')
            .and_then(|p| p.trim().parse::<f32>().ok())
            .map(|p| p / 100.0)
            .or_else(|| ap.parse::<f32>().ok())
            .unwrap_or(1.0);
        (av.clamp(0.0, 1.0) * 255.0).round() as u8
    } else {
        255
    };
    // Convert HSL → RGB (CSS algorithm).
    let h = ((h % 360.0) + 360.0) % 360.0;
    let s = s_pct.clamp(0.0, 100.0) / 100.0;
    let l = l_pct.clamp(0.0, 100.0) / 100.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let to_u8 = |v: f32| ((v + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    Some([to_u8(r1), to_u8(g1), to_u8(b1), a])
}

fn parse_hex_color(s: &str) -> Option<[u8; 4]> {
    let s = s.strip_prefix('#')?;
    let bytes = s.as_bytes();
    match bytes.len() {
        3 => {
            // #rgb → #rrggbb (each digit doubled).
            let r = hex_digit(bytes[0])?;
            let g = hex_digit(bytes[1])?;
            let b = hex_digit(bytes[2])?;
            Some([r * 17, g * 17, b * 17, 255])
        }
        4 => {
            // #rgba → expand each digit to two.
            let r = hex_digit(bytes[0])?;
            let g = hex_digit(bytes[1])?;
            let b = hex_digit(bytes[2])?;
            let a = hex_digit(bytes[3])?;
            Some([r * 17, g * 17, b * 17, a * 17])
        }
        6 => {
            let r = (hex_digit(bytes[0])? << 4) | hex_digit(bytes[1])?;
            let g = (hex_digit(bytes[2])? << 4) | hex_digit(bytes[3])?;
            let b = (hex_digit(bytes[4])? << 4) | hex_digit(bytes[5])?;
            Some([r, g, b, 255])
        }
        8 => {
            let r = (hex_digit(bytes[0])? << 4) | hex_digit(bytes[1])?;
            let g = (hex_digit(bytes[2])? << 4) | hex_digit(bytes[3])?;
            let b = (hex_digit(bytes[4])? << 4) | hex_digit(bytes[5])?;
            let a = (hex_digit(bytes[6])? << 4) | hex_digit(bytes[7])?;
            Some([r, g, b, a])
        }
        _ => None,
    }
}

fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Scanline polygon fill — non-zero winding rule, BGRA output.
// ---------------------------------------------------------------------------

/// Clip a polygon against one half-plane (Sutherland-Hodgman). `inside`
/// tests a vertex; `intersect` finds where edge a→b crosses the boundary
/// (only called when the two endpoints straddle it, so the coordinate
/// along the clip axis differs and the parametric `t` is finite).
fn clip_half(
    poly: &[(f32, f32)],
    inside: impl Fn((f32, f32)) -> bool,
    intersect: impl Fn((f32, f32), (f32, f32)) -> (f32, f32),
) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    let n = poly.len();
    if n == 0 {
        return out;
    }
    for i in 0..n {
        let cur = poly[i];
        let prev = poly[(i + n - 1) % n];
        let cur_in = inside(cur);
        let prev_in = inside(prev);
        if cur_in {
            if !prev_in {
                out.push(intersect(prev, cur));
            }
            out.push(cur);
        } else if prev_in {
            out.push(intersect(prev, cur));
        }
    }
    out
}

/// Clip a polygon to the axis-aligned rect `(x0, y0, x1, y1)`. Used to
/// confine a nested-`<svg>` viewport's shapes to its cell so a single-
/// sheet SVG sprite (Wikipedia's wordmark + sister-project icons) can't
/// bleed one symbol's pixels into the crop window of another.
fn clip_polygon_to_rect(poly: &[(f32, f32)], rect: (f32, f32, f32, f32)) -> Vec<(f32, f32)> {
    let (x0, y0, x1, y1) = rect;
    let mut v = poly.to_vec();
    v = clip_half(
        &v,
        |p| p.0 >= x0,
        |a, b| {
            let t = (x0 - a.0) / (b.0 - a.0);
            (x0, a.1 + t * (b.1 - a.1))
        },
    );
    v = clip_half(
        &v,
        |p| p.0 <= x1,
        |a, b| {
            let t = (x1 - a.0) / (b.0 - a.0);
            (x1, a.1 + t * (b.1 - a.1))
        },
    );
    v = clip_half(
        &v,
        |p| p.1 >= y0,
        |a, b| {
            let t = (y0 - a.1) / (b.1 - a.1);
            (a.0 + t * (b.0 - a.0), y0)
        },
    );
    v = clip_half(
        &v,
        |p| p.1 <= y1,
        |a, b| {
            let t = (y1 - a.1) / (b.1 - a.1);
            (a.0 + t * (b.0 - a.0), y1)
        },
    );
    v
}

fn fill_polygon_nonzero(
    pixels: &mut [u32],
    width: i32,
    height: i32,
    poly: &[(f32, f32)],
    rgba: [u8; 4],
) {
    fill_polygons_nonzero(pixels, width, height, &[poly.to_vec()], rgba);
}

fn fill_polygons_nonzero(
    pixels: &mut [u32],
    width: i32,
    height: i32,
    polys: &[Vec<(f32, f32)>],
    rgba: [u8; 4],
) {
    if polys.iter().all(|poly| poly.len() < 3) {
        return;
    }
    // Pack the colour once. BGRA in memory because that's what the
    // cv_gfx bitmap blit consumes.
    let color = pack_bgra(rgba);

    let Some((ymin, ymax)) = polygon_y_bounds(polys) else {
        return;
    };
    let y_start = ymin.floor().max(0.0) as i32;
    let y_end = ymax.ceil().min(height as f32) as i32;

    for y in y_start..y_end {
        // Compute the centre of the scanline.
        let scan_y = y as f32 + 0.5;
        // For each polygon edge, find its x at scan_y and a signed
        // winding contribution (+1 for downward edges, -1 for upward).
        let mut crossings: Vec<(f32, i32)> = Vec::new();
        for poly in polys {
            collect_winding_crossings(poly, scan_y, &mut crossings);
        }
        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        // Walk left-to-right, accumulating winding number. Whenever
        // the winding is non-zero the span is inside the compound path.
        let mut winding = 0i32;
        let mut span_start: Option<f32> = None;
        for (x, w) in crossings {
            let prev_inside = winding != 0;
            winding += w;
            let new_inside = winding != 0;
            if !prev_inside && new_inside {
                span_start = Some(x);
            }
            if prev_inside && !new_inside {
                if let Some(sx) = span_start.take() {
                    fill_span(pixels, width, height, sx, x, y, color, rgba[3]);
                }
            }
        }
    }
}

fn fill_polygons_evenodd(
    pixels: &mut [u32],
    width: i32,
    height: i32,
    polys: &[Vec<(f32, f32)>],
    rgba: [u8; 4],
) {
    if polys.iter().all(|poly| poly.len() < 3) {
        return;
    }
    let color = pack_bgra(rgba);
    let Some((ymin, ymax)) = polygon_y_bounds(polys) else {
        return;
    };
    let y_start = ymin.floor().max(0.0) as i32;
    let y_end = ymax.ceil().min(height as f32) as i32;

    for y in y_start..y_end {
        let scan_y = y as f32 + 0.5;
        let mut crossings: Vec<f32> = Vec::new();
        for poly in polys {
            collect_evenodd_crossings(poly, scan_y, &mut crossings);
        }
        crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let mut i = 0usize;
        while i + 1 < crossings.len() {
            fill_span(
                pixels,
                width,
                height,
                crossings[i],
                crossings[i + 1],
                y,
                color,
                rgba[3],
            );
            i += 2;
        }
    }
}

fn polygon_y_bounds(polys: &[Vec<(f32, f32)>]) -> Option<(f32, f32)> {
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    let mut any = false;
    for poly in polys {
        if poly.len() < 3 {
            continue;
        }
        for &(_, y) in poly {
            any = true;
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
    }
    any.then_some((ymin, ymax))
}

fn collect_winding_crossings(poly: &[(f32, f32)], scan_y: f32, out: &mut Vec<(f32, i32)>) {
    if poly.len() < 3 {
        return;
    }
    for i in 0..poly.len() {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % poly.len()];
        if let Some(xc) = edge_crossing_x(x0, y0, x1, y1, scan_y) {
            let wind = if y0 < y1 { 1 } else { -1 };
            out.push((xc, wind));
        }
    }
}

fn collect_evenodd_crossings(poly: &[(f32, f32)], scan_y: f32, out: &mut Vec<f32>) {
    if poly.len() < 3 {
        return;
    }
    for i in 0..poly.len() {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % poly.len()];
        if let Some(xc) = edge_crossing_x(x0, y0, x1, y1, scan_y) {
            out.push(xc);
        }
    }
}

fn edge_crossing_x(x0: f32, y0: f32, x1: f32, y1: f32, scan_y: f32) -> Option<f32> {
    // Skip horizontal edges entirely.
    if (y0 - y1).abs() < 1e-6 {
        return None;
    }
    let (low_y, high_y) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
    // A scanline at scan_y crosses this edge iff low_y <= scan_y < high_y.
    // The half-open interval ensures we don't double-count at shared vertices.
    if scan_y < low_y || scan_y >= high_y {
        return None;
    }
    let t = (scan_y - y0) / (y1 - y0);
    Some(x0 + t * (x1 - x0))
}

fn pack_bgra(c: [u8; 4]) -> u32 {
    // cv_gfx::Bitmap stores u32 BGRA little-endian.
    (c[3] as u32) << 24 | (c[0] as u32) << 16 | (c[1] as u32) << 8 | (c[2] as u32)
}

fn fill_span(
    pixels: &mut [u32],
    width: i32,
    height: i32,
    x_start: f32,
    x_end: f32,
    y: i32,
    color: u32,
    alpha: u8,
) {
    if y < 0 || y >= height {
        return;
    }
    let xs = x_start.floor().max(0.0) as i32;
    let xe = x_end.ceil().min(width as f32) as i32;
    if xe <= xs {
        return;
    }
    let row = (y as usize) * (width as usize);
    if alpha == 255 {
        for x in xs..xe {
            pixels[row + x as usize] = color;
        }
    } else {
        let src_a = alpha as f32 / 255.0;
        let src_r = ((color >> 16) & 0xFF) as f32;
        let src_g = ((color >> 8) & 0xFF) as f32;
        let src_b = (color & 0xFF) as f32;
        for x in xs..xe {
            let i = row + x as usize;
            let dst = pixels[i];
            let dr = ((dst >> 16) & 0xFF) as f32;
            let dg = ((dst >> 8) & 0xFF) as f32;
            let db = (dst & 0xFF) as f32;
            let dst_a = ((dst >> 24) & 0xFF) as f32 / 255.0;
            let inv = 1.0 - src_a;
            let out_a = src_a + dst_a * inv;
            if out_a <= 0.0 {
                pixels[i] = 0;
                continue;
            }
            let r = ((src_r * src_a + dr * dst_a * inv) / out_a)
                .round()
                .clamp(0.0, 255.0) as u32;
            let g = ((src_g * src_a + dg * dst_a * inv) / out_a)
                .round()
                .clamp(0.0, 255.0) as u32;
            let b = ((src_b * src_a + db * dst_a * inv) / out_a)
                .round()
                .clamp(0.0, 255.0) as u32;
            let a = (out_a * 255.0).round().clamp(0.0, 255.0) as u32;
            pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

// ---------------------------------------------------------------------------
// Linear-gradient polygon fill — same scanline winding as the solid version,
// but each pixel gets a colour sampled along the gradient axis.
// ---------------------------------------------------------------------------

fn fill_polygon_nonzero_gradient(
    pixels: &mut [u32],
    width: i32,
    height: i32,
    poly: &[(f32, f32)],
    gx1: f32,
    gy1: f32,
    gx2: f32,
    gy2: f32,
    stops: &[GradientStop],
) {
    if poly.len() < 3 || stops.is_empty() {
        return;
    }

    // Axis vector and its squared length. A degenerate gradient
    // (both endpoints equal) collapses to a single stop colour —
    // fall back to a solid fill in that case.
    let dx = gx2 - gx1;
    let dy = gy2 - gy1;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        fill_polygon_nonzero(pixels, width, height, poly, stops[0].rgba);
        return;
    }
    let inv_len2 = 1.0 / len2;

    // Sort stops by offset just in case the input wasn't ordered.
    let mut sorted: Vec<GradientStop> = stops.to_vec();
    sorted.sort_by(|a, b| {
        a.offset
            .partial_cmp(&b.offset)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    // Vertical bounds.
    let mut ymin = poly[0].1;
    let mut ymax = poly[0].1;
    for &(_, y) in poly.iter() {
        if y < ymin {
            ymin = y;
        }
        if y > ymax {
            ymax = y;
        }
    }
    let y_start = ymin.floor().max(0.0) as i32;
    let y_end = ymax.ceil().min(height as f32) as i32;

    for y in y_start..y_end {
        let scan_y = y as f32 + 0.5;
        let mut crossings: Vec<(f32, i32)> = Vec::new();
        for i in 0..poly.len() {
            let (x0, y0) = poly[i];
            let (x1, y1) = poly[(i + 1) % poly.len()];
            if (y0 - y1).abs() < 1e-6 {
                continue;
            }
            let (low_y, high_y) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
            if scan_y < low_y || scan_y >= high_y {
                continue;
            }
            let t = (scan_y - y0) / (y1 - y0);
            let xc = x0 + t * (x1 - x0);
            let wind = if y0 < y1 { 1 } else { -1 };
            crossings.push((xc, wind));
        }
        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        let mut winding = 0i32;
        let mut span_start: Option<f32> = None;
        for (x, w) in crossings {
            let prev_inside = winding != 0;
            winding += w;
            let new_inside = winding != 0;
            if !prev_inside && new_inside {
                span_start = Some(x);
            }
            if prev_inside && !new_inside {
                if let Some(sx) = span_start.take() {
                    fill_span_gradient(
                        pixels, width, height, sx, x, y, gx1, gy1, dx, dy, inv_len2, &sorted,
                    );
                }
            }
        }
    }
}

fn fill_span_gradient(
    pixels: &mut [u32],
    width: i32,
    height: i32,
    x_start: f32,
    x_end: f32,
    y: i32,
    gx1: f32,
    gy1: f32,
    dx: f32,
    dy: f32,
    inv_len2: f32,
    stops: &[GradientStop],
) {
    if y < 0 || y >= height {
        return;
    }
    let xs = x_start.floor().max(0.0) as i32;
    let xe = x_end.ceil().min(width as f32) as i32;
    if xe <= xs {
        return;
    }
    let row = (y as usize) * (width as usize);
    let py = y as f32 + 0.5;
    for x in xs..xe {
        let px = x as f32 + 0.5;
        // Project (px,py) onto the gradient axis. t in [0,1] across the
        // segment from (gx1,gy1) to (gx2,gy2); clamped outside that.
        let t = ((px - gx1) * dx + (py - gy1) * dy) * inv_len2;
        let t = t.max(0.0).min(1.0);
        let rgba = sample_gradient(stops, t);
        let alpha = rgba[3];
        if alpha == 0 {
            continue;
        }
        let color = pack_bgra(rgba);
        let i = row + x as usize;
        if alpha == 255 {
            pixels[i] = color;
        } else {
            // Porter-Duff source-over in straight-alpha (same as fill_span).
            let src_a = alpha as f32 / 255.0;
            let src_r = ((color >> 16) & 0xFF) as f32;
            let src_g = ((color >> 8) & 0xFF) as f32;
            let src_b = (color & 0xFF) as f32;
            let dst = pixels[i];
            let dr = ((dst >> 16) & 0xFF) as f32;
            let dg = ((dst >> 8) & 0xFF) as f32;
            let db = (dst & 0xFF) as f32;
            let dst_a = ((dst >> 24) & 0xFF) as f32 / 255.0;
            let inv = 1.0 - src_a;
            let out_a = src_a + dst_a * inv;
            if out_a <= 0.0 {
                pixels[i] = 0;
                continue;
            }
            let r = ((src_r * src_a + dr * dst_a * inv) / out_a)
                .round()
                .clamp(0.0, 255.0) as u32;
            let g = ((src_g * src_a + dg * dst_a * inv) / out_a)
                .round()
                .clamp(0.0, 255.0) as u32;
            let b = ((src_b * src_a + db * dst_a * inv) / out_a)
                .round()
                .clamp(0.0, 255.0) as u32;
            let a = (out_a * 255.0).round().clamp(0.0, 255.0) as u32;
            pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

fn sample_gradient(stops: &[GradientStop], t: f32) -> [u8; 4] {
    // Stops are pre-sorted by offset. Clamp before the first and after
    // the last; otherwise interpolate linearly between the bracketing pair.
    if t <= stops[0].offset {
        return stops[0].rgba;
    }
    let last = &stops[stops.len() - 1];
    if t >= last.offset {
        return last.rgba;
    }
    for w in stops.windows(2) {
        let a = &w[0];
        let b = &w[1];
        if t >= a.offset && t <= b.offset {
            let span = b.offset - a.offset;
            if span <= 1e-6 {
                return b.rgba;
            }
            let k = (t - a.offset) / span;
            let lerp = |x: u8, y: u8| -> u8 {
                let xf = x as f32;
                let yf = y as f32;
                (xf + (yf - xf) * k).round().max(0.0).min(255.0) as u8
            };
            return [
                lerp(a.rgba[0], b.rgba[0]),
                lerp(a.rgba[1], b.rgba[1]),
                lerp(a.rgba[2], b.rgba[2]),
                lerp(a.rgba[3], b.rgba[3]),
            ];
        }
    }
    last.rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_path() {
        let polys = path_to_polygons("M 10 10 L 20 10 L 20 20 L 10 20 Z");
        assert_eq!(polys.len(), 1);
        assert!(polys[0].len() >= 4);
    }

    #[test]
    fn clip_polygon_confines_to_rect() {
        // A 0..100 square clipped to 10..40 in both axes → all vertices land
        // inside the clip rect (nested-<svg> viewport confinement).
        let square = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let clipped = clip_polygon_to_rect(&square, (10.0, 10.0, 40.0, 40.0));
        assert!(clipped.len() >= 4);
        for (x, y) in &clipped {
            assert!(*x >= 10.0 - 1e-3 && *x <= 40.0 + 1e-3, "x={x} out of clip");
            assert!(*y >= 10.0 - 1e-3 && *y <= 40.0 + 1e-3, "y={y} out of clip");
        }
        // A polygon entirely outside the clip rect is removed.
        let outside = vec![(200.0, 200.0), (300.0, 200.0), (300.0, 300.0)];
        assert!(clip_polygon_to_rect(&outside, (0.0, 0.0, 50.0, 50.0)).len() < 3);
    }

    #[test]
    fn rect_fills() {
        let svg = SvgAttrs {
            width: Some(10.0),
            height: Some(10.0),
            view_box: Some([0.0, 0.0, 10.0, 10.0]),
            fill: None,
            current_color: None,
        };
        let kids = [SvgChild {
            tag: "rect",
            attrs: &[
                ("x", "0"),
                ("y", "0"),
                ("width", "10"),
                ("height", "10"),
                ("fill", "#ff0000"),
            ],
        }];
        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        assert_eq!(img.width, 10);
        assert_eq!(img.height, 10);
        // Center pixel should be red.
        let center = img.pixels[5 * 10 + 5];
        let r = (center >> 16) & 0xFF;
        let g = (center >> 8) & 0xFF;
        let b = center & 0xFF;
        let a = (center >> 24) & 0xFF;
        assert_eq!((r, g, b, a), (255, 0, 0, 255));
    }

    #[test]
    fn circle_renders_non_empty() {
        let svg = SvgAttrs {
            width: Some(20.0),
            height: Some(20.0),
            view_box: Some([0.0, 0.0, 20.0, 20.0]),
            fill: Some([0, 0, 255, 255]),
            current_color: None,
        };
        let kids = [SvgChild {
            tag: "circle",
            attrs: &[("cx", "10"), ("cy", "10"), ("r", "8")],
        }];
        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        let center = img.pixels[10 * 20 + 10];
        let a = (center >> 24) & 0xFF;
        assert_eq!(a, 255, "centre of circle should be fully opaque");
    }

    #[test]
    fn empty_svg_errors() {
        let svg = SvgAttrs {
            width: Some(10.0),
            height: Some(10.0),
            view_box: None,
            fill: None,
            current_color: None,
        };
        let kids: [SvgChild; 0] = [];
        assert!(rasterize_svg_attrs(&svg, &kids).is_err());
    }

    #[test]
    fn currentcolor_uses_svg_computed_color() {
        let svg = SvgAttrs {
            width: Some(8.0),
            height: Some(8.0),
            view_box: Some([0.0, 0.0, 8.0, 8.0]),
            fill: None,
            current_color: Some([68, 71, 70, 255]),
        };
        let kids = [SvgChild {
            tag: "rect",
            attrs: &[
                ("x", "0"),
                ("y", "0"),
                ("width", "8"),
                ("height", "8"),
                ("fill", "currentColor"),
            ],
        }];

        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        let px = img.pixels[0];
        let b = px & 0xFF;
        let g = (px >> 8) & 0xFF;
        let r = (px >> 16) & 0xFF;
        let a = (px >> 24) & 0xFF;
        assert_eq!((r, g, b, a), (68, 71, 70, 255));
    }

    #[test]
    fn parse_transform_translate_then_scale() {
        let m = parse_transform("translate(10, 5) scale(2)").unwrap();
        // (1, 1) → scale → (2, 2) → translate → (12, 7)
        let (x, y) = apply_affine(&m, 1.0, 1.0);
        assert!((x - 12.0).abs() < 1e-4);
        assert!((y - 7.0).abs() < 1e-4);
    }

    #[test]
    fn transform_translate_moves_rect_in_render() {
        let svg = SvgAttrs {
            width: Some(8.0),
            height: Some(8.0),
            view_box: Some([0.0, 0.0, 8.0, 8.0]),
            fill: None,
            current_color: None,
        };
        let kids = [SvgChild {
            tag: "rect",
            attrs: &[
                ("x", "0"),
                ("y", "0"),
                ("width", "2"),
                ("height", "2"),
                ("fill", "#00ff00"),
                ("transform", "translate(4, 3)"),
            ],
        }];
        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        // After translate(4, 3) the rect covers (4..6, 3..5).
        let moved = img.pixels[3 * 8 + 4];
        let untouched = img.pixels[0];
        assert_ne!((moved >> 24) & 0xFF, 0);
        assert_eq!((untouched >> 24) & 0xFF, 0);
    }

    #[test]
    fn linear_gradient_interpolates_red_to_blue() {
        // 20x4 rect; gradient axis runs left→right across the same span.
        // Left edge should read as red, right edge as blue, middle as
        // something in between.
        let svg = SvgAttrs {
            width: Some(20.0),
            height: Some(4.0),
            view_box: Some([0.0, 0.0, 20.0, 4.0]),
            fill: None,
            current_color: None,
        };
        let kids = [
            SvgChild {
                tag: "linearGradient",
                attrs: &[
                    ("id", "g1"),
                    ("x1", "0"),
                    ("y1", "0"),
                    ("x2", "20"),
                    ("y2", "0"),
                    ("gradientUnits", "userSpaceOnUse"),
                    ("tb-stops", "0:#ff0000;1:#0000ff"),
                ],
            },
            SvgChild {
                tag: "rect",
                attrs: &[
                    ("x", "0"),
                    ("y", "0"),
                    ("width", "20"),
                    ("height", "4"),
                    ("fill", "url(#g1)"),
                ],
            },
        ];
        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        let px_left = img.pixels[2 * 20 + 0];
        let px_right = img.pixels[2 * 20 + 19];
        let r_left = (px_left >> 16) & 0xFF;
        let b_left = px_left & 0xFF;
        let r_right = (px_right >> 16) & 0xFF;
        let b_right = px_right & 0xFF;
        assert!(
            r_left > 200 && b_left < 60,
            "left should be ~red, got rgba=#{:08x}",
            px_left
        );
        assert!(
            r_right < 60 && b_right > 200,
            "right should be ~blue, got rgba=#{:08x}",
            px_right
        );
    }

    #[test]
    fn use_offsets_shift_shapes_before_rasterization() {
        let svg = SvgAttrs {
            width: Some(8.0),
            height: Some(8.0),
            view_box: Some([0.0, 0.0, 8.0, 8.0]),
            fill: None,
            current_color: None,
        };
        let kids = [SvgChild {
            tag: "rect",
            attrs: &[
                ("width", "2"),
                ("height", "2"),
                ("fill", "#00ff00"),
                ("tb-use-x", "4"),
                ("tb-use-y", "3"),
            ],
        }];

        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        let moved = img.pixels[3 * 8 + 4];
        let untouched = img.pixels[0];
        assert_ne!((moved >> 24) & 0xFF, 0);
        assert_eq!((untouched >> 24) & 0xFF, 0);
    }

    #[test]
    fn stroke_only_line_renders_visible_pixels() {
        let svg = SvgAttrs {
            width: Some(12.0),
            height: Some(12.0),
            view_box: Some([0.0, 0.0, 12.0, 12.0]),
            fill: None,
            current_color: None,
        };
        let kids = [SvgChild {
            tag: "line",
            attrs: &[
                ("x1", "1"),
                ("y1", "1"),
                ("x2", "11"),
                ("y2", "11"),
                ("fill", "none"),
                ("stroke", "#ff0000"),
                ("stroke-width", "2"),
            ],
        }];
        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        let px = img.pixels[6 * 12 + 6];
        assert_ne!((px >> 24) & 0xFF, 0, "stroked line should paint pixels");
    }

    #[test]
    fn fill_opacity_on_path_produces_low_alpha() {
        // Mirrors hyvechain.com's hex-grid data-URI SVG: a single path
        // with fill='#FFD700' fill-opacity='0.03'. After the walker
        // flattens the nested `<g>` groups, the path arrives at the
        // rasterizer with both attributes already on it. The output
        // should be barely-visible yellow (alpha ≈ 8), not solid
        // gold tiles.
        let svg = SvgAttrs {
            width: Some(28.0),
            height: Some(49.0),
            view_box: Some([0.0, 0.0, 28.0, 49.0]),
            fill: None,
            current_color: None,
        };
        let kids = [SvgChild {
            tag: "path",
            attrs: &[
                ("fill-rule", "evenodd"),
                ("fill", "#FFD700"),
                ("fill-opacity", "0.03"),
                (
                    "d",
                    "M13.99 9.25l13 7.5v15l-13 7.5L1 31.75v-15l12.99-7.5zM3 17.9v12.7l10.99 6.34 11-6.35V17.9l-11-6.34L3 17.9zM0 15l12.98-7.5V0h-2v6.35L0 12.69v2.3zm0 18.5L12.98 41v8h-2v-6.85L0 35.81v-2.3zM15 0v7.5L27.99 15H28v-2.31h-.01L17 6.35V0h-2zm0 49v-8l12.99-7.5H28v2.31h-.01L17 42.15V49h-2z",
                ),
            ],
        }];
        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        // The visual center of the main hex is a cut-out under even-odd
        // compound-path filling. If this is painted, the repeated
        // honeycomb background shows yellow centers instead of Chrome's
        // outline-only texture.
        let center = img.pixels[24 * 28 + 14];
        assert_eq!(
            (center >> 24) & 0xFF,
            0,
            "fill-rule=evenodd should leave the inner hex center transparent, got #{center:08x}"
        );
        // Pick a pixel known to fall inside the painted hex outline.
        let px = img.pixels[17 * 28 + 3];
        let a = (px >> 24) & 0xFF;
        // 3% of 255 ≈ 8. Allow [0, 30] to absorb antialiasing/round
        // errors but reject anything resembling full opacity (255).
        assert!(
            a <= 30,
            "fill-opacity 0.03 should yield alpha ≤ 30, got {} (pixel #{:08x})",
            a,
            px
        );
        assert!(
            ((px >> 16) & 0xFF) > 200 && ((px >> 8) & 0xFF) > 160,
            "low-alpha SVG fill should preserve straight-alpha gold RGB, got pixel #{px:08x}"
        );
    }

    #[test]
    fn style_attribute_stroke_and_opacity_are_honored() {
        let svg = SvgAttrs {
            width: Some(10.0),
            height: Some(10.0),
            view_box: Some([0.0, 0.0, 10.0, 10.0]),
            fill: None,
            current_color: None,
        };
        let kids = [SvgChild {
            tag: "rect",
            attrs: &[
                ("x", "1"),
                ("y", "1"),
                ("width", "8"),
                ("height", "8"),
                ("fill", "none"),
                ("style", "stroke:#00ff00;stroke-width:2;stroke-opacity:0.5"),
            ],
        }];
        let img = rasterize_svg_attrs(&svg, &kids).unwrap();
        let px = img.pixels[1 * 10 + 5];
        let a = (px >> 24) & 0xFF;
        let g = (px >> 8) & 0xFF;
        assert!(
            a > 100 && a < 200,
            "stroke opacity should affect alpha, got {}",
            a
        );
        assert!(g > 100, "stroke color should be greenish, got #{:08x}", px);
    }

    // SVG elliptical arc tests
    #[test]
    fn arc_semicircle_reaches_endpoint() {
        // A 90° arc from (100,0) to (0,100) centred at origin should produce
        // a polyline whose last point is close to (0, 100).
        let mut pts: Vec<(f32, f32)> = vec![(100.0, 0.0)];
        arc_to_polyline(
            (100.0, 0.0),
            100.0, 100.0,   // rx, ry
            0.0,             // x-rotation
            false,           // large-arc
            false,           // sweep (CCW)
            (0.0, 100.0),
            &mut pts,
        );
        let last = pts.last().unwrap();
        assert!(
            (last.0 - 0.0).abs() < 0.5 && (last.1 - 100.0).abs() < 0.5,
            "arc endpoint should be ~(0,100), got ({:.2},{:.2})",
            last.0,
            last.1
        );
    }

    #[test]
    fn arc_full_circle_produces_points() {
        // A full circle via two half-arcs: the combined polyline should be
        // non-trivial (more than 2 points, since the old straight-line stub
        // would give exactly 1 or 2).
        let mut pts: Vec<(f32, f32)> = vec![(100.0, 0.0)];
        // First half: (100,0) → (-100,0), large-arc=true, sweep=true
        arc_to_polyline(
            (100.0, 0.0),
            100.0, 100.0,
            0.0,
            true,
            true,
            (-100.0, 0.0),
            &mut pts,
        );
        // Second half: (-100,0) → (100,0), large-arc=true, sweep=true
        arc_to_polyline(
            (-100.0, 0.0),
            100.0, 100.0,
            0.0,
            true,
            true,
            (100.0, 0.0),
            &mut pts,
        );
        assert!(
            pts.len() > 5,
            "circle arc should produce many intermediate points, got {}",
            pts.len()
        );
    }

    #[test]
    fn arc_path_string_parses_and_curves() {
        // An SVG path with arc commands should yield more than 2 vertices
        // (proving bezier subdivision happened, not just a straight line).
        let polys = path_to_polygons("M 100 0 A 100 100 0 0 1 0 100");
        assert_eq!(polys.len(), 1, "should produce one open subpath");
        assert!(
            polys[0].len() > 2,
            "arc should subdivide into multiple points, got {}",
            polys[0].len()
        );
    }

    // ---- SVG percentage attribute tests (Bug 2 regression) ----

    /// parse_svg_length must resolve "50%" against its reference dimension.
    /// This is the core fix — "50%" of 200.0 should be 100.0, NOT 50.0.
    #[test]
    fn parse_svg_length_resolves_percentage() {
        assert_eq!(parse_svg_length("50%", 200.0), Some(100.0));
        assert_eq!(parse_svg_length("100%", 80.0), Some(80.0));
        assert_eq!(parse_svg_length("25%", 400.0), Some(100.0));
        // Non-percentage falls through to plain number.
        assert_eq!(parse_svg_length("42", 200.0), Some(42.0));
        assert_eq!(parse_svg_length("10px", 200.0), Some(10.0));
    }

    /// A circle with cx="50%" cy="50%" r="40%" should be centred on the
    /// viewport, not at x=50 y=50.  On a 100x100 SVG that means the centre
    /// is at pixel (50,50) and the circle fills most of the viewport.
    #[test]
    fn circle_percentage_cx_cy_r_are_resolved() {
        // 100x100 SVG, viewBox 0 0 100 100.
        // cx="50%" → 50, cy="50%" → 50, r="40%" → 40% of diagonal≈70.7 ≈ 28.28.
        let svg = SvgAttrs {
            width: Some(100.0),
            height: Some(100.0),
            view_box: Some([0.0, 0.0, 100.0, 100.0]),
            fill: Some([255, 0, 0, 255]),
            current_color: None,
        };
        let children = vec![SvgChild {
            tag: "circle",
            attrs: &[
                ("cx", "50%"),
                ("cy", "50%"),
                ("r", "40%"),
            ],
        }];
        let img = rasterize_svg_attrs(&svg, &children).expect("should rasterize");
        // Centre pixel (50,50) must be red — the circle is centred there.
        let centre = img.pixels[50 * 100 + 50];
        let r = ((centre >> 16) & 0xFF) as u8;
        let a = ((centre >> 24) & 0xFF) as u8;
        assert!(a > 0, "centre pixel must be painted, a={a}");
        assert!(r > 200, "centre pixel must be red (circle fill), r={r}");

        // Pixel at (0,0) must be transparent — the circle's percentage radius
        // keeps it well away from the top-left corner.
        let corner = img.pixels[0];
        let corner_a = ((corner >> 24) & 0xFF) as u8;
        assert_eq!(corner_a, 0, "corner must be transparent (circle centred at 50,50)");
    }

    /// A rect with x="10%" y="10%" width="80%" height="80%" should
    /// fill most of the viewport but leave the outer 10% margin transparent.
    #[test]
    fn rect_percentage_dimensions_are_resolved() {
        let svg = SvgAttrs {
            width: Some(100.0),
            height: Some(100.0),
            view_box: Some([0.0, 0.0, 100.0, 100.0]),
            fill: Some([0, 0, 255, 255]),
            current_color: None,
        };
        let children = vec![SvgChild {
            tag: "rect",
            attrs: &[
                ("x", "10%"),
                ("y", "10%"),
                ("width", "80%"),
                ("height", "80%"),
            ],
        }];
        let img = rasterize_svg_attrs(&svg, &children).expect("should rasterize");
        // Centre pixel must be blue.
        let centre = img.pixels[50 * 100 + 50];
        let b = (centre & 0xFF) as u8;
        let a = ((centre >> 24) & 0xFF) as u8;
        assert!(a > 0, "centre must be painted, a={a}");
        assert!(b > 200, "centre must be blue, b={b}");

        // Pixel at (5,5) is inside the 10% margin — must be transparent.
        let margin = img.pixels[5 * 100 + 5];
        let margin_a = ((margin >> 24) & 0xFF) as u8;
        assert_eq!(margin_a, 0, "margin pixel at (5,5) must be transparent");
    }
}
