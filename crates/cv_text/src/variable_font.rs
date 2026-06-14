//! Variable font support — `fvar` axis table + tuple interpolation.
//!
//! V1 implements:
//!   * `FvarAxis` records (tag, min, default, max).
//:   * `normalize_coord` — map a user value through the axis range.
//!   * `apply_avar` — apply Axis Variations (avar) segment map.
//!   * `interpolate_glyph_deltas` — tuple-variation linear blend
//!     across N axes for one glyph outline.
//!
//! That covers what the layout engine needs to support things like
//! `font-variation-settings: "wght" 600, "wdth" 90`. The full HVAR
//! advance-width metric lookup follows when we ship the variable
//! advance plumbing.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisTag(pub [u8; 4]);

impl AxisTag {
    pub fn from_str(s: &str) -> Self {
        let mut t = [b' '; 4];
        for (i, b) in s.bytes().take(4).enumerate() {
            t[i] = b;
        }
        Self(t)
    }
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).unwrap_or("????")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FvarAxis {
    pub tag: AxisTag,
    pub min: f32,
    pub default: f32,
    pub max: f32,
}

/// Map `user_value` to the spec's normalized [-1.0, 1.0] axis space.
pub fn normalize_coord(axis: FvarAxis, user_value: f32) -> f32 {
    let v = user_value.clamp(axis.min, axis.max);
    if v < axis.default {
        if axis.default == axis.min {
            0.0
        } else {
            (v - axis.default) / (axis.default - axis.min)
        }
    } else if v > axis.default {
        if axis.default == axis.max {
            0.0
        } else {
            (v - axis.default) / (axis.max - axis.default)
        }
    } else {
        0.0
    }
}

/// One avar segment map entry (`fromCoord` → `toCoord`).
#[derive(Debug, Clone, Copy)]
pub struct AvarSegment {
    pub from: f32,
    pub to: f32,
}

/// Apply an `avar` segment map to a normalized coord. Performs
/// piecewise-linear interpolation between sorted segments.
pub fn apply_avar(normalized: f32, segments: &[AvarSegment]) -> f32 {
    if segments.is_empty() {
        return normalized;
    }
    // Find the bracketing pair.
    for i in 1..segments.len() {
        if normalized <= segments[i].from {
            let a = segments[i - 1];
            let b = segments[i];
            let span = b.from - a.from;
            if span <= 0.0 {
                return a.to;
            }
            let t = (normalized - a.from) / span;
            return a.to + t * (b.to - a.to);
        }
    }
    // Past the last segment: clamp to its `to`.
    segments.last().unwrap().to
}

/// One glyph delta record from `gvar`. `deltas` is the per-point
/// (dx, dy) offset applied when the design axis sits at `peak`.
#[derive(Debug, Clone)]
pub struct TupleVariation {
    /// Per-axis peak coordinates that this tuple applies at fully.
    pub peak: Vec<f32>,
    /// Per-point (dx, dy) shifts in font units.
    pub deltas: Vec<(f32, f32)>,
}

/// Interpolate glyph point deltas across N axes for the requested
/// coordinate. Returns one (dx, dy) per point.
pub fn interpolate_glyph_deltas(
    coords: &[f32],
    base_points: &[(f32, f32)],
    variations: &[TupleVariation],
) -> Vec<(f32, f32)> {
    let mut out: Vec<(f32, f32)> = base_points.iter().copied().collect();
    for var in variations {
        let mut weight = 1.0;
        for (i, &c) in coords.iter().enumerate() {
            let p = var.peak.get(i).copied().unwrap_or(0.0);
            if p == 0.0 {
                continue;
            }
            // Triangular fade: full at p, zero at 0 and beyond.
            let axis_weight = if p > 0.0 {
                (c / p).clamp(0.0, 1.0)
            } else {
                (c / p).clamp(0.0, 1.0)
            };
            weight *= axis_weight;
        }
        if weight == 0.0 {
            continue;
        }
        for (i, &(dx, dy)) in var.deltas.iter().enumerate() {
            if let Some(p) = out.get_mut(i) {
                p.0 += dx * weight;
                p.1 += dy * weight;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight_axis() -> FvarAxis {
        FvarAxis {
            tag: AxisTag::from_str("wght"),
            min: 100.0,
            default: 400.0,
            max: 900.0,
        }
    }

    #[test]
    fn normalize_default_is_zero() {
        assert_eq!(normalize_coord(weight_axis(), 400.0), 0.0);
    }

    #[test]
    fn normalize_max_is_one() {
        assert!((normalize_coord(weight_axis(), 900.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_min_is_minus_one() {
        assert!((normalize_coord(weight_axis(), 100.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_clamps_out_of_range() {
        let v = normalize_coord(weight_axis(), 1200.0);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn avar_identity_is_passthrough() {
        let segs = [
            AvarSegment {
                from: -1.0,
                to: -1.0,
            },
            AvarSegment { from: 0.0, to: 0.0 },
            AvarSegment { from: 1.0, to: 1.0 },
        ];
        assert_eq!(apply_avar(0.5, &segs), 0.5);
    }

    #[test]
    fn avar_remaps_curve() {
        // Squish 0..1 into 0..0.5.
        let segs = [
            AvarSegment { from: 0.0, to: 0.0 },
            AvarSegment { from: 1.0, to: 0.5 },
        ];
        let v = apply_avar(0.5, &segs);
        assert!((v - 0.25).abs() < 1e-6);
    }

    #[test]
    fn glyph_deltas_with_zero_coords_returns_base() {
        let base = vec![(10.0, 20.0), (30.0, 40.0)];
        let vars = vec![TupleVariation {
            peak: vec![1.0],
            deltas: vec![(5.0, 0.0), (5.0, 0.0)],
        }];
        let out = interpolate_glyph_deltas(&[0.0], &base, &vars);
        assert_eq!(out, base);
    }

    #[test]
    fn glyph_deltas_at_peak_fully_apply() {
        let base = vec![(0.0, 0.0); 2];
        let vars = vec![TupleVariation {
            peak: vec![1.0],
            deltas: vec![(2.0, -1.0), (3.0, 4.0)],
        }];
        let out = interpolate_glyph_deltas(&[1.0], &base, &vars);
        assert_eq!(out, vec![(2.0, -1.0), (3.0, 4.0)]);
    }

    #[test]
    fn glyph_deltas_half_axis_blends_half() {
        let base = vec![(0.0, 0.0)];
        let vars = vec![TupleVariation {
            peak: vec![1.0],
            deltas: vec![(10.0, 0.0)],
        }];
        let out = interpolate_glyph_deltas(&[0.5], &base, &vars);
        assert!((out[0].0 - 5.0).abs() < 1e-6);
    }
}
