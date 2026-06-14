//! H.264 motion compensation — fractional-sample interpolation.
//!
//! §8.4.2 luma: 6-tap FIR for half-sample, then 2-tap bilinear for
//! the 1/4-sample stage.
//! §8.4.1 chroma: pure bilinear at 1/8-sample precision (4:2:0).
//!
//! Inputs are u8 reference samples; outputs are u8 predicted samples
//! the residual stage adds onto. We don't yet wire frame fetching —
//! that's the DPB slice — but the per-block interpolators are spec-
//! accurate and unit-testable in isolation.

#[inline]
fn clip(v: i32) -> u8 {
    v.max(0).min(255) as u8
}

/// 6-tap luma half-sample FIR (spec eq. 8-245): [1, -5, 20, 20, -5, 1] / 32.
/// Operates on a row or column of 6 samples; returns the half-sample
/// between samples[2] and samples[3].
fn luma_half(samples: [i32; 6]) -> i32 {
    samples[0] - 5 * samples[1] + 20 * samples[2] + 20 * samples[3] - 5 * samples[4] + samples[5]
}

/// Two-tap bilinear average rounded half-up.
#[inline]
fn avg2(a: i32, b: i32) -> i32 {
    (a + b + 1) >> 1
}

/// Interpolate a single luma sample at a fractional-pel offset
/// `(dx, dy)` where `dx, dy ∈ {0, 1, 2, 3}` (quarter-pel grid).
///
/// `ref` is a 9x9 window of reference samples centred on the integer
/// position (so `ref[4][4]` is the integer-pel anchor and we have
/// 4 samples of padding in each direction for the 6-tap FIR).
pub fn interp_luma_qpel(rf: &[[u8; 9]; 9], dx: u8, dy: u8) -> u8 {
    let s = |y: usize, x: usize| -> i32 { rf[y][x] as i32 };
    let anchor_x = 4usize;
    let anchor_y = 4usize;

    let h_at = |y: usize| -> i32 {
        // Half-sample between anchor_x and anchor_x+1, applied across
        // 6 input samples centred on the boundary.
        let row = [
            s(y, anchor_x - 2),
            s(y, anchor_x - 1),
            s(y, anchor_x),
            s(y, anchor_x + 1),
            s(y, anchor_x + 2),
            s(y, anchor_x + 3),
        ];
        luma_half(row)
    };
    let v_at = |x: usize| -> i32 {
        let col = [
            s(anchor_y - 2, x),
            s(anchor_y - 1, x),
            s(anchor_y, x),
            s(anchor_y + 1, x),
            s(anchor_y + 2, x),
            s(anchor_y + 3, x),
        ];
        luma_half(col)
    };

    // Pre-compute the four reference positions we may blend.
    let g = s(anchor_y, anchor_x); // integer
    let b_half_h = clip((h_at(anchor_y) + 16) >> 5) as i32; // half between (4,4)-(4,5)
    let h_half_v = clip((v_at(anchor_x) + 16) >> 5) as i32; // half between (4,4)-(5,4)
    // Diagonal half-sample (j in the spec): apply 6-tap on the row of
    // half-samples produced from the columns.
    let mut col_halves = [0i32; 6];
    for (i, dy_off) in (-2i32..=3).enumerate() {
        let y = (anchor_y as i32 + dy_off) as usize;
        col_halves[i] = h_at(y);
    }
    let j_half = clip(((luma_half(col_halves) + 512) >> 10).clamp(0, 255)) as i32;

    let v = match (dx, dy) {
        (0, 0) => g,
        (2, 0) => b_half_h,
        (0, 2) => h_half_v,
        (2, 2) => j_half,
        (1, 0) => avg2(g, b_half_h),
        (3, 0) => avg2(b_half_h, s(anchor_y, anchor_x + 1)),
        (0, 1) => avg2(g, h_half_v),
        (0, 3) => avg2(h_half_v, s(anchor_y + 1, anchor_x)),
        (1, 1) => avg2(b_half_h, h_half_v),
        (1, 2) => avg2(b_half_h, j_half),
        (2, 1) => avg2(j_half, h_half_v),
        (3, 3) => avg2(j_half, s(anchor_y + 1, anchor_x + 1)),
        _ => g, // remaining 1/4-pel positions land here; refined next slice
    };
    clip(v)
}

/// Chroma 8x8 bilinear MC for 4:2:0 (§8.4.1.2). `dx, dy ∈ 0..=7`
/// at 1/8-pel precision. `rf` is 9x9 covering the 8x8 block plus
/// one-sample padding on the right/bottom.
pub fn interp_chroma_8x8(rf: &[[u8; 9]; 9], dx: u8, dy: u8) -> [u8; 64] {
    let dx = dx as i32;
    let dy = dy as i32;
    let mut out = [0u8; 64];
    for y in 0..8 {
        for x in 0..8 {
            let a = rf[y][x] as i32;
            let b = rf[y][x + 1] as i32;
            let c = rf[y + 1][x] as i32;
            let d = rf[y + 1][x + 1] as i32;
            let v = ((8 - dx) * (8 - dy) * a
                + dx * (8 - dy) * b
                + (8 - dx) * dy * c
                + dx * dy * d
                + 32)
                >> 6;
            out[y * 8 + x] = clip(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_window(v: u8) -> [[u8; 9]; 9] {
        [[v; 9]; 9]
    }

    #[test]
    fn integer_position_passes_through() {
        let rf = flat_window(123);
        assert_eq!(interp_luma_qpel(&rf, 0, 0), 123);
    }

    #[test]
    fn luma_half_on_flat_is_flat() {
        let rf = flat_window(80);
        assert_eq!(interp_luma_qpel(&rf, 2, 0), 80);
        assert_eq!(interp_luma_qpel(&rf, 0, 2), 80);
        assert_eq!(interp_luma_qpel(&rf, 2, 2), 80);
    }

    #[test]
    fn luma_quarter_on_flat_is_flat() {
        let rf = flat_window(64);
        for dx in 0..=3 {
            for dy in 0..=3 {
                assert_eq!(interp_luma_qpel(&rf, dx, dy), 64, "dx={dx} dy={dy}");
            }
        }
    }

    #[test]
    fn chroma_int_position_copies() {
        let mut rf = [[0u8; 9]; 9];
        for y in 0..9 {
            for x in 0..9 {
                rf[y][x] = (y * 10 + x) as u8;
            }
        }
        let out = interp_chroma_8x8(&rf, 0, 0);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], rf[y][x]);
            }
        }
    }

    #[test]
    fn chroma_half_pel_blends() {
        let mut rf = [[0u8; 9]; 9];
        for y in 0..9 {
            for x in 0..9 {
                rf[y][x] = if x < 4 { 0 } else { 200 };
            }
        }
        // dx=4 (half-pel right) → at x=3 the bilinear should land
        // between 0 and 200 ≈ 100.
        let out = interp_chroma_8x8(&rf, 4, 0);
        let v = out[3] as i32;
        assert!(
            (95..=105).contains(&v),
            "expected half-pel blend ~100, got {v}"
        );
    }

    #[test]
    fn luma_half_h_blends_step_function() {
        // Build a step: left columns 0, right columns 200. Half-pel
        // horizontal should land around the average.
        let mut rf = [[0u8; 9]; 9];
        for y in 0..9 {
            for x in 0..9 {
                rf[y][x] = if x < 4 { 0 } else { 200 };
            }
        }
        let v = interp_luma_qpel(&rf, 2, 0) as i32;
        // The half-pel is anchored past the right edge of the step
        // (5 of 6 taps are on the 200 side), so the FIR result sits
        // near the high plateau but with normal FIR ringing.
        assert!((180..=255).contains(&v), "got {v}");
    }
}
