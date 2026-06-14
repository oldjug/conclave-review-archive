//! H.264 Intra_16x16 luma prediction.
//!
//! Implements the four modes from §8.3.3:
//!   0 Vertical    — `P[y][x] = top[x]`           (replicate top row)
//!   1 Horizontal  — `P[y][x] = left[y]`          (replicate left col)
//!   2 DC          — average of the available top + left edges
//!   3 Plane       — H/V slope plane prediction (§8.3.3.4)
//!
//! Output is a flat 256-sample 16x16 luma block in row-major order,
//! waiting for the per-sub-block residual (added by the IDCT path).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra16x16Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    Plane = 3,
}

impl Intra16x16Mode {
    pub fn from_raw(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Vertical,
            1 => Self::Horizontal,
            2 => Self::Dc,
            3 => Self::Plane,
            _ => return None,
        })
    }
}

/// Neighborhood for a 16x16 Intra block. `top` covers samples
/// P[-1][0..15]; `left` covers P[0..15][-1]. `top_left` is P[-1][-1]
/// (needed by the Plane mode).
#[derive(Debug, Clone, Copy)]
pub struct Neighbors16x16 {
    pub top: [u8; 16],
    pub left: [u8; 16],
    pub top_left: u8,
    pub top_avail: bool,
    pub left_avail: bool,
}

#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.max(lo).min(hi)
}

pub fn predict_16x16(mode: Intra16x16Mode, n: &Neighbors16x16) -> Option<[u8; 256]> {
    let mut out = [0u8; 256];
    match mode {
        Intra16x16Mode::Vertical => {
            if !n.top_avail {
                return None;
            }
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = n.top[x];
                }
            }
        }
        Intra16x16Mode::Horizontal => {
            if !n.left_avail {
                return None;
            }
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = n.left[y];
                }
            }
        }
        Intra16x16Mode::Dc => {
            let dc: u32 = match (n.top_avail, n.left_avail) {
                (true, true) => {
                    let s: u32 = n.top.iter().map(|&b| b as u32).sum::<u32>()
                        + n.left.iter().map(|&b| b as u32).sum::<u32>();
                    (s + 16) >> 5
                }
                (true, false) => {
                    let s: u32 = n.top.iter().map(|&b| b as u32).sum();
                    (s + 8) >> 4
                }
                (false, true) => {
                    let s: u32 = n.left.iter().map(|&b| b as u32).sum();
                    (s + 8) >> 4
                }
                (false, false) => 128,
            };
            let dc = dc.min(255) as u8;
            for v in out.iter_mut() {
                *v = dc;
            }
        }
        Intra16x16Mode::Plane => {
            if !(n.top_avail && n.left_avail) {
                return None;
            }
            // Spec §8.3.3.4. H and V are signed sums of weighted
            // first-derivative samples across the top row and left
            // column. b and c are scaled slopes; the per-pixel
            // formula 8-92 then samples the plane.
            // Index helpers — the spec references P[-1][-1] for the
            // i=7 case in the H sum and the j=7 case in the V sum,
            // which is the top-left corner sample. We treat any
            // negative offset as the corner.
            let pz = n.top_left as i32;
            let p = |i: i32| -> i32 { if i < 0 { pz } else { n.top[i as usize] as i32 } };
            let q = |i: i32| -> i32 { if i < 0 { pz } else { n.left[i as usize] as i32 } };
            let mut h: i32 = 0;
            for i in 0..=7 {
                h += (i + 1) * (p(8 + i) - p(6 - i));
            }
            let mut v: i32 = 0;
            for j in 0..=7 {
                v += (j + 1) * (q(8 + j) - q(6 - j));
            }
            // For the y=-1, x=-1 corner the spec uses the top-left.
            // The slopes a, b, c.
            let b_slope = (5 * h + 32) >> 6;
            let c_slope = (5 * v + 32) >> 6;
            let a = 16 * (p(15) + q(15));
            for y in 0..16 {
                for x in 0..16 {
                    let val = (a + b_slope * (x as i32 - 7) + c_slope * (y as i32 - 7) + 16) >> 5;
                    out[y * 16 + x] = clip3(0, 255, val) as u8;
                }
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_neighbors(v: u8) -> Neighbors16x16 {
        Neighbors16x16 {
            top: [v; 16],
            left: [v; 16],
            top_left: v,
            top_avail: true,
            left_avail: true,
        }
    }

    #[test]
    fn vertical_replicates_top_row() {
        let mut n = solid_neighbors(0);
        for i in 0..16 {
            n.top[i] = i as u8 * 16;
        }
        let out = predict_16x16(Intra16x16Mode::Vertical, &n).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(out[y * 16 + x], n.top[x]);
            }
        }
    }

    #[test]
    fn horizontal_replicates_left_column() {
        let mut n = solid_neighbors(0);
        for i in 0..16 {
            n.left[i] = i as u8 * 16;
        }
        let out = predict_16x16(Intra16x16Mode::Horizontal, &n).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(out[y * 16 + x], n.left[y]);
            }
        }
    }

    #[test]
    fn dc_averages_32_neighbors() {
        let n = solid_neighbors(100);
        let out = predict_16x16(Intra16x16Mode::Dc, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn dc_falls_back_to_128_when_no_neighbors() {
        let mut n = solid_neighbors(0);
        n.top_avail = false;
        n.left_avail = false;
        let out = predict_16x16(Intra16x16Mode::Dc, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn plane_on_flat_field_is_flat() {
        let n = solid_neighbors(80);
        let out = predict_16x16(Intra16x16Mode::Plane, &n).unwrap();
        // Slopes are zero on a flat input, so all 256 samples → 80.
        for &v in &out {
            assert_eq!(v, 80);
        }
    }

    #[test]
    fn vertical_returns_none_without_top() {
        let mut n = solid_neighbors(50);
        n.top_avail = false;
        assert!(predict_16x16(Intra16x16Mode::Vertical, &n).is_none());
    }

    #[test]
    fn plane_returns_none_without_either_neighbor() {
        let mut n = solid_neighbors(50);
        n.top_avail = false;
        assert!(predict_16x16(Intra16x16Mode::Plane, &n).is_none());
        let mut n = solid_neighbors(50);
        n.left_avail = false;
        assert!(predict_16x16(Intra16x16Mode::Plane, &n).is_none());
    }
}
