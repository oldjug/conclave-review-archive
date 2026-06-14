//! H.264 chroma intra prediction (4:2:0).
//!
//! Per §8.3.4. Chroma blocks are 8x8 (for 4:2:0). The four modes
//! are similar in spirit to Intra_16x16 but the DC averaging
//! splits the block into four 4x4 quadrants — each gets its own DC
//! based on which neighbors are available.
//!
//! Modes:
//!   0 DC          — per-quadrant average (§8.3.4.2)
//!   1 Horizontal  — replicate left column
//!   2 Vertical    — replicate top row
//!   3 Plane       — same shape as Intra_16x16 plane, sized for 8x8

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaIntraMode {
    Dc = 0,
    Horizontal = 1,
    Vertical = 2,
    Plane = 3,
}

impl ChromaIntraMode {
    pub fn from_raw(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Dc,
            1 => Self::Horizontal,
            2 => Self::Vertical,
            3 => Self::Plane,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChromaNeighbors {
    pub top: [u8; 8],
    pub left: [u8; 8],
    pub top_left: u8,
    pub top_avail: bool,
    pub left_avail: bool,
}

#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.max(lo).min(hi)
}

pub fn predict_chroma_8x8(mode: ChromaIntraMode, n: &ChromaNeighbors) -> Option<[u8; 64]> {
    let mut out = [0u8; 64];
    match mode {
        ChromaIntraMode::Vertical => {
            if !n.top_avail {
                return None;
            }
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = n.top[x];
                }
            }
        }
        ChromaIntraMode::Horizontal => {
            if !n.left_avail {
                return None;
            }
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = n.left[y];
                }
            }
        }
        ChromaIntraMode::Dc => {
            // Per spec, the 8x8 block splits into four 4x4 quadrants;
            // each quadrant's DC depends on what edges are available
            // adjacent to it. We enumerate the four quadrants
            // explicitly.
            fn avg(samples: &[u32]) -> u32 {
                let s: u32 = samples.iter().sum();
                let n = samples.len() as u32;
                (s + (n / 2)) / n
            }
            // Top-left quadrant: averages top[0..4] + left[0..4].
            let tl_dc = match (n.top_avail, n.left_avail) {
                (true, true) => avg(&[
                    n.top[0] as u32,
                    n.top[1] as u32,
                    n.top[2] as u32,
                    n.top[3] as u32,
                    n.left[0] as u32,
                    n.left[1] as u32,
                    n.left[2] as u32,
                    n.left[3] as u32,
                ]),
                (true, false) => avg(&[
                    n.top[0] as u32,
                    n.top[1] as u32,
                    n.top[2] as u32,
                    n.top[3] as u32,
                ]),
                (false, true) => avg(&[
                    n.left[0] as u32,
                    n.left[1] as u32,
                    n.left[2] as u32,
                    n.left[3] as u32,
                ]),
                (false, false) => 128,
            };
            // Top-right quadrant: only top[4..8] is "above" it.
            let tr_dc = if n.top_avail {
                avg(&[
                    n.top[4] as u32,
                    n.top[5] as u32,
                    n.top[6] as u32,
                    n.top[7] as u32,
                ])
            } else if n.left_avail {
                avg(&[
                    n.left[0] as u32,
                    n.left[1] as u32,
                    n.left[2] as u32,
                    n.left[3] as u32,
                ])
            } else {
                128
            };
            // Bottom-left quadrant: only left[4..8] is to its left.
            let bl_dc = if n.left_avail {
                avg(&[
                    n.left[4] as u32,
                    n.left[5] as u32,
                    n.left[6] as u32,
                    n.left[7] as u32,
                ])
            } else if n.top_avail {
                avg(&[
                    n.top[0] as u32,
                    n.top[1] as u32,
                    n.top[2] as u32,
                    n.top[3] as u32,
                ])
            } else {
                128
            };
            // Bottom-right: top[4..8] + left[4..8].
            let br_dc = match (n.top_avail, n.left_avail) {
                (true, true) => avg(&[
                    n.top[4] as u32,
                    n.top[5] as u32,
                    n.top[6] as u32,
                    n.top[7] as u32,
                    n.left[4] as u32,
                    n.left[5] as u32,
                    n.left[6] as u32,
                    n.left[7] as u32,
                ]),
                (true, false) => avg(&[
                    n.top[4] as u32,
                    n.top[5] as u32,
                    n.top[6] as u32,
                    n.top[7] as u32,
                ]),
                (false, true) => avg(&[
                    n.left[4] as u32,
                    n.left[5] as u32,
                    n.left[6] as u32,
                    n.left[7] as u32,
                ]),
                (false, false) => 128,
            };
            for y in 0..8 {
                for x in 0..8 {
                    let dc = match (x < 4, y < 4) {
                        (true, true) => tl_dc,
                        (false, true) => tr_dc,
                        (true, false) => bl_dc,
                        (false, false) => br_dc,
                    };
                    out[y * 8 + x] = dc.min(255) as u8;
                }
            }
        }
        ChromaIntraMode::Plane => {
            if !(n.top_avail && n.left_avail) {
                return None;
            }
            let pz = n.top_left as i32;
            let p = |i: i32| -> i32 { if i < 0 { pz } else { n.top[i as usize] as i32 } };
            let q = |i: i32| -> i32 { if i < 0 { pz } else { n.left[i as usize] as i32 } };
            let mut h: i32 = 0;
            for i in 0..=3 {
                h += (i + 1) * (p(4 + i) - p(2 - i));
            }
            let mut v: i32 = 0;
            for j in 0..=3 {
                v += (j + 1) * (q(4 + j) - q(2 - j));
            }
            let b_slope = (34 * h + 32) >> 6;
            let c_slope = (34 * v + 32) >> 6;
            let a = 16 * (p(7) + q(7));
            for y in 0..8 {
                for x in 0..8 {
                    let val = (a + b_slope * (x as i32 - 3) + c_slope * (y as i32 - 3) + 16) >> 5;
                    out[y * 8 + x] = clip3(0, 255, val) as u8;
                }
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(v: u8) -> ChromaNeighbors {
        ChromaNeighbors {
            top: [v; 8],
            left: [v; 8],
            top_left: v,
            top_avail: true,
            left_avail: true,
        }
    }

    #[test]
    fn vertical_replicates_top() {
        let mut n = solid(0);
        for i in 0..8 {
            n.top[i] = i as u8 * 32;
        }
        let out = predict_chroma_8x8(ChromaIntraMode::Vertical, &n).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], n.top[x]);
            }
        }
    }

    #[test]
    fn horizontal_replicates_left() {
        let mut n = solid(0);
        for i in 0..8 {
            n.left[i] = i as u8 * 32;
        }
        let out = predict_chroma_8x8(ChromaIntraMode::Horizontal, &n).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], n.left[y]);
            }
        }
    }

    #[test]
    fn dc_on_solid_is_solid() {
        let n = solid(64);
        let out = predict_chroma_8x8(ChromaIntraMode::Dc, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 64);
        }
    }

    #[test]
    fn dc_no_neighbors_is_128() {
        let mut n = solid(0);
        n.top_avail = false;
        n.left_avail = false;
        let out = predict_chroma_8x8(ChromaIntraMode::Dc, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn plane_on_flat_is_flat() {
        let n = solid(96);
        let out = predict_chroma_8x8(ChromaIntraMode::Plane, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 96);
        }
    }
}
