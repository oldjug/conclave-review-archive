//! H.264 Intra_4x4 luma prediction.
//!
//! Implements the four most common Intra_4x4 prediction modes from
//! §8.3.1.2. A 4x4 sub-block predicts its 16 luma samples from
//! already-reconstructed neighbors above (row P[-1][0..3]) and to
//! the left (column P[0..3][-1]), plus a top-left corner sample.
//!
//! Modes implemented:
//!   0 Vertical   — P[y][x] = top[x]
//!   1 Horizontal — P[y][x] = left[y]
//!   2 DC         — mean of available neighbors (with neighbor
//!                  availability fallbacks per spec)
//!   3 Diagonal-Down-Left
//!
//! Remaining 5 directional modes (4..8) are stubbed as TODOs and
//! return `None` so callers fall back to the interpreter.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra4x4Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    DiagonalDownLeft = 3,
    DiagonalDownRight = 4,
    VerticalRight = 5,
    HorizontalDown = 6,
    VerticalLeft = 7,
    HorizontalUp = 8,
}

impl Intra4x4Mode {
    pub fn from_raw(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Vertical,
            1 => Self::Horizontal,
            2 => Self::Dc,
            3 => Self::DiagonalDownLeft,
            4 => Self::DiagonalDownRight,
            5 => Self::VerticalRight,
            6 => Self::HorizontalDown,
            7 => Self::VerticalLeft,
            8 => Self::HorizontalUp,
            _ => return None,
        })
    }
}

/// Neighborhood for a 4x4 intra prediction. `top` holds the 8 samples
/// to the right + above (top[0..3] are P[-1][0..3], top[4..7] are
/// P[-1][4..7] used by DDL/VL). `top_avail` and `left_avail` gate
/// fallback paths per spec §8.3.1.4.
#[derive(Debug, Clone, Copy)]
pub struct Neighbors4x4 {
    pub top: [u8; 8],
    pub left: [u8; 4],
    pub top_left: u8,
    pub top_avail: bool,
    pub left_avail: bool,
    pub top_right_avail: bool,
}

/// Run the requested Intra_4x4 prediction. Output is a 4x4 block of
/// luma samples in row-major order.
pub fn predict_4x4(mode: Intra4x4Mode, n: &Neighbors4x4) -> Option<[u8; 16]> {
    let mut out = [0u8; 16];
    match mode {
        Intra4x4Mode::Vertical => {
            if !n.top_avail {
                return None;
            }
            for y in 0..4 {
                for x in 0..4 {
                    out[y * 4 + x] = n.top[x];
                }
            }
        }
        Intra4x4Mode::Horizontal => {
            if !n.left_avail {
                return None;
            }
            for y in 0..4 {
                for x in 0..4 {
                    out[y * 4 + x] = n.left[y];
                }
            }
        }
        Intra4x4Mode::Dc => {
            // Per spec §8.3.1.2.3 fallbacks.
            let dc: u32 = match (n.top_avail, n.left_avail) {
                (true, true) => {
                    let s: u32 = n.top[0..4].iter().map(|&b| b as u32).sum::<u32>()
                        + n.left.iter().map(|&b| b as u32).sum::<u32>();
                    (s + 4) >> 3
                }
                (true, false) => {
                    let s: u32 = n.top[0..4].iter().map(|&b| b as u32).sum();
                    (s + 2) >> 2
                }
                (false, true) => {
                    let s: u32 = n.left.iter().map(|&b| b as u32).sum();
                    (s + 2) >> 2
                }
                (false, false) => 128, // §8.3.1.2.3, condition 4
            };
            let dc = dc.min(255) as u8;
            for v in out.iter_mut() {
                *v = dc;
            }
        }
        // ------------------------------------------------------------
        // Modes 4..=8 — directional predictions per H.264 §8.3.1.2.5
        // through 8.3.1.2.9. All require the top, top-left and left
        // neighbors to be available; the spec also requires the
        // top-right (for VerticalLeft) but our V1 falls back to None
        // when that's missing so the caller can decide.
        // ------------------------------------------------------------
        Intra4x4Mode::DiagonalDownRight => {
            if !(n.top_avail && n.left_avail) {
                return None;
            }
            let p_z = n.top_left as u32;
            let p_top = |x: usize| n.top[x] as u32;
            let p_left = |y: usize| n.left[y] as u32;
            for y in 0..4 {
                for x in 0..4 {
                    let v = if x > y {
                        // top
                        let a = if x == y + 1 { p_z } else { p_top(x - y - 2) };
                        let b = p_top(x - y - 1);
                        let c = p_top(x - y);
                        (a + 2 * b + c + 2) >> 2
                    } else if x < y {
                        // left
                        let a = if y == x + 1 { p_z } else { p_left(y - x - 2) };
                        let b = p_left(y - x - 1);
                        let c = p_left(y - x);
                        (a + 2 * b + c + 2) >> 2
                    } else {
                        // diagonal
                        let a = p_left(0);
                        let b = p_z;
                        let c = p_top(0);
                        (a + 2 * b + c + 2) >> 2
                    };
                    out[y * 4 + x] = (v & 0xFF) as u8;
                }
            }
        }
        Intra4x4Mode::VerticalRight => {
            if !(n.top_avail && n.left_avail) {
                return None;
            }
            let p_z = n.top_left as u32;
            let p_top = |x: usize| n.top[x] as u32;
            let p_left = |y: usize| n.left[y] as u32;
            // zVR mapping per spec §8.3.1.2.6 table.
            let avg2 = |a: u32, b: u32| -> u32 { (a + b + 1) >> 1 };
            let avg3 = |a: u32, b: u32, c: u32| -> u32 { (a + 2 * b + c + 2) >> 2 };
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * x - y;
                    let v: u32 = match z {
                        -1 => avg3(p_left(1), p_left(0), p_z),
                        0 => avg2(p_z, p_top(0)),
                        2 => avg2(p_top(0), p_top(1)),
                        4 => avg2(p_top(1), p_top(2)),
                        6 => avg2(p_top(2), p_top(3)),
                        1 => avg3(p_left(0), p_z, p_top(0)),
                        3 => avg3(p_z, p_top(0), p_top(1)),
                        5 => avg3(p_top(0), p_top(1), p_top(2)),
                        7 => avg3(p_top(1), p_top(2), p_top(3)),
                        -2 => avg3(p_left(2), p_left(1), p_left(0)),
                        -3 => avg3(p_left(3), p_left(2), p_left(1)),
                        _ => 128, // spec table doesn't reach this
                    };
                    out[(y as usize) * 4 + x as usize] = (v & 0xFF) as u8;
                }
            }
        }
        Intra4x4Mode::HorizontalDown => {
            if !(n.top_avail && n.left_avail) {
                return None;
            }
            let p_z = n.top_left as u32;
            let p_top = |x: usize| n.top[x] as u32;
            let p_left = |y: usize| n.left[y] as u32;
            let avg2 = |a: u32, b: u32| -> u32 { (a + b + 1) >> 1 };
            let avg3 = |a: u32, b: u32, c: u32| -> u32 { (a + 2 * b + c + 2) >> 2 };
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * y - x;
                    let v: u32 = match z {
                        -1 => avg3(p_top(1), p_top(0), p_z),
                        0 => avg2(p_z, p_left(0)),
                        2 => avg2(p_left(0), p_left(1)),
                        4 => avg2(p_left(1), p_left(2)),
                        6 => avg2(p_left(2), p_left(3)),
                        1 => avg3(p_top(0), p_z, p_left(0)),
                        3 => avg3(p_z, p_left(0), p_left(1)),
                        5 => avg3(p_left(0), p_left(1), p_left(2)),
                        7 => avg3(p_left(1), p_left(2), p_left(3)),
                        -2 => avg3(p_top(2), p_top(1), p_top(0)),
                        -3 => avg3(p_top(3), p_top(2), p_top(1)),
                        _ => 128,
                    };
                    out[(y as usize) * 4 + x as usize] = (v & 0xFF) as u8;
                }
            }
        }
        Intra4x4Mode::VerticalLeft => {
            if !n.top_avail {
                return None;
            }
            let mut t = n.top;
            if !n.top_right_avail {
                for i in 4..8 {
                    t[i] = n.top[3];
                }
            }
            let p_top = |x: usize| t[x] as u32;
            let avg2 = |a: u32, b: u32| -> u32 { (a + b + 1) >> 1 };
            let avg3 = |a: u32, b: u32, c: u32| -> u32 { (a + 2 * b + c + 2) >> 2 };
            // Spec §8.3.1.2.8 table; even y → avg2 of two top samples,
            // odd y → avg3 across three samples.
            for y in 0..4 {
                for x in 0..4 {
                    let v = if y == 0 {
                        avg2(p_top(x), p_top(x + 1))
                    } else if y == 1 {
                        avg3(p_top(x), p_top(x + 1), p_top(x + 2))
                    } else if y == 2 {
                        avg2(p_top(x + 1), p_top(x + 2))
                    } else {
                        avg3(p_top(x + 1), p_top(x + 2), p_top(x + 3))
                    };
                    out[y * 4 + x] = (v & 0xFF) as u8;
                }
            }
        }
        Intra4x4Mode::HorizontalUp => {
            if !n.left_avail {
                return None;
            }
            let p_left = |y: usize| n.left[y] as u32;
            let avg2 = |a: u32, b: u32| -> u32 { (a + b + 1) >> 1 };
            let avg3 = |a: u32, b: u32, c: u32| -> u32 { (a + 2 * b + c + 2) >> 2 };
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = x + 2 * y;
                    let v: u32 = match z {
                        0 => avg2(p_left(0), p_left(1)),
                        1 => avg3(p_left(0), p_left(1), p_left(2)),
                        2 => avg2(p_left(1), p_left(2)),
                        3 => avg3(p_left(1), p_left(2), p_left(3)),
                        4 => avg2(p_left(2), p_left(3)),
                        5 => avg3(p_left(2), p_left(3), p_left(3)),
                        _ => p_left(3),
                    };
                    out[(y as usize) * 4 + x as usize] = (v & 0xFF) as u8;
                }
            }
        }
        Intra4x4Mode::DiagonalDownLeft => {
            if !n.top_avail {
                return None;
            }
            // Per §8.3.1.2.4: when top-right unavailable, P[-1][4..7]
            // are filled with P[-1][3].
            let mut t = n.top;
            if !n.top_right_avail {
                t[4] = n.top[3];
                t[5] = n.top[3];
                t[6] = n.top[3];
                t[7] = n.top[3];
            }
            let p = |x: usize, y: usize| -> u8 {
                // Spec formula 8-50 for x != 3 || y != 3 vs the
                // boundary case at (3,3).
                if x == 3 && y == 3 {
                    let v: u32 = (t[6] as u32) + 3 * (t[7] as u32) + 2;
                    ((v >> 2) & 0xFF) as u8
                } else {
                    let a = t[x + y] as u32;
                    let b = t[x + y + 1] as u32;
                    let c = t[x + y + 2] as u32;
                    let v = (a + 2 * b + c + 2) >> 2;
                    (v & 0xFF) as u8
                }
            };
            for y in 0..4 {
                for x in 0..4 {
                    out[y * 4 + x] = p(x, y);
                }
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_neighbors(v: u8) -> Neighbors4x4 {
        Neighbors4x4 {
            top: [v; 8],
            left: [v; 4],
            top_left: v,
            top_avail: true,
            left_avail: true,
            top_right_avail: true,
        }
    }

    #[test]
    fn vertical_copies_top_row() {
        let mut n = solid_neighbors(0);
        n.top = [10, 20, 30, 40, 0, 0, 0, 0];
        let out = predict_4x4(Intra4x4Mode::Vertical, &n).unwrap();
        for y in 0..4 {
            assert_eq!(out[y * 4 + 0], 10);
            assert_eq!(out[y * 4 + 1], 20);
            assert_eq!(out[y * 4 + 2], 30);
            assert_eq!(out[y * 4 + 3], 40);
        }
    }

    #[test]
    fn horizontal_copies_left_column() {
        let mut n = solid_neighbors(0);
        n.left = [11, 22, 33, 44];
        let out = predict_4x4(Intra4x4Mode::Horizontal, &n).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out[y * 4 + x], n.left[y]);
            }
        }
    }

    #[test]
    fn dc_averages_8_neighbors() {
        let n = solid_neighbors(80);
        let out = predict_4x4(Intra4x4Mode::Dc, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 80);
        }
    }

    #[test]
    fn dc_falls_back_to_128_when_no_neighbors() {
        let mut n = solid_neighbors(0);
        n.top_avail = false;
        n.left_avail = false;
        let out = predict_4x4(Intra4x4Mode::Dc, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn vertical_returns_none_without_top() {
        let mut n = solid_neighbors(0);
        n.top_avail = false;
        assert!(predict_4x4(Intra4x4Mode::Vertical, &n).is_none());
    }

    #[test]
    fn ddr_solid_yields_solid() {
        let n = solid_neighbors(70);
        let out = predict_4x4(Intra4x4Mode::DiagonalDownRight, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 70);
        }
    }

    #[test]
    fn vr_solid_yields_solid() {
        let n = solid_neighbors(120);
        let out = predict_4x4(Intra4x4Mode::VerticalRight, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 120);
        }
    }

    #[test]
    fn hd_solid_yields_solid() {
        let n = solid_neighbors(60);
        let out = predict_4x4(Intra4x4Mode::HorizontalDown, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 60);
        }
    }

    #[test]
    fn vl_solid_yields_solid() {
        let n = solid_neighbors(200);
        let out = predict_4x4(Intra4x4Mode::VerticalLeft, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn hu_solid_yields_solid() {
        let n = solid_neighbors(30);
        let out = predict_4x4(Intra4x4Mode::HorizontalUp, &n).unwrap();
        for &v in &out {
            assert_eq!(v, 30);
        }
    }

    #[test]
    fn ddr_returns_none_without_corner_neighbors() {
        let mut n = solid_neighbors(0);
        n.top_avail = false;
        assert!(predict_4x4(Intra4x4Mode::DiagonalDownRight, &n).is_none());
        let mut n = solid_neighbors(0);
        n.left_avail = false;
        assert!(predict_4x4(Intra4x4Mode::DiagonalDownRight, &n).is_none());
    }

    #[test]
    fn ddl_is_diagonal_smoothing() {
        let mut n = solid_neighbors(0);
        n.top = [100, 100, 100, 100, 100, 100, 100, 100];
        let out = predict_4x4(Intra4x4Mode::DiagonalDownLeft, &n).unwrap();
        // Solid input → solid output (modulo the (3,3) tail-formula
        // which still averages to 100).
        for &v in &out {
            assert_eq!(v, 100);
        }
    }
}
