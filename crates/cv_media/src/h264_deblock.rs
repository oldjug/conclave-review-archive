//! H.264 in-loop deblocking filter — luma path.
//!
//! Per §8.7. For each 4x4 block boundary we compute:
//!   * `boundary_strength` ∈ [0,4]
//!   * `alpha`, `beta` thresholds derived from the average QP across
//!     the boundary
//!   * the per-side filter clip table `tC0` (bS ≤ 3) or fixed
//!     coefficients (bS == 4)
//!
//! and apply the filter to the four samples on each side that
//! straddle the edge (p0,p1,p2,p3 on one side, q0,q1,q2,q3 on the
//! other). This slice implements the bS=1..=3 luma case plus the
//! boundary-strength derivation for intra macroblocks (bS=3 inside
//! an MB, bS=4 at MB edges).
//!
//! Chroma deblock, frame/field weighting, and the per-block-edge
//! activity check ladders are follow-ups.

/// Spec table 8-16 (α threshold) indexed by `indexA = clip3(0, 51,
/// qp_avg + slice_alpha_c0_offset_div2 * 2)`. We surface the full
/// 52-entry table; callers clip the index before lookup.
pub const ALPHA_TABLE: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

/// Spec table 8-17 (β threshold) — indexed by `indexB`.
pub const BETA_TABLE: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// Spec table 8-18 (tC0 for bS=1..=3), 52 rows × 3 bS columns.
pub const TC0_TABLE: [[u8; 3]; 52] = {
    let mut t = [[0u8; 3]; 52];
    // Compact spec table — encode inline.
    let raw: &[(usize, u8, u8, u8)] = &[
        (16, 0, 0, 0),
        (17, 0, 0, 1),
        (18, 0, 0, 1),
        (19, 0, 0, 1),
        (20, 0, 0, 1),
        (21, 0, 1, 1),
        (22, 0, 1, 1),
        (23, 1, 1, 1),
        (24, 1, 1, 1),
        (25, 1, 1, 1),
        (26, 1, 1, 1),
        (27, 1, 1, 2),
        (28, 1, 1, 2),
        (29, 1, 1, 2),
        (30, 1, 1, 2),
        (31, 1, 2, 3),
        (32, 1, 2, 3),
        (33, 2, 2, 3),
        (34, 2, 2, 4),
        (35, 2, 3, 4),
        (36, 2, 3, 4),
        (37, 3, 3, 5),
        (38, 3, 4, 6),
        (39, 3, 4, 6),
        (40, 4, 5, 7),
        (41, 4, 5, 8),
        (42, 4, 6, 9),
        (43, 5, 7, 10),
        (44, 6, 8, 11),
        (45, 6, 8, 13),
        (46, 7, 10, 14),
        (47, 8, 11, 16),
        (48, 9, 12, 18),
        (49, 10, 13, 20),
        (50, 11, 15, 23),
        (51, 13, 17, 25),
    ];
    let mut i = 0;
    while i < raw.len() {
        let (idx, a, b, c) = raw[i];
        t[idx] = [a, b, c];
        i += 1;
    }
    t
};

#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.max(lo).min(hi)
}

/// Boundary strength for a luma 4x4 edge between two adjacent 4x4
/// blocks `p` and `q`. The §8.7.2 derivation is condensed to the
/// inputs the next stage will already have computed.
///
/// `p_is_intra`/`q_is_intra` — true when either side is an Intra
/// prediction. `edge_is_mb_edge` is true at macroblock boundaries.
pub fn boundary_strength(p_is_intra: bool, q_is_intra: bool, edge_is_mb_edge: bool) -> u8 {
    if p_is_intra || q_is_intra {
        if edge_is_mb_edge { 4 } else { 3 }
    } else {
        // Non-intra path: V1 reports bS=2 when either side has coded
        // residuals. The caller can supply the residual flags as a
        // followup; for now we default to 1 (the "no significant
        // residual, motion vector mismatch" case).
        1
    }
}

/// Apply the §8.7.2.1 luma filter across one 4x4 edge, in-place. The
/// caller stages the eight samples around the edge (p3,p2,p1,p0 on
/// one side, q0,q1,q2,q3 on the other) into a mutable slice.
///
/// `qp_avg` — average of the two macroblocks' QPY.
/// `alpha_c0_offset` / `beta_offset` come from the slice header in
/// units of `× 2`.
pub fn filter_luma_edge(
    samples: &mut [u8; 8],
    bs: u8,
    qp_avg: i32,
    alpha_c0_offset: i32,
    beta_offset: i32,
) {
    if bs == 0 {
        return;
    }
    let index_a = clip3(0, 51, qp_avg + alpha_c0_offset);
    let index_b = clip3(0, 51, qp_avg + beta_offset);
    let alpha = ALPHA_TABLE[index_a as usize] as i32;
    let beta = BETA_TABLE[index_b as usize] as i32;
    if alpha == 0 || beta == 0 {
        return;
    }
    let p3 = samples[0] as i32;
    let p2 = samples[1] as i32;
    let p1 = samples[2] as i32;
    let p0 = samples[3] as i32;
    let q0 = samples[4] as i32;
    let q1 = samples[5] as i32;
    let q2 = samples[6] as i32;
    let q3 = samples[7] as i32;
    // Activity check (§8.7.2.1, condition 1+2).
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }
    if bs < 4 {
        let tc0 = TC0_TABLE[index_a as usize][(bs - 1) as usize] as i32;
        let ap = (p2 - p0).abs();
        let aq = (q2 - q0).abs();
        let mut tc = tc0;
        if ap < beta {
            tc += 1;
        }
        if aq < beta {
            tc += 1;
        }
        let delta = clip3(-tc, tc, (((q0 - p0) * 4) + (p1 - q1) + 4) >> 3);
        let p0_new = clip3(0, 255, p0 + delta);
        let q0_new = clip3(0, 255, q0 - delta);
        samples[3] = p0_new as u8;
        samples[4] = q0_new as u8;
        if ap < beta {
            let p1_new = p1 + clip3(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - 2 * p1) >> 1);
            samples[2] = clip3(0, 255, p1_new) as u8;
        }
        if aq < beta {
            let q1_new = q1 + clip3(-tc0, tc0, (q2 + ((p0 + q0 + 1) >> 1) - 2 * q1) >> 1);
            samples[5] = clip3(0, 255, q1_new) as u8;
        }
    } else {
        // bS == 4 — strong filter.
        let small_ap = (p2 - p0).abs() < beta && (p0 - q0).abs() < (alpha >> 2) + 2;
        let small_aq = (q2 - q0).abs() < beta && (p0 - q0).abs() < (alpha >> 2) + 2;
        if small_ap {
            samples[3] = clip3(0, 255, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
            samples[2] = clip3(0, 255, (p2 + p1 + p0 + q0 + 2) >> 2) as u8;
            samples[1] = clip3(0, 255, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
        } else {
            samples[3] = clip3(0, 255, (2 * p1 + p0 + q1 + 2) >> 2) as u8;
        }
        if small_aq {
            samples[4] = clip3(0, 255, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) as u8;
            samples[5] = clip3(0, 255, (p0 + q0 + q1 + q2 + 2) >> 2) as u8;
            samples[6] = clip3(0, 255, (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
        } else {
            samples[4] = clip3(0, 255, (2 * q1 + q0 + p1 + 2) >> 2) as u8;
        }
    }
}

/// Apply the §8.7.2.2 chroma filter across one edge. Chroma blocks
/// in 4:2:0 are 8x8, so each edge straddles 4 samples; only the two
/// nearest samples on each side are modified (no `p1`/`q1` update).
///
/// Same activity gate + threshold tables as luma but with the
/// chroma-specific tweaks: for bS=4 the strong filter degenerates to
/// a simple two-tap average.
pub fn filter_chroma_edge(
    samples: &mut [u8; 4],
    bs: u8,
    qp_avg: i32,
    alpha_c0_offset: i32,
    beta_offset: i32,
) {
    if bs == 0 {
        return;
    }
    let index_a = clip3(0, 51, qp_avg + alpha_c0_offset);
    let index_b = clip3(0, 51, qp_avg + beta_offset);
    let alpha = ALPHA_TABLE[index_a as usize] as i32;
    let beta = BETA_TABLE[index_b as usize] as i32;
    if alpha == 0 || beta == 0 {
        return;
    }
    let p1 = samples[0] as i32;
    let p0 = samples[1] as i32;
    let q0 = samples[2] as i32;
    let q1 = samples[3] as i32;
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }
    if bs < 4 {
        let tc0 = TC0_TABLE[index_a as usize][(bs - 1) as usize] as i32;
        // Chroma path: tc = tc0 + 1 (no p2/q2 widening).
        let tc = tc0 + 1;
        let delta = clip3(-tc, tc, (((q0 - p0) * 4) + (p1 - q1) + 4) >> 3);
        samples[1] = clip3(0, 255, p0 + delta) as u8;
        samples[2] = clip3(0, 255, q0 - delta) as u8;
    } else {
        // Chroma bS=4: two-tap averages, no p3/q3 inputs.
        samples[1] = clip3(0, 255, (2 * p1 + p0 + q1 + 2) >> 2) as u8;
        samples[2] = clip3(0, 255, (2 * q1 + q0 + p1 + 2) >> 2) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_table_endpoints_match_spec() {
        assert_eq!(ALPHA_TABLE[0], 0);
        assert_eq!(ALPHA_TABLE[16], 4);
        assert_eq!(ALPHA_TABLE[28], 20);
        assert_eq!(ALPHA_TABLE[51], 255);
    }

    #[test]
    fn beta_table_endpoints_match_spec() {
        assert_eq!(BETA_TABLE[0], 0);
        assert_eq!(BETA_TABLE[16], 2);
        assert_eq!(BETA_TABLE[28], 7);
        assert_eq!(BETA_TABLE[51], 18);
    }

    #[test]
    fn tc0_spec_rows() {
        // Spot-check a few rows from §8.7.2.3 table 8-18.
        assert_eq!(TC0_TABLE[16], [0, 0, 0]);
        assert_eq!(TC0_TABLE[23], [1, 1, 1]);
        assert_eq!(TC0_TABLE[31], [1, 2, 3]);
        assert_eq!(TC0_TABLE[39], [3, 4, 6]);
        assert_eq!(TC0_TABLE[51], [13, 17, 25]);
    }

    #[test]
    fn boundary_strength_intra_at_mb_edge_is_4() {
        assert_eq!(boundary_strength(true, false, true), 4);
        assert_eq!(boundary_strength(false, true, true), 4);
    }

    #[test]
    fn boundary_strength_intra_inside_mb_is_3() {
        assert_eq!(boundary_strength(true, true, false), 3);
    }

    #[test]
    fn boundary_strength_inter_inter_is_1() {
        assert_eq!(boundary_strength(false, false, false), 1);
    }

    #[test]
    fn filter_skips_at_bs0() {
        let mut s = [10, 20, 30, 40, 50, 60, 70, 80];
        let orig = s;
        filter_luma_edge(&mut s, 0, 28, 0, 0);
        assert_eq!(s, orig);
    }

    #[test]
    fn filter_skips_when_alpha_is_zero() {
        let mut s = [10, 20, 30, 40, 50, 60, 70, 80];
        let orig = s;
        // QP 0 → alpha_table[0] = 0 → no filtering.
        filter_luma_edge(&mut s, 3, 0, 0, 0);
        assert_eq!(s, orig);
    }

    #[test]
    fn chroma_filter_skips_at_bs0() {
        let mut s = [60u8, 80, 100, 120];
        let orig = s;
        filter_chroma_edge(&mut s, 0, 28, 0, 0);
        assert_eq!(s, orig);
    }

    #[test]
    fn chroma_filter_smooths_small_step() {
        let mut s = [100u8, 100, 120, 120];
        filter_chroma_edge(&mut s, 3, 32, 0, 0);
        assert!((s[1] as i32) > 100);
        assert!((s[2] as i32) < 120);
    }

    #[test]
    fn chroma_filter_bs4_does_two_tap_average() {
        let mut s = [100u8, 100, 120, 120];
        filter_chroma_edge(&mut s, 4, 32, 0, 0);
        // p0_new = (2*100 + 100 + 120 + 2) >> 2 = 105
        // q0_new = (2*120 + 120 + 100 + 2) >> 2 = 115
        assert_eq!(s[1], 105);
        assert_eq!(s[2], 115);
    }

    #[test]
    fn filter_smooths_sharp_step_with_bs3() {
        // 20-sample step at QP 32 (alpha=32, beta=8). Diff 20 < alpha
        // so the activity gate passes; flat p1/p2 and q1/q2 keep us
        // inside the smoothing path. Expect both edge samples to
        // converge toward the midpoint.
        let mut s = [100u8, 100, 100, 100, 120, 120, 120, 120];
        filter_luma_edge(&mut s, 3, 32, 0, 0);
        assert!((s[3] as i32) > 100, "p0 should increase, got {}", s[3]);
        assert!((s[4] as i32) < 120, "q0 should decrease, got {}", s[4]);
    }
}
