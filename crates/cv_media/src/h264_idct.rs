//! H.264 integer 4x4 IDCT.
//!
//! Implements the spec's 4x4 inverse transform from §8.5.12. The
//! forward transform is reversible exactly (integer arithmetic) and
//! the inverse runs row-then-column. Output is added to the
//! prediction residual; callers handle saturation to [0, 255] after
//! summing with predicted pixels.

/// Inverse 4x4 transform from spec eq. 8-338…8-340. Input: a 4x4
/// coefficient block in row-major order (already de-zigzagged and
/// dequantized). Output: the reconstructed residual, also 4x4
/// row-major.
pub fn idct_4x4(c: &[i32; 16]) -> [i32; 16] {
    // Horizontal pass — operate on rows.
    let mut h = [0i32; 16];
    for row in 0..4 {
        let i = row * 4;
        let e = c[i] + c[i + 2];
        let f = c[i] - c[i + 2];
        let g = (c[i + 1] >> 1) - c[i + 3];
        let h0 = c[i + 1] + (c[i + 3] >> 1);
        h[i] = e + h0;
        h[i + 1] = f + g;
        h[i + 2] = f - g;
        h[i + 3] = e - h0;
    }
    // Vertical pass — operate on columns of the row-pass result.
    let mut o = [0i32; 16];
    for col in 0..4 {
        let e = h[col] + h[col + 8];
        let f = h[col] - h[col + 8];
        let g = (h[col + 4] >> 1) - h[col + 12];
        let h0 = h[col + 4] + (h[col + 12] >> 1);
        o[col] = e + h0;
        o[col + 4] = f + g;
        o[col + 8] = f - g;
        o[col + 12] = e - h0;
    }
    // Per spec: rounding by adding 32 and shifting right by 6
    // (combined scaling factor from forward + inverse).
    for v in o.iter_mut() {
        *v = (*v + 32) >> 6;
    }
    o
}

/// Saturating add: clamp `pred + residual` to 0..=255 for each pixel.
pub fn add_residual_4x4(pred: &[u8; 16], residual: &[i32; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..16 {
        let v = (pred[i] as i32) + residual[i];
        out[i] = v.clamp(0, 255) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_of_all_zero_is_all_zero() {
        let c = [0i32; 16];
        let out = idct_4x4(&c);
        assert_eq!(out, [0i32; 16]);
    }

    #[test]
    fn idct_recovers_dc_only_block() {
        // Forward DC of value V is 16*V (sum). After scaling by Qstep
        // we'd typically see the spec scale factors, but if we
        // inject a pure DC coefficient that already represents the
        // post-quant value V*64, idct should yield V at every pixel.
        let mut c = [0i32; 16];
        c[0] = 5 * 64; // 5 per pixel, pre-(+32 >> 6)
        let out = idct_4x4(&c);
        for &v in &out {
            assert_eq!(v, 5);
        }
    }

    #[test]
    fn add_residual_saturates() {
        let pred = [250u8; 16];
        let mut residual = [20i32; 16];
        residual[0] = -300; // would underflow
        let out = add_residual_4x4(&pred, &residual);
        assert_eq!(out[0], 0);
        for i in 1..16 {
            assert_eq!(out[i], 255);
        }
    }
}
