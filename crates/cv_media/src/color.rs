//! YUV → RGB color conversion for the video pipeline.
//!
//! Supports BT.601 (SDTV) and BT.709 (HDTV) limited-range YUV.
//! Chroma planes for 4:2:0 are upsampled with bilinear interpolation.
//! Output is BGRA u32 ready for the compositor / present pipeline.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMatrix {
    Bt601,
    Bt709,
}

#[inline]
fn clip(v: i32) -> u8 {
    v.max(0).min(255) as u8
}

/// Convert one limited-range YUV triplet to BGRA u32. The matrices
/// come from ITU-R BT.601 / BT.709; coefficients are fixed-point ×
/// 1024 so we stay in i32 land.
pub fn yuv_to_bgra(y: u8, u: u8, v: u8, matrix: ColorMatrix) -> u32 {
    // Limited range: Y in [16,235], UV in [16,240] with offset 128.
    let yy = (y as i32 - 16).max(0);
    let uu = u as i32 - 128;
    let vv = v as i32 - 128;
    let (r, g, b) = match matrix {
        ColorMatrix::Bt601 => {
            // BT.601: R = 1.164*Y + 1.596*V
            //         G = 1.164*Y - 0.392*U - 0.813*V
            //         B = 1.164*Y + 2.017*U
            let r = (1192 * yy + 1634 * vv + 512) >> 10;
            let g = (1192 * yy - 401 * uu - 832 * vv + 512) >> 10;
            let b = (1192 * yy + 2065 * uu + 512) >> 10;
            (r, g, b)
        }
        ColorMatrix::Bt709 => {
            // BT.709: R = 1.164*Y + 1.793*V
            //         G = 1.164*Y - 0.213*U - 0.534*V
            //         B = 1.164*Y + 2.115*U
            let r = (1192 * yy + 1836 * vv + 512) >> 10;
            let g = (1192 * yy - 218 * uu - 547 * vv + 512) >> 10;
            let b = (1192 * yy + 2165 * uu + 512) >> 10;
            (r, g, b)
        }
    };
    let r = clip(r) as u32;
    let g = clip(g) as u32;
    let b = clip(b) as u32;
    (0xFFu32 << 24) | (r << 16) | (g << 8) | b
}

/// 4:2:0 YUV plane set → tightly packed BGRA bitmap. `y_plane` is
/// width × height; `u_plane` and `v_plane` are (width/2) × (height/2).
/// Chroma is bilinearly upsampled to full resolution before matrix
/// conversion.
pub fn yuv420_to_bgra(
    width: u32,
    height: u32,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    matrix: ColorMatrix,
) -> Vec<u32> {
    assert_eq!(y_plane.len(), (width * height) as usize);
    let cw = width / 2;
    let ch = height / 2;
    assert_eq!(u_plane.len(), (cw * ch) as usize);
    assert_eq!(v_plane.len(), (cw * ch) as usize);
    let mut out = vec![0u32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let y_sample = y_plane[(y * width + x) as usize];
            // Bilinear chroma upsample. Map dest pixel (x,y) to
            // chroma space (x/2, y/2) and bilinearly interpolate
            // among the four surrounding chroma samples.
            let cx_f = (x as f32) * 0.5 - 0.25;
            let cy_f = (y as f32) * 0.5 - 0.25;
            let cx0 = (cx_f.floor() as i32).clamp(0, (cw - 1) as i32) as u32;
            let cy0 = (cy_f.floor() as i32).clamp(0, (ch - 1) as i32) as u32;
            let cx1 = (cx0 + 1).min(cw - 1);
            let cy1 = (cy0 + 1).min(ch - 1);
            let fx = (cx_f - cx0 as f32).clamp(0.0, 1.0);
            let fy = (cy_f - cy0 as f32).clamp(0.0, 1.0);
            let sample =
                |plane: &[u8], px: u32, py: u32| -> f32 { plane[(py * cw + px) as usize] as f32 };
            let u_val = (sample(u_plane, cx0, cy0) * (1.0 - fx) * (1.0 - fy)
                + sample(u_plane, cx1, cy0) * fx * (1.0 - fy)
                + sample(u_plane, cx0, cy1) * (1.0 - fx) * fy
                + sample(u_plane, cx1, cy1) * fx * fy) as u8;
            let v_val = (sample(v_plane, cx0, cy0) * (1.0 - fx) * (1.0 - fy)
                + sample(v_plane, cx1, cy0) * fx * (1.0 - fy)
                + sample(v_plane, cx0, cy1) * (1.0 - fx) * fy
                + sample(v_plane, cx1, cy1) * fx * fy) as u8;
            out[(y * width + x) as usize] = yuv_to_bgra(y_sample, u_val, v_val, matrix);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_yuv_is_black_rgb() {
        let p = yuv_to_bgra(16, 128, 128, ColorMatrix::Bt709);
        let r = (p >> 16) & 0xFF;
        let g = (p >> 8) & 0xFF;
        let b = p & 0xFF;
        assert!(r < 3 && g < 3 && b < 3, "got 0x{p:08X}");
    }

    #[test]
    fn white_yuv_is_white_rgb() {
        let p = yuv_to_bgra(235, 128, 128, ColorMatrix::Bt709);
        let r = (p >> 16) & 0xFF;
        let g = (p >> 8) & 0xFF;
        let b = p & 0xFF;
        assert!(r > 250 && g > 250 && b > 250, "got 0x{p:08X}");
    }

    #[test]
    fn red_chroma_pulls_red_high_bt601() {
        // High V, neutral U, mid Y → red.
        let p = yuv_to_bgra(82, 90, 240, ColorMatrix::Bt601);
        let r = (p >> 16) & 0xFF;
        let g = (p >> 8) & 0xFF;
        let b = p & 0xFF;
        assert!(r > g && r > b, "got R={r} G={g} B={b}");
    }

    #[test]
    fn yuv420_solid_field_round_trips() {
        let w = 4u32;
        let h = 4u32;
        let cw = (w / 2) as usize;
        let ch = (h / 2) as usize;
        let y = vec![128u8; (w * h) as usize]; // grey
        let u = vec![128u8; cw * ch];
        let v = vec![128u8; cw * ch];
        let out = yuv420_to_bgra(w, h, &y, &u, &v, ColorMatrix::Bt709);
        // Grey should produce a neutral-ish output.
        for &p in &out {
            let r = (p >> 16) & 0xFF;
            let g = (p >> 8) & 0xFF;
            let b = p & 0xFF;
            let avg = (r + g + b) / 3;
            assert!(
                (r as i32 - avg as i32).abs() < 3
                    && (g as i32 - avg as i32).abs() < 3
                    && (b as i32 - avg as i32).abs() < 3,
                "expected grey-ish, got 0x{p:08X}"
            );
        }
    }
}
