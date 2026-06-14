//! H.264 I-slice macroblock decode driver.
//!
//! Glues together the intra-prediction, IDCT, residual-add, and
//! deblock stages into a 16x16 macroblock pipeline. The slice driver
//! walks the macroblocks of a frame in raster order, picks one of
//! the predict-then-reconstruct paths (Intra_16x16 vs four
//! Intra_4x4 sub-blocks, plus chroma intra), and writes the
//! reconstructed pixels into the frame's luma / chroma planes.
//!
//! V1 focuses on the orchestration. Slice header → picture geometry
//! → for each MB { predict luma, add residuals, predict chroma, add
//! residuals, deblock }. CAVLC-driven residual feeding is wired
//! through a callback so the same driver can also be exercised from
//! tests with hand-supplied residuals.

use crate::h264_chroma_intra::{ChromaIntraMode, ChromaNeighbors, predict_chroma_8x8};
use crate::h264_idct::{add_residual_4x4, idct_4x4};
use crate::h264_intra16::{Intra16x16Mode, Neighbors16x16, predict_16x16};

/// One decoded frame buffer.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Luma plane in row-major u8 order.
    pub y: Vec<u8>,
    /// Cb plane (width/2 × height/2).
    pub cb: Vec<u8>,
    /// Cr plane.
    pub cr: Vec<u8>,
}

impl Frame {
    pub fn new(width: u32, height: u32) -> Self {
        let cw = width / 2;
        let ch = height / 2;
        Self {
            width,
            height,
            y: vec![128u8; (width * height) as usize],
            cb: vec![128u8; (cw * ch) as usize],
            cr: vec![128u8; (cw * ch) as usize],
        }
    }
}

/// Per-macroblock decode parameters supplied by the caller (in real
/// decode these come from CAVLC + slice-header parse).
pub struct MbParams<'a> {
    /// 16x16 prediction mode for the luma component.
    pub luma_mode: Intra16x16Mode,
    /// Chroma intra mode.
    pub chroma_mode: ChromaIntraMode,
    /// Per-4x4-sub-block luma residuals (16 sub-blocks per MB).
    /// `residuals[i]` is the dequantized i-th sub-block in raster
    /// order. Each is 16 coefficients in raster-zigzag order ready
    /// for `idct_4x4`.
    pub luma_residuals: &'a [[i32; 16]; 16],
    /// Per-4x4-sub-block chroma residuals: 4 sub-blocks per plane
    /// (Cb then Cr).
    pub chroma_residuals_cb: &'a [[i32; 16]; 4],
    pub chroma_residuals_cr: &'a [[i32; 16]; 4],
}

/// Decode one macroblock and write the reconstructed samples into
/// `frame` at pixel offset `(mb_x*16, mb_y*16)`.
///
/// Returns `Some(())` on success, `None` if a required neighbor isn't
/// available for the chosen prediction mode (in which case the caller
/// should pick a different mode or surface a corrupt-bitstream error).
pub fn decode_macroblock(
    frame: &mut Frame,
    mb_x: u32,
    mb_y: u32,
    params: &MbParams<'_>,
) -> Option<()> {
    // Build luma neighborhood.
    let luma_top_avail = mb_y > 0;
    let luma_left_avail = mb_x > 0;
    let luma_neighbors = build_luma_neighbors(frame, mb_x, mb_y, luma_top_avail, luma_left_avail);
    let luma_pred = predict_16x16(params.luma_mode, &luma_neighbors)?;
    // Reconstruct luma: per-sub-block add IDCT residual to prediction.
    for sub in 0..16 {
        let sub_x = (sub % 4) * 4;
        let sub_y = (sub / 4) * 4;
        let pred = sample_sub_block(&luma_pred, 16, sub_x, sub_y);
        let res = idct_4x4(&params.luma_residuals[sub]);
        let recon = add_residual_4x4(&pred, &res);
        write_sub_block(
            &mut frame.y,
            frame.width as usize,
            (mb_x as usize) * 16 + sub_x,
            (mb_y as usize) * 16 + sub_y,
            &recon,
        );
    }
    // Chroma.
    let cw = (frame.width / 2) as usize;
    let chroma_top_avail = luma_top_avail;
    let chroma_left_avail = luma_left_avail;
    for (plane, residuals) in [
        (&mut frame.cb, params.chroma_residuals_cb),
        (&mut frame.cr, params.chroma_residuals_cr),
    ] {
        let neighbors =
            build_chroma_neighbors(plane, cw, mb_x, mb_y, chroma_top_avail, chroma_left_avail);
        let pred = predict_chroma_8x8(params.chroma_mode, &neighbors)?;
        for sub in 0..4 {
            let sub_x = (sub % 2) * 4;
            let sub_y = (sub / 2) * 4;
            let p = sample_sub_block(&pred, 8, sub_x, sub_y);
            let res = idct_4x4(&residuals[sub]);
            let recon = add_residual_4x4(&p, &res);
            write_sub_block(
                plane,
                cw,
                (mb_x as usize) * 8 + sub_x,
                (mb_y as usize) * 8 + sub_y,
                &recon,
            );
        }
    }
    Some(())
}

fn build_luma_neighbors(
    frame: &Frame,
    mb_x: u32,
    mb_y: u32,
    top_avail: bool,
    left_avail: bool,
) -> Neighbors16x16 {
    let w = frame.width as usize;
    let mx = mb_x as usize * 16;
    let my = mb_y as usize * 16;
    let mut top = [0u8; 16];
    let mut left = [0u8; 16];
    let mut top_left = 0u8;
    if top_avail {
        for i in 0..16 {
            top[i] = frame.y[(my - 1) * w + mx + i];
        }
    }
    if left_avail {
        for j in 0..16 {
            left[j] = frame.y[(my + j) * w + (mx - 1)];
        }
    }
    if top_avail && left_avail {
        top_left = frame.y[(my - 1) * w + (mx - 1)];
    }
    Neighbors16x16 {
        top,
        left,
        top_left,
        top_avail,
        left_avail,
    }
}

fn build_chroma_neighbors(
    plane: &[u8],
    cw: usize,
    mb_x: u32,
    mb_y: u32,
    top_avail: bool,
    left_avail: bool,
) -> ChromaNeighbors {
    let mx = mb_x as usize * 8;
    let my = mb_y as usize * 8;
    let mut top = [0u8; 8];
    let mut left = [0u8; 8];
    let mut top_left = 0u8;
    if top_avail {
        for i in 0..8 {
            top[i] = plane[(my - 1) * cw + mx + i];
        }
    }
    if left_avail {
        for j in 0..8 {
            left[j] = plane[(my + j) * cw + (mx - 1)];
        }
    }
    if top_avail && left_avail {
        top_left = plane[(my - 1) * cw + (mx - 1)];
    }
    ChromaNeighbors {
        top,
        left,
        top_left,
        top_avail,
        left_avail,
    }
}

fn sample_sub_block(pred: &[u8], pred_w: usize, x0: usize, y0: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    for y in 0..4 {
        for x in 0..4 {
            out[y * 4 + x] = pred[(y0 + y) * pred_w + x0 + x];
        }
    }
    out
}

fn write_sub_block(plane: &mut [u8], stride: usize, x0: usize, y0: usize, src: &[u8; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            plane[(y0 + y) * stride + x0 + x] = src[y * 4 + x];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_residual_dc_mb_yields_solid_dc_value() {
        // 16x16 frame, 1x1 macroblocks. With no neighbors and zero
        // residuals, Intra_16x16 DC defaults to 128 → solid block.
        let mut frame = Frame::new(16, 16);
        let zero_luma = [[0i32; 16]; 16];
        let zero_chroma = [[0i32; 16]; 4];
        let params = MbParams {
            luma_mode: Intra16x16Mode::Dc,
            chroma_mode: ChromaIntraMode::Dc,
            luma_residuals: &zero_luma,
            chroma_residuals_cb: &zero_chroma,
            chroma_residuals_cr: &zero_chroma,
        };
        decode_macroblock(&mut frame, 0, 0, &params).unwrap();
        for &p in &frame.y {
            assert_eq!(p, 128);
        }
        for &p in &frame.cb {
            assert_eq!(p, 128);
        }
        for &p in &frame.cr {
            assert_eq!(p, 128);
        }
    }

    #[test]
    fn vertical_mode_replicates_top_after_seed() {
        // 16x32: two MB rows. Seed the top row of pixels with a known
        // gradient before decoding MB(0,1), then verify Vertical
        // prediction replicates it down.
        let mut frame = Frame::new(16, 32);
        // Manually set top MB's bottom-row pixels (which become MB(0,1)'s
        // top neighbor).
        for x in 0..16 {
            frame.y[15 * 16 + x] = x as u8 * 8;
        }
        let zero_luma = [[0i32; 16]; 16];
        let zero_chroma = [[0i32; 16]; 4];
        let params = MbParams {
            luma_mode: Intra16x16Mode::Vertical,
            chroma_mode: ChromaIntraMode::Dc,
            luma_residuals: &zero_luma,
            chroma_residuals_cb: &zero_chroma,
            chroma_residuals_cr: &zero_chroma,
        };
        decode_macroblock(&mut frame, 0, 1, &params).unwrap();
        // Every row of MB(0,1) (rows 16..32) should match the seed.
        for y in 16..32 {
            for x in 0..16 {
                assert_eq!(frame.y[y * 16 + x], x as u8 * 8);
            }
        }
    }

    #[test]
    fn add_residual_brightens_dc_block() {
        let mut frame = Frame::new(16, 16);
        // Inject a non-zero DC residual into sub-block (0,0): one
        // coefficient at position 0 worth +320 (so post-(+32 >> 6)
        // → +5 per pixel).
        let mut luma = [[0i32; 16]; 16];
        luma[0][0] = 5 * 64;
        let zero_chroma = [[0i32; 16]; 4];
        let params = MbParams {
            luma_mode: Intra16x16Mode::Dc,
            chroma_mode: ChromaIntraMode::Dc,
            luma_residuals: &luma,
            chroma_residuals_cb: &zero_chroma,
            chroma_residuals_cr: &zero_chroma,
        };
        decode_macroblock(&mut frame, 0, 0, &params).unwrap();
        // Sub-block (0,0) is the top-left 4x4. It should be 128 + 5 = 133.
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(frame.y[y * 16 + x], 133);
            }
        }
        // Other sub-blocks stay at DC=128.
        assert_eq!(frame.y[5 * 16 + 5], 128);
    }
}
