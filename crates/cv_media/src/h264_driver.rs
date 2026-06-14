//! End-to-end H.264 driver: NAL stream → SPS → IDR frame → BGRA.

use crate::color::{ColorMatrix, yuv420_to_bgra};
use crate::h264::{NalHeader, Sps, ebsp_to_rbsp, parse_sps, split_nal_units};
use crate::h264_chroma_intra::ChromaIntraMode;
use crate::h264_intra16::Intra16x16Mode;
use crate::h264_mb_loop::{Frame, MbParams, decode_macroblock};

/// Configuration the driver pulls from the SPS / slice header.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub frame_y: Vec<u8>,
    pub frame_cb: Vec<u8>,
    pub frame_cr: Vec<u8>,
    pub bgra: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverError {
    NoSps,
    BadSps,
    NoIdr,
    DecodeFailed(String),
}

/// Decode the first IDR access unit in `bitstream`. Walks NAL units,
/// finds the SPS (sets geometry), finds the IDR slice, runs the
/// macroblock loop for every MB in the picture, and converts the
/// resulting YUV420 planes to BGRA.
pub fn decode_first_idr(bitstream: &[u8]) -> Result<DecodedFrame, DriverError> {
    // 1. Split into NAL units.
    let nals = split_nal_units(bitstream);
    if nals.is_empty() {
        return Err(DriverError::NoSps);
    }

    // 2. Find SPS, parse it.
    let mut sps: Option<Sps> = None;
    for n in &nals {
        if n.is_empty() {
            continue;
        }
        let hdr = NalHeader::parse(n[0]);
        if hdr.nal_unit_type == crate::h264::nal_type::SPS {
            let rbsp = ebsp_to_rbsp(&n[1..]);
            sps = parse_sps(&rbsp);
            if sps.is_some() {
                break;
            }
        }
    }
    let sps = sps.ok_or(DriverError::NoSps)?;

    // 3. Find first IDR slice.
    let mut idr: Option<&[u8]> = None;
    for n in &nals {
        if n.is_empty() {
            continue;
        }
        let hdr = NalHeader::parse(n[0]);
        if hdr.nal_unit_type == crate::h264::nal_type::SLICE_IDR {
            idr = Some(n);
            break;
        }
    }
    let _idr = idr.ok_or(DriverError::NoIdr)?;

    // 4. Validate geometry.
    if sps.width == 0 || sps.height == 0 || sps.width % 16 != 0 || sps.height % 16 != 0 {
        return Err(DriverError::BadSps);
    }
    let mb_cols = sps.width / 16;
    let mb_rows = sps.height / 16;

    // 5. Run the MB decode loop. For V1 we don't yet have CAVLC
    // coupled per-MB; the driver iterates MBs with zero residuals +
    // Intra_16x16 DC which yields a uniform grey picture but exercises
    // the end-to-end pipeline (SPS → MB loop → YUV → BGRA).
    let mut frame = Frame::new(sps.width, sps.height);
    let zero_luma = [[0i32; 16]; 16];
    let zero_chroma = [[0i32; 16]; 4];
    let params = MbParams {
        luma_mode: Intra16x16Mode::Dc,
        chroma_mode: ChromaIntraMode::Dc,
        luma_residuals: &zero_luma,
        chroma_residuals_cb: &zero_chroma,
        chroma_residuals_cr: &zero_chroma,
    };
    for mb_y in 0..mb_rows {
        for mb_x in 0..mb_cols {
            decode_macroblock(&mut frame, mb_x, mb_y, &params)
                .ok_or_else(|| DriverError::DecodeFailed(format!("MB ({mb_x},{mb_y})")))?;
        }
    }

    // 6. YUV→BGRA.
    let bgra = yuv420_to_bgra(
        sps.width,
        sps.height,
        &frame.y,
        &frame.cb,
        &frame.cr,
        ColorMatrix::Bt709,
    );

    Ok(DecodedFrame {
        width: sps.width,
        height: sps.height,
        frame_y: frame.y,
        frame_cb: frame.cb,
        frame_cr: frame.cr,
        bgra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal NAL byte-stream with our hand-crafted SPS
    /// (640x480 Baseline) and an IDR slice header with no residuals.
    fn synth_bitstream(sps_rbsp: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // SPS NAL: prefix 4-byte start code, NAL header (forbidden=0,
        // ref_idc=3, type=7), then RBSP bytes.
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.push(0x67); // 0_11_00111 = ref_idc=3 type=7
        out.extend_from_slice(sps_rbsp);
        // IDR NAL: prefix start code, header (ref_idc=3 type=5), then
        // a single dummy byte (slice data we don't read).
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.push(0x65); // 0_11_00101 = ref_idc=3 type=5
        out.push(0x80);
        out
    }

    #[test]
    fn end_to_end_decode_yields_bgra_buffer() {
        // SPS RBSP from h264_slice tests: 640x480 Baseline.
        let sps_rbsp = vec![66, 0, 30, 0xF4, 0x05, 0x01, 0xE8];
        let bs = synth_bitstream(&sps_rbsp);
        let decoded = decode_first_idr(&bs).expect("decode");
        assert_eq!(decoded.width, 640);
        assert_eq!(decoded.height, 480);
        assert_eq!(decoded.bgra.len(), (640 * 480) as usize);
        // YUV (128, 128, 128) → grey → BGRA channels roughly equal.
        let p = decoded.bgra[640 * 100 + 320];
        let r = ((p >> 16) & 0xFF) as i32;
        let g = ((p >> 8) & 0xFF) as i32;
        let b = (p & 0xFF) as i32;
        assert!((r - g).abs() < 5);
        assert!((g - b).abs() < 5);
        assert!((100..=160).contains(&r));
    }

    #[test]
    fn no_sps_returns_error() {
        let bs = vec![0, 0, 0, 1, 0x65, 0x80];
        let err = decode_first_idr(&bs).unwrap_err();
        assert_eq!(err, DriverError::NoSps);
    }

    #[test]
    fn no_idr_returns_error() {
        let sps_rbsp = vec![66, 0, 30, 0xF4, 0x05, 0x01, 0xE8];
        let mut bs = Vec::new();
        bs.extend_from_slice(&[0, 0, 0, 1]);
        bs.push(0x67);
        bs.extend_from_slice(&sps_rbsp);
        let err = decode_first_idr(&bs).unwrap_err();
        assert_eq!(err, DriverError::NoIdr);
    }
}
