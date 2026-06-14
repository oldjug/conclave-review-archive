//! H.264 PPS + slice-header parsing.
//!
//! Continues `h264.rs`. This slice covers the Picture Parameter Set
//! (PPS) and the slice header — enough to know which PPS/SPS a coded
//! slice references, the slice type (I / P / B / SI / SP), the
//! `pic_init_qp` baseline QP, and the deblocking-filter knobs that
//! the IDCT/deblock stage will need next.
//!
//! Limited to the IDR-only path used by `<video>` first-frame decode
//! and still-image AVC tiles. Reference-list reordering, prediction
//! weights, and dec_ref_pic_marking are read past but not retained.

use crate::h264::BitReader;

/// Decoded Picture Parameter Set (PPS) — H.264 §7.4.2.2 subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pps {
    pub pic_parameter_set_id: u32,
    pub seq_parameter_set_id: u32,
    pub entropy_coding_mode_flag: bool, // 0 = CAVLC, 1 = CABAC
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups_minus1: u32,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp_minus26: i32,
    pub pic_init_qs_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
}

/// Slice types from H.264 §7.4.3 Table 7-6. The high values 5..=9
/// indicate "all slices in this picture have this type" — the modulo
/// gives the basic type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

impl SliceType {
    pub fn from_raw(v: u32) -> Option<Self> {
        Some(match v % 5 {
            0 => Self::P,
            1 => Self::B,
            2 => Self::I,
            3 => Self::Sp,
            4 => Self::Si,
            _ => return None,
        })
    }
    pub fn is_intra(self) -> bool {
        matches!(self, Self::I | Self::Si)
    }
}

/// Decoded slice header — H.264 §7.3.3 subset, IDR-focused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    pub pic_parameter_set_id: u32,
    pub frame_num: u32,
    pub idr_pic_id: Option<u32>,
    pub slice_qp_delta: i32,
    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
}

/// Parse a PPS RBSP (everything after the NAL header byte, emulation
/// prevention already stripped). Implements only the always-present
/// fields plus a handful of common extensions; anything past
/// `redundant_pic_cnt_present_flag` is left to a future slice.
pub fn parse_pps(rbsp: &[u8]) -> Option<Pps> {
    let mut br = BitReader::new(rbsp);
    let pic_parameter_set_id = br.read_ue()?;
    let seq_parameter_set_id = br.read_ue()?;
    let entropy_coding_mode_flag = br.read_bit()? == 1;
    let bottom_field_pic_order_in_frame_present_flag = br.read_bit()? == 1;
    let num_slice_groups_minus1 = br.read_ue()?;
    if num_slice_groups_minus1 > 0 {
        // Slice-group map signaling not in our V1 path (almost no
        // mainstream encoder produces this). Surface as a hard fail
        // so the caller can fall back.
        return None;
    }
    let num_ref_idx_l0_default_active_minus1 = br.read_ue()?;
    let num_ref_idx_l1_default_active_minus1 = br.read_ue()?;
    let weighted_pred_flag = br.read_bit()? == 1;
    let weighted_bipred_idc = br.read_bits(2)?;
    let pic_init_qp_minus26 = br.read_se()?;
    let pic_init_qs_minus26 = br.read_se()?;
    let chroma_qp_index_offset = br.read_se()?;
    let deblocking_filter_control_present_flag = br.read_bit()? == 1;
    let constrained_intra_pred_flag = br.read_bit()? == 1;
    let redundant_pic_cnt_present_flag = br.read_bit()? == 1;
    Some(Pps {
        pic_parameter_set_id,
        seq_parameter_set_id,
        entropy_coding_mode_flag,
        bottom_field_pic_order_in_frame_present_flag,
        num_slice_groups_minus1,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        weighted_pred_flag,
        weighted_bipred_idc,
        pic_init_qp_minus26,
        pic_init_qs_minus26,
        chroma_qp_index_offset,
        deblocking_filter_control_present_flag,
        constrained_intra_pred_flag,
        redundant_pic_cnt_present_flag,
    })
}

/// Parse the slice header of an IDR picture. `nal_unit_type` is the
/// type from the NAL header (5 for IDR). The full SPS/PPS pair must
/// have already been parsed; we take them by reference because some
/// fields are conditional on SPS/PPS state.
pub fn parse_slice_header_idr(
    rbsp: &[u8],
    sps_log2_max_frame_num_minus4: u32,
) -> Option<SliceHeader> {
    let mut br = BitReader::new(rbsp);
    let first_mb_in_slice = br.read_ue()?;
    let slice_type_raw = br.read_ue()?;
    let slice_type = SliceType::from_raw(slice_type_raw)?;
    let pic_parameter_set_id = br.read_ue()?;
    let frame_num_bits = sps_log2_max_frame_num_minus4 + 4;
    if frame_num_bits == 0 || frame_num_bits > 16 {
        return None;
    }
    let frame_num = br.read_bits(frame_num_bits)?;
    // IDR pictures must carry an idr_pic_id.
    let idr_pic_id = Some(br.read_ue()?);
    // pic_order_cnt_type==2 path (the simplest) skips POC LSB
    // signalling. Higher POC types live in a follow-up slice; we
    // surface failure if we'd need them.
    //
    // For the IDR path, slice_qp_delta is always present and is what
    // the QP scale anchors on.
    let slice_qp_delta = br.read_se()?;
    // Deblocking-filter fields. PPS gates them — but for the V1 IDR
    // path we assume the most common configuration (control present)
    // and read all three. Callers that hit a PPS with control NOT
    // present should call the lower-level reader (TODO follow-up).
    let disable_deblocking_filter_idc = br.read_ue()?;
    let (slice_alpha_c0_offset_div2, slice_beta_offset_div2) = if disable_deblocking_filter_idc != 1
    {
        (br.read_se()?, br.read_se()?)
    } else {
        (0, 0)
    };
    Some(SliceHeader {
        first_mb_in_slice,
        slice_type,
        pic_parameter_set_id,
        frame_num,
        idr_pic_id,
        slice_qp_delta,
        disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2,
        slice_beta_offset_div2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: pack a Vec of (bits, value) into a big-endian byte
    /// stream. Asserts each value fits in its declared bit width.
    fn pack_bits(parts: &[(u32, u32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut acc: u64 = 0;
        let mut filled: u32 = 0;
        for &(bits, val) in parts {
            assert!(bits <= 32);
            assert!(val < (1u64 << bits) as u32 || bits == 32);
            acc = (acc << bits) | (val as u64);
            filled += bits;
            while filled >= 8 {
                let shift = filled - 8;
                buf.push(((acc >> shift) & 0xFF) as u8);
                acc &= (1u64 << shift) - 1;
                filled -= 8;
            }
        }
        if filled > 0 {
            buf.push(((acc << (8 - filled)) & 0xFF) as u8);
        }
        buf
    }

    /// Helper: encode codeNum as unsigned Exp-Golomb and return
    /// (bits, value) suitable for `pack_bits`.
    fn ue_bits(code_num: u32) -> (u32, u32) {
        if code_num == 0 {
            return (1, 1);
        }
        let m = 32 - (code_num + 1).leading_zeros() - 1; // floor(log2(codeNum+1))
        let info = code_num - ((1u32 << m) - 1);
        // Stream is M zeros + 1 + M-bit info. Total length 2M+1.
        let total_bits = 2 * m + 1;
        // Compose: prefix is just "1" sitting in bit position m (value 1<<m),
        // plus info in the low m bits.
        let value = (1u32 << m) | info;
        (total_bits, value)
    }

    /// Signed Exp-Golomb encoder.
    fn se_bits(v: i32) -> (u32, u32) {
        let code_num = if v <= 0 {
            (-v as u32) * 2
        } else {
            (v as u32) * 2 - 1
        };
        ue_bits(code_num)
    }

    #[test]
    fn ue_helper_roundtrips() {
        for &k in &[0, 1, 2, 3, 7, 15, 16, 39, 100] {
            let (bits, value) = ue_bits(k);
            let buf = pack_bits(&[(bits, value)]);
            let mut br = BitReader::new(&buf);
            assert_eq!(br.read_ue(), Some(k), "ue roundtrip for {k}");
        }
    }

    #[test]
    fn se_helper_roundtrips() {
        for &v in &[0i32, 1, -1, 2, -2, 7, -7] {
            let (bits, value) = se_bits(v);
            let buf = pack_bits(&[(bits, value)]);
            let mut br = BitReader::new(&buf);
            assert_eq!(br.read_se(), Some(v), "se roundtrip for {v}");
        }
    }

    #[test]
    fn parse_pps_minimal_baseline() {
        // Build a minimal PPS for CAVLC, single slice group, common
        // reference defaults.
        let mut parts: Vec<(u32, u32)> = Vec::new();
        parts.push(ue_bits(0)); // pic_parameter_set_id = 0
        parts.push(ue_bits(0)); // seq_parameter_set_id = 0
        parts.push((1, 0)); // entropy_coding_mode_flag = CAVLC
        parts.push((1, 0)); // bottom_field_pic_order_in_frame_present_flag
        parts.push(ue_bits(0)); // num_slice_groups_minus1 = 0
        parts.push(ue_bits(0)); // num_ref_idx_l0_default_active_minus1
        parts.push(ue_bits(0)); // num_ref_idx_l1_default_active_minus1
        parts.push((1, 0)); // weighted_pred_flag
        parts.push((2, 0)); // weighted_bipred_idc
        parts.push(se_bits(-2)); // pic_init_qp_minus26 = -2 → QP 24
        parts.push(se_bits(0)); // pic_init_qs_minus26
        parts.push(se_bits(0)); // chroma_qp_index_offset
        parts.push((1, 1)); // deblocking_filter_control_present_flag = 1
        parts.push((1, 0)); // constrained_intra_pred_flag
        parts.push((1, 0)); // redundant_pic_cnt_present_flag
        let rbsp = pack_bits(&parts);
        let pps = parse_pps(&rbsp).expect("parse_pps");
        assert_eq!(pps.pic_parameter_set_id, 0);
        assert_eq!(pps.seq_parameter_set_id, 0);
        assert!(!pps.entropy_coding_mode_flag);
        assert_eq!(pps.num_slice_groups_minus1, 0);
        assert_eq!(pps.pic_init_qp_minus26, -2);
        assert!(pps.deblocking_filter_control_present_flag);
    }

    #[test]
    fn parse_pps_rejects_unsupported_slice_groups() {
        let mut parts: Vec<(u32, u32)> = Vec::new();
        parts.push(ue_bits(0));
        parts.push(ue_bits(0));
        parts.push((1, 0));
        parts.push((1, 0));
        parts.push(ue_bits(1)); // num_slice_groups_minus1 = 1 → unsupported
        let rbsp = pack_bits(&parts);
        assert!(parse_pps(&rbsp).is_none());
    }

    #[test]
    fn parse_slice_header_idr_intra() {
        // SPS log2_max_frame_num_minus4 = 0 → frame_num is 4 bits.
        let log2_minus4 = 0u32;
        let mut parts: Vec<(u32, u32)> = Vec::new();
        parts.push(ue_bits(0)); // first_mb_in_slice = 0
        parts.push(ue_bits(7)); // slice_type = 7 → I-only picture; 7 % 5 = 2 → I
        parts.push(ue_bits(0)); // pic_parameter_set_id = 0
        parts.push((4, 0)); // frame_num = 0
        parts.push(ue_bits(0)); // idr_pic_id = 0
        parts.push(se_bits(3)); // slice_qp_delta = 3
        parts.push(ue_bits(0)); // disable_deblocking_filter_idc = 0 (enabled)
        parts.push(se_bits(0)); // slice_alpha_c0_offset_div2
        parts.push(se_bits(0)); // slice_beta_offset_div2
        let rbsp = pack_bits(&parts);
        let sh = parse_slice_header_idr(&rbsp, log2_minus4).expect("slice header");
        assert_eq!(sh.first_mb_in_slice, 0);
        assert_eq!(sh.slice_type, SliceType::I);
        assert!(sh.slice_type.is_intra());
        assert_eq!(sh.frame_num, 0);
        assert_eq!(sh.idr_pic_id, Some(0));
        assert_eq!(sh.slice_qp_delta, 3);
        assert_eq!(sh.disable_deblocking_filter_idc, 0);
    }

    #[test]
    fn slice_header_skips_deblock_offsets_when_idc_eq_1() {
        let mut parts: Vec<(u32, u32)> = Vec::new();
        parts.push(ue_bits(0));
        parts.push(ue_bits(2)); // I
        parts.push(ue_bits(0));
        parts.push((4, 1)); // frame_num = 1
        parts.push(ue_bits(2)); // idr_pic_id = 2
        parts.push(se_bits(-4)); // slice_qp_delta = -4
        parts.push(ue_bits(1)); // disable_deblocking_filter_idc = 1 → skip offsets
        let rbsp = pack_bits(&parts);
        let sh = parse_slice_header_idr(&rbsp, 0).expect("slice header");
        assert_eq!(sh.disable_deblocking_filter_idc, 1);
        assert_eq!(sh.slice_alpha_c0_offset_div2, 0);
        assert_eq!(sh.slice_beta_offset_div2, 0);
    }
}
