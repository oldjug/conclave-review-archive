//! H.264 / AVC bitstream front-end.
//!
//! Implements the pieces of ITU-T H.264 needed to identify NAL
//! units, strip emulation-prevention bytes, decode unsigned and
//! signed Exp-Golomb codes from the RBSP bit stream, and parse a
//! sufficient subset of the SPS (Sequence Parameter Set) to recover
//! the coded picture size.
//!
//! Out of scope for this slice: PPS, slice headers, CABAC/CAVLC,
//! intra/inter prediction, IDCT, deblock. Those land in follow-up
//! slices once the front-end is locked.

/// Annex B start-code scan: split a byte-stream into NAL units. The
/// returned slices are the *raw* NAL bytes including the NAL header
/// byte — emulation prevention has NOT been stripped yet (that's the
/// next step). Both 3-byte `00 00 01` and 4-byte `00 00 00 01`
/// start codes are recognized.
pub fn split_nal_units(bytes: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut nal_start: Option<usize> = None;
    while i < bytes.len() {
        // Look for a start code at position i.
        let sc_len = start_code_len(bytes, i);
        if sc_len > 0 {
            if let Some(s) = nal_start.take() {
                out.push(&bytes[s..i]);
            }
            i += sc_len;
            nal_start = Some(i);
        } else {
            i += 1;
        }
    }
    if let Some(s) = nal_start {
        out.push(&bytes[s..bytes.len()]);
    }
    out
}

fn start_code_len(bytes: &[u8], i: usize) -> usize {
    if i + 3 < bytes.len()
        && bytes[i] == 0
        && bytes[i + 1] == 0
        && bytes[i + 2] == 0
        && bytes[i + 3] == 1
    {
        return 4;
    }
    if i + 2 < bytes.len() && bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 {
        return 3;
    }
    0
}

/// Per-NAL-unit metadata extracted from the header byte (first byte
/// of the NAL payload after the start code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NalHeader {
    pub forbidden_zero_bit: u8,
    pub nal_ref_idc: u8,
    pub nal_unit_type: u8,
}

impl NalHeader {
    pub fn parse(byte: u8) -> Self {
        Self {
            forbidden_zero_bit: (byte >> 7) & 0x01,
            nal_ref_idc: (byte >> 5) & 0x03,
            nal_unit_type: byte & 0x1F,
        }
    }
}

/// Common nal_unit_type values from H.264 Table 7-1.
pub mod nal_type {
    pub const SLICE_NON_IDR: u8 = 1;
    pub const SLICE_DATA_A: u8 = 2;
    pub const SLICE_DATA_B: u8 = 3;
    pub const SLICE_DATA_C: u8 = 4;
    pub const SLICE_IDR: u8 = 5;
    pub const SEI: u8 = 6;
    pub const SPS: u8 = 7;
    pub const PPS: u8 = 8;
    pub const AUD: u8 = 9;
    pub const END_OF_SEQUENCE: u8 = 10;
    pub const END_OF_STREAM: u8 = 11;
}

/// Strip the H.264 emulation-prevention bytes. Whenever the encoder
/// would otherwise emit `00 00 00`, `00 00 01`, `00 00 02`, or
/// `00 00 03`, it inserts a `03` after the leading `00 00`. The
/// decoder removes them to recover the raw RBSP.
pub fn ebsp_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut i = 0;
    while i < ebsp.len() {
        if i + 2 < ebsp.len() && ebsp[i] == 0 && ebsp[i + 1] == 0 && ebsp[i + 2] == 0x03 {
            out.push(0);
            out.push(0);
            i += 3; // skip the 0x03 emulation byte
        } else {
            out.push(ebsp[i]);
            i += 1;
        }
    }
    out
}

/// MSB-first bit reader over an RBSP byte buffer. Implements the
/// Exp-Golomb (ue/se) codes described in H.264 §9.1.
#[derive(Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize, // absolute bit index from the MSB of data[0]
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    pub fn bits_left(&self) -> usize {
        self.data.len() * 8 - self.bit_pos.min(self.data.len() * 8)
    }

    pub fn read_bit(&mut self) -> Option<u8> {
        if self.bit_pos >= self.data.len() * 8 {
            return None;
        }
        let byte = self.data[self.bit_pos / 8];
        let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
        self.bit_pos += 1;
        Some(bit)
    }

    /// Read `n` bits (n ≤ 32) as a big-endian unsigned integer.
    pub fn read_bits(&mut self, n: u32) -> Option<u32> {
        debug_assert!(n <= 32);
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | (self.read_bit()? as u32);
        }
        Some(v)
    }

    /// Unsigned Exp-Golomb (H.264 §9.1.1).
    pub fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0u32;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 32 {
                return None;
            }
        }
        if leading_zeros == 0 {
            return Some(0);
        }
        let tail = self.read_bits(leading_zeros)?;
        Some((1u32 << leading_zeros) - 1 + tail)
    }

    /// Signed Exp-Golomb (H.264 §9.1.2): map ue→se via the
    /// standard alternating sequence 0, 1, -1, 2, -2, ...
    pub fn read_se(&mut self) -> Option<i32> {
        let v = self.read_ue()?;
        let sign = if v & 1 == 1 { 1 } else { -1 };
        Some(sign * (((v + 1) >> 1) as i32))
    }
}

/// Decoded subset of the SPS. Sufficient to surface picture geometry
/// to the rest of the renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sps {
    pub profile_idc: u8,
    pub level_idc: u8,
    pub seq_parameter_set_id: u32,
    pub log2_max_frame_num_minus4: u32,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    /// True = frames only; False = field-coded interlace can be present.
    pub frame_mbs_only_flag: bool,
    /// Coded picture width in luma samples (after applying mb size).
    pub width: u32,
    pub height: u32,
}

/// Parse SPS RBSP (the bytes after the NAL header byte, with
/// emulation prevention already stripped).
///
/// Implements the subset of H.264 §7.3.2.1.1 needed to extract
/// picture geometry. Profile-specific extensions for High and beyond
/// (`chroma_format_idc`, `bit_depth_luma_minus8`, scaling lists, etc.)
/// are read and discarded so the bit stream stays in sync up to the
/// dimensions we care about.
pub fn parse_sps(rbsp: &[u8]) -> Option<Sps> {
    if rbsp.len() < 4 {
        return None;
    }
    let profile_idc = rbsp[0];
    // constraint_set flags + reserved_zero_2bits in rbsp[1]
    let level_idc = rbsp[2];
    let mut br = BitReader::new(&rbsp[3..]);
    let seq_parameter_set_id = br.read_ue()?;
    // Skip High-profile chroma/bitdepth signalling. The profile_idc
    // values that trigger this block come from H.264 §7.4.2.1.1.
    match profile_idc {
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135 => {
            let chroma_format_idc = br.read_ue()?;
            if chroma_format_idc == 3 {
                br.read_bit()?; // separate_colour_plane_flag
            }
            br.read_ue()?; // bit_depth_luma_minus8
            br.read_ue()?; // bit_depth_chroma_minus8
            br.read_bit()?; // qpprime_y_zero_transform_bypass_flag
            let seq_scaling_matrix_present_flag = br.read_bit()? == 1;
            if seq_scaling_matrix_present_flag {
                // We don't actually need the lists for picture
                // geometry, but we must read past them to keep the
                // bit position correct. Skip up to 8 (or 12) flag
                // bits — and if a list is "present", drain a fixed
                // worst-case number of se() codes so any well-formed
                // SPS we ever see in the wild stays in sync. Since
                // none of the geometry fields below this depend on
                // the lists, we bail to None if we run out of bits.
                let lists = if chroma_format_idc == 3 { 12 } else { 8 };
                for i in 0..lists {
                    let present = br.read_bit()? == 1;
                    if present {
                        let size = if i < 6 { 16 } else { 64 };
                        let mut last_scale: i32 = 8;
                        let mut next_scale: i32 = 8;
                        for _ in 0..size {
                            if next_scale != 0 {
                                let delta = br.read_se()?;
                                next_scale = (last_scale + delta + 256) % 256;
                            }
                            if next_scale != 0 {
                                last_scale = next_scale;
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
    let log2_max_frame_num_minus4 = br.read_ue()?;
    let pic_order_cnt_type = br.read_ue()?;
    match pic_order_cnt_type {
        0 => {
            br.read_ue()?; // log2_max_pic_order_cnt_lsb_minus4
        }
        1 => {
            br.read_bit()?; // delta_pic_order_always_zero_flag
            br.read_se()?; // offset_for_non_ref_pic
            br.read_se()?; // offset_for_top_to_bottom_field
            let n = br.read_ue()?;
            for _ in 0..n {
                br.read_se()?;
            }
        }
        2 => {}
        _ => return None,
    }
    br.read_ue()?; // max_num_ref_frames
    br.read_bit()?; // gaps_in_frame_num_value_allowed_flag
    let pic_width_in_mbs_minus1 = br.read_ue()?;
    let pic_height_in_map_units_minus1 = br.read_ue()?;
    let frame_mbs_only_flag = br.read_bit()? == 1;
    let width = (pic_width_in_mbs_minus1 + 1) * 16;
    let height =
        (pic_height_in_map_units_minus1 + 1) * 16 * (if frame_mbs_only_flag { 1 } else { 2 });
    Some(Sps {
        profile_idc,
        level_idc,
        seq_parameter_set_id,
        log2_max_frame_num_minus4,
        pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1,
        frame_mbs_only_flag,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_finds_three_nals_with_mixed_start_codes() {
        // [0001 NAL_A] [001 NAL_B] [0001 NAL_C]
        let bytes = vec![
            0, 0, 0, 1, 0xAA, 0xBB, // NAL_A
            0, 0, 1, 0xCC, // NAL_B
            0, 0, 0, 1, 0xDD, 0xEE, 0xFF, // NAL_C
        ];
        let nals = split_nal_units(&bytes);
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0], &[0xAA, 0xBB]);
        assert_eq!(nals[1], &[0xCC]);
        assert_eq!(nals[2], &[0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn nal_header_parses_idr_slice() {
        // nal_ref_idc=3, nal_unit_type=5 (IDR) → 0b0_11_00101 = 0x65
        let h = NalHeader::parse(0x65);
        assert_eq!(h.forbidden_zero_bit, 0);
        assert_eq!(h.nal_ref_idc, 3);
        assert_eq!(h.nal_unit_type, nal_type::SLICE_IDR);
    }

    #[test]
    fn ebsp_strips_emulation_byte() {
        // 00 00 03 04 → 00 00 04
        let ebsp = vec![0, 0, 0x03, 0x04];
        let rbsp = ebsp_to_rbsp(&ebsp);
        assert_eq!(rbsp, vec![0, 0, 0x04]);
    }

    #[test]
    fn ebsp_leaves_non_match_alone() {
        let ebsp = vec![0xAB, 0xCD, 0xEF];
        let rbsp = ebsp_to_rbsp(&ebsp);
        assert_eq!(rbsp, ebsp);
    }

    #[test]
    fn ue_decodes_table_values() {
        // codeNum=0  → "1"
        // codeNum=1  → "010"
        // codeNum=2  → "011"
        // codeNum=3  → "00100"
        // Concat: 1 010 011 00100 = 1_010_011_00100 → 0b10100110_0100_0000
        let data = vec![0b1010_0110, 0b0100_0000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_ue(), Some(0));
        assert_eq!(br.read_ue(), Some(1));
        assert_eq!(br.read_ue(), Some(2));
        assert_eq!(br.read_ue(), Some(3));
    }

    #[test]
    fn se_decodes_alternating_sign_sequence() {
        // ue 0,1,2,3,4 → se 0,1,-1,2,-2
        // ue stream: 1 010 011 00100 00101 → 5 codes
        // bits: 1_010_011_00100_00101
        //     = 1010 0110 0100 0010 1
        let data = vec![0b1010_0110, 0b0100_0010, 0b1000_0000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_se(), Some(0));
        assert_eq!(br.read_se(), Some(1));
        assert_eq!(br.read_se(), Some(-1));
        assert_eq!(br.read_se(), Some(2));
        assert_eq!(br.read_se(), Some(-2));
    }

    #[test]
    fn parse_sps_baseline_recovers_geometry() {
        // Hand-built minimal Baseline (profile_idc=66) SPS RBSP:
        //   profile_idc = 66
        //   constraint flags + reserved = 0
        //   level_idc = 30
        //   seq_parameter_set_id = 0          → ue "1"
        //   log2_max_frame_num_minus4 = 0     → ue "1"
        //   pic_order_cnt_type = 0            → ue "1"
        //   log2_max_pic_order_cnt_lsb_minus4 = 0 → ue "1"
        //   max_num_ref_frames = 1            → ue "010"
        //   gaps_in_frame_num_allowed = 0     → 1 bit "0"
        //   pic_width_in_mbs_minus1 = 39      → 40 mbs * 16 = 640 px
        //     ue(39) = unary 5 zeros, then 5-bit binary of (40-32)=8 → "00000_101000"
        //   pic_height_in_map_units_minus1 = 29 → 30 * 16 = 480 px
        //     ue(29) = unary 4 zeros, then 4-bit binary of (30-16)=14 → "00001_1110"
        //   frame_mbs_only_flag = 1
        //   [stop — we don't decode further fields]
        //
        // Bit stream (MSB first):
        //   1 1 1 1 010 0 00000_1_01000 0000_1_1110 1
        //
        // Flat: 1111_0100 | 0000_0101 | 0000_0001 | 1110_1 (+ padding)
        //   = F4 05 01 E8
        let rbsp = vec![66, 0, 30, 0xF4, 0x05, 0x01, 0xE8];
        let sps = parse_sps(&rbsp).expect("parse_sps");
        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.level_idc, 30);
        assert_eq!(sps.pic_width_in_mbs_minus1, 39);
        assert_eq!(sps.pic_height_in_map_units_minus1, 29);
        assert!(sps.frame_mbs_only_flag);
        assert_eq!(sps.width, 640);
        assert_eq!(sps.height, 480);
    }
}
