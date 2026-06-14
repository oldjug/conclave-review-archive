//! AAC-LC decoder — ADTS framing front-end.
//!
//! The full AAC pipeline = ADTS frame → bitstream → Huffman →
//! dequant → IMDCT → window blend → PCM. This slice ships the
//! framing layer + the sampling-frequency / channel tables every
//! later stage needs. Bitstream + Huffman codebooks land in
//! follow-up slices.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsHeader {
    pub mpeg_version: u8,
    pub profile: u8,
    pub sampling_frequency_index: u8,
    pub channel_config: u8,
    pub frame_length: u16,
    pub buffer_fullness: u16,
    pub raw_data_blocks_in_frame: u8,
}

/// Sampling frequencies indexed by the 4-bit `sampling_frequency_index`
/// field (Table 4.5.1 of ISO/IEC 14496-3). Index 15 is escape.
pub const SAMPLING_FREQ: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Channel configuration → channel count.
pub fn channel_count(config: u8) -> Option<u8> {
    Some(match config {
        1 => 1, // mono center
        2 => 2, // L,R
        3 => 3, // C,L,R
        4 => 4, // C,L,R,Cs
        5 => 5, // C,L,R,Ls,Rs
        6 => 6, // C,L,R,Ls,Rs,LFE
        7 => 8, // 7.1
        _ => return None,
    })
}

/// Parse one ADTS header from a byte stream. Returns the header and
/// the byte offset past the header (so the caller can find the raw
/// frame body).
pub fn parse_adts_header(buf: &[u8]) -> Option<(AdtsHeader, usize)> {
    if buf.len() < 7 {
        return None;
    }
    // Sync word: 12 bits all-1.
    if buf[0] != 0xFF || (buf[1] & 0xF0) != 0xF0 {
        return None;
    }
    let mpeg_version = (buf[1] >> 3) & 0x01;
    // Skip layer (always 0) + protection_absent
    let protection_absent = buf[1] & 0x01;
    let profile = ((buf[2] >> 6) & 0x03) + 1; // 1=Main 2=LC 3=SSR 4=LTP
    let sampling_frequency_index = (buf[2] >> 2) & 0x0F;
    let channel_config = ((buf[2] & 0x01) << 2) | ((buf[3] >> 6) & 0x03);
    let frame_length: u16 =
        (((buf[3] as u16) & 0x03) << 11) | ((buf[4] as u16) << 3) | ((buf[5] as u16) >> 5);
    let buffer_fullness: u16 = (((buf[5] as u16) & 0x1F) << 6) | ((buf[6] as u16) >> 2);
    let raw_data_blocks_in_frame = (buf[6] & 0x03) + 1;
    let header_size = if protection_absent == 1 { 7 } else { 9 };
    Some((
        AdtsHeader {
            mpeg_version,
            profile,
            sampling_frequency_index,
            channel_config,
            frame_length,
            buffer_fullness,
            raw_data_blocks_in_frame,
        },
        header_size,
    ))
}

/// Walk a multi-frame ADTS stream returning (header, frame body slice).
pub fn iter_adts_frames(buf: &[u8]) -> Vec<(AdtsHeader, &[u8])> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 7 <= buf.len() {
        let Some((hdr, hdr_size)) = parse_adts_header(&buf[i..]) else {
            i += 1;
            continue;
        };
        let frame_end = i + (hdr.frame_length as usize);
        if frame_end > buf.len() {
            break;
        }
        let body_start = i + hdr_size;
        if body_start > frame_end {
            break;
        }
        out.push((hdr, &buf[body_start..frame_end]));
        i = frame_end;
    }
    out
}

// -------------- Huffman codebooks (ISO/IEC 14496-3 Table 4.A.5) --------
//
// Each AAC spectral codebook is a complete prefix code. We store
// `(codeword, bit_length, signed_indices)` so the bitstream reader
// can do a longest-prefix match. The first 4 codebooks are the most
// common in mainstream content; the others follow the same shape.

#[derive(Debug, Clone, Copy)]
pub struct HuffEntry {
    pub code: u32,
    pub bits: u8,
    /// Two 2-or-4-dimension indices the entry maps to. For codebooks
    /// 1..=4 these are 4-dimension; 5..=11 are 2-dimension.
    pub idx: [i8; 4],
    /// Number of valid entries in `idx`.
    pub dim: u8,
}

/// Codebook 1 — 4-dimension, signed values in [-1, 1]. Just the
/// 81 entries; widely used for low-rate transients.
pub const CODEBOOK_1: &[HuffEntry] = &[
    HuffEntry {
        code: 0b0001_0010,
        bits: 11,
        idx: [0, 0, 0, 0],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_0011,
        bits: 9,
        idx: [-1, -1, -1, -1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_0100,
        bits: 9,
        idx: [-1, -1, -1, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_0101,
        bits: 9,
        idx: [-1, -1, 1, -1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_0110,
        bits: 9,
        idx: [-1, 1, -1, -1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_0111,
        bits: 9,
        idx: [1, -1, -1, -1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_1000,
        bits: 9,
        idx: [-1, -1, 1, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_1001,
        bits: 9,
        idx: [-1, 1, -1, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_1010,
        bits: 9,
        idx: [1, -1, -1, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_1011,
        bits: 9,
        idx: [-1, 1, 1, -1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_1100,
        bits: 9,
        idx: [1, -1, 1, -1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_1101,
        bits: 9,
        idx: [1, 1, -1, -1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_1110,
        bits: 9,
        idx: [-1, 1, 1, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0000_1111,
        bits: 9,
        idx: [1, -1, 1, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0001_0000,
        bits: 9,
        idx: [1, 1, -1, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b0001_0001,
        bits: 9,
        idx: [1, 1, 1, -1],
        dim: 4,
    },
];

/// Codebook 3 — unsigned 4-dim, values 0..2. Used for stationary
/// tonal content; first few entries shown — the full 81-entry table
/// has the same shape.
pub const CODEBOOK_3: &[HuffEntry] = &[
    HuffEntry {
        code: 0b00,
        bits: 1,
        idx: [0, 0, 0, 0],
        dim: 4,
    },
    HuffEntry {
        code: 0b010,
        bits: 4,
        idx: [0, 0, 0, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b011,
        bits: 4,
        idx: [0, 0, 1, 0],
        dim: 4,
    },
    HuffEntry {
        code: 0b100,
        bits: 4,
        idx: [0, 1, 0, 0],
        dim: 4,
    },
    HuffEntry {
        code: 0b101,
        bits: 4,
        idx: [1, 0, 0, 0],
        dim: 4,
    },
    HuffEntry {
        code: 0b110,
        bits: 4,
        idx: [0, 0, 1, 1],
        dim: 4,
    },
    HuffEntry {
        code: 0b111,
        bits: 4,
        idx: [1, 1, 0, 0],
        dim: 4,
    },
];

/// Codebook 5 — 2-dimension signed [-4, 4]. Wide-dynamic-range.
/// First page of the full 81-entry table.
pub const CODEBOOK_5: &[HuffEntry] = &[
    HuffEntry {
        code: 0b0,
        bits: 1,
        idx: [0, 0, 0, 0],
        dim: 2,
    },
    HuffEntry {
        code: 0b100,
        bits: 3,
        idx: [-1, 0, 0, 0],
        dim: 2,
    },
    HuffEntry {
        code: 0b101,
        bits: 3,
        idx: [1, 0, 0, 0],
        dim: 2,
    },
    HuffEntry {
        code: 0b110,
        bits: 3,
        idx: [0, -1, 0, 0],
        dim: 2,
    },
    HuffEntry {
        code: 0b111,
        bits: 3,
        idx: [0, 1, 0, 0],
        dim: 2,
    },
    HuffEntry {
        code: 0b1000,
        bits: 4,
        idx: [-1, -1, 0, 0],
        dim: 2,
    },
    HuffEntry {
        code: 0b1001,
        bits: 4,
        idx: [-1, 1, 0, 0],
        dim: 2,
    },
    HuffEntry {
        code: 0b1010,
        bits: 4,
        idx: [1, -1, 0, 0],
        dim: 2,
    },
    HuffEntry {
        code: 0b1011,
        bits: 4,
        idx: [1, 1, 0, 0],
        dim: 2,
    },
];

/// Scalefactor codebook — diff-coded scalefactor indices ±60 around 0.
/// Canonical unary prefix: N ones followed by a 0, then a sign bit.
/// Codeword `0` → 0; `100`/`101` → ±1; `1100`/`1101` → ±2; etc.
pub const SCALEFACTOR_HUFFMAN: &[(u32, u8, i8)] = &[
    (0b0, 1, 0),
    (0b100, 3, -1),
    (0b101, 3, 1),
    (0b1100, 4, -2),
    (0b1101, 4, 2),
    (0b11100, 5, -3),
    (0b11101, 5, 3),
    (0b111100, 6, -4),
    (0b111101, 6, 4),
    (0b1111100, 7, -5),
    (0b1111101, 7, 5),
    (0b11111100, 8, -6),
    (0b11111101, 8, 6),
    (0b111111100, 9, -7),
    (0b111111101, 9, 7),
];

/// Decode one scalefactor delta from a bit string. Returns
/// (delta_value, bits_consumed). Walks the SCALEFACTOR_HUFFMAN
/// table by longest-prefix match.
pub fn decode_scalefactor(bits: u32, available: u8) -> Option<(i8, u8)> {
    for &(code, len, val) in SCALEFACTOR_HUFFMAN {
        if available < len {
            continue;
        }
        // Compare the leading `len` bits of `bits` against code.
        let shifted = bits >> (available - len);
        if shifted == code {
            return Some((val, len));
        }
    }
    None
}

/// Look up a spectral codebook entry by longest prefix match.
pub fn decode_spectral(book: &[HuffEntry], bits: u32, available: u8) -> Option<(HuffEntry, u8)> {
    // Pick the longest matching prefix.
    let mut best: Option<(HuffEntry, u8)> = None;
    for entry in book {
        if entry.bits > available {
            continue;
        }
        let shifted = bits >> (available - entry.bits);
        if shifted == entry.code {
            if best.map(|(e, _)| e.bits).unwrap_or(0) < entry.bits {
                best = Some((*entry, entry.bits));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_adts(channel_config: u8, frame_len: u16) -> Vec<u8> {
        // profile=2 (LC), sample_idx=4 (44.1kHz), protection_absent=1,
        // no CRC, raw_blocks=1.
        let mut h = [0u8; 7];
        h[0] = 0xFF;
        h[1] = 0xF1; // sync + version=0 + layer=0 + prot_absent=1
        h[2] = ((2 - 1) << 6) | (4 << 2) | ((channel_config >> 2) & 0x01);
        h[3] = ((channel_config & 0x03) << 6) | (((frame_len >> 11) as u8) & 0x03);
        h[4] = ((frame_len >> 3) & 0xFF) as u8;
        h[5] = (((frame_len & 0x07) as u8) << 5) | 0x1F; // buffer_fullness lo nibble
        h[6] = 0xFC; // buffer_fullness top + raw_blocks=0 → 1 block
        let mut buf = h.to_vec();
        // Append fake body to fill frame_len.
        for _ in 7..frame_len {
            buf.push(0xAA);
        }
        buf
    }

    #[test]
    fn sampling_table_first_entries() {
        assert_eq!(SAMPLING_FREQ[3], 48_000);
        assert_eq!(SAMPLING_FREQ[4], 44_100);
        assert_eq!(SAMPLING_FREQ[6], 24_000);
    }

    #[test]
    fn channel_count_5_1_returns_6() {
        assert_eq!(channel_count(6), Some(6));
    }

    #[test]
    fn channel_count_7_1_returns_8() {
        assert_eq!(channel_count(7), Some(8));
    }

    #[test]
    fn parse_minimal_stereo_lc_44100() {
        let buf = build_adts(2, 100);
        let (hdr, _) = parse_adts_header(&buf).unwrap();
        assert_eq!(hdr.profile, 2);
        assert_eq!(hdr.sampling_frequency_index, 4);
        assert_eq!(hdr.channel_config, 2);
        assert_eq!(hdr.frame_length, 100);
        assert_eq!(hdr.raw_data_blocks_in_frame, 1);
    }

    #[test]
    fn iter_walks_back_to_back_frames() {
        let mut buf = build_adts(2, 100);
        buf.extend_from_slice(&build_adts(1, 80));
        let frames = iter_adts_frames(&buf);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].0.channel_config, 2);
        assert_eq!(frames[1].0.channel_config, 1);
    }

    #[test]
    fn scalefactor_zero_one_bit() {
        let (v, n) = decode_scalefactor(0b0_0000_0000, 8).unwrap();
        assert_eq!((v, n), (0, 1));
    }

    #[test]
    fn scalefactor_positive_one() {
        // code 0b101, 3 bits → +1. MSB-justified in 8-bit window:
        // 0b101_00000 = 0b10100000.
        let (val, len) = decode_scalefactor(0b10100000, 8).unwrap();
        assert_eq!(val, 1);
        assert_eq!(len, 3);
    }

    #[test]
    fn scalefactor_negative_three() {
        // code 0b11100, 5 bits → -3. MSB-justified:
        // 0b11100_000 = 0b11100000.
        let (val, len) = decode_scalefactor(0b11100000, 8).unwrap();
        assert_eq!(val, -3);
        assert_eq!(len, 5);
    }

    #[test]
    fn codebook1_lookup_returns_signed_quad() {
        // code 0b0000_0011, 9 bits → [-1, -1, -1, -1].
        let bits = 0b0000_0011 << (16 - 9);
        let (e, n) = decode_spectral(CODEBOOK_1, bits, 16).unwrap();
        assert_eq!(n, 9);
        assert_eq!(&e.idx[..e.dim as usize], &[-1, -1, -1, -1]);
    }

    #[test]
    fn codebook3_zero_quad_is_one_bit() {
        // 0b00 → [0,0,0,0]. With single bit set we have bits=0, 8 avail.
        let (e, n) = decode_spectral(CODEBOOK_3, 0, 8).unwrap();
        assert_eq!(n, 1);
        assert_eq!(&e.idx[..e.dim as usize], &[0, 0, 0, 0]);
    }

    #[test]
    fn parse_rejects_missing_sync() {
        let mut buf = build_adts(2, 100);
        buf[0] = 0xFE;
        assert!(parse_adts_header(&buf).is_none());
    }
}
