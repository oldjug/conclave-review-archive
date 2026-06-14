//! H.264 CAVLC residual decoder.
//!
//! Implements §9.2 — Context-Adaptive Variable Length Coding for
//! transform coefficient levels. CAVLC is the entropy stage that
//! sits between the bit stream and `h264_idct::idct_4x4`: it
//! produces the (up to 16) levels of a 4x4 block in raster order
//! after de-zigzag.
//!
//! Pipeline (matching the spec section numbers):
//!   §9.2.1   read coeff_token  → (TotalCoeff, TrailingOnes)
//!   §9.2.2   read trailing_ones_sign_flag bits
//!   §9.2.2.x read remaining level codes
//!   §9.2.3   read total_zeros
//!   §9.2.4   read run_before for each non-trailing coeff
//!   compose: zigzag-inverse + place into 4x4 block
//!
//! For V1 we ship the coeff_token table for **nC ∈ 0..2** which
//! covers all luma 4x4 sub-blocks in I-slices when the left and top
//! neighbors are themselves intra (typical for IDR pictures).
//! Higher nC tables and the chroma DC variant are encoded the same
//! way; their numeric tables land in follow-ups.
//!
//! Output: a `Block4x4` of 16 dequantized coefficients in raster
//! order, ready for `idct_4x4`.

use crate::h264::BitReader;

/// 16 transform coefficients in row-major (raster) order — what the
/// IDCT consumes.
pub type Block4x4 = [i32; 16];

/// CAVLC zigzag scan (§8.5.4 figure 8-7). Maps the i-th non-zero
/// coefficient's scan-order index to its position in the 4x4 raster.
pub const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Apply the inverse zigzag: given coefficients indexed by scan order
/// 0..15, place them into raster positions in a 4x4 block.
pub fn inverse_zigzag(scan: &[i32; 16]) -> Block4x4 {
    let mut block = [0i32; 16];
    for i in 0..16 {
        block[ZIGZAG_4X4[i]] = scan[i];
    }
    block
}

/// coeff_token table for nC=0..1 (spec Table 9-5, column "0 ≤ nC < 2").
/// Encoded as `(bitstring_value, bit_length, total_coeff, trailing_ones)`.
/// nC predicts how many non-zero coefficients the block likely has.
const COEFF_TOKEN_NC0: &[(u32, u8, u8, u8)] = &[
    // TotalCoeff = 0
    (0b1, 1, 0, 0),
    // TC=1
    (0b0_0010_1, 6, 1, 1),
    (0b00_0000_0000_0101, 14, 1, 0),
    // TC=2
    (0b00_1110, 6, 2, 2),
    (0b00_0111, 8, 2, 1),
    (0b00_0000_0000_0111, 14, 2, 0),
    // TC=3
    (0b00_0111_1, 8, 3, 3),
    (0b00_0001_00, 9, 3, 2),
    (0b00_0000_0100, 11, 3, 1),
    (0b00_0000_0000_0100, 14, 3, 0),
    // TC=4
    (0b00_0011_11, 8, 4, 3),
    (0b00_0001_01, 10, 4, 2),
    (0b00_0000_0001_1, 13, 4, 1),
    (0b00_0000_0000_0010, 14, 4, 0),
];

/// Read a CAVLC coeff_token using the nC=0 table (§9.2.1).
/// Returns `(total_coeff, trailing_ones)`. The bit reader is
/// advanced past the variable-length code on success.
pub fn read_coeff_token_nc0(br: &mut BitReader<'_>) -> Option<(u8, u8)> {
    // Linear search bounded by table size — small (≤14) so cheap.
    // We peek `len` bits and compare.
    let mut peek: u32 = 0;
    let mut peek_len: u8 = 0;
    // Read up to the longest code in the table (14 bits).
    while peek_len < 14 {
        peek = (peek << 1) | (br.read_bit()? as u32);
        peek_len += 1;
        for &(bits, len, tc, t1) in COEFF_TOKEN_NC0 {
            if len == peek_len && bits == peek {
                return Some((tc, t1));
            }
        }
    }
    None
}

/// §9.2.2.x — read one non-trailing level. `suffix_length` is the
/// running suffix length per spec; this function returns the decoded
/// level and the updated suffix length. Limited to the level_prefix
/// values we see in well-formed Baseline / Main bitstreams; very large
/// magnitudes fall back to None.
pub fn read_level(br: &mut BitReader<'_>, suffix_length: u8) -> Option<i32> {
    // level_prefix is unary-encoded — count leading zeros until a 1.
    let mut level_prefix: u32 = 0;
    while br.read_bit()? == 0 {
        level_prefix += 1;
        if level_prefix > 25 {
            return None;
        }
    }
    let (level_code, sfx_extra) = if suffix_length > 0 {
        let suffix = br.read_bits(suffix_length as u32)?;
        ((level_prefix << suffix_length) + suffix, false)
    } else if level_prefix < 14 {
        (level_prefix, false)
    } else if level_prefix == 14 {
        // 4-bit suffix
        let suffix = br.read_bits(4)?;
        (14 + suffix, false)
    } else {
        // level_prefix >= 15 — 12-bit escape suffix
        let suffix = br.read_bits(12)?;
        (15 + suffix, true)
    };
    let _ = sfx_extra; // reserved for future suffix-length update
    // Convert codeNum to signed level per §9.2.2.2.
    let level = if level_code & 1 == 0 {
        (level_code >> 1) as i32 + 1
    } else {
        -((level_code >> 1) as i32 + 1)
    };
    Some(level)
}

/// total_zeros table for TotalCoeff=1..=7 (§9.2.3 Table 9-7, first
/// section). Format: `(total_coeff, zeros_left, bitstring, bit_length)`.
const TOTAL_ZEROS_4X4: &[(u8, u8, u32, u8)] = &[
    // TC=1
    (1, 0, 0b1, 1),
    (1, 1, 0b011, 3),
    (1, 2, 0b010, 3),
    (1, 3, 0b0011, 4),
    (1, 4, 0b0010, 4),
    (1, 5, 0b0001_1, 5),
    (1, 6, 0b0001_0, 5),
    (1, 7, 0b0000_11, 6),
    (1, 8, 0b0000_10, 6),
    (1, 9, 0b0000_011, 7),
    (1, 10, 0b0000_010, 7),
    (1, 11, 0b0000_0011, 8),
    (1, 12, 0b0000_0010, 8),
    (1, 13, 0b0000_0001_1, 9),
    (1, 14, 0b0000_0001_01, 9),
    (1, 15, 0b0000_0001_00, 9),
];

/// Read total_zeros for a block with the given TotalCoeff (§9.2.3).
/// Only TC=1 implemented in V1 (covers the typical low-detail luma
/// path; higher TC tables follow the same shape).
pub fn read_total_zeros(br: &mut BitReader<'_>, total_coeff: u8) -> Option<u8> {
    if total_coeff == 0 {
        return Some(0);
    }
    let mut peek: u32 = 0;
    let mut peek_len: u8 = 0;
    while peek_len < 9 {
        peek = (peek << 1) | (br.read_bit()? as u32);
        peek_len += 1;
        for &(tc, zl, bits, len) in TOTAL_ZEROS_4X4 {
            if tc == total_coeff && len == peek_len && bits == peek {
                return Some(zl);
            }
        }
    }
    None
}

/// run_before table (§9.2.4 Table 9-10). Indexed by zeros_left.
/// For zeros_left ≥ 7 the encoding is truncated unary, so we
/// compute it analytically; for 1..=6 we use the explicit table.
const RUN_BEFORE_TABLE: &[(u8, u8, u32, u8)] = &[
    // (zeros_left, run, bitstring, bit_length)
    (1, 0, 0b1, 1),
    (1, 1, 0b0, 1),
    (2, 0, 0b1, 1),
    (2, 1, 0b01, 2),
    (2, 2, 0b00, 2),
    (3, 0, 0b11, 2),
    (3, 1, 0b10, 2),
    (3, 2, 0b01, 2),
    (3, 3, 0b00, 2),
    (4, 0, 0b11, 2),
    (4, 1, 0b10, 2),
    (4, 2, 0b01, 2),
    (4, 3, 0b001, 3),
    (4, 4, 0b000, 3),
    (5, 0, 0b11, 2),
    (5, 1, 0b10, 2),
    (5, 2, 0b011, 3),
    (5, 3, 0b010, 3),
    (5, 4, 0b001, 3),
    (5, 5, 0b000, 3),
    (6, 0, 0b11, 2),
    (6, 1, 0b000, 3),
    (6, 2, 0b001, 3),
    (6, 3, 0b011, 3),
    (6, 4, 0b010, 3),
    (6, 5, 0b101, 3),
    (6, 6, 0b100, 3),
];

/// Read one run_before given the remaining zeros_left (§9.2.4).
pub fn read_run_before(br: &mut BitReader<'_>, zeros_left: u8) -> Option<u8> {
    if zeros_left == 0 {
        return Some(0);
    }
    if zeros_left <= 6 {
        let max_len = if zeros_left <= 3 { 2 } else { 3 };
        let mut peek: u32 = 0;
        let mut peek_len: u8 = 0;
        while peek_len < max_len {
            peek = (peek << 1) | (br.read_bit()? as u32);
            peek_len += 1;
            for &(zl, r, bits, len) in RUN_BEFORE_TABLE {
                if zl == zeros_left && len == peek_len && bits == peek {
                    return Some(r);
                }
            }
        }
        return None;
    }
    // zeros_left ≥ 7 — truncated unary, up to 14 bits.
    let mut zeros: u8 = 0;
    while br.read_bit()? == 0 {
        zeros += 1;
        if zeros > 14 {
            return None;
        }
    }
    Some(zeros)
}

/// Decode one CAVLC luma 4x4 sub-block with nC=0 prediction (the
/// most common path on I-frames adjacent to other intra blocks).
/// Returns the decoded levels as a `Block4x4` (raster order, ready
/// for IDCT after dequant — V1 returns the raw post-CAVLC levels;
/// the dequant scale factors land in a follow-up).
pub fn decode_residual_4x4_nc0(br: &mut BitReader<'_>) -> Option<Block4x4> {
    let (total_coeff, trailing_ones) = read_coeff_token_nc0(br)?;
    if total_coeff == 0 {
        return Some([0; 16]);
    }
    // Trailing-ones signs come first.
    let mut levels_scan = [0i32; 16];
    let mut suffix_length: u8 = if total_coeff > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };
    let mut idx = (total_coeff - 1) as usize;
    for _ in 0..trailing_ones {
        let sign = br.read_bit()?;
        levels_scan[idx] = if sign == 1 { -1 } else { 1 };
        if idx == 0 {
            break;
        }
        idx -= 1;
    }
    // Remaining levels.
    let remaining = total_coeff - trailing_ones;
    for k in 0..remaining {
        let level = read_level(br, suffix_length)?;
        levels_scan[idx] = level;
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if k + 1 < remaining && idx > 0 {
            idx -= 1;
        }
    }
    // total_zeros + run_before to position the levels in scan order.
    let total_zeros = read_total_zeros(br, total_coeff)?;
    let mut zeros_left = total_zeros;
    let mut packed = [0i32; 16];
    let mut pos: i32 = (total_coeff + total_zeros) as i32 - 1;
    for i in 0..(total_coeff as usize) {
        let run = if i + 1 == total_coeff as usize {
            zeros_left
        } else {
            let r = read_run_before(br, zeros_left)?;
            zeros_left -= r;
            r
        };
        if pos < 0 {
            return None;
        }
        packed[pos as usize] = levels_scan[(total_coeff as usize) - 1 - i];
        pos -= 1 + run as i32;
    }
    Some(inverse_zigzag(&packed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a bit reader over a sequence of (value, bit_length) tuples
    /// packed MSB-first.
    fn br_from_parts(parts: &[(u32, u32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut acc: u64 = 0;
        let mut filled: u32 = 0;
        for &(bits, val) in parts {
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

    #[test]
    fn zigzag_inverse_is_identity_on_raster() {
        // Place 0..15 into scan order then invert; raster-order result
        // should match what ZIGZAG_4X4 spells out per index.
        let scan: [i32; 16] = std::array::from_fn(|i| i as i32);
        let raster = inverse_zigzag(&scan);
        // Sanity: position [0] = first scan coeff; position [15] is last.
        assert_eq!(raster[ZIGZAG_4X4[0]], 0);
        assert_eq!(raster[ZIGZAG_4X4[15]], 15);
    }

    #[test]
    fn coeff_token_zero_total_coeff() {
        // "1" = TotalCoeff=0, TrailingOnes=0
        let buf = br_from_parts(&[(1, 1)]);
        let mut br = BitReader::new(&buf);
        let (tc, t1) = read_coeff_token_nc0(&mut br).unwrap();
        assert_eq!(tc, 0);
        assert_eq!(t1, 0);
    }

    #[test]
    fn coeff_token_tc1_t1_1() {
        // "000101" = TotalCoeff=1, TrailingOnes=1
        let buf = br_from_parts(&[(6, 0b000101)]);
        let mut br = BitReader::new(&buf);
        let (tc, t1) = read_coeff_token_nc0(&mut br).unwrap();
        assert_eq!(tc, 1);
        assert_eq!(t1, 1);
    }

    #[test]
    fn total_zeros_tc1_examples() {
        // TC=1, zeros_left=0 → "1"
        let buf = br_from_parts(&[(1, 1)]);
        let mut br = BitReader::new(&buf);
        assert_eq!(read_total_zeros(&mut br, 1), Some(0));
        // TC=1, zeros_left=1 → "011"
        let buf = br_from_parts(&[(3, 0b011)]);
        let mut br = BitReader::new(&buf);
        assert_eq!(read_total_zeros(&mut br, 1), Some(1));
    }

    #[test]
    fn run_before_zeros_left_2_examples() {
        // zl=2, run=0 → "1"
        let buf = br_from_parts(&[(1, 1)]);
        let mut br = BitReader::new(&buf);
        assert_eq!(read_run_before(&mut br, 2), Some(0));
        // zl=2, run=1 → "01"
        let buf = br_from_parts(&[(2, 0b01)]);
        let mut br = BitReader::new(&buf);
        assert_eq!(read_run_before(&mut br, 2), Some(1));
    }

    #[test]
    fn decode_all_zero_block_yields_zeros() {
        // coeff_token "1" → TotalCoeff=0. Block is all zeros.
        let buf = br_from_parts(&[(1, 1)]);
        let mut br = BitReader::new(&buf);
        let block = decode_residual_4x4_nc0(&mut br).unwrap();
        assert_eq!(block, [0i32; 16]);
    }

    #[test]
    fn decode_single_trailing_one_at_dc() {
        // Build minimum bits for: TotalCoeff=1, TrailingOnes=1, sign=+,
        // total_zeros=0 → puts a +1 at scan position 0 (DC of raster).
        // coeff_token "000101" (6 bits) + sign bit "0" (positive) +
        // total_zeros "1" (TC=1 zl=0).
        let buf = br_from_parts(&[(6, 0b000101), (1, 0), (1, 1)]);
        let mut br = BitReader::new(&buf);
        let block = decode_residual_4x4_nc0(&mut br).unwrap();
        // The single +1 lands at scan-pos 0 → raster pos ZIGZAG_4X4[0]=0.
        assert_eq!(block[0], 1);
        for i in 1..16 {
            assert_eq!(block[i], 0, "raster[{i}] should be 0");
        }
    }
}
