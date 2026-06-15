//! VP8 keyframe (intra-only) decoder — full macroblock reconstruction.
//!
//! This is the real pixel decode for WebP-lossy. It builds on the
//! primitives in [`crate::vp8`] (the boolean entropy decoder, the 4x4
//! inverse DCT, the Walsh-Hadamard transform, the intra predictors and
//! the dequantizer) and adds the parts a header parser leaves out:
//!
//!   * the full default coefficient probability tables for all four
//!     block types (RFC 6386 §13.5 `default_coeff_probs`),
//!   * the dequantization lookup tables (RFC 6386 §14.1
//!     `dc_qlookup` / `ac_qlookup`),
//!   * keyframe per-macroblock mode parsing — the Y mode tree, the
//!     16-subblock B-prediction modes and the chroma mode tree
//!     (RFC 6386 §11, §16, `kf_ymode_*`, `kf_bmode_*`, `kf_uv_mode_*`),
//!   * the per-macroblock coefficient partition decode and the
//!     left/above border bookkeeping the predictors need,
//!   * YUV(I420 / BT.601 limited-range) → RGBA output.
//!
//! Only keyframes (intra frames) are decoded — that is exactly what a
//! still WebP-lossy image is (a single VP8 keyframe). Inter prediction,
//! motion compensation and the in-loop deblocking filter are not part
//! of a still image's reconstruction path; the loop filter only changes
//! reconstructed pixels at block edges and is documented as a follow-up
//! (its absence is a small high-frequency artifact at 16px boundaries,
//! not a blank/black region).
//!
//! References:
//!   - RFC 6386 "VP8 Data Format and Decoding Guide"
//!     <https://www.rfc-editor.org/rfc/rfc6386>
//!   - libvpx reference decoder (the RFC's §20 source attachments)

use crate::png::{ImageError, RgbaImage};
use crate::vp8::{BoolDecoder, idct_4x4, iwht_4x4};

// ---------------------------------------------------------------------------
// Dequantization lookup tables (RFC 6386 §14.1).
// `dc_qlookup` / `ac_qlookup` map a 7-bit quantizer index (0..=127) to the
// multiplier applied to the DC / AC coefficients respectively.
// ---------------------------------------------------------------------------

pub const DC_QLOOKUP: [i32; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23,
    23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
    93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130,
    132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

pub const AC_QLOOKUP: [i32; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140,
    143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205,
    209, 213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

// ---------------------------------------------------------------------------
// Keyframe macroblock prediction modes (RFC 6386 §11.2).
// ---------------------------------------------------------------------------

/// Whole-macroblock luma prediction modes. The fifth (`B_PRED`) signals
/// that the 16 4x4 subblocks are predicted independently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YMode {
    Dc = 0,
    V = 1,
    H = 2,
    Tm = 3,
    B = 4,
}

/// 4x4 subblock prediction modes (RFC 6386 §11.3). All ten are decoded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BMode {
    Dc = 0,
    Tm = 1,
    Ve = 2,
    He = 3,
    Ld = 4,
    Rd = 5,
    Vr = 6,
    Vl = 7,
    Hd = 8,
    Hu = 9,
}

// Mode trees. Each pair of entries is a node: a negative value `-x` is the
// leaf "mode x"; a non-negative value is the index of the next node pair.

// kf_ymode_tree (RFC 6386 §11.2): order is B_PRED, DC_PRED, V_PRED, H_PRED, TM_PRED.
const KF_YMODE_TREE: [i8; 8] = [
    -(YMode::B as i8),
    2,
    4,
    6,
    -(YMode::Dc as i8),
    -(YMode::V as i8),
    -(YMode::H as i8),
    -(YMode::Tm as i8),
];
const KF_YMODE_PROB: [u8; 4] = [145, 156, 163, 128];

// uv_mode_tree (RFC 6386 §11.4): DC, V, H, TM.
const UV_MODE_TREE: [i8; 6] = [
    -(YMode::Dc as i8),
    2,
    -(YMode::V as i8),
    4,
    -(YMode::H as i8),
    -(YMode::Tm as i8),
];
const KF_UV_MODE_PROB: [u8; 3] = [142, 114, 183];

// bmode_tree (RFC 6386 §11.3) over the 10 B modes.
const BMODE_TREE: [i8; 18] = [
    -(BMode::Dc as i8),
    2,
    -(BMode::Tm as i8),
    4,
    -(BMode::Ve as i8),
    6,
    8,
    12,
    -(BMode::He as i8),
    10,
    -(BMode::Rd as i8),
    -(BMode::Vr as i8),
    -(BMode::Ld as i8),
    14,
    -(BMode::Vl as i8),
    16,
    -(BMode::Hd as i8),
    -(BMode::Hu as i8),
];

/// Generic tree walk used by all keyframe mode reads (RFC 6386 §8.2).
fn read_tree(bd: &mut BoolDecoder, tree: &[i8], probs: &[u8], start: usize) -> i8 {
    let mut i = start;
    loop {
        let b = bd.read_bool(u32::from(probs[i >> 1])) as usize;
        let next = tree[i + b];
        if next <= 0 {
            return -next;
        }
        i = next as usize;
    }
}

/// Read a macroblock's segment id from the segment tree probabilities
/// (libvpx decodemv.c::read_mb_features). Returns 0..=3.
fn read_segment_id(bd: &mut BoolDecoder, probs: &[u8; 3]) -> usize {
    if bd.read_bool(u32::from(probs[0])) == 1 {
        2 + bd.read_bool(u32::from(probs[2])) as usize
    } else {
        bd.read_bool(u32::from(probs[1])) as usize
    }
}

// ---------------------------------------------------------------------------
// Per-macroblock state and the decoded segment header we actually need.
// ---------------------------------------------------------------------------

/// Per-block quantization factors after clamping (RFC 6386 §14).
#[derive(Clone, Copy)]
struct Quant {
    y1_dc: i32,
    y1_ac: i32,
    y2_dc: i32,
    y2_ac: i32,
    uv_dc: i32,
    uv_ac: i32,
}

fn clamp_q(idx: i32) -> usize {
    idx.clamp(0, 127) as usize
}

fn build_quant(base_q: i32, deltas: &QDeltas) -> Quant {
    let y1_dc = DC_QLOOKUP[clamp_q(base_q + deltas.y1_dc)];
    let y1_ac = AC_QLOOKUP[clamp_q(base_q)];
    // Y2 (the WHT/second-order block) uses scaled factors per RFC 6386 §14.1.
    let mut y2_dc = DC_QLOOKUP[clamp_q(base_q + deltas.y2_dc)] * 2;
    let mut y2_ac = AC_QLOOKUP[clamp_q(base_q + deltas.y2_ac)] * 155 / 100;
    if y2_ac < 8 {
        y2_ac = 8;
    }
    let uv_dc_raw = DC_QLOOKUP[clamp_q(base_q + deltas.uv_dc)];
    let uv_dc = if uv_dc_raw > 132 { 132 } else { uv_dc_raw };
    let uv_ac = AC_QLOOKUP[clamp_q(base_q + deltas.uv_ac)];
    let _ = &mut y2_dc;
    Quant {
        y1_dc,
        y1_ac,
        y2_dc,
        y2_ac,
        uv_dc,
        uv_ac,
    }
}

/// Quantizer delta fields read from the frame header (RFC 6386 §9.6).
#[derive(Clone, Copy, Default)]
struct QDeltas {
    y1_dc: i32,
    y2_dc: i32,
    y2_ac: i32,
    uv_dc: i32,
    uv_ac: i32,
}

// ---------------------------------------------------------------------------
// Token (coefficient) decode with the FULL per-context probability tables.
// ---------------------------------------------------------------------------

/// Decode one 4x4 block's coefficients into natural (raster) order using the
/// full `[block_type][band][context]` probability tables. Returns the number
/// of the highest non-zero coefficient + 1 (the EOB position), used to set the
/// neighbour "non-zero" context for the next block.
///
/// `first_coeff` is 0 for normal blocks and 1 for Y blocks whose DC came from
/// the Y2 (second-order) block (RFC 6386 §13).
fn decode_coeffs(
    bd: &mut BoolDecoder,
    probs: &CoeffProbs,
    block_type: usize,
    first_coeff: usize,
    ctx: usize,
    dq_dc: i32,
    dq_ac: i32,
    out: &mut [i32; 16],
) -> usize {
    // Transcribed from libvpx `vp8/decoder/detokenize.c::GetCoeffs`. The band
    // for each token is `kBands[n]` where `n` is incremented BEFORE the read;
    // EOB is only re-checked after a non-zero coefficient (never after a zero).
    const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
    // kBands with the +1 sentinel entry (index 16 unused for band selection).
    const KBANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];
    *out = [0; 16];

    let mut n = first_coeff;
    // Initial probability node: prob[block_type][n][ctx].
    let mut p = &probs[block_type][n][ctx];
    // First EOB acts as a coded-block-pattern bit.
    if bd.read_bool(u32::from(p[0])) == 0 {
        return 0;
    }
    loop {
        n += 1;
        if bd.read_bool(u32::from(p[1])) == 0 {
            // Zero coefficient: context 0, no EOB check before the next token.
            p = &probs[block_type][KBANDS[n]][0];
        } else {
            let v: i32 = if bd.read_bool(u32::from(p[2])) == 0 {
                p = &probs[block_type][KBANDS[n]][1];
                1
            } else {
                let val = if bd.read_bool(u32::from(p[3])) == 0 {
                    if bd.read_bool(u32::from(p[4])) == 0 {
                        2
                    } else {
                        3 + bd.read_bool(u32::from(p[5])) as i32
                    }
                } else if bd.read_bool(u32::from(p[6])) == 0 {
                    if bd.read_bool(u32::from(p[7])) == 0 {
                        // cat1: 5 + 1 extra bit (prob 159).
                        5 + bd.read_bool(159) as i32
                    } else {
                        // cat2: 7 + 2 extra bits (probs 165, 145).
                        7 + 2 * bd.read_bool(165) as i32 + bd.read_bool(145) as i32
                    }
                } else {
                    // Categories 3..6 (libvpx kCat3456). `cat` selects the
                    // extra-bit table and the base value 3 + (8 << cat).
                    let bit1 = bd.read_bool(u32::from(p[8])) as usize;
                    let bit0 = bd.read_bool(u32::from(p[9 + bit1])) as usize;
                    let cat = 2 * bit1 + bit0;
                    let tab: &[u8] = match cat {
                        0 => &PCAT3,
                        1 => &PCAT4,
                        2 => &PCAT5,
                        _ => &PCAT6,
                    };
                    let mut acc = 0i32;
                    for &t in tab {
                        acc = acc + acc + bd.read_bool(u32::from(t)) as i32;
                    }
                    acc + 3 + (8 << cat)
                };
                p = &probs[block_type][KBANDS[n]][2];
                val
            };
            let j = ZIGZAG[n - 1];
            let dq = if j == 0 { dq_dc } else { dq_ac };
            out[j] = sign(bd, v) * dq;

            if n == 16 || bd.read_bool(u32::from(p[0])) == 0 {
                return n;
            }
        }
        if n == 16 {
            return 16;
        }
    }
}

/// Apply VP8's uniform-probability sign bit (RFC 6386 §13.2).
#[inline]
fn sign(bd: &mut BoolDecoder, v: i32) -> i32 {
    if bd.read_bool(128) == 1 { -v } else { v }
}

/// Extra-bit "category" probability strings for DCT coefficient categories
/// 3..6 (RFC 6386 §13.2 / libvpx kCat3456). Categories 1 and 2 are inlined.
const PCAT3: [u8; 3] = [173, 148, 140];
const PCAT4: [u8; 4] = [176, 155, 140, 135];
const PCAT5: [u8; 5] = [180, 157, 141, 134, 130];
const PCAT6: [u8; 11] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];

// ---------------------------------------------------------------------------
// Full default coefficient probability table.
// ---------------------------------------------------------------------------

/// `[block_type=4][coeff_band=8][prev_token_ctx=3][token_prob=11]`
/// (RFC 6386 §13.5 `default_coeff_probs`).
type CoeffProbs = [[[[u8; 11]; 3]; 8]; 4];

// The default coefficient probabilities. This is the verbatim
// `default_coeff_probs` table from RFC 6386 §13.5 / libvpx
// `vp8/common/default_coef_probs.h`.
#[rustfmt::skip]
const DEFAULT_COEFF_PROBS: CoeffProbs = [
  [ // block type 0
    [
      [128,128,128,128,128,128,128,128,128,128,128],
      [128,128,128,128,128,128,128,128,128,128,128],
      [128,128,128,128,128,128,128,128,128,128,128],
    ],
    [
      [253,136,254,255,228,219,128,128,128,128,128],
      [189,129,242,255,227,213,255,219,128,128,128],
      [106,126,227,252,214,209,255,255,128,128,128],
    ],
    [
      [  1, 98,248,255,236,226,255,255,128,128,128],
      [181,133,238,254,221,234,255,154,128,128,128],
      [ 78,134,202,247,198,180,255,219,128,128,128],
    ],
    [
      [  1,185,249,255,243,255,128,128,128,128,128],
      [184,150,247,255,236,224,128,128,128,128,128],
      [ 77,110,216,255,236,230,128,128,128,128,128],
    ],
    [
      [  1,101,251,255,241,255,128,128,128,128,128],
      [170,139,241,252,236,209,255,255,128,128,128],
      [ 37,116,196,243,228,255,255,255,128,128,128],
    ],
    [
      [  1,204,254,255,245,255,128,128,128,128,128],
      [207,160,250,255,238,128,128,128,128,128,128],
      [102,103,231,255,211,171,128,128,128,128,128],
    ],
    [
      [  1,152,252,255,240,255,128,128,128,128,128],
      [177,135,243,255,234,225,128,128,128,128,128],
      [ 80,129,211,255,194,224,128,128,128,128,128],
    ],
    [
      [  1,  1,255,128,128,128,128,128,128,128,128],
      [246,  1,255,128,128,128,128,128,128,128,128],
      [255,128,128,128,128,128,128,128,128,128,128],
    ],
  ],
  [ // block type 1
    [
      [198, 35,237,223,193,187,162,160,145,155, 62],
      [131, 45,198,221,172,176,220,157,252,221,  1],
      [ 68, 47,146,208,149,167,221,162,255,223,128],
    ],
    [
      [  1,149,241,255,221,224,255,255,128,128,128],
      [184,141,234,253,222,220,255,199,128,128,128],
      [ 81, 99,181,242,176,190,249,202,255,255,128],
    ],
    [
      [  1,129,232,253,214,197,242,196,255,255,128],
      [ 99,121,210,250,201,198,255,202,128,128,128],
      [ 23, 91,163,242,170,187,247,210,255,255,128],
    ],
    [
      [  1,200,246,255,234,255,128,128,128,128,128],
      [109,178,241,255,231,245,255,255,128,128,128],
      [ 44,130,201,253,205,192,255,255,128,128,128],
    ],
    [
      [  1,132,239,251,219,209,255,165,128,128,128],
      [ 94,136,225,251,218,190,255,255,128,128,128],
      [ 22,100,174,245,186,161,255,199,128,128,128],
    ],
    [
      [  1,182,249,255,232,235,128,128,128,128,128],
      [124,143,241,255,227,234,128,128,128,128,128],
      [ 35, 77,181,251,193,211,255,205,128,128,128],
    ],
    [
      [  1,157,247,255,236,231,255,255,128,128,128],
      [121,141,235,255,225,227,255,255,128,128,128],
      [ 45, 99,188,251,195,217,255,224,128,128,128],
    ],
    [
      [  1,  1,251,255,213,255,128,128,128,128,128],
      [203,  1,248,255,255,128,128,128,128,128,128],
      [137,  1,177,255,224,255,128,128,128,128,128],
    ],
  ],
  [ // block type 2
    [
      [253,  9,248,251,207,208,255,192,128,128,128],
      [175, 13,224,243,193,185,249,198,255,255,128],
      [ 73, 17,171,221,161,179,236,167,255,234,128],
    ],
    [
      [  1, 95,247,253,212,183,255,255,128,128,128],
      [239, 90,244,250,211,209,255,255,128,128,128],
      [155, 77,195,248,188,195,255,255,128,128,128],
    ],
    [
      [  1, 24,239,251,218,219,255,205,128,128,128],
      [201, 51,219,255,196,186,128,128,128,128,128],
      [ 69, 46,190,239,201,218,255,228,128,128,128],
    ],
    [
      [  1,191,251,255,255,128,128,128,128,128,128],
      [223,165,249,255,213,255,128,128,128,128,128],
      [141,124,248,255,255,128,128,128,128,128,128],
    ],
    [
      [  1, 16,248,255,255,128,128,128,128,128,128],
      [190, 36,230,255,236,255,128,128,128,128,128],
      [149,  1,255,128,128,128,128,128,128,128,128],
    ],
    [
      [  1,226,255,128,128,128,128,128,128,128,128],
      [247,192,255,128,128,128,128,128,128,128,128],
      [240,128,255,128,128,128,128,128,128,128,128],
    ],
    [
      [  1,134,252,255,255,128,128,128,128,128,128],
      [213, 62,250,255,255,128,128,128,128,128,128],
      [ 55, 93,255,128,128,128,128,128,128,128,128],
    ],
    [
      [128,128,128,128,128,128,128,128,128,128,128],
      [128,128,128,128,128,128,128,128,128,128,128],
      [128,128,128,128,128,128,128,128,128,128,128],
    ],
  ],
  [ // block type 3
    [
      [202, 24,213,235,186,191,220,160,240,175,255],
      [126, 38,182,232,169,184,228,174,255,187,128],
      [ 61, 46,138,219,151,178,240,170,255,216,128],
    ],
    [
      [  1,112,230,250,199,191,247,159,255,255,128],
      [166,109,228,252,211,215,255,174,128,128,128],
      [ 39, 77,162,232,172,180,245,178,255,255,128],
    ],
    [
      [  1, 52,220,246,198,199,249,220,255,255,128],
      [124, 74,191,243,183,193,250,221,255,255,128],
      [ 24, 71,130,219,154,170,243,182,255,255,128],
    ],
    [
      [  1,182,225,249,219,240,255,224,128,128,128],
      [149,150,226,252,216,205,255,171,128,128,128],
      [ 28,108,170,242,183,194,254,223,255,255,128],
    ],
    [
      [  1, 81,230,252,204,203,255,192,128,128,128],
      [123,102,209,247,188,196,255,233,128,128,128],
      [ 20, 95,153,243,164,173,255,203,128,128,128],
    ],
    [
      [  1,222,248,255,216,213,128,128,128,128,128],
      [168,175,246,252,235,205,255,255,128,128,128],
      [ 47,116,215,255,211,212,255,255,128,128,128],
    ],
    [
      [  1,121,236,253,212,214,255,255,128,128,128],
      [141, 84,213,252,201,202,255,219,128,128,128],
      [ 42, 80,160,240,162,185,255,205,128,128,128],
    ],
    [
      [  1,  1,255,128,128,128,128,128,128,128,128],
      [244,  1,255,128,128,128,128,128,128,128,128],
      [238,  1,255,128,128,128,128,128,128,128,128],
    ],
  ],
];

// ---------------------------------------------------------------------------
// The plane buffer the decoder reconstructs into.
// ---------------------------------------------------------------------------

struct Plane {
    data: Vec<u8>,
    stride: usize,
}

impl Plane {
    fn new(w: usize, h: usize) -> Self {
        Plane {
            data: vec![129u8; w * h],
            stride: w,
        }
    }
    #[inline]
    fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }
    #[inline]
    fn set(&mut self, x: usize, y: usize, v: u8) {
        self.data[y * self.stride + x] = v;
    }
}

// ---------------------------------------------------------------------------
// Frame setup parse (the parts the original parser skipped). Returns the base
// quantizer, the quant deltas, and leaves the boolean decoder positioned at
// the first macroblock's mode data.
// ---------------------------------------------------------------------------

/// Segmentation state (RFC 6386 §9.3, §10). When the segment map is updated,
/// each macroblock reads a `segment_id`; per-segment quantizer adjustments
/// then change that MB's dequant factors.
#[derive(Clone, Copy, Default)]
struct Segmentation {
    enabled: bool,
    update_map: bool,
    /// `true` = the per-segment quantizer values are absolute; `false` = deltas
    /// on the base quantizer.
    abs_values: bool,
    /// Per-segment quantizer (absolute index or delta). 4 segments.
    quant: [i32; 4],
    /// Tree probabilities used to decode each MB's segment id.
    tree_probs: [u8; 3],
}

struct Setup {
    base_q: i32,
    deltas: QDeltas,
    seg: Segmentation,
}

fn parse_setup(bd: &mut BoolDecoder) -> Setup {
    // Color space + clamping (RFC 6386 §9.2).
    let _color_space = bd.read_literal(1);
    let _clamping = bd.read_literal(1);

    // Segmentation (RFC 6386 §9.3, libvpx decodeframe.c).
    let mut seg = Segmentation {
        tree_probs: [255; 3],
        ..Default::default()
    };
    seg.enabled = bd.read_bool(128) != 0;
    if seg.enabled {
        seg.update_map = bd.read_bool(128) != 0;
        let update_data = bd.read_bool(128) != 0;
        if update_data {
            seg.abs_values = bd.read_bool(128) != 0;
            // Per-segment quantizer (4 segments, 7-bit signed magnitude).
            for s in 0..4 {
                if bd.read_bool(128) == 1 {
                    seg.quant[s] = bd.read_signed_literal(7);
                }
            }
            // Per-segment loop-filter level (4 segments, 6-bit signed). Parsed
            // to stay aligned; the loop filter itself is a documented follow-up.
            for _ in 0..4 {
                if bd.read_bool(128) == 1 {
                    let _lf = bd.read_signed_literal(6);
                }
            }
        }
        if seg.update_map {
            for s in 0..3 {
                if bd.read_bool(128) == 1 {
                    seg.tree_probs[s] = bd.read_literal(8) as u8;
                }
            }
        }
    }

    // Loop filter config (RFC 6386 §9.4). Parsed, not applied (no in-loop
    // deblocking yet — a documented follow-up; absence is a small edge artifact).
    let _filter_type = bd.read_bool(128);
    let _filter_level = bd.read_literal(6);
    let _sharpness = bd.read_literal(3);
    let mb_lf = bd.read_bool(128);
    if mb_lf != 0 {
        let update = bd.read_bool(128);
        if update != 0 {
            for _ in 0..4 {
                if bd.read_bool(128) == 1 {
                    let _v = bd.read_signed_literal(6);
                }
            }
            for _ in 0..4 {
                if bd.read_bool(128) == 1 {
                    let _v = bd.read_signed_literal(6);
                }
            }
        }
    }

    // Token partitions (RFC 6386 §9.5). For a single-partition keyframe this
    // is log2(1) = 0; multi-partition reads partition sizes after this field.
    // WebP-lossy still images use a single partition.
    let _log2_parts = bd.read_literal(2);

    // Quantizer indices (RFC 6386 §9.6).
    let base_q = bd.read_literal(7) as i32;
    let mut deltas = QDeltas::default();
    let read_delta = |bd: &mut BoolDecoder| -> i32 {
        if bd.read_bool(128) == 1 {
            bd.read_signed_literal(4)
        } else {
            0
        }
    };
    deltas.y1_dc = read_delta(bd);
    deltas.y2_dc = read_delta(bd);
    deltas.y2_ac = read_delta(bd);
    deltas.uv_dc = read_delta(bd);
    deltas.uv_ac = read_delta(bd);

    // refresh_entropy_probs (keyframe has no golden/altref refresh fields).
    let _refresh_entropy = bd.read_bool(128);

    // Coefficient probability updates (RFC 6386 §9.9). Each of the
    // 4*8*3*11 entries has an update flag governed by `coeff_update_probs`.
    // We must walk these to stay aligned; we apply them onto a working copy.
    // (Most keyframes leave them at defaults; honoring the updates is what
    // makes arbitrary encoders' files decode exactly.)
    // The walk happens in the caller because it owns the probability table.

    Setup { base_q, deltas, seg }
}

/// Walk the coefficient-probability update section (RFC 6386 §9.9), mutating
/// `probs` in place. Must be called immediately after [`parse_setup`].
fn parse_coeff_prob_updates(bd: &mut BoolDecoder, probs: &mut CoeffProbs) {
    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for l in 0..11 {
                    if bd.read_bool(u32::from(COEFF_UPDATE_PROBS[i][j][k][l])) == 1 {
                        probs[i][j][k][l] = bd.read_literal(8) as u8;
                    }
                }
            }
        }
    }
}

/// mb_no_coeff_skip flag + the per-MB skip probability (RFC 6386 §9.10/§9.11).
fn parse_skip_setup(bd: &mut BoolDecoder) -> Option<u8> {
    let mb_no_skip = bd.read_bool(128);
    if mb_no_skip != 0 {
        Some(bd.read_literal(8) as u8)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Top-level keyframe decode.
// ---------------------------------------------------------------------------

/// Decode a VP8 keyframe (the body following the WebP `VP8 ` chunk header,
/// or a raw VP8 frame) into an RGBA image. Returns `Malformed`/`Truncated`
/// on a non-keyframe or corrupt input — never a blank/gradient placeholder.
pub fn decode_keyframe(input: &[u8]) -> Result<RgbaImage, ImageError> {
    let hdr = crate::vp8::parse_frame_header(input)?;
    if !hdr.key_frame {
        return Err(ImageError::Malformed("VP8: not a key frame"));
    }
    let w = hdr.width as usize;
    let h = hdr.height as usize;
    if w == 0 || h == 0 {
        return Err(ImageError::Malformed("VP8: zero dimension"));
    }
    let part_start = hdr.first_partition_off;
    let part_end = (part_start + hdr.first_part_size as usize).min(input.len());
    if part_end <= part_start {
        return Err(ImageError::Truncated);
    }

    // The first partition holds the frame header tail + the per-MB mode
    // data. For single-partition keyframes the coefficient tokens follow in
    // the same boolean stream after the mode data (residuals partition starts
    // at part_end). VP8 actually uses a SECOND boolean decoder for the
    // coefficient/residual data beginning at `part_end`.
    let mut hbd = BoolDecoder::new(&input[part_start..part_end]);
    let setup = parse_setup(&mut hbd);

    // Working copy of the coefficient probabilities, possibly updated.
    let mut probs: CoeffProbs = DEFAULT_COEFF_PROBS;
    parse_coeff_prob_updates(&mut hbd, &mut probs);
    let skip_prob = parse_skip_setup(&mut hbd);

    // Per-segment dequantizers (RFC 6386 §10). Segment 0's quant is the base
    // when segmentation is disabled. When enabled, each segment's base index is
    // either an absolute value or a delta on `base_q`.
    let segment_quant: [Quant; 4] = std::array::from_fn(|s| {
        let seg_base = if setup.seg.enabled {
            if setup.seg.abs_values {
                setup.seg.quant[s]
            } else {
                setup.base_q + setup.seg.quant[s]
            }
        } else {
            setup.base_q
        };
        build_quant(seg_base, &setup.deltas)
    });

    // Residual (token) partition: a separate boolean decoder over the bytes
    // after the first partition (RFC 6386 §9.5 — single partition case).
    if part_end >= input.len() {
        return Err(ImageError::Truncated);
    }
    let mut tbd = BoolDecoder::new(&input[part_end..]);

    // Macroblock grid (round up to 16).
    let mb_w = w.div_ceil(16);
    let mb_h = h.div_ceil(16);

    // Reconstruction planes, padded to MB granularity (+1 col/row of border).
    let yw = mb_w * 16;
    let yh = mb_h * 16;
    let cw = mb_w * 8;
    let ch = mb_h * 8;
    let mut yp = Plane::new(yw, yh);
    let mut up = Plane::new(cw, ch);
    let mut vp = Plane::new(cw, ch);

    // Above-context for the B (4x4) keyframe mode prediction: one mode per
    // 4x4 column across the whole image width, plus a per-MB-row left state.
    let mut above_bmode = vec![BMode::Dc; mb_w * 4];
    // Above non-zero context per 4x4 column for Y/U/V + the Y2 above flag.
    let mut above_nz_y = vec![0u8; mb_w * 4];
    let mut above_nz_u = vec![0u8; mb_w * 2];
    let mut above_nz_v = vec![0u8; mb_w * 2];
    let mut above_nz_y2 = vec![0u8; mb_w];

    let mut coeffs = [0i32; 16];
    let mut residual = [0i32; 16];

    for mby in 0..mb_h {
        let mut left_bmode = [BMode::Dc; 4];
        let mut left_nz_y = [0u8; 4];
        let mut left_nz_u = [0u8; 2];
        let mut left_nz_v = [0u8; 2];
        let mut left_nz_y2 = 0u8;

        for mbx in 0..mb_w {
            // ---- per-MB read order (libvpx decodemv.c::read_mb_modes_mv) ----
            // 1) segment_id (only if the segmentation map is being updated),
            // 2) mb_skip_coeff, 3) prediction modes.
            let segment_id = if setup.seg.enabled && setup.seg.update_map {
                read_segment_id(&mut hbd, &setup.seg.tree_probs)
            } else {
                0
            };
            let quant = &segment_quant[segment_id];

            let mb_skip = match skip_prob {
                Some(p) => hbd.read_bool(u32::from(p)) == 1,
                None => false,
            };

            // ---- macroblock mode parse (from header partition) ----
            let ymode_leaf = read_tree(&mut hbd, &KF_YMODE_TREE, &KF_YMODE_PROB, 0);
            let ymode = match ymode_leaf {
                0 => YMode::Dc,
                1 => YMode::V,
                2 => YMode::H,
                3 => YMode::Tm,
                _ => YMode::B,
            };

            // Per-subblock B modes (only when ymode == B). When the whole MB
            // uses a 16x16 mode the subblock context is derived from it so the
            // neighbours of a B-coded MB still have a sensible above/left.
            let mut bmodes = [BMode::Dc; 16];
            if ymode == YMode::B {
                for sb in 0..16 {
                    let row = sb / 4;
                    let col = sb % 4;
                    let a = if row == 0 {
                        above_bmode[mbx * 4 + col]
                    } else {
                        bmodes[sb - 4]
                    };
                    let l = if col == 0 {
                        left_bmode[row]
                    } else {
                        bmodes[sb - 1]
                    };
                    let probs_b = &KF_BMODE_PROB[a as usize][l as usize];
                    let leaf = read_tree(&mut hbd, &BMODE_TREE, probs_b, 0);
                    bmodes[sb] = bmode_from(leaf);
                }
            } else {
                let derived = derived_bmode(ymode);
                bmodes = [derived; 16];
            }
            // Update above/left B-mode context for the next MB.
            for c in 0..4 {
                above_bmode[mbx * 4 + c] = bmodes[12 + c];
            }
            for r in 0..4 {
                left_bmode[r] = bmodes[r * 4 + 3];
            }

            let uvmode_leaf = read_tree(&mut hbd, &UV_MODE_TREE, &KF_UV_MODE_PROB, 0);
            let uvmode = match uvmode_leaf {
                0 => YMode::Dc,
                1 => YMode::V,
                2 => YMode::H,
                _ => YMode::Tm,
            };

            // ---- coefficient tokens (residual partition) ----
            // RFC 6386 §13: a non-B macroblock carries a Y2 (second-order
            // WHT) block; its 16 Y blocks then omit their DC (first_coeff=1).
            let has_y2 = ymode != YMode::B;
            let mut y2 = [0i32; 16];
            let mut y_blocks = [[0i32; 16]; 16];
            let mut u_blocks = [[0i32; 16]; 4];
            let mut v_blocks = [[0i32; 16]; 4];

            // Per-4x4 "had a non-zero coded coefficient" flags. These drive
            // the entropy context of neighbouring blocks (above/left); they
            // are tracked explicitly rather than re-derived from the
            // coefficient arrays because the DC slot is later overwritten by
            // the Y2 inverse transform.
            let mut nz_y = [0u8; 16];
            let mut nz_u = [0u8; 4];
            let mut nz_v = [0u8; 4];

            if !mb_skip {
                // Y2 block (block type 1) when present.
                if has_y2 {
                    let ctx = (above_nz_y2[mbx] + left_nz_y2) as usize;
                    let nz = decode_coeffs(
                        &mut tbd, &probs, 1, 0, ctx, quant.y2_dc, quant.y2_ac, &mut coeffs,
                    );
                    y2 = coeffs;
                    let flag = (nz > 0) as u8;
                    above_nz_y2[mbx] = flag;
                    left_nz_y2 = flag;
                }

                // 16 Y blocks. Block type 0 if Y2 present (the type-0 table is
                // "luma after Y2"), block type 3 ("luma without Y2") when B.
                let (y_type, first) = if has_y2 { (0usize, 1usize) } else { (3usize, 0usize) };
                for sb in 0..16 {
                    let row = sb / 4;
                    let col = sb % 4;
                    let a = if row == 0 {
                        above_nz_y[mbx * 4 + col]
                    } else {
                        nz_y[sb - 4]
                    };
                    let l = if col == 0 {
                        left_nz_y[row]
                    } else {
                        nz_y[sb - 1]
                    };
                    let nz = decode_coeffs(
                        &mut tbd, &probs, y_type, first, (a + l) as usize, quant.y1_dc,
                        quant.y1_ac, &mut coeffs,
                    );
                    y_blocks[sb] = coeffs;
                    // Entropy context: was there a coded coefficient? (libvpx
                    // sets `*a = *l = (nonzeros > 0)` where the return excludes
                    // the implicit DC for Y-after-Y2.)
                    nz_y[sb] = (nz > 0) as u8;
                }
                for col in 0..4 {
                    above_nz_y[mbx * 4 + col] = nz_y[12 + col];
                }
                for row in 0..4 {
                    left_nz_y[row] = nz_y[row * 4 + 3];
                }

                // Chroma: 4 U blocks then 4 V blocks (block type 2).
                decode_chroma(
                    &mut tbd, &probs, quant, mbx, &mut u_blocks, &mut nz_u, &mut above_nz_u,
                    &mut left_nz_u,
                );
                decode_chroma(
                    &mut tbd, &probs, quant, mbx, &mut v_blocks, &mut nz_v, &mut above_nz_v,
                    &mut left_nz_v,
                );
            } else {
                // Skipped MB: no coded coefficients → clear the nz context.
                // The Y2 context is preserved when the MB has no Y2 block
                // (RFC 6386 §13: a B-pred MB does not touch the Y2 context).
                for col in 0..4 {
                    above_nz_y[mbx * 4 + col] = 0;
                }
                for row in 0..4 {
                    left_nz_y[row] = 0;
                }
                for i in 0..2 {
                    above_nz_u[mbx * 2 + i] = 0;
                    above_nz_v[mbx * 2 + i] = 0;
                    left_nz_u[i] = 0;
                    left_nz_v[i] = 0;
                }
                if has_y2 {
                    above_nz_y2[mbx] = 0;
                    left_nz_y2 = 0;
                }
            }

            // ---- if Y2 present, invert WHT and distribute DC into Y blocks ----
            if has_y2 {
                let mut dcs = [0i32; 16];
                iwht_4x4(&y2, &mut dcs);
                for sb in 0..16 {
                    y_blocks[sb][0] = dcs[sb];
                }
            }

            // ---- reconstruct luma ----
            reconstruct_luma(
                &mut yp, mbx, mby, ymode, &bmodes, &y_blocks, &mut coeffs, &mut residual,
            );
            // ---- reconstruct chroma ----
            reconstruct_chroma(&mut up, mbx, mby, uvmode, &u_blocks, &mut residual);
            reconstruct_chroma(&mut vp, mbx, mby, uvmode, &v_blocks, &mut residual);
        }
    }

    // YUV(I420, BT.601 limited range) → RGBA, cropping to the real W×H.
    let pixels = yuv_planes_to_rgba(&yp, &up, &vp, w, h);
    Ok(RgbaImage {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Decode 4 chroma blocks (one plane's worth) into `blocks` (block type 2),
/// recording per-block non-zero flags in `nz` and updating the above/left
/// entropy-context arrays.
#[allow(clippy::too_many_arguments)]
fn decode_chroma(
    tbd: &mut BoolDecoder,
    probs: &CoeffProbs,
    quant: &Quant,
    mbx: usize,
    blocks: &mut [[i32; 16]; 4],
    nz: &mut [u8; 4],
    above_nz: &mut [u8],
    left_nz: &mut [u8; 2],
) {
    let mut coeffs = [0i32; 16];
    for sb in 0..4 {
        let row = sb / 2;
        let col = sb % 2;
        let a = if row == 0 {
            above_nz[mbx * 2 + col]
        } else {
            nz[sb - 2]
        };
        let l = if col == 0 { left_nz[row] } else { nz[sb - 1] };
        let count = decode_coeffs(
            tbd, probs, 2, 0, (a + l) as usize, quant.uv_dc, quant.uv_ac, &mut coeffs,
        );
        blocks[sb] = coeffs;
        nz[sb] = (count > 0) as u8;
    }
    for col in 0..2 {
        above_nz[mbx * 2 + col] = nz[2 + col];
    }
    for row in 0..2 {
        left_nz[row] = nz[row * 2 + 1];
    }
}

fn bmode_from(leaf: i8) -> BMode {
    match leaf {
        0 => BMode::Dc,
        1 => BMode::Tm,
        2 => BMode::Ve,
        3 => BMode::He,
        4 => BMode::Ld,
        5 => BMode::Rd,
        6 => BMode::Vr,
        7 => BMode::Vl,
        8 => BMode::Hd,
        _ => BMode::Hu,
    }
}

/// When a macroblock uses a 16x16 luma mode, its subblocks expose an
/// equivalent 4x4 mode to neighbouring B-coded MBs (RFC 6386 §11.2).
fn derived_bmode(ymode: YMode) -> BMode {
    match ymode {
        YMode::Dc => BMode::Dc,
        YMode::V => BMode::Ve,
        YMode::H => BMode::He,
        YMode::Tm => BMode::Tm,
        YMode::B => BMode::Dc,
    }
}

// ---------------------------------------------------------------------------
// Reconstruction.
// ---------------------------------------------------------------------------

/// Reconstruct a macroblock's 16x16 luma region into `yp`.
#[allow(clippy::too_many_arguments)]
fn reconstruct_luma(
    yp: &mut Plane,
    mbx: usize,
    mby: usize,
    ymode: YMode,
    bmodes: &[BMode; 16],
    y_blocks: &[[i32; 16]; 16],
    _scratch: &mut [i32; 16],
    residual: &mut [i32; 16],
) {
    let ox = mbx * 16;
    let oy = mby * 16;
    if ymode == YMode::B {
        // Per-4x4-subblock prediction + residual.
        for sb in 0..16 {
            let sx = ox + (sb % 4) * 4;
            let sy = oy + (sb / 4) * 4;
            predict_subblock_4x4(yp, sx, sy, bmodes[sb], mbx, mby, sb);
            idct_4x4(&y_blocks[sb], residual);
            add_residual(yp, sx, sy, residual);
        }
    } else {
        // 16x16 whole-MB prediction, then per-4x4 residual.
        predict_block_16x16(yp, ox, oy, 16, ymode, mbx, mby);
        for sb in 0..16 {
            let sx = ox + (sb % 4) * 4;
            let sy = oy + (sb / 4) * 4;
            idct_4x4(&y_blocks[sb], residual);
            add_residual(yp, sx, sy, residual);
        }
    }
}

/// Reconstruct one chroma plane's 8x8 region.
fn reconstruct_chroma(
    cp: &mut Plane,
    mbx: usize,
    mby: usize,
    uvmode: YMode,
    blocks: &[[i32; 16]; 4],
    residual: &mut [i32; 16],
) {
    let ox = mbx * 8;
    let oy = mby * 8;
    predict_block_16x16(cp, ox, oy, 8, uvmode, mbx, mby);
    for sb in 0..4 {
        let sx = ox + (sb % 2) * 4;
        let sy = oy + (sb / 2) * 4;
        idct_4x4(&blocks[sb], residual);
        add_residual(cp, sx, sy, residual);
    }
}

#[inline]
fn add_residual(plane: &mut Plane, x: usize, y: usize, residual: &[i32; 16]) {
    for r in 0..4 {
        for c in 0..4 {
            let v = i32::from(plane.at(x + c, y + r)) + residual[r * 4 + c];
            plane.set(x + c, y + r, v.clamp(0, 255) as u8);
        }
    }
}

/// 16x16 (luma) or 8x8 (chroma) whole-block intra prediction. `size` is 16 or 8.
fn predict_block_16x16(
    plane: &mut Plane,
    ox: usize,
    oy: usize,
    size: usize,
    mode: YMode,
    mbx: usize,
    mby: usize,
) {
    let have_above = mby > 0;
    let have_left = mbx > 0;
    match mode {
        YMode::Dc => {
            let mut sum = 0i32;
            let mut count = 0i32;
            if have_above {
                for c in 0..size {
                    sum += i32::from(plane.at(ox + c, oy - 1));
                }
                count += size as i32;
            }
            if have_left {
                for r in 0..size {
                    sum += i32::from(plane.at(ox - 1, oy + r));
                }
                count += size as i32;
            }
            let dc = if count == 0 {
                128
            } else {
                let shift = count.trailing_zeros();
                ((sum + (count >> 1)) >> shift) as i32
            };
            let dc = dc.clamp(0, 255) as u8;
            for r in 0..size {
                for c in 0..size {
                    plane.set(ox + c, oy + r, dc);
                }
            }
        }
        YMode::V => {
            for c in 0..size {
                let v = if have_above {
                    plane.at(ox + c, oy - 1)
                } else {
                    127
                };
                for r in 0..size {
                    plane.set(ox + c, oy + r, v);
                }
            }
        }
        YMode::H => {
            for r in 0..size {
                let v = if have_left {
                    plane.at(ox - 1, oy + r)
                } else {
                    129
                };
                for c in 0..size {
                    plane.set(ox + c, oy + r, v);
                }
            }
        }
        YMode::Tm | YMode::B => {
            // TrueMotion: pred = clamp(L[r] + A[c] - corner).
            let corner = if have_above && have_left {
                i32::from(plane.at(ox - 1, oy - 1))
            } else if have_above {
                127
            } else {
                129
            };
            for r in 0..size {
                let l = if have_left {
                    i32::from(plane.at(ox - 1, oy + r))
                } else {
                    129
                };
                for c in 0..size {
                    let a = if have_above {
                        i32::from(plane.at(ox + c, oy - 1))
                    } else {
                        127
                    };
                    let v = (l + a - corner).clamp(0, 255) as u8;
                    plane.set(ox + c, oy + r, v);
                }
            }
        }
    }
}

/// 4x4 subblock intra prediction (RFC 6386 section 12.3). The ten directional
/// predictors are transcribed verbatim from the libvpx reference C
/// (vpx_dsp/intrapred.c: vpx_d45/d135/d117/d153/d63/d207_predictor_4x4_c,
/// which are the VP8 B_LD/B_RD/B_VR/B_HD/B_VL/B_HU modes). Edge naming
/// follows the reference: X = top-left corner (above[-1]); A..H = above[0..7]
/// (above row + above-right extension); I..L = left[0..3]. Unavailable
/// neighbours use the spec border fill (127 above, 129 left).
fn predict_subblock_4x4(
    plane: &mut Plane,
    x: usize,
    y: usize,
    mode: BMode,
    mbx: usize,
    mby: usize,
    sb: usize,
) {
    let row = sb / 4;
    let col = sb % 4;
    let have_above = mby > 0 || row > 0;
    let have_left = mbx > 0 || col > 0;

    let getp = |plane: &Plane, px: usize, py: usize| -> i32 { i32::from(plane.at(px, py)) };

    // above[0..3] directly above; above[4..7] above-right.
    let mut a = [127i32; 8];
    let mut l = [129i32; 4];

    if have_above {
        for i in 0..4 {
            a[i] = getp(plane, x + i, y - 1);
        }
        // Above-right extension (RFC 6386 section 12.3). It is genuinely
        // available only for subblocks in the top row of a macroblock (the
        // macroblock to the above-right is already reconstructed) and away
        // from the image right edge. Otherwise the reference replicates A[3].
        let ar_x = x + 4;
        let ar_ok = row == 0 && (ar_x + 3) < plane.stride;
        for i in 0..4 {
            a[4 + i] = if ar_ok {
                getp(plane, ar_x + i, y - 1)
            } else {
                a[3]
            };
        }
    }
    if have_left {
        for i in 0..4 {
            l[i] = getp(plane, x - 1, y + i);
        }
    }
    let x_corner = if have_above && have_left {
        getp(plane, x - 1, y - 1)
    } else if have_above {
        127
    } else {
        129
    };

    // libvpx edge names.
    let (xx, aa, bb, cc, dd, ee, ff, gg, hh) =
        (x_corner, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    let (ii, jj, kk, ll) = (l[0], l[1], l[2], l[3]);

    let avg2 = |p: i32, q: i32| -> u8 { ((p + q + 1) >> 1) as u8 };
    let avg3 = |p: i32, q: i32, r: i32| -> u8 { ((p + 2 * q + r + 2) >> 2) as u8 };

    let mut out = [0u8; 16];
    // DST(c, r) maps to out[r*4 + c].
    macro_rules! dst {
        ($c:expr, $r:expr) => {
            out[($r) * 4 + ($c)]
        };
    }

    match mode {
        BMode::Dc => {
            let dc = ((aa + bb + cc + dd + ii + jj + kk + ll + 4) >> 3) as u8;
            out = [dc; 16];
        }
        BMode::Tm => {
            let la = [ii, jj, kk, ll];
            let ar = [aa, bb, cc, dd];
            for r in 0..4 {
                for c in 0..4 {
                    dst!(c, r) = (la[r] + ar[c] - xx).clamp(0, 255) as u8;
                }
            }
        }
        BMode::Ve => {
            // B_VE_PRED: vertical with 3-tap smoothing (RFC 6386 section 12.3).
            let e = [avg3(xx, aa, bb), avg3(aa, bb, cc), avg3(bb, cc, dd), avg3(cc, dd, ee)];
            for r in 0..4 {
                for c in 0..4 {
                    dst!(c, r) = e[c];
                }
            }
        }
        BMode::He => {
            let e = [avg3(xx, ii, jj), avg3(ii, jj, kk), avg3(jj, kk, ll), avg3(kk, ll, ll)];
            for r in 0..4 {
                for c in 0..4 {
                    dst!(c, r) = e[r];
                }
            }
        }
        BMode::Ld => {
            // vpx_d45_predictor_4x4_c
            dst!(0, 0) = avg3(aa, bb, cc);
            let v = avg3(bb, cc, dd);
            dst!(1, 0) = v;
            dst!(0, 1) = v;
            let v = avg3(cc, dd, ee);
            dst!(2, 0) = v;
            dst!(1, 1) = v;
            dst!(0, 2) = v;
            let v = avg3(dd, ee, ff);
            dst!(3, 0) = v;
            dst!(2, 1) = v;
            dst!(1, 2) = v;
            dst!(0, 3) = v;
            let v = avg3(ee, ff, gg);
            dst!(3, 1) = v;
            dst!(2, 2) = v;
            dst!(1, 3) = v;
            let v = avg3(ff, gg, hh);
            dst!(3, 2) = v;
            dst!(2, 3) = v;
            dst!(3, 3) = hh as u8;
        }
        BMode::Rd => {
            // vpx_d135_predictor_4x4_c
            dst!(0, 3) = avg3(jj, kk, ll);
            let v = avg3(ii, jj, kk);
            dst!(1, 3) = v;
            dst!(0, 2) = v;
            let v = avg3(xx, ii, jj);
            dst!(2, 3) = v;
            dst!(1, 2) = v;
            dst!(0, 1) = v;
            let v = avg3(aa, xx, ii);
            dst!(3, 3) = v;
            dst!(2, 2) = v;
            dst!(1, 1) = v;
            dst!(0, 0) = v;
            let v = avg3(bb, aa, xx);
            dst!(3, 2) = v;
            dst!(2, 1) = v;
            dst!(1, 0) = v;
            let v = avg3(cc, bb, aa);
            dst!(3, 1) = v;
            dst!(2, 0) = v;
            dst!(3, 0) = avg3(dd, cc, bb);
        }
        BMode::Vr => {
            // vpx_d117_predictor_4x4_c
            let v = avg2(xx, aa);
            dst!(0, 0) = v;
            dst!(1, 2) = v;
            let v = avg2(aa, bb);
            dst!(1, 0) = v;
            dst!(2, 2) = v;
            let v = avg2(bb, cc);
            dst!(2, 0) = v;
            dst!(3, 2) = v;
            dst!(3, 0) = avg2(cc, dd);
            dst!(0, 3) = avg3(kk, jj, ii);
            dst!(0, 2) = avg3(jj, ii, xx);
            let v = avg3(ii, xx, aa);
            dst!(0, 1) = v;
            dst!(1, 3) = v;
            let v = avg3(xx, aa, bb);
            dst!(1, 1) = v;
            dst!(2, 3) = v;
            let v = avg3(aa, bb, cc);
            dst!(2, 1) = v;
            dst!(3, 3) = v;
            dst!(3, 1) = avg3(bb, cc, dd);
        }
        BMode::Vl => {
            // vpx_d63_predictor_4x4_c
            dst!(0, 0) = avg2(aa, bb);
            let v = avg2(bb, cc);
            dst!(1, 0) = v;
            dst!(0, 2) = v;
            let v = avg2(cc, dd);
            dst!(2, 0) = v;
            dst!(1, 2) = v;
            let v = avg2(dd, ee);
            dst!(3, 0) = v;
            dst!(2, 2) = v;
            dst!(3, 2) = avg2(ee, ff);
            dst!(0, 1) = avg3(aa, bb, cc);
            let v = avg3(bb, cc, dd);
            dst!(1, 1) = v;
            dst!(0, 3) = v;
            let v = avg3(cc, dd, ee);
            dst!(2, 1) = v;
            dst!(1, 3) = v;
            let v = avg3(dd, ee, ff);
            dst!(3, 1) = v;
            dst!(2, 3) = v;
            dst!(3, 3) = avg3(ee, ff, gg);
        }
        BMode::Hd => {
            // vpx_d153_predictor_4x4_c
            let v = avg2(ii, xx);
            dst!(0, 0) = v;
            dst!(2, 1) = v;
            let v = avg2(jj, ii);
            dst!(0, 1) = v;
            dst!(2, 2) = v;
            let v = avg2(kk, jj);
            dst!(0, 2) = v;
            dst!(2, 3) = v;
            dst!(0, 3) = avg2(ll, kk);
            dst!(3, 0) = avg3(aa, bb, cc);
            dst!(2, 0) = avg3(xx, aa, bb);
            let v = avg3(ii, xx, aa);
            dst!(1, 0) = v;
            dst!(3, 1) = v;
            let v = avg3(jj, ii, xx);
            dst!(1, 1) = v;
            dst!(3, 2) = v;
            let v = avg3(kk, jj, ii);
            dst!(1, 2) = v;
            dst!(3, 3) = v;
            dst!(1, 3) = avg3(ll, kk, jj);
        }
        BMode::Hu => {
            // vpx_d207_predictor_4x4_c
            dst!(0, 0) = avg2(ii, jj);
            let v = avg2(jj, kk);
            dst!(2, 0) = v;
            dst!(0, 1) = v;
            let v = avg2(kk, ll);
            dst!(2, 1) = v;
            dst!(0, 2) = v;
            dst!(1, 0) = avg3(ii, jj, kk);
            let v = avg3(jj, kk, ll);
            dst!(3, 0) = v;
            dst!(1, 1) = v;
            let v = avg3(kk, ll, ll);
            dst!(3, 1) = v;
            dst!(1, 2) = v;
            let lu = ll as u8;
            dst!(3, 2) = lu;
            dst!(2, 2) = lu;
            dst!(0, 3) = lu;
            dst!(1, 3) = lu;
            dst!(2, 3) = lu;
            dst!(3, 3) = lu;
        }
    }
    for r in 0..4 {
        for c in 0..4 {
            plane.set(x + c, y + r, out[r * 4 + c]);
        }
    }
}

/// Convert reconstructed I420 planes (BT.601 limited range) to RGBA, cropping
/// the padded planes to the real `w`×`h`.
fn yuv_planes_to_rgba(yp: &Plane, up: &Plane, vp: &Plane, w: usize, h: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let yi = i32::from(yp.at(x, y));
            let ui = i32::from(up.at(x / 2, y / 2));
            let vi = i32::from(vp.at(x / 2, y / 2));
            let c = yi - 16;
            let d = ui - 128;
            let e = vi - 128;
            let r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u32;
            let g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u32;
            let b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u32;
            out.push(0xFF00_0000 | (r << 16) | (g << 8) | b);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Keyframe B-mode context probabilities and the coeff-update probabilities.
// These large tables live at the bottom to keep the decode logic readable.
// ---------------------------------------------------------------------------

include!("vp8_tables.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_qlookup_endpoints() {
        assert_eq!(DC_QLOOKUP[0], 4);
        assert_eq!(DC_QLOOKUP[127], 157);
    }

    #[test]
    fn ac_qlookup_endpoints() {
        assert_eq!(AC_QLOOKUP[0], 4);
        assert_eq!(AC_QLOOKUP[127], 284);
    }

    #[test]
    fn ymode_tree_decodes_dc_for_all_zero_probs() {
        // With a deterministic stream we just confirm the tree walk
        // terminates at a valid leaf in 0..=4.
        let buf = vec![0u8; 64];
        let mut bd = BoolDecoder::new(&buf);
        let leaf = read_tree(&mut bd, &KF_YMODE_TREE, &KF_YMODE_PROB, 0);
        assert!((0..=4).contains(&leaf));
    }

    #[test]
    fn coeff_probs_table_shape() {
        // Spot-check a couple of known default entries.
        assert_eq!(DEFAULT_COEFF_PROBS[0][1][0][0], 253);
        assert_eq!(DEFAULT_COEFF_PROBS[0][1][0][1], 136);
        // Block type 3 (luma without Y2) must be populated too — its absence
        // was a real desync hazard for B_PRED macroblocks.
        assert_eq!(DEFAULT_COEFF_PROBS[3][0][0][0], 202);
    }

    #[test]
    fn segment_id_tree_reads_in_range() {
        let buf = vec![0xFFu8; 16];
        let mut bd = BoolDecoder::new(&buf);
        let id = read_segment_id(&mut bd, &[128, 128, 128]);
        assert!(id < 4);
    }

    /// End-to-end: decode the embedded lossy (VP8) WebP keyframe and assert
    /// the top-left quadrant is genuinely red — i.e. real reconstructed
    /// pixels, not a flat placeholder. The fixture is a libwebp-encoded
    /// 64x64 image (red / green / blue / white quadrants).
    #[test]
    fn decodes_lossy_webp_keyframe_to_real_pixels() {
        const WEBP: &[u8] = include_bytes!("../tests/test_lossy_quad.webp");
        let img = crate::webp::decode_webp(WEBP).expect("decode lossy webp");
        assert_eq!(img.width, 64);
        assert_eq!(img.height, 64);
        // Not flat (a placeholder would be a single repeated value).
        let p0 = img.pixels[0];
        assert!(img.pixels.iter().any(|&p| p != p0));
        // Top-left interior pixel is red.
        let p = img.pixels[(16 * 64 + 16) as usize];
        let (r, g, b) = ((p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF);
        assert!(r > 180 && r > g + 80 && r > b + 80, "TL not red: ({r},{g},{b})");
    }
}
