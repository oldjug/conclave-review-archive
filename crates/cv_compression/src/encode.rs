//! DEFLATE compression per RFC 1951, plus gzip (RFC 1952) and zlib
//! (RFC 1950) framing.
//!
//! This is the encoder counterpart to `deflate::inflate` /
//! `gzip::decode_gzip` / `gzip::decode_zlib`. It is the codec backing the
//! `CompressionStream` JS API (Chrome blink `modules/compression`), where
//! `"deflate-raw"` = raw DEFLATE (RFC 1951), `"deflate"` = zlib wrapper
//! (RFC 1950), and `"gzip"` = gzip wrapper (RFC 1952).
//!
//! Strategy: LZ77 back-reference matching against a 32 KiB sliding window
//! with a hash-chain match finder, emitted with the **fixed** Huffman code
//! table (RFC 1951 §3.2.6). Fixed Huffman keeps the encoder small and
//! correct; the output is always a valid DEFLATE stream that any compliant
//! inflater (including ours and zlib/Chrome) decodes back to the input.
//!
//! Correctness is what matters for the web API contract: the bytes must be
//! a well-framed, round-trippable stream — not maximal ratio.

/// Maximum LZ77 back-reference distance (RFC 1951 §3.2.5: 32 768).
const WINDOW_SIZE: usize = 32_768;
/// Minimum match length the encoder will emit as a back-reference. Matches
/// shorter than 3 cost more than literals (RFC 1951 length codes start at 3).
const MIN_MATCH: usize = 3;
/// Maximum match length (RFC 1951 §3.2.5: length code 285 = 258).
const MAX_MATCH: usize = 258;
/// Number of hash-chain head buckets. 15 bits keeps collisions low without a
/// huge table.
const HASH_BITS: usize = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;
/// Cap on hash-chain walk length per position — bounds worst-case time on
/// highly repetitive input while still finding good matches.
const MAX_CHAIN: usize = 256;

/// Bit writer: emits bits LSB-first within each byte, as DEFLATE expects
/// (RFC 1951 §3.1.1).
struct BitWriter {
    out: Vec<u8>,
    bit_buf: u32,
    bit_count: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Write the low `n` bits of `value`, LSB first.
    fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 24);
        self.bit_buf |= (value & ((1u32 << n) - 1)) << self.bit_count;
        self.bit_count += n;
        while self.bit_count >= 8 {
            self.out.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
    }

    /// Write a Huffman code: codes are transmitted MSB-first (RFC 1951
    /// §3.1.1), so reverse the `len`-bit code before emitting LSB-first.
    fn write_code(&mut self, code: u32, len: u32) {
        let mut reversed = 0u32;
        for i in 0..len {
            reversed |= ((code >> i) & 1) << (len - 1 - i);
        }
        self.write_bits(reversed, len);
    }

    /// Flush any residual bits, padding the final partial byte with zeros.
    fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.out.push((self.bit_buf & 0xFF) as u8);
        }
        self.out
    }
}

// --- Fixed Huffman code tables (RFC 1951 §3.2.6) -----------------------------

/// Length/literal symbol (0..=287) -> (code, bit-length) under the fixed tree.
fn fixed_litlen_code(sym: u32) -> (u32, u32) {
    match sym {
        0..=143 => (0b0011_0000 + sym, 8),         // 00110000..10111111
        144..=255 => (0b1_1001_0000 + (sym - 144), 9), // 110010000..111111111
        256..=279 => (sym - 256, 7),                // 0000000..0010111
        280..=287 => (0b1100_0000 + (sym - 280), 8), // 11000000..11000111
        _ => unreachable!("litlen symbol out of range: {sym}"),
    }
}

/// Distance symbol (0..=29) -> (code, 5 bits). Fixed tree uses 5-bit codes
/// that are the symbol value itself, MSB-first.
fn fixed_dist_code(sym: u32) -> (u32, u32) {
    (sym, 5)
}

// Length code base values + extra bits (RFC 1951 §3.2.5, table for codes
// 257..=285). Index = length code - 257.
const LENGTH_BASE: [u32; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Map a match length (3..=258) to its length code (257..=285) and the
/// extra-bit value to transmit after the code.
fn length_to_code(length: usize) -> (u32, u32, u32) {
    debug_assert!((MIN_MATCH..=MAX_MATCH).contains(&length));
    // Walk from the top so the largest base <= length wins.
    let mut idx = 28;
    while idx > 0 && LENGTH_BASE[idx] as usize > length {
        idx -= 1;
    }
    let code = 257 + idx as u32;
    let extra_bits = LENGTH_EXTRA[idx];
    let extra_val = length as u32 - LENGTH_BASE[idx];
    (code, extra_bits, extra_val)
}

/// Map a match distance (1..=32768) to its distance code (0..=29) and the
/// extra-bit value to transmit after the code.
fn distance_to_code(distance: usize) -> (u32, u32, u32) {
    debug_assert!((1..=WINDOW_SIZE).contains(&distance));
    let mut idx = 29;
    while idx > 0 && DIST_BASE[idx] as usize > distance {
        idx -= 1;
    }
    let code = idx as u32;
    let extra_bits = DIST_EXTRA[idx];
    let extra_val = distance as u32 - DIST_BASE[idx];
    (code, extra_bits, extra_val)
}

/// Hash 3 bytes into a chain-table bucket.
fn hash3(a: u8, b: u8, c: u8) -> usize {
    let v = (u32::from(a) << 16) | (u32::from(b) << 8) | u32::from(c);
    // Multiplicative hash; take the high HASH_BITS.
    ((v.wrapping_mul(0x9E37_79B1)) >> (32 - HASH_BITS)) as usize
}

/// Compress `input` to a raw DEFLATE stream (RFC 1951), a single BFINAL
/// block using the fixed Huffman code table.
pub fn deflate(input: &[u8]) -> Vec<u8> {
    let mut bw = BitWriter::new();
    // Block header: BFINAL=1 (1 bit), BTYPE=01 fixed Huffman (2 bits).
    // These header bits are NOT Huffman codes; emit LSB-first directly.
    bw.write_bits(1, 1); // BFINAL = 1
    bw.write_bits(0b01, 2); // BTYPE = 01 (fixed Huffman)

    if input.is_empty() {
        // Just the end-of-block symbol (256).
        let (code, len) = fixed_litlen_code(256);
        bw.write_code(code, len);
        return bw.finish();
    }

    // Hash-chain match finder. `head[h]` = most recent position with hash h;
    // `prev[pos & MASK]` = previous position with the same hash (chain).
    let mut head = vec![usize::MAX; HASH_SIZE];
    let mut prev = vec![usize::MAX; input.len()];

    let emit_literal = |bw: &mut BitWriter, byte: u8| {
        let (code, len) = fixed_litlen_code(u32::from(byte));
        bw.write_code(code, len);
    };

    let n = input.len();
    let mut pos = 0usize;
    while pos < n {
        // Need at least MIN_MATCH bytes ahead to form a match.
        let mut best_len = 0usize;
        let mut best_dist = 0usize;

        if pos + MIN_MATCH <= n {
            let h = hash3(input[pos], input[pos + 1], input[pos + 2]);
            let mut cand = head[h];
            let max_len = (n - pos).min(MAX_MATCH);
            let window_start = pos.saturating_sub(WINDOW_SIZE);
            let mut chain = 0;
            while cand != usize::MAX && cand >= window_start && chain < MAX_CHAIN {
                // Quick reject: compare the byte just past the current best.
                if best_len == 0
                    || (cand + best_len < n && input[cand + best_len] == input[pos + best_len])
                {
                    // Count the actual match length.
                    let mut l = 0usize;
                    while l < max_len && input[cand + l] == input[pos + l] {
                        l += 1;
                    }
                    if l > best_len {
                        best_len = l;
                        best_dist = pos - cand;
                        if l >= max_len {
                            break;
                        }
                    }
                }
                cand = prev[cand];
                chain += 1;
            }
        }

        if best_len >= MIN_MATCH {
            // Emit a length/distance back-reference.
            let (lc, lextra_bits, lextra_val) = length_to_code(best_len);
            let (code, len) = fixed_litlen_code(lc);
            bw.write_code(code, len);
            if lextra_bits > 0 {
                bw.write_bits(lextra_val, lextra_bits);
            }
            let (dc, dextra_bits, dextra_val) = distance_to_code(best_dist);
            let (code, len) = fixed_dist_code(dc);
            bw.write_code(code, len);
            if dextra_bits > 0 {
                bw.write_bits(dextra_val, dextra_bits);
            }

            // Insert hash entries for every position covered by the match so
            // future matches can reference into the middle of it.
            let end = pos + best_len;
            while pos < end {
                if pos + MIN_MATCH <= n {
                    let h = hash3(input[pos], input[pos + 1], input[pos + 2]);
                    prev[pos] = head[h];
                    head[h] = pos;
                }
                pos += 1;
            }
        } else {
            // No usable match: emit a literal and advance one byte.
            emit_literal(&mut bw, input[pos]);
            if pos + MIN_MATCH <= n {
                let h = hash3(input[pos], input[pos + 1], input[pos + 2]);
                prev[pos] = head[h];
                head[h] = pos;
            }
            pos += 1;
        }
    }

    // End-of-block symbol (256).
    let (code, len) = fixed_litlen_code(256);
    bw.write_code(code, len);
    bw.finish()
}

// --- CRC-32 (RFC 1952 / ISO 3309) -------------------------------------------

/// CRC-32 of `data` using the standard (reflected) polynomial 0xEDB88320.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// --- Adler-32 (RFC 1950 §9) --------------------------------------------------

/// Adler-32 checksum of `data` (RFC 1950).
pub fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut a = 1u32;
    let mut b = 0u32;
    // Process in chunks so the running sums never overflow before the mod.
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += u32::from(byte);
            b += a;
        }
        a %= MOD_ADLER;
        b %= MOD_ADLER;
    }
    (b << 16) | a
}

/// Wrap `input` in a gzip member (RFC 1952): 10-byte header, raw DEFLATE
/// payload, then 8-byte trailer = CRC-32(input) LE + ISIZE (input length mod
/// 2^32) LE. Matches the framing produced by Chrome's `"gzip"` format.
pub fn encode_gzip(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // Header.
    out.push(0x1F); // magic
    out.push(0x8B);
    out.push(0x08); // CM = 8 (DEFLATE)
    out.push(0x00); // FLG = 0 (no extra/name/comment/hcrc)
    out.extend_from_slice(&0u32.to_le_bytes()); // MTIME = 0 (Chrome uses 0)
    out.push(0x00); // XFL = 0
    out.push(0xFF); // OS = 255 (unknown) — what Chrome's compression emits
    // Payload.
    out.extend_from_slice(&deflate(input));
    // Trailer.
    out.extend_from_slice(&crc32(input).to_le_bytes());
    out.extend_from_slice(&(input.len() as u32).to_le_bytes());
    out
}

/// Wrap `input` in a zlib stream (RFC 1950): 2-byte header (CMF=0x78,
/// FLG chosen so the 16-bit header is a multiple of 31), raw DEFLATE
/// payload, then 4-byte big-endian Adler-32 trailer. Matches Chrome's
/// `"deflate"` format.
pub fn encode_zlib(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let cmf: u8 = 0x78; // CM=8, CINFO=7 (32K window)
    // FLG: FLEVEL=0, FDICT=0, FCHECK chosen so (cmf*256 + flg) % 31 == 0.
    let mut flg: u8 = 0;
    let rem = ((u16::from(cmf) << 8) | u16::from(flg)) % 31;
    if rem != 0 {
        flg += (31 - rem) as u8;
    }
    out.push(cmf);
    out.push(flg);
    out.extend_from_slice(&deflate(input));
    out.extend_from_slice(&adler32(input).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::inflate;
    use crate::gzip::{decode_gzip, decode_zlib};

    fn roundtrip_raw(data: &[u8]) {
        let compressed = deflate(data);
        let back = inflate(&compressed).expect("inflate");
        assert_eq!(back, data, "raw deflate round-trip mismatch");
    }

    #[test]
    fn empty_roundtrips() {
        roundtrip_raw(b"");
    }

    #[test]
    fn short_literal_roundtrips() {
        roundtrip_raw(b"a");
        roundtrip_raw(b"ab");
        roundtrip_raw(b"abc");
    }

    #[test]
    fn hello_world_roundtrips() {
        roundtrip_raw(b"hello world");
    }

    #[test]
    fn repeated_text_uses_backrefs_and_roundtrips() {
        let data = b"hello world".repeat(50);
        let compressed = deflate(&data);
        // Highly repetitive input must compress well below the original via
        // back-references — proves the LZ77 finder actually fires.
        assert!(
            compressed.len() < data.len() / 2,
            "expected compression, got {} from {}",
            compressed.len(),
            data.len()
        );
        let back = inflate(&compressed).expect("inflate");
        assert_eq!(back, data);
    }

    #[test]
    fn all_byte_values_roundtrip() {
        let data: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        roundtrip_raw(&data);
    }

    #[test]
    fn binary_blob_roundtrips() {
        // Pseudo-random-ish blob (no compressibility) must still round-trip.
        let mut data = Vec::new();
        let mut x: u32 = 0x1234_5678;
        for _ in 0..5000 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            data.push((x >> 16) as u8);
        }
        roundtrip_raw(&data);
    }

    #[test]
    fn gzip_frame_has_magic_and_roundtrips() {
        let data = b"the quick brown fox jumps over the lazy dog".repeat(10);
        let gz = encode_gzip(&data);
        assert_eq!(&gz[0..3], &[0x1F, 0x8B, 0x08], "gzip magic + CM");
        let back = decode_gzip(&gz).expect("decode_gzip");
        assert_eq!(back, data);
    }

    #[test]
    fn gzip_trailer_crc_and_isize_correct() {
        let data = b"hello\n";
        let gz = encode_gzip(data);
        let n = gz.len();
        let crc = u32::from_le_bytes([gz[n - 8], gz[n - 7], gz[n - 6], gz[n - 5]]);
        let isize = u32::from_le_bytes([gz[n - 4], gz[n - 3], gz[n - 2], gz[n - 1]]);
        assert_eq!(crc, crc32(data), "gzip CRC-32 trailer");
        assert_eq!(isize, data.len() as u32, "gzip ISIZE trailer");
    }

    #[test]
    fn zlib_frame_roundtrips_and_header_valid() {
        let data = b"deflate stream contents, repeated. ".repeat(20);
        let zl = encode_zlib(&data);
        // FCHECK: (CMF*256 + FLG) % 31 == 0 (RFC 1950 §2.2).
        let combined = (u16::from(zl[0]) << 8) | u16::from(zl[1]);
        assert_eq!(combined % 31, 0, "zlib FCHECK");
        let back = decode_zlib(&zl).expect("decode_zlib");
        assert_eq!(back, data);
    }

    #[test]
    fn raw_deflate_differs_from_zlib_by_wrapper() {
        let data = b"hello world";
        let raw = deflate(data);
        let zlib = encode_zlib(data);
        // zlib = 2-byte header + raw DEFLATE + 4-byte Adler-32 trailer.
        assert_eq!(zlib.len(), raw.len() + 6, "zlib wrapper is 6 bytes");
        assert_eq!(&zlib[2..2 + raw.len()], &raw[..], "zlib body == raw deflate");
    }

    #[test]
    fn crc32_known_vector() {
        // CRC-32 of "hello\n" is 0x363a3020 (matches gzip.rs decode test).
        assert_eq!(crc32(b"hello\n"), 0x363a_3020);
        // CRC-32 of the empty string is 0.
        assert_eq!(crc32(b""), 0);
        // Standard check vector: CRC-32("123456789") == 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn adler32_known_vector() {
        // Adler-32 of "hello\n" is 0x084B021F (verified against zlib).
        assert_eq!(adler32(b"hello\n"), 0x084B_021F);
        // Adler-32 of the empty string is 1 (RFC 1950: initial value).
        assert_eq!(adler32(b""), 1);
        // Standard check vector from the Adler-32 spec example:
        // Adler-32("Wikipedia") == 0x11E60398.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }
}
