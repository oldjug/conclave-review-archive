//! Poly1305 MAC per RFC 8439 §2.5. Implemented with 64-bit limbs over the
//! 130-bit prime 2^130 - 5. Keyed by a one-time 32-byte key.

pub const KEY_SIZE: usize = 32;
pub const TAG_SIZE: usize = 16;

#[derive(Debug)]
pub struct Poly1305 {
    r: [u64; 3], // 130-bit clamped r, three 44/42-bit limbs
    s: [u32; 4], // 128-bit "s" half of the key (added at the end)
    h: [u64; 3], // 130-bit accumulator
    buffer: [u8; 16],
    buffered: usize,
}

impl Poly1305 {
    pub fn new(key: &[u8; KEY_SIZE]) -> Self {
        // Clamp r per RFC 8439 §2.5.
        let mut r_bytes = [0u8; 16];
        r_bytes.copy_from_slice(&key[..16]);
        r_bytes[3] &= 15;
        r_bytes[7] &= 15;
        r_bytes[11] &= 15;
        r_bytes[15] &= 15;
        r_bytes[4] &= 252;
        r_bytes[8] &= 252;
        r_bytes[12] &= 252;

        let r0 = u32::from_le_bytes(r_bytes[0..4].try_into().unwrap()) as u64;
        let r1 = u32::from_le_bytes(r_bytes[4..8].try_into().unwrap()) as u64;
        let r2 = u32::from_le_bytes(r_bytes[8..12].try_into().unwrap()) as u64;
        let r3 = u32::from_le_bytes(r_bytes[12..16].try_into().unwrap()) as u64;
        // Pack into three limbs ≤ 2^44.
        let r_lim = [
            r0 | (r1 << 32) & ((1u64 << 44) - 1),
            ((r1 >> 12) | (r2 << 20)) & ((1u64 << 44) - 1),
            ((r2 >> 24) | (r3 << 8)) & ((1u64 << 42) - 1),
        ];

        let mut s = [0u32; 4];
        for i in 0..4 {
            s[i] = u32::from_le_bytes(key[16 + i * 4..16 + i * 4 + 4].try_into().unwrap());
        }

        Self {
            r: r_lim,
            s,
            h: [0; 3],
            buffer: [0; 16],
            buffered: 0,
        }
    }

    fn process_block(&mut self, block: &[u8; 16], hibit: u64) {
        let t0 = u32::from_le_bytes(block[0..4].try_into().unwrap()) as u64;
        let t1 = u32::from_le_bytes(block[4..8].try_into().unwrap()) as u64;
        let t2 = u32::from_le_bytes(block[8..12].try_into().unwrap()) as u64;
        let t3 = u32::from_le_bytes(block[12..16].try_into().unwrap()) as u64;

        let mask = (1u64 << 44) - 1;
        // h += block
        self.h[0] += (t0 | (t1 << 32)) & mask;
        self.h[1] += ((t1 >> 12) | (t2 << 20)) & mask;
        self.h[2] += ((t2 >> 24) | (t3 << 8)) | (hibit << 40);

        // h *= r (mod 2^130 - 5)
        let r0 = self.r[0];
        let r1 = self.r[1];
        let r2 = self.r[2];

        let s1 = r1.wrapping_mul(5 << 2);
        let s2 = r2.wrapping_mul(5 << 2);

        let h0 = self.h[0];
        let h1 = self.h[1];
        let h2 = self.h[2];

        // 128-bit products using u128.
        let d0 =
            (h0 as u128) * (r0 as u128) + (h1 as u128) * (s2 as u128) + (h2 as u128) * (s1 as u128);
        let d1 =
            (h0 as u128) * (r1 as u128) + (h1 as u128) * (r0 as u128) + (h2 as u128) * (s2 as u128);
        let d2 =
            (h0 as u128) * (r2 as u128) + (h1 as u128) * (r1 as u128) + (h2 as u128) * (r0 as u128);

        let mut c0 = d0 >> 44;
        self.h[0] = (d0 as u64) & mask;
        let d1 = d1 + c0;
        c0 = d1 >> 44;
        self.h[1] = (d1 as u64) & mask;
        let d2 = d2 + c0;
        let mask42 = (1u64 << 42) - 1;
        c0 = d2 >> 42;
        self.h[2] = (d2 as u64) & mask42;

        // Carry top into bottom, scaled by 5 (since 2^130 ≡ 5 mod p).
        self.h[0] += (c0 as u64) * 5;
        let c = self.h[0] >> 44;
        self.h[0] &= mask;
        self.h[1] += c;
    }

    pub fn update(&mut self, mut data: &[u8]) {
        if self.buffered != 0 {
            let need = 16 - self.buffered;
            let take = need.min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 16 {
                let blk = self.buffer;
                self.process_block(&blk, 1);
                self.buffered = 0;
            }
        }
        while data.len() >= 16 {
            let blk: &[u8; 16] = data[..16].try_into().unwrap();
            self.process_block(blk, 1);
            data = &data[16..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; TAG_SIZE] {
        if self.buffered != 0 {
            // Pad with one 0x01 byte then zeros, hibit = 0.
            let n = self.buffered;
            let mut blk = [0u8; 16];
            blk[..n].copy_from_slice(&self.buffer[..n]);
            blk[n] = 1;
            self.process_block(&blk, 0);
        }

        let mask44 = (1u64 << 44) - 1;
        let mask42 = (1u64 << 42) - 1;

        let mut h0 = self.h[0];
        let mut h1 = self.h[1];
        let mut h2 = self.h[2];

        // Full carry propagation — two passes to ensure h < 2^130.
        // (poly1305-donna does the same.)
        let c = h1 >> 44;
        h1 &= mask44;
        h2 += c;
        let c = h2 >> 42;
        h2 &= mask42;
        h0 += c * 5;
        let c = h0 >> 44;
        h0 &= mask44;
        h1 += c;
        let c = h1 >> 44;
        h1 &= mask44;
        h2 += c;
        let c = h2 >> 42;
        h2 &= mask42;
        h0 += c * 5;
        let c = h0 >> 44;
        h0 &= mask44;
        h1 += c;

        // Compute g = h + (-p) = h + 5 - 2^130.
        let mut g0 = h0 + 5;
        let c = g0 >> 44;
        g0 &= mask44;
        let mut g1 = h1 + c;
        let c = g1 >> 44;
        g1 &= mask44;
        // h2 + c is in [0, 2^42 + small]; subtract 2^42 with wrap so the
        // high bit signals h < p (we underflowed) vs h >= p (we didn't).
        let g2 = (h2 + c).wrapping_sub(1u64 << 42);

        // mask = all 1s if g2 high bit clear (use g, i.e. h was >= p),
        // mask = 0 if g2 high bit set (keep h, i.e. h was < p).
        let select_g = (g2 >> 63).wrapping_sub(1);
        h0 = (h0 & !select_g) | (g0 & select_g);
        h1 = (h1 & !select_g) | (g1 & select_g);
        h2 = (h2 & !select_g) | (g2 & select_g);

        // Repack three (44, 44, 42)-bit limbs to four 32-bit words.
        //   bits 0..32  → h0_32: low 32 of limb 0
        //   bits 32..64 → h1_32: high 12 of limb 0 | low 20 of limb 1
        //   bits 64..96 → h2_32: high 24 of limb 1 | low 8 of limb 2
        //   bits 96..128 → h3_32: bits 8..40 of limb 2
        let h0_32 = h0 as u32;
        let h1_32 = ((h0 >> 32) | (h1 << 12)) as u32;
        let h2_32 = ((h1 >> 20) | (h2 << 24)) as u32;
        let h3_32 = (h2 >> 8) as u32;

        // Add s (mod 2^128).
        let s0 = self.s[0];
        let s1 = self.s[1];
        let s2 = self.s[2];
        let s3 = self.s[3];

        let r0 = (h0_32 as u64) + (s0 as u64);
        let r1 = (h1_32 as u64) + (s1 as u64) + (r0 >> 32);
        let r2 = (h2_32 as u64) + (s2 as u64) + (r1 >> 32);
        let r3 = (h3_32 as u64) + (s3 as u64) + (r2 >> 32);

        let mut tag = [0u8; TAG_SIZE];
        tag[0..4].copy_from_slice(&(r0 as u32).to_le_bytes());
        tag[4..8].copy_from_slice(&(r1 as u32).to_le_bytes());
        tag[8..12].copy_from_slice(&(r2 as u32).to_le_bytes());
        tag[12..16].copy_from_slice(&(r3 as u32).to_le_bytes());
        tag
    }

    pub fn oneshot(key: &[u8; KEY_SIZE], msg: &[u8]) -> [u8; TAG_SIZE] {
        let mut p = Self::new(key);
        p.update(msg);
        p.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for x in b {
            s.push_str(&format!("{x:02x}"));
        }
        s
    }

    /// RFC 8439 §2.5.2 test vector.
    #[test]
    fn rfc8439_2_5_2() {
        let key: [u8; 32] = [
            0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33, 0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5,
            0x06, 0xa8, 0x01, 0x03, 0x80, 0x8a, 0xfb, 0x0d, 0xb2, 0xfd, 0x4a, 0xbf, 0xf6, 0xaf,
            0x41, 0x49, 0xf5, 0x1b,
        ];
        let msg = b"Cryptographic Forum Research Group";
        let tag = Poly1305::oneshot(&key, msg);
        assert_eq!(hex(&tag), "a8061dc1305136c6c22b8baf0c0127a9");
    }
}
