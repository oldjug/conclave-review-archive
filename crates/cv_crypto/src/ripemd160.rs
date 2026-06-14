//! RIPEMD-160 per Dobbertin/Bosselaers/Preneel (1996). 160-bit digest,
//! 64-byte block. Used by Bitcoin P2PKH addresses, Ethereum
//! checksums via `keccak`/`ripemd` chains, and a few older auth
//! flows. Web Crypto doesn't ship it directly, but many JS wallet
//! libraries probe for "ripemd160" digest support.

#[derive(Clone)]
pub struct Ripemd160 {
    h: [u32; 5],
    buf: [u8; 64],
    buf_len: usize,
    msg_len_bits: u64,
}

impl Default for Ripemd160 {
    fn default() -> Self {
        Self::new()
    }
}

impl Ripemd160 {
    pub fn new() -> Self {
        Self {
            h: [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0],
            buf: [0u8; 64],
            buf_len: 0,
            msg_len_bits: 0,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.msg_len_bits = self.msg_len_bits.wrapping_add((data.len() as u64) * 8);
        let mut i = 0;
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            i += take;
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while i + 64 <= data.len() {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[i..i + 64]);
            self.compress(&block);
            i += 64;
        }
        let rem = data.len() - i;
        if rem > 0 {
            self.buf[..rem].copy_from_slice(&data[i..]);
            self.buf_len = rem;
        }
    }

    pub fn finalize(mut self) -> [u8; 20] {
        let mut pad = [0u8; 128];
        pad[0] = 0x80;
        let len_le = self.msg_len_bits.to_le_bytes();
        let cur = self.buf_len;
        let pad_len = if cur < 56 { 56 - cur } else { 120 - cur };
        self.update(&pad[..pad_len]);
        self.update(&len_le);
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 20];
        for (i, w) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    pub fn oneshot(data: &[u8]) -> [u8; 20] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }

    fn compress(&mut self, block: &[u8; 64]) {
        // Parse the block as 16 LE 32-bit words.
        let mut x = [0u32; 16];
        for i in 0..16 {
            x[i] = u32::from_le_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        // Left line message schedule and rotation amounts (RIPEMD-160 spec).
        const R_L: [usize; 80] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0,
            9, 5, 2, 14, 11, 8, 3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10,
            0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2, 4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6,
            15, 13,
        ];
        const R_R: [usize; 80] = [
            5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14,
            15, 8, 12, 4, 9, 1, 2, 15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4,
            1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14, 12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14,
            0, 3, 9, 11,
        ];
        const S_L: [u32; 80] = [
            11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7,
            12, 15, 9, 11, 7, 13, 12, 11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, 11,
            12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12, 9, 15, 5, 11, 6, 8, 13, 12, 5, 12,
            13, 14, 11, 8, 5, 6,
        ];
        const S_R: [u32; 80] = [
            8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7,
            7, 12, 7, 6, 15, 13, 11, 9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, 15, 5,
            8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8, 8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6,
            5, 15, 13, 11, 11,
        ];
        const K_L: [u32; 5] = [0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E];
        const K_R: [u32; 5] = [0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000];

        let f = |j: usize, x: u32, y: u32, z: u32| -> u32 {
            match j {
                0..=15 => x ^ y ^ z,
                16..=31 => (x & y) | ((!x) & z),
                32..=47 => (x | !y) ^ z,
                48..=63 => (x & z) | (y & !z),
                _ => x ^ (y | !z),
            }
        };

        let mut a_l = self.h[0];
        let mut b_l = self.h[1];
        let mut c_l = self.h[2];
        let mut d_l = self.h[3];
        let mut e_l = self.h[4];
        let mut a_r = self.h[0];
        let mut b_r = self.h[1];
        let mut c_r = self.h[2];
        let mut d_r = self.h[3];
        let mut e_r = self.h[4];

        for j in 0..80 {
            let group = j / 16;
            // Left line
            let t = a_l
                .wrapping_add(f(j, b_l, c_l, d_l))
                .wrapping_add(x[R_L[j]])
                .wrapping_add(K_L[group])
                .rotate_left(S_L[j])
                .wrapping_add(e_l);
            a_l = e_l;
            e_l = d_l;
            d_l = c_l.rotate_left(10);
            c_l = b_l;
            b_l = t;
            // Right line uses f from the opposite end.
            let group_r = 4 - group;
            let t = a_r
                .wrapping_add(f(group_r * 16 + j % 16, b_r, c_r, d_r))
                .wrapping_add(x[R_R[j]])
                .wrapping_add(K_R[group])
                .rotate_left(S_R[j])
                .wrapping_add(e_r);
            a_r = e_r;
            e_r = d_r;
            d_r = c_r.rotate_left(10);
            c_r = b_r;
            b_r = t;
        }

        let t = self.h[1].wrapping_add(c_l).wrapping_add(d_r);
        self.h[1] = self.h[2].wrapping_add(d_l).wrapping_add(e_r);
        self.h[2] = self.h[3].wrapping_add(e_l).wrapping_add(a_r);
        self.h[3] = self.h[4].wrapping_add(a_l).wrapping_add(b_r);
        self.h[4] = self.h[0].wrapping_add(b_l).wrapping_add(c_r);
        self.h[0] = t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for byte in b {
            s.push_str(&format!("{:02x}", byte));
        }
        s
    }

    // Official RIPEMD-160 test vectors from Bosselaers/Preneel paper.
    #[test]
    fn empty() {
        assert_eq!(
            hex(&Ripemd160::oneshot(b"")),
            "9c1185a5c5e9fc54612808977ee8f548b2258d31"
        );
    }

    #[test]
    fn abc() {
        assert_eq!(
            hex(&Ripemd160::oneshot(b"abc")),
            "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"
        );
    }
}
