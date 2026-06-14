//! HMAC-SHA256 per FIPS 198-1 / RFC 2104.

use crate::sha256::{self, Sha256};

pub const OUTPUT_SIZE: usize = sha256::OUTPUT_SIZE;
const BLOCK_SIZE: usize = sha256::BLOCK_SIZE;

#[derive(Clone)]
pub struct HmacSha256 {
    inner: Sha256,
    outer_key: [u8; BLOCK_SIZE],
}

impl std::fmt::Debug for HmacSha256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HmacSha256").finish_non_exhaustive()
    }
}

impl HmacSha256 {
    pub fn new(key: &[u8]) -> Self {
        let mut block = [0u8; BLOCK_SIZE];
        if key.len() > BLOCK_SIZE {
            block[..OUTPUT_SIZE].copy_from_slice(&Sha256::oneshot(key));
        } else {
            block[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0u8; BLOCK_SIZE];
        let mut opad = [0u8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            ipad[i] = block[i] ^ 0x36;
            opad[i] = block[i] ^ 0x5c;
        }
        let mut inner = Sha256::new();
        inner.update(&ipad);
        Self {
            inner,
            outer_key: opad,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    pub fn finalize(self) -> [u8; OUTPUT_SIZE] {
        let inner = self.inner.finalize();
        let mut outer = Sha256::new();
        outer.update(&self.outer_key);
        outer.update(&inner);
        outer.finalize()
    }

    pub fn oneshot(key: &[u8], data: &[u8]) -> [u8; OUTPUT_SIZE] {
        let mut h = Self::new(key);
        h.update(data);
        h.finalize()
    }
}

// ---------------------------------------------------------------------------
// HMAC-SHA1 — same construction over SHA-1. Used by `crypto.subtle.sign`
// with `{name: "HMAC", hash: "SHA-1"}` and the WebSocket handshake.

const SHA1_BLOCK_SIZE: usize = 64;
const SHA1_OUTPUT_SIZE: usize = 20;

pub struct HmacSha1 {
    inner: crate::sha1::Sha1,
    outer_key: [u8; SHA1_BLOCK_SIZE],
}

impl HmacSha1 {
    pub fn new(key: &[u8]) -> Self {
        let mut block = [0u8; SHA1_BLOCK_SIZE];
        if key.len() > SHA1_BLOCK_SIZE {
            block[..SHA1_OUTPUT_SIZE].copy_from_slice(&crate::sha1::Sha1::oneshot(key));
        } else {
            block[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0u8; SHA1_BLOCK_SIZE];
        let mut opad = [0u8; SHA1_BLOCK_SIZE];
        for i in 0..SHA1_BLOCK_SIZE {
            ipad[i] = block[i] ^ 0x36;
            opad[i] = block[i] ^ 0x5c;
        }
        let mut inner = crate::sha1::Sha1::new();
        inner.update(&ipad);
        Self {
            inner,
            outer_key: opad,
        }
    }
    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }
    pub fn finalize(self) -> [u8; SHA1_OUTPUT_SIZE] {
        let inner = self.inner.finalize();
        let mut outer = crate::sha1::Sha1::new();
        outer.update(&self.outer_key);
        outer.update(&inner);
        outer.finalize()
    }
    pub fn oneshot(key: &[u8], data: &[u8]) -> [u8; SHA1_OUTPUT_SIZE] {
        let mut h = Self::new(key);
        h.update(data);
        h.finalize()
    }
}

// ---------------------------------------------------------------------------
// HMAC-SHA512 — needed by JWT (HS512), CRAM-SHA512, anything using
// `crypto.subtle.sign({name:"HMAC", hash:"SHA-512"})`.

const SHA512_BLOCK_SIZE: usize = 128;
const SHA512_OUTPUT_SIZE: usize = 64;

pub struct HmacSha512 {
    inner: crate::sha512::Sha512,
    outer_key: [u8; SHA512_BLOCK_SIZE],
}

impl HmacSha512 {
    pub fn new(key: &[u8]) -> Self {
        let mut block = [0u8; SHA512_BLOCK_SIZE];
        if key.len() > SHA512_BLOCK_SIZE {
            block[..SHA512_OUTPUT_SIZE].copy_from_slice(&crate::sha512::Sha512::oneshot(key));
        } else {
            block[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0u8; SHA512_BLOCK_SIZE];
        let mut opad = [0u8; SHA512_BLOCK_SIZE];
        for i in 0..SHA512_BLOCK_SIZE {
            ipad[i] = block[i] ^ 0x36;
            opad[i] = block[i] ^ 0x5c;
        }
        let mut inner = crate::sha512::Sha512::new();
        inner.update(&ipad);
        Self {
            inner,
            outer_key: opad,
        }
    }
    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }
    pub fn finalize(self) -> [u8; SHA512_OUTPUT_SIZE] {
        let inner = self.inner.finalize();
        let mut outer = crate::sha512::Sha512::new();
        outer.update(&self.outer_key);
        outer.update(&inner);
        outer.finalize()
    }
    pub fn oneshot(key: &[u8], data: &[u8]) -> [u8; SHA512_OUTPUT_SIZE] {
        let mut h = Self::new(key);
        h.update(data);
        h.finalize()
    }
}

#[cfg(test)]
mod hmac_extra_tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for byte in b {
            s.push_str(&format!("{:02x}", byte));
        }
        s
    }

    // RFC 2202 test case 1.
    #[test]
    fn hmac_sha1_rfc2202_1() {
        let key = [0x0bu8; 20];
        let data = b"Hi There";
        let tag = HmacSha1::oneshot(&key, data);
        assert_eq!(hex(&tag), "b617318655057264e28bc0b6fb378c8ef146be00");
    }

    // RFC 4231 test case 1.
    #[test]
    fn hmac_sha512_rfc4231_1() {
        let key = [0x0bu8; 20];
        let data = b"Hi There";
        let tag = HmacSha512::oneshot(&key, data);
        assert_eq!(
            hex(&tag),
            "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854"
        );
    }
}

// --------------------------- HMAC-SHA384 ----------------------------

use crate::sha384::{self, Sha384};

pub const HMAC_SHA384_OUTPUT_SIZE: usize = sha384::OUTPUT_SIZE;
const SHA384_BLOCK_SIZE: usize = sha384::BLOCK_SIZE;

#[derive(Clone)]
pub struct HmacSha384 {
    inner: Sha384,
    outer_key: [u8; SHA384_BLOCK_SIZE],
}

impl std::fmt::Debug for HmacSha384 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HmacSha384").finish_non_exhaustive()
    }
}

impl HmacSha384 {
    pub fn new(key: &[u8]) -> Self {
        let mut block = [0u8; SHA384_BLOCK_SIZE];
        if key.len() > SHA384_BLOCK_SIZE {
            block[..HMAC_SHA384_OUTPUT_SIZE].copy_from_slice(&Sha384::oneshot(key));
        } else {
            block[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0u8; SHA384_BLOCK_SIZE];
        let mut opad = [0u8; SHA384_BLOCK_SIZE];
        for i in 0..SHA384_BLOCK_SIZE {
            ipad[i] = block[i] ^ 0x36;
            opad[i] = block[i] ^ 0x5c;
        }
        let mut inner = Sha384::new();
        inner.update(&ipad);
        Self {
            inner,
            outer_key: opad,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    pub fn finalize(self) -> [u8; HMAC_SHA384_OUTPUT_SIZE] {
        let inner = self.inner.finalize();
        let mut outer = Sha384::new();
        outer.update(&self.outer_key);
        outer.update(&inner);
        outer.finalize()
    }

    pub fn oneshot(key: &[u8], data: &[u8]) -> [u8; HMAC_SHA384_OUTPUT_SIZE] {
        let mut h = Self::new(key);
        h.update(data);
        h.finalize()
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

    /// RFC 4231 test case 1.
    #[test]
    fn rfc4231_tc1() {
        let key = [0x0b_u8; 20];
        let data = b"Hi There";
        let mac = HmacSha256::oneshot(&key, data);
        assert_eq!(
            hex(&mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    /// RFC 4231 test case 2: short key.
    #[test]
    fn rfc4231_tc2() {
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let mac = HmacSha256::oneshot(key, data);
        assert_eq!(
            hex(&mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    /// RFC 4231 test case 1 for HMAC-SHA384.
    #[test]
    fn rfc4231_tc1_sha384() {
        let key = [0x0b_u8; 20];
        let data = b"Hi There";
        let mac = HmacSha384::oneshot(&key, data);
        assert_eq!(
            hex(&mac),
            "afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59cfaea9ea9076ede7f4af152e8b2fa9cb6"
        );
    }

    /// RFC 4231 test case 2 for HMAC-SHA384.
    #[test]
    fn rfc4231_tc2_sha384() {
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let mac = HmacSha384::oneshot(key, data);
        assert_eq!(
            hex(&mac),
            "af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e8e2240ca5e69e2c78b3239ecfab21649"
        );
    }

    /// RFC 4231 test case 4 — long key for HMAC-SHA384.
    #[test]
    fn rfc4231_tc4_sha384() {
        let key: Vec<u8> = (0x01..=0x19).collect();
        let data = [0xcd_u8; 50];
        let mac = HmacSha384::oneshot(&key, &data);
        assert_eq!(
            hex(&mac),
            "3e8a69b7783c25851933ab6290af6ca77a9981480850009cc5577c6e1f573b4e6801dd23c4a7d679ccf8a386c674cffb"
        );
    }

    /// RFC 4231 test case 4: long key & data.
    #[test]
    fn rfc4231_tc4() {
        let key: Vec<u8> = (0x01..=0x19).collect();
        let data = [0xcd_u8; 50];
        let mac = HmacSha256::oneshot(&key, &data);
        assert_eq!(
            hex(&mac),
            "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"
        );
    }

    /// RFC 4231 test case 6 — KEY LONGER THAN BLOCK SIZE for HMAC-SHA384.
    /// 131-byte key (vs SHA-384's 128-byte block) exercises the
    /// "hash key first" path in HmacSha384::new.
    #[test]
    fn rfc4231_tc6_sha384() {
        let key = vec![0xaa_u8; 131];
        let data = b"Test Using Larger Than Block-Size Key - Hash Key First";
        let mac = HmacSha384::oneshot(&key, data);
        assert_eq!(
            hex(&mac),
            "4ece084485813e9088d2c63a041bc5b44f9ef1012a2b588f3cd11f05033ac4c60c2ef6ab4030fe8296248df163f44952"
        );
    }

    /// RFC 4231 test case 7 — long key + long data for HMAC-SHA384.
    #[test]
    fn rfc4231_tc7_sha384() {
        let key = vec![0xaa_u8; 131];
        let data = b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.";
        let mac = HmacSha384::oneshot(&key, data);
        assert_eq!(
            hex(&mac),
            "6617178e941f020d351e2f254e8fd32c602420feb0b8fb9adccebb82461e99c5a678cc31e799176d3860e6110c46523e"
        );
    }
}
