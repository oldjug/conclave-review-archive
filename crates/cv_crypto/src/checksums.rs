//! CRC32 (ISO 3309 / ITU-T V.42) and Adler-32 (RFC 1950) checksums.
//! Neither is a cryptographic hash, but they're so widely used by
//! HTTP, PNG, zlib, ZIP, gzip and many JS networking libraries that
//! shipping them in `cv_crypto` is the simplest path.

const CRC32_POLY: u32 = 0xEDB88320; // reversed IEEE 802.3

static CRC32_TABLE: once_cell_table::Table = once_cell_table::Table::new();

mod once_cell_table {
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicBool, Ordering};
    pub struct Table {
        init: AtomicBool,
        data: UnsafeCell<[u32; 256]>,
    }
    unsafe impl Sync for Table {}
    impl Table {
        pub const fn new() -> Self {
            Self {
                init: AtomicBool::new(false),
                data: UnsafeCell::new([0u32; 256]),
            }
        }
        pub fn get(&self) -> &[u32; 256] {
            if !self.init.load(Ordering::Acquire) {
                unsafe {
                    let t = &mut *self.data.get();
                    for i in 0..256u32 {
                        let mut c = i;
                        for _ in 0..8 {
                            c = if c & 1 != 0 {
                                super::CRC32_POLY ^ (c >> 1)
                            } else {
                                c >> 1
                            };
                        }
                        t[i as usize] = c;
                    }
                }
                self.init.store(true, Ordering::Release);
            }
            unsafe { &*self.data.get() }
        }
    }
}

/// Streaming CRC-32 (ISO 3309 / PKZIP / PNG variant).
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    pub fn new() -> Self {
        Self { state: 0xFFFFFFFF }
    }
    pub fn update(&mut self, data: &[u8]) {
        let tbl = CRC32_TABLE.get();
        let mut c = self.state;
        for &b in data {
            c = tbl[((c ^ (b as u32)) & 0xFF) as usize] ^ (c >> 8);
        }
        self.state = c;
    }
    pub fn finalize(self) -> u32 {
        self.state ^ 0xFFFFFFFF
    }
    pub fn oneshot(data: &[u8]) -> u32 {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }
}

/// Streaming Adler-32 per RFC 1950 §9 (zlib trailer).
pub struct Adler32 {
    a: u32,
    b: u32,
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Adler32 {
    pub fn new() -> Self {
        Self { a: 1, b: 0 }
    }
    pub fn update(&mut self, data: &[u8]) {
        // Modulus 65521 ≤ largest prime under 2^16. Process in 5552-byte
        // chunks so the accumulator doesn't overflow before reduction.
        for chunk in data.chunks(5552) {
            for &x in chunk {
                self.a = self.a.wrapping_add(x as u32);
                self.b = self.b.wrapping_add(self.a);
            }
            self.a %= 65521;
            self.b %= 65521;
        }
    }
    pub fn finalize(self) -> u32 {
        (self.b << 16) | self.a
    }
    pub fn oneshot(data: &[u8]) -> u32 {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // PKZIP CRC32 vectors.
    #[test]
    fn crc32_empty() {
        assert_eq!(Crc32::oneshot(b""), 0x00000000);
    }

    #[test]
    fn crc32_quick_brown_fox() {
        assert_eq!(
            Crc32::oneshot(b"The quick brown fox jumps over the lazy dog"),
            0x414FA339,
        );
    }

    #[test]
    fn crc32_123456789() {
        // CRC catalogue check value for CRC-32/ISO-HDLC.
        assert_eq!(Crc32::oneshot(b"123456789"), 0xCBF43926);
    }

    // RFC 1950 Appendix Adler32 vectors.
    #[test]
    fn adler32_empty() {
        assert_eq!(Adler32::oneshot(b""), 1);
    }

    #[test]
    fn adler32_abc() {
        assert_eq!(Adler32::oneshot(b"abc"), 0x024D0127);
    }
}
