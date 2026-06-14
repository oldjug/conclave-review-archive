//! UTF-16 strings — the in-memory form the DOM uses per spec.
//!
//! `String16` owns; `Str16` borrows. The DOM stores ~90% short ASCII so a
//! later optimization will add an inline / refcounted "DOMString" wrapper
//! on top of these. For now the simple owned/borrowed pair is enough.

use core::fmt;

#[derive(Default, Clone, PartialEq, Eq, Hash)]
pub struct String16(Vec<u16>);

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash)]
pub struct Str16([u16]);

impl String16 {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity(cap))
    }

    pub fn from_utf8(s: &str) -> Self {
        Self(s.encode_utf16().collect())
    }

    pub fn from_units(units: Vec<u16>) -> Self {
        Self(units)
    }

    pub fn as_units(&self) -> &[u16] {
        &self.0
    }

    pub fn as_str16(&self) -> &Str16 {
        Str16::from_units(&self.0)
    }

    pub fn push(&mut self, c: char) {
        let mut buf = [0u16; 2];
        let s = c.encode_utf16(&mut buf);
        self.0.extend_from_slice(s);
    }

    pub fn push_str(&mut self, s: &str) {
        self.0.extend(s.encode_utf16());
    }

    pub fn push_str16(&mut self, s: &Str16) {
        self.0.extend_from_slice(&s.0);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Lossy UTF-8 decode. Replaces unpaired surrogates with U+FFFD.
    pub fn to_utf8(&self) -> String {
        String::from_utf16_lossy(&self.0)
    }
}

impl Str16 {
    pub fn from_units(u: &[u16]) -> &Self {
        // SAFETY: `Str16` is `#[repr(transparent)]` over `[u16]`.
        unsafe { &*(std::ptr::from_ref::<[u16]>(u) as *const Str16) }
    }

    pub fn as_units(&self) -> &[u16] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for String16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("String16").field(&self.to_utf8()).finish()
    }
}

impl fmt::Display for String16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for ch in std::char::decode_utf16(self.0.iter().copied()) {
            let c = ch.unwrap_or(char::REPLACEMENT_CHARACTER);
            f.write_str(c.encode_utf8(&mut [0; 4]))?;
        }
        Ok(())
    }
}

impl fmt::Debug for Str16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Str16")
            .field(&String::from_utf16_lossy(&self.0))
            .finish()
    }
}

impl From<&str> for String16 {
    fn from(s: &str) -> Self {
        Self::from_utf8(s)
    }
}

impl From<String> for String16 {
    fn from(s: String) -> Self {
        Self::from_utf8(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii() {
        let s = String16::from_utf8("hello");
        assert_eq!(s.len(), 5);
        assert_eq!(s.to_utf8(), "hello");
    }

    #[test]
    fn roundtrip_bmp_and_supplementary() {
        let s = String16::from_utf8("résumé 🦀");
        assert_eq!(s.to_utf8(), "résumé 🦀");
    }

    #[test]
    fn push_char_surrogate_pair() {
        let mut s = String16::new();
        s.push('🦀');
        assert_eq!(s.len(), 2); // surrogate pair
        assert_eq!(s.to_utf8(), "🦀");
    }
}
