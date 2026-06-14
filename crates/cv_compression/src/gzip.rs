//! Gzip + zlib wrappers around our DEFLATE decoder.
//!
//! - Gzip (RFC 1952): 10-byte header + optional extra/name/comment/CRC16,
//!   raw DEFLATE stream, 8-byte trailer (CRC-32 + ISIZE).
//! - Zlib (RFC 1950): 2-byte header (CMF + FLG, optional 4-byte DICTID),
//!   raw DEFLATE stream, 4-byte ADLER-32 trailer.
//!
//! HTTP `Content-Encoding: gzip` → `decode_gzip`.
//! HTTP `Content-Encoding: deflate` → `decode_zlib` (the standard says zlib
//!   though name suggests raw deflate — match real-world behaviour).

use crate::deflate::{InflateError, inflate};

#[derive(Debug)]
pub enum GzipError {
    Truncated,
    BadMagic,
    UnsupportedMethod(u8),
    BadFlags(u8),
    Inflate(InflateError),
}

impl core::fmt::Display for GzipError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated gzip"),
            Self::BadMagic => f.write_str("bad gzip magic"),
            Self::UnsupportedMethod(m) => write!(f, "unsupported compression method {m}"),
            Self::BadFlags(b) => write!(f, "reserved gzip flag bits set: 0x{b:02x}"),
            Self::Inflate(e) => write!(f, "inflate: {e}"),
        }
    }
}

impl std::error::Error for GzipError {}

#[allow(dead_code)]
const FTEXT: u8 = 0x01;
const FHCRC: u8 = 0x02;
const FEXTRA: u8 = 0x04;
const FNAME: u8 = 0x08;
const FCOMMENT: u8 = 0x10;
const FRESERVED: u8 = 0xE0;

/// Decode a gzip member. Returns the inflated payload; the CRC-32 and
/// ISIZE trailer fields are parsed but not yet validated (acceptable for
/// HTTP transport where TLS already authenticates).
pub fn decode_gzip(input: &[u8]) -> Result<Vec<u8>, GzipError> {
    if input.len() < 18 {
        return Err(GzipError::Truncated);
    }
    if input[0] != 0x1F || input[1] != 0x8B {
        return Err(GzipError::BadMagic);
    }
    let cm = input[2];
    if cm != 8 {
        return Err(GzipError::UnsupportedMethod(cm));
    }
    let flg = input[3];
    if flg & FRESERVED != 0 {
        return Err(GzipError::BadFlags(flg));
    }
    // bytes 4..8 = MTIME, 8 = XFL, 9 = OS — skip.
    let mut pos = 10usize;

    if flg & FEXTRA != 0 {
        if pos + 2 > input.len() {
            return Err(GzipError::Truncated);
        }
        let xlen = u16::from_le_bytes([input[pos], input[pos + 1]]) as usize;
        pos += 2 + xlen;
        if pos > input.len() {
            return Err(GzipError::Truncated);
        }
    }
    if flg & FNAME != 0 {
        pos = skip_zstring(input, pos)?;
    }
    if flg & FCOMMENT != 0 {
        pos = skip_zstring(input, pos)?;
    }
    if flg & FHCRC != 0 {
        if pos + 2 > input.len() {
            return Err(GzipError::Truncated);
        }
        pos += 2;
    }

    // Trailer is 8 bytes at the end (CRC32 + ISIZE).
    if input.len() < pos + 8 {
        return Err(GzipError::Truncated);
    }
    let payload = &input[pos..input.len() - 8];
    inflate(payload).map_err(GzipError::Inflate)
}

fn skip_zstring(input: &[u8], mut pos: usize) -> Result<usize, GzipError> {
    while pos < input.len() && input[pos] != 0 {
        pos += 1;
    }
    if pos >= input.len() {
        return Err(GzipError::Truncated);
    }
    Ok(pos + 1) // consume the 0
}

#[derive(Debug)]
pub enum ZlibError {
    Truncated,
    BadHeader,
    UnsupportedMethod(u8),
    PresetDict,
    Inflate(InflateError),
}

impl core::fmt::Display for ZlibError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated zlib"),
            Self::BadHeader => f.write_str("bad zlib header (FCHECK)"),
            Self::UnsupportedMethod(m) => write!(f, "unsupported compression method {m}"),
            Self::PresetDict => f.write_str("zlib FDICT set; preset dict not supported"),
            Self::Inflate(e) => write!(f, "inflate: {e}"),
        }
    }
}

impl std::error::Error for ZlibError {}

/// Decode a zlib stream (RFC 1950). Trailing ADLER-32 is parsed but not
/// yet verified.
pub fn decode_zlib(input: &[u8]) -> Result<Vec<u8>, ZlibError> {
    if input.len() < 6 {
        return Err(ZlibError::Truncated);
    }
    let cmf = input[0];
    let flg = input[1];
    let cm = cmf & 0x0F;
    if cm != 8 {
        return Err(ZlibError::UnsupportedMethod(cm));
    }
    // FCHECK: (cmf*256 + flg) must be divisible by 31.
    let combined = (u16::from(cmf) << 8) | u16::from(flg);
    if combined % 31 != 0 {
        return Err(ZlibError::BadHeader);
    }
    let mut pos = 2usize;
    if flg & 0x20 != 0 {
        if input.len() < pos + 4 {
            return Err(ZlibError::Truncated);
        }
        // FDICT: 4-byte DICTID followed by compressed data using preset dict.
        // We don't have a dict store, so reject — browsers don't send these.
        return Err(ZlibError::PresetDict);
    }
    if input.len() < pos + 4 {
        return Err(ZlibError::Truncated);
    }
    let payload = &input[pos..input.len() - 4];
    pos = input.len() - 4; // adler trailer at end
    let _adler = u32::from_be_bytes([input[pos], input[pos + 1], input[pos + 2], input[pos + 3]]);
    inflate(payload).map_err(ZlibError::Inflate)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-rolled gzip of "hello\n" produced by `gzip -n`. Includes only
    /// the standard header (no FEXTRA/FNAME/FCOMMENT/FHCRC).
    #[test]
    fn gzip_hello() {
        // Header: 1f 8b 08 00  00 00 00 00  00 03  (CM=deflate, FLG=0, MTIME=0, XFL=0, OS=3=Unix)
        // Raw DEFLATE for "hello\n":
        //   cb 48 cd c9 c9 e7 02 00
        // Trailer: CRC32 little-endian, ISIZE little-endian.
        // CRC32 of "hello\n" = 0x363a3020; ISIZE = 6.
        let mut buf = vec![
            0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xcb, 0x48, 0xcd, 0xc9,
            0xc9, 0xe7, 0x02, 0x00,
        ];
        buf.extend_from_slice(&0x363a3020u32.to_le_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes());
        let out = decode_gzip(&buf).unwrap();
        assert_eq!(out, b"hello\n");
    }

    #[test]
    fn gzip_bad_magic() {
        let buf = vec![0u8; 20];
        assert!(matches!(decode_gzip(&buf), Err(GzipError::BadMagic)));
    }

    #[test]
    fn gzip_truncated() {
        let buf = vec![0x1f, 0x8b, 0x08, 0x00];
        assert!(matches!(decode_gzip(&buf), Err(GzipError::Truncated)));
    }

    /// zlib wrapper around the same "hello\n" DEFLATE stream.
    #[test]
    fn zlib_hello() {
        // CMF=0x78 (deflate, 32K window), FLG=0x9C: (0x78*256+0x9C)=0x789C, %31==0.
        let mut buf = vec![0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0xe7, 0x02, 0x00];
        // ADLER-32 of "hello\n" = 0x08740217 (big-endian).
        buf.extend_from_slice(&0x08740217u32.to_be_bytes());
        let out = decode_zlib(&buf).unwrap();
        assert_eq!(out, b"hello\n");
    }

    #[test]
    fn zlib_bad_fcheck() {
        // Header that doesn't satisfy FCHECK.
        let buf = vec![0x78, 0x9d, 0x00, 0x00, 0x00, 0x00];
        assert!(matches!(decode_zlib(&buf), Err(ZlibError::BadHeader)));
    }
}
