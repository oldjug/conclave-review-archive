//! `cv_compression` — compression codecs.
//!
//! Today: DEFLATE decompression per RFC 1951 (no compression). This
//! unlocks PNG image decoding (`cv_image`), HTTP `gzip` content encoding,
//! and zip / `.crx` reading once `cv_extensions` lands.
//!
//! Reference: RFC 1951, "DEFLATE Compressed Data Format Specification v1.3".

#![allow(dead_code)]

pub mod brotli;
pub mod deflate;
pub mod gzip;

pub use brotli::{BrotliError, decode_brotli};
pub use deflate::{InflateError, inflate};
pub use gzip::{GzipError, ZlibError, decode_gzip, decode_zlib};
