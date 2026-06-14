//! `cv_image` — image decoders.
//!
//! Today: PNG decoder per the W3C PNG Specification. Supports 8-bit color
//! types 0/2/3/4/6 and all five filter modes. Non-interlaced only.
//! JPEG/WebP/AVIF/GIF land in M2+.

#![allow(missing_debug_implementations, unused_assignments)]

pub mod anim_gif;
pub mod avif;
pub mod bmp;
pub mod gif;
pub mod heif;
pub mod ico;
pub mod jpeg;
pub mod jpeg_xl;
pub mod png;
pub mod svg;
pub mod svg_extra;
pub mod tiff;
pub mod vp8;
pub mod webp;

pub use avif::{AvifInfo, parse_avif_info};
pub use bmp::decode_bmp;
pub use gif::decode_gif;
pub use ico::decode_ico;
pub use jpeg::decode_jpeg;
pub use png::{ImageError, RgbaImage, decode_png};
pub use svg::{SvgError, rasterize_svg_attrs};
pub use vp8::{
    decode_i_frame_pixels as decode_vp8_i_frame, parse_frame_header as parse_vp8_header,
};
pub use webp::{WebPInfo, decode_webp, parse_webp_info};
