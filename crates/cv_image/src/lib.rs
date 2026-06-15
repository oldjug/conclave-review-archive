//! `cv_image` — image decoders.
//!
//! Real pixel decode for: PNG (W3C PNG spec), JPEG (baseline), GIF (LZW),
//! BMP, ICO, SVG, and WebP — both VP8L lossless and VP8 lossy (the lossy
//! path is a full RFC 6386 intra-keyframe decoder in [`vp8_decode`]).
//! AVIF/HEIC containers are parsed for dimensions; their AV1/HEVC-intra
//! pixel decode is a documented follow-up (the host renders a sized
//! placeholder, never a fake "decoded" image).

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
pub mod vp8_decode;
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
pub use vp8_decode::decode_keyframe as decode_vp8_keyframe;
pub use webp::{WebPInfo, decode_webp, parse_webp_info};
