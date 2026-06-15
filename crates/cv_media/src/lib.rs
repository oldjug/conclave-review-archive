//! `cv_media` — demux + codec front-end.
//!
//! V1 scope: enough of the H.264 / AVC1 byte stream to parse NAL
//! units, decode SPS (Sequence Parameter Set) headers, and surface
//! the picture geometry needed by the rest of the pipeline. The
//! decoder loop itself (slice-level intra prediction, dequant, IDCT,
//! deblock) is staged behind this front-end.

#![allow(missing_debug_implementations)]

pub mod av1;
pub mod capabilities;
pub mod color;
pub mod dash;
pub mod ebml;
pub mod flac;
pub mod h264;
pub mod h264_cavlc;
pub mod h264_chroma_intra;
pub mod h264_deblock;
pub mod h264_dpb;
pub mod h264_driver;
pub mod h264_idct;
pub mod h264_intra;
pub mod h264_intra16;
pub mod h264_mb_loop;
pub mod h264_mc;
pub mod h264_slice;
pub mod h265;
pub mod hls;
pub mod media_element;
pub mod mp4;
pub mod mse;
pub mod pipeline;
pub mod vorbis;
pub mod vp9;
pub mod webm;
pub mod webvtt;
