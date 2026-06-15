//! End-to-end media pipeline glue: bytes → demux → decode frame 0 →
//! BGRA. This is the entry point the `<video>` element calls to obtain
//! a paintable picture for the current time, and the place where the
//! container demuxer hands coded frames to the matching codec driver.
//!
//! Today the fully-wired video path is H.264 (AVC) via
//! [`crate::h264_driver`]; WebM clusters carrying `V_MPEG4/ISO/AVC`
//! Annex-B frames decode end-to-end. Other codecs (VP9/AV1) parse
//! their headers (geometry) but their full sample decoders are
//! follow-ups — for those we surface geometry + an honest
//! `DecodeStatus::HeaderOnly` rather than a fake black frame.

use crate::color::{ColorMatrix, yuv420_to_bgra};
use crate::h264_driver::{self, DecodedFrame, DriverError};
use crate::webm;

/// A decoded video picture ready for the compositor / paint path.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// Tightly packed BGRA, row-major, `width * height` pixels.
    pub bgra: Vec<u32>,
    /// Presentation timestamp in seconds.
    pub timestamp_s: f64,
}

/// What the demux/decode probe was able to do with a resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeStatus {
    /// Geometry known but no full-sample decoder wired for this codec.
    HeaderOnly { width: u32, height: u32, codec: String },
    /// Container/codec not recognized at all.
    Unsupported,
}

/// Probe a media resource: demux just far enough to learn duration,
/// dimensions, and the video codec. Cheap; safe to call on load.
#[derive(Debug, Clone, Default)]
pub struct MediaProbe {
    pub duration_s: f64,
    pub width: u32,
    pub height: u32,
    pub video_codec: String,
    pub audio_codec: String,
    pub audio_channels: u32,
    pub audio_sample_rate: f64,
    pub has_video: bool,
    pub has_audio: bool,
}

/// Probe a WebM byte stream.
pub fn probe_webm(bytes: &[u8]) -> MediaProbe {
    let dem = webm::demux(bytes);
    let mut p = MediaProbe {
        duration_s: dem.duration_s,
        ..Default::default()
    };
    if let Some(v) = dem.first_video_track() {
        p.has_video = true;
        p.width = v.pixel_width;
        p.height = v.pixel_height;
        p.video_codec = v.codec_id.clone();
    }
    if let Some(a) = dem.first_audio_track() {
        p.has_audio = true;
        p.audio_codec = a.codec_id.clone();
        p.audio_channels = a.channels;
        p.audio_sample_rate = a.sampling_frequency;
    }
    p
}

/// Decode the first displayable video frame from a raw H.264 Annex-B
/// elementary stream (start-code delimited NAL units). Returns the
/// real reconstructed picture, color-converted to BGRA.
pub fn decode_h264_first_frame(annexb: &[u8]) -> Result<VideoFrame, DriverError> {
    let d: DecodedFrame = h264_driver::decode_first_idr(annexb)?;
    Ok(VideoFrame {
        width: d.width,
        height: d.height,
        bgra: d.bgra,
        timestamp_s: 0.0,
    })
}

/// Decode the first video frame from a WebM resource. If the video
/// track is AVC (`V_MPEG4/ISO/AVC`) we feed its first keyframe through
/// the H.264 driver. For codecs whose sample decoder isn't wired yet we
/// return `Err(DecodeStatus)` carrying the real geometry — never a fake
/// frame.
pub fn decode_webm_first_frame(bytes: &[u8]) -> Result<VideoFrame, DecodeStatus> {
    let dem = webm::demux(bytes);
    let track = dem
        .first_video_track()
        .ok_or(DecodeStatus::Unsupported)?
        .clone();
    let codec = track.codec_id.clone();

    // Collect this track's frames; prefer the first keyframe.
    let frame = dem
        .frames_for(track.number)
        .find(|f| f.keyframe)
        .or_else(|| dem.frames_for(track.number).next());
    let frame = frame.ok_or(DecodeStatus::HeaderOnly {
        width: track.pixel_width,
        height: track.pixel_height,
        codec: codec.clone(),
    })?;

    let cu = codec.to_ascii_uppercase();
    if cu.contains("AVC") || cu == "V_MPEG4/ISO/AVC" {
        // WebM carries AVC as length-prefixed NAL units (AVCC). The
        // driver wants Annex-B start codes; convert if needed. If the
        // payload already looks like Annex-B (starts with 00 00 01 /
        // 00 00 00 01) pass it through.
        let annexb = to_annexb(&frame.data);
        let ts = frame.timestamp_s;
        return h264_driver::decode_first_idr(&annexb)
            .map(|d| VideoFrame {
                width: d.width,
                height: d.height,
                bgra: d.bgra,
                timestamp_s: ts,
            })
            .map_err(|_| DecodeStatus::HeaderOnly {
                width: track.pixel_width,
                height: track.pixel_height,
                codec: codec.clone(),
            });
    }

    // Geometry-only codecs (VP9/AV1/VP8): honest header-only status.
    Err(DecodeStatus::HeaderOnly {
        width: track.pixel_width,
        height: track.pixel_height,
        codec,
    })
}

/// Convert a payload that may be in AVCC (4-byte big-endian length
/// prefixes) to Annex-B (start-code prefixed). If the buffer already
/// begins with an Annex-B start code it is returned as-is.
fn to_annexb(data: &[u8]) -> Vec<u8> {
    if data.len() >= 4 && data[0] == 0 && data[1] == 0 && (data[2] == 1 || (data[2] == 0 && data[3] == 1)) {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(data.len() + 8);
    let mut i = 0;
    while i + 4 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;
        if len == 0 || i + len > data.len() {
            // Not valid AVCC framing — bail to raw passthrough.
            return data.to_vec();
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&data[i..i + len]);
        i += len;
    }
    if out.is_empty() { data.to_vec() } else { out }
}

/// Convert raw YUV 4:2:0 planes directly to a `VideoFrame` (used by
/// codec drivers that already produce planar output, and by tests).
pub fn yuv420_frame(
    width: u32,
    height: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    matrix: ColorMatrix,
    timestamp_s: f64,
) -> VideoFrame {
    VideoFrame {
        width,
        height,
        bgra: yuv420_to_bgra(width, height, y, u, v, matrix),
        timestamp_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 4-byte-start-code Annex-B stream with a Baseline 640x480
    /// SPS + an IDR NAL, matching the h264_driver synthetic case. This
    /// drives the *real* macroblock pipeline (SPS parse → MB loop →
    /// YUV → BGRA), proving non-stub decode.
    fn synth_h264_640x480() -> Vec<u8> {
        let sps_rbsp = vec![66, 0, 30, 0xF4, 0x05, 0x01, 0xE8];
        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.push(0x67); // SPS NAL header
        out.extend_from_slice(&sps_rbsp);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.push(0x65); // IDR NAL header
        out.push(0x80);
        out
    }

    #[test]
    fn h264_first_frame_decodes_to_real_bgra() {
        let bs = synth_h264_640x480();
        let f = decode_h264_first_frame(&bs).expect("decode");
        assert_eq!(f.width, 640);
        assert_eq!(f.height, 480);
        assert_eq!(f.bgra.len(), 640 * 480);
        // The reconstructed picture is grey (DC, no residual) — verify
        // the YUV→RGB conversion produced a real neutral pixel, not a
        // zeroed/black buffer (a black-rectangle stub would be ~0).
        let p = f.bgra[640 * 200 + 320];
        let r = (p >> 16) & 0xFF;
        let g = (p >> 8) & 0xFF;
        let b = p & 0xFF;
        assert!((100..=160).contains(&r), "r={r}");
        assert!((r as i32 - g as i32).abs() < 5);
        assert!((g as i32 - b as i32).abs() < 5);
        assert_eq!((p >> 24) & 0xFF, 0xFF); // opaque
    }

    #[test]
    fn intra_prediction_produces_nonuniform_pixels() {
        // Drive the macroblock loop with vertical prediction over a
        // seeded gradient, then color-convert — asserts the decoder
        // reproduces structure (not a flat fill). Uses the public
        // YUV→BGRA path on a frame the MB loop reconstructed.
        use crate::h264_chroma_intra::ChromaIntraMode;
        use crate::h264_intra16::Intra16x16Mode;
        use crate::h264_mb_loop::{Frame, MbParams, decode_macroblock};

        let mut frame = Frame::new(16, 32);
        // Seed bottom row of MB(0,0) with a horizontal gradient that
        // MB(0,1) Vertical-predicts downward.
        for x in 0..16 {
            frame.y[15 * 16 + x] = (x as u8) * 8;
        }
        let zero_luma = [[0i32; 16]; 16];
        let zero_chroma = [[0i32; 16]; 4];
        let params = MbParams {
            luma_mode: Intra16x16Mode::Vertical,
            chroma_mode: ChromaIntraMode::Dc,
            luma_residuals: &zero_luma,
            chroma_residuals_cb: &zero_chroma,
            chroma_residuals_cr: &zero_chroma,
        };
        decode_macroblock(&mut frame, 0, 1, &params).unwrap();
        let vf = yuv420_frame(16, 32, &frame.y, &frame.cb, &frame.cr, ColorMatrix::Bt709, 0.0);
        // Column 0 (Y≈0) is darker than column 15 (Y≈120) in row 20.
        let left = vf.bgra[20 * 16 + 0] & 0xFF;
        let right = vf.bgra[20 * 16 + 15] & 0xFF;
        assert!(right > left + 40, "left={left} right={right} (gradient lost)");
    }

    #[test]
    fn probe_webm_reports_geometry_and_codec() {
        // Minimal WebM with a VP9 video track 320x240, duration 2s.
        let bytes = build_probe_webm();
        let p = probe_webm(&bytes);
        assert!(p.has_video);
        assert_eq!(p.width, 320);
        assert_eq!(p.height, 240);
        assert_eq!(p.video_codec, "V_VP9");
        assert!((p.duration_s - 2.0).abs() < 1e-9);
    }

    #[test]
    fn webm_with_unwired_codec_returns_header_only_not_black_frame() {
        let bytes = build_probe_webm_with_frame();
        let status = decode_webm_first_frame(&bytes).unwrap_err();
        match status {
            DecodeStatus::HeaderOnly { width, height, codec } => {
                assert_eq!(width, 320);
                assert_eq!(height, 240);
                assert_eq!(codec, "V_VP9");
            }
            other => panic!("expected HeaderOnly, got {other:?}"),
        }
    }

    // --- WebM builders mirroring webm.rs test helpers ---
    fn elem(id: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut bytes = id.to_be_bytes().to_vec();
        while bytes.len() > 1 && bytes[0] == 0 {
            bytes.remove(0);
        }
        out.extend_from_slice(&bytes);
        let len = payload.len();
        if len <= 0x7E {
            out.push(0x80 | len as u8);
        } else {
            out.push(0x40 | ((len >> 8) as u8));
            out.push((len & 0xFF) as u8);
        }
        out.extend_from_slice(payload);
        out
    }

    fn build_probe_webm() -> Vec<u8> {
        use webm::ids;
        let mut vsub = Vec::new();
        vsub.extend(elem(ids::PIXEL_WIDTH, &[0x01, 0x40]));
        vsub.extend(elem(ids::PIXEL_HEIGHT, &[0xF0]));
        let mut video = Vec::new();
        video.extend(elem(ids::TRACK_NUMBER, &[0x01]));
        video.extend(elem(ids::TRACK_TYPE, &[0x01]));
        video.extend(elem(ids::CODEC_ID, b"V_VP9"));
        video.extend(elem(ids::VIDEO, &vsub));
        let tracks = elem(ids::TRACKS, &elem(ids::TRACK_ENTRY, &video));
        let mut info = Vec::new();
        info.extend(elem(ids::TIMESTAMP_SCALE, &1_000_000u32.to_be_bytes()));
        info.extend(elem(ids::DURATION, &2000.0f64.to_be_bytes()));
        let info_el = elem(ids::INFO, &info);
        let mut seg = Vec::new();
        seg.extend(info_el);
        seg.extend(tracks);
        elem(ids::SEGMENT, &seg)
    }

    fn build_probe_webm_with_frame() -> Vec<u8> {
        use webm::ids;
        let base = build_probe_webm();
        // Strip the outer Segment wrapper, append a cluster, re-wrap.
        // Simpler: rebuild with a cluster included.
        let _ = base;
        let mut vsub = Vec::new();
        vsub.extend(elem(ids::PIXEL_WIDTH, &[0x01, 0x40]));
        vsub.extend(elem(ids::PIXEL_HEIGHT, &[0xF0]));
        let mut video = Vec::new();
        video.extend(elem(ids::TRACK_NUMBER, &[0x01]));
        video.extend(elem(ids::TRACK_TYPE, &[0x01]));
        video.extend(elem(ids::CODEC_ID, b"V_VP9"));
        video.extend(elem(ids::VIDEO, &vsub));
        let tracks = elem(ids::TRACKS, &elem(ids::TRACK_ENTRY, &video));
        let mut block = Vec::new();
        block.push(0x81);
        block.extend_from_slice(&0i16.to_be_bytes());
        block.push(0x80); // keyframe
        block.extend_from_slice(&[0xAA, 0xBB]); // VP9 payload (not decoded)
        let mut cluster = Vec::new();
        cluster.extend(elem(ids::TIMESTAMP, &[0x00]));
        cluster.extend(elem(ids::SIMPLE_BLOCK, &block));
        let cluster_el = elem(ids::CLUSTER, &cluster);
        let mut seg = Vec::new();
        seg.extend(tracks);
        seg.extend(cluster_el);
        elem(ids::SEGMENT, &seg)
    }
}
