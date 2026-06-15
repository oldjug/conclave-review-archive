//! WebM / Matroska demuxer.
//!
//! Builds on [`crate::ebml`]'s VINT decoder to walk the master-element
//! tree (`Segment` → `Info` / `Tracks` / `Cluster`) and extract the
//! per-track codec metadata plus the coded video/audio frames carried
//! in `SimpleBlock` / `BlockGroup>Block` elements. This is the layer
//! the `<video>` / `<audio>` pipeline pulls samples from.
//!
//! References:
//!   * RFC 8794 — EBML specification (VINT, master elements).
//!   * Matroska element registry
//!     <https://www.matroska.org/technical/elements.html>
//!   * WebM container guidelines <https://www.webmproject.org/docs/container/>
//!
//! Scope: enough of Matroska to demux a real WebM file's track table
//! (codec id, video PixelWidth/PixelHeight, audio SamplingFrequency /
//! Channels), the segment duration (TimestampScale × Duration), and
//! the ordered list of frames per track with their presentation
//! timestamps. Lacing (Xiph/EBML/fixed) of multiple frames inside one
//! block is handled. Out of scope: SeekHead/Cues index, Tags,
//! Chapters — none are needed to decode the timeline front-to-back.

use crate::ebml::read_vint;

/// Matroska element IDs used by the demuxer. Values from the Matroska
/// element registry.
pub mod ids {
    pub const SEGMENT: u64 = 0x1853_8067;
    pub const INFO: u64 = 0x1549_A966;
    pub const TIMESTAMP_SCALE: u64 = 0x2AD7_B1;
    pub const DURATION: u64 = 0x4489;
    pub const TRACKS: u64 = 0x1654_AE6B;
    pub const TRACK_ENTRY: u64 = 0xAE;
    pub const TRACK_NUMBER: u64 = 0xD7;
    pub const TRACK_TYPE: u64 = 0x83;
    pub const CODEC_ID: u64 = 0x86;
    pub const VIDEO: u64 = 0xE0;
    pub const PIXEL_WIDTH: u64 = 0xB0;
    pub const PIXEL_HEIGHT: u64 = 0xBA;
    pub const AUDIO: u64 = 0xE1;
    pub const SAMPLING_FREQUENCY: u64 = 0xB5;
    pub const CHANNELS: u64 = 0x9F;
    pub const CLUSTER: u64 = 0x1F43_B675;
    pub const TIMESTAMP: u64 = 0xE7;
    pub const SIMPLE_BLOCK: u64 = 0xA3;
    pub const BLOCK_GROUP: u64 = 0xA0;
    pub const BLOCK: u64 = 0xA1;
}

/// Matroska TrackType enumeration (Matroska §TrackType).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    Video,
    Audio,
    Subtitle,
    Other(u8),
}

impl TrackType {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => TrackType::Video,
            2 => TrackType::Audio,
            0x11 => TrackType::Subtitle,
            other => TrackType::Other(other),
        }
    }
}

/// One track from the `Tracks` element.
#[derive(Debug, Clone, Default)]
pub struct Track {
    pub number: u64,
    pub track_type: Option<TrackType>,
    /// e.g. `V_VP9`, `V_VP8`, `V_MPEG4/ISO/AVC`, `A_OPUS`, `A_VORBIS`, `A_FLAC`.
    pub codec_id: String,
    pub pixel_width: u32,
    pub pixel_height: u32,
    pub sampling_frequency: f64,
    pub channels: u32,
}

/// One coded frame pulled from a `SimpleBlock` / `Block`.
#[derive(Debug, Clone)]
pub struct Frame {
    pub track_number: u64,
    /// Presentation timestamp in seconds.
    pub timestamp_s: f64,
    /// Keyframe flag (from the SimpleBlock flags; Blocks inside a
    /// BlockGroup are assumed non-keyframe unless flagged).
    pub keyframe: bool,
    /// The raw coded payload (one codec access unit).
    pub data: Vec<u8>,
}

/// Result of demuxing a WebM byte stream.
#[derive(Debug, Clone, Default)]
pub struct WebmDemux {
    pub timestamp_scale_ns: u64,
    pub duration_s: f64,
    pub tracks: Vec<Track>,
    pub frames: Vec<Frame>,
}

impl WebmDemux {
    pub fn first_video_track(&self) -> Option<&Track> {
        self.tracks
            .iter()
            .find(|t| t.track_type == Some(TrackType::Video))
    }

    pub fn first_audio_track(&self) -> Option<&Track> {
        self.tracks
            .iter()
            .find(|t| t.track_type == Some(TrackType::Audio))
    }

    /// All frames belonging to `track_number`, in stream order.
    pub fn frames_for(&self, track_number: u64) -> impl Iterator<Item = &Frame> {
        self.frames
            .iter()
            .filter(move |f| f.track_number == track_number)
    }
}

/// One parsed element header: id, payload byte range, and whether the
/// size was "unknown" (all-ones VINT, used for live/streamed Segments
/// and Clusters).
struct Header {
    id: u64,
    /// Offset of the payload start within the buffer.
    body_start: usize,
    /// Offset just past the payload (clamped to buffer end).
    body_end: usize,
    /// Offset just past this whole element (== body_end here).
    next: usize,
}

/// Read one element header at `pos`. Returns `None` if the buffer is
/// truncated. Handles the "unknown size" sentinel (all size bits set)
/// by treating the payload as extending to `buf.len()` — correct for
/// the last/streamed master element.
fn read_header(buf: &[u8], pos: usize) -> Option<Header> {
    let (id, id_len) = read_vint(&buf[pos..], true)?;
    let after_id = pos + id_len;
    if after_id > buf.len() {
        return None;
    }
    let (size, size_len) = read_vint(&buf[after_id..], false)?;
    let body_start = after_id + size_len;
    // Detect the "unknown size" VINT: all value bits set for the given
    // length. read_vint already masked the marker, so an unknown size
    // equals (1 << (7*size_len)) - 1.
    let unknown = size == (1u64 << (7 * size_len)) - 1;
    let body_end = if unknown {
        buf.len()
    } else {
        (body_start.saturating_add(size as usize)).min(buf.len())
    };
    Some(Header {
        id,
        body_start,
        body_end,
        next: body_end,
    })
}

/// Iterate the child elements directly inside `[start, end)`.
fn children(buf: &[u8], start: usize, end: usize) -> Vec<Header> {
    let mut out = Vec::new();
    let mut pos = start;
    while pos < end {
        let Some(h) = read_header(buf, pos) else { break };
        if h.body_start > end || h.next <= pos {
            break;
        }
        let next = h.next;
        out.push(h);
        pos = next;
    }
    out
}

/// Decode a big-endian unsigned integer from a payload (Matroska uint).
fn read_uint(buf: &[u8]) -> u64 {
    let mut v = 0u64;
    for &b in buf.iter().take(8) {
        v = (v << 8) | b as u64;
    }
    v
}

/// Decode a Matroska float payload (4 or 8 bytes, big-endian IEEE-754).
fn read_float(buf: &[u8]) -> f64 {
    match buf.len() {
        4 => f32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as f64,
        8 => f64::from_be_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]),
        _ => 0.0,
    }
}

/// Demux a complete WebM/Matroska byte stream.
pub fn demux(buf: &[u8]) -> WebmDemux {
    let mut out = WebmDemux {
        timestamp_scale_ns: 1_000_000, // Matroska default: 1 ms.
        ..Default::default()
    };

    // Walk top-level elements; we care about Segment. (The EBML
    // header precedes it and is skipped.)
    for top in children(buf, 0, buf.len()) {
        if top.id == ids::SEGMENT {
            parse_segment(buf, top.body_start, top.body_end, &mut out);
        }
    }

    // Resolve duration: Duration is in TimestampScale units.
    if out.duration_s == 0.0 {
        // Fall back to the last frame timestamp if Info had no Duration.
        out.duration_s = out
            .frames
            .iter()
            .map(|f| f.timestamp_s)
            .fold(0.0, f64::max);
    }
    out
}

fn parse_segment(buf: &[u8], start: usize, end: usize, out: &mut WebmDemux) {
    let mut raw_duration: f64 = 0.0;
    for el in children(buf, start, end) {
        match el.id {
            ids::INFO => {
                for info in children(buf, el.body_start, el.body_end) {
                    match info.id {
                        ids::TIMESTAMP_SCALE => {
                            out.timestamp_scale_ns =
                                read_uint(&buf[info.body_start..info.body_end]).max(1);
                        }
                        ids::DURATION => {
                            raw_duration = read_float(&buf[info.body_start..info.body_end]);
                        }
                        _ => {}
                    }
                }
            }
            ids::TRACKS => {
                for te in children(buf, el.body_start, el.body_end) {
                    if te.id == ids::TRACK_ENTRY {
                        out.tracks
                            .push(parse_track_entry(buf, te.body_start, te.body_end));
                    }
                }
            }
            ids::CLUSTER => {
                parse_cluster(buf, el.body_start, el.body_end, out);
            }
            _ => {}
        }
    }
    if raw_duration > 0.0 {
        out.duration_s = raw_duration * (out.timestamp_scale_ns as f64) / 1.0e9;
    }
}

fn parse_track_entry(buf: &[u8], start: usize, end: usize) -> Track {
    let mut t = Track::default();
    for el in children(buf, start, end) {
        match el.id {
            ids::TRACK_NUMBER => t.number = read_uint(&buf[el.body_start..el.body_end]),
            ids::TRACK_TYPE => {
                t.track_type = Some(TrackType::from_u8(
                    read_uint(&buf[el.body_start..el.body_end]) as u8,
                ));
            }
            ids::CODEC_ID => {
                t.codec_id = String::from_utf8_lossy(&buf[el.body_start..el.body_end])
                    .trim_end_matches('\0')
                    .to_string();
            }
            ids::VIDEO => {
                for v in children(buf, el.body_start, el.body_end) {
                    match v.id {
                        ids::PIXEL_WIDTH => {
                            t.pixel_width = read_uint(&buf[v.body_start..v.body_end]) as u32;
                        }
                        ids::PIXEL_HEIGHT => {
                            t.pixel_height = read_uint(&buf[v.body_start..v.body_end]) as u32;
                        }
                        _ => {}
                    }
                }
            }
            ids::AUDIO => {
                for a in children(buf, el.body_start, el.body_end) {
                    match a.id {
                        ids::SAMPLING_FREQUENCY => {
                            t.sampling_frequency = read_float(&buf[a.body_start..a.body_end]);
                        }
                        ids::CHANNELS => {
                            t.channels = read_uint(&buf[a.body_start..a.body_end]) as u32;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    t
}

fn parse_cluster(buf: &[u8], start: usize, end: usize, out: &mut WebmDemux) {
    let mut cluster_ts: i64 = 0;
    for el in children(buf, start, end) {
        match el.id {
            ids::TIMESTAMP => {
                cluster_ts = read_uint(&buf[el.body_start..el.body_end]) as i64;
            }
            ids::SIMPLE_BLOCK => {
                parse_block(
                    buf,
                    el.body_start,
                    el.body_end,
                    cluster_ts,
                    /*from_simple_block=*/ true,
                    out,
                );
            }
            ids::BLOCK_GROUP => {
                for bg in children(buf, el.body_start, el.body_end) {
                    if bg.id == ids::BLOCK {
                        parse_block(buf, bg.body_start, bg.body_end, cluster_ts, false, out);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Parse a (Simple)Block payload: track-number VINT, signed 16-bit
/// relative timecode, flags byte (keyframe + lacing), then the lacing
/// frame splits. Matroska §Block Structure.
fn parse_block(
    buf: &[u8],
    start: usize,
    end: usize,
    cluster_ts: i64,
    from_simple_block: bool,
    out: &mut WebmDemux,
) {
    if start >= end {
        return;
    }
    let block = &buf[start..end];
    // Track number VINT (size field encoding, marker stripped).
    let Some((track_number, tn_len)) = read_vint(block, false) else {
        return;
    };
    if tn_len + 3 > block.len() {
        return;
    }
    // Signed 16-bit relative timecode (big-endian).
    let rel = i16::from_be_bytes([block[tn_len], block[tn_len + 1]]) as i64;
    let flags = block[tn_len + 2];
    let keyframe = from_simple_block && (flags & 0x80) != 0;
    let lacing = (flags >> 1) & 0x03;
    let mut p = tn_len + 3;

    // Resolve presentation timestamp.
    let ts_units = cluster_ts + rel;
    let timestamp_s = (ts_units as f64) * (out.timestamp_scale_ns as f64) / 1.0e9;

    // Determine the frame byte-ranges per the lacing mode.
    let payload = &block[p..];
    let frames: Vec<&[u8]> = match lacing {
        0 => vec![payload], // no lacing — single frame
        2 => {
            // Fixed-size lacing: 1 byte = frame_count-1, equal sizes.
            if payload.is_empty() {
                return;
            }
            let count = payload[0] as usize + 1;
            let body = &payload[1..];
            if count == 0 || body.len() % count != 0 {
                vec![body]
            } else {
                let sz = body.len() / count;
                (0..count).map(|i| &body[i * sz..(i + 1) * sz]).collect()
            }
        }
        1 => xiph_lacing(payload), // Xiph lacing
        3 => ebml_lacing(payload), // EBML lacing
        _ => vec![payload],
    };
    let _ = &mut p;

    for data in frames {
        out.frames.push(Frame {
            track_number,
            timestamp_s,
            keyframe,
            data: data.to_vec(),
        });
    }
}

/// Xiph lacing: 1 byte count-1, then for each of the first N-1 frames a
/// run of 0xFF.. bytes summed as the size; the last frame takes the
/// remainder. Matroska §Lacing.
fn xiph_lacing(payload: &[u8]) -> Vec<&[u8]> {
    if payload.is_empty() {
        return vec![payload];
    }
    let count = payload[0] as usize + 1;
    let mut sizes = Vec::with_capacity(count);
    let mut i = 1;
    for _ in 0..count.saturating_sub(1) {
        let mut size = 0usize;
        loop {
            if i >= payload.len() {
                return vec![payload];
            }
            let b = payload[i];
            i += 1;
            size += b as usize;
            if b != 0xFF {
                break;
            }
        }
        sizes.push(size);
    }
    let mut out = Vec::with_capacity(count);
    let mut pos = i;
    for &s in &sizes {
        if pos + s > payload.len() {
            return vec![payload];
        }
        out.push(&payload[pos..pos + s]);
        pos += s;
    }
    out.push(&payload[pos..]); // last frame = remainder
    out
}

/// EBML lacing: 1 byte count-1, first size is an unsigned VINT, each
/// subsequent size is a *signed* VINT delta from the previous; the last
/// frame is the remainder. Matroska §Lacing.
fn ebml_lacing(payload: &[u8]) -> Vec<&[u8]> {
    if payload.is_empty() {
        return vec![payload];
    }
    let count = payload[0] as usize + 1;
    let mut i = 1;
    let Some((first, n)) = read_vint(&payload[i..], false) else {
        return vec![payload];
    };
    i += n;
    let mut sizes = vec![first as i64];
    for _ in 0..count.saturating_sub(2) {
        let Some((raw, n)) = read_vint(&payload[i..], false) else {
            return vec![payload];
        };
        // Signed VINT: subtract bias 2^(7*n - 1) - 1.
        let bias = (1i64 << (7 * n - 1)) - 1;
        let delta = raw as i64 - bias;
        let prev = *sizes.last().unwrap();
        sizes.push(prev + delta);
        i += n;
    }
    let mut out = Vec::with_capacity(count);
    let mut pos = i;
    for &s in &sizes {
        if s < 0 || pos + s as usize > payload.len() {
            return vec![payload];
        }
        out.push(&payload[pos..pos + s as usize]);
        pos += s as usize;
    }
    out.push(&payload[pos..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode an element ID (already a canonical multi-byte ID) + a
    /// 1-byte size prefix VINT + payload. IDs here are <= 4 bytes and
    /// emitted verbatim; size uses the 1-byte form (payload < 127).
    fn elem(id: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // Emit ID big-endian using its natural byte length.
        let id_bytes = id_to_bytes(id);
        out.extend_from_slice(&id_bytes);
        // Size: use a VINT. For payload <= 0x7E use 1-byte (0x80 | len).
        let len = payload.len();
        if len <= 0x7E {
            out.push(0x80 | len as u8);
        } else {
            // 2-byte VINT.
            out.push(0x40 | ((len >> 8) as u8));
            out.push((len & 0xFF) as u8);
        }
        out.extend_from_slice(payload);
        out
    }

    fn id_to_bytes(id: u64) -> Vec<u8> {
        // Emit the minimal big-endian bytes (the canonical Matroska ID
        // already carries its length marker bit).
        let mut bytes = id.to_be_bytes().to_vec();
        while bytes.len() > 1 && bytes[0] == 0 {
            bytes.remove(0);
        }
        bytes
    }

    #[test]
    fn vint_roundtrip_id_bytes() {
        assert_eq!(id_to_bytes(0xAE), vec![0xAE]);
        assert_eq!(id_to_bytes(ids::SEGMENT), vec![0x18, 0x53, 0x80, 0x67]);
    }

    #[test]
    fn demux_extracts_track_table_and_geometry() {
        // Build Video track entry: number=1, type=1, codec V_VP9, 320x240.
        let video = {
            let mut v = Vec::new();
            v.extend(elem(ids::TRACK_NUMBER, &[0x01]));
            v.extend(elem(ids::TRACK_TYPE, &[0x01]));
            v.extend(elem(ids::CODEC_ID, b"V_VP9"));
            let mut vsub = Vec::new();
            vsub.extend(elem(ids::PIXEL_WIDTH, &[0x01, 0x40])); // 320
            vsub.extend(elem(ids::PIXEL_HEIGHT, &[0xF0])); // 240
            v.extend(elem(ids::VIDEO, &vsub));
            v
        };
        let track_entry = elem(ids::TRACK_ENTRY, &video);
        let tracks = elem(ids::TRACKS, &track_entry);

        // Info: TimestampScale = 1_000_000 (1ms), Duration = 2000.0 ⇒ 2s.
        let info = {
            let mut i = Vec::new();
            i.extend(elem(ids::TIMESTAMP_SCALE, &1_000_000u32.to_be_bytes()));
            i.extend(elem(ids::DURATION, &2000.0f64.to_be_bytes()));
            i
        };
        let info_el = elem(ids::INFO, &info);

        let mut seg_body = Vec::new();
        seg_body.extend(info_el);
        seg_body.extend(tracks);
        let segment = elem(ids::SEGMENT, &seg_body);

        let dem = demux(&segment);
        assert_eq!(dem.tracks.len(), 1);
        let t = dem.first_video_track().expect("video track");
        assert_eq!(t.number, 1);
        assert_eq!(t.codec_id, "V_VP9");
        assert_eq!(t.pixel_width, 320);
        assert_eq!(t.pixel_height, 240);
        assert!((dem.duration_s - 2.0).abs() < 1e-9, "dur={}", dem.duration_s);
    }

    #[test]
    fn demux_extracts_simpleblock_frames_with_timestamps() {
        // Cluster with timestamp 0, two SimpleBlocks on track 1.
        // SimpleBlock payload: track VINT(0x81), rel timecode (i16 BE),
        // flags byte, then the coded data.
        let make_block = |rel: i16, key: bool, data: &[u8]| -> Vec<u8> {
            let mut b = Vec::new();
            b.push(0x81); // track number 1, 1-byte VINT
            b.extend_from_slice(&rel.to_be_bytes());
            b.push(if key { 0x80 } else { 0x00 }); // keyframe flag, no lacing
            b.extend_from_slice(data);
            b
        };
        let mut cluster = Vec::new();
        cluster.extend(elem(ids::TIMESTAMP, &[0x00]));
        cluster.extend(elem(
            ids::SIMPLE_BLOCK,
            &make_block(0, true, &[0xDE, 0xAD]),
        ));
        cluster.extend(elem(
            ids::SIMPLE_BLOCK,
            &make_block(33, false, &[0xBE, 0xEF, 0x00]),
        ));
        let cluster_el = elem(ids::CLUSTER, &cluster);
        let segment = elem(ids::SEGMENT, &cluster_el);

        let dem = demux(&segment);
        assert_eq!(dem.frames.len(), 2);
        assert_eq!(dem.frames[0].track_number, 1);
        assert!(dem.frames[0].keyframe);
        assert_eq!(dem.frames[0].data, vec![0xDE, 0xAD]);
        assert!((dem.frames[0].timestamp_s - 0.0).abs() < 1e-9);
        // 33 ms cluster-relative @ 1ms scale ⇒ 0.033s.
        assert!((dem.frames[1].timestamp_s - 0.033).abs() < 1e-9);
        assert!(!dem.frames[1].keyframe);
        assert_eq!(dem.frames[1].data, vec![0xBE, 0xEF, 0x00]);
        // Duration falls back to last frame ts when Info absent.
        assert!((dem.duration_s - 0.033).abs() < 1e-9);
    }

    #[test]
    fn fixed_lacing_splits_equal_frames() {
        // SimpleBlock, fixed lacing (lacing bits = 10 ⇒ flags 0x04),
        // count-1 = 2 (3 frames), 6 bytes body ⇒ 2 each.
        let mut b = Vec::new();
        b.push(0x81); // track 1
        b.extend_from_slice(&0i16.to_be_bytes());
        b.push(0x04); // fixed lacing, no keyframe
        b.push(0x02); // frame count - 1 = 2
        b.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let cluster = {
            let mut c = Vec::new();
            c.extend(elem(ids::TIMESTAMP, &[0x00]));
            c.extend(elem(ids::SIMPLE_BLOCK, &b));
            c
        };
        let segment = elem(ids::SEGMENT, &elem(ids::CLUSTER, &cluster));
        let dem = demux(&segment);
        assert_eq!(dem.frames.len(), 3);
        assert_eq!(dem.frames[0].data, vec![1, 2]);
        assert_eq!(dem.frames[1].data, vec![3, 4]);
        assert_eq!(dem.frames[2].data, vec![5, 6]);
    }

    #[test]
    fn blockgroup_block_is_demuxed() {
        let mut blk = Vec::new();
        blk.push(0x81);
        blk.extend_from_slice(&0i16.to_be_bytes());
        blk.push(0x00);
        blk.extend_from_slice(&[0xAA]);
        let bg = elem(ids::BLOCK, &blk);
        let cluster = {
            let mut c = Vec::new();
            c.extend(elem(ids::TIMESTAMP, &[0x00]));
            c.extend(elem(ids::BLOCK_GROUP, &bg));
            c
        };
        let segment = elem(ids::SEGMENT, &elem(ids::CLUSTER, &cluster));
        let dem = demux(&segment);
        assert_eq!(dem.frames.len(), 1);
        assert_eq!(dem.frames[0].data, vec![0xAA]);
        assert!(!dem.frames[0].keyframe); // Block (not SimpleBlock) ⇒ not flagged
    }
}
