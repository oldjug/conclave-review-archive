//! HLS (HTTP Live Streaming, RFC 8216) M3U8 playlist parser.
//!
//! Supports the master playlist (variant streams with codec/bandwidth
//! attributes) and the media playlist (ordered list of segment URIs
//! with target duration). Discontinuity / encryption / byte-range
//! tags are recognised; full TS demux happens elsewhere.

#[derive(Debug, Clone, Default)]
pub struct MasterPlaylist {
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub bandwidth: u32,
    pub codecs: String,
    pub resolution: Option<(u32, u32)>,
    pub uri: String,
}

#[derive(Debug, Clone, Default)]
pub struct MediaPlaylist {
    pub target_duration_s: f32,
    pub media_sequence: u64,
    pub end_list: bool,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub duration_s: f32,
    pub uri: String,
    pub byte_range: Option<(u64, u64)>,
    pub discontinuity: bool,
}

pub fn parse(text: &str) -> Result<Either<MasterPlaylist, MediaPlaylist>, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let header = lines.next().ok_or("empty playlist")?;
    if !header.starts_with("#EXTM3U") {
        return Err("missing #EXTM3U".into());
    }
    let body: Vec<&str> = lines.collect();
    if body.iter().any(|l| l.starts_with("#EXT-X-STREAM-INF")) {
        Ok(Either::Left(parse_master(&body)))
    } else {
        Ok(Either::Right(parse_media(&body)))
    }
}

#[derive(Debug, Clone)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

fn parse_master(body: &[&str]) -> MasterPlaylist {
    let mut out = MasterPlaylist::default();
    let mut pending: Option<(u32, String, Option<(u32, u32)>)> = None;
    for line in body {
        if let Some(rest) = line.strip_prefix("#EXT-X-STREAM-INF:") {
            let mut bandwidth = 0u32;
            let mut codecs = String::new();
            let mut res = None;
            for attr in split_attrs(rest) {
                let (k, v) = attr;
                match k.as_str() {
                    "BANDWIDTH" => bandwidth = v.parse().unwrap_or(0),
                    "CODECS" => codecs = v.trim_matches('"').to_string(),
                    "RESOLUTION" => {
                        if let Some((w, h)) = v.split_once('x') {
                            if let (Ok(w), Ok(h)) = (w.parse(), h.parse()) {
                                res = Some((w, h));
                            }
                        }
                    }
                    _ => {}
                }
            }
            pending = Some((bandwidth, codecs, res));
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        if let Some((bw, codecs, res)) = pending.take() {
            out.variants.push(Variant {
                bandwidth: bw,
                codecs,
                resolution: res,
                uri: line.to_string(),
            });
        }
    }
    out
}

fn parse_media(body: &[&str]) -> MediaPlaylist {
    let mut out = MediaPlaylist::default();
    let mut pending_dur: Option<f32> = None;
    let mut pending_range: Option<(u64, u64)> = None;
    let mut pending_disc = false;
    for line in body {
        if let Some(rest) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
            out.target_duration_s = rest.trim().parse().unwrap_or(0.0);
            continue;
        }
        if let Some(rest) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
            out.media_sequence = rest.trim().parse().unwrap_or(0);
            continue;
        }
        if line.eq_ignore_ascii_case("#EXT-X-ENDLIST") {
            out.end_list = true;
            continue;
        }
        if line.eq_ignore_ascii_case("#EXT-X-DISCONTINUITY") {
            pending_disc = true;
            continue;
        }
        if let Some(rest) = line.strip_prefix("#EXTINF:") {
            let dur = rest.split(',').next().unwrap_or("0").trim();
            pending_dur = dur.parse().ok();
            continue;
        }
        if let Some(rest) = line.strip_prefix("#EXT-X-BYTERANGE:") {
            if let Some((len, off)) = rest.split_once('@') {
                if let (Ok(l), Ok(o)) = (len.trim().parse(), off.trim().parse()) {
                    pending_range = Some((l, o));
                }
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        if let Some(d) = pending_dur.take() {
            out.segments.push(Segment {
                duration_s: d,
                uri: line.to_string(),
                byte_range: pending_range.take(),
                discontinuity: pending_disc,
            });
            pending_disc = false;
        }
    }
    out
}

fn split_attrs(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        let kstart = i;
        while i < bytes.len() && bytes[i] != b'=' {
            i += 1;
        }
        let key = String::from_utf8_lossy(&bytes[kstart..i]).into_owned();
        if i >= bytes.len() {
            break;
        }
        i += 1; // skip '='
        let vstart = i;
        if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i < bytes.len() {
                i += 1;
            }
        } else {
            while i < bytes.len() && bytes[i] != b',' {
                i += 1;
            }
        }
        let val = String::from_utf8_lossy(&bytes[vstart..i]).into_owned();
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        }
        out.push((key.trim().to_string(), val.trim().to_string()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_master_with_one_variant() {
        let text = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1280000,CODECS=\"avc1.42e00a\",RESOLUTION=640x360\nlow.m3u8";
        match parse(text).unwrap() {
            Either::Left(m) => {
                assert_eq!(m.variants.len(), 1);
                assert_eq!(m.variants[0].bandwidth, 1_280_000);
                assert_eq!(m.variants[0].resolution, Some((640, 360)));
                assert_eq!(m.variants[0].uri, "low.m3u8");
            }
            _ => panic!("expected master"),
        }
    }

    #[test]
    fn parses_media_segments() {
        let text = "#EXTM3U\n#EXT-X-TARGETDURATION:6\n#EXTINF:5.0,\nseg1.ts\n#EXTINF:5.0,\nseg2.ts\n#EXT-X-ENDLIST";
        match parse(text).unwrap() {
            Either::Right(m) => {
                assert_eq!(m.segments.len(), 2);
                assert!(m.end_list);
            }
            _ => panic!("expected media"),
        }
    }
}
