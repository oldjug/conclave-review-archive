//! DASH MPD (ISO/IEC 23009-1) — adaptive bitrate manifest parser.
//!
//! Very thin XML walker that surfaces the periods → adaptation sets →
//! representations tree. Caller resolves SegmentTemplate/SegmentList
//! URLs against the BaseURL and consults the resulting Segment list.

#[derive(Debug, Clone, Default)]
pub struct Mpd {
    pub periods: Vec<Period>,
    pub min_buffer_time: f32,
    pub media_presentation_duration_s: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Period {
    pub id: String,
    pub start_s: f32,
    pub adaptation_sets: Vec<AdaptationSet>,
}

#[derive(Debug, Clone, Default)]
pub struct AdaptationSet {
    pub mime_type: String,
    pub representations: Vec<Representation>,
}

#[derive(Debug, Clone, Default)]
pub struct Representation {
    pub id: String,
    pub bandwidth: u32,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub codecs: String,
    pub base_url: String,
}

/// Parse a DASH MPD. Returns an Mpd; missing/garbage tags collapse to
/// empty defaults so callers can still introspect the periods they
/// got.
pub fn parse(xml: &str) -> Mpd {
    let mut mpd = Mpd::default();
    let mut cur_period: Option<Period> = None;
    let mut cur_as: Option<AdaptationSet> = None;
    for tag in iter_tags(xml) {
        let (name, attrs, is_close, is_self_close) = tag;
        let lc = name.to_ascii_lowercase();
        match (lc.as_str(), is_close) {
            ("mpd", false) => {
                if let Some(v) = attr(&attrs, "minBufferTime").and_then(parse_iso_duration) {
                    mpd.min_buffer_time = v;
                }
                if let Some(v) =
                    attr(&attrs, "mediaPresentationDuration").and_then(parse_iso_duration)
                {
                    mpd.media_presentation_duration_s = v;
                }
            }
            ("period", false) => {
                cur_period = Some(Period {
                    id: attr(&attrs, "id").unwrap_or_default(),
                    start_s: attr(&attrs, "start")
                        .and_then(parse_iso_duration)
                        .unwrap_or(0.0),
                    adaptation_sets: Vec::new(),
                });
            }
            ("period", true) => {
                if let Some(p) = cur_period.take() {
                    mpd.periods.push(p);
                }
            }
            ("adaptationset", false) => {
                cur_as = Some(AdaptationSet {
                    mime_type: attr(&attrs, "mimeType").unwrap_or_default(),
                    representations: Vec::new(),
                });
            }
            ("adaptationset", true) => {
                if let (Some(p), Some(a)) = (cur_period.as_mut(), cur_as.take()) {
                    p.adaptation_sets.push(a);
                }
            }
            ("representation", _) => {
                let rep = Representation {
                    id: attr(&attrs, "id").unwrap_or_default(),
                    bandwidth: attr(&attrs, "bandwidth")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0),
                    width: attr(&attrs, "width").and_then(|s| s.parse().ok()),
                    height: attr(&attrs, "height").and_then(|s| s.parse().ok()),
                    codecs: attr(&attrs, "codecs").unwrap_or_default(),
                    base_url: String::new(),
                };
                if let Some(a) = cur_as.as_mut() {
                    a.representations.push(rep);
                }
                let _ = is_self_close; // consumed
            }
            _ => {}
        }
    }
    mpd
}

fn attr(attrs: &[(String, String)], name: &str) -> Option<String> {
    attrs
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

fn parse_iso_duration(s: String) -> Option<f32> {
    // Very small subset: PT123.4S
    let s = s.strip_prefix("PT")?;
    if let Some(rest) = s.strip_suffix('S') {
        return rest.parse().ok();
    }
    None
}

fn iter_tags(xml: &str) -> Vec<(String, Vec<(String, String)>, bool, bool)> {
    let mut out = Vec::new();
    let bytes = xml.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'>' {
                j += 1;
            }
            if j >= bytes.len() {
                break;
            }
            let inner = &xml[i + 1..j];
            let is_close = inner.starts_with('/');
            let is_self_close = inner.ends_with('/');
            let inner = inner.trim_start_matches('/').trim_end_matches('/').trim();
            let mut parts = inner.split_ascii_whitespace();
            let name = parts.next().unwrap_or("").to_string();
            let mut attrs = Vec::new();
            // Crude attr scan: name="value"
            let rest = inner.strip_prefix(&name).unwrap_or("").trim();
            let mut k = 0;
            let rb = rest.as_bytes();
            while k < rb.len() {
                let kstart = k;
                while k < rb.len() && rb[k] != b'=' {
                    k += 1;
                }
                let key = rest[kstart..k].trim().to_string();
                if k >= rb.len() {
                    break;
                }
                k += 1;
                if k >= rb.len() || rb[k] != b'"' {
                    break;
                }
                k += 1;
                let vstart = k;
                while k < rb.len() && rb[k] != b'"' {
                    k += 1;
                }
                let val = rest[vstart..k].to_string();
                if k < rb.len() {
                    k += 1;
                }
                if !key.is_empty() {
                    attrs.push((key, val));
                }
                while k < rb.len() && rb[k].is_ascii_whitespace() {
                    k += 1;
                }
            }
            out.push((name, attrs, is_close, is_self_close));
            i = j + 1;
            continue;
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_mpd() {
        let xml = r#"<MPD minBufferTime="PT2.0S"><Period id="0">
            <AdaptationSet mimeType="video/mp4">
                <Representation id="hd" bandwidth="3000000" width="1280" height="720" codecs="avc1.42e00a"/>
            </AdaptationSet>
        </Period></MPD>"#;
        let m = parse(xml);
        assert_eq!(m.periods.len(), 1);
        assert_eq!(m.periods[0].adaptation_sets.len(), 1);
        assert_eq!(m.periods[0].adaptation_sets[0].representations.len(), 1);
        assert_eq!(
            m.periods[0].adaptation_sets[0].representations[0].bandwidth,
            3_000_000
        );
    }
}
