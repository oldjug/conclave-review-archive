//! WebVTT (Web Video Text Tracks, W3C TTWG) cue parser.

#[derive(Debug, Clone)]
pub struct VttCue {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub id: Option<String>,
}

/// Parse a WebVTT body. The first non-blank line must be `WEBVTT` per
/// spec; cues are blocks separated by blank lines. Settings on the
/// timing line (`align:start position:50%`) are preserved verbatim in
/// future surfaces; for V1 they're skipped.
pub fn parse(text: &str) -> Vec<VttCue> {
    let mut out = Vec::new();
    let mut lines = text.split('\n').map(|l| l.trim_end_matches('\r'));
    let header = lines.next().unwrap_or("");
    if !header.starts_with("WEBVTT") {
        return out;
    }
    let mut cur_block: Vec<String> = Vec::new();
    for line in lines {
        if line.is_empty() {
            if !cur_block.is_empty() {
                if let Some(cue) = parse_block(&cur_block) {
                    out.push(cue);
                }
                cur_block.clear();
            }
            continue;
        }
        cur_block.push(line.to_string());
    }
    if !cur_block.is_empty() {
        if let Some(cue) = parse_block(&cur_block) {
            out.push(cue);
        }
    }
    out
}

fn parse_block(block: &[String]) -> Option<VttCue> {
    // optional id, then timing line "00:00:00.000 --> 00:00:00.000 ...", then text
    let mut idx = 0;
    let mut id = None;
    if !block[0].contains("-->") {
        id = Some(block[0].clone());
        idx = 1;
    }
    if idx >= block.len() {
        return None;
    }
    let timing = &block[idx];
    let (start, end) = parse_timing(timing)?;
    idx += 1;
    let text = block[idx..].join("\n");
    Some(VttCue {
        start_ms: start,
        end_ms: end,
        text,
        id,
    })
}

fn parse_timing(line: &str) -> Option<(u64, u64)> {
    let (left, rest) = line.split_once("-->")?;
    let right = rest.split_whitespace().next()?;
    Some((parse_time(left.trim())?, parse_time(right.trim())?))
}

fn parse_time(s: &str) -> Option<u64> {
    // h:mm:ss.mmm or mm:ss.mmm
    let mut parts = s.split(':').collect::<Vec<_>>();
    if parts.len() == 2 {
        parts.insert(0, "0");
    }
    if parts.len() != 3 {
        return None;
    }
    let h: u64 = parts[0].parse().ok()?;
    let m: u64 = parts[1].parse().ok()?;
    let sec_ms = parts[2];
    let (secs, ms) = match sec_ms.split_once('.') {
        Some((s, ms)) => (s.parse::<u64>().ok()?, ms.parse::<u64>().ok()?),
        None => (sec_ms.parse::<u64>().ok()?, 0),
    };
    Some(h * 3_600_000 + m * 60_000 + secs * 1000 + ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_two_cues() {
        let text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello\n\n00:00:02.000 --> 00:00:03.000\nWorld";
        let cues = parse(text);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].start_ms, 0);
        assert_eq!(cues[0].end_ms, 1500);
        assert_eq!(cues[1].text, "World");
    }
}
