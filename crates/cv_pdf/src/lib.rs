//! `cv_pdf` — PDF reader AND writer + the browser print/PDF-export pipeline.
//!
//! Reader (the byte-level structure of a PDF file):
//!   * Header detection (`%PDF-1.x` / `%PDF-2.0`)
//!   * `startxref` + xref table parsing
//!   * Indirect object lookup by (object number, generation)
//!   * Cross-reference + trailer dictionary
//!   * Stream decode (Flate), object lexer, indirect references, page
//!     tree walk + MediaBox extraction.
//!
//! Writer + print flow (`window.print()` / export-to-PDF):
//!   * [`print_layout`] paginates a laid-out box tree (already cascaded under
//!     the `print` media type) into [`print_layout::PrintPage`]s, honouring
//!     `@page` size/margins and `break-before`/`-after`/`-inside`.
//!   * [`writer`] serialises those pages to a real PDF 1.7 file with selectable
//!     text, filled rectangles and RGB images.

#![allow(missing_debug_implementations)]

pub mod object;
pub mod page;
pub mod print_layout;
pub mod print_preview;
pub mod writer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdfVersion {
    pub major: u8,
    pub minor: u8,
}

pub fn parse_version(buf: &[u8]) -> Option<PdfVersion> {
    let prefix = b"%PDF-";
    if !buf.starts_with(prefix) {
        return None;
    }
    let rest = &buf[prefix.len()..];
    if rest.len() < 3 {
        return None;
    }
    let major = rest[0].checked_sub(b'0')?;
    if rest[1] != b'.' {
        return None;
    }
    let minor = rest[2].checked_sub(b'0')?;
    if major > 9 || minor > 9 {
        return None;
    }
    Some(PdfVersion { major, minor })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XrefEntry {
    pub offset: u64,
    pub generation: u16,
    pub in_use: bool,
}

#[derive(Debug, Default)]
pub struct Xref {
    /// Per-object number → entry.
    pub entries: Vec<Option<XrefEntry>>,
}

impl Xref {
    pub fn lookup(&self, obj_num: u32) -> Option<XrefEntry> {
        self.entries.get(obj_num as usize).copied().flatten()
    }
}

/// Find the byte offset of the last `startxref` token in the file.
/// Per spec the xref table sits at that offset.
pub fn find_startxref(buf: &[u8]) -> Option<u64> {
    let needle = b"startxref";
    let pos = buf.windows(needle.len()).rposition(|w| w == needle)?;
    let rest = &buf[pos + needle.len()..];
    let mut i = 0;
    while i < rest.len() && (rest[i] == b'\n' || rest[i] == b'\r' || rest[i] == b' ') {
        i += 1;
    }
    let mut end = i;
    while end < rest.len() && rest[end].is_ascii_digit() {
        end += 1;
    }
    std::str::from_utf8(&rest[i..end]).ok()?.parse::<u64>().ok()
}

/// Parse a classical (non-stream) xref table starting at `offset`.
/// Sets `entries` for each object number encountered.
pub fn parse_xref(buf: &[u8], offset: u64) -> Option<Xref> {
    let bytes = buf.get((offset as usize)..)?;
    let text = std::str::from_utf8(bytes).ok()?;
    let mut lines = text.lines();
    if lines.next()?.trim() != "xref" {
        return None;
    }
    let mut xref = Xref::default();
    while let Some(header) = lines.next() {
        let header = header.trim();
        if header.starts_with("trailer") || header.is_empty() {
            break;
        }
        let mut parts = header.split_whitespace();
        let start: usize = parts.next()?.parse().ok()?;
        let count: usize = parts.next()?.parse().ok()?;
        if xref.entries.len() < start + count {
            xref.entries.resize(start + count, None);
        }
        for i in 0..count {
            let entry_line = lines.next()?.trim();
            let mut tok = entry_line.split_whitespace();
            let off: u64 = tok.next()?.parse().ok()?;
            let gen_num: u16 = tok.next()?.parse().ok()?;
            let flag = tok.next()?;
            let in_use = flag == "n";
            xref.entries[start + i] = Some(XrefEntry {
                offset: off,
                generation: gen_num,
                in_use,
            });
        }
    }
    Some(xref)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parses_1_7() {
        let buf = b"%PDF-1.7\n...";
        let v = parse_version(buf).unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 7);
    }

    #[test]
    fn version_parses_2_0() {
        let v = parse_version(b"%PDF-2.0\n").unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 0);
    }

    #[test]
    fn version_rejects_missing_prefix() {
        assert!(parse_version(b"PDF-1.7").is_none());
    }

    #[test]
    fn find_startxref_returns_offset() {
        let buf = b"%PDF-1.4\n...\nstartxref\n12345\n%%EOF\n";
        assert_eq!(find_startxref(buf), Some(12345));
    }

    #[test]
    fn parse_xref_table_two_entries() {
        let pdf = b"%PDF-1.4\n\
xref\n\
0 2\n\
0000000000 65535 f\n\
0000000018 00000 n\n\
trailer\n\
<<>>\nstartxref\n9\n%%EOF\n";
        let off = pdf.windows(4).position(|w| w == b"xref").unwrap() as u64;
        let xref = parse_xref(pdf, off).unwrap();
        let e0 = xref.lookup(0).unwrap();
        assert!(!e0.in_use);
        let e1 = xref.lookup(1).unwrap();
        assert!(e1.in_use);
        assert_eq!(e1.offset, 18);
    }

    #[test]
    fn lookup_missing_object_returns_none() {
        let xref = Xref::default();
        assert!(xref.lookup(42).is_none());
    }
}
