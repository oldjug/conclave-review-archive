//! PDF object lexer — names, dicts, arrays, streams.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum PdfObj {
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
    String(Vec<u8>),
    Name(String),
    Array(Vec<PdfObj>),
    Dict(HashMap<String, PdfObj>),
    Stream {
        dict: HashMap<String, PdfObj>,
        data_offset: usize,
        data_len: usize,
    },
    /// Indirect reference: (object num, generation).
    Ref(u32, u16),
}

pub struct PdfLexer<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> PdfLexer<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn seek(&mut self, p: usize) {
        self.pos = p;
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.buf.len() {
            let c = self.buf[self.pos];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == b'\x0C' {
                self.pos += 1;
            } else if c == b'%' {
                while self.pos < self.buf.len() && self.buf[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.buf.get(self.pos).copied()
    }

    pub fn parse(&mut self) -> Option<PdfObj> {
        self.skip_whitespace();
        let c = self.peek()?;
        match c {
            b'/' => self.parse_name().map(PdfObj::Name),
            b'(' => self.parse_string(),
            b'<' => {
                if self.buf.get(self.pos + 1) == Some(&b'<') {
                    self.parse_dict_or_stream()
                } else {
                    self.parse_hex_string()
                }
            }
            b'[' => self.parse_array(),
            b't' | b'f' => self.parse_bool(),
            b'n' => self.parse_null(),
            b'-' | b'+' | b'0'..=b'9' | b'.' => self.parse_number(),
            _ => None,
        }
    }

    fn parse_name(&mut self) -> Option<String> {
        if self.peek() != Some(b'/') {
            return None;
        }
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.buf.len() {
            let c = self.buf[self.pos];
            if c == b' '
                || c == b'\t'
                || c == b'\n'
                || c == b'\r'
                || c == b'/'
                || c == b'<'
                || c == b'['
                || c == b'('
                || c == b'>'
                || c == b']'
            {
                break;
            }
            self.pos += 1;
        }
        std::str::from_utf8(&self.buf[start..self.pos])
            .ok()
            .map(String::from)
    }

    fn parse_string(&mut self) -> Option<PdfObj> {
        if self.peek() != Some(b'(') {
            return None;
        }
        self.pos += 1;
        let mut out = Vec::new();
        let mut depth = 1;
        while self.pos < self.buf.len() && depth > 0 {
            let c = self.buf[self.pos];
            match c {
                b'\\' => {
                    self.pos += 1;
                    if let Some(&e) = self.buf.get(self.pos) {
                        out.push(match e {
                            b'n' => b'\n',
                            b'r' => b'\r',
                            b't' => b'\t',
                            b'b' => 8,
                            b'f' => 12,
                            _ => e,
                        });
                        self.pos += 1;
                    }
                }
                b'(' => {
                    depth += 1;
                    out.push(c);
                    self.pos += 1;
                }
                b')' => {
                    depth -= 1;
                    if depth > 0 {
                        out.push(c);
                    }
                    self.pos += 1;
                }
                _ => {
                    out.push(c);
                    self.pos += 1;
                }
            }
        }
        Some(PdfObj::String(out))
    }

    fn parse_hex_string(&mut self) -> Option<PdfObj> {
        if self.peek() != Some(b'<') {
            return None;
        }
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.buf.len() && self.buf[self.pos] != b'>' {
            self.pos += 1;
        }
        let hex_chars: Vec<u8> = self.buf[start..self.pos]
            .iter()
            .copied()
            .filter(|c| !c.is_ascii_whitespace())
            .collect();
        self.pos += 1;
        let mut out = Vec::new();
        let mut acc: u8 = 0;
        let mut half = false;
        for c in hex_chars {
            let v = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return None,
            };
            if !half {
                acc = v << 4;
                half = true;
            } else {
                out.push(acc | v);
                half = false;
            }
        }
        if half {
            out.push(acc);
        }
        Some(PdfObj::String(out))
    }

    fn parse_dict_or_stream(&mut self) -> Option<PdfObj> {
        self.pos += 2; // skip <<
        let mut dict = HashMap::new();
        loop {
            self.skip_whitespace();
            if self.buf.get(self.pos..self.pos + 2) == Some(b">>") {
                self.pos += 2;
                break;
            }
            let key = self.parse_name()?;
            let value = self.parse()?;
            dict.insert(key, value);
        }
        self.skip_whitespace();
        // Stream detection: dict followed by "stream\n".
        if self.buf.get(self.pos..self.pos + 6) == Some(b"stream") {
            self.pos += 6;
            // Required: line terminator after `stream` (LF or CRLF).
            if self.buf.get(self.pos) == Some(&b'\r') {
                self.pos += 1;
            }
            if self.buf.get(self.pos) == Some(&b'\n') {
                self.pos += 1;
            }
            let data_offset = self.pos;
            let len = match dict.get("Length") {
                Some(PdfObj::Int(n)) => *n as usize,
                _ => return None,
            };
            self.pos += len;
            self.skip_whitespace();
            if self.buf.get(self.pos..self.pos + 9) == Some(b"endstream") {
                self.pos += 9;
            }
            return Some(PdfObj::Stream {
                dict,
                data_offset,
                data_len: len,
            });
        }
        Some(PdfObj::Dict(dict))
    }

    fn parse_array(&mut self) -> Option<PdfObj> {
        self.pos += 1; // [
        let mut items = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                self.pos += 1;
                break;
            }
            items.push(self.parse()?);
        }
        Some(PdfObj::Array(items))
    }

    fn parse_bool(&mut self) -> Option<PdfObj> {
        if self.buf.get(self.pos..self.pos + 4) == Some(b"true") {
            self.pos += 4;
            Some(PdfObj::Bool(true))
        } else if self.buf.get(self.pos..self.pos + 5) == Some(b"false") {
            self.pos += 5;
            Some(PdfObj::Bool(false))
        } else {
            None
        }
    }

    fn parse_null(&mut self) -> Option<PdfObj> {
        if self.buf.get(self.pos..self.pos + 4) == Some(b"null") {
            self.pos += 4;
            Some(PdfObj::Null)
        } else {
            None
        }
    }

    fn parse_number(&mut self) -> Option<PdfObj> {
        let start = self.pos;
        if matches!(self.peek(), Some(b'-') | Some(b'+')) {
            self.pos += 1;
        }
        let mut saw_dot = false;
        while let Some(&c) = self.buf.get(self.pos) {
            if c == b'.' {
                if saw_dot {
                    break;
                }
                saw_dot = true;
                self.pos += 1;
            } else if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.buf[start..self.pos]).ok()?;
        // Lookahead: "N M R" → indirect ref.
        let save = self.pos;
        self.skip_whitespace();
        if let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                let n2_start = self.pos;
                while let Some(&c) = self.buf.get(self.pos) {
                    if c.is_ascii_digit() {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let g_str = &self.buf[n2_start..self.pos];
                self.skip_whitespace();
                if self.buf.get(self.pos) == Some(&b'R') {
                    self.pos += 1;
                    let obj_num: u32 = s.parse().ok()?;
                    let gen_num: u16 = std::str::from_utf8(g_str).ok()?.parse().ok()?;
                    return Some(PdfObj::Ref(obj_num, gen_num));
                }
            }
        }
        self.pos = save;
        if saw_dot {
            Some(PdfObj::Real(s.parse().ok()?))
        } else {
            Some(PdfObj::Int(s.parse().ok()?))
        }
    }
}

/// Decode a Flate-encoded stream using the existing DEFLATE decoder.
pub fn flate_decode(input: &[u8]) -> Result<Vec<u8>, String> {
    // PDF Flate streams use zlib (with header bytes). Strip 2 zlib
    // header bytes if present.
    let body = if input.len() > 2 && (input[0] & 0x0F) == 8 {
        &input[2..]
    } else {
        input
    };
    cv_compression::inflate(body).map_err(|e| format!("{e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_int_real_bool_null() {
        let mut l = PdfLexer::new(b"42 3.14 true false null");
        assert_eq!(l.parse(), Some(PdfObj::Int(42)));
        assert!(matches!(l.parse(), Some(PdfObj::Real(_))));
        assert_eq!(l.parse(), Some(PdfObj::Bool(true)));
        assert_eq!(l.parse(), Some(PdfObj::Bool(false)));
        assert_eq!(l.parse(), Some(PdfObj::Null));
    }

    #[test]
    fn parse_name_and_array() {
        let mut l = PdfLexer::new(b"[/Name1 42 /Name2]");
        match l.parse() {
            Some(PdfObj::Array(v)) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0], PdfObj::Name("Name1".into()));
                assert_eq!(v[1], PdfObj::Int(42));
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn parse_dict() {
        let mut l = PdfLexer::new(b"<< /Length 5 /Type /Catalog >>");
        match l.parse() {
            Some(PdfObj::Dict(d)) => {
                assert_eq!(d.get("Length"), Some(&PdfObj::Int(5)));
                assert_eq!(d.get("Type"), Some(&PdfObj::Name("Catalog".into())));
            }
            _ => panic!("expected dict"),
        }
    }

    #[test]
    fn parse_string_and_escape() {
        let mut l = PdfLexer::new(b"(hello\\nworld)");
        assert_eq!(l.parse(), Some(PdfObj::String(b"hello\nworld".to_vec())));
    }

    #[test]
    fn parse_hex_string() {
        let mut l = PdfLexer::new(b"<48656C6C6F>");
        assert_eq!(l.parse(), Some(PdfObj::String(b"Hello".to_vec())));
    }

    #[test]
    fn parse_indirect_ref() {
        let mut l = PdfLexer::new(b"7 0 R");
        assert_eq!(l.parse(), Some(PdfObj::Ref(7, 0)));
    }

    #[test]
    fn parse_stream() {
        let buf = b"<< /Length 3 >>\nstream\nabc\nendstream";
        let mut l = PdfLexer::new(buf);
        match l.parse() {
            Some(PdfObj::Stream { data_len, .. }) => assert_eq!(data_len, 3),
            _ => panic!("expected stream"),
        }
    }
}
