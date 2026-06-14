//! Percent-encoding helpers. Encode sets from
//! <https://url.spec.whatwg.org/#percent-encoded-bytes>.

#[derive(Copy, Clone, Debug)]
pub enum EncodeSet {
    /// C0 controls and U+007F+.
    C0Control,
    /// C0 + space + `"`, `<`, `>`, `` ` ``.
    Fragment,
    /// Fragment + `#`.
    Query,
    /// Query + `?`, `` ` ``, `{`, `}`.
    SpecialQuery,
    /// Query + `?`, `` ` ``, `{`, `}`.
    Path,
    /// Path + `/`, `:`, `;`, `=`, `@`, `[`, `]`, `\`, `^`, `|`.
    Userinfo,
}

#[inline]
fn in_c0_control(c: u8) -> bool {
    c < 0x20 || c == 0x7F
}

fn needs_encode(c: u8, set: EncodeSet) -> bool {
    if in_c0_control(c) || c >= 0x80 {
        return true;
    }
    match set {
        EncodeSet::C0Control => false,
        EncodeSet::Fragment => matches!(c, b' ' | b'"' | b'<' | b'>' | b'`'),
        EncodeSet::Query => matches!(c, b' ' | b'"' | b'#' | b'<' | b'>'),
        EncodeSet::SpecialQuery => matches!(c, b' ' | b'"' | b'#' | b'<' | b'>' | b'\''),
        EncodeSet::Path => matches!(
            c,
            b' ' | b'"' | b'#' | b'<' | b'>' | b'?' | b'`' | b'{' | b'}'
        ),
        EncodeSet::Userinfo => {
            matches!(
                c,
                b' ' | b'"'
                    | b'#'
                    | b'<'
                    | b'>'
                    | b'?'
                    | b'`'
                    | b'{'
                    | b'}'
                    | b'/'
                    | b':'
                    | b';'
                    | b'='
                    | b'@'
                    | b'['
                    | b'\\'
                    | b']'
                    | b'^'
                    | b'|'
            )
        }
    }
}

pub fn encode_into(out: &mut String, bytes: &[u8], set: EncodeSet) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for &b in bytes {
        if needs_encode(b, set) {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0F) as usize] as char);
        } else {
            out.push(b as char);
        }
    }
}

pub fn encode(bytes: &[u8], set: EncodeSet) -> String {
    let mut out = String::with_capacity(bytes.len());
    encode_into(&mut out, bytes, set);
    out
}

pub fn decode(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_encode_space() {
        assert_eq!(encode(b"hello world", EncodeSet::Path), "hello%20world");
    }

    #[test]
    fn decode_basic() {
        assert_eq!(decode("hello%20world"), b"hello world");
        assert_eq!(decode("a%2Bb"), b"a+b");
        assert_eq!(decode("bad%XX"), b"bad%XX"); // invalid escapes preserved
    }
}
