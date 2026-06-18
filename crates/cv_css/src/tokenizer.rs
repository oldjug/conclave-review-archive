//! CSS Syntax 3 tokenizer — practical subset.
//!
//! Reference: <https://www.w3.org/TR/css-syntax-3/#tokenization>.

use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CssToken {
    Ident(String),
    Function(String),  // `<ident>(`
    AtKeyword(String), // `@<ident>`
    Hash(String),      // `#<ident>` — for IDs and colors
    String(String),
    Number(f64),
    Percent(f64),
    Dimension {
        value: f64,
        unit: String,
    },
    Whitespace,
    Colon,
    Semicolon,
    Comma,
    LeftBrace,
    RightBrace,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    /// `<` `>` `+` `~` `*` `=` `/` `.` etc. Used as combinators / generic delimiters.
    Delim(char),
    Url(String),
    /// `!` followed by ident — used for `!important`.
    Bang,
    Eof,
}

impl fmt::Display for CssToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ident(s) => write!(f, "ident({s})"),
            Self::Function(s) => write!(f, "function({s})"),
            Self::AtKeyword(s) => write!(f, "@{s}"),
            Self::Hash(s) => write!(f, "#{s}"),
            Self::String(s) => write!(f, "string({s})"),
            Self::Number(n) => write!(f, "number({n})"),
            Self::Percent(n) => write!(f, "percent({n})"),
            Self::Dimension { value, unit } => write!(f, "dim({value}{unit})"),
            Self::Whitespace => f.write_str("ws"),
            Self::Colon => f.write_str(":"),
            Self::Semicolon => f.write_str(";"),
            Self::Comma => f.write_str(","),
            Self::LeftBrace => f.write_str("{"),
            Self::RightBrace => f.write_str("}"),
            Self::LeftParen => f.write_str("("),
            Self::RightParen => f.write_str(")"),
            Self::LeftBracket => f.write_str("["),
            Self::RightBracket => f.write_str("]"),
            Self::Delim(c) => write!(f, "delim({c:?})"),
            Self::Url(s) => write!(f, "url({s})"),
            Self::Bang => f.write_str("!"),
            Self::Eof => f.write_str("eof"),
        }
    }
}

pub fn tokenize(src: &str) -> Vec<CssToken> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b' ' | b'\t' | b'\n' | b'\r' | 0x0C => {
                i += 1;
                while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r' | 0x0C) {
                    i += 1;
                }
                out.push(CssToken::Whitespace);
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                // Comment.
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
            }
            b'{' => {
                out.push(CssToken::LeftBrace);
                i += 1;
            }
            b'}' => {
                out.push(CssToken::RightBrace);
                i += 1;
            }
            b'(' => {
                out.push(CssToken::LeftParen);
                i += 1;
            }
            b')' => {
                out.push(CssToken::RightParen);
                i += 1;
            }
            b'[' => {
                out.push(CssToken::LeftBracket);
                i += 1;
            }
            b']' => {
                out.push(CssToken::RightBracket);
                i += 1;
            }
            b':' => {
                out.push(CssToken::Colon);
                i += 1;
            }
            b';' => {
                out.push(CssToken::Semicolon);
                i += 1;
            }
            b',' => {
                out.push(CssToken::Comma);
                i += 1;
            }
            b'!' => {
                out.push(CssToken::Bang);
                i += 1;
            }
            b'#' => {
                i += 1;
                // CSS Syntax §4.3.1: a `#` followed by a name code point or a
                // valid escape produces a <hash-token> whose value is the result
                // of "consume a name" (§4.3.11). This decodes CSS escapes
                // (`#\30 nextIsWhiteSpace` → id "0nextIsWhiteSpace", `#zero\0` →
                // id "zero\u{FFFD}") AND accepts digit-leading bodies (hex colors
                // like `#336699`). Name code points are ident bytes (alnum / _ /
                // - / non-ASCII); a backslash that is a valid escape is consumed
                // via consume_escape.
                let mut name = String::new();
                while i < bytes.len() {
                    let b = bytes[i];
                    if is_valid_escape(bytes, i) {
                        if let Some(ch) = consume_escape(bytes, &mut i) {
                            name.push(ch);
                            continue;
                        }
                        break;
                    }
                    if is_ident_byte(b) {
                        if b.is_ascii() {
                            name.push(b as char);
                            i += 1;
                        } else {
                            name.push(decode_one(bytes, &mut i));
                        }
                    } else {
                        break;
                    }
                }
                out.push(CssToken::Hash(name));
            }
            b'@' => {
                i += 1;
                let name = consume_ident_chars(bytes, &mut i);
                if name.is_empty() {
                    out.push(CssToken::Delim('@'));
                } else {
                    out.push(CssToken::AtKeyword(name));
                }
            }
            b'"' | b'\'' => {
                // CSS Syntax §4.3.5 string token, with §4.3.7 escape
                // decoding. `\HHHHHH` (1–6 hex digits) followed by
                // optional whitespace is a Unicode codepoint escape;
                // `\<non-hex>` is the literal character. Without
                // decoding, Font Awesome's `content: "\f135"` would
                // store the 5 literal chars `\f135` and render as
                // garbled text instead of the icon codepoint U+F135.
                // (The backslash in Orbitron looks like a V at small
                // sizes — visible bug on every Font Awesome icon.)
                let quote = b;
                i += 1;
                let mut decoded = String::new();
                while i < bytes.len() && bytes[i] != quote {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 1;
                        // Collect up to 6 hex digits.
                        let mut hex = String::new();
                        while hex.len() < 6 && i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                            hex.push(bytes[i] as char);
                            i += 1;
                        }
                        if !hex.is_empty() {
                            // One optional whitespace char terminates
                            // the hex escape (so `\f135 ` reads as one
                            // escape then keeps the next char in the
                            // string).
                            if i < bytes.len()
                                && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r' | 0x0c)
                            {
                                i += 1;
                            }
                            if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                                if let Some(c) = char::from_u32(cp) {
                                    decoded.push(c);
                                } else {
                                    // Invalid codepoint → U+FFFD per spec.
                                    decoded.push('\u{FFFD}');
                                }
                            }
                        } else if i < bytes.len() {
                            // `\<non-hex>` → that literal character.
                            // Multi-byte UTF-8: consume the full
                            // continuation.
                            let start = i;
                            i += 1;
                            while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                                i += 1;
                            }
                            decoded.push_str(&String::from_utf8_lossy(&bytes[start..i]));
                        }
                    } else {
                        // Plain character. Walk one UTF-8 sequence.
                        let start = i;
                        i += 1;
                        while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                            i += 1;
                        }
                        decoded.push_str(&String::from_utf8_lossy(&bytes[start..i]));
                    }
                }
                out.push(CssToken::String(decoded));
                if i < bytes.len() {
                    i += 1; // closing quote
                }
            }
            b'0'..=b'9' => {
                let (val, end) = consume_number(bytes, i);
                i = end;
                if bytes.get(i) == Some(&b'%') {
                    i += 1;
                    out.push(CssToken::Percent(val));
                } else if is_ident_start(bytes, i) {
                    let unit = consume_ident_chars(bytes, &mut i);
                    out.push(CssToken::Dimension { value: val, unit });
                } else {
                    out.push(CssToken::Number(val));
                }
            }
            b'+' | b'-'
                if bytes
                    .get(i + 1)
                    .is_some_and(|c| c.is_ascii_digit() || *c == b'.') =>
            {
                let (val, end) = consume_number(bytes, i);
                i = end;
                if bytes.get(i) == Some(&b'%') {
                    i += 1;
                    out.push(CssToken::Percent(val));
                } else if is_ident_start(bytes, i) {
                    let unit = consume_ident_chars(bytes, &mut i);
                    out.push(CssToken::Dimension { value: val, unit });
                } else {
                    out.push(CssToken::Number(val));
                }
            }
            b'.' if bytes.get(i + 1).is_some_and(|c| c.is_ascii_digit()) => {
                let (val, end) = consume_number(bytes, i);
                i = end;
                if bytes.get(i) == Some(&b'%') {
                    i += 1;
                    out.push(CssToken::Percent(val));
                } else if is_ident_start(bytes, i) {
                    let unit = consume_ident_chars(bytes, &mut i);
                    out.push(CssToken::Dimension { value: val, unit });
                } else {
                    out.push(CssToken::Number(val));
                }
            }
            b if is_ident_start_byte(b) => {
                let name = consume_ident_chars(bytes, &mut i);
                // url(<token>) special-case for unquoted URLs.
                if name.eq_ignore_ascii_case("url") && bytes.get(i) == Some(&b'(') {
                    i += 1;
                    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
                        i += 1;
                    }
                    let url = if bytes.get(i) == Some(&b'"') || bytes.get(i) == Some(&b'\'') {
                        let q = bytes[i];
                        i += 1;
                        let s = i;
                        while i < bytes.len() && bytes[i] != q {
                            i += 1;
                        }
                        let s = String::from_utf8_lossy(&bytes[s..i.min(bytes.len())]).to_string();
                        if i < bytes.len() {
                            i += 1;
                        }
                        s
                    } else {
                        let s = i;
                        while i < bytes.len() && bytes[i] != b')' {
                            i += 1;
                        }
                        String::from_utf8_lossy(&bytes[s..i.min(bytes.len())])
                            .trim()
                            .to_string()
                    };
                    while i < bytes.len() && bytes[i] != b')' {
                        i += 1;
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                    out.push(CssToken::Url(url));
                } else if bytes.get(i) == Some(&b'(') {
                    i += 1;
                    out.push(CssToken::Function(name));
                } else {
                    out.push(CssToken::Ident(name));
                }
            }
            _ => {
                // Decode UTF-8 to get char for Delim.
                let ch = decode_one(bytes, &mut i);
                out.push(CssToken::Delim(ch));
            }
        }
    }
    out.push(CssToken::Eof);
    out
}

fn is_ident_start_byte(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b == b'-' || b >= 0x80
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b >= 0x80
}

fn is_valid_escape(bytes: &[u8], i: usize) -> bool {
    matches!(bytes.get(i), Some(b'\\'))
        && matches!(bytes.get(i + 1), Some(next) if !matches!(next, b'\n' | b'\r' | 0x0C))
}

fn is_ident_start(bytes: &[u8], i: usize) -> bool {
    let Some(&b) = bytes.get(i) else { return false };
    if is_ident_start_byte(b) {
        return true;
    }
    if is_valid_escape(bytes, i) {
        return true;
    }
    if b == b'-' {
        if let Some(&n) = bytes.get(i + 1) {
            return is_ident_start_byte(n) || n == b'-' || is_valid_escape(bytes, i + 1);
        }
    }
    false
}

fn is_ident_continue(bytes: &[u8], i: usize) -> bool {
    bytes
        .get(i)
        .is_some_and(|b| is_ident_byte(*b) || is_valid_escape(bytes, i))
}

fn hex_value(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u32),
        b'a'..=b'f' => Some((b - b'a' + 10) as u32),
        b'A'..=b'F' => Some((b - b'A' + 10) as u32),
        _ => None,
    }
}

fn consume_escape(bytes: &[u8], i: &mut usize) -> Option<char> {
    if !is_valid_escape(bytes, *i) {
        return None;
    }
    *i += 1; // skip backslash
    let mut value: u32 = 0;
    let mut digits = 0;
    while *i < bytes.len() && digits < 6 {
        let Some(v) = hex_value(bytes[*i]) else { break };
        value = (value << 4) | v;
        *i += 1;
        digits += 1;
    }
    if digits > 0 {
        if *i < bytes.len() && matches!(bytes[*i], b' ' | b'\t' | b'\n' | b'\r' | 0x0C) {
            *i += 1;
        }
        // css-syntax §4.3.7: if the code point is zero, a surrogate, or greater
        // than the maximum allowed code point, return U+FFFD REPLACEMENT
        // CHARACTER. (char::from_u32 already rejects surrogates / out-of-range,
        // but accepts 0, so guard zero explicitly.)
        if value == 0 {
            return Some('\u{FFFD}');
        }
        return char::from_u32(value).or(Some('\u{FFFD}'));
    }
    if *i >= bytes.len() {
        return None;
    }
    Some(decode_one(bytes, i))
}

fn consume_ident_chars(bytes: &[u8], i: &mut usize) -> String {
    let mut out = String::new();
    if *i < bytes.len() && bytes[*i] == b'-' {
        out.push('-');
        *i += 1;
    }
    while *i < bytes.len() && is_ident_continue(bytes, *i) {
        if let Some(ch) = consume_escape(bytes, i) {
            out.push(ch);
        } else if bytes[*i].is_ascii() {
            out.push(bytes[*i] as char);
            *i += 1;
        } else {
            out.push(decode_one(bytes, i));
        }
    }
    out
}

fn consume_number(bytes: &[u8], mut i: usize) -> (f64, usize) {
    let start = i;
    if matches!(bytes.get(i), Some(b'+' | b'-')) {
        i += 1;
    }
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if bytes.get(i) == Some(&b'.') && bytes.get(i + 1).is_some_and(u8::is_ascii_digit) {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        let save = i;
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if !bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i = save;
        } else {
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
    }
    let s = std::str::from_utf8(&bytes[start..i]).unwrap_or("0");
    let v = s.parse::<f64>().unwrap_or(0.0);
    (v, i)
}

fn decode_one(bytes: &[u8], i: &mut usize) -> char {
    let b = bytes[*i];
    *i += 1;
    if b < 0x80 {
        return b as char;
    }
    let extra = if b >= 0xF0 {
        3
    } else if b >= 0xE0 {
        2
    } else if b >= 0xC0 {
        1
    } else {
        return '\u{FFFD}';
    };
    let mut buf = [0u8; 4];
    buf[0] = b;
    for k in 0..extra {
        match bytes.get(*i) {
            Some(&n) if n & 0xC0 == 0x80 => {
                buf[k + 1] = n;
                *i += 1;
            }
            _ => return '\u{FFFD}',
        }
    }
    std::str::from_utf8(&buf[..=extra])
        .ok()
        .and_then(|s| s.chars().next())
        .unwrap_or('\u{FFFD}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn css_string_escapes_decode_to_unicode_codepoints() {
        // Font Awesome and every modern icon font defines
        // `.fa-rocket::before { content: "\f135" }`. Before this fix
        // the tokenizer stored the raw 5 characters `\f135` and the
        // icon rendered as garbled text (the backslash in display
        // fonts often looks like a V at small sizes — every Font
        // Awesome icon on the affected site showed as "V###" garbage).
        let t = tokenize(r#"a{content:"\f135"}"#);
        let str_tok = t
            .into_iter()
            .find_map(|tk| {
                if let CssToken::String(s) = tk {
                    Some(s)
                } else {
                    None
                }
            })
            .expect("string token");
        assert_eq!(str_tok, "\u{f135}");
        assert_eq!(str_tok.chars().count(), 1);

        // Trailing whitespace terminates the hex escape (so
        // `\f135 more` reads as U+F135 then "more").
        let t = tokenize(r#"a{content:"\f135 more"}"#);
        let s = t
            .into_iter()
            .find_map(|tk| {
                if let CssToken::String(s) = tk {
                    Some(s)
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(s, "\u{f135}more");

        // `\<non-hex>` is a literal character escape.
        let t = tokenize(r#"a{content:"\""}"#);
        let s = t
            .into_iter()
            .find_map(|tk| {
                if let CssToken::String(s) = tk {
                    Some(s)
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(s, "\"");
    }

    #[test]
    fn idents_and_braces() {
        let t = tokenize("body { color: red; }");
        // Filter whitespace for easier assertion.
        let v: Vec<_> = t
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace))
            .collect();
        assert!(matches!(v[0], CssToken::Ident(ref s) if s == "body"));
        assert!(matches!(v[1], CssToken::LeftBrace));
        assert!(matches!(v[2], CssToken::Ident(ref s) if s == "color"));
        assert!(matches!(v[3], CssToken::Colon));
        assert!(matches!(v[4], CssToken::Ident(ref s) if s == "red"));
        assert!(matches!(v[5], CssToken::Semicolon));
        assert!(matches!(v[6], CssToken::RightBrace));
    }

    #[test]
    fn ident_consumes_tailwind_escaped_class_fragments() {
        let t = tokenize(r#".md\:ml-\[250px\].border-hyve-border\/50{}"#);
        let v: Vec<_> = t
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace | CssToken::Eof))
            .collect();
        assert!(matches!(v[0], CssToken::Delim('.')));
        assert!(matches!(v[1], CssToken::Ident(ref s) if s == "md:ml-[250px]"));
        assert!(matches!(v[2], CssToken::Delim('.')));
        assert!(matches!(v[3], CssToken::Ident(ref s) if s == "border-hyve-border/50"));
    }

    #[test]
    fn ident_decodes_hex_escape() {
        let t = tokenize(r#".w-\[\32 50px\]{}"#);
        let v: Vec<_> = t
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace | CssToken::Eof))
            .collect();
        assert!(matches!(v[1], CssToken::Ident(ref s) if s == "w-[250px]"));
    }

    #[test]
    fn hash_class_dot() {
        let t = tokenize("#main .nav a.active");
        let v: Vec<_> = t
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace))
            .collect();
        assert!(matches!(v[0], CssToken::Hash(ref s) if s == "main"));
        assert!(matches!(v[1], CssToken::Delim('.')));
        assert!(matches!(v[2], CssToken::Ident(ref s) if s == "nav"));
        assert!(matches!(v[3], CssToken::Ident(ref s) if s == "a"));
        assert!(matches!(v[4], CssToken::Delim('.')));
    }

    #[test]
    fn numbers_units_percents() {
        let t = tokenize("12px 0.5em 75% -1.5rem");
        let v: Vec<_> = t
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace))
            .collect();
        assert!(
            matches!(&v[0], CssToken::Dimension { value, unit } if (*value - 12.0).abs() < 1e-9 && unit == "px")
        );
        assert!(
            matches!(&v[1], CssToken::Dimension { value, unit } if (*value - 0.5).abs() < 1e-9 && unit == "em")
        );
        assert!(matches!(&v[2], CssToken::Percent(v) if (v - 75.0).abs() < 1e-9));
        assert!(
            matches!(&v[3], CssToken::Dimension { value, unit } if (*value + 1.5).abs() < 1e-9 && unit == "rem")
        );
    }

    #[test]
    fn strings_and_urls() {
        let t = tokenize(r#"content: "hi"; background: url(foo.png);"#);
        assert!(
            t.iter()
                .any(|x| matches!(x, CssToken::String(s) if s == "hi"))
        );
        assert!(
            t.iter()
                .any(|x| matches!(x, CssToken::Url(s) if s == "foo.png"))
        );
    }

    #[test]
    fn at_rule_keyword() {
        let t = tokenize("@media (max-width: 600px) { body {} }");
        assert!(matches!(&t[0], CssToken::AtKeyword(s) if s == "media"));
    }
}
