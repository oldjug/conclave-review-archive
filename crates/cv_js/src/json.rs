//! Minimal JSON parser + serializer per ECMA-404.
//!
//! Used by the runtime's `JSON.parse` / `JSON.stringify` built-ins.

use crate::interp::Value;
use crate::ordered::OrderedMap as HashMap;
use std::cell::RefCell;
use std::rc::Rc;

pub fn parse(src: &str) -> Result<Value, String> {
    let mut p = Parser {
        bytes: src.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing content at byte {}", p.pos));
    }
    Ok(v)
}

pub fn stringify(v: &Value) -> String {
    let mut out = String::new();
    stringify_into(v, &mut out);
    out
}

/// Pretty-printed `JSON.stringify(value, replacer, indent)`. `indent`
/// is the per-level indent string (e.g. `"  "` for 2-space). Falls
/// back to the compact form for indent == 0.
pub fn stringify_pretty(v: &Value, indent: &str) -> String {
    if indent.is_empty() {
        return stringify(v);
    }
    let mut out = String::new();
    stringify_pretty_into(v, &mut out, indent, 0);
    out
}

fn stringify_pretty_into(v: &Value, out: &mut String, indent: &str, depth: usize) {
    use crate::ordered::OrderedMap as HashMap;
    let push_nl = |o: &mut String, d: usize| {
        o.push('\n');
        for _ in 0..d {
            o.push_str(indent);
        }
    };
    match v {
        Value::Object(o) => {
            let b = o.borrow();
            if b.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            // Per ECMA-262 §25.5.2 SerializeJSONObject: iterate keys in
            // [[OwnPropertyKeys]] order (= insertion order for our objects),
            // NOT alphabetical. Previously the pretty path called `.sort()`
            // which silently reshuffled — broke canonical JSON, log diffs,
            // and any consumer that compares stringified JSON.
            let keys: Vec<&String> = b
                .keys()
                .filter(|k| !crate::interp::is_internal_key(k))
                .collect();
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_nl(out, depth + 1);
                let mut esc = String::new();
                let _ = std::fmt::Write::write_fmt(&mut esc, format_args!("{k:?}"));
                out.push_str(&esc);
                out.push_str(": ");
                stringify_pretty_into(b.get(*k).unwrap_or(&Value::Null), out, indent, depth + 1);
            }
            push_nl(out, depth);
            out.push('}');
            let _ = HashMap::<u8, u8>::new();
        }
        Value::Array(a) => {
            let b = a.borrow();
            if b.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, v) in b.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_nl(out, depth + 1);
                stringify_pretty_into(v, out, indent, depth + 1);
            }
            push_nl(out, depth);
            out.push(']');
        }
        // Scalars reuse the compact path so float / bool / null /
        // string formatting stays consistent.
        other => stringify_into(other, out),
    }
}

fn stringify_into(v: &Value, out: &mut String) {
    match v {
        Value::Undefined
        | Value::Hole
        | Value::NativeFunction(_)
        | Value::Function(_)
        | Value::BcClosure(_) => {
            // JSON.stringify(undefined) returns undefined as a value, but
            // when called at the top level its return type is a string —
            // ECMA-262 §24.5.2 specifies the result is undefined (i.e. not
            // a string at all). For our use case we emit an empty string.
            out.push_str("null");
        }
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_nan() || n.is_infinite() {
                out.push_str("null");
            } else if *n == 0.0 {
                // Both +0 and -0 serialize as "0" per ECMA-262 §25.5.2:
                // SerializeJSONProperty → Number::toString drops the sign
                // on zero. Previously `JSON.stringify(-0)` emitted "-0"
                // (Rust's default), causing diff-based tests/canon JSON
                // hashes to diverge from Chrome.
                out.push('0');
            } else if *n == n.trunc() && n.abs() < 1e16 {
                out.push_str(&format!("{:.0}", n));
            } else {
                // Reuse the spec-shaped exponential-or-decimal formatter
                // from the interpreter so JSON and Number-to-String
                // agree on `1e21`/`1e-7` thresholds.
                out.push_str(&crate::interp::format_number_es_pub(*n));
            }
        }
        Value::BigInt(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_string(s, out),
        Value::Array(a) => {
            out.push('[');
            let v = a.borrow();
            for (i, item) in v.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                stringify_into(item, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            let map = o.borrow();
            let mut first = true;
            for (k, val) in map.iter() {
                if crate::interp::is_internal_key(k) {
                    continue; // hide engine-internal [[Prototype]] slot
                }
                if matches!(
                    val,
                    Value::NativeFunction(_) | Value::Function(_) | Value::Undefined
                ) {
                    continue; // JSON.stringify drops these properties
                }
                if !first {
                    out.push(',');
                }
                first = false;
                write_string(k, out);
                out.push(':');
                stringify_into(val, out);
            }
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        self.skip_ws();
        if self.pos >= self.bytes.len() {
            return Err("unexpected end".into());
        }
        let b = self.bytes[self.pos];
        match b {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(|s| Value::str(s)),
            b't' | b'f' => self.parse_bool(),
            b'n' => self.parse_null(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(format!("unexpected byte {other:?} at {}", self.pos)),
        }
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.pos += 1; // {
        let mut map: HashMap<String, Value> = HashMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(Rc::new(RefCell::new(map))));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at {}", self.pos));
            }
            self.pos += 1;
            let v = self.parse_value()?;
            map.insert(key, v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(Rc::new(RefCell::new(map))));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.pos += 1; // [
        let mut items: Vec<Value> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(Rc::new(RefCell::new(items))));
        }
        loop {
            let v = self.parse_value()?;
            items.push(v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(Rc::new(RefCell::new(items))));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at {}", self.pos));
        }
        self.pos += 1;
        let mut s = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                return Err("unterminated string".into());
            }
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(s),
                b'\\' => {
                    if self.pos >= self.bytes.len() {
                        return Err("bad escape".into());
                    }
                    let esc = self.bytes[self.pos];
                    self.pos += 1;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'b' => s.push('\u{08}'),
                        b'f' => s.push('\u{0C}'),
                        b'u' => {
                            if self.pos + 4 > self.bytes.len() {
                                return Err("bad \\u escape".into());
                            }
                            let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                                .map_err(|_| "bad \\u escape utf8".to_string())?;
                            let code = u32::from_str_radix(hex, 16)
                                .map_err(|_| "bad \\u hex".to_string())?;
                            self.pos += 4;
                            // Per ECMA-262 §25.5.1: JSON strings are UTF-16
                            // sequences. A high-surrogate \uD800-\uDBFF must
                            // combine with the immediately-following
                            // low-surrogate \uDC00-\uDFFF into a single
                            // astral code point. Previously each half
                            // produced U+FFFD on its own, corrupting any
                            // JSON containing emoji / astral CJK / math
                            // symbols / private-use chars in `\u\u` form.
                            let combined: Option<char> = if (0xD800..=0xDBFF).contains(&code) {
                                // Try to consume a following \uDCxx low
                                // surrogate. Peek 6 bytes for `\uXXXX`.
                                if self.pos + 6 <= self.bytes.len()
                                    && self.bytes[self.pos] == b'\\'
                                    && self.bytes[self.pos + 1] == b'u'
                                {
                                    if let Ok(lo_hex) = std::str::from_utf8(
                                        &self.bytes[self.pos + 2..self.pos + 6],
                                    ) {
                                        if let Ok(lo) = u32::from_str_radix(lo_hex, 16) {
                                            if (0xDC00..=0xDFFF).contains(&lo) {
                                                let astral = 0x10000
                                                    + (((code - 0xD800) << 10)
                                                        | (lo - 0xDC00));
                                                self.pos += 6;
                                                char::from_u32(astral)
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                char::from_u32(code)
                            };
                            if let Some(c) = combined {
                                s.push(c);
                            } else {
                                s.push('\u{FFFD}');
                            }
                        }
                        other => return Err(format!("bad escape \\{}", other as char)),
                    }
                }
                _ => {
                    // Read this UTF-8 character.
                    let start = self.pos - 1;
                    let extra = if b >= 0xF0 {
                        3
                    } else if b >= 0xE0 {
                        2
                    } else if b >= 0xC0 {
                        1
                    } else {
                        0
                    };
                    if start + 1 + extra > self.bytes.len() {
                        return Err("bad utf8".into());
                    }
                    self.pos = start + 1 + extra;
                    let slice = &self.bytes[start..start + 1 + extra];
                    let ch = std::str::from_utf8(slice)
                        .ok()
                        .and_then(|s| s.chars().next())
                        .unwrap_or('\u{FFFD}');
                    s.push(ch);
                }
            }
        }
    }

    fn parse_bool(&mut self) -> Result<Value, String> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Value::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Value::Bool(false))
        } else {
            Err(format!("expected bool at {}", self.pos))
        }
    }

    fn parse_null(&mut self) -> Result<Value, String> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Value::Null)
        } else {
            Err(format!("expected null at {}", self.pos))
        }
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "bad number utf8".to_string())?;
        let v: f64 = s
            .parse()
            .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        Ok(Value::Number(v))
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_object() {
        let v = parse(r#"{"a":1,"b":"two","c":[true,false,null]}"#).unwrap();
        if let Value::Object(o) = v {
            let m = o.borrow();
            assert!(matches!(m.get("a"), Some(Value::Number(n)) if (n - 1.0).abs() < 1e-9));
            assert!(matches!(m.get("b"), Some(Value::String(s)) if &**s == "two"));
            assert!(matches!(m.get("c"), Some(Value::Array(_))));
        } else {
            panic!("parse_simple_object: expected Value::Object, got {:?}", v);
        }
    }

    #[test]
    fn stringify_roundtrip() {
        // HashMap iteration order isn't stable, so compare by re-parsing.
        let v = parse(r#"{"x":1.5,"y":[1,2,3],"z":"hi"}"#).unwrap();
        let s = stringify(&v);
        let v2 = parse(&s).unwrap();
        if let (Value::Object(a), Value::Object(b)) = (&v, &v2) {
            let am = a.borrow();
            let bm = b.borrow();
            assert_eq!(am.len(), bm.len());
            for (k, va) in am.iter() {
                let vb = bm.get(k).expect("key missing");
                match (va, vb) {
                    (Value::Number(x), Value::Number(y)) => assert!((x - y).abs() < 1e-9),
                    (Value::String(x), Value::String(y)) => assert_eq!(x, y),
                    (Value::Array(_), Value::Array(_)) => {}
                    _ => panic!(
                        "stringify_roundtrip: type mismatch at key '{}': parsed value type changed from {:?} to {:?}",
                        k, va, vb
                    ),
                }
            }
        } else {
            panic!(
                "stringify_roundtrip: expected both values to be objects after roundtrip, got {:?} and {:?}",
                v, v2
            );
        }
    }

    #[test]
    fn handles_escapes() {
        let v = parse(r#""hi\n\tworld""#).unwrap();
        assert!(matches!(&v, Value::String(s) if &**s == "hi\n\tworld"));
        assert_eq!(stringify(&v), "\"hi\\n\\tworld\"");
    }

    // ── Bug 2: pretty-print must keep insertion order, not sort keys ─────────
    #[test]
    fn stringify_pretty_preserves_insertion_order() {
        use crate::ordered::OrderedMap as HashMap;
        let mut m = HashMap::new();
        m.insert("b".to_string(), Value::Number(1.0));
        m.insert("a".to_string(), Value::Number(2.0));
        use std::cell::RefCell;
        use std::rc::Rc;
        let v = Value::Object(Rc::new(RefCell::new(m)));
        let s = stringify_pretty(&v, "  ");
        let b_pos = s.find("\"b\"").expect("key b missing");
        let a_pos = s.find("\"a\"").expect("key a missing");
        assert!(
            b_pos < a_pos,
            "key 'b' should appear before 'a' (insertion order), got: {s:?}"
        );
    }

    // ── Bug 3: JSON.stringify(-0) must emit "0", not "-0" ────────────────────
    #[test]
    fn stringify_negative_zero() {
        // ECMA-262 §25.5.2 SerializeJSONProperty: both +0 and -0 produce "0".
        let neg_zero = Value::Number(-0.0_f64);
        assert_eq!(
            stringify(&neg_zero),
            "0",
            "JSON.stringify(-0) should be \"0\", got {:?}",
            stringify(&neg_zero)
        );
        // Also verify via the pretty path (scalars delegate to compact).
        assert_eq!(
            stringify_pretty(&neg_zero, "  "),
            "0",
            "JSON.stringify(-0, null, 2) should be \"0\""
        );
    }

    // ── Bug 4: surrogate pairs must combine into a single astral code point ──
    #[test]
    fn parse_string_surrogate_pairs_combine() {
        // U+1F600 (😀) encoded as 😀 in JSON.
        let v = parse(r#""😀""#).unwrap();
        assert!(
            matches!(&v, Value::String(s) if &**s == "😀"),
            "surrogate pair should decode to 😀, got {:?}",
            v
        );
    }

    #[test]
    fn parse_string_lone_high_surrogate_becomes_replacement() {
        // A high surrogate NOT followed by a low surrogate → U+FFFD.
        let v = parse(r#""\uD800""#).unwrap();
        assert!(
            matches!(&v, Value::String(s) if &**s == "\u{FFFD}"),
            "lone high surrogate should become U+FFFD, got {:?}",
            v
        );
    }

    // ── Explicit test names requested in the bug report ──────────────────────

    /// `JSON.stringify({b:1,a:2}, null, 2)` must have "b" before "a".
    #[test]
    fn json_stringify_pretty_preserves_insertion_order() {
        use crate::ordered::OrderedMap as HashMap;
        use std::cell::RefCell;
        use std::rc::Rc;
        let mut m = HashMap::new();
        m.insert("b".to_string(), Value::Number(1.0));
        m.insert("a".to_string(), Value::Number(2.0));
        let v = Value::Object(Rc::new(RefCell::new(m)));
        let s = stringify_pretty(&v, "  ");
        let b_pos = s.find("\"b\"").expect("key b missing");
        let a_pos = s.find("\"a\"").expect("key a missing");
        assert!(
            b_pos < a_pos,
            "key 'b' should appear before 'a' (insertion order), got: {s:?}"
        );
    }

    /// `JSON.parse("\"\\uD83D\\uDE00\"")` must return the 😀 emoji (U+1F600).
    #[test]
    fn json_parse_surrogate_pair_emoji() {
        // The JSON string contains the two-unit sequence 😀 which
        // encodes U+1F600 (😀). Per ECMA-262 §24.5.1 the parser must combine
        // the high+low surrogate pair into the single astral scalar.
        let v = parse(r#""😀""#).unwrap();
        assert!(
            matches!(&v, Value::String(s) if &**s == "😀"),
            "\\uD83D\\uDE00 should decode to 😀 (U+1F600), got {:?}",
            v
        );
    }
}
