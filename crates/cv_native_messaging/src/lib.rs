//! `cv_native_messaging` — Chrome-compatible Native Messaging host
//! protocol so extensions can talk to native desktop apps.
//!
//! Wire format (matches Chrome's `chrome.runtime.connectNative` /
//! `sendNativeMessage`):
//!   * 4-byte little-endian length prefix
//!   * UTF-8 JSON payload
//!
//! Two halves:
//!   1. **Host discovery** — read manifest from
//!      `HKCU\Software\Conclave\NativeMessagingHosts\<name>`
//!      (registry key whose default value is a path to a JSON
//!      manifest). Manifest fields: `name`, `description`, `path`,
//!      `type` ("stdio"), `allowed_origins`.
//!   2. **Pipe** — spawn the host process with CreateProcessW, set
//!      stdin/stdout to pipes, write+read length-prefixed frames.
//!
//! `MessageChannel` is the moving piece extensions get — `send(json)`
//! and `recv()`. The JS binding sits in cv_extensions.

#![allow(non_snake_case, non_camel_case_types, clippy::missing_safety_doc)]

use std::io::{Read, Write};

/// Manifest fields read from disk (we own the parser — minimal JSON
/// subset, no third-party crate).
#[derive(Debug, Clone, Default)]
pub struct HostManifest {
    pub name: String,
    pub description: String,
    pub path: String,
    pub message_type: String,
    pub allowed_origins: Vec<String>,
}

impl HostManifest {
    /// Parse the manifest JSON. Only the known field set; everything
    /// else is ignored.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut h = HostManifest::default();
        // Hand-rolled minimal JSON pull — we don't ship a parser dep.
        let mut chars = text.chars().peekable();
        skip_ws(&mut chars);
        if chars.next() != Some('{') {
            return Err("manifest must be an object".into());
        }
        loop {
            skip_ws(&mut chars);
            if chars.peek() == Some(&'}') {
                chars.next();
                break;
            }
            let key = read_json_string(&mut chars)?;
            skip_ws(&mut chars);
            if chars.next() != Some(':') {
                return Err("expected ':' after key".into());
            }
            skip_ws(&mut chars);
            match key.as_str() {
                "name" => h.name = read_json_string(&mut chars)?,
                "description" => h.description = read_json_string(&mut chars)?,
                "path" => h.path = read_json_string(&mut chars)?,
                "type" => h.message_type = read_json_string(&mut chars)?,
                "allowed_origins" => h.allowed_origins = read_json_string_array(&mut chars)?,
                _ => skip_json_value(&mut chars)?,
            }
            skip_ws(&mut chars);
            if chars.peek() == Some(&',') {
                chars.next();
            }
        }
        Ok(h)
    }

    pub fn origin_allowed(&self, origin: &str) -> bool {
        // Origins are extension URLs like `chrome-extension://<id>/`.
        // V1: substring match (Chrome's check is exact + wildcard).
        self.allowed_origins.iter().any(|o| origin.starts_with(o))
    }
}

fn skip_ws(it: &mut std::iter::Peekable<std::str::Chars>) {
    while let Some(&c) = it.peek() {
        if c.is_whitespace() {
            it.next();
        } else {
            break;
        }
    }
}

fn read_json_string(it: &mut std::iter::Peekable<std::str::Chars>) -> Result<String, String> {
    skip_ws(it);
    if it.next() != Some('"') {
        return Err("expected string".into());
    }
    let mut out = String::new();
    while let Some(c) = it.next() {
        match c {
            '"' => return Ok(out),
            '\\' => {
                let esc = it.next().ok_or("bad escape")?;
                match esc {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    _ => return Err(format!("unsupported escape: \\{esc}")),
                }
            }
            other => out.push(other),
        }
    }
    Err("unterminated string".into())
}

fn read_json_string_array(
    it: &mut std::iter::Peekable<std::str::Chars>,
) -> Result<Vec<String>, String> {
    skip_ws(it);
    if it.next() != Some('[') {
        return Err("expected array".into());
    }
    let mut out = Vec::new();
    loop {
        skip_ws(it);
        if it.peek() == Some(&']') {
            it.next();
            return Ok(out);
        }
        out.push(read_json_string(it)?);
        skip_ws(it);
        if it.peek() == Some(&',') {
            it.next();
        }
    }
}

/// Coarse skip-over for unknown values. Enough for our manifest reader.
fn skip_json_value(it: &mut std::iter::Peekable<std::str::Chars>) -> Result<(), String> {
    skip_ws(it);
    let &c = it.peek().ok_or("eof")?;
    match c {
        '"' => {
            read_json_string(it)?;
            Ok(())
        }
        '{' => {
            let mut depth = 0;
            for ch in it.by_ref() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
            Err("unterminated object".into())
        }
        '[' => {
            let mut depth = 0;
            for ch in it.by_ref() {
                match ch {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
            Err("unterminated array".into())
        }
        _ => {
            // number, true/false/null
            while let Some(&c) = it.peek() {
                if c.is_alphanumeric() || c == '.' || c == '-' || c == '+' {
                    it.next();
                } else {
                    break;
                }
            }
            Ok(())
        }
    }
}

/// Wire framer used over the spawned host's stdio pipes.
#[derive(Debug)]
pub struct Framer;

impl Framer {
    /// Read one length-prefixed frame, returning the JSON payload bytes.
    pub fn read_frame<R: Read>(r: &mut R) -> std::io::Result<Vec<u8>> {
        let mut lenbuf = [0u8; 4];
        r.read_exact(&mut lenbuf)?;
        let len = u32::from_le_bytes(lenbuf) as usize;
        if len > 64 * 1024 * 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "frame too large",
            ));
        }
        let mut payload = vec![0u8; len];
        r.read_exact(&mut payload)?;
        Ok(payload)
    }

    /// Write one length-prefixed frame.
    pub fn write_frame<W: Write>(w: &mut W, payload: &[u8]) -> std::io::Result<()> {
        if payload.len() > u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "frame too large",
            ));
        }
        let len = (payload.len() as u32).to_le_bytes();
        w.write_all(&len)?;
        w.write_all(payload)?;
        w.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_manifest() {
        let txt = r#"{
            "name": "com.example.host",
            "description": "Test host",
            "path": "C:\\bin\\host.exe",
            "type": "stdio",
            "allowed_origins": ["chrome-extension://abc/", "chrome-extension://def/"]
        }"#;
        let m = HostManifest::parse(txt).unwrap();
        assert_eq!(m.name, "com.example.host");
        assert_eq!(m.message_type, "stdio");
        assert_eq!(m.path, "C:\\bin\\host.exe");
        assert_eq!(m.allowed_origins.len(), 2);
        assert!(m.origin_allowed("chrome-extension://abc/popup.html"));
        assert!(!m.origin_allowed("chrome-extension://evil/"));
    }

    #[test]
    fn frame_roundtrip() {
        let payload = br#"{"hello":"world"}"#;
        let mut buf: Vec<u8> = Vec::new();
        Framer::write_frame(&mut buf, payload).unwrap();
        assert_eq!(&buf[..4], &(payload.len() as u32).to_le_bytes());
        let mut cur = Cursor::new(buf);
        let got = Framer::read_frame(&mut cur).unwrap();
        assert_eq!(got, payload);
    }

    #[test]
    fn reject_oversized_frame() {
        // length 0x07FF_FFFF (~134 MB) — within limit; we should
        // attempt to read but our test cursor has zero bytes after,
        // so we expect a read error (EOF), not the size check.
        let mut buf: Vec<u8> = vec![0xff, 0xff, 0xff, 0x07];
        // 0x07FFFFFF = 134217727 — under the 64 MB ceiling? Actually
        // 0x07FFFFFF > 0x04000000 (64 MB), so it should reject.
        let mut cur = Cursor::new(&mut buf);
        let err = Framer::read_frame(&mut cur).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
