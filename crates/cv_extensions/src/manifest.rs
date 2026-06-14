//! Manifest V3 parser — schema validation + permission set.
//!
//! V1 parses the JSON subset Chrome documents as Required + the most
//! popular optional fields. Each field is exposed as a typed value
//! the runtime consults at registration time.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub manifest_version: u32,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub permissions: Vec<String>,
    pub host_permissions: Vec<String>,
    pub background: Option<BackgroundService>,
    pub content_scripts: Vec<ContentScript>,
    pub action: Option<BrowserAction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackgroundService {
    pub service_worker: String,
    pub type_module: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentScript {
    pub matches: Vec<String>,
    pub js: Vec<String>,
    pub css: Vec<String>,
    pub run_at: ContentScriptRunAt,
    pub all_frames: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentScriptRunAt {
    DocumentStart,
    DocumentEnd,
    DocumentIdle,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BrowserAction {
    pub default_title: Option<String>,
    pub default_icon: HashMap<String, String>,
    pub default_popup: Option<String>,
}

/// Minimal JSON value (we don't pull serde).
#[derive(Debug, Clone)]
enum Jv {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Jv>),
    Obj(Vec<(String, Jv)>),
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }
    fn consume(&mut self, c: u8) -> bool {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn parse_value(&mut self) -> Option<Jv> {
        self.skip_ws();
        let c = self.peek()?;
        Some(match c {
            b'"' => Jv::Str(self.parse_string()?),
            b'{' => self.parse_obj()?,
            b'[' => self.parse_array()?,
            b't' | b'f' => self.parse_bool()?,
            b'n' => self.parse_null()?,
            b'-' | b'0'..=b'9' => self.parse_num()?,
            _ => return None,
        })
    }
    fn parse_string(&mut self) -> Option<String> {
        if !self.consume(b'"') {
            return None;
        }
        let mut out = String::new();
        while let Some(c) = self.peek() {
            self.pos += 1;
            match c {
                b'"' => return Some(out),
                b'\\' => {
                    let e = self.peek()?;
                    self.pos += 1;
                    out.push(match e {
                        b'n' => '\n',
                        b't' => '\t',
                        b'\\' => '\\',
                        b'"' => '"',
                        b'/' => '/',
                        _ => e as char,
                    });
                }
                _ => out.push(c as char),
            }
        }
        None
    }
    fn parse_num(&mut self) -> Option<Jv> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9') | Some(b'.')) {
            self.pos += 1;
        }
        std::str::from_utf8(&self.src[start..self.pos])
            .ok()?
            .parse::<f64>()
            .ok()
            .map(Jv::Num)
    }
    fn parse_bool(&mut self) -> Option<Jv> {
        if self.src[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Some(Jv::Bool(true))
        } else if self.src[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Some(Jv::Bool(false))
        } else {
            None
        }
    }
    fn parse_null(&mut self) -> Option<Jv> {
        if self.src[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Some(Jv::Null)
        } else {
            None
        }
    }
    fn parse_array(&mut self) -> Option<Jv> {
        if !self.consume(b'[') {
            return None;
        }
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            if self.consume(b']') {
                return Some(Jv::Arr(out));
            }
            out.push(self.parse_value()?);
            self.skip_ws();
            if !self.consume(b',') && self.peek() != Some(b']') {
                return None;
            }
        }
    }
    fn parse_obj(&mut self) -> Option<Jv> {
        if !self.consume(b'{') {
            return None;
        }
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            if self.consume(b'}') {
                return Some(Jv::Obj(out));
            }
            let k = self.parse_string()?;
            self.skip_ws();
            if !self.consume(b':') {
                return None;
            }
            let v = self.parse_value()?;
            out.push((k, v));
            self.skip_ws();
            if !self.consume(b',') && self.peek() != Some(b'}') {
                return None;
            }
        }
    }
}

fn jv_str(jv: &Jv) -> Option<&str> {
    if let Jv::Str(s) = jv { Some(s) } else { None }
}
fn jv_arr(jv: &Jv) -> Option<&Vec<Jv>> {
    if let Jv::Arr(a) = jv { Some(a) } else { None }
}
fn jv_obj(jv: &Jv) -> Option<&Vec<(String, Jv)>> {
    if let Jv::Obj(o) = jv { Some(o) } else { None }
}
fn jv_num(jv: &Jv) -> Option<f64> {
    if let Jv::Num(n) = jv { Some(*n) } else { None }
}
fn jv_bool(jv: &Jv) -> Option<bool> {
    if let Jv::Bool(b) = jv { Some(*b) } else { None }
}
fn obj_get<'a>(obj: &'a [(String, Jv)], key: &str) -> Option<&'a Jv> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

pub fn parse_manifest(src: &str) -> Option<Manifest> {
    let mut p = Parser {
        src: src.as_bytes(),
        pos: 0,
    };
    let v = p.parse_value()?;
    let obj = jv_obj(&v)?;
    let manifest_version = obj_get(obj, "manifest_version").and_then(jv_num)? as u32;
    if manifest_version != 3 {
        return None; // V2 unsupported per project policy
    }
    let name = obj_get(obj, "name").and_then(jv_str)?.to_string();
    let version = obj_get(obj, "version").and_then(jv_str)?.to_string();
    let description = obj_get(obj, "description")
        .and_then(jv_str)
        .map(String::from);
    let permissions = obj_get(obj, "permissions")
        .and_then(jv_arr)
        .map(|a| a.iter().filter_map(jv_str).map(String::from).collect())
        .unwrap_or_default();
    let host_permissions = obj_get(obj, "host_permissions")
        .and_then(jv_arr)
        .map(|a| a.iter().filter_map(jv_str).map(String::from).collect())
        .unwrap_or_default();
    let background = obj_get(obj, "background").and_then(jv_obj).and_then(|o| {
        Some(BackgroundService {
            service_worker: obj_get(o, "service_worker").and_then(jv_str)?.to_string(),
            type_module: obj_get(o, "type")
                .and_then(jv_str)
                .map(|t| t == "module")
                .unwrap_or(false),
        })
    });
    let content_scripts = obj_get(obj, "content_scripts")
        .and_then(jv_arr)
        .map(|arr| {
            arr.iter()
                .filter_map(jv_obj)
                .map(|o| ContentScript {
                    matches: obj_get(o, "matches")
                        .and_then(jv_arr)
                        .map(|a| a.iter().filter_map(jv_str).map(String::from).collect())
                        .unwrap_or_default(),
                    js: obj_get(o, "js")
                        .and_then(jv_arr)
                        .map(|a| a.iter().filter_map(jv_str).map(String::from).collect())
                        .unwrap_or_default(),
                    css: obj_get(o, "css")
                        .and_then(jv_arr)
                        .map(|a| a.iter().filter_map(jv_str).map(String::from).collect())
                        .unwrap_or_default(),
                    run_at: match obj_get(o, "run_at")
                        .and_then(jv_str)
                        .unwrap_or("document_idle")
                    {
                        "document_start" => ContentScriptRunAt::DocumentStart,
                        "document_end" => ContentScriptRunAt::DocumentEnd,
                        _ => ContentScriptRunAt::DocumentIdle,
                    },
                    all_frames: obj_get(o, "all_frames").and_then(jv_bool).unwrap_or(false),
                })
                .collect()
        })
        .unwrap_or_default();
    let action = obj_get(obj, "action")
        .and_then(jv_obj)
        .map(|o| BrowserAction {
            default_title: obj_get(o, "default_title")
                .and_then(jv_str)
                .map(String::from),
            default_icon: obj_get(o, "default_icon")
                .and_then(jv_obj)
                .map(|icons| {
                    icons
                        .iter()
                        .filter_map(|(k, v)| jv_str(v).map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default(),
            default_popup: obj_get(o, "default_popup")
                .and_then(jv_str)
                .map(String::from),
        });
    Some(Manifest {
        manifest_version,
        name,
        version,
        description,
        permissions,
        host_permissions,
        background,
        content_scripts,
        action,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_v2_manifest() {
        let m = parse_manifest(r#"{"manifest_version": 2, "name": "Old", "version": "1"}"#);
        assert!(m.is_none());
    }

    #[test]
    fn parses_minimal_v3() {
        let src = r#"{
            "manifest_version": 3,
            "name": "Hello",
            "version": "1.0"
        }"#;
        let m = parse_manifest(src).unwrap();
        assert_eq!(m.name, "Hello");
        assert_eq!(m.version, "1.0");
        assert!(m.permissions.is_empty());
    }

    #[test]
    fn parses_permissions_and_host_permissions() {
        let src = r#"{
            "manifest_version": 3,
            "name": "Reader",
            "version": "1.2.3",
            "permissions": ["storage", "scripting"],
            "host_permissions": ["https://*.example.com/*"]
        }"#;
        let m = parse_manifest(src).unwrap();
        assert_eq!(m.permissions, vec!["storage", "scripting"]);
        assert_eq!(m.host_permissions, vec!["https://*.example.com/*"]);
    }

    #[test]
    fn parses_background_service_worker() {
        let src = r#"{
            "manifest_version": 3,
            "name": "BG",
            "version": "1",
            "background": { "service_worker": "sw.js", "type": "module" }
        }"#;
        let m = parse_manifest(src).unwrap();
        let bg = m.background.unwrap();
        assert_eq!(bg.service_worker, "sw.js");
        assert!(bg.type_module);
    }

    #[test]
    fn parses_content_scripts_with_run_at() {
        let src = r#"{
            "manifest_version": 3,
            "name": "CS",
            "version": "1",
            "content_scripts": [{
                "matches": ["<all_urls>"],
                "js": ["c.js"],
                "run_at": "document_start",
                "all_frames": true
            }]
        }"#;
        let m = parse_manifest(src).unwrap();
        let cs = &m.content_scripts[0];
        assert_eq!(cs.matches, vec!["<all_urls>"]);
        assert_eq!(cs.run_at, ContentScriptRunAt::DocumentStart);
        assert!(cs.all_frames);
    }

    #[test]
    fn parses_action_with_icon_map() {
        let src = r#"{
            "manifest_version": 3,
            "name": "Action",
            "version": "1",
            "action": {
                "default_title": "Click me",
                "default_popup": "popup.html",
                "default_icon": { "16": "16.png", "32": "32.png" }
            }
        }"#;
        let m = parse_manifest(src).unwrap();
        let a = m.action.unwrap();
        assert_eq!(a.default_title.as_deref(), Some("Click me"));
        assert_eq!(a.default_icon.get("16").map(String::as_str), Some("16.png"));
    }
}
