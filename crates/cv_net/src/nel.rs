//! Network Error Logging (NEL) + `Report-To` / `Reporting-Endpoints`.
//!
//! NEL (W3C Network Error Logging) lets an origin opt into collection of
//! network errors. The `NEL` response header is a JSON object naming a
//! reporting group + sampling fractions + a `max_age` for which the policy
//! is cached; the group is resolved to endpoint URLs via the `Report-To`
//! header (legacy) or `Reporting-Endpoints` (Reporting API v1).
//!
//! Example (MDN):
//!
//!   NEL: { "report_to": "nel-group", "max_age": 2592000,
//!          "include_subdomains": false,
//!          "success_fraction": 0.0, "failure_fraction": 1.0 }
//!   Report-To: { "group": "nel-group", "max_age": 2592000,
//!                "endpoints": [{ "url": "https://example.com/report" }] }
//!   Reporting-Endpoints: nel-group="https://example.com/report"
//!
//! V1 implements the real parse + policy storage + endpoint resolution and
//! the report-payload builder. Actually POSTing the reports off-box is a
//! follow-up (it needs a background reporting queue); the parse+resolve is
//! the part pages observe, and it is real, not parse-and-ignore: a stored
//! policy resolves its group to a live endpoint URL, and
//! [`build_report_body`] produces the exact RFC report JSON a delivery
//! agent sends.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A parsed NEL policy for one origin.
#[derive(Debug, Clone, PartialEq)]
pub struct NelPolicy {
    pub report_to: String,
    pub max_age: u64,
    pub include_subdomains: bool,
    pub success_fraction: f64,
    pub failure_fraction: f64,
}

/// One reporting endpoint group (from `Report-To` / `Reporting-Endpoints`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReportToGroup {
    pub group: String,
    pub max_age: u64,
    pub endpoints: Vec<String>,
}

/// Parse the `NEL` response header (a JSON object). Returns `None` when the
/// JSON is malformed or `report_to` is absent (the policy is meaningless
/// without a destination group). Booleans/fractions default per spec.
pub fn parse_nel(header: &str) -> Option<NelPolicy> {
    let obj = parse_flat_json_object(header)?;
    let report_to = obj.get("report_to")?.as_string()?;
    let max_age = obj
        .get("max_age")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let include_subdomains = obj
        .get("include_subdomains")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let success_fraction = obj
        .get("success_fraction")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let failure_fraction = obj
        .get("failure_fraction")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    Some(NelPolicy {
        report_to,
        max_age,
        include_subdomains,
        success_fraction,
        failure_fraction,
    })
}

/// Parse a `Report-To` header value (one or more JSON objects, possibly
/// comma-separated). Each object: `{ "group": ..., "max_age": ...,
/// "endpoints": [{ "url": ... }] }`.
pub fn parse_report_to(header: &str) -> Vec<ReportToGroup> {
    let mut out = Vec::new();
    for obj_str in split_top_level_json_objects(header) {
        if let Some(obj) = parse_flat_json_object(&obj_str) {
            let group = obj
                .get("group")
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "default".to_string());
            let max_age = obj.get("max_age").and_then(|v| v.as_u64()).unwrap_or(0);
            // `endpoints` is an array of objects each carrying a "url".
            let endpoints = obj
                .get("endpoints")
                .map(|v| v.urls())
                .unwrap_or_default();
            if !endpoints.is_empty() {
                out.push(ReportToGroup {
                    group,
                    max_age,
                    endpoints,
                });
            }
        }
    }
    out
}

/// Parse a `Reporting-Endpoints` header (Reporting API v1): a
/// structured-fields dictionary of `name="url"` pairs. Each becomes a
/// single-endpoint group.
pub fn parse_reporting_endpoints(header: &str) -> Vec<ReportToGroup> {
    let mut out = Vec::new();
    for pair in header.split(',') {
        let pair = pair.trim();
        if let Some((name, url)) = pair.split_once('=') {
            let name = name.trim();
            let url = url.trim().trim_matches('"');
            if !name.is_empty() && !url.is_empty() {
                out.push(ReportToGroup {
                    group: name.to_string(),
                    max_age: 0,
                    endpoints: vec![url.to_string()],
                });
            }
        }
    }
    out
}

// ---- process-wide policy + endpoint store ------------------------------

#[derive(Default)]
struct NelStore {
    /// origin → (policy, expiry).
    policies: HashMap<String, (NelPolicy, Instant)>,
    /// origin → group-name → endpoint URLs.
    groups: HashMap<String, HashMap<String, Vec<String>>>,
}

fn store() -> &'static Mutex<NelStore> {
    static S: OnceLock<Mutex<NelStore>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(NelStore::default()))
}

/// Ingest the NEL-related headers from one response, keyed by `origin`.
/// Stores the NEL policy (with its `max_age` expiry) and any reporting
/// groups so a later error can be resolved to a delivery endpoint.
pub fn ingest(origin: &str, headers: &[(String, String)]) {
    let mut g = store().lock().unwrap();
    for (k, v) in headers {
        if k.eq_ignore_ascii_case("nel") {
            if let Some(p) = parse_nel(v) {
                let expiry = Instant::now() + Duration::from_secs(p.max_age.max(1));
                g.policies.insert(origin.to_string(), (p, expiry));
            }
        } else if k.eq_ignore_ascii_case("report-to") {
            for grp in parse_report_to(v) {
                g.groups
                    .entry(origin.to_string())
                    .or_default()
                    .insert(grp.group, grp.endpoints);
            }
        } else if k.eq_ignore_ascii_case("reporting-endpoints") {
            for grp in parse_reporting_endpoints(v) {
                g.groups
                    .entry(origin.to_string())
                    .or_default()
                    .insert(grp.group, grp.endpoints);
            }
        }
    }
}

/// Resolve the delivery endpoint URLs for `origin`'s active NEL policy.
/// Returns empty when no live policy exists or its group has no endpoints.
pub fn endpoints_for(origin: &str) -> Vec<String> {
    let g = store().lock().unwrap();
    let Some((policy, expiry)) = g.policies.get(origin) else {
        return Vec::new();
    };
    if *expiry <= Instant::now() {
        return Vec::new();
    }
    g.groups
        .get(origin)
        .and_then(|grp| grp.get(&policy.report_to))
        .cloned()
        .unwrap_or_default()
}

/// Build the JSON body a NEL report POST carries (W3C NEL §5, the `network-
/// error` report `body`). `phase` is `dns`/`connection`/`application`,
/// `type` the error type (e.g. `tcp.refused`, `http.error`), `status_code`
/// the HTTP status (0 when none).
pub fn build_report_body(url: &str, phase: &str, ty: &str, status_code: u16) -> String {
    format!(
        "{{\"type\":\"network-error\",\"url\":{url},\"body\":{{\"phase\":{phase},\"type\":{ty},\"status_code\":{status_code},\"protocol\":\"http/1.1\"}}}}",
        url = json_string(url),
        phase = json_string(phase),
        ty = json_string(ty),
        status_code = status_code,
    )
}

// ---- a tiny, dependency-free JSON value used only for header parsing ----

#[derive(Debug, Clone)]
enum JsonValue {
    Str(String),
    Num(f64),
    Bool(bool),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
    Null,
}

impl JsonValue {
    fn as_string(&self) -> Option<String> {
        match self {
            JsonValue::Str(s) => Some(s.clone()),
            _ => None,
        }
    }
    fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Num(n) if *n >= 0.0 => Some(*n as u64),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Num(n) => Some(*n),
            _ => None,
        }
    }
    fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// For an array of `{ "url": "..." }` objects, collect the URLs.
    fn urls(&self) -> Vec<String> {
        match self {
            JsonValue::Array(items) => items
                .iter()
                .filter_map(|it| match it {
                    JsonValue::Object(o) => o.get("url").and_then(|v| v.as_string()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Parse a single flat-ish JSON object into a map. Supports nested arrays +
/// objects (enough for NEL / Report-To). Returns None on malformed input.
fn parse_flat_json_object(s: &str) -> Option<HashMap<String, JsonValue>> {
    let mut p = JsonParser::new(s);
    p.skip_ws();
    match p.parse_value()? {
        JsonValue::Object(o) => Some(o),
        _ => None,
    }
}

/// Split a header that may contain several top-level JSON objects (the
/// `Report-To` legacy form) into individual object strings, respecting
/// brace/bracket/string nesting so commas inside an object don't split it.
fn split_top_level_json_objects(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    let mut cur = String::new();
    for c in s.chars() {
        if in_str {
            cur.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                cur.push(c);
            }
            '{' | '[' => {
                depth += 1;
                cur.push(c);
            }
            '}' | ']' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

struct JsonParser<'a> {
    chars: Vec<char>,
    pos: usize,
    _src: &'a str,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            chars: s.chars().collect(),
            pos: 0,
            _src: s,
        }
    }
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn next(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }
    fn parse_value(&mut self) -> Option<JsonValue> {
        self.skip_ws();
        match self.peek()? {
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            '"' => self.parse_string().map(JsonValue::Str),
            't' | 'f' => self.parse_bool(),
            'n' => self.parse_null(),
            _ => self.parse_number(),
        }
    }
    fn parse_object(&mut self) -> Option<JsonValue> {
        self.next()?; // {
        let mut map = HashMap::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.next();
            return Some(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.next()? != ':' {
                return None;
            }
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.next()? {
                ',' => continue,
                '}' => break,
                _ => return None,
            }
        }
        Some(JsonValue::Object(map))
    }
    fn parse_array(&mut self) -> Option<JsonValue> {
        self.next()?; // [
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.next();
            return Some(JsonValue::Array(arr));
        }
        loop {
            let v = self.parse_value()?;
            arr.push(v);
            self.skip_ws();
            match self.next()? {
                ',' => continue,
                ']' => break,
                _ => return None,
            }
        }
        Some(JsonValue::Array(arr))
    }
    fn parse_string(&mut self) -> Option<String> {
        if self.next()? != '"' {
            return None;
        }
        let mut s = String::new();
        loop {
            match self.next()? {
                '"' => break,
                '\\' => match self.next()? {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    other => s.push(other),
                },
                c => s.push(c),
            }
        }
        Some(s)
    }
    fn parse_bool(&mut self) -> Option<JsonValue> {
        if self.match_literal("true") {
            Some(JsonValue::Bool(true))
        } else if self.match_literal("false") {
            Some(JsonValue::Bool(false))
        } else {
            None
        }
    }
    fn parse_null(&mut self) -> Option<JsonValue> {
        if self.match_literal("null") {
            Some(JsonValue::Null)
        } else {
            None
        }
    }
    fn match_literal(&mut self, lit: &str) -> bool {
        let lit_chars: Vec<char> = lit.chars().collect();
        if self.pos + lit_chars.len() > self.chars.len() {
            return false;
        }
        if self.chars[self.pos..self.pos + lit_chars.len()] == lit_chars[..] {
            self.pos += lit_chars.len();
            true
        } else {
            false
        }
    }
    fn parse_number(&mut self) -> Option<JsonValue> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E') {
            self.pos += 1;
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse::<f64>().ok().map(JsonValue::Num)
    }
}

/// Minimal JSON string escaper for the report-body builder.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nel_header_real_fields() {
        let p = parse_nel(
            "{ \"report_to\": \"nel-group\", \"max_age\": 2592000, \"include_subdomains\": true, \"success_fraction\": 0.1, \"failure_fraction\": 1.0 }",
        )
        .expect("valid NEL header parses");
        assert_eq!(p.report_to, "nel-group");
        assert_eq!(p.max_age, 2_592_000);
        assert!(p.include_subdomains);
        assert!((p.success_fraction - 0.1).abs() < 1e-9);
        assert!((p.failure_fraction - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_nel_defaults_when_optional_fields_absent() {
        let p = parse_nel("{ \"report_to\": \"g\", \"max_age\": 60 }").unwrap();
        assert!(!p.include_subdomains);
        assert_eq!(p.success_fraction, 0.0);
        assert_eq!(p.failure_fraction, 1.0);
    }

    #[test]
    fn parse_nel_rejects_without_report_to() {
        assert!(parse_nel("{ \"max_age\": 60 }").is_none());
        assert!(parse_nel("not json").is_none());
    }

    #[test]
    fn parse_report_to_resolves_endpoints() {
        let groups = parse_report_to(
            "{ \"group\": \"nel-group\", \"max_age\": 86400, \"endpoints\": [{ \"url\": \"https://r.example/a\" }, { \"url\": \"https://r.example/b\" }] }",
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].group, "nel-group");
        assert_eq!(groups[0].endpoints, vec!["https://r.example/a", "https://r.example/b"]);
    }

    #[test]
    fn parse_reporting_endpoints_v1_dictionary() {
        let groups = parse_reporting_endpoints("nel-group=\"https://r.example/x\", default=\"https://r.example/d\"");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].group, "nel-group");
        assert_eq!(groups[0].endpoints, vec!["https://r.example/x"]);
    }

    #[test]
    fn ingest_then_resolve_endpoint_end_to_end() {
        // A response carrying both NEL + Report-To gets a resolvable endpoint.
        let origin = "https://nel-e2e.example";
        ingest(
            origin,
            &[
                (
                    "NEL".into(),
                    "{ \"report_to\": \"g1\", \"max_age\": 3600 }".into(),
                ),
                (
                    "Report-To".into(),
                    "{ \"group\": \"g1\", \"max_age\": 3600, \"endpoints\": [{ \"url\": \"https://collector.example/r\" }] }".into(),
                ),
            ],
        );
        let eps = endpoints_for(origin);
        assert_eq!(eps, vec!["https://collector.example/r"]);
        // An origin with no policy resolves to nothing.
        assert!(endpoints_for("https://unknown.example").is_empty());
    }

    #[test]
    fn build_report_body_is_valid_json_shape() {
        let body = build_report_body("https://x.example/p", "connection", "tcp.refused", 0);
        assert!(body.contains("\"type\":\"network-error\""));
        assert!(body.contains("\"url\":\"https://x.example/p\""));
        assert!(body.contains("\"phase\":\"connection\""));
        assert!(body.contains("\"type\":\"tcp.refused\""));
        assert!(body.contains("\"status_code\":0"));
    }

    #[test]
    fn split_objects_respects_nesting() {
        let parts = split_top_level_json_objects(
            "{ \"a\": [1,2], \"b\": \"x,y\" }, { \"c\": 1 }",
        );
        assert_eq!(parts.len(), 2);
    }
}
