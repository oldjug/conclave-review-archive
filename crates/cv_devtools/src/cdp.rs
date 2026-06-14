//! Chrome DevTools Protocol message router.
//!
//! Every CDP request is JSON of the form:
//!   { "id": N, "method": "Domain.method", "params": {...} }
//! Every CDP response is:
//!   { "id": N, "result": {...} }  OR  { "id": N, "error": {...} }
//! Domain events push:
//!   { "method": "Domain.event", "params": {...} }
//!
//! V1 handles request parsing + dispatch via a handler map. The
//! WebSocket transport (using `cv_net::websocket`) wraps this in a
//! follow-up.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CdpRequest {
    pub id: u64,
    pub method: String,
    pub params_raw: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdpResponse {
    Success { id: u64, result_raw: String },
    Error { id: u64, code: i32, message: String },
}

pub type Handler = Box<dyn Fn(&str) -> Result<String, (i32, String)> + Send + Sync>;

#[derive(Default)]
pub struct Router {
    handlers: HashMap<String, Handler>,
    /// Notification queue — handlers and external code push events here.
    pub events: Vec<(String, String)>, // (method, params_raw)
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("handlers", &self.handlers.len())
            .field("events", &self.events.len())
            .finish()
    }
}

impl Router {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, method: &str, h: Handler) {
        self.handlers.insert(method.to_string(), h);
    }

    pub fn dispatch(&self, req: &CdpRequest) -> CdpResponse {
        match self.handlers.get(&req.method) {
            Some(h) => match h(&req.params_raw) {
                Ok(result_raw) => CdpResponse::Success {
                    id: req.id,
                    result_raw,
                },
                Err((code, message)) => CdpResponse::Error {
                    id: req.id,
                    code,
                    message,
                },
            },
            None => CdpResponse::Error {
                id: req.id,
                code: -32601,
                message: format!("method not found: {}", req.method),
            },
        }
    }

    pub fn emit(&mut self, method: &str, params_raw: &str) {
        self.events
            .push((method.to_string(), params_raw.to_string()));
    }
}

/// Pre-built standard domains. The handlers here are stubs that
/// produce well-formed but empty results, so DevTools can probe
/// capabilities before the real implementations arrive.
pub fn standard_router() -> Router {
    let mut r = Router::new();
    r.register("Page.enable", Box::new(|_| Ok("{}".into())));
    r.register("Runtime.enable", Box::new(|_| Ok("{}".into())));
    r.register("DOM.enable", Box::new(|_| Ok("{}".into())));
    r.register("CSS.enable", Box::new(|_| Ok("{}".into())));
    r.register("Network.enable", Box::new(|_| Ok("{}".into())));
    r.register(
        "Browser.getVersion",
        Box::new(|_| Ok(r#"{"product":"Conclave/1.0","userAgent":"Conclave/1.0"}"#.into())),
    );
    r
}

/// Tiny CDP request parser — assumes well-formed JSON.
pub fn parse_request(json: &str) -> Option<CdpRequest> {
    let id = extract_number(json, "id")?;
    let method = extract_string(json, "method")?;
    let params_raw = extract_object(json, "params").unwrap_or_else(|| "{}".into());
    Some(CdpRequest {
        id: id as u64,
        method,
        params_raw,
    })
}

fn extract_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let after = &after[1..];
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

fn extract_number(json: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let end = after
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

fn extract_object(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('{') {
        return None;
    }
    let mut depth = 0;
    for (i, c) in after.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(after[..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_request_extracts_id_method_and_params() {
        let req = parse_request(r#"{"id":42,"method":"Page.enable","params":{}}"#).unwrap();
        assert_eq!(req.id, 42);
        assert_eq!(req.method, "Page.enable");
        assert_eq!(req.params_raw, "{}");
    }

    #[test]
    fn dispatch_unknown_method_returns_minus_32601() {
        let r = Router::new();
        let resp = r.dispatch(&CdpRequest {
            id: 1,
            method: "Foo.bar".into(),
            params_raw: "{}".into(),
        });
        match resp {
            CdpResponse::Error { code, .. } => assert_eq!(code, -32601),
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn dispatch_registered_method_returns_success() {
        let mut r = Router::new();
        r.register("Test.echo", Box::new(|p| Ok(p.to_string())));
        let resp = r.dispatch(&CdpRequest {
            id: 7,
            method: "Test.echo".into(),
            params_raw: r#"{"x":1}"#.into(),
        });
        match resp {
            CdpResponse::Success { id, result_raw } => {
                assert_eq!(id, 7);
                assert_eq!(result_raw, r#"{"x":1}"#);
            }
            _ => panic!("expected success"),
        }
    }

    #[test]
    fn standard_router_includes_browser_get_version() {
        let r = standard_router();
        let resp = r.dispatch(&CdpRequest {
            id: 1,
            method: "Browser.getVersion".into(),
            params_raw: "{}".into(),
        });
        match resp {
            CdpResponse::Success { result_raw, .. } => {
                assert!(result_raw.contains("Conclave"));
            }
            _ => panic!("expected success"),
        }
    }

    #[test]
    fn emit_queues_event() {
        let mut r = Router::new();
        r.emit("Page.frameNavigated", r#"{"url":"about:blank"}"#);
        assert_eq!(r.events.len(), 1);
        assert_eq!(r.events[0].0, "Page.frameNavigated");
    }
}
