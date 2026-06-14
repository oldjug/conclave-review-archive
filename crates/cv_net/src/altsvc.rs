//! `Alt-Svc` (RFC 7838) — alternative-service advertisement.
//!
//! Servers advertise `Alt-Svc: h3=":443"; ma=86400` to nudge clients
//! onto HTTP/3 the next time they connect. We parse and cache the
//! advertised alternates so a future connection to the same origin can
//! prefer the advertised protocol.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct AltSvcEntry {
    pub protocol: String,
    pub host: String,
    pub port: u16,
    pub expires: Instant,
}

/// Parse one or more `Alt-Svc` header values.
pub fn parse(value: &str) -> Vec<AltSvcEntry> {
    let mut out = Vec::new();
    for alt in value.split(',') {
        let alt = alt.trim();
        if alt.eq_ignore_ascii_case("clear") {
            continue;
        }
        let mut parts = alt.split(';').map(str::trim);
        let token = match parts.next() {
            Some(t) if !t.is_empty() => t,
            _ => continue,
        };
        let (proto, target) = match token.split_once('=') {
            Some(x) => x,
            None => continue,
        };
        let target = target.trim().trim_matches('"');
        let (host, port) = if let Some(rest) = target.strip_prefix(':') {
            (String::new(), rest.parse::<u16>().unwrap_or(0))
        } else if let Some((h, p)) = target.rsplit_once(':') {
            (h.to_string(), p.parse::<u16>().unwrap_or(0))
        } else {
            (target.to_string(), 0)
        };
        if port == 0 {
            continue;
        }
        let mut ma = 86_400u64;
        for kv in parts {
            if let Some((k, v)) = kv.split_once('=') {
                if k.eq_ignore_ascii_case("ma") {
                    if let Ok(n) = v.parse::<u64>() {
                        ma = n;
                    }
                }
            }
        }
        out.push(AltSvcEntry {
            protocol: proto.to_string(),
            host,
            port,
            expires: Instant::now() + Duration::from_secs(ma),
        });
    }
    out
}

/// Process-wide cache keyed by `(host, port)` of the origin that
/// advertised the alternates.
type AltSvcCache = OnceLock<Mutex<HashMap<(String, u16), Vec<AltSvcEntry>>>>;
static CACHE: AltSvcCache = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<(String, u16), Vec<AltSvcEntry>>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn store(origin_host: &str, origin_port: u16, entries: Vec<AltSvcEntry>) {
    if let Ok(mut g) = cache().lock() {
        g.insert((origin_host.to_string(), origin_port), entries);
    }
}

pub fn lookup(origin_host: &str, origin_port: u16) -> Vec<AltSvcEntry> {
    if let Ok(g) = cache().lock() {
        if let Some(list) = g.get(&(origin_host.to_string(), origin_port)) {
            let now = Instant::now();
            return list.iter().filter(|e| e.expires > now).cloned().collect();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_h3_basic() {
        let v = parse("h3=\":443\"; ma=86400");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].protocol, "h3");
        assert_eq!(v[0].port, 443);
    }

    #[test]
    fn parse_multi() {
        let v = parse("h2=\":443\"; ma=60, h3=\"alt.example:8443\"");
        assert_eq!(v.len(), 2);
        assert_eq!(v[1].host, "alt.example");
        assert_eq!(v[1].port, 8443);
    }
}
