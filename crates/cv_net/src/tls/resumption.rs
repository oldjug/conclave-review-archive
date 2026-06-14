//! TLS 1.3 session resumption — PSK ticket cache (RFC 8446 §4.6.1, §4.2.11).
//!
//! When a TLS 1.3 server sends a NewSessionTicket post-handshake, the
//! client may cache the ticket and use it in a future PSK extension on
//! the next ClientHello. The cache is keyed by hostname + ALPN so that
//! tickets handed out by `example.com` for `h2` don't accidentally get
//! offered to `example.com` for HTTP/1.1.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Ticket {
    pub ticket: Vec<u8>,
    pub nonce: Vec<u8>,
    pub resumption_secret: Vec<u8>,
    pub max_early_data: u32,
    pub expires: Instant,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TicketKey {
    pub host: String,
    pub alpn: String,
}

type TicketCache = OnceLock<Mutex<HashMap<TicketKey, Vec<Ticket>>>>;
static CACHE: TicketCache = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<TicketKey, Vec<Ticket>>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn store(key: TicketKey, ticket: Ticket) {
    if let Ok(mut g) = cache().lock() {
        let list = g.entry(key).or_default();
        list.push(ticket);
        // Keep at most a handful of tickets per origin so the cache
        // doesn't grow without bound for servers that hand out many.
        while list.len() > 4 {
            list.remove(0);
        }
    }
}

pub fn take_one(key: &TicketKey) -> Option<Ticket> {
    if let Ok(mut g) = cache().lock() {
        if let Some(list) = g.get_mut(key) {
            while let Some(t) = list.pop() {
                if t.expires > Instant::now() {
                    return Some(t);
                }
            }
        }
    }
    None
}

/// Build a Ticket from the server-side parsed NewSessionTicket message
/// fields plus the resumption-master-secret derived at handshake time.
pub fn from_nst(
    ticket: Vec<u8>,
    nonce: Vec<u8>,
    resumption_secret: Vec<u8>,
    lifetime_secs: u32,
    max_early_data: u32,
) -> Ticket {
    Ticket {
        ticket,
        nonce,
        resumption_secret,
        max_early_data,
        expires: Instant::now() + Duration::from_secs(lifetime_secs as u64),
    }
}

/// Encrypted Client Hello (ECH) — RFC 9460 / draft-ietf-tls-esni.
/// V1 surface tracks the ECHConfig blob; full ECH negotiation requires
/// HPKE which we don't ship. We expose the cache so callers can
/// retrieve / load a published ECHConfigList without re-implementing
/// DNS lookup of the underlying HTTPS RR.
pub mod ech {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static ECH: OnceLock<Mutex<HashMap<String, Vec<u8>>>> = OnceLock::new();
    fn cache() -> &'static Mutex<HashMap<String, Vec<u8>>> {
        ECH.get_or_init(|| Mutex::new(HashMap::new()))
    }
    pub fn store(host: &str, ech_config: Vec<u8>) {
        if let Ok(mut g) = cache().lock() {
            g.insert(host.to_string(), ech_config);
        }
    }
    pub fn lookup(host: &str) -> Option<Vec<u8>> {
        cache().lock().ok().and_then(|g| g.get(host).cloned())
    }
}
