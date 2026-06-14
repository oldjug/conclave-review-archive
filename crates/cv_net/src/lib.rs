//! `cv_net` — networking stack.
//!
//! M0a scope: synchronous DNS via `GetAddrInfoW`, blocking TCP via WinSock,
//! HTTP/1.1 client. Enough to pull a page over plain HTTP for the M0 demo.
//!
//! M0b will add TLS 1.3 (over our `cv_crypto`) for HTTPS, and M5+ adds
//! HTTP/2, HTTP/3, async, and connection pooling. The public surface
//! (`Client`, `Response`) is shaped so those land without churn.

#![allow(
    dead_code,
    missing_debug_implementations,
    unreachable_pub,
    unused_assignments,
    unused_imports
)]

pub mod altsvc;
pub mod cache;
pub mod cookies;
pub mod dns;
pub mod doh;
pub mod fetch_enforce;
pub mod hsts;
pub mod http1;
pub mod http2;
pub mod http3;
pub mod psl;
pub mod proxy;
pub mod quic;
pub mod safe_browsing;
pub mod security;
pub mod socket;
pub mod stun;
pub mod sys;
pub mod tls;
pub mod trust;
pub mod webrtc;
pub mod websocket;

pub use cache::{CacheControl, CachedEntry, HttpCache};
pub use cookies::{CookieJar, SameSite};
pub use http1::{Client, Request, Response};
pub use security::{
    CorsDecision, CspDirective, Origin, Policy, RequestMode, cors_decision, parse_csp,
    validate_cors_response,
};
pub use socket::{Socket, SocketError};
pub use websocket::{WebSocket, WsFrame};

#[derive(Debug)]
pub enum NetError {
    Dns(String),
    Socket(SocketError),
    Http(String),
    Url(String),
}

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dns(s) => write!(f, "dns: {s}"),
            Self::Socket(e) => write!(f, "socket: {e:?}"),
            Self::Http(s) => write!(f, "http: {s}"),
            Self::Url(s) => write!(f, "url: {s}"),
        }
    }
}

impl std::error::Error for NetError {}

impl From<SocketError> for NetError {
    fn from(e: SocketError) -> Self {
        Self::Socket(e)
    }
}
