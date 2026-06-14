//! Origin — the tuple `(scheme, host, port)` per HTML §7.5.

use crate::Scheme;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Origin {
    pub scheme: Scheme,
    pub host: String,
    pub port: Option<u16>,
}

impl Origin {
    /// Per spec, an origin is a tuple origin (scheme/host/port) or an opaque
    /// origin. We model only tuple origins for now — opaque origins arrive
    /// with iframes / data: / sandboxed contexts.
    pub fn new(scheme: Scheme, host: String, port: Option<u16>) -> Self {
        Self { scheme, host, port }
    }

    /// "Same-origin" check: matching scheme/host/port.
    pub fn same_origin(&self, other: &Self) -> bool {
        self == other
    }
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}://{}", self.scheme.as_str(), self.host)?;
        if let Some(p) = self.port {
            if Some(p) != self.scheme.default_port() {
                write!(f, ":{p}")?;
            }
        }
        Ok(())
    }
}
