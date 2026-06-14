//! Proxy support — manual proxy + PAC parsing.
//!
//! V1 surface: a ProxyConfig that gives a URL the proxy resolver to
//! forward to. Manual mode is an absolute proxy URL (e.g.
//! `http://localhost:8888`); PAC mode is a JavaScript text whose
//! `FindProxyForURL(url, host)` we evaluate via the cv_js engine when
//! caller plumbs an interp in. For the V1 builder we only honour the
//! manual path because PAC requires the JS engine which we don't
//! cross-link into cv_net.

use cv_url::Url;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyMode {
    Direct,
    Manual(Url),
    Pac(String),
}

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub mode: ProxyMode,
    pub bypass: Vec<String>,
}

impl ProxyConfig {
    pub fn direct() -> Self {
        Self {
            mode: ProxyMode::Direct,
            bypass: Vec::new(),
        }
    }

    pub fn manual(url: Url) -> Self {
        Self {
            mode: ProxyMode::Manual(url),
            bypass: Vec::new(),
        }
    }

    /// Determine the proxy URL to use for the given destination, or
    /// `None` if the request should bypass and connect directly.
    pub fn resolve(&self, dest: &Url) -> Option<&Url> {
        let host = dest.host.as_str();
        for pattern in &self.bypass {
            if host_matches(host, pattern) {
                return None;
            }
        }
        self.proxy_url()
    }

    fn proxy_url(&self) -> Option<&Url> {
        match &self.mode {
            ProxyMode::Direct => None,
            ProxyMode::Manual(u) => Some(u),
            ProxyMode::Pac(_) => None,
        }
    }
}

/// Match a host against a PAC-style pattern. `*.example.com` matches
/// every subdomain of example.com; bare `example.com` matches exactly.
pub fn host_matches(host: &str, pattern: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix("*.") {
        return host == suffix || host.ends_with(&format!(".{}", suffix));
    }
    host == pattern
}

/// Parse a `PROXY host:port; DIRECT; SOCKS5 ...` style PAC return
/// string. Each entry is normalized to an absolute URL when possible.
pub fn parse_pac_result(s: &str) -> Vec<Option<String>> {
    s.split(';')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut it = p.split_ascii_whitespace();
            let kind = it.next().unwrap_or("").to_ascii_uppercase();
            let target = it.next().unwrap_or("");
            match kind.as_str() {
                "DIRECT" => None,
                "PROXY" | "HTTP" => Some(format!("http://{target}")),
                "HTTPS" => Some(format!("https://{target}")),
                "SOCKS" | "SOCKS5" => Some(format!("socks5://{target}")),
                "SOCKS4" => Some(format!("socks4://{target}")),
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_url_passes_through() {
        let proxy = Url::parse("http://localhost:8080").unwrap();
        let cfg = ProxyConfig::manual(proxy);
        let dest = Url::parse("https://example.com/a").unwrap();
        assert!(cfg.resolve(&dest).is_some());
    }

    #[test]
    fn bypass_short_circuits() {
        let mut cfg = ProxyConfig::manual(Url::parse("http://localhost:8080").unwrap());
        cfg.bypass.push("*.internal".into());
        let dest = Url::parse("https://api.internal/").unwrap();
        assert!(cfg.resolve(&dest).is_none());
    }

    #[test]
    fn pac_result_direct_and_proxy() {
        let v = parse_pac_result("PROXY 10.0.0.1:8080; DIRECT");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].as_deref(), Some("http://10.0.0.1:8080"));
        assert_eq!(v[1], None);
    }
}
