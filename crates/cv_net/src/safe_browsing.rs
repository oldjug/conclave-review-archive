//! Safe Browsing v4 local hash-prefix list lookup.
//!
//! Real Safe Browsing keeps a local cache of 32-bit prefix hashes for
//! known malicious URLs. The client computes SHA-256 of canonical URL
//! variants and checks the first 4 bytes against the local list; a hit
//! triggers a full-hash verification round-trip against the API. This
//! module provides the prefix list data structure and the
//! canonicalization-based hashing layer.

use std::collections::HashSet;

fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    let mut h = cv_crypto::sha256::Sha256::new();
    h.update(input);
    h.finalize()
}

#[derive(Default, Debug, Clone)]
pub struct PrefixList {
    prefixes: HashSet<[u8; 4]>,
}

impl PrefixList {
    pub fn new() -> Self {
        Self {
            prefixes: HashSet::new(),
        }
    }

    /// Bulk-insert from a sorted little-endian-packed buffer of 4-byte
    /// prefixes (the wire format Safe Browsing serves).
    pub fn ingest_packed(&mut self, packed: &[u8]) {
        for chunk in packed.chunks_exact(4) {
            let mut k = [0u8; 4];
            k.copy_from_slice(chunk);
            self.prefixes.insert(k);
        }
    }

    /// Test whether `url` matches any known prefix. The URL is
    /// canonicalized per Safe Browsing rules (lowercased host, %hex
    /// folded, default port stripped) before hashing.
    pub fn looks_unsafe(&self, url: &str) -> bool {
        for canon in canonical_forms(url) {
            let hash = sha256_bytes(canon.as_bytes());
            let mut p = [0u8; 4];
            p.copy_from_slice(&hash[..4]);
            if self.prefixes.contains(&p) {
                return true;
            }
        }
        false
    }

    pub fn len(&self) -> usize {
        self.prefixes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prefixes.is_empty()
    }
}

/// Strip scheme + leading slashes, lowercase the host portion, and
/// produce the path-truncation set Safe Browsing computes.
pub fn canonical_forms(url: &str) -> Vec<String> {
    let body = url
        .splitn(2, "://")
        .nth(1)
        .unwrap_or(url)
        .trim_end_matches('/');
    let lower = body.to_ascii_lowercase();
    let (host, path) = match lower.find('/') {
        Some(i) => (&lower[..i], &lower[i..]),
        None => (lower.as_str(), ""),
    };
    let mut out = Vec::new();
    out.push(format!("{host}{path}"));
    let mut parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    while !parts.is_empty() {
        parts.pop();
        let joined = if parts.is_empty() {
            String::new()
        } else {
            format!("/{}", parts.join("/"))
        };
        out.push(format!("{host}{joined}"));
    }
    let host_labels: Vec<&str> = host.split('.').collect();
    for i in 1..host_labels.len().saturating_sub(1) {
        let sub = host_labels[i..].join(".");
        out.push(sub);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_hit_for_known_canonical_form() {
        let mut list = PrefixList::new();
        let hash = sha256_bytes(b"evil.example.com");
        list.ingest_packed(&hash[..4]);
        assert!(list.looks_unsafe("https://evil.example.com/"));
    }

    #[test]
    fn no_hit_for_unrelated_url() {
        let list = PrefixList::new();
        assert!(!list.looks_unsafe("https://example.com/"));
    }
}
