//! HSTS (HTTP Strict Transport Security) per RFC 6797.
//!
//! When a response over HTTPS carries
//!     Strict-Transport-Security: max-age=N[; includeSubDomains][; preload]
//! we remember (host, expiry, include_subdomains) so subsequent
//! navigations to `http://host/...` are upgraded to `https://host/...`
//! before the connection happens. This prevents downgrade attacks
//! against a known-HTTPS host.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// One stored HSTS entry.
#[derive(Clone, Debug)]
pub struct HstsEntry {
    /// UNIX seconds at which this entry expires. 0 = expired/deleted.
    pub expiry_unix: u64,
    /// True when the `includeSubDomains` flag was present.
    pub include_subdomains: bool,
}

fn store() -> &'static Mutex<HashMap<String, HstsEntry>> {
    static STORE: OnceLock<Mutex<HashMap<String, HstsEntry>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record an entry for `host` based on a response header value.
/// Returns true when the header was syntactically valid (and applied)
/// or false if it had no `max-age`.
pub fn record(host: &str, header_value: &str) -> bool {
    let Some((max_age, include_subdomains)) = parse(header_value) else {
        return false;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut map = store().lock().unwrap();
    if max_age == 0 {
        // max-age=0 is the spec-mandated way to forget an entry.
        map.remove(&host.to_ascii_lowercase());
    } else {
        map.insert(
            host.to_ascii_lowercase(),
            HstsEntry {
                expiry_unix: now.saturating_add(max_age),
                include_subdomains,
            },
        );
    }
    true
}

/// True when `host` (or any HSTS-protecting ancestor) currently
/// requires HTTPS. Callers should upgrade `http://host/...` → `https://...`
/// before opening a connection.
pub fn must_upgrade(host: &str) -> bool {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let lc = host.to_ascii_lowercase();
    // Preload list — bundled high-value Chromium HSTS-preload entries
    // (`includeSubDomains` = true, no `expires`). We keep the list
    // short to avoid bloating the binary; the runtime store still
    // absorbs entries from response headers.
    if is_preloaded(&lc) {
        return true;
    }
    let map = store().lock().unwrap();
    if let Some(e) = map.get(&lc) {
        if e.expiry_unix > now {
            return true;
        }
    }
    // Walk parent domains, honouring includeSubDomains entries.
    let mut s = lc.as_str();
    while let Some(dot) = s.find('.') {
        s = &s[dot + 1..];
        if is_preloaded(s) {
            return true;
        }
        if let Some(e) = map.get(s) {
            if e.expiry_unix > now && e.include_subdomains {
                return true;
            }
        }
    }
    false
}

/// True if `host` (lowercased) is on the bundled HSTS preload list.
/// Every entry on the list has `includeSubDomains` semantics — a hit
/// on an ancestor domain forces HTTPS for every subdomain.
fn is_preloaded(host: &str) -> bool {
    // High-traffic Chromium preload subset (every entry has
    // includeSubDomains=true on the upstream list). We deliberately
    // keep this small — the binary footprint of the full ~150k-entry
    // list would dwarf the engine. Additions land through response
    // headers at runtime.
    matches!(
        host,
        "google.com"
            | "youtube.com"
            | "gmail.com"
            | "googlemail.com"
            | "android.com"
            | "googleapis.com"
            | "gstatic.com"
            | "googletagmanager.com"
            | "doubleclick.net"
            | "facebook.com"
            | "twitter.com"
            | "github.com"
            | "githubusercontent.com"
            | "githubassets.com"
            | "stackoverflow.com"
            | "stackexchange.com"
            | "wikipedia.org"
            | "wikimedia.org"
            | "mozilla.org"
            | "mozilla.com"
            | "mozilla.net"
            | "cloudflare.com"
            | "cloudflareinsights.com"
            | "cloudfront.net"
            | "amazon.com"
            | "amazonaws.com"
            | "aws.amazon.com"
            | "apple.com"
            | "icloud.com"
            | "microsoft.com"
            | "live.com"
            | "office.com"
            | "office365.com"
            | "windows.com"
            | "outlook.com"
            | "msftauth.net"
            | "azure.com"
            | "azurewebsites.net"
            | "office.net"
            | "msn.com"
            | "bing.com"
            | "linkedin.com"
            | "twitch.tv"
            | "tiktok.com"
            | "reddit.com"
            | "pinterest.com"
            | "instagram.com"
            | "whatsapp.com"
            | "messenger.com"
            | "fb.com"
            | "fbcdn.net"
            | "paypal.com"
            | "paypalobjects.com"
            | "stripe.com"
            | "shopify.com"
            | "cloudflare.net"
            | "duckduckgo.com"
            | "1password.com"
            | "lastpass.com"
            | "bitwarden.com"
            | "protonmail.com"
            | "proton.me"
            | "signal.org"
            | "telegram.org"
            | "discord.com"
            | "discordapp.com"
            | "slack.com"
            | "zoom.us"
            | "dropbox.com"
            | "drive.google.com"
            | "docs.google.com"
            | "mail.google.com"
            | "accounts.google.com"
            | "play.google.com"
            | "developer.mozilla.org"
            | "letsencrypt.org"
            | "digicert.com"
            | "sectigo.com"
    )
}

/// Parse `max-age=N` and optional `includeSubDomains`. Returns
/// `Some((max_age, includes))` on a valid header.
fn parse(value: &str) -> Option<(u64, bool)> {
    let mut max_age: Option<u64> = None;
    let mut includes = false;
    for raw_part in value.split(';') {
        let part = raw_part.trim();
        if part.is_empty() {
            continue;
        }
        let lc = part.to_ascii_lowercase();
        if let Some(rest) = lc.strip_prefix("max-age") {
            let rest = rest.trim_start_matches([' ', '=']);
            let rest = rest.trim_matches('"').trim();
            max_age = rest.parse::<u64>().ok();
        } else if lc == "includesubdomains" {
            includes = true;
        }
        // preload / unknown tokens — accepted, ignored.
    }
    max_age.map(|m| (m, includes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple() {
        assert_eq!(parse("max-age=63072000"), Some((63072000, false)));
    }

    #[test]
    fn parse_includes() {
        assert_eq!(
            parse("max-age=63072000; includeSubDomains; preload"),
            Some((63072000, true))
        );
    }

    #[test]
    fn parse_quoted() {
        assert_eq!(parse("max-age=\"3600\""), Some((3600, false)));
    }

    #[test]
    fn parse_missing_max_age() {
        assert_eq!(parse("includeSubDomains"), None);
    }

    #[test]
    fn preload_list_forces_upgrade_without_record() {
        // No entry has been recorded for google.com in this test, but
        // the preload list should still demand HTTPS — directly and
        // via the includeSubDomains-style ancestor walk.
        assert!(must_upgrade("google.com"));
        assert!(must_upgrade("mail.google.com"));
        assert!(must_upgrade("github.com"));
        // A host that is NOT on the bundled list and has no record
        // should not be upgraded.
        assert!(!must_upgrade("nonexistent-domain-for-test.example"));
    }

    #[test]
    fn record_and_query() {
        record("test-host-a.example", "max-age=3600");
        assert!(must_upgrade("test-host-a.example"));
        // includeSubDomains propagates to subdomains.
        record("test-host-b.example", "max-age=3600; includeSubDomains");
        assert!(must_upgrade("sub.test-host-b.example"));
        // Without includeSubDomains, only the exact host is covered.
        assert!(!must_upgrade("sub.test-host-a.example"));
        // max-age=0 clears the entry.
        record("test-host-a.example", "max-age=0");
        assert!(!must_upgrade("test-host-a.example"));
    }
}
