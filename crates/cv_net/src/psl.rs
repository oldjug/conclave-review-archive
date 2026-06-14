//! Public Suffix List — registrable-domain / public-suffix computation
//! (Milestone 1.5 of the master design).
//!
//! Replaces the old last-two-labels heuristic, which returned `co.uk` and
//! `github.io` as "registrable domains" — wrong, and a genuine security bug:
//! cookie same-site scoping, HTTP-cache partitioning, and site isolation all
//! key on the registrable domain, so the heuristic let unrelated sites under a
//! shared suffix appear same-site.
//!
//! This implements the PSL algorithm exactly: the prevailing rule is the
//! longest matching rule, `*` matches any single label, `!` exception rules
//! win over wildcards and subtract their leftmost label, and the default rule
//! `*` makes the rightmost label a public suffix when nothing else matches.
//!
//! The rule TABLE below is a substantial, correct subset of publicsuffix.org's
//! list — the common ICANN multi-level suffixes plus the named private
//! suffixes. Single-label TLDs (`com`, `org`, `uk`, …) need NO rule: the
//! default rule handles them. Embedding the full ~9000-rule list is a
//! mechanical codegen step (`tools/psl_gen` over `public_suffix_list.dat`) that
//! does not change this algorithm or its results on the entries present.

use std::collections::HashSet;
use std::sync::OnceLock;

/// Multi-level public suffixes (ICANN + private). Single-label TLDs are omitted
/// — the default `*` rule covers them. Only suffixes the last-two-labels rule
/// would get WRONG need to be here.
const NORMAL: &[&str] = &[
    // United Kingdom
    "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "sch.uk", "ac.uk", "gov.uk", "nhs.uk",
    "police.uk", "mod.uk",
    // Australia
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au",
    // Japan
    "co.jp", "or.jp", "ne.jp", "ac.jp", "ad.jp", "ed.jp", "go.jp", "gr.jp", "lg.jp",
    // Brazil
    "com.br", "net.br", "org.br", "gov.br", "edu.br",
    // New Zealand
    "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz",
    // South Africa
    "co.za", "org.za", "net.za", "gov.za", "ac.za",
    // India
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in", "gov.in", "ac.in", "edu.in", "res.in",
    // China
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn", "ac.cn",
    // Mexico
    "com.mx", "org.mx", "gob.mx", "edu.mx", "net.mx",
    // Turkey
    "com.tr", "net.tr", "org.tr", "gov.tr", "edu.tr",
    // Korea
    "co.kr", "or.kr", "ne.kr", "re.kr", "go.kr", "ac.kr",
    // Singapore / Hong Kong / Indonesia
    "com.sg", "net.sg", "org.sg", "gov.sg", "edu.sg",
    "com.hk", "net.hk", "org.hk", "gov.hk", "edu.hk", "idv.hk",
    "co.id", "or.id", "ac.id", "go.id", "web.id",
    // Russia / Ukraine / Poland / Spain / Italy
    "com.ru", "net.ru", "org.ru",
    "com.ua", "net.ua", "org.ua",
    "com.pl", "net.pl", "org.pl", "gov.pl", "edu.pl",
    "com.es", "org.es", "gob.es", "edu.es",
    "edu.it", "gov.it",
    // Private suffixes (each subdomain is its own site) — the named cases + the
    // common hosting/CDN/SaaS suffixes that must be partitioned.
    "github.io", "githubusercontent.com", "gitlab.io",
    "s3.amazonaws.com", "s3-website.amazonaws.com",
    "cloudfront.net", "elasticbeanstalk.com",
    "herokuapp.com", "herokussl.com",
    "appspot.com", "firebaseapp.com", "web.app", "cloudfunctions.net",
    "azurewebsites.net", "blob.core.windows.net",
    "pages.dev", "workers.dev", "netlify.app", "vercel.app", "now.sh",
    "blogspot.com", "wordpress.com", "tumblr.com", "wixsite.com",
];

/// Wildcard rules `*.X`, stored as the parent `X`. Any single label before `X`
/// is itself a public suffix (e.g. `*.ck` → `foo.ck` is a public suffix).
const WILDCARD: &[&str] = &[
    "ck", "jm", "kw", "bd", "mm", "np", "pg", "ke", "er", "fj", "fk", "gu", "il", "kh", "mz",
    "ni", "platform.sh",
];

/// Exception rules `!X`, stored as `X`. They win over wildcards and subtract
/// their leftmost label (e.g. `!www.ck` → `www.ck` is registrable, not a suffix).
const EXCEPTION: &[&str] = &["www.ck", "city.kawasaki.jp", "city.kitakyushu.jp"];

fn normal() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| NORMAL.iter().copied().collect())
}
fn wildcard() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| WILDCARD.iter().copied().collect())
}
fn exception() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| EXCEPTION.iter().copied().collect())
}

/// Number of trailing labels that form the public suffix of `labels`.
fn public_suffix_labels(labels: &[&str]) -> usize {
    let n = labels.len();
    if n == 0 {
        return 0;
    }
    // 1. Exception rules win and subtract their leftmost label.
    for k in (2..=n).rev() {
        let suffix = labels[n - k..].join(".");
        if exception().contains(suffix.as_str()) {
            return k - 1;
        }
    }
    // 2. Longest matching normal-or-wildcard rule.
    let mut best = 0usize;
    for k in 1..=n {
        let suffix = labels[n - k..].join(".");
        if normal().contains(suffix.as_str()) {
            best = best.max(k);
        }
        if k >= 2 {
            // `*.parent` matches if the last k-1 labels equal `parent`.
            let parent = labels[n - k + 1..].join(".");
            if wildcard().contains(parent.as_str()) {
                best = best.max(k);
            }
        }
    }
    // 3. Default rule `*`: the rightmost label is a public suffix.
    best.max(1)
}

/// The public suffix (effective TLD) of `host`, lowercased input assumed.
/// For `www.example.co.uk` → `co.uk`; for `user.github.io` → `github.io`.
pub fn public_suffix(host: &str) -> String {
    let labels: Vec<&str> = host.split('.').filter(|l| !l.is_empty()).collect();
    if labels.is_empty() {
        return host.to_string();
    }
    let k = public_suffix_labels(&labels);
    labels[labels.len() - k..].join(".")
}

/// The registrable domain (eTLD+1) of `host`: the public suffix plus one more
/// label. For `www.example.co.uk` → `example.co.uk`; `user.github.io` →
/// `user.github.io`. If `host` IS a public suffix (or has no extra label), the
/// host is returned unchanged.
pub fn registrable_domain(host: &str) -> String {
    let labels: Vec<&str> = host.split('.').filter(|l| !l.is_empty()).collect();
    if labels.is_empty() {
        return host.to_string();
    }
    let ps = public_suffix_labels(&labels);
    let take = (ps + 1).min(labels.len());
    labels[labels.len() - take..].join(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_gtld_via_default_rule() {
        assert_eq!(registrable_domain("example.com"), "example.com");
        assert_eq!(registrable_domain("www.example.com"), "example.com");
        assert_eq!(registrable_domain("a.b.c.example.com"), "example.com");
        assert_eq!(public_suffix("www.example.com"), "com");
    }

    #[test]
    fn multi_level_cctld() {
        assert_eq!(registrable_domain("www.example.co.uk"), "example.co.uk");
        assert_eq!(public_suffix("www.example.co.uk"), "co.uk");
        assert_eq!(registrable_domain("foo.bar.com.au"), "bar.com.au");
        assert_eq!(registrable_domain("shop.example.co.jp"), "example.co.jp");
    }

    #[test]
    fn private_suffix_each_subdomain_is_its_own_site() {
        // The whole point: two GitHub Pages sites are NOT same-site.
        assert_eq!(registrable_domain("alice.github.io"), "alice.github.io");
        assert_eq!(registrable_domain("bob.github.io"), "bob.github.io");
        assert_ne!(
            registrable_domain("alice.github.io"),
            registrable_domain("bob.github.io"),
            "distinct GitHub Pages users must be distinct sites"
        );
        assert_eq!(
            registrable_domain("my-bucket.s3.amazonaws.com"),
            "my-bucket.s3.amazonaws.com"
        );
    }

    #[test]
    fn wildcard_rule() {
        // *.ck → foo.ck is itself a public suffix; bar.foo.ck is registrable.
        assert_eq!(public_suffix("bar.foo.ck"), "foo.ck");
        assert_eq!(registrable_domain("bar.foo.ck"), "bar.foo.ck");
    }

    #[test]
    fn exception_rule_beats_wildcard() {
        // !www.ck → www.ck is registrable despite *.ck.
        assert_eq!(public_suffix("www.ck"), "ck");
        assert_eq!(registrable_domain("www.ck"), "www.ck");
    }

    #[test]
    fn host_that_is_a_public_suffix() {
        assert_eq!(registrable_domain("co.uk"), "co.uk");
        assert_eq!(registrable_domain("com"), "com");
    }

    #[test]
    fn single_label_host() {
        assert_eq!(registrable_domain("localhost"), "localhost");
        assert_eq!(public_suffix("localhost"), "localhost");
    }
}
