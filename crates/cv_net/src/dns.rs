//! DNS — V1 uses the Windows resolver via `GetAddrInfoW`.
//!
//! Our own from-scratch DNS-over-UDP/TCP/HTTPS client lands in M0b; for the
//! M0a fetcher demo, deferring to the system resolver is the right call:
//! it gets us a working pipeline immediately and the public surface
//! (`resolve`) doesn't change when we swap in our resolver.
//!
//! ## M6.4 — transparent TTL cache + in-flight single-flight
//!
//! Every connect used to re-resolve through `GetAddrInfoW`. That is wasteful:
//! resolution is per-host and changes rarely, yet a busy page opens dozens of
//! sockets to the same handful of hosts. This module now wraps the raw system
//! call in a process-global TTL cache (default 60s, `CV_DNS_TTL_SECS`) and a
//! per-host single-flight gate so concurrent resolves of the SAME host collapse
//! into ONE underlying lookup (the rest wait on a `Condvar` and reuse the
//! result). Failures are negative-cached BRIEFLY (default 3s,
//! `CV_DNS_NEG_TTL_SECS`) to avoid hammering a dead host.
//!
//! The cache is keyed by HOST only (DNS is per-host); the requested `port` is
//! re-stamped onto the cached `IpAddr`s on a hit, so `resolve(host, port)`'s
//! public signature + result stay byte-identical to the uncached path. The
//! whole cache can be disabled with `CV_DNS_CACHE=0` as an escape hatch.

use crate::sys;
use std::collections::HashMap;
use std::ptr;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpAddr {
    V4([u8; 4]),
    V6([u8; 16]),
}

impl IpAddr {
    pub fn is_v4(&self) -> bool {
        matches!(self, Self::V4(_))
    }
}

impl std::fmt::Display for IpAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V4(o) => write!(f, "{}.{}.{}.{}", o[0], o[1], o[2], o[3]),
            Self::V6(o) => {
                for i in (0..16).step_by(2) {
                    if i > 0 {
                        f.write_str(":")?;
                    }
                    write!(f, "{:x}", (u16::from(o[i]) << 8) | u16::from(o[i + 1]))?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedAddr {
    pub ip: IpAddr,
    pub port: u16,
}

// ---------------------------------------------------------------------------
// Configuration (env-overridable, read once)
// ---------------------------------------------------------------------------

/// Master switch. When false, `resolve` always goes straight to the underlying
/// resolver with no caching and no single-flight — a clean escape hatch.
/// Default ON. Disable with `CV_DNS_CACHE=0`.
fn cache_enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("CV_DNS_CACHE")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off")))
            .unwrap_or(true)
    })
}

/// Positive-entry TTL. Default 60s. Override with `CV_DNS_TTL_SECS`.
fn positive_ttl() -> Duration {
    static F: OnceLock<Duration> = OnceLock::new();
    *F.get_or_init(|| {
        let secs = std::env::var("CV_DNS_TTL_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(60);
        Duration::from_secs(secs)
    })
}

/// Negative-entry TTL — keep failures briefly so we don't hammer a dead host,
/// but expire fast so transient failures self-heal. Default 3s. Override with
/// `CV_DNS_NEG_TTL_SECS`.
fn negative_ttl() -> Duration {
    static F: OnceLock<Duration> = OnceLock::new();
    *F.get_or_init(|| {
        let secs = std::env::var("CV_DNS_NEG_TTL_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(3);
        Duration::from_secs(secs)
    })
}

// ---------------------------------------------------------------------------
// Cache state
// ---------------------------------------------------------------------------

/// A cached resolution for a host. We store the bare `IpAddr`s (no port — DNS
/// is per-host) plus the expiry instant. A `Err` form negative-caches a
/// failure for the (short) negative TTL window.
#[derive(Debug, Clone)]
struct Entry {
    /// `Ok(ips)` on success, `Err(msg)` on a negative-cached failure.
    result: Result<Vec<IpAddr>, String>,
    /// When this entry stops being valid.
    expires: Instant,
}

impl Entry {
    fn is_fresh(&self, now: Instant) -> bool {
        self.expires > now
    }
}

/// Shared cache + single-flight bookkeeping, behind one mutex so the cache map
/// and the in-flight set move atomically. The `Condvar` wakes waiters when a
/// leader finishes (or fails) a lookup.
struct Shared {
    /// host -> cached entry (positive or negative).
    cache: HashMap<String, Entry>,
    /// Hosts with a resolve currently in flight (single-flight leaders).
    inflight: HashMap<String, ()>,
}

struct CacheState {
    shared: Mutex<Shared>,
    cv: Condvar,
}

fn state() -> &'static CacheState {
    static STATE: OnceLock<CacheState> = OnceLock::new();
    STATE.get_or_init(|| CacheState {
        shared: Mutex::new(Shared {
            cache: HashMap::new(),
            inflight: HashMap::new(),
        }),
        cv: Condvar::new(),
    })
}

// ---------------------------------------------------------------------------
// Test seam — an injectable underlying resolver + a call counter.
// ---------------------------------------------------------------------------
//
// Production always uses `resolve_uncached` (the real GetAddrInfoW path). Tests
// install a fake via `set_test_resolver` so they never touch the live network,
// and read `underlying_call_count` to assert the cache actually elides calls.

#[cfg(test)]
mod test_hook {
    //! Host-SCOPED test resolver. The real test suite runs all cv_net tests in
    //! parallel and several of them (`http1`, `websocket`) legitimately resolve
    //! `127.0.0.1` for loopback sockets. To avoid cross-test interference, the
    //! installed fake only intercepts hosts a test explicitly registers (the
    //! synthetic `*.test` names); every other host falls through to the REAL
    //! resolver and is NOT counted. So even concurrently-running loopback tests
    //! see normal resolution and never touch our counter.
    use super::IpAddr;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    type ResolverFn = Box<dyn Fn(&str) -> Result<Vec<IpAddr>, String> + Send + Sync>;

    struct Hook {
        /// Only hosts in this set are intercepted + counted.
        scoped_hosts: Vec<String>,
        resolver: Option<ResolverFn>,
    }

    static RESOLVER: OnceLock<Mutex<Hook>> = OnceLock::new();
    static CALLS: AtomicU64 = AtomicU64::new(0);

    fn slot() -> &'static Mutex<Hook> {
        RESOLVER.get_or_init(|| {
            Mutex::new(Hook {
                scoped_hosts: Vec::new(),
                resolver: None,
            })
        })
    }

    /// Install a fake resolver that intercepts ONLY `scoped_hosts`.
    pub fn set_test_resolver<F>(scoped_hosts: &[&str], f: F)
    where
        F: Fn(&str) -> Result<Vec<IpAddr>, String> + Send + Sync + 'static,
    {
        let mut g = slot().lock().unwrap_or_else(|e| e.into_inner());
        g.scoped_hosts = scoped_hosts.iter().map(|s| s.to_string()).collect();
        g.resolver = Some(Box::new(f));
    }

    pub fn clear_test_resolver() {
        let mut g = slot().lock().unwrap_or_else(|e| e.into_inner());
        g.scoped_hosts.clear();
        g.resolver = None;
    }

    /// Returns the override result iff `host` is a registered scoped host;
    /// `None` means "fall through to the real resolver". Bumps the call count
    /// only when the override actually fires (i.e. for a scoped host).
    pub fn maybe_invoke(host: &str) -> Option<Result<Vec<IpAddr>, String>> {
        let g = slot().lock().unwrap_or_else(|e| e.into_inner());
        match (&g.resolver, g.scoped_hosts.iter().any(|h| h == host)) {
            (Some(f), true) => {
                CALLS.fetch_add(1, Ordering::SeqCst);
                Some(f(host))
            }
            _ => None,
        }
    }

    pub fn underlying_call_count() -> u64 {
        CALLS.load(Ordering::SeqCst)
    }

    pub fn reset_call_count() {
        CALLS.store(0, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Resolve `host:port` to a list of addresses, transparently cached.
///
/// Behaviour matches the old uncached resolver bit-for-bit: same `Result`,
/// same address set, with the requested `port` stamped onto every entry. The
/// cache is purely an optimization — it elides redundant `GetAddrInfoW` calls
/// within the TTL window and collapses concurrent same-host resolves into one.
pub fn resolve(host: &str, port: u16) -> Result<Vec<ResolvedAddr>, String> {
    if !cache_enabled() {
        // Escape hatch: behave exactly like the legacy path (port already
        // stamped by `resolve_uncached`).
        return resolve_uncached(host, port);
    }
    resolve_host_cached(host).map(|ips| stamp(ips, port))
}

/// Stamp a per-host IP list with the requested port to form `ResolvedAddr`s.
fn stamp(ips: Vec<IpAddr>, port: u16) -> Vec<ResolvedAddr> {
    ips.into_iter().map(|ip| ResolvedAddr { ip, port }).collect()
}

/// Cached + single-flight resolution of a HOST to its bare `IpAddr`s.
///
/// Flow:
///  1. Fresh cache hit (positive or negative) -> return it, no lookup.
///  2. Miss/expired, nobody in flight -> become the leader, mark in-flight,
///     drop the lock, do the (possibly slow) underlying lookup, store the
///     result, clear in-flight, notify waiters.
///  3. Miss/expired, someone else in flight -> wait on the condvar, then
///     re-check the cache (the leader's result lands there).
fn resolve_host_cached(host: &str) -> Result<Vec<IpAddr>, String> {
    let st = state();
    let mut shared = st.shared.lock().unwrap_or_else(|e| e.into_inner());

    loop {
        let now = Instant::now();

        // (1) Fresh cache hit?
        if let Some(entry) = shared.cache.get(host) {
            if entry.is_fresh(now) {
                return entry.result.clone();
            }
        }

        // (2) Is a lookup for this host already in flight?
        if shared.inflight.contains_key(host) {
            // (3) Wait for the leader. Use a bounded wait so a leader that
            // somehow vanishes (panic mid-lookup is guarded, but be safe) can't
            // wedge us forever — on a spurious/timed wakeup we loop and re-check.
            let (g, _timeout) = st
                .cv
                .wait_timeout(shared, Duration::from_secs(5))
                .unwrap_or_else(|e| e.into_inner());
            shared = g;
            continue;
        }

        // We are the leader. Claim the in-flight slot and release the lock
        // across the (slow) underlying lookup so other hosts aren't blocked.
        shared.inflight.insert(host.to_string(), ());
        drop(shared);

        // Perform the real lookup OUTSIDE the lock. Guard against a panic in
        // the underlying resolver leaking the in-flight marker (which would
        // wedge every waiter) via a drop-guard that clears it + notifies.
        let mut guard = InflightGuard {
            host: host.to_string(),
            armed: true,
        };
        let result = resolve_uncached_host(host);
        // Disarm: we do the store + notify ourselves below under the lock, so
        // the guard's Drop must no-op (but still run, freeing its String).
        guard.armed = false;

        let mut shared = st.shared.lock().unwrap_or_else(|e| e.into_inner());
        let ttl = match &result {
            Ok(_) => positive_ttl(),
            Err(_) => negative_ttl(),
        };
        shared.cache.insert(
            host.to_string(),
            Entry {
                result: result.clone(),
                expires: Instant::now() + ttl,
            },
        );
        shared.inflight.remove(host);
        // Wake every waiter; each re-checks the cache and picks up our result.
        st.cv.notify_all();
        return result;
    }
}

/// Drop-guard that clears the in-flight marker + wakes waiters if the leader
/// unwinds before it can store a result. Normal path `mem::forget`s it.
struct InflightGuard {
    host: String,
    armed: bool,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let st = state();
        let mut shared = st.shared.lock().unwrap_or_else(|e| e.into_inner());
        shared.inflight.remove(&self.host);
        drop(shared);
        // Wake waiters so they re-check and retry (cache will be empty/stale,
        // so the next one becomes a fresh leader).
        st.cv.notify_all();
    }
}

/// Underlying host resolution returning bare `IpAddr`s (no port). In tests an
/// injected resolver may shadow the real system call.
fn resolve_uncached_host(host: &str) -> Result<Vec<IpAddr>, String> {
    #[cfg(test)]
    {
        if let Some(r) = test_hook::maybe_invoke(host) {
            return r;
        }
    }
    // Real path: resolve via the system resolver with a placeholder port, then
    // strip the port back off (the cache is per-host). Port 0 is fine for
    // GetAddrInfoW — we only consume the IPs.
    resolve_uncached(host, 0).map(|addrs| addrs.into_iter().map(|a| a.ip).collect())
}

/// The raw Windows resolver path (the pre-M6.4 `resolve`). Kept as a private
/// fn so the cache wraps it and tests can bypass it via the test hook. Always
/// makes a real `GetAddrInfoW` call.
fn resolve_uncached(host: &str, port: u16) -> Result<Vec<ResolvedAddr>, String> {
    crate::socket::ensure_wsa_started();
    let host_w: Vec<u16> = host.encode_utf16().chain(std::iter::once(0)).collect();
    let port_str = port.to_string();
    let port_w: Vec<u16> = port_str.encode_utf16().chain(std::iter::once(0)).collect();

    let hints = sys::addrinfoW {
        ai_flags: 0,
        ai_family: sys::AF_UNSPEC,
        ai_socktype: sys::SOCK_STREAM,
        ai_protocol: sys::IPPROTO_TCP,
        ai_addrlen: 0,
        ai_canonname: ptr::null_mut(),
        ai_addr: ptr::null_mut(),
        ai_next: ptr::null_mut(),
    };
    let mut result: *mut sys::addrinfoW = ptr::null_mut();
    let rc = unsafe {
        sys::GetAddrInfoW(
            host_w.as_ptr(),
            port_w.as_ptr(),
            &raw const hints,
            &raw mut result,
        )
    };
    if rc != 0 {
        return Err(format!("GetAddrInfoW failed: rc={rc}"));
    }

    let mut out = Vec::new();
    let mut cur = result;
    while !cur.is_null() {
        let ai = unsafe { &*cur };
        if let Some(addr) = addr_from_sockaddr(ai.ai_addr, ai.ai_family) {
            out.push(ResolvedAddr { ip: addr, port });
        }
        cur = ai.ai_next;
    }
    unsafe { sys::FreeAddrInfoW(result) };
    if out.is_empty() {
        return Err(format!("no addresses for {host}"));
    }
    Ok(out)
}

fn addr_from_sockaddr(sa: *const sys::sockaddr, family: i32) -> Option<IpAddr> {
    if sa.is_null() {
        return None;
    }
    match family {
        f if f == sys::AF_INET => {
            let sin = unsafe { &*sa.cast::<sys::sockaddr_in>() };
            Some(IpAddr::V4(sin.sin_addr))
        }
        f if f == sys::AF_INET6 => {
            let sin6 = unsafe { &*sa.cast::<sys::sockaddr_in6>() };
            Some(IpAddr::V6(sin6.sin6_addr))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};

    // The cache + test-resolver + call-counter are process-global, and the
    // env-driven config is read once via OnceLock. Tests that share that state
    // must run serially; a single mutex serializes them and also gives each a
    // clean slate.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Wipe all cache + single-flight state so a test starts clean. (Internal
    /// helper, test-only — production never clears the cache.)
    fn reset_cache() {
        let st = state();
        let mut shared = st.shared.lock().unwrap_or_else(|e| e.into_inner());
        shared.cache.clear();
        shared.inflight.clear();
    }

    fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4([a, b, c, d])
    }

    // ----- (existing) live resolver smoke test, network-gated -------------
    // Kept but only runs when explicitly opted-in, so the default `cargo test`
    // never touches real DNS.
    #[test]
    fn resolves_localhost_live() {
        if std::env::var("CV_DNS_LIVE_TEST").is_err() {
            return; // skip unless opted-in; do not hit the resolver in CI
        }
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        test_hook::clear_test_resolver();
        reset_cache();
        crate::socket::ensure_wsa_started();
        let addrs = resolve("localhost", 80).expect("resolve");
        assert!(!addrs.is_empty(), "localhost must resolve");
        assert!(addrs.iter().any(|a| match &a.ip {
            IpAddr::V4(o) => o == &[127, 0, 0, 1],
            IpAddr::V6(o) => o == &[0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }));
    }

    // (a) a 2nd resolve within TTL returns cached addrs WITHOUT re-calling the
    //     underlying resolver.
    #[test]
    fn second_resolve_within_ttl_is_cached() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_cache();
        test_hook::reset_call_count();
        test_hook::set_test_resolver(&["example.test"], |_host| Ok(vec![v4(1, 2, 3, 4)]));

        let a = resolve("example.test", 443).expect("first");
        let b = resolve("example.test", 443).expect("second");

        assert_eq!(test_hook::underlying_call_count(), 1, "only ONE underlying call");
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].ip, v4(1, 2, 3, 4));
        assert_eq!(b[0].ip, v4(1, 2, 3, 4));

        test_hook::clear_test_resolver();
    }

    // (b) after TTL expiry the underlying resolver is called again.
    #[test]
    fn expired_entry_re_resolves() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_cache();
        test_hook::reset_call_count();
        test_hook::set_test_resolver(&["expire.test"], |_host| Ok(vec![v4(9, 9, 9, 9)]));

        // First resolve populates the cache.
        let _ = resolve("expire.test", 80).expect("first");
        assert_eq!(test_hook::underlying_call_count(), 1);

        // Force expiry by back-dating the entry's expires instant into the past.
        {
            let st = state();
            let mut shared = st.shared.lock().unwrap_or_else(|e| e.into_inner());
            let e = shared.cache.get_mut("expire.test").expect("entry present");
            e.expires = Instant::now() - Duration::from_secs(1);
        }

        // Second resolve must miss and re-call.
        let _ = resolve("expire.test", 80).expect("second");
        assert_eq!(
            test_hook::underlying_call_count(),
            2,
            "expired entry forces a fresh underlying call"
        );

        test_hook::clear_test_resolver();
    }

    // (c) the requested port is re-stamped correctly on a hit.
    #[test]
    fn port_is_restamped_on_hit() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_cache();
        test_hook::reset_call_count();
        test_hook::set_test_resolver(&["port.test"], |_host| {
            Ok(vec![v4(5, 6, 7, 8), IpAddr::V6([0; 16])])
        });

        let a = resolve("port.test", 443).expect("first");
        // Different port on the cached hit — must be re-stamped, no new call.
        let b = resolve("port.test", 8080).expect("second");

        assert_eq!(test_hook::underlying_call_count(), 1, "cached, one call");
        assert!(a.iter().all(|r| r.port == 443), "first stamped 443");
        assert!(b.iter().all(|r| r.port == 8080), "hit re-stamped 8080");
        // Same IPs, just a different port.
        assert_eq!(a[0].ip, b[0].ip);
        assert_eq!(a[1].ip, b[1].ip);

        test_hook::clear_test_resolver();
    }

    // (d) negative-cache: a failure is cached briefly then retried.
    #[test]
    fn failure_is_negative_cached_then_retried() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_cache();
        test_hook::reset_call_count();
        test_hook::set_test_resolver(&["dead.test"], |_host| Err("nope".to_string()));

        let r1 = resolve("dead.test", 80);
        assert!(r1.is_err(), "first lookup fails");
        assert_eq!(test_hook::underlying_call_count(), 1);

        // Second call within the negative window: served from the negative
        // cache, NOT re-resolved.
        let r2 = resolve("dead.test", 80);
        assert!(r2.is_err(), "negative-cached failure");
        assert_eq!(
            test_hook::underlying_call_count(),
            1,
            "negative cache elides the second call"
        );

        // Expire the negative entry -> a retry must re-call the resolver.
        {
            let st = state();
            let mut shared = st.shared.lock().unwrap_or_else(|e| e.into_inner());
            let e = shared.cache.get_mut("dead.test").expect("neg entry present");
            e.expires = Instant::now() - Duration::from_secs(1);
        }
        let r3 = resolve("dead.test", 80);
        assert!(r3.is_err());
        assert_eq!(
            test_hook::underlying_call_count(),
            2,
            "expired negative entry forces a retry"
        );

        test_hook::clear_test_resolver();
    }

    // (e) concurrent same-host resolves trigger exactly ONE underlying call
    //     (single-flight / in-flight dedup).
    #[test]
    fn concurrent_same_host_single_flight() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_cache();
        test_hook::reset_call_count();

        // The resolver blocks until all threads are parked inside resolve(),
        // so without single-flight every thread would call it.
        static GATE: AtomicU64 = AtomicU64::new(0);
        GATE.store(0, Ordering::SeqCst);
        test_hook::set_test_resolver(&["burst.test"], |_host| {
            // Signal that the (single) underlying call has begun, then hold
            // briefly so concurrent callers must be waiting on the condvar.
            GATE.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(120));
            Ok(vec![v4(7, 7, 7, 7)])
        });

        const N: usize = 8;
        let barrier = Arc::new(Barrier::new(N));
        let mut handles = Vec::new();
        for _ in 0..N {
            let b = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                b.wait();
                resolve("burst.test", 443).expect("resolve")
            }));
        }
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Exactly one underlying call despite N concurrent resolves.
        assert_eq!(
            test_hook::underlying_call_count(),
            1,
            "single-flight collapses concurrent resolves into one call"
        );
        // Every caller got the same correct, port-stamped answer.
        for r in &results {
            assert_eq!(r.len(), 1);
            assert_eq!(r[0].ip, v4(7, 7, 7, 7));
            assert_eq!(r[0].port, 443);
        }

        test_hook::clear_test_resolver();
    }
}
