//! Origin-Private File System (OPFS) + Cache Storage durability + storage
//! estimate — the disk layer for M6.5.
//!
//! Three capabilities share this module because they all live under the same
//! per-user `%APPDATA%\Conclave\` tree and all reuse the corruption-safe,
//! atomic-write disk idioms first shipped for IndexedDB (`idb_persist`):
//!
//! * **OPFS** (`navigator.storage.getDirectory()`): a real on-disk directory
//!   tree under `…\Conclave\opfs\`. A `FileSystemDirectoryHandle` maps to a
//!   real subdirectory and a `FileSystemFileHandle` to a real file, so a write
//!   in one run is readable in the next (table-stakes persistence). The JS
//!   handle objects themselves are built in `main.rs` (they need the engine's
//!   `native_fn` / promise / byte helpers); this module owns *only* the
//!   filesystem mechanics: path resolution, name sanitization, create/read/
//!   write/remove/list, all bounds-checked and panic-free.
//!
//! * **Cache Storage durability** (`caches`): the in-memory `caches` store
//!   (cache-name → url → Response `Value`) is persisted to `…\Conclave\
//!   cache_storage\<cache>.tbc`, one file per named cache, and reloaded on
//!   startup so a PWA's offline cache survives a restart. Entry `Value`s are
//!   serialized with the SAME tagged-binary codec IndexedDB uses, via
//!   [`crate::idb_persist::encode_value_blob`] — we do NOT re-author it.
//!
//! * **`navigator.storage.estimate()`**: `usage` is the real byte total of
//!   every file under the origin's `opfs` + `cache_storage` + `indexeddb`
//!   dirs; `quota` is a sane large constant. So a write grows `usage`.
//!
//! ## Safety / fallback
//!
//! Every path is sandboxed under the origin root: entry names are sanitized
//! and `.` / `..` / separators are rejected, so no handle can escape the OPFS
//! tree. A missing / unwritable `%APPDATA%` (or, under `cargo test`, the lack
//! of an explicit test override) makes every op a silent no-op or `None` —
//! never a panic. Tests point the root at a temp dir via [`set_root_for_test`].

#![allow(unreachable_pub)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};

// Process-/thread-local override for the `Conclave` ROOT directory, used by
// tests so the real `%APPDATA%` is never touched. When `Some`, every subdir
// (`opfs`, `cache_storage`, `indexeddb`) resolves inside it.
thread_local! {
    static ROOT_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Point the OPFS / cache-storage / estimate layer at `root` for tests, so the
/// real per-user `%APPDATA%\Conclave` is never polluted. Pass `None` to
/// restore the default. Set BEFORE any op in the test.
pub fn set_root_for_test(root: Option<PathBuf>) {
    ROOT_OVERRIDE.with(|d| *d.borrow_mut() = root);
}

/// The `Conclave` root directory (NOT created here — subdir helpers create
/// the leaf they need). Honors the test override. Returns `None` if it can't
/// be resolved.
fn root_dir() -> Option<PathBuf> {
    if let Some(d) = ROOT_OVERRIDE.with(|d| d.borrow().clone()) {
        return Some(d);
    }
    // Under `cargo test`, with no explicit override the real `%APPDATA%` is
    // NEVER touched — every op degrades to a no-op (memory-only), exactly like
    // `idb_persist`. Durability tests opt in via `set_root_for_test`.
    #[cfg(test)]
    {
        None
    }
    #[cfg(not(test))]
    {
        let appdata = std::env::var_os("APPDATA")?;
        let mut p = PathBuf::from(appdata);
        p.push("Conclave");
        Some(p)
    }
}

/// Resolve (and create) a named subdir of the root, e.g. `opfs`,
/// `cache_storage`. `None` if the root is unavailable or mkdir fails.
fn subdir(name: &str) -> Option<PathBuf> {
    let mut p = root_dir()?;
    p.push(name);
    if std::fs::create_dir_all(&p).is_ok() {
        Some(p)
    } else {
        None
    }
}

/// The OPFS root directory (`…\Conclave\opfs`), created if missing.
pub fn opfs_root_dir() -> Option<PathBuf> {
    subdir("opfs")
}

/// The Cache Storage directory (`…\Conclave\cache_storage`), created if
/// missing.
pub fn cache_storage_dir() -> Option<PathBuf> {
    subdir("cache_storage")
}

// ── Name sanitization (sandbox guard) ────────────────────────────────────────

/// Validate a single OPFS path component. The WHATWG spec forbids `""`,
/// `"."`, `".."`, and any name containing `/`. We additionally reject `\`,
/// NUL, and a couple of Windows-illegal characters so the on-disk name is a
/// safe single component that can never traverse out of the OPFS tree.
/// Returns the name unchanged if valid, else `None`.
pub fn validate_component(name: &str) -> Option<&str> {
    if name.is_empty() || name == "." || name == ".." {
        return None;
    }
    if name
        .chars()
        .any(|c| matches!(c, '/' | '\\' | '\0' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
    {
        return None;
    }
    Some(name)
}

// ── OPFS filesystem ops (all sandboxed, panic-free) ──────────────────────────

/// A handle's on-disk path. `rel` is a list of already-validated components
/// from the OPFS root. Returns `None` if the root is unavailable.
fn opfs_path(rel: &[String]) -> Option<PathBuf> {
    let mut p = opfs_root_dir()?;
    for comp in rel {
        // Defense-in-depth: re-validate even though callers validate on entry.
        validate_component(comp)?;
        p.push(comp);
    }
    Some(p)
}

/// Ensure the directory at `rel` exists (create=true). Returns `false` (no
/// panic) if the root is unavailable or mkdir fails.
pub fn dir_ensure(rel: &[String]) -> bool {
    match opfs_path(rel) {
        Some(p) => std::fs::create_dir_all(&p).is_ok(),
        None => false,
    }
}

/// True if a directory exists at `rel`.
pub fn dir_exists(rel: &[String]) -> bool {
    match opfs_path(rel) {
        Some(p) => p.is_dir(),
        None => false,
    }
}

/// True if a regular file exists at `rel`.
pub fn file_exists(rel: &[String]) -> bool {
    match opfs_path(rel) {
        Some(p) => p.is_file(),
        None => false,
    }
}

/// Create an empty file at `rel` if it does not exist (mkdir its parent first).
/// Idempotent: an existing file is left untouched. Returns `false` on failure.
pub fn file_create(rel: &[String]) -> bool {
    let path = match opfs_path(rel) {
        Some(p) => p,
        None => return false,
    };
    if path.exists() {
        return path.is_file();
    }
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return false;
        }
    }
    // create_new would race; OpenOptions create is fine — we only get here when
    // the file is absent.
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .is_ok()
}

/// Read the whole file at `rel`, or `None` if absent / unreadable.
pub fn file_read(rel: &[String]) -> Option<Vec<u8>> {
    let path = opfs_path(rel)?;
    std::fs::read(&path).ok()
}

/// Overwrite (or create) the file at `rel` with `bytes`, atomically (tmp +
/// rename) so a crash mid-write can't truncate it. Creates parent dirs.
/// Returns `false` on any failure (no panic).
pub fn file_write_all(rel: &[String], bytes: &[u8]) -> bool {
    let path = match opfs_path(rel) {
        Some(p) => p,
        None => return false,
    };
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return false;
        }
    }
    atomic_write(&path, bytes)
}

/// Current size in bytes of the file at `rel` (0 if absent).
pub fn file_size(rel: &[String]) -> u64 {
    match opfs_path(rel) {
        Some(p) => std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0),
        None => 0,
    }
}

/// Truncate (or extend with zero bytes) the file at `rel` to `new_len`.
/// Returns `false` on failure.
pub fn file_truncate(rel: &[String], new_len: u64) -> bool {
    let path = match opfs_path(rel) {
        Some(p) => p,
        None => return false,
    };
    let mut data = std::fs::read(&path).unwrap_or_default();
    let n = new_len as usize;
    if data.len() > n {
        data.truncate(n);
    } else {
        data.resize(n, 0u8);
    }
    atomic_write(&path, &data)
}

/// Remove the entry (file or directory) named `name` from directory `parent`.
/// `recursive` allows removing a non-empty directory (WHATWG
/// `removeEntry(name, { recursive })`). Returns `false` if absent or on error.
pub fn remove_entry(parent: &[String], name: &str, recursive: bool) -> bool {
    if validate_component(name).is_none() {
        return false;
    }
    let mut rel: Vec<String> = parent.to_vec();
    rel.push(name.to_string());
    let path = match opfs_path(&rel) {
        Some(p) => p,
        None => return false,
    };
    if path.is_dir() {
        if recursive {
            std::fs::remove_dir_all(&path).is_ok()
        } else {
            std::fs::remove_dir(&path).is_ok()
        }
    } else if path.is_file() {
        std::fs::remove_file(&path).is_ok()
    } else {
        false
    }
}

/// List the immediate children of directory `rel` as `(name, is_dir)` pairs.
/// Empty vec if the dir is absent / unreadable.
pub fn dir_entries(rel: &[String]) -> Vec<(String, bool)> {
    let path = match opfs_path(rel) {
        Some(p) => p,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&path) {
        for entry in rd.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                let is_dir = entry.path().is_dir();
                out.push((name.to_string(), is_dir));
            }
        }
    }
    out
}

// ── Atomic write (shared) ────────────────────────────────────────────────────

/// Atomically write `bytes` to `path` via a sibling `.tmp` + rename. Returns
/// `true` on success. Best-effort; never panics.
fn atomic_write(path: &Path, bytes: &[u8]) -> bool {
    let mut tmp = path.to_path_buf();
    let mut name = tmp
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    tmp.set_file_name(name);
    if std::fs::write(&tmp, bytes).is_ok() {
        if std::fs::rename(&tmp, path).is_ok() {
            return true;
        }
        let _ = std::fs::remove_file(&tmp);
    }
    false
}

// ── Storage estimate ─────────────────────────────────────────────────────────

/// Recursively sum the byte size of every regular file under `dir`.
fn dir_size_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for entry in rd.flatten() {
                let p = entry.path();
                match entry.file_type() {
                    Ok(ft) if ft.is_dir() => stack.push(p),
                    Ok(ft) if ft.is_file() => {
                        total = total.saturating_add(
                            std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0),
                        );
                    }
                    _ => {}
                }
            }
        }
    }
    total
}

/// `navigator.storage.estimate().usage` — the real on-disk byte total of the
/// origin's persistent storage (OPFS + Cache Storage + IndexedDB). Reflects
/// actual usage: writing a file makes this grow. `0` if nothing is stored or
/// the root is unavailable.
pub fn usage_bytes() -> u64 {
    let root = match root_dir() {
        Some(r) => r,
        None => return 0,
    };
    let mut total = 0u64;
    for sub in ["opfs", "cache_storage", "indexeddb"] {
        let p = root.join(sub);
        if p.is_dir() {
            total = total.saturating_add(dir_size_bytes(&p));
        }
    }
    total
}

/// `navigator.storage.estimate().quota` — a sane large constant (10 GiB). We
/// do not enforce it; it exists so quota-aware code (`usage / quota`) behaves
/// sensibly, matching Chrome's "generous origin quota" model.
pub const QUOTA_BYTES: u64 = 10 * 1024 * 1024 * 1024;

// ── Cache Storage durability (one file per named cache) ──────────────────────
//
// Each named cache persists as `cache_storage\<sanitized>_<hash>.tbc`:
//   MAGIC "TBCAC" | u32 version | u32 entry-count | [ url-string | value-blob ]*
// where each value-blob is `idb_persist::encode_value_blob(response)`. A
// corrupt/truncated/version-mismatched file decodes to an EMPTY cache (the
// in-memory copy stays authoritative; the next put overwrites the file).

const CACHE_MAGIC: &[u8; 5] = b"TBCAC";
const CACHE_VERSION: u32 = 1;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Stable, filesystem-safe filename for a cache name (mirrors
/// `idb_persist::db_filename`): a sanitized human-readable prefix + FNV-1a
/// hash, so arbitrary cache names (which may contain `/`, `:` …) map to a
/// single safe component and collide only on a full 64-bit hash collision.
fn cache_filename(cache_name: &str) -> String {
    let h = fnv1a64(cache_name.as_bytes());
    let mut prefix = String::new();
    for c in cache_name.chars().take(32) {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            prefix.push(c);
        } else {
            prefix.push('_');
        }
    }
    format!("{prefix}_{h:016x}.tbc")
}

fn cache_file_path(cache_name: &str) -> Option<PathBuf> {
    let mut p = cache_storage_dir()?;
    p.push(cache_filename(cache_name));
    Some(p)
}

fn wr_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Serialize one named cache's entries (`url -> Response Value`) to bytes.
pub fn encode_cache(entries: &[(String, cv_js::Value)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(CACHE_MAGIC);
    wr_u32(&mut out, CACHE_VERSION);
    wr_u32(&mut out, entries.len() as u32);
    for (url, val) in entries {
        wr_u32(&mut out, url.len() as u32);
        out.extend_from_slice(url.as_bytes());
        let blob = crate::idb_persist::encode_value_blob(val);
        wr_u32(&mut out, blob.len() as u32);
        out.extend_from_slice(&blob);
    }
    out
}

/// Deserialize one named cache's entries. Returns an EMPTY vec for a missing /
/// wrong magic, version mismatch, or any truncation/corruption (never panics).
pub fn decode_cache(bytes: &[u8]) -> Vec<(String, cv_js::Value)> {
    fn inner(bytes: &[u8]) -> Option<Vec<(String, cv_js::Value)>> {
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize| -> Option<&[u8]> {
            let end = pos.checked_add(n)?;
            let s = bytes.get(*pos..end)?;
            *pos = end;
            Some(s)
        };
        if take(&mut pos, CACHE_MAGIC.len())? != CACHE_MAGIC {
            return None;
        }
        let rd_u32 = |pos: &mut usize| -> Option<u32> {
            let b = take(pos, 4)?;
            Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        if rd_u32(&mut pos)? != CACHE_VERSION {
            return None;
        }
        let count = rd_u32(&mut pos)? as usize;
        // A corrupt count can't claim more entries than there are bytes.
        if count > bytes.len() {
            return None;
        }
        let mut out = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            let url_len = rd_u32(&mut pos)? as usize;
            let url_bytes = take(&mut pos, url_len)?;
            let url = std::str::from_utf8(url_bytes).ok()?.to_string();
            let blob_len = rd_u32(&mut pos)? as usize;
            let blob = take(&mut pos, blob_len)?;
            let val = crate::idb_persist::decode_value_blob(blob)?;
            out.push((url, val));
        }
        Some(out)
    }
    inner(bytes).unwrap_or_default()
}

/// Persist one named cache to disk (atomic). No-op if the dir is unavailable.
pub fn persist_cache(cache_name: &str, entries: &[(String, cv_js::Value)]) {
    if let Some(path) = cache_file_path(cache_name) {
        let _ = atomic_write(&path, &encode_cache(entries));
    }
}

/// Load one named cache's entries from disk (empty if absent/corrupt).
pub fn load_cache(cache_name: &str) -> Vec<(String, cv_js::Value)> {
    let path = match cache_file_path(cache_name) {
        Some(p) => p,
        None => return Vec::new(),
    };
    match std::fs::read(&path) {
        Ok(bytes) => decode_cache(&bytes),
        Err(_) => Vec::new(),
    }
}

/// Delete one named cache's on-disk file (used by `caches.delete(name)`).
pub fn delete_cache(cache_name: &str) {
    if let Some(path) = cache_file_path(cache_name) {
        let _ = std::fs::remove_file(&path);
    }
}

/// Names of every persisted named cache (recovered from each file's payload is
/// not possible — the filename hashes the name — so we DON'T store the name in
/// the cache file; instead startup reload enumerates files and the caller's
/// in-memory store learns names from `caches.open`). For startup, we list the
/// raw filenames so the loader can pre-warm; but since we can't reverse the
/// hash, named-cache enumeration relies on `caches.open(name)` lazily loading.
/// This helper returns the on-disk filenames (for the test/usage probes).
pub fn persisted_cache_filenames() -> Vec<String> {
    let dir = match cache_storage_dir() {
        Some(d) => d,
        None => return Vec::new(),
    };
    let mut names = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("tbc") {
                if let Some(n) = p.file_name().and_then(|s| s.to_str()) {
                    names.push(n.to_string());
                }
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use cv_js::OrderedMap as HashMap;
    use cv_js::Value;

    fn temp_root(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("tb_opfs_test_{}_{}_{}", tag, std::process::id(), n));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn response_value(status: f64, body: &str) -> Value {
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("status".into(), Value::Number(status));
        m.insert("_body".into(), Value::str(body.to_string()));
        let mut headers: HashMap<String, Value> = HashMap::new();
        headers.insert(
            "content-type".into(),
            Value::str("text/plain".to_string()),
        );
        m.insert(
            "headers".into(),
            Value::Object(Rc::new(RefCell::new(headers))),
        );
        Value::Object(Rc::new(RefCell::new(m)))
    }

    #[test]
    fn value_blob_round_trip_via_idb_codec() {
        let v = response_value(200.0, "hello world");
        let bytes = crate::idb_persist::encode_value_blob(&v);
        let back = crate::idb_persist::decode_value_blob(&bytes).expect("decode");
        // Status + body round-trip through the shared codec.
        if let Value::Object(o) = &back {
            let b = o.borrow();
            assert!(matches!(b.get("status"), Some(Value::Number(n)) if (*n - 200.0).abs() < 1e-9));
            assert_eq!(
                b.get("_body").map(|v| v.to_display_string()).as_deref(),
                Some("hello world")
            );
        } else {
            panic!("not an object: {back:?}");
        }
    }

    #[test]
    fn cache_round_trip_and_corruption_safe() {
        let entries = vec![
            (
                "https://example.com/a".to_string(),
                response_value(200.0, "AAA"),
            ),
            (
                "https://example.com/b".to_string(),
                response_value(404.0, "missing"),
            ),
        ];
        let bytes = encode_cache(&entries);
        let back = decode_cache(&bytes);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].0, "https://example.com/a");
        assert_eq!(back[1].0, "https://example.com/b");
        assert_eq!(
            back[1].1.clone().to_display_string(),
            response_value(404.0, "missing").to_display_string()
        );

        // Corruption ⇒ empty, never a panic.
        assert!(decode_cache(b"").is_empty());
        assert!(decode_cache(b"not-a-cache-file").is_empty());
        let mut trunc = bytes.clone();
        trunc.truncate(trunc.len() / 2);
        assert!(decode_cache(&trunc).is_empty());
    }

    #[test]
    fn cache_persist_load_via_disk() {
        let root = temp_root("cache");
        set_root_for_test(Some(root.clone()));

        let entries = vec![(
            "https://example.com/data".to_string(),
            response_value(200.0, "persisted-body"),
        )];
        persist_cache("v1", &entries);

        let loaded = load_cache("v1");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "https://example.com/data");
        if let Value::Object(o) = &loaded[0].1 {
            assert_eq!(
                o.borrow().get("_body").map(|v| v.to_display_string()).as_deref(),
                Some("persisted-body")
            );
        } else {
            panic!("not object");
        }

        // A never-written cache loads empty.
        assert!(load_cache("never").is_empty());

        delete_cache("v1");
        assert!(load_cache("v1").is_empty());

        set_root_for_test(None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn opfs_write_read_remove_round_trip() {
        let root = temp_root("opfs_rw");
        set_root_for_test(Some(root.clone()));

        let rel = vec!["dir1".to_string(), "file.txt".to_string()];
        assert!(!file_exists(&rel));
        assert!(file_write_all(&rel, b"hello opfs"));
        assert!(file_exists(&rel));
        assert_eq!(file_read(&rel).as_deref(), Some(&b"hello opfs"[..]));
        assert_eq!(file_size(&rel), 10);

        // Truncate to 5 bytes.
        assert!(file_truncate(&rel, 5));
        assert_eq!(file_read(&rel).as_deref(), Some(&b"hello"[..]));

        // dir entries.
        let entries = dir_entries(&["dir1".to_string()]);
        assert!(entries.iter().any(|(n, d)| n == "file.txt" && !*d));

        // remove the file.
        assert!(remove_entry(&["dir1".to_string()], "file.txt", false));
        assert!(!file_exists(&rel));

        set_root_for_test(None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn opfs_persists_across_simulated_restart() {
        let root = temp_root("opfs_persist");
        // "Run 1": write a file.
        set_root_for_test(Some(root.clone()));
        let rel = vec!["sub".to_string(), "keep.bin".to_string()];
        assert!(file_write_all(&rel, b"durable-bytes"));
        set_root_for_test(None);

        // "Run 2": fresh resolution pointed at the same root reads it back.
        set_root_for_test(Some(root.clone()));
        assert!(file_exists(&rel));
        assert_eq!(file_read(&rel).as_deref(), Some(&b"durable-bytes"[..]));
        set_root_for_test(None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn name_validation_rejects_traversal() {
        assert!(validate_component("ok_name-1.txt").is_some());
        assert!(validate_component("").is_none());
        assert!(validate_component(".").is_none());
        assert!(validate_component("..").is_none());
        assert!(validate_component("a/b").is_none());
        assert!(validate_component("a\\b").is_none());
        assert!(validate_component("a:b").is_none());
    }

    #[test]
    fn usage_grows_after_write() {
        let root = temp_root("usage");
        set_root_for_test(Some(root.clone()));

        let before = usage_bytes();
        assert!(file_write_all(&["u.dat".to_string()], &vec![7u8; 1234]));
        let after = usage_bytes();
        assert!(
            after >= before + 1234,
            "usage must grow by at least the written bytes: before={before} after={after}"
        );

        set_root_for_test(None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_root_is_no_panic() {
        // Point root at a path under a file so subdir mkdir fails.
        let root = temp_root("noroot");
        std::fs::write(&root, b"x").unwrap();
        set_root_for_test(Some(root.join("sub")));
        assert!(!file_write_all(&["a.txt".to_string()], b"x"));
        assert!(file_read(&["a.txt".to_string()]).is_none());
        assert_eq!(usage_bytes(), 0);
        persist_cache("c", &[]); // must not panic
        assert!(load_cache("c").is_empty());
        set_root_for_test(None);
        let _ = std::fs::remove_file(&root);
    }
}
