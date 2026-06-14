//! IndexedDB durability — persist the in-memory `REG` database tree to disk
//! and reload it on startup (M6.2).
//!
//! The in-memory IndexedDB backend (`main.rs`, the `indexedDB` global) keeps
//! every database as a `cv_js::Value` tree:
//!
//! ```text
//! db = { _stores: { name -> { _keyPath, _autoKey, _rows: [{key, value}] } } }
//! ```
//!
//! This module makes that tree *durable*: after every mutation we write the
//! owning database to `%APPDATA%\Conclave\indexeddb\<db-name-hash>.idb`
//! (one file per database, atomic write-then-rename, mirroring the
//! `cookies.tsv` / `localStorage.tsv` idiom that ships next to it), and at
//! `indexedDB.open(name)` time we reload that file so a reopened database
//! already contains its persisted stores, rows, keyPath, autoIncrement flag
//! and the live autoIncrement counter.
//!
//! ## Encoding — hand-rolled tagged binary (NOT JSON)
//!
//! We deliberately do NOT reuse the engine's `JSON.stringify`/`parse`:
//!
//! * JSON has no `undefined`. IndexedDB key comparison (`idb_key_cmp`,
//!   spec §2.4.3) ranks `undefined` (rank 0) *below* `number`/`string`, and
//!   distinguishes it from `null` (rank 4). A JSON round-trip would erase the
//!   `Undefined`↔`Null` distinction and silently reorder keys after a
//!   restart. A tagged binary preserves the exact `cv_js::Value` variant.
//! * JSON.stringify also drops native functions / `undefined` *properties*,
//!   so a whole-DB stringify would not be reversible without re-running
//!   `make_store` anyway. We persist ONLY the data fields, by construction.
//!
//! The format is a tag byte + payload per value, the same shape as the
//! paint-cache disk layer (`paint_cache::disk`). A numeric key `5` reloads
//! as `Number(5.0)`, never `String("5")`, so `idb_key_cmp` ordering and key
//! equality survive a restart byte-for-value-identically.
//!
//! ## File layout
//!
//! ```text
//! MAGIC "TBIDB" | u32 version | <database payload>
//! ```
//!
//! A missing file ⇒ a fresh (empty) database, exactly as today. A corrupt,
//! truncated, or version-mismatched file is treated as an empty database and
//! never panics — durability is purely additive to the in-memory behavior.

// This is a binary-internal module (consumed only by `main.rs`), so its `pub`
// items are not reachable from outside the crate; the `pub` is for the in-crate
// call sites and tests. Same situation as the sibling `retained_dl` module.
#![allow(unreachable_pub)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use cv_js::OrderedMap as HashMap;
use cv_js::Value;

/// Magic bytes identifying a Conclave IndexedDB database file.
const MAGIC: &[u8; 5] = b"TBIDB";
/// On-disk schema version. Bump on ANY format change so an engine update
/// invalidates every stale file (they fail the version guard on load and are
/// treated as an empty DB, then overwritten on the next mutation).
const DISK_VERSION: u32 = 1;

// ── Value tags (tagged-binary encoding for the constrained IDB Value set) ────
const TAG_UNDEFINED: u8 = 0;
const TAG_NULL: u8 = 1;
const TAG_BOOL: u8 = 2;
const TAG_NUMBER: u8 = 3;
const TAG_STRING: u8 = 4;
const TAG_ARRAY: u8 = 5;
const TAG_OBJECT: u8 = 6;

// Process-global override for the IndexedDB directory, used by tests so the
// real `%APPDATA%` is never touched. When `Some`, `db_file_path` resolves
// inside it instead of `%APPDATA%\Conclave\indexeddb`.
thread_local! {
    static DIR_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Point the IndexedDB persistence layer at `dir` for tests (so the real
/// per-user directory is never polluted). Pass `None` to restore the default.
/// Set BEFORE any persist/load op in the test.
pub fn set_dir_for_test(dir: Option<PathBuf>) {
    DIR_OVERRIDE.with(|d| *d.borrow_mut() = dir);
}

/// The directory IndexedDB database files live in, creating it if missing.
/// Returns `None` if it can't be resolved/created (then every persist op is a
/// silent no-op and the DB stays memory-only — never a panic). Honors the
/// test override.
fn idb_dir() -> Option<PathBuf> {
    if let Some(d) = DIR_OVERRIDE.with(|d| d.borrow().clone()) {
        if std::fs::create_dir_all(&d).is_ok() {
            return Some(d);
        }
        return None;
    }
    // Under `cargo test`, the real per-user `%APPDATA%` is NEVER touched: with
    // no explicit override, persistence is a silent no-op (memory-only). This
    // keeps the existing in-memory `indexeddb_*` tests behaving exactly as
    // before, and forces durability tests to opt in via `set_dir_for_test`.
    #[cfg(test)]
    {
        return None;
    }
    #[cfg(not(test))]
    {
        let appdata = std::env::var_os("APPDATA")?;
        let mut p = PathBuf::from(appdata);
        p.push("Conclave");
        p.push("indexeddb");
        if std::fs::create_dir_all(&p).is_ok() {
            Some(p)
        } else {
            None
        }
    }
}

/// A stable, filesystem-safe filename for a database name. The raw name can
/// contain characters illegal in a path (`/`, `:`, …), so we hash it (FNV-1a
/// 64-bit) and keep a short sanitized prefix for human readability. Two
/// distinct names collide only on a full 64-bit hash collision.
fn db_filename(db_name: &str) -> String {
    let h = fnv1a64(db_name.as_bytes());
    let mut prefix = String::new();
    for c in db_name.chars().take(32) {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            prefix.push(c);
        } else {
            prefix.push('_');
        }
    }
    format!("{prefix}_{h:016x}.idb")
}

/// Absolute path of the file backing database `db_name`, or `None` if the
/// directory is unavailable.
fn db_file_path(db_name: &str) -> Option<PathBuf> {
    let mut p = idb_dir()?;
    p.push(db_filename(db_name));
    Some(p)
}

/// Engine-internal property keys carry a `\u{1}` sentinel prefix (mirrors
/// `cv_js::interp::is_internal_key`, inlined here to avoid widening the cv_js
/// public surface). Such keys (e.g. the `[[Prototype]]` slot) are never part
/// of JS-observable stored data, so we drop them from persisted objects.
fn is_internal_key(k: &str) -> bool {
    k.starts_with('\u{1}')
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ── Binary writer helpers (little-endian, length-prefixed) ───────────────────
fn wr_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn wr_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn wr_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn wr_str(out: &mut Vec<u8>, s: &str) {
    wr_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

// ── Binary reader (bounds-checked; any failure ⇒ None ⇒ empty DB) ────────────
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f64(&mut self) -> Option<f64> {
        let b = self.take(8)?;
        Some(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn str(&mut self) -> Option<String> {
        let len = self.u32()? as usize;
        let b = self.take(len)?;
        std::str::from_utf8(b).ok().map(|s| s.to_string())
    }
    fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }
}

/// Recursion guard so a maliciously / accidentally deeply-nested value can't
/// blow the stack on load. IndexedDB values are app data; 128 levels is far
/// beyond any real structure clone tree.
const MAX_DEPTH: u32 = 128;

// ── Reusable single-Value blob codec (shared with `caches` durability) ───────
//
// The Cache Storage durability layer (`caches`, M6.5) stores a Response-shaped
// `cv_js::Value` tree (status/headers/body — strings + numbers, the same
// constrained set IndexedDB stores). Rather than re-author the tagged-binary
// serializer there, it reuses the SAME `encode_value`/`decode_value` below via
// these two thin wrappers, which add their own magic+version envelope so a
// cache blob can never be mistaken for an IDB database file (and a future
// format bump invalidates stale blobs identically).

/// Magic bytes for a standalone single-`Value` blob (cache entries, etc.).
const VALUE_BLOB_MAGIC: &[u8; 5] = b"TBVAL";

/// Encode a single `cv_js::Value` into a self-describing blob (magic + version
/// + value). Reuses the IndexedDB `encode_value` so the constrained value set
/// (Undefined/Null/Bool/Number/String/Array/Object) round-trips identically.
pub fn encode_value_blob(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(VALUE_BLOB_MAGIC);
    wr_u32(&mut out, DISK_VERSION);
    encode_value(&mut out, v, 0);
    out
}

/// Decode a single-`Value` blob produced by [`encode_value_blob`]. Returns
/// `None` for a missing/wrong magic, a version mismatch, or any
/// truncation/corruption (caller then treats it as absent — never a panic).
pub fn decode_value_blob(bytes: &[u8]) -> Option<Value> {
    let mut r = Reader::new(bytes);
    let magic = r.take(VALUE_BLOB_MAGIC.len())?;
    if magic != VALUE_BLOB_MAGIC {
        return None;
    }
    let disk_ver = r.u32()?;
    if disk_ver != DISK_VERSION {
        return None;
    }
    decode_value(&mut r, 0)
}

// ── Value <-> bytes (the constrained IndexedDB value set) ────────────────────

/// Serialize one `cv_js::Value` (constrained to the IDB-storable set) into
/// `out`. Functions / BigInt etc. are not part of stored data; they are
/// encoded as `Null` so the format stays total (this never happens for real
/// stored keys/values, which the in-memory backend already constrains).
fn encode_value(out: &mut Vec<u8>, v: &Value, depth: u32) {
    if depth >= MAX_DEPTH {
        out.push(TAG_NULL);
        return;
    }
    match v {
        Value::Undefined | Value::Hole => out.push(TAG_UNDEFINED),
        Value::Null => out.push(TAG_NULL),
        Value::Bool(b) => {
            out.push(TAG_BOOL);
            out.push(u8::from(*b));
        }
        Value::Number(n) => {
            out.push(TAG_NUMBER);
            wr_f64(out, *n);
        }
        Value::String(s) => {
            out.push(TAG_STRING);
            wr_str(out, s);
        }
        Value::Array(a) => {
            out.push(TAG_ARRAY);
            let items = a.borrow();
            wr_u32(out, items.len() as u32);
            for item in items.iter() {
                encode_value(out, item, depth + 1);
            }
        }
        Value::Object(o) => {
            out.push(TAG_OBJECT);
            let map = o.borrow();
            // Persist only own, non-internal, data-valued properties — this is
            // exactly the JSON.stringify property filter, so stored object
            // *values* round-trip with the same observable shape. Native
            // functions never appear inside stored IDB values, but we drop
            // them defensively to keep the encoding total.
            let mut entries: Vec<(&String, &Value)> = Vec::new();
            for (k, val) in map.iter() {
                if is_internal_key(k) {
                    continue;
                }
                if matches!(
                    val,
                    Value::NativeFunction(_)
                        | Value::Function(_)
                        | Value::BcClosure(_)
                        | Value::Undefined
                ) {
                    continue;
                }
                entries.push((k, val));
            }
            wr_u32(out, entries.len() as u32);
            for (k, val) in entries {
                wr_str(out, k);
                encode_value(out, val, depth + 1);
            }
        }
        // BigInt / functions are not IDB-storable values; encode as Null so the
        // format stays total. (Unreachable for real stored data.)
        _ => out.push(TAG_NULL),
    }
}

/// Deserialize one `cv_js::Value` from `r`. Returns `None` on any malformed /
/// truncated input so the whole load degrades to an empty DB.
fn decode_value(r: &mut Reader, depth: u32) -> Option<Value> {
    if depth >= MAX_DEPTH {
        return None;
    }
    let tag = r.u8()?;
    Some(match tag {
        TAG_UNDEFINED => Value::Undefined,
        TAG_NULL => Value::Null,
        TAG_BOOL => Value::Bool(r.u8()? != 0),
        TAG_NUMBER => Value::Number(r.f64()?),
        TAG_STRING => Value::str(r.str()?),
        TAG_ARRAY => {
            let n = r.u32()? as usize;
            // Guard against an absurd length from a corrupt file: an array
            // header can't claim more items than there are bytes remaining
            // (each item is ≥ 1 byte).
            if n > r.bytes.len() {
                return None;
            }
            let mut items = Vec::with_capacity(n.min(1024));
            for _ in 0..n {
                items.push(decode_value(r, depth + 1)?);
            }
            Value::Array(Rc::new(RefCell::new(items)))
        }
        TAG_OBJECT => {
            let n = r.u32()? as usize;
            if n > r.bytes.len() {
                return None;
            }
            let mut map: HashMap<String, Value> = HashMap::new();
            for _ in 0..n {
                let k = r.str()?;
                let val = decode_value(r, depth + 1)?;
                map.insert(k, val);
            }
            Value::Object(Rc::new(RefCell::new(map)))
        }
        _ => return None,
    })
}

// ── A database's persistable state ───────────────────────────────────────────

/// The serializable projection of one IndexedDB database. Captures everything
/// JS-observable state depends on after a restart: the version, and per store
/// the keyPath, autoIncrement flag, the next autoIncrement counter, and every
/// row's typed key + value.
// NOTE: `cv_js::Value` deliberately does not implement `PartialEq` (object
// identity vs structural equality is ambiguous for JS values), so these
// snapshot structs cannot derive it. Tests compare via `snapshots_equal` /
// `values_equal` below, which do a structural deep-compare.
#[derive(Debug, Clone)]
pub struct DbSnapshot {
    /// The database name. Stored in the payload (the filename is a lossy hash)
    /// so `indexedDB.databases()` can enumerate a persisted-but-not-yet-opened
    /// database with its real name.
    pub name: String,
    pub version: u32,
    pub stores: Vec<StoreSnapshot>,
}

#[derive(Debug, Clone)]
pub struct StoreSnapshot {
    pub name: String,
    pub key_path: Option<String>,
    pub auto_increment: bool,
    /// Next autoIncrement integer (the live counter — 1 for a fresh store).
    pub auto_key: f64,
    pub rows: Vec<RowSnapshot>,
}

#[derive(Debug, Clone)]
pub struct RowSnapshot {
    pub key: Value,
    pub value: Value,
}

impl DbSnapshot {
    /// Encode this snapshot to the on-disk byte format (magic + version +
    /// payload). The result is what [`atomic_write`] persists.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        wr_u32(&mut out, DISK_VERSION);
        wr_str(&mut out, &self.name);
        wr_u32(&mut out, self.version);
        wr_u32(&mut out, self.stores.len() as u32);
        for s in &self.stores {
            wr_str(&mut out, &s.name);
            match &s.key_path {
                Some(kp) => {
                    out.push(1);
                    wr_str(&mut out, kp);
                }
                None => out.push(0),
            }
            out.push(u8::from(s.auto_increment));
            wr_f64(&mut out, s.auto_key);
            wr_u32(&mut out, s.rows.len() as u32);
            for row in &s.rows {
                encode_value(&mut out, &row.key, 0);
                encode_value(&mut out, &row.value, 0);
            }
        }
        out
    }

    /// Decode a snapshot from on-disk bytes. Returns `None` for a missing
    /// magic, a version mismatch, or any truncation/corruption — the caller
    /// then treats the database as empty (never a panic).
    pub fn from_bytes(bytes: &[u8]) -> Option<DbSnapshot> {
        let mut r = Reader::new(bytes);
        let magic = r.take(MAGIC.len())?;
        if magic != MAGIC {
            return None;
        }
        let disk_ver = r.u32()?;
        if disk_ver != DISK_VERSION {
            return None;
        }
        let name = r.str()?;
        let version = r.u32()?;
        let store_count = r.u32()? as usize;
        if store_count > bytes.len() {
            return None;
        }
        let mut stores = Vec::with_capacity(store_count.min(1024));
        for _ in 0..store_count {
            let name = r.str()?;
            let has_kp = r.u8()?;
            let key_path = match has_kp {
                0 => None,
                1 => Some(r.str()?),
                _ => return None,
            };
            let auto_increment = r.u8()? != 0;
            let auto_key = r.f64()?;
            let row_count = r.u32()? as usize;
            if row_count > bytes.len() {
                return None;
            }
            let mut rows = Vec::with_capacity(row_count.min(4096));
            for _ in 0..row_count {
                let key = decode_value(&mut r, 0)?;
                let value = decode_value(&mut r, 0)?;
                rows.push(RowSnapshot { key, value });
            }
            stores.push(StoreSnapshot {
                name,
                key_path,
                auto_increment,
                auto_key,
                rows,
            });
        }
        // Trailing bytes are tolerated (forward-compat) but the structured
        // prefix must have parsed cleanly. We do not require `at_end`.
        let _ = r.at_end();
        Some(DbSnapshot {
            name,
            version,
            stores,
        })
    }
}

// ── REG <-> snapshot bridges ─────────────────────────────────────────────────
//
// These translate between the live `cv_js::Value` database tree (as built by
// `make_database` / `make_store` in main.rs) and a `DbSnapshot`. They read /
// write only the data fields; the native methods (`get`/`put`/…) are owned by
// main.rs and re-attached there on load.

/// Build a [`DbSnapshot`] from a live database `Value` (the `Value::Object`
/// stored in `REG`). Returns `None` if the value is not a well-formed database
/// object (then nothing is persisted).
pub fn snapshot_from_db(db_name: &str, db: &Value) -> Option<DbSnapshot> {
    let db_obj = match db {
        Value::Object(o) => o,
        _ => return None,
    };
    let db_b = db_obj.borrow();
    let version = match db_b.get("version") {
        Some(Value::Number(n)) => *n as u32,
        _ => 1,
    };
    let stores_val = db_b.get("_stores")?.clone();
    let stores_obj = match &stores_val {
        Value::Object(o) => o,
        _ => return None,
    };
    let mut stores = Vec::new();
    for (name, store_val) in stores_obj.borrow().iter() {
        let store_obj = match store_val {
            Value::Object(o) => o,
            _ => continue,
        };
        let s = store_obj.borrow();
        let key_path = match s.get("_keyPath") {
            Some(Value::String(kp)) => Some(kp.to_string()),
            _ => None,
        };
        let auto_increment = matches!(s.get("autoIncrement"), Some(Value::Bool(true)));
        let auto_key = match s.get("_autoKey") {
            Some(Value::Number(n)) => *n,
            _ => 1.0,
        };
        let mut rows = Vec::new();
        if let Some(Value::Array(a)) = s.get("_rows") {
            for row in a.borrow().iter() {
                if let Value::Object(ro) = row {
                    let rm = ro.borrow();
                    let key = rm.get("key").cloned().unwrap_or(Value::Undefined);
                    let value = rm.get("value").cloned().unwrap_or(Value::Undefined);
                    rows.push(RowSnapshot { key, value });
                }
            }
        }
        stores.push(StoreSnapshot {
            name: name.clone(),
            key_path,
            auto_increment,
            auto_key,
            rows,
        });
    }
    Some(DbSnapshot {
        name: db_name.to_string(),
        version,
        stores,
    })
}

// ── Disk I/O ─────────────────────────────────────────────────────────────────

/// Atomically write `bytes` to `path` via a sibling `.tmp` file + rename, so a
/// crash mid-write can't corrupt the database file. Best-effort: any I/O error
/// is swallowed (the in-memory copy remains authoritative; we retry on the next
/// mutation).
fn atomic_write(path: &std::path::Path, bytes: &[u8]) {
    let mut tmp = path.to_path_buf();
    let mut name = tmp
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    tmp.set_file_name(name);
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Persist a live database `Value` (its `_stores`/rows/keyPath/autoKey/version)
/// to its on-disk file, atomically. No-op (never a panic) if the directory is
/// unavailable or the value isn't a well-formed database object. This is the
/// flush-on-change entry point main.rs calls after every mutation.
pub fn persist_db(db_name: &str, db: &Value) {
    let path = match db_file_path(db_name) {
        Some(p) => p,
        None => return, // %APPDATA% unavailable ⇒ memory-only, no panic.
    };
    let snap = match snapshot_from_db(db_name, db) {
        Some(s) => s,
        None => return,
    };
    atomic_write(&path, &snap.to_bytes());
}

/// Load the persisted [`DbSnapshot`] for `db_name`, or `None` if there is no
/// file, the directory is unavailable, or the file is corrupt / a version
/// mismatch (all of which mean "fresh, empty database"). Never panics.
pub fn load_snapshot(db_name: &str) -> Option<DbSnapshot> {
    let path = db_file_path(db_name)?;
    let bytes = std::fs::read(&path).ok()?;
    DbSnapshot::from_bytes(&bytes)
}

/// Delete the on-disk file for `db_name` (used by `indexedDB.deleteDatabase`).
/// Best-effort; a missing file is fine.
pub fn delete_db_file(db_name: &str) {
    if let Some(path) = db_file_path(db_name) {
        let _ = std::fs::remove_file(&path);
    }
}

/// List the names of every persisted database (used by `indexedDB.databases()`
/// so a database persisted in a prior run is enumerable before it is opened).
/// We can't recover the exact original name from the hashed filename, so we
/// read the name back out of each file's payload.
pub fn persisted_db_names() -> Vec<String> {
    let dir = match idb_dir() {
        Some(d) => d,
        None => return Vec::new(),
    };
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("idb") {
                continue;
            }
            if let Ok(bytes) = std::fs::read(&p) {
                if let Some(snap) = DbSnapshot::from_bytes(&bytes) {
                    names.push(snap.name);
                }
            }
        }
    }
    names
}

/// Structural deep-equality for the constrained IDB `Value` set. Objects
/// compare order-sensitively on their non-internal data properties (the
/// observable shape JSON would serialize). Used by the round-trip tests; a
/// number compares bit-for-value (so `5` ≠ `5.0000001`).
#[cfg(test)]
pub(crate) fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::Hole, Value::Hole) => true,
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => x.to_bits() == y.to_bits() || x == y,
        (Value::String(x), Value::String(y)) => x.as_str() == y.as_str(),
        (Value::Array(x), Value::Array(y)) => {
            let xb = x.borrow();
            let yb = y.borrow();
            xb.len() == yb.len()
                && xb.iter().zip(yb.iter()).all(|(p, q)| values_equal(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            let collect = |o: &Rc<RefCell<HashMap<String, Value>>>| {
                o.borrow()
                    .iter()
                    .filter(|(k, _)| !is_internal_key(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>()
            };
            let xs = collect(x);
            let ys = collect(y);
            xs.len() == ys.len()
                && xs
                    .iter()
                    .zip(ys.iter())
                    .all(|((ka, va), (kb, vb))| ka == kb && values_equal(va, vb))
        }
        _ => false,
    }
}

#[cfg(test)]
pub(crate) fn snapshots_equal(a: &DbSnapshot, b: &DbSnapshot) -> bool {
    if a.name != b.name || a.version != b.version || a.stores.len() != b.stores.len() {
        return false;
    }
    a.stores.iter().zip(b.stores.iter()).all(|(s, t)| {
        s.name == t.name
            && s.key_path == t.key_path
            && s.auto_increment == t.auto_increment
            && s.auto_key.to_bits() == t.auto_key.to_bits()
            && s.rows.len() == t.rows.len()
            && s.rows.iter().zip(t.rows.iter()).all(|(r, q)| {
                values_equal(&r.key, &q.key) && values_equal(&r.value, &q.value)
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "tb_idb_test_{}_{}_{}",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn obj(pairs: &[(&str, Value)]) -> Value {
        let mut m: HashMap<String, Value> = HashMap::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        Value::Object(Rc::new(RefCell::new(m)))
    }

    fn arr(items: Vec<Value>) -> Value {
        Value::Array(Rc::new(RefCell::new(items)))
    }

    fn sample_snapshot() -> DbSnapshot {
        DbSnapshot {
            name: "sample".into(),
            version: 3,
            stores: vec![
                StoreSnapshot {
                    name: "users".into(),
                    key_path: Some("id".into()),
                    auto_increment: true,
                    auto_key: 5.0,
                    rows: vec![
                        RowSnapshot {
                            key: Value::Number(1.0),
                            value: obj(&[
                                ("id", Value::Number(1.0)),
                                ("name", Value::str("Alice")),
                                ("active", Value::Bool(true)),
                                ("tags", arr(vec![Value::str("a"), Value::str("b")])),
                                (
                                    "meta",
                                    obj(&[("score", Value::Number(9.5)), ("note", Value::Null)]),
                                ),
                            ]),
                        },
                        RowSnapshot {
                            key: Value::Number(4.0),
                            value: obj(&[
                                ("id", Value::Number(4.0)),
                                ("name", Value::str("Dave")),
                            ]),
                        },
                    ],
                },
                StoreSnapshot {
                    name: "kv".into(),
                    key_path: None,
                    auto_increment: false,
                    auto_key: 1.0,
                    rows: vec![
                        RowSnapshot {
                            key: Value::str("greeting"),
                            value: Value::str("hello"),
                        },
                        RowSnapshot {
                            key: Value::Number(10.0),
                            value: Value::str("ten"),
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn snapshot_bytes_round_trip_value_identical() {
        let snap = sample_snapshot();
        let bytes = snap.to_bytes();
        let back = DbSnapshot::from_bytes(&bytes).expect("decode ok");
        assert!(
            snapshots_equal(&snap, &back),
            "snapshot must round-trip byte-for-value:\n {snap:?}\n vs\n {back:?}"
        );
    }

    #[test]
    fn numeric_key_stays_number_not_string() {
        let snap = DbSnapshot {
            name: "s".into(),
            version: 1,
            stores: vec![StoreSnapshot {
                name: "s".into(),
                key_path: None,
                auto_increment: false,
                auto_key: 1.0,
                rows: vec![RowSnapshot {
                    key: Value::Number(5.0),
                    value: Value::str("five"),
                }],
            }],
        };
        let back = DbSnapshot::from_bytes(&snap.to_bytes()).unwrap();
        let k = &back.stores[0].rows[0].key;
        assert!(
            matches!(k, Value::Number(n) if (*n - 5.0).abs() < 1e-12),
            "numeric key 5 must reload as Number(5), got {k:?}"
        );
    }

    #[test]
    fn undefined_vs_null_key_type_preserved() {
        // idb_key_cmp ranks Undefined(0) below Null(4); the encoding must not
        // collapse them (JSON would).
        let snap = DbSnapshot {
            name: "s".into(),
            version: 1,
            stores: vec![StoreSnapshot {
                name: "s".into(),
                key_path: None,
                auto_increment: false,
                auto_key: 1.0,
                rows: vec![
                    RowSnapshot {
                        key: Value::Undefined,
                        value: Value::Null,
                    },
                    RowSnapshot {
                        key: Value::Null,
                        value: Value::Undefined,
                    },
                ],
            }],
        };
        let back = DbSnapshot::from_bytes(&snap.to_bytes()).unwrap();
        assert!(matches!(back.stores[0].rows[0].key, Value::Undefined));
        // The value Undefined inside an object would be dropped, but a top-level
        // row value of Undefined round-trips as Undefined here.
        assert!(matches!(back.stores[0].rows[0].value, Value::Null));
        assert!(matches!(back.stores[0].rows[1].key, Value::Null));
    }

    #[test]
    fn corrupt_file_returns_none() {
        assert!(DbSnapshot::from_bytes(b"").is_none());
        assert!(DbSnapshot::from_bytes(b"not-a-tbidb-file").is_none());
        // Right magic, wrong version.
        let mut bad = Vec::new();
        bad.extend_from_slice(MAGIC);
        wr_u32(&mut bad, DISK_VERSION + 1);
        assert!(DbSnapshot::from_bytes(&bad).is_none());
        // Right magic+version then truncated mid-store.
        let mut trunc = sample_snapshot().to_bytes();
        trunc.truncate(trunc.len() / 2);
        assert!(
            DbSnapshot::from_bytes(&trunc).is_none(),
            "truncated file must decode to None, never panic"
        );
    }

    #[test]
    fn persist_and_load_via_disk() {
        let dir = temp_dir("disk");
        set_dir_for_test(Some(dir.clone()));

        // Build a live database Value the way make_database/make_store do.
        let store = obj(&[
            ("name", Value::str("items")),
            ("_keyPath", Value::str("id")),
            ("keyPath", Value::str("id")),
            ("autoIncrement", Value::Bool(true)),
            ("_autoKey", Value::Number(7.0)),
            (
                "_rows",
                arr(vec![
                    obj(&[
                        ("key", Value::Number(1.0)),
                        ("value", obj(&[("id", Value::Number(1.0)), ("n", Value::str("one"))])),
                    ]),
                    obj(&[
                        ("key", Value::Number(6.0)),
                        ("value", obj(&[("id", Value::Number(6.0)), ("n", Value::str("six"))])),
                    ]),
                ]),
            ),
        ]);
        let stores = obj(&[("items", store)]);
        let db = obj(&[
            ("name", Value::str("mydb")),
            ("version", Value::Number(2.0)),
            ("_stores", stores),
        ]);

        persist_db("mydb", &db);
        let snap = load_snapshot("mydb").expect("loaded");
        assert_eq!(snap.name, "mydb");
        assert_eq!(snap.version, 2);
        assert_eq!(snap.stores.len(), 1);
        let s = &snap.stores[0];
        assert_eq!(s.name, "items");
        assert_eq!(s.key_path.as_deref(), Some("id"));
        assert!(s.auto_increment);
        assert!((s.auto_key - 7.0).abs() < 1e-12);
        assert_eq!(s.rows.len(), 2);
        assert!(matches!(s.rows[1].key, Value::Number(n) if (n - 6.0).abs() < 1e-12));

        // A different (never-persisted) name has no file ⇒ None ⇒ fresh DB.
        assert!(load_snapshot("never_written").is_none());

        // databases() enumeration recovers the real name from the payload.
        let names = persisted_db_names();
        assert!(names.contains(&"mydb".to_string()), "got {names:?}");

        set_dir_for_test(None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_appdata_is_no_panic() {
        // Point at a path under a file (so create_dir_all fails) and confirm
        // persist/load are silent no-ops.
        let dir = temp_dir("nodir");
        // Create a *file* where the dir should be, so mkdir fails.
        std::fs::write(&dir, b"x").unwrap();
        set_dir_for_test(Some(dir.join("sub")));
        let db = obj(&[
            ("name", Value::str("x")),
            ("version", Value::Number(1.0)),
            ("_stores", obj(&[])),
        ]);
        persist_db("x", &db); // must not panic
        assert!(load_snapshot("x").is_none());
        set_dir_for_test(None);
        let _ = std::fs::remove_file(&dir);
    }
}
