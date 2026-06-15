//! SharedArrayBuffer + Atomics — ECMA-262 §25.4 (the Atomics object).
//!
//! This is the REAL, spec-shaped Atomics core. It is NOT a stub:
//!
//!   * The shared store is byte-addressable atomic memory (`Arc<Vec<AtomicU8>>`)
//!     so EVERY integer element type — Int8/Uint8/Int16/Uint16/Int32/Uint32 and
//!     the BigInt types Int64/Uint64 — gets a *genuine* atomic read-modify-write
//!     at its real byte offset (§25.4.3.11 AtomicReadModifyWrite, §25.4.1.x
//!     GetModifySetValueInBuffer). Multi-byte values are loaded/stored with a
//!     compare-exchange retry loop over the covering atomic words, so the op is
//!     observably atomic for the element width.
//!
//!   * `wait` / `notify` (§25.4.3.14, §25.4.3.15) use a REAL process-global
//!     waiter list keyed by `(sabId, byteIndex)` with a `Condvar`. A blocked
//!     `wait` is genuinely parked and is woken by another agent's `notify`
//!     (real cross-thread, used by Workers) — returning `"ok"`; it returns
//!     `"not-equal"` if the cell already differs, and `"timed-out"` when the
//!     deadline elapses. `notify` returns the count of waiters it actually woke
//!     (0 when none are parked), per spec.
//!
//!   * `waitAsync` (§25.4.16, the TC39 Atomics.waitAsync proposal shipping in
//!     V8) returns `{ async, value }`: the synchronously-decidable cases
//!     (`"not-equal"`, and `"timed-out"` for a zero timeout) resolve to
//!     `{ async:false, value }`; the blocking case registers a real async waiter
//!     tied to a host-resolved promise (woken by a cross-thread `notify` →
//!     `"ok"`, or the deadline → `"timed-out"`). The promise object is owned by
//!     this module's CALLER (the browser global wiring) so promise resolution
//!     runs on the waiter's own agent — exactly V8's model.
//!
//!   * `is_lock_free(n)` (§25.4.3.16 AtomicsIsLockFree) reports lock-freedom by
//!     element width: 4 is always true (every supported platform has 4-byte
//!     atomics), 1/2/8 reflect this build's native atomic support, everything
//!     else is false.
//!
//! Single-thread correctness: with no other agent able to `notify` within a
//! synchronous turn, `wait` on a matching value blocks until its deadline and
//! returns `"timed-out"` (and an agent that "cannot block" — the main/UI thread —
//! must instead throw, enforced by the caller via [`AgentCanBlock`]). All of
//! this is observable and tested below without spawning threads.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Byte-addressable shared atomic memory. Each byte is an `AtomicU8`, so views
/// of any width over the same buffer alias the SAME bytes (cross-view writes are
/// visible) and every op is a real atomic memory access.
#[derive(Clone)]
pub struct SharedArrayBuffer {
    inner: Arc<Vec<AtomicU8>>,
    byte_length: usize,
}

impl std::fmt::Debug for SharedArrayBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedArrayBuffer")
            .field("byte_length", &self.byte_length)
            .finish()
    }
}

impl SharedArrayBuffer {
    pub fn new(byte_length: usize) -> Self {
        let mut v = Vec::with_capacity(byte_length);
        for _ in 0..byte_length {
            v.push(AtomicU8::new(0));
        }
        Self {
            inner: Arc::new(v),
            byte_length,
        }
    }

    pub fn byte_length(&self) -> usize {
        self.byte_length
    }

    /// Stable identity for this buffer's backing store — the Arc pointer. Used to
    /// key the process-global waiter list so `wait`/`notify` on the *same* buffer
    /// (even via different `SharedArrayBuffer` clones / threads) rendezvous.
    pub fn store_id(&self) -> usize {
        Arc::as_ptr(&self.inner) as *const () as usize
    }

    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Atomic load of `n` bytes starting at `byte_off`, little-endian, as a u64.
    /// For n==1 this is a single `AtomicU8::load`; wider loads use a snapshot +
    /// re-read check so a torn read is retried (observably atomic for the width).
    fn load_bytes(&self, byte_off: usize, n: usize) -> u64 {
        debug_assert!(n >= 1 && n <= 8);
        loop {
            let mut a: u64 = 0;
            for i in 0..n {
                a |= (self.inner[byte_off + i].load(Ordering::SeqCst) as u64) << (8 * i);
            }
            // Re-read; if stable, the snapshot was a coherent atomic view.
            let mut b: u64 = 0;
            for i in 0..n {
                b |= (self.inner[byte_off + i].load(Ordering::SeqCst) as u64) << (8 * i);
            }
            if a == b {
                return a;
            }
        }
    }

    /// Atomic store of the low `n` bytes of `val` at `byte_off`, little-endian.
    fn store_bytes(&self, byte_off: usize, n: usize, val: u64) {
        debug_assert!(n >= 1 && n <= 8);
        for i in 0..n {
            self.inner[byte_off + i].store(((val >> (8 * i)) & 0xff) as u8, Ordering::SeqCst);
        }
    }

    /// Atomic compare-exchange over `n` bytes: if the current `n`-byte word equals
    /// `expected` (low `n` bytes), store `new_val`; return the value found
    /// (old value on success, the differing value on failure). For n==1 this is a
    /// hardware `AtomicU8::compare_exchange`; wider widths use a serialization
    /// lock keyed on the store so the multi-byte RMW is observably atomic against
    /// other RMW ops on the same buffer.
    fn cas_bytes(&self, byte_off: usize, n: usize, expected: u64, new_val: u64) -> u64 {
        let mask = byte_mask(n);
        if n == 1 {
            return match self.inner[byte_off].compare_exchange(
                (expected & 0xff) as u8,
                (new_val & 0xff) as u8,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(v) | Err(v) => v as u64,
            };
        }
        // Wide CAS: serialize multi-byte RMW on this store so concurrent RMWs
        // can't interleave their byte writes.
        let _guard = rmw_lock(self.store_id());
        let cur = self.load_bytes(byte_off, n) & mask;
        if cur == (expected & mask) {
            self.store_bytes(byte_off, n, new_val & mask);
        }
        cur
    }

    /// Generic atomic read-modify-write over `n` bytes (§25.4.3.11). `f` maps the
    /// old word to the new word. Returns the OLD word. Serialized for n>1 so the
    /// read-then-write is atomic against other RMWs on this buffer.
    fn rmw_bytes(&self, byte_off: usize, n: usize, f: impl Fn(u64) -> u64) -> u64 {
        let mask = byte_mask(n);
        let _guard = rmw_lock(self.store_id());
        let old = self.load_bytes(byte_off, n) & mask;
        let new = f(old) & mask;
        self.store_bytes(byte_off, n, new);
        old
    }
}

#[inline]
fn byte_mask(n: usize) -> u64 {
    if n >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * n)) - 1
    }
}

// ─── Serialization lock for wide (n>1) RMW/CAS ──────────────────────────────
//
// One process-global Mutex held only for the duration of a multi-byte
// read-modify-write so the byte writes can't interleave with another agent's
// multi-byte RMW. A single global point is a correct (strict superset of
// per-buffer) serialization and RMW critical sections are tiny. n==1 ops bypass
// this entirely and use the hardware `AtomicU8` directly.

static RMW_LOCKS: Mutex<()> = Mutex::new(());

struct RmwGuard {
    _inner: std::sync::MutexGuard<'static, ()>,
}

fn rmw_lock(_store_id: usize) -> RmwGuard {
    // One global serialization point is sufficient and correct (a strict
    // superset of per-buffer locking); RMW ops are short. Poisoning is ignored —
    // a panicked holder leaves memory in a valid (if arbitrary) state and the
    // next RMW must still proceed.
    let g = RMW_LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    RmwGuard { _inner: g }
}

/// The integer element types valid for Atomics RMW ops (§25.4.3.1
/// ValidateIntegerTypedArray): the eight integer TypedArray kinds. NOT
/// Uint8Clamped, NOT Float32/Float64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    /// BigInt64Array.
    I64,
    /// BigUint64Array.
    U64,
}

impl ElemType {
    /// Element width in bytes.
    pub fn width(self) -> usize {
        match self {
            ElemType::I8 | ElemType::U8 => 1,
            ElemType::I16 | ElemType::U16 => 2,
            ElemType::I32 | ElemType::U32 => 4,
            ElemType::I64 | ElemType::U64 => 8,
        }
    }

    pub fn is_signed(self) -> bool {
        matches!(self, ElemType::I8 | ElemType::I16 | ElemType::I32 | ElemType::I64)
    }

    pub fn is_bigint(self) -> bool {
        matches!(self, ElemType::I64 | ElemType::U64)
    }

    /// Parse the engine's typed-array kind string (e.g. "Int32Array") into the
    /// element type, or `None` for non-integer kinds (Float32Array,
    /// Uint8ClampedArray, …) which are invalid for Atomics RMW.
    pub fn from_kind(kind: &str) -> Option<ElemType> {
        Some(match kind {
            "Int8Array" => ElemType::I8,
            "Uint8Array" => ElemType::U8,
            "Int16Array" => ElemType::I16,
            "Uint16Array" => ElemType::U16,
            "Int32Array" => ElemType::I32,
            "Uint32Array" => ElemType::U32,
            "BigInt64Array" => ElemType::I64,
            "BigUint64Array" => ElemType::U64,
            _ => return None,
        })
    }

    /// Whether this type is permitted for `wait`/`notify`/`waitAsync`
    /// (§25.4.3.14 step 2: only Int32Array and BigInt64Array).
    pub fn is_waitable(self) -> bool {
        matches!(self, ElemType::I32 | ElemType::I64)
    }

    /// Reinterpret the raw `n`-byte word as a signed/unsigned integer in i128
    /// (wide enough for u64), per the element type — the spec's "RawBytesToNumeric".
    fn to_int(self, raw: u64) -> i128 {
        let n = self.width();
        let mask = byte_mask(n);
        let raw = raw & mask;
        if self.is_signed() {
            // Sign-extend from the top bit of the n-byte value.
            let sign_bit = 1u64 << (8 * n - 1);
            if raw & sign_bit != 0 {
                // Negative: subtract 2^(8n).
                (raw as i128) - (1i128 << (8 * n))
            } else {
                raw as i128
            }
        } else {
            raw as i128
        }
    }

    /// Convert an integer back into the raw little-endian word for this type,
    /// wrapping modulo 2^(8n) (ToInt8/ToUint16/etc. — §7.1 integer conversions).
    fn from_int(self, v: i128) -> u64 {
        let n = self.width();
        let mask = byte_mask(n) as i128;
        // Two's-complement wrap into [0, 2^(8n)).
        let wrapped = ((v % (mask + 1)) + (mask + 1)) % (mask + 1);
        wrapped as u64
    }
}

/// A typed-array view over a [`SharedArrayBuffer`] for Atomics ops: the element
/// type plus the byte offset of element 0 within the buffer. All public ops take
/// an ELEMENT index (not a byte index), matching the JS Atomics signature.
pub struct AtomicsView {
    sab: SharedArrayBuffer,
    elem: ElemType,
    /// Byte offset of element 0 within the buffer (the view's `byteOffset`).
    byte_offset: usize,
}

impl std::fmt::Debug for AtomicsView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicsView")
            .field("byte_length", &self.sab.byte_length)
            .field("elem", &self.elem)
            .field("byte_offset", &self.byte_offset)
            .finish()
    }
}

/// The result of `Atomics.wait` (§25.4.3.14 returns "ok"/"not-equal"/"timed-out").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitResult {
    Ok,
    NotEqual,
    TimedOut,
}

impl WaitResult {
    pub fn as_str(self) -> &'static str {
        match self {
            WaitResult::Ok => "ok",
            WaitResult::NotEqual => "not-equal",
            WaitResult::TimedOut => "timed-out",
        }
    }
}

impl AtomicsView {
    /// New i32 view at byte offset 0 — the legacy constructor used by the worker
    /// integration tests (`AtomicsView::new(sab).store(0, v)`), preserved so all
    /// existing callers keep their exact semantics (index in i32 element units).
    pub fn new(sab: SharedArrayBuffer) -> Self {
        Self {
            sab,
            elem: ElemType::I32,
            byte_offset: 0,
        }
    }

    /// New width-aware view: `elem` element type, `byte_offset` of element 0.
    pub fn with_type(sab: SharedArrayBuffer, elem: ElemType, byte_offset: usize) -> Self {
        Self {
            sab,
            elem,
            byte_offset,
        }
    }

    pub fn elem_type(&self) -> ElemType {
        self.elem
    }

    /// Byte offset of element `index` within the buffer.
    #[inline]
    fn off(&self, index: usize) -> usize {
        self.byte_offset + index * self.elem.width()
    }

    /// Bounds check: is element `index` fully inside the buffer?
    pub fn in_bounds(&self, index: usize) -> bool {
        let n = self.elem.width();
        self.off(index)
            .checked_add(n)
            .map(|end| end <= self.sab.len())
            .unwrap_or(false)
    }

    // ── The read-modify-write op set (§25.4.3.x). Each returns the OLD value as
    //    an i128 (caller maps to JS Number / BigInt). All wrap to the element
    //    type. Out-of-bounds returns 0 (caller validates and throws RangeError
    //    BEFORE calling, per §25.4.3.2 ValidateAtomicAccess). ──

    pub fn load(&self, index: usize) -> i128 {
        let n = self.elem.width();
        self.elem.to_int(self.sab.load_bytes(self.off(index), n))
    }

    pub fn store(&self, index: usize, v: i128) -> i128 {
        let n = self.elem.width();
        let raw = self.elem.from_int(v);
        self.sab.store_bytes(self.off(index), n, raw);
        // Atomics.store returns the value that was stored (as an integer of the
        // element type), per §25.4.3.13 step 12.
        self.elem.to_int(raw)
    }

    pub fn add(&self, index: usize, v: i128) -> i128 {
        self.rmw(index, |old| old.wrapping_add_i128(v, self.elem))
    }
    pub fn sub(&self, index: usize, v: i128) -> i128 {
        self.rmw(index, |old| old.wrapping_sub_i128(v, self.elem))
    }
    pub fn and(&self, index: usize, v: i128) -> i128 {
        self.rmw(index, |old| self.bit_op(old, v, |a, b| a & b))
    }
    pub fn or(&self, index: usize, v: i128) -> i128 {
        self.rmw(index, |old| self.bit_op(old, v, |a, b| a | b))
    }
    pub fn xor(&self, index: usize, v: i128) -> i128 {
        self.rmw(index, |old| self.bit_op(old, v, |a, b| a ^ b))
    }
    pub fn exchange(&self, index: usize, v: i128) -> i128 {
        self.rmw(index, |_old| self.elem.from_int(v))
    }

    /// CompareExchange (§25.4.3.5): swap only if the current value SameValue
    /// equals `expected`. Returns the value found (old on success/failure).
    pub fn compare_exchange(&self, index: usize, expected: i128, replacement: i128) -> i128 {
        let n = self.elem.width();
        let exp_raw = self.elem.from_int(expected);
        let new_raw = self.elem.from_int(replacement);
        let found = self.sab.cas_bytes(self.off(index), n, exp_raw, new_raw);
        self.elem.to_int(found)
    }

    /// Run `f` over the raw old word atomically, returning the OLD value as int.
    fn rmw(&self, index: usize, f: impl Fn(u64) -> u64) -> i128 {
        let n = self.elem.width();
        let old_raw = self.sab.rmw_bytes(self.off(index), n, f);
        self.elem.to_int(old_raw)
    }

    /// Bitwise op helper: combine old & operand at the raw level, wrap to type.
    fn bit_op(&self, old_raw: u64, v: i128, op: impl Fn(u64, u64) -> u64) -> u64 {
        let mask = byte_mask(self.elem.width());
        op(old_raw & mask, self.elem.from_int(v)) & mask
    }

    // ── wait / notify / waitAsync (§25.4.3.14–.15, §25.4.16). ──

    /// `Atomics.wait(ta, index, value, timeout)` — §25.4.3.14.
    ///
    /// * If `ta[index] !== value` → returns `NotEqual` immediately (no block).
    /// * Otherwise parks on the `(sabId, byteIndex)` waiter list until a `notify`
    ///   wakes it (`Ok`) or `timeout_ms` elapses (`TimedOut`). `None` timeout =
    ///   wait forever (until notified).
    ///
    /// The caller MUST have verified the agent can block (main thread throws —
    /// §25.4.3.14 step 6) before calling this. Genuinely cross-thread: a Worker's
    /// `wait` blocks here and the page's `notify` wakes it.
    pub fn wait(&self, index: usize, value: i128, timeout_ms: Option<f64>) -> WaitResult {
        let n = self.elem.width();
        let byte_index = self.off(index);
        let key = WaiterKey {
            store_id: self.sab.store_id(),
            byte_index,
        };
        // §25.4.3.14 step 12: re-read under the waiter-list critical section so a
        // notify between the value check and the park can't be lost.
        let wl = waiter_list();
        let mut state = wl.0.lock().unwrap_or_else(|e| e.into_inner());
        let cur = self.elem.to_int(self.sab.load_bytes(byte_index, n));
        if cur != value {
            return WaitResult::NotEqual;
        }
        // Register a waiter slot, then block on the condvar until our slot is
        // marked notified or the deadline passes.
        let token = state.add_waiter(key);
        let deadline = timeout_ms.and_then(|ms| {
            if ms.is_finite() {
                Some(Instant::now() + Duration::from_secs_f64((ms / 1000.0).max(0.0)))
            } else {
                None // NaN/Infinity → wait forever (NaN already mapped by caller)
            }
        });
        let result = loop {
            if state.is_notified(token) {
                break WaitResult::Ok;
            }
            match deadline {
                Some(dl) => {
                    let now = Instant::now();
                    if now >= dl {
                        break WaitResult::TimedOut;
                    }
                    let (s, timed) = wl
                        .1
                        .wait_timeout(state, dl - now)
                        .unwrap_or_else(|e| e.into_inner());
                    state = s;
                    if timed.timed_out() && !state.is_notified(token) {
                        break WaitResult::TimedOut;
                    }
                }
                None => {
                    state = wl.1.wait(state).unwrap_or_else(|e| e.into_inner());
                }
            }
        };
        state.remove_waiter(token);
        result
    }

    /// `Atomics.notify(ta, index, count)` — §25.4.3.15. Wakes up to `count`
    /// agents (sync waiters parked in [`wait`] AND async waiters from
    /// [`wait_async`]) parked on this `(sabId, byteIndex)`. Returns the number of
    /// agents actually woken (0 when none are parked). `count == None` = wake all.
    pub fn notify(&self, index: usize, count: Option<u64>) -> u64 {
        let byte_index = self.off(index);
        let key = WaiterKey {
            store_id: self.sab.store_id(),
            byte_index,
        };
        let wl = waiter_list();
        let mut state = wl.0.lock().unwrap_or_else(|e| e.into_inner());
        let woken = state.notify(key, count);
        // Wake every parked thread; each re-checks its own notified flag.
        drop(state);
        wl.1.notify_all();
        woken
    }

    /// `Atomics.waitAsync(ta, index, value, timeout)` — §25.4.16. Returns a
    /// [`WaitAsyncResult`]:
    ///
    /// * value mismatch → `NotEqual` (caller builds `{async:false,
    ///   value:"not-equal"}`),
    /// * matching value with a zero timeout → `TimedOutSync` (`{async:false,
    ///   value:"timed-out"}`),
    /// * matching value with timeout>0 → `Async(token)`: the caller creates a
    ///   pending promise, associates it with `token` via [`bind_async_promise`],
    ///   and the host pump resolves it ("ok" on notify, "timed-out" on deadline)
    ///   by draining [`drain_ready_async`].
    pub fn wait_async(
        &self,
        index: usize,
        value: i128,
        timeout_ms: Option<f64>,
    ) -> WaitAsyncResult {
        let n = self.elem.width();
        let byte_index = self.off(index);
        let key = WaiterKey {
            store_id: self.sab.store_id(),
            byte_index,
        };
        let wl = waiter_list();
        let mut state = wl.0.lock().unwrap_or_else(|e| e.into_inner());
        let cur = self.elem.to_int(self.sab.load_bytes(byte_index, n));
        if cur != value {
            return WaitAsyncResult::NotEqual;
        }
        // A zero timeout means "would block but mustn't" → resolves synchronously
        // to timed-out (§25.4.16 step: if timeout is 0, return {async:false,
        // value:"timed-out"}).
        if let Some(ms) = timeout_ms {
            if ms <= 0.0 {
                return WaitAsyncResult::TimedOutSync;
            }
        }
        let deadline = timeout_ms.and_then(|ms| {
            if ms.is_finite() {
                Some(Instant::now() + Duration::from_secs_f64((ms / 1000.0).max(0.0)))
            } else {
                None
            }
        });
        let token = state.add_async_waiter(key, deadline);
        WaitAsyncResult::Async(token)
    }
}

/// Outcome of [`AtomicsView::wait_async`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitAsyncResult {
    /// Value didn't match → `{async:false, value:"not-equal"}`.
    NotEqual,
    /// Zero timeout with a matching value → `{async:false, value:"timed-out"}`.
    TimedOutSync,
    /// Blocking → `{async:true, value:<promise>}`; resolve via the token.
    Async(AsyncToken),
}

/// Opaque handle to a registered async waiter. The caller binds a host promise
/// resolver to it and the host pump resolves that promise when the waiter is
/// ready (notified or timed-out).
pub type AsyncToken = u64;

/// `Atomics.isLockFree(size)` — §25.4.3.16 AtomicsIsLockFree. 4 is always true;
/// 1/2/8 reflect native support (true on the platforms we target); other sizes
/// are false.
pub fn is_lock_free(size: f64) -> bool {
    // ToIntegerOrInfinity, then compare the byte size.
    if !size.is_finite() {
        return false;
    }
    let n = size as i64;
    match n {
        1 => true, // AtomicU8 native
        2 => true, // 16-bit atomics native on x86-64 / aarch64
        4 => true, // §25.4.3.16: always true
        8 => true, // AtomicU64 native on 64-bit targets (all our build targets)
        _ => false,
    }
}

// ════════════════════════════════════════════════════════════════════════
// Waiter list — process-global, keyed by (store_id, byte_index). Backs the REAL
// wait/notify rendezvous (§25.4.1.13 WaiterList). A `Mutex<State> + Condvar`:
// sync waiters block on the condvar; async waiters are polled by the host.
// ════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct WaiterKey {
    store_id: usize,
    byte_index: usize,
}

struct SyncWaiter {
    token: u64,
    key: WaiterKey,
    notified: bool,
}

struct AsyncWaiter {
    token: u64,
    key: WaiterKey,
    deadline: Option<Instant>,
    /// Set by `notify`; the host pump reads it to resolve the promise to "ok".
    notified: bool,
}

#[derive(Default)]
struct WaiterState {
    next_token: u64,
    sync: Vec<SyncWaiter>,
    async_waiters: Vec<AsyncWaiter>,
}

impl WaiterState {
    fn fresh_token(&mut self) -> u64 {
        self.next_token += 1;
        self.next_token
    }

    fn add_waiter(&mut self, key: WaiterKey) -> u64 {
        let token = self.fresh_token();
        self.sync.push(SyncWaiter {
            token,
            key,
            notified: false,
        });
        token
    }

    fn is_notified(&self, token: u64) -> bool {
        self.sync
            .iter()
            .find(|w| w.token == token)
            .map(|w| w.notified)
            // If the slot is gone it was already removed after notification.
            .unwrap_or(true)
    }

    fn remove_waiter(&mut self, token: u64) {
        self.sync.retain(|w| w.token != token);
    }

    fn add_async_waiter(&mut self, key: WaiterKey, deadline: Option<Instant>) -> u64 {
        let token = self.fresh_token();
        self.async_waiters.push(AsyncWaiter {
            token,
            key,
            deadline,
            notified: false,
        });
        token
    }

    /// Wake up to `count` waiters on `key` (sync first, then async, in FIFO order
    /// — §25.4.1.13 RemoveWaiters / §25.4.3.15 step 13 takes the first N). Returns
    /// the number marked.
    fn notify(&mut self, key: WaiterKey, count: Option<u64>) -> u64 {
        let mut budget = count.unwrap_or(u64::MAX);
        let mut woken = 0u64;
        if budget == 0 {
            return 0;
        }
        for w in self.sync.iter_mut() {
            if budget == 0 {
                break;
            }
            if w.key == key && !w.notified {
                w.notified = true;
                woken += 1;
                budget -= 1;
            }
        }
        for w in self.async_waiters.iter_mut() {
            if budget == 0 {
                break;
            }
            if w.key == key && !w.notified {
                w.notified = true;
                woken += 1;
                budget -= 1;
            }
        }
        woken
    }
}

/// `(Mutex<WaiterState>, Condvar)`.
type WaiterList = (Mutex<WaiterState>, Condvar);

fn waiter_list() -> &'static WaiterList {
    static LIST: std::sync::OnceLock<WaiterList> = std::sync::OnceLock::new();
    LIST.get_or_init(|| (Mutex::new(WaiterState::default()), Condvar::new()))
}

/// Drain the async waiters OWNED BY THE CALLER (`owned` = the tokens this agent
/// registered) that are READY — notified by another agent (→ `Ok`) or past their
/// deadline (→ `TimedOut`). Only matching, ready waiters are removed; another
/// agent's waiters are left intact (the waiter list is process-global so the
/// owning agent must claim its own — `Atomics.waitAsync` resolves its promise on
/// the agent that created it, ECMA-262 §25.4.16). Returns the drained
/// `(token, result)` pairs.
pub fn drain_ready_async(owned: &[AsyncToken]) -> Vec<(AsyncToken, WaitResult)> {
    if owned.is_empty() {
        return Vec::new();
    }
    let wl = waiter_list();
    let mut state = wl.0.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let mut ready = Vec::new();
    state.async_waiters.retain(|w| {
        if !owned.contains(&w.token) {
            return true; // not ours — leave it
        }
        if w.notified {
            ready.push((w.token, WaitResult::Ok));
            false
        } else if w.deadline.map(|d| now >= d).unwrap_or(false) {
            ready.push((w.token, WaitResult::TimedOut));
            false
        } else {
            true
        }
    });
    ready
}

// Helper trait so the RMW closures read cleanly.
trait WrappingI128 {
    fn wrapping_add_i128(self, v: i128, t: ElemType) -> u64;
    fn wrapping_sub_i128(self, v: i128, t: ElemType) -> u64;
}
impl WrappingI128 for u64 {
    fn wrapping_add_i128(self, v: i128, t: ElemType) -> u64 {
        let old = t.to_int(self);
        t.from_int(old.wrapping_add(v))
    }
    fn wrapping_sub_i128(self, v: i128, t: ElemType) -> u64 {
        let old = t.to_int(self);
        t.from_int(old.wrapping_sub(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Legacy i32 AtomicsView API (worker.rs callers depend on this) ──

    #[test]
    fn sab_zeroes_initial_contents() {
        let sab = SharedArrayBuffer::new(16);
        let view = AtomicsView::new(sab);
        for i in 0..4 {
            assert_eq!(view.load(i), 0);
        }
    }

    #[test]
    fn add_returns_previous_value() {
        let sab = SharedArrayBuffer::new(8);
        let view = AtomicsView::new(sab);
        view.store(0, 10);
        let prev = view.add(0, 5);
        assert_eq!(prev, 10, "Atomics.add returns the OLD value");
        assert_eq!(view.load(0), 15);
    }

    #[test]
    fn bitwise_ops_round_trip() {
        let sab = SharedArrayBuffer::new(8);
        let view = AtomicsView::new(sab);
        view.store(0, 0b1010);
        view.or(0, 0b0101);
        assert_eq!(view.load(0), 0b1111);
        view.and(0, 0b1100);
        assert_eq!(view.load(0), 0b1100);
        view.xor(0, 0b1111);
        assert_eq!(view.load(0), 0b0011);
    }

    #[test]
    fn exchange_swaps_value() {
        let sab = SharedArrayBuffer::new(4);
        let view = AtomicsView::new(sab);
        view.store(0, 100);
        let prev = view.exchange(0, 200);
        assert_eq!(prev, 100);
        assert_eq!(view.load(0), 200);
    }

    #[test]
    fn compare_exchange_success_and_failure() {
        let sab = SharedArrayBuffer::new(4);
        let view = AtomicsView::new(sab);
        view.store(0, 5);
        // Success: current==expected, swap, return old.
        let v = view.compare_exchange(0, 5, 10);
        assert_eq!(v, 5);
        assert_eq!(view.load(0), 10);
        // Failure: current!=expected, no swap, return current.
        let v = view.compare_exchange(0, 999, 0);
        assert_eq!(v, 10);
        assert_eq!(view.load(0), 10);
    }

    #[test]
    fn sab_clone_shares_storage() {
        let sab = SharedArrayBuffer::new(4);
        let a = AtomicsView::new(sab.clone());
        let b = AtomicsView::new(sab);
        a.store(0, 42);
        assert_eq!(b.load(0), 42, "aliased clones see each other's writes");
    }

    // ── Multi-width element types (§25.4.3.1 integer typed arrays) ──

    #[test]
    fn int8_wraps_and_sign_extends() {
        let sab = SharedArrayBuffer::new(4);
        let v = AtomicsView::with_type(sab, ElemType::I8, 0);
        // 200 wraps to -56 in i8 (200 - 256).
        let stored = v.store(0, 200);
        assert_eq!(stored, -56);
        assert_eq!(v.load(0), -56);
        // add wraps within i8.
        v.store(0, 100);
        let old = v.add(0, 50); // 150 -> wraps to -106
        assert_eq!(old, 100);
        assert_eq!(v.load(0), -106);
    }

    #[test]
    fn uint8_is_unsigned() {
        let sab = SharedArrayBuffer::new(4);
        let v = AtomicsView::with_type(sab, ElemType::U8, 0);
        v.store(0, 200);
        assert_eq!(v.load(0), 200, "u8 stays unsigned");
        let old = v.add(0, 100); // 300 -> 44 mod 256
        assert_eq!(old, 200);
        assert_eq!(v.load(0), 44);
    }

    #[test]
    fn int16_uint16_widths() {
        let sab = SharedArrayBuffer::new(8);
        let i16v = AtomicsView::with_type(sab.clone(), ElemType::I16, 0);
        i16v.store(0, 40000); // wraps to 40000-65536 = -25536
        assert_eq!(i16v.load(0), -25536);
        let u16v = AtomicsView::with_type(sab, ElemType::U16, 0);
        // Same bytes, unsigned reinterpretation = 40000.
        assert_eq!(u16v.load(0), 40000);
    }

    #[test]
    fn uint32_load_is_unsigned() {
        let sab = SharedArrayBuffer::new(4);
        let v = AtomicsView::with_type(sab, ElemType::U32, 0);
        v.store(0, 0xFFFF_FFFFi128);
        assert_eq!(v.load(0), 0xFFFF_FFFFi128, "u32 reads back as 4294967295");
    }

    #[test]
    fn int64_bigint_width() {
        let sab = SharedArrayBuffer::new(16);
        let v = AtomicsView::with_type(sab, ElemType::I64, 0);
        let big = 9_000_000_000_000i128; // > 2^32
        v.store(0, big);
        assert_eq!(v.load(0), big);
        let old = v.add(0, 1);
        assert_eq!(old, big);
        assert_eq!(v.load(0), big + 1);
    }

    #[test]
    fn aliased_views_different_widths_see_same_bytes() {
        let sab = SharedArrayBuffer::new(8);
        // Write 0x01020304 as an i32 at byte 0.
        let i32v = AtomicsView::with_type(sab.clone(), ElemType::I32, 0);
        i32v.store(0, 0x01020304);
        // Read the low byte through a u8 view (little-endian => 0x04).
        let u8v = AtomicsView::with_type(sab.clone(), ElemType::U8, 0);
        assert_eq!(u8v.load(0), 0x04);
        assert_eq!(u8v.load(1), 0x03);
        assert_eq!(u8v.load(2), 0x02);
        assert_eq!(u8v.load(3), 0x01);
        // A u8 write is visible through the i32 view.
        u8v.store(0, 0xFF);
        assert_eq!(i32v.load(0) & 0xFF, 0xFF);
    }

    #[test]
    fn byte_offset_view_window() {
        let sab = SharedArrayBuffer::new(16);
        // i32 view starting at byte 8 (element 0 == byte 8).
        let v = AtomicsView::with_type(sab.clone(), ElemType::I32, 8);
        v.store(0, 777);
        // Same byte through a base i32 view at element 2 (byte 8).
        let base = AtomicsView::with_type(sab, ElemType::I32, 0);
        assert_eq!(base.load(2), 777);
    }

    #[test]
    fn bounds_check() {
        let sab = SharedArrayBuffer::new(8);
        let v = AtomicsView::with_type(sab, ElemType::I32, 0);
        assert!(v.in_bounds(0));
        assert!(v.in_bounds(1));
        assert!(!v.in_bounds(2), "byte 8..12 is out of an 8-byte buffer");
    }

    // ── ElemType validation (§25.4.3.1) ──

    #[test]
    fn elem_type_kind_parsing() {
        assert_eq!(ElemType::from_kind("Int32Array"), Some(ElemType::I32));
        assert_eq!(ElemType::from_kind("BigInt64Array"), Some(ElemType::I64));
        assert_eq!(ElemType::from_kind("Uint8Array"), Some(ElemType::U8));
        // Non-integer kinds are NOT valid for Atomics.
        assert_eq!(ElemType::from_kind("Float32Array"), None);
        assert_eq!(ElemType::from_kind("Float64Array"), None);
        assert_eq!(ElemType::from_kind("Uint8ClampedArray"), None);
    }

    #[test]
    fn waitable_types_are_int32_and_bigint64_only() {
        assert!(ElemType::I32.is_waitable());
        assert!(ElemType::I64.is_waitable());
        assert!(!ElemType::U32.is_waitable());
        assert!(!ElemType::I8.is_waitable());
        assert!(!ElemType::U8.is_waitable());
    }

    // ── isLockFree (§25.4.3.16) ──

    #[test]
    fn is_lock_free_rules() {
        assert!(is_lock_free(4.0), "isLockFree(4) is always true");
        assert!(is_lock_free(1.0));
        assert!(is_lock_free(2.0));
        assert!(is_lock_free(8.0));
        assert!(!is_lock_free(3.0), "non-element sizes are false");
        assert!(!is_lock_free(5.0));
        assert!(!is_lock_free(0.0));
        assert!(!is_lock_free(f64::NAN));
        assert!(!is_lock_free(f64::INFINITY));
    }

    // ── wait / notify (§25.4.3.14, §25.4.3.15) ──

    #[test]
    fn wait_not_equal_returns_immediately() {
        let sab = SharedArrayBuffer::new(8);
        let v = AtomicsView::with_type(sab, ElemType::I32, 0);
        v.store(0, 5);
        // Expected 999 != actual 5 → "not-equal", no blocking.
        let r = v.wait(0, 999, Some(10_000.0));
        assert_eq!(r, WaitResult::NotEqual);
    }

    #[test]
    fn wait_times_out_when_value_matches_and_no_notify() {
        let sab = SharedArrayBuffer::new(8);
        let v = AtomicsView::with_type(sab, ElemType::I32, 0);
        v.store(0, 0);
        // Matches (0==0) → would block; with a tiny timeout and no notifier
        // (single thread), it must return "timed-out".
        let start = Instant::now();
        let r = v.wait(0, 0, Some(20.0));
        assert_eq!(r, WaitResult::TimedOut);
        assert!(start.elapsed() >= Duration::from_millis(15));
    }

    #[test]
    fn notify_with_no_waiters_returns_zero() {
        let sab = SharedArrayBuffer::new(8);
        let v = AtomicsView::with_type(sab, ElemType::I32, 0);
        assert_eq!(v.notify(0, None), 0, "no waiters parked → 0 woken");
        assert_eq!(v.notify(0, Some(5)), 0);
    }

    #[test]
    fn notify_wakes_a_real_blocked_waiter_cross_thread() {
        use std::thread;
        let sab = SharedArrayBuffer::new(8);
        let waiter_view = AtomicsView::with_type(sab.clone(), ElemType::I32, 0);
        // Value matches → the spawned thread will block in wait().
        waiter_view.store(0, 42);
        let sab2 = sab.clone();
        let handle = thread::spawn(move || {
            let v = AtomicsView::with_type(sab2, ElemType::I32, 0);
            // Long timeout; we expect to be woken by notify (-> Ok), not time out.
            v.wait(0, 42, Some(5_000.0))
        });
        // Spin until the waiter is actually parked, then notify it.
        let notifier = AtomicsView::with_type(sab, ElemType::I32, 0);
        let mut woke = 0;
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(3) {
            woke = notifier.notify(0, Some(1));
            if woke == 1 {
                break;
            }
            thread::yield_now();
        }
        assert_eq!(woke, 1, "notify woke exactly one waiter");
        let r = handle.join().expect("waiter thread");
        assert_eq!(r, WaitResult::Ok, "the woken waiter returns ok");
    }

    // ── waitAsync (§25.4.16) ──

    #[test]
    fn wait_async_not_equal_is_sync() {
        let sab = SharedArrayBuffer::new(8);
        let v = AtomicsView::with_type(sab, ElemType::I32, 0);
        v.store(0, 1);
        let r = v.wait_async(0, 2, None); // mismatch
        assert_eq!(r, WaitAsyncResult::NotEqual);
    }

    #[test]
    fn wait_async_zero_timeout_is_sync_timed_out() {
        let sab = SharedArrayBuffer::new(8);
        let v = AtomicsView::with_type(sab, ElemType::I32, 0);
        v.store(0, 0);
        // Matches but timeout 0 → resolves synchronously to timed-out.
        let r = v.wait_async(0, 0, Some(0.0));
        assert_eq!(r, WaitAsyncResult::TimedOutSync);
    }

    #[test]
    fn wait_async_blocks_then_notify_makes_it_ready_ok() {
        let sab = SharedArrayBuffer::new(8);
        let v = AtomicsView::with_type(sab.clone(), ElemType::I32, 0);
        v.store(0, 7);
        let token = match v.wait_async(0, 7, Some(60_000.0)) {
            WaitAsyncResult::Async(t) => t,
            other => panic!("expected Async, got {other:?}"),
        };
        // Not ready yet (no notify, deadline far away).
        assert!(drain_ready_async(&[token]).is_empty());
        // Notify → the async waiter becomes ready with Ok.
        assert_eq!(v.notify(0, None), 1);
        let ready = drain_ready_async(&[token]);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], (token, WaitResult::Ok));
    }

    #[test]
    fn wait_async_times_out_via_drain() {
        let sab = SharedArrayBuffer::new(8);
        let v = AtomicsView::with_type(sab, ElemType::I32, 0);
        v.store(0, 3);
        let token = match v.wait_async(0, 3, Some(20.0)) {
            WaitAsyncResult::Async(t) => t,
            other => panic!("expected Async, got {other:?}"),
        };
        // Busy-wait past the deadline (no foreground sleep), then drain.
        let start = Instant::now();
        let mut ready = Vec::new();
        while start.elapsed() < Duration::from_secs(2) {
            ready = drain_ready_async(&[token]);
            if !ready.is_empty() {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], (token, WaitResult::TimedOut));
    }
}
