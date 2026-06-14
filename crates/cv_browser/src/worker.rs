//! M8a — REAL Web Workers.
//!
//! A `new Worker(url)` spawns a dedicated OS thread that builds its OWN
//! `cv_js::Interp` (the interp is `!Send` — its global `Env` is `Rc<RefCell<…>>`
//! — so the page interp can never move; we build a *fresh* realm on the worker
//! thread, exactly like the renderer/compositor threads do). The worker installs
//! the worker-valid global subset (NO DOM/window/document/storage), runs the
//! worker script, then enters a message-pump loop.
//!
//! ## Crossing the `!Send` boundary
//! A `cv_js::Value` is `!Send` (every container is an `Rc`). `postMessage` must
//! therefore SERIALIZE the value to a `Send` byte form on the sender thread and
//! REBUILD a fresh `Value` on the receiver thread. We reuse the existing
//! tagged-binary codec (`idb_persist::encode_value_blob` / `decode_value_blob`),
//! which covers exactly the structured-clone data set
//! (Undefined/Null/Bool/Number/String/Array/Object). ArrayBuffer/TypedArray/Date
//! ride through as plain objects carrying sentinel props (`_isArrayBuffer` +
//! `_bytes`, `_typedarray` + `_bytes`, `_isDate` + `_time`) — the codec preserves
//! them byte-for-value-identically, so they round-trip with zero new tags.
//!
//! ## SharedArrayBuffer
//! A SAB is the ONE message that produces real concurrency: it is shared by Arc
//! REFERENCE, never copied. In the value tree it is replaced by a placeholder
//! `{_isSharedArrayBuffer:true,_sabId,byteLength}` (carried through the byte
//! codec), and the real `cv_js::sab::SharedArrayBuffer` (an `Arc<Vec<AtomicI32>>`,
//! `Send`) rides out-of-band in `ClonePayload.sabs`. On receive the same id is
//! re-registered in the process-global SAB registry (a no-op if already present),
//! so both threads' `Atomics` resolve the SAME atomic words.
//!
//! ## Non-cloneable values
//! `encode_value` silently encodes Function/BcClosure as `Null` — WRONG for
//! postMessage. So we run a cloneability pass FIRST that throws `DataCloneError`
//! (code 25) on Function/NativeFunction/BcClosure/Symbol/circular refs, never a
//! silent null.
//!
//! ## "Main thread" = the RENDERER thread
//! The page interp runs on the off-main renderer thread (`offmain_renderer`), not
//! the Win32 UI thread. So worker→main delivery means worker→renderer: the
//! renderer self-clocks at 16ms and DRAINS a renderer-thread-local registry each
//! tick (the proven `REQUESTED_NAV` idiom). Replies are fired as a MAIN-THREAD
//! TASK via `enqueue_microtask`, never synchronously inside `try_recv`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use cv_js::Value;

use crate::scheduler::{Scheduler, SchedRef};
use crate::{
    build_crypto_object, drain_scheduler, fetch_body_for_url, install_shared_memory, make_sab_wrapper,
    process_now_ms, read_data_url_bytes, sab_lookup, sab_ref_of, sab_register_with_id,
};

// ── Process-global worker id allocator ───────────────────────────────────────
static WORKER_NEXT_ID: AtomicU32 = AtomicU32::new(1);

pub fn next_worker_id_pub() -> u32 {
    WORKER_NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// `true` when real-thread Workers are enabled (default ON). Off → the existing
/// loopback stub stays verbatim as the safe fallback. Mirrors
/// `observers_real_enabled()` / `offmain_compositor_enabled()`.
pub fn worker_real_enabled() -> bool {
    thread_local! {
        static EN: bool = std::env::var("CV_REAL_WORKERS")
            .map(|v| v != "0")
            .unwrap_or(true);
    }
    EN.with(|v| *v)
}

// ── The Send message types ───────────────────────────────────────────────────

/// A structured-clone payload that is `Send`: a self-describing byte blob plus
/// the SAB Arcs carried out-of-band (shared by reference, not serialized).
#[derive(Debug)]
pub struct ClonePayload {
    pub blob: Vec<u8>,
    /// (`sab_id`, the shared buffer). The blob carries placeholder objects
    /// referencing these ids; on receive we re-register the Arc under its id.
    pub sabs: Vec<(u64, cv_js::sab::SharedArrayBuffer)>,
}

// `ClonePayload` is `Send`: `Vec<u8>` is `Send`, and `SharedArrayBuffer` is
// `Arc<Vec<AtomicI32>>` (Clone + Send + Sync). Assert it at compile time.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<ClonePayload>();
};

/// Main(renderer) → worker.
pub enum WorkerMsg {
    Post { payload: ClonePayload },
    Terminate,
}

/// Worker → main(renderer).
pub enum MainMsg {
    Post { payload: ClonePayload },
    Error { name: String, message: String, stack: String },
    Closed,
}

// ── Cloneability + serialization ─────────────────────────────────────────────

/// A DataCloneError thrown when a value cannot be structured-cloned. Mirrors
/// `make_data_clone_error` (code 25) but kept local so the worker module is
/// self-contained.
fn data_clone_error(msg: &str) -> cv_js::JsError {
    crate::make_data_clone_error_pub(msg)
}

/// Walk the value, REJECTING anything not structured-cloneable BEFORE encoding
/// (functions would otherwise be silently nulled by the codec). Collect SAB
/// handles into `sabs` and substitute a placeholder object for each so the byte
/// codec carries no shared bytes. Returns the rewritten (SAB-substituted) value.
fn cloneability_pass(
    v: &Value,
    sabs: &mut Vec<(u64, cv_js::sab::SharedArrayBuffer)>,
    seen: &mut Vec<usize>,
) -> Result<Value, cv_js::JsError> {
    use std::cell::RefCell;
    use std::rc::Rc;
    use cv_js::OrderedMap as HashMap;
    match v {
        Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_) => Err(
            data_clone_error("A function could not be cloned."),
        ),
        // Symbols are represented as `@@symbol(...)` / `@@sym:...` strings.
        Value::String(s) if cv_js::is_symbol_key(s) => {
            Err(data_clone_error("A Symbol could not be cloned."))
        }
        Value::Array(a) => {
            let id = Rc::as_ptr(a) as usize;
            if seen.contains(&id) {
                return Err(data_clone_error("Circular reference detected."));
            }
            seen.push(id);
            let items: Vec<Value> = a.borrow().clone();
            let mut out = Vec::with_capacity(items.len());
            for item in &items {
                out.push(cloneability_pass(item, sabs, seen)?);
            }
            seen.pop();
            Ok(Value::Array(Rc::new(RefCell::new(out))))
        }
        Value::Object(o) => {
            // A SharedArrayBuffer wrapper — collect its Arc out-of-band and
            // substitute the placeholder (which the byte codec preserves).
            if let Some((sab_id, byte_len)) = sab_ref_of(v) {
                if let Some(buf) = sab_lookup(sab_id) {
                    if !sabs.iter().any(|(id, _)| *id == sab_id) {
                        sabs.push((sab_id, buf));
                    }
                }
                // The placeholder is exactly the wrapper object; encode it as-is.
                return Ok(make_sab_wrapper(sab_id, byte_len));
            }
            let id = Rc::as_ptr(o) as usize;
            if seen.contains(&id) {
                return Err(data_clone_error("Circular reference detected."));
            }
            seen.push(id);
            let pairs: Vec<(String, Value)> =
                o.borrow().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            let mut copy: HashMap<String, Value> = HashMap::new();
            for (k, val) in pairs {
                // Skip internal (\u{1}…) and symbol-keyed (@@…) props: they are
                // not structured-clone data and the codec drops internals anyway.
                if k.starts_with('\u{1}') || cv_js::is_symbol_key(&k) {
                    continue;
                }
                // A function-valued data prop also makes the object non-cloneable.
                if matches!(
                    val,
                    Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
                ) {
                    return Err(data_clone_error("A function could not be cloned."));
                }
                copy.insert(k, cloneability_pass(&val, sabs, seen)?);
            }
            seen.pop();
            Ok(Value::Object(Rc::new(RefCell::new(copy))))
        }
        // Primitives (number/string/bool/null/undefined/bigint) clone by value.
        other => Ok(other.clone()),
    }
}

/// Encode a value for postMessage: cloneability pass + SAB collect + byte codec.
/// Errors are real `DataCloneError` throws (synchronous in the caller).
pub fn encode_for_postmessage(v: &Value) -> Result<ClonePayload, cv_js::JsError> {
    let mut sabs: Vec<(u64, cv_js::sab::SharedArrayBuffer)> = Vec::new();
    let mut seen: Vec<usize> = Vec::new();
    let rewritten = cloneability_pass(v, &mut sabs, &mut seen)?;
    let blob = crate::idb_persist::encode_value_blob(&rewritten);
    Ok(ClonePayload { blob, sabs })
}

/// Decode a payload into a fresh `Value` in the receiving interp's heap, then
/// re-bind every SAB placeholder to the SAME shared Arc (re-registered under its
/// original id). Returns `Undefined` on a malformed blob (never panics).
pub fn decode_for_postmessage(payload: &ClonePayload) -> Value {
    // Re-register the shared Arcs under their original ids so this thread's
    // Atomics resolve the same storage (idempotent if already present).
    for (id, buf) in &payload.sabs {
        sab_register_with_id(*id, buf.clone());
    }
    crate::idb_persist::decode_value_blob(&payload.blob).unwrap_or(Value::Undefined)
    // The placeholder objects decode straight back into SAB wrappers (they ARE
    // `{_isSharedArrayBuffer,_sabId,byteLength}` objects), already bound to the
    // re-registered Arc — no tree-walk fix-up needed.
}

// ── The worker thread ────────────────────────────────────────────────────────

/// Per-worker shared shutdown flag. `terminate()` sets it; the pump checks it
/// each loop turn for prompt cancel.
pub type ShutdownFlag = Arc<AtomicBool>;

/// Spawn a real worker thread. Returns `Some(handle parts)` on success, `None`
/// if the thread builder fails (caller then reports `onerror`). Never panics.
#[allow(clippy::type_complexity)]
pub fn spawn_worker(
    id: u32,
    script_src: String,
    base_url: String,
    to_worker_rx: Receiver<WorkerMsg>,
    from_worker_tx: Sender<MainMsg>,
    shutdown: ShutdownFlag,
) -> Option<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name(format!("tb-worker-{id}"))
        // 64MB: big enough for the recursive tree-walk interp (renderer uses
        // 512MB, compositor 8MB — a sane middle for a worker).
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            worker_main(
                script_src,
                base_url,
                to_worker_rx,
                from_worker_tx,
                shutdown,
            )
        })
        .map_err(|e| cv_js::diag_log(&format!("[worker] spawn failed: {e}")))
        .ok()
}

/// Worker-thread entry. Builds a FRESH interp + worker global scope, runs the
/// script, then pumps messages until terminate/close/disconnect.
fn worker_main(
    script_src: String,
    base_url: String,
    to_worker_rx: Receiver<WorkerMsg>,
    from_worker_tx: Sender<MainMsg>,
    shutdown: ShutdownFlag,
) {
    use std::time::Duration;

    // A FRESH realm on THIS thread (the !Send global Env is created here).
    let mut interp = cv_js::Interp::new();
    interp.install_math();
    interp.install_json();
    interp.install_basic_globals();
    interp.install_date();
    interp.install_promise();
    // M9.1: workers inherit the creator's isolation; pass `true` so the JS
    // SharedArrayBuffer constructor stays available exactly as today (the M8a
    // worker tests build SABs Rust-side + use Atomics, but a worker may also
    // construct one — keep it enabled in the worker/test realm).
    install_shared_memory(&interp, true);

    // Worker-local scheduler + event loop (setTimeout/setInterval/queueMicrotask
    // fire ON the worker thread).
    let sched: SchedRef = Scheduler::new_ref();
    crate::install_event_loop(&interp, sched.clone());

    // Closing flag (worker-side self-terminate via `close()`).
    let closing = std::rc::Rc::new(std::cell::Cell::new(false));

    // Worker global scope: self/globalThis/postMessage/onmessage/addEventListener/
    // close/importScripts + TextEncoder/TextDecoder/crypto/performance/
    // structuredClone. NO DOM/window/document/storage.
    let onmessage_handlers = install_worker_global_scope(
        &mut interp,
        &from_worker_tx,
        &base_url,
        closing.clone(),
    );

    // Run the worker script. A throw here → fatal: report onerror + die.
    if let Err(e) = interp.run(&script_src) {
        let (name, message, stack) = describe_throw(&e);
        let _ = from_worker_tx.send(MainMsg::Error { name, message, stack });
        return;
    }
    interp.drain_microtasks();
    // The initial script may have scheduled timers/microtasks (e.g. setTimeout
    // 0 to defer setup) — drain once so they fire before the first message.
    drain_scheduler(&mut interp, &sched, 256);

    // ── Message-pump loop ──
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        match to_worker_rx.recv_timeout(Duration::from_millis(16)) {
            Ok(WorkerMsg::Post { payload }) => {
                let data = decode_for_postmessage(&payload);
                dispatch_message(&mut interp, &onmessage_handlers, data, &from_worker_tx);
                interp.drain_microtasks();
                drain_scheduler(&mut interp, &sched, 64);
            }
            Ok(WorkerMsg::Terminate) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Frame tick: drain the worker's own scheduler + microtasks so
                // setTimeout-in-worker callbacks fire even with no messages.
                drain_scheduler(&mut interp, &sched, 256);
                interp.drain_microtasks();
            }
        }
        // close() finishes the current task + microtask drain, then stops.
        if closing.get() {
            let _ = from_worker_tx.send(MainMsg::Closed);
            break;
        }
    }
    // Falling out of the loop drops `interp` + the worker Env ON this thread
    // (correct: the !Send graph never crosses back). `from_worker_tx` drops, so
    // the renderer's drain sees `Disconnected` and removes the registry entry.
}

/// Fire every registered `message` handler with a `MessageEvent {data,type}`.
/// `this` is the worker global `self`. A throw in a handler → onerror (does NOT
/// kill the worker).
fn dispatch_message(
    interp: &mut cv_js::Interp,
    handlers: &OnMessageHandlers,
    data: Value,
    from_worker_tx: &Sender<MainMsg>,
) {
    use std::cell::RefCell;
    use std::rc::Rc;
    use cv_js::OrderedMap as HashMap;
    let mut event: HashMap<String, Value> = HashMap::new();
    event.insert("data".into(), data);
    event.insert("type".into(), Value::String("message".into()));
    let event = Value::Object(Rc::new(RefCell::new(event)));

    let self_obj = interp.get_global("self").unwrap_or(Value::Undefined);
    // Merge `self.onmessage = fn` (read from the worker global) with the
    // addEventListener accumulators.
    let mut cbs: Vec<Value> = handler_prop(&self_obj, "onmessage");
    cbs.extend(handlers.borrow().iter().cloned());
    for cb in cbs {
        if matches!(cb, Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)) {
            if let Err(e) =
                interp.call_value_with_this(cb, self_obj.clone(), vec![event.clone()])
            {
                let (name, message, stack) = describe_throw(&e);
                let _ = from_worker_tx.send(MainMsg::Error { name, message, stack });
            }
        }
    }
}

/// Extract (name, message, stack) from a thrown value for an ErrorEvent.
fn describe_throw(e: &cv_js::JsError) -> (String, String, String) {
    match e {
        cv_js::JsError::Internal(s) => ("Error".into(), s.clone(), format!("Error: {s}")),
        cv_js::JsError::Throw(v) => {
            if let Value::Object(o) = v {
                let b = o.borrow();
                let get = |k: &str| {
                    b.get(k)
                        .map(|x| x.to_display_string())
                        .filter(|s| !s.is_empty())
                };
                let name = get("name").unwrap_or_else(|| "Error".into());
                let message = get("message").unwrap_or_default();
                let stack = get("stack").unwrap_or_else(|| format!("{name}: {message}"));
                (name, message, stack)
            } else {
                let s = v.to_display_string();
                ("Error".into(), s.clone(), format!("Error: {s}"))
            }
        }
    }
}

/// The shared `message`-handler list (both `self.onmessage =` and
/// `addEventListener("message", …)` accumulate here).
type OnMessageHandlers = std::rc::Rc<std::cell::RefCell<Vec<Value>>>;

/// Install the dedicated-worker global scope on `interp`. Returns the shared
/// handler list the pump dispatches to.
fn install_worker_global_scope(
    interp: &mut cv_js::Interp,
    from_worker_tx: &Sender<MainMsg>,
    base_url: &str,
    closing: std::rc::Rc<std::cell::Cell<bool>>,
) -> OnMessageHandlers {
    use std::cell::RefCell;
    use std::rc::Rc;
    use cv_js::OrderedMap as HashMap;

    let handlers: OnMessageHandlers = Rc::new(RefCell::new(Vec::new()));

    // `self` === `globalThis` === the realm global object.
    let global = interp.global_object();

    // postMessage(value): worker → main encode path.
    {
        let tx = from_worker_tx.clone();
        let post = cv_js::native_fn("postMessage", move |args| {
            let data = args.into_iter().next().unwrap_or(Value::Undefined);
            // Cloneability pass throws DataCloneError synchronously on bad input.
            let payload = encode_for_postmessage(&data)?;
            let _ = tx.send(MainMsg::Post { payload });
            Ok(Value::Undefined)
        });
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("postMessage".into(), post);
        }
    }

    // close(): set the closing flag; the pump stops after the current task.
    {
        let closing2 = closing.clone();
        let close = cv_js::native_fn("close", move |_| {
            closing2.set(true);
            Ok(Value::Undefined)
        });
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("close".into(), close);
        }
    }

    // addEventListener("message", cb) — accumulate; also `removeEventListener`.
    {
        let h = handlers.clone();
        let add = cv_js::native_fn("addEventListener", move |args| {
            let evt = args.first().map(|v| v.to_display_string()).unwrap_or_default();
            let cb = args.get(1).cloned().unwrap_or(Value::Null);
            if evt == "message"
                && matches!(
                    cb,
                    Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
                )
            {
                h.borrow_mut().push(cb);
            }
            Ok(Value::Undefined)
        });
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("addEventListener".into(), add);
        }
    }
    {
        let remove = cv_js::native_fn("removeEventListener", |_| Ok(Value::Undefined));
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("removeEventListener".into(), remove);
        }
    }

    // importScripts(...urls): SYNCHRONOUS fetch + run per url.
    {
        let base = base_url.to_string();
        // NB: `native_fn_with_interp` lets us run code in THIS interp.
        let import = cv_js::native_fn_with_interp("importScripts", move |interp, args| {
            for arg in &args {
                let url = arg.to_display_string();
                if url.is_empty() {
                    continue;
                }
                let src = match fetch_worker_script(&url, &base) {
                    Some(s) => s,
                    None => {
                        return Err(crate::make_data_clone_error_pub(&format!(
                            "Failed to load worker script: {url}"
                        )));
                    }
                };
                interp.run(&src)?;
                interp.drain_microtasks();
            }
            Ok(Value::Undefined)
        });
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("importScripts".into(), import);
        }
    }

    // Worker-valid web APIs: TextEncoder/TextDecoder, crypto, performance,
    // structuredClone. (SAB + Atomics already installed by install_shared_memory.)
    install_worker_text_codecs(interp);
    interp.define_global("crypto", build_crypto_object());
    interp.define_global("performance", build_worker_performance());
    install_worker_structured_clone(interp);

    // `self` / `globalThis` resolve to the realm global. `globalThis` is set up
    // by Interp::new; alias `self` to the same object so `self.X` and bare `X`
    // agree (DedicatedWorkerGlobalScope).
    interp.define_global("self", global.clone());
    interp.define_global("globalThis", global.clone());

    // onmessage settable cell: writing `self.onmessage = fn` must register the
    // handler. We expose a getter/setter pair by storing the value directly on
    // the global and having the pump read it. Simpler + robust: a native
    // accessor is overkill — instead the pump reads `self.onmessage` directly,
    // so seed it as Null and let JS overwrite it. The pump merges the
    // `onmessage` cell with the addEventListener list at dispatch time.
    if let Value::Object(g) = &global {
        g.borrow_mut().insert("onmessage".into(), Value::Null);
        g.borrow_mut().insert("onerror".into(), Value::Null);
    }

    handlers
}

/// Build a minimal worker `performance` object (now + timeOrigin), same
/// process-start origin as the page.
fn build_worker_performance() -> Value {
    use std::cell::RefCell;
    use std::rc::Rc;
    use cv_js::OrderedMap as HashMap;
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert(
        "now".into(),
        cv_js::native_fn("now", |_| Ok(Value::Number(process_now_ms()))),
    );
    let time_origin = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as f64;
    m.insert("timeOrigin".into(), Value::Number(time_origin));
    m.insert("mark".into(), cv_js::native_fn("mark", |_| Ok(Value::Undefined)));
    m.insert("measure".into(), cv_js::native_fn("measure", |_| Ok(Value::Undefined)));
    Value::Object(Rc::new(RefCell::new(m)))
}

/// TextEncoder/TextDecoder for the worker realm (UTF-8). Returns a Uint8Array
/// (built via the worker's own Uint8Array ctor) from `encode`, and a string from
/// `decode`. Mirrors the page's block but standalone.
fn install_worker_text_codecs(interp: &cv_js::Interp) {
    use std::cell::RefCell;
    use std::rc::Rc;
    use cv_js::OrderedMap as HashMap;

    let enc_ctor = cv_js::native_fn("TextEncoder", |_| {
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("encoding".into(), Value::String("utf-8".into()));
        m.insert(
            "encode".into(),
            cv_js::native_fn_with_interp("encode", |interp, args| {
                let s = args.first().map(|v| v.to_display_string()).unwrap_or_default();
                let arr: Vec<Value> = s.bytes().map(|b| Value::Number(f64::from(b))).collect();
                let bytes_arr = Value::Array(Rc::new(RefCell::new(arr)));
                if let Some(ctor) = interp.get_global("Uint8Array") {
                    if let Ok(ta) = interp.call_value(ctor, vec![bytes_arr.clone()]) {
                        if matches!(ta, Value::Object(_)) {
                            return Ok(ta);
                        }
                    }
                }
                Ok(bytes_arr)
            }),
        );
        Ok(Value::Object(Rc::new(RefCell::new(m))))
    });
    let mut enc_wrap: HashMap<String, Value> = HashMap::new();
    enc_wrap.insert("_construct".into(), enc_ctor);
    interp.define_global("TextEncoder", Value::Object(Rc::new(RefCell::new(enc_wrap))));

    let dec_ctor = cv_js::native_fn("TextDecoder", |args| {
        let encoding = args
            .first()
            .map(|v| v.to_display_string())
            .unwrap_or_else(|| "utf-8".into());
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("encoding".into(), Value::str(encoding));
        m.insert(
            "decode".into(),
            cv_js::native_fn("decode", |args| {
                let bytes: Vec<u8> = match args.first() {
                    Some(Value::Array(a)) => a.borrow().iter().map(|v| v.to_number() as u8).collect(),
                    Some(Value::Object(o)) => {
                        let m = o.borrow();
                        if let Some(Value::Array(a)) = m.get("_bytes") {
                            a.borrow().iter().map(|v| v.to_number() as u8).collect()
                        } else {
                            Vec::new()
                        }
                    }
                    _ => Vec::new(),
                };
                Ok(Value::str(String::from_utf8_lossy(&bytes).to_string()))
            }),
        );
        Ok(Value::Object(Rc::new(RefCell::new(m))))
    });
    let mut dec_wrap: HashMap<String, Value> = HashMap::new();
    dec_wrap.insert("_construct".into(), dec_ctor);
    interp.define_global("TextDecoder", Value::Object(Rc::new(RefCell::new(dec_wrap))));
}

/// `structuredClone` for the worker realm: encode + decode through the same
/// postMessage codec (deep copy with the same cloneable set). Throws
/// DataCloneError on non-cloneable input.
fn install_worker_structured_clone(interp: &cv_js::Interp) {
    let sc = cv_js::native_fn("structuredClone", |args| {
        let v = args.first().cloned().unwrap_or(Value::Undefined);
        let payload = encode_for_postmessage(&v)?;
        Ok(decode_for_postmessage(&payload))
    });
    interp.define_global("structuredClone", sc);
}

/// Install the worker TextEncoder/TextDecoder pair on `interp`. Exposed so the
/// Service Worker sub-realm (service_worker.rs) reuses the exact worker codecs.
pub fn install_worker_text_codecs_pub(interp: &cv_js::Interp) {
    install_worker_text_codecs(interp);
}

/// Install the worker `structuredClone` on `interp`. Exposed for the SW realm.
pub fn install_worker_structured_clone_pub(interp: &cv_js::Interp) {
    install_worker_structured_clone(interp);
}

/// Fetch a worker script source (importScripts / initial load). Supports
/// `data:` URLs (the oracle uses these — no network), `file:`, and `http(s):`,
/// resolving relative refs against `base`.
pub fn fetch_worker_script(url: &str, base: &str) -> Option<String> {
    // data: URLs resolve in-process (no network) — used by the oracle.
    if url.starts_with("data:") {
        let parsed = cv_url::Url::parse(url).ok()?;
        return read_data_url_bytes(&parsed).map(|b| String::from_utf8_lossy(&b).into_owned());
    }
    // Resolve against the base URL.
    let resolved = if let Ok(base_url) = cv_url::Url::parse(base) {
        base_url.resolve(url).ok()?
    } else {
        cv_url::Url::parse(url).ok()?
    };
    if resolved.scheme == cv_url::Scheme::Data {
        return read_data_url_bytes(&resolved).map(|b| String::from_utf8_lossy(&b).into_owned());
    }
    fetch_body_for_url(&resolved, 30_000)
        .ok()
        .map(|b| String::from_utf8_lossy(&b).into_owned())
}

// ── Renderer-side: the Worker handle + registry + inbox drain ────────────────

/// The page-side handle for one worker. Lives in the page interp as a JS object
/// AND is referenced by the renderer-thread registry (so the inbox drain reaches
/// the same onmessage/onerror cells). All `Rc` fields are renderer-thread-local
/// and never cross the boundary; `to_worker_tx` / `shutdown` are the only `Send`
/// bits, used to talk to the worker thread.
pub struct WorkerHandle {
    pub id: u32,
    pub to_worker_tx: Sender<WorkerMsg>,
    pub shutdown: ShutdownFlag,
    pub join: Option<std::thread::JoinHandle<()>>,
    /// The Worker JS object (for `this` binding when firing onmessage/onerror).
    pub js_object: Value,
    /// `onmessage =` cell + `addEventListener("message")` accumulators.
    pub onmessage: std::rc::Rc<std::cell::RefCell<Vec<Value>>>,
    /// `onerror =` cell + `addEventListener("error")` accumulators.
    pub onerror: std::rc::Rc<std::cell::RefCell<Vec<Value>>>,
    pub terminated: bool,
}

thread_local! {
    /// Renderer-thread registry of live workers (mirrors the REQUESTED_NAV
    /// idiom). Each entry pairs the from_worker receiver with the handle so the
    /// ticker drain can fire onmessage/onerror as MAIN-THREAD TASKS.
    static WORKER_REGISTRY: std::cell::RefCell<
        Vec<(u32, Receiver<MainMsg>, std::rc::Rc<std::cell::RefCell<WorkerHandle>>)>,
    > = const { std::cell::RefCell::new(Vec::new()) };
}

/// Register a worker on the renderer thread.
pub fn register_worker(
    id: u32,
    rx: Receiver<MainMsg>,
    handle: std::rc::Rc<std::cell::RefCell<WorkerHandle>>,
) {
    WORKER_REGISTRY.with(|r| r.borrow_mut().push((id, rx, handle)));
}

/// Drain all worker inboxes (renderer thread). Fires each `Worker.onmessage` /
/// `Worker.onerror` as a MAIN-THREAD TASK (`enqueue_microtask`) so they run in
/// the page event loop, NOT synchronously inside `try_recv`. Removes
/// disconnected/closed workers. Returns `true` if any work was enqueued.
pub fn drain_worker_inboxes(interp: &mut cv_js::Interp) -> bool {
    use std::cell::RefCell;
    use std::rc::Rc;
    use cv_js::OrderedMap as HashMap;

    let mut enqueued = false;
    // Snapshot the registry entries we need (id + a clone of the handle Rc + the
    // messages drained) WITHOUT holding the thread_local borrow across the
    // interp calls (enqueue_microtask is a cheap push, but stay re-entrancy safe).
    let mut to_remove: Vec<u32> = Vec::new();
    let mut deliveries: Vec<(Rc<RefCell<WorkerHandle>>, MainMsg)> = Vec::new();

    WORKER_REGISTRY.with(|r| {
        let reg = r.borrow();
        for (id, rx, handle) in reg.iter() {
            // Drop anything from a terminated worker (post-terminate no-op).
            let terminated = handle.borrow().terminated;
            loop {
                match rx.try_recv() {
                    Ok(msg) => {
                        if terminated {
                            // Discard: never deliver after terminate.
                            if matches!(msg, MainMsg::Closed) {
                                to_remove.push(*id);
                            }
                            continue;
                        }
                        if matches!(msg, MainMsg::Closed) {
                            to_remove.push(*id);
                        }
                        deliveries.push((handle.clone(), msg));
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        to_remove.push(*id);
                        break;
                    }
                }
            }
        }
    });

    for (handle, msg) in deliveries {
        match msg {
            MainMsg::Post { payload } => {
                let data = decode_for_postmessage(&payload);
                let mut event: HashMap<String, Value> = HashMap::new();
                event.insert("data".into(), data);
                event.insert("type".into(), Value::String("message".into()));
                let event = Value::Object(Rc::new(RefCell::new(event)));
                let (this_obj, cbs) = {
                    let h = handle.borrow();
                    // Merge `worker.onmessage = fn` (read from the JS object's
                    // `onmessage` prop) with the addEventListener accumulators.
                    let mut cbs = handler_prop(&h.js_object, "onmessage");
                    cbs.extend(h.onmessage.borrow().iter().cloned());
                    (h.js_object.clone(), cbs)
                };
                for cb in cbs {
                    if matches!(
                        cb,
                        Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
                    ) {
                        enqueue_with_this(interp, cb, this_obj.clone(), vec![event.clone()]);
                        enqueued = true;
                    }
                }
            }
            MainMsg::Error { name, message, stack } => {
                let mut ev: HashMap<String, Value> = HashMap::new();
                ev.insert("message".into(), Value::str(message.clone()));
                ev.insert("filename".into(), Value::String("".into()));
                ev.insert("lineno".into(), Value::Number(0.0));
                ev.insert("type".into(), Value::String("error".into()));
                let mut err: HashMap<String, Value> = HashMap::new();
                err.insert("name".into(), Value::str(name));
                err.insert("message".into(), Value::str(message));
                err.insert("stack".into(), Value::str(stack));
                err.insert("_isError".into(), Value::Bool(true));
                ev.insert("error".into(), Value::Object(Rc::new(RefCell::new(err))));
                let event = Value::Object(Rc::new(RefCell::new(ev)));
                let (this_obj, cbs) = {
                    let h = handle.borrow();
                    let mut cbs = handler_prop(&h.js_object, "onerror");
                    cbs.extend(h.onerror.borrow().iter().cloned());
                    (h.js_object.clone(), cbs)
                };
                for cb in cbs {
                    if matches!(
                        cb,
                        Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
                    ) {
                        enqueue_with_this(interp, cb, this_obj.clone(), vec![event.clone()]);
                        enqueued = true;
                    }
                }
            }
            MainMsg::Closed => {
                handle.borrow_mut().terminated = true;
            }
        }
    }

    if !to_remove.is_empty() {
        WORKER_REGISTRY.with(|r| {
            r.borrow_mut().retain(|(id, _, _)| !to_remove.contains(id));
        });
    }

    enqueued
}

/// Read a single callable from a JS object's event-handler property
/// (`worker.onmessage = fn`). Returns an empty Vec if it isn't a callable.
fn handler_prop(obj: &Value, key: &str) -> Vec<Value> {
    if let Value::Object(o) = obj {
        if let Some(cb) = o.borrow().get(key) {
            if matches!(
                cb,
                Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
            ) {
                return vec![cb.clone()];
            }
        }
    }
    Vec::new()
}

/// Enqueue a callback as a main-thread task with a bound `this`. The interp's
/// `enqueue_microtask` doesn't take a `this`, so we wrap the call in a native
/// thunk that invokes `call_value_with_this` when the microtask runs.
fn enqueue_with_this(interp: &cv_js::Interp, cb: Value, this_obj: Value, args: Vec<Value>) {
    let thunk = cv_js::native_fn_with_interp("__tb_worker_dispatch", move |interp, _| {
        interp.call_value_with_this(cb.clone(), this_obj.clone(), args.clone())
    });
    interp.enqueue_microtask(thunk, Vec::new());
}

/// Count of live workers (test helper).
pub fn live_worker_count() -> usize {
    WORKER_REGISTRY.with(|r| r.borrow().len())
}

/// Clear the registry (test isolation between cases on the same thread).
pub fn clear_worker_registry() {
    WORKER_REGISTRY.with(|r| r.borrow_mut().clear());
}

// ════════════════════════════════════════════════════════════════════════
// THE ORACLE — provably real (off-thread) workers. All in-process, bounded,
// no live network (worker source passed inline). Each test spawns a real
// worker OS thread and exercises a minimal renderer-style drain loop.
// ════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod worker_oracle {
    use super::*;
    use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
    use std::time::{Duration, Instant};
    use cv_js::Value;

    /// Spawn a real worker thread with `src` as the worker script. Returns the
    /// send/recv channels + shutdown + join handle. The base URL is irrelevant
    /// for inline source (no importScripts), so we pass "about:blank".
    fn spawn_test_worker(
        src: &str,
    ) -> (
        Sender<WorkerMsg>,
        Receiver<MainMsg>,
        ShutdownFlag,
        std::thread::JoinHandle<()>,
    ) {
        let (to_tx, to_rx) = std::sync::mpsc::channel::<WorkerMsg>();
        let (from_tx, from_rx) = std::sync::mpsc::channel::<MainMsg>();
        let shutdown: ShutdownFlag = Arc::new(AtomicBool::new(false));
        let join = spawn_worker(
            next_worker_id_pub(),
            src.to_string(),
            "about:blank".to_string(),
            to_rx,
            from_tx,
            shutdown.clone(),
        )
        .expect("worker thread spawned");
        (to_tx, from_rx, shutdown, join)
    }

    /// Encode a JS value built on a throwaway interp into a payload, post it.
    fn post(to_tx: &Sender<WorkerMsg>, v: &Value) {
        let payload = encode_for_postmessage(v).expect("cloneable");
        to_tx.send(WorkerMsg::Post { payload }).expect("send");
    }

    /// Drive the from_worker receiver up to `timeout`, returning the FIRST
    /// `MainMsg::Post` decoded into a Value, or `None` on timeout.
    fn await_reply(from_rx: &Receiver<MainMsg>, timeout: Duration) -> Option<Value> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.checked_duration_since(Instant::now())?;
            match from_rx.recv_timeout(remaining.min(Duration::from_millis(200))) {
                Ok(MainMsg::Post { payload }) => return Some(decode_for_postmessage(&payload)),
                Ok(MainMsg::Error { name, message, .. }) => {
                    panic!("worker error {name}: {message}");
                }
                Ok(MainMsg::Closed) => return None,
                Err(RecvTimeoutError::Timeout) => {
                    if Instant::now() >= deadline {
                        return None;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        }
    }

    /// Build a small object `{key: number}` value (on a throwaway interp's heap).
    fn obj_num(key: &str, n: f64) -> Value {
        use std::cell::RefCell;
        use std::rc::Rc;
        use cv_js::OrderedMap as HashMap;
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert(key.to_string(), Value::Number(n));
        Value::Object(Rc::new(RefCell::new(m)))
    }

    fn as_number(v: &Value) -> f64 {
        match v {
            Value::Number(n) => *n,
            _ => f64::NAN,
        }
    }

    // (a) REAL-COMPUTE round-trip — the non-loopback proof. The worker SUMS
    // 1..=n and posts the COMPUTED sum (55), not the echoed input {n:10}.
    #[test]
    fn worker_real_compute_roundtrip() {
        let src = "self.onmessage = function(e){ var n=e.data.n, s=0; for(var i=1;i<=n;i++) s+=i; self.postMessage(s); };";
        let (to_tx, from_rx, shutdown, join) = spawn_test_worker(src);
        post(&to_tx, &obj_num("n", 10.0));
        let reply = await_reply(&from_rx, Duration::from_secs(3)).expect("reply within 3s");
        // 55 = the COMPUTED sum, NOT the {n:10} object a loopback would echo.
        assert_eq!(as_number(&reply), 55.0, "worker computed 1+..+10 = 55");
        assert!(
            !matches!(reply, Value::Object(_)),
            "a loopback stub would deliver an object, not the number 55"
        );
        shutdown.store(true, Ordering::Release);
        let _ = to_tx.send(WorkerMsg::Terminate);
        let _ = join.join();
    }

    // (b) OFF-MAIN-THREAD via SharedArrayBuffer + Atomics: the worker (a
    // DISTINCT thread) flips word[1] from 0→1; the test main thread never
    // writes it, yet observes the transition — proving the OTHER thread ran.
    #[test]
    fn worker_runs_off_main_thread() {
        // Build a SAB on the test (main) thread, store a sentinel at [0].
        let sab = cv_js::sab::SharedArrayBuffer::new(16);
        let id = crate::sab_register(sab.clone());
        crate::sab_lookup(id)
            .map(|s| cv_js::sab::AtomicsView::new(s).store(0, 7))
            .unwrap();
        let wrapper = crate::make_sab_wrapper(id, 16);

        // Worker: when it receives the SAB, it stores [0]+100 into [1] and posts.
        let src = "self.onmessage = function(e){ var v=Atomics.load(e.data,0); Atomics.store(e.data,1,v+100); self.postMessage('done'); };";
        let (to_tx, from_rx, shutdown, join) = spawn_test_worker(src);
        // Before posting, [1] is 0 (we never wrote it on this thread).
        assert_eq!(cv_js::sab::AtomicsView::new(sab.clone()).load(1), 0);
        post(&to_tx, &wrapper);
        let reply = await_reply(&from_rx, Duration::from_secs(3)).expect("worker replied");
        assert_eq!(reply.to_display_string(), "done");
        // The OTHER thread wrote 7+100=107 into the SHARED word [1].
        assert_eq!(
            cv_js::sab::AtomicsView::new(sab).load(1),
            107,
            "the worker thread (not this test thread) wrote the shared word"
        );
        shutdown.store(true, Ordering::Release);
        let _ = to_tx.send(WorkerMsg::Terminate);
        let _ = join.join();
    }

    // (c) structuredClone fidelity: a nested object/array + ArrayBuffer + Date
    // echoes back deep-equal; numbers stay numbers, ArrayBuffer bytes + Date
    // time survive the boundary.
    #[test]
    fn worker_structured_clone_fidelity() {
        use std::cell::RefCell;
        use std::rc::Rc;
        use cv_js::OrderedMap as HashMap;
        // Build {a:[1,2,{b:"x"}], buf:<ArrayBuffer 3 bytes>, d:<Date 12345>}.
        let inner = {
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("b".into(), Value::String("x".into()));
            Value::Object(Rc::new(RefCell::new(m)))
        };
        let arr = Value::Array(Rc::new(RefCell::new(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            inner,
        ])));
        let buf = {
            let bytes = Value::Array(Rc::new(RefCell::new(vec![
                Value::Number(10.0),
                Value::Number(20.0),
                Value::Number(30.0),
            ])));
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("_isArrayBuffer".into(), Value::Bool(true));
            m.insert("byteLength".into(), Value::Number(3.0));
            m.insert("_bytes".into(), bytes);
            Value::Object(Rc::new(RefCell::new(m)))
        };
        let date = {
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("_isDate".into(), Value::Bool(true));
            m.insert("_time".into(), Value::Number(12345.0));
            Value::Object(Rc::new(RefCell::new(m)))
        };
        let payload_val = {
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("a".into(), arr);
            m.insert("buf".into(), buf);
            m.insert("d".into(), date);
            Value::Object(Rc::new(RefCell::new(m)))
        };

        let src = "self.onmessage = function(e){ self.postMessage(e.data); };";
        let (to_tx, from_rx, shutdown, join) = spawn_test_worker(src);
        post(&to_tx, &payload_val);
        let reply = await_reply(&from_rx, Duration::from_secs(3)).expect("echo reply");

        // Deep structural checks on the returned value.
        let Value::Object(o) = &reply else {
            panic!("expected object back, got {reply:?}");
        };
        let b = o.borrow();
        // a:[1,2,{b:"x"}] — numbers stay numbers, nesting preserved.
        let Some(Value::Array(a)) = b.get("a") else {
            panic!("a is not an array");
        };
        let a = a.borrow();
        assert_eq!(as_number(&a[0]), 1.0);
        assert_eq!(as_number(&a[1]), 2.0);
        let Value::Object(nested) = &a[2] else {
            panic!("a[2] not object");
        };
        assert_eq!(
            nested.borrow().get("b").map(|v| v.to_display_string()),
            Some("x".to_string())
        );
        // buf: ArrayBuffer bytes identical.
        let Some(Value::Object(buf2)) = b.get("buf") else {
            panic!("buf missing");
        };
        let buf2 = buf2.borrow();
        assert!(matches!(buf2.get("_isArrayBuffer"), Some(Value::Bool(true))));
        let Some(Value::Array(bytes2)) = buf2.get("_bytes") else {
            panic!("buf bytes missing");
        };
        let got: Vec<f64> = bytes2.borrow().iter().map(as_number).collect();
        assert_eq!(got, vec![10.0, 20.0, 30.0], "ArrayBuffer bytes round-trip");
        // d: Date time identical.
        let Some(Value::Object(d2)) = b.get("d") else {
            panic!("date missing");
        };
        assert_eq!(d2.borrow().get("_time").map(as_number), Some(12345.0));

        shutdown.store(true, Ordering::Release);
        let _ = to_tx.send(WorkerMsg::Terminate);
        let _ = join.join();
    }

    // (d) NON-CLONEABLE → DataCloneError, thrown synchronously by the
    // cloneability pass (NOT silently nulled by the codec).
    #[test]
    fn worker_postmessage_function_throws_dataclone() {
        let func = cv_js::native_fn("noop", |_| Ok(Value::Undefined));
        let err = encode_for_postmessage(&func).expect_err("function must reject");
        match err {
            cv_js::JsError::Throw(Value::Object(o)) => {
                let b = o.borrow();
                assert_eq!(
                    b.get("name").map(|v| v.to_display_string()),
                    Some("DataCloneError".to_string())
                );
                assert_eq!(b.get("code").map(as_number), Some(25.0));
            }
            other => panic!("expected DataCloneError throw, got {other:?}"),
        }
        // Also: an object carrying a function property is non-cloneable.
        use std::cell::RefCell;
        use std::rc::Rc;
        use cv_js::OrderedMap as HashMap;
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("fn".into(), cv_js::native_fn("f", |_| Ok(Value::Undefined)));
        let obj = Value::Object(Rc::new(RefCell::new(m)));
        assert!(
            encode_for_postmessage(&obj).is_err(),
            "object with a function prop is non-cloneable"
        );
    }

    // (e) terminate() stops delivery: post {n:5} (worker would reply 15),
    // terminate, then post {n:99}; NO reply for {n:99} ever arrives.
    #[test]
    fn worker_terminate_stops_delivery() {
        let src = "self.onmessage = function(e){ var n=e.data.n, s=0; for(var i=1;i<=n;i++) s+=i; self.postMessage(s); };";
        let (to_tx, from_rx, shutdown, join) = spawn_test_worker(src);
        post(&to_tx, &obj_num("n", 5.0));
        // The pre-terminate reply (15) may or may not have arrived; drain it.
        let _first = await_reply(&from_rx, Duration::from_millis(800));

        // Terminate: prompt cancel + graceful stop.
        shutdown.store(true, Ordering::Release);
        let _ = to_tx.send(WorkerMsg::Terminate);
        // Post AFTER terminate — must never produce a reply.
        let payload = encode_for_postmessage(&obj_num("n", 99.0)).unwrap();
        let _ = to_tx.send(WorkerMsg::Post { payload });

        // Drain for a bounded window: the only acceptable value is 15 (if the
        // pre-terminate reply was still in flight). 5050 (=1+..+99→4950? no,
        // sum 1..99 = 4950) must NEVER appear.
        let deadline = Instant::now() + Duration::from_millis(1200);
        while let Some(rem) = deadline.checked_duration_since(Instant::now()) {
            match from_rx.recv_timeout(rem.min(Duration::from_millis(200))) {
                Ok(MainMsg::Post { payload }) => {
                    let v = as_number(&decode_for_postmessage(&payload));
                    assert_ne!(v, 4950.0, "post-terminate {{n:99}} must not be processed");
                    assert_eq!(v, 15.0, "only the pre-terminate reply (15) is allowed");
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        let _ = join.join();
    }

    // (f) SharedArrayBuffer shared by REFERENCE: worker does Atomics.add(view,0,41)
    // then posts "done"; main initialized [0]=1, so after the reply main reads 42.
    // A copy would leave main's view at 1.
    #[test]
    fn worker_sab_shared_by_reference() {
        let sab = cv_js::sab::SharedArrayBuffer::new(8);
        let id = crate::sab_register(sab.clone());
        cv_js::sab::AtomicsView::new(sab.clone()).store(0, 1);
        let wrapper = crate::make_sab_wrapper(id, 8);

        let src = "self.onmessage = function(e){ Atomics.add(e.data,0,41); self.postMessage('done'); };";
        let (to_tx, from_rx, shutdown, join) = spawn_test_worker(src);
        post(&to_tx, &wrapper);
        let reply = await_reply(&from_rx, Duration::from_secs(3)).expect("worker replied");
        assert_eq!(reply.to_display_string(), "done");
        assert_eq!(
            cv_js::sab::AtomicsView::new(sab).load(0),
            42,
            "1 + 41 written by the worker through the SHARED Arc (not a copy)"
        );
        shutdown.store(true, Ordering::Release);
        let _ = to_tx.send(WorkerMsg::Terminate);
        let _ = join.join();
    }

    // Worker errors in the INITIAL script → MainMsg::Error (becomes onerror).
    #[test]
    fn worker_initial_script_error_reports_onerror() {
        let src = "throw new Error('boom');";
        let (to_tx, from_rx, shutdown, join) = spawn_test_worker(src);
        let mut saw_error = false;
        let deadline = Instant::now() + Duration::from_secs(2);
        while let Some(rem) = deadline.checked_duration_since(Instant::now()) {
            match from_rx.recv_timeout(rem.min(Duration::from_millis(200))) {
                Ok(MainMsg::Error { message, .. }) => {
                    assert!(message.contains("boom"), "error message carried: {message}");
                    saw_error = true;
                    break;
                }
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        assert!(saw_error, "initial-script throw reported as a worker error");
        shutdown.store(true, Ordering::Release);
        let _ = to_tx.send(WorkerMsg::Terminate);
        let _ = join.join();
    }

    // fetch_worker_script resolves a data: URL in-process (the blob/data path).
    #[test]
    fn fetch_worker_script_resolves_data_url() {
        let url = "data:text/javascript,self.postMessage(1)";
        let src = fetch_worker_script(url, "about:blank").expect("data url resolves");
        assert_eq!(src, "self.postMessage(1)");
    }

    // The worker realm OMITS DOM: reading `document`/`window` is a ReferenceError,
    // surfaced as a worker error.
    #[test]
    fn worker_has_no_dom() {
        let src = "self.onmessage = function(e){ try { var d = document; self.postMessage('HAS_DOM'); } catch(err){ self.postMessage('NO_DOM:'+err.name); } };";
        let (to_tx, from_rx, shutdown, join) = spawn_test_worker(src);
        post(&to_tx, &obj_num("x", 1.0));
        let reply = await_reply(&from_rx, Duration::from_secs(3)).expect("reply");
        let s = reply.to_display_string();
        assert!(
            s.starts_with("NO_DOM:"),
            "worker must not expose document; got {s}"
        );
        shutdown.store(true, Ordering::Release);
        let _ = to_tx.send(WorkerMsg::Terminate);
        let _ = join.join();
    }

    // CV_REAL_WORKERS flag default is ON (or honors an explicit override).
    #[test]
    fn worker_flag_default_on() {
        // Cannot mutate process env safely in parallel tests; just assert the
        // function is callable and returns a bool consistent with the env.
        let expected = std::env::var("CV_REAL_WORKERS")
            .map(|v| v != "0")
            .unwrap_or(true);
        assert_eq!(worker_real_enabled(), expected);
    }
}
