//! M8b — REAL Service Workers.
//!
//! A `navigator.serviceWorker.register(scriptURL, {scope})` builds a SECOND
//! `cv_js::Interp` — a SW *sub-realm* — co-located ON the renderer thread (NOT a
//! worker thread). The sub-realm runs the SW script, fires `install`/`activate`
//! lifecycle events (draining `waitUntil` promises for the precache step), and
//! is stored in a renderer-thread-local `SW_REGISTRY` keyed by scope.
//!
//! ## Why a same-thread sub-realm, not an M8a worker thread
//! `fetch_body_for_url` (main.rs) is a SYNCHRONOUS, blocking renderer-thread
//! call. SW fetch interception happens INSIDE it. If the SW lived on a worker
//! thread (M8a), the renderer would have to send a FetchEvent and BLOCK on
//! `recv()` for `respondWith` — but `respondWith` commonly resolves a promise
//! (`caches.match(req).then(...)`) settled by the worker draining ITS scheduler,
//! risking a hang. And the M6 caches `store` is a renderer-thread-local
//! `Rc<RefCell<..>>` (NOT `Send`) — a worker-thread SW literally cannot touch
//! it. A same-thread sub-realm runs `respondWith` resolution fully synchronously
//! on one thread, shares the cache `store` Rc by reference, and never blocks.
//!
//! ## No deadlock / no hang
//! When the fetch handler is invoked we settle the `respondWith` value with a
//! BOUNDED `drain_microtasks` + `drain_scheduler` loop (256 iters). If the
//! promise is still pending after the budget → treat as "no response" → fall
//! through to network. No lock is ever held across a SW call; the registry
//! borrow is released (data cloned out) before the realm is entered. Errors at
//! every step degrade to a normal network fetch — never panic, never spin.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::OnceLock;

use cv_js::OrderedMap as HashMap;
use cv_js::{Value, native_fn, native_fn_with_interp};

use crate::scheduler::{SchedRef, Scheduler};
use crate::{
    build_crypto_object, cache_store_shared, drain_scheduler, fetch_body_for_url,
    install_event_loop, make_response_object, process_now_ms,
};
use crate::worker::{decode_for_postmessage, encode_for_postmessage, fetch_worker_script};

/// `true` when real Service Workers are enabled (default ON). Off → the existing
/// registration stub stays verbatim as the safe fallback. Mirrors
/// `worker_real_enabled()` / `caches_persist_enabled()`.
pub fn service_workers_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        !matches!(
            std::env::var("CV_SERVICE_WORKERS").ok().as_deref(),
            Some("0") | Some("false") | Some("off")
        )
    })
}

// ── The SW realm: a distinct Interp + its lifecycle handler lists ─────────────

/// A SW lifecycle/state phase (mirrors the spec ServiceWorker.state machine).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SwState {
    Installing,
    Installed,
    Activating,
    Activated,
    Redundant,
}

impl SwState {
    fn as_str(self) -> &'static str {
        match self {
            SwState::Installing => "installing",
            SwState::Installed => "installed",
            SwState::Activating => "activating",
            SwState::Activated => "activated",
            SwState::Redundant => "redundant",
        }
    }
}

/// The per-type event handler accumulator (install/activate/fetch/message), the
/// `skipWaiting` flag cell, and the `waitUntil` promise list shared with events.
struct SwHandlers {
    /// type ("install"/"activate"/"fetch"/"message") -> listener callbacks.
    by_type: Rc<RefCell<HashMap<String, Vec<Value>>>>,
    /// `self.skipWaiting()` set this.
    skip_waiting: Rc<Cell<bool>>,
}

/// One registered Service Worker — its sub-realm Interp + scheduler + handlers +
/// lifecycle state. All `Rc`/`Interp` fields are renderer-thread-local and never
/// cross a thread boundary (the SW runs on the same thread as the page).
pub struct SwRegistration {
    pub scope: String,
    pub script_url: String,
    /// The SW sub-realm interpreter (a DISTINCT realm from the page).
    interp: RefCell<cv_js::Interp>,
    sched: SchedRef,
    handlers: SwHandlers,
    pub state: Cell<SwState>,
    /// Guard against a SW's own `fetch()` re-entering interception for the same
    /// realm (a SW fetching its own scope must hit network, not itself).
    intercepting: Rc<Cell<bool>>,
}

thread_local! {
    /// Renderer-thread registry of controlling Service Workers, keyed implicitly
    /// by scope (longest-prefix match at lookup). Mirrors the WORKER_REGISTRY
    /// idiom. Empty on a non-SW page → `sw_try_intercept` early-returns on one
    /// cheap borrow + is_empty check (zero common-path overhead).
    static SW_REGISTRY: RefCell<Vec<Rc<SwRegistration>>> = const { RefCell::new(Vec::new()) };

    /// The page-side `navigator.serviceWorker` "message" handler list (page
    /// gets a real addEventListener("message") accumulator + onmessage). A SW
    /// `client.postMessage(v)` fires these on the PAGE interp synchronously.
    static SW_PAGE_MESSAGE_HANDLERS: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
}

/// True if any SW is registered (the common-path fast check).
pub fn sw_registry_empty() -> bool {
    SW_REGISTRY.with(|r| r.borrow().is_empty())
}

/// Clear the SW registry + page message handlers (test isolation).
pub fn clear_sw_registry() {
    SW_REGISTRY.with(|r| r.borrow_mut().clear());
    SW_PAGE_MESSAGE_HANDLERS.with(|r| r.borrow_mut().clear());
}

/// Count of registered SWs (test helper).
pub fn sw_registration_count() -> usize {
    SW_REGISTRY.with(|r| r.borrow().len())
}

// ── Promise inspection (the synchronous respondWith-settle primitive) ─────────

/// Read (state, value) of a promise-shaped value via the public `_isPromise` /
/// `_state` / `_value` fields (interp.rs make_pending_promise shape). Returns
/// None for a non-promise.
fn inspect_promise(v: &Value) -> Option<(String, Value)> {
    if let Value::Object(o) = v {
        let m = o.borrow();
        if matches!(m.get("_isPromise"), Some(Value::Bool(true))) {
            let state = m
                .get("_state")
                .map(|x| x.to_display_string())
                .unwrap_or_default();
            let value = m.get("_value").cloned().unwrap_or(Value::Undefined);
            return Some((state, value));
        }
    }
    None
}

/// Synchronously settle a (possibly-promise) value by draining the SW realm's
/// microtasks + scheduler up to `budget` iterations. Returns:
/// - `Some(Ok(v))`   — a plain value, or a fulfilled promise's value.
/// - `Some(Err(reason))` — a rejected promise.
/// - `None`          — still pending after the budget (caller falls through).
fn settle_value(
    interp: &mut cv_js::Interp,
    sched: &SchedRef,
    v: Value,
    budget: usize,
) -> Option<Result<Value, Value>> {
    // Non-promise: already settled.
    let Some((state0, val0)) = inspect_promise(&v) else {
        return Some(Ok(v));
    };
    match state0.as_str() {
        "fulfilled" => {
            // The fulfilled value may itself be a thenable (chained promise) —
            // settle recursively (bounded).
            if inspect_promise(&val0).is_some() {
                return settle_value(interp, sched, val0, budget);
            }
            return Some(Ok(val0));
        }
        "rejected" => return Some(Err(val0)),
        _ => {}
    }
    // Pending — drive the event loop until it settles or the budget runs out.
    for _ in 0..budget {
        interp.drain_microtasks();
        let fired = drain_scheduler(interp, sched, 64);
        if let Some((state, val)) = inspect_promise(&v) {
            match state.as_str() {
                "fulfilled" => {
                    if inspect_promise(&val).is_some() {
                        return settle_value(interp, sched, val, budget);
                    }
                    return Some(Ok(val));
                }
                "rejected" => return Some(Err(val)),
                _ => {}
            }
        }
        if !fired {
            // Nothing left to fire and still pending → drain one more microtask
            // round, then bail if still unsettled (no infinite spin).
            interp.drain_microtasks();
            if let Some((state, val)) = inspect_promise(&v) {
                match state.as_str() {
                    "fulfilled" => {
                        if inspect_promise(&val).is_some() {
                            return settle_value(interp, sched, val, budget);
                        }
                        return Some(Ok(val));
                    }
                    "rejected" => return Some(Err(val)),
                    _ => return None,
                }
            }
            return None;
        }
    }
    None
}

// ── Response body extraction (handles BOTH Response shapes) ───────────────────

/// Extract response bytes from a settled `respondWith` value. Handles:
/// - a `make_response_object`-shaped Response: prefer its stored body, else
///   settle `text()`/`arrayBuffer()`.
/// - a cache plain object `{_body: String|bytes, ...}`.
/// Returns `None` (→ network fall-through) if no body can be extracted.
fn extract_response_bytes(
    interp: &mut cv_js::Interp,
    sched: &SchedRef,
    resp: &Value,
) -> Option<Vec<u8>> {
    let Value::Object(o) = resp else {
        // A bare string Response body (uncommon but tolerated).
        if let Value::String(s) = resp {
            return Some(s.to_string().into_bytes());
        }
        return None;
    };
    // 1) Cache plain form `{_body: ...}` (opfs persisted shape).
    {
        let b = o.borrow();
        if let Some(body) = b.get("_body") {
            return Some(value_to_bytes(body));
        }
    }
    // 2) make_response_object form: call text() (returns a settled promise) and
    //    settle it. arrayBuffer() yields an ArrayBuffer wrapper {_bytes}.
    let text_fn = o.borrow().get("text").cloned();
    if let Some(text_fn) = text_fn {
        if matches!(
            text_fn,
            Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
        ) {
            if let Ok(p) = interp.call_value_with_this(text_fn, resp.clone(), Vec::new()) {
                if let Some(Ok(settled)) = settle_value(interp, sched, p, 256) {
                    return Some(value_to_bytes(&settled));
                }
            }
        }
    }
    let ab_fn = o.borrow().get("arrayBuffer").cloned();
    if let Some(ab_fn) = ab_fn {
        if matches!(
            ab_fn,
            Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
        ) {
            if let Ok(p) = interp.call_value_with_this(ab_fn, resp.clone(), Vec::new()) {
                if let Some(Ok(settled)) = settle_value(interp, sched, p, 256) {
                    return Some(value_to_bytes(&settled));
                }
            }
        }
    }
    None
}

/// Convert a JS value to bytes: a String → its UTF-8; an ArrayBuffer/typed-array
/// wrapper → its `_bytes` array; a number array → its bytes.
fn value_to_bytes(v: &Value) -> Vec<u8> {
    match v {
        Value::String(s) => s.to_string().into_bytes(),
        Value::Array(a) => a.borrow().iter().map(|x| x.to_number() as u8).collect(),
        Value::Object(o) => {
            let b = o.borrow();
            if let Some(Value::Array(bytes)) = b.get("_bytes") {
                return bytes.borrow().iter().map(|x| x.to_number() as u8).collect();
            }
            if let Some(body) = b.get("_body") {
                return value_to_bytes(body);
            }
            v.to_display_string().into_bytes()
        }
        other => other.to_display_string().into_bytes(),
    }
}

// ── The SW sub-realm builder (worker recipe MINUS DOM) ────────────────────────

/// Build a fresh SW sub-realm `Interp` + scheduler + `ServiceWorkerGlobalScope`,
/// run the script, and return the registration with handler lists populated.
/// Lifecycle (install/activate) is fired by `register_service_worker`. Returns
/// `Err(message)` if the script throws on first run.
fn build_sw_realm(
    script_src: &str,
    script_url: &str,
    scope: &str,
) -> Result<Rc<SwRegistration>, String> {
    let mut interp = cv_js::Interp::new();
    interp.install_math();
    interp.install_json();
    interp.install_basic_globals();
    interp.install_date();
    interp.install_promise();
    // M9.1: service workers inherit isolation; pass `true` to keep the JS
    // SharedArrayBuffer constructor available (non-regressing for M8a tests).
    crate::install_shared_memory(&interp, true);

    let sched: SchedRef = Scheduler::new_ref();
    install_event_loop(&interp, sched.clone());

    let by_type: Rc<RefCell<HashMap<String, Vec<Value>>>> = Rc::new(RefCell::new(HashMap::new()));
    let skip_waiting = Rc::new(Cell::new(false));
    let intercepting = Rc::new(Cell::new(false));

    install_sw_global_scope(
        &mut interp,
        script_url,
        scope,
        by_type.clone(),
        skip_waiting.clone(),
        intercepting.clone(),
    );

    // Run the SW script: registers self.addEventListener("install"/...) handlers.
    if let Err(e) = interp.run(script_src) {
        return Err(describe_throw(&e));
    }
    interp.drain_microtasks();
    drain_scheduler(&mut interp, &sched, 64);

    Ok(Rc::new(SwRegistration {
        scope: scope.to_string(),
        script_url: script_url.to_string(),
        interp: RefCell::new(interp),
        sched,
        handlers: SwHandlers {
            by_type,
            skip_waiting,
        },
        state: Cell::new(SwState::Installing),
        intercepting,
    }))
}

fn describe_throw(e: &cv_js::JsError) -> String {
    match e {
        cv_js::JsError::Internal(s) => s.clone(),
        cv_js::JsError::Throw(v) => v.to_display_string(),
    }
}

/// Install `ServiceWorkerGlobalScope` on the SW sub-realm (modeled on
/// `install_worker_global_scope` but SW-flavored: per-type listeners, lifecycle
/// events, respondWith, skipWaiting, clients, the SHARED caches, fetch,
/// importScripts). NO DOM/window/document.
fn install_sw_global_scope(
    interp: &mut cv_js::Interp,
    script_url: &str,
    scope: &str,
    by_type: Rc<RefCell<HashMap<String, Vec<Value>>>>,
    skip_waiting: Rc<Cell<bool>>,
    intercepting: Rc<Cell<bool>>,
) {
    let global = interp.global_object();

    // self === globalThis === the realm global object. NO window/document.
    interp.define_global("self", global.clone());
    interp.define_global("globalThis", global.clone());

    // addEventListener(type, cb) — accumulate into the per-type list.
    {
        let h = by_type.clone();
        let add = native_fn("addEventListener", move |args| {
            let evt = args.first().map(|v| v.to_display_string()).unwrap_or_default();
            let cb = args.get(1).cloned().unwrap_or(Value::Null);
            if matches!(
                cb,
                Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
            ) {
                h.borrow_mut().entry(evt).or_default().push(cb);
            }
            Ok(Value::Undefined)
        });
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("addEventListener".into(), add);
        }
    }
    {
        let h = by_type.clone();
        let remove = native_fn("removeEventListener", move |args| {
            let evt = args.first().map(|v| v.to_display_string()).unwrap_or_default();
            let cb = args.get(1).cloned().unwrap_or(Value::Null);
            if let Some(list) = h.borrow_mut().get_mut(&evt) {
                list.retain(|c| !values_same_callable(c, &cb));
            }
            Ok(Value::Undefined)
        });
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("removeEventListener".into(), remove);
        }
    }

    // skipWaiting(): set the realm flag; returns a resolved promise.
    {
        let sw = skip_waiting.clone();
        let f = native_fn("skipWaiting", move |_| {
            sw.set(true);
            Ok(cv_js::interp::make_settled_promise(true, Value::Undefined))
        });
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("skipWaiting".into(), f);
        }
    }

    // clients: V1-minimal (we control the one page on register).
    {
        let mut clients: HashMap<String, Value> = HashMap::new();
        clients.insert(
            "claim".into(),
            native_fn("claim", |_| {
                Ok(cv_js::interp::make_settled_promise(true, Value::Undefined))
            }),
        );
        clients.insert(
            "matchAll".into(),
            native_fn("matchAll", |_| {
                Ok(cv_js::interp::make_settled_promise(
                    true,
                    Value::Array(Rc::new(RefCell::new(Vec::new()))),
                ))
            }),
        );
        clients.insert(
            "get".into(),
            native_fn("get", |_| {
                Ok(cv_js::interp::make_settled_promise(true, Value::Undefined))
            }),
        );
        if let Value::Object(g) = &global {
            g.borrow_mut()
                .insert("clients".into(), Value::Object(Rc::new(RefCell::new(clients))));
        }
    }

    // caches: the SHARED M6 Cache API bound over the same `store` Rc as the page
    // (cache_store_shared). Uses REAL settled promises so caches.match(req)
    // .then(r => r || fetch(req)) chains correctly and is settle-able.
    if let Value::Object(g) = &global {
        g.borrow_mut().insert("caches".into(), build_sw_caches());
    }

    // fetch(req): REAL network — calls fetch_body_for_url (renderer thread, same
    // as page fetch) and returns a make_response_object settled promise. The
    // `intercepting` guard ensures the SW's own fetch does NOT re-enter SW
    // interception (network, not itself).
    {
        let f = native_fn_with_interp("fetch", move |_interp, args| {
            let url_str = match args.first() {
                Some(Value::Object(o)) => o
                    .borrow()
                    .get("url")
                    .map(|u| u.to_display_string())
                    .unwrap_or_default(),
                Some(v) => v.to_display_string(),
                None => String::new(),
            };
            let parsed = match cv_url::Url::parse(&url_str) {
                Ok(u) => u,
                Err(e) => {
                    return Ok(cv_js::interp::make_settled_promise(
                        false,
                        Value::str(format!("Failed to fetch: {e}")),
                    ));
                }
            };
            // fetch_body_for_url consults the SW registry, but the per-realm
            // `intercepting` flag (set during interception) prevents the SW from
            // intercepting its OWN fetch — a global SW_FETCH_DEPTH guard handles
            // the cross-realm case. Here we go straight to network.
            match fetch_body_for_url(&parsed, 30_000) {
                Ok(body) => Ok(cv_js::interp::make_settled_promise(
                    true,
                    make_response_object(body, 200, "OK".into(), Vec::new(), url_str),
                )),
                Err(e) => Ok(cv_js::interp::make_settled_promise(
                    false,
                    Value::str(format!("Failed to fetch: {e}")),
                )),
            }
        });
        if let Value::Object(g) = &global {
            g.borrow_mut().insert("fetch".into(), f);
        }
    }
    let _ = intercepting;

    // importScripts(...urls): synchronous fetch+run (reuse worker path).
    {
        let base = script_url.to_string();
        let import = native_fn_with_interp("importScripts", move |interp, args| {
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

    // TextEncoder/TextDecoder, crypto, performance, structuredClone.
    crate::worker::install_worker_text_codecs_pub(interp);
    interp.define_global("crypto", build_crypto_object());
    interp.define_global("performance", build_sw_performance());
    crate::worker::install_worker_structured_clone_pub(interp);

    // self.registration — minimal (scope + live state slots filled in lifecycle).
    {
        let mut reg: HashMap<String, Value> = HashMap::new();
        reg.insert("scope".into(), Value::str(scope.to_string()));
        reg.insert("installing".into(), Value::Null);
        reg.insert("waiting".into(), Value::Null);
        reg.insert("active".into(), Value::Null);
        if let Value::Object(g) = &global {
            g.borrow_mut()
                .insert("registration".into(), Value::Object(Rc::new(RefCell::new(reg))));
        }
    }

    // onmessage/oninstall/... settable cells (merged with addEventListener lists
    // at dispatch time).
    if let Value::Object(g) = &global {
        let mut b = g.borrow_mut();
        b.insert("onmessage".into(), Value::Null);
        b.insert("oninstall".into(), Value::Null);
        b.insert("onactivate".into(), Value::Null);
        b.insert("onfetch".into(), Value::Null);
    }
}

fn values_same_callable(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::NativeFunction(x), Value::NativeFunction(y)) => Rc::ptr_eq(x, y),
        (Value::Function(x), Value::Function(y)) => Rc::ptr_eq(x, y),
        (Value::BcClosure(x), Value::BcClosure(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

fn build_sw_performance() -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert(
        "now".into(),
        native_fn("now", |_| Ok(Value::Number(process_now_ms()))),
    );
    let time_origin = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as f64;
    m.insert("timeOrigin".into(), Value::Number(time_origin));
    m.insert("mark".into(), native_fn("mark", |_| Ok(Value::Undefined)));
    m.insert("measure".into(), native_fn("measure", |_| Ok(Value::Undefined)));
    Value::Object(Rc::new(RefCell::new(m)))
}

// ── The SW-realm Cache API (shared store, REAL promises) ──────────────────────

/// URL key extraction (mirrors `extract_cache_url` in main.rs).
fn extract_cache_url(v: &Value) -> String {
    match v {
        Value::Object(o) => {
            let b = o.borrow();
            if let Some(u) = b.get("url") {
                return u.to_display_string();
            }
            if let Some(u) = b.get("href") {
                return u.to_display_string();
            }
            v.to_display_string()
        }
        _ => v.to_display_string(),
    }
}

/// Build a `caches` object for the SW realm over the SHARED store Rc, using REAL
/// settled promises (so `.then` chains + can be synchronously settled).
fn build_sw_caches() -> Value {
    let store = cache_store_shared();
    let mk = |v: Value| cv_js::interp::make_settled_promise(true, v);

    let mut m: HashMap<String, Value> = HashMap::new();

    // caches.open(name) → resolved promise of a Cache object over the named map.
    {
        let store = store.clone();
        let open = native_fn("open", move |args| {
            let name = args.first().map(|v| v.to_display_string()).unwrap_or_default();
            let entries: Rc<RefCell<HashMap<String, Value>>> = {
                let mut s = store.borrow_mut();
                if let Some(existing) = s.get(&name) {
                    existing.clone()
                } else {
                    let mut map: HashMap<String, Value> = HashMap::new();
                    if crate::caches_persist_enabled_pub() {
                        for (url, val) in crate::opfs::load_cache(&name) {
                            map.insert(url, val);
                        }
                    }
                    let rc = Rc::new(RefCell::new(map));
                    s.insert(name.clone(), rc.clone());
                    rc
                }
            };
            Ok(cv_js::interp::make_settled_promise(
                true,
                build_sw_cache_object(name, entries),
            ))
        });
        m.insert("open".into(), open);
    }

    // caches.match(req) — search all named caches; first hit or undefined.
    {
        let store = store.clone();
        let match_fn = native_fn("match", move |args| {
            let key = args.first().map(extract_cache_url).unwrap_or_default();
            let hit = store
                .borrow()
                .values()
                .find_map(|entries| entries.borrow().get(&key).cloned());
            Ok(cv_js::interp::make_settled_promise(
                true,
                hit.unwrap_or(Value::Undefined),
            ))
        });
        m.insert("match".into(), match_fn);
    }

    // caches.has(name)
    {
        let store = store.clone();
        m.insert(
            "has".into(),
            native_fn("has", move |args| {
                let name = args.first().map(|v| v.to_display_string()).unwrap_or_default();
                Ok(cv_js::interp::make_settled_promise(
                    true,
                    Value::Bool(store.borrow().contains_key(&name)),
                ))
            }),
        );
    }

    // caches.keys()
    {
        let store = store.clone();
        m.insert(
            "keys".into(),
            native_fn("keys", move |_| {
                let ks: Vec<Value> = store
                    .borrow()
                    .keys()
                    .map(|k| Value::str(k.clone()))
                    .collect();
                Ok(mk(Value::Array(Rc::new(RefCell::new(ks)))))
            }),
        );
    }

    // caches.delete(name)
    {
        let store = store.clone();
        m.insert(
            "delete".into(),
            native_fn("delete", move |args| {
                let name = args.first().map(|v| v.to_display_string()).unwrap_or_default();
                let removed = store.borrow_mut().remove(&name).is_some();
                if crate::caches_persist_enabled_pub() {
                    crate::opfs::delete_cache(&name);
                }
                Ok(cv_js::interp::make_settled_promise(true, Value::Bool(removed)))
            }),
        );
    }

    Value::Object(Rc::new(RefCell::new(m)))
}

/// Build a single `Cache` object (put/match/add/addAll/delete/keys) over the
/// given per-name entries map, using REAL settled promises.
fn build_sw_cache_object(name: String, entries: Rc<RefCell<HashMap<String, Value>>>) -> Value {
    let mut c: HashMap<String, Value> = HashMap::new();

    let put_entries = entries.clone();
    let put_name = name.clone();
    c.insert(
        "put".into(),
        native_fn("put", move |args| {
            let key = args.first().map(extract_cache_url).unwrap_or_default();
            let val = args.get(1).cloned().unwrap_or(Value::Undefined);
            put_entries.borrow_mut().insert(key, val);
            flush_named_cache(&put_name, &put_entries);
            Ok(cv_js::interp::make_settled_promise(true, Value::Undefined))
        }),
    );

    let match_entries = entries.clone();
    c.insert(
        "match".into(),
        native_fn("match", move |args| {
            let key = args.first().map(extract_cache_url).unwrap_or_default();
            let v = match_entries
                .borrow()
                .get(&key)
                .cloned()
                .unwrap_or(Value::Undefined);
            Ok(cv_js::interp::make_settled_promise(true, v))
        }),
    );

    // add(req)/addAll(reqs): REAL fetch + put (offline-first precache). Each URL
    // is fetched via the network and its body stored as a make_response_object.
    let add_entries = entries.clone();
    let add_name = name.clone();
    c.insert(
        "add".into(),
        native_fn("add", move |args| {
            let key = args.first().map(extract_cache_url).unwrap_or_default();
            sw_cache_add_one(&add_entries, &add_name, &key);
            Ok(cv_js::interp::make_settled_promise(true, Value::Undefined))
        }),
    );
    let addall_entries = entries.clone();
    let addall_name = name.clone();
    c.insert(
        "addAll".into(),
        native_fn("addAll", move |args| {
            if let Some(Value::Array(a)) = args.first() {
                for item in a.borrow().iter() {
                    let key = extract_cache_url(item);
                    sw_cache_add_one(&addall_entries, &addall_name, &key);
                }
            }
            Ok(cv_js::interp::make_settled_promise(true, Value::Undefined))
        }),
    );

    let delete_entries = entries.clone();
    let delete_name = name.clone();
    c.insert(
        "delete".into(),
        native_fn("delete", move |args| {
            let key = args.first().map(extract_cache_url).unwrap_or_default();
            let removed = delete_entries.borrow_mut().remove(&key).is_some();
            if removed {
                flush_named_cache(&delete_name, &delete_entries);
            }
            Ok(cv_js::interp::make_settled_promise(true, Value::Bool(removed)))
        }),
    );

    let keys_entries = entries;
    c.insert(
        "keys".into(),
        native_fn("keys", move |_| {
            let ks: Vec<Value> = keys_entries
                .borrow()
                .keys()
                .map(|k| Value::str(k.clone()))
                .collect();
            Ok(cv_js::interp::make_settled_promise(
                true,
                Value::Array(Rc::new(RefCell::new(ks))),
            ))
        }),
    );

    Value::Object(Rc::new(RefCell::new(c)))
}

/// Fetch one URL over the network and store it as a Response in the cache map.
fn sw_cache_add_one(entries: &Rc<RefCell<HashMap<String, Value>>>, name: &str, url: &str) {
    if url.is_empty() {
        return;
    }
    if let Ok(parsed) = cv_url::Url::parse(url) {
        if let Ok(body) = fetch_body_for_url(&parsed, 30_000) {
            let resp = make_response_object(body, 200, "OK".into(), Vec::new(), url.to_string());
            entries.borrow_mut().insert(url.to_string(), resp);
            flush_named_cache(name, entries);
        }
    }
}

fn flush_named_cache(name: &str, entries: &Rc<RefCell<HashMap<String, Value>>>) {
    if !crate::caches_persist_enabled_pub() {
        return;
    }
    let snapshot: Vec<(String, Value)> = entries
        .borrow()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    crate::opfs::persist_cache(name, &snapshot);
}

// ── Lifecycle: fire install then activate, draining waitUntil ─────────────────

/// Construct an ExtendableEvent (install/activate) with a `waitUntil` that
/// collects promises into the shared list, and dispatch every listener of
/// `event_type`. Then drain the SW realm until the waitUntil promises settle
/// (bounded). Returns false if a waitUntil promise rejected (abort activation).
fn fire_lifecycle_event(reg: &SwRegistration, event_type: &str) -> bool {
    let waituntil_promises: Rc<RefCell<Vec<Value>>> = Rc::new(RefCell::new(Vec::new()));

    // Snapshot listeners (drop the by_type borrow before entering the realm).
    let mut listeners: Vec<Value> = reg
        .handlers
        .by_type
        .borrow()
        .get(event_type)
        .cloned()
        .unwrap_or_default();
    // Merge the on<type> property (self.oninstall = fn).
    {
        let interp = reg.interp.borrow();
        let on_name = format!("on{event_type}");
        if let Some(cb) = interp.get_global(&on_name) {
            if matches!(
                cb,
                Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
            ) {
                listeners.insert(0, cb);
            }
        }
    }

    let event = {
        let wp = waituntil_promises.clone();
        let mut ev: HashMap<String, Value> = HashMap::new();
        ev.insert("type".into(), Value::str(event_type.to_string()));
        ev.insert(
            "waitUntil".into(),
            native_fn("waitUntil", move |args| {
                if let Some(p) = args.first() {
                    wp.borrow_mut().push(p.clone());
                }
                Ok(Value::Undefined)
            }),
        );
        Value::Object(Rc::new(RefCell::new(ev)))
    };

    let self_obj = reg
        .interp
        .borrow()
        .get_global("self")
        .unwrap_or(Value::Undefined);

    for cb in listeners {
        let mut interp = reg.interp.borrow_mut();
        if let Err(e) = interp.call_value_with_this(cb, self_obj.clone(), vec![event.clone()]) {
            cv_js::diag_log(&format!(
                "[sw] {event_type} listener threw: {}",
                describe_throw(&e)
            ));
        }
    }

    // Drain the waitUntil promises (the precache / cleanup step), bounded.
    let promises: Vec<Value> = waituntil_promises.borrow().clone();
    let mut all_ok = true;
    for p in promises {
        let mut interp = reg.interp.borrow_mut();
        match settle_value(&mut interp, &reg.sched, p, 256) {
            Some(Ok(_)) => {}
            Some(Err(_)) => all_ok = false, // a waitUntil rejection → abort
            None => all_ok = false,         // still pending past budget → abort
        }
    }
    {
        let mut interp = reg.interp.borrow_mut();
        interp.drain_microtasks();
    }
    all_ok
}

/// Register a Service Worker (the real flow). Returns the JS
/// `ServiceWorkerRegistration` value (a live object), or an `Err(reason)` for a
/// rejected registration promise. All synchronous on the renderer thread.
pub fn register_service_worker(
    page_interp: &cv_js::Interp,
    script_url: &str,
    scope: &str,
    page_base: &str,
) -> Result<Value, String> {
    // 1) Fetch the SW script (data: in-process; real URLs via fetch_body_for_url).
    let src = fetch_worker_script(script_url, page_base)
        .ok_or_else(|| format!("Failed to fetch service worker script: {script_url}"))?;

    // 2/3/4) Build the sub-realm + run the script.
    let reg = build_sw_realm(&src, script_url, scope)?;

    // 5) Fire install (waitUntil precache). On settle: installing → installed.
    reg.state.set(SwState::Installing);
    let install_ok = fire_lifecycle_event(&reg, "install");
    if !install_ok {
        // V1: a waitUntil rejection aborts activation; keep no controller.
        reg.state.set(SwState::Redundant);
        return Err("service worker install failed (waitUntil rejected)".to_string());
    }
    reg.state.set(SwState::Installed);

    // 6/7) skipWaiting is implicit (one client, no prior controller) → activate.
    reg.state.set(SwState::Activating);
    let activate_ok = fire_lifecycle_event(&reg, "activate");
    if !activate_ok {
        reg.state.set(SwState::Redundant);
        return Err("service worker activate failed (waitUntil rejected)".to_string());
    }
    reg.state.set(SwState::Activated);

    // 8) Mark controlling: insert into the registry (longest-scope match later).
    SW_REGISTRY.with(|r| {
        // Replace any prior registration for the SAME scope (update path).
        let mut reg_list = r.borrow_mut();
        reg_list.retain(|e| e.scope != scope);
        reg_list.push(reg.clone());
    });

    // 9) Build + return the JS ServiceWorkerRegistration object.
    let js_reg = build_js_registration(page_interp, &reg);

    // Set navigator.serviceWorker.controller to the active SW (clients.claim
    // makes the already-loaded page controlled immediately — V1, one client).
    set_page_controller(page_interp, &reg);

    Ok(js_reg)
}

/// Build the JS ServiceWorkerRegistration object returned by register(). Its
/// .active/.installing/.waiting reflect realm state; .unregister/.update/.scope
/// are live; .active.postMessage routes structuredClone(v) into the SW realm.
fn build_js_registration(_page_interp: &cv_js::Interp, reg: &Rc<SwRegistration>) -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert("scope".into(), Value::str(reg.scope.clone()));
    m.insert("installing".into(), Value::Null);
    m.insert("waiting".into(), Value::Null);
    m.insert(
        "active".into(),
        build_js_service_worker(reg, SwState::Activated),
    );

    // unregister(): drop the registry entry + clear the controller.
    {
        let scope = reg.scope.clone();
        m.insert(
            "unregister".into(),
            native_fn_with_interp("unregister", move |interp, _| {
                SW_REGISTRY.with(|r| r.borrow_mut().retain(|e| e.scope != scope));
                clear_page_controller(interp);
                Ok(cv_js::interp::make_settled_promise(true, Value::Bool(true)))
            }),
        );
    }

    // update(): re-fetch + re-run the script (V1: rebuild + re-fire lifecycle).
    {
        let scope = reg.scope.clone();
        let script_url = reg.script_url.clone();
        m.insert(
            "update".into(),
            native_fn_with_interp("update", move |interp, _| {
                let base = crate::current_window_href_pub(interp).unwrap_or_default();
                match register_service_worker(interp, &script_url, &scope, &base) {
                    Ok(new_reg) => Ok(cv_js::interp::make_settled_promise(true, new_reg)),
                    Err(e) => Ok(cv_js::interp::make_settled_promise(false, Value::str(e))),
                }
            }),
        );
    }

    m.insert(
        "addEventListener".into(),
        native_fn("addEventListener", |_| Ok(Value::Undefined)),
    );
    m.insert(
        "update_via_cache".into(),
        Value::String("imports".into()),
    );

    Value::Object(Rc::new(RefCell::new(m)))
}

/// Build a JS ServiceWorker object (.scriptURL/.state/.postMessage). postMessage
/// routes structuredClone(v) into the SW realm's "message" handlers (page→SW).
fn build_js_service_worker(reg: &Rc<SwRegistration>, state: SwState) -> Value {
    let mut sw: HashMap<String, Value> = HashMap::new();
    sw.insert("scriptURL".into(), Value::str(reg.script_url.clone()));
    sw.insert("state".into(), Value::String(state.as_str().into()));

    let reg_for_post = reg.clone();
    sw.insert(
        "postMessage".into(),
        native_fn_with_interp("postMessage", move |page_interp, args| {
            let data = args.into_iter().next().unwrap_or(Value::Undefined);
            post_message_to_sw(page_interp, &reg_for_post, data);
            Ok(Value::Undefined)
        }),
    );
    sw.insert(
        "addEventListener".into(),
        native_fn("addEventListener", |_| Ok(Value::Undefined)),
    );
    Value::Object(Rc::new(RefCell::new(sw)))
}

// ── postMessage: page → SW (synchronous, same thread) ─────────────────────────

/// Route a page→SW postMessage: encode via structuredClone, build a MessageEvent
/// with a `source` client (whose postMessage fires page handlers — SW→page),
/// dispatch the SW realm's "message" listeners, drain SW microtasks.
fn post_message_to_sw(page_interp: &cv_js::Interp, reg: &Rc<SwRegistration>, data: Value) {
    // Encode + decode through the structured-clone codec so the SW realm gets
    // its OWN copy (the two realms share nothing by reference).
    let decoded = match encode_for_postmessage(&data) {
        Ok(payload) => decode_for_postmessage(&payload),
        Err(_) => Value::Undefined,
    };

    // A `source` client whose postMessage fires the PAGE's serviceWorker
    // "message" handlers (SW→page). The page interp is captured so SW→page is
    // also synchronous on this thread.
    let source = build_sw_client_source();

    let mut ev: HashMap<String, Value> = HashMap::new();
    ev.insert("data".into(), decoded);
    ev.insert("type".into(), Value::String("message".into()));
    ev.insert("source".into(), source.clone());
    ev.insert(
        "origin".into(),
        Value::str(crate::current_window_href_pub(page_interp).unwrap_or_default()),
    );
    ev.insert(
        "ports".into(),
        Value::Array(Rc::new(RefCell::new(Vec::new()))),
    );
    let event = Value::Object(Rc::new(RefCell::new(ev)));

    // Dispatch the SW realm's "message" listeners (+ self.onmessage).
    let mut listeners: Vec<Value> = reg
        .handlers
        .by_type
        .borrow()
        .get("message")
        .cloned()
        .unwrap_or_default();
    let self_obj = {
        let interp = reg.interp.borrow();
        if let Some(cb) = interp.get_global("onmessage") {
            if matches!(
                cb,
                Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
            ) {
                listeners.insert(0, cb);
            }
        }
        interp.get_global("self").unwrap_or(Value::Undefined)
    };

    for cb in listeners {
        let mut interp = reg.interp.borrow_mut();
        if let Err(e) = interp.call_value_with_this(cb, self_obj.clone(), vec![event.clone()]) {
            cv_js::diag_log(&format!("[sw] message listener threw: {}", describe_throw(&e)));
        }
    }
    {
        let mut interp = reg.interp.borrow_mut();
        interp.drain_microtasks();
        drain_scheduler(&mut interp, &reg.sched, 64);
    }
}

/// Build a `client` object whose .postMessage(v) fires the PAGE's
/// navigator.serviceWorker "message" handlers (SW→page).
fn build_sw_client_source() -> Value {
    let mut client: HashMap<String, Value> = HashMap::new();
    client.insert("id".into(), Value::String("client-0".into()));
    client.insert("type".into(), Value::String("window".into()));
    client.insert(
        "postMessage".into(),
        native_fn_with_interp("postMessage", move |sw_interp, args| {
            let data = args.into_iter().next().unwrap_or(Value::Undefined);
            // Encode in the SW realm, decode into the page realm (own copy).
            let decoded = match encode_for_postmessage(&data) {
                Ok(payload) => decode_for_postmessage(&payload),
                Err(_) => Value::Undefined,
            };
            // We're currently executing INSIDE the SW realm interp (sw_interp).
            // The page "message" handlers must run on the PAGE interp. They were
            // registered there via navigator.serviceWorker.addEventListener.
            // We can fire them on the SAME interp object only if it is the page
            // interp — but during page→SW→page, sw_interp is the SW realm. So we
            // store the page handlers' callbacks and the page interp is reached
            // by deferring: we fire them via the page-message thread-local list,
            // calling each on sw_interp is WRONG. Instead, enqueue onto a pending
            // delivery list the page drains. For the synchronous round-trip in
            // V1 (single thread), we fire them directly on sw_interp because the
            // handler closures capture only page globals via the shared Value
            // graph — but to keep realms isolated we route through the page
            // delivery queue.
            deliver_message_to_page(sw_interp, decoded);
            Ok(Value::Undefined)
        }),
    );
    Value::Object(Rc::new(RefCell::new(client)))
}

thread_local! {
    /// SW→page messages awaiting delivery on the PAGE interp. Filled by a SW
    /// client.postMessage; drained by `drain_sw_page_messages` on the page tick
    /// AND synchronously after a page→SW postMessage round-trip.
    static SW_PAGE_INBOX: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
}

/// Called from the SW realm (client.postMessage). We cannot fire page handlers
/// on the SW interp, so queue the (already page-realm-decoded) data for the page
/// to drain. The page drains synchronously right after a postMessage round-trip
/// (see `flush_sw_page_inbox`) and on every tick.
fn deliver_message_to_page(_sw_interp: &mut cv_js::Interp, data: Value) {
    SW_PAGE_INBOX.with(|q| q.borrow_mut().push(data));
}

/// Drain the SW→page inbox: fire the page's navigator.serviceWorker "message"
/// handlers on the PAGE interp. Returns the number delivered.
pub fn flush_sw_page_inbox(page_interp: &mut cv_js::Interp) -> usize {
    let pending: Vec<Value> = SW_PAGE_INBOX.with(|q| std::mem::take(&mut *q.borrow_mut()));
    if pending.is_empty() {
        return 0;
    }
    let handlers: Vec<Value> = SW_PAGE_MESSAGE_HANDLERS.with(|h| h.borrow().clone());
    let mut delivered = 0;
    for data in pending {
        let mut ev: HashMap<String, Value> = HashMap::new();
        ev.insert("data".into(), data);
        ev.insert("type".into(), Value::String("message".into()));
        let event = Value::Object(Rc::new(RefCell::new(ev)));
        for cb in &handlers {
            if matches!(
                cb,
                Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
            ) {
                let _ = page_interp.call_value_with_this(cb.clone(), Value::Undefined, vec![event.clone()]);
                delivered += 1;
            }
        }
    }
    page_interp.drain_microtasks();
    delivered
}

/// Register a page-side navigator.serviceWorker "message" handler.
pub fn add_page_message_handler(cb: Value) {
    SW_PAGE_MESSAGE_HANDLERS.with(|h| h.borrow_mut().push(cb));
}

// ── Controller wiring on the page ─────────────────────────────────────────────

fn set_page_controller(page_interp: &cv_js::Interp, reg: &Rc<SwRegistration>) {
    if let Some(Value::Object(nav)) = page_interp.get_global("navigator") {
        let sw_container = nav.borrow().get("serviceWorker").cloned();
        if let Some(Value::Object(c)) = sw_container {
            c.borrow_mut().insert(
                "controller".into(),
                build_js_service_worker(reg, SwState::Activated),
            );
        }
    }
}

fn clear_page_controller(page_interp: &cv_js::Interp) {
    if let Some(Value::Object(nav)) = page_interp.get_global("navigator") {
        let sw_container = nav.borrow().get("serviceWorker").cloned();
        if let Some(Value::Object(c)) = sw_container {
            c.borrow_mut().insert("controller".into(), Value::Null);
        }
    }
}

// ── THE INTERCEPTION POINT ─────────────────────────────────────────────────────

thread_local! {
    /// Re-entrancy depth guard for SW fetch interception. A SW's own fetch()
    /// (cache-miss network) re-enters fetch_body_for_url; this counter ensures
    /// the inner call goes to the network and does NOT recursively try to
    /// SW-intercept its own request (no self-interception loop).
    static SW_INTERCEPT_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Try to serve `url` from a controlling Service Worker's fetch handler. Returns
/// `Some(bytes)` on a served response, `None` to fall through to network (no SW
/// for the scope / no fetch listener / no respondWith / handler error / pending
/// past budget / rejected / no extractable body). Synchronous, renderer-thread.
pub fn sw_try_intercept(url: &cv_url::Url) -> Option<Vec<u8>> {
    // Fast common-path: empty registry → one cheap borrow, return None.
    if SW_REGISTRY.with(|r| r.borrow().is_empty()) {
        return None;
    }
    // Re-entrancy guard: a SW's own fetch() must hit network, not re-intercept.
    if SW_INTERCEPT_DEPTH.with(|d| d.get()) > 0 {
        return None;
    }

    let url_str = url.to_string();

    // Find the controlling registration: longest scope prefix, activated, with
    // >=1 fetch listener. Clone the Rc out so we don't hold the registry borrow
    // across the realm call.
    let reg = SW_REGISTRY.with(|r| {
        let reg_list = r.borrow();
        let mut best: Option<Rc<SwRegistration>> = None;
        let mut best_len = 0usize;
        for e in reg_list.iter() {
            if e.state.get() != SwState::Activated {
                continue;
            }
            if !scope_matches(&e.scope, &url_str) {
                continue;
            }
            let has_fetch = e
                .handlers
                .by_type
                .borrow()
                .get("fetch")
                .map(|l| !l.is_empty())
                .unwrap_or(false);
            if !has_fetch {
                continue;
            }
            if e.scope.len() >= best_len {
                best_len = e.scope.len();
                best = Some(e.clone());
            }
        }
        best
    })?;

    // Build the FetchEvent: a Request {url,method,...}, respondWith capturing a
    // shared cell, waitUntil (no-op for fetch in V1), preventDefault.
    let responded: Rc<RefCell<Option<Value>>> = Rc::new(RefCell::new(None));

    let request = {
        let mut req: HashMap<String, Value> = HashMap::new();
        req.insert("url".into(), Value::str(url_str.clone()));
        req.insert("method".into(), Value::String("GET".into()));
        req.insert("mode".into(), Value::String("cors".into()));
        req.insert("credentials".into(), Value::String("same-origin".into()));
        req.insert(
            "headers".into(),
            Value::Object(Rc::new(RefCell::new(HashMap::new()))),
        );
        Value::Object(Rc::new(RefCell::new(req)))
    };

    let event = {
        let resp_cell = responded.clone();
        let mut ev: HashMap<String, Value> = HashMap::new();
        ev.insert("type".into(), Value::String("fetch".into()));
        ev.insert("request".into(), request);
        ev.insert(
            "respondWith".into(),
            native_fn("respondWith", move |args| {
                // respondWith may be called once; later calls are ignored (V1).
                if resp_cell.borrow().is_none() {
                    *resp_cell.borrow_mut() = Some(args.into_iter().next().unwrap_or(Value::Undefined));
                }
                Ok(Value::Undefined)
            }),
        );
        ev.insert(
            "waitUntil".into(),
            native_fn("waitUntil", |_| Ok(Value::Undefined)),
        );
        ev.insert(
            "preventDefault".into(),
            native_fn("preventDefault", |_| Ok(Value::Undefined)),
        );
        ev.insert(
            "clients".into(),
            Value::Object(Rc::new(RefCell::new(HashMap::new()))),
        );
        Value::Object(Rc::new(RefCell::new(ev)))
    };

    // Dispatch the fetch listeners. Enter the interception guard so the SW's own
    // fetch() (cache-miss network) does NOT re-intercept.
    let listeners: Vec<Value> = reg
        .handlers
        .by_type
        .borrow()
        .get("fetch")
        .cloned()
        .unwrap_or_default();

    let self_obj = reg
        .interp
        .borrow()
        .get_global("self")
        .unwrap_or(Value::Undefined);

    SW_INTERCEPT_DEPTH.with(|d| d.set(d.get() + 1));
    reg.intercepting.set(true);

    for cb in listeners {
        // A listener calling respondWith sets the cell; stop after the first.
        {
            let mut interp = reg.interp.borrow_mut();
            if let Err(e) = interp.call_value_with_this(cb, self_obj.clone(), vec![event.clone()]) {
                cv_js::diag_log(&format!(
                    "[sw] fetch listener threw: {}",
                    describe_throw(&e)
                ));
            }
        }
        if responded.borrow().is_some() {
            break;
        }
    }

    // Settle the respondWith value (Response or Promise<Response>) + extract.
    let result = (|| {
        let r = responded.borrow().clone()?;
        let mut interp = reg.interp.borrow_mut();
        match settle_value(&mut interp, &reg.sched, r, 256) {
            Some(Ok(resp)) => extract_response_bytes(&mut interp, &reg.sched, &resp),
            Some(Err(_)) => None, // rejected → network (V1 resilience)
            None => None,         // pending past budget → network (no hang)
        }
    })();

    reg.intercepting.set(false);
    SW_INTERCEPT_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));

    result
}

/// True if `url` is in `scope` (scope is a URL-prefix; "/" matches everything
/// same-origin; an absolute scope must be a prefix of the request URL).
fn scope_matches(scope: &str, url: &str) -> bool {
    if scope == "/" || scope.is_empty() {
        return true;
    }
    if url.starts_with(scope) {
        return true;
    }
    // A path-only scope ("/app/") matches if the URL's path starts with it.
    if scope.starts_with('/') {
        if let Ok(u) = cv_url::Url::parse(url) {
            return u.path.starts_with(scope);
        }
    }
    false
}

// ════════════════════════════════════════════════════════════════════════════
// THE ORACLE — provably real Service Workers. In-process, bounded, no live
// network (SW source = data: URL; "network" = a thread-local test fixture). Each
// test proves served-from-SW vs served-from-network via a sentinel + counter.
// ════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod sw_oracle {
    use super::*;

    /// A page interp with navigator.serviceWorker installed (so register +
    /// controller wiring have somewhere to land). Returns the interp.
    fn page_interp_with_navigator() -> cv_js::Interp {
        let interp = cv_js::Interp::new();
        interp.install_promise();
        // Minimal navigator with a serviceWorker container.
        let mut nav: HashMap<String, Value> = HashMap::new();
        let mut sw_container: HashMap<String, Value> = HashMap::new();
        sw_container.insert("controller".into(), Value::Null);
        nav.insert(
            "serviceWorker".into(),
            Value::Object(Rc::new(RefCell::new(sw_container))),
        );
        interp.define_global("navigator", Value::Object(Rc::new(RefCell::new(nav))));
        // window.location.href for current_window_href_pub (update path).
        interp
    }

    fn data_url(src: &str) -> String {
        // Base64 so the inline JS source (quotes/newlines/braces) survives URL
        // parsing intact (the parser fragments on `#`, splits on `,`, etc.).
        format!(
            "data:text/javascript;base64,{}",
            crate::base64_encode_pub(src.as_bytes())
        )
    }

    fn setup() {
        clear_sw_registry();
        crate::test_net_fixture_clear();
    }

    // (a) ★ FETCH INTERCEPTION — served from SW cache, NOT network.
    #[test]
    fn sw_intercepts_and_serves_cached_body() {
        setup();
        let page = page_interp_with_navigator();
        let sw_src = r#"
            self.addEventListener('install', function(e){
              e.waitUntil(caches.open('v1').then(function(c){
                return c.put('https://x.test/app.js', new Response('FROM_SW_CACHE'));
              }));
            });
            self.addEventListener('fetch', function(e){
              e.respondWith(caches.match(e.request).then(function(r){ return r || fetch(e.request); }));
            });
        "#;
        // Provide a Response ctor in the SW realm? No — we use a plain object.
        // Replace `new Response(x)` with a plain {_body:x} via a shim install.
        let sw_src = sw_src.replace("new Response('FROM_SW_CACHE')", "{_body:'FROM_SW_CACHE'}");

        crate::test_net_fixture_set("https://x.test/app.js", b"FROM_NETWORK");

        let reg = register_service_worker(
            &page,
            &data_url(&sw_src),
            "https://x.test/",
            "https://x.test/",
        );
        assert!(reg.is_ok(), "registration succeeded: {reg:?}");

        let url = cv_url::Url::parse("https://x.test/app.js").unwrap();
        let bytes = sw_try_intercept(&url).expect("SW intercepted");
        assert_eq!(
            String::from_utf8_lossy(&bytes),
            "FROM_SW_CACHE",
            "served the SW-precached body, not the network body"
        );
        assert_eq!(
            crate::test_net_fixture_count("https://x.test/app.js"),
            0,
            "network was NOT touched (served from SW cache)"
        );
        setup();
    }

    // (a2) THE END-TO-END PROOF: a controlling SW serves a page request through
    // the REAL fetch_body_for_url chokepoint (not just sw_try_intercept). Only
    // exercises the flag-on path (default ON); under CV_SERVICE_WORKERS=0 the
    // guard short-circuits and this would hit the fixture — so we assert the
    // served bytes only when the flag is on.
    #[test]
    fn sw_intercepts_through_fetch_body_for_url() {
        if !service_workers_enabled() {
            return; // flag-off: the chokepoint guard is compiled out of the path
        }
        setup();
        let page = page_interp_with_navigator();
        let sw_src = r#"
            self.addEventListener('install', function(e){
              e.waitUntil(caches.open('v1').then(function(c){
                return c.put('https://e2e.test/data', {_body:'SW_BYTES'});
              }));
            });
            self.addEventListener('fetch', function(e){
              e.respondWith(caches.match(e.request).then(function(r){ return r || fetch(e.request); }));
            });
        "#;
        crate::test_net_fixture_set("https://e2e.test/data", b"NET_BYTES");
        let reg = register_service_worker(
            &page,
            &data_url(sw_src),
            "https://e2e.test/",
            "https://e2e.test/",
        );
        assert!(reg.is_ok(), "registration ok: {reg:?}");

        let url = cv_url::Url::parse("https://e2e.test/data").unwrap();
        let bytes = fetch_body_for_url(&url, 5_000).expect("fetch_body_for_url returned");
        assert_eq!(
            String::from_utf8_lossy(&bytes),
            "SW_BYTES",
            "fetch_body_for_url served the SW-cached body via interception"
        );
        assert_eq!(
            crate::test_net_fixture_count("https://e2e.test/data"),
            0,
            "the network fixture was NOT touched (served from SW)"
        );
        setup();
    }

    // (b) install + activate FIRE (observable flags inside the SW realm).
    #[test]
    fn sw_install_and_activate_fire() {
        setup();
        let page = page_interp_with_navigator();
        let sw_src = r#"
            self.addEventListener('install', function(e){ globalThis.__installed = true; });
            self.addEventListener('activate', function(e){ globalThis.__activated = true; });
        "#;
        let reg = register_service_worker(
            &page,
            &data_url(sw_src),
            "https://y.test/",
            "https://y.test/",
        );
        assert!(reg.is_ok(), "registration ok: {reg:?}");

        // Read the flags out of the SW realm.
        let r = SW_REGISTRY.with(|reg| reg.borrow().last().cloned()).unwrap();
        let installed = r
            .interp
            .borrow()
            .get_global("__installed")
            .map(|v| v.to_bool())
            .unwrap_or(false);
        let activated = r
            .interp
            .borrow()
            .get_global("__activated")
            .map(|v| v.to_bool())
            .unwrap_or(false);
        assert!(installed, "install event fired");
        assert!(activated, "activate event fired");
        assert_eq!(r.state.get(), SwState::Activated);
        setup();
    }

    // (c) NO respondWith → network fall-through (graceful).
    #[test]
    fn sw_no_respondwith_falls_through() {
        setup();
        let page = page_interp_with_navigator();
        // fetch handler that NEVER calls respondWith (passthrough).
        let sw_src = "self.addEventListener('fetch', function(e){ var x = 1; });";
        let reg = register_service_worker(
            &page,
            &data_url(sw_src),
            "https://z.test/",
            "https://z.test/",
        );
        assert!(reg.is_ok(), "registration ok: {reg:?}");

        let url = cv_url::Url::parse("https://z.test/page").unwrap();
        let intercepted = sw_try_intercept(&url);
        assert!(
            intercepted.is_none(),
            "no respondWith → None → caller falls through to network"
        );
        setup();
    }

    // (d) NO SW registered → sw_try_intercept returns None (common path).
    #[test]
    fn sw_no_registration_returns_none() {
        setup();
        let url = cv_url::Url::parse("https://nope.test/page").unwrap();
        assert!(
            sw_try_intercept(&url).is_none(),
            "empty registry → None (byte-identical common path)"
        );
        setup();
    }

    // (d2) NO SW registered → fetch_body_for_url goes STRAIGHT to (fixture)
    // network: exact bytes + counter==1, byte-identical to today's behavior.
    // This drives the REAL chokepoint, proving the SW guard adds zero behavior
    // change on the common path.
    #[test]
    fn sw_no_registration_fetch_unchanged() {
        setup();
        crate::test_net_fixture_set("https://plain.test/page", b"FROM_NETWORK");
        let url = cv_url::Url::parse("https://plain.test/page").unwrap();
        let bytes = fetch_body_for_url(&url, 5_000).expect("network served");
        assert_eq!(String::from_utf8_lossy(&bytes), "FROM_NETWORK");
        assert_eq!(
            crate::test_net_fixture_count("https://plain.test/page"),
            1,
            "the network was hit exactly once (no SW interception)"
        );
        setup();
    }

    // (e) postMessage page↔SW round-trip (synchronous, structuredClone-encoded).
    #[test]
    fn sw_postmessage_roundtrip() {
        setup();
        let mut page = page_interp_with_navigator();
        let sw_src = r#"
            self.addEventListener('message', function(e){
              e.source.postMessage('pong:' + e.data);
            });
        "#;
        let reg = register_service_worker(
            &page,
            &data_url(sw_src),
            "https://m.test/",
            "https://m.test/",
        );
        assert!(reg.is_ok(), "registration ok: {reg:?}");

        // Page registers a serviceWorker message handler that records the reply.
        let recorder = native_fn_with_interp("recorder", |interp, args| {
            let data = if let Some(Value::Object(o)) = args.first() {
                o.borrow().get("data").cloned().unwrap_or(Value::Undefined)
            } else {
                Value::Undefined
            };
            interp.define_global("__reply", data);
            Ok(Value::Undefined)
        });
        add_page_message_handler(recorder);

        // Page calls registration.active.postMessage('ping').
        let active = if let Value::Object(o) = reg.as_ref().unwrap() {
            o.borrow().get("active").cloned().unwrap()
        } else {
            panic!("no registration");
        };
        let post = if let Value::Object(o) = &active {
            o.borrow().get("postMessage").cloned().unwrap()
        } else {
            panic!("no active SW");
        };
        page.call_value_with_this(post, active.clone(), vec![Value::str("ping".to_string())])
            .unwrap();

        // Drain the SW→page inbox on the page interp.
        flush_sw_page_inbox(&mut page);

        let reply = page.get_global("__reply").map(|v| v.to_display_string());
        assert_eq!(
            reply.as_deref(),
            Some("pong:ping"),
            "page→SW→page round-trip delivered the reply"
        );
        setup();
    }

    // (f) Handler throw → not-responded → network fall-through (graceful).
    #[test]
    fn sw_fetch_handler_throw_falls_through() {
        setup();
        let page = page_interp_with_navigator();
        let sw_src = "self.addEventListener('fetch', function(e){ throw new Error('boom'); });";
        let reg = register_service_worker(
            &page,
            &data_url(sw_src),
            "https://t.test/",
            "https://t.test/",
        );
        assert!(reg.is_ok());
        let url = cv_url::Url::parse("https://t.test/x").unwrap();
        assert!(
            sw_try_intercept(&url).is_none(),
            "a throwing fetch handler degrades to network"
        );
        setup();
    }

    // (g) respondWith a plain Response object (not a promise) → served directly.
    #[test]
    fn sw_respondwith_direct_response() {
        setup();
        let page = page_interp_with_navigator();
        let sw_src = "self.addEventListener('fetch', function(e){ e.respondWith({_body:'DIRECT'}); });";
        let reg = register_service_worker(
            &page,
            &data_url(sw_src),
            "https://d.test/",
            "https://d.test/",
        );
        assert!(reg.is_ok());
        let url = cv_url::Url::parse("https://d.test/x").unwrap();
        let bytes = sw_try_intercept(&url).expect("served");
        assert_eq!(String::from_utf8_lossy(&bytes), "DIRECT");
        setup();
    }

    // (h) flag default ON (honors override).
    #[test]
    fn sw_flag_default_on() {
        let expected = !matches!(
            std::env::var("CV_SERVICE_WORKERS").ok().as_deref(),
            Some("0") | Some("false") | Some("off")
        );
        assert_eq!(service_workers_enabled(), expected);
    }

    // (i) SW realm has NO document (distinct realm, no DOM).
    #[test]
    fn sw_realm_has_no_dom() {
        setup();
        let page = page_interp_with_navigator();
        let sw_src = r#"
            self.addEventListener('message', function(e){
              try { var d = document; e.source.postMessage('HAS_DOM'); }
              catch(err){ e.source.postMessage('NO_DOM'); }
            });
        "#;
        let reg = register_service_worker(
            &page,
            &data_url(sw_src),
            "https://nodom.test/",
            "https://nodom.test/",
        );
        assert!(reg.is_ok());
        let r = SW_REGISTRY.with(|reg| reg.borrow().last().cloned()).unwrap();
        // document should be undefined/absent in the SW realm.
        let has_doc = r.interp.borrow().get_global("document").is_some();
        assert!(!has_doc, "SW realm must not expose document");
        setup();
    }
}
