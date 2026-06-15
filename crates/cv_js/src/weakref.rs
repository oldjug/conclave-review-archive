//! `WeakRef` and `FinalizationRegistry` — weak references backed by the REAL
//! tracing GC (ECMA-262 §26.1, §26.2; processing model §9.9).
//!
//! These are NOT stubs. The semantics are tied to actual GC reachability:
//!
//!   * A [`WeakRef`] holds its target through a `std::rc::Weak` (so it does NOT
//!     keep the target alive) plus a `collected` flag. `WeakRef.prototype.deref`
//!     (§26.1.3.2) returns the target while it is still reachable and `undefined`
//!     once the GC has determined it unreachable.
//!
//!   * A [`FinalizationRegistry`] (§26.2) stores `(weak target, held value,
//!     weak unregister token)` registrations. After a collection determines a
//!     target unreachable, its held value is enqueued; the cleanup job
//!     (§9.9.3 CleanupFinalizationRegistry → §26.2.1.1 step) invokes the
//!     registry's `cleanupCallback` with each held value. `unregister`
//!     (§26.2.3.4) removes matching registrations and returns whether any were
//!     removed.
//!
//! ## How "collected" is decided
//!
//! The cv_js GC (`interp::gc_collect`) is a tracing mark-sweep over the
//! `Rc<RefCell<…>>` JS object graph. Its mark phase records the pointer of every
//! object/array REACHABLE FROM THE ROOTS in `GcMark { objs, arrs }`. A WeakRef /
//! FinalizationRegistry target is therefore "live" iff its pointer is in that
//! marked-live set after a collection, and "collected" otherwise — exactly V8's
//! definition (unreachable from roots, NOT merely refcount==0; the cv_js sweep
//! *clears* rather than frees, so `Weak::upgrade` alone is insufficient).
//!
//! [`process_after_mark`] is invoked by the GC right after the mark phase
//! (before the sweep, while held values / tokens are still readable) to flip
//! `collected` flags and enqueue finalizers.
//!
//! ## Rooting discipline
//!
//! * WeakRef targets are held ONLY weakly → never rooted (the whole point).
//! * FinalizationRegistry **held values** must outlive the target, so
//!   [`pending_held_value_roots`] feeds every not-yet-finalized held value (and
//!   every enqueued-but-not-yet-run held value) into the GC root set. The GC
//!   seeds these so a held value that references live objects keeps them alive
//!   until the cleanup callback runs.
//!
//! Always-on: additive, reachable only via the `WeakRef` /
//! `FinalizationRegistry` globals.

use crate::interp::{
    current_native_this, is_symbol_key, make_temporal_error, native_ctor_pure, native_fn, Interp,
    JsError, Value,
};
use crate::ordered::OrderedMap as HashMap;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::{Rc, Weak};

/// A weak handle to a value that "can be held weakly" (ECMA-262 §7.4
/// CanBeHeldWeakly): an Object, an Array, or a Symbol. Objects/Arrays are held
/// through `std::rc::Weak` and tracked by GC reachability. Symbols are not part
/// of the traced Rc graph (they are string keys here); per spec a *registered*
/// symbol is never collected, so holding it strongly is observationally correct
/// for that case (documented in the module footer).
enum WeakTarget {
    Obj { weak: Weak<RefCell<HashMap<String, Value>>>, ptr: usize },
    Arr { weak: Weak<RefCell<Vec<Value>>>, ptr: usize },
    /// A symbol target — strong (registered symbols never collect).
    Symbol(Value),
}

impl WeakTarget {
    /// Build a weak target from a JS value, or `None` if the value cannot be
    /// held weakly (§7.4: not Object/Array/Symbol).
    fn from_value(v: &Value) -> Option<WeakTarget> {
        match v {
            Value::Object(o) => Some(WeakTarget::Obj {
                weak: Rc::downgrade(o),
                ptr: Rc::as_ptr(o) as *const () as usize,
            }),
            Value::Array(a) => Some(WeakTarget::Arr {
                weak: Rc::downgrade(a),
                ptr: Rc::as_ptr(a) as *const () as usize,
            }),
            Value::String(s) if is_symbol_key(s) => Some(WeakTarget::Symbol(v.clone())),
            _ => None,
        }
    }

    /// The pointer identity used to compare against the GC's marked-live set.
    /// `None` for symbols (never compared — symbols never collect here).
    fn ptr(&self) -> Option<usize> {
        match self {
            WeakTarget::Obj { ptr, .. } | WeakTarget::Arr { ptr, .. } => Some(*ptr),
            WeakTarget::Symbol(_) => None,
        }
    }

    /// Reconstitute the strong `Value`, or `None` if the underlying Rc is gone.
    fn upgrade(&self) -> Option<Value> {
        match self {
            WeakTarget::Obj { weak, .. } => weak.upgrade().map(Value::Object),
            WeakTarget::Arr { weak, .. } => weak.upgrade().map(Value::Array),
            WeakTarget::Symbol(v) => Some(v.clone()),
        }
    }
}

/// One `WeakRef`. `collected` is flipped by the GC once the target is found
/// unreachable; once set, `deref` returns `undefined` forever (a target is
/// never resurrected — §9.9 KeptAlive does not un-collect).
struct WeakRefCell {
    target: WeakTarget,
    collected: bool,
}

/// One `(target, held, unregisterToken)` registration in a
/// `FinalizationRegistry` (§26.2). `finalized` guards against double-enqueue.
struct Registration {
    target: WeakTarget,
    held: Value,
    /// Pointer identity of the unregister token, if any (§26.2.3.4 matches by
    /// SameValue, i.e. pointer identity for objects).
    unregister_token: Option<usize>,
    /// Whether the symbol unregister token (no pointer) should match — we store
    /// the symbol value to SameValue-compare.
    unregister_symbol: Option<Value>,
    finalized: bool,
}

/// A `FinalizationRegistry` instance (§26.2): the cleanup callback plus its live
/// registrations.
struct FinReg {
    callback: Value,
    registrations: Vec<Registration>,
}

thread_local! {
    /// All live `WeakRef` cells, indexed by the `_weakRefId` stored on the JS
    /// object. (Slots are never reused; a dropped WeakRef JS object leaves a
    /// dormant cell — bounded by total WeakRefs created, like the GC registries.)
    static WEAK_REFS: RefCell<Vec<WeakRefCell>> = const { RefCell::new(Vec::new()) };
    /// All live `FinalizationRegistry` instances, indexed by `_finRegId`.
    static FIN_REGS: RefCell<Vec<FinReg>> = const { RefCell::new(Vec::new()) };
    /// Held values whose target has been collected and whose cleanup callback
    /// has NOT yet run. `(finreg_id, held)`. Drained by `run_cleanup`.
    static PENDING_CLEANUP: RefCell<Vec<(usize, Value)>> = const { RefCell::new(Vec::new()) };
}

/// Allocate a WeakRef cell, returning its id.
fn alloc_weakref(target: WeakTarget) -> usize {
    WEAK_REFS.with(|r| {
        let mut r = r.borrow_mut();
        r.push(WeakRefCell {
            target,
            collected: false,
        });
        r.len() - 1
    })
}

/// Allocate a FinalizationRegistry, returning its id.
fn alloc_finreg(callback: Value) -> usize {
    FIN_REGS.with(|r| {
        let mut r = r.borrow_mut();
        r.push(FinReg {
            callback,
            registrations: Vec::new(),
        });
        r.len() - 1
    })
}

/// ECMA-262 §7.4 CanBeHeldWeakly: an Object/Array, or a Symbol that is not in
/// the GlobalSymbolRegistry. We treat all symbol keys as weakly-holdable (the
/// registered-vs-not distinction only affects collectability, handled above).
fn can_be_held_weakly(v: &Value) -> bool {
    matches!(v, Value::Object(_) | Value::Array(_))
        || matches!(v, Value::String(s) if is_symbol_key(s))
}

fn type_error(msg: &str) -> JsError {
    make_temporal_error("TypeError", msg.to_string())
}

// ============================================================================
// GC integration — called by interp::gc_collect after the mark phase.
// ============================================================================

/// After the GC mark phase, flip `collected` on any WeakRef whose target is not
/// in the marked-live set, and enqueue finalizers for any FinalizationRegistry
/// registration whose target is not in the marked-live set.
///
/// `live_objs` / `live_arrs` are the pointer sets of every object/array
/// reachable from the GC roots (`GcMark.objs` / `GcMark.arrs`). A target is
/// "collected" iff it is an Object/Array whose pointer is absent from these.
///
/// Runs BEFORE the sweep clears containers, so held values + tokens are intact.
pub fn process_after_mark(live_objs: &HashSet<usize>, live_arrs: &HashSet<usize>) {
    // Is this target pointer still reachable from roots?
    let is_live = |t: &WeakTarget| -> bool {
        match t {
            WeakTarget::Obj { ptr, .. } => live_objs.contains(ptr),
            WeakTarget::Arr { ptr, .. } => live_arrs.contains(ptr),
            // Symbols are not in the traced graph and never collect here.
            WeakTarget::Symbol(_) => true,
        }
    };

    WEAK_REFS.with(|r| {
        for cell in r.borrow_mut().iter_mut() {
            if !cell.collected && !is_live(&cell.target) {
                cell.collected = true;
            }
        }
    });

    let mut newly_enqueued: Vec<(usize, Value)> = Vec::new();
    FIN_REGS.with(|r| {
        for (id, fr) in r.borrow_mut().iter_mut().enumerate() {
            for reg in fr.registrations.iter_mut() {
                if !reg.finalized && !is_live(&reg.target) {
                    reg.finalized = true;
                    newly_enqueued.push((id, reg.held.clone()));
                }
            }
        }
    });
    if !newly_enqueued.is_empty() {
        PENDING_CLEANUP.with(|p| p.borrow_mut().extend(newly_enqueued));
    }
}

/// Every held value the GC must keep alive: held values of registrations whose
/// target is not yet collected, AND held values already enqueued for cleanup but
/// not yet delivered (§9.9 — held values are roots until their cleanup runs).
///
/// Targets are deliberately NOT included (they are weak).
pub fn pending_held_value_roots() -> Vec<Value> {
    let mut roots = Vec::new();
    FIN_REGS.with(|r| {
        for fr in r.borrow().iter() {
            // The cleanup callback itself stays reachable while the registry has
            // outstanding work (it is also held by user code, but root it to be
            // safe when only the registry object reaches it).
            if fr.registrations.iter().any(|reg| !reg.finalized) {
                roots.push(fr.callback.clone());
            }
            for reg in fr.registrations.iter() {
                if !reg.finalized {
                    roots.push(reg.held.clone());
                }
            }
        }
    });
    PENDING_CLEANUP.with(|p| {
        for (_, held) in p.borrow().iter() {
            roots.push(held.clone());
        }
    });
    roots
}

/// Whether any finalization cleanup work is pending (host can decide to pump).
pub fn has_pending_cleanup() -> bool {
    PENDING_CLEANUP.with(|p| !p.borrow().is_empty())
}

/// Number of WeakRefs whose target is still live (for tests / diagnostics).
pub fn live_weakref_count() -> usize {
    WEAK_REFS.with(|r| r.borrow().iter().filter(|c| !c.collected).count())
}

// ============================================================================
// JS surface.
// ============================================================================

/// `WeakRef` constructor (§26.1.1) + `WeakRef.prototype.deref` (§26.1.3.2).
fn build_weakref_ctor() -> Value {
    native_ctor_pure("WeakRef", 1, |args| {
        let target = args.first().cloned().unwrap_or(Value::Undefined);
        // §26.1.1.1 step 3: if CanBeHeldWeakly(target) is false, throw TypeError.
        let wt = WeakTarget::from_value(&target)
            .ok_or_else(|| type_error("WeakRef: target must be an object or symbol"))?;
        let id = alloc_weakref(wt);
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("_weakRefId".into(), Value::Number(id as f64));
        // §26.1.3.2 WeakRef.prototype.deref.
        m.insert(
            "deref".into(),
            native_fn("deref", move |_args| {
                let id = weakref_id_of_this()?;
                Ok(WEAK_REFS.with(|r| {
                    let mut r = r.borrow_mut();
                    let cell = &mut r[id];
                    if cell.collected {
                        return Value::Undefined;
                    }
                    // Defensive: if the underlying Rc is fully gone (the acyclic
                    // free path, independent of the tracing sweep) treat it as
                    // collected too.
                    match cell.target.upgrade() {
                        Some(v) => v,
                        None => {
                            cell.collected = true;
                            Value::Undefined
                        }
                    }
                }))
            }),
        );
        Ok(Value::Object(Rc::new(RefCell::new(m))))
    })
}

/// Read the `_weakRefId` off the `this` object of a `deref` call.
fn weakref_id_of_this() -> Result<usize, JsError> {
    match current_native_this() {
        Value::Object(o) => {
            let b = o.borrow();
            match b.get("_weakRefId") {
                Some(Value::Number(n)) => Ok(*n as usize),
                _ => Err(type_error("WeakRef.prototype.deref called on incompatible receiver")),
            }
        }
        _ => Err(type_error("WeakRef.prototype.deref called on non-object")),
    }
}

/// Read the `_finRegId` off the `this` object of a registry method call.
fn finreg_id_of_this() -> Result<usize, JsError> {
    match current_native_this() {
        Value::Object(o) => {
            let b = o.borrow();
            match b.get("_finRegId") {
                Some(Value::Number(n)) => Ok(*n as usize),
                _ => Err(type_error(
                    "FinalizationRegistry method called on incompatible receiver",
                )),
            }
        }
        _ => Err(type_error(
            "FinalizationRegistry method called on non-object",
        )),
    }
}

/// `FinalizationRegistry` constructor (§26.2.1) + prototype `register` /
/// `unregister` (§26.2.3).
fn build_finreg_ctor() -> Value {
    native_ctor_pure("FinalizationRegistry", 1, |args| {
        let cb = args.first().cloned().unwrap_or(Value::Undefined);
        // §26.2.1.1 step 2: if IsCallable(cleanupCallback) is false, throw.
        if !is_callable(&cb) {
            return Err(type_error(
                "FinalizationRegistry: cleanup callback must be a function",
            ));
        }
        let id = alloc_finreg(cb);
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("_finRegId".into(), Value::Number(id as f64));
        m.insert("register".into(), native_fn("register", finreg_register));
        m.insert(
            "unregister".into(),
            native_fn("unregister", finreg_unregister),
        );
        Ok(Value::Object(Rc::new(RefCell::new(m))))
    })
}

fn is_callable(v: &Value) -> bool {
    match v {
        Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_) => true,
        Value::Object(o) => {
            let b = o.borrow();
            b.contains_key("_call") || b.contains_key("_construct")
        }
        _ => false,
    }
}

/// §26.2.3.2 FinalizationRegistry.prototype.register(target, heldValue [, unregisterToken]).
fn finreg_register(args: Vec<Value>) -> Result<Value, JsError> {
    let id = finreg_id_of_this()?;
    let target = args.first().cloned().unwrap_or(Value::Undefined);
    let held = args.get(1).cloned().unwrap_or(Value::Undefined);
    let token = args.get(2).cloned().unwrap_or(Value::Undefined);

    // step 4: if CanBeHeldWeakly(target) is false, throw TypeError.
    let wt = WeakTarget::from_value(&target)
        .ok_or_else(|| type_error("register: target must be an object or symbol"))?;
    // step 5: if SameValue(target, heldValue) is true, throw TypeError.
    if same_value(&target, &held) {
        return Err(type_error("register: held value must not be the target"));
    }
    // step 6: if CanBeHeldWeakly(unregisterToken) is false and it isn't
    // undefined, throw TypeError.
    let (unregister_token, unregister_symbol) = if matches!(token, Value::Undefined) {
        (None, None)
    } else if can_be_held_weakly(&token) {
        match &token {
            Value::Object(o) => (Some(Rc::as_ptr(o) as *const () as usize), None),
            Value::Array(a) => (Some(Rc::as_ptr(a) as *const () as usize), None),
            // Symbol token — match by SameValue on the symbol key string.
            _ => (None, Some(token.clone())),
        }
    } else {
        return Err(type_error(
            "register: unregister token must be an object or symbol",
        ));
    };

    FIN_REGS.with(|r| {
        let mut r = r.borrow_mut();
        r[id].registrations.push(Registration {
            target: wt,
            held,
            unregister_token,
            unregister_symbol,
            finalized: false,
        });
    });
    Ok(Value::Undefined)
}

/// §26.2.3.4 FinalizationRegistry.prototype.unregister(unregisterToken).
/// Returns `true` if any registration was removed, else `false`.
fn finreg_unregister(args: Vec<Value>) -> Result<Value, JsError> {
    let id = finreg_id_of_this()?;
    let token = args.first().cloned().unwrap_or(Value::Undefined);
    // step 3: if CanBeHeldWeakly(unregisterToken) is false, throw TypeError.
    if !can_be_held_weakly(&token) {
        return Err(type_error(
            "unregister: unregister token must be an object or symbol",
        ));
    }
    let token_ptr = match &token {
        Value::Object(o) => Some(Rc::as_ptr(o) as *const () as usize),
        Value::Array(a) => Some(Rc::as_ptr(a) as *const () as usize),
        _ => None,
    };
    let removed = FIN_REGS.with(|r| {
        let mut r = r.borrow_mut();
        let before = r[id].registrations.len();
        r[id].registrations.retain(|reg| {
            let matches_ptr = token_ptr.is_some() && reg.unregister_token == token_ptr;
            let matches_sym = reg
                .unregister_symbol
                .as_ref()
                .is_some_and(|s| same_value(s, &token));
            !(matches_ptr || matches_sym)
        });
        before != r[id].registrations.len()
    });
    Ok(Value::Bool(removed))
}

/// SameValue (§7.2.11) restricted to the value kinds we need to compare:
/// pointer identity for Object/Array, key equality for symbols, structural for
/// primitives.
fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => Rc::ptr_eq(x, y),
        (Value::Array(x), Value::Array(y)) => Rc::ptr_eq(x, y),
        (Value::String(x), Value::String(y)) => x.as_str() == y.as_str(),
        (Value::Number(x), Value::Number(y)) => x.to_bits() == y.to_bits(),
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Null, Value::Null) | (Value::Undefined, Value::Undefined) => true,
        _ => false,
    }
}

/// Install `WeakRef` + `FinalizationRegistry` globals (called from
/// `Interp::install_basic_globals`).
pub fn install(interp: &Interp) {
    interp.define_global("WeakRef", build_weakref_ctor());
    interp.define_global("FinalizationRegistry", build_finreg_ctor());
}

// ============================================================================
// Cleanup driving — invoked from the host message loop / microtask drain.
// ============================================================================

impl Interp {
    /// Run pending FinalizationRegistry cleanup callbacks (§9.9.3
    /// CleanupFinalizationRegistry). Each enqueued `(registry, heldValue)` pair
    /// invokes the registry's cleanup callback with `heldValue`. Returns the
    /// number of callbacks run. Idempotent / safe to call repeatedly.
    pub fn run_finalization_cleanup(&mut self) -> usize {
        let mut count = 0;
        loop {
            let next = PENDING_CLEANUP.with(|p| {
                let mut p = p.borrow_mut();
                if p.is_empty() {
                    None
                } else {
                    Some(p.remove(0))
                }
            });
            let Some((reg_id, held)) = next else { break };
            let cb = FIN_REGS.with(|r| r.borrow().get(reg_id).map(|fr| fr.callback.clone()));
            if let Some(cb) = cb {
                // A throwing cleanup callback must not abort the others (the
                // host job swallows it — §9.9.4.1 HostEnqueueFinalizationRegistry-
                // CleanupJob). We ignore the result/error.
                let _ = self.call_value(cb, vec![held]);
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::Interp;

    /// A fresh, fully-installed interpreter (WeakRef/FinalizationRegistry come
    /// in via `install_basic_globals`).
    fn interp() -> Interp {
        let mut i = Interp::new();
        i.install_basic_globals();
        i
    }

    /// The string value of the LAST `console.log` line.
    fn last_log(i: &Interp) -> String {
        i.output.last().cloned().unwrap_or_default()
    }

    /// Force a full GC with an explicit root set, then run finalizers, so the
    /// test is fully deterministic (no reliance on the periodic pump).
    fn collect_then_finalize(i: &mut Interp, roots: &[Value]) {
        i.gc_collect(roots);
        i.run_finalization_cleanup();
    }

    #[test]
    fn weakref_keeps_target_visible_while_strongly_reachable() {
        let mut i = interp();
        // Target held by a global var → reachable across GC.
        i.run("globalThis.__t = {tag:'live'}; globalThis.__w = new WeakRef(globalThis.__t);")
            .unwrap();
        i.gc_collect(&[]);
        i.run("console.log(globalThis.__w.deref() ? globalThis.__w.deref().tag : 'GONE');")
            .unwrap();
        assert_eq!(last_log(&i), "live", "live target must remain deref-able");
    }

    #[test]
    fn weakref_derefs_undefined_after_target_collected() {
        let mut i = interp();
        // Hold the target strongly, take a WeakRef, confirm deref works, then
        // drop the only strong ref. After a GC the target is unreachable from
        // roots → deref() must be undefined. (Proves the WeakRef itself does
        // NOT keep the target alive.)
        i.run(
            "globalThis.__t = {tag:'doomed'}; \
             globalThis.__w = new WeakRef(globalThis.__t); \
             console.log(globalThis.__w.deref() ? globalThis.__w.deref().tag : 'GONE');",
        )
        .unwrap();
        assert_eq!(last_log(&i), "doomed", "deref returns target while strongly held");
        // Drop the strong reference. The WeakRef now holds the only (weak) link.
        i.run("delete globalThis.__t;").unwrap();
        i.gc_collect(&[]);
        i.run("console.log(globalThis.__w.deref() === undefined ? 'UNDEF' : 'STILL');")
            .unwrap();
        assert_eq!(last_log(&i), "UNDEF", "deref must be undefined after GC");
    }

    #[test]
    fn finalization_callback_fires_with_held_value() {
        let mut i = interp();
        i.run(
            "globalThis.__log = []; \
             globalThis.__r = new FinalizationRegistry(h => { globalThis.__log.push(h); }); \
             globalThis.__r.register({}, 'held-token');",
        )
        .unwrap();
        // Target ({}) is unreachable immediately; registry + log are global.
        collect_then_finalize(&mut i, &[]);
        i.run("console.log(globalThis.__log.length + ':' + (globalThis.__log[0]||''));")
            .unwrap();
        assert_eq!(last_log(&i), "1:held-token", "callback fires once with held value");
    }

    #[test]
    fn finalization_does_not_fire_while_target_alive() {
        let mut i = interp();
        i.run(
            "globalThis.__log = []; \
             globalThis.__keep = {}; \
             globalThis.__r = new FinalizationRegistry(h => { globalThis.__log.push(h); }); \
             globalThis.__r.register(globalThis.__keep, 'should-not-fire');",
        )
        .unwrap();
        collect_then_finalize(&mut i, &[]);
        i.run("console.log(globalThis.__log.length);").unwrap();
        assert_eq!(last_log(&i), "0", "callback must NOT fire while target reachable");
    }

    #[test]
    fn unregister_prevents_callback() {
        let mut i = interp();
        i.run(
            "globalThis.__log = []; \
             globalThis.__tok = {}; \
             globalThis.__r = new FinalizationRegistry(h => { globalThis.__log.push(h); }); \
             globalThis.__r.register({}, 'held', globalThis.__tok); \
             console.log(globalThis.__r.unregister(globalThis.__tok));",
        )
        .unwrap();
        assert_eq!(last_log(&i), "true", "unregister returns true when it removes");
        collect_then_finalize(&mut i, &[]);
        i.run("console.log(globalThis.__log.length);").unwrap();
        assert_eq!(last_log(&i), "0", "unregistered target must not fire");
    }

    #[test]
    fn unregister_unknown_token_returns_false() {
        let mut i = interp();
        i.run(
            "globalThis.__r = new FinalizationRegistry(()=>{}); \
             console.log(globalThis.__r.unregister({}));",
        )
        .unwrap();
        assert_eq!(last_log(&i), "false");
    }

    #[test]
    fn register_target_equals_held_throws() {
        let mut i = interp();
        i.run(
            "globalThis.__r = new FinalizationRegistry(()=>{}); \
             let o = {}; \
             try { globalThis.__r.register(o, o); console.log('NO-THROW'); } \
             catch (e) { console.log(e instanceof TypeError ? 'TypeError' : 'other'); }",
        )
        .unwrap();
        assert_eq!(last_log(&i), "TypeError", "register(o,o) must throw TypeError");
    }

    #[test]
    fn weakref_of_primitive_throws() {
        let mut i = interp();
        i.run(
            "try { new WeakRef(42); console.log('NO-THROW'); } \
             catch(e){ console.log(e instanceof TypeError ? 'TypeError' : 'other'); }",
        )
        .unwrap();
        assert_eq!(last_log(&i), "TypeError", "WeakRef(42) must throw TypeError");
    }

    #[test]
    fn finreg_non_callable_throws() {
        let mut i = interp();
        i.run(
            "try { new FinalizationRegistry(123); console.log('NO-THROW'); } \
             catch(e){ console.log(e instanceof TypeError ? 'TypeError' : 'other'); }",
        )
        .unwrap();
        assert_eq!(last_log(&i), "TypeError", "FinalizationRegistry(123) must throw");
    }

    #[test]
    fn held_value_object_survives_until_cleanup() {
        // The held value is itself an object referencing live data; it must NOT
        // be collected before its cleanup callback runs (§9.9 roots discipline).
        let mut i = interp();
        i.run(
            "globalThis.__seen = 'NONE'; \
             globalThis.__r = new FinalizationRegistry(h => { globalThis.__seen = h.payload; }); \
             globalThis.__r.register({}, { payload: 'survived' });",
        )
        .unwrap();
        collect_then_finalize(&mut i, &[]);
        i.run("console.log(globalThis.__seen);").unwrap();
        assert_eq!(last_log(&i), "survived", "held object survives until cleanup");
    }

    #[test]
    fn second_gc_after_revive_does_not_double_fire() {
        // Once a target collects + the finalizer fires, a subsequent GC must not
        // re-enqueue it (the registration is marked finalized).
        let mut i = interp();
        i.run(
            "globalThis.__n = 0; \
             globalThis.__r = new FinalizationRegistry(() => { globalThis.__n++; }); \
             globalThis.__r.register({}, 'x');",
        )
        .unwrap();
        collect_then_finalize(&mut i, &[]);
        collect_then_finalize(&mut i, &[]);
        i.run("console.log(globalThis.__n);").unwrap();
        assert_eq!(last_log(&i), "1", "finalizer fires exactly once");
    }
}
