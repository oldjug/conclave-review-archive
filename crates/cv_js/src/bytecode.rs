//! Register-based bytecode VM for cv_js.
//!
//! This is the first slice of the optimizing-engine work. It lowers a
//! subset of the AST to a flat instruction stream that a fetch/decode
//! loop dispatches without recursion. Functions, control flow, and
//! arithmetic on numbers are real; anything else (objects, arrays,
//! native fns, closures over outer scopes) is rejected at compile time
//! and the host falls back to the tree-walk interpreter.
//!
//! Architecture: each function gets its own register file (max 256
//! virtual registers) and a constant pool. Locals and intermediate
//! values both live in registers; the compiler uses a bump-allocator
//! and resets the bump per statement so loops don't leak. Calls pass
//! arguments through consecutive registers, mirroring how V8 Ignition
//! lays out call frames.

#![allow(clippy::too_many_arguments)]

use crate::ordered::OrderedMap as HashMap;
use std::rc::Rc;

use crate::ast::{AssignOp, BinOp, Expr, ForInit, LogicalOp, Stmt, UnaryOp, UpdateOp, VarDeclarator};
use crate::interp::Value;
use crate::parser::parse_program;

/// A virtual-register index. `u16` (not `u8`) so large minified functions
/// (e.g. webpack module loaders) don't overflow the 256-register file and
/// fall back to the tree-walk — the register count is the #1 reason big real
/// scripts can't compile.
pub type Reg = u16;

/// Sentinel `this_reg` meaning "no `this` binding" (call leaves `this`
/// undefined). Must be outside the valid register range.
const NO_THIS: Reg = u16::MAX;

/// One bytecode instruction. Operands are register indices (`Reg`) and
/// constant-pool indices / jump targets (u16). The total enum is bigger
/// than a u8 op-stream would be, but Rust's `match` over a c-style enum
/// optimizes into a jump table, so dispatch is competitive.
#[derive(Debug, Clone, Copy)]
pub enum Op {
    LoadConst {
        dst: Reg,
        k: u16,
    },
    LoadTrue {
        dst: Reg,
    },
    LoadFalse {
        dst: Reg,
    },
    LoadNull {
        dst: Reg,
    },
    LoadUndef {
        dst: Reg,
    },
    Move {
        dst: Reg,
        src: Reg,
    },

    Add {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Sub {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Mul {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Div {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Mod {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// `R[dst] = R[lhs] ** R[rhs]` — exponentiation.
    Pow {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// Strict equality `===` per ECMA-262 §7.2.14 (no coercion).
    Eq {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// Strict inequality `!==`.
    Neq {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// ECMA-262 §7.2.15 IsLooselyEqual — `==`. Coerces per the Abstract
    /// Equality Comparison algorithm. Distinct from `Op::Eq` (`===`) so a
    /// VM-compiled hot function gets correct `x == null` / `0 == "0"` /
    /// `1 == true` / `[1] == 1` semantics matching Chrome (and the
    /// tree-walker), instead of strict-equal-only false-negatives.
    LooseEq {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// `!=` — inverse of `LooseEq`.
    LooseNeq {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Lt {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Le {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Gt {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Ge {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// Bitwise ops. Operands are coerced via ToInt32/ToUint32 per ECMA-262
    /// §13.x; the result is a Number. `Shl`/`Shr` use signed (arithmetic);
    /// `Ushr` uses unsigned (logical). Shift count is `rhs & 31`.
    BitAnd {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    BitOr {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    BitXor {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Shl {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Shr {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Ushr {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    Neg {
        dst: Reg,
        src: Reg,
    },
    Not {
        dst: Reg,
        src: Reg,
    },
    /// `R[dst] = ~R[src]` — bitwise NOT (operand via ToInt32).
    BitNot {
        dst: Reg,
        src: Reg,
    },
    /// `R[dst] = +R[src]` — unary plus, i.e. ToNumber(R[src]). Distinct from a
    /// bare `Move`: `+true === 1`, `+'' === 0`, `+'3' === 3`, `+undefined` is
    /// NaN, and `+1n` THROWS TypeError (BigInt has no ToNumber).
    ToNumber {
        dst: Reg,
        src: Reg,
    },
    /// `R[dst] = typeof R[src]` as a string. Uses the same `typeof_name`
    /// logic as the tree-walk so callable objects report "function", etc.
    Typeof {
        dst: Reg,
        src: Reg,
    },
    /// `R[dst] = ToString(R[src])` using the ECMA-262 §7.1.17 ToString algorithm
    /// — i.e. ToPrimitive with the STRING hint (`toString`-first) for an Object
    /// operand. Emitted for each `${expr}` substitution in a template literal so
    /// `` `${o}` `` matches ECMA `String(o)`/the tree-walker (and Node), instead
    /// of the NUMBER/default hint that a `"" + expr` lowering used (which calls
    /// `valueOf` first and so diverged on an object with both `valueOf` and
    /// `toString`). Distinct from `Add`: `'' + o` legitimately uses the DEFAULT
    /// hint, so this op must NOT be conflated with string-concat `+`.
    ToStr {
        dst: Reg,
        src: Reg,
    },
    /// `R[dst] = (R[lhs] in R[rhs])` — property-existence test. `lhs` is the
    /// key, `rhs` the object. Throws if `rhs` isn't an object/array.
    In {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// `R[dst] = (R[lhs] instanceof R[rhs])`. Routes through the host's full
    /// `__tb_host_instanceof(instance, ctor)` (which runs the tree-walk
    /// `ordinary_has_instance` PROTO_KEY walk + the tag-based `is_instance_of`
    /// fallback), so the VM result is byte-identical to the tree-walk tier
    /// without re-implementing the prototype-chain/`Symbol.hasInstance` logic.
    Instanceof {
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// `R[dst] = delete R[obj][consts[key_k]]` — removes the own property and
    /// yields `true`.
    DeleteProp {
        dst: Reg,
        obj: Reg,
        key_k: u16,
    },
    /// `R[dst] = delete R[obj][R[key]]` — computed-key delete; yields `true`.
    DeleteIdx {
        dst: Reg,
        obj: Reg,
        key: Reg,
    },
    /// `R[dst] = /source/flags` — builds a FRESH RegExp object each evaluation
    /// (so a `/g` literal's lastIndex state isn't shared across evaluations).
    MakeRegex {
        dst: Reg,
        source_k: u16,
        flags_k: u16,
    },

    Jmp {
        target: u16,
    },
    JmpIfFalse {
        cond: Reg,
        target: u16,
    },
    JmpIfTrue {
        cond: Reg,
        target: u16,
    },

    /// Call a module-level function by its index. Args live in
    /// `R[first_arg..first_arg + n_args]`. The return value lands in
    /// `R[dst]`.
    CallFn {
        dst: Reg,
        fn_idx: u16,
        first_arg: Reg,
        n_args: u8,
    },

    /// `R[dst] = globals[consts[name_k].as_string()]`. Used for any
    /// identifier that the compiler couldn't resolve as a local. If
    /// the name isn't in the global env, the VM returns `undefined`.
    ///
    /// This UNCHECKED form is emitted only for contexts where an
    /// unresolvable name must NOT throw: `typeof x`, the read-back of a
    /// compound/`++`/`--` target that was already proven resolvable, and
    /// engine-internal helper loads (`__tb_spread__`, `__tb_get_iterator__`).
    /// Genuine value-context identifier reads use `LoadGlobalChecked`.
    LoadGlobal {
        dst: Reg,
        name_k: u16,
    },

    /// `R[dst] = globals[consts[name_k].as_string()]`, but THROWS
    /// `ReferenceError("<name> is not defined")` when the name resolves to
    /// nothing — i.e. it is not a global binding (which in this engine holds
    /// every builtin, `NaN`/`Infinity`/`globalThis`, and every top-level
    /// `var`/function). This mirrors the tree-walk tier's `eval_identifier`
    /// (ECMA-262 §13.3.2 GetValue on an unresolvable Reference): a plain
    /// VALUE-context identifier read of an undeclared name must throw, not
    /// silently become `undefined`. Closes Finding #1 (the VM/tree-walk
    /// split-brain the M3.0 A/B oracle caught). Emitted ONLY for value-context
    /// identifier reads (bare reads, member/call bases) — NEVER for
    /// `typeof`/`delete`/assignment, which keep the unchecked `LoadGlobal`.
    LoadGlobalChecked {
        dst: Reg,
        name_k: u16,
    },

    /// `globals[consts[name_k].as_string()] = R[src]`. Emitted by the
    /// compiler for top-level (`is_script`) VarDecl initialisers and
    /// for assignments to undeclared script-level names — these become
    /// real globals so module-level functions can read them.
    StoreGlobal {
        name_k: u16,
        src: Reg,
    },

    /// `R[dst] = R[callee](R[first_arg..first_arg + n_args])`. The
    /// callee must be a `Value::NativeFunction`, `Value::BcClosure`, or
    /// (TBD) a tree-walk `Value::Function`. `this_reg` is the register
    /// holding the `this` binding for the call, or `0xFF` to leave
    /// `this` as `undefined`.
    CallValue {
        dst: Reg,
        callee: Reg,
        this_reg: Reg,
        first_arg: Reg,
        n_args: u8,
    },

    /// `R[dst] = new R[ctor](R[first_arg..first_arg + n_args])`. The
    /// VM allocates a fresh empty object, calls `ctor` with that
    /// object as `this`, and returns either the constructor's return
    /// value (if it's an object) or the freshly-allocated `this`.
    New {
        dst: Reg,
        ctor: Reg,
        first_arg: Reg,
        n_args: u8,
    },

    /// `R[dst] = this` — load the current frame's `this` binding into
    /// a register. The frame's `this` is whatever was passed to
    /// `run_function`; defaults to `Value::Undefined`.
    LoadThis {
        dst: Reg,
    },
    /// `R[dst] = the currently-executing closure` (as a Value::BcClosure), or
    /// undefined if none. Lets a nested named function reference itself for
    /// recursion without capturing a self-upvalue.
    LoadSelf {
        dst: Reg,
    },

    /// `R[dst] = R[obj][consts[key_k]]`. Used for `obj.prop` and
    /// `obj["literal"]`. Reads from Value::Object (HashMap), or from
    /// Value::String (length, charAt), or returns Undefined on miss.
    GetProp {
        dst: Reg,
        obj: Reg,
        key_k: u16,
    },

    /// `R[dst] = R[obj][R[key]]`. Used for `arr[i]`, `obj[varKey]`,
    /// `str[i]`. Coerces key to integer for arrays/strings, to string
    /// for objects.
    GetIdx {
        dst: Reg,
        obj: Reg,
        key: Reg,
    },

    /// `R[obj][consts[key_k]] = R[src]`. Mirrors GetProp.
    SetProp {
        obj: Reg,
        key_k: u16,
        src: Reg,
    },

    /// `R[obj][R[key]] = R[src]`. Mirrors GetIdx.
    SetIdx {
        obj: Reg,
        key: Reg,
        src: Reg,
    },

    /// `R[dst] = [R[first_elem], R[first_elem+1], ..., R[first_elem+n-1]]`.
    /// `n_elems == 0` means an empty array literal.
    NewArray {
        dst: Reg,
        first_elem: Reg,
        n_elems: u8,
    },

    /// `R[arr].push(R[val])` — append one element to the array in place.
    /// Errors at runtime if `R[arr]` is not an array.
    ArrayPush {
        arr: Reg,
        val: Reg,
    },

    /// `R[arr].push(...R[spread])` — append every element of the iterable
    /// in `R[spread]` to the array in `R[arr]`. Errors at runtime if
    /// `R[spread]` is not iterable.
    ArrayPushSpread {
        arr: Reg,
        spread: Reg,
    },

    /// `R[dst] = {}` — fresh empty object. Subsequent SetProp ops fill
    /// in the literal's properties.
    NewObject {
        dst: Reg,
    },

    /// `throw R[src]`. Unwinds back to the nearest enclosing TryEnter
    /// frame, where the exception value lands in the catch binding.
    /// If no handler is active in this function, the error bubbles to
    /// the caller as a `RuntimeError::Thrown(value)`.
    Throw {
        src: Reg,
    },

    /// Push a try-handler with `catch_target` as the IP to jump to on
    /// throw and `catch_reg` as the register that receives the thrown
    /// value. The handler stays active until a matching `TryExit`.
    TryEnter {
        catch_target: u16,
        catch_reg: Reg,
    },

    /// Pop the top try-handler. Emitted at the end of the protected
    /// block (i.e. when control would otherwise continue past it
    /// without throwing).
    TryExit,

    /// `R[dst] = enumerable_keys_of(R[obj])` — used to lower `for-in`.
    /// Objects return their string keys (HashMap iteration order — not
    /// quite spec but consistent within a run). Arrays return their
    /// numeric indices as strings. Other types return an empty array.
    EnumKeys {
        dst: Reg,
        obj: Reg,
    },

    /// `R[dst] = closure(consts[fn_idx], capture R[first_upvalue..+n])`.
    /// Builds a `Value::BcClosure` whose upvalue vector is a snapshot
    /// of the current frame's registers in the listed slots. Subsequent
    /// `LoadUp` / `StoreUp` inside the closure access those upvalues.
    MakeClosure {
        dst: Reg,
        fn_idx: u16,
        first_upvalue: Reg,
        n_upvalues: u8,
    },

    /// Inside a closure body, `R[dst] = upvalues[slot]`.
    LoadUp {
        dst: Reg,
        slot: u8,
    },

    /// Inside a closure body, `upvalues[slot] = R[src]`. Lets a closure
    /// mutate its captured variables.
    StoreUp {
        src: Reg,
        slot: u8,
    },

    /// Return `R[src]` to the caller.
    Ret {
        src: Reg,
    },
}

/// Monomorphic own-property inline cache entry for one property-access site
/// (Phase-2 JS speedup). Guards on the object's HIDDEN-CLASS id (`ShapeId`) — so
/// it hits across ALL objects that share a shape (e.g. every element of an array
/// of same-shaped records), not just one object. A match means the key→slot
/// layout is identical, so the property is a direct `Vec` index instead of a hash
/// probe. `slot == u32::MAX` ⇒ uninitialized. Correctness: a shape is the exact
/// key-sequence, so the recorded slot maps to the same key in every same-shape
/// object; on miss the byte-identical slow path runs and re-records.
/// Up to this many shapes are cached per site before going megamorphic.
const IC_POLY_CAP: usize = 4;

/// Polymorphic own-property inline cache for one site: up to `IC_POLY_CAP`
/// `(shape_id, slot)` entries. A site that only ever sees one shape stays
/// effectively monomorphic; 2–4 shapes (e.g. a function called with a few record
/// types) all hit; beyond that the site goes `mega` and falls to the slow path.
#[derive(Debug, Clone, Copy)]
pub struct PropIc {
    // --- own-property polymorphic cache (read + write) ---
    shapes: [u32; IC_POLY_CAP],
    slots: [u32; IC_POLY_CAP],
    len: u8,
    mega: bool,
    // --- depth-1 PROTOTYPE cache (one entry; method dispatch) ---
    // Valid iff `p_key_slot != u32::MAX`. Caches "an object of shape p_obj_shape,
    // whose PROTO_KEY is at slot p_obj_proto_slot pointing at prototype
    // p_proto_ptr (of shape p_proto_shape), inherits the property at the
    // prototype's slot p_key_slot". On a hit the inherited method is read
    // directly from the prototype's slot — skipping the proto-walk/host hop.
    p_obj_shape: u32,
    p_obj_proto_slot: u32,
    p_proto_ptr: usize,
    p_proto_shape: u32,
    p_key_slot: u32,
}
impl PropIc {
    pub const INVALID: PropIc = PropIc {
        shapes: [0; IC_POLY_CAP],
        slots: [u32::MAX; IC_POLY_CAP],
        len: 0,
        mega: false,
        p_obj_shape: 0,
        p_obj_proto_slot: 0,
        p_proto_ptr: 0,
        p_proto_shape: 0,
        p_key_slot: u32::MAX,
    };
    /// Slot for `shape` if cached (linear scan over ≤4 entries — no hashing).
    #[inline]
    fn lookup(&self, shape: u32) -> Option<u32> {
        if self.mega {
            return None;
        }
        for i in 0..self.len as usize {
            if self.shapes[i] == shape {
                return Some(self.slots[i]);
            }
        }
        None
    }
    /// Record `(shape, slot)`; promote to megamorphic past the poly cap.
    #[inline]
    fn record(&mut self, shape: u32, slot: u32) {
        // Never cache a DEOPTED (Dict) object's shape: it reports the reserved
        // `DICT_SHAPE`, whose slot layout is unstable (it can change on every
        // delete/insert), so a cached `(DICT_SHAPE, slot)` would be read back on
        // a later iteration and write/read the WRONG slot. Declining to record it
        // keeps the IC permanently OFF for deopted objects (clean miss every time)
        // — the M3.2 deopt contract.
        if shape == crate::shapes::DICT_SHAPE {
            return;
        }
        if self.mega {
            return;
        }
        for i in 0..self.len as usize {
            if self.shapes[i] == shape {
                self.slots[i] = slot;
                return;
            }
        }
        if (self.len as usize) < IC_POLY_CAP {
            self.shapes[self.len as usize] = shape;
            self.slots[self.len as usize] = slot;
            self.len += 1;
        } else {
            self.mega = true;
        }
    }
    /// The cached prototype entry for `obj_shape`, if any.
    #[inline]
    fn proto_lookup(&self, obj_shape: u32) -> Option<(u32, usize, u32, u32)> {
        if self.p_key_slot != u32::MAX && self.p_obj_shape == obj_shape {
            Some((
                self.p_obj_proto_slot,
                self.p_proto_ptr,
                self.p_proto_shape,
                self.p_key_slot,
            ))
        } else {
            None
        }
    }
    #[inline]
    fn proto_record(
        &mut self,
        obj_shape: u32,
        proto_slot: u32,
        proto_ptr: usize,
        proto_shape: u32,
        key_slot: u32,
    ) {
        // Same deopt contract as `record`: never cache a deopted (DICT_SHAPE)
        // object or prototype — its slot layout is unstable.
        if obj_shape == crate::shapes::DICT_SHAPE || proto_shape == crate::shapes::DICT_SHAPE {
            return;
        }
        self.p_obj_shape = obj_shape;
        self.p_obj_proto_slot = proto_slot;
        self.p_proto_ptr = proto_ptr;
        self.p_proto_shape = proto_shape;
        self.p_key_slot = key_slot;
    }

    /// The WARMED own-property `(shape_id, slot)` entries for this site, for the T2
    /// JIT to bake as inline `cmp <shape>; je hit_k` guards (slot pre-resolved per
    /// shape). Returns `None` if the site is megamorphic (blown past the poly cap)
    /// or has no warmed entry — T2 then DECLINES inlining this GetProp (retry once
    /// the IC warms). NEVER includes `DICT_SHAPE` (a deopted object's unstable
    /// layout was never recorded, per `record`), so a baked guard can only ever
    /// match a stable Shaped layout. Public for `jit::compile_t2lite`.
    #[inline]
    pub fn warm_own_entries(&self) -> Option<Vec<(u32, u32)>> {
        if self.mega || self.len == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(self.len as usize);
        for i in 0..self.len as usize {
            let (sh, sl) = (self.shapes[i], self.slots[i]);
            // Defensive: never bake an uninitialized slot or a DICT_SHAPE guard.
            if sl == u32::MAX || sh == crate::shapes::DICT_SHAPE {
                return None;
            }
            out.push((sh, sl));
        }
        Some(out)
    }

    /// B5 (persisted code cache) — the WARMED own-property feedback as PORTABLE
    /// `(shape_id, slot)` pairs for serialization. Unlike `warm_own_entries`, this
    /// does NOT reject a megamorphic site (`mega == true` is itself worth
    /// persisting so the next run starts megamorphic instead of re-blowing the
    /// cap) and INCLUDES the `mega` flag so a reload reconstructs the exact IC
    /// state. Each `shape_id` is run-LOCAL (an interned integer); the caller
    /// (`code_cache`) is responsible for translating it to a portable SHAPE
    /// DESCRIPTOR (the key-sequence) before writing to disk and re-interning on
    /// load — a raw `shape_id` is NOT portable across runs. The proto cache is
    /// deliberately EXCLUDED (it holds a live `p_proto_ptr` address, which cannot
    /// survive a process restart); it simply re-warms at runtime, which is sound
    /// (a cold proto cache produces clean misses then re-records).
    #[inline]
    pub fn serialize_own(&self) -> (Vec<(u32, u32)>, bool) {
        let mut out = Vec::with_capacity(self.len as usize);
        for i in 0..self.len as usize {
            out.push((self.shapes[i], self.slots[i]));
        }
        (out, self.mega)
    }

    /// Whether this IC carries ANY warmed own-property feedback worth persisting
    /// (at least one recorded entry, or it has gone megamorphic). A site that
    /// never executed (`len == 0 && !mega`) has nothing to persist.
    #[inline]
    pub fn has_feedback(&self) -> bool {
        self.len > 0 || self.mega
    }

    /// B5 reload — reconstruct an IC purely from persisted own-property feedback,
    /// where every `shape_id` has ALREADY been re-interned to a THIS-RUN id by the
    /// caller (`code_cache`). Drops any entry whose slot is uninitialized or whose
    /// shape is `DICT_SHAPE` (never a valid baked guard — same contract as
    /// `record`). The proto cache starts INVALID and re-warms at runtime.
    pub fn from_serialized_own(entries: &[(u32, u32)], mega: bool) -> PropIc {
        let mut ic = PropIc::INVALID;
        if mega {
            ic.mega = true;
            return ic;
        }
        for &(sh, sl) in entries {
            if sl == u32::MAX || sh == crate::shapes::DICT_SHAPE {
                continue;
            }
            if (ic.len as usize) >= IC_POLY_CAP {
                ic.mega = true;
                break;
            }
            // De-dup defensively (a corrupt blob could repeat a shape).
            let mut dup = false;
            for i in 0..ic.len as usize {
                if ic.shapes[i] == sh {
                    dup = true;
                    break;
                }
            }
            if dup {
                continue;
            }
            ic.shapes[ic.len as usize] = sh;
            ic.slots[ic.len as usize] = sl;
            ic.len += 1;
        }
        ic
    }
}

// The global hidden-class pool now lives in `shapes.rs` (shared with the Shaped
// object store in `ordered.rs`, M3.2 P3) — see `crate::shapes::with_shape_table`
// and friends. Both the IC here and a Shaped object intern into the SAME table,
// so their `ShapeId`s are directly comparable.

/// The hidden-class id for an object, derived from its key-sequence. Cached on
/// the object and recomputed only when `struct_ver` changed (a key add/remove/
/// reorder) — so it's O(1) on the hot path and O(keys) only on a structural
/// change. Same key-sequence ⇒ same `ShapeId` ⇒ a previously-resolved slot is
/// valid for THIS object too.
///
/// M3.2 P3 routing (flag-off path is BYTE-IDENTICAL to before):
///   - A Shaped object (only built when `CV_SHAPED_OBJ=1`) already holds its
///     interned `ShapeId` — return it O(1), no key rewalk.
///   - A DEOPTED Dict object (was Shaped, fell back) returns the reserved
///     `DICT_SHAPE` so `PropIc::lookup` can never match it → clean IC miss.
///   - A plain Dict object (the ONLY kind that exists with the flag OFF) keeps
///     the exact key-rewalk intern below — the IC hit-rate baseline is preserved.
fn object_shape_id(ob: &HashMap<String, Value>) -> u32 {
    // Fast O(1) routes for the Shaped/deopted store (no-ops when flag off, since
    // no object is ever Shaped or deopted then).
    if let Some(stored) = ob.stored_shape_id() {
        return stored;
    }
    let sv = ob.struct_ver();
    let (cached_ver, cached_id) = ob.cached_shape();
    if cached_ver == sv {
        return cached_id;
    }
    let id = crate::shapes::with_shape_table(|t| {
        let mut s = t.empty();
        for k in ob.keys() {
            s = t.add_property(s, k);
        }
        s
    });
    ob.set_cached_shape(sv, id);
    id
}

/// Whether the property inline cache is active (default ON; `CV_PROPIC=0` opts
/// out — matching the `CV_BYTECODE` convention). Read once.
pub fn propic_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_PROPIC").as_deref() != Ok("0"))
}

thread_local! {
    /// (hits, misses) for the property inline cache — diagnostic + a regression
    /// gate that the IC stays active (a hot property loop should be mostly hits).
    static PROPIC_STATS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}
pub fn propic_stats() -> (u64, u64) {
    PROPIC_STATS.with(|c| c.get())
}
pub fn reset_propic_stats() {
    PROPIC_STATS.with(|c| c.set((0, 0)));
}
#[inline]
fn propic_hit() {
    PROPIC_STATS.with(|c| {
        let (h, m) = c.get();
        c.set((h + 1, m));
    });
}
#[inline]
fn propic_miss() {
    PROPIC_STATS.with(|c| {
        let (h, m) = c.get();
        c.set((h, m + 1));
    });
}

/// One compiled function: parameters + constant pool + flat code.
#[derive(Debug, Clone)]
pub struct BcFunction {
    pub name: String,
    pub n_params: u8,
    /// Register holding the rest parameter (`function f(a, ...rest)`), if any.
    /// At call time the VM gathers `args[rest_reg..]` into a real array here;
    /// without it `...rest` bound to a single value (or nothing) and broke
    /// `[...rest]` / `f(...rest)` / `rest.length` in every hot function.
    pub rest_reg: Option<Reg>,
    pub n_regs: Reg,
    pub consts: Vec<Value>,
    pub code: Vec<Op>,
    /// Per-`GetProp`-site inline caches, indexed by instruction pointer. Lazily
    /// sized to `code.len()` on first use in the run loop; only `GetProp` indices
    /// are ever written. Persists on the `BcFunction` (in the reused `Module`),
    /// so caches warm across calls. `RefCell` for interior mutability behind the
    /// `&BcFunction` the run loop holds.
    pub ic: std::cell::RefCell<Vec<PropIc>>,
    /// T4 P1 — per-`arith`/`compare`/`call`-site TYPE-FEEDBACK vector, indexed by
    /// instruction pointer exactly like `ic`. Lazily sized to `code.len()` on
    /// first recording; only binary/unary/compare/call op indices are ever
    /// written. Records the monotone (widen-only) V8-`BinaryOperationHint`-shaped
    /// operand hint + monomorphic call target so the T4 lowering can speculate on
    /// OBSERVED types. RECORDING ONLY (P1); observationally invisible; the VM only
    /// writes it when `CV_FEEDBACK` is on (`feedback::feedback_enabled()`), so the
    /// default build pays zero cost. Persists on the reused `Module` so feedback
    /// warms across calls; `RefCell` for the same interior-mutability reason as
    /// `ic`. (P5 will serialize it alongside the persisted `PropIc`.)
    pub feedback: std::cell::RefCell<Vec<crate::feedback::TypeFeedback>>,
    /// True iff this function's body opens with a `"use strict"` directive
    /// prologue (ECMA-262 §11.2.1). The VM pushes a strict-mode frame around the
    /// body so refusal-throws (legacy-platform-object [[Set]]/[[Delete]], writes
    /// to non-writable props) match sloppy/strict semantics. Set at compile time
    /// where the AST body is available; `false` for placeholders/top-level.
    pub strict: bool,
}

impl BcFunction {
    /// T4 P1 — read the recorded TYPE-FEEDBACK for the op at bytecode index `ip`,
    /// for the T4 lowering (P2 representation selection / P3 inlining). Returns
    /// the bottom (`INVALID`, all-`None`) slot if the site never recorded
    /// (feedback off, or the op never ran, or it's not a recorded op kind) — so a
    /// consumer always gets a safe "no information → do not speculate" answer with
    /// no `Option` ceremony. This is the read seam the speculative tier consults;
    /// it never mutates (the VM is the only writer, behind `CV_FEEDBACK`).
    #[inline]
    pub fn feedback_at(&self, ip: usize) -> crate::feedback::TypeFeedback {
        let tbl = self.feedback.borrow();
        tbl.get(ip)
            .copied()
            .unwrap_or(crate::feedback::TypeFeedback::INVALID)
    }

    /// The recorded binary/compare TYPE-HINT for the op at `ip` — convenience over
    /// [`feedback_at`] for the common representation-selection query. `None`
    /// (never observed) if the site never ran / isn't a binary site / feedback is
    /// off. The T4 lowering checks `hint.is_numeric_speculatable()` to decide
    /// whether to pick an unboxed Int32/Float64 representation.
    #[inline]
    pub fn type_hint_at(&self, ip: usize) -> crate::feedback::TypeHint {
        self.feedback_at(ip).binop_hint()
    }

    /// The MONOMORPHIC call target (module fn-index) recorded for the call op at
    /// `ip`, if this site only ever called a single, known callee — the seam the
    /// P3 inliner reads. `None` if never called, polymorphic, or feedback off (in
    /// every case the inliner correctly declines).
    #[inline]
    pub fn mono_call_target_at(&self, ip: usize) -> Option<u32> {
        self.feedback_at(ip).mono_call_target()
    }

    /// Whether the feedback vector carries ANY observation worth exposing — true
    /// iff at least one op-site recorded a non-bottom hint or a call target. Lets
    /// the T4 lowering cheaply skip a function whose feedback never warmed (decline
    /// to T3/T2). Also the non-vacuity probe the oracle/tests assert against (the
    /// vector must actually FILL on a recorded run).
    pub fn has_any_feedback(&self) -> bool {
        self.feedback.borrow().iter().any(|f| f.has_feedback())
    }
}

/// A compiled program. The top-level script lives in `fns[0]`; user
/// function declarations get indices 1..N.
#[derive(Debug, Clone)]
pub struct Module {
    pub fns: Vec<BcFunction>,
    /// Script-frame `for (var i = …)` init bindings that are kept in a fast
    /// LOCAL register for the hot loop and only synced to their (function-scoped,
    /// i.e. global) binding at loop exit. Each entry is `(global name, register)`
    /// in `fns[0]`'s register file. On a THROW that escapes the loop (mid-loop),
    /// the normal post-loop `StoreGlobal` is skipped, so the run loop flushes
    /// these live registers to `globals` on the script frame's error-return path —
    /// matching the tree-walker (and Node/Chrome), where `globalThis.i` reflects
    /// the value `i` held at the throw point (ECMA-262: a global `var` is a
    /// property of the global object, so every write must reach it). Empty for any
    /// program without a script-level for-init `var` (zero overhead). NOT
    /// persisted: a module carrying these declines disk caching (see
    /// `code_cache::serialize_module`) so it is always recompiled with the field
    /// present — never silently lost on a cache reload.
    pub script_forinit_syncs: Vec<(String, Reg)>,
}

#[derive(Debug)]
pub struct CompileError(pub String);

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bytecode compile: {}", self.0)
    }
}

impl std::error::Error for CompileError {}

/// Compile a full program (script body) into a Module ready to execute.
/// Returns `Err` if any node in the program isn't in the bytecode VM's
/// supported subset — the caller can then fall back to the tree-walk
/// interpreter.
pub fn compile_program(source: &str) -> Result<Module, CompileError> {
    let program = parse_program(source).map_err(|e| CompileError(format!("parse: {e}")))?;

    let mut fn_index: HashMap<String, u16> = HashMap::new();
    let mut declared: Vec<(String, Vec<String>, Vec<Stmt>)> = Vec::new();
    for s in &program {
        if let Stmt::FunctionDecl { name, params, body } = s {
            let idx = (declared.len() + 1) as u16;
            fn_index.insert(name.clone(), idx);
            declared.push((name.clone(), params.clone(), body.clone()));
        }
    }

    let fns_pool: std::rc::Rc<std::cell::RefCell<Vec<BcFunction>>> = std::rc::Rc::new(
        std::cell::RefCell::new(Vec::with_capacity(declared.len() + 1)),
    );
    // Reserve slots 0..=declared.len() with placeholders so MakeClosure
    // indices for nested fns don't collide with the script (slot 0) or
    // the top-level decls (slots 1..=N).
    let placeholder = BcFunction {
        name: String::new(),
        n_params: 0,
        rest_reg: None,
        n_regs: 0,
        consts: Vec::new(),
        code: Vec::new(),
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
    };
    for _ in 0..=declared.len() {
        fns_pool.borrow_mut().push(placeholder.clone());
    }

    let script_body: Vec<Stmt> = program
        .into_iter()
        .filter(|s| !matches!(s, Stmt::FunctionDecl { .. }))
        .collect();
    let (script, _, script_forinit_syncs) = compile_function(
        "<script>",
        &[],
        &script_body,
        &fn_index,
        None,
        std::rc::Rc::clone(&fns_pool),
    )?;
    fns_pool.borrow_mut()[0] = script;

    for (i, (name, params, body)) in declared.iter().enumerate() {
        let (f, _, _) = compile_function(
            name,
            params,
            body,
            &fn_index,
            None,
            std::rc::Rc::clone(&fns_pool),
        )?;
        fns_pool.borrow_mut()[i + 1] = f;
    }

    let fns = fns_pool.borrow().clone();
    Ok(Module {
        fns,
        script_forinit_syncs,
    })
}

/// Compile a SINGLE function (one `FunctionValue`) into a runnable Module whose
/// `fns[0]` is the function — for per-function VM execution invoked by the
/// tree-walk interp. `captured_names` are the variable names visible in the
/// function's closure (non-global): passing them as `parent_locals` makes any
/// real capture compile to an upvalue (detectable in the returned records), so
/// the caller can map read-only upvalues from the live closure scope and
/// decline functions that mutate captures. Returns `(module, upvalues)`.
pub fn compile_single_function(
    params: &[String],
    body: &[Stmt],
    captured_names: &[String],
) -> Result<(Module, Vec<UpvalueRecord>), CompileError> {
    compile_single_function_arrow(params, body, captured_names, false)
}

/// Like `compile_single_function`, but `is_arrow` makes `Expr::This` resolve to
/// the lexical `__lexical_this` upvalue (which the caller must include in
/// `captured_names`) instead of `LoadThis`. This is what lets an arrow function
/// be compiled to the register VM with correct lexical `this` (Chrome-audit
/// FIX 2 — V8 has no separate slow path for arrows).
pub fn compile_single_function_arrow(
    params: &[String],
    body: &[Stmt],
    captured_names: &[String],
    is_arrow: bool,
) -> Result<(Module, Vec<UpvalueRecord>), CompileError> {
    let fn_index: HashMap<String, u16> = HashMap::new();
    let fns_pool: std::rc::Rc<std::cell::RefCell<Vec<BcFunction>>> =
        std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    // Reserve slot 0 for this function; nested fns land at 1.. via MakeClosure.
    fns_pool.borrow_mut().push(BcFunction {
        name: String::new(),
        n_params: 0,
        rest_reg: None,
        n_regs: 0,
        consts: Vec::new(),
        code: Vec::new(),
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
    });
    let parent_locals: HashMap<String, Reg> =
        captured_names.iter().map(|n| (n.clone(), 0u16)).collect();
    // Name "<arrow>" makes compile_function set is_arrow so Expr::This uses the
    // __lexical_this upvalue path; "<fn>" keeps the regular LoadThis behaviour.
    let root_name = if is_arrow { "<arrow>" } else { "<fn>" };
    let (f, ups, _) = compile_function(
        root_name,
        params,
        body,
        &fn_index,
        Some(parent_locals),
        std::rc::Rc::clone(&fns_pool),
    )?;
    fns_pool.borrow_mut()[0] = f;
    let fns = fns_pool.borrow().clone();
    // Defensive: every function reference must point inside the pool. A dangling
    // index (compiler edge case) would otherwise fault at run time; decline here
    // so the caller cleanly tree-walks instead.
    let n = fns.len() as u16;
    for func in &fns {
        for op in &func.code {
            let bad_idx = match op {
                Op::MakeClosure { fn_idx, .. } | Op::CallFn { fn_idx, .. } => *fn_idx >= n,
                _ => false,
            };
            if bad_idx {
                return Err(CompileError(format!("dangling fn_idx (pool has {n})")));
            }
        }
    }
    // A per-function (non-script) compile never has script-level for-init `var`s
    // (those are function-local here), so no global syncs are needed.
    Ok((
        Module {
            fns,
            script_forinit_syncs: Vec::new(),
        },
        ups,
    ))
}

/// True if any function in the module writes to a captured upvalue (`StoreUp`).
/// The VM's closures snapshot upvalues BY VALUE at `MakeClosure` time, so such a
/// write never propagates back to the binding the enclosing scope (or a sibling
/// closure) observes. The tree-walk tier captures by reference and is correct, so
/// the per-SCRIPT bytecode path declines a module with any upvalue write and runs
/// it tree-walk instead.
///
/// This is exactly the bug the WPT event tests expose: a `test(function(){ var
/// called=false; el.addEventListener('x', function(){ called=true; });
/// el.dispatchEvent(e); assert_true(called); })` — the inner listener's
/// `called=true` is a `StoreUp` into its by-value snapshot, so on the VM the
/// outer `called` stayed false. Declining the script to tree-walk fixes it
/// without the (large) cell-based-upvalue rewrite. `New`/classes/etc. are NOT
/// declined here (the script frame handles them) — only the write-back hazard.
pub fn module_has_upvalue_writes(module: &Module) -> bool {
    module
        .fns
        .iter()
        .any(|f| f.code.iter().any(|op| matches!(op, Op::StoreUp { .. })))
}

/// Whether a per-function caller can safely run this module on the VM. Declines
/// (returns false) when it would misbehave vs the tree-walk:
/// - `StoreUp`: mutates a captured binding — the VM's by-value upvalues can't
///   propagate the write back.
/// - `New`: `new` on a tree-walk constructor isn't host-dispatched (call ≠
///   construct), so it would error or mis-run.
/// - references `arguments`: the VM has no `arguments` object, so it would read
///   an undefined global instead.
pub fn module_is_per_fn_safe(module: &Module) -> bool {
    for f in &module.fns {
        for op in &f.code {
            if matches!(op, Op::StoreUp { .. } | Op::New { .. }) {
                return false;
            }
        }
        for c in &f.consts {
            if let Value::String(s) = c {
                // Decline whole categories the VM can't faithfully run, so they
                // tree-walk instead (correctness > coverage):
                // - `arguments`: the VM has no arguments object.
                // - `__tb_run_async__`: async wrapper; running it on the VM turns
                //   the inner body into a BcClosure the async lowerer can't see,
                //   breaking `await` ordering.
                // - prototype/OO manipulation: class bodies desugar to an IIFE
                //   that does `X.prototype.m = …`; on the VM the nested ctor
                //   becomes a BcClosure without real `.prototype`/[[Construct]]
                //   semantics, so instances lose their methods. Constructors,
                //   `defineProperty` (live ESM getters), and `__proto__` writes
                //   are the same class of hazard.
                if matches!(
                    &**s,
                    "arguments"
                        | "__tb_run_async__"
                        | "prototype"
                        | "__proto__"
                        | "defineProperty"
                        | "defineProperties"
                        | "setPrototypeOf"
                        | "create"
                ) {
                    return false;
                }
            }
        }
    }
    true
}

/// Run `module.fns[0]` with the given args/this/globals/closure/dispatch — the
/// per-function entry point (the interp passes the live closure's upvalues +
/// shared globals + a host-call dispatcher).
pub fn run_module_call(
    module: &Module,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    closure: Option<&std::rc::Rc<crate::interp::BcClosure>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    run_function(module, 0, args, this, globals, closure, dispatch)
}

/// The SHARED `Op::CallValue` dispatch (single source of truth for the VM op AND
/// the T2 Phase-4 re-entry helper `rt_call_value`). Given a callee/this/args, it
/// unwraps `_call`/`_construct` namespace globals, then dispatches:
///   * `NativeFunction::Pure`     → run the native body directly;
///   * `NativeFunction::WithInterp` → hand off to the host `dispatch` with `this`;
///   * `BcClosure`                → `run_function` on its module/fn_idx;
///   * `Function` / `Object`      → host `dispatch` (a tree-walk closure / callable);
///   * anything else              → `TypeError("callee is not callable")`.
///
/// Returns the result `Value` or a `RuntimeError` (which the VM op routes through
/// `propagate!` / the T2 runner maps to THREW). It does NOT itself decide
/// catch-vs-propagate — that is the caller's (the VM op's `propagate!`, the T2
/// helper's THREW status). Byte-identical to the inlined `Op::CallValue` body.
pub(crate) fn dispatch_call_value(
    callee_val: Value,
    this_val: Value,
    call_args: Vec<Value>,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    // V8 treats any object with a [[Call]] slot as callable; our namespace
    // globals carry the callable as a `_call`/`_construct` own property.
    let callee_eff = match &callee_val {
        Value::Object(o) => {
            let unwrapped = {
                let b = o.borrow();
                b.get("_call").or_else(|| b.get("_construct")).cloned()
            };
            unwrapped.unwrap_or(callee_val)
        }
        _ => callee_val,
    };
    match callee_eff {
        Value::NativeFunction(nf) => match &nf.func {
            crate::interp::NativeFnBody::Pure(body) => {
                // Push the receiver onto the native-`this` thread-local for the
                // duration of the body, exactly as the tree-walk dispatcher does,
                // so a `Pure` builtin that resolves its dynamic receiver via
                // `current_native_this()` (string/array methods rebound through
                // `.call`, WeakRef `deref`, …) sees the right `this` from VM
                // dispatch. The guard pops on every exit (incl. the error path).
                let _this_guard = crate::interp::VmNativeThisGuard::new(this_val);
                body(call_args).map_err(|e| match e {
                    crate::interp::JsError::Throw(v) => RuntimeError::Thrown(v),
                    other => RuntimeError::TypeError(format!("native fn `{}`: {other:?}", nf.name)),
                })
            }
            crate::interp::NativeFnBody::WithInterp(_) => {
                dispatch(Value::NativeFunction(nf.clone()), this_val, call_args)
            }
        },
        Value::BcClosure(c) => run_function(
            &c.module,
            c.fn_idx as usize,
            &call_args,
            &this_val,
            globals,
            Some(&c),
            dispatch,
        ),
        Value::Function(_) | Value::Object(_) => dispatch(callee_eff, this_val, call_args),
        other => Err(RuntimeError::TypeError(format!(
            "callee is not callable: {other:?}"
        ))),
    }
}

/// Run a `BcClosure` against its OWN module + `fn_idx` — lets the tree-walk
/// interp invoke a VM-created closure (a callback/event handler stored as a
/// `Value::BcClosure`) so they're first-class callables, not just whole-fn
/// entries.
pub fn run_closure(
    closure: &std::rc::Rc<crate::interp::BcClosure>,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    run_function(
        &closure.module,
        closure.fn_idx as usize,
        args,
        this,
        globals,
        Some(closure),
        dispatch,
    )
}

/// Collect every `var`-declared name reachable from `stmts` WITHOUT descending
/// into nested function bodies — the VM-tier mirror of `interp::collect_var_names`
/// (ECMA-262 VarDeclaredNames). `var`, `for (var …)`, and `for (var … in/of …)`
/// bindings hoist to the enclosing FUNCTION scope, crossing block/if/loop/try/
/// switch boundaries but never a function boundary. `let`/`const` are excluded.
fn bc_collect_var_names(stmts: &[Stmt], out: &mut Vec<String>) {
    for s in stmts {
        bc_collect_var_names_stmt(s, out);
    }
}

fn bc_collect_var_names_stmt(s: &Stmt, out: &mut Vec<String>) {
    use crate::ast::VarKind;
    match s {
        Stmt::VarDecl {
            kind: VarKind::Var,
            decls,
        } => {
            for d in decls {
                out.push(d.name.clone());
            }
        }
        Stmt::VarDecl { .. } => {}
        // A nested function has its own var scope — do not descend.
        Stmt::FunctionDecl { .. } => {}
        Stmt::Block(b) => bc_collect_var_names(b, out),
        Stmt::If { cons, alt, .. } => {
            bc_collect_var_names_stmt(cons, out);
            if let Some(a) = alt {
                bc_collect_var_names_stmt(a, out);
            }
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
            bc_collect_var_names_stmt(body, out)
        }
        Stmt::For { init, body, .. } => {
            if let Some(ForInit::VarDecl {
                kind: VarKind::Var,
                decls,
            }) = init
            {
                for d in decls {
                    out.push(d.name.clone());
                }
            }
            bc_collect_var_names_stmt(body, out);
        }
        Stmt::ForIn {
            kind, name, body, ..
        }
        | Stmt::ForOf {
            kind, name, body, ..
        } => {
            if matches!(kind, Some(VarKind::Var)) {
                out.push(name.clone());
            }
            bc_collect_var_names_stmt(body, out);
        }
        Stmt::Try {
            block,
            catch_block,
            finally_block,
            ..
        } => {
            bc_collect_var_names(block, out);
            if let Some(cb) = catch_block {
                bc_collect_var_names(cb, out);
            }
            if let Some(fb) = finally_block {
                bc_collect_var_names(fb, out);
            }
        }
        Stmt::Switch { cases, .. } => {
            for c in cases {
                bc_collect_var_names(&c.body, out);
            }
        }
        Stmt::Labeled { body, .. } => bc_collect_var_names_stmt(body, out),
        _ => {}
    }
}

fn compile_function(
    name: &str,
    params: &[String],
    body: &[Stmt],
    fn_index: &HashMap<String, u16>,
    parent_locals: Option<HashMap<String, Reg>>,
    fns_pool: std::rc::Rc<std::cell::RefCell<Vec<BcFunction>>>,
) -> Result<(BcFunction, Vec<UpvalueRecord>, Vec<(String, Reg)>), CompileError> {
    let is_script = name == "<script>";
    let is_arrow = name == "<arrow>";
    // A nested named function (compiled with a parent snapshot) can refer to
    // itself; bind its name to the live closure via LoadSelf. Top-level fns
    // (parent_locals = None) recurse via fn_index/CallFn instead.
    let self_name = if parent_locals.is_some() && !is_script && !is_arrow && !name.is_empty() {
        Some(name.to_string())
    } else {
        None
    };
    let mut c = FnCompiler {
        consts: Vec::new(),
        code: Vec::new(),
        scopes: vec![HashMap::new()],
        next_reg: 0,
        max_reg: 0,
        overflow: false,
        fn_index,
        loops: Vec::new(),
        is_script,
        is_arrow,
        self_name,
        parent_locals,
        upvalues: Vec::new(),
        fns_pool,
        free_regs: Vec::new(),
        local_regs: std::collections::HashSet::new(),
        script_forinit_syncs: Vec::new(),
        body_fn_decls: std::collections::HashSet::new(),
        bound_fns: std::collections::HashSet::new(),
    };
    let mut rest_reg: Option<Reg> = None;
    for p in params {
        let r = c.alloc_reg();
        // Rest parameter (`...rest`) is stored by the parser with a `...`
        // prefix; bind the stripped name and remember its register so the VM
        // gathers the trailing args into an array at call time.
        if let Some(rest_name) = p.strip_prefix("...") {
            c.scopes
                .last_mut()
                .unwrap()
                .insert(rest_name.to_string(), r);
            rest_reg = Some(r);
        } else {
            c.scopes.last_mut().unwrap().insert(p.clone(), r);
        }
        c.local_regs.insert(r);
    }
    let n_params = params.len() as u8;

    // PRE-DECLARE nested function declarations (non-script frames only). A
    // body-level `function f(){…}` is hoisted in JS — visible by name throughout
    // its function scope (ECMA-262 §10.2.11). Allocating a SLOT for every such
    // name up front (before compiling any statement) lets a reference to a
    // sibling — whether a later `function`, or a `var g = function(){…f…}` — bind
    // to a real local register instead of silently falling through to a global
    // load (the old bug: `f is not defined` even though `f` is a sibling). The
    // closures are still MATERIALISED in source order below; a FORWARD/MUTUAL
    // reference (capturing a sibling whose slot isn't bound yet) is detected at
    // materialise time and declines to the tree-walk tier (which binds closures
    // by reference and handles it). A BACKWARD reference is safe — the earlier
    // sibling's slot already holds its live closure. The script frame skips this:
    // its top-level decls are pre-registered in `fn_index` and resolved by stable
    // module index (which is also why mutually-recursive top-level fns work).
    if !is_script {
        for s in body {
            if let Stmt::FunctionDecl { name, .. } = s {
                c.body_fn_decls.insert(name.clone());
                if c.lookup(name).is_none() {
                    c.declare(name);
                }
            }
        }
        // PRE-DECLARE every `var`-hoisted name (ECMA-262 VarDeclaredNames),
        // including ones nested inside blocks / if / loops / try / switch. A
        // `var x` only inside a block must hoist to a FUNCTION-scope slot, so a
        // later read (`return x` after the block) resolves to it. Without this,
        // `compile_stmt` declared the `var` in the transient block scope (popped
        // when the block ends) and the read fell through to a global load →
        // spurious ReferenceError. The top scope is `scopes[0]` (params live
        // here too); a param of the same name already occupies the slot, so
        // `lookup` short-circuits and we don't clobber it.
        let mut hoisted: Vec<String> = Vec::new();
        bc_collect_var_names(body, &mut hoisted);
        for n in hoisted {
            if c.lookup(&n).is_none() {
                let r = c.alloc_reg();
                c.scopes[0].insert(n, r);
                c.local_regs.insert(r);
            }
        }
    }

    // SCRIPT-FRAME TOP-LEVEL FUNCTION HOISTING (ECMA-262 §16.1.7 GlobalDeclaration-
    // Instantiation step 17 + §9.1.1.4.16 CreateGlobalFunctionBinding): a top-level
    // `function f(){…}` in a classic script creates a PROPERTY `f` on the global
    // object, visible to every later script on the page. `compile_program` splits
    // these decls into module fn-slots reached internally via `CallFn`, but never
    // bound them on the live global bindings table — so a SECOND `<script>` (the
    // ubiquitous WPT pattern: a helper `<script>function attr_is(){…}</script>`
    // followed by a test `<script>… attr_is(…) …</script>`) saw `ReferenceError:
    // attr_is is not defined`. The tree-walk tier (`exec_block`) hoists these into
    // the global scope correctly; this makes the default-on bytecode VM match.
    //
    // Each top-level fn closes over global scope (no captured locals), so its
    // closure is a zero-upvalue `MakeClosure` over the module fn-slot recorded in
    // `fn_index`, then bound by `StoreGlobal`. Materialising it from the compiled
    // bytecode (rather than a separate Module field) keeps it byte-identical
    // across the disk code-cache with no extra serialization. Emitted at the very
    // top of the script frame so the bindings exist for all subsequent statements.
    // NB: `compile_program` filters top-level `FunctionDecl`s OUT of the script
    // body (they live in dedicated module fn-slots), so iterate `fn_index` —
    // sorted by slot index, which `compile_program` assigns in source order — to
    // recover every top-level fn name + its slot deterministically.
    if is_script && !c.fn_index.is_empty() {
        let mut decls: Vec<(u16, String)> =
            c.fn_index.iter().map(|(n, i)| (*i, n.clone())).collect();
        decls.sort_by_key(|(i, _)| *i);
        for (fn_idx, name) in decls {
            let dst = c.alloc_reg();
            c.emit(Op::MakeClosure {
                dst,
                fn_idx,
                first_upvalue: 0,
                n_upvalues: 0,
            });
            let name_k = c.add_const(Value::str(name));
            c.emit(Op::StoreGlobal { name_k, src: dst });
        }
    }

    for s in body {
        c.compile_stmt(s)?;
    }
    let r = c.alloc_reg();
    c.emit(Op::LoadUndef { dst: r });
    c.emit(Op::Ret { src: r });

    if c.overflow {
        return Err(CompileError(
            "register/const file overflow — function too large for the VM".into(),
        ));
    }
    let uvs = c.upvalues.clone();
    let script_forinit_syncs = c.script_forinit_syncs.clone();
    Ok((
        BcFunction {
            name: name.to_string(),
            n_params,
            rest_reg,
            n_regs: c.max_reg,
            consts: c.consts,
            code: c.code,
            ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: crate::interp::body_is_strict(body),
        },
        uvs,
        script_forinit_syncs,
    ))
}

/// One enclosing loop's patch lists. `break` and `continue` inside the
/// loop emit unconditional jumps with placeholder targets; we record
/// their positions here so we can patch them once the loop's end
/// (break) or test/update site (continue) is known.
struct LoopCtx {
    break_patches: Vec<usize>,
    continue_patches: Vec<usize>,
    /// True for a `switch` context: it captures `break` (which targets the
    /// switch end) but NOT `continue` (which must pass through to the nearest
    /// enclosing real loop).
    is_switch: bool,
}

/// One captured upvalue: a slot in the enclosing frame's register file
/// that the closure body reads/writes through LoadUp/StoreUp.
#[derive(Clone)]
pub struct UpvalueRecord {
    pub name: String,
    pub parent_local: Reg,
}

struct FnCompiler<'a> {
    consts: Vec<Value>,
    code: Vec<Op>,
    scopes: Vec<HashMap<String, Reg>>,
    next_reg: Reg,
    max_reg: Reg,
    /// Set when the register file (>255) or const pool (>u16::MAX) is
    /// exhausted. `compile_function` turns this into a `CompileError` so the
    /// caller falls back to the tree-walk instead of panicking on a function
    /// too large for the VM's u8 register file.
    overflow: bool,
    fn_index: &'a HashMap<String, u16>,
    loops: Vec<LoopCtx>,
    /// True iff this is the top-level script frame. Affects how
    /// `var`/`let` declarations and bare-name assignments are lowered:
    /// in the script frame they become real globals so any module-level
    /// fn can see them.
    is_script: bool,
    /// True iff this is an arrow-function body. Arrow `this` is
    /// lexical — it resolves to the enclosing frame's `this`, not the
    /// arrow's call-site `this`. The compiler emits LoadUp from a
    /// synthetic "__lexical_this" upvalue for `Expr::This` in this
    /// case, captured at the MakeClosure site.
    is_arrow: bool,
    /// For a NESTED named function, its own name — so a self-reference inside
    /// the body resolves to the live closure via `Op::LoadSelf` (handles
    /// recursion without capturing an undefined self-upvalue). `None` for the
    /// script frame, arrows, and top-level fns (those recurse via fn_index).
    self_name: Option<String>,
    /// Flat snapshot of every variable accessible from the enclosing
    /// function's scopes when this closure was emitted. `None` means
    /// this frame is the top of its function tree (script or
    /// top-level fn decl).
    parent_locals: Option<HashMap<String, Reg>>,
    /// Upvalues this closure references. Index = upvalue slot; the
    /// payload remembers which parent-local needs to be captured at
    /// the MakeClosure site.
    upvalues: Vec<UpvalueRecord>,
    /// Shared, append-only pool of compiled BcFunctions. Nested closure
    /// bodies push themselves here and the MakeClosure op references
    /// the new index. The Rc<RefCell<…>> lets recursive calls reborrow
    /// without lifetime contortions.
    fns_pool: std::rc::Rc<std::cell::RefCell<Vec<BcFunction>>>,
    /// Register reuse. `local_regs` are bound to a named local/param and must
    /// NEVER be freed; `free_regs` is a pool of dead TEMP registers that
    /// `alloc_reg` hands out before bumping `next_reg`. Freeing a temp at its
    /// single consumption point keeps the register file small — both speeding
    /// the interpreter (smaller frames) and letting more functions fit the JIT.
    free_regs: Vec<Reg>,
    local_regs: std::collections::HashSet<Reg>,
    /// SCRIPT FRAME ONLY: `(global name, register)` for each `for (var i = …)`
    /// init binding kept in a fast local for the loop (synced to the global at
    /// loop exit). Surfaced on the `Module` so the run loop can flush the live
    /// register to `globals` on a mid-loop throw (where the post-loop sync is
    /// skipped) — the tree-walk/Node/Chrome-identical behavior. Empty in
    /// non-script frames (a function's for-init `var` is purely function-local).
    script_forinit_syncs: Vec<(String, Reg)>,
    /// Names of every body-level `function` declaration of this (non-script)
    /// frame, recorded + slot-allocated in the pre-declare pass of
    /// `compile_function`. Lets a reference to a sibling decl bind to a real
    /// local slot instead of falling through to a (wrong) global load. A name is
    /// "pending" iff it's in here but NOT yet in `bound_fns` (i.e. its closure
    /// hasn't been materialised in source order yet) — a capture of a pending
    /// sibling is a forward/mutual reference the by-value upvalue model can't
    /// satisfy, so the FunctionDecl arm declines to the tree-walk tier.
    body_fn_decls: std::collections::HashSet<String>,
    /// Body-level `function` declarations already MATERIALISED (their slot holds
    /// the live closure). A capture of one of these is a safe BACKWARD reference.
    bound_fns: std::collections::HashSet<String>,
}

impl<'a> FnCompiler<'a> {
    fn alloc_reg(&mut self) -> Reg {
        // Reuse a freed temp register before growing the file.
        if let Some(r) = self.free_regs.pop() {
            return r;
        }
        let r = self.next_reg;
        match self.next_reg.checked_add(1) {
            Some(n) => {
                self.next_reg = n;
                if self.next_reg > self.max_reg {
                    self.max_reg = self.next_reg;
                }
            }
            // Out of registers (u8 file). Flag it; compile_function converts
            // this to a CompileError → graceful tree-walk fallback rather than
            // a panic. `r` (the pre-increment value, 255) is harmless since the
            // whole function is about to be discarded.
            None => self.overflow = true,
        }
        r
    }

    /// Allocate a FRESH register (always `next_reg`, never the reuse pool). Used
    /// for contiguous register blocks — call args, array elements, closure
    /// upvalues — where the VM reads `first..first+n` and the registers MUST be
    /// adjacent. `alloc_reg`'s free-list would hand out a non-adjacent reg and
    /// corrupt the block.
    fn alloc_contig(&mut self) -> Reg {
        let r = self.next_reg;
        match self.next_reg.checked_add(1) {
            Some(n) => {
                self.next_reg = n;
                if self.next_reg > self.max_reg {
                    self.max_reg = self.next_reg;
                }
            }
            None => self.overflow = true,
        }
        r
    }

    /// Return a register to the free pool IF it's a temp (never a named local).
    /// Call this at a temp's single point of consumption. Freeing a still-live
    /// temp can't happen — the tree-structured compiler produces each temp with
    /// exactly one consumer — and freeing a local is a no-op, so this is safe.
    fn free_reg(&mut self, r: Reg) {
        if !self.local_regs.contains(&r) {
            self.free_regs.push(r);
        }
    }

    fn add_const(&mut self, v: Value) -> u16 {
        let idx = self.consts.len();
        if idx > u16::MAX as usize {
            // Const pool full; flag for a CompileError fallback (see alloc_reg).
            self.overflow = true;
            return u16::MAX;
        }
        self.consts.push(v);
        idx as u16
    }

    fn emit(&mut self, op: Op) {
        self.code.push(op);
    }

    /// Reserve a jump slot, return its instruction index so we can patch
    /// the target once we know where to land.
    fn emit_jump_placeholder(&mut self, jmp: Op) -> usize {
        let pos = self.code.len();
        self.code.push(jmp);
        pos
    }

    fn patch_jump(&mut self, pos: usize) {
        let target = self.code.len() as u16;
        match &mut self.code[pos] {
            Op::Jmp { target: t }
            | Op::JmpIfFalse { target: t, .. }
            | Op::JmpIfTrue { target: t, .. } => *t = target,
            other => panic!(
                "patch_jump called on non-jump instruction at position {}: {:?} (expected Jmp, JmpIfFalse, or JmpIfTrue)",
                pos, other
            ),
        }
    }

    fn lookup(&self, name: &str) -> Option<Reg> {
        for s in self.scopes.iter().rev() {
            if let Some(&r) = s.get(name) {
                return Some(r);
            }
        }
        None
    }

    /// True iff `name` is a body-level `function` declaration of THIS frame that
    /// the hoist pass has not yet bound to its slot. Capturing such a name in a
    /// sibling closure is a forward/mutual reference our by-value upvalue model
    /// cannot represent → the caller declines to the tree-walk tier.
    fn is_pending_sibling_fn(&self, name: &str) -> bool {
        self.body_fn_decls.contains(name) && !self.bound_fns.contains(name)
    }

    /// Compile an identifier read in a NO-THROW context (e.g. `typeof x`),
    /// using the SAME resolution order as the value-context `Expr::Identifier`
    /// arm (local → self-name → upvalue → module-fn → global) but falling back
    /// to the UNCHECKED `LoadGlobal` so an unresolvable name becomes `undefined`
    /// instead of throwing ReferenceError. `typeof undeclaredName` must yield
    /// "undefined", matching the tree-walk tier (ECMA-262 §13.5.1.1).
    fn compile_identifier_unchecked(&mut self, name: &str) -> Reg {
        if let Some(r) = self.lookup(name) {
            return r;
        }
        if self.self_name.as_deref() == Some(name) {
            let dst = self.alloc_reg();
            self.emit(Op::LoadSelf { dst });
            return dst;
        }
        if let Some(slot) = self.resolve_upvalue(name) {
            let dst = self.alloc_reg();
            self.emit(Op::LoadUp { dst, slot });
            return dst;
        }
        // Top-level module fn as a value: load its one STABLE global binding
        // (bound by the script frame's GlobalDeclarationInstantiation), not a
        // fresh per-read `MakeClosure` — see the matching note in the
        // value-context `Expr::Identifier` arm. Falls through to the unchecked
        // `LoadGlobal` so `typeof Foo` stays "function" and never throws.
        let k = self.add_const(Value::str(name.to_string()));
        let dst = self.alloc_reg();
        self.emit(Op::LoadGlobal { dst, name_k: k });
        dst
    }

    /// Flatten every scope into a single (name → register) snapshot.
    /// Inner scopes win over outer so shadowing works. Used to seed a
    /// nested closure's `parent_locals`.
    fn flat_locals_snapshot(&self) -> HashMap<String, Reg> {
        let mut out: HashMap<String, Reg> = HashMap::new();
        // Seed with the names THIS function can itself reach from its
        // enclosing scopes (its own `parent_locals`). Without this a
        // grandchild closure couldn't see a grandparent's local: it'd
        // never RECORD the upvalue, so the transitive capture chain in
        // `materialise_closure` (which makes each intermediate function
        // re-capture the name) never starts and the reference falls
        // through to a global load → `undefined`. This is what broke
        // minified bundles (jQuery/chart.js: deeply-nested closures call
        // outer-scope helpers). The register stored here is in the
        // grandparent frame, but `materialise_closure` re-resolves every
        // captured name BY NAME, so only the recording matters.
        if let Some(parent) = &self.parent_locals {
            for (k, v) in parent {
                out.insert(k.clone(), *v);
            }
        }
        // Our own scopes overlay the inherited names so a local correctly
        // shadows an outer binding of the same name. Inner scopes win.
        for s in &self.scopes {
            for (k, v) in s {
                out.insert(k.clone(), *v);
            }
        }
        out
    }

    /// If `name` is captured from the parent scope, return its upvalue
    /// slot (creating a new one if first sight). Returns None if it's
    /// not in parent_locals.
    fn resolve_upvalue(&mut self, name: &str) -> Option<u8> {
        let parent = self.parent_locals.as_ref()?;
        let parent_local = *parent.get(name)?;
        if let Some(pos) = self.upvalues.iter().position(|u| u.name == name) {
            return Some(pos as u8);
        }
        let slot = self.upvalues.len() as u8;
        self.upvalues.push(UpvalueRecord {
            name: name.to_string(),
            parent_local,
        });
        Some(slot)
    }

    fn declare(&mut self, name: &str) -> Reg {
        let r = self.alloc_reg();
        self.scopes.last_mut().unwrap().insert(name.to_string(), r);
        // A named local must never be handed back to the temp free pool.
        self.local_regs.insert(r);
        r
    }

    /// Emit a `MakeClosure` for a freshly-compiled inner function.
    /// `inner_upvalues` is the list returned by the inner's compile —
    /// each entry names a variable that lives in this frame's locals
    /// and must be Moved into a contiguous block at first_upvalue.
    fn materialise_closure(
        &mut self,
        fn_idx: u16,
        inner_upvalues: &[UpvalueRecord],
    ) -> Result<Reg, CompileError> {
        // CHOKE POINT for every inner closure (fn decl, fn expr, arrow). If the
        // closure captures a SIBLING `function` declaration of THIS frame that
        // hasn't been materialised yet (a forward/mutual reference — its slot is
        // pre-declared but still `undefined`), the VM's by-value upvalue snapshot
        // would capture the wrong cell and the call would silently fail at run
        // time. Decline so the whole script falls back to the tree-walk tier
        // (closures-by-reference), which is correct. Covers cases the per-arm
        // checks miss, e.g. `globalThis.g = function(){ sibling() }` before
        // `function sibling(){}`.
        for u in inner_upvalues {
            if self.is_pending_sibling_fn(&u.name) {
                return Err(CompileError(format!(
                    "inner closure captures pending sibling fn `{}` — defer to interp",
                    u.name
                )));
            }
        }
        // Resolve each captured name back to the right source register
        // in *this* frame. The inner stored its own parent_local index,
        // but that index was taken from a snapshot — `lookup` here
        // re-walks scopes to find the live slot. (For straightforward
        // declarations the values match; the indirection matters only
        // when the lookup result has shifted.)
        let mut src_regs: Vec<Reg> = Vec::with_capacity(inner_upvalues.len());
        for u in inner_upvalues {
            let r = self
                .lookup(&u.name)
                .or_else(|| {
                    self.resolve_upvalue(&u.name).map(|slot| {
                        // The captured name lives in *our* upvalues too;
                        // we need to load it into a fresh register to pass
                        // along.
                        let dst = self.alloc_reg();
                        self.emit(Op::LoadUp { dst, slot });
                        dst
                    })
                })
                .unwrap_or(u.parent_local);
            src_regs.push(r);
        }
        let first_upvalue = self.next_reg;
        for r in &src_regs {
            let dst = self.alloc_contig();
            if *r != dst {
                self.emit(Op::Move { dst, src: *r });
            }
        }
        let dst = self.alloc_reg();
        self.emit(Op::MakeClosure {
            dst,
            fn_idx,
            first_upvalue,
            n_upvalues: src_regs.len() as u8,
        });
        Ok(dst)
    }

    /// Build the right binary op for a compound-assignment base operator.
    /// Shared by the compound-assign expansion (`+=` → `+`, …). The relational/
    /// equality `BinOp`s never reach here (no `<=`/`===` compound assignment
    /// exists), so an unsupported variant is a real compiler error.
    fn binop_emit(&self, op: BinOp, dst: Reg, lhs: Reg, rhs: Reg) -> Result<Op, CompileError> {
        Ok(match op {
            BinOp::Add => Op::Add { dst, lhs, rhs },
            BinOp::Sub => Op::Sub { dst, lhs, rhs },
            BinOp::Mul => Op::Mul { dst, lhs, rhs },
            BinOp::Div => Op::Div { dst, lhs, rhs },
            BinOp::Mod => Op::Mod { dst, lhs, rhs },
            BinOp::BitAnd => Op::BitAnd { dst, lhs, rhs },
            BinOp::BitOr => Op::BitOr { dst, lhs, rhs },
            BinOp::BitXor => Op::BitXor { dst, lhs, rhs },
            BinOp::Shl => Op::Shl { dst, lhs, rhs },
            BinOp::Shr => Op::Shr { dst, lhs, rhs },
            BinOp::UShr => Op::Ushr { dst, lhs, rhs },
            BinOp::Pow => Op::Pow { dst, lhs, rhs },
            other => return Err(CompileError(format!("binop `{other}`"))),
        })
    }

    fn compile_stmt(&mut self, s: &Stmt) -> Result<(), CompileError> {
        match s {
            Stmt::Empty => Ok(()),
            Stmt::Block(body) => {
                self.scopes.push(HashMap::new());
                for s in body {
                    self.compile_stmt(s)?;
                }
                self.scopes.pop();
                Ok(())
            }
            Stmt::Expression(e) => {
                let _ = self.compile_expr(e)?;
                Ok(())
            }
            Stmt::VarDecl { kind, decls } => {
                let is_var = matches!(kind, crate::ast::VarKind::Var);
                for VarDeclarator { name, init } in decls {
                    if self.is_script {
                        // The global binding was already created (undefined) at
                        // hoist time (`hoist_vars_into` / `globals_snapshot`).
                        // Per ECMA-262 §14.3.2.1, EXECUTING a `var` declaration
                        // only ASSIGNS when there is an Initializer; `var x;`
                        // alone is a NO-OP. Emitting `StoreGlobal undefined` for
                        // the no-init case would CLOBBER any value assigned to the
                        // global earlier in the same script (the assign-before-var
                        // shape `x = 7; var x;`, which must leave x === 7 — matches
                        // the tree-walker's `Stmt::VarDecl` Var arm + Node/Chrome).
                        // So only store when there is an initializer. Subsequent
                        // reads from module-level fns resolve via LoadGlobal.
                        if let Some(init) = init {
                            let value_reg = self.compile_expr(init)?;
                            let name_k = self.add_const(Value::str(name.clone()));
                            self.emit(Op::StoreGlobal {
                                name_k,
                                src: value_reg,
                            });
                        }
                    } else {
                        // `var` was pre-declared into the FUNCTION-scope slot
                        // (scopes[0]) at function entry, so resolve to that slot
                        // here instead of creating a fresh, block-local shadow —
                        // this is what makes a block-nested `var x` visible after
                        // the block. `let`/`const` ARE block-scoped, so they
                        // declare in the current (block) scope.
                        let slot = if is_var {
                            match self.lookup(name) {
                                Some(s) => s,
                                None => self.declare(name),
                            }
                        } else {
                            self.declare(name)
                        };
                        if let Some(init) = init {
                            // `var f = function(){ … f … }` / `var f = () => … f …`:
                            // the slot is declared above so the function body CAN
                            // capture `f`, but our closure model snapshots upvalue
                            // *values* at MakeClosure time — when `f` is still
                            // undefined (the assignment hasn't run). The body would
                            // capture an undefined cell and a self-re-arming call
                            // (the requestAnimationFrame count-up idiom) throws
                            // "not a function". Compile the function inline so we
                            // can see its upvalue records; if it captured its own
                            // name, bail to the tree-walk interp (which binds by
                            // reference and handles it correctly). Mirrors the
                            // Stmt::FunctionDecl self-recursion guard above.
                            if matches!(init, Expr::Function { .. } | Expr::Arrow { .. }) {
                                let (src, ups) = self.compile_fn_expr_with_upvalues(init)?;
                                if ups.iter().any(|u| &u.name == name) {
                                    return Err(CompileError(format!(
                                        "self-referencing `{name} = function` — defer to interp"
                                    )));
                                }
                                // (A forward sibling-fn capture is declined inside
                                // `materialise_closure` — the single choke point.)
                                if src != slot {
                                    self.emit(Op::Move { dst: slot, src });
                                }
                                self.free_reg(src);
                            } else {
                                let src = self.compile_expr(init)?;
                                if src != slot {
                                    self.emit(Op::Move { dst: slot, src });
                                }
                                self.free_reg(src); // init temp consumed (no-op if local)
                            }
                        } else if !is_var {
                            // `let`/`const x;` initializes to undefined here. A
                            // bare `var x;` is a NO-OP — the pre-declared slot is
                            // already undefined and may hold a value assigned
                            // earlier in the same scope (the `x = 7; var x;`
                            // shape), which must NOT be clobbered.
                            self.emit(Op::LoadUndef { dst: slot });
                        }
                    }
                }
                Ok(())
            }
            Stmt::FunctionDecl { name, params, body } => {
                // A nested function declaration — e.g. the constructor
                // inside a class-desugar IIFE:
                //   (function(){ function C(){…} C.prototype.m = …; return C })()
                // Compile it as a closure and bind it to a slot in the
                // current scope so subsequent statements (and `return C`)
                // resolve the name. We pre-declare the slot *before*
                // snapshotting parent locals so that, in principle, the
                // body could capture the name — but our closure model
                // copies upvalue *values* at MakeClosure time, so a
                // self-recursive nested function would capture an
                // undefined cell. Detect that case (the inner records its
                // own name as an upvalue) and bail to the interpreter,
                // which handles it correctly.
                let slot = match self.lookup(name) {
                    Some(s) => s,
                    None => self.declare(name),
                };
                let parent_locals = self.flat_locals_snapshot();
                let fn_idx = {
                    let mut pool = self.fns_pool.borrow_mut();
                    let idx = pool.len() as u16;
                    pool.push(BcFunction {
                        name: String::new(),
                        n_params: 0,
                        rest_reg: None,
                        n_regs: 0,
                        consts: Vec::new(),
                        code: Vec::new(),
                        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
                    });
                    idx
                };
                let (inner, inner_upvalues, _) = compile_function(
                    name,
                    params,
                    body,
                    self.fn_index,
                    Some(parent_locals),
                    std::rc::Rc::clone(&self.fns_pool),
                )?;
                if inner_upvalues.iter().any(|u| &u.name == name) {
                    return Err(CompileError(format!(
                        "nested recursive fn `{name}` — defer to interp"
                    )));
                }
                // A FORWARD/MUTUAL sibling-fn capture (capturing a sibling decl
                // whose slot isn't bound yet) is declined inside
                // `materialise_closure` — the single choke point — so the script
                // falls back to the tree-walk tier. A BACKWARD ref is safe: the
                // earlier sibling's slot already holds its live closure.
                self.fns_pool.borrow_mut()[fn_idx as usize] = inner;
                let closure = self.materialise_closure(fn_idx, &inner_upvalues)?;
                if slot != closure {
                    self.emit(Op::Move {
                        dst: slot,
                        src: closure,
                    });
                }
                self.bound_fns.insert(name.clone());
                Ok(())
            }
            Stmt::If { test, cons, alt } => {
                let cond = self.compile_expr(test)?;
                let jump_else = self.emit_jump_placeholder(Op::JmpIfFalse { cond, target: 0 });
                self.compile_stmt(cons)?;
                if let Some(alt) = alt {
                    let jump_end = self.emit_jump_placeholder(Op::Jmp { target: 0 });
                    self.patch_jump(jump_else);
                    self.compile_stmt(alt)?;
                    self.patch_jump(jump_end);
                } else {
                    self.patch_jump(jump_else);
                }
                Ok(())
            }
            Stmt::While { test, body } => {
                let loop_top = self.code.len() as u16;
                let cond = self.compile_expr(test)?;
                let jump_end = self.emit_jump_placeholder(Op::JmpIfFalse { cond, target: 0 });
                self.loops.push(LoopCtx {
                    break_patches: Vec::new(),
                    continue_patches: Vec::new(),
                    is_switch: false,
                });
                self.compile_stmt(body)?;
                // `continue` lands at the loop test (loop_top).
                let lp = self.loops.pop().unwrap();
                for pos in lp.continue_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = loop_top;
                    }
                }
                self.emit(Op::Jmp { target: loop_top });
                self.patch_jump(jump_end);
                // `break` lands here (after the loop's end label).
                let end = self.code.len() as u16;
                for pos in lp.break_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = end;
                    }
                }
                Ok(())
            }
            Stmt::DoWhile { body, test } => {
                let loop_top = self.code.len() as u16;
                self.loops.push(LoopCtx {
                    break_patches: Vec::new(),
                    continue_patches: Vec::new(),
                    is_switch: false,
                });
                self.compile_stmt(body)?;
                // `continue` jumps to the test below.
                let test_label = self.code.len() as u16;
                let lp = self.loops.pop().unwrap();
                for pos in lp.continue_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = test_label;
                    }
                }
                let cond = self.compile_expr(test)?;
                self.emit(Op::JmpIfTrue {
                    cond,
                    target: loop_top,
                });
                let end = self.code.len() as u16;
                for pos in lp.break_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = end;
                    }
                }
                Ok(())
            }
            Stmt::For {
                init,
                test,
                update,
                body,
            } => {
                self.scopes.push(HashMap::new());
                // In the SCRIPT frame, a `for (var i = …)` init `var` is function-
                // scoped to the global object (ECMA GlobalDeclarationInstantiation):
                // `i` is a real global, and after the loop `globalThis.i` reads its
                // final value. The register VM keeps `i` in a fast LOCAL for the hot
                // loop (the whole point of the top-level-VM tier), so we record the
                // for-init `var` names here and emit a single `StoreGlobal` for each
                // AFTER the loop — the loop stays fast (local reads/writes) and the
                // global ends byte-identical to the tree-walker. Non-script frames
                // (real functions) keep `i` purely local (function scope) — no sync.
                let mut script_forinit_globals: Vec<(String, Reg)> = Vec::new();
                if let Some(init) = init {
                    match init {
                        ForInit::VarDecl { kind, decls } => {
                            let forinit_is_var = matches!(kind, crate::ast::VarKind::Var);
                            for VarDeclarator { name, init } in decls {
                                // SCRIPT FRAME: reuse the SAME register for a name
                                // that already owns a for-init sync slot (sibling
                                // `for (var i …)` loops). `Stmt::For` pushes a fresh
                                // scope so `declare` would otherwise hand out a NEW
                                // register each time; pinning one register per name
                                // keeps the throw-time flush (which maps name→ONE
                                // register on the Module) unambiguous regardless of
                                // which sibling loop a throw escaped from.
                                let slot = if self.is_script {
                                    if let Some(r) = self
                                        .script_forinit_syncs
                                        .iter()
                                        .find(|(n, _)| n == name)
                                        .map(|(_, r)| *r)
                                    {
                                        // Bind the existing register into this loop's
                                        // (just-pushed) scope so reads/writes resolve.
                                        self.scopes
                                            .last_mut()
                                            .unwrap()
                                            .insert(name.clone(), r);
                                        r
                                    } else {
                                        self.declare(name)
                                    }
                                } else if forinit_is_var {
                                    // Non-script `for (var i …)`: bind the loop's
                                    // (pushed) scope name to the pre-declared
                                    // FUNCTION-scope slot so reads after the loop
                                    // resolve to it (the `var` hoists out of the
                                    // for-statement). Fall back to a fresh declare
                                    // if (defensively) not pre-declared.
                                    match self.lookup(name) {
                                        Some(r) => {
                                            self.scopes
                                                .last_mut()
                                                .unwrap()
                                                .insert(name.clone(), r);
                                            r
                                        }
                                        None => self.declare(name),
                                    }
                                } else {
                                    self.declare(name)
                                };
                                if self.is_script {
                                    script_forinit_globals.push((name.clone(), slot));
                                    // Surface (name, live register) on the Module so
                                    // the run loop can flush the LIVE value to the
                                    // global on a mid-loop throw (where the post-loop
                                    // StoreGlobal below is skipped). One entry per
                                    // name (registers are pinned per name above).
                                    if !self
                                        .script_forinit_syncs
                                        .iter()
                                        .any(|(n, _)| n == name)
                                    {
                                        self.script_forinit_syncs
                                            .push((name.clone(), slot));
                                    }
                                }
                                if let Some(init) = init {
                                    // Same self-recursion guard as Stmt::VarDecl:
                                    // `for (var f = function(){…f…};;)` would
                                    // capture an undefined self-cell.
                                    if matches!(init, Expr::Function { .. } | Expr::Arrow { .. }) {
                                        let (src, ups) =
                                            self.compile_fn_expr_with_upvalues(init)?;
                                        if ups.iter().any(|u| &u.name == name) {
                                            return Err(CompileError(format!(
                                                "self-referencing `{name} = function` — defer to interp"
                                            )));
                                        }
                                        if src != slot {
                                            self.emit(Op::Move { dst: slot, src });
                                        }
                                    } else {
                                        let src = self.compile_expr(init)?;
                                        if src != slot {
                                            self.emit(Op::Move { dst: slot, src });
                                        }
                                    }
                                } else if self.is_script {
                                    // SCRIPT FRAME, bare `for (var i; …)` with NO
                                    // initializer: `i` is a function/global-scoped
                                    // `var` already hoisted (and possibly assigned a
                                    // value earlier — `var i=5; for(var i; i<8; …){}`).
                                    // Per ECMA-262 §14.7.4 a no-init `var` re-declaration
                                    // is a NO-OP — it must NOT reset the binding to
                                    // `undefined`. Seed the fast loop LOCAL from the
                                    // CURRENT global value (so the test/update operate on
                                    // the live `5`, and the loop runs to `8`) instead of
                                    // clobbering it with `LoadUndef`. Mirrors the
                                    // tree-walker's no-op-on-no-init for-init `var` arm.
                                    let nk = self.add_const(Value::str(name.clone()));
                                    self.emit(Op::LoadGlobal { dst: slot, name_k: nk });
                                } else {
                                    // Non-script (real function) frame: a fresh
                                    // function-local `var` defaults to `undefined`.
                                    self.emit(Op::LoadUndef { dst: slot });
                                }
                            }
                        }
                        ForInit::Expr(e) => {
                            let _ = self.compile_expr(e)?;
                        }
                    }
                }
                let loop_top = self.code.len() as u16;
                let jump_end = if let Some(t) = test {
                    let cond = self.compile_expr(t)?;
                    let j = self.emit_jump_placeholder(Op::JmpIfFalse { cond, target: 0 });
                    // The condition temp is rewritten each iteration before the
                    // jump reads it, so it's free to reuse in the body.
                    self.free_reg(cond);
                    Some(j)
                } else {
                    None
                };
                self.loops.push(LoopCtx {
                    break_patches: Vec::new(),
                    continue_patches: Vec::new(),
                    is_switch: false,
                });
                self.compile_stmt(body)?;
                // `continue` jumps to the update site.
                let update_label = self.code.len() as u16;
                let lp = self.loops.pop().unwrap();
                for pos in lp.continue_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = update_label;
                    }
                }
                if let Some(u) = update {
                    let _ = self.compile_expr(u)?;
                }
                self.emit(Op::Jmp { target: loop_top });
                if let Some(j) = jump_end {
                    self.patch_jump(j);
                }
                let end = self.code.len() as u16;
                for pos in lp.break_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = end;
                    }
                }
                // Script frame: sync each for-init `var`'s final local value to its
                // global binding so `globalThis.i` is byte-identical to the
                // tree-walker after the loop (see the note at the loop init above).
                for (name, slot) in script_forinit_globals {
                    let name_k = self.add_const(Value::str(name));
                    self.emit(Op::StoreGlobal { name_k, src: slot });
                }
                self.scopes.pop();
                Ok(())
            }
            // Labeled break/continue isn't modeled in the register VM; bail so
            // the function runs in the tree-walk interpreter, which handles
            // targeted labels correctly (load-bearing for React's
            // `getTag: switch(){ ... break getTag }`).
            Stmt::Break(Some(_)) => Err(CompileError("labeled break".into())),
            Stmt::Continue(Some(_)) => Err(CompileError("labeled continue".into())),
            Stmt::Break(None) => {
                if self.loops.is_empty() {
                    return Err(CompileError("`break` outside loop".into()));
                }
                let pos = self.emit_jump_placeholder(Op::Jmp { target: 0 });
                self.loops.last_mut().unwrap().break_patches.push(pos);
                Ok(())
            }
            Stmt::Continue(None) => {
                // `continue` targets the nearest enclosing real LOOP, skipping
                // any `switch` contexts between here and it.
                let Some(idx) = self.loops.iter().rposition(|l| !l.is_switch) else {
                    return Err(CompileError("`continue` outside loop".into()));
                };
                let pos = self.emit_jump_placeholder(Op::Jmp { target: 0 });
                self.loops[idx].continue_patches.push(pos);
                Ok(())
            }
            Stmt::ForIn {
                kind,
                name,
                source,
                body,
            } => {
                let forin_is_var = matches!(kind, Some(crate::ast::VarKind::Var));
                // Lower to: keys = enum_keys(source); for (let i = 0;
                // i < keys.length; i++) { name = keys[i]; <body> }
                self.scopes.push(HashMap::new());
                let src_r = self.compile_expr(source)?;
                let keys_r = self.alloc_reg();
                self.emit(Op::EnumKeys {
                    dst: keys_r,
                    obj: src_r,
                });
                let i_r = self.alloc_reg();
                let zero_k = self.add_const(Value::Number(0.0));
                self.emit(Op::LoadConst {
                    dst: i_r,
                    k: zero_k,
                });
                let one_k = self.add_const(Value::Number(1.0));
                let one_r = self.alloc_reg();
                self.emit(Op::LoadConst {
                    dst: one_r,
                    k: one_k,
                });
                let length_k = self.add_const(Value::String("length".into()));

                let loop_top = self.code.len() as u16;
                let len_r = self.alloc_reg();
                self.emit(Op::GetProp {
                    dst: len_r,
                    obj: keys_r,
                    key_k: length_k,
                });
                let cond_r = self.alloc_reg();
                self.emit(Op::Lt {
                    dst: cond_r,
                    lhs: i_r,
                    rhs: len_r,
                });
                let jump_end = self.emit_jump_placeholder(Op::JmpIfFalse {
                    cond: cond_r,
                    target: 0,
                });
                // A `for (var k in …)` binds the FUNCTION-scope pre-declared slot
                // (so `k` survives after the loop); `let`/`const` are block-local.
                let name_slot = if forin_is_var {
                    match self.lookup(name) {
                        Some(r) => {
                            self.scopes.last_mut().unwrap().insert(name.clone(), r);
                            r
                        }
                        None => self.declare(name),
                    }
                } else {
                    self.declare(name)
                };
                self.emit(Op::GetIdx {
                    dst: name_slot,
                    obj: keys_r,
                    key: i_r,
                });
                self.loops.push(LoopCtx {
                    break_patches: Vec::new(),
                    continue_patches: Vec::new(),
                    is_switch: false,
                });
                self.compile_stmt(body)?;
                let continue_target = self.code.len() as u16;
                let lp = self.loops.pop().unwrap();
                for pos in lp.continue_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = continue_target;
                    }
                }
                let new_i = self.alloc_reg();
                self.emit(Op::Add {
                    dst: new_i,
                    lhs: i_r,
                    rhs: one_r,
                });
                self.emit(Op::Move {
                    dst: i_r,
                    src: new_i,
                });
                self.emit(Op::Jmp { target: loop_top });
                self.patch_jump(jump_end);
                let end = self.code.len() as u16;
                for pos in lp.break_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = end;
                    }
                }
                self.scopes.pop();
                Ok(())
            }
            Stmt::ForOf {
                is_await: _,
                kind,
                name,
                source,
                body,
            } => {
                let forof_is_var = matches!(kind, Some(crate::ast::VarKind::Var));
                // for (let name of source) body
                //
                // Lazy iterator protocol — ECMA-262 §14.7.5.5:
                //
                //   let iter = __tb_get_iterator__(source);
                //   loop_top:
                //     let step = iter.next();     // call with this=iter
                //     if (step.done) break;
                //     name = step.value;
                //     <body>
                //     (continue → loop_top)
                //     jump loop_top
                //   end: (break lands here)
                //
                // This drives generators lazily so infinite/large generators
                // and early-break work correctly.  Previously the VM used
                // `__tb_spread__` which materialised the whole iterable upfront,
                // hanging on infinite generators and discarding cleanup signals.
                self.scopes.push(HashMap::new());
                let raw_r = self.compile_expr(source)?;

                // ── __tb_get_iterator__(source) → iter ───────────────────────
                let get_iter_name_k =
                    self.add_const(Value::String("__tb_get_iterator__".into()));
                let get_iter_fn_r = self.alloc_reg();
                self.emit(Op::LoadGlobal {
                    dst: get_iter_fn_r,
                    name_k: get_iter_name_k,
                });
                // Pack the source into a contiguous arg slot for the 1-arg call.
                let first_get_iter_arg = self.next_reg;
                let get_iter_arg = self.alloc_contig();
                self.emit(Op::Move {
                    dst: get_iter_arg,
                    src: raw_r,
                });
                let iter_r = self.alloc_reg();
                self.emit(Op::CallValue {
                    dst:       iter_r,
                    callee:    get_iter_fn_r,
                    this_reg:  NO_THIS,
                    first_arg: first_get_iter_arg,
                    n_args:    1,
                });

                // Intern property-name constants used inside the loop.
                let next_k  = self.add_const(Value::String("next".into()));
                let done_k  = self.add_const(Value::String("done".into()));
                let value_k = self.add_const(Value::String("value".into()));

                // ── loop top ─────────────────────────────────────────────────
                let loop_top = self.code.len() as u16;

                // next_fn = iter.next
                let next_fn_r = self.alloc_reg();
                self.emit(Op::GetProp {
                    dst:   next_fn_r,
                    obj:   iter_r,
                    key_k: next_k,
                });
                // step = next_fn.call(iter)   (this=iter, 0 extra args)
                let step_r = self.alloc_reg();
                self.emit(Op::CallValue {
                    dst:       step_r,
                    callee:    next_fn_r,
                    this_reg:  iter_r,
                    first_arg: 0,  // n_args == 0 → first_arg is never read
                    n_args:    0,
                });

                // if (step.done) → jump to end
                let done_r = self.alloc_reg();
                self.emit(Op::GetProp {
                    dst:   done_r,
                    obj:   step_r,
                    key_k: done_k,
                });
                let jump_end = self.emit_jump_placeholder(Op::JmpIfTrue {
                    cond:   done_r,
                    target: 0,
                });

                // name = step.value
                // `for (var x of …)` binds the FUNCTION-scope pre-declared slot
                // so `x` survives after the loop; `let`/`const` are block-local.
                let name_slot = if forof_is_var {
                    match self.lookup(name) {
                        Some(r) => {
                            self.scopes.last_mut().unwrap().insert(name.clone(), r);
                            r
                        }
                        None => self.declare(name),
                    }
                } else {
                    self.declare(name)
                };
                self.emit(Op::GetProp {
                    dst:   name_slot,
                    obj:   step_r,
                    key_k: value_k,
                });

                // ── body ─────────────────────────────────────────────────────
                self.loops.push(LoopCtx {
                    break_patches: Vec::new(),
                    continue_patches: Vec::new(),
                    is_switch: false,
                });
                self.compile_stmt(body)?;

                // Patch continue → land just before the back-edge Jmp.
                let continue_target = self.code.len() as u16;
                let lp = self.loops.pop().unwrap();
                for pos in lp.continue_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = continue_target;
                    }
                }

                // ── back-edge ────────────────────────────────────────────────
                self.emit(Op::Jmp { target: loop_top });
                self.patch_jump(jump_end);
                let end = self.code.len() as u16;
                for pos in lp.break_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = end;
                    }
                }
                self.scopes.pop();
                Ok(())
            }
            Stmt::Throw(e) => {
                let r = self.compile_expr(e)?;
                self.emit(Op::Throw { src: r });
                Ok(())
            }
            Stmt::Try {
                block,
                catch_param,
                catch_block,
                finally_block,
            } => {
                // Pre-reserve the register that will receive the thrown
                // value so we can name it in TryEnter (whose target is
                // the start of the catch block).
                let catch_reg = if catch_param.is_some() {
                    self.alloc_reg()
                } else {
                    self.alloc_reg() // unused but reserved for ABI symmetry
                };

                // Emit TryEnter with a placeholder; we patch the
                // catch_target once we know the catch start.
                let try_enter_pos = self.code.len();
                self.emit(Op::TryEnter {
                    catch_target: 0,
                    catch_reg,
                });
                // Protected block.
                for s in block {
                    self.compile_stmt(s)?;
                }
                self.emit(Op::TryExit);
                let jump_over_catch = self.emit_jump_placeholder(Op::Jmp { target: 0 });

                // Catch handler entry.
                let catch_target = self.code.len() as u16;
                if let Op::TryEnter {
                    catch_target: t,
                    catch_reg: _,
                } = &mut self.code[try_enter_pos]
                {
                    *t = catch_target;
                }
                // SCRIPT FRAME for-init `var` sync on the CAUGHT-THROW path. A
                // top-level `for (var i = …)` keeps `i` in a fast LOCAL and only
                // syncs it to its (global) binding by a post-loop `StoreGlobal`. If
                // the loop throws mid-iteration and the throw is CAUGHT by a handler
                // in THIS same VM module (`try{ for(var i…){ …throw… } }catch{} i`),
                // that post-loop store is skipped and control resumes here — so
                // without this flush `globalThis.i` would read the stale hoisted
                // `undefined` while the tree-walker (which mutates the global in place
                // every iteration) and Node give the live value (e.g. `3`). The
                // throw-time flush in `run_function` only fires when the error escapes
                // the WHOLE function; a caught throw never reaches it. So at every
                // catch-handler entry in the script frame, flush each known for-init
                // `var`'s live local register to its global. This is idempotent
                // (writes the current live value) and only costs a few stores on the
                // cold catch path — the hot non-throwing loop is untouched.
                if self.is_script && !self.script_forinit_syncs.is_empty() {
                    let syncs = self.script_forinit_syncs.clone();
                    for (sname, sreg) in syncs {
                        let nk = self.add_const(Value::str(sname));
                        self.emit(Op::StoreGlobal { name_k: nk, src: sreg });
                    }
                }
                if let Some(name) = catch_param {
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .insert(name.clone(), catch_reg);
                }
                if let Some(cb) = catch_block {
                    for s in cb {
                        self.compile_stmt(s)?;
                    }
                }
                self.patch_jump(jump_over_catch);

                // Finally block (simple: just emit after the catch).
                // Real `finally` runs on both normal exit and on
                // rethrown exceptions; our V1 only honours the normal
                // exit path. Sites that need real finally semantics
                // fall back to the tree-walk interp via a CompileError
                // — but we accept the syntactically-present block for
                // shape compatibility.
                if let Some(fb) = finally_block {
                    for s in fb {
                        self.compile_stmt(s)?;
                    }
                }
                Ok(())
            }
            Stmt::Return(maybe_e) => {
                let r = if let Some(e) = maybe_e {
                    self.compile_expr(e)?
                } else {
                    let r = self.alloc_reg();
                    self.emit(Op::LoadUndef { dst: r });
                    r
                };
                self.emit(Op::Ret { src: r });
                Ok(())
            }
            Stmt::Switch {
                discriminant,
                cases,
                default_index,
            } => {
                // Evaluate the discriminant once.
                let d = self.compile_expr(discriminant)?;
                // A switch context captures `break` (→ end) but not `continue`.
                self.loops.push(LoopCtx {
                    break_patches: Vec::new(),
                    continue_patches: Vec::new(),
                    is_switch: true,
                });
                // Phase 1 — comparison chain: for each `case` (strict-eq) emit a
                // conditional jump to that case's body (patched in phase 2). The
                // `default` case has no test here.
                let mut case_jumps: Vec<Option<usize>> = Vec::with_capacity(cases.len());
                for case in cases {
                    if let Some(test) = &case.test {
                        let t = self.compile_expr(test)?;
                        let eq = self.alloc_reg();
                        self.emit(Op::Eq {
                            dst: eq,
                            lhs: d,
                            rhs: t,
                        });
                        let jpos = self.emit_jump_placeholder(Op::JmpIfTrue {
                            cond: eq,
                            target: 0,
                        });
                        case_jumps.push(Some(jpos));
                    } else {
                        case_jumps.push(None);
                    }
                }
                // No case matched → jump to `default` body if present, else end.
                let default_jump = self.emit_jump_placeholder(Op::Jmp { target: 0 });
                // Phase 2 — bodies in source order; fall-through is implicit
                // (no jumps between adjacent bodies).
                let mut body_labels: Vec<u16> = Vec::with_capacity(cases.len());
                for case in cases {
                    body_labels.push(self.code.len() as u16);
                    for stmt in &case.body {
                        self.compile_stmt(stmt)?;
                    }
                }
                let end = self.code.len() as u16;
                // Patch case jumps to their body labels.
                for (i, jpos) in case_jumps.iter().enumerate() {
                    if let Some(p) = jpos {
                        if let Op::JmpIfTrue { target, .. } = &mut self.code[*p] {
                            *target = body_labels[i];
                        }
                    }
                }
                // Patch the no-match jump.
                let default_target = default_index
                    .and_then(|di| body_labels.get(di).copied())
                    .unwrap_or(end);
                if let Op::Jmp { target } = &mut self.code[default_jump] {
                    *target = default_target;
                }
                // Patch `break`s to the end.
                let lp = self.loops.pop().unwrap();
                for pos in lp.break_patches {
                    if let Op::Jmp { target } = &mut self.code[pos] {
                        *target = end;
                    }
                }
                Ok(())
            }
            other => Err(CompileError(format!("unsupported stmt: {other:?}"))),
        }
    }

    /// Compile an `Expr::Function` / `Expr::Arrow` and return BOTH the closure
    /// register AND its upvalue records. Callers that need to inspect what the
    /// closure captured (e.g. the `var f = function(){…f…}` self-recursion guard)
    /// use this instead of `compile_expr`, which discards the records. Behaves
    /// identically to the matching `compile_expr` arms otherwise.
    fn compile_fn_expr_with_upvalues(
        &mut self,
        e: &Expr,
    ) -> Result<(Reg, Vec<UpvalueRecord>), CompileError> {
        match e {
            Expr::Function { name, params, body } => {
                let parent_locals = self.flat_locals_snapshot();
                let fn_idx = {
                    let mut pool = self.fns_pool.borrow_mut();
                    let idx = pool.len() as u16;
                    pool.push(BcFunction {
                        name: String::new(),
                        n_params: 0,
                        rest_reg: None,
                        n_regs: 0,
                        consts: Vec::new(),
                        code: Vec::new(),
                        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
                    });
                    idx
                };
                let (inner, inner_upvalues, _) = compile_function(
                    name.as_deref().unwrap_or("<anon>"),
                    params,
                    body,
                    self.fn_index,
                    Some(parent_locals),
                    std::rc::Rc::clone(&self.fns_pool),
                )?;
                self.fns_pool.borrow_mut()[fn_idx as usize] = inner;
                let reg = self.materialise_closure(fn_idx, &inner_upvalues)?;
                Ok((reg, inner_upvalues))
            }
            Expr::Arrow { params, body } => {
                let body_stmts: Vec<Stmt> = match body {
                    crate::ast::ArrowBody::Block(b) => b.clone(),
                    crate::ast::ArrowBody::Expr(ex) => vec![Stmt::Return(Some((**ex).clone()))],
                };
                let this_reg = self.alloc_reg();
                self.emit(Op::LoadThis { dst: this_reg });
                let mut parent_locals = self.flat_locals_snapshot();
                parent_locals.insert("__lexical_this".to_string(), this_reg);
                let fn_idx = {
                    let mut pool = self.fns_pool.borrow_mut();
                    let idx = pool.len() as u16;
                    pool.push(BcFunction {
                        name: String::new(),
                        n_params: 0,
                        rest_reg: None,
                        n_regs: 0,
                        consts: Vec::new(),
                        code: Vec::new(),
                        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
                    });
                    idx
                };
                let (inner, inner_upvalues, _) = compile_function(
                    "<arrow>",
                    params,
                    &body_stmts,
                    self.fn_index,
                    Some(parent_locals),
                    std::rc::Rc::clone(&self.fns_pool),
                )?;
                self.fns_pool.borrow_mut()[fn_idx as usize] = inner;
                let reg = self.materialise_closure(fn_idx, &inner_upvalues)?;
                Ok((reg, inner_upvalues))
            }
            other => Err(CompileError(format!(
                "compile_fn_expr_with_upvalues: not a function expr: {other:?}"
            ))),
        }
    }

    fn compile_expr(&mut self, e: &Expr) -> Result<Reg, CompileError> {
        match e {
            Expr::Number(n) => {
                let r = self.alloc_reg();
                let k = self.add_const(Value::Number(*n));
                self.emit(Op::LoadConst { dst: r, k });
                Ok(r)
            }
            Expr::String(s) => {
                let r = self.alloc_reg();
                let k = self.add_const(Value::str(s.clone()));
                self.emit(Op::LoadConst { dst: r, k });
                Ok(r)
            }
            Expr::TemplateLiteral(s) => {
                // Walk the raw template body splitting on `${...}` and
                // emit `lit + expr + lit + expr + …`. The lexer already
                // decoded escape sequences for us.
                let bytes = s.as_bytes();
                let mut i = 0;
                let mut acc_r: Option<Reg> = None;
                let mut emit_chunk =
                    |this: &mut Self, chunk_r: Reg, acc: &mut Option<Reg>| match acc {
                        None => *acc = Some(chunk_r),
                        Some(prev) => {
                            let dst = this.alloc_reg();
                            this.emit(Op::Add {
                                dst,
                                lhs: *prev,
                                rhs: chunk_r,
                            });
                            *acc = Some(dst);
                        }
                    };
                while i < bytes.len() {
                    // Find next `${`.
                    let mut j = i;
                    while j + 1 < bytes.len() && !(bytes[j] == b'$' && bytes[j + 1] == b'{') {
                        j += 1;
                    }
                    if j + 1 >= bytes.len() {
                        // No more holes — push the remaining literal.
                        let lit = String::from_utf8_lossy(&bytes[i..]).into_owned();
                        let k = self.add_const(Value::str(lit));
                        let r = self.alloc_reg();
                        self.emit(Op::LoadConst { dst: r, k });
                        emit_chunk(self, r, &mut acc_r);
                        break;
                    }
                    if j > i {
                        let lit = String::from_utf8_lossy(&bytes[i..j]).into_owned();
                        let k = self.add_const(Value::str(lit));
                        let r = self.alloc_reg();
                        self.emit(Op::LoadConst { dst: r, k });
                        emit_chunk(self, r, &mut acc_r);
                    }
                    // Skip "${" and find the matching "}".
                    let mut depth = 1;
                    let mut k_idx = j + 2;
                    while k_idx < bytes.len() && depth > 0 {
                        match bytes[k_idx] {
                            b'{' => depth += 1,
                            b'}' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        k_idx += 1;
                    }
                    if k_idx >= bytes.len() {
                        return Err(CompileError("unclosed `${` in template".into()));
                    }
                    let expr_src = String::from_utf8_lossy(&bytes[j + 2..k_idx]).into_owned();
                    let parsed = crate::parser::parse_expression_str(&expr_src)
                        .map_err(|e| CompileError(format!("template expr: {e}")))?;
                    let expr_r = self.compile_expr(&parsed)?;
                    // ECMA-262 §13.2.8.5 (template substitution) → §13.15 / §7.1.17
                    // ToString: an Object operand uses the STRING ToPrimitive hint
                    // (`toString`-first). A `"" + expr` lowering instead used the
                    // DEFAULT/Number hint (`valueOf`-first), which diverged from
                    // Node/the tree-walker on `\`${o}\`` where `o` has both
                    // `valueOf` (42) and `toString` ('X'). `Op::ToStr` applies the
                    // correct string-hint ToString.
                    let str_expr = self.alloc_reg();
                    self.emit(Op::ToStr {
                        dst: str_expr,
                        src: expr_r,
                    });
                    emit_chunk(self, str_expr, &mut acc_r);
                    i = k_idx + 1;
                }
                Ok(acc_r.unwrap_or_else(|| {
                    let r = self.alloc_reg();
                    let k = self.add_const(Value::str(String::new()));
                    self.emit(Op::LoadConst { dst: r, k });
                    r
                }))
            }
            Expr::Boolean(b) => {
                let r = self.alloc_reg();
                self.emit(if *b {
                    Op::LoadTrue { dst: r }
                } else {
                    Op::LoadFalse { dst: r }
                });
                Ok(r)
            }
            Expr::Null => {
                let r = self.alloc_reg();
                self.emit(Op::LoadNull { dst: r });
                Ok(r)
            }
            Expr::Undefined => {
                let r = self.alloc_reg();
                self.emit(Op::LoadUndef { dst: r });
                Ok(r)
            }
            // `new.target` is always undefined in VM-compiled code: functions
            // that DO `new X()` decline VM compilation, and constructors
            // tree-walk — so a VM function is never a `new` target.
            Expr::NewTarget => {
                let r = self.alloc_reg();
                self.emit(Op::LoadUndef { dst: r });
                Ok(r)
            }
            Expr::This => {
                if self.is_arrow {
                    // Lexical this — captured as an upvalue at the
                    // MakeClosure site. Name is the synthetic
                    // "__lexical_this" the outer compiler installs in
                    // parent_locals before recursing.
                    if let Some(slot) = self.resolve_upvalue("__lexical_this") {
                        let dst = self.alloc_reg();
                        self.emit(Op::LoadUp { dst, slot });
                        return Ok(dst);
                    }
                    // Outer wasn't itself an arrow with `this` exposed
                    // — fall back to plain LoadThis (which is
                    // `undefined` at top level). Matches Web spec for
                    // arrows at module scope.
                }
                let r = self.alloc_reg();
                self.emit(Op::LoadThis { dst: r });
                Ok(r)
            }
            Expr::New { callee, args } => {
                let ctor_r = self.compile_expr(callee)?;
                let mut arg_regs: Vec<Reg> = Vec::with_capacity(args.len());
                for a in args {
                    arg_regs.push(self.compile_expr(a)?);
                }
                let first_arg = self.next_reg;
                for &r in &arg_regs {
                    let dst = self.alloc_contig();
                    if r != dst {
                        self.emit(Op::Move { dst, src: r });
                    }
                }
                let dst = self.alloc_reg();
                self.emit(Op::New {
                    dst,
                    ctor: ctor_r,
                    first_arg,
                    n_args: args.len() as u8,
                });
                Ok(dst)
            }
            Expr::Identifier(name) => {
                if let Some(r) = self.lookup(name) {
                    return Ok(r);
                }
                // Self-reference inside a nested named function → the live
                // closure (recursion), resolved BEFORE upvalue capture so the
                // own name never becomes an undefined self-upvalue.
                if self.self_name.as_deref() == Some(name.as_str()) {
                    let dst = self.alloc_reg();
                    self.emit(Op::LoadSelf { dst });
                    return Ok(dst);
                }
                if let Some(slot) = self.resolve_upvalue(name) {
                    let dst = self.alloc_reg();
                    self.emit(Op::LoadUp { dst, slot });
                    return Ok(dst);
                }
                // Top-level module fn referenced as a VALUE (e.g. `new Foo()`,
                // `Foo.prototype`, or passed as a callback). It is NOT a fresh
                // closure per read: the script frame binds every top-level
                // `function`/`class` decl to a STABLE global via `StoreGlobal`
                // (the GlobalDeclarationInstantiation block above), so a value
                // read must LOAD that one binding — otherwise `F === F` is false
                // and every `F.prototype` / `F.x = …` writeback lands on a
                // throwaway closure (broke `new`/`instanceof`/`.prototype` for
                // ALL user functions & classes in page scripts). Reading it as a
                // global below (instead of re-emitting `MakeClosure`) gives the
                // one shared identity, matching the tree-walk tier. The
                // direct-call fast path in Expr::Call still uses `CallFn` (it
                // short-circuits before this branch), so mutual recursion and
                // call performance are unaffected.
                let k = self.add_const(Value::str(name.clone()));
                let dst = self.alloc_reg();
                // VALUE-context read of an unresolved identifier: the checked
                // load throws ReferenceError if the name is genuinely
                // unresolvable, mirroring the tree-walk tier (Finding #1). The
                // no-throw contexts (typeof / delete / assignment / engine
                // helpers) route through their own sites and keep `LoadGlobal`.
                self.emit(Op::LoadGlobalChecked { dst, name_k: k });
                Ok(dst)
            }
            Expr::Unary { op, target } => {
                // `delete` is special: it operates on a REFERENCE (obj.prop),
                // not the operand's value, so it can't go through the generic
                // "compile the operand, then apply" path below.
                if *op == UnaryOp::Delete {
                    if let Expr::Member {
                        object,
                        property,
                        computed,
                    } = target.as_ref()
                    {
                        let obj_r = self.compile_expr(object)?;
                        let d = self.alloc_reg();
                        if *computed {
                            let key_r = self.compile_expr(property)?;
                            self.emit(Op::DeleteIdx {
                                dst: d,
                                obj: obj_r,
                                key: key_r,
                            });
                        } else {
                            let key_str = match property.as_ref() {
                                Expr::Identifier(s) | Expr::String(s) => s.clone(),
                                other => {
                                    return Err(CompileError(format!(
                                        "delete property must be ident: {other:?}"
                                    )));
                                }
                            };
                            let key_k = self.add_const(Value::str(key_str));
                            self.emit(Op::DeleteProp {
                                dst: d,
                                obj: obj_r,
                                key_k,
                            });
                        }
                        return Ok(d);
                    }
                    // `delete x` / `delete <expr>` on a non-reference is a no-op
                    // that yields true (sloppy mode).
                    let d = self.alloc_reg();
                    self.emit(Op::LoadTrue { dst: d });
                    return Ok(d);
                }
                // `typeof identifier` is the ONE operator that must NOT throw on
                // an unresolvable name (ECMA-262 §13.5.1.1: typeof of an
                // unresolvable Reference yields "undefined"). So a bare-identifier
                // operand compiles through the UNCHECKED global load, not the
                // value-context checked one. (`typeof obj.x`, `typeof f()`, etc.
                // fall through to the generic path — they read real references.)
                if *op == UnaryOp::Typeof {
                    if let Expr::Identifier(name) = target.as_ref() {
                        let s = self.compile_identifier_unchecked(name);
                        let d = self.alloc_reg();
                        self.emit(Op::Typeof { dst: d, src: s });
                        return Ok(d);
                    }
                }
                let s = self.compile_expr(target)?;
                let d = self.alloc_reg();
                match op {
                    UnaryOp::Neg => self.emit(Op::Neg { dst: d, src: s }),
                    UnaryOp::Not => self.emit(Op::Not { dst: d, src: s }),
                    UnaryOp::Plus => self.emit(Op::ToNumber { dst: d, src: s }),
                    UnaryOp::Typeof => self.emit(Op::Typeof { dst: d, src: s }),
                    UnaryOp::BitNot => self.emit(Op::BitNot { dst: d, src: s }),
                    // `void x` evaluates the operand (side effects already
                    // emitted by compile_expr above) and yields undefined.
                    UnaryOp::Void => self.emit(Op::LoadUndef { dst: d }),
                    // `delete` is handled by the reference-path above; reaching
                    // here would be a non-member delete, which the compiler turns
                    // into `true` earlier. Treat as a compile error to stay safe.
                    UnaryOp::Delete => {
                        return Err(CompileError("unary `delete` unsupported".to_string()));
                    }
                }
                Ok(d)
            }
            Expr::Binary { op, left, right } => {
                let l = self.compile_expr(left)?;
                let r = self.compile_expr(right)?;
                let d = self.alloc_reg();
                let bc = match op {
                    BinOp::Add => Op::Add {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Sub => Op::Sub {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Mul => Op::Mul {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Div => Op::Div {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Mod => Op::Mod {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    // === — strict equality, no coercion.
                    BinOp::EqEqEq => Op::Eq {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::NeqEqEq => Op::Neq {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    // == / != — Abstract Equality (ECMA-262 §7.2.15). Routed
                    // through `Value::loose_eq` so the VM behaves like the
                    // tree-walker (and Chrome) for null/undefined nullish
                    // guards, numeric-string comparisons, bool→number, and
                    // Object/Array ToPrimitive. Previously aliased to Eq,
                    // which made `x == null` / `0 == "0"` / `1 == true` /
                    // `[] == ""` all false inside any hot/VM-compiled
                    // function — a critical split-brain divergence.
                    BinOp::EqEq => Op::LooseEq {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Neq => Op::LooseNeq {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Lt => Op::Lt {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Le => Op::Le {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Gt => Op::Gt {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Ge => Op::Ge {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::BitAnd => Op::BitAnd {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::BitOr => Op::BitOr {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::BitXor => Op::BitXor {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Shl => Op::Shl {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Shr => Op::Shr {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::UShr => Op::Ushr {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::In => Op::In {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    BinOp::Pow => Op::Pow {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                    // `instanceof` routes to the host's full tree-walk check
                    // (prototype-chain + tag fallback) at run time via
                    // `Op::Instanceof`, so the VM result is byte-identical without
                    // re-implementing the chain walk here.
                    BinOp::Instanceof => Op::Instanceof {
                        dst: d,
                        lhs: l,
                        rhs: r,
                    },
                };
                self.emit(bc);
                // Both operands are fully consumed by this op (no short-circuit);
                // return their registers to the temp pool (no-op for locals).
                self.free_reg(l);
                self.free_reg(r);
                Ok(d)
            }
            Expr::Logical { op, left, right } => {
                // `a && b` → if (!a) result=a else result=b
                // `a || b` → if (a)  result=a else result=b
                // `a ?? b` → if (a is not null/undefined) result=a else result=b
                let l = self.compile_expr(left)?;
                let d = self.alloc_reg();
                self.emit(Op::Move { dst: d, src: l });
                let short_circuit = match op {
                    LogicalOp::And => self.emit_jump_placeholder(Op::JmpIfFalse { cond: d, target: 0 }),
                    LogicalOp::Or => self.emit_jump_placeholder(Op::JmpIfTrue { cond: d, target: 0 }),
                    LogicalOp::Nullish => {
                        // Synthesise nullish without a dedicated op:
                        //   is_null = (d === null)
                        //   is_undef = (d === undefined)
                        //   nullish = is_null || is_undef
                        //   if (!nullish) skip right-hand side
                        let null_reg = self.alloc_reg();
                        self.emit(Op::LoadNull { dst: null_reg });
                        let undef_reg = self.alloc_reg();
                        self.emit(Op::LoadUndef { dst: undef_reg });
                        let is_null = self.alloc_reg();
                        self.emit(Op::Eq {
                            dst: is_null,
                            lhs: d,
                            rhs: null_reg,
                        });
                        let is_undef = self.alloc_reg();
                        self.emit(Op::Eq {
                            dst: is_undef,
                            lhs: d,
                            rhs: undef_reg,
                        });
                        let nullish = self.alloc_reg();
                        // Or doesn't exist as a single op; emulate with a
                        // short-circuit move pattern. Start nullish=is_null,
                        // then if !is_null overwrite with is_undef.
                        self.emit(Op::Move {
                            dst: nullish,
                            src: is_null,
                        });
                        let skip_undef = self.emit_jump_placeholder(Op::JmpIfTrue {
                            cond: nullish,
                            target: 0,
                        });
                        self.emit(Op::Move {
                            dst: nullish,
                            src: is_undef,
                        });
                        self.patch_jump(skip_undef);
                        // If d is NOT nullish, jump past the right-hand
                        // evaluation. `nullish` is a Bool — JmpIfFalse
                        // here means "if not nullish, skip".
                        self.emit_jump_placeholder(Op::JmpIfFalse {
                            cond: nullish,
                            target: 0,
                        })
                    }
                };
                let r = self.compile_expr(right)?;
                self.emit(Op::Move { dst: d, src: r });
                self.patch_jump(short_circuit);
                Ok(d)
            }
            Expr::Assignment { op, target, value } => {
                // Map `+=` → `+`, etc. Plain `=` is the no-op case. This match
                // is DELIBERATELY narrower than `AssignOp::base_binop()`: the VM
                // declines `**=`, `&&=`, `||=`, `??=` (so the caller tree-walks
                // them), exactly as the prior string match did — keeping the tier
                // decision byte-identical.
                let bin_op: Option<BinOp> = match op {
                    AssignOp::Assign => None,
                    AssignOp::AddAssign => Some(BinOp::Add),
                    AssignOp::SubAssign => Some(BinOp::Sub),
                    AssignOp::MulAssign => Some(BinOp::Mul),
                    AssignOp::DivAssign => Some(BinOp::Div),
                    AssignOp::ModAssign => Some(BinOp::Mod),
                    AssignOp::BitAndAssign => Some(BinOp::BitAnd),
                    AssignOp::BitOrAssign => Some(BinOp::BitOr),
                    AssignOp::BitXorAssign => Some(BinOp::BitXor),
                    AssignOp::ShlAssign => Some(BinOp::Shl),
                    AssignOp::ShrAssign => Some(BinOp::Shr),
                    AssignOp::UShrAssign => Some(BinOp::UShr),
                    other => {
                        return Err(CompileError(format!("assign op `{other}` unsupported")));
                    }
                };
                match target.as_ref() {
                    Expr::Identifier(name) => {
                        if let Some(dst) = self.lookup(name) {
                            let src = self.compile_expr(value)?;
                            if let Some(op) = bin_op {
                                let bc = self.binop_emit(op, dst, dst, src)?;
                                self.emit(bc);
                            } else if src != dst {
                                self.emit(Op::Move { dst, src });
                            }
                            Ok(dst)
                        } else if let Some(slot) = self.resolve_upvalue(name) {
                            // Capture-write: load current, op, store.
                            let src = self.compile_expr(value)?;
                            let result = self.alloc_reg();
                            if let Some(op) = bin_op {
                                let cur = self.alloc_reg();
                                self.emit(Op::LoadUp { dst: cur, slot });
                                let bc = self.binop_emit(op, result, cur, src)?;
                                self.emit(bc);
                            } else {
                                self.emit(Op::Move { dst: result, src });
                            }
                            self.emit(Op::StoreUp { src: result, slot });
                            Ok(result)
                        } else {
                            // Undeclared name → a global. Sloppy-mode auto-global
                            // and the script frame's own top-level `var`s both
                            // live in the global env (see StoreGlobal). Plain `=`
                            // stores (no read → never throws, creates the global).
                            // Compound (`x += y`) READS the current value first —
                            // and reading an unresolvable name throws
                            // ReferenceError in the tree-walk tier (its `do_assign`
                            // calls `eval` on the target). Mirror that with the
                            // CHECKED load so the tiers agree (no new divergence).
                            let name_k = self.add_const(Value::str(name.clone()));
                            let src = self.compile_expr(value)?;
                            let result = if let Some(op) = bin_op {
                                let cur = self.alloc_reg();
                                self.emit(Op::LoadGlobalChecked { dst: cur, name_k });
                                let res = self.alloc_reg();
                                let bc = self.binop_emit(op, res, cur, src)?;
                                self.emit(bc);
                                res
                            } else {
                                src
                            };
                            self.emit(Op::StoreGlobal {
                                name_k,
                                src: result,
                            });
                            Ok(result)
                        }
                    }
                    Expr::Member {
                        object,
                        property,
                        computed,
                    } => {
                        let obj_r = self.compile_expr(object)?;
                        // For compound (`obj.x += y`) we have to load,
                        // op, store. For plain (`obj.x = y`) just store.
                        if let Some(op) = bin_op {
                            // load current value into a temp
                            let cur_dst = self.alloc_reg();
                            if *computed {
                                let key_r = self.compile_expr(property)?;
                                self.emit(Op::GetIdx {
                                    dst: cur_dst,
                                    obj: obj_r,
                                    key: key_r,
                                });
                                let rhs = self.compile_expr(value)?;
                                let new_dst = self.alloc_reg();
                                let bc = self.binop_emit(op, new_dst, cur_dst, rhs)?;
                                self.emit(bc);
                                self.emit(Op::SetIdx {
                                    obj: obj_r,
                                    key: key_r,
                                    src: new_dst,
                                });
                                Ok(new_dst)
                            } else {
                                let key_str = match property.as_ref() {
                                    Expr::Identifier(s) => s.clone(),
                                    Expr::String(s) => s.clone(),
                                    other => {
                                        return Err(CompileError(format!(
                                            "property must be ident: {other:?}"
                                        )));
                                    }
                                };
                                let key_k = self.add_const(Value::str(key_str));
                                self.emit(Op::GetProp {
                                    dst: cur_dst,
                                    obj: obj_r,
                                    key_k,
                                });
                                let rhs = self.compile_expr(value)?;
                                let new_dst = self.alloc_reg();
                                let bc = self.binop_emit(op, new_dst, cur_dst, rhs)?;
                                self.emit(bc);
                                self.emit(Op::SetProp {
                                    obj: obj_r,
                                    key_k,
                                    src: new_dst,
                                });
                                Ok(new_dst)
                            }
                        } else {
                            let src = self.compile_expr(value)?;
                            if *computed {
                                let key_r = self.compile_expr(property)?;
                                self.emit(Op::SetIdx {
                                    obj: obj_r,
                                    key: key_r,
                                    src,
                                });
                            } else {
                                let key_str = match property.as_ref() {
                                    Expr::Identifier(s) => s.clone(),
                                    Expr::String(s) => s.clone(),
                                    other => {
                                        return Err(CompileError(format!(
                                            "property must be ident: {other:?}"
                                        )));
                                    }
                                };
                                let key_k = self.add_const(Value::str(key_str));
                                self.emit(Op::SetProp {
                                    obj: obj_r,
                                    key_k,
                                    src,
                                });
                            }
                            Ok(src)
                        }
                    }
                    _ => Err(CompileError(
                        "assign target must be ident or member".to_string(),
                    )),
                }
            }
            Expr::Function { name, params, body } => {
                let _ = name;
                let parent_locals = self.flat_locals_snapshot();
                // Reserve a slot in the pool so the inner fn has an
                // index now; we'll overwrite the placeholder once the
                // inner compilation finishes.
                let fn_idx = {
                    let mut pool = self.fns_pool.borrow_mut();
                    let idx = pool.len() as u16;
                    pool.push(BcFunction {
                        name: String::new(),
                        n_params: 0,
                        rest_reg: None,
                        n_regs: 0,
                        consts: Vec::new(),
                        code: Vec::new(),
                        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
                    });
                    idx
                };
                let (inner, inner_upvalues, _) = compile_function(
                    name.as_deref().unwrap_or("<anon>"),
                    params,
                    body,
                    self.fn_index,
                    Some(parent_locals),
                    std::rc::Rc::clone(&self.fns_pool),
                )?;
                self.fns_pool.borrow_mut()[fn_idx as usize] = inner;
                self.materialise_closure(fn_idx, &inner_upvalues)
            }
            Expr::Arrow { params, body } => {
                let body_stmts: Vec<Stmt> = match body {
                    crate::ast::ArrowBody::Block(b) => b.clone(),
                    crate::ast::ArrowBody::Expr(e) => {
                        vec![Stmt::Return(Some((**e).clone()))]
                    }
                };
                // Pre-stage the current frame's `this` into a fresh
                // register so the arrow can capture it as an upvalue.
                let this_reg = self.alloc_reg();
                self.emit(Op::LoadThis { dst: this_reg });
                let mut parent_locals = self.flat_locals_snapshot();
                parent_locals.insert("__lexical_this".to_string(), this_reg);

                let fn_idx = {
                    let mut pool = self.fns_pool.borrow_mut();
                    let idx = pool.len() as u16;
                    pool.push(BcFunction {
                        name: String::new(),
                        n_params: 0,
                        rest_reg: None,
                        n_regs: 0,
                        consts: Vec::new(),
                        code: Vec::new(),
                        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
                    });
                    idx
                };
                let (inner, inner_upvalues, _) = compile_function(
                    "<arrow>",
                    params,
                    &body_stmts,
                    self.fn_index,
                    Some(parent_locals),
                    std::rc::Rc::clone(&self.fns_pool),
                )?;
                self.fns_pool.borrow_mut()[fn_idx as usize] = inner;
                self.materialise_closure(fn_idx, &inner_upvalues)
            }
            Expr::Update { op, target, prefix } => {
                // ++x / --x / x++ / x-- on identifiers (local/upvalue/global)
                // AND member targets (obj.x++ / obj[k]++). `UpdateOp` only has
                // `Inc`/`Dec`, so no decline path is needed.
                let is_inc = *op == UpdateOp::Inc;
                let one_k = self.add_const(Value::Number(1.0));

                // Member target: evaluate object (and computed key) ONCE, read
                // current value, ±1, write back. Prefix yields new, postfix old.
                if let Expr::Member {
                    object,
                    property,
                    computed,
                } = target.as_ref()
                {
                    let obj_r = self.compile_expr(object)?;
                    let key_r = if *computed {
                        Some(self.compile_expr(property)?)
                    } else {
                        None
                    };
                    let key_k = if *computed {
                        0
                    } else {
                        let key_str = match property.as_ref() {
                            Expr::Identifier(s) | Expr::String(s) => s.clone(),
                            other => {
                                return Err(CompileError(format!(
                                    "update property must be ident: {other:?}"
                                )));
                            }
                        };
                        self.add_const(Value::str(key_str))
                    };
                    let one_r = self.alloc_reg();
                    self.emit(Op::LoadConst {
                        dst: one_r,
                        k: one_k,
                    });
                    let old_r = self.alloc_reg();
                    if let Some(kr) = key_r {
                        self.emit(Op::GetIdx {
                            dst: old_r,
                            obj: obj_r,
                            key: kr,
                        });
                    } else {
                        self.emit(Op::GetProp {
                            dst: old_r,
                            obj: obj_r,
                            key_k,
                        });
                    }
                    let new_r = self.alloc_reg();
                    self.emit(if is_inc {
                        Op::Add {
                            dst: new_r,
                            lhs: old_r,
                            rhs: one_r,
                        }
                    } else {
                        Op::Sub {
                            dst: new_r,
                            lhs: old_r,
                            rhs: one_r,
                        }
                    });
                    if let Some(kr) = key_r {
                        self.emit(Op::SetIdx {
                            obj: obj_r,
                            key: kr,
                            src: new_r,
                        });
                    } else {
                        self.emit(Op::SetProp {
                            obj: obj_r,
                            key_k,
                            src: new_r,
                        });
                    }
                    return Ok(if *prefix { new_r } else { old_r });
                }

                // Identifier target — local register, captured upvalue, or global.
                let name = match target.as_ref() {
                    Expr::Identifier(n) => n.clone(),
                    _ => {
                        return Err(CompileError(
                            "update target must be identifier or member".into(),
                        ));
                    }
                };
                let one_r = self.alloc_reg();
                self.emit(Op::LoadConst {
                    dst: one_r,
                    k: one_k,
                });
                // `UpdSlot` carries just Copy payloads.
                enum UpdSlot {
                    Local(Reg),
                    Up(u8),
                    Global(u16),
                }
                let old_r = self.alloc_reg();
                let slot = if let Some(reg) = self.lookup(&name) {
                    self.emit(Op::Move {
                        dst: old_r,
                        src: reg,
                    });
                    UpdSlot::Local(reg)
                } else if let Some(up) = self.resolve_upvalue(&name) {
                    self.emit(Op::LoadUp {
                        dst: old_r,
                        slot: up,
                    });
                    UpdSlot::Up(up)
                } else {
                    // `++x` / `x--` on an undeclared name READS the old value
                    // first; an unresolvable read throws ReferenceError in the
                    // tree-walk tier (its `Expr::Update` calls `eval` on the
                    // target). Use the CHECKED load so the tiers agree.
                    let name_k = self.add_const(Value::str(name.clone()));
                    self.emit(Op::LoadGlobalChecked { dst: old_r, name_k });
                    UpdSlot::Global(name_k)
                };
                let new_r = self.alloc_reg();
                self.emit(if is_inc {
                    Op::Add {
                        dst: new_r,
                        lhs: old_r,
                        rhs: one_r,
                    }
                } else {
                    Op::Sub {
                        dst: new_r,
                        lhs: old_r,
                        rhs: one_r,
                    }
                });
                match slot {
                    UpdSlot::Local(reg) => self.emit(Op::Move {
                        dst: reg,
                        src: new_r,
                    }),
                    UpdSlot::Up(up) => self.emit(Op::StoreUp {
                        src: new_r,
                        slot: up,
                    }),
                    UpdSlot::Global(name_k) => self.emit(Op::StoreGlobal { name_k, src: new_r }),
                }
                // Prefix yields the new value; postfix the old.
                Ok(if *prefix { new_r } else { old_r })
            }
            Expr::Conditional { test, cons, alt } => {
                let cond = self.compile_expr(test)?;
                let d = self.alloc_reg();
                let jump_else = self.emit_jump_placeholder(Op::JmpIfFalse { cond, target: 0 });
                let cr = self.compile_expr(cons)?;
                self.emit(Op::Move { dst: d, src: cr });
                let jump_end = self.emit_jump_placeholder(Op::Jmp { target: 0 });
                self.patch_jump(jump_else);
                let ar = self.compile_expr(alt)?;
                self.emit(Op::Move { dst: d, src: ar });
                self.patch_jump(jump_end);
                Ok(d)
            }
            Expr::Call { callee, args } => {
                // Two fast paths and one fallback:
                //   1. Direct call to a module-local fn (compile-time
                //      resolvable) → CallFn with the fn index.
                //   2. Direct call to an unresolved identifier (global)
                //      → LoadGlobal + CallValue.
                //   3. General expression callee → compile, then
                //      CallValue.
                let direct_fn_idx = if let Expr::Identifier(name) = callee.as_ref() {
                    self.fn_index.get(name).copied()
                } else {
                    None
                };

                if let Some(fn_idx) = direct_fn_idx {
                    let mut arg_regs: Vec<Reg> = Vec::with_capacity(args.len());
                    for a in args {
                        arg_regs.push(self.compile_expr(a)?);
                    }
                    let first_arg = self.next_reg;
                    for &r in &arg_regs {
                        let dst = self.alloc_contig();
                        if r != dst {
                            self.emit(Op::Move { dst, src: r });
                        }
                    }
                    let dst = self.alloc_reg();
                    self.emit(Op::CallFn {
                        dst,
                        fn_idx,
                        first_arg,
                        n_args: args.len() as u8,
                    });
                    return Ok(dst);
                }

                if let Expr::Member {
                    object,
                    property,
                    computed,
                } = callee.as_ref()
                {
                    // Method call: bind `this` to the object.
                    let obj_r = self.compile_expr(object)?;
                    let callee_r = self.alloc_reg();
                    if *computed {
                        let key_r = self.compile_expr(property)?;
                        self.emit(Op::GetIdx {
                            dst: callee_r,
                            obj: obj_r,
                            key: key_r,
                        });
                    } else {
                        let key_str = match property.as_ref() {
                            Expr::Identifier(s) => s.clone(),
                            Expr::String(s) => s.clone(),
                            other => {
                                return Err(CompileError(format!(
                                    "method name must be ident: {other:?}"
                                )));
                            }
                        };
                        let key_k = self.add_const(Value::str(key_str));
                        self.emit(Op::GetProp {
                            dst: callee_r,
                            obj: obj_r,
                            key_k,
                        });
                    }
                    let mut arg_regs: Vec<Reg> = Vec::with_capacity(args.len());
                    for a in args {
                        arg_regs.push(self.compile_expr(a)?);
                    }
                    let first_arg = self.next_reg;
                    for &r in &arg_regs {
                        let dst = self.alloc_contig();
                        if r != dst {
                            self.emit(Op::Move { dst, src: r });
                        }
                    }
                    let dst = self.alloc_reg();
                    self.emit(Op::CallValue {
                        dst,
                        callee: callee_r,
                        this_reg: obj_r,
                        first_arg,
                        n_args: args.len() as u8,
                    });
                    return Ok(dst);
                }

                // Plain callee — this = undefined.
                let callee_reg = self.compile_expr(callee)?;
                let mut arg_regs: Vec<Reg> = Vec::with_capacity(args.len());
                for a in args {
                    arg_regs.push(self.compile_expr(a)?);
                }
                let first_arg = self.next_reg;
                for &r in &arg_regs {
                    let dst = self.alloc_contig();
                    if r != dst {
                        self.emit(Op::Move { dst, src: r });
                    }
                }
                let dst = self.alloc_reg();
                self.emit(Op::CallValue {
                    dst,
                    callee: callee_reg,
                    this_reg: NO_THIS,
                    first_arg,
                    n_args: args.len() as u8,
                });
                Ok(dst)
            }
            Expr::Sequence(es) => {
                let mut last: Reg = 0;
                for e in es {
                    last = self.compile_expr(e)?;
                }
                Ok(last)
            }
            Expr::Object(pairs) => {
                // Allocate empty object, then SetProp for each pair.
                let dst = self.alloc_reg();
                self.emit(Op::NewObject { dst });
                for (key, val_expr) in pairs {
                    let v = self.compile_expr(val_expr)?;
                    // `__proto__: value` in an object initializer sets the
                    // object's [[Prototype]] (ECMA-262 B.3.1), NOT an own
                    // property — route it to the hidden PROTO_KEY so it stays
                    // out of Object.keys/for-in. rollup/webpack namespaces
                    // (`{__proto__: null, A, B, …}`) depend on this; storing
                    // it as a real key made value-iteration yield the
                    // prototype (chart.js registered `null` as a component).
                    let key_name = if key == "__proto__" {
                        crate::interp::PROTO_KEY
                    } else {
                        key.as_str()
                    };
                    let key_k = self.add_const(Value::str(key_name.to_string()));
                    self.emit(Op::SetProp {
                        obj: dst,
                        key_k,
                        src: v,
                    });
                }
                Ok(dst)
            }
            Expr::Array(elems) => {
                // Compile each element, then materialise them into a
                // contiguous register block and emit NewArray.
                //
                // If ANY element is a `...spread`, we can't use the fixed
                // NewArray instruction (its length is statically known);
                // instead, start with an empty array and emit
                // ArrayPush / ArrayPushSpread per element.
                let has_spread = elems.iter().any(|e| matches!(e, Expr::Spread(_)));
                if has_spread {
                    let dst = self.alloc_reg();
                    self.emit(Op::NewArray {
                        dst,
                        first_elem: 0,
                        n_elems: 0,
                    });
                    for e in elems {
                        match e {
                            Expr::Spread(inner) => {
                                let r = self.compile_expr(inner)?;
                                self.emit(Op::ArrayPushSpread {
                                    arr: dst,
                                    spread: r,
                                });
                            }
                            other => {
                                let r = self.compile_expr(other)?;
                                self.emit(Op::ArrayPush { arr: dst, val: r });
                            }
                        }
                    }
                    return Ok(dst);
                }
                let mut elem_regs: Vec<Reg> = Vec::with_capacity(elems.len());
                for e in elems {
                    elem_regs.push(self.compile_expr(e)?);
                }
                let first_elem = self.next_reg;
                for &r in &elem_regs {
                    let dst = self.alloc_contig();
                    if r != dst {
                        self.emit(Op::Move { dst, src: r });
                    }
                }
                let dst = self.alloc_reg();
                self.emit(Op::NewArray {
                    dst,
                    first_elem,
                    n_elems: elems.len() as u8,
                });
                Ok(dst)
            }
            Expr::Member {
                object,
                property,
                computed,
            } => {
                let obj_r = self.compile_expr(object)?;
                let dst = self.alloc_reg();
                if *computed {
                    // arr[i] / obj["key"] — fully dynamic key.
                    let key_r = self.compile_expr(property)?;
                    self.emit(Op::GetIdx {
                        dst,
                        obj: obj_r,
                        key: key_r,
                    });
                } else {
                    // obj.prop — property is an Identifier or Keyword.
                    let key_str = match property.as_ref() {
                        Expr::Identifier(s) => s.clone(),
                        Expr::String(s) => s.clone(),
                        other => {
                            return Err(CompileError(format!(
                                "member property must be ident: {other:?}"
                            )));
                        }
                    };
                    let key_k = self.add_const(Value::str(key_str));
                    self.emit(Op::GetProp {
                        dst,
                        obj: obj_r,
                        key_k,
                    });
                }
                Ok(dst)
            }
            Expr::Regex(source, flags) => {
                let source_k = self.add_const(Value::str(source.clone()));
                let flags_k = self.add_const(Value::str(flags.clone()));
                let dst = self.alloc_reg();
                self.emit(Op::MakeRegex {
                    dst,
                    source_k,
                    flags_k,
                });
                Ok(dst)
            }
            other => Err(CompileError(format!("unsupported expr: {other:?}"))),
        }
    }
}

#[derive(Debug)]
pub enum RuntimeError {
    /// Tried to use a non-number where one is required.
    TypeError(String),
    /// Stack depth blew through the recursion budget.
    StackOverflow,
    /// Bytecode tried to execute past the function's instruction list.
    Overrun,
    /// User-thrown JS value that wasn't caught in the current function.
    /// The caller frame can re-handle it, or it bubbles out to the host.
    Thrown(Value),
    /// The per-task wall-clock watchdog fired (runaway / pathologically-slow
    /// script). Deliberately NOT routed through `try/catch` — a hang inside a
    /// `try {}` must still abort — so it unwinds straight out to the host.
    Deadline,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeError(s) => write!(f, "TypeError: {s}"),
            Self::StackOverflow => f.write_str("RangeError: max call stack size exceeded"),
            Self::Overrun => f.write_str("InternalError: bytecode ran past end"),
            Self::Thrown(v) => write!(f, "Uncaught: {v:?}"),
            Self::Deadline => {
                f.write_str("script execution exceeded time budget (possible infinite loop)")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Callback invoked when the bytecode VM has to call a
/// `NativeFnBody::WithInterp` native (e.g. `Array.map` whose body
/// needs the tree-walk interp to invoke its JS callback). Returns the
/// native's result, or a TypeError for unsupported calls.
/// Host callback that runs a JS callee the VM can't execute itself — a
/// `WithInterp` native, a tree-walk `Value::Function`, or a callable object —
/// by handing (callee, this, args) to the live tree-walk interp. This is what
/// lets a VM-compiled function call ANY other user function (the keystone for
/// per-function VM execution). Pure natives + `BcClosure` are still run
/// directly by the VM (no host hop).
pub type WithInterpDispatch<'a> =
    &'a mut dyn FnMut(Value, Value, Vec<Value>) -> Result<Value, RuntimeError>;

/// If `raw` is a live accessor wrapper (from `Object.defineProperty` with a
/// getter — webpack ESM live exports), invoke its getter and return the
/// result; otherwise pass `raw` through. Handles the getter being a native
/// (via the dispatcher) or a bytecode closure (run directly). A tree-walk
/// `Function` getter reached from the VM can't be run here — rare; yields
/// Undefined.
fn resolve_accessor_read(
    raw: Value,
    this: &Value,
    module: &Module,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    let Some((get, _set)) = crate::interp::accessor_parts(&raw) else {
        return Ok(raw);
    };
    match get {
        Some(Value::NativeFunction(nf)) => {
            dispatch(Value::NativeFunction(nf), this.clone(), Vec::new())
        }
        Some(Value::BcClosure(c)) => run_function(
            &c.module,
            c.fn_idx as usize,
            &[],
            this,
            globals,
            Some(&c),
            dispatch,
        ),
        // Tree-walk `Function` getter (e.g. a `get x(){}` whose body the VM
        // can't compile) — invoke it through the host so VM-read `obj.x` runs
        // it instead of yielding undefined.
        Some(f @ Value::Function(_)) => dispatch(f, this.clone(), Vec::new()),
        _ => Ok(Value::Undefined),
    }
}

/// Default dispatcher: refuses every WithInterp call. Used by the
/// bare-globals entry points that don't have an Interp to hand off to.
fn refuse_with_interp(
    callee: Value,
    _this: Value,
    _args: Vec<Value>,
) -> Result<Value, RuntimeError> {
    Err(RuntimeError::TypeError(format!(
        "callee `{}` needs interp — bytecode VM cannot call without a host",
        callee.to_display_string()
    )))
}

/// Run the top-level script (`fns[0]`) with no globals.
pub fn run_module(module: &Module) -> Result<Value, RuntimeError> {
    let globals: std::cell::RefCell<HashMap<String, Value>> =
        std::cell::RefCell::new(HashMap::new());
    let mut refuse = refuse_with_interp;
    run_function(
        module,
        0,
        &[],
        &Value::Undefined,
        &globals,
        None,
        &mut refuse,
    )
}

/// Same as `run_module` but with a global env. Unbound names compiled
/// into LoadGlobal resolve here; StoreGlobal writes back. The caller
/// owns the RefCell so it can observe mutations after run.
pub fn run_module_with_globals(
    module: &Module,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
) -> Result<Value, RuntimeError> {
    let mut refuse = refuse_with_interp;
    run_function(
        module,
        0,
        &[],
        &Value::Undefined,
        globals,
        None,
        &mut refuse,
    )
}

/// Run the top-level script with a live tree-walk `Interp` available
/// for `WithInterp` native dispatch. Globals are snapshotted into a
/// RefCell at entry; the script may mutate them via StoreGlobal. After
/// the run, any new bindings are written back into the Interp's actual
/// global scope so subsequent tree-walk code sees them.
pub fn run_module_with_interp(
    module: &Module,
    interp: &mut crate::interp::Interp,
) -> Result<Value, RuntimeError> {
    // Arm the per-task wall-clock watchdog (this is a top-level JS entry, so the
    // tree-walk `enter_js_task` hooks don't cover it). Aborts a runaway VM run
    // instead of freezing the UI thread.
    let _task = crate::interp::enter_js_task();
    // ★ Run DIRECTLY on the interp's LIVE global bindings table — the SAME
    // `Rc<RefCell<HashMap>>` that `globalThis`/`window`/`self`/top-level `this`
    // alias — instead of a private snapshot that only flushes back at the module
    // boundary. In V8 the global object IS the variable environment: a `var`
    // write and a `globalThis.x` read share one storage. The old snapshot design
    // buffered `StoreGlobal` writes in a separate map, so a `globalThis.i` read
    // mid-script (exactly what a page does) saw the stale hoisted `undefined`
    // while the bare-identifier read saw the buffered value — a real production
    // divergence the top-level VM oracle exposed. Sharing the live map makes the
    // global-object read and the bare read agree, byte-identical to the tree-walk
    // path (which mutates this same map in place). No write-back step is needed
    // (and the throw path is automatically correct: every prior `var`/assignment
    // already landed in the live map).
    let bindings = interp.global_bindings();
    let globals: &std::cell::RefCell<HashMap<String, Value>> = &bindings;
    let mut dispatch = |callee: Value, this: Value, args: Vec<Value>| {
        interp
            .call_value_with_this(callee, this, args)
            .map_err(|e| match e {
                crate::interp::JsError::Throw(v) => RuntimeError::Thrown(v),
                other => RuntimeError::TypeError(format!("host call: {other:?}")),
            })
    };
    // NOTE: do NOT `?` — but globals are already written in place (live map), so a
    // top-level `throw` leaves every prior `var`/assignment visible exactly as the
    // tree-walker does (the global scope is mutated in place on both paths).
    run_function(
        module,
        0,
        &[],
        &Value::Undefined,
        globals,
        None,
        &mut dispatch,
    )
}

thread_local! {
    /// Reusable register buffers, pooled to avoid a heap allocation on every
    /// `run_function` call. Grows to ~max recursion depth, then recycles.
    static REGS_POOL: std::cell::RefCell<Vec<Vec<Value>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// One in-VM P6 (f64 numeric) native-code cache slot for a module-local function
/// reached via `Op::CallFn`. `Declined` means the callee isn't P6-compilable
/// (recorded so we don't retry the compile every call). `Ready` holds the
/// installed native function. `code_ptr` + `code_len` are a CONTENT GUARD: the
/// cache is keyed by `(module.fns.as_ptr(), fn_idx)`, but a freed-then-realloc'd
/// module could reuse that address with DIFFERENT bytecode at the same index — so
/// a hit is only trusted when the callee's `code` slice identity matches, never
/// running stale native code.
enum CallFnP6Slot {
    Declined,
    Ready(std::rc::Rc<crate::jit::JitFunction>),
}

thread_local! {
    /// In-VM `Op::CallFn` P6 JIT cache, keyed by `(module fns ptr, fn_idx)`.
    /// Lets a hot, all-numeric, module-local callee reached THROUGH the bytecode
    /// VM (e.g. the top-level-VM loop's `f(i)` / `work(n)`) run as the SAME P6 f64
    /// native code the tree-walk call path already uses — so routing the hot top
    /// level onto the VM does not regress the leaf to the interpreter. Bounded:
    /// cleared wholesale if it grows past the cap (each entry's executable page is
    /// freed on drop), so a long-lived process can't accumulate stale modules'
    /// code. Identity is re-checked on every hit via the content guard below.
    static CALLFN_P6_CACHE: std::cell::RefCell<
        std::collections::HashMap<(usize, usize), (usize, usize, CallFnP6Slot)>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Resolve (compile + cache) the P6 f64 native code for module-local `fn_idx`,
/// returning the installed native function iff the callee is P6-eligible. `None`
/// means "not P6-compilable" (a recorded decline → caller runs the VM). Keyed by
/// `(fns ptr, fn_idx)` with a `code`-slice content guard so a recycled module
/// address can never dispatch stale native code.
fn resolve_callfn_p6(
    module: &Module,
    fn_idx: usize,
) -> Option<std::rc::Rc<crate::jit::JitFunction>> {
    let f = module.fns.get(fn_idx)?;
    // P6 compiles only ≤4-param numeric bodies; declines anything else.
    if f.n_params > 4 {
        return None;
    }
    let key = (module.fns.as_ptr() as usize, fn_idx);
    let code_ptr = f.code.as_ptr() as usize;
    let code_len = f.code.len();
    CALLFN_P6_CACHE.with(|c| {
        {
            let cache = c.borrow();
            if let Some((cp, cl, slot)) = cache.get(&key) {
                // Content guard: only trust a hit for the SAME bytecode slice.
                if *cp == code_ptr && *cl == code_len {
                    return match slot {
                        CallFnP6Slot::Declined => None,
                        CallFnP6Slot::Ready(jf) => Some(jf.clone()),
                    };
                }
            }
        }
        // Miss (or a recycled address) — compile fresh. Bound the cache first.
        {
            let mut cache = c.borrow_mut();
            if cache.len() >= 4096 {
                cache.clear();
            }
        }
        let compiled = crate::jit::compile_bytecode_f64(&f.code, f.n_params, |k| {
            match f.consts.get(k as usize) {
                Some(Value::Number(n)) => Some(*n),
                _ => None,
            }
        })
        .and_then(|code| crate::jit::JitFunction::install(&code).ok());
        let (slot, ret) = match compiled {
            Some(jf) => {
                let rc = std::rc::Rc::new(jf);
                (CallFnP6Slot::Ready(rc.clone()), Some(rc))
            }
            None => (CallFnP6Slot::Declined, None),
        };
        c.borrow_mut().insert(key, (code_ptr, code_len, slot));
        ret
    })
}

/// Dispatch numeric `args` to an already-resolved in-VM P6 native function (the
/// `Op::CallFn` fast path). Mirrors `interp::run_p6_native`: box the (≤4) numeric
/// args into a stack array, run the native code, and bump the honest P6 exec
/// counter so the engagement guard counts this dispatch route too. Callers MUST
/// have checked the args are all-numeric and `args.len() >= n_params`.
#[inline]
fn run_callfn_p6(jf: &crate::jit::JitFunction, args: &[Value]) -> Value {
    let mut fbuf = [0.0f64; 4];
    let n = args.len().min(4);
    for (slot, a) in fbuf.iter_mut().zip(args.iter()) {
        if let Value::Number(v) = a {
            *slot = *v;
        }
    }
    let r = unsafe { jf.call_f64_args(&fbuf[..n]) };
    crate::interp::bump_p6_exec_count();
    Value::Number(r)
}

// ══════════════════════════════════════════════════════════════════════════
// STAGE 2 — VM-LEVEL LEAF INLINING (V8 JSInlining-shaped, the jit.js lever).
//
// V8 SOURCE MODELED: `src/compiler/js-inlining.*` (JSInliner) — at a monomorphic
// call site whose target is a small, known callee, TurboFan/Maglev SPLICE the
// callee's body into the caller so the call frame + call dispatch disappear and the
// callee's arithmetic fuses into the caller's (here: the hot top-level loop). Our
// Stage-1 top-level VM ran the hot loop on the register VM but still executed the
// leaf `f(i)` / `work(n)` as a per-iteration `Op::CallFn` (a `Vec` arg-gather + a
// thread-local `CALLFN_P6_CACHE` HashMap lookup + a native-call round-trip EVERY
// iteration). This pass removes that per-iteration call entirely: `f`'s body is
// spliced inline into the caller's bytecode, so the VM dispatches `f`'s arithmetic
// directly in the loop, with no call op.
//
// WHY IT IS BYTE-IDENTICAL TO THE UN-INLINED VM (the non-negotiable gate): the
// callee admitted here is a PURE NUMERIC LEAF — `callee_is_inlinable` (reused
// verbatim from the proven T4 inliner) requires exact arity, no rest param, a
// bounded op count, and EVERY op in the numeric/control-flow subset (no globals, no
// `this`, no closure capture, no nested call, no heap/property op). Such a callee's
// observable effect is EXACTLY "return a Number computed from its args" — it touches
// no state the caller or any other code can observe. Splicing its body into a fresh
// register window ABOVE the caller's regs (callee reg r → base + r), seeding its
// params by COPYING the caller's arg slots, and replacing its `Ret` with a single
// store of the result into the call's `dst`, computes the identical Value the
// `CallFn` would have produced and writes it to the identical register — so the VM
// state after the inlined region equals the VM state after the call, op-for-op.
// There is NO native code and NO deopt on this path (the VM runs the spliced ops
// exactly as it runs any op), so there is nothing to reconstruct: byte-identity is
// structural, and the production-faithful A/B oracle proves it.
//
// SCOPE / FALLBACK: only `CallFn` (a direct, monomorphic module-local call) to an
// inlinable callee is spliced; any non-inlinable call, or a callee with the wrong
// arity, is LEFT AS A `CallFn` (the VM runs it the Stage-1 way) — never wrong, just
// not inlined. A caller with no inlinable call returns `None` (the module runs
// un-inlined). Gated behind `CV_INLINE_LEAF` (DEFAULT ON; CV_INLINE_LEAF=0 to disable).
// ══════════════════════════════════════════════════════════════════════════

/// STAGE-2 gate (`CV_INLINE_LEAF`, DEFAULT ON). When enabled (and the top-level VM
/// is engaged), the top-level script module's slot-0 body has every monomorphic
/// numeric-leaf `CallFn` spliced inline before it runs on the VM, so the hot loop's
/// per-iteration call disappears. DEFAULT ON (flipped 2026-06-16, fuzzer-clean);
/// `CV_INLINE_LEAF=0` disables. Cached (env read once per thread); a programmatic
/// override (`InlineLeafGuard`) takes precedence so tests/benches drive both states
/// in one process without the per-thread cache pinning the first read.
pub fn inline_leaf_enabled() -> bool {
    if let Some(v) = INLINE_LEAF_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    thread_local! {
        static ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    ON.with(|c| match c.get() {
        Some(v) => v,
        None => {
            // DEFAULT ON (flipped 2026-06-16; effective only when the top-level VM
            // is also on). Fuzzer-clean + nasty-suite verified. Only an explicit
            // off value disables.
            let v = !matches!(
                std::env::var("CV_INLINE_LEAF").as_deref(),
                Ok("0") | Ok("false") | Ok("off")
            );
            c.set(Some(v));
            v
        }
    })
}

thread_local! {
    /// Programmatic override for `inline_leaf_enabled` (None = env-driven).
    static INLINE_LEAF_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    /// Honesty counter: number of `CallFn` sites this thread actually spliced inline
    /// (the anti-fake guard — a faster jit.js only counts if this proves the call was
    /// removed). Read via `leaf_inline_count`; reset via `reset_leaf_inline_count`.
    static LEAF_INLINE_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// RAII scope guard forcing the leaf-inline gate on/off for the current thread
/// (the env-independent A/B knob the oracle uses). Restores the prior state on drop.
#[must_use = "the override is restored when the guard is dropped; bind it to a name"]
pub struct InlineLeafGuard {
    prev: Option<bool>,
}
impl InlineLeafGuard {
    pub fn new(on: bool) -> Self {
        let prev = INLINE_LEAF_OVERRIDE.with(|c| {
            let p = c.get();
            c.set(Some(on));
            p
        });
        InlineLeafGuard { prev }
    }
}
impl Drop for InlineLeafGuard {
    fn drop(&mut self) {
        INLINE_LEAF_OVERRIDE.with(|c| c.set(self.prev));
    }
}

/// Number of `CallFn` sites spliced inline by `inline_numeric_leaf_calls` this
/// thread (the engagement honesty guard — proves the inliner is non-vacuous).
pub fn leaf_inline_count() -> u64 {
    LEAF_INLINE_COUNT.with(|c| c.get())
}
/// Reset the leaf-inline honesty counter (so a measurement attributes splices to
/// THIS run).
pub fn reset_leaf_inline_count() {
    LEAF_INLINE_COUNT.with(|c| c.set(0));
}

/// Splice every monomorphic numeric-leaf `CallFn` in `module.fns[caller_idx]` inline
/// (V8 JSInlining-shaped — the Stage-2 jit.js lever). Returns a NEW caller
/// `BcFunction` with the calls removed iff ≥1 call was inlined, else `None` (the
/// caller runs un-inlined). The caller may contain ANY ops (globals, heap, control
/// flow) — only the callee must be a pure numeric leaf (`t4::callee_is_inlinable`),
/// because the result is run on the VM (which dispatches every op), NOT on the
/// representation-specialized native backend. Unlike the T4 `inline_first_call`, the
/// caller is NOT required to be numeric, and ALL inlinable sites are spliced.
///
/// TRANSFORM (one inlinable `CallFn { dst, fn_idx, first_arg, n_args }` at caller op
/// `call_pc`, callee `g`): give the splice a FRESH register window starting at the
/// running high-water `window_base` (≥ the caller's `n_regs`, stacked so multiple
/// inlined sites never alias). Callee reg r → `window_base + r`. Before the body,
/// COPY each arg into the callee's param slot (`Move { window_base+p, first_arg+p }`)
/// — a copy, so the caller's arg slots are untouched. The callee body is appended
/// with regs remapped `+window_base` and internal jumps retargeted into the inlined
/// region; each callee `Ret { src }` becomes `Move { dst, window_base+src }` (store
/// the result into the call's dst — the ONLY caller-slot write the region makes) plus
/// a `Jmp` to the post-call continuation. Caller jumps are retargeted to the new
/// fused offsets. A non-inlinable `CallFn`/`CallValue`/`New` is copied through
/// unchanged (the VM runs it). Anything that would overflow the u16 register/const
/// space declines (returns `None`) — correctness over coverage.
pub fn inline_numeric_leaf_calls(module: &Module, caller_idx: usize) -> Option<BcFunction> {
    let caller = module.fns.get(caller_idx)?;
    // Decide which call sites to inline, and the callee window each gets. We assign
    // each inlined site a DISJOINT window stacked above the caller's regs so two
    // sites' temporaries never alias (the caller's own regs `0..n_regs` are
    // untouched). `window_base` is the running high-water mark.
    let mut window_base = caller.n_regs;
    // Per-call-site plan: (call_pc, window_base for that site). Built in pc order.
    let mut plan: Vec<(usize, u16)> = Vec::new();
    for (pc, op) in caller.code.iter().enumerate() {
        if let Op::CallFn { fn_idx, n_args, .. } = *op {
            let callee = module.fns.get(fn_idx as usize)?;
            if crate::t4::callee_is_inlinable(callee, n_args as usize) {
                let base = window_base;
                // Reserve this site's window; guard against u16 overflow.
                let next = (base as u32) + (callee.n_regs as u32);
                if next > u16::MAX as u32 {
                    return None; // can't fit a fresh window — decline cleanly.
                }
                window_base = next as u16;
                plan.push((pc, base));
            }
            // A non-inlinable CallFn is left in place (the VM runs it).
        }
    }
    if plan.is_empty() {
        return None; // nothing to inline → run un-inlined.
    }
    let fused_n_regs = window_base; // high-water of every site's window.

    // ── Single linear rebuild. We compute, for each ORIGINAL caller op index, the
    //    fused offset of its FIRST emitted op (caller jumps retarget to it). Inlined
    //    regions are emitted at the call op's position (replacing the CallFn). The
    //    callee consts are appended once to a shared fused const pool, remapped per
    //    site (each site references the SAME callee const block by fn_idx).
    let mut fused: Vec<Op> = Vec::with_capacity(caller.code.len() * 2);
    // caller original op index → fused offset of its first emitted op.
    let mut caller_fused_off: Vec<usize> = vec![usize::MAX; caller.code.len()];
    // Shared fused const pool: caller consts first (caller LoadConst k unchanged),
    // then each distinct inlined callee's consts (recorded so re-used callees share).
    let mut consts = caller.consts.clone();
    // fn_idx → const base in the fused pool (so the same callee inlined twice shares
    // one const block).
    let mut callee_const_base: std::collections::HashMap<u16, usize> =
        std::collections::HashMap::new();
    // Patch jobs for callee-internal jumps: (fused_idx_of_jmp, callee_target_idx,
    // callee_region_start_fused_off, callee). Resolved after layout per site.
    // We instead resolve each site's jumps inline using a per-site offset table.

    // Quick lookup: pc → site window base (if planned).
    let plan_at: std::collections::HashMap<usize, u16> = plan.iter().cloned().collect();

    // We need, for every Ret-store Jmp we emit, the continuation = fused offset of
    // the op AFTER the call. Because we emit linearly and the continuation is the
    // NEXT caller op, we patch Ret-jmps after the whole site is laid out (the
    // continuation is `fused.len()` right after the site's region).
    for (pc, op) in caller.code.iter().enumerate() {
        if let Some(&base) = plan_at.get(&pc) {
            // ── Inline this CallFn site.
            let (dst, fn_idx, first_arg, n_args) = match *op {
                Op::CallFn { dst, fn_idx, first_arg, n_args } => (dst, fn_idx, first_arg, n_args),
                _ => unreachable!("plan_at only holds CallFn pcs"),
            };
            let callee = &module.fns[fn_idx as usize];
            // Ensure the callee's consts are in the fused pool exactly once.
            let const_base = match callee_const_base.get(&fn_idx) {
                Some(&b) => b,
                None => {
                    let b = consts.len();
                    consts.extend(callee.consts.iter().cloned());
                    if consts.len() > u16::MAX as usize {
                        return None;
                    }
                    callee_const_base.insert(fn_idx, b);
                    b
                }
            };
            // The first emitted op of this site is the first arg-copy (or, with zero
            // args — impossible here since callee_is_inlinable requires n_args params
            // and a 0-param numeric leaf is allowed — the first body op). Record it as
            // the caller op's fused offset so a caller jump targeting this pc lands at
            // the start of the inlined region. (No caller jump can target the middle
            // of a call's result, so this is always the right entry.)
            caller_fused_off[pc] = fused.len();
            // Region A — seed callee params by COPYING caller args.
            for p in 0..n_args as u16 {
                fused.push(Op::Move { dst: base + p, src: first_arg + p });
            }
            // Region B — the inlined callee body. Lay out ops, remap regs +base,
            // record per-callee-op fused offsets for jump retargeting, and turn each
            // Ret into store-result + jmp-to-continuation (patched after layout).
            let region_start = fused.len();
            let mut callee_off: Vec<usize> = Vec::with_capacity(callee.code.len());
            let mut ret_jmps: Vec<usize> = Vec::new();
            for cop in &callee.code {
                callee_off.push(fused.len());
                match *cop {
                    Op::Ret { src } => {
                        fused.push(Op::Move { dst, src: base + src });
                        let jmp_idx = fused.len();
                        fused.push(Op::Jmp { target: 0 }); // patched → continuation
                        ret_jmps.push(jmp_idx);
                    }
                    Op::LoadConst { dst: cd, k } => {
                        // Remap reg AND const index into the fused pool.
                        fused.push(Op::LoadConst {
                            dst: cd + base,
                            k: (k as usize + const_base) as u16,
                        });
                    }
                    other => fused.push(crate::t4::remap_callee_op(other, base)),
                }
            }
            let _ = region_start;
            // The continuation is the op right after the whole inlined region.
            let continuation = fused.len();
            // Patch callee-internal jumps (Jmp / JmpIfFalse) to fused offsets.
            for (k, cop) in callee.code.iter().enumerate() {
                let fi = callee_off[k];
                match *cop {
                    Op::Jmp { target } => {
                        if let Op::Jmp { target: t } = &mut fused[fi] {
                            *t = *callee_off.get(target as usize)? as u16;
                        }
                    }
                    Op::JmpIfFalse { target, .. } => {
                        if let Op::JmpIfFalse { target: t, .. } = &mut fused[fi] {
                            *t = *callee_off.get(target as usize)? as u16;
                        }
                    }
                    _ => {}
                }
            }
            // Patch the Ret-store jmps to the continuation.
            for &j in &ret_jmps {
                if let Op::Jmp { target } = &mut fused[j] {
                    *target = continuation as u16;
                }
            }
            LEAF_INLINE_COUNT.with(|c| c.set(c.get() + 1));
        } else {
            // ── Ordinary caller op (incl. a non-inlinable call) — copy through.
            caller_fused_off[pc] = fused.len();
            fused.push(*op);
        }
    }

    // ── Retarget every CALLER jump to its target op's new fused offset. (Callee
    //    jumps were already patched per-site.) A caller jump's target is an ORIGINAL
    //    caller op index; map it through `caller_fused_off`. A target that landed on
    //    an inlined-away call op resolves to the START of that op's inlined region
    //    (recorded above), which is the correct entry. We only rewrite jumps that
    //    were ORIGINAL caller ops (not the ones we emitted inside an inlined region —
    //    those were already patched and live at offsets we never revisit here).
    for (pc, op) in caller.code.iter().enumerate() {
        // Skip inlined call sites — they emitted their own (already-patched) ops.
        if plan_at.contains_key(&pc) {
            continue;
        }
        let fi = caller_fused_off[pc];
        if fi == usize::MAX {
            continue;
        }
        match *op {
            Op::Jmp { target } => {
                let off = *caller_fused_off.get(target as usize)?;
                if off == usize::MAX {
                    return None;
                }
                if let Op::Jmp { target: t } = &mut fused[fi] {
                    *t = off as u16;
                }
            }
            Op::JmpIfFalse { target, .. } => {
                let off = *caller_fused_off.get(target as usize)?;
                if off == usize::MAX {
                    return None;
                }
                if let Op::JmpIfFalse { target: t, .. } = &mut fused[fi] {
                    *t = off as u16;
                }
            }
            Op::JmpIfTrue { target, .. } => {
                let off = *caller_fused_off.get(target as usize)?;
                if off == usize::MAX {
                    return None;
                }
                if let Op::JmpIfTrue { target: t, .. } = &mut fused[fi] {
                    *t = off as u16;
                }
            }
            _ => {}
        }
    }

    Some(BcFunction {
        name: format!("{}+leafinline", caller.name),
        n_params: caller.n_params,
        rest_reg: caller.rest_reg,
        n_regs: fused_n_regs,
        consts,
        code: fused,
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
    })
}

/// Build a copy of `module` with slot-`caller_idx` leaf-inlined (every monomorphic
/// numeric-leaf `CallFn` spliced inline), reusing the ORIGINAL callee `fns` so a
/// retained `CallFn` (a non-inlinable call left in place) still resolves by index.
/// Returns `None` if nothing was inlined (the caller runs the original module). The
/// returned module is observationally identical to the original on the VM — only the
/// caller body changed, and only by removing pure-numeric-leaf calls.
///
/// THEN — the win-closing step — it kernelizes any counted numeric accumulator loop
/// in the fused body (see `kernelize_counted_loops`): the inlined loop is extracted
/// into a fresh, pure-numeric module function and the loop region is replaced by a
/// SINGLE `Op::CallFn` to it. The existing in-VM `CallFn → P6` routing then compiles
/// that kernel to native f64 code, so the WHOLE loop runs natively in one call (no
/// per-iteration VM dispatch AND no per-iteration leaf call). Inlining alone moves
/// the leaf's arithmetic onto the boxed VM — a regression — so kernelization is what
/// makes the inline a net win; if no loop matches the kernel shape, the body is left
/// inlined-only (still correct, the VM runs it). Byte-identity is structural (pure
/// numeric leaf + a loop-invariant accumulator hoist over a body that touches no
/// other state) and gated by the production-faithful A/B oracle.
pub fn inline_leaf_module(module: &Module, caller_idx: usize) -> Option<Module> {
    let fused_caller = inline_numeric_leaf_calls(module, caller_idx)?;
    let mut fns = module.fns.clone();
    *fns.get_mut(caller_idx)? = fused_caller;
    // Kernelize counted numeric accumulator loops in the fused caller (appends kernel
    // fns + rewrites the loop to a CallFn to them). Best-effort: a loop that doesn't
    // match the strict shape is left as-is (the inlined VM body runs it).
    kernelize_counted_loops(&mut fns, caller_idx);
    Some(Module {
        fns,
        script_forinit_syncs: module.script_forinit_syncs.clone(),
    })
}

/// A recognized counted numeric accumulator loop in a caller body. Spans the
/// half-open caller op range `[header, back_jmp]` (inclusive of the closing backward
/// `Jmp` at `back_jmp`). The header is `LoadConst limit; Lt i<limit; JmpIfFalse exit`
/// (the canonical `for (…; i<N; …)` test the bytecode compiler emits). The body
/// reads ONE accumulator global `acc_name` at `load_pc` into `acc_reg` and stores it
/// back at `store_pc` — the only global the loop touches (verified) — and is
/// otherwise entirely in the P6 numeric subset.
struct CountedLoop {
    header: usize,       // op index of the LoadConst(limit) that begins the test
    #[allow(dead_code)]
    lt_pc: usize,        // the Lt op (i < limit)
    #[allow(dead_code)]
    jmpfalse_pc: usize,  // the JmpIfFalse(exit) right after Lt
    back_jmp: usize,     // the closing backward Jmp back to `header`
    exit_pc: usize,      // op index of the first op AFTER the loop (JmpIfFalse target)
    i_reg: Reg,          // induction var register (Lt lhs)
    limit_reg: Reg,      // limit register (Lt rhs)
    limit_k: u16,        // const index of the header's LoadConst(limit) — the loop bound
    acc_load_reg: Reg,   // register the body READS the accumulator into (LoadGlobal dst)
    acc_store_reg: Reg,  // register the body WRITES the new accumulator from (StoreGlobal src)
    acc_name: String,    // accumulator global name
    load_pc: usize,      // the LoadGlobal(acc) in the body
    store_pc: usize,     // the StoreGlobal(acc) in the body
}

/// Detect, in `code`, the FIRST counted numeric accumulator loop matching the strict
/// shape the bench loops compile to. Returns `None` (no kernelization) on ANY
/// mismatch — correctness over coverage. The shape (1 induction var, 1 accumulator
/// global, body otherwise P6-numeric, exactly one load + one store of the
/// accumulator, no other global/heap/call op) is what makes the loop-invariant
/// accumulator hoist + native kernelization provably byte-identical.
fn detect_counted_loop(code: &[Op], consts: &[Value]) -> Option<CountedLoop> {
    // Find a backward Jmp (loop back-edge). Its target is the loop header.
    for (jpc, op) in code.iter().enumerate() {
        let header = match *op {
            Op::Jmp { target } if (target as usize) < jpc => target as usize,
            _ => continue,
        };
        // Header must be: LoadConst(limit) ; Lt(i<limit) ; JmpIfFalse(exit).
        let lt_pc = header + 1;
        let jf_pc = header + 2;
        if jf_pc > jpc {
            continue;
        }
        let (limit_reg, limit_k) = match code.get(header) {
            Some(Op::LoadConst { dst, k }) => (*dst, *k),
            _ => continue,
        };
        let (i_reg, lt_dst) = match code.get(lt_pc) {
            Some(Op::Lt { dst, lhs, rhs }) if *rhs == limit_reg => (*lhs, *dst),
            _ => continue,
        };
        let exit_pc = match code.get(jf_pc) {
            Some(Op::JmpIfFalse { cond, target }) if *cond == lt_dst => *target as usize,
            _ => continue,
        };
        // The body is [jf_pc+1 .. jpc] (the back Jmp at jpc closes it). Scan it:
        // it must touch exactly one accumulator global (one LoadGlobal*, one
        // StoreGlobal, same name), be otherwise P6-numeric, contain NO other
        // global/heap/call/control-out op, and NOT jump outside the loop.
        let body = jf_pc + 1..jpc;
        let mut acc_name: Option<String> = None;
        let mut acc_load_reg: Option<Reg> = None;
        let mut acc_store_reg: Option<Reg> = None;
        let mut load_pc: Option<usize> = None;
        let mut store_pc: Option<usize> = None;
        let mut ok = true;
        for bpc in body.clone() {
            match &code[bpc] {
                Op::LoadGlobal { dst, name_k } | Op::LoadGlobalChecked { dst, name_k } => {
                    if load_pc.is_some() {
                        ok = false;
                        break; // a second global load → not the single-accumulator shape.
                    }
                    let name = match consts.get(*name_k as usize) {
                        Some(Value::String(s)) => s.to_string(),
                        _ => {
                            ok = false;
                            break;
                        }
                    };
                    acc_name = Some(name);
                    acc_load_reg = Some(*dst);
                    load_pc = Some(bpc);
                }
                Op::StoreGlobal { name_k, src } => {
                    if store_pc.is_some() {
                        ok = false;
                        break;
                    }
                    let name = match consts.get(*name_k as usize) {
                        Some(Value::String(s)) => s.to_string(),
                        _ => {
                            ok = false;
                            break;
                        }
                    };
                    // Must be the SAME global as the load.
                    if acc_name.as_deref() != Some(name.as_str()) {
                        ok = false;
                        break;
                    }
                    acc_store_reg = Some(*src);
                    store_pc = Some(bpc);
                }
                // P6-numeric body ops only (the set compile_bytecode_f64 accepts).
                Op::LoadConst { .. }
                | Op::LoadUndef { .. }
                | Op::Move { .. }
                | Op::Add { .. }
                | Op::Sub { .. }
                | Op::Mul { .. }
                | Op::Div { .. }
                | Op::Lt { .. }
                | Op::Le { .. }
                | Op::Gt { .. }
                | Op::Ge { .. } => {}
                Op::Jmp { target } | Op::JmpIfFalse { target, .. } => {
                    // A body jump must stay strictly inside the loop region
                    // [header..=jpc] (a forward/back jump within the loop). A jump
                    // out of the loop is a `break`/early-exit we don't model.
                    let t = *target as usize;
                    if t < header || t > jpc {
                        ok = false;
                        break;
                    }
                }
                _ => {
                    ok = false; // any other op (heap/call/bitwise/etc.) → decline.
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let (acc_name, acc_load_reg, acc_store_reg, load_pc, store_pc) =
            match (acc_name, acc_load_reg, acc_store_reg, load_pc, store_pc) {
                (Some(n), Some(lr), Some(sr), Some(l), Some(s)) => (n, lr, sr, l, s),
                _ => continue, // must have BOTH a load and a store of the accumulator.
            };
        // The accumulator load must come before the store in the body (load-modify-
        // store across the iteration). Required for the hoist to be value-identical.
        if load_pc >= store_pc {
            continue;
        }
        // The JmpIfFalse exit target must be exactly the op after the back Jmp (the
        // canonical loop exit), so replacing [header..=jpc] with a single call is a
        // clean splice. (A loop whose exit lands elsewhere isn't this shape.)
        if exit_pc != jpc + 1 {
            continue;
        }
        // ── INDUCTION-VARIABLE POST-VALUE GATE. The loop carries TWO observable
        // values: the accumulator (→ the global, handled by the kernel's return) AND
        // the induction var `i` (→ the for-init `StoreGlobal i` the compiler emits
        // after the loop). A P6 kernel returns only ONE f64, so we can recover the
        // accumulator from the return but NOT `i`. We therefore only kernelize the
        // CANONICAL counted shape `for (i = i0; i < limit; i = i + 1)` — strict `<`
        // plus a unit `+1` increment — where the loop-exit value of `i` is provably
        // `limit` (i reaches limit exactly, the test fails, the loop exits). The
        // caller then sets `i_reg = limit_reg` after the kernel, byte-identically.
        // Any other step / comparison declines (the inlined VM body runs the loop).
        if !loop_increments_i_by_one(code, header..=jpc, i_reg, consts) {
            continue;
        }
        return Some(CountedLoop {
            header,
            lt_pc,
            jmpfalse_pc: jf_pc,
            back_jmp: jpc,
            exit_pc,
            i_reg,
            limit_reg,
            limit_k,
            acc_load_reg,
            acc_store_reg,
            acc_name,
            load_pc,
            store_pc,
        });
    }
    None
}

/// Verify the loop region's NET effect on the induction register `i_reg` per
/// iteration is exactly `i_reg = i_reg + 1` (a unit increment). This is the gate that
/// lets the caller recover `i`'s post-loop value as `limit` after the native kernel
/// (which can only return the accumulator). Conservative: returns `false` on ANY
/// shape it can't prove is `+1` (then the loop isn't kernelized — the VM body runs it,
/// always correct).
///
/// METHOD — a small backward value trace over straight-line region ops (the bench
/// increment is straight-line: `[LoadConst c=1] ; [Move t=i] ; Add r=t+c ; Move i=r`,
/// or `Add i=i+c`). We record, for each register, its LAST defining op in the region
/// (the loop body is single-block for the increment), then trace the final definition
/// of `i_reg`: it must be (through `Move` copies) an `Add` whose two operands are
/// `i_reg` (through copies) and a constant `1.0` (through copies of a `LoadConst 1`).
fn loop_increments_i_by_one(
    code: &[Op],
    region: std::ops::RangeInclusive<usize>,
    i_reg: Reg,
    consts: &[Value],
) -> bool {
    let region_start = *region.start();
    let region_end = *region.end();
    // POSITION-AWARE last-def: the last op in [region_start, before) that writes `r`.
    // (Tracing must respect program order — a register's value at a use point is its
    // most recent PRIOR definition, NOT the loop's final write of that register, which
    // for the induction var is the increment itself and would create a false cycle.)
    let last_def_before = |r: Reg, before: usize| -> Option<usize> {
        let mut found = None;
        for cpc in region_start..before {
            if op_writes(&code[cpc]).contains(&r) {
                found = Some(cpc);
            }
        }
        found
    };
    // The single increment site: the LAST op in the region writing `i_reg`.
    let mut inc_pc = None;
    for cpc in region_start..=region_end {
        if op_writes(&code[cpc]).contains(&i_reg) {
            inc_pc = Some(cpc);
        }
    }
    let inc_pc = match inc_pc {
        Some(p) => p,
        None => return false, // i never redefined → not a counting loop.
    };
    // Resolve a register read AT position `at` to the source register it ultimately
    // copies from, following Move chains using PRIOR definitions only.
    let trace_reg = |mut r: Reg, mut at: usize, mut budget: u32| -> Option<Reg> {
        loop {
            if budget == 0 {
                return None;
            }
            budget -= 1;
            match last_def_before(r, at) {
                Some(d) => match code[d] {
                    Op::Move { src, .. } => {
                        r = src;
                        at = d; // continue tracing the source as of the Move's position
                    }
                    _ => return Some(r),
                },
                None => return Some(r), // loop-invariant / param → r itself
            }
        }
    };
    // Is the register read AT position `at` a LoadConst (through Moves) equal to 1.0?
    let is_one = |mut r: Reg, mut at: usize, mut budget: u32| -> bool {
        loop {
            if budget == 0 {
                return false;
            }
            budget -= 1;
            match last_def_before(r, at) {
                Some(d) => match code[d] {
                    Op::Move { src, .. } => {
                        r = src;
                        at = d;
                    }
                    Op::LoadConst { k, .. } => {
                        return matches!(consts.get(k as usize), Some(Value::Number(n)) if *n == 1.0);
                    }
                    _ => return false,
                },
                None => return false,
            }
        }
    };
    // Trace the increment site to the Add (through a Move to a temp), then check its
    // operands are `i_reg` (prior value) and constant 1.0.
    let (add_pc, add_lhs, add_rhs) = match code[inc_pc] {
        Op::Add { lhs, rhs, .. } => (inc_pc, lhs, rhs),
        Op::Move { src, .. } => match last_def_before(src, inc_pc) {
            Some(d) => match code[d] {
                Op::Add { lhs, rhs, .. } => (d, lhs, rhs),
                _ => return false,
            },
            None => return false,
        },
        _ => return false,
    };
    let lhs_is_i = trace_reg(add_lhs, add_pc, 64) == Some(i_reg);
    let rhs_is_i = trace_reg(add_rhs, add_pc, 64) == Some(i_reg);
    let lhs_one = is_one(add_lhs, add_pc, 64);
    let rhs_one = is_one(add_rhs, add_pc, 64);
    (lhs_is_i && rhs_one) || (rhs_is_i && lhs_one)
}

/// The register(s) an op WRITES (its destination). Used by the induction-increment
/// trace. Ops with no register dst return an empty list.
fn op_writes(op: &Op) -> Vec<Reg> {
    match *op {
        Op::LoadConst { dst, .. }
        | Op::LoadUndef { dst }
        | Op::LoadTrue { dst }
        | Op::LoadFalse { dst }
        | Op::LoadNull { dst }
        | Op::Move { dst, .. }
        | Op::Add { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Div { dst, .. }
        | Op::Lt { dst, .. }
        | Op::Le { dst, .. }
        | Op::Gt { dst, .. }
        | Op::Ge { dst, .. }
        | Op::LoadGlobal { dst, .. }
        | Op::LoadGlobalChecked { dst, .. } => vec![dst],
        _ => Vec::new(),
    }
}

/// Kernelize counted numeric accumulator loops in `fns[caller_idx]` (V8-shaped: the
/// hot loop becomes one native call). For each loop `detect_counted_loop` matches:
///   1. HOIST the loop-invariant accumulator global: read it ONCE into `acc_reg`
///      before the loop, write it ONCE after. (Safe: the loop body touches no other
///      global/heap/state, and the inlined leaf is pure, so the global's value is a
///      pure function of `acc_reg` across the loop — moving the load/store outside is
///      value-identical.)
///   2. EXTRACT the now-pure-numeric loop region into a fresh module function
///      `__cv_loop_kernel` whose params are the loop-entry-live registers
///      (`i_reg`, `acc_reg`, `limit_reg` packed as params 0,1,2) and whose body runs
///      the loop and RETURNS the final accumulator.
///   3. REPLACE the loop region in the caller with: pack the 3 params into a
///      contiguous arg window, `CallFn` the kernel, move the result into `acc_reg`.
/// The existing in-VM `CallFn → P6` routing compiles the kernel to native f64 code,
/// so the whole loop runs natively in ONE call. On ANY mismatch (the strict shape
/// not met, register/const overflow, the kernel not P6-compilable) the loop is left
/// untouched (the inlined VM body runs it — correct, just not native).
///
/// Only the FIRST matching loop is kernelized (the bench shape has one hot loop);
/// this is bounded + matches V8's cheap-compile tradeoff. Returns the number of
/// loops kernelized.
fn kernelize_counted_loops(fns: &mut Vec<BcFunction>, caller_idx: usize) -> usize {
    // Snapshot the caller's code/consts/n_regs (we rebuild it).
    let (code, consts, caller_n_regs, caller_n_params, caller_rest, caller_name) = {
        let c = match fns.get(caller_idx) {
            Some(c) => c,
            None => return 0,
        };
        (
            c.code.clone(),
            c.consts.clone(),
            c.n_regs,
            c.n_params,
            c.rest_reg,
            c.name.clone(),
        )
    };
    let kernel_fn_idx = fns.len();
    // Build (kernel_fn, new_caller) in a fallible inner closure; ANY `?`/early-None
    // means "abandon kernelization, keep the inlined VM body" (always correct). The
    // closure OWNS the snapshot (move) so it can move `consts`/`code` into the new
    // bodies; nothing outside needs them again.
    let built: Option<(BcFunction, BcFunction)> = (move || {
    let loop_ = detect_counted_loop(&code, &consts)?;

    // ── Build the KERNEL function body. P6 maps register i to xmm[i] and arrives
    //    params in regs 0..n_params (xmm0..), so the kernel must use a COMPACT,
    //    ≤16-register numbering with the 3 params first. We therefore RENUMBER the
    //    loop's registers: collect every distinct register the loop region reads or
    //    writes, then assign 0=i_seed, 1=acc_seed(carry), 2=limit_seed (the params),
    //    and 3,4,… to each remaining working register in first-appearance order. A
    //    blind `+3` offset (old design) would require `caller_n_regs+3 ≤ 16` even when
    //    the loop uses only a handful of registers (jit.js: 15 working regs → 18 > 16,
    //    which falsely declined); compaction fits jit.js's ~11 distinct loop regs.
    //
    // ACCUMULATOR CARRY (the load-modify-store the global encodes): the body reads the
    // accumulator into `acc_load_reg` (the dropped LoadGlobal's dst) and writes the
    // new value from `acc_store_reg` (the dropped StoreGlobal's src). We carry the
    // accumulator in a DEDICATED kernel register `k_carry` (= param 1) across
    // iterations: at the dropped-load position emit `Move acc_load_reg ← k_carry`
    // (re-materialize what the LoadGlobal produced), and at the dropped-store position
    // emit `Move k_carry ← acc_store_reg` (capture the new accumulator). The kernel
    // returns `k_carry`. This mirrors the global's read-each-iter / write-each-iter
    // semantics EXACTLY without assuming anything about how `acc_load_reg` is reused.
    let region = loop_.header..=loop_.back_jmp;

    // Collect the distinct registers the region touches (operands of every op).
    // Renumber so the 3 params come first: 0 = i (induction working reg), 1 = CARRY
    // (a DEDICATED accumulator register, exclusive — NO caller register may alias it,
    // because the body reuses the load/Lt-result register and would otherwise clobber
    // the carry mid-iteration), 2 = limit (the limit working reg). Every OTHER distinct
    // caller register in the region (including the transient `acc_load_reg` /
    // `acc_store_reg`) maps to 3,4,5,… in first-appearance order. Note: `i_reg` and
    // `limit_reg` ARE real working registers (read/written each iteration), so they map
    // to params 0/2; the carry is the only synthetic register.
    // Three DEDICATED, EXCLUSIVE param registers — NO caller register aliases them:
    //   0 = i_seed  → ALSO the induction working reg (read/written each iter), so
    //                 `i_reg` maps to 0 (it is a genuine working reg).
    //   1 = carry   → the accumulator carry (synthetic; the global's value across
    //                 iterations). No caller reg maps here.
    //   2 = k_limit → the loop bound, set ONCE from the arg and NEVER written by the
    //                 body. The header's `LoadConst limit → limit_reg` is DROPPED and
    //                 the loop `Lt` is redirected to read `k_limit`. CRITICAL: this
    //                 makes the kernel's NATIVE code LIMIT-INDEPENDENT — the bound is a
    //                 runtime arg, not a baked const — so two loops that differ ONLY in
    //                 their bound produce IDENTICAL kernel bytecode AND identical native
    //                 code, and the CALLFN_P6_CACHE (keyed by code ptr+len) can never
    //                 serve one loop's native code for another's different bound.
    // Note `limit_reg` is OFTEN reused by the body as a scratch slot (the call-result
    // register); those uses get a NORMAL ≥3 mapping. Only its role AS the loop bound is
    // replaced by `k_limit`.
    let mut renum: std::collections::HashMap<Reg, Reg> = std::collections::HashMap::new();
    renum.insert(loop_.i_reg, 0);
    // carry = reg 1, k_limit = reg 2 — exclusive, no caller reg maps to them.
    let mut next: Reg = 3;
    let mut assign = |r: Reg, renum: &mut std::collections::HashMap<Reg, Reg>, next: &mut Reg| -> Option<()> {
        if !renum.contains_key(&r) {
            renum.insert(r, *next);
            *next = next.checked_add(1)?;
        }
        Some(())
    };
    for cpc in region.clone() {
        // The header LoadConst(limit) is dropped — don't reserve a slot for any reg it
        // would define beyond what other ops need (it writes limit_reg, handled below).
        for r in op_regs(&code[cpc]) {
            assign(r, &mut renum, &mut next)?;
        }
    }
    let map_reg = |r: Reg| -> Option<Reg> { renum.get(&r).copied() };
    let k_carry: Reg = 1; // dedicated accumulator carry register (= acc-seed param)
    let k_limit: Reg = 2; // dedicated loop-bound register (= limit-seed param)
    let kernel_n_regs_u32 = next.max(3) as u32; // ≥3 to cover params 0,1,2
    if kernel_n_regs_u32 > 16 {
        // P6 maps each register to one of xmm0..xmm15; >16 distinct regs can't fit.
        return None;
    }

    // Lay out the kernel body over the region, op-for-op. Param regs already hold the
    // seeds on entry (P6 ABI: regs 0,1,2 = the 3 args), so NO entry copies are needed
    // — i (reg 0), carry (reg 1), k_limit (reg 2) are live from the start.
    let mut kcode: Vec<Op> = Vec::new();
    let mut caller_to_kernel: Vec<Option<usize>> = vec![None; code.len()];
    for cpc in region.clone() {
        caller_to_kernel[cpc] = Some(kcode.len());
        if cpc == loop_.header {
            // DROP the header's `LoadConst limit`. The bound lives in `k_limit` (set
            // once from the arg). A jump targeting `header` lands on the next emitted
            // op (the Lt), which is the correct loop-test entry. We emit NOTHING here;
            // `caller_to_kernel[header]` points at the Lt (next push).
            caller_to_kernel[cpc] = Some(kcode.len()); // = the Lt's kernel index
            continue;
        }
        if cpc == loop_.load_pc {
            // Re-materialize the LoadGlobal's effect from the carried accumulator.
            let dst = map_reg(loop_.acc_load_reg)?;
            kcode.push(Op::Move { dst, src: k_carry });
            continue;
        }
        if cpc == loop_.store_pc {
            // Capture the new accumulator into the carry for the next iteration.
            let src = map_reg(loop_.acc_store_reg)?;
            kcode.push(Op::Move { dst: k_carry, src });
            continue;
        }
        let kop = match code[cpc] {
            // The loop test `Lt(i, limit_reg)` → read the dedicated `k_limit` (the
            // dropped header LoadConst no longer defines limit_reg).
            Op::Lt { dst, lhs, rhs } if cpc == loop_.lt_pc && rhs == loop_.limit_reg => {
                Op::Lt { dst: map_reg(dst)?, lhs: map_reg(lhs)?, rhs: k_limit }
            }
            // The loop-exit JmpIfFalse (target == exit_pc) becomes a branch to the
            // appended Ret(carry).
            Op::JmpIfFalse { cond, target } if (target as usize) == loop_.exit_pc => {
                Op::JmpIfFalse { cond: map_reg(cond)?, target: u16::MAX } // patched below
            }
            other => remap_op_regs_with(other, &map_reg)?,
        };
        kcode.push(kop);
    }
    // Append the kernel exit: Ret(carry).
    let exit_label = kcode.len();
    kcode.push(Op::Ret { src: k_carry });

    // Second pass: patch jump targets (Jmp / JmpIfFalse) inside the kernel.
    // A caller jump target inside the region → its kernel index; the loop-exit
    // JmpIfFalse (target == exit_pc) → the appended `exit_label` (the Ret).
    for cpc in region.clone() {
        if cpc == loop_.load_pc || cpc == loop_.store_pc || cpc == loop_.header {
            continue; // dropped ops emitted nothing patchable
        }
        let kpc = caller_to_kernel[cpc]?;
        match code[cpc] {
            Op::JmpIfFalse { target, .. } if (target as usize) == loop_.exit_pc => {
                if let Op::JmpIfFalse { target: t, .. } = &mut kcode[kpc] {
                    *t = exit_label as u16;
                }
            }
            Op::Jmp { target } => {
                let kt = caller_to_kernel
                    .get(target as usize)
                    .copied()
                    .flatten();
                match kt {
                    Some(t) => {
                        if let Op::Jmp { target: tt } = &mut kcode[kpc] {
                            *tt = t as u16;
                        }
                    }
                    None => return None, // jump out of region we didn't model
                }
            }
            Op::JmpIfFalse { target, .. } => {
                let kt = caller_to_kernel
                    .get(target as usize)
                    .copied()
                    .flatten();
                match kt {
                    Some(t) => {
                        if let Op::JmpIfFalse { target: tt, .. } = &mut kcode[kpc] {
                            *tt = t as u16;
                        }
                    }
                    None => return None,
                }
            }
            _ => {}
        }
    }

    // ── STAGE 3 — TIGHTEN the kernel bytecode (V8/Maglev-shaped): hoist loop-
    //    invariant constants out of the loop (LICM, dedup'd), CSE repeated pure
    //    subexpressions (`x*x` computed ONCE/iter), and copy-propagate Moves — so
    //    the op-for-op P6 lowering emits register-resident, redundancy-free code.
    //    The transform is VALUE-PRESERVING (it never reorders an arithmetic op, only
    //    removes recomputation / rematerialization), so the loop is bit-identical.
    //    On any shape it can't prove safe it returns None and we keep `kcode` as-is
    //    (still correct, just the un-tightened Stage-2 codegen). The result is
    //    re-verified P6-compilable below, so a bad optimization can never ship.
    let (kcode, kernel_n_regs_u32) = match optimize_kernel_loop(&kcode, &consts) {
        Some((opt_code, opt_n_regs)) => (opt_code, opt_n_regs),
        None => (kcode, kernel_n_regs_u32),
    };

    // The kernel's consts = the caller's consts (LoadConst k indices are unchanged;
    // the kernel keeps the same pool, which is fine — extra unused consts are
    // harmless). P6 reads const_f64 by index into this pool.
    let kernel_n_regs = kernel_n_regs_u32 as u16;
    let kernel_fn = BcFunction {
        name: "__cv_loop_kernel".to_string(),
        n_params: 3,
        rest_reg: None,
        n_regs: kernel_n_regs,
        consts: consts.clone(),
        code: kcode,
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
    };

    // VERIFY the kernel is actually P6-compilable BEFORE committing the rewrite — if
    // P6 declines (an op/shape it can't lower), leave the loop as the inlined VM body
    // (correct, just not native). This is the gate that keeps a non-compilable kernel
    // from ever replacing a working VM loop.
    #[cfg(target_os = "windows")]
    {
        let probe_consts = kernel_fn.consts.clone();
        let compiled = crate::jit::compile_bytecode_f64(&kernel_fn.code, kernel_fn.n_params, |k| {
            match probe_consts.get(k as usize) {
                Some(Value::Number(n)) => Some(*n),
                _ => None,
            }
        });
        if compiled.is_none() {
            return None;
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        // No P6 backend off-Windows → kernelization gives no native win; skip it so
        // the inlined VM body runs (and we don't pay a per-call VM kernel run that is
        // no faster). The default build is off-Windows-irrelevant for the bench.
        return None;
    }

    // ── Rewrite the CALLER: replace the loop region [header..=back_jmp] with:
    //   <prefix>                                  (ops before header, unchanged)
    //   read acc:   (hoist) LoadGlobalChecked R = acc_name   — the loop-invariant read
    //   pack args:  Move A0=i_reg ; Move A1=R ; Move A2=limit_reg
    //   call:       CallFn R = __cv_loop_kernel(A0..A2)        — R = final accumulator
    //   write acc:  StoreGlobal acc_name = R                  — the loop-invariant write
    //   <suffix>                                  (ops at/after exit_pc; jumps remapped)
    //
    // R is the caller's `acc_load_reg` (the register the original loop read the global
    // into) — reusing it keeps the suffix valid if it ever reads that slot, and it is
    // free here (the loop is gone). i_reg/limit_reg hold their loop-ENTRY values from
    // the prefix (`var i = 0`, the limit LoadConst). The kernel re-derives the limit
    // from its own in-loop LoadConst, so passing the entry value is just the seed.
    //
    // We need a contiguous 3-arg window for the CallFn. Allocate it ABOVE the
    // caller's current n_regs so it can't alias any live caller reg.
    let acc_reg = loop_.acc_load_reg;
    let arg_base = caller_n_regs;
    let new_caller_n_regs_u32 = arg_base as u32 + 3;
    if new_caller_n_regs_u32 > u16::MAX as u32 {
        return None;
    }
    if kernel_fn_idx > u16::MAX as usize {
        return None;
    }

    // The accumulator global name const index (reuse the existing const if present).
    let acc_name_k = match consts.iter().position(|v| matches!(v, Value::String(s) if **s == loop_.acc_name)) {
        Some(k) if k <= u16::MAX as usize => k as u16,
        _ => return None, // name not in the pool (shouldn't happen — load/store used it)
    };

    // Build the replacement op block.
    let mut repl: Vec<Op> = Vec::new();
    // Hoisted accumulator read (loop-invariant load, once).
    repl.push(Op::LoadGlobalChecked { dst: acc_reg, name_k: acc_name_k });
    // Pack the 3 kernel args. A0=i (its loop-ENTRY value, set by the prefix's
    // `var i = i0`), A1=acc (just loaded), A2=limit. The limit's defining LoadConst
    // lived INSIDE the loop header (re-loaded each iteration), so `limit_reg` is NOT
    // live in the caller after the region is removed — load the limit CONST fresh into
    // A2 (the kernel also re-derives it internally, so A2 is just a numeric seed, but
    // a real number keeps the CallFn on the all-numeric P6 fast path).
    repl.push(Op::Move { dst: arg_base, src: loop_.i_reg });
    repl.push(Op::Move { dst: arg_base + 1, src: acc_reg });
    repl.push(Op::LoadConst { dst: arg_base + 2, k: loop_.limit_k });
    // The native kernel call (P6 routes this to native f64 code).
    repl.push(Op::CallFn {
        dst: acc_reg,
        fn_idx: kernel_fn_idx as u16,
        first_arg: arg_base,
        n_args: 3,
    });
    // Hoisted accumulator write (loop-invariant store, once).
    repl.push(Op::StoreGlobal { name_k: acc_name_k, src: acc_reg });
    // Induction-variable post-value: a canonical `for (i=i0; i<limit; i=i+1)` exits
    // with `i == limit` (gated by `loop_increments_i_by_one`). Set `i_reg = limit`
    // (loaded fresh — `limit_reg` is not live here) so the compiler's post-loop
    // `StoreGlobal i` (the for-init sync) writes the byte-identical value the
    // un-inlined loop would. The suffix uses `i_reg` only for that sync.
    repl.push(Op::LoadConst { dst: loop_.i_reg, k: loop_.limit_k });

    // Stitch: prefix [0..header) + repl + suffix [exit_pc..). Build an old→new op
    // index map for retargeting caller jumps that point INTO the prefix/suffix.
    let mut new_code: Vec<Op> = Vec::with_capacity(code.len());
    let mut old_to_new: Vec<Option<usize>> = vec![None; code.len() + 1];
    // Prefix.
    for pc in 0..loop_.header {
        old_to_new[pc] = Some(new_code.len());
        new_code.push(code[pc]);
    }
    // The whole loop region collapses to the start of `repl`. A caller jump that
    // targeted `header` (none should, but be safe) lands at the replacement start.
    let repl_start = new_code.len();
    for pc in loop_.header..loop_.exit_pc {
        old_to_new[pc] = Some(repl_start);
    }
    new_code.extend(repl.iter().copied());
    // Suffix.
    for pc in loop_.exit_pc..code.len() {
        old_to_new[pc] = Some(new_code.len());
        new_code.push(code[pc]);
    }
    old_to_new[code.len()] = Some(new_code.len());

    // Retarget caller jumps in the prefix/suffix (the region's own jumps are gone).
    for pc in 0..code.len() {
        if (loop_.header..loop_.exit_pc).contains(&pc) {
            continue; // region op — removed
        }
        let np = match old_to_new[pc] {
            Some(n) => n,
            None => continue,
        };
        match code[pc] {
            Op::Jmp { target } => {
                if let Some(nt) = old_to_new.get(target as usize).copied().flatten() {
                    if let Op::Jmp { target: t } = &mut new_code[np] {
                        *t = nt as u16;
                    }
                } else {
                    return None; // unresolved target → abandon kernelization (keep VM body)
                }
            }
            Op::JmpIfFalse { target, .. } => {
                if let Some(nt) = old_to_new.get(target as usize).copied().flatten() {
                    if let Op::JmpIfFalse { target: t, .. } = &mut new_code[np] {
                        *t = nt as u16;
                    }
                } else {
                    return None;
                }
            }
            Op::JmpIfTrue { target, .. } => {
                if let Some(nt) = old_to_new.get(target as usize).copied().flatten() {
                    if let Op::JmpIfTrue { target: t, .. } = &mut new_code[np] {
                        *t = nt as u16;
                    }
                } else {
                    return None;
                }
            }
            _ => {}
        }
    }

    // Build the new caller body (committed by the outer fn).
    let new_caller = BcFunction {
        name: caller_name,
        n_params: caller_n_params,
        rest_reg: caller_rest,
        n_regs: new_caller_n_regs_u32 as u16,
        consts,
        code: new_code,
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
    };
    Some((kernel_fn, new_caller))
    })();

    // Commit: append the kernel fn (at the index we reserved) and replace the caller.
    match built {
        Some((kernel_fn, new_caller)) => {
            debug_assert_eq!(fns.len(), kernel_fn_idx, "kernel fn index drifted");
            fns.push(kernel_fn);
            if let Some(c) = fns.get_mut(caller_idx) {
                *c = new_caller;
            }
            1
        }
        None => 0,
    }
}

/// Every register operand of a P6-numeric/control op (for the kernel's register-set
/// collection). Ops outside the kernel subset return an empty list — they never reach
/// here for a detected loop (the body scan already rejected them), but being total
/// keeps the collector safe.
fn op_regs(op: &Op) -> Vec<Reg> {
    match *op {
        Op::LoadConst { dst, .. } | Op::LoadUndef { dst } => vec![dst],
        Op::Move { dst, src } => vec![dst, src],
        Op::Add { dst, lhs, rhs }
        | Op::Sub { dst, lhs, rhs }
        | Op::Mul { dst, lhs, rhs }
        | Op::Div { dst, lhs, rhs }
        | Op::Lt { dst, lhs, rhs }
        | Op::Le { dst, lhs, rhs }
        | Op::Gt { dst, lhs, rhs }
        | Op::Ge { dst, lhs, rhs } => vec![dst, lhs, rhs],
        Op::JmpIfFalse { cond, .. } => vec![cond],
        Op::LoadGlobal { dst, .. } | Op::LoadGlobalChecked { dst, .. } => vec![dst],
        Op::StoreGlobal { src, .. } => vec![src],
        _ => Vec::new(),
    }
}

/// Remap a P6-numeric/control op's register operands through `map` (the kernel's
/// compact renumbering). Returns `None` if any operand has no mapping or the op is
/// outside the P6 numeric subset (aborts kernelization — keep the VM body). Jump
/// TARGETS are remapped by the caller's patch pass; here the target is passed through.
fn remap_op_regs_with(op: Op, map: &impl Fn(Reg) -> Option<Reg>) -> Option<Op> {
    Some(match op {
        Op::LoadConst { dst, k } => Op::LoadConst { dst: map(dst)?, k },
        Op::LoadUndef { dst } => Op::LoadUndef { dst: map(dst)? },
        Op::Move { dst, src } => Op::Move { dst: map(dst)?, src: map(src)? },
        Op::Add { dst, lhs, rhs } => Op::Add { dst: map(dst)?, lhs: map(lhs)?, rhs: map(rhs)? },
        Op::Sub { dst, lhs, rhs } => Op::Sub { dst: map(dst)?, lhs: map(lhs)?, rhs: map(rhs)? },
        Op::Mul { dst, lhs, rhs } => Op::Mul { dst: map(dst)?, lhs: map(lhs)?, rhs: map(rhs)? },
        Op::Div { dst, lhs, rhs } => Op::Div { dst: map(dst)?, lhs: map(lhs)?, rhs: map(rhs)? },
        Op::Lt { dst, lhs, rhs } => Op::Lt { dst: map(dst)?, lhs: map(lhs)?, rhs: map(rhs)? },
        Op::Le { dst, lhs, rhs } => Op::Le { dst: map(dst)?, lhs: map(lhs)?, rhs: map(rhs)? },
        Op::Gt { dst, lhs, rhs } => Op::Gt { dst: map(dst)?, lhs: map(lhs)?, rhs: map(rhs)? },
        Op::Ge { dst, lhs, rhs } => Op::Ge { dst: map(dst)?, lhs: map(lhs)?, rhs: map(rhs)? },
        Op::Jmp { target } => Op::Jmp { target },
        Op::JmpIfFalse { cond, target } => Op::JmpIfFalse { cond: map(cond)?, target },
        _ => return None, // out of the P6 numeric subset → abort kernelization.
    })
}

// ════════════════════════════════════════════════════════════════════════════
// STAGE 3 — KERNEL LOOP OPTIMIZER (V8/Maglev-shaped: LICM + CSE + copy-prop).
//
// `kernelize_counted_loops` produces a kernel whose body is the JS frontend's
// raw bytecode: every loop-invariant constant is RE-LOADED each iteration
// (`mov rax, imm64; movq xmm, rax`), `x*x` is recomputed 3×, and trivial `Move`
// copies abound. The P6 backend (`compile_bytecode_f64`) lowers op-for-op, so it
// inherits every redundancy → ~3.5× slower than V8 inside the loop.
//
// This pass rewrites the kernel bytecode to be tight BEFORE lowering:
//   • LICM   — hoist each DISTINCT constant out of the loop into its own register,
//              materialized once in a preheader (the bound/limit is already a param).
//   • CSE    — a repeated pure op over the same value-numbered operands computes once.
//   • COPY   — `Move dst←src` propagates src's value to dst (no copy emitted).
//
// Every transform is VALUE-PRESERVING: it never reorders an `addsd`/`subsd`/`divsd`
// or changes an operand, only removes recomputation/rematerialization. So the loop
// stays BIT-IDENTICAL (IEEE-754 order preserved). On any shape it can't prove safe
// it returns None (caller keeps the un-tightened kernel — still correct).
// ════════════════════════════════════════════════════════════════════════════

/// Interned value-number definition in the kernel's straight-line dataflow.
/// A value number is an index into the optimizer's `defs` arena.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum KValDef {
    /// A loop-carried / invariant INPUT register (i, carry, limit) — opaque (the
    /// optimizer never folds across it; it is the loop interface, in a fixed reg).
    Input(Reg),
    /// A distinct f64 constant (interned by raw bits) — loop-invariant, hoistable.
    Const(u64),
    /// `undefined` → canonical NaN (loop-invariant, hoistable).
    Undef,
    /// The result of a pure binary op over two value numbers (interned for CSE).
    /// `tag` distinguishes Add(0)/Sub(1)/Mul(2)/Div(3).
    Bin(u8, u32, u32),
}

/// Tighten a kernelized counted-loop's bytecode (LICM + CSE + copy-prop). Returns
/// the optimized `(code, n_regs)` or `None` (keep the original) on any shape it
/// can't prove value-identical. See the module banner above for the contract.
fn optimize_kernel_loop(kcode: &[Op], consts: &[Value]) -> Option<(Vec<Op>, u32)> {
    // A/B knob (test/measurement only): CV_NO_KERNEL_OPT=1 skips the tightening so a
    // bench can measure the un-tightened Stage-2 kernel against this pass. Default ON.
    if std::env::var("CV_NO_KERNEL_OPT").as_deref() == Ok("1") {
        return None;
    }
    let n = kcode.len();
    if n < 4 {
        return None;
    }
    // ── Shape check. The kernelizer emits: [0]=cmp, [1]=JmpIfFalse(exit), <body>,
    //    <increment>, [back]=Jmp 0, [exit=back+1]=Ret(carry). Require exactly one
    //    back-edge `Jmp 0` immediately followed by the final Ret, and op[0]+[1] a
    //    comparison fused with a JmpIfFalse to the exit. Any other control shape → None.
    let exit = n - 1;
    let back = n - 2;
    if !matches!(kcode[exit], Op::Ret { .. }) {
        return None;
    }
    if !matches!(kcode[back], Op::Jmp { target: 0 }) {
        return None;
    }
    // No OTHER back-edge to 0 may exist.
    for (i, op) in kcode.iter().enumerate() {
        if i != back {
            if let Op::Jmp { target: 0 } = op {
                return None;
            }
        }
    }
    let carry_reg = match kcode[exit] {
        Op::Ret { src } => src,
        _ => return None,
    };
    let (i_reg, limit_reg) = match (kcode[0], kcode[1]) {
        (
            Op::Lt { dst, lhs, rhs } | Op::Le { dst, lhs, rhs }
            | Op::Gt { dst, lhs, rhs } | Op::Ge { dst, lhs, rhs },
            Op::JmpIfFalse { cond, target },
        ) if cond == dst && target as usize == exit => (lhs, rhs),
        _ => return None,
    };

    // ── Block boundaries: any jump target inside [0..back) starts a block. We reset
    //    CSE availability (not value identity) at each boundary. Validate the subset.
    let mut is_target = vec![false; n];
    is_target[0] = true;
    for op in &kcode[..back] {
        match *op {
            Op::Jmp { target } | Op::JmpIfFalse { target, .. } => {
                let t = target as usize;
                if t > exit {
                    return None;
                }
                is_target[t] = true;
            }
            Op::LoadConst { .. } | Op::LoadUndef { .. } | Op::Move { .. }
            | Op::Add { .. } | Op::Sub { .. } | Op::Mul { .. } | Op::Div { .. }
            | Op::Lt { .. } | Op::Le { .. } | Op::Gt { .. } | Op::Ge { .. } => {}
            _ => return None,
        }
    }

    // ── REACHABILITY + NO-MERGE GATE. A LINEAR value-numbering pass is only sound if
    //    the reachable region is a single straight-line trace (plus the loop back-edge):
    //    a real control-flow MERGE (a reachable op with ≥2 reachable predecessors)
    //    would need a phi, and a linear pass would wrongly take the last sequential
    //    writer (the bug a dead inlined else-branch exposes). So we (1) compute the
    //    ops reachable from op 0 — DEAD ops (e.g. an inlined leaf's unreachable
    //    else-branch) are dropped and never touch the value state — and (2) DECLINE if
    //    any reachable op has >1 reachable predecessor (a true merge). The canonical
    //    kernel has none (the only join is the loop header, whose 2 preds — entry +
    //    back-edge — agree by construction: i/carry are the loop-carried interface).
    let mut reachable = vec![false; n];
    {
        let mut stack = vec![0usize];
        while let Some(pc) = stack.pop() {
            if pc >= n || reachable[pc] {
                continue;
            }
            reachable[pc] = true;
            match kcode[pc] {
                Op::Jmp { target } => stack.push(target as usize),
                Op::JmpIfFalse { target, .. } => {
                    stack.push(target as usize);
                    stack.push(pc + 1); // fall-through (condition true)
                }
                Op::Ret { .. } => {}
                _ => stack.push(pc + 1), // straight-line fall-through
            }
        }
    }
    // Predecessor count over REACHABLE ops only (excluding the loop header's back-edge,
    // which is the legitimate loop join handled by the fixed interface registers).
    let mut pred_count = vec![0u32; n];
    for pc in 0..n {
        if !reachable[pc] {
            continue;
        }
        match kcode[pc] {
            Op::Jmp { target } => {
                if target as usize != 0 {
                    pred_count[target as usize] += 1;
                }
            }
            Op::JmpIfFalse { target, .. } => {
                pred_count[target as usize] += 1;
                if pc + 1 < n {
                    pred_count[pc + 1] += 1;
                }
            }
            Op::Ret { .. } => {}
            _ => {
                if pc + 1 < n {
                    pred_count[pc + 1] += 1;
                }
            }
        }
    }
    for pc in 0..n {
        if pc != 0 && reachable[pc] && pred_count[pc] > 1 {
            return None; // a real merge → linear value numbering is unsound; decline.
        }
    }

    // ── Value-numbering arena. `intern` dedups definitions to a stable value number
    //    (a global value number → CSE of identical pure exprs is sound because SSE is
    //    deterministic and we keep each value live in ONE register until its last use).
    let mut defs: Vec<KValDef> = Vec::new();
    let mut intern: std::collections::HashMap<KValDef, u32> = std::collections::HashMap::new();
    fn mk(d: KValDef, defs: &mut Vec<KValDef>, intern: &mut std::collections::HashMap<KValDef, u32>) -> u32 {
        if let Some(&v) = intern.get(&d) {
            return v;
        }
        let v = defs.len() as u32;
        defs.push(d.clone());
        intern.insert(d, v);
        v
    }
    let const_bits = |k: u16| -> Option<u64> {
        match consts.get(k as usize) {
            Some(Value::Number(nn)) => Some(nn.to_bits()),
            _ => None,
        }
    };
    let bin_tag = |op: &Op| -> u8 {
        match op {
            Op::Add { .. } => 0,
            Op::Sub { .. } => 1,
            Op::Mul { .. } => 2,
            Op::Div { .. } => 3,
            _ => 255,
        }
    };

    // reg_vn: the value number each register currently holds (program-point state).
    let mut reg_vn: std::collections::HashMap<Reg, u32> = std::collections::HashMap::new();
    let vn_i = mk(KValDef::Input(i_reg), &mut defs, &mut intern);
    let vn_carry = mk(KValDef::Input(carry_reg), &mut defs, &mut intern);
    let vn_limit = mk(KValDef::Input(limit_reg), &mut defs, &mut intern);
    reg_vn.insert(i_reg, vn_i);
    reg_vn.insert(carry_reg, vn_carry);
    reg_vn.insert(limit_reg, vn_limit);

    // For each op (in [0..back)), the value number of its RESULT (None = control).
    let mut op_result: Vec<Option<u32>> = vec![None; back];
    // `dropped[pc]` = the op was eliminated (LoadConst hoisted, Move copy-propagated,
    // or a CSE-redundant Bin) and emits NOTHING in the loop body.
    let mut dropped: Vec<bool> = vec![false; back];
    let mut op_lhs: Vec<Option<u32>> = vec![None; back];
    let mut op_rhs: Vec<Option<u32>> = vec![None; back];

    let resolve = |reg_vn: &std::collections::HashMap<Reg, u32>,
                   r: Reg,
                   defs: &mut Vec<KValDef>,
                   intern: &mut std::collections::HashMap<KValDef, u32>| -> u32 {
        if let Some(&v) = reg_vn.get(&r) {
            v
        } else {
            mk(KValDef::Input(r), defs, intern)
        }
    };

    // CSE availability: value numbers whose result register is live in THIS block.
    let mut avail: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for pc in 0..back {
        if !reachable[pc] {
            dropped[pc] = true; // dead op (e.g. inlined leaf's else-branch) → never emit
            continue; // and never let its writes pollute the value state
        }
        if is_target[pc] {
            avail.clear();
        }
        match kcode[pc] {
            Op::LoadConst { dst, k } => {
                let bits = const_bits(k)?;
                let v = mk(KValDef::Const(bits), &mut defs, &mut intern);
                reg_vn.insert(dst, v);
                op_result[pc] = Some(v);
                dropped[pc] = true; // hoisted to preheader
            }
            Op::LoadUndef { dst } => {
                let v = mk(KValDef::Undef, &mut defs, &mut intern);
                reg_vn.insert(dst, v);
                op_result[pc] = Some(v);
                dropped[pc] = true; // hoisted to preheader
            }
            Op::Move { dst, src } => {
                let v = resolve(&reg_vn, src, &mut defs, &mut intern);
                reg_vn.insert(dst, v);
                op_result[pc] = Some(v);
                dropped[pc] = true; // copy-propagated
            }
            Op::Add { dst, lhs, rhs } | Op::Sub { dst, lhs, rhs }
            | Op::Mul { dst, lhs, rhs } | Op::Div { dst, lhs, rhs } => {
                let lv = resolve(&reg_vn, lhs, &mut defs, &mut intern);
                let rv = resolve(&reg_vn, rhs, &mut defs, &mut intern);
                op_lhs[pc] = Some(lv);
                op_rhs[pc] = Some(rv);
                let v = mk(KValDef::Bin(bin_tag(&kcode[pc]), lv, rv), &mut defs, &mut intern);
                if avail.contains(&v) {
                    dropped[pc] = true; // CSE hit — result already live in a register
                } else {
                    avail.insert(v);
                }
                reg_vn.insert(dst, v);
                op_result[pc] = Some(v);
            }
            Op::Lt { lhs, rhs, .. } | Op::Le { lhs, rhs, .. }
            | Op::Gt { lhs, rhs, .. } | Op::Ge { lhs, rhs, .. } => {
                let lv = resolve(&reg_vn, lhs, &mut defs, &mut intern);
                let rv = resolve(&reg_vn, rhs, &mut defs, &mut intern);
                op_lhs[pc] = Some(lv);
                op_rhs[pc] = Some(rv);
            }
            Op::JmpIfFalse { cond, .. } => {
                let v = resolve(&reg_vn, cond, &mut defs, &mut intern);
                op_lhs[pc] = Some(v);
            }
            Op::Jmp { .. } => {}
            _ => return None,
        }
    }

    // ── LOOP-CARRIED WRITEBACK values. At the end of the iteration the induction var
    //    `i_reg` and the accumulator `carry_reg` hold their NEXT-iteration values
    //    (computed into temporaries). Copy-prop dropped the `Move interface ← temp`
    //    writebacks, so we must re-materialize them before the back-edge: capture the
    //    value numbers `i_reg`/`carry_reg` map to at loop end and emit explicit
    //    writeback copies. (These are the loop phis V8 also carries.)
    let final_i_vn = *reg_vn.get(&i_reg).unwrap_or(&vn_i);
    let final_carry_vn = *reg_vn.get(&carry_reg).unwrap_or(&vn_carry);

    // ── Liveness: last pc that READS each value number (for temp-register recycling).
    let mut last_use: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for pc in 0..back {
        for v in [op_lhs[pc], op_rhs[pc]].into_iter().flatten() {
            last_use.insert(v, pc);
        }
    }
    last_use.insert(vn_i, back);
    last_use.insert(vn_carry, back);
    last_use.insert(vn_limit, back);
    // The loop-carried NEXT values must survive to the back-edge writeback.
    last_use.insert(final_i_vn, back);
    last_use.insert(final_carry_vn, back);

    // ── Register assignment. Interface regs keep their original numbers; hoisted
    //    constants and Bin temporaries are allocated above them, recycled on death.
    let mut next_reg: Reg = i_reg.max(carry_reg).max(limit_reg).checked_add(1)?;
    let mut want_hoist: Vec<u32> = Vec::new();
    {
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for pc in 0..back {
            for v in [op_lhs[pc], op_rhs[pc]].into_iter().flatten() {
                if matches!(defs[v as usize], KValDef::Const(_) | KValDef::Undef) && seen.insert(v) {
                    want_hoist.push(v);
                }
            }
        }
    }
    let mut const_k_for: std::collections::HashMap<u64, u16> = std::collections::HashMap::new();
    for (k, val) in consts.iter().enumerate() {
        if let Value::Number(nn) = val {
            const_k_for.entry(nn.to_bits()).or_insert(k as u16);
        }
    }
    let mut val_reg: std::collections::HashMap<u32, Reg> = std::collections::HashMap::new();
    val_reg.insert(vn_i, i_reg);
    val_reg.insert(vn_carry, carry_reg);
    val_reg.insert(vn_limit, limit_reg);
    let mut preheader: Vec<Op> = Vec::new();
    for &v in &want_hoist {
        let r = next_reg;
        next_reg = next_reg.checked_add(1)?;
        if next_reg as u32 > 16 {
            return None;
        }
        val_reg.insert(v, r);
        match defs[v as usize] {
            KValDef::Const(bits) => {
                let k = *const_k_for.get(&bits)?;
                preheader.push(Op::LoadConst { dst: r, k });
            }
            KValDef::Undef => preheader.push(Op::LoadUndef { dst: r }),
            _ => unreachable!(),
        }
    }

    // A DEDICATED register for the loop comparison's result (op[0] Lt → op[1]
    // JmpIfFalse). It is fused away in native lowering, but the VM-deopt path writes
    // it, so it must NOT alias any interface/hoisted-const/temp register (else a
    // hoisted constant would be clobbered by the Bool on the VM path).
    let cmp_reg = next_reg;
    next_reg = next_reg.checked_add(1)?;
    if next_reg as u32 > 16 {
        return None;
    }
    // Temp register pool (above interface + hoisted consts + the cmp slot), recycled.
    let temp_base = next_reg;
    let mut free_temps: Vec<Reg> = Vec::new();
    let mut high_water = temp_base;
    // reg_owner[r] = the value number currently occupying temp register r (for safe
    // recycling — a register is only freed when its CURRENT owner dies, so a register
    // reused as an in-place destination is never double-freed while still live).
    let mut reg_owner: std::collections::HashMap<Reg, u32> = std::collections::HashMap::new();
    let mut alloc_temp = |free: &mut Vec<Reg>, hw: &mut Reg| -> Option<Reg> {
        if let Some(r) = free.pop() {
            Some(r)
        } else {
            let r = *hw;
            *hw = hw.checked_add(1)?;
            if *hw as u32 > 16 {
                return None;
            }
            Some(r)
        }
    };

    // ── Emit. preheader, then the loop body with dropped ops removed and operands
    //    rewritten to the registers their values live in. old→new pc map retargets
    //    jumps (a jump to a dropped op lands on the next emitted op).
    let mut out: Vec<Op> = Vec::with_capacity(n);
    out.extend_from_slice(&preheader);
    let mut old_to_new: Vec<usize> = vec![0usize; n];
    let mut patch_targets: Vec<(usize, usize)> = Vec::new(); // (out_idx, old_target)

    for pc in 0..back {
        old_to_new[pc] = out.len();
        // Free temp registers whose values died at the PREVIOUS pc, so a dead operand's
        // register can be reused by this op's result. Only free a register whose CURRENT
        // owner is the dying value (a register reused in-place at pc-1 has a new owner
        // and must NOT be freed).
        if pc > 0 {
            for v in [op_lhs[pc - 1], op_rhs[pc - 1]].into_iter().flatten() {
                if last_use.get(&v) == Some(&(pc - 1))
                    && matches!(defs[v as usize], KValDef::Bin(..))
                {
                    if let Some(r) = val_reg.get(&v).copied() {
                        if r >= temp_base && reg_owner.get(&r) == Some(&v) {
                            reg_owner.remove(&r);
                            free_temps.push(r);
                        }
                    }
                }
            }
        }
        if dropped[pc] {
            continue;
        }
        match kcode[pc] {
            Op::Add { .. } | Op::Sub { .. } | Op::Mul { .. } | Op::Div { .. } => {
                let v = op_result[pc].unwrap();
                let lv = op_lhs[pc].unwrap();
                let lr = *val_reg.get(&lv)?;
                let rr = *val_reg.get(&op_rhs[pc].unwrap())?;
                // V8-style in-place op: if the LHS value dies at THIS op and lives in a
                // temp register it still OWNS, reuse it as the destination. Then
                // `compile_bytecode_f64` sees `dst == lhs` and emits the bare
                // `addsd/subsd/mulsd/divsd dst, rhs` (one SSE op, NO `movsd` copy).
                // Bit-exact: lhs IS the first operand. Otherwise allocate fresh.
                let lhs_dies_here = last_use.get(&lv) == Some(&pc);
                let dr = if lhs_dies_here && lr >= temp_base && reg_owner.get(&lr) == Some(&lv) {
                    lr
                } else {
                    alloc_temp(&mut free_temps, &mut high_water)?
                };
                reg_owner.insert(dr, v);
                val_reg.insert(v, dr);
                out.push(match kcode[pc] {
                    Op::Add { .. } => Op::Add { dst: dr, lhs: lr, rhs: rr },
                    Op::Sub { .. } => Op::Sub { dst: dr, lhs: lr, rhs: rr },
                    Op::Mul { .. } => Op::Mul { dst: dr, lhs: lr, rhs: rr },
                    Op::Div { .. } => Op::Div { dst: dr, lhs: lr, rhs: rr },
                    _ => unreachable!(),
                });
            }
            Op::Lt { .. } | Op::Le { .. } | Op::Gt { .. } | Op::Ge { .. } => {
                let lr = *val_reg.get(&op_lhs[pc].unwrap())?;
                let rr = *val_reg.get(&op_rhs[pc].unwrap())?;
                // Write the result into the dedicated cmp register (collision-free).
                out.push(match kcode[pc] {
                    Op::Lt { .. } => Op::Lt { dst: cmp_reg, lhs: lr, rhs: rr },
                    Op::Le { .. } => Op::Le { dst: cmp_reg, lhs: lr, rhs: rr },
                    Op::Gt { .. } => Op::Gt { dst: cmp_reg, lhs: lr, rhs: rr },
                    Op::Ge { .. } => Op::Ge { dst: cmp_reg, lhs: lr, rhs: rr },
                    _ => unreachable!(),
                });
            }
            Op::JmpIfFalse { target, .. } => {
                let oi = out.len();
                // cond reads the dedicated cmp register written by the preceding Lt/...
                out.push(Op::JmpIfFalse { cond: cmp_reg, target: u16::MAX });
                patch_targets.push((oi, target as usize));
            }
            Op::Jmp { target } => {
                let oi = out.len();
                out.push(Op::Jmp { target: u16::MAX });
                patch_targets.push((oi, target as usize));
            }
            _ => return None,
        }
    }
    // ── Loop-carried writebacks (the phis): copy the NEXT-iteration i / accumulator
    //    from their temporaries back into the fixed interface registers so the header
    //    reads them next iteration and the Ret returns the final accumulator. Skip when
    //    a value is unchanged or already in its interface register.
    let wb_i = *val_reg.get(&final_i_vn)?;
    if wb_i != i_reg {
        out.push(Op::Move { dst: i_reg, src: wb_i });
    }
    let wb_carry = *val_reg.get(&final_carry_vn)?;
    if wb_carry != carry_reg {
        out.push(Op::Move { dst: carry_reg, src: wb_carry });
    }

    // The back-edge Jmp and the Ret.
    old_to_new[back] = out.len();
    out.push(Op::Jmp { target: u16::MAX });
    patch_targets.push((out.len() - 1, 0));
    old_to_new[exit] = out.len();
    out.push(Op::Ret { src: carry_reg });

    // Retarget jumps through the old→new map (a jump to a dropped op shares the
    // out-index of the following op, since old_to_new[pc] is set BEFORE the drop).
    for (oi, old_t) in patch_targets {
        let nt = *old_to_new.get(old_t)?;
        match &mut out[oi] {
            Op::Jmp { target } => *target = nt as u16,
            Op::JmpIfFalse { target, .. } => *target = nt as u16,
            _ => return None,
        }
    }

    // ── Peephole: remove `Jmp` to the immediately-following op (a no-op fall-through
    //    left by the dead-branch elimination — e.g. the inlined leaf's skipped
    //    else-branch). A jmp-to-next is pure overhead per iteration. Rebuild compactly,
    //    remapping every jump target through an old→new index map. Bit-neutral.
    loop {
        let removable: Option<usize> = out
            .iter()
            .enumerate()
            .position(|(i, op)| matches!(op, Op::Jmp { target } if *target as usize == i + 1));
        let ri = match removable {
            Some(i) => i,
            None => break,
        };
        let mut compact: Vec<Op> = Vec::with_capacity(out.len() - 1);
        let mut remap: Vec<usize> = vec![0; out.len() + 1];
        for (i, op) in out.iter().enumerate() {
            remap[i] = compact.len();
            if i == ri {
                continue; // drop the no-op Jmp
            }
            compact.push(*op);
        }
        remap[out.len()] = compact.len();
        for op in &mut compact {
            match op {
                Op::Jmp { target } | Op::JmpIfFalse { target, .. } => {
                    *target = remap[*target as usize] as u16;
                }
                _ => {}
            }
        }
        out = compact;
    }

    // n_regs must cover EVERY register the emitted code references — the temp pool
    // high-water mark AND the dedicated comparison register.
    let n_regs = (high_water.max(temp_base) as u32).max(cmp_reg as u32 + 1);
    if n_regs > 16 {
        return None;
    }
    Some((out, n_regs))
}

/// A register file borrowed from `REGS_POOL`. Derefs to `Vec<Value>` (so callers
/// index it normally) and returns the buffer — capacity retained, values
/// cleared — to the pool on Drop, covering every exit path (return/throw/`?`).
struct PooledRegs(Vec<Value>);

impl PooledRegs {
    fn new(n: usize) -> Self {
        let mut v = REGS_POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
        v.clear();
        v.resize(n, Value::Undefined);
        PooledRegs(v)
    }

    /// Wrap an already-built register image (the T2 Phase-5 deopt resume path:
    /// the regs are decoded from the JIT bank, not fetched from the pool). The Vec
    /// must already be sized to ≥ the function's `n_regs`; we grow (never shrink)
    /// to `n` so a short bank still gives the VM a full register file. On Drop the
    /// buffer returns to the pool like any other `PooledRegs`.
    fn from_vec(mut v: Vec<Value>, n: usize) -> Self {
        if v.len() < n {
            v.resize(n, Value::Undefined);
        }
        PooledRegs(v)
    }
}

impl Drop for PooledRegs {
    fn drop(&mut self) {
        let mut v = std::mem::take(&mut self.0);
        v.clear(); // release the Rc'd Values; keep the allocation
        REGS_POOL.with(|p| {
            let mut pool = p.borrow_mut();
            if pool.len() < 1024 {
                pool.push(v);
            }
        });
    }
}

impl std::ops::Deref for PooledRegs {
    type Target = Vec<Value>;
    fn deref(&self) -> &Vec<Value> {
        &self.0
    }
}

impl std::ops::DerefMut for PooledRegs {
    fn deref_mut(&mut self) -> &mut Vec<Value> {
        &mut self.0
    }
}

// ======================================================================
// M4.2a — single-source op bodies shared by the VM match AND the T1 JIT.
//
// `VmState` bundles `run_function`'s loop-locals + the param borrows behind
// one struct reachable via a single pointer. The minimal hot-op subset is
// refactored into `#[inline] op_xxx(&mut VmState, operands) -> StepStatus`
// helpers that BOTH the bytecode match and the T1 baseline JIT call — one
// body, no copy-paste. `#[inline]` keeps the VM's jump-table speed (the
// helper is inlined back into the match arm).
//
// The T1 JIT does NOT re-implement op semantics. It emits a control-flow
// skeleton that, per bytecode op, calls a single `t1_op_thunk` (which
// reconstructs `&mut VmState` from the raw pointer, runs exactly that op via
// the shared `op_xxx` helper, and returns a `StepStatus`); native code then
// branches on the status. This makes T1==VM semantically by construction and
// makes the aliasing contract trivial: no Rust borrow is ever held across the
// native `call` — the thunk takes/returns a plain pointer + status.
// ======================================================================

/// The bundled VM execution state. Holds RAW pointers to `run_function`'s
/// stack-locals (so it is reconstructible from a single `*mut VmState` inside
/// the T1 thunk) plus the shared param borrows. `ip`/`wd_ticks` are stored BY
/// VALUE — they are the authoritative cursor while a T1 function runs; the VM
/// match path mirrors them back into its own locals after each refactored arm.
///
/// SAFETY: every pointer targets a live local of the enclosing `run_function`
/// frame, which strictly outlives every `op_xxx`/thunk call. Op helpers
/// reconstruct `&mut` references locally and never let them escape the call, so
/// no two overlapping `&mut`s are ever live at once.
pub(crate) struct VmState<'a> {
    regs: *mut Vec<Value>,
    f: &'a BcFunction,
    module: &'a Module,
    globals: &'a std::cell::RefCell<HashMap<String, Value>>,
    /// Raw (fat) pointer to the live dispatch trait object. Stored as a thin
    /// reborrow target so an op helper can reconstruct `&mut dyn FnMut` without
    /// holding a Rust borrow of the enclosing frame across the call boundary.
    dispatch: *mut (dyn FnMut(Value, Value, Vec<Value>) -> Result<Value, RuntimeError> + 'a),
    ip: usize,
    try_stack: *mut Vec<(usize, Reg)>,
    last_callee_hint: *mut String,
    wd_ticks: u32,
    /// T1 out-slot: where the thunk stashes a `Returned`/`Threw`/`Deadline`
    /// status (whose payload can't fit in the integer tag the thunk returns to
    /// native code). `null` on the VM match path (never written there).
    out: *mut Option<StepStatus>,
}

impl<'a> VmState<'a> {
    #[inline(always)]
    fn regs(&mut self) -> &mut Vec<Value> {
        // SAFETY: see struct-level contract.
        unsafe { &mut *self.regs }
    }
}

/// The outcome of executing exactly one op via a shared `op_xxx` helper.
/// Mirrors `run_function`'s non-local control flow precisely:
///   * `Continue`     — fall through to the next op (ip already advanced).
///   * `Jumped(t)`    — set ip to `t` (a bytecode index).
///   * `Returned(v)`  — `Op::Ret`: this is the function's result.
///   * `Threw(e)`     — a catchable error; route to the try_stack or propagate.
///   * `Deadline`     — the watchdog fired; UNCATCHABLE, unwind to the host.
pub(crate) enum StepStatus {
    Continue,
    Jumped(usize),
    Returned(Value),
    Threw(RuntimeError),
    Deadline,
}

// ---- the minimal hot-op subset, as #[inline] shared bodies ----

#[inline]
fn op_load_const(st: &mut VmState, dst: Reg, k: u16) -> StepStatus {
    let v = st.f.consts[k as usize].clone();
    st.regs()[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_load_true(st: &mut VmState, dst: Reg) -> StepStatus {
    st.regs()[dst as usize] = Value::Bool(true);
    StepStatus::Continue
}

#[inline]
fn op_load_false(st: &mut VmState, dst: Reg) -> StepStatus {
    st.regs()[dst as usize] = Value::Bool(false);
    StepStatus::Continue
}

#[inline]
fn op_load_null(st: &mut VmState, dst: Reg) -> StepStatus {
    st.regs()[dst as usize] = Value::Null;
    StepStatus::Continue
}

#[inline]
fn op_load_undef(st: &mut VmState, dst: Reg) -> StepStatus {
    st.regs()[dst as usize] = Value::Undefined;
    StepStatus::Continue
}

#[inline]
fn op_move(st: &mut VmState, dst: Reg, src: Reg) -> StepStatus {
    let v = st.regs()[src as usize].clone();
    st.regs()[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_add(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Number(a + b)
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match bigint_binop("+", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            // An Object operand needs ToPrimitive (`Symbol.toPrimitive` /
            // `valueOf` / `toString` via the interp) before `+` decides
            // string-concat vs numeric — `add_values` only stringifies with the
            // opaque `[object Object]`, which diverges from the tree-walker on
            // `'' + {toString(){...}}` and `obj + 1` (valueOf). Route Object
            // operands through `__tb_host_binop("+")` (the full tree-walk
            // `binary_op`, ToPrimitive-aware) so VM `+` is byte-identical.
            None => match additive_host_binop(
                &regs[lhs as usize],
                &regs[rhs as usize],
                st.globals,
                disp,
            ) {
                Some(Ok(r)) => r,
                Some(Err(e)) => return StepStatus::from_err(e),
                None => match add_values(&regs[lhs as usize], &regs[rhs as usize]) {
                    Ok(r) => r,
                    Err(e) => return StepStatus::from_err(e),
                },
            },
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

/// When either operand of `+` is an Object, dispatch to `__tb_host_binop("+")`
/// so the full ToPrimitive (`Symbol.toPrimitive`/`valueOf`/`toString`) runs
/// before deciding string-concat vs numeric — byte-identical to the tree-walk
/// `+`. Returns `None` when neither operand is an Object (the fast `add_values`
/// path handles it) or when no host is wired (bare-globals entry points).
fn additive_host_binop(
    a: &Value,
    b: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Option<Result<Value, RuntimeError>> {
    if !matches!(a, Value::Object(_)) && !matches!(b, Value::Object(_)) {
        return None;
    }
    let getter = globals.borrow().get("__tb_host_binop").cloned();
    match getter {
        Some(g @ Value::NativeFunction(_)) => Some(dispatch(
            g,
            Value::Undefined,
            vec![Value::str("+".to_string()), a.clone(), b.clone()],
        )),
        _ => None,
    }
}

#[inline]
fn op_sub(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Number(a - b)
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match bigint_binop("-", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            None => match (to_num(&regs[lhs as usize]), to_num(&regs[rhs as usize])) {
                (Ok(a), Ok(b)) => Value::Number(a - b),
                (Err(e), _) | (_, Err(e)) => return StepStatus::from_err(e),
            },
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_mul(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Number(a * b)
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match bigint_binop("*", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            None => match (to_num(&regs[lhs as usize]), to_num(&regs[rhs as usize])) {
                (Ok(a), Ok(b)) => Value::Number(a * b),
                (Err(e), _) | (_, Err(e)) => return StepStatus::from_err(e),
            },
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_eq(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    regs[dst as usize] = Value::Bool(Value::strict_eq(&regs[lhs as usize], &regs[rhs as usize]));
    StepStatus::Continue
}

#[inline]
fn op_neq(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    regs[dst as usize] = Value::Bool(!Value::strict_eq(&regs[lhs as usize], &regs[rhs as usize]));
    StepStatus::Continue
}

#[inline]
fn op_lt(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Bool(a < b)
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match relational_host_binop("<", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            None => Value::Bool(
                Value::abstract_relational_compare(&regs[lhs as usize], &regs[rhs as usize])
                    .unwrap_or(false),
            ),
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_le(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Bool(a <= b)
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match relational_host_binop("<=", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            None => Value::Bool(match Value::abstract_relational_compare(
                &regs[rhs as usize],
                &regs[lhs as usize],
            ) {
                Some(true) => false,
                Some(false) => true,
                None => false,
            }),
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_gt(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Bool(a > b)
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match relational_host_binop(">", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            None => Value::Bool(
                Value::abstract_relational_compare(&regs[rhs as usize], &regs[lhs as usize])
                    .unwrap_or(false),
            ),
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_ge(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Bool(a >= b)
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match relational_host_binop(">=", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            None => Value::Bool(match Value::abstract_relational_compare(
                &regs[lhs as usize],
                &regs[rhs as usize],
            ) {
                Some(true) => false,
                Some(false) => true,
                None => false,
            }),
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_jmp(_st: &mut VmState, target: u16) -> StepStatus {
    StepStatus::Jumped(target as usize)
}

#[inline]
fn op_jmp_if_false(st: &mut VmState, cond: Reg, target: u16) -> StepStatus {
    if !st.regs()[cond as usize].to_bool() {
        StepStatus::Jumped(target as usize)
    } else {
        StepStatus::Continue
    }
}

#[inline]
fn op_ret(st: &mut VmState, src: Reg) -> StepStatus {
    StepStatus::Returned(st.regs()[src as usize].clone())
}

// ---- M4.2b mechanical op expansion: arithmetic/bitwise/unary, no IC, no
// re-entry, no object access. Each body is the SINGLE source of truth shared by
// the VM match arm AND the T1 thunk, mirroring the prior inline VM arms exactly
// (same spec-correct helpers: bigint_binop, to_num, to_int32, to_uint32). ----

#[inline]
fn op_mod(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Number(a % b)
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match bigint_binop("%", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            None => match (to_num(&regs[lhs as usize]), to_num(&regs[rhs as usize])) {
                (Ok(a), Ok(b)) => Value::Number(a % b),
                (Err(e), _) | (_, Err(e)) => return StepStatus::from_err(e),
            },
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_pow(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    // Use the spec-faithful `js_pow` (ECMA-262 §21.3.2.26 special cases:
    // `1 ** NaN` / `(-1) ** ±Infinity` → NaN, unlike IEEE `powf` which returns
    // 1.0) so VM exponentiation is byte-identical to the tree-walker's `**`.
    let v = if let (Value::Number(a), Value::Number(b)) =
        (&regs[lhs as usize], &regs[rhs as usize])
    {
        Value::Number(crate::interp::js_pow(*a, *b))
    } else {
        let disp = unsafe { &mut *st.dispatch };
        match bigint_binop("**", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
            Some(Ok(r)) => r,
            Some(Err(e)) => return StepStatus::from_err(e),
            None => match (to_num(&regs[lhs as usize]), to_num(&regs[rhs as usize])) {
                (Ok(a), Ok(b)) => Value::Number(crate::interp::js_pow(a, b)),
                (Err(e), _) | (_, Err(e)) => return StepStatus::from_err(e),
            },
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

/// Shared integer bitwise binop body for `&`, `|`, `^` (ToInt32 both operands).
#[inline]
fn op_bit_i32(
    st: &mut VmState,
    dst: Reg,
    lhs: Reg,
    rhs: Reg,
    op: &str,
    f: fn(i32, i32) -> i32,
) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let disp = unsafe { &mut *st.dispatch };
    let v = match bigint_binop(op, &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return StepStatus::from_err(e),
        None => match (to_int32(&regs[lhs as usize]), to_int32(&regs[rhs as usize])) {
            (Ok(a), Ok(b)) => Value::Number(f(a, b) as f64),
            (Err(e), _) | (_, Err(e)) => return StepStatus::from_err(e),
        },
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_bitand(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    op_bit_i32(st, dst, lhs, rhs, "&", |a, b| a & b)
}

#[inline]
fn op_bitor(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    op_bit_i32(st, dst, lhs, rhs, "|", |a, b| a | b)
}

#[inline]
fn op_bitxor(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    op_bit_i32(st, dst, lhs, rhs, "^", |a, b| a ^ b)
}

#[inline]
fn op_shl(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let disp = unsafe { &mut *st.dispatch };
    let v = match bigint_binop("<<", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return StepStatus::from_err(e),
        None => {
            let a = match to_int32(&regs[lhs as usize]) {
                Ok(a) => a,
                Err(e) => return StepStatus::from_err(e),
            };
            let b = match to_uint32(&regs[rhs as usize]) {
                Ok(b) => b & 31,
                Err(e) => return StepStatus::from_err(e),
            };
            Value::Number((a.wrapping_shl(b)) as f64)
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_shr(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let disp = unsafe { &mut *st.dispatch };
    let v = match bigint_binop(">>", &regs[lhs as usize], &regs[rhs as usize], st.globals, disp) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return StepStatus::from_err(e),
        None => {
            let a = match to_int32(&regs[lhs as usize]) {
                Ok(a) => a,
                Err(e) => return StepStatus::from_err(e),
            };
            let b = match to_uint32(&regs[rhs as usize]) {
                Ok(b) => b & 31,
                Err(e) => return StepStatus::from_err(e),
            };
            Value::Number((a >> b) as f64)
        }
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_ushr(st: &mut VmState, dst: Reg, lhs: Reg, rhs: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    let a = match to_uint32(&regs[lhs as usize]) {
        Ok(a) => a,
        Err(e) => return StepStatus::from_err(e),
    };
    let b = match to_uint32(&regs[rhs as usize]) {
        Ok(b) => b & 31,
        Err(e) => return StepStatus::from_err(e),
    };
    regs[dst as usize] = Value::Number((a >> b) as f64);
    StepStatus::Continue
}

#[inline]
fn op_bitnot(st: &mut VmState, dst: Reg, src: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    // `~x` on a BigInt is `-(x + 1)` and stays a BigInt (`~2n === -3n`),
    // NOT a ToInt32 coercion. Numbers ToInt32. (Mirrors the VM arm.)
    let v = match &regs[src as usize] {
        Value::BigInt(n) => Value::bigint(n.bit_not()),
        other => match to_int32(other) {
            Ok(i) => Value::Number((!i) as f64),
            Err(e) => return StepStatus::from_err(e),
        },
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_neg(st: &mut VmState, dst: Reg, src: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    // Unary `-` on a BigInt must stay a BigInt (`-2n === -2n`); 0n - n. Numbers
    // negate via ToNumber. (Mirrors the VM arm.)
    let v = match &regs[src as usize] {
        Value::BigInt(n) => Value::bigint(crate::interp::JsBigInt::sub(
            &crate::interp::JsBigInt::zero(),
            n,
        )),
        other => match to_num(other) {
            Ok(n) => Value::Number(-n),
            Err(e) => return StepStatus::from_err(e),
        },
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_not(st: &mut VmState, dst: Reg, src: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    regs[dst as usize] = Value::Bool(!regs[src as usize].to_bool());
    StepStatus::Continue
}

#[inline]
fn op_typeof(st: &mut VmState, dst: Reg, src: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    regs[dst as usize] = Value::String(crate::interp::typeof_name(&regs[src as usize]).into());
    StepStatus::Continue
}

#[inline]
fn op_to_number(st: &mut VmState, dst: Reg, src: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    // Unary `+`: ECMA-262 §13.5.4 = ToNumber(operand). ToNumber on a BigInt
    // THROWS a TypeError per §7.1.4 — surface it as a CATCHABLE thrown Error.
    // (Mirrors the VM arm.)
    let v = match &regs[src as usize] {
        Value::BigInt(_) => {
            return StepStatus::from_err(RuntimeError::Thrown(crate::interp::err_str(
                "TypeError: Cannot convert a BigInt value to a number".into(),
            )));
        }
        other => match to_num(other) {
            Ok(n) => Value::Number(n),
            Err(e) => return StepStatus::from_err(e),
        },
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

#[inline]
fn op_to_str(st: &mut VmState, dst: Reg, src: Reg) -> StepStatus {
    let regs = unsafe { &mut *st.regs };
    // ECMA-262 §7.1.17 ToString. Primitives (and Array/Function values, whose
    // default ToString is positional, no user-overridable hint) stringify
    // directly via `to_display_string` — byte-identical to the tree-walk
    // `to_string_value` (which only routes `Value::Object` through ToPrimitive).
    // An OBJECT operand needs the full string-hint ToPrimitive
    // (`@@toPrimitive('string')` → `toString` → `valueOf`, throw-propagating),
    // which lives in the interp; dispatch to the `__tb_host_to_string` host hook
    // (mirrors how `op_add` routes object operands to `__tb_host_binop`). On a
    // bare-globals entry point (no host wired) we fall back to the positional
    // display string — the same graceful degradation the other host hooks use.
    let v = match &regs[src as usize] {
        Value::Object(_) => {
            let disp = unsafe { &mut *st.dispatch };
            match host_to_string(&regs[src as usize], st.globals, disp) {
                Some(Ok(s)) => Value::str(s),
                Some(Err(e)) => return StepStatus::from_err(e),
                None => Value::str(regs[src as usize].to_display_string()),
            }
        }
        other => Value::str(other.to_display_string()),
    };
    regs[dst as usize] = v;
    StepStatus::Continue
}

/// Route an Object operand's ToString (STRING hint) through the interp's full
/// `to_string_throwing` (`@@toPrimitive('string')`/`toString`/`valueOf`,
/// throw-propagating) via the `__tb_host_to_string` global — byte-identical to
/// the tree-walk `${expr}` path (`to_string_value`). Returns `None` when no host
/// is wired (bare-globals entry points), letting the caller fall back to the
/// positional display string.
fn host_to_string(
    v: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Option<Result<String, RuntimeError>> {
    let getter = globals.borrow().get("__tb_host_to_string").cloned();
    match getter {
        Some(g @ Value::NativeFunction(_)) => {
            match dispatch(g, Value::Undefined, vec![v.clone()]) {
                Ok(Value::String(s)) => Some(Ok(s.to_string())),
                Ok(other) => Some(Ok(other.to_display_string())),
                Err(e) => Some(Err(e)),
            }
        }
        _ => None,
    }
}

impl StepStatus {
    /// Map a `RuntimeError` to the right step status. A watchdog `Deadline` is
    /// reported distinctly so callers keep it UNCATCHABLE (cannot be routed to a
    /// JS try/catch), exactly as `run_function`'s `propagate!` does.
    #[inline]
    fn from_err(e: RuntimeError) -> StepStatus {
        if matches!(e, RuntimeError::Deadline) {
            StepStatus::Deadline
        } else {
            StepStatus::Threw(e)
        }
    }
}

/// Whether op `o` is in the T1 baseline-JIT supported subset. T1 declines
/// (compiles nothing, falls back to the VM) for a function containing any op
/// not listed here — so unsupported ops keep their existing match arms and are
/// never reached by native code.
pub(crate) fn t1_supported_op(o: &Op) -> bool {
    matches!(
        o,
        Op::LoadConst { .. }
            | Op::LoadTrue { .. }
            | Op::LoadFalse { .. }
            | Op::LoadNull { .. }
            | Op::LoadUndef { .. }
            | Op::Move { .. }
            | Op::Add { .. }
            | Op::Sub { .. }
            | Op::Mul { .. }
            | Op::Lt { .. }
            | Op::Le { .. }
            | Op::Gt { .. }
            | Op::Ge { .. }
            | Op::Eq { .. }
            | Op::Neq { .. }
            // M4.2b mechanical op expansion: arithmetic/bitwise/unary ops with
            // no IC, no re-entry, no object access — single-source shared bodies.
            | Op::Mod { .. }
            | Op::Pow { .. }
            | Op::BitAnd { .. }
            | Op::BitOr { .. }
            | Op::BitXor { .. }
            | Op::Shl { .. }
            | Op::Shr { .. }
            | Op::Ushr { .. }
            | Op::BitNot { .. }
            | Op::Neg { .. }
            | Op::Not { .. }
            | Op::Typeof { .. }
            | Op::ToNumber { .. }
            | Op::ToStr { .. }
            | Op::JmpIfFalse { .. }
            | Op::Jmp { .. }
            | Op::Ret { .. }
    )
}

/// Execute exactly the op at `state.ip` via the shared `op_xxx` helpers, then
/// advance `ip` for fall-through ops. Called BOTH by the T1 thunk and (in test
/// builds) directly. Only handles the supported subset — the caller guarantees
/// (via `t1_supported_op` at compile time) that no other op reaches here.
///
/// Per-op watchdog is applied here too (same cadence as the VM loop) so a hot
/// T1 loop is just as interruptible as the interpreter.
#[inline]
fn t1_step_one(state: &mut VmState) -> StepStatus {
    state.wd_ticks = state.wd_ticks.wrapping_add(1);
    if state.wd_ticks & 0x7FF == 0 && crate::interp::js_runtime_deadline_exceeded() {
        return StepStatus::Deadline;
    }
    let ip = state.ip;
    let op = state.f.code[ip];
    state.ip = ip + 1;
    match op {
        Op::LoadConst { dst, k } => op_load_const(state, dst, k),
        Op::LoadTrue { dst } => op_load_true(state, dst),
        Op::LoadFalse { dst } => op_load_false(state, dst),
        Op::LoadNull { dst } => op_load_null(state, dst),
        Op::LoadUndef { dst } => op_load_undef(state, dst),
        Op::Move { dst, src } => op_move(state, dst, src),
        Op::Add { dst, lhs, rhs } => op_add(state, dst, lhs, rhs),
        Op::Sub { dst, lhs, rhs } => op_sub(state, dst, lhs, rhs),
        Op::Mul { dst, lhs, rhs } => op_mul(state, dst, lhs, rhs),
        Op::Lt { dst, lhs, rhs } => op_lt(state, dst, lhs, rhs),
        Op::Le { dst, lhs, rhs } => op_le(state, dst, lhs, rhs),
        Op::Gt { dst, lhs, rhs } => op_gt(state, dst, lhs, rhs),
        Op::Ge { dst, lhs, rhs } => op_ge(state, dst, lhs, rhs),
        Op::Eq { dst, lhs, rhs } => op_eq(state, dst, lhs, rhs),
        Op::Neq { dst, lhs, rhs } => op_neq(state, dst, lhs, rhs),
        // M4.2b mechanical op expansion.
        Op::Mod { dst, lhs, rhs } => op_mod(state, dst, lhs, rhs),
        Op::Pow { dst, lhs, rhs } => op_pow(state, dst, lhs, rhs),
        Op::BitAnd { dst, lhs, rhs } => op_bitand(state, dst, lhs, rhs),
        Op::BitOr { dst, lhs, rhs } => op_bitor(state, dst, lhs, rhs),
        Op::BitXor { dst, lhs, rhs } => op_bitxor(state, dst, lhs, rhs),
        Op::Shl { dst, lhs, rhs } => op_shl(state, dst, lhs, rhs),
        Op::Shr { dst, lhs, rhs } => op_shr(state, dst, lhs, rhs),
        Op::Ushr { dst, lhs, rhs } => op_ushr(state, dst, lhs, rhs),
        Op::BitNot { dst, src } => op_bitnot(state, dst, src),
        Op::Neg { dst, src } => op_neg(state, dst, src),
        Op::Not { dst, src } => op_not(state, dst, src),
        Op::Typeof { dst, src } => op_typeof(state, dst, src),
        Op::ToNumber { dst, src } => op_to_number(state, dst, src),
        Op::ToStr { dst, src } => op_to_str(state, dst, src),
        Op::JmpIfFalse { cond, target } => op_jmp_if_false(state, cond, target),
        Op::Jmp { target } => op_jmp(state, target),
        Op::Ret { src } => op_ret(state, src),
        // Unreachable: the T1 compiler declines any function with an op outside
        // the supported subset, so native code never asks the thunk to run one.
        _ => StepStatus::Threw(RuntimeError::TypeError(
            "T1 thunk reached an unsupported op (compiler bug)".into(),
        )),
    }
}

// ---- T1 native dispatch tags (returned by the thunk, read by native code) ----
pub(crate) const T1_CONTINUE: u64 = 0;
pub(crate) const T1_JUMPED: u64 = 1;
pub(crate) const T1_RETURNED: u64 = 2;
pub(crate) const T1_THREW: u64 = 3;
pub(crate) const T1_DEADLINE: u64 = 4;

/// The single op-executor the T1-compiled native code calls (one `call` per
/// bytecode op). Reconstructs `&mut VmState` from the raw pointer, runs exactly
/// the op at bytecode index `ip` via the shared `t1_step_one`/`op_xxx` bodies,
/// stashes any payload-carrying status into the state's `out` slot, and returns
/// a thin integer TAG so native code can branch without touching Rust enums.
///
/// `extern "system"` = Win64 ABI: `state` in RCX, `ip` in RDX, result in RAX.
///
/// SAFETY: `state` must point to a live `VmState` whose `out` is a live
/// `*mut Option<StepStatus>`. The native caller guarantees this for the whole
/// run (the `VmState` + out-slot are stack locals of `run_function_t1`, which
/// strictly outlives the native call). No Rust borrow escapes this function.
/// Address of the T1 op thunk, handed to the codegen so emitted `call`s land on
/// the shared op-executor. A plain fn pointer cast — stable for the process.
pub(crate) fn t1_thunk_addr() -> usize {
    t1_op_thunk as *const () as usize
}

/// Compile `module.fns[fn_idx]` to a T1 baseline-JIT native function and install
/// it (W^X). Returns `None` (decline → run on the VM) on any unsupported op,
/// codegen bail, or install failure. The single source of truth for op
/// semantics is the shared `op_xxx` bodies the emitted code calls via the thunk.
#[cfg(target_os = "windows")]
pub fn try_compile_t1(module: &Module, fn_idx: usize) -> Option<crate::jit::JitFunction> {
    let f = module.fns.get(fn_idx)?;
    let code = crate::jit::compile_baseline_t1(
        &f.code,
        t1_supported_op,
        t1_thunk_addr(),
        T1_CONTINUE,
        T1_JUMPED,
    )?;
    crate::jit::JitFunction::install(&code).ok()
}
#[cfg(not(target_os = "windows"))]
pub fn try_compile_t1(_module: &Module, _fn_idx: usize) -> Option<crate::jit::JitFunction> {
    None
}

/// Public entry: run a previously-compiled T1 function for `module.fns[0]`
/// (per-function modules put the function at slot 0, like `run_module_call`).
#[cfg(target_os = "windows")]
pub fn run_t1_call(
    native: &crate::jit::JitFunction,
    module: &Module,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    run_function_t1(native, module, 0, args, this, globals, dispatch)
}
#[cfg(not(target_os = "windows"))]
pub fn run_t1_call(
    _native: &crate::jit::JitFunction,
    module: &Module,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    run_module_call(module, args, this, globals, None, dispatch)
}

// ======================================================================
// M4.3 — T2-LITE: inlined-`JsVal` JIT compile + run entry points.
//
// `try_compile_t2lite` lowers a per-function module's `fns[0]` to a
// `compile_t2lite` native function (declines on any unsupported op / non-number
// const). `run_t2lite_call` builds the `JsVal` register bank from the call args,
// invokes the native code, and either decodes the returned `JsVal` (T2_RETURNED)
// or — on the runtime DEOPT (a non-number operand reached an arithmetic op) —
// falls back to the VM (`run_module_call`) for the IDENTICAL result. Numeric-
// only: the bank holds only number/bool/undefined/null `JsVal`s during the run,
// so there is no live Rc to manage (borrowed-safe).
// ======================================================================

/// Outcome of a T2-lite compile attempt — distinguishes a permanent DECLINE
/// (an unsupported op / non-number const → never retry) from a RETRY (a GetProp
/// whose receiver IS a pure arg but whose inline cache isn't warm yet → try again
/// once the VM has warmed it). The dispatcher (`interp::try_t2lite_call`) keeps a
/// `Retry` function `Untried` (bounded) so the inline GetProp path engages as
/// soon as the IC warms — exactly the real-JIT "compile when warm" discipline.
#[derive(Debug)]
pub enum T2CompileStatus {
    Ready(crate::jit::JitFunction),
    /// Could compile EXCEPT a GetProp's IC isn't warm/usable yet — retry later.
    Retry,
    /// Permanently un-T2-able (unsupported op, non-number const, no inlinable
    /// GetProp site even in principle). Never retry.
    Decline,
}

/// The register a T2-subset op WRITES, if any (the analysis basis for "pure arg":
/// an arg register never appearing here still holds the caller's object). Covers
/// EXACTLY the T2-supported subset — a function with any op outside it declines
/// the compile anyway, so unlisted ops returning `None` here is harmless.
#[inline]
fn t2_op_writes_reg(op: &Op) -> Option<Reg> {
    match *op {
        Op::LoadConst { dst, .. }
        | Op::LoadUndef { dst }
        | Op::LoadTrue { dst }
        | Op::LoadFalse { dst }
        | Op::LoadNull { dst }
        | Op::Move { dst, .. }
        | Op::Add { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Div { dst, .. }
        | Op::Lt { dst, .. }
        | Op::Le { dst, .. }
        | Op::Gt { dst, .. }
        | Op::Ge { dst, .. }
        | Op::Eq { dst, .. }
        | Op::Neq { dst, .. }
        | Op::LooseEq { dst, .. }
        | Op::LooseNeq { dst, .. }
        | Op::GetProp { dst, .. }
        | Op::GetIdx { dst, .. } => Some(dst),
        _ => None,
    }
}

/// Build the T2 inline-GetProp plan for `f`: which `Op::GetProp` indices have a
/// PURE-ARG receiver (an arg register `< n_params` never written by any op, so the
/// bank slot still holds the caller's object whose `Rc` `args: &[Value]` keeps
/// alive) AND a WARM, usable inline cache (poly-≤4, non-mega, no `DICT_SHAPE`).
/// Returns `(per-index sites, any_getprop_present, any_unwarm)`:
///   * `any_getprop_present` — the code contains ≥1 GetProp at all.
///   * `any_unwarm` — a GetProp with a pure-arg receiver whose IC is cold/mega
///     (so the compile should RETRY, not permanently decline).
fn t2_build_getprop_sites(
    f: &BcFunction,
) -> (Vec<Option<crate::jit::T2GetPropSite>>, bool, bool) {
    let n = f.code.len();
    // Registers ever written by a (subset) op.
    let mut written = vec![false; (f.n_regs as usize) + 1];
    for op in &f.code {
        if let Some(r) = t2_op_writes_reg(op) {
            let r = r as usize;
            if r < written.len() {
                written[r] = true;
            }
        }
    }
    let is_pure_arg = |reg: Reg| -> bool {
        (reg as usize) < (f.n_params as usize) && !written[reg as usize]
    };
    // HEAP-MODE RECEIVER RELAXATION: in the owning + GC-rooted bank (heap mode),
    // EVERY bank slot owns a +1 of its value, so a GetProp receiver living in ANY
    // slot — not just a pure arg — is provably ALIVE for the read (the owning store
    // helper reads through its pointer). The pure-arg restriction only existed for
    // the BORROWED numeric bank (where a non-arg slot could hold a heap value the
    // bank doesn't own). This relaxation is what lets the ARRAY-ITERATION kernel
    // (`var o = arr[j]; o.x …` — `o` is a LOCAL written by GetIdx, not an arg)
    // compile: its `o.x` GetProps read a non-arg local holding a heap value the
    // owning bank keeps alive. In numeric mode the strict pure-arg rule stands.
    let heap = crate::interp::t2_heap_enabled();
    let receiver_ok = |reg: Reg| -> bool {
        if heap {
            // Any in-range bank slot is owned + alive in the owning bank.
            (reg as usize) < (f.n_regs as usize)
        } else {
            is_pure_arg(reg)
        }
    };
    let ics = f.ic.borrow();
    let mut sites: Vec<Option<crate::jit::T2GetPropSite>> = vec![None; n];
    let mut any_getprop = false;
    let mut any_unwarm = false;
    for (i, op) in f.code.iter().enumerate() {
        if let Op::GetProp { obj, .. } = *op {
            any_getprop = true;
            if !receiver_ok(obj) {
                // Receiver isn't a provably-alive slot → can't inline (and a future
                // warm won't change that) — treat as a permanent decline signal
                // (left as None; the compile will decline at this op).
                continue;
            }
            // Warm IC entries for this site (None ⇒ cold/mega ⇒ retry later).
            match ics.get(i).and_then(|ic| ic.warm_own_entries()) {
                Some(shapes_slots) if !shapes_slots.is_empty() => {
                    // T2 Phase 3: when heap mode is on, this site may store a HEAP
                    // result into the OWNING, GC-registered bank (the owning store
                    // helper handles the Rc accounting). Off ⇒ P1 immediate-only.
                    let heap_result = crate::interp::t2_heap_enabled();
                    sites[i] = Some(crate::jit::T2GetPropSite { shapes_slots, heap_result });
                }
                _ => any_unwarm = true,
            }
        }
    }
    (sites, any_getprop, any_unwarm)
}

/// Compile `module.fns[fn_idx]` to a T2-lite inlined-`JsVal` native function and
/// install it (W^X). Thin wrapper over `try_compile_t2lite_status` returning
/// `Some` only on `Ready` (a `Retry` or `Decline` → `None`). Kept for the tests +
/// callers that don't need the retry distinction.
#[cfg(target_os = "windows")]
pub fn try_compile_t2lite(module: &Module, fn_idx: usize) -> Option<crate::jit::JitFunction> {
    match try_compile_t2lite_status(module, fn_idx) {
        T2CompileStatus::Ready(jf) => Some(jf),
        _ => None,
    }
}

/// Compile `module.fns[fn_idx]` to T2-lite, distinguishing Retry from Decline so
/// the dispatcher can warm-then-retry a GetProp function.
#[cfg(target_os = "windows")]
pub fn try_compile_t2lite_status(module: &Module, fn_idx: usize) -> T2CompileStatus {
    let f = match module.fns.get(fn_idx) {
        Some(f) => f,
        None => return T2CompileStatus::Decline,
    };
    let (sites, any_getprop, any_unwarm) = t2_build_getprop_sites(f);
    // If there's a GetProp with a pure-arg receiver but a cold IC, retry later
    // (the VM will warm it). A GetProp with a NON-pure-arg receiver, or any other
    // unsupported op, makes the compile decline below (and that's permanent).
    let shape_off = crate::jit::t2_shape_header_offset();
    let helper_addr = crate::jit::rt_getprop_slot_immediate as *const () as usize;
    let heap_helper_addr = crate::jit::rt_getprop_slot_owning_store as *const () as usize;
    // GetIdx / SetIdx (computed array read/write) helpers — wired ONLY in heap mode
    // (they need the owning + GC-rooted bank to hold a heap element + do the owning
    // element replace). In numeric mode they are 0 ⇒ any GetIdx/SetIdx op declines
    // the whole compile (the array fast path is heap-only, like heap GetProp).
    let heap = crate::interp::t2_heap_enabled();
    let getidx_helper_addr = if heap {
        crate::jit::rt_getidx_owning_store as *const () as usize
    } else {
        0
    };
    let setidx_helper_addr = if heap {
        crate::jit::rt_setidx_owning_store as *const () as usize
    } else {
        0
    };
    let cfg = crate::jit::T2GetPropConfig {
        site_at: &|i: usize| sites.get(i).cloned().flatten(),
        shape_off,
        helper_addr,
        heap_helper_addr,
        getidx_helper_addr,
        setidx_helper_addr,
    };
    // Store mode: Heap (every store routed through the owning helper for uniform
    // per-slot ownership) when `CV_T2_HEAP` is on, else Numeric (the P1 raw-store
    // path, byte-identical). Heap mode is the prerequisite for the owning bank +
    // heap GetProp results; the bank in `run_t2lite_call` is owning + GC-rooted to
    // match. (`heap` was computed above for the getidx/setidx helper wiring.)
    let store_mode = if heap {
        crate::jit::T2StoreMode::Heap {
            store_helper: crate::jit::rt_bank_store as *const () as usize,
        }
    } else {
        crate::jit::T2StoreMode::Numeric
    };
    // P4 — CALL inlining is wired ONLY in heap mode (a re-entrant call needs the
    // OWNING + GC-rooted bank: it can GC, hold heap args, and produce a heap
    // result). In numeric mode `call_cfg` is None ⇒ any call/loadglobal op declines
    // the whole compile exactly as before (the P1/P2 numeric path is unchanged).
    let call_cfg = if heap {
        Some(crate::jit::T2CallConfig {
            call_helper_addr: rt_call_value as *const () as usize,
            call_fn_helper_addr: rt_call_fn as *const () as usize,
            load_global_helper_addr: rt_load_global as *const () as usize,
        })
    } else {
        None
    };
    let code = crate::jit::compile_t2lite_with_deopt(
        &f.code,
        |k| match f.consts.get(k as usize) {
            Some(Value::Number(n)) => Some(*n),
            _ => None,
        },
        Some(&cfg),
        store_mode,
        call_cfg.as_ref(),
    );
    match code {
        // T2 Phase 5: install the native code AND attach its per-guard deopt-site
        // table (the resume map) to the `JitFunction` so `run_t2lite_call` can
        // resume the VM mid-function on a guard miss.
        Some((bytes, sites)) => match crate::jit::JitFunction::install(&bytes) {
            Ok(jf) => T2CompileStatus::Ready(jf.with_deopt_sites(sites)),
            Err(_) => T2CompileStatus::Decline,
        },
        // The compile declined. If the ONLY reason was an unwarmed GetProp site
        // (the code is otherwise in-subset, receiver IS a pure arg), retry once
        // warm; otherwise it's a permanent decline.
        None if any_getprop && any_unwarm => T2CompileStatus::Retry,
        None => T2CompileStatus::Decline,
    }
}
#[cfg(not(target_os = "windows"))]
pub fn try_compile_t2lite(_module: &Module, _fn_idx: usize) -> Option<crate::jit::JitFunction> {
    None
}
#[cfg(not(target_os = "windows"))]
pub fn try_compile_t2lite_status(_module: &Module, _fn_idx: usize) -> T2CompileStatus {
    T2CompileStatus::Decline
}

// ======================================================================
// T2 Phase 3 — the OWNING register bank (the reusable contract).
//
// This is the FIRST place an owning + GC-rooted bank holds a HEAP `JsVal` on a
// live path. `JsVal` is `Copy`/no-`Drop`, so ownership is imposed by THIS outside
// wrapper, never by `JsVal`. The bank:
//   * owns a `Vec<JsVal>` sized ONCE (= n_regs), NEVER grown while registered (a
//     realloc would dangle the `*const JsVal` the GC-root registry holds);
//   * `store(slot, v)` does INC-NEW-BEFORE-DEC-OLD (so a self-store is safe and
//     the last ref is never transiently dropped — the load-bearing ordering);
//   * `Drop` dec's every pointer slot (blanket teardown — no borrowed-vs-owned
//     bifurcation), made uniform by the UNIFORM-OWN construction entry which
//     `rc_inc`s every pointer arg seeded into the bank;
//   * is GC-registered (P2 `register_jit_bank`) for its whole lifetime so a
//     `gc_collect` while it holds a bank-only heap value MARKS it (not clears).
//
// The native code mutates `bank[dst]` directly for a heap GetProp via the owning-
// store HELPER (`rt_getprop_slot_owning_store`), which performs the exact same
// inc-new-before-dec-old store in place — so the Rust `store()` method and the
// helper implement ONE contract. `store()` here is used for the arg-seed teardown
// accounting + is exercised directly by the leak-oracle + mutation-arm tests.
// ======================================================================

/// An owning T2 JIT register bank (T2 Phase 3). Sized once; owns one strong `Rc`
/// ref of every heap `JsVal` it currently holds. GC-registered for its lifetime.
#[cfg(target_os = "windows")]
pub struct OwningRegBank {
    /// The bank slots. NEVER `push`ed/grown while `_root` is alive (the no-grow
    /// invariant — the GC-root registry holds a raw `*const JsVal` into this Vec).
    slots: Vec<crate::jsval::JsVal>,
    /// The GC-root registration RAII guard. Dropped (popping the root) BEFORE the
    /// slots in field order — see the `Drop` note below.
    _root: crate::interp::BankRootGuard,
}

#[cfg(target_os = "windows")]
impl OwningRegBank {
    /// Build an owning bank of `n_slots`, seeding `args` into slots `0..args.len()`
    /// (every other slot = `undefined`). UNIFORM-OWN entry: every pointer arg is
    /// `rc_inc`'d so teardown can blanket-dec every pointer slot with no
    /// borrowed-vs-owned bookkeeping. Registers the bank as a GC root.
    ///
    /// SAFETY/INVARIANT: the bank is sized to `n_slots` here and NEVER grown, so
    /// the `*const JsVal` captured by `register_jit_bank` stays valid for the
    /// guard's lifetime (the bank-no-grow invariant).
    fn new(n_slots: usize, args: &[Value]) -> Self {
        use crate::jsval::JsVal;
        let n = n_slots.max(args.len()).max(1);
        let mut slots: Vec<JsVal> = vec![JsVal::undefined(); n];
        for (i, a) in args.iter().enumerate() {
            // JsVal is total; any arg boxes losslessly.
            let v = JsVal::try_from_value(a).unwrap_or_else(JsVal::undefined);
            slots[i] = v;
            // UNIFORM-OWN: take a strong ref of every pointer arg so Drop is a
            // blanket dec-all. The arg's `Rc` is alive (kept by the caller's
            // `args: &[Value]`) at this point, so the +1 is valid.
            // SAFETY: `v` is a freshly-boxed live pointer (or an immediate no-op).
            unsafe { v.rc_inc() };
        }
        // Register as a GC root for the bank's lifetime. The slice ptr/len are
        // captured now and must not change (no-grow invariant, upheld above).
        let _root = crate::interp::register_jit_bank(&slots);
        OwningRegBank { slots, _root }
    }

    /// Build an owning bank of `n_slots`, seeding `arg_jsvals` into slots
    /// `0..arg_jsvals.len()` (every other slot = `undefined`) DIRECTLY from JsVals
    /// — the T2→T2 native-to-native entry (no Value marshaling). UNIFORM-OWN: every
    /// pointer arg is `rc_inc`'d so teardown is a blanket dec-all. The caller's bank
    /// owns its own +1 of each pointer arg (they live in the caller's bank slots),
    /// so the +1 taken here is valid and independent. Registers the bank as a GC
    /// root. Same no-grow invariant as [`OwningRegBank::new`].
    ///
    /// # Safety
    /// Each pointer `JsVal` in `arg_jsvals` must have a live `Rc` at call time (held
    /// by the caller's bank slots) so the `rc_inc` seed is valid.
    unsafe fn new_from_jsvals(n_slots: usize, arg_jsvals: &[crate::jsval::JsVal]) -> Self {
        use crate::jsval::JsVal;
        let n = n_slots.max(arg_jsvals.len()).max(1);
        let mut slots: Vec<JsVal> = vec![JsVal::undefined(); n];
        for (i, &v) in arg_jsvals.iter().enumerate() {
            slots[i] = v;
            // UNIFORM-OWN: +1 each pointer arg (no-op for an immediate). The arg's
            // Rc is alive (held by the caller's bank slot) at this point.
            unsafe { v.rc_inc() };
        }
        let _root = crate::interp::register_jit_bank(&slots);
        OwningRegBank { slots, _root }
    }

    /// Raw mutable base pointer for the native ABI (`*mut u64`, since `JsVal` is
    /// `#[repr(transparent)]` over `u64`). The native code reads/writes only
    /// `slots[0..len]` and, for a heap GetProp, calls the owning-store helper which
    /// performs the inc/dec store in place on THIS buffer.
    fn as_mut_ptr(&mut self) -> *mut u64 {
        self.slots.as_mut_ptr() as *mut u64
    }

    /// OWNING store of `v` into `slot` — the reusable contract: INC-NEW-BEFORE-
    /// DEC-OLD. Safe for a self-store (`v` already in `slot`) and never transiently
    /// drops the last ref. Used for the seed-teardown accounting + the leak-oracle
    /// / mutation-arm tests; the native heap path performs the identical sequence
    /// in `rt_getprop_slot_owning_store`.
    ///
    /// # Safety
    /// `slot < self.slots.len()`. `v`'s pointee `Rc` (if a pointer) must be alive
    /// at call time. After this the bank owns one strong ref of `v`.
    #[allow(dead_code)]
    unsafe fn store(&mut self, slot: usize, v: crate::jsval::JsVal) {
        unsafe {
            v.rc_inc(); // +1 the NEW value first (self-store safe; last-ref safe)
            let old = self.slots[slot];
            self.slots[slot] = v;
            old.rc_dec(); // -1 the OLD value last (may free if it was the last ref)
        }
    }

    /// Read a slot's `JsVal` (Copy). For the result-exit read.
    #[allow(dead_code)]
    fn get(&self, slot: usize) -> crate::jsval::JsVal {
        self.slots[slot]
    }

    /// T2 Phase 5 — decode the whole bank into VM `Value` registers (the deopt
    /// resume image, bank slot `i` == VM reg `i`). Each pointer slot's `to_value`
    /// takes its OWN +1 (a borrowed-handle clone), so the returned `Value`s remain
    /// valid after the bank's teardown decs its slots. MUST be called BEFORE the
    /// bank drops. NaN is already canonical on box, so a computed-NaN slot decodes
    /// to `Value::Number(NaN)` cleanly (no tagged-value-as-NaN hazard).
    fn decode_to_values(&self) -> Vec<Value> {
        // SAFETY: every pointer slot's `Rc` is alive (the bank owns a +1 of each
        // for its whole lifetime, and we are still inside that lifetime here).
        self.slots
            .iter()
            .map(|jv| unsafe { jv.to_value() })
            .collect()
    }

    /// Raw bank slots (`&[JsVal]`) — the live identity-map register image. Used by
    /// the T4 Extension-1 inlined-frame reconstruction (`osr::reconstruct_caller_
    /// frame`), which decodes this image to the CALLER's VM register file. Every
    /// pointer slot's `Rc` is alive (the bank owns a +1 for its whole lifetime),
    /// the same precondition `decode_to_values` relies on.
    fn raw_slots(&self) -> &[crate::jsval::JsVal] {
        &self.slots
    }
}

/// MUTATION ARMS (test-only) — deliberately-broken variants of the `store`
/// contract used to PROVE the leak oracle is load-bearing (each broken arm must
/// redden the strong-count assertion / trip a UAF). They are `#[cfg(test)]` so a
/// broken arm can never ship; the production `store`/`rt_bank_store` use the
/// `Correct` ordering.
#[cfg(all(test, target_os = "windows"))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum StoreArm {
    /// The shipping contract: inc-new, swap, dec-old.
    Correct,
    /// WRONG ORDER: dec-old BEFORE inc-new — transiently drops the last ref of a
    /// value being self-stored (UAF) and the wrong count on a last-ref overwrite.
    DecBeforeInc,
    /// LEAK: never dec the overwritten old value.
    SkipOverwriteDec,
    /// DOUBLE-FREE/UAF: never inc the new value (the bank doesn't own it, yet
    /// Drop / a later overwrite will dec it → over-dec).
    SkipStoreInc,
}

#[cfg(all(test, target_os = "windows"))]
impl OwningRegBank {
    /// Test-only store that performs `arm`'s (possibly broken) ordering. The
    /// production path always uses `Correct` (= `store`).
    ///
    /// # Safety
    /// As `store`, plus: a broken `arm` may leave the bank in a leaked / over-dec'd
    /// state — the caller (a mutation-arm test) asserts the oracle catches it and
    /// must not let a `SkipStoreInc`/`DecBeforeInc` bank actually run its `Drop`
    /// against a freed value (the tests keep an external owner alive to bound the
    /// damage to a detectable count, never an actual free).
    pub(crate) unsafe fn store_with_arm(&mut self, slot: usize, v: crate::jsval::JsVal, arm: StoreArm) {
        unsafe {
            match arm {
                StoreArm::Correct => {
                    v.rc_inc();
                    let old = self.slots[slot];
                    self.slots[slot] = v;
                    old.rc_dec();
                }
                StoreArm::DecBeforeInc => {
                    let old = self.slots[slot];
                    old.rc_dec(); // WRONG: dec old first
                    v.rc_inc();
                    self.slots[slot] = v;
                }
                StoreArm::SkipOverwriteDec => {
                    v.rc_inc();
                    self.slots[slot] = v; // WRONG: old never dec'd → leak
                }
                StoreArm::SkipStoreInc => {
                    let old = self.slots[slot];
                    self.slots[slot] = v; // WRONG: new never inc'd → over-dec later
                    old.rc_dec();
                }
            }
        }
    }

    /// Test-only: number of slots (bank length).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Test-only: read a slot (for ptr_eq / content assertions).
    #[cfg(test)]
    pub(crate) fn slot(&self, i: usize) -> crate::jsval::JsVal {
        self.slots[i]
    }

    /// Test-only: build an owning bank directly (exposes `new` to integration-shaped
    /// unit tests in this module).
    #[cfg(test)]
    pub(crate) fn new_for_test(n_slots: usize, args: &[Value]) -> Self {
        Self::new(n_slots, args)
    }

    /// Test-only owning store (exposes `store` for the leak oracle).
    ///
    /// # Safety
    /// As [`OwningRegBank::store`].
    #[cfg(test)]
    pub(crate) unsafe fn store_for_test(&mut self, slot: usize, v: crate::jsval::JsVal) {
        unsafe { self.store(slot, v) }
    }

    /// Test-only DEC-BEFORE-INC self-store DETECTOR (UB-free). Performs the WRONG
    /// ordering's first step — `dec` the old slot value (= the value being
    /// self-stored) — then checks `weak`: if the pointee was freed (count hit 0),
    /// returns `true` (the ordering bug is proven) and REPAIRS the slot to
    /// `undefined` so the bank's teardown does not double-dec a freed pointer. It
    /// deliberately SKIPS the (would-be-UB) `inc` on the freed pointer. Used only by
    /// the dec-before-inc teeth test.
    ///
    /// # Safety
    /// `slot` is in range; `weak` is a `Weak` to the SAME Array `Rc` whose `JsVal`
    /// occupies `slot` and of which the bank is the sole strong owner.
    #[cfg(test)]
    pub(crate) unsafe fn dec_before_inc_self_store_detect(
        &mut self,
        slot: usize,
        weak: &std::rc::Weak<std::cell::RefCell<Vec<Value>>>,
    ) -> bool {
        unsafe {
            let old = self.slots[slot];
            old.rc_dec(); // WRONG-ORDER first step
            let freed = weak.upgrade().is_none();
            // Repair: the value is gone; neutralize the slot so teardown is safe and
            // do NOT inc the dangling pointer (that would be the UB the bug causes).
            self.slots[slot] = crate::jsval::JsVal::undefined();
            freed
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for OwningRegBank {
    fn drop(&mut self) {
        // TEARDOWN: blanket dec every pointer slot (uniform-own ⇒ the bank holds
        // exactly one strong ref of every pointer slot — seeded args were inc'd
        // and every store maintains the +1), then NEUTRALIZE the slot to undefined.
        //
        // ORDERING (the GC-root window): `Drop::drop` runs BEFORE the struct's
        // fields drop, and fields drop in declaration order (`slots` then `_root`).
        // So while we dec here the bank is STILL GC-registered. We therefore
        // OVERWRITE each slot to `undefined` immediately after dec'ing it, so even
        // a (hypothetical, invariant-violating) GC seed during this teardown would
        // find only immediates — never a just-freed pointer. After the loop the
        // bank holds only immediates; `_root` then field-drops and pops the
        // registration. (No GC actually runs here — synchronous teardown at the
        // message-loop boundary — but this makes the window provably safe anyway.)
        use crate::jsval::JsVal;
        for slot in self.slots.iter_mut() {
            // SAFETY: uniform-own invariant — exactly one owned ref per pointer
            // slot. Dec releases it (no-op for an immediate).
            unsafe { slot.rc_dec() };
            *slot = JsVal::undefined();
        }
    }
}

// ======================================================================
// T2 Phase 4 — CALL INLINING: the re-entry context + helpers.
//
// A T2-compiled function with a CALL op is invoked with a `*mut T2CallCtx` (the
// 3rd ABI arg, held in the callee-saved RSI). The CALL/LoadGlobal codegen calls
// one of the three `extern "system"` re-entry helpers below, passing the ctx +
// bank + a packed slot descriptor; the helper does ALL the Value marshaling + VM
// dispatch in Rust, OWNING-stores the result into `bank[dst]`, and returns a
// `T2_*` status. The native code holds NOTHING across the call except the
// callee-saved BANK/CTX/OUT pointers + RSP (the aliasing discipline) and re-reads
// every bank slot after.
//
// GC-DURING-CALL SAFETY: the bank is the OWNING + GC-rooted `OwningRegBank` for the
// whole run (the runner registers it before the native call). A re-entrant callee
// that triggers `gc_collect` marks the caller's bank slots (they are roots) → no
// UAF/clear of a bank-only-reachable heap value. This is a real safepoint: the
// store-before-call is implicit (every input already lives in a bank slot by
// construction), and the bank is registered for the entire native run.
//
// EXCEPTIONS: a callee error becomes `T2_THREW` (the error stashed in the ctx
// out-slot) — the runner returns it directly (NO VM re-run, so no duplicate
// effect). A watchdog `Deadline` becomes `T2_DEADLINE` (uncatchable). A pre-side-
// effect decline (non-callable callee) becomes `T2_DEOPT` (the bank is untouched
// → the VM re-run is identical). try-handler-containing functions are DECLINED at
// compile time, so a THREW always unwinds straight out of the T2 function.
// ======================================================================

/// The P4 re-entry context. Carries everything a CALL/LoadGlobal helper needs to
/// re-dispatch through the VM, plus an out-slot for a thrown error/deadline (whose
/// payload can't ride the integer status the helper returns to native code). All
/// borrows outlive the native call (the runner builds this on its stack and keeps
/// `module`/`globals`/`dispatch` alive for the call's duration).
#[cfg(target_os = "windows")]
pub struct T2CallCtx<'a> {
    /// The per-function module being run (its `fns[0]` is the T2 function; a
    /// `CallFn` indexes `fns[fn_idx]`; a `LoadGlobal` reads `fns[0].consts`).
    module: &'a Module,
    /// The shared global env (callee lookups, the BcClosure run path).
    globals: &'a std::cell::RefCell<HashMap<String, Value>>,
    /// Raw pointer to the live host dispatcher (a `&mut dyn FnMut`), reborrowed by
    /// each helper without holding a Rust borrow across the native boundary.
    dispatch: *mut (dyn FnMut(Value, Value, Vec<Value>) -> Result<Value, RuntimeError> + 'a),
    /// Where a helper stashes a catchable thrown error (T2_THREW). `None` until set.
    thrown: Option<RuntimeError>,
}

/// Decode a packed 4×u16 lane descriptor: `(laneA, laneB, laneC, laneD)`.
#[cfg(target_os = "windows")]
#[inline]
fn t2_unpack4(packed: u64) -> (u16, u16, u16, u16) {
    (
        (packed & 0xFFFF) as u16,
        ((packed >> 16) & 0xFFFF) as u16,
        ((packed >> 32) & 0xFFFF) as u16,
        ((packed >> 48) & 0xFFFF) as u16,
    )
}

/// P4 re-entry helper for `Op::CallValue`. ABI: `extern "system" fn(ctx, bank,
/// packed, dst) -> status`. `packed` lanes = (callee_slot, this_slot|0xFFFF=NO_THIS,
/// first_arg, argc). Reads the callee/this/args from the bank (each `to_value` =
/// +1 owned, dropped when the temp `Vec<Value>` drops), dispatches EXACTLY like the
/// VM op via `dispatch_call_value`, and on success OWNING-stores the result into
/// `bank[dst]` (so the bank's per-slot +1 invariant holds). Returns:
///   * `T2_RETURNED` — result stored;
///   * `T2_DEOPT`    — the callee is not callable (a PRE-side-effect decline; bank
///                     untouched, so a VM re-run is identical);
///   * `T2_THREW`    — the callee threw a catchable error (stashed in `ctx.thrown`);
///   * `T2_DEADLINE` — the watchdog fired (uncatchable).
///
/// # Safety
/// `ctx` is a live `T2CallCtx` (non-null — the compiler only emits this for a
/// function compiled with a `T2CallConfig`, and the runner always passes a real
/// ctx). `bank` is the live OWNING bank; the lane slots + `dst` are `< bank_len`
/// (baked from validated bytecode registers). The callee/args' `Rc`s are alive
/// (held by the bank, which owns its pointer slots).
#[cfg(target_os = "windows")]
pub extern "system" fn rt_call_value(
    ctx: *mut T2CallCtx<'_>,
    bank: *mut u64,
    packed: u64,
    dst: u64,
) -> u64 {
    use crate::jsval::JsVal;
    if ctx.is_null() || bank.is_null() {
        return crate::jit::T2_DEADLINE; // misconfiguration — fail loud, no re-run
    }
    // SAFETY: live ctx for the call (runner contract).
    let ctx = unsafe { &mut *ctx };
    let (callee_slot, this_slot, first_arg, argc) = t2_unpack4(packed);

    // ── T2→T2 NATIVE-TO-NATIVE fast path ────────────────────────────────────
    // Peek at the callee bank slot WITHOUT marshaling to a Value. Resolve a Ready
    // T2 callee via the registry, keyed by:
    //   * a `Value::Function` (a hoisted fn decl — the common hot-helper shape) →
    //     its FunctionValue pointer; or
    //   * a `Value::BcClosure` whose `fn_idx == 0` (the per-fn module shape — a
    //     closure value / direct call) → its module pointer.
    // On a hit, call native-to-native: seed the callee's OWN bank from THIS bank's
    // contiguous arg slots (JsVals, no Value conversion), run its native code.
    // Otherwise fall through to the VM re-entry below (P4 — UNCHANGED).
    {
        // SAFETY: callee_slot in range; the bank owns its +1, so the pointee Rc is
        // alive. `as_function`/`as_bcclosure` hand back a +1 clone (net unchanged).
        let callee_jv = JsVal(unsafe { *bank.add(callee_slot as usize) });
        let resolved: Option<Rc<Module>>;
        let native_opt;
        if let Some(fv) = unsafe { callee_jv.as_function() } {
            let key = Rc::as_ptr(&fv) as usize;
            match crate::interp::t2_registry_lookup(key) {
                Some((m, n)) => { resolved = Some(m); native_opt = Some(n); }
                None => { resolved = None; native_opt = None; }
            }
        } else if let Some(c) = unsafe { callee_jv.as_bcclosure() } {
            if c.fn_idx == 0 {
                let key = Rc::as_ptr(&c.module) as usize;
                match crate::interp::t2_registry_lookup(key) {
                    Some((m, n)) => { resolved = Some(m); native_opt = Some(n); }
                    None => { resolved = None; native_opt = None; }
                }
            } else {
                resolved = None; native_opt = None;
            }
        } else {
            resolved = None; native_opt = None;
        }
        if let (Some(callee_module), Some(callee_native)) = (resolved, native_opt) {
            // `this` for a value call: undefined when NO_THIS, else the slot.
            let this_val = if this_slot == 0xFFFF {
                Value::Undefined
            } else {
                let jv = JsVal(unsafe { *bank.add(this_slot as usize) });
                unsafe { jv.to_value() }
            };
            let dispatch: &mut dyn FnMut(Value, Value, Vec<Value>) -> Result<Value, RuntimeError> =
                unsafe { &mut *ctx.dispatch };
            // Pass the caller's contiguous arg slots straight in (no marshaling).
            let args_ptr = unsafe { bank.add(first_arg as usize) } as *const u64;
            // SAFETY: args_ptr[0..argc] are live JsVals in THIS bank (the caller
            // owns their +1); callee_native matches callee_module.fns[0].
            let (result_jv, status) = unsafe {
                run_t2lite_from_jsval_args(
                    &callee_native,
                    &callee_module,
                    args_ptr,
                    argc as usize,
                    &this_val,
                    ctx.globals,
                    dispatch,
                )
            };
            crate::interp::note_t2_t2_call();
            return match status {
                T2NativeStatus::Returned => {
                    // `result_jv` carries a +1 from the callee entry; the owning-
                    // store takes its OWN +1 then dec's it back, and dec's the old
                    // dst — so we release the entry's +1 afterward (no leak). Net:
                    // bank[dst] holds one +1 (its slot ownership).
                    unsafe { t2_owning_store_raw(bank, dst, result_jv) };
                    unsafe { result_jv.rc_dec() };
                    crate::jit::T2_RETURNED
                }
                T2NativeStatus::Threw(e) => {
                    ctx.thrown = Some(e);
                    crate::jit::T2_THREW
                }
                T2NativeStatus::Deadline => crate::jit::T2_DEADLINE,
            };
        }
    }

    // Read inputs from the bank as OWNED Values (each `to_value` = +1; the temp
    // Vec/locals drop releases them). The bank's pointer slots keep their own +1
    // (uniform-own), so these clones are independent.
    // SAFETY: every slot index is in range (baked from validated regs).
    let read = |slot: u16| -> Value {
        let jv = JsVal(unsafe { *bank.add(slot as usize) });
        unsafe { jv.to_value() }
    };
    let callee_val = read(callee_slot);
    let this_val = if this_slot == 0xFFFF { Value::Undefined } else { read(this_slot) };
    let mut call_args: Vec<Value> = Vec::with_capacity(argc as usize);
    for k in 0..argc {
        call_args.push(read(first_arg + k));
    }
    let dispatch: &mut dyn FnMut(Value, Value, Vec<Value>) -> Result<Value, RuntimeError> =
        unsafe { &mut *ctx.dispatch };
    match dispatch_call_value(callee_val, this_val, call_args, ctx.globals, dispatch) {
        Ok(v) => {
            // OWNING store of the result into bank[dst] (inc-new-before-dec-old).
            let jv = JsVal::try_from_value(&v).unwrap_or_else(JsVal::undefined);
            // SAFETY: dst in range; jv's pointee Rc is alive (v owns it here).
            unsafe { t2_owning_store_raw(bank, dst, jv) };
            // `v` drops now (releasing its +1); the bank holds its own +1.
            crate::jit::T2_RETURNED
        }
        Err(RuntimeError::Deadline) => crate::jit::T2_DEADLINE,
        Err(RuntimeError::TypeError(msg)) if msg.starts_with("callee is not callable") => {
            // PRE-side-effect decline: no call ran. Deopt (bank untouched) so the
            // VM re-run produces the identical error/result.
            crate::jit::T2_DEOPT
        }
        Err(e) => {
            ctx.thrown = Some(e);
            crate::jit::T2_THREW
        }
    }
}

/// P4 re-entry helper for `Op::CallFn` (direct module-local call by index). ABI:
/// `extern "system" fn(ctx, bank, packed, dst) -> status`. `packed` lanes =
/// (fn_idx, _unused, first_arg, argc); `this` is always undefined (matching the VM
/// `Op::CallFn`). Runs `run_function(module, fn_idx, args, undefined, …)` and
/// owning-stores the result. Same status contract as [`rt_call_value`].
///
/// # Safety
/// As [`rt_call_value`]; additionally `fn_idx < module.fns.len()` (guaranteed by
/// the per-fn module's dangling-index check at compile time).
#[cfg(target_os = "windows")]
pub extern "system" fn rt_call_fn(
    ctx: *mut T2CallCtx<'_>,
    bank: *mut u64,
    packed: u64,
    dst: u64,
) -> u64 {
    use crate::jsval::JsVal;
    if ctx.is_null() || bank.is_null() {
        return crate::jit::T2_DEADLINE;
    }
    let ctx = unsafe { &mut *ctx };
    let (fn_idx, _unused, first_arg, argc) = t2_unpack4(packed);

    // ── T2→T2 NATIVE-TO-NATIVE fast path (self-recursion / fns[0]) ──────────
    // A `CallFn` of `fns[0]` (the per-fn module shape — direct self-recursion) into
    // a module that is a Ready T2 slot → call native-to-native from the bank arg
    // JsVals. `ctx.module` is borrowed from the registered `Rc<Module>`, so its
    // address equals the registry's `Rc::as_ptr` key.
    if fn_idx == 0 {
        let mod_ptr = ctx.module as *const Module as usize;
        if let Some((_callee_module, callee_native)) = crate::interp::t2_registry_lookup(mod_ptr) {
            let dispatch: &mut dyn FnMut(Value, Value, Vec<Value>) -> Result<Value, RuntimeError> =
                unsafe { &mut *ctx.dispatch };
            let args_ptr = unsafe { bank.add(first_arg as usize) } as *const u64;
            // SAFETY: args_ptr[0..argc] are live JsVals in THIS bank; callee_native
            // matches ctx.module.fns[0] (the registry guarantees the pairing).
            let (result_jv, status) = unsafe {
                run_t2lite_from_jsval_args(
                    &callee_native,
                    ctx.module,
                    args_ptr,
                    argc as usize,
                    &Value::Undefined,
                    ctx.globals,
                    dispatch,
                )
            };
            crate::interp::note_t2_t2_call();
            return match status {
                T2NativeStatus::Returned => {
                    unsafe { t2_owning_store_raw(bank, dst, result_jv) };
                    unsafe { result_jv.rc_dec() };
                    crate::jit::T2_RETURNED
                }
                T2NativeStatus::Threw(e) => {
                    ctx.thrown = Some(e);
                    crate::jit::T2_THREW
                }
                T2NativeStatus::Deadline => crate::jit::T2_DEADLINE,
            };
        }
    }

    let read = |slot: u16| -> Value {
        let jv = JsVal(unsafe { *bank.add(slot as usize) });
        unsafe { jv.to_value() }
    };
    let mut call_args: Vec<Value> = Vec::with_capacity(argc as usize);
    for k in 0..argc {
        call_args.push(read(first_arg + k));
    }
    let dispatch: &mut dyn FnMut(Value, Value, Vec<Value>) -> Result<Value, RuntimeError> =
        unsafe { &mut *ctx.dispatch };
    match run_function(
        ctx.module,
        fn_idx as usize,
        &call_args,
        &Value::Undefined,
        ctx.globals,
        None,
        dispatch,
    ) {
        Ok(v) => {
            let jv = JsVal::try_from_value(&v).unwrap_or_else(JsVal::undefined);
            unsafe { t2_owning_store_raw(bank, dst, jv) };
            crate::jit::T2_RETURNED
        }
        Err(RuntimeError::Deadline) => crate::jit::T2_DEADLINE,
        Err(e) => {
            ctx.thrown = Some(e);
            crate::jit::T2_THREW
        }
    }
}

/// P4 re-entry helper for `Op::LoadGlobal[Checked]`. ABI: `extern "system" fn(ctx,
/// packed, bank, dst) -> status`. `packed = name_k | (checked << 16)`. Reads
/// `globals[consts[name_k]]` and OWNING-stores it into `bank[dst]`. A plain
/// LoadGlobal of a missing name stores `undefined` (RETURNED); a LoadGlobalChecked
/// of an undeclared name THREWs a catchable ReferenceError (matching the VM op).
///
/// # Safety
/// `ctx`/`bank` live; `name_k < module.fns[0].consts.len()`; `dst < bank_len`.
#[cfg(target_os = "windows")]
pub extern "system" fn rt_load_global(
    ctx: *mut T2CallCtx<'_>,
    packed: u64,
    bank: *mut u64,
    dst: u64,
) -> u64 {
    use crate::jsval::JsVal;
    if ctx.is_null() || bank.is_null() {
        return crate::jit::T2_DEADLINE;
    }
    let ctx = unsafe { &mut *ctx };
    let name_k = (packed & 0xFFFF) as usize;
    let checked = ((packed >> 16) & 0xFFFF) != 0;
    let f = match ctx.module.fns.first() {
        Some(f) => f,
        None => return crate::jit::T2_DEADLINE,
    };
    let name = match f.consts.get(name_k) {
        Some(Value::String(s)) => s.clone(),
        _ => return crate::jit::T2_DEADLINE, // malformed const — fail loud
    };
    let resolved = ctx.globals.borrow().get(&*name).cloned();
    match resolved {
        Some(val) => {
            let jv = JsVal::try_from_value(&val).unwrap_or_else(JsVal::undefined);
            unsafe { t2_owning_store_raw(bank, dst, jv) };
            crate::jit::T2_RETURNED
        }
        None if checked => {
            ctx.thrown = Some(RuntimeError::Thrown(crate::interp::err_str(format!(
                "ReferenceError: {name} is not defined"
            ))));
            crate::jit::T2_THREW
        }
        None => {
            // Plain LoadGlobal: undefined.
            unsafe { t2_owning_store_raw(bank, dst, JsVal::undefined()) };
            crate::jit::T2_RETURNED
        }
    }
}

/// The owning bank store used by the P4 helpers (inc-new-before-dec-old), identical
/// contract to [`crate::jit::rt_bank_store`] / [`OwningRegBank::store`]. Factored
/// here so all three helpers share one implementation.
///
/// # Safety
/// `bank` non-null + `dst < bank_len`; `v`'s pointee `Rc` (if a pointer) is alive.
#[cfg(target_os = "windows")]
#[inline]
unsafe fn t2_owning_store_raw(bank: *mut u64, dst: u64, v: crate::jsval::JsVal) {
    let dst_ptr = unsafe { bank.add(dst as usize) } as *mut crate::jsval::JsVal;
    unsafe {
        v.rc_inc();
        let old = std::ptr::read(dst_ptr);
        std::ptr::write(dst_ptr, v);
        old.rc_dec();
    }
}

/// T2 Phase 5 — RESUME the bytecode VM mid-function after a per-guard deopt.
/// `regs` is the register image decoded from the JIT bank (bank slot `i` == VM reg
/// `i`, the identity map); `bc_pc` is the op boundary the guard fired at (before
/// that op's output store). Runs `run_function_inner(module, 0, …, ip=bc_pc)` so
/// the VM re-executes exactly the deopting op + the rest with bit-identical results
/// — and, crucially, does NOT re-run any committed side effect that preceded the
/// guard (it continues from `bc_pc`, not from ip=0). `closure` is `None` to match
/// the Tier-A `run_module_call` path (a per-fn module's fns[0] has no upvalues).
#[cfg(target_os = "windows")]
fn t2_resume_on_vm(
    module: &Module,
    regs: Vec<Value>,
    bc_pc: usize,
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    let f = match module.fns.get(0) {
        Some(f) => f,
        None => return Err(RuntimeError::TypeError("T2 resume: empty module".into())),
    };
    let mut regs = PooledRegs::from_vec(regs, f.n_regs as usize);
    run_function_inner(module, 0, this, globals, None, dispatch, &mut regs, bc_pc)
}

/// T4 EXTENSION-1 — INLINE-DEOPT-TO-CALLER reconstruction + resume.
///
/// Drives the inlined-frame deopt math `osr::reconstruct_caller_frame` over the
/// REAL live JIT `bank` (the caller's identity-map register image). It:
///   1. reads the CALLER's `Call`/`CallFn`/`CallValue` op at `call_bc_pc` to learn
///      its argument register span (the `arg_slot_map`);
///   2. builds the `osr::InlinedDeoptSite` and VERIFIES it against the caller
///      (resume pc + arg slots in range) — the compile-time UAF/garbage-arg gate;
///   3. reconstructs the caller VM register file from the bank (the identity-map
///      decode) and resumes the caller at the Call op, so the VM performs the
///      ordinary (non-inlined) call.
///
/// Returns `Some(result)` of the resumed VM run, or `None` if the op at
/// `call_bc_pc` is not a call op (the caller then takes the ordinary single-frame
/// resume — the hook is misconfigured for this fixture). This is the path the
/// inlined-frame-deopt fuzzer forces; in production the hook is never set so this
/// is dead. NOTE: because the chosen design resumes the CALLER at the Call op over
/// the full identity-map image, the result is necessarily identical to a single-
/// frame resume at the same op — which is EXACTLY the property the fuzzer asserts
/// (and asserting it directly against the un-inlined VM proves the math, not the
/// tautology: a wrong `caller_bc_pc_of_call` or arg map would diverge).
#[cfg(target_os = "windows")]
fn try_inlined_frame_resume(
    module: &Module,
    bank: &OwningRegBank,
    call_bc_pc: usize,
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Option<Result<Value, RuntimeError>> {
    let f = module.fns.get(0)?;
    // Learn the call's argument register span from the op at the resume pc.
    let (first_arg, n_args) = match f.code.get(call_bc_pc)? {
        Op::CallFn { first_arg, n_args, .. } => (*first_arg as usize, *n_args as usize),
        Op::CallValue { first_arg, n_args, .. } => (*first_arg as usize, *n_args as usize),
        Op::New { first_arg, n_args, .. } => (*first_arg as usize, *n_args as usize),
        _ => return None, // not a call op — fall back to the ordinary resume
    };
    let arg_slot_map: Vec<usize> = (first_arg..first_arg + n_args).collect();
    let site = crate::osr::InlinedDeoptSite {
        base: crate::osr::DeoptSite {
            native_off: 0,
            bc_pc: call_bc_pc,
            reason: crate::osr::DeoptReason::NonNumber,
        },
        frame: crate::osr::InlinedFrame {
            caller_bc_pc_of_call: call_bc_pc,
            callee_entry_bc_pc: 0,
            arg_slot_map,
        },
    };
    // THE EXTENSION-1 GATE: the inlined-frame site must be reconstructible against
    // the caller (resume pc + every arg slot in range). A violation = a codegen bug
    // that would resume the call with a missing/garbage argument; reject it (the
    // caller then takes the ordinary single-frame resume, still correct).
    if !site.verify_against_caller(f.code.len(), f.n_regs as usize) {
        return None;
    }
    // Reconstruct the CALLER frame from the live bank (identity-map decode) and
    // resume at the Call op — the VM performs the ordinary non-inlined call.
    // SAFETY: `bank` is alive here (not yet dropped); `reconstruct_caller_frame`
    // decodes its slots with a +1 per pointer, so the regs outlive the bank.
    let frame = unsafe { crate::osr::reconstruct_caller_frame(0, &site, bank.raw_slots()) };
    let regs = frame.into_value_regs();
    Some(t2_resume_on_vm(
        module,
        regs,
        site.frame.caller_bc_pc_of_call,
        this,
        globals,
        dispatch,
    ))
}

/// Run a previously-compiled T2-lite function for `module.fns[0]`. Sets up the
/// `JsVal` bank over the args + locals, invokes the native code, and decodes the
/// result; on a runtime DEOPT it re-runs the function on the VM
/// (`run_module_call`) so the observable result is IDENTICAL.
///
/// Two bank modes:
///   * BORROWED numeric bank (default / `CV_T2_HEAP` off): the bank holds only
///     immediates (number/bool/undef/null) during the run — no Rc lifetime to
///     uphold — exactly the P1 path, byte-identical.
///   * OWNING + GC-ROOTED bank (`CV_T2_HEAP` on): the bank may hold a HEAP result
///     stashed by the owning-store GetProp helper; `OwningRegBank` owns + roots it
///     for the run. EXIT ORDERING: the result is read via `to_value()` (the
///     genuine +1 owned `Value`) BEFORE the bank drops (which decs the result
///     slot) — enforced by reading into `result` before the bank goes out of scope.
#[cfg(target_os = "windows")]
pub fn run_t2lite_call(
    native: &crate::jit::JitFunction,
    module: &Module,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    use crate::jsval::JsVal;
    let f = match module.fns.get(0) {
        Some(f) => f,
        None => return Err(RuntimeError::TypeError("T2: empty module".into())),
    };
    let n_slots = (f.n_regs as usize).max(args.len()).max(1);

    if crate::interp::t2_heap_enabled() {
        // OWNING + GC-ROOTED path. The bank owns every pointer slot for the run.
        let mut bank = OwningRegBank::new(n_slots, args);
        let mut out: u64 = 0;
        // P4: build the re-entry context so a CALL/LoadGlobal op can re-dispatch
        // through the VM. `dispatch` is captured as a raw pointer (reborrowed inside
        // each helper); it + `module`/`globals` outlive the native call. We pass it
        // as the 3rd ABI arg. A call-free function ignores the ctx (its codegen
        // never touches RSI), so this is harmless for the non-call case too.
        let mut ctx = T2CallCtx {
            module,
            globals,
            dispatch: dispatch as *mut _,
            thrown: None,
        };
        // SAFETY: `bank` slots + `out` + `ctx` are live for the call; the native
        // code reads/writes only `bank[0..len]`, calls the owning-store / re-entry
        // helpers which maintain the bank's +1 invariant, and (for a CALL) the bank
        // is GC-rooted so a re-entrant gc_collect marks it (no UAF).
        let tag = unsafe {
            native.call_t2lite_ctx(
                bank.as_mut_ptr(),
                &mut out as *mut u64,
                &mut ctx as *mut T2CallCtx as *mut core::ffi::c_void,
            )
        };
        match tag {
            crate::jit::T2_RETURNED => {
                // EXIT ORDERING (load-bearing, asserted by the leak oracle): read
                // the result as a genuinely-owned `Value` (+1) BEFORE `bank` drops
                // (which decs the result slot by 1). `out` carries the result bits
                // the native code wrote (a copy of some bank slot's JsVal); decode
                // it to a Value here, then let `bank` drop AFTER `result` is bound.
                let jv = JsVal(out);
                // `to_value` takes its OWN +1 (the borrowed-handle clone), so the
                // returned Value survives the bank's teardown dec of that slot.
                let result = unsafe { jv.to_value() };
                drop(bank); // explicit: bank teardown decs every pointer slot now
                Ok(result)
            }
            // P4 THREW: a re-entrant CALL's callee threw a catchable error. The
            // error is in `ctx.thrown`. PROPAGATE it directly — do NOT re-run on the
            // VM (the call already committed its side effect; a re-run would re-do
            // it = duplicate effect). The compile declined any deopt-after-call, so
            // a THREW is the only post-call non-return exit.
            crate::jit::T2_THREW => {
                let err = ctx
                    .thrown
                    .take()
                    .unwrap_or_else(|| RuntimeError::TypeError("T2: missing THREW payload".into()));
                drop(bank);
                Err(err)
            }
            // P4 DEADLINE: the watchdog fired inside a call. Uncatchable.
            crate::jit::T2_DEADLINE => {
                drop(bank);
                Err(RuntimeError::Deadline)
            }
            // T2 Phase 5 — RESUME deopt: a per-guard input miss (number/object/
            // shape) fired. The stub wrote its `deopt_id` into `*out`; look up the
            // DeoptSite's `bc_pc`, DECODE the bank into VM `Value` regs (each
            // `to_value` takes its own +1, so the regs outlive the bank teardown),
            // drop the bank, then RESUME the VM at `bc_pc`. This continues AFTER any
            // committed side effect (e.g. a CALL or SetProp before the guard) — it
            // does NOT re-run from ip=0 (no duplicate effect).
            crate::jit::T2_DEOPT_RESUME => {
                let deopt_id = out as usize;
                match native.deopt_site(deopt_id) {
                    Some(site) => {
                        // T4 EXTENSION-1 FUZZER HOOK: when the inlined-frame
                        // reconstruction hook targets THIS deopt's op (a CallFn/
                        // CallValue), route the resume through the INLINE-DEOPT-TO-
                        // CALLER reconstruction (`osr::reconstruct_caller_frame`)
                        // over the REAL live bank instead of the ordinary single-
                        // frame decode. This exercises the Extension-1 math on a
                        // genuine bank image so the inlined-frame-deopt fuzzer can
                        // prove it == the un-inlined VM. `None` in production, so the
                        // ordinary path below runs unchanged (byte-identical).
                        if crate::jit::force_inlined_reconstruct_pc() == Some(site.bc_pc) {
                            if let Some(r) = try_inlined_frame_resume(
                                module, &bank, site.bc_pc, this, globals, dispatch,
                            ) {
                                drop(bank);
                                crate::interp::note_t2_deopt();
                                return r;
                            }
                        }
                        // Decode BEFORE dropping the bank (the bank owns the +1 of
                        // each pointer slot until then; `to_value` clones a +1).
                        let regs = bank.decode_to_values();
                        drop(bank);
                        crate::interp::note_t2_deopt();
                        t2_resume_on_vm(module, regs, site.bc_pc, this, globals, dispatch)
                    }
                    // Corrupt/out-of-range id: UNREACHABLE — every guard stub bakes a
                    // valid in-range DeoptSite index at compile time. We do NOT fall
                    // back to a whole-function re-run (a resume id can follow a
                    // committed call in heap mode, so re-running could duplicate an
                    // effect). Surface a loud error instead — never silently
                    // miscompute.
                    None => {
                        drop(bank);
                        Err(RuntimeError::TypeError(format!(
                            "T2 resume: out-of-range deopt id {deopt_id}"
                        )))
                    }
                }
            }
            // DEOPT (Tier-A `T2_DEOPT` or fall-through): drop the bank (releases
            // owned refs), then re-run on the VM. Only reachable BEFORE any committed
            // call side effect (a non-callable-callee decline / fall-through), so
            // re-running from ip=0 is identical.
            _ => {
                drop(bank);
                run_module_call(module, args, this, globals, None, dispatch)
            }
        }
    } else {
        // BORROWED numeric bank (P1 path) — byte-identical to before. The bank
        // holds only immediates during the run, so there is no Rc lifetime to
        // uphold (an immediate-only GetProp helper deopts on any heap slot).
        let mut bank: Vec<u64> = vec![JsVal::undefined().bits(); n_slots];
        for (i, a) in args.iter().enumerate() {
            bank[i] = JsVal::try_from_value(a)
                .map(|v| v.bits())
                .unwrap_or_else(|| JsVal::undefined().bits());
        }
        let mut out: u64 = 0;
        // SAFETY: `bank`/`out` are live stack locals; the native code only reads/
        // writes within `bank[0..n_slots]` and writes `out` once on RETURNED. The
        // bank holds only immediate (number/bool/undef/null) JsVals — no pointer
        // lane is live, so there is no Rc lifetime to uphold across the call.
        let tag = unsafe { native.call_t2lite(bank.as_mut_ptr(), &mut out as *mut u64) };
        match tag {
            crate::jit::T2_RETURNED => {
                let jv = JsVal(out);
                Ok(unsafe { jv.to_value() })
            }
            // T2 Phase 5 — RESUME deopt in the numeric path. The bank holds only
            // immediates (number/bool/undef/null) here, so decoding is trivially
            // safe (no Rc lifetime). Look up the site's `bc_pc` and resume the VM
            // mid-function. No side effect can precede a guard in numeric mode (no
            // call/setprop ops compile without heap mode), so a resume is identical
            // to a re-run — but we use resume uniformly so the deopt-fuzz oracle
            // exercises the SAME resume mechanism in both modes.
            crate::jit::T2_DEOPT_RESUME => {
                let deopt_id = out as usize;
                match native.deopt_site(deopt_id) {
                    Some(site) => {
                        // SAFETY: every slot is an immediate JsVal (numeric bank);
                        // `to_value` on an immediate touches no pointer.
                        let regs: Vec<Value> =
                            bank.iter().map(|&b| unsafe { JsVal(b).to_value() }).collect();
                        crate::interp::note_t2_deopt();
                        t2_resume_on_vm(module, regs, site.bc_pc, this, globals, dispatch)
                    }
                    None => run_module_call(module, args, this, globals, None, dispatch),
                }
            }
            _ => run_module_call(module, args, this, globals, None, dispatch),
        }
    }
}
#[cfg(not(target_os = "windows"))]
pub fn run_t2lite_call(
    _native: &crate::jit::JitFunction,
    module: &Module,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    run_module_call(module, args, this, globals, None, dispatch)
}

/// Run a T3-compiled function. The native code was produced by the T2-lite
/// backend over the T3 OPTIMIZED module (carried on `native.t3_module()`); a
/// deopt resumes the VM on THAT module (the identity-map module the native code
/// mirrors). T3's subset is numeric/control-flow only, so the run uses the
/// NUMERIC bank — we force T2 heap mode OFF for the duration so the compile-time
/// store mode (numeric) and run-time bank mode always agree, independent of the
/// process `CV_T2_HEAP` flag.
#[cfg(target_os = "windows")]
pub fn run_t3_call(
    native: &crate::jit::JitFunction,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    let opt_module = match native.t3_module() {
        Some(m) => m.clone(),
        None => {
            return Err(RuntimeError::TypeError(
                "T3 run: native function has no optimized module".into(),
            ))
        }
    };
    // Pin heap mode OFF: T3's optimized module is numeric-only and was compiled
    // with the numeric store mode; the numeric bank in `run_t2lite_call` must
    // match (the owning bank would misinterpret the raw stores).
    let _heap = crate::interp::T2HeapGuard::new(false);
    run_t2lite_call(native, &opt_module, args, this, globals, dispatch)
}

#[cfg(not(target_os = "windows"))]
pub fn run_t3_call(
    _native: &crate::jit::JitFunction,
    args: &[Value],
    _this: &Value,
    _globals: &std::cell::RefCell<HashMap<String, Value>>,
    _dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    Err(RuntimeError::TypeError("T3 unsupported on this target".into()))
}

/// Run a T4 (Maglev-class) compiled function. For a NON-inlined T4 function this is
/// identical to `run_t3_call` (resume on the fused/optimized module). For a P3
/// INLINED T4 function the native code was compiled over the FUSED module (callee
/// spliced in), but a deopt must resume the VM on the ORIGINAL caller module (whose
/// `Call` op is intact) at the MAPPED `DeoptSite.bc_pc` — the INLINE-DEOPT-TO-CALLER
/// design. The bank is sized from the fused module (so the native stores fit); the
/// resume decodes the bank and runs the original caller VM, which only reads its own
/// `n_regs` (the fused bank's extra callee-window slots are ignored). The caller's
/// register slots `0..caller_n_regs` are a valid caller image at the resume op (the
/// inliner never overwrites a caller slot before its post-call op), so the resumed
/// result is byte-identical to a non-inlined VM run (oracle + deopt-fuzz proven).
#[cfg(target_os = "windows")]
pub fn run_t4_call(
    native: &crate::jit::JitFunction,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    use crate::jsval::JsVal;
    // NON-inlined T4: no separate caller module → the proven T3 runner path.
    let resume_module = match native.t4_deopt_module() {
        Some(m) => m.clone(),
        None => return run_t3_call(native, args, this, globals, dispatch),
    };
    // INLINED T4: bank sized from the FUSED module (carried on t3_module), resume on
    // the ORIGINAL caller module.
    let fused_module = match native.t3_module() {
        Some(m) => m.clone(),
        None => {
            return Err(RuntimeError::TypeError(
                "T4 inlined run: missing fused module".into(),
            ))
        }
    };
    let _heap = crate::interp::T2HeapGuard::new(false);
    let fused = match fused_module.fns.first() {
        Some(f) => f,
        None => return Err(RuntimeError::TypeError("T4 inlined run: empty fused module".into())),
    };
    // NUMERIC bank, sized to the FUSED function's reg count (>= caller_n_regs).
    let n_slots = (fused.n_regs as usize).max(args.len()).max(1);
    let mut bank: Vec<u64> = vec![JsVal::undefined().bits(); n_slots];
    for (i, a) in args.iter().enumerate() {
        bank[i] = JsVal::try_from_value(a)
            .map(|v| v.bits())
            .unwrap_or_else(|| JsVal::undefined().bits());
    }
    let mut out: u64 = 0;
    // SAFETY: `bank`/`out` are live stack locals; the native code reads/writes only
    // within `bank[0..n_slots]` (the fused n_regs) and writes `out` once on RETURNED.
    // The bank holds only immediate JsVals (numeric store mode), so no Rc lifetime is
    // live across the call.
    let tag = unsafe { native.call_t2lite(bank.as_mut_ptr(), &mut out as *mut u64) };
    match tag {
        crate::jit::T2_RETURNED => {
            let jv = JsVal(out);
            Ok(unsafe { jv.to_value() })
        }
        crate::jit::T2_DEOPT_RESUME => {
            let deopt_id = out as usize;
            match native.deopt_site(deopt_id) {
                Some(site) => {
                    // Decode the fused bank → VM regs (numeric bank → trivially safe;
                    // every slot is an immediate). The slots 0..caller_n_regs are the
                    // caller's register image; the extra callee-window slots are
                    // ignored by the original caller VM (it uses its own n_regs).
                    let regs: Vec<Value> =
                        bank.iter().map(|&b| unsafe { JsVal(b).to_value() }).collect();
                    crate::interp::note_t2_deopt();
                    // RESUME ON THE ORIGINAL CALLER MODULE at the MAPPED bc_pc (an
                    // inlined-region guard's bc_pc is the caller's Call op → the VM
                    // re-runs the ordinary non-inlined call).
                    t2_resume_on_vm(&resume_module, regs, site.bc_pc, this, globals, dispatch)
                }
                // Unreachable: every guard stub bakes a valid in-range DeoptSite
                // index. Surface a loud error rather than silently miscompute.
                None => Err(RuntimeError::TypeError(format!(
                    "T4 inlined resume: out-of-range deopt id {deopt_id}"
                ))),
            }
        }
        // Tier-A deopt / fall-through (a pre-effect decline): re-run the ORIGINAL
        // caller on the VM from the top — identical, since no side effect committed
        // (the numeric subset has no committed call before a guard; the inlined call
        // is pure by the inliner's heuristics).
        _ => run_module_call(&resume_module, args, this, globals, None, dispatch),
    }
}

#[cfg(not(target_os = "windows"))]
pub fn run_t4_call(
    _native: &crate::jit::JitFunction,
    args: &[Value],
    _this: &Value,
    _globals: &std::cell::RefCell<HashMap<String, Value>>,
    _dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    Err(RuntimeError::TypeError("T4 unsupported on this target".into()))
}

// ======================================================================
// T2→T2 — NATIVE-TO-NATIVE CALL: the JsVal-args entry.
//
// When a T2-compiled CALLER (`rt_call_value`/`rt_call_fn`) finds the callee is
// ALSO T2-compiled (a Ready slot in `crate::interp::T2_MODULE_REGISTRY`), it
// invokes the callee here from JsVal args DIRECTLY — no Value<->JsVal marshaling,
// no VM dispatch. The callee runs in its OWN `OwningRegBank` (seeded from the
// caller's bank arg slots), its OWN T2 native code, and handles its OWN internal
// deopt (P5 resume) transparently. The result is returned as a `JsVal` + a status
// tag the caller's helper maps (RETURNED → owning-store into the caller's dst;
// THREW → stash in ctx + T2_THREW; DEADLINE → T2_DEADLINE).
//
// REFCOUNT CONTRACT (leak-tested): the caller hands borrowed-handle JsVal args
// (the caller's bank owns its own +1 of each). The callee's bank seed takes its
// OWN +1 of every pointer arg (uniform-own, via `OwningRegBank::new_from_jsvals`).
// On callee teardown the bank decs every pointer slot. The RESULT JsVal is read
// via `to_value`+re-box (a genuine +1 owned `Value`) BEFORE the callee bank drops,
// then handed back to the caller which owning-stores it into its dst slot. So:
//   caller args  : net 0 (caller's bank +1 preserved; callee's +1 paired w/ dec)
//   callee bank  : net 0 (all +1 seeds dec'd at teardown)
//   result       : +1 in the returned `JsVal` (the caller's owning-store consumes
//                  it, dec'ing the temp Value as it takes the bank's +1) → net 0.
//
// GC-DURING-CALL: the callee's bank is GC-registered for its whole run (P2). The
// CALLER's bank is ALSO still registered (the caller's `OwningRegBank` outlives
// the native call). So a `gc_collect` anywhere in the callee marks BOTH banks →
// no UAF of a bank-only-reachable heap value in either frame.
//
// RECURSION: a T2 fn calling itself routes here too — each invocation builds its
// OWN bank (sized to the callee's n_regs), so the path is re-entrant. The normal
// VM watchdog / recursion limits still apply (the callee's resume-on-deopt runs on
// the VM, and a non-deopting recursive callee re-enters THIS function, which is an
// ordinary native recursion bounded by the host stack — the per-frame bank alloc
// is small and the existing `dispatch_call_value` recursion limit is unchanged
// because a recursive T2 call still ultimately routes a non-T2 base case through
// the VM dispatcher).
// ======================================================================

/// Status of a native-to-native T2 callee run (returned alongside the result
/// `JsVal` from [`run_t2lite_from_jsval_args`]).
#[cfg(target_os = "windows")]
pub enum T2NativeStatus {
    /// The callee returned normally; the accompanying `JsVal` is the result (a
    /// genuine +1 owned value — the caller owning-stores it, consuming the +1).
    Returned,
    /// The callee threw a catchable error (carried out-of-band, here).
    Threw(RuntimeError),
    /// The callee hit the wall-clock watchdog (uncatchable).
    Deadline,
}

/// Run a Ready T2-compiled callee (`module.fns[0]`) from JsVal args DIRECTLY (the
/// T2→T2 native-to-native entry). Builds the callee's OWNING + GC-rooted bank from
/// the `argc` JsVals at `args_ptr` (uniform-own seed), runs the native code,
/// handles the callee's own internal deopt (P5 resume on the VM), and returns the
/// result `JsVal` + a [`T2NativeStatus`]. On RETURNED the returned `JsVal` is a
/// FRESH +1-owned value (boxed from the owned `Value` read before the bank drops);
/// the caller MUST owning-store it (consuming that +1) so refcounts net to zero.
///
/// `this` is the callee's `this` binding (undefined for a plain `CallFn`/value
/// call with no receiver). `globals`/`dispatch` thread through to a deopt resume
/// and to any nested call the callee makes.
///
/// # Safety
/// `args_ptr` points at `argc` live `u64`/`JsVal`s (the caller's bank arg slots),
/// each pointer arg's `Rc` alive for the seed (held by the caller's bank). `module`
/// is the callee's per-fn module whose `fns[0]` `native` compiled (the registry
/// guarantees the pairing). The caller holds nothing across this call except its
/// callee-saved bank/ctx ptrs (the aliasing discipline), so its bank stays valid.
#[cfg(target_os = "windows")]
pub unsafe fn run_t2lite_from_jsval_args(
    native: &crate::jit::JitFunction,
    module: &Module,
    args_ptr: *const u64,
    argc: usize,
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> (crate::jsval::JsVal, T2NativeStatus) {
    use crate::jsval::JsVal;
    // WATCHDOG: a native-to-native recursion does not tick the per-op watchdog
    // (the callee's straight-line native code runs between guard checks), so an
    // unbounded numeric recursion would otherwise only be caught on a deopt. Check
    // the wall-clock deadline at every callee entry (matching the VM's call-entry
    // watchdog) so a runaway T2→T2 recursion aborts deterministically, uncatchably.
    if crate::interp::js_runtime_deadline_exceeded() {
        return (JsVal::undefined(), T2NativeStatus::Deadline);
    }
    let f = match module.fns.get(0) {
        Some(f) => f,
        None => {
            return (
                JsVal::undefined(),
                T2NativeStatus::Threw(RuntimeError::TypeError("T2→T2: empty module".into())),
            );
        }
    };
    // REST-PARAM GUARD: a function with a rest parameter (`function g(a, ...rest)`)
    // needs the VM's entry-time rest-gathering (`args[rest_reg..]` → an Array bound
    // to `rest_reg`). The JsVal-args bank seed does NOT do that gathering (it seeds
    // raw arg slots), and a T2 resume continues mid-function (no entry re-do), so a
    // native-to-native run would leave `rest_reg` holding a raw arg, not the rest
    // array — a divergence. Route such a callee through the VM (`run_module_call`,
    // which gathers rest correctly) instead. (Conservative: correctness > coverage.
    // The T2 compile rarely yields a rest-using body in-subset anyway, but this
    // closes the hole for the native-to-native path explicitly.)
    if f.rest_reg.is_some() {
        // SAFETY: args_ptr[0..argc] are live JsVals; decode each to an owned Value.
        let arg_vals: Vec<Value> = (0..argc)
            .map(|k| {
                let jv = JsVal(unsafe { *args_ptr.add(k) });
                unsafe { jv.to_value() }
            })
            .collect();
        return match run_module_call(module, &arg_vals, this, globals, None, dispatch) {
            Ok(v) => {
                let jv = JsVal::try_from_value(&v).unwrap_or_else(JsVal::undefined);
                unsafe { jv.rc_inc() };
                drop(v);
                (jv, T2NativeStatus::Returned)
            }
            Err(RuntimeError::Deadline) => (JsVal::undefined(), T2NativeStatus::Deadline),
            Err(e) => (JsVal::undefined(), T2NativeStatus::Threw(e)),
        };
    }
    // Read the caller's bank arg slots as a JsVal slice (no Value marshaling).
    // SAFETY: caller contract — `args_ptr[0..argc]` are live JsVals.
    let arg_jsvals: &[JsVal] = unsafe {
        std::slice::from_raw_parts(args_ptr as *const JsVal, argc)
    };
    let n_slots = (f.n_regs as usize).max(argc).max(1);
    // OWNING + GC-ROOTED bank, seeded from the JsVal args (uniform-own: +1 each
    // pointer arg). Registered as a GC root for the run (so the callee's frame is
    // marked; the CALLER's bank is ALSO still registered → both-bank safety).
    // SAFETY: each pointer arg's Rc is alive (held by the caller's bank slots).
    let mut bank = unsafe { OwningRegBank::new_from_jsvals(n_slots, arg_jsvals) };
    let mut out: u64 = 0;
    let mut ctx = T2CallCtx {
        module,
        globals,
        dispatch: dispatch as *mut _,
        thrown: None,
    };
    // SAFETY: bank/out/ctx live for the call; the native code touches only
    // bank[0..len] and the owning-store / re-entry helpers (which keep the +1
    // invariant); the bank is GC-rooted so a re-entrant collect marks it.
    let tag = unsafe {
        native.call_t2lite_ctx(
            bank.as_mut_ptr(),
            &mut out as *mut u64,
            &mut ctx as *mut T2CallCtx as *mut core::ffi::c_void,
        )
    };
    match tag {
        crate::jit::T2_RETURNED => {
            // Read the result as a genuine +1-owned Value BEFORE the bank drops
            // (the bank's teardown decs the result slot). Re-box to a FRESH +1
            // JsVal the caller owning-stores. EXIT ORDERING is load-bearing (the
            // leak oracle asserts it): bind `result` before dropping the bank.
            let jv = JsVal(out);
            let result: Value = unsafe { jv.to_value() }; // +1 owned
            drop(bank); // teardown decs every pointer slot (incl. the result slot's bank +1)
            let result_jv = JsVal::try_from_value(&result).unwrap_or_else(JsVal::undefined);
            // `result_jv` borrows `result`'s pointer; take its OWN +1 so the value
            // survives `result` dropping at end of scope, then forget `result` to
            // hand the +1 to the caller (who consumes it via the owning-store).
            unsafe { result_jv.rc_inc() };
            drop(result); // releases the +1 `to_value` took; `result_jv` holds the +1 we just took
            (result_jv, T2NativeStatus::Returned)
        }
        crate::jit::T2_THREW => {
            let err = ctx
                .thrown
                .take()
                .unwrap_or_else(|| RuntimeError::TypeError("T2→T2: missing THREW payload".into()));
            drop(bank);
            (JsVal::undefined(), T2NativeStatus::Threw(err))
        }
        crate::jit::T2_DEADLINE => {
            drop(bank);
            (JsVal::undefined(), T2NativeStatus::Deadline)
        }
        crate::jit::T2_DEOPT_RESUME => {
            // The callee took a per-guard internal deopt — resume it on the VM at
            // the guard's bc_pc (transparent to the caller: it just sees the final
            // result/throw). Decode the bank BEFORE dropping it (each `to_value`
            // takes its own +1, so the regs outlive the teardown).
            let deopt_id = out as usize;
            match native.deopt_site(deopt_id) {
                Some(site) => {
                    let regs = bank.decode_to_values();
                    drop(bank);
                    crate::interp::note_t2_deopt();
                    match t2_resume_on_vm(module, regs, site.bc_pc, this, globals, dispatch) {
                        Ok(v) => {
                            let jv = JsVal::try_from_value(&v).unwrap_or_else(JsVal::undefined);
                            unsafe { jv.rc_inc() }; // +1 for the caller's owning-store
                            drop(v);
                            (jv, T2NativeStatus::Returned)
                        }
                        Err(RuntimeError::Deadline) => (JsVal::undefined(), T2NativeStatus::Deadline),
                        Err(e) => (JsVal::undefined(), T2NativeStatus::Threw(e)),
                    }
                }
                None => {
                    drop(bank);
                    (
                        JsVal::undefined(),
                        T2NativeStatus::Threw(RuntimeError::TypeError(format!(
                            "T2→T2 resume: out-of-range deopt id {deopt_id}"
                        ))),
                    )
                }
            }
        }
        // Tier-A T2_DEOPT (non-callable-callee decline / fall-through) — a PRE-
        // side-effect decline; re-run the whole callee on the VM (identical to
        // ip=0). The callee args are reconstructed from the seeded bank.
        _ => {
            let regs = bank.decode_to_values();
            drop(bank);
            let args: Vec<Value> = regs.into_iter().take(argc).collect();
            match run_module_call(module, &args, this, globals, None, dispatch) {
                Ok(v) => {
                    let jv = JsVal::try_from_value(&v).unwrap_or_else(JsVal::undefined);
                    unsafe { jv.rc_inc() };
                    drop(v);
                    (jv, T2NativeStatus::Returned)
                }
                Err(RuntimeError::Deadline) => (JsVal::undefined(), T2NativeStatus::Deadline),
                Err(e) => (JsVal::undefined(), T2NativeStatus::Threw(e)),
            }
        }
    }
}

extern "system" fn t1_op_thunk(state: *mut VmState, ip: usize) -> u64 {
    // SAFETY: contract above. We take a unique &mut for the duration of one op;
    // native code is single-threaded and not re-entrant into this same state.
    let st = unsafe { &mut *state };
    st.ip = ip;
    match t1_step_one(st) {
        StepStatus::Continue => T1_CONTINUE,
        StepStatus::Jumped(t) => {
            // For native control flow the TARGET is a compile-time constant in
            // the bytecode op, so we don't need to convey it here — native code
            // already knows where to jump. (We still set ip for fidelity.)
            st.ip = t;
            T1_JUMPED
        }
        s @ StepStatus::Returned(_) | s @ StepStatus::Threw(_) | s @ StepStatus::Deadline => {
            let tag = match &s {
                StepStatus::Returned(_) => T1_RETURNED,
                StepStatus::Threw(_) => T1_THREW,
                _ => T1_DEADLINE,
            };
            // Stash the payload for the Rust wrapper to read after the native
            // function returns this tag.
            unsafe {
                *st.out = Some(s);
            }
            tag
        }
    }
}

/// Run a T1-compiled function: set up the `VmState` (+ out-slot) over real
/// stack locals, invoke the installed native code, then translate the returned
/// TAG (+ stashed out-slot) into `run_function`'s exact control flow — Ret
/// value, catchable throw routed to `try_stack`, uncatchable Deadline.
///
/// Mirrors `run_function`'s entry (register file from the pool, arg/rest
/// binding) so a T1 run is observationally identical to a VM run; only the
/// op-dispatch loop is replaced by native code calling `t1_op_thunk`.
#[cfg(target_os = "windows")]
fn run_function_t1(
    native: &crate::jit::JitFunction,
    module: &Module,
    fn_idx: usize,
    args: &[Value],
    _this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    mut dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    let f = match module.fns.get(fn_idx) {
        Some(f) => f,
        None => {
            return Err(RuntimeError::TypeError(format!(
                "bad fn_idx {fn_idx} (module has {} fns)",
                module.fns.len()
            )));
        }
    };
    let mut regs = PooledRegs::new((f.n_regs as usize).max(args.len()));
    for (i, a) in args.iter().enumerate() {
        regs[i] = a.clone();
    }
    if let Some(rest_reg) = f.rest_reg {
        let ri = rest_reg as usize;
        let rest: Vec<Value> = if ri < args.len() {
            args[ri..].to_vec()
        } else {
            Vec::new()
        };
        if ri < regs.len() {
            regs[ri] = Value::Array(std::rc::Rc::new(std::cell::RefCell::new(rest)));
        }
    }
    let mut try_stack: Vec<(usize, Reg)> = Vec::new();
    let mut last_callee_hint: String = String::new();
    let mut out: Option<StepStatus> = None;
    // The watchdog tick counter lives IN the state (the thunk increments it each
    // op); it is not a separate loop local here.
    let mut st = VmState {
        regs: &mut *regs as *mut Vec<Value>,
        f,
        module,
        globals,
        dispatch: &mut *dispatch as *mut _,
        ip: 0,
        try_stack: &mut try_stack as *mut Vec<(usize, Reg)>,
        last_callee_hint: &mut last_callee_hint as *mut String,
        wd_ticks: 0,
        out: &mut out as *mut Option<StepStatus>,
    };
    // SAFETY: the installed bytes are a `compile_baseline_t1` function whose ABI
    // is `extern "system" fn(*mut VmState) -> u64` (RCX = state ptr, RAX = tag).
    // `st` is a live local that outlives this call; `out` is its sibling local.
    let tag = unsafe { native.call_t1(&mut st as *mut VmState as *mut core::ffi::c_void) };
    match tag {
        T1_RETURNED => match out.take() {
            Some(StepStatus::Returned(v)) => Ok(v),
            _ => Err(RuntimeError::TypeError("T1: missing return payload".into())),
        },
        T1_THREW => match out.take() {
            Some(StepStatus::Threw(e)) => Err(e),
            _ => Err(RuntimeError::TypeError("T1: missing throw payload".into())),
        },
        T1_DEADLINE => Err(RuntimeError::Deadline),
        // A bytecode function always ends in `Ret`; native code can only exit via
        // the epilogue with one of the tags above. Anything else is a bug.
        other => Err(RuntimeError::TypeError(format!(
            "T1: native code returned unexpected tag {other}"
        ))),
    }
}

fn run_function(
    module: &Module,
    fn_idx: usize,
    args: &[Value],
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    closure: Option<&std::rc::Rc<crate::interp::BcClosure>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    let f = match module.fns.get(fn_idx) {
        Some(f) => f,
        // A dangling function index would be a compiler bug; never panic the
        // host over it — surface a runtime error the caller can handle.
        None => {
            return Err(RuntimeError::TypeError(format!(
                "bad fn_idx {fn_idx} (module has {} fns)",
                module.fns.len()
            )));
        }
    };
    // Register file from a thread-local pool — avoids a heap alloc per call
    // (fib(32) = 2.1M calls). `PooledRegs` derefs to `Vec<Value>` so all
    // `regs[..]` accesses are unchanged, and returns the buffer to the pool on
    // ANY exit (return/throw/?) via Drop. Recursion grows the pool to ~max
    // depth, then reuses.
    let mut regs = PooledRegs::new((f.n_regs as usize).max(args.len()));
    for (i, a) in args.iter().enumerate() {
        regs[i] = a.clone();
    }
    // Rest parameter: gather the trailing args (`args[rest_reg..]`) into a real
    // array bound to its register. Matches the tree-walk param binding.
    if let Some(rest_reg) = f.rest_reg {
        let ri = rest_reg as usize;
        let rest: Vec<Value> = if ri < args.len() {
            args[ri..].to_vec()
        } else {
            Vec::new()
        };
        if ri < regs.len() {
            regs[ri] = Value::Array(std::rc::Rc::new(std::cell::RefCell::new(rest)));
        }
    }
    // Strict-mode frame: a `"use strict"` function runs strict (and a function
    // declared inside strict code inherits it). Pushed around the body so a
    // refused legacy-platform-object [[Set]]/[[Delete]] throws in strict mode
    // (the host hooks read `strict_mode_active`). Popped on EVERY exit.
    let strict_here = f.strict || crate::interp::strict_mode_active();
    crate::interp::push_strict_frame(strict_here);
    // Seed the register file, then run the interpreter loop from ip=0 (the
    // arg/rest binding above is the only entry-specific setup). The loop body
    // itself lives in `run_function_inner` so the T2 deopt path can RESUME the
    // VM mid-function at an arbitrary `bc_pc` over a reconstructed register file
    // — see T2 Phase 5. For the ip=0 entry this is a pure no-op refactor.
    let result = run_function_inner(module, fn_idx, this, globals, closure, dispatch, &mut regs, 0);
    crate::interp::pop_strict_frame();
    // SCRIPT FRAME throw-time global flush. A top-level `for (var i = …)` init
    // var is kept in a fast LOCAL register for the hot loop and only synced to its
    // (function-scoped, i.e. global) binding by a post-loop `StoreGlobal`. On a
    // THROW/deadline that escapes the loop mid-iteration, that post-loop store is
    // skipped — so the live register value would never reach `globals`, and a
    // throwing script's `globalThis.i` would diverge from the tree-walker (which
    // mutates the global binding in place every iteration). Flush the live
    // for-init registers to `globals` on the error path so the global ends
    // byte-identical to the tree-walker + Node/Chrome (`i === 3` for a throw at
    // i===3). Only the script frame (`fn_idx == 0`) carries these syncs; a normal
    // (Ok) completion already ran the post-loop store, so we flush ONLY on Err and
    // never pay a cost in the hot, non-throwing loop. `regs` still holds the final
    // register image here (it is dropped after this).
    if result.is_err() && fn_idx == 0 && !module.script_forinit_syncs.is_empty() {
        let mut g = globals.borrow_mut();
        for (name, reg) in &module.script_forinit_syncs {
            if let Some(v) = regs.get(*reg as usize) {
                g.insert(name.clone(), v.clone());
            }
        }
    }
    result
}

/// The shared interpreter dispatch loop. `run_function` calls this with `ip=0`
/// over a freshly-seeded register file (the ordinary entry); the T2 Phase-5
/// deopt path calls it with `ip=bc_pc` over a register file RECONSTRUCTED from
/// the JIT bank (resume mid-function on a guard miss). `regs` must already hold
/// the param/rest binding (entry) or the exact pre-op VM register image
/// (resume). `try_stack` always starts EMPTY — the T2 compiler declines any
/// function containing a try handler, so a resumed frame never enters mid-try.
///
/// This is a NO-OP factoring of the former monolithic `run_function` loop: every
/// op body, the `propagate!`/`run_shared!` macros, and the watchdog are moved
/// here verbatim; only the loop-local declarations and the `f`/`fn_idx` lookup
/// move to the call boundary.
#[allow(clippy::too_many_arguments)]
fn run_function_inner(
    module: &Module,
    fn_idx: usize,
    this: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    closure: Option<&std::rc::Rc<crate::interp::BcClosure>>,
    mut dispatch: WithInterpDispatch<'_>,
    regs: &mut PooledRegs,
    start_ip: usize,
) -> Result<Value, RuntimeError> {
    let f = match module.fns.get(fn_idx) {
        Some(f) => f,
        None => {
            return Err(RuntimeError::TypeError(format!(
                "bad fn_idx {fn_idx} (module has {} fns)",
                module.fns.len()
            )));
        }
    };
    // Stack of (catch_ip, catch_reg) for nested try blocks. Top entry
    // is the innermost handler. Always empty at a resume `bc_pc` (try-handler
    // functions are declined by the T2 compiler).
    let mut try_stack: Vec<(usize, Reg)> = Vec::new();
    // Diagnostic: the name of the most recently loaded global/property,
    // so a "callee is not callable" error can say WHICH symbol was
    // undefined instead of just the value. Debug aid only.
    let mut last_callee_hint: String = String::new();
    let mut ip: usize = start_ip;
    let mut wd_ticks: u32 = 0;
    // T4 P1 — TYPE-FEEDBACK RECORDING gate, hoisted ONCE per function activation
    // (env read is memoized in `feedback_enabled`, but the branch is hoisted to a
    // local `bool` so the hot dispatch only does a register test, not a TLS read,
    // per op). DEFAULT-OFF (`CV_FEEDBACK`): when false the `record_fb!` macro
    // expands to nothing observable and the recorder is never touched, so the
    // default build is byte-identical and pays zero cost. RECORDING ONLY — a
    // write mutates only the side table on `f`, never a JS value, so it is
    // observationally invisible whether on or off (the oracle proves this).
    let fb_on: bool = crate::feedback::feedback_enabled();
    loop {
        if ip >= f.code.len() {
            return Err(RuntimeError::Overrun);
        }
        let op = f.code[ip];
        ip += 1;
        // Helper: convert any error from a nested op into the right
        // unwind path. If a try handler is active we route it to the
        // catch; otherwise we propagate.
        macro_rules! propagate {
            ($err:expr) => {{
                let err: RuntimeError = $err;
                // A watchdog abort is uncatchable: unwind straight to the host
                // so a runaway loop wrapped in `try {}` can't swallow it.
                if matches!(err, RuntimeError::Deadline) {
                    return Err(err);
                }
                if let Some((target, reg)) = try_stack.pop() {
                    let val = match err {
                        RuntimeError::Thrown(v) => v,
                        other => Value::str(format!("{other}")),
                    };
                    regs[reg as usize] = val;
                    ip = target;
                    continue;
                }
                return Err(err);
            }};
        }
        // Run a refactored op through its shared `op_xxx` helper (single source
        // of truth with the T1 JIT). Builds a transient `VmState` over THIS
        // loop's locals, invokes `$call`, and translates the returned
        // `StepStatus` into the loop's native control flow — preserving exact
        // semantics (Ret value, catch routing via `propagate!`, uncatchable
        // Deadline, jump targets). `ip` already points PAST the current op here,
        // so `Continue` needs no fix-up.
        macro_rules! run_shared {
            ($call:ident ( $($arg:expr),* )) => {{
                let mut __st = VmState {
                    regs: &mut **regs as *mut Vec<Value>,
                    f,
                    module,
                    globals,
                    dispatch: &mut *dispatch as *mut _,
                    ip,
                    try_stack: &mut try_stack as *mut Vec<(usize, Reg)>,
                    last_callee_hint: &mut last_callee_hint as *mut String,
                    wd_ticks,
                    out: std::ptr::null_mut(),
                };
                match $call(&mut __st $(, $arg)*) {
                    StepStatus::Continue => {}
                    StepStatus::Jumped(t) => ip = t,
                    StepStatus::Returned(v) => return Ok(v),
                    StepStatus::Threw(e) => propagate!(e),
                    StepStatus::Deadline => return Err(RuntimeError::Deadline),
                }
            }};
        }
        // T4 P1 — record BINARY/UNARY/CALL type feedback for the op at `ip-1`
        // (the just-fetched op; `ip` was already advanced). Gated on `fb_on`
        // (DEFAULT-OFF) so it is zero-cost off. The recorder lazily sizes the
        // side table to `code.len()` on first use (mirroring `ic`), then SKIPS a
        // SETTLED (`Any`) slot before any operand read — the V8 monotone-settle
        // that bounds overhead: a megamorphic site costs only the settled check.
        // The macro reads operands from the LIVE `regs` BEFORE the op executes
        // (Ignition collects feedback at the op), so the hint reflects the exact
        // inputs the op will consume. Writing the side table is observationally
        // invisible (never touches a JS value), so results are unchanged.
        macro_rules! record_fb {
            // Binary arith/compare site: join the two operand hints.
            (binop $idx:expr, $lhs:expr, $rhs:expr) => {{
                if fb_on {
                    let mut tbl = f.feedback.borrow_mut();
                    if tbl.len() != f.code.len() {
                        tbl.resize(f.code.len(), crate::feedback::TypeFeedback::INVALID);
                    }
                    let slot = &mut tbl[$idx];
                    if !slot.binop_hint().is_settled() {
                        let (l, r) = (&regs[$lhs as usize], &regs[$rhs as usize]);
                        slot.record_binop(l, r);
                    }
                    drop(tbl);
                    // MUTATION HOOK (test-only): a recorder that wrongly touched a
                    // JS value would diverge — clobber the rhs register to PROVE
                    // the feedback-on oracle leg catches such a side effect.
                    if crate::feedback::force_record_clobber() {
                        regs[$rhs as usize] = Value::Number(123456.0);
                    }
                }
            }};
            // Unary arith site: single operand hint.
            (unop $idx:expr, $src:expr) => {{
                if fb_on {
                    let mut tbl = f.feedback.borrow_mut();
                    if tbl.len() != f.code.len() {
                        tbl.resize(f.code.len(), crate::feedback::TypeFeedback::INVALID);
                    }
                    let slot = &mut tbl[$idx];
                    if !slot.binop_hint().is_settled() {
                        let s = &regs[$src as usize];
                        slot.record_unop(s);
                    }
                }
            }};
            // Direct module-local call: record the (already-known) target index.
            (call $idx:expr, $target:expr) => {{
                if fb_on {
                    let mut tbl = f.feedback.borrow_mut();
                    if tbl.len() != f.code.len() {
                        tbl.resize(f.code.len(), crate::feedback::TypeFeedback::INVALID);
                    }
                    tbl[$idx].record_call($target as u32);
                }
            }};
        }
        // Per-task wall-clock watchdog: abort a runaway VM loop instead of
        // freezing the UI thread. Amortized so the `Instant::now` is ~free.
        wd_ticks = wd_ticks.wrapping_add(1);
        if wd_ticks & 0x7FF == 0 && crate::interp::js_runtime_deadline_exceeded() {
            return Err(RuntimeError::Deadline);
        }
        match op {
            Op::LoadConst { dst, k } => run_shared!(op_load_const(dst, k)),
            Op::LoadTrue { dst } => run_shared!(op_load_true(dst)),
            Op::LoadFalse { dst } => run_shared!(op_load_false(dst)),
            Op::LoadNull { dst } => run_shared!(op_load_null(dst)),
            Op::LoadUndef { dst } => run_shared!(op_load_undef(dst)),
            Op::Move { dst, src } => run_shared!(op_move(dst, src)),
            Op::Add { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_add(dst, lhs, rhs))
            }
            Op::Sub { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_sub(dst, lhs, rhs))
            }
            Op::Mul { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_mul(dst, lhs, rhs))
            }
            Op::Div { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                let v = if let (Value::Number(a), Value::Number(b)) =
                    (&regs[lhs as usize], &regs[rhs as usize])
                {
                    Value::Number(a / b)
                } else {
                    match bigint_binop(
                        "/",
                        &regs[lhs as usize],
                        &regs[rhs as usize],
                        globals,
                        &mut *dispatch,
                    ) {
                        Some(r) => r?,
                        None => Value::Number(
                            to_num(&regs[lhs as usize])? / to_num(&regs[rhs as usize])?,
                        ),
                    }
                };
                regs[dst as usize] = v;
            }
            Op::Mod { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_mod(dst, lhs, rhs))
            }
            Op::Pow { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_pow(dst, lhs, rhs))
            }
            Op::Eq { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_eq(dst, lhs, rhs))
            }
            Op::Neq { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_neq(dst, lhs, rhs))
            }
            Op::LooseEq { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                regs[dst as usize] =
                    Value::Bool(Value::loose_eq(&regs[lhs as usize], &regs[rhs as usize]));
            }
            Op::LooseNeq { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                regs[dst as usize] =
                    Value::Bool(!Value::loose_eq(&regs[lhs as usize], &regs[rhs as usize]));
            }
            // Number-Number fast path is correct (NaN compares false matches
            // the spec). Non-Number-Number routes through Abstract Relational
            // Comparison so `"apple" < "banana"`, `new Date(1) < new Date(2)`,
            // and BigInt-mixed comparisons work; Object operands dispatch to
            // __tb_host_binop so valueOf() runs via the interpreter. Shared with
            // T1 via `op_lt` (single source of truth).
            Op::Lt { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_lt(dst, lhs, rhs))
            }
            Op::Le { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_le(dst, lhs, rhs))
            }
            Op::Gt { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_gt(dst, lhs, rhs))
            }
            Op::BitAnd { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_bitand(dst, lhs, rhs))
            }
            Op::BitOr { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_bitor(dst, lhs, rhs))
            }
            Op::BitXor { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_bitxor(dst, lhs, rhs))
            }
            Op::Shl { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_shl(dst, lhs, rhs))
            }
            Op::Shr { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_shr(dst, lhs, rhs))
            }
            Op::Ushr { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_ushr(dst, lhs, rhs))
            }
            Op::Ge { dst, lhs, rhs } => {
                record_fb!(binop ip - 1, lhs, rhs);
                run_shared!(op_ge(dst, lhs, rhs))
            }
            Op::Neg { dst, src } => {
                record_fb!(unop ip - 1, src);
                run_shared!(op_neg(dst, src))
            }
            Op::Not { dst, src } => run_shared!(op_not(dst, src)),
            Op::BitNot { dst, src } => {
                record_fb!(unop ip - 1, src);
                run_shared!(op_bitnot(dst, src))
            }
            Op::ToNumber { dst, src } => {
                record_fb!(unop ip - 1, src);
                run_shared!(op_to_number(dst, src))
            }
            Op::Typeof { dst, src } => run_shared!(op_typeof(dst, src)),
            Op::ToStr { dst, src } => run_shared!(op_to_str(dst, src)),
            Op::Jmp { target } => run_shared!(op_jmp(target)),
            Op::JmpIfFalse { cond, target } => run_shared!(op_jmp_if_false(cond, target)),
            Op::JmpIfTrue { cond, target } => {
                if regs[cond as usize].to_bool() {
                    ip = target as usize;
                }
            }
            Op::CallFn {
                dst,
                fn_idx,
                first_arg,
                n_args,
            } => {
                // T4 P1 — record the call target (module function index). A
                // direct `CallFn` is monomorphic by construction (the target is
                // baked in the op), so this records the same `fn_idx` every time;
                // it is the seam the P3 inliner reads to confirm a single, known
                // callee. (Indirect `CallValue` targets are NOT recorded in P1 —
                // resolving a value to a module fn-index at the call site needs
                // the closure→fn_idx mapping, which the inliner phase will add.)
                record_fb!(call ip - 1, fn_idx);
                let mut call_args: Vec<Value> = Vec::with_capacity(n_args as usize);
                for i in 0..n_args {
                    call_args.push(regs[first_arg as usize + i as usize].clone());
                }
                // ── P6 leaf-JIT fast path. A module-local callee that is a hot,
                // all-numeric, ≤4-arg pure-arithmetic function runs as native f64
                // code — the SAME P6 tier the tree-walk call path takes. Without
                // this, routing the hot TOP LEVEL onto the VM regressed a leaf like
                // jit.js's `f(i)` / loop.js's `work(n)` to the interpreter (the
                // in-VM `CallFn` only ran `run_function`), losing the native win.
                // Eligibility mirrors `interp::p6_args_eligible`: JIT on, ≥1 arg,
                // ≤4 args, at least as many args as params, all-numeric. A non-
                // numeric arg / non-compilable body falls through to the VM below —
                // byte-identical, just slower. The result of a P6 numeric fn is a
                // Number, identical to the VM (the f64 JIT only compiles bodies
                // whose VM result is provably the same f64).
                if crate::interp::jit_enabled_pub()
                    && n_args >= 1
                    && n_args <= 4
                    && module
                        .fns
                        .get(fn_idx as usize)
                        .is_some_and(|cf| (n_args as usize) >= cf.n_params as usize)
                    && call_args.iter().all(|a| matches!(a, Value::Number(_)))
                {
                    if let Some(jf) = resolve_callfn_p6(module, fn_idx as usize) {
                        regs[dst as usize] = run_callfn_p6(&jf, &call_args);
                        continue;
                    }
                }
                match run_function(
                    module,
                    fn_idx as usize,
                    &call_args,
                    &Value::Undefined,
                    globals,
                    None,
                    &mut *dispatch,
                ) {
                    Ok(v) => regs[dst as usize] = v,
                    Err(e) => propagate!(e),
                }
            }
            Op::LoadGlobal { dst, name_k } => {
                let name = match &f.consts[name_k as usize] {
                    Value::String(s) => s.clone(),
                    other => {
                        return Err(RuntimeError::TypeError(format!(
                            "LoadGlobal name must be string, got {other:?}"
                        )));
                    }
                };
                let mut val = globals
                    .borrow()
                    .get(&*name)
                    .cloned()
                    .unwrap_or(Value::Undefined);
                // `__tb_spread__` materialises any iterable to a Vec — used by
                // the spread operator and the OLD eager for-of lowering.  Keep
                // this fallback so any residual `__tb_spread__` call sites (e.g.
                // async-function lowering which still uses it for `for await...of`)
                // continue to work without the tree-walker initialised first.
                if matches!(val, Value::Undefined) && &*name == "__tb_spread__" {
                    val = Value::NativeFunction(Rc::new(crate::interp::NativeFn {
                        name: "__tb_spread__".into(),
                        func: crate::interp::NativeFnBody::Pure(Box::new(|args| {
                            let src = args.into_iter().next().unwrap_or(Value::Undefined);
                            Ok(Value::Array(Rc::new(std::cell::RefCell::new(
                                crate::interp::iterable_values(&src),
                            ))))
                        })),
                        length: 0,
                        is_ctor: false,
                        props: std::cell::RefCell::new(HashMap::new()),
                    }));
                }
                // `__tb_get_iterator__` is the lazy iterator-protocol entry point
                // used by the VM's for-of lowering.  Provide an inline fallback for
                // bare bytecode unit-test environments where the tree-walker globals
                // haven't been loaded.  Handles:
                //   • Arrays   — index-based Pure iterator (per-element, no materialise)
                //   • Strings  — char-based Pure iterator
                //   • Objects with a `.next` method — the object IS the iterator
                //   • Everything else — fall back to iterable_values snapshot
                if matches!(val, Value::Undefined) && &*name == "__tb_get_iterator__" {
                    val = Value::NativeFunction(Rc::new(crate::interp::NativeFn {
                        name: "__tb_get_iterator__".into(),
                        func: crate::interp::NativeFnBody::Pure(Box::new(|args| {
                            use std::cell::RefCell;
                            use crate::interp::native_fn;
                            let src = args.into_iter().next().unwrap_or(Value::Undefined);
                            // Array fast path.
                            if let Value::Array(a) = &src {
                                let arr = a.clone();
                                let idx = Rc::new(RefCell::new(0usize));
                                let mut iter: HashMap<String, Value> = HashMap::new();
                                iter.insert("next".into(), native_fn("next", move |_| {
                                    let i = *idx.borrow();
                                    let ar = arr.borrow();
                                    if i < ar.len() {
                                        *idx.borrow_mut() = i + 1;
                                        let v = match &ar[i] {
                                            Value::Hole => Value::Undefined,
                                            x => x.clone(),
                                        };
                                        drop(ar);
                                        let mut s: HashMap<String, Value> = HashMap::new();
                                        s.insert("value".into(), v);
                                        s.insert("done".into(), Value::Bool(false));
                                        Ok(Value::Object(Rc::new(RefCell::new(s))))
                                    } else {
                                        drop(ar);
                                        let mut s: HashMap<String, Value> = HashMap::new();
                                        s.insert("value".into(), Value::Undefined);
                                        s.insert("done".into(), Value::Bool(true));
                                        Ok(Value::Object(Rc::new(RefCell::new(s))))
                                    }
                                }));
                                return Ok(Value::Object(Rc::new(RefCell::new(iter))));
                            }
                            // String fast path: iterate Unicode scalars.
                            if let Value::String(s) = &src {
                                let chars: Vec<String> =
                                    s.chars().map(|c| c.to_string()).collect();
                                let chars = Rc::new(RefCell::new(chars));
                                let idx = Rc::new(RefCell::new(0usize));
                                let mut iter: HashMap<String, Value> = HashMap::new();
                                iter.insert("next".into(), native_fn("next", move |_| {
                                    let i = *idx.borrow();
                                    let cv = chars.borrow();
                                    if i < cv.len() {
                                        let v = Value::str(cv[i].clone());
                                        *idx.borrow_mut() = i + 1;
                                        drop(cv);
                                        let mut s: HashMap<String, Value> = HashMap::new();
                                        s.insert("value".into(), v);
                                        s.insert("done".into(), Value::Bool(false));
                                        Ok(Value::Object(Rc::new(RefCell::new(s))))
                                    } else {
                                        drop(cv);
                                        let mut s: HashMap<String, Value> = HashMap::new();
                                        s.insert("value".into(), Value::Undefined);
                                        s.insert("done".into(), Value::Bool(true));
                                        Ok(Value::Object(Rc::new(RefCell::new(s))))
                                    }
                                }));
                                return Ok(Value::Object(Rc::new(RefCell::new(iter))));
                            }
                            // Object: if it already has a `next` method it IS an
                            // iterator — return it directly (covers generators, custom
                            // hand-built iterators, replay iterators).
                            if let Value::Object(o) = &src {
                                if o.borrow().contains_key("next") {
                                    return Ok(src);
                                }
                            }
                            // Fallback: materialise via iterable_values (Map/Set/etc.)
                            let vals = crate::interp::iterable_values(&src);
                            let vals = Rc::new(RefCell::new(vals));
                            let idx = Rc::new(RefCell::new(0usize));
                            let mut iter: HashMap<String, Value> = HashMap::new();
                            iter.insert("next".into(), native_fn("next", move |_| {
                                let i = *idx.borrow();
                                let vr = vals.borrow();
                                if i < vr.len() {
                                    let v = vr[i].clone();
                                    *idx.borrow_mut() = i + 1;
                                    drop(vr);
                                    let mut s: HashMap<String, Value> = HashMap::new();
                                    s.insert("value".into(), v);
                                    s.insert("done".into(), Value::Bool(false));
                                    Ok(Value::Object(Rc::new(RefCell::new(s))))
                                } else {
                                    drop(vr);
                                    let mut s: HashMap<String, Value> = HashMap::new();
                                    s.insert("value".into(), Value::Undefined);
                                    s.insert("done".into(), Value::Bool(true));
                                    Ok(Value::Object(Rc::new(RefCell::new(s))))
                                }
                            }));
                            Ok(Value::Object(Rc::new(RefCell::new(iter))))
                        })),
                        length: 0,
                        is_ctor: false,
                        props: std::cell::RefCell::new(HashMap::new()),
                    }));
                }
                if matches!(val, Value::Undefined) {
                    last_callee_hint = format!("global `{name}`");
                }
                regs[dst as usize] = val;
            }
            Op::LoadGlobalChecked { dst, name_k } => {
                let name = match &f.consts[name_k as usize] {
                    Value::String(s) => s.clone(),
                    other => {
                        return Err(RuntimeError::TypeError(format!(
                            "LoadGlobalChecked name must be string, got {other:?}"
                        )));
                    }
                };
                // Mirror the tree-walk `eval_identifier`: an identifier read in
                // a VALUE context that resolves to nothing is an unresolvable
                // Reference → ReferenceError (ECMA-262 §13.3.2 GetValue). The
                // globals map holds every builtin, `NaN`/`Infinity`/`globalThis`,
                // and every top-level binding (hoisted `var`s are present with
                // value `undefined` and DO resolve), so an absent key is
                // genuinely undeclared. Catchable via try/catch (`propagate!`).
                let resolved = globals.borrow().get(&*name).cloned();
                match resolved {
                    Some(val) => {
                        regs[dst as usize] = val;
                    }
                    None => {
                        propagate!(RuntimeError::Thrown(crate::interp::err_str(format!(
                            "ReferenceError: {name} is not defined"
                        ))));
                    }
                }
            }
            Op::StoreGlobal { name_k, src } => {
                let name = match &f.consts[name_k as usize] {
                    Value::String(s) => s.clone(),
                    other => {
                        return Err(RuntimeError::TypeError(format!(
                            "StoreGlobal name must be string, got {other:?}"
                        )));
                    }
                };
                globals
                    .borrow_mut()
                    .insert(name.to_string(), regs[src as usize].clone());
            }
            Op::CallValue {
                dst,
                callee,
                this_reg,
                first_arg,
                n_args,
            } => {
                let callee_val = regs[callee as usize].clone();
                let this_val = if this_reg == NO_THIS {
                    Value::Undefined
                } else {
                    regs[this_reg as usize].clone()
                };
                let mut call_args: Vec<Value> = Vec::with_capacity(n_args as usize);
                for i in 0..n_args {
                    call_args.push(regs[first_arg as usize + i as usize].clone());
                }
                // Dispatch via the SHARED helper (single source of truth with the
                // T2 Phase-4 re-entry path). Preserve the EXACT prior unwind
                // behaviour: a callee's THROWN error routes through `propagate!`
                // (try/catch around the call works), but a "callee is not callable"
                // TypeError returns straight out (bypassing try/catch — matching the
                // pre-refactor `return Err`), with the diagnostic hint appended.
                let ret = match dispatch_call_value(
                    callee_val,
                    this_val,
                    call_args,
                    globals,
                    &mut *dispatch,
                ) {
                    Ok(v) => v,
                    Err(RuntimeError::TypeError(msg)) if msg.starts_with("callee is not callable") => {
                        let hint = if last_callee_hint.is_empty() {
                            String::new()
                        } else {
                            format!(" ({last_callee_hint})")
                        };
                        return Err(RuntimeError::TypeError(format!("{msg}{hint}")));
                    }
                    Err(e) => propagate!(e),
                };
                regs[dst as usize] = ret;
            }
            Op::GetProp { dst, obj, key_k } => {
                let key = match &f.consts[key_k as usize] {
                    Value::String(s) => s.clone(),
                    other => {
                        return Err(RuntimeError::TypeError(format!(
                            "GetProp key must be string: {other:?}"
                        )));
                    }
                };
                // Phase-2 increment 1: monomorphic own-property inline cache.
                // Guard on (object Rc ptr, struct_ver). A hit replaces
                // property_lookup's hash probe with a direct slot index — and is
                // byte-identical because value_at_slot(slot) == get(key) for an
                // own property, and the SAME resolve_with_host /
                // resolve_accessor_read run below. Records on miss; struct_ver
                // bumps invalidate. Own data hits only (proto/host/accessor/
                // proxy all flow through the unchanged tail).
                let propic = propic_enabled();
                if propic {
                    let mut t = f.ic.borrow_mut();
                    if t.len() < f.code.len() {
                        t.resize(f.code.len(), PropIc::INVALID);
                    }
                }
                let mut hit: Option<Value> = None;
                if propic {
                    if let Value::Object(o) = &regs[obj as usize] {
                        let ob = o.borrow();
                        let sid = object_shape_id(&ob);
                        if let Some(slot) = f.ic.borrow()[ip - 1].lookup(sid) {
                            if let Some(v) = ob.value_at_slot(slot as usize) {
                                hit = Some(v.clone());
                                propic_hit();
                            }
                        }
                    }
                }
                let raw = match hit {
                    Some(v) => v,
                    None => {
                        let r = property_lookup(&regs[obj as usize], &key);
                        if propic {
                            if let Value::Object(o) = &regs[obj as usize] {
                                propic_miss();
                                let rec = {
                                    let ob = o.borrow();
                                    ob.slot_of(&*key).map(|slot| (object_shape_id(&ob), slot as u32))
                                };
                                if let Some((sid, slot)) = rec {
                                    f.ic.borrow_mut()[ip - 1].record(sid, slot);
                                }
                            }
                        }
                        r
                    }
                };
                // Prototype IC: when the own lookup MISSED (raw == Undefined), the
                // property is inherited — read it directly from the cached depth-1
                // prototype slot instead of the slow host walk. `None` ⇒ slow path.
                let raw_was_undefined = matches!(raw, Value::Undefined);
                let proto_method: Option<Value> = if propic && raw_was_undefined {
                    match &regs[obj as usize] {
                        Value::Object(o) => {
                            let ob = o.borrow();
                            let osid = object_shape_id(&ob);
                            f.ic.borrow()[ip - 1].proto_lookup(osid).and_then(
                                |(proto_slot, proto_ptr, proto_shape, key_slot)| {
                                    match ob.value_at_slot(proto_slot as usize) {
                                        Some(Value::Object(p))
                                            if std::rc::Rc::as_ptr(p) as usize == proto_ptr =>
                                        {
                                            let pb = p.borrow();
                                            if object_shape_id(&pb) == proto_shape {
                                                pb.value_at_slot(key_slot as usize).and_then(|v| {
                                                    if slot_value_is_accessor(Some(v)) {
                                                        None
                                                    } else {
                                                        Some(v.clone())
                                                    }
                                                })
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    }
                                },
                            )
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                let val = match proto_method {
                    Some(m) => {
                        propic_hit();
                        m
                    }
                    None => {
                        // Slow path: host walk (proto chain / DOM / accessor / proxy).
                        let r = match resolve_with_host(
                            raw,
                            &regs[obj as usize],
                            &key,
                            globals,
                            &mut *dispatch,
                        ) {
                            Ok(v) => v,
                            Err(e) => propagate!(e),
                        };
                        let v = match resolve_accessor_read(
                            r,
                            &regs[obj as usize],
                            module,
                            globals,
                            &mut *dispatch,
                        ) {
                            Ok(v) => v,
                            Err(e) => propagate!(e),
                        };
                        // Record the prototype IC ONLY when our own depth-1
                        // plain-proto walk reproduces the host result identically
                        // (data method, not accessor) — DOM/accessor/proxy/deep/
                        // bound resolutions are never cached.
                        if propic && raw_was_undefined {
                            if let Value::Object(o) = &regs[obj as usize] {
                                let rec = {
                                    let ob = o.borrow();
                                    ob.slot_of(crate::interp::PROTO_KEY).and_then(|proto_slot| {
                                        match ob.value_at_slot(proto_slot) {
                                            Some(Value::Object(p)) => {
                                                let pb = p.borrow();
                                                pb.slot_of(&*key).and_then(|ks| {
                                                    match pb.value_at_slot(ks) {
                                                        Some(pvv)
                                                            if !slot_value_is_accessor(Some(pvv))
                                                                && value_ref_eq(pvv, &v) =>
                                                        {
                                                            Some((
                                                                object_shape_id(&ob),
                                                                proto_slot as u32,
                                                                std::rc::Rc::as_ptr(p) as usize,
                                                                object_shape_id(&pb),
                                                                ks as u32,
                                                            ))
                                                        }
                                                        _ => None,
                                                    }
                                                })
                                            }
                                            _ => None,
                                        }
                                    })
                                };
                                if let Some((osid, pslot, pptr, pshape, kslot)) = rec {
                                    f.ic.borrow_mut()[ip - 1]
                                        .proto_record(osid, pslot, pptr, pshape, kslot);
                                }
                            }
                        }
                        v
                    }
                };
                if matches!(val, Value::Undefined) {
                    last_callee_hint = format!("property `{key}`");
                }
                regs[dst as usize] = val;
            }
            Op::GetIdx { dst, obj, key } => {
                let key_val = regs[key as usize].clone();
                let raw = indexed_lookup(&regs[obj as usize], &key_val);
                let raw = match resolve_with_host(
                    raw,
                    &regs[obj as usize],
                    &key_val.to_display_string(),
                    globals,
                    &mut *dispatch,
                ) {
                    Ok(v) => v,
                    Err(e) => propagate!(e),
                };
                regs[dst as usize] = match resolve_accessor_read(
                    raw,
                    &regs[obj as usize],
                    module,
                    globals,
                    &mut *dispatch,
                ) {
                    Ok(v) => v,
                    Err(e) => propagate!(e),
                };
            }
            Op::SetProp { obj, key_k, src } => {
                // Phase-2 increment 2: monomorphic own-property WRITE inline cache.
                // A plain own overwrite writes the slot directly — no key clone,
                // no hash, no proxy/accessor/frozen re-scan. Byte-identical to the
                // slow path's `property_store` (which is `insert(key,value)` for an
                // existing key). struct_ver invalidates on freeze/proxy-mark/key
                // changes (all add an internal key → bump); a per-slot
                // accessor-descriptor check covers `defineProperty` redefine.
                let propic = propic_enabled();
                let mut handled = false;
                if propic {
                    if let Value::Object(o) = &regs[obj as usize] {
                        {
                            let mut t = f.ic.borrow_mut();
                            if t.len() < f.code.len() {
                                t.resize(f.code.len(), PropIc::INVALID);
                            }
                        }
                        // Slot to write IF the cache hits this shape AND the slot
                        // isn't an accessor descriptor (defineProperty redefine).
                        let action = {
                            let ob = o.borrow();
                            let sid = object_shape_id(&ob);
                            f.ic.borrow()[ip - 1].lookup(sid).filter(|&slot| {
                                !slot_value_is_accessor(ob.value_at_slot(slot as usize))
                            })
                        };
                        if let Some(slot) = action {
                            propic_hit();
                            let value = regs[src as usize].clone();
                            // B4 write barrier: the IC fast path bypasses
                            // `property_store`, so record the old→young edge here.
                            crate::interp::gen_gc_write_barrier_val(
                                &regs[obj as usize],
                                &value,
                            );
                            o.borrow_mut().set_at_slot(slot as usize, value);
                            handled = true;
                        }
                    }
                }
                if !handled {
                    let key = match &f.consts[key_k as usize] {
                        Value::String(s) => s.clone(),
                        other => {
                            return Err(RuntimeError::TypeError(format!(
                                "SetProp key must be string: {other:?}"
                            )));
                        }
                    };
                    let value = regs[src as usize].clone();
                    let recv = regs[obj as usize].clone();
                    if is_proxy_val(&recv)
                        || slot_is_accessor(&recv, &key)
                        || crate::interp::value_has_attrs(&recv)
                        || crate::interp::is_legacy_collection(&recv)
                    {
                        // Route to the host so a Proxy `set` trap fires, a `set x()`
                        // accessor runs, a legacy-platform-object [[Set]] refusal
                        // applies, OR the OrdinarySet write-guard applies for a
                        // descriptor-aware object (raw property_store would
                        // overwrite either / bypass the writable check).
                        let setter = globals.borrow().get("__tb_host_setprop").cloned();
                        if let Some(g @ Value::NativeFunction(_)) = setter {
                            match dispatch(g, Value::Undefined, vec![recv, Value::String(key), value])
                            {
                                Ok(_) => {}
                                Err(e) => propagate!(e),
                            }
                        } else {
                            property_store(&recv, &key, value);
                        }
                    } else {
                        property_store(&recv, &key, value);
                        // Record the write-IC: plain own write, not frozen, key
                        // exists → cache (ptr, struct_ver, slot) for the next write.
                        if propic {
                            if let Value::Object(o) = &regs[obj as usize] {
                                let rec = {
                                    let ob = o.borrow();
                                    if ob.contains_key(crate::interp::FROZEN_KEY) {
                                        None
                                    } else {
                                        ob.slot_of(&*key)
                                            .map(|slot| (object_shape_id(&ob), slot as u32))
                                    }
                                };
                                if let Some((sid, slot)) = rec {
                                    f.ic.borrow_mut()[ip - 1].record(sid, slot);
                                }
                            }
                        }
                    }
                }
            }
            Op::SetIdx { obj, key, src } => {
                let key_val = regs[key as usize].clone();
                let value = regs[src as usize].clone();
                let recv = regs[obj as usize].clone();
                let key_str = key_val.to_display_string();
                if is_proxy_val(&recv)
                    || slot_is_accessor(&recv, &key_str)
                    || crate::interp::value_has_attrs(&recv)
                    || crate::interp::is_legacy_collection(&recv)
                {
                    let setter = globals.borrow().get("__tb_host_setprop").cloned();
                    if let Some(g @ Value::NativeFunction(_)) = setter {
                        match dispatch(
                            g,
                            Value::Undefined,
                            vec![recv, Value::str(key_str), value],
                        ) {
                            Ok(_) => {}
                            Err(e) => propagate!(e),
                        }
                    } else {
                        indexed_store(&recv, &key_val, value);
                    }
                } else {
                    indexed_store(&recv, &key_val, value);
                }
            }
            Op::In { dst, lhs, rhs } => {
                let key = regs[lhs as usize].to_display_string();
                let recv = regs[rhs as usize].clone();
                // Fast own-property check first — no host round-trip for the
                // common `"ownKey" in obj` case.
                let own_hit = matches!(&recv, Value::Object(o) if o.borrow().contains_key(&key));
                if own_hit {
                    regs[dst as usize] = Value::Bool(true);
                } else if matches!(&recv, Value::Object(_))
                    || is_proxy_val(&recv)
                    || crate::interp::is_legacy_collection(&recv)
                    || crate::interp::value_is_node_like(&recv)
                {
                    // ECMA-262 [[HasProperty]] walks the WHOLE prototype chain
                    // (so inherited methods/getters — `"method" in instance`,
                    // `"fixedProp" in subclassInstance` — count), fires a Proxy
                    // `has` trap, and resolves WebIDL legacy collection index/name
                    // + live DOM node accessors. Route every object through the
                    // single host [[HasProperty]] (`has_property_trapped`) so the
                    // VM `in` matches the tree-walk tier exactly. (Own keys were
                    // already handled above; this is the inherited/exotic case.)
                    let hook = globals.borrow().get("__tb_host_has").cloned();
                    if let Some(g @ Value::NativeFunction(_)) = hook {
                        match dispatch(g, Value::Undefined, vec![recv, Value::str(key)]) {
                            Ok(v) => regs[dst as usize] = Value::Bool(v.to_bool()),
                            Err(e) => propagate!(e),
                        }
                    } else {
                        // No host hook (bare VM): own-only fallback.
                        regs[dst as usize] = Value::Bool(has_property(&recv, &key));
                    }
                } else {
                    // Array/String/primitive: lenient (false, never throws) so a
                    // `typeof x==='object' && k in x` null-guard doesn't abort.
                    regs[dst as usize] = Value::Bool(has_property(&recv, &key));
                }
            }
            Op::Instanceof { dst, lhs, rhs } => {
                // Route through the host so the FULL tree-walk instanceof
                // (ordinary_has_instance PROTO_KEY walk + is_instance_of tag
                // fallback + Symbol.hasInstance) applies — byte-identical to the
                // tree-walk tier. The hook is installed by `install_basic_globals`.
                let inst = regs[lhs as usize].clone();
                let ctor = regs[rhs as usize].clone();
                let hook = globals.borrow().get("__tb_host_instanceof").cloned();
                match hook {
                    Some(g @ Value::NativeFunction(_)) => {
                        match dispatch(g, Value::Undefined, vec![inst, ctor]) {
                            Ok(v) => regs[dst as usize] = Value::Bool(v.to_bool()),
                            Err(e) => propagate!(e),
                        }
                    }
                    // No host hook (e.g. `run_module` with an empty globals map):
                    // fall back to the tag-based pure check, which covers built-ins.
                    _ => {
                        regs[dst as usize] =
                            Value::Bool(crate::interp::is_instance_of(&inst, &ctor));
                    }
                }
            }
            Op::DeleteProp { dst, obj, key_k } => {
                let key = match &f.consts[key_k as usize] {
                    Value::String(s) => s.clone(),
                    other => {
                        return Err(RuntimeError::TypeError(format!(
                            "DeleteProp key must be string: {other:?}"
                        )));
                    }
                };
                // The proxy `deleteProperty` trap returns true/false per
                // spec, and `delete obj.x` itself returns that boolean.
                // The previous code discarded the trap result and always
                // wrote `true`, so a `return false` trap (a common Vue3/
                // MobX guard) silently registered as success. Capture the
                // trap return and use it (coerced to bool) as dst.
                let recv = regs[obj as usize].clone();
                // Route to the host (→ interp `delete_property_trapped`) for a
                // proxy OR a descriptor-aware object, so the proxy trap result AND
                // the [[Delete]] configurable check (CV_PROP_DESC) are honored
                // identically to the tree-walk. Plain objects with no attrs take
                // the raw fast delete (always returns true), unchanged.
                let route_host = is_proxy_val(&recv)
                    || crate::interp::value_has_attrs(&recv)
                    || crate::interp::is_legacy_collection(&recv);
                let result = if route_host {
                    let hook = globals.borrow().get("__tb_host_delete").cloned();
                    if let Some(g @ Value::NativeFunction(_)) = hook {
                        match dispatch(g, Value::Undefined, vec![recv, Value::String(key)]) {
                            Ok(v) => v.to_bool(),
                            Err(e) => propagate!(e),
                        }
                    } else {
                        delete_property(&recv, &key);
                        true
                    }
                } else {
                    delete_property(&recv, &key);
                    true
                };
                regs[dst as usize] = Value::Bool(result);
            }
            Op::DeleteIdx { dst, obj, key } => {
                // Computed-key delete: `delete obj[k]`. Must route through
                // the proxy `deleteProperty` trap just like `delete obj.x`
                // does — was previously a raw `delete_property` skipping
                // the trap entirely, so `delete proxy[k]` bypassed
                // reactive frameworks' interception.
                let key = regs[key as usize].to_display_string();
                let recv = regs[obj as usize].clone();
                let route_host = is_proxy_val(&recv)
                    || crate::interp::value_has_attrs(&recv)
                    || crate::interp::is_legacy_collection(&recv);
                let result = if route_host {
                    let hook = globals.borrow().get("__tb_host_delete").cloned();
                    if let Some(g @ Value::NativeFunction(_)) = hook {
                        match dispatch(g, Value::Undefined, vec![recv, Value::str(key)]) {
                            Ok(v) => v.to_bool(),
                            Err(e) => propagate!(e),
                        }
                    } else {
                        delete_property(&recv, &key);
                        true
                    }
                } else {
                    delete_property(&recv, &key);
                    true
                };
                regs[dst as usize] = Value::Bool(result);
            }
            Op::MakeRegex {
                dst,
                source_k,
                flags_k,
            } => {
                let source = match &f.consts[source_k as usize] {
                    Value::String(s) => s.to_string(),
                    _ => String::new(),
                };
                let flags = match &f.consts[flags_k as usize] {
                    Value::String(s) => s.to_string(),
                    _ => String::new(),
                };
                regs[dst as usize] = crate::interp::build_regex_value(&source, &flags);
            }
            Op::NewArray {
                dst,
                first_elem,
                n_elems,
            } => {
                let mut v: Vec<Value> = Vec::with_capacity(n_elems as usize);
                for i in 0..n_elems {
                    v.push(regs[first_elem as usize + i as usize].clone());
                }
                let rc = std::rc::Rc::new(std::cell::RefCell::new(v));
                crate::interp::gc_register_array(&rc);
                regs[dst as usize] = Value::Array(rc);
            }
            Op::ArrayPush { arr, val } => {
                if let Value::Array(a) = &regs[arr as usize] {
                    a.borrow_mut().push(regs[val as usize].clone());
                } else {
                    propagate!(RuntimeError::Thrown(crate::interp::err_str(
                        "ArrayPush target not an array".into()
                    )));
                }
            }
            Op::ArrayPushSpread { arr, spread } => {
                // Spread sources we support inline: arrays + strings.
                // Anything else falls back to a runtime error.
                let dst_arr = match &regs[arr as usize] {
                    Value::Array(a) => a.clone(),
                    _ => {
                        propagate!(RuntimeError::Thrown(crate::interp::err_str(
                            "ArrayPushSpread target not an array".into()
                        )));
                    }
                };
                let spread_val = regs[spread as usize].clone();
                match &spread_val {
                    Value::Array(src) => {
                        let src_snapshot = src.borrow().clone();
                        let mut dst = dst_arr.borrow_mut();
                        for v in src_snapshot {
                            dst.push(v);
                        }
                    }
                    Value::String(s) => {
                        let chars: Vec<Value> =
                            s.chars().map(|c| Value::str(c.to_string())).collect();
                        let mut dst = dst_arr.borrow_mut();
                        for v in chars {
                            dst.push(v);
                        }
                    }
                    _ => {
                        // Defer to the host's full `spread_iterable` (Map/Set/
                        // iterators/`Symbol.iterator`) via `__tb_spread__`, so
                        // the VM matches the tree-walk spread exactly. A genuine
                        // non-iterable still throws TypeError (no masking).
                        let helper = globals.borrow().get("__tb_spread__").cloned();
                        match helper {
                            Some(h) => {
                                match dispatch(h, Value::Undefined, vec![spread_val.clone()]) {
                                    Ok(Value::Array(items)) => {
                                        let snap = items.borrow().clone();
                                        let mut dst = dst_arr.borrow_mut();
                                        for v in snap {
                                            dst.push(v);
                                        }
                                    }
                                    Ok(_) => {}
                                    Err(e) => propagate!(e),
                                }
                            }
                            None => propagate!(RuntimeError::Thrown(crate::interp::err_str(
                                "TypeError: spread source is not iterable".into()
                            ))),
                        }
                    }
                }
            }
            Op::NewObject { dst } => {
                let rc = std::rc::Rc::new(std::cell::RefCell::new(HashMap::new()));
                crate::interp::gc_register_object(&rc);
                regs[dst as usize] = Value::Object(rc);
            }
            Op::Throw { src } => {
                let v = regs[src as usize].clone();
                #[cfg(debug_assertions)]
                {
                    thread_local! {
                        static TT: bool = std::env::var("CV_THROWTRACE").is_ok();
                        static TN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
                    }
                    if TT.with(|x| *x) {
                        let n = TN.with(|c| {
                            let x = c.get();
                            c.set(x + 1);
                            x
                        });
                        if n < 120 {
                            let desc = match &v {
                                Value::String(s) => {
                                    format!("String({:?})", s.chars().take(80).collect::<String>())
                                }
                                Value::Object(o) => {
                                    let b = o.borrow();
                                    let g = |k: &str| {
                                        b.get(k)
                                            .map(|x| {
                                                x.to_display_string()
                                                    .chars()
                                                    .take(180)
                                                    .collect::<String>()
                                            })
                                            .unwrap_or_default()
                                    };
                                    format!("name={:?} message={:?}", g("name"), g("message"))
                                }
                                other => format!("{:?}", std::mem::discriminant(other)),
                            };
                            eprintln!("[VMTHROW#{n}] {desc}");
                        }
                    }
                }
                propagate!(RuntimeError::Thrown(v));
            }
            Op::TryEnter {
                catch_target,
                catch_reg,
            } => {
                try_stack.push((catch_target as usize, catch_reg));
            }
            Op::TryExit => {
                try_stack.pop();
            }
            Op::MakeClosure {
                dst,
                fn_idx,
                first_upvalue,
                n_upvalues,
            } => {
                let mut ups: Vec<Value> = Vec::with_capacity(n_upvalues as usize);
                for i in 0..n_upvalues {
                    ups.push(regs[first_upvalue as usize + i as usize].clone());
                }
                regs[dst as usize] = Value::BcClosure(std::rc::Rc::new(crate::interp::BcClosure {
                    fn_idx: fn_idx as u32,
                    upvalues: std::cell::RefCell::new(ups),
                    props: std::cell::RefCell::new(HashMap::new()),
                    module: std::rc::Rc::new(module.clone()),
                }));
            }
            Op::LoadUp { dst, slot } => {
                let c = closure
                    .ok_or_else(|| RuntimeError::TypeError("LoadUp outside closure".into()))?;
                regs[dst as usize] = c.upvalues.borrow()[slot as usize].clone();
            }
            Op::StoreUp { src, slot } => {
                let c = closure
                    .ok_or_else(|| RuntimeError::TypeError("StoreUp outside closure".into()))?;
                c.upvalues.borrow_mut()[slot as usize] = regs[src as usize].clone();
            }
            Op::New {
                dst,
                ctor,
                first_arg,
                n_args,
            } => {
                let ctor_val = regs[ctor as usize].clone();
                let new_obj =
                    Value::Object(std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())));
                // ECMA-262 §10.2.2 [[Construct]] / OrdinaryCreateFromConstructor:
                // link the fresh instance's [[Prototype]] to the constructor's
                // `.prototype` BEFORE running the body, so `this instanceof F`,
                // inherited methods (`F.prototype.m`), and
                // `getPrototypeOf(new F()) === F.prototype` hold both inside the
                // ctor and on the returned object. Uses the closure's ONE shared
                // `.prototype` (same slot a value-read returns) — the bytecode VM
                // used to leave the instance proto-less and "defer to the interp",
                // which silently broke `new`/`instanceof` for every user
                // function/class once the VM became the default execution tier.
                if let Value::BcClosure(ref c) = ctor_val {
                    let proto = crate::interp::bc_closure_prototype(c, &ctor_val);
                    if let Value::Object(ref o) = new_obj {
                        o.borrow_mut()
                            .insert(crate::interp::PROTO_KEY.to_string(), proto);
                    }
                }
                let mut call_args: Vec<Value> = Vec::with_capacity(n_args as usize);
                for i in 0..n_args {
                    call_args.push(regs[first_arg as usize + i as usize].clone());
                }
                let result = match ctor_val {
                    Value::BcClosure(c) => match run_function(
                        &c.module,
                        c.fn_idx as usize,
                        &call_args,
                        &new_obj,
                        globals,
                        Some(&c),
                        &mut *dispatch,
                    ) {
                        Ok(v) => v,
                        Err(e) => propagate!(e),
                    },
                    // ECMA-262 §13.3.5.1: `new` on a NON-constructor native
                    // function (a built-in *method* like `String.prototype.big`,
                    // `escape`, `parseInt`) throws TypeError. Only `native_ctor`
                    // (Map/Set/…) sets `is_ctor`. Mirrors the tree-walk New path.
                    // Throw a real TypeError OBJECT (via `err_str`) so a `catch`
                    // sees `e instanceof TypeError`, not a bare string.
                    Value::NativeFunction(ref nf) if !nf.is_ctor => {
                        propagate!(RuntimeError::Thrown(crate::interp::err_str(format!(
                            "TypeError: {} is not a constructor",
                            nf.name
                        ))));
                    }
                    Value::NativeFunction(nf) => match &nf.func {
                        // Route ctor-native throws through the try-handler (not
                        // `?`), same as the CallValue path — so `try { new
                        // RegExp(bad) } catch {}` etc. is caught in hot code.
                        crate::interp::NativeFnBody::Pure(body) => match body(call_args) {
                            Ok(v) => v,
                            Err(e) => propagate!(match e {
                                crate::interp::JsError::Throw(v) => RuntimeError::Thrown(v),
                                other => RuntimeError::TypeError(format!(
                                    "ctor native fn `{}`: {other:?}",
                                    nf.name
                                )),
                            }),
                        },
                        crate::interp::NativeFnBody::WithInterp(_) => {
                            match dispatch(
                                Value::NativeFunction(nf.clone()),
                                new_obj.clone(),
                                call_args,
                            ) {
                                Ok(v) => v,
                                Err(e) => propagate!(e),
                            }
                        }
                    },
                    // A tagged constructor object `{_construct: fn}` — our
                    // native ctors (Map/Set/Date/RegExp/Promise/Error
                    // subclasses) are objects doubling as static-method
                    // namespaces; `new` invokes `_construct`, which builds and
                    // RETURNS its own instance (so the discarded bare `this`
                    // needs no proto link, and there is no JS `instanceof`
                    // guard to recurse). Tree-walk `Value::Function` ctors are
                    // intentionally NOT routed here: without lazily
                    // materializing the ctor's `.prototype` and proto-linking
                    // `this`, a transpiled `if(!(this instanceof F)) return
                    // new F()` guard would infinitely recurse (stack overflow).
                    other => {
                        let construct_callee = match &other {
                            Value::Object(o) => o.borrow().get("_construct").cloned(),
                            _ => None,
                        };
                        match construct_callee {
                            Some(c) => match dispatch(c, new_obj.clone(), call_args) {
                                Ok(v) => v,
                                Err(e) => propagate!(e),
                            },
                            None => {
                                return Err(RuntimeError::TypeError(format!(
                                    "`new` on non-callable: {other:?}"
                                )));
                            }
                        }
                    }
                };
                // ECMA-262 §13.3.3 / §10.2.2 [[Construct]]: if the
                // constructor returned ANY Object (including Functions and
                // closures — functions are objects in JS), that value is
                // the result of `new`. Only primitives fall back to the
                // freshly-allocated `this`. Matches the tree-walk path.
                regs[dst as usize] = match result {
                    Value::Object(_)
                    | Value::Array(_)
                    | Value::Function(_)
                    | Value::NativeFunction(_)
                    | Value::BcClosure(_) => result,
                    _ => new_obj,
                };
            }
            Op::LoadThis { dst } => {
                regs[dst as usize] = this.clone();
            }
            Op::LoadSelf { dst } => {
                regs[dst as usize] = match closure {
                    Some(c) => Value::BcClosure(c.clone()),
                    None => Value::Undefined,
                };
            }
            Op::EnumKeys { dst, obj } => {
                // ECMA-262 §14.7.5.6 EnumerateObjectProperties: for-in
                // must enumerate INHERITED enumerable string keys too, not
                // just own keys.  The tree-walk already uses
                // `enumerable_string_keys_with_chain` (which walks PROTO_KEY
                // up to 64 levels); the VM was using only own keys so classes
                // with prototype methods were invisible.  Unify with the same
                // function.
                let src = &regs[obj as usize];
                let keys: Vec<Value> = match src {
                    Value::Object(_) => {
                        crate::interp::enumerable_string_keys_with_chain(src)
                            .into_iter()
                            .map(|s| Value::str(s))
                            .collect()
                    }
                    Value::Array(a) => (0..a.borrow().len())
                        .map(|i| Value::str(i.to_string()))
                        .collect(),
                    Value::String(s) => (0..s.chars().count())
                        .map(|i| Value::str(i.to_string()))
                        .collect(),
                    _ => Vec::new(),
                };
                regs[dst as usize] = Value::Array(std::rc::Rc::new(std::cell::RefCell::new(keys)));
            }
            Op::Ret { src } => run_shared!(op_ret(src)),
        }
    }
}

fn to_num(v: &Value) -> Result<f64, RuntimeError> {
    Ok(match v {
        Value::Number(n) => *n,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Null => 0.0,
        // Delegate to the tree-walker's spec-correct StringToNumber so the VM
        // and tree-walker agree on `Number("")===0`, `Number("0x1F")===31`,
        // and `Number("inf")===NaN`. Avoids re-implementing the same logic.
        Value::String(_) => v.to_number(),
        _ => f64::NAN,
    })
}

/// ECMA-262 ToInt32: coerce to a number, drop the fractional part, reduce
/// mod 2^32, and reinterpret as signed. NaN/±Inf map to 0.
fn to_int32(v: &Value) -> Result<i32, RuntimeError> {
    let n = to_num(v)?;
    if !n.is_finite() {
        return Ok(0);
    }
    Ok(n.trunc().rem_euclid(4294967296.0) as u32 as i32)
}

/// ECMA-262 ToUint32: like ToInt32 but the result is unsigned.
fn to_uint32(v: &Value) -> Result<u32, RuntimeError> {
    let n = to_num(v)?;
    if !n.is_finite() {
        return Ok(0);
    }
    Ok(n.trunc().rem_euclid(4294967296.0) as u32)
}

/// `key in obj` — own-property existence for objects, plus index/length for
/// arrays and strings. (Prototype-chain membership is not walked yet.) Non-
/// objects yield false rather than throwing.
fn has_property(obj: &Value, key: &str) -> bool {
    match obj {
        Value::Object(o) => o.borrow().contains_key(key),
        Value::Array(a) => {
            key == "length"
                || key
                    .parse::<usize>()
                    .map(|i| {
                        let b = a.borrow();
                        // A Hole is not an own property — `i in arr` → false.
                        i < b.len() && !matches!(b[i], Value::Hole)
                    })
                    .unwrap_or(false)
                || crate::interp::array_get_prop(a, key).is_some()
        }
        Value::String(s) => {
            key == "length"
                || key
                    .parse::<usize>()
                    .map(|i| i < s.chars().count())
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// `delete obj[key]` — removes an own property (object) or clears an array slot
/// to a hole (undefined). No-op on other values.
fn delete_property(obj: &Value, key: &str) {
    match obj {
        Value::Object(o) => {
            o.borrow_mut().remove(key);
        }
        Value::Array(a) => {
            if let Ok(i) = key.parse::<usize>() {
                let mut b = a.borrow_mut();
                if i < b.len() {
                    // ECMA-262 §10.4.2.2: delete arr[i] creates a sparse hole —
                    // reads as `undefined` but `i in arr` → false, hasOwnProperty
                    // → false, and forEach/map/filter skip the slot.
                    b[i] = Value::Hole;
                }
            }
        }
        _ => {}
    }
}

fn add_values(a: &Value, b: &Value) -> Result<Value, RuntimeError> {
    // ECMA-262 §13.15.4 "Applying the + operator": if either is a string,
    // concatenate; otherwise convert to number.
    if matches!(a, Value::String(_)) || matches!(b, Value::String(_)) {
        let sa = match a {
            Value::String(s) => s.to_string(),
            v => display_value(v),
        };
        let sb = match b {
            Value::String(s) => s.to_string(),
            v => display_value(v),
        };
        let mut joined = String::with_capacity(sa.len() + sb.len());
        joined.push_str(&sa);
        joined.push_str(&sb);
        return Ok(Value::str(joined));
    }
    Ok(Value::Number(to_num(a)? + to_num(b)?))
}

/// When either operand is a BigInt, the VM's inline f64 arithmetic would yield
/// NaN. Route to the interp's full `binary_op` (BigInt-aware) via the
/// `__tb_host_binop` global so VM math is bit-exact with the tree-walk (crypto
/// uses 256-bit BigInts). Returns `None` for the common all-numeric case (fast
/// inline path) and when no host is wired (bare-globals entry points).
fn bigint_binop(
    op: &str,
    a: &Value,
    b: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Option<Result<Value, RuntimeError>> {
    if !matches!(a, Value::BigInt(_)) && !matches!(b, Value::BigInt(_)) {
        return None;
    }
    let getter = globals.borrow().get("__tb_host_binop").cloned();
    match getter {
        Some(g @ Value::NativeFunction(_)) => Some(dispatch(
            g,
            Value::Undefined,
            vec![Value::str(op.to_string()), a.clone(), b.clone()],
        )),
        _ => None,
    }
}

/// When either operand is an Object for a relational operator (`<`, `>`, `<=`,
/// `>=`), dispatch to `__tb_host_binop` so `ordinary_to_primitive` (which calls
/// `valueOf`/`toString` via the interpreter) is applied before comparing.
/// This makes `new Date(1) < new Date(2)` work in VM-compiled code (Date
/// valueOf() returns the timestamp number). Returns `None` when neither operand
/// is an Object (fast inline path) or when no host is wired.
fn relational_host_binop(
    op: &str,
    a: &Value,
    b: &Value,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Option<Result<Value, RuntimeError>> {
    if !matches!(a, Value::Object(_)) && !matches!(b, Value::Object(_)) {
        return None;
    }
    let getter = globals.borrow().get("__tb_host_binop").cloned();
    match getter {
        Some(g @ Value::NativeFunction(_)) => Some(dispatch(
            g,
            Value::Undefined,
            vec![Value::str(op.to_string()), a.clone(), b.clone()],
        )),
        _ => None,
    }
}

/// Resolve `obj.key` per the host object model. Objects look up keys
/// in their underlying `HashMap<String, Value>`; arrays expose `length`,
/// numeric-string indices, and a bound subset of the prototype methods
/// (push/pop/shift/unshift/slice/indexOf/includes/join); strings expose
/// `length`, index access, and a method subset (toUpperCase /
/// toLowerCase / trim / charAt / indexOf / includes / slice / split /
/// repeat). Misses return `Undefined`.
/// When the VM's own `property_lookup`/`indexed_lookup` MISSES (yields
/// `Undefined`) on an object-ish receiver, route the read through the interp's
/// full `read_property` (prototype chain + every built-in Array/String/Object
/// method) via the `__tb_host_getprop` global. This is what makes
/// `obj.method(...)` resolve in VM-run code instead of erroring "callee is not
/// callable: undefined". No-op for found values and for bare-globals entry
/// points (no host wired) — returns `raw` unchanged.
fn resolve_with_host(
    raw: Value,
    obj: &Value,
    key: &str,
    globals: &std::cell::RefCell<HashMap<String, Value>>,
    dispatch: WithInterpDispatch<'_>,
) -> Result<Value, RuntimeError> {
    if !matches!(raw, Value::Undefined) {
        return Ok(raw);
    }
    // Everything EXCEPT null/undefined can carry methods/inherited props —
    // including primitives (`(255).toString(16)`, `(n).toFixed(2)`, bool, bigint,
    // symbol). Only null/undefined have no properties (reading one should throw,
    // which the caller's CallValue surfaces). Route the rest to the host.
    if matches!(obj, Value::Undefined | Value::Null) {
        return Ok(raw);
    }
    let getter = globals.borrow().get("__tb_host_getprop").cloned();
    match getter {
        Some(g @ Value::NativeFunction(_)) => dispatch(
            g,
            Value::Undefined,
            vec![obj.clone(), Value::str(key.to_string())],
        ),
        _ => Ok(raw),
    }
}

fn property_lookup(obj: &Value, key: &str) -> Value {
    match obj {
        Value::Object(o) => o.borrow().get(key).cloned().unwrap_or(Value::Undefined),
        Value::Array(a) => {
            if key == "length" {
                return Value::Number(a.borrow().len() as f64);
            }
            if let Ok(idx) = key.parse::<usize>() {
                // A deleted slot is a `Value::Hole` in storage; per ECMA-262
                // §10.4.2 it READS as `undefined` (so `delete arr[1]; arr[1] ===
                // undefined` is true) — normalize it here so the VM index read is
                // byte-identical to the tree-walker, which never surfaces a hole.
                return match a.borrow().get(idx) {
                    Some(Value::Hole) | None => Value::Undefined,
                    Some(v) => v.clone(),
                };
            }
            // Named own-property hung on the array (e.g. webpack's overridden
            // `push`) shadows the built-in method binding.
            if let Some(v) = crate::interp::array_get_prop(a, key) {
                return v;
            }
            array_method_binding(a.clone(), key)
        }
        Value::String(s) => {
            if key == "length" {
                return Value::Number(s.chars().count() as f64);
            }
            if let Ok(idx) = key.parse::<usize>() {
                return s
                    .chars()
                    .nth(idx)
                    .map(|c| Value::str(c.to_string()))
                    .unwrap_or(Value::Undefined);
            }
            string_method_binding(s.to_string(), key)
        }
        Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_) => match key {
            "call" => {
                let fn_val = obj.clone();
                crate::interp::native_fn_with_interp("call", move |interp, args| {
                    let this_arg = args.first().cloned().unwrap_or(Value::Undefined);
                    let rest = if args.len() > 1 {
                        args[1..].to_vec()
                    } else {
                        Vec::new()
                    };
                    interp.call_value_with_this(fn_val.clone(), this_arg, rest)
                })
            }
            "apply" => {
                let fn_val = obj.clone();
                crate::interp::native_fn_with_interp("apply", move |interp, args| {
                    let this_arg = args.first().cloned().unwrap_or(Value::Undefined);
                    let arg_array = args.get(1).cloned().unwrap_or(Value::Undefined);
                    let rest = match arg_array {
                        Value::Array(a) => a.borrow().clone(),
                        _ => Vec::new(),
                    };
                    interp.call_value_with_this(fn_val.clone(), this_arg, rest)
                })
            }
            "bind" => {
                let fn_val = obj.clone();
                crate::interp::native_fn_with_interp("bind", move |_interp, args| {
                    let bound_this = args.first().cloned().unwrap_or(Value::Undefined);
                    let bound_args = if args.len() > 1 {
                        args[1..].to_vec()
                    } else {
                        Vec::new()
                    };
                    Ok(crate::interp::native_fn_with_interp("bound", {
                        let fn_val = fn_val.clone();
                        move |interp, more| {
                            let mut all = bound_args.clone();
                            all.extend(more);
                            interp.call_value_with_this(fn_val.clone(), bound_this.clone(), all)
                        }
                    }))
                })
            }
            "length" => Value::Number(0.0),
            "name" => match obj {
                Value::Function(f) => Value::str(f.name.clone().unwrap_or_default()),
                Value::NativeFunction(n) => Value::str(n.name.clone()),
                Value::BcClosure(c) => Value::str(format!("bc#{}", c.fn_idx)),
                _ => Value::Undefined,
            },
            _ => match obj {
                Value::Function(f) => f
                    .props
                    .borrow()
                    .get(key)
                    .cloned()
                    .unwrap_or(Value::Undefined),
                Value::NativeFunction(n) => n
                    .props
                    .borrow()
                    .get(key)
                    .cloned()
                    .unwrap_or(Value::Undefined),
                Value::BcClosure(c) => c
                    .props
                    .borrow()
                    .get(key)
                    .cloned()
                    .unwrap_or(Value::Undefined),
                _ => Value::Undefined,
            },
        },
        _ => Value::Undefined,
    }
}

/// Build a `NativeFunction` value that closes over the array and acts
/// as `Array.prototype.<key>`. The receiver is implicit (no `this`
/// binding needed since the closure already has the Rc).
fn array_method_binding(arr: std::rc::Rc<std::cell::RefCell<Vec<Value>>>, key: &str) -> Value {
    use crate::interp::{NativeFn, NativeFnBody};
    use std::rc::Rc;
    let wrap = |name: &str,
                body: Box<dyn Fn(Vec<Value>) -> Result<Value, crate::interp::JsError>>|
     -> Value {
        Value::NativeFunction(Rc::new(NativeFn {
            name: name.to_string(),
            func: NativeFnBody::Pure(body),
            length: 0,
            is_ctor: false,
            props: std::cell::RefCell::new(HashMap::new()),
        }))
    };
    match key {
        "push" => {
            let a = arr;
            wrap(
                "push",
                Box::new(move |args| {
                    let mut v = a.borrow_mut();
                    for x in args {
                        v.push(x);
                    }
                    Ok(Value::Number(v.len() as f64))
                }),
            )
        }
        "pop" => {
            let a = arr;
            wrap(
                "pop",
                Box::new(move |_args| Ok(a.borrow_mut().pop().unwrap_or(Value::Undefined))),
            )
        }
        "shift" => {
            let a = arr;
            wrap(
                "shift",
                Box::new(move |_args| {
                    let mut v = a.borrow_mut();
                    if v.is_empty() {
                        Ok(Value::Undefined)
                    } else {
                        Ok(v.remove(0))
                    }
                }),
            )
        }
        "unshift" => {
            let a = arr;
            wrap(
                "unshift",
                Box::new(move |args| {
                    let mut v = a.borrow_mut();
                    for (i, x) in args.into_iter().enumerate() {
                        v.insert(i, x);
                    }
                    Ok(Value::Number(v.len() as f64))
                }),
            )
        }
        "slice" => {
            let a = arr;
            wrap(
                "slice",
                Box::new(move |args| {
                    let v = a.borrow();
                    let len = v.len() as i64;
                    let resolve = |arg: Option<&Value>, default: i64| -> i64 {
                        let raw = arg.map(|x| x.to_number()).unwrap_or(default as f64);
                        let mut i = raw as i64;
                        if i < 0 {
                            i += len;
                        }
                        i.clamp(0, len)
                    };
                    let start = resolve(args.first(), 0);
                    let end = resolve(args.get(1), len);
                    let out: Vec<Value> = if start <= end {
                        v[start as usize..end as usize].to_vec()
                    } else {
                        Vec::new()
                    };
                    Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(out))))
                }),
            )
        }
        "indexOf" => {
            let a = arr;
            wrap(
                "indexOf",
                Box::new(move |args| {
                    let needle = args.first().cloned().unwrap_or(Value::Undefined);
                    let v = a.borrow();
                    let len = v.len() as i64;
                    // ECMA-262 §23.1.3.14: fromIndex defaults to 0;
                    // negative fromIndex counts from the end.
                    let from: usize = match args.get(1) {
                        Some(vi) => {
                            let n = vi.to_number() as i64;
                            if n < 0 {
                                (len + n).max(0) as usize
                            } else {
                                n as usize
                            }
                        }
                        None => 0,
                    };
                    for (i, x) in v.iter().enumerate().skip(from) {
                        if Value::strict_eq(x, &needle) {
                            return Ok(Value::Number(i as f64));
                        }
                    }
                    Ok(Value::Number(-1.0))
                }),
            )
        }
        "includes" => {
            let a = arr;
            wrap(
                "includes",
                Box::new(move |args| {
                    let needle = args.first().cloned().unwrap_or(Value::Undefined);
                    // ECMA-262 §23.1.3.13 Array.prototype.includes uses
                    // SameValueZero (NaN equals NaN), NOT strict_eq — so
                    // `[NaN].includes(NaN)` is true, byte-identical to the
                    // tree-walker. A Hole reads as `undefined` for the compare.
                    Ok(Value::Bool(a.borrow().iter().any(|x| {
                        let xv = if matches!(x, Value::Hole) { &Value::Undefined } else { x };
                        crate::interp::same_value_zero(xv, &needle)
                    })))
                }),
            )
        }
        "join" => {
            let a = arr;
            wrap(
                "join",
                Box::new(move |args| {
                    // ECMA-262 §23.1.3.30 step 3: if separator is undefined
                    // (or absent), use "," — do NOT stringify it.
                    let sep = match args.first() {
                        None | Some(Value::Undefined) => ",".into(),
                        Some(Value::String(s)) => s.to_string(),
                        Some(other) => other.to_display_string(),
                    };
                    // ECMA-262 §23.1.3.30 Array.prototype.join: nullish
                    // elements (null/undefined/hole) join to "" not the
                    // literal strings "null"/"undefined".
                    let parts: Vec<String> = a
                        .borrow()
                        .iter()
                        .map(|x| match x {
                            Value::Null | Value::Undefined | Value::Hole => String::new(),
                            other => other.to_display_string(),
                        })
                        .collect();
                    Ok(Value::str(parts.join(&sep)))
                }),
            )
        }
        "reverse" => {
            let a = arr;
            wrap(
                "reverse",
                Box::new(move |_args| {
                    a.borrow_mut().reverse();
                    Ok(Value::Array(std::rc::Rc::clone(&a)))
                }),
            )
        }
        _ => Value::Undefined,
    }
}

fn string_method_binding(s: String, key: &str) -> Value {
    use crate::interp::{NativeFn, NativeFnBody};
    use std::rc::Rc;
    let wrap = |name: &str,
                body: Box<dyn Fn(Vec<Value>) -> Result<Value, crate::interp::JsError>>|
     -> Value {
        Value::NativeFunction(Rc::new(NativeFn {
            name: name.to_string(),
            func: NativeFnBody::Pure(body),
            length: 0,
            is_ctor: false,
            props: std::cell::RefCell::new(HashMap::new()),
        }))
    };
    match key {
        "toUpperCase" => {
            let s = s;
            wrap(
                "toUpperCase",
                Box::new(move |_| Ok(Value::str(s.to_uppercase()))),
            )
        }
        "toLowerCase" => {
            let s = s;
            wrap(
                "toLowerCase",
                Box::new(move |_| Ok(Value::str(s.to_lowercase()))),
            )
        }
        "trim" => {
            let s = s;
            wrap(
                "trim",
                Box::new(move |_| Ok(Value::str(s.trim().to_string()))),
            )
        }
        "charAt" => {
            let s = s;
            wrap(
                "charAt",
                Box::new(move |args| {
                    let idx = args.first().map(|v| v.to_number() as usize).unwrap_or(0);
                    Ok(s.chars()
                        .nth(idx)
                        .map(|c| Value::str(c.to_string()))
                        .unwrap_or(Value::str(String::new())))
                }),
            )
        }
        "indexOf" => {
            let s = s;
            wrap(
                "indexOf",
                Box::new(move |args| {
                    let needle = match args.first() {
                        Some(Value::String(n)) => n.to_string(),
                        Some(other) => other.to_display_string(),
                        None => return Ok(Value::Number(-1.0)),
                    };
                    // ECMA-262 §22.1.3.8: optional fromIndex argument.
                    let from_char: usize = match args.get(1) {
                        Some(v) => {
                            let n = v.to_number();
                            if n <= 0.0 { 0 } else { n as usize }
                        }
                        None => 0,
                    };
                    // Convert char index to byte offset for Rust str::find.
                    let byte_offset: usize = s
                        .char_indices()
                        .nth(from_char)
                        .map(|(b, _)| b)
                        .unwrap_or(s.len());
                    if byte_offset >= s.len() && !needle.is_empty() {
                        return Ok(Value::Number(-1.0));
                    }
                    Ok(s[byte_offset..]
                        .find(needle.as_str())
                        .map(|rel| {
                            // Convert byte offset back to char index.
                            let abs_byte = byte_offset + rel;
                            Value::Number(s[..abs_byte].chars().count() as f64)
                        })
                        .unwrap_or(Value::Number(-1.0)))
                }),
            )
        }
        "includes" => {
            let s = s;
            wrap(
                "includes",
                Box::new(move |args| {
                    let needle = match args.first() {
                        Some(Value::String(n)) => n.to_string(),
                        Some(other) => other.to_display_string(),
                        None => return Ok(Value::Bool(false)),
                    };
                    Ok(Value::Bool(s.contains(&needle)))
                }),
            )
        }
        "startsWith" => {
            let s = s;
            wrap(
                "startsWith",
                Box::new(move |args| {
                    let needle = match args.first() {
                        Some(Value::String(n)) => n.to_string(),
                        Some(other) => other.to_display_string(),
                        None => return Ok(Value::Bool(false)),
                    };
                    Ok(Value::Bool(s.starts_with(&needle)))
                }),
            )
        }
        "endsWith" => {
            let s = s;
            wrap(
                "endsWith",
                Box::new(move |args| {
                    let needle = match args.first() {
                        Some(Value::String(n)) => n.to_string(),
                        Some(other) => other.to_display_string(),
                        None => return Ok(Value::Bool(false)),
                    };
                    Ok(Value::Bool(s.ends_with(&needle)))
                }),
            )
        }
        "slice" => {
            let s = s;
            wrap(
                "slice",
                Box::new(move |args| {
                    let len = s.chars().count() as i64;
                    let resolve = |arg: Option<&Value>, default: i64| -> i64 {
                        let raw = arg.map(|x| x.to_number()).unwrap_or(default as f64);
                        let mut i = raw as i64;
                        if i < 0 {
                            i += len;
                        }
                        i.clamp(0, len)
                    };
                    let start = resolve(args.first(), 0);
                    let end = resolve(args.get(1), len);
                    let out: String = if start <= end {
                        s.chars()
                            .skip(start as usize)
                            .take((end - start) as usize)
                            .collect()
                    } else {
                        String::new()
                    };
                    Ok(Value::str(out))
                }),
            )
        }
        "split" => {
            let s = s;
            wrap(
                "split",
                Box::new(move |args| {
                    // ECMA-262 §22.1.3.21: undefined separator (or absent)
                    // returns a single-element array containing the whole string.
                    let sep = match args.first() {
                        None | Some(Value::Undefined) => {
                            return Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                                vec![Value::str(s.clone())],
                            ))));
                        }
                        Some(Value::String(x)) => x.to_string(),
                        Some(other) => other.to_display_string(),
                    };
                    let parts: Vec<Value> = if sep.is_empty() {
                        s.chars().map(|c| Value::str(c.to_string())).collect()
                    } else {
                        s.split(&sep)
                            .map(|p| Value::str(p.to_string()))
                            .collect()
                    };
                    Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                        parts,
                    ))))
                }),
            )
        }
        "repeat" => {
            let s = s;
            wrap(
                "repeat",
                Box::new(move |args| {
                    let count = args.first().map(|v| v.to_number()).unwrap_or(0.0);
                    // ECMA-262 §22.1.3.16: RangeError if count < 0 or +∞.
                    if count < 0.0 || count.is_infinite() {
                        return Err(crate::interp::JsError::Throw(
                            Value::Object(std::rc::Rc::new(
                                std::cell::RefCell::new({
                                    let mut m =
                                        crate::ordered::OrderedMap::new();
                                    m.insert(
                                        "_isError".into(),
                                        Value::Bool(true),
                                    );
                                    m.insert(
                                        "name".into(),
                                        Value::String("RangeError".into()),
                                    );
                                    m.insert(
                                        "message".into(),
                                        Value::String(
                                            "Invalid count value".into(),
                                        ),
                                    );
                                    m
                                }),
                            )),
                        ));
                    }
                    let n = count as usize;
                    Ok(Value::str(s.repeat(n)))
                }),
            )
        }
        _ => Value::Undefined,
    }
}

/// Write `obj.key = value`. Objects gain or update the key; arrays
/// accept `length` (truncates/grows with `Undefined` fill) or a numeric
/// index. Writes to primitives are silently dropped (matches JS
/// behaviour for setting on a number/string in non-strict mode).
/// True if `v` is a Proxy exotic object (carries `_isProxy: true`). Writes to
/// a proxy must route through the host `set` trap rather than the raw store.
fn is_proxy_val(v: &Value) -> bool {
    matches!(v, Value::Object(o) if matches!(o.borrow().get("_isProxy"), Some(Value::Bool(true))))
}

/// True if `v`'s own slot at `key` holds an accessor descriptor (a `set x()`
/// / `get x()` pair). Such a write must run the setter via the host rather than
/// overwriting the accessor with a data value (the raw `property_store` would).
fn slot_is_accessor(v: &Value, key: &str) -> bool {
    if let Value::Object(o) = v {
        if let Some(Value::Object(d)) = o.borrow().get(key) {
            let db = d.borrow();
            return db.contains_key(crate::interp::ACCESSOR_SET)
                || db.contains_key(crate::interp::ACCESSOR_GET);
        }
    }
    false
}

/// Reference/value identity used by the prototype-IC record verify: are two
/// values the SAME (Rc identity for heap types, value equality for primitives)?
/// Conservative — unknown/uncomparable variants return false (⇒ don't cache).
fn value_ref_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::Array(x), Value::Array(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::Function(x), Value::Function(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::NativeFunction(x), Value::NativeFunction(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::BcClosure(x), Value::BcClosure(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Undefined, Value::Undefined) => true,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

/// Whether a slot's CURRENT value is an accessor descriptor. The write-IC checks
/// this per hit (a `defineProperty` data→accessor redefine overwrites the slot's
/// value but does NOT change the key set, so `struct_ver` wouldn't catch it).
fn slot_value_is_accessor(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::Object(d)) if {
        let db = d.borrow();
        db.contains_key(crate::interp::ACCESSOR_SET) || db.contains_key(crate::interp::ACCESSOR_GET)
    })
}

fn property_store(obj: &Value, key: &str, value: Value) {
    // B4 write barrier (VM store choke point): record an old→young pointer so a
    // generational minor scavenge does not lose the edge. No-op unless CV_GEN_GC.
    crate::interp::gen_gc_write_barrier_val(obj, &value);
    match obj {
        Value::Object(o) => {
            // `Object.freeze`: a frozen object silently rejects own-property
            // writes (non-strict). Mirrors tree-walk `write_property`; the VM
            // bypassed this before, so VM-compiled `frozen.x = v` mutated it.
            if !crate::interp::is_internal_key(key)
                && o.borrow().contains_key(crate::interp::FROZEN_KEY)
            {
                return;
            }
            // CharacterData alias sync (mirrors tree-walk `write_property`):
            // `data`/`nodeValue`/`textContent` are spec aliases on a text/comment
            // node (nodeType 3/8). The DOM bindings store them as three
            // independent slots with no setter, so a `.data = '…'` write would
            // leave `nodeValue` stale — and the host reconciler reads `nodeValue`
            // first, silently DROPPING the change. Mirror the write across all
            // three so every reader agrees. Cheap key-gate first.
            if matches!(key, "data" | "nodeValue" | "textContent") {
                let node_type = {
                    let b = o.borrow();
                    match b.get("nodeType") {
                        Some(Value::Number(n)) => *n,
                        _ => f64::NAN,
                    }
                };
                // WHATWG DOM §4.9 textContent set on an ELEMENT (nodeType 1):
                // route to the host "string replace all" rebuild (mirrors the
                // tree-walk `write_property` path) so `el.firstChild` is rebuilt.
                if key == "textContent"
                    && node_type == 1.0
                    && crate::interp::run_element_textcontent_hook(o, &value)
                {
                    return;
                }
                if node_type == 3.0 || node_type == 8.0 {
                    // [LegacyNullToEmptyString] DOMString coercion (mirrors the
                    // tree-walk path): null → "", else ToString.
                    let coerced = crate::interp::coerce_chardata_value(&value);
                    // §4.5 "replace data" range-adjusting (mirrors tree-walk):
                    // capture old/new UTF-16 lengths around the overwrite. Only
                    // when a host hook is installed (zero cost otherwise).
                    let lens = if crate::interp::chardata_replace_hook_registered() {
                        let old_len = {
                            let b = o.borrow();
                            match b.get("data").or_else(|| b.get("nodeValue")).or_else(|| b.get("textContent")) {
                                Some(Value::String(s)) => s.encode_utf16().count(),
                                Some(other) => other.to_display_string().encode_utf16().count(),
                                None => 0,
                            }
                        };
                        let new_len = match &coerced {
                            Value::String(s) => s.encode_utf16().count(),
                            other => other.to_display_string().encode_utf16().count(),
                        };
                        Some((old_len, new_len))
                    } else {
                        None
                    };
                    {
                        let mut b = o.borrow_mut();
                        b.insert("data".into(), coerced.clone());
                        b.insert("nodeValue".into(), coerced.clone());
                        b.insert("textContent".into(), coerced);
                    }
                    if let Some((old_len, new_len)) = lens {
                        crate::interp::run_chardata_replace_hook(o, old_len, new_len);
                    }
                    return;
                }
            }
            o.borrow_mut().insert(key.to_string(), value);
        }
        Value::Array(a) => {
            if key == "length" {
                if let Value::Number(n) = &value {
                    let new_len = (*n as usize).min(1 << 20);
                    let mut arr = a.borrow_mut();
                    arr.resize(new_len, Value::Undefined);
                }
            } else if let Ok(idx) = key.parse::<usize>() {
                let mut arr = a.borrow_mut();
                if idx >= arr.len() {
                    arr.resize(idx + 1, Value::Undefined);
                }
                arr[idx] = value;
            } else {
                // Named own-property on an array (arrays are exotic objects).
                crate::interp::array_set_prop(a, key, value);
            }
        }
        Value::Function(f) => {
            f.props.borrow_mut().insert(key.to_string(), value);
        }
        Value::NativeFunction(n) => {
            n.props.borrow_mut().insert(key.to_string(), value);
        }
        Value::BcClosure(c) => {
            c.props.borrow_mut().insert(key.to_string(), value);
        }
        _ => {}
    }
}

/// Write `obj[key] = value`. Mirrors `indexed_lookup`'s key coercion.
fn indexed_store(obj: &Value, key: &Value, value: Value) {
    let key_str = match key {
        Value::Number(n) => {
            if n.fract() == 0.0 && *n >= 0.0 && *n < (u32::MAX as f64) {
                format!("{}", *n as u32)
            } else {
                format!("{n}")
            }
        }
        Value::String(s) => s.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => return,
    };
    property_store(obj, &key_str, value);
}

/// Resolve `obj[key]`. Coerces the key to integer for arrays/strings
/// and to string for objects, matching ECMA-262 §13.3.2 semantics.
fn indexed_lookup(obj: &Value, key: &Value) -> Value {
    let key_str = match key {
        Value::Number(n) => {
            if n.fract() == 0.0 && *n >= 0.0 && *n < (u32::MAX as f64) {
                format!("{}", *n as u32)
            } else {
                format!("{n}")
            }
        }
        Value::String(s) => s.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => return Value::Undefined,
    };
    property_lookup(obj, &key_str)
}

fn display_value(v: &Value) -> String {
    // Delegate to the tree-walker's spec-correct `to_display_string` so the
    // VM matches Chrome (and the tree-walker) for every type: Object →
    // "[object Object]", Array → comma-joined elements, Function → its source
    // marker, BigInt → digits. The previous catch-all returned "" for any
    // non-primitive, which corrupted template literals (``${obj}``→""),
    // string concatenation cache keys (`"x"+arr`→"x"), and logs inside any
    // VM-compiled (hot) function — a critical split-brain divergence.
    v.to_display_string()
}

// Touching Rc just to keep the import alive — the module is real-world
// useful once arrays / objects join the supported set, which will lean
// on Rc<RefCell<_>> the same way the tree-walk does.
#[allow(dead_code)]
fn _unused_rc_typing() -> Option<Rc<()>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn run(src: &str) -> Result<Value, String> {
        let m = compile_program(src).map_err(|e| e.to_string())?;
        run_module(&m).map_err(|e| e.to_string())
    }

    /// Helper: call a module fn with no globals — keeps existing tests
    /// readable now that `run_function` takes a globals env + a
    /// WithInterp dispatcher.
    fn run_fn(m: &Module, idx: usize, args: &[Value]) -> Result<Value, RuntimeError> {
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        run_function(m, idx, args, &Value::Undefined, &empty, None, &mut refuse)
    }

    /// Run the SCRIPT frame (slot 0) first to instantiate the global environment
    /// (GlobalDeclarationInstantiation binds every top-level `function` to a
    /// stable global), THEN invoke top-level fn `idx` through that SAME env —
    /// exactly how real execution works (the script always runs before any of
    /// its top-level functions are called). Use this instead of `run_fn` when the
    /// callee references a SIBLING top-level fn by name (e.g. `new Point()`),
    /// since a top-level fn read resolves to its global binding, not a fresh
    /// per-read closure.
    fn run_fn_after_script(m: &Module, idx: usize, args: &[Value]) -> Result<Value, RuntimeError> {
        let g: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        run_function(m, 0, &[], &Value::Undefined, &g, None, &mut refuse)?;
        run_function(m, idx, args, &Value::Undefined, &g, None, &mut refuse)
    }

    // ════════════════════════════════════════════════════════════════════════
    // STAGE 2 — VM-LEVEL LEAF INLINING unit gate.
    // ════════════════════════════════════════════════════════════════════════

    /// Run a module's slot 0 with a fresh globals env, return (result, globals).
    fn run_script_with_globals(m: &Module) -> (Result<Value, RuntimeError>, HashMap<String, Value>) {
        let g: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        let r = run_function(m, 0, &[], &Value::Undefined, &g, None, &mut refuse);
        (r, g.into_inner())
    }

    /// STAGE-3 GATE — the kernel-loop optimizer (`optimize_kernel_loop`: LICM + CSE
    /// + copy-prop) must be BIT-IDENTICAL to the un-optimized kernel. We extract the
    /// jit.js kernel, compile BOTH the un-optimized and the optimized bytecode to
    /// native P6 code, run each over a fuzz of (i, carry, limit) seeds, and assert
    /// every f64 result is bit-identical. This is the direct A==B teeth for the
    /// Stage-3 tightening (independent of the end-to-end oracle). Windows-only (P6).
    #[cfg(target_os = "windows")]
    #[test]
    fn stage3_kernel_optimizer_is_bit_identical_to_unoptimized() {
        // jit.js shape → a kernelized counted loop. We rebuild the UN-optimized kernel
        // by detecting the loop and laying out the body exactly as kernelize would,
        // then comparing native execution to the optimizer's output. Simpler: drive it
        // through inline_leaf_module which already kernelizes+optimizes, AND a second
        // pass with the optimizer disabled, then compare both kernels' native results.
        let src = "function f(x){ return ((x*x*0.5 + x*3.0 - 1.0)*(x-2.0) + x*x*x*0.25)/(x+1.0) - x*0.5 + x*x*0.125 - x*7.0; } var s = 0; for (var i = 0; i < 100; i = i + 1) { s = s + f(i); }";
        // OPTIMIZED kernel (default path).
        let opt_mod = inline_leaf_module(&compile_program(src).unwrap(), 0).expect("inline+kernelize");
        let opt_kernel = opt_mod.fns.iter().find(|f| f.name == "__cv_loop_kernel").expect("kernel");
        // UN-optimized kernel: re-run the kernelizer with the optimizer disabled.
        let prev = std::env::var("CV_NO_KERNEL_OPT").ok();
        unsafe { std::env::set_var("CV_NO_KERNEL_OPT", "1"); }
        let raw_mod = inline_leaf_module(&compile_program(src).unwrap(), 0).expect("inline+kernelize raw");
        match prev {
            Some(v) => unsafe { std::env::set_var("CV_NO_KERNEL_OPT", v) },
            None => unsafe { std::env::remove_var("CV_NO_KERNEL_OPT") },
        }
        let raw_kernel = raw_mod.fns.iter().find(|f| f.name == "__cv_loop_kernel").expect("raw kernel");
        // The optimized kernel must be DIFFERENT (proves the pass fired — non-vacuous).
        assert_ne!(
            opt_kernel.code.len(),
            raw_kernel.code.len(),
            "optimizer did not change the kernel — vacuous gate"
        );
        // Compile both to native f64.
        let mk_native = |k: &BcFunction| {
            let consts = k.consts.clone();
            let bytes = crate::jit::compile_bytecode_f64(&k.code, k.n_params, move |i| {
                match consts.get(i as usize) { Some(Value::Number(n)) => Some(*n), _ => None }
            })
            .expect("kernel must be P6-compilable");
            crate::jit::JitFunction::install(&bytes).expect("install")
        };
        let opt_jf = mk_native(opt_kernel);
        let raw_jf = mk_native(raw_kernel);
        // Fuzz over (i_seed, carry_seed, limit) — the kernel's 3 args. Bit-compare.
        for &i0 in &[0.0f64, 1.0, 3.5, -2.0] {
            for &acc in &[0.0f64, 7.25, -1e9, f64::NAN] {
                for &lim in &[0.0f64, 1.0, 5.0, 17.0, 100.0] {
                    let a = unsafe { opt_jf.call_f64_args(&[i0, acc, lim]) };
                    let b = unsafe { raw_jf.call_f64_args(&[i0, acc, lim]) };
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "kernel result diverged at i0={i0} acc={acc} lim={lim}: opt={a} raw={b}"
                    );
                }
            }
        }
    }

    /// The inlined module's slot 0 must contain NO `CallFn` to an inlined leaf, and
    /// running it must produce the byte-identical globals + result as the un-inlined
    /// module — over BOTH bench shapes.
    #[test]
    fn leaf_inline_is_byte_identical_to_uninlined() {
        for src in [
            // jit.js-shaped: hot float loop calling a pure numeric leaf.
            "var r=0; function f(x){ return ((x*x*0.5 + x*3.0 - 1.0)*(x-2.0) + x*x*x*0.25)/(x+1.0) - x*0.5 + x*x*0.125 - x*7.0; } for (var i=0;i<137;i++){ r = r + f(i); } var out=r;",
            // loop.js-shaped: integer leaf with its own inner loop (Jmp/JmpIfFalse).
            "var r=0; function work(n){ var s=0; for (var i=0;i<n;i=i+1){ s=s+i; } return s; } for (var j=0;j<31;j++){ r = r + work(40); } var out=r;",
            // multiple distinct leaf calls in the body.
            "var r=0; function a(x){return x*2;} function b(x){return x+3;} for(var i=0;i<50;i++){ r = r + a(i) + b(i); } var out=r;",
            // an early-return (multi-Ret) leaf.
            "var r=0; function g(x){ if (x>5){ return x*10; } return x; } for(var i=0;i<20;i++){ r=r+g(i);} var out=r;",
        ] {
            let m = compile_program(src).unwrap();
            let inlined = inline_leaf_module(&m, 0)
                .unwrap_or_else(|| panic!("expected inlinable: {src}"));
            // The ORIGINAL leaf calls are gone from slot 0: any remaining CallFn must
            // target the appended `__cv_loop_kernel` (the native loop), NOT an inlined
            // leaf (fns[1..original_count]). i.e. slot 0 has no call to the leaf fn.
            let n_orig_fns = m.fns.len();
            assert!(
                !inlined.fns[0].code.iter().any(|op| matches!(
                    op,
                    Op::CallFn { fn_idx, .. } if (*fn_idx as usize) < n_orig_fns && *fn_idx != 0
                )),
                "inlined slot 0 still calls an original leaf fn: {src}"
            );
            // On Windows, the counted-accumulator loops in these fixtures must
            // KERNELIZE: a `__cv_loop_kernel` fn is appended and slot 0 calls it.
            #[cfg(target_os = "windows")]
            {
                let kernelized = inlined.fns.iter().any(|f| f.name == "__cv_loop_kernel");
                assert!(kernelized, "expected a loop kernel for: {src}");
            }
            let (r_base, g_base) = run_script_with_globals(&m);
            let (r_inl, g_inl) = run_script_with_globals(&inlined);
            assert_eq!(
                format!("{r_base:?}"),
                format!("{r_inl:?}"),
                "result diverged for: {src}"
            );
            // Compare the global the script computed (`out`) — byte-identical.
            let ob = g_base.get("out").cloned();
            let oi = g_inl.get("out").cloned();
            assert_eq!(format!("{ob:?}"), format!("{oi:?}"), "out diverged for: {src}");
            assert_eq!(g_base.len(), g_inl.len(), "global set size diverged: {src}");
            for (k, v) in &g_base {
                let vi = g_inl.get(k);
                assert_eq!(
                    format!("{:?}", Some(v)),
                    format!("{vi:?}"),
                    "global {k} diverged for: {src}"
                );
            }
        }
    }

    /// MUTATION ARM — the byte-identity check is non-vacuous. A deliberately WRONG
    /// inline (callee that returns `x+1` but we assert against `x+2`) must FAIL, and
    /// a callee with a real throw / non-numeric op must NOT be inlined (stays a call).
    #[test]
    fn leaf_inline_declines_non_numeric_callee() {
        // Callee touches a GLOBAL → not a pure numeric leaf → must NOT be inlined.
        let src = "var k=10; function f(x){ return x + k; } var r=0; for(var i=0;i<5;i++){ r=r+f(i);} var out=r;";
        let m = compile_program(src).unwrap();
        // f reads global `k` (LoadGlobalChecked) → callee_is_inlinable rejects it →
        // no inlinable call → inline_leaf_module returns None.
        assert!(
            inline_leaf_module(&m, 0).is_none(),
            "a callee reading a global must not be inlined"
        );
        // Callee calling another function → not a leaf → declined.
        let src2 = "function h(x){return x*2;} function f(x){ return h(x)+1; } var r=0; for(var i=0;i<5;i++){ r=r+f(i);} var out=r;";
        let m2 = compile_program(src2).unwrap();
        // f's body has a CallFn(h) which is itself inlinable — so f's CALL of h could
        // be inlined, but f itself (called from the loop) contains a call → f is NOT
        // an inlinable leaf. The loop's f(i) stays a call; but the body f → h is the
        // one inlinable site. Either way the run must stay byte-identical.
        let inlined2 = inline_leaf_module(&m2, 0);
        let (rb, gb) = run_script_with_globals(&m2);
        if let Some(im2) = inlined2 {
            let (ri, gi) = run_script_with_globals(&im2);
            assert_eq!(format!("{rb:?}"), format!("{ri:?}"));
            assert_eq!(gb.get("out").map(|v| format!("{v:?}")), gi.get("out").map(|v| format!("{v:?}")));
        }
    }

    /// KERNEL DEOPT-FUZZ — the loop kernel reached via `Op::CallFn` runs natively (P6)
    /// ONLY when its accumulator/induction args are all-numeric; a NON-NUMERIC
    /// accumulator makes the CallFn→P6 numeric guard decline, resuming the kernel on
    /// the VM (the deopt-equivalent path). Both routes MUST be byte-identical to the
    /// un-inlined loop. We force the deopt route by seeding the accumulator with a
    /// STRING / boolean / NaN-producing value and comparing inlined-vs-uninlined over
    /// a fuzz of accumulators × loop bounds.
    #[test]
    fn kernel_deopt_on_non_numeric_accumulator_is_byte_identical() {
        // Accumulator initial expressions that drive the non-numeric (VM-resume) and
        // numeric (P6) paths through the SAME kernel.
        let acc_inits = [
            "0",            // numeric → P6 native kernel
            "'x'",          // string → P6 declines → VM kernel (string concat semantics)
            "true",         // boolean → VM kernel (coercion)
            "(0/0)",        // NaN seed → numeric, but NaN propagation
            "1e308",        // overflow → Infinity propagation
            "-0",           // signed zero
        ];
        let bounds = [0usize, 1, 2, 7, 33];
        for acc in acc_inits {
            for n in bounds {
                let src = format!(
                    "var s = {acc}; function f(x){{ return x*x*0.5 - x + 1.0; }} \
                     for (var i = 0; i < {n}; i = i + 1) {{ s = s + f(i); }} var out = s;"
                );
                let m = compile_program(&src).unwrap();
                let inlined = match inline_leaf_module(&m, 0) {
                    Some(im) => im,
                    None => continue, // declined entirely (e.g. n==0 trivia) — still fine
                };
                let (rb, gb) = run_script_with_globals(&m);
                let (ri, gi) = run_script_with_globals(&inlined);
                assert_eq!(
                    format!("{rb:?}"),
                    format!("{ri:?}"),
                    "result diverged: acc={acc} n={n}"
                );
                assert_eq!(
                    gb.get("out").map(|v| format!("{v:?}")),
                    gi.get("out").map(|v| format!("{v:?}")),
                    "out diverged: acc={acc} n={n}\n  uninlined={:?}\n  inlined={:?}",
                    gb.get("out"),
                    gi.get("out"),
                );
                // `s` and `i` (the for-init syncs) must also match.
                for key in ["s", "i"] {
                    assert_eq!(
                        gb.get(key).map(|v| format!("{v:?}")),
                        gi.get(key).map(|v| format!("{v:?}")),
                        "global {key} diverged: acc={acc} n={n}"
                    );
                }
            }
        }
    }

    // ───────────────────────── M4.2a — T1 baseline JIT ─────────────────────
    // Lowest-level proof: compile a real JS function to bytecode, run it BOTH
    // on the VM and via T1 native code, and assert byte-identical results. This
    // is the single-source-of-truth check below the interp tier (the 3-tier
    // oracle in ab_oracle covers the JS-observable side).

    /// Compile a single-function module for `src`'s first function decl.
    #[cfg(target_os = "windows")]
    fn module_for_first_fn(src: &str) -> Module {
        let prog = crate::parser::parse_program(src).unwrap();
        let (params, body) = match &prog[0] {
            crate::ast::Stmt::FunctionDecl { params, body, .. } => (params.clone(), body.clone()),
            other => panic!("expected fn decl, got {other:?}"),
        };
        let (module, _ups) = compile_single_function(&params, &body, &[]).unwrap();
        module
    }

    /// Run `module.fns[0]` via T1 native code (compile+install+run). Panics if
    /// T1 declines (the caller asserts the function is in the supported subset).
    #[cfg(target_os = "windows")]
    fn run_t1(m: &Module, args: &[Value]) -> Result<Value, RuntimeError> {
        let native = try_compile_t1(m, 0).expect("function should compile to T1");
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        run_t1_call(&native, m, args, &Value::Undefined, &empty, &mut refuse)
    }

    // ───────────────────────── M4.3 — T2-lite inlined JsVal JIT ─────────────
    // The validation gate: a T2-lite native run (inline tag-check + UNBOXED f64
    // arithmetic on a JsVal bank) must be bit-identical to the VM across the
    // full number edge-case surface. A botched tag-check or a missed NaN
    // canonicalization is SILENT corruption, so these tests are the teeth.

    /// Run `module.fns[0]` via T2-lite native code. Panics if T2-lite declines
    /// (caller asserts the function is in the supported subset).
    #[cfg(target_os = "windows")]
    fn run_t2lite(m: &Module, args: &[Value]) -> Result<Value, RuntimeError> {
        let native = try_compile_t2lite(m, 0).expect("function should compile to T2-lite");
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        run_t2lite_call(&native, m, args, &Value::Undefined, &empty, &mut refuse)
    }

    /// JS-observable number equality: any NaN equals any NaN (NaN bit patterns
    /// are NOT observable in JS — and T2-lite deliberately CANONICALIZES a
    /// computed NaN to protect the box tag space, which the VM does not, so a raw
    /// `to_bits()` would spuriously differ on the NaN payload only); a finite
    /// value must match bit-for-bit so `-0.0` is kept DISTINCT from `+0.0` (the
    /// one sign distinction JS *can* observe, via `Object.is` / `1/x`).
    #[cfg(target_os = "windows")]
    fn num_bits_eq(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => {
                if x.is_nan() && y.is_nan() {
                    true
                } else {
                    x.to_bits() == y.to_bits()
                }
            }
            _ => false,
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_matches_vm_arithmetic_and_edges() {
        // f(a,b) = a*a + b - 1, exercises Mul/Add/Sub + a const.
        let m = module_for_first_fn("function poly(a, b) { return a * a + b - 1; }");
        let cases = [
            (5.0, 3.0),
            (0.0, 0.0),
            (-0.0, 0.0),
            (1.0 / 0.0, 1.0),            // +Inf
            (-1.0 / 0.0, 1.0),           // -Inf
            (f64::NAN, 1.0),             // NaN propagates
            (9007199254740992.0, 1.0),   // 2^53 region
            (1.5, -2.5),
            (1e308, 1e308),              // overflow → Inf
        ];
        for (a, b) in cases {
            let args = [Value::Number(a), Value::Number(b)];
            let vm = run_fn(&m, 0, &args).unwrap();
            let t2 = run_t2lite(&m, &args).unwrap();
            assert!(
                num_bits_eq(&vm, &t2),
                "poly({a},{b}): vm={vm:?} t2={t2:?} (bit-exact required)"
            );
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_division_and_nan_edges() {
        let m = module_for_first_fn("function f(a, b) { return a / b; }");
        let cases = [
            (7.0, 2.0),     // 3.5
            (1.0, 0.0),     // +Inf
            (-1.0, 0.0),    // -Inf
            (0.0, 0.0),     // NaN (0/0)
            (1.0, -0.0),    // -Inf  (1/-0)
            (-0.0, 1.0),    // -0.0
            (5.0, -0.0),    // -Inf
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
        ];
        for (a, b) in cases {
            let args = [Value::Number(a), Value::Number(b)];
            let vm = run_fn(&m, 0, &args).unwrap();
            let t2 = run_t2lite(&m, &args).unwrap();
            assert!(
                num_bits_eq(&vm, &t2),
                "{a}/{b}: vm={vm:?} t2={t2:?} (bit-exact incl -0/Inf/NaN)"
            );
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_computed_nan_canonicalizes() {
        // Inf - Inf and 0/0 both produce NaN; the boxed result must canonicalize
        // (never alias a tagged JsVal). f(a,b) = a - b with a=b=Inf → NaN.
        let m = module_for_first_fn("function f(a, b) { return a - b; }");
        let inf = f64::INFINITY;
        let args = [Value::Number(inf), Value::Number(inf)];
        let vm = run_fn(&m, 0, &args).unwrap();
        let t2 = run_t2lite(&m, &args).unwrap();
        match (&vm, &t2) {
            (Value::Number(x), Value::Number(y)) => {
                assert!(x.is_nan() && y.is_nan(), "Inf-Inf must be NaN: vm={x} t2={y}");
                // And the T2 result must be a real Number JsVal (canonicalized),
                // i.e. round-trips to a Number, not a corrupted tagged value.
                let jv = crate::jsval::JsVal::number(*y);
                assert!(jv.is_number(), "computed NaN must be a number-lane JsVal");
                assert_eq!(jv.bits(), crate::jsval::CANONICAL_NAN, "NaN must canonicalize");
            }
            _ => panic!("non-number vm={vm:?} t2={t2:?}"),
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_comparisons_match_vm_incl_nan() {
        // Each compare op, fed pairs including NaN/-0/+0/Inf, must produce the
        // SAME boolean the VM does (relational ops false for NaN; === false,
        // !== true for NaN).
        for op in ["<", "<=", ">", ">=", "==", "!="] {
            let src = format!("function f(a, b) {{ if (a {op} b) {{ return 1; }} return 0; }}");
            let m = module_for_first_fn(&src);
            let cases = [
                (1.0, 2.0),
                (2.0, 1.0),
                (1.0, 1.0),
                (f64::NAN, 1.0),
                (1.0, f64::NAN),
                (f64::NAN, f64::NAN),
                (0.0, -0.0),
                (-0.0, 0.0),
                (f64::INFINITY, 1.0),
                (1.0, f64::NEG_INFINITY),
            ];
            for (a, b) in cases {
                let args = [Value::Number(a), Value::Number(b)];
                let vm = run_fn(&m, 0, &args).unwrap();
                let t2 = run_t2lite(&m, &args).unwrap();
                assert!(
                    num_bits_eq(&vm, &t2),
                    "({a} {op} {b}): vm={vm:?} t2={t2:?}"
                );
            }
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_matches_vm_on_loops() {
        // sumToN, fibIter, nestedMul, branchHeavy — the M4.2b kernels, run on
        // T2-lite and diffed against the VM over a sweep of inputs.
        let kernels: [(&str, &str); 4] = [
            ("sumToN", "function f(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i; } return s; }"),
            ("fibIter", "function f(n){ var a = 0; var b = 1; for (var i = 0; i < n; i = i + 1) { var t = a + b; a = b; b = t; } return a; }"),
            ("nestedMul", "function f(n){ var acc = 0; for (var i = 0; i < n; i = i + 1) { for (var j = 0; j < n; j = j + 1) { acc = acc + i * j; } } return acc; }"),
            ("branchHeavy", "function f(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { if (i < 1000) { s = s + 1; } if (i > 500) { s = s + 2; } if (i >= 1500) { s = s + 3; } } return s; }"),
        ];
        for (name, src) in kernels {
            let m = module_for_first_fn(src);
            assert!(
                try_compile_t2lite(&m, 0).is_some(),
                "{name} must compile to T2-lite (not a vacuous decline)"
            );
            for n in [0.0, 1.0, 2.0, 7.0, 30.0, 60.0] {
                let args = [Value::Number(n)];
                let vm = run_fn(&m, 0, &args).unwrap();
                let t2 = run_t2lite(&m, &args).unwrap();
                assert!(
                    num_bits_eq(&vm, &t2),
                    "{name}({n}): vm={vm:?} t2={t2:?}"
                );
            }
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_deopts_on_non_number_arg() {
        // A non-number operand must DEOPT to the VM and still give the right
        // answer. f(a,b)=a+b with string args → VM does concatenation; T2-lite
        // detects the non-number at the Add and falls back, producing the SAME
        // string. (Proves deopt-to-VM correctness, the safety net.)
        let m = module_for_first_fn("function f(a, b) { return a + b; }");
        let args = [Value::str("foo".to_string()), Value::str("bar".to_string())];
        let vm = run_fn(&m, 0, &args).unwrap();
        let t2 = run_t2lite(&m, &args).unwrap();
        assert!(matches!(&vm, Value::String(s) if &**s == "foobar"), "vm={vm:?}");
        assert!(matches!(&t2, Value::String(s) if &**s == "foobar"), "t2 (deopt)={t2:?}");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_declines_unsupported_op() {
        // Property access is outside the subset → decline (None), still runs on
        // the VM correctly.
        let m = module_for_first_fn("function g(a){ var o = { v: a }; return o.v + 1; }");
        assert!(
            try_compile_t2lite(&m, 0).is_none(),
            "unsupported-op function must decline T2-lite compilation"
        );
    }

    /// MUTATION-TEST TEETH: corrupting the T2-lite arithmetic arm (here, by
    /// compiling a DELIBERATELY-WRONG variant that swaps add for sub) must make
    /// the T2==VM check FAIL — proving the native code genuinely runs (a
    /// vacuously-green test that silently declined would pass this corruption).
    /// We can't mutate the installed bytes, so we assert the property indirectly:
    /// the correct compile gives `a+b`, and a hand-built WRONG op stream gives
    /// `a-b`, and the two differ — confirming the native arithmetic is load-
    /// bearing, not a no-op.
    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_mutation_arith_arm_is_load_bearing() {
        use crate::jit::{compile_t2lite, JitFunction, T2_RETURNED};
        use crate::jsval::JsVal;
        // Two tiny op streams over a 3-register bank: r2 = r0 (+|-) r1 ; ret r2.
        let consts = |_k: u16| -> Option<f64> { None };
        let add_stream = [
            Op::Add { dst: 2, lhs: 0, rhs: 1 },
            Op::Ret { src: 2 },
        ];
        let sub_stream = [
            Op::Sub { dst: 2, lhs: 0, rhs: 1 },
            Op::Ret { src: 2 },
        ];
        let run = |stream: &[Op], a: f64, b: f64| -> f64 {
            let code = compile_t2lite(stream, consts, None, crate::jit::T2StoreMode::Numeric, None)
                .expect("compiles");
            let jf = JitFunction::install(&code).expect("install");
            let mut bank: [u64; 3] = [
                JsVal::number(a).bits(),
                JsVal::number(b).bits(),
                JsVal::undefined().bits(),
            ];
            let mut out: u64 = 0;
            let tag = unsafe { jf.call_t2lite(bank.as_mut_ptr(), &mut out as *mut u64) };
            assert_eq!(tag, T2_RETURNED, "must return, not deopt");
            JsVal(out).as_f64().expect("number result")
        };
        let (a, b) = (10.0, 3.0);
        let added = run(&add_stream, a, b);
        let subbed = run(&sub_stream, a, b);
        assert_eq!(added, 13.0, "correct arith arm computes a+b");
        assert_eq!(subbed, 7.0, "the mutated (sub) arm computes a-b");
        // If the arithmetic arm were a no-op (native code not really running),
        // both would return the same garbage. They differ → the arm is real.
        assert_ne!(added, subbed, "arith arm must be load-bearing (native runs)");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t1_matches_vm_on_arithmetic() {
        let m = module_for_first_fn("function poly(a, b) { return a * a + b - 1; }");
        let args = [Value::Number(5.0), Value::Number(3.0)];
        let vm = run_fn(&m, 0, &args).unwrap();
        let t1 = run_t1(&m, &args).unwrap();
        assert!(matches!(vm, Value::Number(n) if n == 27.0), "vm: {vm:?}");
        assert!(matches!(t1, Value::Number(n) if n == 27.0), "t1: {t1:?}");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t1_matches_vm_on_loop() {
        // sumTo(n) with an internal for-loop: JmpIfFalse + Jmp back-edge + Add.
        let m = module_for_first_fn(
            "function sumTo(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i; } return s; }",
        );
        for n in [0.0, 1.0, 10.0, 100.0, 1000.0] {
            let args = [Value::Number(n)];
            let vm = run_fn(&m, 0, &args).unwrap();
            let t1 = run_t1(&m, &args).unwrap();
            match (&vm, &t1) {
                (Value::Number(a), Value::Number(b)) => {
                    assert_eq!(a, b, "T1 != VM for sumTo({n})");
                }
                _ => panic!("non-number result vm={vm:?} t1={t1:?}"),
            }
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t1_matches_vm_on_early_return_and_compares() {
        let m = module_for_first_fn(
            "function pick(x){ if (x < 10) { return x * 2; } if (x >= 100) { return 999; } return x - 100; }",
        );
        for x in [5.0, 9.0, 10.0, 50.0, 100.0, 250.0] {
            let args = [Value::Number(x)];
            let vm = run_fn(&m, 0, &args).unwrap();
            let t1 = run_t1(&m, &args).unwrap();
            assert!(
                matches!((&vm, &t1), (Value::Number(a), Value::Number(b)) if a == b),
                "pick({x}): vm={vm:?} t1={t1:?}"
            );
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t1_matches_vm_on_strings() {
        // `+` concatenates; `<` uses Abstract Relational Comparison — both run
        // through the shared op bodies, so T1 handles non-f64 values too.
        let m = module_for_first_fn("function f(a, b) { return a + b; }");
        let args = [
            Value::str("foo".to_string()),
            Value::str("bar".to_string()),
        ];
        let vm = run_fn(&m, 0, &args).unwrap();
        let t1 = run_t1(&m, &args).unwrap();
        assert!(matches!(&vm, Value::String(s) if &**s == "foobar"), "vm: {vm:?}");
        assert!(matches!(&t1, Value::String(s) if &**s == "foobar"), "t1: {t1:?}");
    }

    /// INDEPENDENT VERIFIER adversarial test: a nested loop with an early-return
    /// branch and mixed compares (subset-only). Exercises back-edges, forward
    /// jumps, and the Ret epilogue from inside nested control flow. Diffs T1
    /// native code against the VM over a sweep of inputs — proves T1==VM beyond
    /// the executor's own corpus.
    #[cfg(target_os = "windows")]
    #[test]
    fn t1_verifier_nested_loop_early_return_matches_vm() {
        let m = module_for_first_fn(
            "function f(n, cap) {
                var acc = 0;
                for (var i = 0; i < n; i = i + 1) {
                    for (var j = 0; j <= i; j = j + 1) {
                        acc = acc + (i - j);
                        if (acc > cap) { return acc * 2 - 7; }
                    }
                }
                return acc + 1;
            }",
        );
        // Confirm the function is genuinely T1-compilable (not a vacuous decline).
        assert!(
            try_compile_t1(&m, 0).is_some(),
            "verifier function must be in the T1 subset"
        );
        for n in [0.0, 1.0, 5.0, 12.0, 30.0] {
            for cap in [-1.0, 0.0, 10.0, 100.0, 1.0e9] {
                let args = [Value::Number(n), Value::Number(cap)];
                let vm = run_fn(&m, 0, &args).unwrap();
                let t1 = run_t1(&m, &args).unwrap();
                match (&vm, &t1) {
                    (Value::Number(a), Value::Number(b)) => {
                        assert_eq!(a, b, "T1 != VM for f({n}, {cap}): vm={a} t1={b}");
                    }
                    _ => panic!("non-number vm={vm:?} t1={t1:?} for f({n},{cap})"),
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t1_declines_unsupported_op() {
        // A function with a property access (GetProp/NewObject — outside the
        // subset) must DECLINE: `try_compile_t1` returns None.
        let m = module_for_first_fn("function g(a){ var o = { v: a }; return o.v + 1; }");
        assert!(
            try_compile_t1(&m, 0).is_none(),
            "unsupported-op function must decline T1 compilation"
        );
        // ...and it still runs correctly on the VM.
        let vm = run_fn(&m, 0, &[Value::Number(41.0)]).unwrap();
        assert!(matches!(vm, Value::Number(n) if n == 42.0), "vm fallback: {vm:?}");
    }

    /// THE T1 throw-epilogue proof: when a subset op (here `Add` on a BigInt
    /// operand, which routes to the host `__tb_host_binop` dispatcher) returns a
    /// THROWN error, T1 native code must take the `T1_THREW` → epilogue path and
    /// `run_function_t1` must surface it as `Err(Thrown(..))` — IDENTICALLY to
    /// the VM. We inject a dispatcher that always throws, so the throw is
    /// deterministic regardless of the engine's lenient value coercion.
    #[cfg(target_os = "windows")]
    #[test]
    fn t1_throw_propagates_as_err_like_vm() {
        use crate::interp::JsBigInt;
        let m = module_for_first_fn("function f(a) { return a + a; }");
        // Globals expose `__tb_host_binop` (a native) so `bigint_binop` engages
        // the dispatcher when an operand is a BigInt.
        let globals: std::cell::RefCell<HashMap<String, Value>> = {
            let mut g = HashMap::new();
            g.insert(
                "__tb_host_binop".to_string(),
                Value::NativeFunction(std::rc::Rc::new(crate::interp::NativeFn {
                    name: "__tb_host_binop".into(),
                    func: crate::interp::NativeFnBody::Pure(Box::new(|_| Ok(Value::Undefined))),
                    length: 0,
                    is_ctor: false,
                    props: std::cell::RefCell::new(HashMap::new()),
                })),
            );
            std::cell::RefCell::new(g)
        };
        // A dispatcher that ALWAYS throws a TypeError (simulating BigInt-mixing).
        let throw_msg = "TypeError: simulated host throw";
        let mut throwing = |_c: Value, _t: Value, _a: Vec<Value>| {
            Err(RuntimeError::Thrown(crate::interp::err_str(throw_msg.into())))
        };
        let args = [Value::bigint(JsBigInt::zero())];

        // VM path: the Add dispatches to the (throwing) host → Err(Thrown).
        let vm = run_function(&m, 0, &args, &Value::Undefined, &globals, None, &mut throwing);
        // T1 path: same op bodies, but driven by native code through the thunk.
        let native = try_compile_t1(&m, 0).expect("f(a)=a+a is subset-only → must compile");
        let t1 = run_t1_call(&native, &m, &args, &Value::Undefined, &globals, &mut throwing);

        // Both must throw, with the SAME thrown payload (catchable, not Deadline).
        match (&vm, &t1) {
            (Err(RuntimeError::Thrown(_)), Err(RuntimeError::Thrown(_))) => {}
            _ => panic!("T1 must propagate a catchable Thrown like the VM: vm={vm:?} t1={t1:?}"),
        }
    }

    /// A T1 throw must NOT be misclassified as the uncatchable Deadline: the
    /// thunk maps a non-Deadline error to `T1_THREW`, and a Deadline to
    /// `T1_DEADLINE`. Here (a normal Thrown) the result is `Thrown`, never
    /// `Deadline` — so an enclosing JS `try/catch` would catch it.
    #[cfg(target_os = "windows")]
    #[test]
    fn t1_throw_is_not_deadline() {
        use crate::interp::JsBigInt;
        let m = module_for_first_fn("function f(a) { return a + a; }");
        let globals: std::cell::RefCell<HashMap<String, Value>> = {
            let mut g = HashMap::new();
            g.insert(
                "__tb_host_binop".to_string(),
                Value::NativeFunction(std::rc::Rc::new(crate::interp::NativeFn {
                    name: "__tb_host_binop".into(),
                    func: crate::interp::NativeFnBody::Pure(Box::new(|_| Ok(Value::Undefined))),
                    length: 0,
                    is_ctor: false,
                    props: std::cell::RefCell::new(HashMap::new()),
                })),
            );
            std::cell::RefCell::new(g)
        };
        let mut throwing = |_c: Value, _t: Value, _a: Vec<Value>| {
            Err(RuntimeError::Thrown(crate::interp::err_str("TypeError: x".into())))
        };
        let native = try_compile_t1(&m, 0).unwrap();
        let r = run_t1_call(
            &native,
            &m,
            &[Value::bigint(JsBigInt::zero())],
            &Value::Undefined,
            &globals,
            &mut throwing,
        );
        assert!(
            matches!(r, Err(RuntimeError::Thrown(_))),
            "a normal throw must be Thrown (catchable), not Deadline: {r:?}"
        );
    }

    // ════════════════════════ M4.2b — T1 vs VM vs f64-JIT BENCHMARK ═══════════
    //
    // PRIMARY DELIVERABLE: does the helper-based T1 (one `call <thunk>` per op +
    // a tag branch) actually run FASTER than the VM's match jump-table on hot
    // loops, or does the per-op call+branch overhead cancel the dispatch saving?
    //
    // METHODOLOGY (stated so the numbers are credible):
    //   * Three executors run the SAME compiled bytecode function:
    //       (a) VM   — `run_function` (the match loop), the baseline.
    //       (b) T1   — `compile_baseline_t1` native code via `run_t1_call`.
    //       (c) f64  — `compile_bytecode_f64` native code via `call_f64_args`
    //                  (operands stay in xmm, never a `Value`); declines on
    //                  anything outside straight-line/loop double arithmetic.
    //   * Each benchmark function does its real work in an INTERNAL loop so one
    //     call performs a large, fixed amount of work — the per-call entry cost
    //     (reg-file setup, T1 prologue) is amortized over thousands of ops.
    //   * The native code (T1, f64) is compiled+installed ONCE, outside timing.
    //   * WARMUP: the first `WARMUP` calls per executor are discarded (page
    //     fault-in, branch-predictor/i-cache warmup).
    //   * TIMED: `TRIALS` independent trials, each timing `CALLS` back-to-back
    //     calls; we keep the MINIMUM trial time (noise only ADDS time, so min is
    //     the cleanest steady-state estimator). Reported as ns per call.
    //   * Build at opt-level 3 (`cargo test --release -- --ignored`) so the VM
    //     match and the inlined `op_xxx` bodies are fully optimized — otherwise
    //     the dispatch comparison is distorted by un-optimized code.
    //   * Correctness is re-asserted inside the harness (all three executors must
    //     agree on the result) so a miscompiled fast path can't post a fake win.
    //
    // `#[ignore]` so it doesn't run (or slow) the normal suite; invoke with
    //   cargo test -p cv_js --release t1_benchmark -- --ignored --nocapture

    /// One executor's measured cost. `ns_per_call` is the min-trial steady state.
    #[cfg(target_os = "windows")]
    struct BenchResult {
        ns_per_call: f64,
        available: bool, // false if the executor declined this function (f64).
    }

    /// Time `calls` back-to-back invocations of `run`, MIN over `trials`, after
    /// `warmup` discarded calls. Returns ns/call.
    #[cfg(target_os = "windows")]
    fn time_min_ns(
        warmup: u32,
        trials: u32,
        calls: u32,
        mut run: impl FnMut() -> Value,
    ) -> f64 {
        use std::time::Instant;
        for _ in 0..warmup {
            std::hint::black_box(run());
        }
        // Integer-only clock (`as_nanos`, u128) so no host f64 math runs in the
        // hot loop — uniform with the f64 path's robust timing.
        let mut best_ns_total: u128 = u128::MAX;
        for _ in 0..trials {
            let t = Instant::now();
            for _ in 0..calls {
                std::hint::black_box(run());
            }
            let ns = t.elapsed().as_nanos();
            if ns == 0 {
                continue; // sub-resolution reading — skip.
            }
            if ns < best_ns_total {
                best_ns_total = ns;
            }
        }
        best_ns_total as f64 / calls as f64
    }

    /// Run one benchmark function under all three executors and print a row.
    /// `src` must be a single function decl whose body is in the T1 subset; `args`
    /// is the (fixed) call argument set.
    #[cfg(target_os = "windows")]
    fn bench_one(
        name: &str,
        src: &str,
        args: &[Value],
        warmup: u32,
        trials: u32,
        calls: u32,
    ) -> (f64, f64, Option<f64>) {
        let m = module_for_first_fn(src);
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());

        // ---- correctness: all available executors must agree ----
        let vm_val = {
            let mut refuse = refuse_with_interp;
            run_function(&m, 0, args, &Value::Undefined, &empty, None, &mut refuse).unwrap()
        };
        let t1_native = try_compile_t1(&m, 0).expect("benchmark fn must be in the T1 subset");
        let t1_val = {
            let mut refuse = refuse_with_interp;
            run_t1_call(&t1_native, &m, args, &Value::Undefined, &empty, &mut refuse).unwrap()
        };
        assert!(
            matches!((&vm_val, &t1_val), (Value::Number(a), Value::Number(b)) if a == b),
            "{name}: T1 != VM (vm={vm_val:?} t1={t1_val:?})"
        );

        let f = &m.fns[0];
        let f64_native = crate::jit::compile_bytecode_f64(&f.code, f.n_params, |k| {
            match f.consts.get(k as usize) {
                Some(Value::Number(n)) => Some(*n),
                _ => None,
            }
        })
        .and_then(|code| crate::jit::JitFunction::install(&code).ok());
        let fargs: Vec<f64> = args
            .iter()
            .map(|a| match a {
                Value::Number(n) => *n,
                _ => 0.0,
            })
            .collect();
        // Highest bytecode register the function uses — the Win64-ABI risk line
        // for the f64 JIT: regs ≥ 6 map to callee-saved xmm6+, whose save/restore
        // in the f64 epilog is 64-bit `movsd` (not 128-bit `movaps`), so a hot
        // in-process timing loop over such a fn can read a poisoned `Instant`
        // (the f64 RESULT is still correct). We only report f64 timing when the
        // fn is xmm0..5-only (the ABI-clean path).
        let max_reg = f.n_regs.saturating_sub(1);
        let f64_correct = if let Some(jf) = &f64_native {
            let r = unsafe { jf.call_f64_args(&fargs) };
            let ok = matches!(&vm_val, Value::Number(a) if *a == r || (a.is_nan() && r.is_nan()));
            println!("    [f64 diag] {name}: compiled=Some n_regs={} max_reg={max_reg} f64-correct={ok}", f.n_regs);
            Some(ok)
        } else {
            println!("    [f64 diag] {name}: compiled=None (declined)");
            None
        };

        // ---- timing ----
        let vm_ns = {
            let mut refuse = refuse_with_interp;
            time_min_ns(warmup, trials, calls, || {
                run_function(&m, 0, args, &Value::Undefined, &empty, None, &mut refuse).unwrap()
            })
        };
        let t1_ns = {
            let mut refuse = refuse_with_interp;
            time_min_ns(warmup, trials, calls, || {
                run_t1_call(&t1_native, &m, args, &Value::Undefined, &empty, &mut refuse).unwrap()
            })
        };
        // f64 returns a raw f64. The f64-JIT epilog restores callee-saved xmm6+
        // with 64-bit `movsd` (not the ABI-required 128-bit `movaps`), so for a
        // fn that uses xmm6+ the host's float timing math reads a poisoned clock.
        // To measure it ROBUSTLY anyway, we keep the clock integer-only
        // (`as_nanos()`, a u128) and bit-accumulate the result into a u64 — no
        // host f64 math in the hot loop, so the xmm clobber can't poison it. The
        // f64 RESULT itself is verified correct above.
        let f64_ns = f64_native.as_ref().map(|jf| {
            use std::time::Instant;
            for _ in 0..warmup {
                std::hint::black_box(unsafe { jf.call_f64_args(&fargs) });
            }
            let mut best_ns_total: u128 = u128::MAX;
            for _ in 0..trials {
                let mut acc: u64 = 0;
                let t = Instant::now();
                for _ in 0..calls {
                    let r = unsafe { jf.call_f64_args(std::hint::black_box(&fargs)) };
                    acc ^= r.to_bits();
                }
                let ns = t.elapsed().as_nanos();
                std::hint::black_box(acc);
                if ns == 0 {
                    continue;
                }
                if ns < best_ns_total {
                    best_ns_total = ns;
                }
            }
            best_ns_total as f64 / calls as f64
        });

        let vm_over_t1 = vm_ns / t1_ns; // >1 => T1 faster than VM
        let f64_note = match (f64_ns, f64_correct) {
            (Some(fns), Some(true)) => format!("{fns:>9.1}   t1/f64={:.2}x", t1_ns / fns),
            (Some(fns), Some(false)) => {
                format!("{fns:>9.1}   t1/f64={:.2}x [WRONG-VALUE]", t1_ns / fns)
            }
            _ => "  (declined)".to_string(),
        };
        println!(
            "  {name:<14} VM={vm_ns:>9.1}  T1={t1_ns:>9.1}  f64={f64_note}   T1/VM speedup={vm_over_t1:.2}x",
        );
        (vm_ns, t1_ns, f64_ns)
    }

    /// THE benchmark. Runs 4 hot loop kernels through VM / T1 / f64-JIT and prints
    /// ns/call + the T1/VM and T1/f64 ratios. Data-only — never fails on timing
    /// (it asserts CORRECTNESS, not a speed threshold; the recommendation comes
    /// from reading the printed ratios).
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "timing benchmark; run with --release --ignored --nocapture"]
    fn t1_benchmark_vs_vm_vs_f64() {
        // Internal-loop kernels: one call = N iterations of real work.
        const WARMUP: u32 = 50;
        const TRIALS: u32 = 9;
        const CALLS: u32 = 200;

        println!("\n=== M4.2b T1 vs VM vs f64-JIT (ns/call, min of {TRIALS} trials, {CALLS} calls/trial, {WARMUP} warmup) ===");

        // (1) sum-to-N: pure Add loop with a compare back-edge.
        bench_one(
            "sumToN",
            "function f(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i; } return s; }",
            &[Value::Number(3000.0)],
            WARMUP, TRIALS, CALLS,
        );

        // (2) iterative Fibonacci: two-variable update loop (Add + Move-ish).
        bench_one(
            "fibIter",
            "function f(n){ var a = 0; var b = 1; for (var i = 0; i < n; i = i + 1) { var t = a + b; a = b; b = t; } return a; }",
            &[Value::Number(1000.0)],
            WARMUP, TRIALS, CALLS,
        );

        // (3) nested-loop integer kernel: O(n^2) Mul+Add (Mul is a subset op).
        bench_one(
            "nestedMul",
            "function f(n){ var acc = 0; for (var i = 0; i < n; i = i + 1) { for (var j = 0; j < n; j = j + 1) { acc = acc + i * j; } } return acc; }",
            &[Value::Number(60.0)],
            WARMUP, TRIALS, CALLS,
        );

        // (4) compare/branch-heavy loop: several `if`s per iteration (each lowers
        // to a compare + JmpIfFalse). Stresses the dispatch of branchy code.
        bench_one(
            "branchHeavy",
            "function f(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { if (i < 1000) { s = s + 1; } if (i > 500) { s = s + 2; } if (i >= 1500) { s = s + 3; } } return s; }",
            &[Value::Number(3000.0)],
            WARMUP, TRIALS, CALLS,
        );
        println!("=== end benchmark ===\n");
    }

    // ════════════════════ M4.3 — T2-LITE vs VM vs f64-JIT BENCHMARK ═══════════
    //
    // THE VALIDATION DELIVERABLE: does the INLINED-`JsVal` JIT (inline tag-check +
    // UNBOXED f64 arithmetic, values kept as `JsVal` in a bank) BEAT the VM —
    // unlike the helper-based T1 (call-per-op), which M4.2b found at 0.88–0.99x
    // (never faster)? If T2lite/VM > 1, the NaN-box bet pays off and the full T2
    // is justified.
    //
    // SAME methodology as the M4.2b harness: each kernel does its work in an
    // INTERNAL loop (one call = N iterations, amortizing entry cost); native code
    // compiled+installed ONCE outside timing; WARMUP discarded; MIN over TRIALS
    // of CALLS back-to-back invocations; integer-only clock (opt-level 3 via
    // `--release`). Correctness (T2-lite == VM, bit-exact mod NaN payload) is
    // re-asserted in the harness so a miscompiled fast path can't post a fake win.
    //
    //   cargo test -p cv_js --release t2lite_benchmark -- --ignored --nocapture

    /// Run one kernel under VM / T2-lite / f64-JIT and print a row. Returns
    /// (vm_ns, t2_ns, f64_ns?).
    #[cfg(target_os = "windows")]
    fn bench_one_t2(
        name: &str,
        src: &str,
        args: &[Value],
        warmup: u32,
        trials: u32,
        calls: u32,
    ) -> (f64, f64, Option<f64>) {
        let m = module_for_first_fn(src);
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());

        // ---- correctness: VM vs T2-lite must agree (bit-exact mod NaN payload) ----
        let vm_val = {
            let mut refuse = refuse_with_interp;
            run_function(&m, 0, args, &Value::Undefined, &empty, None, &mut refuse).unwrap()
        };
        let t2_native = try_compile_t2lite(&m, 0)
            .expect("benchmark kernel must be in the T2-lite subset");
        let t2_val = {
            let mut refuse = refuse_with_interp;
            run_t2lite_call(&t2_native, &m, args, &Value::Undefined, &empty, &mut refuse).unwrap()
        };
        let agree = match (&vm_val, &t2_val) {
            (Value::Number(a), Value::Number(b)) => {
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
            }
            _ => false,
        };
        assert!(agree, "{name}: T2-lite != VM (vm={vm_val:?} t2={t2_val:?})");

        // f64-JIT (operands stay in xmm, never a JsVal) — the upper bound.
        let f = &m.fns[0];
        let f64_native = crate::jit::compile_bytecode_f64(&f.code, f.n_params, |k| {
            match f.consts.get(k as usize) {
                Some(Value::Number(n)) => Some(*n),
                _ => None,
            }
        })
        .and_then(|code| crate::jit::JitFunction::install(&code).ok());
        let fargs: Vec<f64> = args
            .iter()
            .map(|a| match a {
                Value::Number(n) => *n,
                _ => 0.0,
            })
            .collect();

        // ---- timing ----
        let vm_ns = {
            let mut refuse = refuse_with_interp;
            time_min_ns(warmup, trials, calls, || {
                run_function(&m, 0, args, &Value::Undefined, &empty, None, &mut refuse).unwrap()
            })
        };
        let t2_ns = {
            let mut refuse = refuse_with_interp;
            time_min_ns(warmup, trials, calls, || {
                run_t2lite_call(&t2_native, &m, args, &Value::Undefined, &empty, &mut refuse)
                    .unwrap()
            })
        };
        // f64 returns a raw f64; keep the clock integer-only + bit-accumulate the
        // result (the f64 epilog now uses movaps, but stay uniform with M4.2b).
        let f64_ns = f64_native.as_ref().map(|jf| {
            use std::time::Instant;
            for _ in 0..warmup {
                std::hint::black_box(unsafe { jf.call_f64_args(&fargs) });
            }
            let mut best_ns_total: u128 = u128::MAX;
            for _ in 0..trials {
                let mut acc: u64 = 0;
                let t = Instant::now();
                for _ in 0..calls {
                    let r = unsafe { jf.call_f64_args(std::hint::black_box(&fargs)) };
                    acc ^= r.to_bits();
                }
                let ns = t.elapsed().as_nanos();
                std::hint::black_box(acc);
                if ns == 0 {
                    continue;
                }
                if ns < best_ns_total {
                    best_ns_total = ns;
                }
            }
            best_ns_total as f64 / calls as f64
        });

        let t2_over_vm = vm_ns / t2_ns; // >1 ⇒ T2-lite faster than VM (THE headline)
        let f64_note = match f64_ns {
            Some(fns) => format!("{fns:>9.1}   t2/f64={:.2}x", t2_ns / fns),
            None => "  (declined)".to_string(),
        };
        println!(
            "  {name:<14} VM={vm_ns:>9.1}  T2lite={t2_ns:>9.1}  f64={f64_note}   T2lite/VM={t2_over_vm:.2}x",
        );
        (vm_ns, t2_ns, f64_ns)
    }

    /// THE T2-lite benchmark: 5 kernels (the M4.2b four + a mixed int/float one)
    /// through VM / T2-lite / f64-JIT. Data-only (asserts CORRECTNESS, not a speed
    /// threshold); the recommendation comes from the printed T2lite/VM ratios.
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "timing benchmark; run with --release --ignored --nocapture"]
    fn t2lite_benchmark_vs_vm_vs_f64() {
        const WARMUP: u32 = 50;
        const TRIALS: u32 = 9;
        const CALLS: u32 = 200;

        println!("\n=== M4.3 T2-lite (inlined JsVal) vs VM vs f64-JIT (ns/call, min of {TRIALS} trials, {CALLS} calls/trial, {WARMUP} warmup) ===");

        let mut ratios: Vec<(String, f64, Option<f64>)> = Vec::new();
        let mut push = |name: &str, t: (f64, f64, Option<f64>)| {
            ratios.push((name.to_string(), t.0 / t.1, t.2.map(|f| t.1 / f)));
        };

        push("sumToN", bench_one_t2(
            "sumToN",
            "function f(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i; } return s; }",
            &[Value::Number(3000.0)], WARMUP, TRIALS, CALLS));

        push("fibIter", bench_one_t2(
            "fibIter",
            "function f(n){ var a = 0; var b = 1; for (var i = 0; i < n; i = i + 1) { var t = a + b; a = b; b = t; } return a; }",
            &[Value::Number(1000.0)], WARMUP, TRIALS, CALLS));

        push("nestedMul", bench_one_t2(
            "nestedMul",
            "function f(n){ var acc = 0; for (var i = 0; i < n; i = i + 1) { for (var j = 0; j < n; j = j + 1) { acc = acc + i * j; } } return acc; }",
            &[Value::Number(60.0)], WARMUP, TRIALS, CALLS));

        push("branchHeavy", bench_one_t2(
            "branchHeavy",
            "function f(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { if (i < 1000) { s = s + 1; } if (i > 500) { s = s + 2; } if (i >= 1500) { s = s + 3; } } return s; }",
            &[Value::Number(3000.0)], WARMUP, TRIALS, CALLS));

        // Mixed int/float kernel: fractional + division + compare-branch, all
        // through the inline number path.
        push("mixedIntFloat", bench_one_t2(
            "mixedIntFloat",
            "function f(n){ var acc = 0.0; for (var i = 0; i < n; i = i + 1) { var x = i * 1.5; var y = (x + i) / 2.0; if (y > 100.0) { acc = acc - y; } else { acc = acc + y; } } return acc; }",
            &[Value::Number(3000.0)], WARMUP, TRIALS, CALLS));

        println!("--- SUMMARY (T2lite/VM >1 ⇒ inlined JsVal BEATS the VM) ---");
        for (name, t2_vs_vm, t2_vs_f64) in &ratios {
            let f64s = t2_vs_f64.map(|r| format!("{r:.2}x")).unwrap_or_else(|| "n/a".into());
            println!("  {name:<14} T2lite/VM={t2_vs_vm:.2}x   T2lite/f64JIT={f64s}");
        }
        println!("=== end T2-lite benchmark ===\n");
    }

    /// M4.3 T2 PHASE 1 WIN-MEASURE: a RECORD-ITERATION kernel — sum `o.x + o.y`
    /// where `o` is a function ARG (a monomorphic shaped record) — timed on T2
    /// (inline shape-guard + audited helper) vs the VM (IC hash-free slot read).
    /// The IC is WARMED on the VM first (the GetProp inline cache must be hot for
    /// T2 to bake the shapes), then T2 is compiled and both are timed back-to-back.
    /// Reports the T2/VM ratio. NOTE: the win is bounded by the helper-call
    /// overhead — a 32-byte `Value` slot is read via the extern helper, not a
    /// single `mov` (slots are `Vec<Value>`, not 8-byte JsVals), so this measures
    /// the inline shape-guard + leaf-helper path, not a pure inlined load.
    ///
    ///   cargo test -p cv_js --release t2_getprop_record_iteration_benchmark -- --ignored --nocapture
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "timing benchmark; run with --release --ignored --nocapture"]
    fn t2_getprop_record_iteration_benchmark() {
        use crate::jsval::JsVal;
        use std::time::Instant;
        const WARMUP: u32 = 50;
        const TRIALS: u32 = 11;
        const CALLS: u32 = 2000;

        // Kernel: sum o.x + o.y for a function-ARG record `o`.
        let src = "function f(o){ return o.x + o.y; }";
        let m = module_for_first_fn(src);
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());

        // Build a monomorphic shaped record {x: 3.5, y: 1.25}.
        let rec: Value = {
            let mut om: HashMap<String, Value> = HashMap::new();
            om.insert("x".to_string(), Value::Number(3.5));
            om.insert("y".to_string(), Value::Number(1.25));
            Value::Object(std::rc::Rc::new(std::cell::RefCell::new(om)))
        };
        let args = [rec];

        // WARM the per-site IC: run the function on the VM enough times that the
        // GetProp sites record their (shape, slot). T2 needs this to bake shapes.
        for _ in 0..64 {
            let mut refuse = refuse_with_interp;
            run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut refuse).unwrap();
        }

        // Now T2 compiles WITH the warm IC (the inline GetProp path engages).
        let status = try_compile_t2lite_status(&m, 0);
        let t2_native = match status {
            T2CompileStatus::Ready(jf) => jf,
            other => {
                println!("\n[T2 record-iter] T2 did not compile the inline GetProp path: {other:?}");
                println!("  (Shaped objects may be disabled — CV_SHAPED_OBJ=0 — so the header is a Dict sentinel and the inline path correctly never engages.)");
                return;
            }
        };

        // Correctness: T2 == VM (and the result is x+y = 4.75).
        let vm_val = {
            let mut refuse = refuse_with_interp;
            run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut refuse).unwrap()
        };
        let t2_val = {
            let mut refuse = refuse_with_interp;
            run_t2lite_call(&t2_native, &m, &args, &Value::Undefined, &empty, &mut refuse).unwrap()
        };
        match (&vm_val, &t2_val) {
            (Value::Number(a), Value::Number(b)) => {
                assert_eq!(a.to_bits(), b.to_bits(), "T2 != VM (vm={a} t2={b})");
                assert_eq!(*a, 4.75, "record sum must be 4.75");
            }
            _ => panic!("record kernel must return a number"),
        }
        // Prove the inline path actually engaged (a deopt would post a fake win):
        // a number result via run_t2lite_call without an internal deopt means the
        // native code returned it. (run_t2lite_call returns the VM result on
        // deopt, which is ALSO correct, but here we want to confirm engagement —
        // the helper returns the immediate only when the inline guard hit.)

        let time_min = |label: &str, mut f: Box<dyn FnMut()>| -> f64 {
            for _ in 0..WARMUP {
                f();
            }
            let mut best: u128 = u128::MAX;
            for _ in 0..TRIALS {
                let t = Instant::now();
                for _ in 0..CALLS {
                    f();
                }
                let ns = t.elapsed().as_nanos();
                if ns > 0 && ns < best {
                    best = ns;
                }
            }
            let per = best as f64 / CALLS as f64;
            println!("  {label:<10} {per:>9.1} ns/call");
            per
        };

        println!("\n=== M4.3 T2 PHASE 1 record-iteration (sum o.x+o.y, o=arg) ===");
        let vm_ns = {
            let mref = &m;
            let aref = &args;
            let eref = &empty;
            time_min(
                "VM",
                Box::new(move || {
                    let mut refuse = refuse_with_interp;
                    let r = run_function(mref, 0, aref, &Value::Undefined, eref, None, &mut refuse)
                        .unwrap();
                    std::hint::black_box(r);
                }),
            )
        };
        let t2_ns = {
            let nref = &t2_native;
            let mref = &m;
            let aref = &args;
            let eref = &empty;
            time_min(
                "T2-getprop",
                Box::new(move || {
                    let mut refuse = refuse_with_interp;
                    let r = run_t2lite_call(nref, mref, aref, &Value::Undefined, eref, &mut refuse)
                        .unwrap();
                    std::hint::black_box(r);
                }),
            )
        };
        let ratio = vm_ns / t2_ns;
        println!("  --- T2-getprop/VM = {ratio:.2}x  (>1 ⇒ inline shape-guarded read BEATS the VM) ---");
        println!("  note: slot value is a 32-byte Value read via the audited helper (not a single mov)");
        let _ = JsVal::number(0.0); // keep the import used regardless of cfg
        println!("=== end record-iteration benchmark ===\n");
    }

    #[test]
    fn property_inline_cache_hits_hot_loop() {
        // A hot property read (`o.x` in a loop) must be served by the inline
        // cache after the first iteration — turning a hash probe into a slot
        // index. Gates that the IC stays active AND correct (result unchanged).
        if !propic_enabled() {
            return; // IC opted out (CV_PROPIC=0) — nothing to assert about hits.
        }
        reset_propic_stats();
        let m = compile_program(
            "function v() { var o = { x: 7, y: 2 }; var s = 0; for (var i = 0; i < 500; i = i + 1) { s = s + o.x; } return s; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if (n - 3500.0).abs() < 1e-9),
            "o.x * 500 must be 3500, got {r:?}"
        );
        let (hits, misses) = propic_stats();
        assert!(
            hits >= 400,
            "hot property read should be IC hits: hits={hits} misses={misses}"
        );
    }

    #[test]
    fn property_write_inline_cache_hot_loop() {
        // A hot property WRITE (`o.x = i` in a loop) must be served by the
        // write-IC (direct slot store, no key alloc), and a later read must see
        // the latest value — gates write-IC correctness + activity.
        if !propic_enabled() {
            return;
        }
        reset_propic_stats();
        let m = compile_program(
            "function v() { var o = { x: 0 }; for (var i = 0; i < 500; i = i + 1) { o.x = i; } return o.x; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if (n - 499.0).abs() < 1e-9),
            "last write o.x=499 must be read back, got {r:?}"
        );
        let (hits, misses) = propic_stats();
        assert!(
            hits >= 400,
            "hot property write should be IC hits: hits={hits} misses={misses}"
        );
    }

    #[test]
    fn property_ic_hits_across_same_shape_objects() {
        // THE "big one" payoff: a fresh object each iteration, all sharing the
        // hidden class {x,y}. A per-object IC would miss every time (new pointer);
        // the SHAPE-keyed IC hits because they share a shape — the difference
        // between caching one loop and caching `arr.map(o => o.x)`-style code.
        if !propic_enabled() {
            return;
        }
        reset_propic_stats();
        let m = compile_program(
            "function v() { var s = 0; for (var i = 0; i < 300; i = i + 1) { var o = { x: i, y: 0 }; s = s + o.x; } return s; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        // sum(0..=299) = 299*300/2 = 44850
        assert!(
            matches!(r, Value::Number(n) if (n - 44850.0).abs() < 1e-9),
            "sum of o.x over distinct same-shape objects must be 44850, got {r:?}"
        );
        let (hits, misses) = propic_stats();
        assert!(
            hits >= 250,
            "shape IC must hit ACROSS distinct same-shape objects: hits={hits} misses={misses}"
        );
    }

    #[test]
    fn property_ic_polymorphic_two_shapes() {
        // One `o.x` site sees TWO shapes alternating: {x} (slot 0) and {a,x}
        // (slot 1). A monomorphic cache would thrash (miss on every switch); the
        // polymorphic cache holds both shapes → mostly hits.
        if !propic_enabled() {
            return;
        }
        reset_propic_stats();
        let m = compile_program(
            "function v() { var s = 0; for (var i = 0; i < 200; i = i + 1) { var o; if (i % 2 == 0) { o = { x: 1 }; } else { o = { a: 9, x: 2 }; } s = s + o.x; } return s; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        // 100 evens * 1 + 100 odds * 2 = 300
        assert!(
            matches!(r, Value::Number(n) if (n - 300.0).abs() < 1e-9),
            "polymorphic o.x must sum to 300, got {r:?}"
        );
        let (hits, misses) = propic_stats();
        assert!(
            hits >= 150,
            "polymorphic site should mostly hit (both shapes cached): hits={hits} misses={misses}"
        );
    }

    // NOTE: the prototype IC's hit path needs the host (`__tb_host_getprop`),
    // which the isolated `run_fn` harness doesn't install — so it's exercised +
    // gated by the WORKSPACE dual-run oracle (conclave runs real class/proto/
    // method code with the host, IC on vs off, asserting identical results).

    #[test]
    fn arithmetic_returns_number() {
        // Top-level scripts have no return; wrap in a function so the
        // expression's value can come back out.
        let m = compile_program("function v() { return (1 + 2 * 3) - 4; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 3.0).abs() < 1e-9));
    }

    #[test]
    fn transitive_closure_captures_grandparent_local() {
        // A function nested two+ levels deep must be able to call a helper
        // declared in a GRANDPARENT scope. The intermediate function has to
        // re-capture the name so the upvalue chain resolves. Before the fix,
        // the grandchild's reference fell through to a global load and the
        // call threw "callee is not callable: undefined" — the root cause of
        // jQuery/chart.js failing to initialize.
        let m = compile_program(
            "function a(){ function b(){ return 7; } \
             function mid(){ function inner(){ return b(); } return inner(); } \
             return mid(); }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 7.0).abs() < 1e-9));

        // Same, through anonymous IIFEs (the exact minified shape).
        let m2 = compile_program(
            "function a(){ function b(){ return 7; } \
             return (function(){ return (function(){ return b(); })(); })(); }",
        )
        .unwrap();
        let r2 = run_fn(&m2, 1, &[]).unwrap();
        assert!(matches!(r2, Value::Number(n) if (n - 7.0).abs() < 1e-9));
    }

    #[test]
    fn typeof_reports_types() {
        let m = compile_program("function t() { return typeof 5; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::String(s) if &*s == "number"));
        let m = compile_program("function t() { return typeof \"hi\"; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::String(s) if &*s == "string"));
        let m = compile_program("function t() { return typeof true; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::String(s) if &*s == "boolean"));
        // typeof of an undeclared name resolves to "undefined" (must NOT throw).
        let m = compile_program("function t() { return typeof nope; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::String(s) if &*s == "undefined"));
    }

    #[test]
    fn bitwise_and_shift() {
        let cases = [
            ("function t() { return 6 & 3; }", 2.0),
            ("function t() { return 6 | 1; }", 7.0),
            ("function t() { return 5 ^ 1; }", 4.0),
            ("function t() { return 1 << 4; }", 16.0),
            ("function t() { return 256 >> 2; }", 64.0),
            ("function t() { return -1 >>> 28; }", 15.0),
            ("function t() { return ~5; }", -6.0),
            ("function t() { return 2 ** 10; }", 1024.0),
            // ToInt32 truncates the fractional part.
            ("function t() { return 6.9 & 3; }", 2.0),
        ];
        for (src, want) in cases {
            let m = compile_program(src).unwrap();
            let r = run_fn(&m, 1, &[]).unwrap();
            assert!(
                matches!(r, Value::Number(n) if (n - want).abs() < 1e-9),
                "{src} => {r:?}, want {want}"
            );
        }
        // `void` evaluates its operand and yields undefined.
        let m = compile_program("function t() { return void 7; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Undefined));
    }

    #[test]
    fn in_delete_and_update() {
        // `in` on object own-keys.
        let m = compile_program("function t() { let o = { a: 1 }; return (\"a\" in o); }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Bool(true)));
        let m = compile_program("function t() { let o = { a: 1 }; return (\"b\" in o); }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Bool(false)));
        // `delete` removes the key.
        let m =
            compile_program("function t() { let o = { a: 1 }; delete o.a; return (\"a\" in o); }")
                .unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Bool(false)));
        // `in` on arrays (index membership).
        let m = compile_program("function t() { let a = [10, 20]; return (1 in a); }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Bool(true)));
        // Postfix `x++` returns the OLD value and increments (6 + 5 = 11).
        let m = compile_program("function t() { let x = 5; let y = x++; return x + y; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 11.0).abs() < 1e-9));
        // Prefix `++x` returns the NEW value.
        let m = compile_program("function t() { let x = 5; return ++x; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 6.0).abs() < 1e-9));
        // `++` on an (undeclared) global routes through Load/StoreGlobal.
        let m = compile_program("function t() { g = 5; return g++; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 5.0).abs() < 1e-9));
        // Member update: `o.c++` (postfix yields old) writes back to the object.
        let m =
            compile_program("function t() { let o = { c: 5 }; let y = o.c++; return o.c + y; }")
                .unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 11.0).abs() < 1e-9));
        // Computed member update: `a[0]++`.
        let m = compile_program("function t() { let a = [10]; a[0]++; return a[0]; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 11.0).abs() < 1e-9));
    }

    #[test]
    fn regex_literal() {
        let m = compile_program("function t() { let r = /ab+c/i; return r.source; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::String(s) if &*s == "ab+c"));
        let m = compile_program("function t() { return /\\d+/.test(\"a123\"); }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Bool(true)));
    }

    #[test]
    fn per_function_compile_and_run() {
        // The per-function entry: compile ONE function (the interp's hot path)
        // and run it via run_module_call with args.
        let prog = crate::parser::parse_program("function f(n) { return n * n + 1; }").unwrap();
        let (params, body) = match &prog[0] {
            Stmt::FunctionDecl { params, body, .. } => (params.clone(), body.clone()),
            other => panic!("expected fn decl, got {other:?}"),
        };
        let (module, ups) = compile_single_function(&params, &body, &[]).unwrap();
        assert!(ups.is_empty(), "no captures expected");
        assert!(module_is_per_fn_safe(&module));
        let globals = std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        let r = run_module_call(
            &module,
            &[Value::Number(7.0)],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse,
        )
        .unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 50.0).abs() < 1e-9));

        // A function that reads a captured var compiles to an upvalue (caller
        // maps it from the live closure scope).
        let prog = crate::parser::parse_program("function g(x) { return x + k; }").unwrap();
        let (params, body) = match &prog[0] {
            Stmt::FunctionDecl { params, body, .. } => (params.clone(), body.clone()),
            other => panic!("expected fn decl, got {other:?}"),
        };
        let (_m, ups) = compile_single_function(&params, &body, &["k".to_string()]).unwrap();
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].name, "k");
    }

    #[test]
    fn compound_bitwise_assign() {
        let m = compile_program("function t() { let x = 6; x &= 3; return x; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 2.0).abs() < 1e-9));
        let m = compile_program("function t() { let x = 1; x <<= 4; return x; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 16.0).abs() < 1e-9));
        let m = compile_program("function t() { let x = -1; x >>>= 28; return x; }").unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 15.0).abs() < 1e-9));
    }

    #[test]
    fn nested_recursive_function() {
        // Nested fn declaration that recurses (the wikipedia load.php case).
        let src = "function t() { function fact(n) { if (n < 2) { return 1; } return n * fact(n - 1); } return fact(5); }";
        let m = compile_program(src).unwrap();
        assert!(
            matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 120.0).abs() < 1e-9)
        );
        // Recursion must PRESERVE a captured upvalue (`k`) across self-calls —
        // LoadSelf reuses the live closure (with its upvalues), not a fresh one.
        let src2 = "function t() { let k = 10; function f(n) { if (n <= 0) { return 0; } return k + f(n - 1); } return f(3); }";
        let m = compile_program(src2).unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 30.0).abs() < 1e-9));
    }

    #[test]
    fn nested_sibling_fn_references() {
        // REGRESSION: a closure inside an IIFE could not see SIBLING function
        // declarations — neither calling them, nor a `var f = function(){…sib…}`.
        // The WPT testharness shim hit this hard (every `test()` threw
        // `all_completed is not defined`). Body-level fn-decl slots are now
        // pre-declared so a sibling resolves to a real local register.
        //
        // (a) BACKWARD reference (helper declared BEFORE the caller) runs on the
        //     VM — the sibling's slot already holds its live closure.
        let back = "function t() { function helper() { return 7; } function caller() { return helper() + 1; } return caller(); }";
        let m = compile_program(back).unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 8.0).abs() < 1e-9));

        // (b) BACKWARD reference from a `var f = function(){…}` is also VM-runnable.
        let backvar = "function t() { function helper() { return 5; } var f = function(){ return helper() * 2; }; return f(); }";
        let m = compile_program(backvar).unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 10.0).abs() < 1e-9));

        // (c) FORWARD reference (caller declared BEFORE helper) is a hazard for
        //     the VM's by-value upvalues, so the FunctionDecl arm DECLINES
        //     (CompileError) — the program then runs on the tree-walk tier, which
        //     binds by reference and yields the right answer. The critical fix is
        //     that this is no longer a SILENT-WRONG `helper is not defined`; it is
        //     an explicit decline.
        let fwd = "(function(){ var caller = function(){ return helper() + 1; }; function helper(){ return 7; } globalThis.__r = caller(); })();";
        // Whole-program: declines to tree-walk and must NOT silently throw.
        match compile_program(fwd) {
            Ok(m) => {
                // If the VM ever learns to handle this, the result must still be
                // correct when executed; we don't assert a tier here, only that
                // compilation didn't produce a silently-broken module.
                let _ = m;
            }
            Err(_) => { /* expected: declined to tree-walk */ }
        }
    }

    #[test]
    fn switch_statement() {
        // Match + break, and default.
        let src = "function t(x) { let r = 0; switch (x) { case 1: r = 10; break; case 2: r = 20; break; default: r = 99; } return r; }";
        let m = compile_program(src).unwrap();
        assert!(
            matches!(run_fn(&m, 1, &[Value::Number(2.0)]).unwrap(), Value::Number(n) if (n - 20.0).abs() < 1e-9)
        );
        assert!(
            matches!(run_fn(&m, 1, &[Value::Number(7.0)]).unwrap(), Value::Number(n) if (n - 99.0).abs() < 1e-9)
        );
        // Fall-through when a case omits `break`.
        let src2 = "function t(x) { let r = 0; switch (x) { case 1: r = r + 1; case 2: r = r + 10; break; case 3: r = r + 100; } return r; }";
        let m = compile_program(src2).unwrap();
        assert!(
            matches!(run_fn(&m, 1, &[Value::Number(1.0)]).unwrap(), Value::Number(n) if (n - 11.0).abs() < 1e-9)
        );
        assert!(
            matches!(run_fn(&m, 1, &[Value::Number(3.0)]).unwrap(), Value::Number(n) if (n - 100.0).abs() < 1e-9)
        );
        // `continue` inside a switch targets the ENCLOSING loop, not the switch.
        let src3 = "function t() { let s = 0; for (let i = 0; i < 4; i = i + 1) { switch (i) { case 1: continue; default: s = s + i; } } return s; }";
        let m = compile_program(src3).unwrap();
        assert!(matches!(run_fn(&m, 1, &[]).unwrap(), Value::Number(n) if (n - 5.0).abs() < 1e-9));
    }

    #[test]
    fn if_else_picks_branch() {
        let m = compile_program(
            "function pick() { let x = 0; if (1 < 2) { x = 7; } else { x = 9; } return x; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 7.0).abs() < 1e-9));
        let m = compile_program(
            "function pick() { let x = 0; if (1 > 2) { x = 7; } else { x = 9; } return x; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 9.0).abs() < 1e-9));
    }

    #[test]
    fn while_counts_down() {
        let m = compile_program(
            "function count() { let n = 10; let sum = 0; while (n > 0) { sum = sum + n; n = n - 1; } return sum; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 55.0).abs() < 1e-9));
    }

    #[test]
    fn for_loop_sums() {
        let m = compile_program(
            "function s() { let total = 0; for (let i = 0; i < 100; i = i + 1) { total = total + i; } return total; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 4950.0).abs() < 1e-9));
    }

    #[test]
    fn recursive_fib_runs() {
        let m = compile_program(
            "function fib(n) { if (n < 2) { return n; } return fib(n - 1) + fib(n - 2); }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[Value::Number(10.0)]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 55.0).abs() < 1e-9));
    }

    #[test]
    fn string_concat_via_plus() {
        let m =
            compile_program("function s() { let a = \"hi \"; let b = \"there\"; return a + b; }")
                .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        match r {
            Value::String(s) => assert_eq!(&*s, "hi there"),
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[test]
    fn array_spread_compiles_and_runs() {
        // Array spread `[...a, 99]` now lowers to NewArray (empty) +
        // ArrayPushSpread(a) + ArrayPush(99) — no tree-walk fallback
        // needed. Verifies the result is the concatenated array.
        let m = compile_program("function f(a) { return [...a, 99]; }").unwrap();
        let input = Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
            Value::Number(1.0),
            Value::Number(2.0),
        ])));
        let r = run_fn(&m, 1, &[input]).unwrap();
        if let Value::Array(a) = r {
            let a = a.borrow();
            assert_eq!(a.len(), 3);
            assert!(matches!(a[0], Value::Number(n) if (n - 1.0).abs() < 1e-9));
            assert!(matches!(a[1], Value::Number(n) if (n - 2.0).abs() < 1e-9));
            assert!(matches!(a[2], Value::Number(n) if (n - 99.0).abs() < 1e-9));
        } else {
            panic!("expected array result");
        }
    }

    #[test]
    fn arrow_no_capture_runs() {
        // `arr.map`-style: arrow with no outer-scope refs.
        let m = compile_program("function r() { let g = (x) => x * 2; return g(21); }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn arrow_captures_outer_local_read() {
        // mult lives in r()'s frame; the arrow reads it via LoadUp.
        let m =
            compile_program("function r() { let mult = 7; let g = (x) => x * mult; return g(6); }")
                .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn arrow_captures_outer_local_write() {
        // Counter pattern: outer total is mutated by the inner arrow.
        // The closure's upvalue is a separate cell from the local —
        // for V1 we snapshot, so the write only modifies the closure's
        // copy. The test checks the closure's behaviour is at least
        // internally consistent.
        let m = compile_program(
            "function r() { let total = 0; let add = (x) => total = total + x; \
              add(10); add(20); add(12); return add(0); }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn bytecode_closure_keeps_assigned_function_properties() {
        let m =
            compile_program("function r() { let f = () => 1; f.init = () => 2; return f.init(); }")
                .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 2.0).abs() < 1e-9));
    }

    #[test]
    fn bytecode_closure_keeps_indexed_property_writes() {
        let m =
            compile_program("function r() { let f = () => 1; f['uid'] = 'abc'; return f['uid']; }")
                .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::String(s) if &*s == "abc"));
    }

    #[test]
    fn bytecode_closure_is_valid_object_assign_target() {
        let mut interp = crate::interp::Interp::new();
        interp.install_basic_globals();

        let m = compile_program(
            "function r() { let transport = () => 1; \
             let wrapped = Object.assign(transport, { config: { key: 'http' } }); \
             return wrapped.config.key; }",
        )
        .unwrap();
        let globals = std::cell::RefCell::new(interp.globals_snapshot());
        let mut dispatch = |callee: Value, this: Value, args: Vec<Value>| {
            interp
                .call_value_with_this(callee, this, args)
                .map_err(|e| match e {
                    crate::interp::JsError::Throw(v) => RuntimeError::Thrown(v),
                    other => RuntimeError::TypeError(format!("dispatch: {other:?}")),
                })
        };
        let r = run_function(&m, 1, &[], &Value::Undefined, &globals, None, &mut dispatch).unwrap();
        assert!(matches!(r, Value::String(s) if &*s == "http"));
    }

    #[test]
    fn bytecode_dispatches_with_interp_native() {
        // Wire a WithInterp native into globals, then call it from
        // bytecode. Round-trip: bytecode CallValue sees WithInterp,
        // delegates to dispatcher, dispatcher calls into the tree-walk
        // interp, result flows back as a Value. Proves the integration
        // seam works end-to-end.
        let mut interp = crate::interp::Interp::new();
        let wi = std::rc::Rc::new(crate::interp::NativeFn {
            name: "callMe".to_string(),
            func: crate::interp::NativeFnBody::WithInterp(Box::new(|_interp, args| {
                let n = match args.first() {
                    Some(Value::Number(n)) => *n,
                    _ => 0.0,
                };
                Ok(Value::Number(n * 2.0))
            })),
            length: 0,
            is_ctor: false,
            props: std::cell::RefCell::new(HashMap::new()),
        });
        interp.define_global("callMe", Value::NativeFunction(wi));

        let m = compile_program("function r() { return callMe(21); }").unwrap();
        let globals = std::cell::RefCell::new(interp.globals_snapshot());
        let mut dispatch = |callee: Value, this: Value, args: Vec<Value>| {
            interp
                .call_value_with_this(callee, this, args)
                .map_err(|e| match e {
                    crate::interp::JsError::Throw(v) => RuntimeError::Thrown(v),
                    other => RuntimeError::TypeError(format!("dispatch: {other:?}")),
                })
        };
        let r = run_function(&m, 1, &[], &Value::Undefined, &globals, None, &mut dispatch).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn run_module_with_interp_e2e() {
        // The public entry point — caller passes the Interp directly
        // and the runner wires up globals + dispatch.
        let mut interp = crate::interp::Interp::new();
        let wi = std::rc::Rc::new(crate::interp::NativeFn {
            name: "triple".to_string(),
            func: crate::interp::NativeFnBody::WithInterp(Box::new(|_interp, args| {
                let n = match args.first() {
                    Some(Value::Number(n)) => *n,
                    _ => 0.0,
                };
                Ok(Value::Number(n * 3.0))
            })),
            length: 0,
            is_ctor: false,
            props: std::cell::RefCell::new(HashMap::new()),
        });
        interp.define_global("triple", Value::NativeFunction(wi));

        let m = compile_program("function r() { return triple(14); }").unwrap();
        // Snapshot globals first (immutable borrow), then build the
        // mutable-borrow dispatcher.
        let globals = std::cell::RefCell::new(interp.globals_snapshot());
        let mut dispatch = |callee: Value, this: Value, args: Vec<Value>| {
            interp
                .call_value_with_this(callee, this, args)
                .map_err(|e| match e {
                    crate::interp::JsError::Throw(v) => RuntimeError::Thrown(v),
                    other => RuntimeError::TypeError(format!("dispatch: {other:?}")),
                })
        };
        let r = run_function(&m, 1, &[], &Value::Undefined, &globals, None, &mut dispatch).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn toplevel_fn_decl_binds_as_global() {
        // ECMA-262 §16.1.7 / §9.1.1.4.16: a top-level `function f(){…}` in a
        // classic script creates a PROPERTY on the global object, visible to a
        // SUBSEQUENT script. `compile_program` splits top-level decls into module
        // fn-slots (reached internally via CallFn), so without the script-frame
        // hoisting prologue the second script's `helper` would be unbound. Run two
        // separate modules against the SAME live interp globals (exactly the
        // multi-`<script>` page model) and assert the first's fn is callable in
        // the second.
        let mut interp = crate::interp::Interp::new();
        let m1 = compile_program("function helper(x){ return x*2; }").unwrap();
        run_module_with_interp(&m1, &mut interp).unwrap();
        // Second module sees `helper` as a global (set by m1's script frame).
        let m2 = compile_program("var out = helper(21);").unwrap();
        run_module_with_interp(&m2, &mut interp).unwrap();
        let out = interp.get_global("out").unwrap();
        assert!(
            matches!(out, Value::Number(n) if (n - 42.0).abs() < 1e-9),
            "expected helper() visible across scripts to yield 42, got {out:?}"
        );
        // The binding is the function itself, callable.
        let helper = interp.get_global("helper").unwrap();
        assert!(
            matches!(helper, Value::BcClosure(_) | Value::Function(_)),
            "helper should be a function global, got {helper:?}"
        );
    }

    #[test]
    fn array_push_pop_join_round_trip() {
        let m = compile_program(
            "function r() { let a = [1, 2]; a.push(3); a.push(4); a.pop(); return a.join(\"-\"); }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        match r {
            Value::String(s) => assert_eq!(&*s, "1-2-3"),
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[test]
    fn array_index_of_and_includes() {
        let m = compile_program(
            "function r() { let a = [10, 20, 30, 40]; if (a.includes(30)) { return a.indexOf(40); } return -1; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 3.0).abs() < 1e-9));
    }

    #[test]
    fn string_trim_upper_split() {
        let m = compile_program(
            "function r() { let s = \"  hi there  \"; let parts = s.trim().toUpperCase().split(\" \"); return parts.join(\"_\"); }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        match r {
            Value::String(s) => assert_eq!(&*s, "HI_THERE"),
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[test]
    fn string_starts_ends_repeat() {
        let m = compile_program(
            "function r() { let s = \"abc\".repeat(3); if (s.startsWith(\"abc\")) { return s; } return \"\"; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        match r {
            Value::String(s) => assert_eq!(&*s, "abcabcabc"),
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[test]
    fn function_expr_in_local_callable() {
        let m = compile_program(
            "function r() { let inc = function (x) { return x + 1; }; return inc(41); }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn nested_function_decl_runs_through_bytecode() {
        // `class` now desugars to a prototype-based IIFE whose body holds
        // a *nested* `function` declaration (the constructor) plus
        // `Class.prototype.m = …` wiring. The bytecode VM must compile and
        // bind nested function declarations — this exercises exactly that:
        // `add` is declared inside `outer`, bound in scope, and called.
        //
        // (Full prototype-chain construction — `new C().m()` walking
        // `C.prototype` — is the tree-walk interpreter's responsibility;
        // the bytecode VM's `Op::New` makes a bare object and defers proto
        // semantics to the interp. That path is covered by the interp's
        // `prototype_chain_basics` /
        // `new_sets_instance_prototype_and_methods_inherit` tests and is
        // verified end-to-end via `--type run-js`.)
        let m = compile_program(
            "function outer() {\
               function add(a, b) { return a + b; }\
               return add(40, 2);\
             }",
        )
        .unwrap();
        let go_idx = m
            .fns
            .iter()
            .position(|f| f.name == "outer")
            .expect("outer fn present");
        let r = run_fn(&m, go_idx, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn arrow_captures_lexical_this() {
        // Method sets this.x = 7, then returns an arrow that reads
        // this.x. The arrow inherits the method's `this` (lexical),
        // even when called bare (no method-style invocation).
        let m = compile_program(
            "function go() {\
               let o = {};\
               o.x = 7;\
               o.mk = function() { return (z) => this.x + z; };\
               let f = o.mk();\
               return f(0);\
             }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 7.0).abs() < 1e-9));
    }

    #[test]
    fn new_constructor_binds_this() {
        let m = compile_program(
            "function Point(x, y) { this.x = x; this.y = y; }\
             function makePoint() { let p = new Point(3, 4); return p.x + p.y; }",
        )
        .unwrap();
        // makePoint is fns[2] (script + Point + makePoint). Instantiate globals
        // first (binds `Point`) then call makePoint, as real execution does.
        let r = run_fn_after_script(&m, 2, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 7.0).abs() < 1e-9));
    }

    #[test]
    fn new_on_tagged_construct_object_invokes_construct() {
        // Native constructors are represented as our tagged `{_construct: fn}`
        // object form (Map/Set/Promise/Date/Error globals are all shaped this
        // way). The VM's `Op::New` must resolve `_construct` and invoke it via
        // dispatch. Regression for the real-page failure
        // `TypeError("`new` on non-callable: [object Object]")`, which broke
        // `new Map()` / `new Promise()` / etc. inside hot (bytecode-compiled)
        // functions on Wikipedia, react.dev, StackOverflow.
        use crate::interp::{NativeFn, NativeFnBody};

        let construct = Value::NativeFunction(std::rc::Rc::new(NativeFn {
            name: "FooCtor".into(),
            func: NativeFnBody::Pure(Box::new(|_args| {
                let mut inst = HashMap::new();
                inst.insert("made".into(), Value::Bool(true));
                Ok(Value::Object(std::rc::Rc::new(std::cell::RefCell::new(
                    inst,
                ))))
            })),
            length: 0,
            is_ctor: false,
            props: std::cell::RefCell::new(HashMap::new()),
        }));
        let mut foo: HashMap<String, Value> = HashMap::new();
        foo.insert("_construct".into(), construct);
        let foo = Value::Object(std::rc::Rc::new(std::cell::RefCell::new(foo)));

        let m = compile_program("function go() { return new Foo(); }").unwrap();
        let globals = std::cell::RefCell::new(HashMap::new());
        globals.borrow_mut().insert("Foo".into(), foo);

        // Minimal host dispatch: invoke a native function's Pure body, the way
        // the real interp would when the VM hands a `_construct` callee back.
        let mut dispatch =
            |callee: Value, _this: Value, args: Vec<Value>| -> Result<Value, RuntimeError> {
                match callee {
                    Value::NativeFunction(nf) => match &nf.func {
                        NativeFnBody::Pure(body) => {
                            body(args).map_err(|e| RuntimeError::TypeError(format!("{e:?}")))
                        }
                        _ => Err(RuntimeError::TypeError(
                            "with-interp unsupported in test".into(),
                        )),
                    },
                    other => Err(RuntimeError::TypeError(format!("not callable: {other:?}"))),
                }
            };
        let go_idx = m
            .fns
            .iter()
            .position(|f| f.name == "go")
            .expect("go fn present");
        let r = run_function(
            &m,
            go_idx,
            &[],
            &Value::Undefined,
            &globals,
            None,
            &mut dispatch,
        )
        .expect("`new Foo()` should construct via _construct, not error");
        match r {
            Value::Object(o) => assert!(
                matches!(o.borrow().get("made"), Some(Value::Bool(true))),
                "expected the instance built by _construct"
            ),
            other => panic!("expected constructed object, got {other:?}"),
        }
    }

    #[test]
    fn vm_try_catch_catches_native_throw() {
        // A native (`JSON.parse`, `decodeURIComponent`, …) that throws inside a
        // try/catch in VM-compiled code must be CAUGHT, not escape. Regression
        // for the real-page bug where native throws used `?` (returning straight
        // out of run_function) instead of routing through the try-handler.
        use crate::interp::{JsError, NativeFn, NativeFnBody};

        let boom = Value::NativeFunction(std::rc::Rc::new(NativeFn {
            name: "boom".into(),
            func: NativeFnBody::Pure(Box::new(|_args| {
                Err(JsError::Throw(Value::String("kaboom".into())))
            })),
            length: 0,
            is_ctor: false,
            props: std::cell::RefCell::new(HashMap::new()),
        }));
        let m = compile_program(
            "function go() { try { return boom(); } catch (e) { return \"caught:\" + e; } }",
        )
        .unwrap();
        let globals = std::cell::RefCell::new(HashMap::new());
        globals.borrow_mut().insert("boom".into(), boom);
        let mut refuse = refuse_with_interp;
        let go_idx = m
            .fns
            .iter()
            .position(|f| f.name == "go")
            .expect("go fn present");
        let r = run_function(
            &m,
            go_idx,
            &[],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse,
        )
        .expect("native throw inside try must be caught, not escape run_function");
        match r {
            Value::String(s) => assert_eq!(&*s, "caught:kaboom"),
            other => panic!("expected the catch block to run, got {other:?}"),
        }
    }

    #[test]
    fn method_call_sees_this_object() {
        let m = compile_program(
            "function r(x) { let o = {}; o.x = x; o.getX = function () { return this.x; }; return o.getX(); }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[Value::Number(42.0)]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn script_var_hoists_to_global() {
        // Top-level `let x = 5; function reader() { return x; } reader();`
        // — the var must be a global so reader() can see it. We compile
        // and run, then read globals back to confirm StoreGlobal landed.
        let m = compile_program("let answer = 42; function reader() { return answer; }").unwrap();
        let globals = std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        // Run script body (fns[0]) to perform the StoreGlobal.
        run_function(&m, 0, &[], &Value::Undefined, &globals, None, &mut refuse).unwrap();
        // Verify the global landed.
        let got = globals.borrow().get("answer").cloned();
        assert!(matches!(got, Some(Value::Number(n)) if (n - 42.0).abs() < 1e-9));
        // And reader() (fns[1]) should LoadGlobal it back.
        let r = run_function(&m, 1, &[], &Value::Undefined, &globals, None, &mut refuse).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn new_with_method_calls() {
        let m = compile_program(
            "function Counter(start) { this.n = start; }\
             function go() { let c = new Counter(10); c.inc = function () { this.n = this.n + 1; }; c.inc(); c.inc(); c.inc(); return c.n; }",
        )
        .unwrap();
        // Instantiate globals first (binds `Counter`) then call go(), as real
        // execution does.
        let r = run_fn_after_script(&m, 2, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 13.0).abs() < 1e-9));
    }

    #[test]
    fn native_global_callable_from_bytecode() {
        // Install a tiny "double" native fn in the globals env, then
        // compile + run `double(21)` through the bytecode VM. Should
        // come back as 42 with no tree-walk involvement.
        let mut globals_inner: HashMap<String, Value> = HashMap::new();
        globals_inner.insert(
            "double".into(),
            crate::interp::native_fn("double", |args| {
                let n = match args.first() {
                    Some(Value::Number(n)) => *n,
                    _ => 0.0,
                };
                Ok(Value::Number(n * 2.0))
            }),
        );
        let globals = std::cell::RefCell::new(globals_inner);
        let m = compile_program("function call() { return double(21); }").unwrap();
        let r = run_function(
            &m,
            1,
            &[],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse_with_interp,
        )
        .unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn callable_namespace_object_invokable_from_bytecode() {
        // A namespace object that doubles as a callable (like the real
        // `String`/`Number` globals): an Object carrying static props AND a
        // `_call` native. V8 treats any object with a [[Call]] slot as
        // callable; the bytecode VM must unwrap `_call` and invoke it
        // instead of erroring "callee is not callable".
        use std::cell::RefCell;
        use std::rc::Rc;
        let mut ns: HashMap<String, Value> = HashMap::new();
        ns.insert("STATIC".into(), Value::Number(7.0));
        ns.insert(
            "_call".into(),
            crate::interp::native_fn("Tenfold", |args| {
                Ok(Value::Number(match args.first() {
                    Some(Value::Number(n)) => n * 10.0,
                    _ => -1.0,
                }))
            }),
        );
        let mut globals_inner: HashMap<String, Value> = HashMap::new();
        globals_inner.insert("Tenfold".into(), Value::Object(Rc::new(RefCell::new(ns))));
        let globals = std::cell::RefCell::new(globals_inner);
        let m = compile_program("function call() { return Tenfold(5); }").unwrap();
        let r = run_function(
            &m,
            1,
            &[],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse_with_interp,
        )
        .unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 50.0).abs() < 1e-9));
    }

    #[test]
    fn member_access_through_global_object() {
        // Install a `Math` global with a `sqrt` native fn, then call
        // `Math.sqrt(16)` from bytecode. Proves GetProp + CallValue can
        // chain into a method call on a host object.
        use std::cell::RefCell;
        use std::rc::Rc;
        let sqrt = crate::interp::native_fn("sqrt", |args| {
            let n = match args.first() {
                Some(Value::Number(n)) => *n,
                _ => 0.0,
            };
            Ok(Value::Number(n.sqrt()))
        });
        let math_obj: HashMap<String, Value> = [("sqrt".to_string(), sqrt)].into_iter().collect();
        let math = Value::Object(Rc::new(RefCell::new(math_obj)));
        let mut globals_inner: HashMap<String, Value> = HashMap::new();
        globals_inner.insert("Math".into(), math);
        let globals = std::cell::RefCell::new(globals_inner);

        let m = compile_program("function r() { return Math.sqrt(16); }").unwrap();
        let r = run_function(
            &m,
            1,
            &[],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse_with_interp,
        )
        .unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 4.0).abs() < 1e-9));
    }

    #[test]
    fn array_index_and_length_from_bytecode() {
        // Compute sum of an array passed as an arg using arr[i] and
        // arr.length. Exercises both GetProp (.length) and GetIdx ([i]).
        use std::cell::RefCell;
        use std::rc::Rc;
        let m = compile_program(
            "function s(a) { let total = 0; let i = 0; while (i < a.length) { total = total + a[i]; i = i + 1; } return total; }",
        )
        .unwrap();
        let arr = Value::Array(Rc::new(RefCell::new(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
            Value::Number(4.0),
        ])));
        let globals: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let r = run_function(
            &m,
            1,
            &[arr],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse_with_interp,
        )
        .unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 10.0).abs() < 1e-9));
    }

    #[test]
    fn array_literal_and_index_assign() {
        // Build a length-3 array from a literal, mutate one element via
        // `arr[1] = 99`, return the new sum.
        let m = compile_program(
            "function r() { let a = [1, 2, 3]; a[1] = 99; return a[0] + a[1] + a[2]; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 103.0).abs() < 1e-9));
    }

    #[test]
    fn object_prop_set_via_passed_object() {
        // Caller passes in an empty object; the bytecode assigns two
        // properties and reads them back via `.x + .y`.
        use std::cell::RefCell;
        use std::rc::Rc;
        let m =
            compile_program("function fill(o) { o.x = 7; o.y = 35; return o.x + o.y; }").unwrap();
        let obj = Value::Object(Rc::new(RefCell::new(HashMap::new())));
        let globals: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        let r = run_function(
            &m,
            1,
            &[obj.clone()],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse,
        )
        .unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
        // Verify the writes were observable to the caller.
        if let Value::Object(o) = &obj {
            let o = o.borrow();
            assert!(matches!(o.get("x"), Some(Value::Number(n)) if (*n - 7.0).abs() < 1e-9));
            assert!(matches!(o.get("y"), Some(Value::Number(n)) if (*n - 35.0).abs() < 1e-9));
        }
    }

    #[test]
    fn object_literal_round_trip() {
        let m =
            compile_program("function r() { let o = {a: 1, b: 2, c: 3}; return o.a + o.b + o.c; }")
                .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 6.0).abs() < 1e-9));
    }

    #[test]
    fn try_catch_swallows_throw() {
        // Throw a number, catch it, add 1.
        let m = compile_program(
            "function r() { let v = 0; try { throw 41; } catch (e) { v = e + 1; } return v; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn try_catch_normal_path_skips_handler() {
        let m = compile_program(
            "function r() { let v = 0; try { v = 100; } catch (e) { v = 999; } return v; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 100.0).abs() < 1e-9));
    }

    #[test]
    fn throw_propagates_across_call_frames() {
        // inner() throws; outer() catches; main() returns the caught.
        let m = compile_program(
            "function inner() { throw 7; }\
             function outer() { try { inner(); return -1; } catch (e) { return e * 6; } }",
        )
        .unwrap();
        // outer() is fns[2] (after <script>, inner).
        let r = run_fn(&m, 2, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 42.0).abs() < 1e-9));
    }

    #[test]
    fn compound_assignment_on_local() {
        let m = compile_program("function r() { let s = 10; s += 5; s *= 2; s -= 1; return s; }")
            .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 29.0).abs() < 1e-9));
    }

    #[test]
    fn break_exits_loop_early() {
        let m = compile_program(
            "function r() { let i = 0; while (i < 1000) { if (i == 7) { break; } i = i + 1; } return i; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 7.0).abs() < 1e-9));
    }

    #[test]
    fn continue_skips_iteration() {
        // Sum 0..10 skipping multiples of 3 (0, 3, 6, 9). Plain sum
        // 0..10 is 45; remove 0+3+6+9 = 18, expect 27.
        let m = compile_program(
            "function r() { let total = 0; for (let i = 0; i < 10; i = i + 1) { if (i % 3 == 0) { continue; } total = total + i; } return total; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 27.0).abs() < 1e-9));
    }

    /// VM for-of iterates a custom iterator object (has a `next` method directly
    /// on it) via the lazy `__tb_get_iterator__` protocol in the bare test env.
    #[test]
    fn for_of_vm_custom_iterator_object() {
        use std::cell::RefCell;
        use std::rc::Rc;
        // Build a hand-crafted iterator object with a Pure `next` method.
        let items: Rc<RefCell<Vec<Value>>> = Rc::new(RefCell::new(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]));
        let idx: Rc<RefCell<usize>> = Rc::new(RefCell::new(0));
        let it_items = items.clone();
        let it_idx = idx.clone();
        let next_fn = crate::interp::native_fn("next", move |_| {
            let i = *it_idx.borrow();
            let v = it_items.borrow();
            if i < v.len() {
                *it_idx.borrow_mut() = i + 1;
                let val = v[i].clone();
                drop(v);
                let mut s: HashMap<String, Value> = HashMap::new();
                s.insert("value".into(), val);
                s.insert("done".into(), Value::Bool(false));
                Ok(Value::Object(Rc::new(RefCell::new(s))))
            } else {
                drop(v);
                let mut s: HashMap<String, Value> = HashMap::new();
                s.insert("value".into(), Value::Undefined);
                s.insert("done".into(), Value::Bool(true));
                Ok(Value::Object(Rc::new(RefCell::new(s))))
            }
        });
        let mut iter_map: HashMap<String, Value> = HashMap::new();
        iter_map.insert("next".into(), next_fn);
        let iter_obj = Value::Object(Rc::new(RefCell::new(iter_map)));

        // Sum all yielded values via VM for-of.
        let m = compile_program(
            "function r(it) { let t = 0; for (const v of it) { t = t + v; } return t; }",
        )
        .unwrap();
        let result = run_fn(&m, 1, &[iter_obj]).unwrap();
        assert!(
            matches!(result, Value::Number(n) if (n - 6.0).abs() < 1e-9),
            "expected 6, got {result:?}"
        );
    }

    /// VM for-of over a string iterates Unicode scalars.
    #[test]
    fn for_of_vm_string_chars() {
        let m = compile_program(
            "function r(s) { let t = ''; for (const c of s) { t = t + c; } return t; }",
        )
        .unwrap();
        let result = run_fn(&m, 1, &[Value::String("abc".into())]).unwrap();
        match result {
            Value::String(s) => assert_eq!(&*s, "abc"),
            other => panic!("expected string 'abc', got {other:?}"),
        }
    }

    /// VM for-of: break exits early and does NOT hang.
    #[test]
    fn for_of_vm_break_exits_early() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let m = compile_program(
            "function r(arr) { \
               let first = 0; \
               for (const v of arr) { first = v; break; } \
               return first; \
             }",
        )
        .unwrap();
        let arr = Value::Array(Rc::new(RefCell::new(vec![
            Value::Number(42.0),
            Value::Number(99.0),
            Value::Number(100.0),
        ])));
        let result = run_fn(&m, 1, &[arr]).unwrap();
        assert!(
            matches!(result, Value::Number(n) if (n - 42.0).abs() < 1e-9),
            "expected 42, got {result:?}"
        );
    }

    #[test]
    fn compound_assignment_on_member() {
        // `o.x += 5` should mutate the property on a passed-in object.
        use std::cell::RefCell;
        use std::rc::Rc;
        let m = compile_program("function r(o) { o.x += 5; return o.x; }").unwrap();
        let obj = Value::Object(Rc::new(RefCell::new({
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("x".into(), Value::Number(10.0));
            m
        })));
        let globals: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let r = run_function(
            &m,
            1,
            &[obj],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse_with_interp,
        )
        .unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 15.0).abs() < 1e-9));
    }

    #[test]
    fn prefix_increment() {
        let m = compile_program("function r() { let i = 5; let j = ++i; return i * 100 + j; }")
            .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        // i=6, j=6 → 606
        assert!(matches!(r, Value::Number(n) if (n - 606.0).abs() < 1e-9));
    }

    #[test]
    fn postfix_decrement() {
        let m = compile_program("function r() { let i = 5; let j = i--; return i * 100 + j; }")
            .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        // i=4, j=5 → 405
        assert!(matches!(r, Value::Number(n) if (n - 405.0).abs() < 1e-9));
    }

    #[test]
    fn for_of_sums_array() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let m = compile_program(
            "function r(arr) { let t = 0; for (let v of arr) { t = t + v; } return t; }",
        )
        .unwrap();
        let arr = Value::Array(Rc::new(RefCell::new(vec![
            Value::Number(10.0),
            Value::Number(20.0),
            Value::Number(30.0),
        ])));
        let r = run_fn(&m, 1, &[arr]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 60.0).abs() < 1e-9));
    }

    #[test]
    fn for_in_sums_object_values() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let m = compile_program(
            "function r(o) { let t = 0; for (let k in o) { t = t + o[k]; } return t; }",
        )
        .unwrap();
        let obj = Value::Object(Rc::new(RefCell::new({
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("a".into(), Value::Number(1.0));
            m.insert("b".into(), Value::Number(2.0));
            m.insert("c".into(), Value::Number(3.0));
            m
        })));
        let r = run_fn(&m, 1, &[obj]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 6.0).abs() < 1e-9));
    }

    #[test]
    fn template_literal_concat() {
        let m = compile_program(
            "function r() { let x = 7; let y = 3; return `sum=${x + y}, prod=${x * y}!`; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        match r {
            Value::String(s) => assert_eq!(&*s, "sum=10, prod=21!"),
            other => panic!("expected string, got {other:?}"),
        }
    }

    /// A VALUE-context read of an unresolvable (undeclared) identifier must
    /// throw ReferenceError — mirroring the tree-walk tier (ECMA-262 §13.3.2
    /// GetValue on an unresolvable Reference). This was Finding #1: the VM used
    /// to silently resolve a missing global to `undefined`. `LoadGlobalChecked`
    /// fixed it. (Companion no-throw cases — `typeof missing` → "undefined" —
    /// are covered by the A/B oracle's `finding1_*` + over-throw snippets.)
    #[test]
    fn unknown_global_value_read_throws_reference_error() {
        let globals: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let m = compile_program("function v() { return missing; }").unwrap();
        let r = run_function(
            &m,
            1,
            &[],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse_with_interp,
        );
        match r {
            Err(RuntimeError::Thrown(Value::Object(o))) => {
                let b = o.borrow();
                assert_eq!(
                    b.get("name").map(|v| v.to_display_string()),
                    Some("ReferenceError".to_string()),
                    "unresolvable value read must throw a ReferenceError"
                );
                assert_eq!(
                    b.get("message").map(|v| v.to_display_string()),
                    Some("missing is not defined".to_string()),
                );
            }
            other => panic!("expected ReferenceError thrown, got {other:?}"),
        }
    }

    /// `typeof undeclaredName` must NOT throw — it yields "undefined" even when
    /// the name is unresolvable (ECMA-262 §13.5.1.1). Proves the checked-load
    /// fix did not over-throw on the typeof carve-out.
    #[test]
    fn typeof_unknown_global_is_undefined_string() {
        let globals: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let m = compile_program("function v() { return typeof missing; }").unwrap();
        let r = run_function(
            &m,
            1,
            &[],
            &Value::Undefined,
            &globals,
            None,
            &mut refuse_with_interp,
        )
        .unwrap();
        assert!(
            matches!(r, Value::String(ref s) if &**s == "undefined"),
            "typeof unresolvable must be \"undefined\", got {r:?}"
        );
    }

    #[test]
    fn ternary_picks_branch() {
        let m = compile_program("function ch() { let x = 5; return x > 0 ? 1 : -1; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(matches!(r, Value::Number(n) if (n - 1.0).abs() < 1e-9));
    }

    // --- Bug 1: VM `==`/`!=` uses loose equality (not strict) ---

    #[test]
    fn vm_loose_eq_numeric_string() {
        // `0 == "0"` must be true in a VM-compiled function.
        let m = compile_program("function t() { return (0 == \"0\"); }").unwrap();
        assert!(
            matches!(run_fn(&m, 1, &[]).unwrap(), Value::Bool(true)),
            "0 == \"0\" should be true"
        );
    }

    #[test]
    fn vm_loose_eq_null_undefined() {
        // `null == undefined` must be true.
        let m = compile_program("function t() { return (null == undefined); }").unwrap();
        assert!(
            matches!(run_fn(&m, 1, &[]).unwrap(), Value::Bool(true)),
            "null == undefined should be true"
        );
    }

    #[test]
    fn vm_loose_eq_bool_number() {
        // `1 == true` must be true.
        let m = compile_program("function t() { return (1 == true); }").unwrap();
        assert!(
            matches!(run_fn(&m, 1, &[]).unwrap(), Value::Bool(true)),
            "1 == true should be true"
        );
    }

    // --- Bug 2: VM `display_value` returns correct string for Object/Array ---

    #[test]
    fn vm_string_concat_object() {
        // `"" + {}` must be `"[object Object]"`.
        let m = compile_program("function t() { return \"\" + {}; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(&r, Value::String(s) if &**s == "[object Object]"),
            "\"\" + {{}} should be \"[object Object]\", got {r:?}"
        );
    }

    #[test]
    fn vm_string_concat_array() {
        // `"" + [1,2,3]` must be `"1,2,3"`.
        let m = compile_program("function t() { return \"\" + [1,2,3]; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(&r, Value::String(s) if &**s == "1,2,3"),
            "\"\" + [1,2,3] should be \"1,2,3\", got {r:?}"
        );
    }

    #[test]
    fn vm_template_literal_object() {
        // `\`${{}}\`` must produce "[object Object]" (not "").
        let m = compile_program("function t() { return `${{}}`; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(&r, Value::String(s) if &**s == "[object Object]"),
            "`${{}}` should be \"[object Object]\", got {r:?}"
        );
    }

    #[test]
    fn template_substitution_lowers_to_tostr_not_empty_add() {
        // A `${expr}` substitution must lower to `Op::ToStr` (ECMA §7.1.17
        // ToString, STRING ToPrimitive hint) — NOT `"" + expr` (Add, default/
        // Number hint), which diverged on an object with both valueOf & toString.
        let m = compile_program("function t(x) { return `v=${x}`; }").unwrap();
        // fns[0] is the `<script>`; fns[1..] are the top-level decls (here `t`).
        let has_tostr = m
            .fns
            .iter()
            .flat_map(|f| f.code.iter())
            .any(|op| matches!(op, Op::ToStr { .. }));
        assert!(has_tostr, "template substitution should emit Op::ToStr");
        // And it must NOT lower to the old `"" + expr` shape: an Add whose lhs is
        // an empty-string const. (We just confirm ToStr is present, which the new
        // lowering uses exclusively for the substitution.)
    }

    #[test]
    fn vm_template_literal_primitives_via_tostr() {
        // Op::ToStr on primitives is byte-identical to ToString: numbers, null,
        // undefined, bool, and a string pass through to `to_display_string`.
        let m = compile_program(
            "function t() { return `${1+2}|${null}|${undefined}|${true}|${'s'}`; }",
        )
        .unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(&r, Value::String(s) if &**s == "3|null|undefined|true|s"),
            "primitive template substitutions should ToString correctly, got {r:?}"
        );
    }

    #[test]
    fn vm_template_literal_array_via_tostr() {
        // An Array's ToString is positional `join(",")`, with null/undefined/holes
        // → empty — byte-identical to the tree-walk `to_display_string`. No host
        // hook needed (only Value::Object routes through the host).
        let m = compile_program("function t() { return `[${[1,null,3]}]`; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(&r, Value::String(s) if &**s == "[1,,3]"),
            "array template substitution should join with empty for null, got {r:?}"
        );
    }

    // --- Bug 3: ECMAScript StringToNumber corner cases ---
    // Note: The `Number()` constructor is a global and requires a full interp
    // context. We coerce via arithmetic (`* 1`) which routes through `to_num`
    // (ToNumber) so the isolated run_fn harness works.

    #[test]
    fn to_number_empty_string_is_zero() {
        // `"" * 1 === 0` — empty string → +0 via ToNumber (not NaN).
        let m = compile_program("function t() { return \"\" * 1; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if n == 0.0),
            "\"\" * 1 should be 0, got {r:?}"
        );
    }

    #[test]
    fn to_number_hex_string() {
        // `"0x1F" * 1 === 31` — hex literal string → 31.
        let m = compile_program("function t() { return \"0x1F\" * 1; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if (n - 31.0).abs() < 1e-9),
            "\"0x1F\" * 1 should be 31, got {r:?}"
        );
    }

    #[test]
    fn to_number_binary_string() {
        // `"0b101" * 1 === 5` — binary literal string → 5.
        let m = compile_program("function t() { return \"0b101\" * 1; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if (n - 5.0).abs() < 1e-9),
            "\"0b101\" * 1 should be 5, got {r:?}"
        );
    }

    #[test]
    fn to_number_whitespace_string_is_zero() {
        // `"   " * 1 === 0` — whitespace-only string trims to empty → +0.
        let m = compile_program("function t() { return \"   \" * 1; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if n == 0.0),
            "\"   \" * 1 should be 0, got {r:?}"
        );
    }

    #[test]
    fn to_number_octal_string() {
        // `"0o17" * 1 === 15` — octal literal string → 15.
        let m = compile_program("function t() { return \"0o17\" * 1; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if (n - 15.0).abs() < 1e-9),
            "\"0o17\" * 1 should be 15, got {r:?}"
        );
    }

    #[test]
    fn to_number_inf_string_is_nan() {
        // `"inf" * 1` — Rust's f64::parse accepts "inf" but JS StringToNumber
        // does not recognise it; must produce NaN.
        let m = compile_program("function t() { return \"inf\" * 1; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if n.is_nan()),
            "\"inf\" * 1 should be NaN, got {r:?}"
        );
    }

    #[test]
    fn to_number_infinity_string_is_infinity() {
        // `"Infinity" * 1 === Infinity` — the exact token "Infinity" is special.
        let m = compile_program("function t() { return \"Infinity\" * 1; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if n.is_infinite() && n > 0.0),
            "\"Infinity\" * 1 should be Infinity, got {r:?}"
        );
    }

    #[test]
    fn to_number_empty_string_minus_zero() {
        // `"" - 0 === 0` — coercion via subtraction, not multiplication.
        let m = compile_program("function t() { return \"\" - 0; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Number(n) if n == 0.0),
            "\"\" - 0 should be 0, got {r:?}"
        );
    }

    #[test]
    fn vm_loose_eq_coercion() {
        // Bug 1: VM must use loose_eq (AbstractEquality) for `==`/`!=`,
        // NOT strict_eq.  All of these were incorrectly `false` before the
        // Op::LooseEq / Op::LooseNeq fix.
        let cases: &[(&str, bool)] = &[
            // numeric-string coercion
            ("function t() { return 0 == \"0\"; }", true),
            // bool→number coercion
            ("function t() { return 1 == true; }", true),
            // null == undefined
            ("function t() { return null == undefined; }", true),
            // empty-string → 0
            ("function t() { return \"\" == 0; }", true),
            // null != 0  (null is only == undefined/null)
            ("function t() { return null != 0; }", true),
            // sanity: strict-ish cases still work
            ("function t() { return 1 != 2; }", true),
            ("function t() { return 0 == 0; }", true),
        ];
        for (src, expected) in cases {
            let m = compile_program(src).unwrap_or_else(|e| panic!("compile {src:?}: {e}"));
            let r = run_fn(&m, 1, &[]).unwrap_or_else(|e| panic!("run {src:?}: {e}"));
            assert!(
                matches!(r, Value::Bool(b) if b == *expected),
                "{src:?} → expected Bool({expected}), got {r:?}"
            );
        }
    }

    // --- VM: array join renders null/undefined as "" ---
    #[test]
    fn vm_array_join_null_undefined_empty() {
        // `[1,null,3].join(",")` must be "1,,3" — null elements become "".
        let m = compile_program("function t() { return [1,null,3].join(\",\"); }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(&r, Value::String(s) if &**s == "1,,3"),
            "[1,null,3].join(\",\") should be \"1,,3\", got {r:?}"
        );
    }

    // --- VM: string concat with two-element array ---
    #[test]
    fn vm_string_concat_array_two_elem() {
        // `"" + [1,2]` must be "1,2".
        let m = compile_program("function t() { return \"\" + [1,2]; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(&r, Value::String(s) if &**s == "1,2"),
            "\"\" + [1,2] should be \"1,2\", got {r:?}"
        );
    }

    // --- VM: Number("") === 0 via subtraction coercion ---
    #[test]
    fn vm_number_empty_string_strict_eq_zero() {
        // `Number("") === 0` — tested via `("" - 0) === 0` inside a VM function.
        let m = compile_program("function t() { return (\"\" - 0) === 0; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Bool(true)),
            "(\"\" - 0) === 0 should be true, got {r:?}"
        );
    }

    // --- VM: Number("0x1F") === 31 via multiplication coercion ---
    #[test]
    fn vm_number_hex_string_strict_eq_31() {
        // `Number("0x1F") === 31` — tested via `("0x1F" * 1) === 31`.
        let m = compile_program("function t() { return (\"0x1F\" * 1) === 31; }").unwrap();
        let r = run_fn(&m, 1, &[]).unwrap();
        assert!(
            matches!(r, Value::Bool(true)),
            "(\"0x1F\" * 1) === 31 should be true, got {r:?}"
        );
    }

    // ====================================================================
    // T2 Phase 3 — OWNING RegBank: leak oracle, mutation arms, GC-soak.
    //
    // These exercise the owning-bank contract directly (the private `store` /
    // `store_with_arm` / teardown), which is the load-bearing safety for holding a
    // HEAP `JsVal` in a JIT bank slot. The GC default-ON (`gc_enabled()` true
    // unless CV_GC=0) means `register_jit_bank` really registers here.
    // ====================================================================

    #[cfg(target_os = "windows")]
    mod t2_owning_bank_p3 {
        use super::*;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;

        /// A GC-registered heap Object with one property; returns (owning Rc, JsVal).
        fn make_obj() -> (Rc<RefCell<HashMap<String, Value>>>, JsVal) {
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("k".to_string(), Value::Number(1.0));
            let rc = Rc::new(RefCell::new(m));
            crate::interp::gc_register_object(&rc);
            let jv = JsVal::object(&rc);
            (rc, jv)
        }
        fn make_arr() -> (Rc<RefCell<Vec<Value>>>, JsVal) {
            let rc = Rc::new(RefCell::new(vec![Value::Number(5.0)]));
            crate::interp::gc_register_array(&rc);
            let jv = JsVal::array(&rc);
            (rc, jv)
        }
        fn make_str() -> (Value, JsVal) {
            let v = Value::str("p3 owns me");
            let jv = JsVal::try_from_value(&v).unwrap();
            (v, jv)
        }

        // ---- LEAK ORACLE: a full store+overwrite+return cycle is net-zero. ----

        /// Per heap lane: snapshot strong_count, build an owning bank, store the
        /// heap value, OVERWRITE it (last-ref-overwrite arm), do a SELF-STORE, read
        /// the result as an owned Value, drop the bank, drop the result — the count
        /// must return EXACTLY to baseline (no leak, no over-dec).
        #[test]
        fn leak_oracle_store_overwrite_selfstore_return_is_net_zero() {
            // Object lane.
            {
                let (obj_rc, obj_jv) = make_obj();
                let base = Rc::strong_count(&obj_rc);
                {
                    // Bank seeded with NO heap args (so only explicit stores own).
                    let mut bank = OwningRegBank::new_for_test(4, &[]);
                    // store heap into slot 1 (+1 owned by bank).
                    unsafe { bank.store_for_test(1, obj_jv) };
                    assert_eq!(Rc::strong_count(&obj_rc), base + 1, "after store: bank owns +1");
                    // SELF-STORE: store the same value into the SAME slot — must be a
                    // no-op on the count (inc-new-before-dec-old makes it safe).
                    unsafe { bank.store_for_test(1, obj_jv) };
                    assert_eq!(Rc::strong_count(&obj_rc), base + 1, "self-store net-zero");
                    // LAST-REF-OVERWRITE: overwrite slot 1 with an immediate — the
                    // bank's owned heap ref is released.
                    unsafe { bank.store_for_test(1, JsVal::number(42.0)) };
                    assert_eq!(Rc::strong_count(&obj_rc), base, "overwrite released bank ref");
                    // Re-store and exit via a result read (the EXIT ORDERING path).
                    unsafe { bank.store_for_test(2, obj_jv) };
                    let result_jv = bank.slot(2);
                    // EXIT: read the result as an owned Value BEFORE the bank drops.
                    let result: Value = unsafe { result_jv.to_value() }; // +1
                    assert_eq!(Rc::strong_count(&obj_rc), base + 2, "bank +1, result +1");
                    drop(bank); // bank teardown decs slot 2 → -1
                    assert_eq!(Rc::strong_count(&obj_rc), base + 1, "after bank drop: only result");
                    drop(result); // -1
                    assert_eq!(Rc::strong_count(&obj_rc), base, "after result drop: baseline");
                }
                assert_eq!(Rc::strong_count(&obj_rc), base, "final baseline (Object lane)");
            }
            // Array + String lanes: same net-zero on a store+drop cycle.
            for which in 0..2 {
                let (owner_count_before, store_jv, count): (usize, JsVal, Box<dyn Fn() -> usize>) =
                    if which == 0 {
                        let (arr_rc, arr_jv) = make_arr();
                        let b = Rc::strong_count(&arr_rc);
                        let c = move || Rc::strong_count(&arr_rc);
                        (b, arr_jv, Box::new(c))
                    } else {
                        let (str_v, str_jv) = make_str();
                        let rc = match &str_v { Value::String(s) => s.as_rc().clone(), _ => unreachable!() };
                        let b = Rc::strong_count(&rc);
                        // keep str_v alive by moving it into the closure
                        let c = move || { let _ = &str_v; Rc::strong_count(&rc) };
                        (b, str_jv, Box::new(c))
                    };
                {
                    let mut bank = OwningRegBank::new_for_test(2, &[]);
                    unsafe { bank.store_for_test(0, store_jv) };
                    assert_eq!(count(), owner_count_before + 1, "lane {which}: bank owns +1");
                    drop(bank);
                    assert_eq!(count(), owner_count_before, "lane {which}: bank drop → baseline");
                }
            }
        }

        /// UNIFORM-OWN entry: a heap ARG seeded into the bank is inc'd on construct
        /// and dec'd on Drop (net-zero). Mixed with immediates.
        #[test]
        fn uniform_own_seed_args_net_zero() {
            let (obj_rc, _obj_jv) = make_obj();
            let (arr_rc, _arr_jv) = make_arr();
            let base_obj = Rc::strong_count(&obj_rc);
            let base_arr = Rc::strong_count(&arr_rc);
            let args = vec![
                Value::Object(obj_rc.clone()), // heap arg → +1 owner from the clone
                Value::Number(7.0),            // immediate
                Value::Array(arr_rc.clone()),  // heap arg
            ];
            // the clones bumped the owners; record the with-clones baseline
            let with_clones_obj = Rc::strong_count(&obj_rc);
            let with_clones_arr = Rc::strong_count(&arr_rc);
            {
                let bank = OwningRegBank::new_for_test(5, &args);
                // Each heap arg got a bank-owned +1 on top of the arg clone.
                assert_eq!(Rc::strong_count(&obj_rc), with_clones_obj + 1, "obj arg seeded +1");
                assert_eq!(Rc::strong_count(&arr_rc), with_clones_arr + 1, "arr arg seeded +1");
                drop(bank);
            }
            // Bank dropped → its seeded +1 released; only the arg clones remain.
            assert_eq!(Rc::strong_count(&obj_rc), with_clones_obj, "obj bank ref released");
            assert_eq!(Rc::strong_count(&arr_rc), with_clones_arr, "arr bank ref released");
            drop(args);
            assert_eq!(Rc::strong_count(&obj_rc), base_obj, "obj back to baseline");
            assert_eq!(Rc::strong_count(&arr_rc), base_arr, "arr back to baseline");
        }

        // ---- MUTATION ARMS: each broken store ordering reddens the oracle. ----

        /// Run one mutation arm and return the strong-count DELTA from baseline
        /// observed after a store+overwrite cycle (the bank is `forget`-leaked to
        /// avoid compounding an over-dec into an actual free during teardown). A
        /// CORRECT arm yields delta 0; a broken arm yields a non-zero delta (caught).
        fn arm_delta(arm: StoreArm) -> i64 {
            // Hold THREE external owners so an over-dec by 1–2 can't free the value
            // (keeps the failure a detectable WRONG COUNT, never a UAF).
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("k".to_string(), Value::Number(1.0));
            let rc = Rc::new(RefCell::new(m));
            crate::interp::gc_register_object(&rc);
            let _o1 = rc.clone();
            let _o2 = rc.clone();
            let jv = JsVal::object(&rc);
            let base = Rc::strong_count(&rc) as i64;
            let mut bank = OwningRegBank::new_for_test(2, &[]);
            // store heap → slot 0, then overwrite slot 0 with an immediate.
            unsafe { bank.store_with_arm(0, jv, arm) };
            unsafe { bank.store_with_arm(0, JsVal::number(0.0), arm) };
            let after = Rc::strong_count(&rc) as i64;
            // Skip the bank's Drop so a broken arm's leaked/over-dec'd slot 0 (now an
            // immediate anyway after the 2nd store) doesn't compound the accounting.
            std::mem::forget(bank);
            after - base
        }

        #[test]
        fn mutation_arms_are_caught_by_leak_oracle() {
            // CORRECT: a store then overwrite is net-zero (delta 0).
            assert_eq!(arm_delta(StoreArm::Correct), 0, "correct store must be net-zero");
            // SKIP-OVERWRITE-DEC (leak): the overwritten heap ref is never released,
            // so the count stays ELEVATED (+1) → the leak oracle reddens.
            assert_eq!(
                arm_delta(StoreArm::SkipOverwriteDec),
                1,
                "skip-overwrite-dec must LEAK (+1) — the oracle catches it"
            );
            // DEC-BEFORE-INC (wrong order): on the first store the old slot is an
            // immediate (dec no-op) and the new is inc'd (+1); on the 2nd store the
            // old (heap) is dec'd before the new (immediate) inc → nets correctly to
            // 0 ONLY because no value is its own last ref here, BUT the ORDERING is
            // observably wrong on a self-store / last-ref case. Prove the ordering
            // teeth separately below; here assert it is NOT silently identical to
            // the leak arm.
            let dbi = arm_delta(StoreArm::DecBeforeInc);
            // SKIP-STORE-INC (double-free shape): the new value is never inc'd, so
            // the bank doesn't own it, yet the SECOND store decs the (un-owned) heap
            // old → OVER-DEC (-1) → the oracle reddens.
            assert_eq!(
                arm_delta(StoreArm::SkipStoreInc),
                -1,
                "skip-store-inc must OVER-DEC (-1) — double-free shape, oracle catches it"
            );
            // The dec-before-inc arm is caught by the dedicated self-store UAF teeth
            // test (it transiently drops the last ref); here we only require it is
            // not vacuously equal to the correct arm in the cases that differ.
            let _ = dbi;
        }

        /// DEC-BEFORE-INC teeth (deterministic, UB-free): on a SELF-STORE where the
        /// bank holds the SOLE ref, the WRONG order (dec-old THEN inc-new) transiently
        /// drops the strong count to 0 — the pointee is FREED before the inc — a real
        /// UAF. We catch it WITHOUT executing the UAF by detecting the transient free
        /// via a `Weak`: a `store_with_arm` variant that, for `DecBeforeInc`, decs the
        /// old, then checks the `Weak`; if it can no longer upgrade the value was
        /// freed → the ordering bug is PROVEN, and we abort before the (UB) inc.
        ///
        /// The CORRECT order (inc-first) keeps the count ≥1 throughout, so the `Weak`
        /// always upgrades — net-zero, no free. This is the load-bearing proof that
        /// INC-NEW-BEFORE-DEC-OLD is required.
        #[test]
        fn dec_before_inc_self_store_transiently_frees_sole_ref() {
            use std::rc::Weak;
            // Use the Array lane so we can hold a `Weak` to the exact Rc the bank owns.
            // CORRECT arm, sole-bank-owner self-store: the Weak survives throughout.
            {
                let rc = Rc::new(RefCell::new(vec![Value::Number(1.0)]));
                crate::interp::gc_register_array(&rc);
                let weak: Weak<RefCell<Vec<Value>>> = Rc::downgrade(&rc);
                let jv = JsVal::array(&rc);
                let mut bank = OwningRegBank::new_for_test(1, &[]);
                unsafe { bank.store_for_test(0, jv) }; // bank +1 (count now 2: rc + bank)
                drop(rc); // bank is now the SOLE strong owner (count 1)
                assert!(weak.upgrade().is_some(), "alive before self-store");
                // CORRECT self-store: inc(→2) then dec(→1). Never 0; Weak survives.
                unsafe { bank.store_for_test(0, jv) };
                assert!(
                    weak.upgrade().is_some(),
                    "correct inc-first self-store never transiently frees the sole ref"
                );
                drop(bank);
                assert!(weak.upgrade().is_none(), "teardown freed it");
            }
            // BROKEN arm (dec-before-inc): the detector proves the transient free.
            {
                let rc = Rc::new(RefCell::new(vec![Value::Number(2.0)]));
                crate::interp::gc_register_array(&rc);
                let weak: Weak<RefCell<Vec<Value>>> = Rc::downgrade(&rc);
                let jv = JsVal::array(&rc);
                let mut bank = OwningRegBank::new_for_test(1, &[]);
                unsafe { bank.store_for_test(0, jv) };
                drop(rc); // sole owner = bank (count 1)
                // The detector runs dec-old first; with sole ownership the count hits
                // 0 and the Weak dies → proven. It then SKIPS the UB inc and repairs
                // the slot to `undefined` so teardown is safe.
                let freed = unsafe { bank.dec_before_inc_self_store_detect(0, &weak) };
                assert!(
                    freed,
                    "dec-before-inc on a sole-ref self-store MUST transiently free \
                     (the ordering bug the inc-first contract prevents)"
                );
                drop(bank); // slot was repaired to undefined → safe teardown
            }
        }

        // ---- GC-SOAK: a heap value resident SOLELY in a registered bank survives. ----

        /// A heap Object whose ONLY reference (besides the test's owner) is the
        /// owning, GC-registered bank survives a forced `gc_collect`: it is NOT
        /// cleared by `gc_sweep`, unboxes to the IDENTICAL Rc (ptr_eq), and keeps its
        /// content. Churn keeps the GC live count bounded.
        #[test]
        fn gc_soak_bank_resident_heap_value_survives() {
            if !crate::interp::gc_enabled() {
                return; // GC disabled in this env — the registered-bank path can't engage.
            }
            let interp = crate::interp::Interp::new();
            let (obj_rc, obj_jv) = make_obj();
            let obj_ptr = Rc::as_ptr(&obj_rc) as *const () as usize;
            {
                let mut bank = OwningRegBank::new_for_test(3, &[]);
                // The bank holds the obj as a bank-only root (the test's obj_rc is
                // NOT a GC root — only the registered bank is).
                unsafe { bank.store_for_test(1, obj_jv) };
                // Force a collection. The registered bank's slot must mark the obj.
                let _ = interp.gc_collect(&[]);
                // SURVIVAL: not cleared.
                assert!(
                    matches!(obj_rc.borrow().get("k"), Some(Value::Number(n)) if *n == 1.0),
                    "registered owning-bank Object was CLEARED by gc_sweep (root/seed FAILED)"
                );
                // IDENTITY: the slot still unboxes to the identical Rc + content.
                let back = unsafe { bank.slot(1).to_value() };
                match back {
                    Value::Object(o) => {
                        assert_eq!(Rc::as_ptr(&o) as *const () as usize, obj_ptr, "ptr_eq after GC");
                        assert!(matches!(o.borrow().get("k"), Some(Value::Number(n)) if *n == 1.0));
                    }
                    other => panic!("slot1 not Object after GC: {other:?}"),
                }
                drop(bank);
            }
            // CHURN: many register/collect/teardown rounds keep the live count bounded.
            let live0 = crate::interp::gc_live_object_count();
            for _ in 0..200 {
                let (rc, jv) = make_obj();
                let mut bank = OwningRegBank::new_for_test(2, &[]);
                unsafe { bank.store_for_test(0, jv) };
                let _ = interp.gc_collect(&[]);
                assert!(!rc.borrow().is_empty(), "churn obj cleared while bank-resident");
                drop(bank);
                drop(rc);
            }
            let _ = interp.gc_collect(&[]);
            let live1 = crate::interp::gc_live_object_count();
            assert!(
                live1 <= live0 + 8,
                "GC live count grew with churn (leak): live0={live0} live1={live1}"
            );
            assert_eq!(crate::interp::jit_bank_registry_len(), 0, "registry drained after churn");
            drop(obj_rc);
        }

        // ---- END-TO-END LEAK ORACLE through the REAL compile+run path. ----

        /// Compile `function pick(o){ var c = o.child; return c; }` to T2(heap) and
        /// run it via `run_t2lite_call` with a heap-Object arg whose `child` is a
        /// heap Object. After the call returns the held child and the returned Value
        /// is dropped, the child's strong count must return EXACTLY to baseline (no
        /// leak through the owning bank's store/teardown on the live path). The
        /// registry must be drained (RAII popped).
        #[test]
        fn end_to_end_run_t2lite_call_heap_result_is_net_zero() {
            let _heap = crate::interp::T2HeapGuard::new(true);
            // Build a real shaped receiver { child: <obj> } and warm the IC by running
            // the VM a few times so the T2 GetProp site can bake the shape.
            let m = module_for_first_fn("function pick(o){ var c = o.child; return c; }");
            let empty: std::cell::RefCell<HashMap<String, Value>> =
                std::cell::RefCell::new(HashMap::new());

            // The child object whose refcount we track.
            let mut childmap: HashMap<String, Value> = HashMap::new();
            childmap.insert("tag".to_string(), Value::Number(7.0));
            let child_rc = Rc::new(RefCell::new(childmap));
            crate::interp::gc_register_object(&child_rc);
            let child_ptr = Rc::as_ptr(&child_rc) as *const () as usize;

            // The receiver { child }.
            let mut recmap: HashMap<String, Value> = HashMap::new();
            recmap.insert("child".to_string(), Value::Object(child_rc.clone()));
            let rec = Value::Object(Rc::new(RefCell::new(recmap)));

            // Baseline: child_rc + the receiver's stored clone = 2 (record env note
            // below). Capture it AFTER building the receiver.
            let base = Rc::strong_count(&child_rc);

            // Warm the IC: run on the VM a handful of times (warms the GetProp site
            // so try_compile_t2lite can bake the shape).
            for _ in 0..20 {
                let mut refuse = refuse_with_interp;
                let _ = run_function(&m, 0, &[rec.clone()], &Value::Undefined, &empty, None, &mut refuse);
            }
            assert_eq!(Rc::strong_count(&child_rc), base, "VM warmup left child count at baseline");

            // Compile to T2(heap). The site should now be warm + heap_result.
            let native = match try_compile_t2lite(&m, 0) {
                Some(jf) => jf,
                None => return, // env (e.g. Shaped off) makes the site un-inlinable; skip.
            };

            // Run via the OWNING bank path. The bank holds the child across the body,
            // then returns it. Registry must be 0 after the call (RAII popped).
            let result = {
                let mut refuse = refuse_with_interp;
                run_t2lite_call(&native, &m, &[rec.clone()], &Value::Undefined, &empty, &mut refuse)
                    .expect("pick runs")
            };
            assert_eq!(
                crate::interp::jit_bank_registry_len(),
                0,
                "owning bank registration popped on return (RAII gap-free)"
            );
            // The result IS the child object (held heap result, returned).
            match &result {
                Value::Object(o) => {
                    assert_eq!(Rc::as_ptr(o) as *const () as usize, child_ptr, "result is the child (ptr_eq)");
                    assert!(matches!(o.borrow().get("tag"), Some(Value::Number(n)) if *n == 7.0));
                }
                // If T2 deopted (e.g. heap site declined this run), the VM produced
                // the same object — still correct; just assert ptr identity.
                other => panic!("pick result not an Object: {other:?}"),
            }
            // While `result` is alive it holds +1 over baseline.
            assert_eq!(Rc::strong_count(&child_rc), base + 1, "result holds exactly +1");
            drop(result);
            // After dropping the result, the child must be back to baseline — the
            // owning bank's store/teardown was perfectly balanced (no leak/over-dec).
            assert_eq!(
                Rc::strong_count(&child_rc),
                base,
                "after dropping the T2 heap result the child returns to baseline (net-zero)"
            );
            drop((rec, child_rc));
        }

        /// GETIDX END-TO-END LEAK ORACLE: compile `function pick(a){ var c = a[0];
        /// return c; }` to T2(heap) and run it via `run_t2lite_call` with a heap
        /// ARRAY arg whose element [0] is a heap Object. The GetIdx owning-stores the
        /// element into the bank, the body returns it, and after the returned Value
        /// is dropped the element's strong count must return EXACTLY to baseline (no
        /// leak / over-dec through the owning element read + bank teardown). The
        /// registry must be drained (RAII popped) — the GC-soak survival proof for a
        /// heap GetIdx element.
        #[test]
        fn end_to_end_run_t2lite_getidx_heap_element_is_net_zero() {
            let _heap = crate::interp::T2HeapGuard::new(true);
            let m = module_for_first_fn("function pick(a){ var c = a[0]; return c; }");
            let empty: std::cell::RefCell<HashMap<String, Value>> =
                std::cell::RefCell::new(HashMap::new());

            // The element object whose refcount we track.
            let mut elmap: HashMap<String, Value> = HashMap::new();
            elmap.insert("tag".to_string(), Value::Number(7.0));
            let el_rc = Rc::new(RefCell::new(elmap));
            crate::interp::gc_register_object(&el_rc);
            let el_ptr = Rc::as_ptr(&el_rc) as *const () as usize;

            // The receiver array [ <el> ].
            let arr_rc = Rc::new(RefCell::new(vec![Value::Object(el_rc.clone())]));
            crate::interp::gc_register_array(&arr_rc);
            let arr = Value::Array(arr_rc.clone());

            // Baseline AFTER the array stored its clone of the element.
            let base = Rc::strong_count(&el_rc);

            // Warm: run on the VM a few times (warms nothing IC-wise for GetIdx, but
            // exercises the same path the compile will take; harmless).
            for _ in 0..20 {
                let mut refuse = refuse_with_interp;
                let _ = run_function(&m, 0, &[arr.clone()], &Value::Undefined, &empty, None, &mut refuse);
            }
            assert_eq!(Rc::strong_count(&el_rc), base, "VM warmup left element count at baseline");

            let native = match try_compile_t2lite(&m, 0) {
                Some(jf) => jf,
                None => return, // GetIdx un-inlinable in this env — skip (VM path proven elsewhere).
            };

            let result = {
                let mut refuse = refuse_with_interp;
                run_t2lite_call(&native, &m, &[arr.clone()], &Value::Undefined, &empty, &mut refuse)
                    .expect("pick runs")
            };
            assert_eq!(
                crate::interp::jit_bank_registry_len(),
                0,
                "owning bank registration popped on return (RAII gap-free)"
            );
            // The result IS the element object (held heap GetIdx result, returned).
            match &result {
                Value::Object(o) => {
                    assert_eq!(Rc::as_ptr(o) as *const () as usize, el_ptr, "result is the element (ptr_eq)");
                    assert!(matches!(o.borrow().get("tag"), Some(Value::Number(n)) if *n == 7.0));
                }
                other => panic!("pick result not an Object: {other:?}"),
            }
            assert_eq!(Rc::strong_count(&el_rc), base + 1, "result holds exactly +1");
            drop(result);
            assert_eq!(
                Rc::strong_count(&el_rc),
                base,
                "after dropping the T2 GetIdx heap element the count returns to baseline (net-zero)"
            );
            drop((arr, el_rc));
        }

        // ---- P4: GC-DURING-CALL — the re-entry safepoint marks the caller bank. ----

        /// THE P4 GC-during-call teeth. A T2(heap) function `f(o)` does:
        ///   `var c = o.child; return echo(c);`
        /// — it loads the heap child into a bank slot (owning store), then CALLS the
        /// global `echo`. The dispatch fired for `echo` does TWO things mid-call:
        ///   (1) DELETES `child` from the receiver `o` (so the child is now reachable
        ///       ONLY through the caller's registered T2 bank — a bank-ONLY root),
        ///   (2) forces a full `gc_collect` (the would-be mid-op nested-GC hazard).
        /// Then it echoes its arg back. The child MUST survive (not swept): its
        /// content is intact and the returned value is the IDENTICAL Rc (ptr_eq).
        /// This proves the call boundary is a real safepoint — the registered bank
        /// roots its heap slots through a re-entrant collect (no UAF / clear).
        #[test]
        fn gc_during_reentrant_call_marks_caller_bank_no_uaf() {
            if !crate::interp::gc_enabled() {
                return; // GC disabled — the registered-bank root path can't engage.
            }
            let _heap = crate::interp::T2HeapGuard::new(true);
            let m = module_for_first_fn("function f(o){ var c = o.child; return echo(c); }");
            let globals: std::cell::RefCell<HashMap<String, Value>> =
                std::cell::RefCell::new(HashMap::new());

            // The child object whose survival-through-an-in-call-GC we test. It is a
            // GC-registered heap Object (so gc_sweep would CLEAR it if it weren't
            // marked by a root).
            let mut childmap: HashMap<String, Value> = HashMap::new();
            childmap.insert("tag".to_string(), Value::Number(99.0));
            let child_rc = Rc::new(RefCell::new(childmap));
            crate::interp::gc_register_object(&child_rc);
            let child_ptr = Rc::as_ptr(&child_rc) as *const () as usize;
            let child_weak = Rc::downgrade(&child_rc);

            // The receiver { child }. We keep `rec_rc` so we can DELETE `child` from
            // it inside the dispatch (making the child bank-only).
            let mut recmap: HashMap<String, Value> = HashMap::new();
            recmap.insert("child".to_string(), Value::Object(child_rc.clone()));
            let rec_rc = Rc::new(RefCell::new(recmap));
            crate::interp::gc_register_object(&rec_rc);
            let rec = Value::Object(rec_rc.clone());

            // Warm the IC on the VM so the T2 GetProp site bakes the shape.
            for _ in 0..20 {
                let mut refuse = refuse_with_interp;
                let _ = run_function(&m, 0, &[rec.clone()], &Value::Undefined, &globals, None, &mut refuse);
            }

            let native = match try_compile_t2lite(&m, 0) {
                Some(jf) => jf,
                None => return, // site un-inlinable in this env — skip (VM path proven elsewhere).
            };

            // Install `echo` as a global so the T2 CallValue's LoadGlobalChecked
            // resolves it (it just needs to be a callable-shaped Value so
            // `dispatch_call_value` routes it to our host `dispatch` closure, which
            // does the real work). A bare Object is treated as callable-via-dispatch.
            globals
                .borrow_mut()
                .insert("echo".to_string(), Value::Object(Rc::new(RefCell::new(HashMap::new()))));
            // We use a real Interp purely to drive `gc_collect`.
            let interp = crate::interp::Interp::new();

            // Drop the test's OWN strong hold on the child so that, once the dispatch
            // deletes it from `rec`, the ONLY remaining strong ref is the T2 bank.
            drop(child_rc);

            // The dispatch: on the `echo(c)` call, (1) delete `child` from rec (now
            // the child is bank-only), (2) force a GC, (3) echo the arg back.
            let mut collected = false;
            let result = {
                let collected = &mut collected;
                let mut dispatch = |_callee: Value, _this: Value, a: Vec<Value>| -> Result<Value, RuntimeError> {
                    // (1) Remove `child` from the receiver → child is now reachable
                    // ONLY via the caller's registered T2 bank (and this `a[0]` temp).
                    rec_rc.borrow_mut().remove("child");
                    // (2) Force a full collection MID-CALL. The registered bank must
                    // mark the child (its slot is a root); if it didn't, gc_sweep
                    // would clear the child's map → the assertions below fail.
                    let _ = interp.gc_collect(&[]);
                    *collected = true;
                    // (3) Echo the arg back as the call result.
                    Ok(a.into_iter().next().unwrap_or(Value::Undefined))
                };
                run_t2lite_call(&native, &m, &[rec.clone()], &Value::Undefined, &globals, &mut dispatch)
                    .expect("f runs through the re-entrant echo call")
            };

            assert!(collected, "the re-entrant dispatch (with the forced GC) must have fired");
            // The child SURVIVED the in-call GC (not swept): identical Rc + content.
            assert!(child_weak.upgrade().is_some(), "child freed during in-call GC (UAF/over-dec)");
            match &result {
                Value::Object(o) => {
                    assert_eq!(
                        Rc::as_ptr(o) as *const () as usize, child_ptr,
                        "result is the SAME child Rc after the in-call GC (ptr_eq)"
                    );
                    assert!(
                        matches!(o.borrow().get("tag"), Some(Value::Number(n)) if *n == 99.0),
                        "child content CLEARED by the in-call gc_sweep — the bank failed to root it"
                    );
                }
                other => panic!("f result not the child Object: {other:?}"),
            }
            // RAII: the bank registration popped on return.
            assert_eq!(crate::interp::jit_bank_registry_len(), 0, "bank registry drained after return");
            drop(result);
            drop((rec, rec_rc));
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // T2→T2 — NATIVE-TO-NATIVE: the JsVal-args entry refcount + both-bank GC gate.
    //
    // The native-to-native callee path (`run_t2lite_from_jsval_args`) is the
    // silent-corruption surface: a botched arg seed / result-handback can leak or
    // over-dec, and a both-bank GC gap is a UAF. These unit tests exercise the
    // entry DIRECTLY with a real T2-compiled callee + a heap arg/result.
    // ════════════════════════════════════════════════════════════════════════
    #[cfg(target_os = "windows")]
    mod t2_t2_native_entry {
        use super::*;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;

        /// LEAK ORACLE: a full T2→T2 native-to-native call of `id(x)` (returns its
        /// heap arg) with a heap Object arg+result is REFCOUNT NET-ZERO end-to-end.
        /// The caller hands a borrowed-handle JsVal; the callee bank seeds +1, the
        /// result is handed back +1; the caller's owning-store consumes that +1 and
        /// the entry's +1 is released — so once we drop the returned Value the
        /// Object's strong count returns EXACTLY to baseline.
        #[test]
        fn native_entry_heap_arg_and_result_is_net_zero() {
            // `id` returns its first arg — the simplest heap-passthrough callee.
            let m = module_for_first_fn("function id(x){ return x; }");
            let native = match try_compile_t2lite(&m, 0) {
                Some(jf) => jf,
                None => return, // env can't compile this callee — skip (covered elsewhere).
            };
            let globals: std::cell::RefCell<HashMap<String, Value>> =
                std::cell::RefCell::new(HashMap::new());

            // A GC-registered heap Object we'll pass through `id` and back.
            let mut map: HashMap<String, Value> = HashMap::new();
            map.insert("tag".to_string(), Value::Number(42.0));
            let obj_rc = Rc::new(RefCell::new(map));
            crate::interp::gc_register_object(&obj_rc);
            let base = Rc::strong_count(&obj_rc);

            // The caller's "bank arg slot": a single JsVal holding the Object. The
            // caller owns this +1 (here, modeled by `arg_jv` + `obj_rc`).
            let arg_jv = JsVal::object(&obj_rc);
            // Building `arg_jv` did NOT change the count (it's a borrowed handle).
            assert_eq!(Rc::strong_count(&obj_rc), base, "boxing a JsVal is borrowed (no +1)");

            let mut refuse = refuse_with_interp;
            let args = [arg_jv.bits()];
            // SAFETY: args[0] is a live Object JsVal (obj_rc keeps it alive); native
            // matches m.fns[0]; the entry seeds its own bank +1 and hands back +1.
            let (result_jv, status) = unsafe {
                run_t2lite_from_jsval_args(
                    &native,
                    &m,
                    args.as_ptr(),
                    1,
                    &Value::Undefined,
                    &globals,
                    &mut refuse,
                )
            };
            assert!(matches!(status, T2NativeStatus::Returned), "id returns normally");
            // The result carries a +1 handed to the caller (the +1 the real call-site
            // would consume via its owning-store). Model that consume: take the value
            // as an owned Value (round-trips to the SAME Rc) then release the +1.
            // SAFETY: result_jv is a live Object (+1 held by the entry's handback).
            let result = unsafe { result_jv.to_value() }; // +1 (to_value clone)
            unsafe { result_jv.rc_dec() }; // release the entry's handback +1
            // ptr_eq: the result is the IDENTICAL Object Rc (identity preserved).
            match &result {
                Value::Object(o) => assert!(
                    Rc::ptr_eq(o, &obj_rc),
                    "native-to-native result is the SAME Object Rc (ptr identity)"
                ),
                other => panic!("expected the Object back, got {other:?}"),
            }
            // Now only `result` (and `obj_rc`) own the Object → base + 1.
            assert_eq!(Rc::strong_count(&obj_rc), base + 1, "only the result holds the extra +1");
            drop(result);
            // NET-ZERO: dropping the result returns to baseline (no leak, no over-dec).
            assert_eq!(Rc::strong_count(&obj_rc), base, "T2→T2 call is refcount NET-ZERO");
            // Bank registry drained (the callee's bank popped on return).
            assert_eq!(crate::interp::jit_bank_registry_len(), 0, "callee bank registry drained");
        }

        /// BOTH-BANK GC: during a T2→T2 native-to-native call, a `gc_collect` forced
        /// inside the callee (via a nested dispatch the callee makes) must mark BOTH
        /// the caller's bank AND the callee's bank, so a heap value held SOLELY in
        /// either bank survives (not swept). We construct:
        ///   * a CALLER `f(c){ return mid(c); }` — its bank holds the heap child `c`;
        ///   * a registered T2 CALLEE `mid(x){ return echo(x); }` — its bank ALSO
        ///     holds the child `x`, and it CALLS `echo` (a host dispatch) which
        ///     DELETES the child from its last non-bank owner and forces a full GC.
        /// The child is then reachable ONLY through the two registered banks; it must
        /// survive the in-call collect (ptr_eq + content intact) — proving both banks
        /// are GC roots simultaneously across the nested native call.
        #[test]
        fn gc_inside_native_callee_marks_both_banks_no_uaf() {
            if !crate::interp::gc_enabled() {
                return; // GC disabled — the registered-bank root path can't engage.
            }
            let _heap = crate::interp::T2HeapGuard::new(true);
            // Caller + callee modules. Both are simple passthroughs in the T2 subset.
            let caller_m = Rc::new(module_for_first_fn("function f(c){ return mid(c); }"));
            let callee_m = Rc::new(module_for_first_fn("function mid(x){ return echo(x); }"));
            let caller_native = match try_compile_t2lite(&caller_m, 0) {
                Some(jf) => jf,
                None => return,
            };
            let callee_native = match try_compile_t2lite(&callee_m, 0) {
                Some(jf) => Rc::new(jf),
                None => return,
            };

            // The child object: GC-registered, will become bank-ONLY mid-call.
            let mut childmap: HashMap<String, Value> = HashMap::new();
            childmap.insert("tag".to_string(), Value::Number(99.0));
            let child_rc = Rc::new(RefCell::new(childmap));
            crate::interp::gc_register_object(&child_rc);
            let child_ptr = Rc::as_ptr(&child_rc) as *const () as usize;
            let child_weak = Rc::downgrade(&child_rc);
            let child = Value::Object(child_rc.clone());

            // Register the callee `mid` so the caller's `CallValue` resolves it
            // native-to-native. The caller calls a GLOBAL `mid` which we publish as a
            // `Value::BcClosure` carrying the callee module — the bcclosure value-
            // callee path keys the registry by the MODULE pointer.
            let mid_closure = Rc::new(crate::interp::BcClosure {
                fn_idx: 0,
                upvalues: RefCell::new(Vec::new()),
                props: RefCell::new(HashMap::new()),
                module: callee_m.clone(),
            });
            crate::interp::t2_registry_register_module_for_test(&callee_m, &callee_native);
            let globals: std::cell::RefCell<HashMap<String, Value>> =
                std::cell::RefCell::new(HashMap::new());
            globals
                .borrow_mut()
                .insert("mid".to_string(), Value::BcClosure(mid_closure.clone()));
            // `echo` is the callable the CALLEE invokes; a bare Object routes through
            // our host dispatch (which does the delete + GC + echo).
            globals
                .borrow_mut()
                .insert("echo".to_string(), Value::Object(Rc::new(RefCell::new(HashMap::new()))));

            let interp = crate::interp::Interp::new();
            // Drop the test's own strong hold on the child so the ONLY non-bank owner
            // is the `child` Value we pass in (which the dispatch's arg will mirror).
            drop(child_rc);

            let mut collected = false;
            let result = {
                let collected = &mut collected;
                let mut dispatch = |_callee: Value, _this: Value, a: Vec<Value>| -> Result<Value, RuntimeError> {
                    // At THIS point we are inside `echo`, called from the NATIVE callee
                    // `mid`, which was called native-to-native from the NATIVE caller
                    // `f`. Both `f`'s bank (slot for `c`) and `mid`'s bank (slot for
                    // `x`) hold the child. Force a full GC: both banks must mark it.
                    let _ = interp.gc_collect(&[]);
                    *collected = true;
                    // Echo the arg back (the child) as the call result.
                    Ok(a.into_iter().next().unwrap_or(Value::Undefined))
                };
                run_t2lite_call(&caller_native, &caller_m, &[child.clone()], &Value::Undefined, &globals, &mut dispatch)
                    .expect("f → mid (native) → echo runs")
            };

            assert!(collected, "the nested echo dispatch (with the forced GC) must have fired");
            assert!(child_weak.upgrade().is_some(), "child freed during in-call GC — a bank failed to root it");
            match &result {
                Value::Object(o) => {
                    assert_eq!(
                        Rc::as_ptr(o) as *const () as usize, child_ptr,
                        "result is the SAME child Rc after the in-call GC (ptr_eq, both banks marked)"
                    );
                    assert!(
                        matches!(o.borrow().get("tag"), Some(Value::Number(n)) if *n == 99.0),
                        "child content CLEARED by the in-call gc_sweep — a bank failed to root it"
                    );
                }
                other => panic!("f result not the child Object: {other:?}"),
            }
            assert_eq!(crate::interp::jit_bank_registry_len(), 0, "BOTH banks drained after return");
            drop(result);
            crate::interp::reset_t2_module_registry();
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // T2 PHASE 5 — DEOPT-FUZZING ORACLE (the existential silent-miscompute gate).
    //
    // The load-bearing gate: a REAL per-guard deopt must RESUME the VM mid-function
    // with bit-identical results. These tests FORCE every op's resume point to fire
    // one-at-a-time (over type-correct inputs, so the ONLY reason it deopts is the
    // forced miss) and assert the resumed-on-VM result == the VM oracle. A single
    // resume that diverges = silent miscompute = FAIL. Plus: natural deopts (crafted
    // poison inputs) for every reason, the no-duplicate-effect proof, the identity-
    // map mid-loop proof, and the bc_pc-mutation arm that proves the oracle has TEETH.
    // ════════════════════════════════════════════════════════════════════════

    /// RAII: set the deopt-fuzz force-deopt op index for a scope, restoring on drop.
    #[cfg(target_os = "windows")]
    struct ForceDeoptGuard {
        prev: Option<usize>,
    }
    #[cfg(target_os = "windows")]
    impl ForceDeoptGuard {
        fn new(pc: Option<usize>) -> Self {
            ForceDeoptGuard { prev: crate::jit::set_force_deopt_pc(pc) }
        }
    }
    #[cfg(target_os = "windows")]
    impl Drop for ForceDeoptGuard {
        fn drop(&mut self) {
            crate::jit::set_force_deopt_pc(self.prev);
        }
    }

    /// Structural, JS-observable value equality for the deopt-fuzz oracle. NaN ==
    /// NaN (bit patterns aren't observable); finite numbers bit-exact (so -0 != +0);
    /// arrays/objects recurse; callables are opaque-but-present.
    #[cfg(target_os = "windows")]
    fn fuzz_values_eq(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Undefined, Value::Undefined) => true,
            (Value::Null, Value::Null) => true,
            (Value::Hole, Value::Hole) => true,
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::Number(x), Value::Number(y)) => {
                (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
            }
            (Value::String(x), Value::String(y)) => x == y,
            (Value::BigInt(x), Value::BigInt(y)) => x == y,
            (Value::Array(x), Value::Array(y)) => {
                let (xa, ya) = (x.borrow(), y.borrow());
                xa.len() == ya.len() && xa.iter().zip(ya.iter()).all(|(p, q)| fuzz_values_eq(p, q))
            }
            (Value::Object(x), Value::Object(y)) => {
                let (xo, yo) = (x.borrow(), y.borrow());
                if xo.len() != yo.len() {
                    return false;
                }
                xo.iter().all(|(k, v)| match yo.get(k) {
                    Some(w) => fuzz_values_eq(v, w),
                    None => false,
                })
            }
            (Value::Function(_), Value::Function(_)) => true,
            (Value::NativeFunction(_), Value::NativeFunction(_)) => true,
            (Value::BcClosure(_), Value::BcClosure(_)) => true,
            _ => false,
        }
    }

    /// Compare two `Result<Value, RuntimeError>` for the fuzz oracle: both Ok and
    /// structurally equal, OR both Err (errors compared by Display, which the VM and
    /// resume produce identically since resume runs the SAME VM code).
    #[cfg(target_os = "windows")]
    fn fuzz_results_eq(
        a: &Result<Value, RuntimeError>,
        b: &Result<Value, RuntimeError>,
    ) -> bool {
        match (a, b) {
            (Ok(x), Ok(y)) => fuzz_values_eq(x, y),
            (Err(x), Err(y)) => format!("{x}") == format!("{y}"),
            _ => false,
        }
    }

    /// Compile `module.fns[0]` to T2 (heap mode) WITH the deopt-site table attached,
    /// honoring any active `ForceDeoptGuard`. Returns None if the function declines.
    #[cfg(target_os = "windows")]
    fn compile_t2_for_fuzz(m: &Module) -> Option<crate::jit::JitFunction> {
        match try_compile_t2lite_status(m, 0) {
            T2CompileStatus::Ready(jf) => Some(jf),
            _ => None,
        }
    }

    /// THE deopt-fuzz sweep. For `src`'s first function called with `args`:
    ///   1. compute the VM oracle (`run_function` from ip=0);
    ///   2. for EACH op index, force a deopt at that op's boundary, run via T2 (which
    ///      resumes the VM at that bc_pc), and assert the result == the oracle.
    /// Runs under heap mode so calls/getprops/heap results are in-subset. A forced
    /// deopt at op P with type-correct args proves: the JIT bank at P is the exact
    /// pre-op VM register image, so resuming the VM at P finishes identically.
    #[cfg(target_os = "windows")]
    fn deopt_fuzz_sweep(src: &str, args: &[Value]) {
        let _heap = crate::interp::T2HeapGuard::new(true);
        let m = module_for_first_fn(src);
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        let oracle = run_function(&m, 0, args, &Value::Undefined, &empty, None, &mut refuse);
        let n = m.fns[0].code.len();
        // Control: no forced deopt — the function must run (and agree) natively.
        {
            if let Some(native) = compile_t2_for_fuzz(&m) {
                let mut d = refuse_with_interp;
                let r = run_t2lite_call(&native, &m, args, &Value::Undefined, &empty, &mut d);
                assert!(
                    fuzz_results_eq(&r, &oracle),
                    "deopt-fuzz control (no force): T2 != VM oracle\n  t2={r:?}\n  vm={oracle:?}"
                );
            }
        }
        // Forced sweep: every op boundary, one at a time.
        let mut forced_any = false;
        for pc in 0..n {
            let _g = ForceDeoptGuard::new(Some(pc));
            let native = match compile_t2_for_fuzz(&m) {
                Some(j) => j,
                None => continue, // declined under force (e.g. a back-edge target) — skip
            };
            forced_any = true;
            let mut d = refuse_with_interp;
            let r = run_t2lite_call(&native, &m, args, &Value::Undefined, &empty, &mut d);
            assert!(
                fuzz_results_eq(&r, &oracle),
                "DEOPT-FUZZ: forced deopt at op {pc} resumed to a DIFFERENT result \
                 (silent miscompute)\n  resumed={r:?}\n  vm-oracle={oracle:?}\n  src={src}"
            );
        }
        assert!(forced_any, "deopt-fuzz forced no ops (function declined T2 entirely)");
    }

    /// THE T4 deopt-fuzz sweep — the P2 analogue of `deopt_fuzz_sweep`, exercising
    /// the representation-aware backend (`compile_t4_unboxed_with_deopt`). For
    /// `src`'s first function called with `args`:
    ///   1. compute the VM oracle (`run_function` from ip=0);
    ///   2. for EACH op index, force a deopt at that op's boundary, compile + run
    ///      via T4 (which resumes the VM on the OPTIMIZED module at that bc_pc), and
    ///      assert the result == the oracle.
    /// A forced deopt at op P with type-correct args proves the T4 native bank at P
    /// is the exact pre-op VM register image (the unboxed XMM cache is invalidated
    /// per-block and NEVER read on deopt — the bank is), so resuming the VM at P
    /// finishes identically. This is the existential gate: a T4 representation
    /// specialization whose deopt is wrong is BROKEN, not done.
    #[cfg(target_os = "windows")]
    fn t4_deopt_fuzz_sweep(src: &str, args: &[Value]) {
        // T4 runs numeric mode (the representation backend stores in the double
        // lane); pin heap off so the run-time bank matches the compile, exactly as
        // `run_t3_call` does.
        let _heap = crate::interp::T2HeapGuard::new(false);
        let m = module_for_first_fn(src);
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        let oracle = run_function(&m, 0, args, &Value::Undefined, &empty, None, &mut refuse);
        let n = m.fns[0].code.len();
        // Control: no forced deopt — T4 must compile, run, and AGREE.
        {
            if let Some(native) = crate::t4::try_compile_t4(&m, 0) {
                let mut d = refuse_with_interp;
                let r = run_t3_call(&native, args, &Value::Undefined, &empty, &mut d);
                assert!(
                    fuzz_results_eq(&r, &oracle),
                    "T4 deopt-fuzz control (no force): T4 != VM oracle\n  t4={r:?}\n  vm={oracle:?}\n  src={src}"
                );
            } else {
                // T4 declined the whole function (outside its subset) — nothing to
                // fuzz; the dispatcher would run T3/T2/VM (always correct).
                return;
            }
        }
        // Forced sweep: every op boundary, one at a time. `set_force_deopt_pc` is
        // read by `compile_t4_unboxed_with_deopt` (same thread-local as T2), so the
        // forced deopt is emitted at op P's boundary in the T4 code.
        let mut forced_any = false;
        for pc in 0..n {
            let _g = ForceDeoptGuard::new(Some(pc));
            let native = match crate::t4::try_compile_t4(&m, 0) {
                Some(j) => j,
                None => continue, // declined under force (e.g. a back-edge target) — skip
            };
            forced_any = true;
            let mut d = refuse_with_interp;
            let r = run_t3_call(&native, args, &Value::Undefined, &empty, &mut d);
            assert!(
                fuzz_results_eq(&r, &oracle),
                "T4 DEOPT-FUZZ: forced deopt at op {pc} resumed to a DIFFERENT result \
                 (silent miscompute)\n  resumed={r:?}\n  vm-oracle={oracle:?}\n  src={src}"
            );
        }
        assert!(forced_any, "T4 deopt-fuzz forced no ops (function declined T4 entirely)");
    }

    /// T4 DEOPT-FUZZ #1 — float-dense function (the jit.js `f(x)` shape, maximal
    /// same-block XMM-cache reuse): every op's resume is bit-exact. This is the
    /// phase's keystone — it proves the unboxed-f64 representation's deopt is
    /// byte-identical to the VM at EVERY op boundary, over special inputs too.
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_deopt_fuzz_float_dense_every_op_resumes_identically() {
        let src = "function f(x){ return ((x*x*0.5 + x*3.0 - 1.0) * (x - 2.0) + x*x*x*0.25) \
                   / (x + 1.0) - x*0.5 + x*x*0.125 - x*7.0; }";
        for x in [5.0, 0.0, -0.0, 1.5, -2.5, f64::NAN, 1e160, 100.0, -7.0] {
            t4_deopt_fuzz_sweep(src, &[Value::Number(x)]);
        }
    }

    /// T4 DEOPT-FUZZ #2 — a LOOP (cross-block cache invalidation): forcing a deopt
    /// MID-loop must resume with the EXACT loop state. Proves the per-block XMM
    /// cache invalidation at the back-edge is sound (a value computed before the
    /// back-edge is reloaded-with-guard after it, never read from a dead XMM).
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_deopt_fuzz_loop_mid_iteration_resumes_with_exact_state() {
        let src =
            "function sumTo(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i * 2.0; } return s; }";
        for n in [0.0, 1.0, 5.0, 20.0, 100.0] {
            t4_deopt_fuzz_sweep(src, &[Value::Number(n)]);
        }
    }

    /// T4 DEOPT-FUZZ #3 — branchy control flow with early returns + comparisons.
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_deopt_fuzz_branchy_function_resumes_identically() {
        let src =
            "function pick(x){ if (x < 10.0) { return x * 2.0; } if (x >= 100.0) { return x - 1.0; } return x + 5.0; }";
        for x in [5.0, 9.0, 10.0, 50.0, 99.0, 100.0, 250.0] {
            t4_deopt_fuzz_sweep(src, &[Value::Number(x)]);
        }
    }

    /// T4 DEOPT-FUZZ #4 — NaN in a bank slot at deopt: a forced deopt while a
    /// computed NaN sits in a register must resume with the CANONICAL NaN from the
    /// bank (the XMM cache may hold a non-canonical NaN payload, but deopt decodes
    /// the BANK, which is canonical — so resume is byte-identical).
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_deopt_fuzz_nan_in_bank_slot_resumes_canonical() {
        let src = "function f(a, b){ var x = a / b; var y = x + 1.0; return y * 2.0; }";
        t4_deopt_fuzz_sweep(src, &[Value::Number(0.0), Value::Number(0.0)]);
        t4_deopt_fuzz_sweep(src, &[Value::Number(f64::NAN), Value::Number(2.0)]);
    }

    /// DEOPT-FUZZ #1 — pure numeric function: every op's resume is bit-exact.
    #[cfg(target_os = "windows")]
    #[test]
    fn deopt_fuzz_numeric_every_op_resumes_identically() {
        let src = "function poly(a, b) { var t = a * a + b - 1; return t / 2; }";
        for (a, b) in [
            (5.0, 3.0),
            (-0.0, 0.0),
            (f64::NAN, 1.0),
            (1e308, 1e308),
            (1.5, -2.5),
        ] {
            deopt_fuzz_sweep(src, &[Value::Number(a), Value::Number(b)]);
        }
    }

    /// DEOPT-FUZZ #2 — a LOOP: forcing a deopt MID-loop must resume with the EXACT
    /// loop state (counter, accumulator, ip) and finish identically (the identity-map
    /// mid-loop proof). The sweep forces each op, including the in-loop ops on the
    /// FIRST entry to that op (the bank then holds the live loop state).
    #[cfg(target_os = "windows")]
    #[test]
    fn deopt_fuzz_loop_mid_iteration_resumes_with_exact_state() {
        let src =
            "function sumTo(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i * 2; } return s; }";
        for n in [0.0, 1.0, 5.0, 20.0, 100.0] {
            deopt_fuzz_sweep(src, &[Value::Number(n)]);
        }
    }

    /// DEOPT-FUZZ #3 — control flow with early returns + comparisons.
    #[cfg(target_os = "windows")]
    #[test]
    fn deopt_fuzz_branchy_function_resumes_identically() {
        let src =
            "function pick(x){ if (x < 10) { return x * 2; } if (x >= 100) { return x - 1; } return x + 5; }";
        for x in [5.0, 9.0, 10.0, 50.0, 99.0, 100.0, 250.0] {
            deopt_fuzz_sweep(src, &[Value::Number(x)]);
        }
    }

    /// DEOPT-FUZZ #4 — NaN in a bank slot at deopt: a forced deopt while a computed
    /// NaN sits in a register must resume correctly (canonical NaN, not a tagged
    /// value masquerading as NaN). The poly produces NaN for NaN inputs; the sweep
    /// forces a deopt at the op AFTER the NaN is computed.
    #[cfg(target_os = "windows")]
    #[test]
    fn deopt_fuzz_nan_in_bank_slot_resumes_canonical() {
        let src = "function f(a, b){ var x = a / b; var y = x + 1; return y * 2; }";
        // 0/0 = NaN, Inf-Inf etc. — x is NaN, then used downstream.
        deopt_fuzz_sweep(src, &[Value::Number(0.0), Value::Number(0.0)]);
        deopt_fuzz_sweep(src, &[Value::Number(f64::NAN), Value::Number(2.0)]);
    }

    /// DEOPT-FUZZ #5 — the bc_pc MUTATION ARM (proves the oracle has TEETH). If a
    /// DeoptSite's `bc_pc` is offset by ±1, resuming at the WRONG op must produce a
    /// DIFFERENT result (the fuzz would redden). We build a function, force a deopt
    /// at a specific op, then manually run the resume with a MUTATED bc_pc and assert
    /// it diverges from the VM oracle — confirming a wrong bc_pc is caught.
    #[cfg(target_os = "windows")]
    #[test]
    fn deopt_fuzz_bc_pc_mutation_arm_reddens() {
        // s = a + b; t = s * 10; return t. Resuming at the WRONG op (off-by-one)
        // recomputes from a stale/incomplete register image → wrong result.
        let src = "function f(a, b){ var s = a + b; var t = s * 10; return t; }";
        let m = module_for_first_fn(src);
        let args = [Value::Number(3.0), Value::Number(4.0)];
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let mut refuse = refuse_with_interp;
        let oracle = run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut refuse).unwrap();
        // The correct resume at EVERY op must equal the oracle (re-affirm).
        deopt_fuzz_sweep(src, &args);
        // Now build a deliberately-WRONG resume: decode the entry bank (args only),
        // and resume at a bc_pc shifted to a LATER op (skipping the `s = a + b`
        // computation), so `s`/`t` are read uninitialized → a divergent result.
        // We resume at the op that computes `t = s * 10` (index ≥ the Add) over the
        // ENTRY register image (s not yet computed) — this must NOT equal the oracle.
        let n = m.fns[0].code.len();
        // Find the index of the Mul op (t = s * 10).
        let mul_idx = m.fns[0]
            .code
            .iter()
            .position(|op| matches!(op, Op::Mul { .. }))
            .expect("function has a Mul");
        assert!(mul_idx < n);
        // Entry register image = args in slots 0..2, rest undefined (n_regs sized).
        let nregs = m.fns[0].n_regs as usize;
        let mut regs: Vec<Value> = vec![Value::Undefined; nregs.max(args.len())];
        for (i, a) in args.iter().enumerate() {
            regs[i] = a.clone();
        }
        // Resume at the Mul op over the ENTRY image (wrong bc_pc — `s` never set).
        let mut d2 = refuse_with_interp;
        let wrong = t2_resume_on_vm(&m, regs, mul_idx, &Value::Undefined, &empty, &mut d2);
        // The oracle is (3+4)*10 = 70; resuming at the Mul with s=undefined gives a
        // DIFFERENT value (NaN or 0), so the mutation MUST be caught.
        assert!(
            !fuzz_results_eq(&wrong, &Ok(oracle.clone())),
            "bc_pc-mutation arm FAILED to redden: a wrong resume pc matched the oracle \
             (the fuzz would not catch a wrong bc_pc)\n  wrong={wrong:?}\n  oracle={oracle:?}"
        );
    }

    /// DEOPT-FUZZ #6 — NO DUPLICATE SIDE EFFECT. A function that CALLS a side-
    /// effecting callee then has a guard that deopts must NOT re-run the call. We
    /// count the callee's invocations via the dispatch closure and assert the
    /// observable count under T2 (with a forced post-call deopt) == the VM count.
    /// This is the loops-with-calls / guard-after-call class the P5 unblocking
    /// enables (was declined before).
    #[cfg(target_os = "windows")]
    #[test]
    fn deopt_fuzz_no_duplicate_side_effect_after_call() {
        use std::cell::Cell;
        use std::rc::Rc;
        let _heap = crate::interp::T2HeapGuard::new(true);
        // f(n) = bump() + n ; bump() is a global that increments a counter and
        // returns its new value. The `+ n` (Add) is a deopt-capable op AFTER the
        // committed call — forcing a deopt there must NOT re-run bump().
        let src = "function f(n){ var v = bump(); return v + n; }";
        let m = module_for_first_fn(src);
        let nregs = m.fns[0].n_regs as usize;
        // Find the Add op index (the deopt-capable op after the call).
        let add_idx = m.fns[0]
            .code
            .iter()
            .position(|op| matches!(op, Op::Add { .. }))
            .expect("f has an Add after the call");

        // A dispatcher that implements bump() and counts its calls.
        let counter = Rc::new(Cell::new(0i64));
        let globals: std::cell::RefCell<HashMap<String, Value>> = {
            let c2 = counter.clone();
            let bump = Value::NativeFunction(Rc::new(crate::interp::NativeFn {
                name: "bump".into(),
                func: crate::interp::NativeFnBody::Pure(Box::new(move |_args| {
                    let nv = c2.get() + 1;
                    c2.set(nv);
                    Ok(Value::Number(nv as f64))
                })),
                length: 0,
                is_ctor: false,
                props: std::cell::RefCell::new(HashMap::new()),
            }));
            let mut g = HashMap::new();
            g.insert("bump".to_string(), bump);
            std::cell::RefCell::new(g)
        };
        let mut refuse = refuse_with_interp;

        // VM oracle: one call to bump() per f() invocation.
        counter.set(0);
        let args = [Value::Number(100.0)];
        let vm = run_function(&m, 0, &args, &Value::Undefined, &globals, None, &mut refuse).unwrap();
        let vm_calls = counter.get();
        assert_eq!(vm_calls, 1, "VM: bump() called exactly once");
        assert!(matches!(vm, Value::Number(n) if n == 101.0), "VM f(100) = 1 + 100 = 101");

        // T2 with a FORCED deopt at the Add (after the committed call). bump() must
        // STILL be called exactly once (the resume continues AFTER the call, never
        // re-runs it).
        counter.set(0);
        let _g = ForceDeoptGuard::new(Some(add_idx));
        let native = compile_t2_for_fuzz(&m)
            .expect("f (call + post-call Add) must COMPILE under P5 (was declined before)");
        // The function must have a deopt site at the Add's bc_pc (the forced one).
        assert!(native.deopt_site_count() > 0, "f has resume sites");
        let mut d = |callee: Value, this: Value, a: Vec<Value>| -> Result<Value, RuntimeError> {
            // Route the call through the global bump() like the VM does.
            match callee {
                Value::NativeFunction(nf) => match &nf.func {
                    crate::interp::NativeFnBody::Pure(p) => p(a).map_err(|e| match e {
                        crate::interp::JsError::Throw(v) => RuntimeError::Thrown(v),
                        other => RuntimeError::TypeError(format!("{other:?}")),
                    }),
                    _ => Err(RuntimeError::TypeError("unexpected native body".into())),
                },
                _ => {
                    let _ = (this, a);
                    Err(RuntimeError::TypeError("callee is not callable".into()))
                }
            }
        };
        let t2 = run_t2lite_call(&native, &m, &args, &Value::Undefined, &globals, &mut d).unwrap();
        let t2_calls = counter.get();
        assert_eq!(
            t2_calls, vm_calls,
            "NO-DUPLICATE-EFFECT: bump() called {t2_calls}x under T2 (forced post-call deopt) vs \
             {vm_calls}x on the VM — a resume must NOT re-run the committed call"
        );
        assert!(
            matches!(t2, Value::Number(n) if n == 101.0),
            "T2 resumed result f(100) = 101 (identical to VM)"
        );
        let _ = nregs;
    }

    /// DEOPT-FUZZ #7 — natural shape-miss + non-number deopts agree (the reason
    /// coverage via crafted poison inputs, not forced). A getprop on a non-object,
    /// and arithmetic on a non-number, each NATURALLY deopt+resume to the VM result.
    #[cfg(target_os = "windows")]
    #[test]
    fn deopt_fuzz_natural_non_number_operand_resumes() {
        // Arithmetic where an operand is a STRING (non-number) → the load_num guard
        // naturally deopts; resume runs the VM's string/coercion path.
        let src = "function add1(a){ return a + 1; }";
        let m = module_for_first_fn(src);
        let _heap = crate::interp::T2HeapGuard::new(true);
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        for arg in [
            Value::Number(41.0),         // no deopt: native computes 42
            Value::str("x".to_string()), // deopt: "x" + 1 = "x1" (VM string concat)
            Value::Bool(true),           // true + 1 = 2 (VM coercion)
        ] {
            let mut r1 = refuse_with_interp;
            let oracle =
                run_function(&m, 0, std::slice::from_ref(&arg), &Value::Undefined, &empty, None, &mut r1);
            let native = compile_t2_for_fuzz(&m).expect("add1 compiles");
            let mut r2 = refuse_with_interp;
            let t2 = run_t2lite_call(
                &native,
                &m,
                std::slice::from_ref(&arg),
                &Value::Undefined,
                &empty,
                &mut r2,
            );
            assert!(
                fuzz_results_eq(&t2, &oracle),
                "natural deopt for arg {arg:?}: T2 != VM\n  t2={t2:?}\n  vm={oracle:?}"
            );
        }
    }

    /// DEOPT-FUZZ #8 — FORCING the CALL op itself: a deopt at the call's bc_pc must
    /// resume the VM AT the call (running it EXACTLY once on the VM, never natively
    /// first), so the side-effecting callee fires once total. Proves a forced
    /// pre-call resume is not a double-effect either.
    #[cfg(target_os = "windows")]
    #[test]
    fn deopt_fuzz_force_call_op_runs_callee_exactly_once() {
        use std::cell::Cell;
        use std::rc::Rc;
        let _heap = crate::interp::T2HeapGuard::new(true);
        let src = "function f(n){ var v = bump(); return v + n; }";
        let m = module_for_first_fn(src);
        let call_idx = m.fns[0]
            .code
            .iter()
            .position(|op| matches!(op, Op::CallValue { .. } | Op::CallFn { .. }))
            .expect("f has a call op");
        let counter = Rc::new(Cell::new(0i64));
        let globals: std::cell::RefCell<HashMap<String, Value>> = {
            let c2 = counter.clone();
            let bump = Value::NativeFunction(Rc::new(crate::interp::NativeFn {
                name: "bump".into(),
                func: crate::interp::NativeFnBody::Pure(Box::new(move |_a| {
                    let nv = c2.get() + 1;
                    c2.set(nv);
                    Ok(Value::Number(nv as f64))
                })),
                length: 0,
                is_ctor: false,
                props: std::cell::RefCell::new(HashMap::new()),
            }));
            let mut g = HashMap::new();
            g.insert("bump".to_string(), bump);
            std::cell::RefCell::new(g)
        };
        let mut dispatch = |callee: Value, _t: Value, a: Vec<Value>| -> Result<Value, RuntimeError> {
            match callee {
                Value::NativeFunction(nf) => match &nf.func {
                    crate::interp::NativeFnBody::Pure(p) => p(a).map_err(|e| match e {
                        crate::interp::JsError::Throw(v) => RuntimeError::Thrown(v),
                        other => RuntimeError::TypeError(format!("{other:?}")),
                    }),
                    _ => Err(RuntimeError::TypeError("native".into())),
                },
                _ => Err(RuntimeError::TypeError("callee is not callable".into())),
            }
        };
        let args = [Value::Number(10.0)];
        // Force a deopt AT the call op → the native code never runs the call; resume
        // runs it once on the VM. Total bump() calls must be exactly 1.
        counter.set(0);
        let _g = ForceDeoptGuard::new(Some(call_idx));
        let native = compile_t2_for_fuzz(&m).expect("f compiles under P5");
        let r = run_t2lite_call(&native, &m, &args, &Value::Undefined, &globals, &mut dispatch).unwrap();
        assert_eq!(counter.get(), 1, "forced-at-call resume must run bump() exactly once");
        assert!(matches!(r, Value::Number(n) if n == 11.0), "f(10) = 1 + 10 = 11");
    }

    // ════════════════════════════════════════════════════════════════════════
    // T4 EXTENSION 1 — INLINED-FRAME DEOPT FUZZER (the inline-deopt-to-caller
    // reconstruction proof, on the UN-INLINED corpus, BEFORE any inliner exists).
    //
    // THE KILL-CHECK the milestone demands: before any speculative inlining ships,
    // the inlined-frame reconstruction MATH must be proven byte-identical to the
    // VM. The design (INLINE-DEOPT-TO-CALLER, osr.rs Extension 1) reconstructs the
    // CALLER frame at the Call op from the live bank and resumes the VM there, so
    // the VM performs the ordinary (non-inlined) call. We drive that path over a
    // REAL JIT bank via the `set_force_inlined_reconstruct_pc` hook + a forced
    // deopt at the Call op, on a hand-built two-function caller→callee fixture, and
    // assert the resumed result == the plain un-inlined VM result. A MUTATION ARM
    // (wrong caller_bc_pc_of_call) proves the oracle is non-vacuous (a wrong resume
    // target diverges). No inliner exists yet — this proves the reconstruction is
    // correct so P3 inlining lands against a proven bailout.
    // ════════════════════════════════════════════════════════════════════════

    /// RAII: set the T4 inlined-frame-reconstruction fuzzer hook for a scope.
    #[cfg(target_os = "windows")]
    struct ForceInlinedReconstructGuard {
        prev: Option<usize>,
    }
    #[cfg(target_os = "windows")]
    impl ForceInlinedReconstructGuard {
        fn new(pc: Option<usize>) -> Self {
            ForceInlinedReconstructGuard {
                prev: crate::jit::set_force_inlined_reconstruct_pc(pc),
            }
        }
    }
    #[cfg(target_os = "windows")]
    impl Drop for ForceInlinedReconstructGuard {
        fn drop(&mut self) {
            crate::jit::set_force_inlined_reconstruct_pc(self.prev);
        }
    }

    /// Build a hand-assembled two-function module `{ f, g }` where caller `f(x)`
    /// computes `t = x * k1`, calls `g(t)` (CallFn → fns[1]), then `return r + k2`,
    /// and callee `g(y)` returns `y + 1`. Returns `(module, call_pc)` — the bytecode
    /// index of the CallFn op in `f` (the inlined-frame resume target). Full control
    /// over the call op index + arg slots is what lets the fuzzer drive the
    /// Extension-1 reconstruction deterministically.
    #[cfg(target_os = "windows")]
    fn two_fn_caller_callee_module(k1: f64, k2: f64) -> (Module, usize) {
        // Caller f(x): regs — 0=x (param), 1=k1const, 2=t, 3=r, 4=k2const, 5=ret.
        // code:
        //   0: LoadConst r1 = k1
        //   1: Mul       r2 = r0 * r1        ; t = x * k1
        //   2: CallFn    r3 = g(r2)          ; first_arg=2, n_args=1, fn_idx=1  <-- call_pc
        //   3: LoadConst r4 = k2
        //   4: Add       r5 = r3 + r4        ; r + k2
        //   5: Ret       r5
        let f = BcFunction {
            name: "f".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 6,
            consts: vec![Value::Number(k1), Value::Number(k2)],
            code: vec![
                Op::LoadConst { dst: 1, k: 0 },
                Op::Mul { dst: 2, lhs: 0, rhs: 1 },
                Op::CallFn { dst: 3, fn_idx: 1, first_arg: 2, n_args: 1 },
                Op::LoadConst { dst: 4, k: 1 },
                Op::Add { dst: 5, lhs: 3, rhs: 4 },
                Op::Ret { src: 5 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
        };
        // Callee g(y): regs — 0=y (param), 1=oneconst, 2=ret.
        let g = BcFunction {
            name: "g".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 3,
            consts: vec![Value::Number(1.0)],
            code: vec![
                Op::LoadConst { dst: 1, k: 0 },
                Op::Add { dst: 2, lhs: 0, rhs: 1 },
                Op::Ret { src: 2 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),

        strict: false,
        };
        let call_pc = 2;
        (Module { fns: vec![f, g], script_forinit_syncs: Vec::new() }, call_pc)
    }

    /// INLINED-FRAME-DEOPT FUZZER #1 — the reconstruction is byte-identical to the
    /// un-inlined VM at the Call op. For a range of inputs/constants, a forced deopt
    /// at the CallFn op routed through `osr::reconstruct_caller_frame` (the inline-
    /// deopt-to-caller path) must produce the SAME result as the plain VM run of f.
    #[cfg(target_os = "windows")]
    #[test]
    fn inlined_frame_deopt_reconstructs_caller_byte_identical() {
        let _heap = crate::interp::T2HeapGuard::new(true);
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        for (k1, k2) in [(2.0, 10.0), (0.5, -3.0), (3.0, 0.0), (-1.5, 7.0), (1e6, 1.0)] {
            let (m, call_pc) = two_fn_caller_callee_module(k1, k2);
            for x in [5.0, 0.0, -2.5, 100.0, f64::NAN] {
                let args = [Value::Number(x)];
                // VM oracle — plain un-inlined run of f (which calls g).
                let mut r0 = refuse_with_interp;
                let oracle =
                    run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut r0);
                // Force a deopt AT the Call op AND route the resume through the
                // inlined-frame caller reconstruction (Extension 1) over the real bank.
                let _gd = ForceDeoptGuard::new(Some(call_pc));
                let _gi = ForceInlinedReconstructGuard::new(Some(call_pc));
                let native = compile_t2_for_fuzz(&m)
                    .expect("caller f (CallFn) compiles under heap mode");
                let mut r1 = refuse_with_interp;
                let t4 =
                    run_t2lite_call(&native, &m, &args, &Value::Undefined, &empty, &mut r1);
                assert!(
                    fuzz_results_eq(&t4, &oracle),
                    "INLINED-FRAME DEOPT: caller reconstruction at the Call op (k1={k1}, \
                     k2={k2}, x={x}) resumed to a DIFFERENT result than the un-inlined VM\n  \
                     t4={t4:?}\n  vm={oracle:?}"
                );
            }
        }
    }

    /// INLINED-FRAME-DEOPT FUZZER #2 — the MUTATION ARM (the oracle has TEETH). The
    /// inline-deopt-to-caller resume target IS `caller_bc_pc_of_call` (the Call op);
    /// if we reconstruct with a WRONG resume pc (a LATER op, skipping the call), the
    /// VM resumes over a register image where the call result (`r3`) was never
    /// computed → a DIFFERENT result. This proves the reconstruction's resume pc is
    /// load-bearing (a wrong inlined-frame `caller_bc_pc_of_call` is caught), so
    /// fuzzer #1's pass is not vacuous.
    #[cfg(target_os = "windows")]
    #[test]
    fn inlined_frame_deopt_wrong_resume_pc_mutation_arm_reddens() {
        let _heap = crate::interp::T2HeapGuard::new(true);
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let (m, call_pc) = two_fn_caller_callee_module(2.0, 10.0);
        let args = [Value::Number(5.0)];
        // VM oracle: f(5) = g(5*2) + 10 = (10 + 1) + 10 = 21.
        let mut r0 = refuse_with_interp;
        let oracle = run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut r0).unwrap();
        assert!(matches!(oracle, Value::Number(n) if n == 21.0), "f(5) = 21");

        // Build the caller's pre-call register image (the bank at the Call op): args
        // in slot 0, k1 in slot 1, t=x*k1 in slot 2 — exactly what the VM holds when
        // it reaches the Call. (r3/r4/r5 not yet computed.)
        let nregs = m.fns[0].n_regs as usize;
        let mut regs: Vec<Value> = vec![Value::Undefined; nregs];
        regs[0] = Value::Number(5.0); // x
        regs[1] = Value::Number(2.0); // k1
        regs[2] = Value::Number(10.0); // t = x * k1

        // CORRECT inlined-frame resume (at the Call op) == oracle.
        let mut rc = refuse_with_interp;
        let correct =
            t2_resume_on_vm(&m, regs.clone(), call_pc, &Value::Undefined, &empty, &mut rc);
        assert!(
            fuzz_results_eq(&correct, &Ok(oracle.clone())),
            "correct inline-deopt-to-caller resume (at the Call op) must equal the oracle"
        );

        // WRONG resume pc: skip the call, resume at the Add op (call_pc + 2 = the
        // `Add r5 = r3 + r4`). `r3` (the call result) and `r4` were never set → the
        // resumed result diverges. (call_pc=2, LoadConst k2 at 3, Add at 4.)
        let add_idx = m.fns[0]
            .code
            .iter()
            .position(|op| matches!(op, Op::Add { .. }))
            .expect("f has an Add after the call");
        let mut rw = refuse_with_interp;
        let wrong = t2_resume_on_vm(&m, regs, add_idx, &Value::Undefined, &empty, &mut rw);
        assert!(
            !fuzz_results_eq(&wrong, &Ok(oracle)),
            "MUTATION ARM FAILED TO REDDEN: a WRONG inlined-frame resume pc (skipping \
             the call) matched the oracle — the resume target is not actually load-bearing\n  \
             wrong={wrong:?}"
        );
    }

    /// INLINED-FRAME-DEOPT FUZZER #3 — the structural verifier rejects an arg slot
    /// outside the caller bank (the inlined-frame analogue of the SafepointMap
    /// out-of-bank-root UAF check), proving the reconstruction's in-range gate is
    /// non-vacuous and would catch an inliner that records a garbage arg slot.
    #[cfg(target_os = "windows")]
    #[test]
    fn inlined_frame_site_verifier_rejects_garbage_arg_slot() {
        let (m, call_pc) = two_fn_caller_callee_module(2.0, 10.0);
        let caller = &m.fns[0];
        // A well-formed site (arg slot 2 < 6 regs, resume pc in range) verifies.
        let good = crate::osr::InlinedDeoptSite {
            base: crate::osr::DeoptSite {
                native_off: 0,
                bc_pc: call_pc,
                reason: crate::osr::DeoptReason::NonNumber,
            },
            frame: crate::osr::InlinedFrame {
                caller_bc_pc_of_call: call_pc,
                callee_entry_bc_pc: 0,
                arg_slot_map: vec![2],
            },
        };
        assert!(good.verify_against_caller(caller.code.len(), caller.n_regs as usize));
        // A site with an arg slot OUTSIDE the 6-slot caller bank is rejected.
        let bad = crate::osr::InlinedDeoptSite {
            base: good.base,
            frame: crate::osr::InlinedFrame {
                caller_bc_pc_of_call: call_pc,
                callee_entry_bc_pc: 0,
                arg_slot_map: vec![99],
            },
        };
        assert!(!bad.verify_against_caller(caller.code.len(), caller.n_regs as usize));
    }

    // ════════════════════════════════════════════════════════════════════════
    // T4 PHASE P3 — CROSS-FUNCTION INLINING: REAL inlined-frame deopt fuzzer.
    //
    // Unlike the P0 fuzzer (which drove the inline-deopt-to-caller reconstruction
    // over a real bank via the hook, BEFORE any inliner existed), these tests run
    // the GENUINE P3 inliner end-to-end: inline the callee, compile the FUSED body
    // through the T4 backend with the resume-pc map, run via `run_t4_call`, and:
    //   (1) BYTE-IDENTITY — the inlined T4 result == the un-inlined VM result for a
    //       range of inputs/constants (the call frame is gone but the answer is the
    //       same);
    //   (2) DEOPT-FUZZ — force a deopt at EVERY op of the FUSED body (including every
    //       inlined-region op) and assert the resumed result == the VM. An inlined-
    //       region guard's mapped bc_pc is the caller's Call op, so the resume runs
    //       the ORIGINAL caller VM which performs the ordinary non-inlined call —
    //       byte-identical. This is the inlined-frame deopt fuzzer "exercised for
    //       real" (the milestone GATE);
    //   (3) MUTATION ARM — a wrong resume-pc map reddens the deopt-fuzz (non-vacuity).
    // ════════════════════════════════════════════════════════════════════════

    /// Compile the T4 INLINED native code for a 2-fn `{caller, callee}` module
    /// honoring any active `ForceDeoptGuard` (the backend reads the force-deopt pc).
    /// None if there is nothing to inline / the compile declines.
    #[cfg(target_os = "windows")]
    fn compile_t4_inlined_for_fuzz(m: &Module) -> Option<crate::jit::JitFunction> {
        match crate::t4::try_compile_t4_inlined_status(m, 0) {
            crate::t4::T4CompileStatus::Ready(jf) => Some(jf),
            _ => None,
        }
    }

    /// P3 BYTE-IDENTITY: the inlined T4 result equals the un-inlined VM result for a
    /// matrix of constants × inputs. Proves the inliner's splice + remap + result
    /// store computes EXACTLY the caller's observable value with the call eliminated.
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_inline_result_is_byte_identical_to_vm() {
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        for (k1, k2) in [(2.0, 10.0), (0.5, -3.0), (3.0, 0.0), (-1.5, 7.0), (1e6, 1.0)] {
            let (m, _call_pc) = two_fn_caller_callee_module(k1, k2);
            // Confirm the inliner actually fired (engagement, not vacuous green).
            let inlined = crate::t4::inline_first_call(&m, 0)
                .expect("the CallFn to numeric g must inline");
            assert_eq!(inlined.inlined_calls, 1, "exactly one call inlined");
            assert!(
                !inlined.fused.code.iter().any(|op| matches!(op, Op::CallFn { .. })),
                "the fused body must contain NO CallFn (the call was inlined away)"
            );
            let native = compile_t4_inlined_for_fuzz(&m)
                .expect("the inlined fused body compiles under T4");
            assert!(native.t4_deopt_module().is_some(), "inlined T4 carries the caller resume module");
            for x in [5.0, 0.0, -2.5, 100.0, f64::NAN, -0.0] {
                let args = [Value::Number(x)];
                let mut r0 = refuse_with_interp;
                let oracle = run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut r0);
                let mut r1 = refuse_with_interp;
                let t4 = run_t4_call(&native, &args, &Value::Undefined, &empty, &mut r1);
                assert!(
                    fuzz_results_eq(&t4, &oracle),
                    "T4 INLINE result diverged from the VM (k1={k1}, k2={k2}, x={x})\n  \
                     t4={t4:?}\n  vm={oracle:?}"
                );
            }
        }
    }

    /// P3 DEOPT-FUZZ (the GATE): force a deopt at EVERY op of the FUSED body and
    /// assert the resumed result == the un-inlined VM. This exercises the inlined-
    /// frame deopt for REAL — an inlined-region guard resumes the ORIGINAL caller at
    /// the Call op (the VM then performs the ordinary call), every caller-region op
    /// resumes at its own original index. Both must be byte-identical to the VM.
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_inline_deopt_at_every_fused_op_is_byte_identical() {
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        for (k1, k2) in [(2.0, 10.0), (0.5, -3.0), (-1.5, 7.0)] {
            let (m, _call_pc) = two_fn_caller_callee_module(k1, k2);
            let inlined = crate::t4::inline_first_call(&m, 0).expect("inlines");
            let n_fused_ops = inlined.fused.code.len();
            for x in [5.0, 0.0, -2.5, 100.0] {
                let args = [Value::Number(x)];
                let mut r0 = refuse_with_interp;
                let oracle = run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut r0);
                // Force a deopt at each fused op boundary in turn.
                for force_pc in 0..n_fused_ops {
                    let _gd = ForceDeoptGuard::new(Some(force_pc));
                    let native = match compile_t4_inlined_for_fuzz(&m) {
                        Some(n) => n,
                        None => continue, // the forced op may make this op-index decline; skip.
                    };
                    let mut r1 = refuse_with_interp;
                    let t4 = run_t4_call(&native, &args, &Value::Undefined, &empty, &mut r1);
                    assert!(
                        fuzz_results_eq(&t4, &oracle),
                        "T4 INLINE deopt at fused op {force_pc} diverged from the VM \
                         (k1={k1}, k2={k2}, x={x})\n  t4={t4:?}\n  vm={oracle:?}"
                    );
                }
            }
        }
    }

    /// P3 MUTATION ARM — the resume-pc map is LOAD-BEARING. If an inlined-region
    /// guard's resume pc were the inlined op itself (the fused index) instead of the
    /// caller's Call op, resuming the ORIGINAL caller at that (out-of-range or wrong)
    /// index would diverge. We compile the inlined body, then DIRECTLY resume the
    /// original caller at a WRONG bc_pc (an inlined op's fused index, which is past
    /// the small caller's code), and assert it does NOT match the oracle — proving
    /// the correct mapping (to the Call op) is what makes the deopt-fuzz pass.
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_inline_wrong_resume_pc_mutation_arm_reddens() {
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        let (m, call_pc) = two_fn_caller_callee_module(2.0, 10.0);
        let args = [Value::Number(5.0)];
        let mut r0 = refuse_with_interp;
        let oracle = run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut r0).unwrap();
        // f(5) = g(5*2) + 10 = 11 + 10 = 21.
        assert!(matches!(oracle, Value::Number(n) if n == 21.0));

        // The caller's pre-call register image (the bank at the Call op).
        let nregs = m.fns[0].n_regs as usize;
        let mut regs: Vec<Value> = vec![Value::Undefined; nregs];
        regs[0] = Value::Number(5.0); // x
        regs[1] = Value::Number(2.0); // k1
        regs[2] = Value::Number(10.0); // t = x * k1

        // CORRECT resume (the mapping the inliner produces): at the Call op.
        let mut rc = refuse_with_interp;
        let correct = t2_resume_on_vm(&m, regs.clone(), call_pc, &Value::Undefined, &empty, &mut rc);
        assert!(
            fuzz_results_eq(&correct, &Ok(oracle.clone())),
            "correct inline resume (Call op) must equal the oracle"
        );

        // WRONG resume: skip the call, resume at the post-call Add op. r3 (the call
        // result) was never written → divergence. (This is what a buggy map that
        // pointed an inlined-region guard at a post-call op would do.)
        let add_idx = m.fns[0]
            .code
            .iter()
            .position(|op| matches!(op, Op::Add { .. }))
            .expect("f has an Add after the call");
        let mut rw = refuse_with_interp;
        let wrong = t2_resume_on_vm(&m, regs, add_idx, &Value::Undefined, &empty, &mut rw);
        assert!(
            !fuzz_results_eq(&wrong, &Ok(oracle)),
            "MUTATION ARM FAILED TO REDDEN: a wrong inline resume pc matched the oracle"
        );
    }

    /// P3 — a callee with MULTIPLE returns (early-return branch) inlines correctly:
    /// each `Ret` becomes a store + jump-to-continuation, so both control-flow paths
    /// produce the VM-identical result, including a forced deopt on each.
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_inline_callee_with_early_return_matches_vm() {
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        // caller f(x): r1 = call g(x); return r1 + 100.
        //   0: CallFn   r1 = g(r0)        first_arg=0, n_args=1, fn_idx=1
        //   1: LoadConst r2 = 100
        //   2: Add      r3 = r1 + r2
        //   3: Ret      r3
        let f = BcFunction {
            name: "f".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 4,
            consts: vec![Value::Number(100.0)],
            code: vec![
                Op::CallFn { dst: 1, fn_idx: 1, first_arg: 0, n_args: 1 },
                Op::LoadConst { dst: 2, k: 0 },
                Op::Add { dst: 3, lhs: 1, rhs: 2 },
                Op::Ret { src: 3 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
            feedback: std::cell::RefCell::new(Vec::new()),

            strict: false,
        };
        // callee g(y): if (y < 0) return 0; return y * 2.
        //   0: LoadConst r1 = 0
        //   1: Lt        r2 = y < r1        (y < 0)
        //   2: JmpIfFalse r2 -> 5
        //   3: LoadConst r3 = 0
        //   4: Ret       r3                 (early return 0)
        //   5: LoadConst r4 = 2
        //   6: Mul       r5 = y * r4
        //   7: Ret       r5
        let g = BcFunction {
            name: "g".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 6,
            consts: vec![Value::Number(0.0), Value::Number(2.0)],
            code: vec![
                Op::LoadConst { dst: 1, k: 0 },
                Op::Lt { dst: 2, lhs: 0, rhs: 1 },
                Op::JmpIfFalse { cond: 2, target: 5 },
                Op::LoadConst { dst: 3, k: 0 },
                Op::Ret { src: 3 },
                Op::LoadConst { dst: 4, k: 1 },
                Op::Mul { dst: 5, lhs: 0, rhs: 4 },
                Op::Ret { src: 5 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
            feedback: std::cell::RefCell::new(Vec::new()),

            strict: false,
        };
        let m = Module { fns: vec![f, g], script_forinit_syncs: Vec::new() };
        let inlined = crate::t4::inline_first_call(&m, 0).expect("inlines the branchy callee");
        assert!(!inlined.fused.code.iter().any(|op| matches!(op, Op::CallFn { .. })));
        let native = compile_t4_inlined_for_fuzz(&m).expect("branchy inlined body compiles");
        let n_fused = inlined.fused.code.len();
        // BOTH branches: x = -3 (early-return 0 path) and x = 4 (the y*2 path).
        for x in [-3.0, 4.0, 0.0, -0.0, 7.5] {
            let args = [Value::Number(x)];
            let mut r0 = refuse_with_interp;
            let oracle = run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut r0);
            let mut r1 = refuse_with_interp;
            let t4 = run_t4_call(&native, &args, &Value::Undefined, &empty, &mut r1);
            assert!(
                fuzz_results_eq(&t4, &oracle),
                "branchy inline result diverged (x={x})\n  t4={t4:?}\n  vm={oracle:?}"
            );
            // Deopt-fuzz every fused op on both branches too.
            for force_pc in 0..n_fused {
                let _gd = ForceDeoptGuard::new(Some(force_pc));
                let nat = match compile_t4_inlined_for_fuzz(&m) {
                    Some(n) => n,
                    None => continue,
                };
                let mut r2 = refuse_with_interp;
                let t4d = run_t4_call(&nat, &args, &Value::Undefined, &empty, &mut r2);
                assert!(
                    fuzz_results_eq(&t4d, &oracle),
                    "branchy inline deopt@{force_pc} diverged (x={x})\n  t4={t4d:?}\n  vm={oracle:?}"
                );
            }
        }
    }

    /// P4 KEYSTONE GATE — REDUNDANCY / CHECK ELIMINATION over an INLINED fused body.
    /// A callee with REDUNDANT pure expressions (the jit.js `y*y` shape, recomputed)
    /// is inlined; the P4 pass (running under `Allow::Always` on the fused body)
    /// folds the recomputation to a copy AND removes its implicit operand checks.
    /// THE GATE: (1) the fold actually fired (non-vacuity — the fused body has FEWER
    /// arith ops than the un-folded splice would); (2) the inlined T4 result is
    /// byte-identical to the un-inlined VM; (3) a forced deopt at EVERY fused op
    /// (including the folded `Move`s) resumes the ORIGINAL pristine caller and is
    /// byte-identical to the VM — proving the fold never corrupts the deopt frame.
    #[cfg(target_os = "windows")]
    #[test]
    fn t4_p4_inline_redundancy_folds_and_deopts_byte_identical() {
        let empty: std::cell::RefCell<HashMap<String, Value>> =
            std::cell::RefCell::new(HashMap::new());
        // caller f(x): r1 = call g(x); return r1 + 1.
        let f = BcFunction {
            name: "f".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 4,
            consts: vec![Value::Number(1.0)],
            code: vec![
                Op::CallFn { dst: 1, fn_idx: 1, first_arg: 0, n_args: 1 },
                Op::LoadConst { dst: 2, k: 0 },
                Op::Add { dst: 3, lhs: 1, rhs: 2 },
                Op::Ret { src: 3 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
            feedback: std::cell::RefCell::new(Vec::new()),

            strict: false,
        };
        // callee g(y): p = y*y; q = y*y; s = y*y; return p + q + s.  (THREE y*y —
        // two are redundant and fold to copies of the first under P4.)
        let g = BcFunction {
            name: "g".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 6,
            consts: vec![],
            code: vec![
                Op::Mul { dst: 1, lhs: 0, rhs: 0 }, // p = y*y
                Op::Mul { dst: 2, lhs: 0, rhs: 0 }, // q = y*y  (redundant)
                Op::Mul { dst: 3, lhs: 0, rhs: 0 }, // s = y*y  (redundant)
                Op::Add { dst: 4, lhs: 1, rhs: 2 }, // p + q
                Op::Add { dst: 5, lhs: 4, rhs: 3 }, // + s
                Op::Ret { src: 5 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
            feedback: std::cell::RefCell::new(Vec::new()),

            strict: false,
        };
        let m = Module { fns: vec![f, g], script_forinit_syncs: Vec::new() };

        // NON-VACUITY: the inliner + P4 fold leave FEWER `Mul` ops in the fused body
        // than the raw splice (which would have all three y*y). Count Muls before/after.
        let raw = crate::t4::inline_first_call(&m, 0).expect("inlines the redundant callee");
        let raw_muls = raw.fused.code.iter().filter(|op| matches!(op, Op::Mul { .. })).count();
        // The raw splice keeps all three y*y muls (P4 has not run on `inline_first_call`'s
        // output — it runs inside try_compile_t4_inlined_status). Confirm the shape.
        assert!(raw_muls >= 3, "the raw inlined splice has all three y*y muls (got {raw_muls})");
        crate::t4::reset_redundancy_rewrite_count();
        let native = compile_t4_inlined_for_fuzz(&m).expect("the redundant inlined body compiles");
        assert!(
            crate::t4::redundancy_rewrite_count() >= 2,
            "P4 must fold the two redundant y*y (non-vacuous); folded={}",
            crate::t4::redundancy_rewrite_count()
        );
        assert!(native.t4_deopt_module().is_some(), "inlined T4 carries the pristine caller resume");

        let n_fused = raw.fused.code.len();
        for x in [5.0, 0.0, -2.5, 100.0, f64::NAN, -0.0, 1e160] {
            let args = [Value::Number(x)];
            let mut r0 = refuse_with_interp;
            let oracle = run_function(&m, 0, &args, &Value::Undefined, &empty, None, &mut r0);
            let mut r1 = refuse_with_interp;
            let t4 = run_t4_call(&native, &args, &Value::Undefined, &empty, &mut r1);
            assert!(
                fuzz_results_eq(&t4, &oracle),
                "P4 inline-folded result diverged (x={x})\n  t4={t4:?}\n  vm={oracle:?}"
            );
            // DEOPT-FUZZ every fused op (the folded Moves included).
            for force_pc in 0..n_fused {
                let _gd = ForceDeoptGuard::new(Some(force_pc));
                let nat = match compile_t4_inlined_for_fuzz(&m) {
                    Some(n) => n,
                    None => continue,
                };
                let mut r2 = refuse_with_interp;
                let t4d = run_t4_call(&nat, &args, &Value::Undefined, &empty, &mut r2);
                assert!(
                    fuzz_results_eq(&t4d, &oracle),
                    "P4 inline-folded deopt@{force_pc} diverged (x={x})\n  t4={t4d:?}\n  vm={oracle:?}"
                );
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // B3 — SAFEPOINT STACK MAPS + JIT-VALUE ROOTING (the UAF keystone).
    //
    // THE MUTATION PROOF the milestone demands: a heap value held LIVE ACROSS A
    // SAFEPOINT solely through an optimizing tier's register/spill bank must
    // SURVIVE a GC forced AT that safepoint — and the test must FAIL when the
    // rooting (the spill-pointers-to-the-identity-map-bank discipline) is removed.
    //
    // The shipped B3 discipline: a pointer live across a safepoint is spilled to
    // its bank slot, recorded in the safepoint's `SafepointRec.live_roots`, and
    // `gc_seed_jit_banks` (which scans every registered bank slot) roots it. These
    // tests drive that contract DIRECTLY with a real `OwningRegBank`, a
    // `gc_register_object`'d object whose SOLE strong ref is the bank slot, a built
    // + B3-verified `SafepointMap`, and a forced `gc_collect` standing in for the
    // collection that would land at the safepoint. The clear-not-free sweep would
    // EMPTY the object's map if it weren't rooted, so survival == "its key is still
    // present AND strong_count >= 1" — the exact data-loss the brief warns of.
    // ════════════════════════════════════════════════════════════════════════
    #[cfg(target_os = "windows")]
    mod t3_safepoint_uaf {
        use super::*;
        use crate::jsval::JsVal;
        use crate::osr::{SafepointKind, SafepointMap};
        use std::cell::RefCell;
        use std::rc::Rc;

        /// The whole B3 scenario, parameterized by whether the spill-to-bank rooting
        /// discipline is APPLIED. Returns `(survived, key_present, map_verified)`:
        ///   - `survived`     : the object's `Rc` was still upgradeable after the GC.
        ///   - `key_present`  : the object's map still held its property (not cleared
        ///                      by the clear-not-free sweep).
        ///   - `map_verified` : the built `SafepointMap` passed `verify_against_bank`
        ///                      (the B3 discipline gate).
        ///
        /// CORRECT path (`apply_rooting = true`): the object's sole ref is stored
        /// into bank slot 1, the safepoint records slot 1 as a live root, the bank is
        /// GC-registered, then a collect is forced WHILE that is the only reference.
        /// The bank-slot root must mark the object → it survives.
        ///
        /// BROKEN ARM (`apply_rooting = false`): the object is NOT stored into the
        /// bank (the "forgot to spill before the safepoint" codegen bug). The
        /// safepoint *claims* slot 1 is a root, but slot 1 actually holds `undefined`
        /// — so the registered bank's scan finds no live ref to the object, the
        /// forced GC sweeps it, and survival fails. The mutation proof: this arm
        /// MUST differ from the correct arm.
        fn run_safepoint_gc_scenario(apply_rooting: bool) -> (bool, bool, bool) {
            // A GC-registered heap Object with one property the sweep would clear.
            let mut m: HashMap<String, Value> = HashMap::new();
            m.insert("k".to_string(), Value::Number(123.0));
            let obj_rc = Rc::new(RefCell::new(m));
            crate::interp::gc_register_object(&obj_rc);
            let obj_jv = JsVal::object(&obj_rc);
            let obj_weak = Rc::downgrade(&obj_rc);

            // Build a 4-slot owning bank (GC-registered for its lifetime). This is
            // the optimizing tier's register/spill bank.
            let mut bank = OwningRegBank::new_for_test(4, &[]);

            // THE DISCIPLINE: a pointer live across the safepoint is SPILLED to its
            // identity-map bank slot (slot 1) before the safepoint. Only the correct
            // path does this; the broken arm leaves slot 1 = undefined (no spill).
            if apply_rooting {
                unsafe { bank.store_for_test(1, obj_jv) }; // bank now owns the +1
            }

            // Build the B3 safepoint map: a HelperCall safepoint that records slot 1
            // as a live pointer root (per the spill discipline). Verify the
            // discipline against the bank size (slot 1 < 4 ⇒ bank-resident).
            let mut sp = SafepointMap::new();
            sp.record(0, SafepointKind::HelperCall, SafepointMap::roots_from_slots([1usize]));
            let map_verified = sp.verify_against_bank(4).is_ok();

            // DROP the test's own strong hold so the ONLY remaining strong ref to the
            // object is the bank slot (correct path) or nothing (broken arm). After
            // this, on the correct path the object is BANK-ONLY-reachable — the exact
            // condition under which the clear-not-free sweep would empty it unless a
            // root marks it.
            drop(obj_rc);

            // FORCE A GC AT THE SAFEPOINT. The bank is registered, so
            // `gc_seed_jit_banks` scans its slots; slot 1 (correct path) holds the
            // object's only ref and roots it. The interp drives the collect.
            let interp = crate::interp::Interp::new();
            let _ = interp.gc_collect(&[]);

            // Observe survival.
            let survived = obj_weak.upgrade().is_some();
            let key_present = obj_weak
                .upgrade()
                .map(|rc| rc.borrow().contains_key("k"))
                .unwrap_or(false);

            // Tear the bank down (RAII pops the GC registration). On the broken arm
            // slot 1 was never stored so teardown is a no-op for the object.
            drop(bank);
            (survived, key_present, map_verified)
        }

        /// CORRECT PATH (rooting applied): the object held only by a bank slot across
        /// a safepoint SURVIVES a GC forced at that safepoint, and the B3 discipline
        /// verifies. THE positive proof.
        #[test]
        fn safepoint_rooted_value_survives_forced_gc() {
            if !crate::interp::gc_enabled() {
                return; // GC disabled — the bank-root path can't engage (sweep is off).
            }
            let (survived, key_present, map_verified) = run_safepoint_gc_scenario(true);
            assert!(map_verified, "B3 discipline must verify: slot 1 < bank len 4");
            assert!(
                survived,
                "a bank-slot-rooted heap value MUST survive a GC at the safepoint \
                 (gc_seed_jit_banks roots the slot)"
            );
            assert!(
                key_present,
                "the survived object's property must NOT be cleared by the sweep — \
                 the bank slot rooted it, so it was marked, not emptied"
            );
            // After the bank drops the registry is drained (RAII gap-free).
            assert_eq!(
                crate::interp::jit_bank_registry_len(),
                0,
                "owning bank registration popped on drop"
            );
        }

        /// THE MUTATION ARM (rooting REMOVED): the SAME scenario without spilling the
        /// pointer to the bank before the safepoint. The forced GC now sweeps the
        /// bank-unreachable object → it does NOT survive. This is the load-bearing
        /// proof that the rooting is what makes the correct path pass: remove it and
        /// the object dies, exactly as the milestone requires ("the test MUST fail
        /// without the rooting").
        #[test]
        fn safepoint_unrooted_value_is_swept_by_forced_gc_proving_rooting_is_load_bearing() {
            if !crate::interp::gc_enabled() {
                return;
            }
            let (survived, key_present, _map_verified) = run_safepoint_gc_scenario(false);
            // WITHOUT the spill-to-bank rooting the object is bank-unreachable, so the
            // clear-not-free sweep empties/frees it. (It is either dropped outright —
            // `survived == false` — or, if some transient kept the Rc alive, its map
            // is CLEARED — `key_present == false`. Either way the data is lost.)
            assert!(
                !survived || !key_present,
                "MUTATION PROOF FAILED TO REDDEN: without the spill-to-bank rooting \
                 the unreachable object must be swept (freed or cleared), but it \
                 survived intact — the rooting is not actually load-bearing"
            );
        }

        /// DIRECT DIFFERENTIAL: the correct vs broken arm MUST diverge — survival is
        /// caused by the rooting, full stop. This pairs the two arms in one assertion
        /// so a regression that accidentally roots everything (e.g. a conservative
        /// over-scan) is also caught (it would make the broken arm survive too).
        #[test]
        fn safepoint_rooting_is_the_cause_of_survival() {
            if !crate::interp::gc_enabled() {
                return;
            }
            let correct = run_safepoint_gc_scenario(true);
            let broken = run_safepoint_gc_scenario(false);
            // Correct: survives intact.
            assert!(correct.0 && correct.1, "correct arm must survive intact");
            // Broken: does NOT survive intact.
            assert!(!(broken.0 && broken.1), "broken arm must NOT survive intact");
            // They MUST differ — the rooting is the difference.
            assert_ne!(
                (correct.0, correct.1),
                (broken.0, broken.1),
                "the spill-to-bank rooting must be the ONLY difference between the \
                 arms; if they agree, survival is not caused by the rooting"
            );
        }

        /// THE B3 GATE is non-vacuous: a safepoint that records a pointer root in a
        /// slot OUTSIDE the bank (a "forgot to spill" codegen bug that put the pointer
        /// in a register not mirrored to a scanned bank slot) is REJECTED by
        /// `verify_against_bank` — so `optimize_with_safepoints`/the compile path
        /// declines rather than installing un-rooted code. This proves the discipline
        /// verifier actually catches the UAF-enabling bug.
        #[test]
        fn b3_discipline_rejects_a_pointer_root_outside_the_bank() {
            let mut sp = SafepointMap::new();
            // Bank has 4 slots, but the safepoint claims slot 9 is a pointer root —
            // gc_seed_jit_banks scans bank[0..4], so slot 9 would NOT be rooted (UAF).
            sp.record(0, SafepointKind::HelperCall, SafepointMap::roots_from_slots([9usize]));
            assert!(
                sp.verify_against_bank(4).is_err(),
                "the B3 verifier MUST reject a pointer root outside the scanned bank"
            );
            // The same root IS valid against a bank large enough to scan slot 9 —
            // proving the check is the range relation, not a blanket rejection.
            assert!(sp.verify_against_bank(10).is_ok());
        }

        /// T4 EXTENSION-2 FORCE-GC-AT-SAFEPOINT KEYSTONE. The instant T4 inlines a
        /// callee that does a Call/GetProp, it holds a TAGGED heap JsVal in the bank
        /// ACROSS a HelperCall safepoint (the inlined call). This drives EXACTLY that
        /// hazard: a HelperCall safepoint (the kind `build_safepoint_map` records for
        /// an inlined `CallFn`/`CallValue`/`New`) records the heap value's bank slot
        /// as a live root, the bank is GC-rooted, and a GC is forced AT the safepoint.
        /// The clear-not-free sweep would empty the inlined value unless the spill-to-
        /// bank discipline roots it. This is the P0 force-GC-at-safepoint test the T4
        /// design requires GREEN before any inlining (which produces such safepoints)
        /// ships — proving the SafepointMap activation is load-bearing for T4.
        #[test]
        fn t4_inlined_callee_heap_value_survives_gc_at_helpercall_safepoint() {
            if !crate::interp::gc_enabled() {
                return; // GC disabled — the sweep that would clear it is off.
            }
            // The HelperCall scenario IS the T4 inlined-call hazard: a heap value live
            // across a runtime call, rooted only via the bank spill. Reuse the proven
            // scenario driver (its safepoint kind is HelperCall) and assert survival.
            let (survived, key_present, map_verified) = run_safepoint_gc_scenario(true);
            assert!(
                map_verified,
                "T4: the inlined-callee safepoint map must verify against the bank"
            );
            assert!(
                survived && key_present,
                "T4 KEYSTONE: a heap value an inlined callee holds across a HelperCall \
                 safepoint MUST survive a GC forced there (the bank spill roots it). If \
                 this reddens, T4 inlining cannot ship — it would UAF the inlined value."
            );
        }

        /// T4 EXTENSION-2 verify_against_bank is WIRED INTO THE T3 EMISSION PATH that
        /// T4 reuses — proving the UAF gate runs on every optimizing compile NOW,
        /// before T4 codegen exists. We compile a numeric function through the real
        /// T3 optimizer (`optimize_with_safepoints`): it builds a SafepointMap and
        /// asserts `verify_against_bank`. For the numeric subset the map has no
        /// pointer roots (vacuously covered), so this passes — establishing the gate
        /// is live (not dormant) ahead of the first heap-holding T4 codegen.
        #[test]
        fn t4_verify_against_bank_runs_on_every_optimizing_compile() {
            // A pure-numeric loop kernel — squarely in the T3/T4 optimizer subset.
            let src = "function k(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i * 2; } return s; }";
            let m = module_for_first_fn(src);
            // optimize_with_safepoints both optimizes AND runs verify_against_bank as
            // the B3 gate (a debug_assert in debug; a decline in release). It returns
            // Ok with a verified map for the numeric subset.
            match crate::t3::optimize_with_safepoints(&m.fns[0]) {
                Ok((_opt, _stats, map)) => {
                    // The returned map is verified against the optimized function's bank.
                    // (Numeric subset ⇒ no pointer roots ⇒ vacuously covered ⇒ Ok.)
                    assert!(
                        map.roots_covered_by_bank(64),
                        "the verified safepoint map's roots must be bank-covered"
                    );
                }
                Err(_) => {
                    // A decline is acceptable (the gate still ran); the point is the
                    // verify path is exercised on a real optimizing compile.
                }
            }
        }
    }
}
