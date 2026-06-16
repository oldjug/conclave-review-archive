//! `cv_js` — JavaScript engine (V1: lexer + parser + tree-walk interpreter).
//!
//! ECMA-262 is enormous; this crate ships in slices:
//!   - **Slice 1 (this commit):** lexer covering tokens needed for
//!     expressions, declarations, function definitions, control flow.
//!   - Slice 2: AST + recursive-descent parser.
//!   - Slice 3: tree-walk interpreter with a basic object model.
//!   - Slice 4: built-ins (Number, String, Array, Math, JSON).
//!   - Slice 5+: bytecode VM, JIT, full Web bindings.
//!
//! The goal of slice 1 is to provide a token stream `cv_html` can use to
//! at least *not crash* on inline `<script>` content, plus a basis for
//! the parser.

#![allow(
    dead_code,
    missing_debug_implementations,
    non_camel_case_types,
    unused_assignments,
    unused_imports,
    unused_mut
)]

pub mod ab_oracle;
pub mod ast;
pub mod async_lower;
pub mod bytecode;
/// B5 — persisted bytecode + type-feedback code cache. Serializes a compiled
/// `Module` (bytecode + warmed PropIc feedback) to disk keyed by a hash of
/// `(source, engine version, flags, shape-assumptions digest)`; validates or
/// recompiles on load. Gated by `CV_CODE_CACHE`, DEFAULT OFF. Native-code
/// persistence is the documented follow-on (the B1 code cage is its groundwork).
pub mod code_cache;
/// T4 (Maglev-class) PHASE P1 — binary/compare TYPE-FEEDBACK VECTOR. A
/// per-bytecode-slot monotone (widen-only) type-hint lattice (V8
/// `BinaryOperationHint`/`CompareOperationHint` shape) recorded by the bytecode
/// VM and exposed to the T4 lowering. RECORDING ONLY (no specialization); gated by
/// `CV_FEEDBACK`, DEFAULT OFF, observationally invisible. P2 consumes it; P5
/// persists it.
pub mod feedback;
pub mod gc;
pub mod interp;
pub mod jit;
/// Code cage — process-wide RX arena for the optimizing JIT (Windows only; the
/// per-page JIT install is the default + fallback). Gated by `CV_CODE_CAGE`.
#[cfg(target_os = "windows")]
pub mod jit_cage;
pub mod json;
pub mod jsval;
pub mod lexer;
pub mod m3_harness;
pub mod ordered;
pub use ordered::OrderedMap;
/// Per-property attribute (writable/enumerable/configurable + accessor) side
/// table — the ECMA-262 descriptor model implemented without growing the
/// per-object `OrderedMap`. Gated by `CV_PROP_DESC` (default OFF during
/// development; goal default-ON once the A/B oracle is green corpus-wide).
pub mod propattrs;
pub mod osr;
pub mod parser;
pub mod regex;
pub mod runtime_features;
/// Unicode property data for RegExp `\p{...}` / `\P{...}` property escapes
/// (ECMA-262 §22.2.1). UCD-derived code-point range tables (General_Category,
/// binary properties, common Scripts) consumed by `regex.rs` under the `u`/`v`
/// flag. Always on (additive; only reachable via `\p`/`\P` with the `u` flag).
pub mod unicode_props;
pub mod sab;
pub mod shapes;
/// `Temporal` — the TC39 Temporal date/time API (Stage 3, shipping in V8).
/// Real ISO-8601 calendar/time arithmetic (leap years, month-overflow
/// constrain/reject, calendar difference, ns-exact Instants). Always on
/// (additive: only reachable via the `Temporal` global). ZonedDateTime is a
/// UTC-offset core; named-IANA-zone DST + non-ISO calendars are documented
/// followups (see module footer), not stubs.
pub mod temporal;
/// `WeakRef` + `FinalizationRegistry` (ECMA-262 §26.1, §26.2) — weak references
/// backed by the real tracing GC. A WeakRef does not keep its target alive
/// (`deref` returns `undefined` once the GC determines the target unreachable);
/// a FinalizationRegistry enqueues held values for its cleanup callback after a
/// target is collected, and `unregister` removes registrations. Always on
/// (additive; only reachable via the `WeakRef`/`FinalizationRegistry` globals).
pub mod weakref;
/// T3 — the optimizing tier (B2 of PHASE B): bytecode → SSA-ish IR →
/// conservative semantics-preserving passes → linear-scan regalloc → the proven
/// T2-lite backend. Gated by `CV_T3` / `ForcedTier::T3`, DEFAULT OFF.
pub mod t3;

/// T4 (Maglev-class) speculative optimizing tier — PHASE P2: representation
/// selection + unboxed Float64. Reuses T3's optimizer + the T2-lite deopt
/// keystone, but emits through the representation-aware backend
/// (`jit::compile_t4_unboxed_with_deopt`) so same-block numeric operands skip the
/// per-op reload + tag-check + unbox. Gated by `CV_T4` / `ForcedTier::T4`, DEFAULT
/// OFF (byte-identical default build until soak).
pub mod t4;

/// Differential FUZZER (test-only): a seeded grammar generates thousands of varied
/// top-level JS programs and diffs the tree-walker against the top-level register-VM
/// path (`CV_TOPLEVEL_VM`) through the production-faithful oracle. The permanent
/// regression gate proving the VM tier is byte-identical for the WHOLE construct
/// grammar, not just a fixed corpus.
#[cfg(test)]
mod toplevel_vm_fuzz;

pub use ast::{ArrowBody, Expr, ForInit, Stmt, VarDeclarator, VarKind};
pub use bytecode::{
    BcFunction, CompileError, InlineLeafGuard, Module, Op, RuntimeError,
    compile_program as bc_compile, inline_leaf_enabled, inline_leaf_module,
    inline_numeric_leaf_calls, leaf_inline_count, reset_leaf_inline_count,
    run_module as bc_run, run_module_with_globals as bc_run_with_globals,
};
pub use ab_oracle::{
    Divergence, DivergenceKind, ThrownError, TierOutcome, assert_tiers_agree,
};
pub use interp::{
    ACCESSOR_GET, ACCESSOR_SET, BankRootGuard, ForcedTier, Interp, JsBigInt, JsError, NativeFn,
    NativeFnBody, NoP6JitGuard, PROTO_KEY, TierGuard, Value, current_native_this, diag_log,
    forced_tier,
    gc_enabled, gc_live_object_count, gc_register_array, gc_register_object, gen_gc_enabled,
    is_symbol_key, jit_bank_registry_len,
    js_stack_snapshot, make_pending_promise, make_settled_promise, make_temporal_error, native_ctor,
    native_ctor_pure, native_fn, native_fn_n, native_fn_with_interp, p6_exec_count,
    parse_bigint_from_string,
    register_jit_bank, resolve_promise,
    reset_bc_fn_cache, reset_call_inline_cache, reset_p6_exec_count, reset_t1_exec_count,
    reset_t2_exec_count, reset_t3_exec_count, reset_t4_exec_count, reset_toplevel_vm_took_count,
    set_force_tier, t1_exec_count, t2_exec_count, t2_heap_enabled, t3_exec_count, t4_exec_count,
    TopLevelVmGuard, toplevel_vm_enabled, toplevel_vm_took_count,
};
pub use lexer::{Keyword, Punct, Token, TokenKind, tokenize};
pub use parser::{ParseError, parse_program};
