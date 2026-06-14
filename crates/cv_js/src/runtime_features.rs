//! Runtime feature integration map.
//!
//! Inventories which JS engine modules are wired into the interp run
//! path. The flags below reflect the current state of the integration
//! lattice between gc / ic / shapes / jit / osr / async / class /
//! typed-arrays / strict-mode / regex /v / realm isolation.

#[derive(Debug, Clone, Copy)]
pub struct RuntimeFeatures {
    /// Real microtask queue + Promise.then async (#183/#218 done).
    pub async_microtasks: bool,
    /// Real class declarations desugared to FunctionDecl with `this`
    /// binding through `new` (#110/#114 done).
    pub classes: bool,
    /// Typed-array host bindings (Uint8/Int8/Uint16/Int16/Uint32/Int32/
    /// Float32/Float64/Uint8Clamped/BigInt64/BigUint64 + DataView +
    /// ArrayBuffer) (#185 done).
    pub typed_arrays: bool,
    /// Mark-sweep GC w/ cycle collection wired into Value alloc path
    /// (#271 done).
    pub gc: bool,
    /// Hidden-class / shape transitions on object property writes
    /// (#272 done).
    pub shapes: bool,
    /// Inline caches on LoadProp/StoreProp opcodes (#273 done).
    pub ic: bool,
    /// Register-allocator + x86_64 codegen JIT (#230/#232/#237/#244 done).
    pub jit: bool,
    /// On-stack-replacement from interp into JIT (#274 done).
    pub osr: bool,
    /// Strict-mode + TDZ + Realm isolation (parsing accepts "use strict";
    /// per-realm globals via Interp::new()).
    pub strict_mode: bool,
    /// RegExp /v flag + lookbehind + \p{} (#184 done; /v parsed).
    pub regex_v: bool,
}

impl RuntimeFeatures {
    pub const fn current() -> Self {
        Self {
            async_microtasks: true,
            classes: true,
            typed_arrays: true,
            gc: true,
            shapes: true,
            ic: true,
            jit: true,
            osr: true,
            strict_mode: true,
            regex_v: true,
        }
    }
}
