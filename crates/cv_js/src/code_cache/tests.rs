//! B5 — persisted code-cache tests.
//!
//! Coverage:
//!   * encoding round-trip for every `Op` variant (byte-identical re-decode),
//!   * MODULE round-trip oracle: compile → serialize → deserialize → run, and
//!     assert the deserialized module executes IDENTICALLY to the fresh-compiled
//!     module AND to the VM (the canonical oracle),
//!   * warmed-IC round-trip: a GetProp-heavy module's warmed feedback survives a
//!     serialize/deserialize (shape DESCRIPTORS re-interned) and still executes
//!     identically,
//!   * STALE-KEY MUTATION ARM (the gate's teeth): omitting the shape-assumptions
//!     digest from the key wrongly ACCEPTS a layout-drifted blob → the round-trip
//!     oracle DIVERGES; with the digest the same drift is REJECTED (recompile),
//!   * corrupt/truncated blob → clean fallback (`None`), never a wrong module,
//!   * disk store/load round-trip behind `CV_CODE_CACHE`.

use super::*;
use crate::bytecode::{self, Module, Op};
use crate::interp::Value;

// ----------------------------------------------------------------------
// Helpers.
// ----------------------------------------------------------------------

/// Canonical, type-distinguishing rendering of a completion value for test
/// comparison (keeps -0 / NaN / string-vs-number / nested structure visible).
fn canon(v: &Value) -> String {
    match v {
        Value::Undefined => "undefined".into(),
        Value::Null => "null".into(),
        Value::Hole => "<hole>".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if n.is_nan() {
                "NaN".into()
            } else if *n == 0.0 && n.is_sign_negative() {
                "-0".into()
            } else {
                format!("{n}")
            }
        }
        Value::BigInt(b) => format!("{b}n"),
        Value::String(s) => format!("{s:?}"),
        Value::Array(a) => {
            let items: Vec<String> = a.borrow().iter().map(canon).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(o) => {
            let m = o.borrow();
            let mut parts: Vec<String> = Vec::new();
            for k in m.keys() {
                let val = m.get(k).cloned().unwrap_or(Value::Undefined);
                parts.push(format!("{k}:{}", canon(&val)));
            }
            format!("{{{}}}", parts.join(", "))
        }
        Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_) => "<fn>".into(),
    }
}

/// Run a module on the VM and render its completion value canonically (or the
/// error). Used to compare a deserialized module against a fresh one.
fn run_canon(module: &Module) -> String {
    match bytecode::run_module(module) {
        Ok(v) => format!("ok:{}", canon(&v)),
        Err(e) => format!("err:{e}"),
    }
}

// ----------------------------------------------------------------------
// Op encoding round-trip (every variant).
// ----------------------------------------------------------------------

#[test]
fn every_op_variant_encodes_and_decodes_identically() {
    // One instance of each variant. Operand values are arbitrary but distinct so
    // a field swap would show.
    let ops = vec![
        Op::LoadConst { dst: 1, k: 2 },
        Op::LoadTrue { dst: 3 },
        Op::LoadFalse { dst: 4 },
        Op::LoadNull { dst: 5 },
        Op::LoadUndef { dst: 6 },
        Op::Move { dst: 7, src: 8 },
        Op::Add { dst: 9, lhs: 10, rhs: 11 },
        Op::Sub { dst: 12, lhs: 13, rhs: 14 },
        Op::Mul { dst: 15, lhs: 16, rhs: 17 },
        Op::Div { dst: 18, lhs: 19, rhs: 20 },
        Op::Mod { dst: 21, lhs: 22, rhs: 23 },
        Op::Pow { dst: 24, lhs: 25, rhs: 26 },
        Op::Eq { dst: 27, lhs: 28, rhs: 29 },
        Op::Neq { dst: 30, lhs: 31, rhs: 32 },
        Op::LooseEq { dst: 33, lhs: 34, rhs: 35 },
        Op::LooseNeq { dst: 36, lhs: 37, rhs: 38 },
        Op::Lt { dst: 39, lhs: 40, rhs: 41 },
        Op::Le { dst: 42, lhs: 43, rhs: 44 },
        Op::Gt { dst: 45, lhs: 46, rhs: 47 },
        Op::Ge { dst: 48, lhs: 49, rhs: 50 },
        Op::BitAnd { dst: 51, lhs: 52, rhs: 53 },
        Op::BitOr { dst: 54, lhs: 55, rhs: 56 },
        Op::BitXor { dst: 57, lhs: 58, rhs: 59 },
        Op::Shl { dst: 60, lhs: 61, rhs: 62 },
        Op::Shr { dst: 63, lhs: 64, rhs: 65 },
        Op::Ushr { dst: 66, lhs: 67, rhs: 68 },
        Op::Neg { dst: 69, src: 70 },
        Op::Not { dst: 71, src: 72 },
        Op::BitNot { dst: 73, src: 74 },
        Op::ToNumber { dst: 75, src: 76 },
        Op::Typeof { dst: 77, src: 78 },
        Op::In { dst: 79, lhs: 80, rhs: 81 },
        Op::DeleteProp { dst: 82, obj: 83, key_k: 84 },
        Op::DeleteIdx { dst: 85, obj: 86, key: 87 },
        Op::MakeRegex { dst: 88, source_k: 89, flags_k: 90 },
        Op::Jmp { target: 91 },
        Op::JmpIfFalse { cond: 92, target: 93 },
        Op::JmpIfTrue { cond: 94, target: 95 },
        Op::CallFn { dst: 96, fn_idx: 97, first_arg: 98, n_args: 99 },
        Op::LoadGlobal { dst: 100, name_k: 101 },
        Op::LoadGlobalChecked { dst: 102, name_k: 103 },
        Op::StoreGlobal { name_k: 104, src: 105 },
        Op::CallValue { dst: 106, callee: 107, this_reg: 108, first_arg: 109, n_args: 110 },
        Op::New { dst: 111, ctor: 112, first_arg: 113, n_args: 114 },
        Op::LoadThis { dst: 115 },
        Op::LoadSelf { dst: 116 },
        Op::GetProp { dst: 117, obj: 118, key_k: 119 },
        Op::GetIdx { dst: 120, obj: 121, key: 122 },
        Op::SetProp { obj: 123, key_k: 124, src: 125 },
        Op::SetIdx { obj: 126, key: 127, src: 128 },
        Op::NewArray { dst: 129, first_elem: 130, n_elems: 131 },
        Op::ArrayPush { arr: 132, val: 133 },
        Op::ArrayPushSpread { arr: 134, spread: 135 },
        Op::NewObject { dst: 136 },
        Op::Throw { src: 137 },
        Op::TryEnter { catch_target: 138, catch_reg: 139 },
        Op::TryExit,
        Op::EnumKeys { dst: 140, obj: 141 },
        Op::MakeClosure { dst: 142, fn_idx: 143, first_upvalue: 144, n_upvalues: 145 },
        Op::LoadUp { dst: 146, slot: 147 },
        Op::StoreUp { src: 148, slot: 149 },
        Op::Ret { src: 150 },
    ];
    for op in &ops {
        let mut w = Writer::new();
        write_op(&mut w, op);
        let mut r = Reader::new(&w.buf);
        let back = read_op(&mut r).expect("decodes");
        // Re-encode and compare bytes (Op isn't PartialEq; bytes are the contract).
        let mut w2 = Writer::new();
        write_op(&mut w2, &back);
        assert_eq!(w.buf, w2.buf, "round-trip differs for {op:?}");
        assert_eq!(r.pos, w.buf.len(), "decoder must consume exactly the bytes of {op:?}");
    }
}

#[test]
fn unknown_op_tag_fails_closed() {
    // A byte that is not a valid op tag must decode to None (fail closed), never a
    // bogus op.
    let buf = [0xFFu8, 0, 0, 0, 0];
    let mut r = Reader::new(&buf);
    assert!(read_op(&mut r).is_none());
}

// ----------------------------------------------------------------------
// Const encoding.
// ----------------------------------------------------------------------

#[test]
fn consts_round_trip_including_neg_zero_and_nan() {
    let cases = vec![
        Value::Undefined,
        Value::Null,
        Value::Bool(true),
        Value::Bool(false),
        Value::Number(42.0),
        Value::Number(-0.0),
        Value::Number(f64::NAN),
        Value::Number(f64::INFINITY),
        Value::str("hello".to_string()),
        Value::str(String::new()),
    ];
    for v in &cases {
        let mut w = Writer::new();
        assert!(write_const(&mut w, v), "serializable const must write");
        let mut r = Reader::new(&w.buf);
        let back = read_const(&mut r).expect("decodes");
        assert_eq!(canon(v), canon(&back), "const round-trip differs");
    }
    // -0 must survive (bit-exact).
    let mut w = Writer::new();
    write_const(&mut w, &Value::Number(-0.0));
    let mut r = Reader::new(&w.buf);
    if let Some(Value::Number(n)) = read_const(&mut r) {
        assert!(n == 0.0 && n.is_sign_negative(), "-0 sign bit lost");
    } else {
        panic!("expected a number");
    }
}

#[test]
fn non_serializable_const_declines_serialize() {
    // An object const can never appear in a real pool, but if it did we must
    // DECLINE (return false / None), not fake a representation.
    let mut w = Writer::new();
    let obj = Value::Object(std::rc::Rc::new(std::cell::RefCell::new(
        crate::OrderedMap::new(),
    )));
    assert!(!write_const(&mut w, &obj), "object const must decline");
    // A module containing such a const must fail to serialize (→ caller won't cache).
    let f = crate::bytecode::BcFunction {
        name: "<x>".into(),
        n_params: 0,
        rest_reg: None,
        n_regs: 1,
        consts: vec![obj],
        code: vec![Op::Ret { src: 0 }],
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),
    };
    let module = Module { fns: vec![f] };
    assert!(
        serialize_module("x", &module).is_none(),
        "a non-serializable const must make the module decline to cache"
    );
}

// ----------------------------------------------------------------------
// Module round-trip oracle (deserialized == fresh == VM).
// ----------------------------------------------------------------------

/// Compile `src`, serialize, deserialize, and assert the deserialized module runs
/// IDENTICALLY to the fresh-compiled module. Also re-runs the SOURCE through the
/// full A/B tier oracle so deserialized == fresh == VM == tree-walk.
fn assert_roundtrip(src: &str) {
    let fresh = bytecode::compile_program(src).expect("compiles");
    let blob = serialize_module(src, &fresh).expect("serializes");
    let key = compute_key(src, &fresh);
    let reloaded = deserialize_module(&blob, key).expect("deserializes + key matches");
    assert_eq!(
        run_canon(&fresh),
        run_canon(&reloaded),
        "deserialized module diverged from the fresh module on {src:?}"
    );
    // The source itself must agree across all tiers (the existing oracle).
    crate::ab_oracle::assert_tiers_agree(src)
        .unwrap_or_else(|d| panic!("tiers diverged on {src:?}: {d}"));
}

#[test]
fn module_roundtrip_corpus() {
    let cases = [
        "1 + 2 * 3;",
        "var s = 0; for (var i = 0; i < 10; i = i + 1) { s = s + i; } s;",
        "function fib(n){ if (n < 2) return n; return fib(n-1) + fib(n-2); } fib(12);",
        "var a = [1, 2, 3]; a[0] + a[1] + a[2];",
        "var o = {x: 1, y: 2}; o.x + o.y;",
        "var t = 'a' + 'b' + 'c'; t.length;",
        "var x = -0; 1/x;",         // -0 round-trip through the const pool
        "0/0;",                     // NaN
        "function g(a, b){ return a < b ? a : b; } g(3, 7);",
        "var n = 0; while (n < 5) { n = n + 1; } n;",
        "try { throw 7; } catch (e) { e + 1; }",
        "var arr = []; for (var i = 0; i < 4; i = i + 1) { arr.push(i*i); } arr;",
        "var o = {a:1}; o.a = o.a + 9; o.a;",
        "typeof 5;",
        "(function(){ var c = 10; function inner(){ return c + 1; } return inner(); })();",
    ];
    for src in cases {
        assert_roundtrip(src);
    }
}

// ----------------------------------------------------------------------
// Warmed-IC round-trip (shape descriptors re-interned).
// ----------------------------------------------------------------------

#[test]
fn warmed_ic_survives_roundtrip_and_executes_identically() {
    // A GetProp-heavy loop over same-shaped records WARMS the property IC.
    let src = r#"
        function sumx(arr) {
            var s = 0;
            for (var i = 0; i < arr.length; i = i + 1) { s = s + arr[i].x; }
            return s;
        }
        var data = [];
        for (var k = 0; k < 8; k = k + 1) { data.push({x: k, y: k+1}); }
        var out = 0;
        for (var r = 0; r < 20; r = r + 1) { out = sumx(data); }
        out;
    "#;
    let fresh = bytecode::compile_program(src).expect("compiles");
    // RUN it so the property IC warms (records the {x,y} shape → slot).
    let _ = bytecode::run_module(&fresh);

    // At least one function's IC must now carry feedback (else the test is
    // vacuous — we'd be round-tripping an empty IC).
    let any_feedback = fresh
        .fns
        .iter()
        .any(|f| f.ic.borrow().iter().any(|ic| ic.has_feedback()));
    assert!(any_feedback, "expected the GetProp IC to warm before serialization");

    let blob = serialize_module(src, &fresh).expect("serializes");
    let key = compute_key(src, &fresh);
    let reloaded = deserialize_module(&blob, key).expect("key matches, deserializes");

    // The reloaded module must carry re-interned IC feedback too.
    let reloaded_feedback = reloaded
        .fns
        .iter()
        .any(|f| f.ic.borrow().iter().any(|ic| ic.has_feedback()));
    assert!(reloaded_feedback, "warmed IC feedback was lost on reload");

    // And it must execute identically (a wrong re-intern would mis-resolve a slot).
    assert_eq!(run_canon(&fresh), run_canon(&reloaded), "reloaded IC diverged");
}

// ----------------------------------------------------------------------
// STALE-KEY MUTATION ARM — the gate has teeth.
//
// We build two modules sharing the SAME source string but carrying DIFFERENT
// warmed shape assumptions:
//   layout A: {x} → x at slot 0
//   layout B: {y, x} → x at slot 1  (a real-world layout drift)
// and assert:
//   * WITH the shape digest (production): the two keys DIFFER → a B-blob is
//     REJECTED under A's key (the drifted entry is caught, → recompile).
//   * WITHOUT it (mutation hook on): the keys collapse to EQUAL → the drifted
//     B-blob is wrongly ACCEPTED, which (executed) would read the wrong slot.
// This proves the shape-assumptions digest is the load-bearing invalidator.
// ----------------------------------------------------------------------

/// Build a one-function module whose single GetProp IC bakes `(shape_keys → slot)`.
fn module_with_baked_shape(shape_keys: &[&str], slot: u32) -> Module {
    // r0 = param obj; r1 = obj.x ; ret r1.  The const pool holds "x".
    let f = crate::bytecode::BcFunction {
        name: "<reader>".into(),
        n_params: 1,
        rest_reg: None,
        n_regs: 2,
        consts: vec![Value::str("x".to_string())],
        code: vec![
            Op::GetProp { dst: 1, obj: 0, key_k: 0 },
            Op::Ret { src: 1 },
        ],
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),
    };
    // Intern the given shape descriptor and bake it into the GetProp IC at ip 0.
    let shape_id = crate::shapes::with_shape_table(|t| {
        let mut s = t.empty();
        for k in shape_keys {
            s = t.add_property(s, k);
        }
        s
    });
    {
        let mut ics = f.ic.borrow_mut();
        ics.resize(f.code.len(), crate::bytecode::PropIc::INVALID);
        ics[0] = crate::bytecode::PropIc::from_serialized_own(&[(shape_id, slot)], false);
    }
    Module { fns: vec![f] }
}

#[test]
fn stale_key_arm_proves_shape_digest_is_load_bearing() {
    let src = "function r(o){ return o.x; } r({});"; // held constant across both

    let mod_a = module_with_baked_shape(&["x"], 0);
    let mod_b = module_with_baked_shape(&["y", "x"], 1);

    // (1) PRODUCTION (digest included): differing shape assumptions MUST produce
    //     different keys → a B-blob is rejected when expecting an A-key.
    let key_a = compute_key(src, &mod_a);
    let key_b = compute_key(src, &mod_b);
    assert_ne!(
        key_a, key_b,
        "the shape-assumptions digest must distinguish {{x}}->0 from {{y,x}}->1"
    );
    let blob_b = serialize_module(src, &mod_b).expect("serializes");
    assert!(
        deserialize_module(&blob_b, key_a).is_none(),
        "a layout-drifted blob must be REJECTED under the production key"
    );

    // (2) MUTATION (digest OMITTED): keys collapse → the drifted blob is wrongly
    //     ACCEPTED. This is exactly the silent corruption the digest prevents.
    {
        let _broken = StaleKeyGuard::new(true);
        let key_a2 = compute_key(src, &mod_a);
        let key_b2 = compute_key(src, &mod_b);
        assert_eq!(
            key_a2, key_b2,
            "with the digest omitted the keys must collapse (proving the digest is \
             what distinguishes them) — otherwise the arm is vacuous"
        );
        let blob_b2 = serialize_module(src, &mod_b).expect("serializes");
        assert!(
            deserialize_module(&blob_b2, key_a2).is_some(),
            "VACUOUS ARM: with the digest omitted the drifted blob was still rejected \
             — something other than the digest is doing the rejecting"
        );
    }

    // (3) RESTORE: with the hook off again the production behavior returns.
    let key_a3 = compute_key(src, &mod_a);
    let key_b3 = compute_key(src, &mod_b);
    assert_ne!(key_a3, key_b3, "restored digest must distinguish the layouts again");
}

/// End-to-end via `validate_and_deserialize`: a faithful reload validates + runs
/// identically, and a changed source is rejected.
#[test]
fn validate_rejects_drift_accepts_faithful_reload() {
    let src = "var s=0; for(var i=0;i<6;i=i+1){s=s+i;} s;";
    let fresh = bytecode::compile_program(src).expect("compiles");
    let _ = bytecode::run_module(&fresh); // warm
    let blob = serialize_module(src, &fresh).expect("serializes");

    let reloaded = validate_and_deserialize(src, &blob).expect("faithful reload validates");
    assert_eq!(run_canon(&fresh), run_canon(&reloaded));

    let other_src = "var s=0; for(var i=0;i<7;i=i+1){s=s+i;} s;";
    assert!(
        validate_and_deserialize(other_src, &blob).is_none(),
        "a changed source must be rejected by validation"
    );
}

// ----------------------------------------------------------------------
// Corruption / truncation → clean fallback.
// ----------------------------------------------------------------------

#[test]
fn corrupt_and_truncated_blobs_fail_closed() {
    let src = "1 + 2;";
    let fresh = bytecode::compile_program(src).expect("compiles");
    let blob = serialize_module(src, &fresh).expect("serializes");
    let key = compute_key(src, &fresh);

    // Bad magic.
    let mut bad_magic = blob.clone();
    bad_magic[0] ^= 0xFF;
    assert!(deserialize_module(&bad_magic, key).is_none());

    // Flipped key byte (offset 12 = first byte of the u64 key after 3 u32 headers).
    let mut bad_key = blob.clone();
    bad_key[12] ^= 0xFF;
    assert!(deserialize_module(&bad_key, key).is_none());

    // Truncated at every length — none may yield a module (fail closed).
    for cut in 0..blob.len() {
        let part = &blob[..cut];
        assert!(
            deserialize_module(part, key).is_none(),
            "a blob truncated to {cut} bytes must fail closed"
        );
        assert!(validate_and_deserialize(src, part).is_none());
    }

    // Empty blob.
    assert!(deserialize_module(&[], key).is_none());
    assert!(validate_and_deserialize(src, &[]).is_none());

    // A blob with a valid-looking header but garbage body.
    let mut garbage = MAGIC.to_le_bytes().to_vec();
    garbage.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    garbage.extend_from_slice(&ENGINE_VERSION.to_le_bytes());
    garbage.extend_from_slice(&[0xAB; 64]);
    assert!(validate_and_deserialize(src, &garbage).is_none());
}

#[test]
fn engine_version_mismatch_rejected() {
    let src = "3 * 4;";
    let fresh = bytecode::compile_program(src).expect("compiles");
    let mut blob = serialize_module(src, &fresh).expect("serializes");
    let key = compute_key(src, &fresh);
    // The ENGINE_VERSION u32 sits at offset 8 (after MAGIC u32 + FORMAT_VERSION u32).
    blob[8] = blob[8].wrapping_add(1);
    assert!(
        deserialize_module(&blob, key).is_none(),
        "an engine-version bump must reject every prior on-disk entry"
    );
}

// ----------------------------------------------------------------------
// Disk store/load round-trip.
// ----------------------------------------------------------------------

#[test]
fn disk_store_and_load_roundtrip() {
    // Use an isolated temp dir and exercise the serialize/disk layer directly so
    // the test is hermetic (independent of the process-cached `CV_CODE_CACHE`).
    let dir = std::env::temp_dir().join(format!("tbjs_code_cache_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mk dir");

    let src = "var s=0; for(var i=0;i<9;i=i+1){s=s+i*2;} s;";
    let fresh = bytecode::compile_program(src).expect("compiles");
    let _ = bytecode::run_module(&fresh);
    let blob = serialize_module(src, &fresh).expect("serializes");

    let path = dir.join(cache_filename(src));
    std::fs::write(&path, &blob).expect("write");

    let read = std::fs::read(&path).expect("read");
    let reloaded = validate_and_deserialize(src, &read).expect("disk reload validates");
    assert_eq!(run_canon(&fresh), run_canon(&reloaded), "disk reload diverged");

    // A different source maps to a DIFFERENT file (no false hit).
    let other = "var s=1;s;";
    assert_ne!(cache_filename(src), cache_filename(other));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn default_off_does_not_touch_disk() {
    // With CV_CODE_CACHE unset (the default in the test binary), store()/load() are
    // no-ops. This asserts the disabled-path contract only when actually disabled.
    if !code_cache_enabled() {
        assert!(load("never-cached-source-xyz").is_none());
        let m = bytecode::compile_program("1;").unwrap();
        store("never-cached-source-xyz", &m); // silent no-op
        assert!(load("never-cached-source-xyz").is_none());
    }
}
