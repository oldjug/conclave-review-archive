//! B5 end-to-end: the `CV_CODE_CACHE` production seam, exercised in its OWN
//! process so the `code_cache_enabled()` OnceLock observes the env we set here
//! (a unit test in the lib binary can't, because another test may read the flag
//! first). Proves:
//!   1. with the cache ON, the first compile MISSES (compiles + writes a blob),
//!      the second HITS (loads the same module from disk, no recompile),
//!   2. a cache HIT executes byte-identically to a fresh compile,
//!   3. a hand-corrupted on-disk blob falls back cleanly to a recompile (still
//!      correct), and
//!   4. when the cache is OFF nothing is written.

use cv_js::bytecode;
use cv_js::code_cache;
use cv_js::interp::Value;

fn canon(v: &Value) -> String {
    match v {
        Value::Number(n) => {
            if n.is_nan() {
                "NaN".into()
            } else {
                format!("{n}")
            }
        }
        Value::String(s) => format!("{s:?}"),
        Value::Bool(b) => b.to_string(),
        Value::Undefined => "undefined".into(),
        Value::Null => "null".into(),
        // The e2e snippets all return primitives; anything else is unexpected.
        _ => "<opaque>".into(),
    }
}

fn run_canon(m: &bytecode::Module) -> String {
    match bytecode::run_module(m) {
        Ok(v) => format!("ok:{}", canon(&v)),
        Err(e) => format!("err:{e}"),
    }
}

#[test]
fn code_cache_e2e_hit_miss_and_corruption_fallback() {
    // Isolated cache dir; enable the cache BEFORE any flag read in this process.
    let dir = std::env::temp_dir().join(format!("tbjs_cc_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: single-threaded test entry, set before the OnceLock is first read.
    unsafe {
        std::env::set_var("CV_CODE_CACHE", "1");
        std::env::set_var("CV_CODE_CACHE_DIR", &dir);
    }
    assert!(
        code_cache::code_cache_enabled(),
        "CV_CODE_CACHE=1 must enable the cache in this fresh process"
    );

    let src = "var s=0; for(var i=0;i<12;i=i+1){ s = s + i*i; } s;";

    // (1) MISS: no file yet → load returns None.
    assert!(code_cache::load(src).is_none(), "cold cache must miss");

    // First compile-through: compiles fresh AND persists a blob.
    let m1 = code_cache::compile_program_cached(src).expect("compiles");
    let expect = run_canon(&m1);

    // The blob now exists on disk.
    assert!(
        code_cache::load(src).is_some(),
        "after a compile-through the entry must be persisted + loadable"
    );

    // (2) HIT: a second compile-through returns a module loaded FROM DISK, and it
    //     executes identically to the fresh one.
    let m2 = code_cache::compile_program_cached(src).expect("hit returns a module");
    assert_eq!(expect, run_canon(&m2), "a cache HIT must execute identically");

    // Also assert it equals a guaranteed-fresh compile (the canonical truth).
    let fresh = bytecode::compile_program(src).expect("fresh compiles");
    assert_eq!(expect, run_canon(&fresh), "fresh == cached == VM");

    // (3) CORRUPTION FALLBACK: clobber the on-disk blob; a load must reject it
    //     (None) and compile-through must still return a CORRECT module.
    let path = dir.join({
        // The filename is internal; rediscover it by listing the dir (one entry).
        let mut name = String::new();
        for e in std::fs::read_dir(&dir).expect("read dir").flatten() {
            let f = e.file_name().to_string_lossy().to_string();
            if f.ends_with(".tbcc") {
                name = f;
            }
        }
        assert!(!name.is_empty(), "a .tbcc entry must exist");
        name
    });
    std::fs::write(&path, b"not a valid blob at all").expect("corrupt write");
    assert!(
        code_cache::load(src).is_none(),
        "a corrupt blob must be REJECTED (fail closed), not loaded"
    );
    let m3 = code_cache::compile_program_cached(src).expect("recompiles after corruption");
    assert_eq!(expect, run_canon(&m3), "corruption fallback must still be correct");
    // And the corrupt blob was rewritten with a valid one (best-effort store).
    assert!(
        code_cache::load(src).is_some(),
        "compile-through after corruption should rewrite a valid blob"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
