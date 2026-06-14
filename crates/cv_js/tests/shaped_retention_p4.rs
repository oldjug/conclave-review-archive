//! M3.2 P4 — Shaped-object RETENTION measurement (the go/no-go evidence for P5).
//!
//! This is an INTEGRATION test (its own process), which is what makes it a valid
//! measurement: `ordered::shaped_obj_enabled()` and `shaped_stats_enabled()` are
//! process-global `OnceLock`s read ONCE. In a unit-test binary, dozens of other
//! tests have already initialized those locks (with the flag off), so an
//! after-the-fact `set_var` would be ignored. Here, this file compiles to its OWN
//! binary; we set `CV_SHAPED_OBJ=1` + `CV_SHAPED_STATS=1` at the very top of the
//! ONE test, before any code path touches the locks, so the gate genuinely turns
//! on and the counters genuinely fire on the real interpreter's object allocations.
//!
//! What it measures: for each representative workload, it resets the global
//! retention counters, runs the workload through the REAL `Interp` (the same
//! `OrderedMap::new()` path every JS object takes), then snapshots
//! `ordered::shaped_stats()`. The reported Shaped-retention % and deopt-trigger
//! breakdown are therefore REAL counts of real allocations — not estimates.
//!
//! The workloads:
//!   (a) the curated object-model corpus (the oracle's `OBJECT_MODEL_SNIPPETS`
//!       analogue — property add/delete, prototypes, accessors, freeze, proxy,
//!       class, JSON, Map/Set, typed arrays);
//!   (b) SYNTHETIC real-bundle object-shape patterns: plain config objects, class
//!       instances, React-element-like records, a webpack chunk-map (mixed
//!       integer+string keys — THE P4 risk), event/promise-like objects, and
//!       JSON.parse output shapes;
//!   (c) the in-tree test262 object-model subset (if the checkout is present).
//!
//! Run with output:
//!   cargo test -p cv_js --test shaped_retention_p4 -- --nocapture

use cv_js::interp::Interp;
use cv_js::ordered::{reset_shaped_stats, shaped_obj_enabled, shaped_stats, shaped_stats_enabled, ShapedStats};

/// One named workload + its per-run JS sources.
struct Workload {
    name: &'static str,
    /// Each entry is run in a FRESH `Interp` (so an object's whole lifetime is
    /// observed: born, mutated, possibly deopted, then dropped at scope end).
    sources: Vec<String>,
}

/// The per-`Interp` ENGINE BASELINE: how many Shaped objects an interpreter
/// allocates (and how many deopt) just from `install_basic_globals` +
/// `install_json` + running an empty script — with ZERO user objects. The
/// engine's own infrastructure (Error class objects, Map/Set/Proxy internals,
/// the global object, …) gets stamped with exotic sentinels and deopts BY
/// DESIGN; that is fixed overhead, not a property of user code. Subtracting it
/// isolates the retention of the objects the WORKLOAD actually allocates.
fn engine_baseline() -> ShapedStats {
    reset_shaped_stats();
    let mut i = Interp::new();
    i.install_basic_globals();
    i.install_json();
    let _ = i.run_completion_value(";");
    shaped_stats()
}

/// Run every source in `w` through a fresh interpreter and return the DELTA of
/// the retention counters attributable to this workload (reset before, snapshot
/// after). Each source gets its own `Interp` so we measure full object
/// lifecycles, not a shared global heap.
fn measure(w: &Workload) -> ShapedStats {
    reset_shaped_stats();
    for src in &w.sources {
        let mut i = Interp::new();
        i.install_basic_globals();
        i.install_json();
        // We deliberately ignore per-snippet errors: an engine gap on an exotic
        // snippet must not abort the MEASUREMENT (the objects it allocated before
        // the error are already counted). The retention question is about what
        // got allocated, not whether every snippet completed.
        let _ = i.run_completion_value(src);
    }
    shaped_stats()
}

/// User-attributable retention: subtract `n_interps` copies of the engine
/// baseline (one per fresh `Interp` the workload spun up) from a workload's
/// totals, then compute retention over the remainder. Clamped at 0 so a workload
/// that under-counts (e.g. an early error) never reports a negative population.
/// Returns `(user_created, user_retained, user_rate)`.
fn user_attributable(w_stats: &ShapedStats, n_interps: u64, base: &ShapedStats) -> (u64, u64, f64) {
    let base_created = base.created_shaped.saturating_mul(n_interps);
    let base_retained = base.retained_shaped().saturating_mul(n_interps);
    let user_created = w_stats.created_shaped.saturating_sub(base_created);
    let user_retained = w_stats.retained_shaped().saturating_sub(base_retained);
    let rate = if user_created == 0 {
        1.0
    } else {
        user_retained as f64 / user_created as f64
    };
    (user_created, user_retained, rate)
}

/// Pretty-print a workload's retention line + trigger breakdown. `base` +
/// `n_interps` let it additionally report the USER-attributable retention (raw
/// minus the engine baseline), which is the number that drives the go/no-go.
fn report(w: &Workload, s: &ShapedStats, base: &ShapedStats, n_interps: u64) {
    let total_deopt = s.total_deopted();
    let pct = s.retention_rate() * 100.0;
    let (uc, ur, urate) = user_attributable(s, n_interps, base);
    println!(
        "\n[{name}]  created_shaped={cs}  retained={ret}  raw-retention={pct:.1}%   \
         USER: created={uc} retained={ur} retention={urate:.1}%",
        name = w.name,
        cs = s.created_shaped,
        ret = s.retained_shaped(),
        pct = pct,
        uc = uc,
        ur = ur,
        urate = urate * 100.0,
    );
    if total_deopt > 0 {
        println!(
            "    deopts (total {total_deopt}): int-key={ik} sentinel={ss} non-str={ns} cap={cap} \
             remove={rm} clear={cl} drain={dr} retain={rt} into-iter={ii}",
            ik = s.deopted_integer_key,
            ss = s.deopted_sentinel_stamp,
            ns = s.deopted_non_string_key,
            cap = s.deopted_cap_exceeded,
            rm = s.deopted_remove,
            cl = s.deopted_clear,
            dr = s.deopted_drain,
            rt = s.deopted_retain,
            ii = s.deopted_into_iter,
        );
    }
}

/// The curated object-model corpus — the same categories the A/B oracle's
/// `OBJECT_MODEL_SNIPPETS` covers (kept in sync deliberately): the hot paths the
/// flat-slot rewrite touches.
fn object_model_workload() -> Workload {
    let sources = vec![
        // property add / reassign + enumeration order (plain string keys → Shaped)
        "var o = {}; o.b = 1; o.a = 2; o.c = 3; Object.keys(o).join(',');",
        // delete → forces a deopt (remove)
        "var o = {b:1, a:2}; delete o.b; o.c = 3; Object.keys(o).join(',');",
        // integer keys mixed in → forces a deopt (int-key)
        "var o = {}; o[2]=1; o.x=1; o[0]=1; o[1]=1; o.y=1; Object.keys(o).join(',');",
        "var o = {x:1}; o.x = 2; o.x = 3; o.x;",
        "var o = {a:1,b:2,c:3}; var r=[]; for (var k in o) r.push(k+'='+o[k]); r.join(',');",
        // prototype chain + __proto__ (PROTO_KEY stays Shaped)
        "var base = {greet: function(){return 'hi';}}; var o = {}; o.__proto__ = base; o.greet();",
        "var base = {x: 10}; var o = Object.create(base); o.y = 20; '' + o.x + ',' + o.y;",
        "function A(){this.x=1;} A.prototype.y=2; var a = new A(); '' + a.x + ',' + a.y;",
        "var o = {a:1}; Object.getPrototypeOf(o) === Object.prototype;",
        // getters / setters / accessors (accessor stamp → deopt sentinel)
        "var o = { _v: 5, get v(){ return this._v * 2; }, set v(n){ this._v = n; } }; o.v = 10; o.v;",
        "var o = {}; Object.defineProperty(o, 'p', { get: function(){ return 42; } }); o.p;",
        "var log=[]; var o = { get a(){ log.push('get'); return 1; } }; o.a; o.a; log.join(',');",
        // freeze (frozen stamp → deopt sentinel)
        "var o = {x:1}; Object.freeze(o); o.x = 99; o.x;",
        "var o = {x:1}; Object.freeze(o); Object.isFrozen(o);",
        // proxy (proxy stamp → deopt sentinel)
        "var p = new Proxy({}, { get: function(t,k){ return 'G:'+k; } }); p.foo;",
        "var store={}; var p = new Proxy(store, { set: function(t,k,v){ t[k]=v*2; return true; } }); p.x=5; store.x;",
        // class + extends + super
        "class A { constructor(){ this.x = 1; } m(){ return 'A.m'; } } class B extends A { constructor(){ super(); this.y = 2; } m(){ return super.m()+'/B.m'; } } var b = new B(); '' + b.x + ',' + b.y + ',' + b.m();",
        "class S { static n = 5; static get d(){ return S.n*2; } } '' + S.n + ',' + S.d;",
        // JSON round-trip (JSON.parse output shapes)
        "JSON.stringify(JSON.parse('{\"b\":1,\"a\":[2,3],\"n\":null}'));",
        "var o = {z:1, a:2, m:[1,2,{q:3}]}; JSON.parse(JSON.stringify(o)).m[2].q;",
        // Map / Set (these stamp _isMap/_isSet sentinels → deopt by design)
        "var m = new Map(); m.set('a',1); m.set('b',2); m.set('a',9); '' + m.size + ',' + m.get('a');",
        "var s = new Set([1,2,2,3,3,3]); var out=[]; s.forEach(function(v){out.push(v);}); '' + s.size;",
        // typed arrays (_typedarray/_bytes sentinels → deopt by design)
        "var ta = new Int32Array(4); ta[0]=10; ta[1]=20; ta[2]=ta[0]+ta[1]; '' + ta[2] + ',' + ta.length;",
        // array methods that drive object shape changes
        "var o = {}; ['a','b','c'].forEach(function(k,i){ o[k]=i; }); JSON.stringify(o);",
    ];
    Workload {
        name: "object-model corpus",
        sources: sources.into_iter().map(String::from).collect(),
    }
}

/// SYNTHETIC real-bundle object-shape patterns — what real code (config objects,
/// framework records, bundlers) actually allocates at scale. Each pattern is run
/// in a LOOP so the count is statistically meaningful (a single object would be
/// noise next to the engine's own bootstrap allocations).
fn synthetic_real_bundle_workloads() -> Vec<Workload> {
    vec![
        // 1) Plain config / option objects {a,b,c,d} — THE common case. All
        //    string keys, no delete, no integer key → should stay 100% Shaped.
        Workload {
            name: "plain config objects {a,b,c,d}",
            sources: vec![
                "var out=0; for (var i=0;i<2000;i=i+1){ var cfg={width:100,height:50,color:'red',label:'x'}; out=out+cfg.width; } out;".into(),
            ],
        },
        // 2) Class instances (this.x=...; this.y=...) — the OO allocation shape.
        Workload {
            name: "class instances (this.x=.. constructor)",
            sources: vec![
                "function Point(x,y){ this.x=x; this.y=y; this.z=0; } var s=0; for(var i=0;i<2000;i=i+1){ var p=new Point(i,i+1); s=s+p.x; } s;".into(),
                "class Vec{ constructor(a,b,c){ this.a=a; this.b=b; this.c=c; } } var s=0; for(var i=0;i<2000;i=i+1){ var v=new Vec(i,i,i); s=s+v.a; } s;".into(),
            ],
        },
        // 3) React-element-like records {type,key,props,ref}. All string keys.
        Workload {
            name: "React-element-like {type,key,props,ref}",
            sources: vec![
                "function el(type,key,props){ return {type:type, key:key, props:props, ref:null, $$typeof:'react.element'}; } var n=0; for(var i=0;i<2000;i=i+1){ var e=el('div', 'k'+i, {className:'c'}); n=n+(e.type.length); } n;".into(),
            ],
        },
        // 4) Webpack chunk-map-like object with MIXED integer + string keys —
        //    THE P4 risk. Integer keys force a deopt the instant they appear.
        Workload {
            name: "webpack chunk-map (MIXED int+string keys)",
            sources: vec![
                // module id -> factory; ids are numeric strings (webpack uses
                // numeric module ids), so the first integer key deopts the map.
                // Build MANY such maps so the integer-key deopt is the headline.
                "var c=0; for(var n=0;n<1000;n=n+1){ var modules={}; for(var i=0;i<8;i=i+1){ modules[i]=i; } modules.runtime=1; c=c+modules.runtime; } c;".into(),
                // the chunk-loaded map {0:1, 1:1, app:1} idiom, allocated repeatedly
                "var t=0; for(var i=0;i<1000;i=i+1){ var installed={}; installed[0]=1; installed[1]=0; installed.app=1; t=t+installed.app; } t;".into(),
            ],
        },
        // 5) Event / promise-like objects (string keys, may use defineProperty).
        Workload {
            name: "event/promise-like objects",
            sources: vec![
                "function evt(type,target){ return {type:type, target:target, bubbles:true, cancelable:false, timeStamp:0, defaultPrevented:false}; } var n=0; for(var i=0;i<2000;i=i+1){ var e=evt('click', null); if(e.bubbles) n=n+1; } n;".into(),
                "function deferred(){ var d={state:'pending', value:undefined, onFulfilled:null, onRejected:null}; return d; } var n=0; for(var i=0;i<2000;i=i+1){ var d=deferred(); if(d.state==='pending') n=n+1; } n;".into(),
            ],
        },
        // 6) JSON.parse output shapes (API response records). All string keys.
        Workload {
            name: "JSON.parse output shapes (API records)",
            sources: vec![
                "var sum=0; for(var i=0;i<1000;i=i+1){ var o=JSON.parse('{\"id\":1,\"name\":\"alice\",\"email\":\"a@b.c\",\"active\":true}'); sum=sum+o.id; } sum;".into(),
                // nested object array (typical API list response)
                "var n=0; for(var i=0;i<500;i=i+1){ var r=JSON.parse('{\"page\":1,\"total\":3,\"items\":[{\"k\":1},{\"k\":2},{\"k\":3}]}'); n=n+r.items.length; } n;".into(),
            ],
        },
    ]
}

/// In-tree test262 object-model subset — the SAME directory list the A/B oracle
/// uses. Walks each dir (sorted, capped), filters out negative/module/async/
/// unsupported by a minimal frontmatter check, assembles harness + body, and
/// runs each through one interpreter. Returns `None` if the checkout is absent.
fn test262_workload() -> Option<Workload> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("conformance")
        .join("tmp")
        .join("test262");
    if !root.join("harness").join("sta.js").exists() {
        return None;
    }
    const DIRS: &[&str] = &[
        "built-ins/Object/keys",
        "built-ins/Object/getOwnPropertyNames",
        "built-ins/Object/values",
        "built-ins/Object/entries",
        "built-ins/Object/freeze",
        "built-ins/Object/getPrototypeOf",
        "built-ins/Object/create",
        "built-ins/Array/prototype/map",
        "built-ins/Array/prototype/filter",
        "built-ins/Array/prototype/forEach",
        "language/expressions/object",
        "language/expressions/property-accessors",
    ];
    const PER_DIR_CAP: usize = 18;

    let assert_js = std::fs::read_to_string(root.join("harness").join("assert.js")).ok();
    let sta_js = std::fs::read_to_string(root.join("harness").join("sta.js")).ok();
    let (assert_js, sta_js) = match (assert_js, sta_js) {
        (Some(a), Some(s)) => (a, s),
        _ => return None,
    };

    let mut sources = Vec::new();
    for dir in DIRS {
        let full = root.join("test").join(dir);
        let mut files: Vec<std::path::PathBuf> = match std::fs::read_dir(&full) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.extension().map(|x| x == "js").unwrap_or(false)
                        && !p
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.contains("_FIXTURE"))
                            .unwrap_or(false)
                })
                .collect(),
            Err(_) => continue,
        };
        files.sort();
        files.truncate(PER_DIR_CAP);
        for path in files {
            let body = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            // Minimal frontmatter filter (mirrors the oracle's runnability gate).
            let block = body
                .split_once("/*---")
                .and_then(|(_, r)| r.split_once("---*/"))
                .map(|(b, _)| b)
                .unwrap_or("");
            let negative = block.contains("negative:");
            let flags = block
                .lines()
                .find(|l| l.trim_start().starts_with("flags:"))
                .unwrap_or("");
            let is_module = flags.contains("module");
            let is_async = flags.contains("async") || block.contains("includes: [asyncHelpers");
            let is_raw = flags.contains("raw");
            let includes_line = block
                .lines()
                .find(|l| l.trim_start().starts_with("includes:"))
                .unwrap_or("");
            // Only sta.js/assert.js are auto-prepended; any other include → skip.
            let needs_other_include = includes_line.contains('[')
                && !includes_line
                    .split('[')
                    .nth(1)
                    .map(|r| {
                        r.split(']')
                            .next()
                            .unwrap_or("")
                            .split(',')
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty())
                            .all(|s| s == "sta.js" || s == "assert.js")
                    })
                    .unwrap_or(true);
            let unsupported = needs_other_include
                || body.contains("$262")
                || body.contains("createRealm")
                || body.contains("detachArrayBuffer");
            if negative || is_module || is_async || unsupported {
                continue;
            }
            let src = if is_raw {
                body
            } else {
                format!("{assert_js}\n{sta_js}\n{body}")
            };
            sources.push(src);
        }
    }
    if sources.is_empty() {
        return None;
    }
    Some(Workload {
        name: "test262 object-model subset",
        sources,
    })
}

#[test]
fn shaped_retention_p4_report() {
    // CRITICAL: turn the gate + stats ON before ANY code touches the OnceLocks.
    // Edition 2024: `set_var` is `unsafe`; sound here because this is the FIRST
    // thing the (single-threaded) test binary does — no other thread is reading
    // the env, and the OnceLocks have not been initialized yet.
    unsafe {
        std::env::set_var("CV_SHAPED_OBJ", "1");
        std::env::set_var("CV_SHAPED_STATS", "1");
    }
    assert!(
        shaped_obj_enabled(),
        "CV_SHAPED_OBJ must be ON for the retention measurement to mean anything"
    );
    assert!(
        shaped_stats_enabled(),
        "CV_SHAPED_STATS must be ON or the counters never fire"
    );

    println!("\n========== M3.2 P4 — Shaped-object RETENTION measurement ==========");
    println!("(CV_SHAPED_OBJ=1, CV_SHAPED_STATS=1; counts are REAL allocations of the live Interp)");

    // The per-Interp engine baseline (one install_basic_globals + install_json,
    // no user objects). The engine's own exotic infrastructure deopts BY DESIGN;
    // subtracting it isolates the retention of the objects USER code allocates,
    // which is what the go/no-go is actually about.
    let base = engine_baseline();
    println!(
        "\n[engine baseline / Interp]  created_shaped={cs}  retained={ret}  raw-retention={pct:.1}%",
        cs = base.created_shaped,
        ret = base.retained_shaped(),
        pct = base.retention_rate() * 100.0,
    );
    println!(
        "    (deopts by trigger: int-key={ik} sentinel={ss} non-str={ns} cap={cap} remove={rm} \
         clear={cl} drain={dr} retain={rt} into-iter={ii})",
        ik = base.deopted_integer_key,
        ss = base.deopted_sentinel_stamp,
        ns = base.deopted_non_string_key,
        cap = base.deopted_cap_exceeded,
        rm = base.deopted_remove,
        cl = base.deopted_clear,
        dr = base.deopted_drain,
        rt = base.deopted_retain,
        ii = base.deopted_into_iter,
    );
    println!("    NOTE: every per-snippet deopt count above this line is dominated by this");
    println!("    fixed per-Interp baseline; the USER column subtracts it out.");

    // ---- (b) synthetic real-bundle patterns: report each, keep user rates ----
    // Two aggregates: PLAIN-object workloads (the common case — the go signal)
    // and the mixed-int-key chunk-map (the hybrid's evidence, kept separate).
    let mut plain_user_created = 0u64;
    let mut plain_user_retained = 0u64;
    let mut webpack_int_deopts = 0u64;
    let mut webpack_user_rate = 1.0;
    let mut user_rates: Vec<(&'static str, f64, u64, u64)> = Vec::new();

    let synth = synthetic_real_bundle_workloads();
    for w in &synth {
        let s = measure(w);
        let n = w.sources.len() as u64;
        report(w, &s, &base, n);
        let (uc, ur, urate) = user_attributable(&s, n, &base);
        if w.name.contains("webpack") {
            webpack_int_deopts = s.deopted_integer_key;
            webpack_user_rate = urate;
        } else {
            plain_user_created += uc;
            plain_user_retained += ur;
        }
        user_rates.push((w.name, urate, uc, s.deopted_integer_key));
    }

    // ---- (a) the curated object-model corpus ----
    // This corpus deliberately includes Map/Set/Proxy/typed-array constructors,
    // which the engine tags with exotic sentinels and deopts BY DESIGN; its raw
    // retention is therefore NOT a plain-object signal (it's a stress mix). We
    // report it for the deopt-trigger breakdown but do not fold it into the
    // plain-object go gate.
    let om = object_model_workload();
    let om_s = measure(&om);
    let om_n = om.sources.len() as u64;
    report(&om, &om_s, &base, om_n);

    // ---- (c) test262 object-model subset (if present) ----
    // Same caveat as the corpus: many files construct exotics. Reported for the
    // breakdown, not folded into the plain-object gate.
    if let Some(w) = test262_workload() {
        println!("\n(test262 subset: {} runnable files)", w.sources.len());
        let s = measure(&w);
        let n = w.sources.len() as u64;
        report(&w, &s, &base, n);
    } else {
        println!("\n[test262 object-model subset] checkout absent — skipped (synthetic + corpus still measured)");
    }

    let plain_rate = if plain_user_created == 0 {
        1.0
    } else {
        plain_user_retained as f64 / plain_user_created as f64
    };
    println!("\n---------- PLAIN-OBJECT AGGREGATE (the common case; engine baseline removed) ----------");
    println!(
        "plain-object user created_shaped={plain_user_created}  retained={plain_user_retained}  \
         retention={:.1}%",
        plain_rate * 100.0
    );
    println!(
        "mixed-int-key chunk-map: user-retention={:.1}%  (int-key deopts={webpack_int_deopts}) \
         — the hybrid's target",
        webpack_user_rate * 100.0
    );

    // Teeth: the measurement must have actually exercised the Shaped path on a
    // non-trivial number of USER plain objects (not just the engine baseline).
    assert!(
        plain_user_created > 10_000,
        "the measurement attributed only {plain_user_created} Shaped objects to plain user \
         code — too few to draw a conclusion; the gate/workloads did not take effect"
    );

    // ---- the P4 go/no-go assertion (turns the numbers into a gate) ----
    // GO criterion: the COMMON-CASE plain-object workloads (config, class
    // instances, React elements, event/promise objects, JSON records) retain
    // user objects at a HIGH rate. Each is checked individually + in aggregate.
    for (name, urate, uc, int_key_deopts) in &user_rates {
        if name.contains("webpack") {
            // Expected: integer keys force deopts here — the hybrid's evidence.
            assert!(
                *int_key_deopts > 0,
                "[{name}] expected integer-key deopts (the mixed-key risk) but saw none"
            );
            continue;
        }
        // Only gate workloads that actually allocated a meaningful user population.
        if *uc < 1000 {
            continue;
        }
        assert!(
            *urate >= 0.97,
            "[{name}] plain USER-object retention {:.1}% < 97% — common real objects are \
             deopting; investigate before flipping the default",
            urate * 100.0
        );
    }

    // Aggregate PLAIN-object retention is the headline GO figure.
    assert!(
        plain_rate >= 0.97,
        "aggregate PLAIN-object retention {:.1}% < 97% — recommend the dense-int hybrid \
         (P4.5) before flipping CV_SHAPED_OBJ default ON",
        plain_rate * 100.0
    );

    // The chunk-map must show the int-key deopt signature (the hybrid evidence),
    // confirming the ONE low-retention case is exactly the predicted mixed-key one.
    assert!(
        webpack_int_deopts > 0,
        "the webpack chunk-map showed NO integer-key deopts — the P4 mixed-key \
         hypothesis is unconfirmed"
    );
}
