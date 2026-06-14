//! M3.2 P5 — bounded-ShapeTable + no-leak SOAK (the default-on safety gate).
//!
//! P5 flips `CV_SHAPED_OBJ` default ON. The risk of a shared, append-only
//! `ShapeTable` is that a long session could mint shapes WITHOUT BOUND (one per
//! distinct key sequence forever) or leak Shaped objects whose references were
//! dropped. This soak proves neither happens.
//!
//! It is an INTEGRATION test (its own process), which is what makes the global
//! `OnceLock` gates (`shaped_obj_enabled`, `gc_enabled`, `shaped_stats_enabled`)
//! deterministic: this single test sets the env at the very top, before any code
//! path reads those locks. (The default is now ON, so the soak validates the
//! shipping configuration; setting `CV_SHAPED_OBJ=1` explicitly just pins it
//! regardless of how the runner is invoked.)
//!
//! What it simulates: a sustained session that creates + mutates + DROPS many
//! objects across a BOUNDED set of shapes — plain config objects of varied key
//! sets, class instances, frozen objects (deopt by sentinel), webpack-style
//! integer-key maps (deopt by int-key), and churny dynamic-key bags that exceed
//! the slot cap (deopt by cap) — over a long loop, with periodic GC.
//!
//! What it asserts:
//!   (1) BOUNDED: the global `ShapeTable` size after a 50x-larger churn is no
//!       bigger than after a small warmup (+ a tiny constant) — it does NOT grow
//!       with the number of OBJECTS allocated. Dynamic-key/integer-key bags deopt
//!       rather than minting unbounded shapes.
//!   (2) NO LEAK: the count of LIVE registered JS objects after a final GC is
//!       bounded (dropped-reference objects are reclaimed) — it does not grow
//!       with the total number of objects ever created.
//!   (3) NO PANIC: the whole soak completes.
//!
//! Run with output:
//!   cargo test -p cv_js --test shaped_soak_p5 -- --nocapture

use cv_js::interp::{gc_enabled, gc_live_object_count, Interp, Value};
use cv_js::ordered::{shaped_obj_enabled, shaped_stats, ShapedStats};
use cv_js::shapes::global_shape_count;

/// One round of the workload: build + mutate + read many objects across the
/// bounded set of shape patterns, all in `iters` scopes that DROP at the end of
/// each interpreter run. Each source allocates objects in a loop, so a single
/// call exercises many object lifetimes. Returns the snapshot stats delta is the
/// caller's job; this just runs the churn.
fn churn_round(iters: usize) {
    // A FIXED palette of shape patterns. The number of DISTINCT shapes these
    // can mint is fixed (bounded by the palette), no matter how many times we
    // run them — that is the property the bound depends on.
    let sources: &[&str] = &[
        // plain config objects, several distinct key sets (each a distinct shape)
        "var s=0; for(var i=0;i<200;i=i+1){ var o={width:1,height:2,color:'r',label:'x'}; s=s+o.width; } s;",
        "var s=0; for(var i=0;i<200;i=i+1){ var o={x:i,y:i+1,z:0}; s=s+o.x; } s;",
        "var s=0; for(var i=0;i<200;i=i+1){ var o={type:'div',key:'k',props:null,ref:null}; s=s+o.type.length; } s;",
        "var s=0; for(var i=0;i<200;i=i+1){ var o={a:1,b:2,c:3,d:4,e:5,f:6}; s=s+o.f; } s;",
        // class instances (constructor this.x=.. shape; reuses the same shape)
        "function P(x,y){this.x=x;this.y=y;this.z=0;} var s=0; for(var i=0;i<200;i=i+1){ var p=new P(i,i); s=s+p.x; } s;",
        "class V{constructor(a,b,c){this.a=a;this.b=b;this.c=c;}} var s=0; for(var i=0;i<200;i=i+1){ var v=new V(i,i,i); s=s+v.a; } s;",
        // frozen objects — deopt by sentinel (no shape growth past the base shape)
        "var n=0; for(var i=0;i<200;i=i+1){ var o={p:1,q:2}; Object.freeze(o); n=n+o.p; } n;",
        // webpack-style integer-key maps — deopt on the FIRST integer key, so
        // they NEVER mint per-object shapes (the unbounded-shape risk's antidote)
        "var c=0; for(var i=0;i<200;i=i+1){ var m={}; m[0]=1; m[1]=2; m[2]=3; m.app=1; c=c+m.app; } c;",
        // churny DYNAMIC-key bag that blows past the slot cap (128) → deopt by
        // cap, so it does NOT mint one shape per added key without bound
        "var t=0; for(var i=0;i<20;i=i+1){ var bag={}; for(var k=0;k<160;k=k+1){ bag['f'+k]=k; } t=t+bag.f0; } t;",
        // JSON.parse output shapes (API records — all string keys, one shape)
        "var sum=0; for(var i=0;i<200;i=i+1){ var o=JSON.parse('{\"id\":1,\"name\":\"a\",\"ok\":true}'); sum=sum+o.id; } sum;",
    ];

    for _ in 0..iters {
        for src in sources {
            // Fresh Interp per source so the objects it allocates fall out of
            // scope and become collectable when the Interp drops — exactly the
            // "create, use, drop" lifecycle of a real session.
            let mut i = Interp::new();
            i.install_basic_globals();
            i.install_json();
            // Ignore per-snippet errors: the MEASUREMENT is about allocation +
            // shape behavior, not whether every snippet completes.
            let _ = i.run_completion_value(src);
            // Reclaim this round's garbage so live counts reflect reachable-only.
            let _ = i.gc_collect(&[]);
        }
    }
}

#[test]
fn shaped_soak_bounded_and_no_leak() {
    // CRITICAL: pin the gates ON before ANY code touches their OnceLocks. Edition
    // 2024: `set_var` is `unsafe`; sound here because this is the FIRST thing the
    // single-threaded test binary does — no other thread reads the env yet.
    unsafe {
        std::env::set_var("CV_SHAPED_OBJ", "1"); // the default is now on; pin it
        std::env::set_var("CV_GC", "1"); // need reclamation for the leak metric
        std::env::set_var("CV_SHAPED_STATS", "1"); // observe deopt triggers
    }
    assert!(
        shaped_obj_enabled(),
        "CV_SHAPED_OBJ must be ON for the soak to exercise the Shaped store"
    );
    assert!(
        gc_enabled(),
        "CV_GC must be ON for the leak metric (reclamation) to be meaningful"
    );

    println!("\n========== M3.2 P5 — bounded-ShapeTable + no-leak SOAK ==========");

    // ---- WARMUP: run the palette enough to mint EVERY shape it ever will. ----
    // After this, the ShapeTable holds the program's full (finite) shape set.
    const WARMUP_ROUNDS: usize = 2;
    churn_round(WARMUP_ROUNDS);
    let shapes_after_warmup = global_shape_count();
    let _ = std::hint::black_box(shapes_after_warmup);
    println!("ShapeTable size after warmup ({WARMUP_ROUNDS} rounds): {shapes_after_warmup}");

    // ---- SUSTAINED CHURN: 50x more rounds. If shapes were minted per-object, ----
    // this would balloon the table by ~50x. Bounded ⇒ it barely moves.
    const SOAK_ROUNDS: usize = 100;
    churn_round(SOAK_ROUNDS);
    let shapes_after_soak = global_shape_count();
    println!(
        "ShapeTable size after soak ({SOAK_ROUNDS} more rounds, {}x warmup): {shapes_after_soak}",
        SOAK_ROUNDS / WARMUP_ROUNDS
    );

    // Report the deopt-trigger breakdown — proves the dynamic/int/over-cap bags
    // deopted (rather than minting shapes), which is WHY the table stayed bounded.
    let s: ShapedStats = shaped_stats();
    println!(
        "deopts over soak: int-key={ik} sentinel={ss} cap={cap} (created_shaped={cs})",
        ik = s.deopted_integer_key,
        ss = s.deopted_sentinel_stamp,
        cap = s.deopted_cap_exceeded,
        cs = s.created_shaped,
    );

    // ---- ASSERTION (1): the table is BOUNDED. ----
    // It may grow by a tiny constant between warmup and soak (e.g. a shape only
    // reachable on a later code path), but it must NOT scale with object count.
    // The soak allocated ~50x the warmup's objects; allow at most a small
    // ADDITIVE slack (the empty-root + a handful of late shapes), never a factor.
    let slack = 64usize;
    assert!(
        shapes_after_soak <= shapes_after_warmup + slack,
        "ShapeTable grew from {shapes_after_warmup} to {shapes_after_soak} across a 50x churn \
         (slack {slack}) — shapes are being minted per-object (UNBOUNDED); the bound is broken"
    );
    // Also a hard absolute ceiling: the bounded palette can't produce thousands
    // of shapes (the churny bag deopts at the 128 cap, so its chain is ≤128).
    assert!(
        shapes_after_soak < 1000,
        "ShapeTable holds {shapes_after_soak} shapes — far more than the bounded palette can \
         justify; a dynamic-key path is minting shapes without deopting"
    );

    // The over-cap and integer-key deopts MUST have fired (they are the mechanism
    // that keeps the table bounded under churny/dynamic keys).
    assert!(
        s.deopted_cap_exceeded > 0,
        "the over-cap churny bag did NOT deopt (cap deopts = 0) — an unbounded-shape path exists"
    );
    assert!(
        s.deopted_integer_key > 0,
        "the integer-key webpack maps did NOT deopt (int-key deopts = 0) — unexpected"
    );

    // ---- ASSERTION (2): NO LEAK — live objects reclaimed, count bounded. ----
    // Each churn round dropped its objects + GC'd. A separate fresh Interp ran
    // last; its objects are also droppable. Run one more whole round, GC, and
    // confirm the LIVE registered-object count is small (does NOT scale with the
    // ~tens-of-thousands of objects the soak created). A leak would leave them
    // all live.
    let live_before = gc_live_object_count();
    churn_round(2);
    let live_after = gc_live_object_count();
    println!("live GC objects: before extra round={live_before}  after={live_after}");
    // Tens of thousands of objects were created across the soak; a leak would
    // show them all here. A bounded reclaimer keeps this in the low hundreds.
    assert!(
        live_after < 5_000,
        "after the soak + GC, {live_after} JS objects are still LIVE — that scales with the \
         total created (a LEAK); dropped Shaped objects are not being reclaimed"
    );

    // ---- ASSERTION (3): NO PANIC — reaching here means the soak completed. ----
    // The Shaped store must have actually been exercised (not silently a no-op).
    assert!(
        s.created_shaped > 1_000,
        "the soak created only {} Shaped objects — the Shaped path did not take effect",
        s.created_shaped
    );

    // Final sanity: an object built right now is genuinely Shaped (the default is
    // on), proving this whole soak measured the shipping configuration.
    let mut i = Interp::new();
    i.install_basic_globals();
    let v = i
        .run_completion_value("var o={alpha:1,beta:2,gamma:3}; o;")
        .expect("build a plain object");
    if let Value::Object(o) = &v {
        assert!(
            o.borrow().stored_shape_id().is_some(),
            "a plain object is NOT Shaped under the default — the flip did not take effect"
        );
        assert_ne!(
            o.borrow().stored_shape_id(),
            Some(cv_js::shapes::DICT_SHAPE),
            "a plain object DEOPTED unexpectedly under the default"
        );
    } else {
        panic!("expected an object completion value");
    }

    println!("\nSOAK PASSED: ShapeTable bounded ({shapes_after_warmup} -> {shapes_after_soak}), \
              no leak (live={live_after}), no panic.");
}
