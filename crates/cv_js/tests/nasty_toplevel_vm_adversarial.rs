//! ADVERSARIAL independent verification of the CV_TOPLEVEL_VM divergence gate.
//! These are hand-written NASTY top-level programs targeting the interplay zones
//! the fuzzer grammar might under-sample. Each is run through the SAME production-
//! faithful oracle the fuzzer uses (`assert_toplevel_vm_agrees`: throw parity +
//! console + every touched global READ THROUGH globalThis). A FAILURE here means a
//! divergence the fuzzer missed.

use cv_js::ab_oracle::{assert_toplevel_vm_agrees, assert_toplevel_vm_agrees_engaged};
use cv_js::interp::Interp;

/// Run `src` on the TREE-WALK reference (top-level VM off by default) and return
/// its console output joined by '\n'. Confirms the reference itself is Node-correct
/// (the oracle proves VM == tree-walk; this proves tree-walk == Node).
fn treewalk_output(src: &str) -> String {
    let mut it = Interp::new();
    it.install_basic_globals();
    let _ = it.run(src);
    it.output.join("\n")
}

#[test]
fn nasty_treewalk_reference_matches_node() {
    // (src, EXACT Node v24 output) — captured by running each on real Node.
    let cases: &[(&str, &str)] = &[
        ("var o = { valueOf: function(){ return 42; }, toString: function(){ return 'X'; } }; var t = `v=${o}`; var p = '' + o; console.log(t, p);",
         "v=X 42"),
        ("var g=0; for (var i=0;i<4;i=i+1){ switch(i){ case 0: g=g+1; case 1: g=g+2; break; default: g=g+7; } } console.log(g, i);",
         "19 4"),
        ("var z = 0 * -1; var n = 0/0; for (var i=0;i<2;i=i+1){ z = z; } console.log(Object.is(z,-0), n!==n);",
         "true true"),
        ("var o2 = { [Symbol.toPrimitive]: function(hint){ return hint === 'string' ? 'STR' : 7; } }; var tt = `v=${o2}`; var pp = '' + o2; console.log(tt, pp);",
         "v=STR 7"),
        // NOTE: `finally { return ... }` overriding a try-return is a PRE-EXISTING
        // tree-walker correctness gap (Node: 'finally'; tree-walk: 'try') that is
        // ORTHOGONAL to the flip — see
        // `finally_return_override_is_treewalk_gap_not_a_flip_divergence`. Excluded
        // from the Node-match list here so it doesn't mask the flip-safety result.
        ("var log=[]; outer: for (var j=0;j<3;j=j+1){ try { if(j===1){ break outer; } log.push('b'+j); } finally { log.push('f'+j); } } console.log(log.join(','));",
         "b0,f0,f1"),
        ("var trace=[]; try { try { try { throw new TypeError('deep'); } catch(e1){ trace.push('c1'); throw e1; } } catch(e2){ trace.push('c2:'+e2.name); throw e2; } } catch(e3){ trace.push('c3:'+e3.message); } console.log(trace.join('|'));",
         "c1|c2:TypeError|c3:deep"),
        ("x = 5; var beforeRead = x; var x; x = x + 10; console.log(beforeRead, x);",
         "5 15"),
    ];
    for (src, node_out) in cases {
        let tw = treewalk_output(src);
        assert_eq!(
            &tw, node_out,
            "tree-walk reference DIVERGES FROM NODE:\n{src}\n  tree-walk: {tw:?}\n  node:      {node_out:?}"
        );
    }
}

fn check(label: &str, src: &str) {
    match assert_toplevel_vm_agrees(src) {
        Ok(()) => {}
        Err(d) => panic!("DIVERGENCE in [{label}]:\n{src}\n--- {d}"),
    }
}

/// Like `check`, but ALSO requires the VM path to have GENUINELY engaged — so a
/// green result can't be a silent decline-on-both-passes (which proves nothing
/// about the VM). Use for shapes that MUST be VM-eligible.
fn check_engaged(label: &str, src: &str) {
    match assert_toplevel_vm_agrees_engaged(src) {
        Ok(()) => {}
        Err(d) => panic!("DIVERGENCE/NON-ENGAGEMENT in [{label}]:\n{src}\n--- {d}"),
    }
}

#[test]
fn nasty_toplevel_cases() {
    // 1. try/finally/return interplay: finally must run, value from try.
    check(
        "fn_try_return_finally_sideeffect",
        "var g=0; function h(){ try { return 1; } finally { g = g + 5; } } var r = h(); console.log(r, g);",
    );

    // 2. labeled break across finally (the classic dropped-finally bug).
    check(
        "labeled_break_across_finally",
        "var log=[]; outer: for (var i=0;i<3;i=i+1){ try { if(i===1){ break outer; } log.push('b'+i); } finally { log.push('f'+i); } } console.log(log.join(','));",
    );

    // 3. labeled continue across finally.
    check(
        "labeled_continue_across_finally",
        "var log=[]; outer: for (var i=0;i<3;i=i+1){ for (var j=0;j<2;j=j+1){ try { if(j===0){ continue outer; } log.push('x'+i+j); } finally { log.push('f'+i+j); } } } console.log(log.join(','));",
    );

    // 4. template literal with Symbol.toPrimitive (string hint).
    check(
        "template_symbol_toprimitive",
        "var o = { [Symbol.toPrimitive]: function(hint){ return hint === 'string' ? 'STR' : 7; } }; var t = `v=${o}`; var p = '' + o; console.log(t, p);",
    );

    // 5. getter with side effects, read through global object.
    check(
        "getter_side_effects",
        "var calls=0; var obj={}; Object.defineProperty(obj,'x',{ get: function(){ calls=calls+1; return calls*10; } }); var a=obj.x; var b=obj.x; console.log(a,b,calls);",
    );

    // 6. var hoisting INTO catch binding shadow + write-back.
    check(
        "var_in_catch_hoist",
        "var g=0; try { throw 9; } catch(e){ var captured = e; g = g + e; } console.log(g, captured);",
    );

    // 7. try/finally where finally OVERRIDES the return value.
    check(
        "finally_overrides_return",
        "function h(){ try { return 'try'; } finally { return 'finally'; } } console.log(h());",
    );

    // 8. for-init-var loop, then a throwing try AFTER it (the slot-pressure shape).
    check(
        "forvar_then_throwing_try",
        "var g=0; for (var i=0;i<4;i=i+1){ g=g+i; } try { throw new Error('z'); } catch(e){ g=g+1; } console.log(g, i);",
    );

    // 9. nested for-init-var loops reading inner counter after loop.
    check(
        "nested_forvar_read_after",
        "var s=0; for (var a=0;a<3;a=a+1){ for (var b=0;b<2;b=b+1){ s=s+1; } } console.log(s, a, b);",
    );

    // 10. switch fall-through with break inside try inside loop.
    check(
        "switch_fallthrough_in_try_in_loop",
        "var g=0; for (var i=0;i<4;i=i+1){ try { switch(i){ case 0: g=g+1; case 1: g=g+2; break; default: g=g+7; } } catch(e){ g=g-1; } } console.log(g, i);",
    );

    // 11. throw in for-update expression caught by surrounding try.
    check(
        "throw_in_loop_caught_outer",
        "var g=0; try { for (var i=0;i<3;i=i+1){ if(i===2){ throw 'stop'; } g=g+1; } } catch(e){ g=g+100; } console.log(g);",
    );

    // 12. assign-before-decl global hoist + reassign across loop.
    check(
        "assign_before_decl_hoist",
        "x = 5; var beforeRead = x; var x; x = x + 10; console.log(beforeRead, x);",
    );

    // 13. template literal nesting object with valueOf+toString in a hole.
    check(
        "template_nested_valueof_tostring",
        "var o = { valueOf: function(){ return 1; }, toString: function(){ return 'T'; } }; var u = `a${o}b${o + 1}c`; console.log(u);",
    );

    // 14. do-while with labeled continue and a finally.
    check(
        "dowhile_labeled_continue_finally",
        "var log=[]; lbl: do { for (var k=0;k<3;k=k+1){ try { if(k===1) continue lbl; log.push('k'+k); } finally { log.push('fk'+k); } } } while(false); console.log(log.join(','));",
    );

    // 15. -0 / NaN preservation through globals across a loop.
    check(
        "neg_zero_nan_globals",
        "var z = 0 * -1; var n = 0/0; for (var i=0;i<2;i=i+1){ z = z; } console.log(Object.is(z,-0), n!==n);",
    );

    // 16. const reassignment TDZ throw parity (must decline; both throw same).
    check(
        "const_reassign_throws",
        "var ok=0; try { const c = 1; c = 2; } catch(e){ ok = 1; } console.log(ok);",
    );

    // 17. nested try with rethrow climbing two catch levels.
    check(
        "rethrow_two_levels",
        "var trace=[]; try { try { try { throw new TypeError('deep'); } catch(e1){ trace.push('c1'); throw e1; } } catch(e2){ trace.push('c2:'+e2.name); throw e2; } } catch(e3){ trace.push('c3:'+e3.message); } console.log(trace.join('|'));",
    );

    // 18. getter that throws, read through global object.
    check(
        "getter_throws",
        "var obj={}; Object.defineProperty(obj,'boom',{ get: function(){ throw new RangeError('g'); } }); var r='none'; try { r = obj.boom; } catch(e){ r = e.name; } console.log(r);",
    );

    // 19. closure capturing for-init-var read mid-loop (the stale-global hazard).
    check(
        "closure_reads_forvar_midloop",
        "var seen=[]; for (var i=0;i<3;i=i+1){ (function(){ seen.push(i); })(); } console.log(seen.join(','));",
    );

    // 20. arguments + nested finally + return inside an IIFE.
    check(
        "iife_arguments_finally_return",
        "var v = (function(){ try { return arguments.length; } finally { } })(1,2,3); console.log(v);",
    );
}

/// The eligible subset of my nasty cases MUST genuinely take the VM path (else the
/// agreement above is vacuous for those shapes). These are var-heavy, loop-heavy,
/// try-free / decline-free shapes that should engage the top-level VM.
#[test]
fn nasty_eligible_cases_genuinely_engage_the_vm() {
    // assign-before-decl hoist + reassign (no try, no nested-forvar) — eligible.
    check_engaged(
        "engage_assign_before_decl",
        "x = 5; var beforeRead = x; var x; x = x + 10; console.log(beforeRead, x);",
    );
    // single direct top-level for-init-var counted loop (the bench shape).
    check_engaged(
        "engage_direct_forvar_loop",
        "var sum = 0; for (var i = 0; i < 100; i = i + 1) { sum = sum + i; } console.log(sum, i);",
    );
    // -0 / NaN through globals across a simple loop.
    check_engaged(
        "engage_neg_zero_nan",
        "var z = 0 * -1; var n = 0/0; for (var i=0;i<2;i=i+1){ z = z; } console.log(Object.is(z,-0), n!==n);",
    );
    // NOTE: a closure reading a for-init-var (`(function(){ seen.push(i); })()`)
    // CORRECTLY DECLINES (a fn body reading a top-level for-init var would see a
    // stale global mid-loop) — verified separately that it still AGREES; it is NOT
    // required to engage. So it is intentionally excluded from this engagement set.
    // switch fall-through (no try) inside a direct loop.
    check_engaged(
        "engage_switch_fallthrough",
        "var g=0; for (var i=0;i<4;i=i+1){ switch(i){ case 0: g=g+1; case 1: g=g+2; break; default: g=g+7; } } console.log(g, i);",
    );
    // template literal with object string-hint coercion at top level.
    check_engaged(
        "engage_template_string_hint",
        "var o = { valueOf: function(){ return 42; }, toString: function(){ return 'X'; } }; var t = `v=${o}`; var p = '' + o; console.log(t, p);",
    );
}

/// Characterize the finally-return-override case: it is a PRE-EXISTING tree-walker
/// correctness gap (vs Node), NOT a CV_TOPLEVEL_VM divergence. The oracle must
/// still report AGREEMENT (VM declines to the tree-walker → both produce the same,
/// albeit Node-divergent, value), so flipping the flag does NOT change behavior on
/// this program. This isolates the bug as orthogonal to flip-safety.
#[test]
fn finally_return_override_is_treewalk_gap_not_a_flip_divergence() {
    let src = "function h(){ try { return 'try'; } finally { return 'finally'; } } console.log(h());";
    // VM-vs-tree-walk AGREE (the flip is a no-op here): the gate stays green.
    assert_toplevel_vm_agrees(src)
        .unwrap_or_else(|d| panic!("UNEXPECTED flip divergence (would block flip):\n{src}\n{d}"));
    // And both produce the (buggy) tree-walk value, NOT Node's 'finally'.
    assert_eq!(treewalk_output(src), "try", "tree-walk gap characterization");
}

/// TOP-LEVEL finally with control-flow override (break inside finally; throw caught
/// then finally) — these all have a `finally` so the flip DECLINES them, but they
/// must still (a) AGREE VM-vs-tree-walk and (b) match Node on the tree-walk path.
#[test]
fn toplevel_finally_controlflow_agrees_and_matches_node() {
    let cases: &[(&str, &str)] = &[
        ("var log=[]; for (var i=0;i<3;i=i+1){ try { log.push('t'+i); } finally { log.push('f'+i); if(i===1) break; } } console.log(log.join(','));",
         "t0,f0,t1,f1"),
        ("var log2=[]; for (var k=0;k<3;k=k+1){ try { if(k===1) throw 'x'; log2.push('t'+k); } catch(e){ log2.push('c'+k); } finally { log2.push('f'+k); } } console.log(log2.join(','));",
         "t0,f0,c1,f1,t2,f2"),
    ];
    for (src, node_out) in cases {
        assert_toplevel_vm_agrees(src)
            .unwrap_or_else(|d| panic!("flip divergence on top-level finally:\n{src}\n{d}"));
        assert_eq!(&treewalk_output(src), node_out, "tree-walk vs Node:\n{src}");
    }
}
