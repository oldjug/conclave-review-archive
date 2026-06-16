//! Differential FUZZER — top-level register-VM (`CV_TOPLEVEL_VM`) vs tree-walk
//! byte-identity gate.
//!
//! ───────────────────────────── WHY THIS EXISTS ────────────────────────────
//! Three rounds of "the test corpus passes, it's safe to flip `CV_TOPLEVEL_VM`"
//! turned out FALSE when probed harder:
//!   1. a false-green oracle (it read globals via the post-flush snapshot, not the
//!      production `globalThis.X` read a page sees mid-run),
//!   2. then 22 curated-corpus divergences once the oracle was made
//!      production-faithful,
//!   3. then `try`/`finally` + abrupt-completion divergences BEYOND any corpus.
//!
//! A FIXED test list cannot prove byte-identity for a whole LANGUAGE. A fuzzer
//! can. This is how real engines (V8/SpiderMonkey/JSC) validate a new tier: a
//! grammar generates a torrent of varied programs and every one is diffed against
//! the reference. Here the REFERENCE is the tree-walker (`interp.rs`) and the
//! CANDIDATE is the top-level register-VM path (`try_run_toplevel_vm`); the
//! observation is exactly the production-faithful one the curated oracle uses
//! (`ab_oracle::assert_toplevel_vm_agrees`): completion / `console.*` output /
//! throw parity / the final value of every touched global READ THROUGH the global
//! object.
//!
//! ───────────────────────────── DESIGN ─────────────────────────────────────
//! * Deterministic, seeded `SplitMix64`/`xorshift` RNG (no `rand` dep, no clock,
//!   no I/O — `Date`/`random` are unavailable in the workflow, but this runs in
//!   Rust test code where a seeded RNG is the right tool).
//! * A recursive grammar (`Gen`) emits whole top-level programs drawn from the
//!   construct families that have bitten us AND broadly across the language:
//!   var/let/const + hoisting + assign-before-decl, for/while/do loops with
//!   break/continue (labeled + plain), try/catch/finally (incl. break/continue/
//!   throw/return crossing the boundary), nested functions/closures/arrows,
//!   generators, classes, arithmetic (int+float, ToPrimitive/coercion, pow/mod
//!   special values, NaN/Infinity/-0), template literals, array/object literals +
//!   methods, Proxy/Map/Set/WeakRef, `arguments`, getters/setters, switch,
//!   throw-to-outer-catch.
//! * Every generated program ALWAYS PARSES (the grammar only emits well-formed
//!   syntax) and is bounded in depth + statement count so the batch runtime stays
//!   short and the machine stays safe (in-process, bounded, offline — never a
//!   spawned process loop).
//! * For each program: `assert_toplevel_vm_agrees` — a divergence is a FAILURE
//!   with the minimal reproducer printed.
//!
//! The broad batch (`fuzz_toplevel_vm_broad_batch`) is a permanent regression gate.
//! Non-vacuity is proven by `fuzz_catches_reintroduced_bug` (it un-declines a
//! known-divergent construct via a fault-injection hook and shows the fuzzer turns
//! RED, then restores). NO STUBS — every program runs real code through both real
//! tiers.

#![cfg(test)]

use crate::ab_oracle::{assert_toplevel_vm_agrees, Divergence};

// ───────────────────────────── SEEDED RNG ─────────────────────────────────

/// A tiny, fast, deterministic PRNG (`SplitMix64`). Same seed ⇒ same stream ⇒
/// the whole batch is reproducible bit-for-bit across runs and machines. No
/// external crate, no entropy source, no clock.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, n)` (n must be > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    /// True with probability `num/den`.
    fn chance(&mut self, num: u32, den: u32) -> bool {
        self.below(den as usize) < num as usize
    }

    /// Pick one of `choices` uniformly.
    fn pick<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        &choices[self.below(choices.len())]
    }
}

// ───────────────────────────── GENERATOR ──────────────────────────────────

/// Recursive program generator. Carries a fuel budget (max nodes) + a depth cap
/// so generation always terminates and the program is small enough to run two
/// tiers + the global-object probe quickly. Tracks declared global names so it can
/// READ them back (assign-before-decl, hoisting, redeclaration), and a label stack
/// for labeled break/continue.
struct Gen<'r> {
    rng: &'r mut Rng,
    out: String,
    fuel: i32,
    depth: u32,
    /// Top-level `var` names already introduced (for read-back / reassignment).
    globals: Vec<String>,
    /// Top-level function names introduced (read ONLY via a direct call — eligibility
    /// declines a value-position use, which is correct but vacuous to generate often).
    fns: Vec<String>,
    /// Active loop/switch labels available to break/continue.
    labels: Vec<String>,
    /// Whether we're currently inside a loop or switch (plain break/continue legal).
    loop_depth: u32,
    /// Monotonic counter for fresh identifier suffixes.
    counter: u32,
    /// PROTECTED names — loop counters and while/do counters. They are globals (so
    /// readable through `globalThis`) but are NEVER assignment targets in a body,
    /// so a body can never mutate a loop counter and create a runaway loop. (A
    /// runaway loop is not a correctness divergence — both tiers time out on the
    /// wall-clock watchdog — but it makes the batch slow and the line count
    /// wall-clock-non-deterministic, so the grammar simply never emits one.)
    protected: Vec<String>,
    /// When true, the grammar is ALLOWED to emit DECLINING constructs (let/const,
    /// block-nested fns, escaping callbacks, Proxy/Map/Set/WeakRef, getters/setters,
    /// generators, classes, method-rebind, defineProperty accessors). These make
    /// the program VM-INELIGIBLE — it tree-walks on both passes — which still
    /// validates that the DECLINE fallback is byte-identical. When false, the
    /// grammar stays within the VM-eligible subset so the program ENGAGES the VM
    /// path (so a healthy fraction of the batch genuinely exercises the VM tier).
    diverge_ok: bool,
    /// In non-diverge mode, a for-init-`var` loop CO-OCCURRING with a `try` declines
    /// (the fuzzer-found write-back bug). To keep eligible programs engaging, each
    /// non-diverge program picks ONE of {avoid all trys, avoid all for-loops} so it
    /// never produces the declining combination — giving full coverage of BOTH the
    /// try family and the loop family across the batch, just not mixed in one
    /// program. (Diverge-mode ignores these and may mix freely → declines.)
    avoid_try: bool,
    avoid_for: bool,
}

/// The maximum recursion depth of statement/expression nesting. Keeps programs
/// shallow enough that two-tier execution + the probe is sub-millisecond.
const MAX_DEPTH: u32 = 5;

impl<'r> Gen<'r> {
    fn new(rng: &'r mut Rng, fuel: i32) -> Self {
        Gen {
            rng,
            out: String::new(),
            fuel,
            depth: 0,
            globals: Vec::new(),
            fns: Vec::new(),
            labels: Vec::new(),
            loop_depth: 0,
            counter: 0,
            protected: Vec::new(),
            diverge_ok: false,
            avoid_try: false,
            avoid_for: false,
        }
    }

    /// A name that may be READ (an assignable global OR a protected loop counter).
    /// Returns `None` if nothing is available.
    fn pick_readable(&mut self) -> Option<String> {
        let total = self.globals.len() + self.protected.len();
        if total == 0 {
            return None;
        }
        let idx = self.rng.below(total);
        if idx < self.globals.len() {
            Some(self.globals[idx].clone())
        } else {
            Some(self.protected[idx - self.globals.len()].clone())
        }
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    fn spend(&mut self) -> bool {
        self.fuel -= 1;
        self.fuel > 0 && self.depth < MAX_DEPTH
    }

    /// Generate a whole top-level program. Always ends by logging the running
    /// state of a few touched globals so the observation set is non-empty (the
    /// oracle then reads each through `globalThis`), giving the fuzzer teeth on
    /// the global-visibility axis specifically.
    fn program(&mut self) -> String {
        // Seed a couple of base globals so reads/reassigns have material.
        let base = self.fresh("g");
        self.globals.push(base.clone());
        self.out.push_str(&format!("var {base} = 0;\n"));

        let n = 2 + self.rng.below(6); // 2..7 top-level statements
        for _ in 0..n {
            if self.fuel <= 0 {
                break;
            }
            self.statement();
        }

        // Trailing observation: log a handful of globals so the oracle's
        // touched-global / globalThis-read legs always have something to compare.
        if !self.globals.is_empty() {
            let k = 1 + self.rng.below(self.globals.len().min(4));
            let mut line = String::from("console.log(");
            for i in 0..k {
                if i > 0 {
                    line.push_str(", ");
                }
                line.push_str(&self.globals[i % self.globals.len()]);
            }
            line.push_str(");\n");
            self.out.push_str(&line);
        }
        std::mem::take(&mut self.out)
    }

    /// Emit one statement (with indentation by depth).
    fn statement(&mut self) {
        if !self.spend() {
            // Out of fuel: a harmless terminal statement.
            self.emit_assign_existing();
            return;
        }
        // In diverge-mode, occasionally emit a whole DIVERGING-CONSTRUCT family
        // (Proxy/Map/Set/WeakRef, generator, class, getter/setter, arguments) so the
        // decline paths for those get fuzz coverage (they must tree-walk on both
        // passes — byte-identical fallback).
        if self.diverge_ok && self.depth == 0 && self.rng.chance(22, 100) {
            self.stmt_diverging_construct();
            return;
        }
        // Weighted statement menu. Numbers chosen so the BITTEN families
        // (loops/try/closures/coercion) get heavy coverage.
        let choice = self.rng.below(100);
        match choice {
            0..=14 => self.stmt_var_decl(),
            15..=22 => self.stmt_assign(),
            // For-loops: declined when this program also has a try (avoid_for keeps
            // try-heavy eligible programs free of for-init-var loops).
            23..=37 => {
                if self.avoid_for {
                    self.stmt_while_loop();
                } else {
                    self.stmt_for_loop();
                }
            }
            38..=45 => self.stmt_while_loop(),
            46..=50 => self.stmt_do_while(),
            // Trys: declined when this program also has a for-init-var loop
            // (avoid_try keeps loop-heavy eligible programs free of trys).
            51..=63 => {
                if self.avoid_try {
                    self.stmt_if();
                } else {
                    self.stmt_try();
                }
            }
            64..=70 => self.stmt_if(),
            71..=76 => self.stmt_switch(),
            77..=83 => self.stmt_function_decl(),
            84..=88 => self.stmt_console_log(),
            // A self-catching throw is a throwing try-body → DECLINES. Keep it in
            // diverge-mode (decline-fallback coverage); in the eligible subset emit a
            // console.log instead (or, for try-heavy programs, a real catch).
            89..=92 => {
                if self.diverge_ok {
                    self.stmt_throw();
                } else if self.avoid_for {
                    // try-heavy eligible: a locally-caught throw is a throwing try-
                    // body → DECLINES even without a for-loop, so still use a plain
                    // log. (Throw-to-catch is covered in diverge-mode.)
                    self.stmt_console_log();
                } else {
                    self.stmt_console_log();
                }
            }
            93..=96 => self.stmt_block(),
            // Labeled nested loops are for-init-var loops → route to a plain if for
            // try-heavy (avoid_for) eligible programs.
            _ => {
                if self.avoid_for {
                    self.stmt_if();
                } else {
                    self.stmt_labeled_loop();
                }
            }
        }
    }

    fn indent(&self) -> String {
        "  ".repeat(self.depth as usize)
    }

    // ── var / let / const ──────────────────────────────────────────────────
    fn stmt_var_decl(&mut self) {
        let ind = self.indent();
        // Mostly `var` (the eligible shape); in diverge-mode sometimes let/const
        // (correctly declined — must STILL agree byte-for-byte via the tree-walk
        // fallback).
        let kind = if !self.diverge_ok || self.rng.chance(70, 100) {
            "var"
        } else if self.rng.chance(50, 100) {
            "let"
        } else {
            "const"
        };
        let name = self.fresh("v");
        let e = self.expr(0);
        self.out.push_str(&format!("{ind}{kind} {name} = {e};\n"));
        if kind == "var" {
            self.globals.push(name);
        }
    }

    // ── assignment to an existing or fresh global ──────────────────────────
    fn stmt_assign(&mut self) {
        if self.globals.is_empty() {
            self.stmt_var_decl();
            return;
        }
        let ind = self.indent();
        let name = self.globals[self.rng.below(self.globals.len())].clone();
        let op = self.rng.pick(&["=", "+=", "-=", "*=", "|=", "&="]).to_string();
        let e = self.expr(0);
        self.out.push_str(&format!("{ind}{name} {op} {e};\n"));
    }

    fn emit_assign_existing(&mut self) {
        let ind = self.indent();
        if let Some(name) = self.globals.first().cloned() {
            self.out.push_str(&format!("{ind}{name} = {name} + 1;\n"));
        } else {
            self.out.push_str(&format!("{ind};\n"));
        }
    }

    // ── for loop (counted, the perf-critical shape + break/continue) ───────
    fn stmt_for_loop(&mut self) {
        // A for-init-`var` loop NESTED inside another loop hits the for-init-var
        // write-back DECLINE (the fuzzer-found VM bug). In non-diverge mode, when
        // already inside a loop, emit a WHILE loop instead (its counter is a plain
        // `var` assignment, not a for-init, so no write-back bug → stays eligible).
        // In diverge-mode, the nested for is allowed (it declines → tree-walk
        // fallback, which the oracle proves byte-identical).
        if self.loop_depth > 0 && !self.diverge_ok {
            self.stmt_while_loop();
            return;
        }
        let ind = self.indent();
        let i = self.fresh("i");
        let bound = 1 + self.rng.below(6);
        let step = if self.rng.chance(70, 100) { 1 } else { 2 };
        // Top-level for-init `var` (NOT let — let at top level is declined but
        // tested separately; here we keep the counted numeric loop eligible).
        self.out.push_str(&format!(
            "{ind}for (var {i} = 0; {i} < {bound}; {i} = {i} + {step}) {{\n"
        ));
        // The loop counter is PROTECTED (readable, never an assignment target) so
        // the body can never mutate it into a runaway loop.
        self.protected.push(i.clone());
        self.depth += 1;
        self.loop_depth += 1;
        let body_n = 1 + self.rng.below(3);
        for _ in 0..body_n {
            if self.rng.chance(18, 100) {
                self.emit_break_or_continue(&i);
            } else {
                self.statement();
            }
        }
        self.loop_depth -= 1;
        self.depth -= 1;
        self.out.push_str(&format!("{ind}}}\n"));
    }

    fn emit_break_or_continue(&mut self, ctr: &str) {
        let ind = self.indent();
        let kw = if self.rng.chance(50, 100) { "break" } else { "continue" };
        // Guard the jump so the loop still does real work most iterations.
        let g = 1 + self.rng.below(3);
        self.out.push_str(&format!("{ind}if ({ctr} === {g}) {{ {kw}; }}\n"));
    }

    fn stmt_while_loop(&mut self) {
        let ind = self.indent();
        let c = self.fresh("w");
        let bound = 1 + self.rng.below(6);
        self.out.push_str(&format!("{ind}var {c} = 0;\n"));
        self.out.push_str(&format!("{ind}while ({c} < {bound}) {{\n"));
        self.depth += 1;
        self.loop_depth += 1;
        // Always advance the counter so the loop terminates.
        self.out.push_str(&format!("{}{c} = {c} + 1;\n", self.indent()));
        // PROTECTED only AFTER the advance line is emitted (the advance is the
        // generator's own write, not a body assignment), so the body can't touch it.
        self.protected.push(c.clone());
        let body_n = self.rng.below(2);
        for _ in 0..body_n {
            self.statement();
        }
        if self.rng.chance(20, 100) {
            self.emit_break_or_continue(&c);
        }
        self.loop_depth -= 1;
        self.depth -= 1;
        self.out.push_str(&format!("{ind}}}\n"));
    }

    fn stmt_do_while(&mut self) {
        let ind = self.indent();
        let c = self.fresh("d");
        let bound = 1 + self.rng.below(5);
        self.out.push_str(&format!("{ind}var {c} = 0;\n"));
        self.out.push_str(&format!("{ind}do {{\n"));
        self.depth += 1;
        self.loop_depth += 1;
        self.out.push_str(&format!("{}{c} = {c} + 1;\n", self.indent()));
        self.protected.push(c.clone());
        let body_n = self.rng.below(2);
        for _ in 0..body_n {
            self.statement();
        }
        self.loop_depth -= 1;
        self.depth -= 1;
        self.out.push_str(&format!("{ind}}} while ({c} < {bound});\n"));
    }

    // ── labeled loop (labeled break/continue crossing nesting) ─────────────
    fn stmt_labeled_loop(&mut self) {
        // A labeled outer for + nested for is two nested for-init-var loops →
        // DECLINES. To keep labeled break/continue VM-eligible in non-diverge mode,
        // only emit at the top level and use a WHILE for the inner loop (no nested
        // for-init-var). In diverge-mode (or nested), the for/for shape is allowed
        // (declines → tree-walk fallback).
        let use_while_inner = !self.diverge_ok && self.loop_depth == 0;
        if self.loop_depth > 0 && !self.diverge_ok {
            // Nested labeled-loop would decline; emit a plain (now while) loop body.
            self.stmt_while_loop();
            return;
        }
        let ind = self.indent();
        let lbl = self.fresh("L");
        let outer = self.fresh("oi");
        let inner = self.fresh("ii");
        self.out.push_str(&format!("{ind}{lbl}: for (var {outer} = 0; {outer} < 3; {outer} = {outer} + 1) {{\n"));
        self.protected.push(outer.clone());
        self.depth += 1;
        self.loop_depth += 1;
        self.labels.push(lbl.clone());
        let i2 = self.indent();
        if use_while_inner {
            // WHILE inner (no nested for-init-var) — stays eligible.
            self.out.push_str(&format!("{i2}var {inner} = 0;\n"));
            self.out.push_str(&format!("{i2}while ({inner} < 3) {{\n"));
            self.depth += 1;
            let i3 = self.indent();
            self.out.push_str(&format!("{i3}{inner} = {inner} + 1;\n"));
            let kw = if self.rng.chance(50, 100) { "break" } else { "continue" };
            self.out.push_str(&format!("{i3}if ({outer} + {inner} === 3) {{ {kw} {lbl}; }}\n"));
            if let Some(g) = self.globals.first().cloned() {
                self.out.push_str(&format!("{i3}{g} = {g} + 1;\n"));
            }
            self.depth -= 1;
            self.protected.push(inner.clone());
        } else {
            self.out.push_str(&format!("{i2}for (var {inner} = 0; {inner} < 3; {inner} = {inner} + 1) {{\n"));
            self.depth += 1;
            let i3 = self.indent();
            let kw = if self.rng.chance(50, 100) { "break" } else { "continue" };
            self.out.push_str(&format!("{i3}if ({outer} + {inner} === 2) {{ {kw} {lbl}; }}\n"));
            if let Some(g) = self.globals.first().cloned() {
                self.out.push_str(&format!("{i3}{g} = {g} + 1;\n"));
            }
            self.depth -= 1;
        }
        self.out.push_str(&format!("{i2}}}\n"));
        self.labels.pop();
        self.loop_depth -= 1;
        self.depth -= 1;
        self.out.push_str(&format!("{ind}}}\n"));
    }

    // ── DIVERGING CONSTRUCT FAMILIES (diverge-mode only; all DECLINE) ──────────
    //
    // Each emits a self-contained, syntactically-valid snippet that lands a result
    // in a fresh global, exercising a construct family whose top-level-VM lowering
    // diverges and is therefore DECLINED. The oracle proves the tree-walk fallback
    // is byte-identical on both passes.
    fn stmt_diverging_construct(&mut self) {
        let ind = self.indent();
        let g = self.fresh("dv");
        // Every snippet computes a PRIMITIVE result inside an IIFE and stores only
        // that primitive into the observed global `{g}`. The IIFE keeps the program
        // VM-INELIGIBLE (the diverging construct lives in a function body the decline
        // scan descends into) WITHOUT leaking an engine-internal object (Map/Set/
        // WeakRef carry process-global internal counters like `_weakRefId` that are
        // NOT spec-observable and differ between the oracle's two passes — comparing
        // them would be a false divergence). So the global is always a plain number/
        // string, deep-compared safely.
        let body = match self.rng.below(10) {
            0 => "var m = new Map(); m.set('a', 1); m.set('b', 2); return m.size;".to_string(),
            1 => "var s = new Set(); s.add(1); s.add(2); s.add(2); return s.size;".to_string(),
            2 => "var o = { v: 7 }; var wr = new WeakRef(o); return wr.deref().v;".to_string(),
            3 => "var p = new Proxy({}, { get: function(t, k){ return 42; } }); return p.anything;".to_string(),
            4 => "function* gen(){ var a = yield 1; yield a + 10; } var it = gen(); var r1 = it.next(); var r2 = it.next(5); return r1.value + r2.value;".to_string(),
            5 => "class C { constructor(n){ this.n = n; } dbl(){ return this.n * 2; } } var c = new C(8); return c.dbl() + (c instanceof C ? 1 : 0);".to_string(),
            6 => "var obj = {}; var hidden = 9; Object.defineProperty(obj, 'x', { get: function(){ return hidden; } }); return obj.x;".to_string(),
            7 => "var s = 0; for (var i = 0; i < arguments.length; i = i + 1) { s = s + arguments[i]; } return s;".to_string(),
            8 => "return ''.toUpperCase.call('hi').length;".to_string(),
            _ => "var k = {}; var wm = new WeakMap(); wm.set(k, 5); return wm.get(k);".to_string(),
        };
        // The `arguments` snippet (case 7) needs args; the rest ignore them.
        self.out.push_str(&format!("{ind}var {g} = (function(){{ {body} }})(1, 2, 3);\n"));
        self.globals.push(g);
    }

    // ── try / catch / finally (incl. abrupt completion crossing boundary) ──
    //
    // VM-ELIGIBLE shape (non-diverge): a `try` with a CATCH (no finally) whose body
    // throws or does plain work and is caught locally — never crosses the boundary
    // abruptly. DIVERGING shapes (diverge-mode): a `finally` block, a rethrow to an
    // outer catch, or a break/continue/throw crossing the try boundary. The latter
    // are correctly DECLINED, so the program tree-walks on both passes (validating
    // the decline fallback is byte-identical).
    fn stmt_try(&mut self) {
        let ind = self.indent();
        // To make a rethrow / boundary-crossing throw actually OBSERVABLE (not a
        // top-level uncaught throw), wrap diverging trys in an OUTER catch.
        let diverging = self.diverge_ok && self.rng.chance(60, 100);
        let outer_e = if diverging {
            let e = self.fresh("oe");
            self.out.push_str(&format!("{ind}try {{\n"));
            self.depth += 1;
            Some(e)
        } else {
            None
        };

        let tind = self.indent();
        self.out.push_str(&format!("{tind}try {{\n"));
        self.depth += 1;
        // Body: in the ELIGIBLE shape the body must NOT complete abruptly (no throw /
        // break / continue / return crossing the boundary — those all DECLINE), so it
        // is plain work. In diverge-mode the body may throw / break / continue
        // (declines → tree-walk fallback, which the oracle proves byte-identical).
        let body_n = 1 + self.rng.below(2);
        for _ in 0..body_n {
            let r = self.rng.below(100);
            if diverging && r < 30 {
                let tv = self.throw_value();
                self.out.push_str(&format!("{}throw {};\n", self.indent(), tv));
            } else if diverging && r < 45 && self.loop_depth > 0 {
                let kw = if self.rng.chance(50, 100) { "break" } else { "continue" };
                self.out.push_str(&format!("{}{kw};\n", self.indent()));
            } else {
                self.statement();
            }
        }
        self.depth -= 1;

        // Catch — always present in the eligible shape; in diverge-mode it may
        // rethrow to the outer catch.
        let has_catch = !diverging || self.rng.chance(70, 100);
        if has_catch {
            let e = self.fresh("e");
            self.out.push_str(&format!("{tind}}} catch ({e}) {{\n"));
            self.depth += 1;
            if let Some(g) = self.globals.first().cloned() {
                self.out.push_str(&format!("{}{g} = {g} + 100;\n", self.indent()));
            }
            if diverging && self.rng.chance(40, 100) {
                self.out.push_str(&format!("{}throw {e};\n", self.indent()));
            }
            self.depth -= 1;
        }
        // Finally — ONLY in diverge-mode (declines).
        if diverging && (self.rng.chance(60, 100) || !has_catch) {
            self.out.push_str(&format!("{tind}}} finally {{\n"));
            self.depth += 1;
            if let Some(g) = self.globals.first().cloned() {
                self.out.push_str(&format!("{}{g} = {g} + 1000;\n", self.indent()));
            }
            self.depth -= 1;
        }
        self.out.push_str(&format!("{tind}}}\n"));

        if let Some(e) = outer_e {
            self.depth -= 1;
            self.out.push_str(&format!("{ind}}} catch ({e}) {{\n"));
            self.depth += 1;
            if let Some(g) = self.globals.first().cloned() {
                self.out.push_str(&format!("{}{g} = {g} + 10000;\n", self.indent()));
            }
            self.depth -= 1;
            self.out.push_str(&format!("{ind}}}\n"));
        }
    }

    fn throw_value(&mut self) -> String {
        let r = self.rng.below(5);
        match r {
            0 => "new Error('boom')".to_string(),
            1 => "new TypeError('te')".to_string(),
            2 => "new RangeError('re')".to_string(),
            3 => "'str-throw'".to_string(),
            _ => format!("{}", self.rng.below(100)),
        }
    }

    // ── if / else ──────────────────────────────────────────────────────────
    fn stmt_if(&mut self) {
        let ind = self.indent();
        let cond = self.expr(0);
        self.out.push_str(&format!("{ind}if ({cond}) {{\n"));
        self.depth += 1;
        self.statement();
        self.depth -= 1;
        if self.rng.chance(50, 100) {
            self.out.push_str(&format!("{ind}}} else {{\n"));
            self.depth += 1;
            self.statement();
            self.depth -= 1;
        }
        self.out.push_str(&format!("{ind}}}\n"));
    }

    // ── switch ───────────────────────────────────────────────────────────
    fn stmt_switch(&mut self) {
        let ind = self.indent();
        let disc = if let Some(g) = self.globals.first().cloned() {
            format!("{g} % 4")
        } else {
            "1".to_string()
        };
        self.out.push_str(&format!("{ind}switch ({disc}) {{\n"));
        self.depth += 1;
        self.loop_depth += 1; // break is legal inside switch
        let cases = 2 + self.rng.below(3);
        for c in 0..cases {
            self.out.push_str(&format!("{}case {c}:\n", self.indent()));
            if let Some(g) = self.globals.first().cloned() {
                self.out.push_str(&format!("{}  {g} = {g} + {};\n", self.indent(), c + 1));
            }
            // Sometimes fall through (omit break).
            if self.rng.chance(70, 100) {
                self.out.push_str(&format!("{}  break;\n", self.indent()));
            }
        }
        self.out.push_str(&format!("{}default:\n", self.indent()));
        if let Some(g) = self.globals.first().cloned() {
            self.out.push_str(&format!("{}  {g} = {g} + 7;\n", self.indent()));
        }
        self.loop_depth -= 1;
        self.depth -= 1;
        self.out.push_str(&format!("{ind}}}\n"));
    }

    // ── function declaration (called directly only) ────────────────────────
    fn stmt_function_decl(&mut self) {
        // A function declaration nested inside a block/loop/if (depth > 0) has Annex
        // B global-hoisting semantics the VM declines. In non-diverge mode, only emit
        // fn decls at the TOP LEVEL (depth 0, eligible); deeper, emit an assignment
        // instead. In diverge-mode, allow the nested decl (it declines → fallback).
        if self.depth > 0 && !self.diverge_ok {
            self.stmt_assign();
            return;
        }
        let ind = self.indent();
        let name = self.fresh("f");
        let p = self.fresh("p");
        self.out.push_str(&format!("{ind}function {name}({p}) {{\n"));
        self.depth += 1;
        // A pure-ish body that returns a numeric/string expression.
        let r = self.expr_local(&p);
        // In diverge-mode, sometimes a try/finally with return inside (the bitten
        // return-crossing-finally case) — this declines (a finally anywhere in a
        // top-level-reachable body declines). Otherwise a plain return (eligible).
        if self.diverge_ok && self.rng.chance(35, 100) {
            self.out.push_str(&format!("{}try {{ return {r}; }} finally {{ }}\n", self.indent()));
        } else {
            self.out.push_str(&format!("{}return {r};\n", self.indent()));
        }
        self.depth -= 1;
        self.out.push_str(&format!("{ind}}}\n"));
        self.fns.push(name.clone());
        // Immediately CALL it (the only eligible use) and store into a global.
        let g = self.fresh("rf");
        let arg = self.rng.below(10);
        self.out.push_str(&format!("{ind}var {g} = {name}({arg});\n"));
        self.globals.push(g);
    }

    fn stmt_console_log(&mut self) {
        let ind = self.indent();
        let e = self.expr(0);
        self.out.push_str(&format!("{ind}console.log({e});\n"));
    }

    fn stmt_throw(&mut self) {
        // Only throw at top level rarely AND only when there's an enclosing catch
        // is hard to guarantee here; a bare top-level throw is fine (throw parity).
        // To keep most programs producing global state, wrap in a self-catching try.
        let ind = self.indent();
        let e = self.fresh("ce");
        let tv = self.throw_value();
        self.out.push_str(&format!("{ind}try {{ throw {tv}; }} catch ({e}) {{\n"));
        self.depth += 1;
        if let Some(g) = self.globals.first().cloned() {
            self.out.push_str(&format!("{}{g} = {g} + 1;\n", self.indent()));
        }
        self.depth -= 1;
        self.out.push_str(&format!("{ind}}}\n"));
    }

    fn stmt_block(&mut self) {
        let ind = self.indent();
        self.out.push_str(&format!("{ind}{{\n"));
        self.depth += 1;
        let n = 1 + self.rng.below(2);
        for _ in 0..n {
            self.statement();
        }
        self.depth -= 1;
        self.out.push_str(&format!("{ind}}}\n"));
    }

    // ── EXPRESSIONS ────────────────────────────────────────────────────────

    /// An expression usable at top level. `d` is the expression nesting depth.
    fn expr(&mut self, d: u32) -> String {
        self.fuel -= 1;
        if self.fuel <= 0 || d >= 3 {
            return self.atom();
        }
        let r = self.rng.below(100);
        match r {
            0..=24 => {
                let a = self.expr(d + 1);
                let b = self.expr(d + 1);
                let op = self.rng.pick(&[
                    "+", "-", "*", "/", "%", "**", "&", "|", "^", "<<", ">>", ">>>",
                ]);
                format!("({a} {op} {b})")
            }
            25..=39 => {
                let a = self.expr(d + 1);
                let b = self.expr(d + 1);
                let op = self.rng.pick(&["<", ">", "<=", ">=", "===", "!==", "==", "!="]);
                format!("({a} {op} {b})")
            }
            40..=49 => {
                let a = self.expr(d + 1);
                let b = self.expr(d + 1);
                let op = self.rng.pick(&["&&", "||", "??"]);
                format!("({a} {op} {b})")
            }
            50..=57 => {
                let t = self.expr(d + 1);
                let c = self.expr(d + 1);
                let a = self.expr(d + 1);
                format!("({t} ? {c} : {a})")
            }
            58..=64 => {
                let a = self.expr(d + 1);
                let op = self.rng.pick(&["-", "!", "~", "+", "typeof "]);
                format!("({op}{a})")
            }
            65..=72 => self.coercion_expr(d),
            73..=80 => self.template_literal(d),
            81..=87 => self.array_or_object_expr(d),
            88..=93 => self.method_call_expr(d),
            _ => self.atom(),
        }
    }

    /// A simple expression referencing the given local parameter (used in fn bodies).
    fn expr_local(&mut self, p: &str) -> String {
        let r = self.rng.below(6);
        match r {
            0 => format!("{p} * {p}"),
            1 => format!("{p} + 1"),
            2 => format!("{p} % 3"),
            3 => format!("{p} * 0.5 - 1.0"),
            4 => format!("{p} > 2 ? {p} : -{p}"),
            _ => format!("{p}"),
        }
    }

    /// Coercion / ToPrimitive / special numeric values (the bitten arithmetic
    /// edge cases): NaN, Infinity, -0, pow/mod specials, valueOf/toString objects.
    fn coercion_expr(&mut self, _d: u32) -> String {
        // valueOf/toString OBJECT-literal coercion uses inline `function(){…}`
        // members. Those object literals are fine VALUES, but an object-literal
        // containing a function as a top-level expression argument can be treated as
        // an escaping callback by the decline scan, so only emit the function-member
        // forms in diverge-mode; the pure-numeric special-value forms (NaN/Infinity/
        // -0/pow/mod) stay VM-eligible and are the bitten arithmetic edge cases.
        if self.diverge_ok && self.rng.chance(35, 100) {
            let r = self.rng.below(3);
            return match r {
                0 => "('' + {toString:function(){return 'TS';}})".to_string(),
                1 => "({valueOf:function(){return 7;}} + 1)".to_string(),
                _ => "(`v=${ {valueOf:function(){return 42;},toString:function(){return 'X';}} }`)".to_string(),
            };
        }
        let r = self.rng.below(13);
        match r {
            0 => "(0/0)".to_string(),               // NaN
            1 => "(1/0)".to_string(),               // Infinity
            2 => "(-1/0)".to_string(),              // -Infinity
            3 => "(-0)".to_string(),                // -0
            4 => "(0 * -1)".to_string(),            // -0
            5 => "(1 ** (0/0))".to_string(),        // 1 ** NaN === NaN (ECMA, not IEEE)
            6 => "((0/0) ** 0)".to_string(),        // NaN ** 0 === 1
            7 => "(5 % 0)".to_string(),             // NaN
            8 => "((-5) % 3)".to_string(),          // -2 (sign of dividend)
            9 => "('5' * 2)".to_string(),           // 10 (string→number)
            10 => "('5' + 2)".to_string(),          // '52' (number→string)
            11 => "(true + 1)".to_string(),         // 2
            _ => "(null + 1)".to_string(),          // 1
        }
    }

    /// Template literal with interpolated holes (the bitten regex-in-hole / hint).
    fn template_literal(&mut self, d: u32) -> String {
        let a = self.expr(d + 1);
        let r = self.rng.below(3);
        match r {
            0 => format!("(`val=${{{a}}}`)"),
            1 => format!("(`${{{a}}}-${{{a}}}`)"),
            _ => format!("(`pre ${{{a}}} post`).length"),
        }
    }

    /// Array / object literal + a method. The callback-taking forms (map/filter/
    /// reduce) pass an escaping `function(){…}` which is correctly DECLINED, so they
    /// are only emitted in diverge-mode; the callback-free forms stay VM-eligible.
    fn array_or_object_expr(&mut self, _d: u32) -> String {
        if self.diverge_ok && self.rng.chance(40, 100) {
            let r = self.rng.below(3);
            return match r {
                0 => "[1,2,3].reduce(function(a,b){return a+b;},0)".to_string(),
                1 => "[1,2,3].map(function(x){return x*2;}).length".to_string(),
                _ => "[1,2,3].filter(function(x){return x>1;}).length".to_string(),
            };
        }
        let r = self.rng.below(7);
        match r {
            0 => "[1,2,3].length".to_string(),
            1 => "[3,1,2].sort()[0]".to_string(),
            2 => "[NaN].includes(NaN)".to_string(),
            3 => "Object.keys({a:1,b:2}).length".to_string(),
            4 => "({a:1,b:2}).a".to_string(),
            5 => "[1,2,3].indexOf(2)".to_string(),
            _ => "[[1],[2]].flat().length".to_string(),
        }
    }

    /// A method call on a string/number/array primitive (coercion-sensitive).
    fn method_call_expr(&mut self, _d: u32) -> String {
        // NOTE: deliberately NO `Math.*` — `Math` is not in the oracle harness's
        // `install_basic_globals()` (and `Math.random` would be non-deterministic
        // anyway), so a `Math.x` call throws "Math is not defined" identically on
        // BOTH tiers. Only constructs the harness actually provides are emitted.
        let r = self.rng.below(7);
        match r {
            0 => "'hello'.toUpperCase().length".to_string(),
            1 => "'a,b,c'.split(',').length".to_string(),
            2 => "(3.14159).toFixed(2).length".to_string(),
            3 => "parseInt('42', 10)".to_string(),
            4 => "Number('3.5')".to_string(),
            5 => "String(123).length".to_string(),
            _ => "(-7).toString().length".to_string(),
        }
    }

    /// A leaf expression: a literal, a read of an existing global, or a call.
    fn atom(&mut self) -> String {
        let r = self.rng.below(100);
        if r < 30 {
            // Read an existing readable name (global or loop counter) — assign-
            // before-decl / hoisting / loop-counter-read coverage.
            if let Some(name) = self.pick_readable() {
                return name;
            }
            format!("{}", self.rng.below(50))
        } else if r < 50 {
            format!("{}", self.rng.below(50))
        } else if r < 62 {
            let n = self.rng.below(1000) as f64 / 100.0;
            format!("{n}")
        } else if r < 70 {
            "true".to_string()
        } else if r < 76 {
            "false".to_string()
        } else if r < 80 {
            "null".to_string()
        } else if r < 84 {
            "undefined".to_string()
        } else if r < 92 {
            format!("'{}'", self.fresh("s"))
        } else {
            format!("{}", self.rng.below(10))
        }
    }
}

/// Configure a generator's per-program mode flags from its own RNG stream.
fn configure_mode(g: &mut Gen<'_>) {
    // ~35% DIVERGE-mode (may emit declining constructs → tree-walk both passes,
    // validating the decline fallback). The other ~65% stay VM-eligible so they
    // ENGAGE the VM path. Eligible programs pick ONE of {no trys, no for-loops} so
    // they never hit the for-init-var + try co-occurrence decline.
    g.diverge_ok = g.rng.chance(35, 100);
    if !g.diverge_ok {
        if g.rng.chance(50, 100) {
            g.avoid_try = true; // loop-heavy eligible program
        } else {
            g.avoid_for = true; // try-heavy eligible program
        }
    }
}

/// Generate a program AND report whether it was diverge-mode (for diagnostics).
fn generate_program_with_mode(seed: u64) -> (String, bool) {
    let mut rng = Rng::new(seed);
    let fuel = 40 + (seed % 30) as i32;
    let mut g = Gen::new(&mut rng, fuel);
    configure_mode(&mut g);
    let mode = g.diverge_ok;
    (g.program(), mode)
}

/// Generate one program for `seed`. Pure function of the seed (reproducible).
fn generate_program(seed: u64) -> String {
    let mut rng = Rng::new(seed);
    // Fuel scales mildly with seed so programs vary in size; bounded so two-tier
    // execution stays fast.
    let fuel = 40 + (seed % 30) as i32;
    let mut g = Gen::new(&mut rng, fuel);
    configure_mode(&mut g);
    g.program()
}

// ───────────────────────────── THE FUZZER LOOP ────────────────────────────

/// Run `count` seeded programs (base seed `base`) through the production-faithful
/// top-level-VM oracle. Returns `Ok(())` on zero divergences, or the FIRST
/// divergence with its minimal reproducer (the generating seed + the program
/// source + the structured `Divergence`).
fn run_fuzz(base: u64, count: u64) -> Result<u64, String> {
    let mut ran = 0u64;
    for i in 0..count {
        let seed = base.wrapping_add(i).wrapping_mul(0x100_0001).wrapping_add(i);
        let src = generate_program(seed);
        // The generator only emits well-formed syntax; if a program fails to parse
        // that is itself a generator bug worth surfacing (so do not silently skip).
        if crate::parser::parse_program(&src).is_err() {
            return Err(format!(
                "GENERATOR BUG: produced unparseable program (seed {seed}):\n{src}"
            ));
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_toplevel_vm_agrees(&src)
        })) {
            Ok(Ok(())) => {}
            Ok(Err(d)) => {
                return Err(format_divergence(seed, &src, &d));
            }
            Err(_) => {
                return Err(format!(
                    "PANIC while comparing tiers (seed {seed}):\n--- minimal reproducer ---\n{src}\n--------------------------"
                ));
            }
        }
        ran += 1;
    }
    Ok(ran)
}

fn format_divergence(seed: u64, src: &str, d: &Divergence) -> String {
    format!(
        "TOP-LEVEL VM DIVERGENCE (seed {seed})\n\
         --- minimal reproducer ---\n{src}\n\
         --------------------------\n{d}"
    )
}

// ───────────────────────────── THE TESTS ──────────────────────────────────

/// Smoke: the generator only ever emits programs that PARSE. (A generator that
/// emitted garbage would make the oracle vacuously skip — proven non-vacuous by
/// running a sizeable sample through the parser.)
#[test]
fn fuzz_generator_only_emits_parseable_programs() {
    for i in 0..500u64 {
        let seed = 0xABCD_0000u64.wrapping_add(i);
        let src = generate_program(seed);
        crate::parser::parse_program(&src)
            .unwrap_or_else(|e| panic!("generator emitted unparseable program (seed {seed}):\n{src}\n{e}"));
    }
}

/// ★ THE PERMANENT REGRESSION GATE — the broad batch. A deterministic, seeded,
/// bounded fuzz of THOUSANDS of varied top-level programs, each diffed between the
/// tree-walker and the top-level register-VM through the production-faithful
/// oracle (completion / console / throw / globalThis-read of every touched
/// global). ZERO divergences is the bar; any divergence prints the minimal
/// reproducer and fails. This is what proves `CV_TOPLEVEL_VM` is flip-safe for the
/// whole construct grammar, not just a fixed corpus.
#[test]
fn fuzz_toplevel_vm_broad_batch() {
    // 3000 programs at base seed 0xC0FFEE. Bounded (each program is shallow + small)
    // so the default gate runs in a few seconds, in-process, offline. THOUSANDS of
    // varied programs is a far stronger gate than any fixed corpus; the `#[ignore]`d
    // `fuzz_toplevel_vm_deep_stress` sweeps 20k+ across several seed bands as the
    // pre-flip soak gate.
    const BASE_SEED: u64 = 0x00C0_FFEE;
    const COUNT: u64 = 3000;
    match run_fuzz(BASE_SEED, COUNT) {
        Ok(ran) => {
            assert_eq!(ran, COUNT, "every generated program must have been compared");
        }
        Err(report) => panic!("\n{report}\n"),
    }
}

/// DEEP STRESS (ignored): a much larger, multi-base-seed sweep (20k programs over
/// several disjoint seed bands) run on demand to flush residual divergences the
/// default 4k batch might miss. Bounded + in-process; `--ignored` keeps it off the
/// fast CI path while remaining available as the soak gate before the default flip.
#[test]
#[ignore]
fn fuzz_toplevel_vm_deep_stress() {
    for base in [0x00C0_FFEEu64, 0xDEAD_BEEF, 0x1234_5678, 0xFACE_F00D, 0x0BAD_C0DE] {
        match run_fuzz(base, 4000) {
            Ok(ran) => assert_eq!(ran, 4000),
            Err(report) => panic!("\n[deep-stress base {base:#x}]\n{report}\n"),
        }
    }
}

/// BATCH NON-VACUITY — prove the broad batch is not vacuously green by all
/// programs DECLINING to the tree-walker on both passes. A healthy fraction of the
/// generated grammar must actually be VM-ELIGIBLE and TAKE the top-level VM path
/// (`toplevel_vm_took_count() > 0`), so the batch genuinely exercises the VM tier.
/// (The remaining declining programs are still valuable: they prove the decline
/// fallback is byte-identical too.)
#[test]
fn fuzz_batch_engages_the_vm_path() {
    const COUNT: u64 = 1500;
    let mut engaged = 0u64;
    let mut eligible = 0u64;
    for i in 0..COUNT {
        let seed = 0x00C0_FFEEu64
            .wrapping_add(i)
            .wrapping_mul(0x100_0001)
            .wrapping_add(i);
        let src = generate_program(seed);
        let stmts = match crate::parser::parse_program(&src) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if crate::interp::toplevel_vm_eligible_for_test(&stmts) {
            eligible += 1;
        }
        // Actually run with the VM ON and count whether the path was taken.
        let _g = crate::interp::TopLevelVmGuard::new(true);
        crate::interp::reset_toplevel_vm_took_count();
        crate::interp::reset_bc_fn_cache();
        let mut interp = crate::interp::Interp::new();
        interp.install_basic_globals();
        let _ = interp.run(&src);
        if crate::interp::toplevel_vm_took_count() > 0 {
            engaged += 1;
        }
    }
    // The grammar is var-heavy + loop-heavy, so a large fraction must be eligible
    // AND engage. A low number would mean the batch silently tree-walked everything
    // (a vacuously-green gate). Require at least 25% engagement — comfortably met
    // (~40-60% typical) while leaving headroom for grammar tweaks.
    let pct = (engaged as f64) / (COUNT as f64) * 100.0;
    println!(
        "[batch non-vacuity] eligible={eligible}/{COUNT}, VM-engaged={engaged}/{COUNT} ({pct:.1}%)"
    );
    assert!(
        engaged * 4 >= COUNT,
        "batch non-vacuity FAIL: only {engaged}/{COUNT} programs took the VM path — \
         the broad batch would be vacuously green (all tree-walked). Grammar must \
         produce VM-eligible programs."
    );
}

/// DIAGNOSTIC (ignored): does the top-level VM handle a LOCALLY-CAUGHT throw
/// (`try { throw X } catch(e) {…}`, no finally, no rethrow) byte-identically? If
/// yes, the `block_can_abruptly_complete` decline is over-conservative for this
/// (very common) shape and could be tightened. We probe by forcing eligibility via
/// a (new) hook and running the production oracle. NOTE: this only PROBES; the
/// production decline is unchanged unless the result says it's safe to tighten.
#[test]
#[ignore]
fn fuzz_diag_locally_caught_throw_vm_parity() {
    use crate::interp::ForceEligibleBug;
    let cases = [
        "var g=0; try { throw 0; } catch(e) { g = g + 1; } console.log(g);",
        "var g=0; try { throw new Error('x'); } catch(e) { g = g + e.message.length; } console.log(g);",
        "var g=0; for (var i=0;i<3;i=i+1){ try { if (i===1) throw i; g = g + 10; } catch(e) { g = g + 100; } } console.log(g);",
        "var g=0; try { var a = [1,2,3]; throw a.length; } catch(e) { g = e; } console.log(g);",
    ];
    for src in cases {
        let _inject = crate::interp::ForceEligibleGuard::new(ForceEligibleBug::LocallyCaughtThrow);
        match assert_toplevel_vm_agrees(src) {
            Ok(()) => println!("[locally-caught-throw] OK (VM byte-identical): {src}"),
            Err(d) => println!("[locally-caught-throw] DIVERGES: {src}\n{d}"),
        }
    }
}

/// DIAGNOSTIC (ignored): print non-diverge programs that DECLINE, to find grammar
/// constructs that unexpectedly leave the VM-eligible subset.
#[test]
#[ignore]
fn fuzz_diag_why_nondiverge_declines() {
    let mut shown = 0;
    for i in 0..1500u64 {
        let seed = 0x00C0_FFEEu64
            .wrapping_add(i)
            .wrapping_mul(0x100_0001)
            .wrapping_add(i);
        let (src, mode) = generate_program_with_mode(seed);
        if mode {
            continue;
        }
        let stmts = crate::parser::parse_program(&src).unwrap();
        if !crate::interp::toplevel_vm_eligible_for_test(&stmts) {
            println!("=== DECLINED non-diverge (seed {seed}) ===\n{src}\n");
            shown += 1;
            if shown >= 12 {
                break;
            }
        }
    }
    println!("shown {shown} declined non-diverge programs");
}

/// DIAGNOSTIC (ignored): dump a sample of ELIGIBLE generated programs (each ending
/// in a `console.log` of touched globals) so they can be cross-checked against Node
/// — confirming the tree-walker REFERENCE itself is Node-correct on these shapes
/// (not merely that the two tiers agree with each other). Run with `--nocapture` and
/// pipe each `=== PROGRAM` block to `node` to compare console output.
#[test]
#[ignore]
fn fuzz_diag_dump_eligible_for_node_crosscheck() {
    let mut shown = 0;
    for i in 0..3000u64 {
        let seed = 0x00C0_FFEEu64
            .wrapping_add(i)
            .wrapping_mul(0x100_0001)
            .wrapping_add(i);
        let src = generate_program(seed);
        let stmts = crate::parser::parse_program(&src).unwrap();
        if !crate::interp::toplevel_vm_eligible_for_test(&stmts) {
            continue;
        }
        // Capture the tree-walk console output (the reference) for side-by-side.
        let out = {
            let _g = crate::interp::TopLevelVmGuard::new(false);
            crate::interp::reset_bc_fn_cache();
            let mut it = crate::interp::Interp::new();
            it.install_basic_globals();
            let _ = it.run(&src);
            it.output.clone()
        };
        println!("=== PROGRAM seed={seed} ===\n{src}\n--- treewalk-output: {out:?}\n");
        shown += 1;
        if shown >= 20 {
            break;
        }
    }
    println!("dumped {shown} eligible programs for Node cross-check");
}

/// ★ NON-VACUITY PROOF — the fuzzer must CATCH a reintroduced bug, else a green
/// batch proves nothing.
///
/// We can't just delete a `toplevel_vm_eligible` decline from source inside a test
/// (it's a compile-time function). Instead the engine exposes a TEST-ONLY
/// fault-injection hook (`set_toplevel_vm_force_eligible`) that forces a chosen
/// construct family to be treated as VM-eligible even though it diverges — exactly
/// equivalent to "un-declining" it. With that hook armed for the `try`/`finally`
/// family (the construct that bit us in round 3), the fuzzer's batch MUST turn RED
/// on a program that exercises it. We then disarm the hook and confirm the batch is
/// GREEN again — proving (a) the fuzzer has teeth and (b) the decline is what keeps
/// it green.
#[test]
fn fuzz_catches_reintroduced_bug() {
    use crate::interp::ForceEligibleBug;

    /// With `bug` injected (a known-divergent construct family un-declined), search
    /// a band of seeds (filtered to programs that actually contain `marker`) for a
    /// divergence. Returns the caught reproducer, or None if the fuzzer had no teeth.
    fn search_with_bug(bug: ForceEligibleBug, marker: &str) -> Option<String> {
        let _inject = crate::interp::ForceEligibleGuard::new(bug);
        // Scan a wide seed band: the EXPENSIVE oracle only runs on the (few)
        // programs that contain `marker`, and the loop STOPS at the first divergence
        // (found quickly), so this is cheap despite the large bound.
        for i in 0..4000u64 {
            let seed = 0x00C0_FFEEu64
                .wrapping_add(i)
                .wrapping_mul(0x100_0001)
                .wrapping_add(i);
            let src = generate_program(seed);
            if !src.contains(marker) {
                continue;
            }
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert_toplevel_vm_agrees(&src)
            })) {
                Ok(Ok(())) => {}
                Ok(Err(d)) => return Some(format_divergence(seed, &src, &d)),
                Err(_) => return Some(format!("panic on seed {seed}:\n{src}")),
            }
        }
        None
    }

    // ── (1) try/finally un-declined (the round-3 bug) MUST be caught ──────────
    let caught_tf = search_with_bug(ForceEligibleBug::TryFinally, "finally");
    assert!(
        caught_tf.is_some(),
        "NON-VACUITY FAILURE: with try/finally un-declined, the fuzzer found NO \
         divergence — it has no teeth. The decline must be what keeps it green."
    );
    println!(
        "[non-vacuity] fuzzer CAUGHT the reintroduced try/finally bug:\n{}",
        caught_tf.unwrap()
    );

    // ── (2) for-init-var write-back DECLINE is LIVE (non-vacuous) ─────────────
    // This is the OTHER fuzzer-discovered family. Its write-back divergence is
    // extremely register-pressure-threshold-sensitive (the originals were found in
    // the full mixed grammar; an isolated reproducer is not deterministic), so the
    // robust non-vacuity proof for THIS decline is that it is LIVE — it actually
    // changes eligibility for the diverging shapes: each known-divergent shape is
    // DECLINED in production (probe off) and becomes ELIGIBLE under the probe
    // (decline suppressed). A dead/no-op decline could not flip eligibility. Each
    // declined shape must ALSO still AGREE (the tree-walk fallback is byte-identical)
    // — that is the production-correctness half. The divergence itself was caught by
    // the fuzzer during development (reproducers logged in the report).
    let fi_shapes = [
        // nested for-init-var loops.
        "var g=0; for (var i2=0;i2<2;i2=i2+1){ for (var i3=0;i3<3;i3=i3+1){ g=g+1; } } console.log(i3);",
        // for-init-var loop containing a do-while (the i10 shape).
        "var g=0; for (var i10=0;i10<3;i10=i10+1){ var d=0; do { d=d+1; } while (d<1); } console.log(i10);",
        // for-init-var loop inside a try (the i9 shape).
        "var g=0; try { for (var i9=0;i9<2;i9=i9+2){ g=g+1; } } catch(e){} console.log(i9);",
        // for-init-var loop co-occurring with a try (the i3-with-try shape).
        "var g=0; for (var k=0;k<3;k=k+1){ g=g+1; } try { g=g+1; } catch(e){} console.log(k);",
    ];
    let mut flips = 0;
    for src in fi_shapes {
        let stmts = crate::parser::parse_program(src).expect("parse");
        // Production: DECLINED.
        let declined_in_prod = !crate::interp::toplevel_vm_eligible_for_test(&stmts);
        // Under the probe (decline suppressed): ELIGIBLE.
        let eligible_under_probe = {
            let _inject =
                crate::interp::ForceEligibleGuard::new(ForceEligibleBug::ForInitVarLoopInTry);
            crate::interp::toplevel_vm_eligible_for_test(&stmts)
        };
        assert!(
            declined_in_prod && eligible_under_probe,
            "for-init-var decline must be LIVE (declined in prod, eligible under probe) \
             for shape:\n{src}\n(declined_in_prod={declined_in_prod}, \
             eligible_under_probe={eligible_under_probe})"
        );
        // The declined shape must still AGREE via the tree-walk fallback.
        assert_toplevel_vm_agrees(src).unwrap_or_else(|d| {
            panic!("declined for-init-var shape must agree via fallback:\n{src}\n{d}")
        });
        flips += 1;
    }
    println!(
        "[non-vacuity] for-init-var DECLINE is LIVE: it flipped eligibility for \
         {flips}/{} known-divergent shapes (declined in prod, eligible under probe).",
        fi_shapes.len()
    );

    // ── (3) Hooks disarmed (production restored): the SAME bands are GREEN ────
    // Proves the DECLINES — not luck — are what make the fuzzer pass.
    for i in 0..800u64 {
        let seed = 0x00C0_FFEEu64
            .wrapping_add(i)
            .wrapping_mul(0x100_0001)
            .wrapping_add(i);
        let src = generate_program(seed);
        if !src.contains("finally") && !src.contains("for (") {
            continue;
        }
        assert_toplevel_vm_agrees(&src).unwrap_or_else(|d| {
            panic!(
                "after disarming the fault-injection hooks, the production path must \
                 be byte-identical again, but seed {seed} diverged:\n{src}\n{d}"
            )
        });
    }
}
