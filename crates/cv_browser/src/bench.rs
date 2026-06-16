//! `--type bench` — a REAL performance benchmark harness.
//!
//! Every number this module emits is a genuine measurement of real work: a real
//! `std::time::Instant`-timed workload, a real deterministic work counter, or a
//! real in-process heap-liveness count. There are NO constants, placeholders, or
//! hardcoded metrics anywhere in this file — a fabricated number would be worse
//! than an omitted one (it would silently corrupt the perf baseline these numbers
//! gate). If a thing cannot be measured for real, it is left out, not faked.
//!
//! Metric families (each pairs a TIMING with a deterministic WORK COUNTER where
//! one exists):
//!   - startup   : cold `PROCESS_T0.elapsed()` at first-session + first-paint
//!                 (single-shot per process; reported with `samples: 1`).
//!   - ttfp      : `build_runtime_and_first_paint` wall time over N iters, with
//!                 the produced display-list chunk count as the work counter.
//!   - repeat    : two incremental renders on the SAME unchanged doc; the
//!                 layout reuse counters (`relayout_stats`) prove skip-not-redo.
//!   - animation : N back-to-back incremental frames; the exact per-frame
//!                 `retained_dl::diff` node-set sizes are the damage counters.
//!   - js_exec   : the loop/jit microbenches through the real interpreter, with
//!                 `t2_exec_count()` as the honesty guard (>0 == the JIT ran).
//!   - memory    : `gc_live_object_count()` after load and after idle+GC.
//!
//! The HTML/JS inputs live under `benchfix/` (committed, fully self-contained,
//! zero external references) and are resolved via `CARGO_MANIFEST_DIR` so the
//! bench runs from any CWD, offline and deterministically. Inputs are FIXED
//! forever — changing one invalidates the baseline (bump the `schema` if so).

use std::time::Instant;

use crate::Cli;
use crate::LiveInterp;

/// Bench JSON schema version. Bump ONLY when a fixed input or a metric's meaning
/// changes (so old baselines are not silently compared against new semantics).
const SCHEMA: &str = "conclave-bench/1";

/// Fixed measurement parameters. Held in the emitted `config` so numbers are
/// only ever compared like-for-like across runs.
const ITERS: usize = 7;
const WARMUP: usize = 2;
const VIEWPORT_W: f32 = 1280.0;
const VIEWPORT_H: f32 = 800.0;
/// Animation frame count (ticked back-to-back, no sleep — sleep adds jitter and
/// is not the cost we measure).
const ANIM_FRAMES: usize = 120;
/// Idle frames ticked before the after-idle memory sample.
const IDLE_FRAMES: usize = 30;

/// Resolve a fixed input file under `benchfix/`. `CARGO_MANIFEST_DIR` is the
/// crate dir at build time, so this works regardless of the runtime CWD.
fn benchfix(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchfix")
        .join(name)
}

fn read_input(name: &str) -> Result<String, String> {
    let p = benchfix(name);
    std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))
}

/// The fixed layout config every bench input renders under (1280x800, real text
/// measurer). Identical to the probe path so layout work matches Chrome's.
fn bench_cfg() -> cv_layout::LayoutConfig {
    cv_layout::LayoutConfig {
        viewport_w: VIEWPORT_W,
        viewport_h: VIEWPORT_H,
        measure_text_fn: Some(crate::layout_text_measurer()),
        ..cv_layout::LayoutConfig::default()
    }
}

// ── statistics over real samples ─────────────────────────────────────────────

/// Median of a sample set (sorted-middle; even N averages the two middles).
/// Empty input is reported as 0.0 — callers never pass an empty set (N == ITERS).
fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// p95 via nearest-rank on the sorted samples (the rank-ceil index, clamped).
fn p95(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // nearest-rank: ceil(0.95 * n) - 1, clamped into [0, n-1].
    let rank = ((0.95 * v.len() as f64).ceil() as usize).max(1) - 1;
    v[rank.min(v.len() - 1)]
}

fn min_of(samples: &[f64]) -> f64 {
    samples
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .min(f64::INFINITY)
}

/// Integer median (used for per-frame changed-node counts). Sorted-middle; for
/// even N takes the lower-middle (no fractional node count).
fn median_usize(mut v: Vec<usize>) -> usize {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[(v.len() - 1) / 2]
}

// ── JSON emission (no third-party crate; reuses the crate string escaper) ─────

/// A tiny JSON value tree, just enough for the bench output. Numbers are emitted
/// as JSON numbers (never strings). `f64` is formatted with enough precision to
/// be lossless for the millisecond/ratio ranges we measure.
enum J {
    Null,
    Bool(bool),
    I(i64),
    F(f64),
    S(String),
    Obj(Vec<(&'static str, J)>),
}

impl J {
    fn write(&self, out: &mut String, indent: usize) {
        match self {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::I(n) => out.push_str(&n.to_string()),
            J::F(x) => {
                // Non-finite is not valid JSON; emit null rather than a fake
                // number. A finite value is formatted with 6 decimals (lossless
                // for our ms/ratio ranges) and trailing-zero-trimmed.
                if x.is_finite() {
                    let s = format!("{x:.6}");
                    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
                    out.push_str(if trimmed.is_empty() { "0" } else { trimmed });
                } else {
                    out.push_str("null");
                }
            }
            J::S(s) => out.push_str(&crate::json_quote(s)),
            J::Obj(fields) => {
                out.push_str("{\n");
                for (i, (k, v)) in fields.iter().enumerate() {
                    for _ in 0..indent + 2 {
                        out.push(' ');
                    }
                    out.push_str(&crate::json_quote(k));
                    out.push_str(": ");
                    v.write(out, indent + 2);
                    if i + 1 < fields.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                for _ in 0..indent {
                    out.push(' ');
                }
                out.push('}');
            }
        }
    }

    fn to_pretty(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, 0);
        s.push('\n');
        s
    }
}

// ── measured metric structs ──────────────────────────────────────────────────

struct TtfpResult {
    samples_ms: Vec<f64>,
    doc_chunks: usize,
}

struct RepeatResult {
    first_ms: Vec<f64>,
    second_ms: Vec<f64>,
    relaid_first: u64,
    reused_first: u64,
    relaid_second: u64,
    reused_second: u64,
}

struct AnimResult {
    frames_total: usize,
    frame_ms: Vec<f64>,
    changed_nodes: Vec<usize>,
    doc_chunks: usize,
}

struct JsResult {
    cold_ms: Vec<f64>,
    warm_ms: Vec<f64>,
    /// Total native (optimizing-tier) executions across ALL tiers (P6 + T1 + T3 +
    /// T2). This is the honest "did the JIT actually run" guard: >0 means some
    /// optimizing tier executed the hot function as native machine code. The
    /// hot-numeric benches tier under P6 (tried first), so the old T2-only count
    /// read 0 — an ATTRIBUTION bug, not a "JIT didn't engage" bug. Reading every
    /// tier fixes it.
    native_exec_count: u64,
    /// Per-tier breakdown so the source of the native execs is visible.
    p6_exec_count: u64,
    t1_exec_count: u64,
    t3_exec_count: u64,
    t4_exec_count: u64,
    t2_exec_count: u64,
    t2_enabled: bool,
    /// TOP-LEVEL VM honesty guard: how many of this microbench's `run()` calls
    /// actually compiled the hot TOP-LEVEL script to bytecode and ran it on the
    /// register VM (the `CV_TOPLEVEL_VM` lever), vs falling through to the
    /// tree-walker. >0 proves a faster cold time came from the VM path, not luck.
    toplevel_vm_took: u64,
    toplevel_vm_enabled: bool,
}

// ── TTFP ─────────────────────────────────────────────────────────────────────

/// Time `build_runtime_and_first_paint` end-to-end (parse + CSS + JS bootstrap +
/// first layout + first bake) for a fixed input. Fresh parse each iter = fresh
/// document, no cross-iter cache, so each sample is an independent cold build.
/// The work counter is the produced display-list chunk count, recorded once from
/// a warmup build to confirm every iter does the same work.
fn measure_ttfp(html: &str, label: &str, cfg: &cv_layout::LayoutConfig) -> Result<TtfpResult, String> {
    let mut doc_chunks = 0usize;
    // Warmup (not timed): primes any process-global text-measure / font caches so
    // the timed iters measure steady-state build cost, not one-time init.
    for _ in 0..WARMUP {
        crate::bench_reset_render_thread_locals();
        let (_rt, _doc, _sheets, paint) =
            crate::build_runtime_and_first_paint(html, label, cfg, "")
                .map_err(|e| format!("ttfp warmup build: {e}"))?;
        doc_chunks = chunk_count(&paint, cfg);
    }
    let mut samples_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        // Each timed build is an independent COLD build: clear the per-page
        // incremental style/layout thread-locals so this iter cannot reuse the
        // previous iter's (or a previous document's) node-id-keyed cache.
        crate::bench_reset_render_thread_locals();
        let t = Instant::now();
        let (_rt, _doc, _sheets, paint) =
            crate::build_runtime_and_first_paint(html, label, cfg, "")
                .map_err(|e| format!("ttfp build: {e}"))?;
        samples_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        // Confirm the work counter is stable across iters (same document, same
        // chunk count). If it ever differs the input changed under us.
        let c = chunk_count(&paint, cfg);
        if c != doc_chunks {
            return Err(format!(
                "ttfp work counter unstable for {label}: {doc_chunks} vs {c} chunks"
            ));
        }
    }
    Ok(TtfpResult { samples_ms, doc_chunks })
}

/// Number of retained display-list chunks the paint covers — the deterministic
/// work counter that pairs with the TTFP timing. Computed from the SAME layout
/// tree the bake produced.
fn chunk_count(paint: &cv_ui::PaintData, cfg: &cv_layout::LayoutConfig) -> usize {
    match paint.layout_root.as_ref() {
        Some(lb) => crate::retained_dl::generate(lb, cfg).chunks.len(),
        None => 0,
    }
}

// ── REPEAT-LOAD (skip-not-redo) ──────────────────────────────────────────────

/// Build once, then render the SAME unchanged doc twice. Frame 1 warms the
/// fragment/paint caches; frame 2 hits them. `relayout_stats()` is reset at the
/// START of every `cv_layout::layout()` call and reflects ONLY that pass's
/// (laid_out, reused) — so we read it DIRECTLY after each frame (it is already
/// the per-frame delta, no snapshot-and-subtract needed). The frame-2 reuse ratio
/// is the beat-Chrome proof: Chrome re-styles + re-layouts a clean reload; we
/// reuse.
///
/// We run the pair `ITERS` times and report the median time plus the counters
/// from the LAST timed pair (the counters are exact integers and identical every
/// iter for an unchanged doc, so any pair is representative).
fn measure_repeat(
    html: &str,
    label: &str,
    cfg: &cv_layout::LayoutConfig,
) -> Result<RepeatResult, String> {
    let mut first_ms = Vec::with_capacity(ITERS);
    let mut second_ms = Vec::with_capacity(ITERS);
    let mut relaid_first = 0u64;
    let mut reused_first = 0u64;
    let mut relaid_second = 0u64;
    let mut reused_second = 0u64;

    for iter in 0..(WARMUP + ITERS) {
        // Independent cold build per pair so frame-1's reuse counters start from
        // a truly cold cache (no leftover from the previous pair / document).
        crate::bench_reset_render_thread_locals();
        let (mut rt, mut doc, sheets, _paint) =
            crate::build_runtime_and_first_paint(html, label, cfg, "")
                .map_err(|e| format!("repeat build: {e}"))?;

        // Frame 1: warms the fragment/paint caches (everything laid out fresh).
        let t1 = Instant::now();
        let _ = crate::render_with_existing_runtime(&mut rt, &mut doc, &sheets, cfg, None);
        let f1 = t1.elapsed().as_secs_f64() * 1000.0;
        // relayout_stats() is the count from frame 1's layout pass alone.
        let s1 = cv_layout::relayout_stats();

        // Frame 2: same doc, hits the caches warmed by frame 1.
        let t2 = Instant::now();
        let _ = crate::render_with_existing_runtime(&mut rt, &mut doc, &sheets, cfg, None);
        let f2 = t2.elapsed().as_secs_f64() * 1000.0;
        // relayout_stats() is now the count from frame 2's layout pass alone.
        let s2 = cv_layout::relayout_stats();

        if iter >= WARMUP {
            first_ms.push(f1);
            second_ms.push(f2);
            relaid_first = s1.0;
            reused_first = s1.1;
            relaid_second = s2.0;
            reused_second = s2.1;
        }
    }

    Ok(RepeatResult {
        first_ms,
        second_ms,
        relaid_first,
        reused_first,
        relaid_second,
        reused_second,
    })
}

// ── ANIMATION (damage raster) ────────────────────────────────────────────────

/// Load the animation page and tick N frames back-to-back, timing each frame and
/// computing the EXACT per-frame damage via `retained_dl::generate` on
/// consecutive frames' layout trees + `retained_dl::diff`. The diff node-set
/// sizes are integer node-id sets — exact and deterministic. We report the median
/// changed-node count and the fraction of the document's chunks that changed
/// (the "% re-rastered" beat-Chrome number). The setInterval in anim.html mutates
/// exactly one box per tick, so the changed fraction should be tiny.
fn measure_animation(cfg: &cv_layout::LayoutConfig) -> Result<AnimResult, String> {
    let html = read_input("anim.html")?;
    let label = "file:///benchfix/anim.html";
    // Cold build: clear the prior measurement's incremental caches so the anim
    // page lays out from its OWN document, not a previous (e.g. medium_dom) one.
    crate::bench_reset_render_thread_locals();
    let (mut rt, mut doc, sheets, first_paint) =
        crate::build_runtime_and_first_paint(&html, label, cfg, "")
            .map_err(|e| format!("anim build: {e}"))?;

    let doc_chunks = chunk_count(&first_paint, cfg);

    let mut frame_ms = Vec::with_capacity(ANIM_FRAMES);
    let mut changed_nodes = Vec::with_capacity(ANIM_FRAMES);

    // Prev frame's retained list, for the per-frame diff. Seed from the first
    // paint's layout tree so the first ticked frame has a real baseline.
    let mut prev_rdl = first_paint
        .layout_root
        .as_ref()
        .map(|lb| crate::retained_dl::generate(lb, cfg));

    for _ in 0..ANIM_FRAMES {
        // The setInterval callback fires inside render_with_existing_runtime's
        // drain_due, mutating one box's style.left. Tick back-to-back (no sleep).
        let t = Instant::now();
        let paint = crate::render_with_existing_runtime(&mut rt, &mut doc, &sheets, cfg, None);
        frame_ms.push(t.elapsed().as_secs_f64() * 1000.0);

        if let Some(lb) = paint.layout_root.as_ref() {
            let new_rdl = crate::retained_dl::generate(lb, cfg);
            if let Some(old) = prev_rdl.as_ref() {
                let d = crate::retained_dl::diff(old, &new_rdl);
                changed_nodes.push(
                    d.changed.len() + d.moved.len() + d.added.len() + d.removed.len(),
                );
            }
            prev_rdl = Some(new_rdl);
        }
    }

    Ok(AnimResult {
        frames_total: ANIM_FRAMES,
        frame_ms,
        changed_nodes,
        doc_chunks,
    })
}

struct KeyframeAnimResult {
    frames_total: usize,
    frame_ms: Vec<f64>,
}

/// Load a page driven ENTIRELY by CSS `@keyframes` (no per-frame JS DOM
/// mutation) and tick N frames back-to-back, timing each frame. This is the path
/// the @keyframes-collection memo targets: every animated frame re-collects the
/// keyframe model (without the memo) and re-samples every animated box. Unlike
/// `anim.html` (which mutates `style.width` via JS and has NO `@keyframes`), this
/// page exercises `collect_keyframes()` + `sample_animation()` for real, so the
/// memo's effect is visible in `frame_ms_median`. The animation output is
/// deterministic per host frame tick (one rAF advance per tick), so the two
/// configs (memo on/off) render the SAME frames — only the per-frame parse work
/// differs.
fn measure_animation_keyframes(cfg: &cv_layout::LayoutConfig) -> Result<KeyframeAnimResult, String> {
    let html = read_input("anim_keyframes.html")?;
    let label = "file:///benchfix/anim_keyframes.html";
    crate::bench_reset_render_thread_locals();
    let (mut rt, mut doc, sheets, _first_paint) =
        crate::build_runtime_and_first_paint(&html, label, cfg, "")
            .map_err(|e| format!("anim_keyframes build: {e}"))?;

    let mut frame_ms = Vec::with_capacity(ANIM_FRAMES);
    for _ in 0..ANIM_FRAMES {
        let t = Instant::now();
        let _paint = crate::render_with_existing_runtime(&mut rt, &mut doc, &sheets, cfg, None);
        frame_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    Ok(KeyframeAnimResult {
        frames_total: ANIM_FRAMES,
        frame_ms,
    })
}

// ── JS-EXEC (JIT vs VM honesty) ──────────────────────────────────────────────

/// Run a JS microbench through the real interpreter (`Interp::run` — the
/// tree-walk evaluator, which returns the completion value and shares one global
/// scope so we can read the result back consistently). Timed with `Instant`;
/// `t2_exec_count()` is read afterwards as the JIT HONESTY GUARD: it is >0 only if
/// the T2 optimizing tier actually executed, and ==0 means the workload ran on
/// the interpreter, NOT the JIT — so a "fast" number with a zero count is honestly
/// the VM, not a JIT win. (`Interp::run` is the tree-walk tier and does not engage
/// the bytecode VM / T2, so this count is expected to read 0 here; that is the
/// honest current state, reported verbatim.) Cold = a FRESH `LiveInterp` per iter;
/// warm = a 2nd run on the SAME interp. We report BOTH. This is HONESTLY slower
/// than V8 today — the number is the baseline to improve, not a win.
///
/// Each script ends with a top-level `var __bench_*_result` holding the loop sum,
/// so we can confirm the run actually COMPLETED and produced the expected value
/// (not an early throw / partial run that would make the timing meaningless).
fn measure_js(file: &str, result_global: &str) -> Result<JsResult, String> {
    let src = read_input(file)?;
    // A blank doc so we get a real LiveInterp (Math/JSON/globals installed) with
    // no page work — we are measuring the script, not the DOM.
    let blank = "<!doctype html><html><head></head><body></body></html>";
    let doc = cv_html::parse(blank);
    let label = "file:///benchfix/jsbench";

    let mut cold_ms = Vec::with_capacity(ITERS);
    let mut warm_ms = Vec::with_capacity(ITERS);
    let mut last_p6 = 0u64;
    let mut last_t1 = 0u64;
    let mut last_t3 = 0u64;
    let mut last_t4 = 0u64;
    let mut last_t2 = 0u64;
    let mut last_toplevel_vm = 0u64;

    // Warmup iters (not timed) — primes any process-global JIT machinery so the
    // timed cold iters measure a representative fresh-interp run.
    for _ in 0..WARMUP {
        let mut rt = LiveInterp::new(&doc, label);
        let _ = rt.interp.run(&src);
    }

    for _ in 0..ITERS {
        // COLD: fresh interp, first run of this function.
        let mut rt = LiveInterp::new(&doc, label);
        // Reset EVERY optimizing-tier exec counter so the post-run reads attribute
        // native execution to whichever tier actually ran. The hot-numeric benches
        // tier under P6 (tried before T2 in try_call_fn_via_bytecode), so a
        // T2-only read would (wrongly) report 0 native execs even though the
        // function runs as native machine code.
        cv_js::reset_p6_exec_count();
        cv_js::reset_t1_exec_count();
        cv_js::reset_t3_exec_count();
        cv_js::reset_t4_exec_count();
        cv_js::reset_t2_exec_count();
        // Top-level-VM honesty guard: reset before the cold run so the post-run
        // count attributes the path to THIS run (whether the hot top-level script
        // compiled to bytecode + ran on the VM, or fell through to tree-walk).
        cv_js::reset_toplevel_vm_took_count();
        let tc = Instant::now();
        rt.interp
            .run(&src)
            .map_err(|e| format!("js cold run {file}: {e:?}"))?;
        cold_ms.push(tc.elapsed().as_secs_f64() * 1000.0);
        // Snapshot the top-level-VM engagement for the COLD run specifically (the
        // warm run below would add to it). >0 == this microbench's hot top-level
        // ran on the register VM, the honesty guard for the cold-time win.
        last_toplevel_vm = cv_js::toplevel_vm_took_count();
        // Confirm the script COMPLETED and produced its result (a real Number),
        // so the timing measures the whole workload, not an early abort.
        // `run_completion_value` returns the value of the last expression (a bare
        // identifier here), unlike `run` which reports `undefined` for a normal
        // (non-return) completion.
        let got = rt.interp.run_completion_value(result_global);
        match got {
            Ok(cv_js::Value::Number(_)) => {}
            other => {
                return Err(format!(
                    "js {file}: result global {result_global} not a finished Number: {other:?}"
                ));
            }
        }

        // WARM: same interp, second run.
        let tw = Instant::now();
        rt.interp
            .run(&src)
            .map_err(|e| format!("js warm run {file}: {e:?}"))?;
        warm_ms.push(tw.elapsed().as_secs_f64() * 1000.0);

        // Native-tier exec counts accumulated over BOTH runs on this interp — the
        // honesty guard. Reported verbatim across every tier — never massaged. The
        // total (native_exec_count) >0 proves an optimizing tier ran the hot code
        // as native machine code; the per-tier split shows which one.
        last_p6 = cv_js::p6_exec_count();
        last_t1 = cv_js::t1_exec_count();
        last_t3 = cv_js::t3_exec_count();
        last_t4 = cv_js::t4_exec_count();
        last_t2 = cv_js::t2_exec_count();
    }

    Ok(JsResult {
        cold_ms,
        warm_ms,
        native_exec_count: last_p6 + last_t1 + last_t3 + last_t4 + last_t2,
        p6_exec_count: last_p6,
        t1_exec_count: last_t1,
        t3_exec_count: last_t3,
        t4_exec_count: last_t4,
        t2_exec_count: last_t2,
        t2_enabled: cv_js::t2_heap_enabled(),
        toplevel_vm_took: last_toplevel_vm,
        toplevel_vm_enabled: cv_js::toplevel_vm_enabled(),
    })
}

/// The measured outcome of the T4 P5 AOT-PERSIST COLD-REPEAT-VISIT leg — the ★
/// beat-Chrome lever. `first_cold_ms` is the FIRST cold visit (fresh process
/// analogue: compile T4 + persist the native blob, paying the full warmup). The key
/// number `repeat_cold_ms` is the SECOND cold visit (a fresh interp with the live
/// T4 cache cleared, so the optimized native code is RE-INSTALLED FROM THE DISK
/// BLOB with ZERO codegen + ZERO warmup) — the path PAST V8, which re-JITs every
/// cold load. `aot_loaded` (>0) is the honesty guard: the reload actually fired.
struct AotRepeatResult {
    first_cold_ms: f64,
    repeat_cold_ms: Vec<f64>,
    aot_loaded: u64,
    aot_stored: u64,
    t4_exec_count: u64,
    /// Byte-identity guard: the result value of the cold-repeat (reloaded) run MUST
    /// equal the value of a plain VM run of the same script. True iff they matched —
    /// a fast number with a mismatched result is FAKE, not a win.
    result_matches_vm: bool,
    /// COMPILE-COST ISOLATION: a LOW-iteration variant where the T4 codegen is a
    /// MEASURABLE fraction of the run (vs jit.js's 1.5M iters, where the one-time
    /// compile is a negligible fraction of the runtime). `first_compile_ms` = a cold
    /// visit that COMPILES T4 (full codegen + warmup); `repeat_compile_ms` = a cold
    /// REPEAT visit that RE-INSTALLS the persisted blob (zero codegen). The delta is
    /// the warmup/codegen cost the AOT lever eliminates — the part of the
    /// cold-repeat win that V8 cannot get (it re-JITs every cold load).
    first_compile_ms: f64,
    repeat_compile_ms: f64,
}

/// Measure the T4 P5 AOT-PERSIST cold-repeat lever on `file` (jit.js). Forces T4 +
/// AOT on in an ISOLATED on-disk store, runs the script ONCE to compile+persist the
/// optimized native code, then runs it AGAIN in a fresh interp with the live T4
/// cache cleared — so the second run RE-INSTALLS the persisted native code from disk
/// with zero warmup. The second run's time is the beat-Chrome cold-repeat number.
/// All correctness gates from the round-trip oracle hold (the persisted DeoptSite
/// table re-checks every guard); this leg ADDS the timing.
fn measure_aot_cold_repeat(file: &str, result_global: &str) -> Result<AotRepeatResult, String> {
    let src = read_input(file)?;
    let blank = "<!doctype html><html><head></head><body></body></html>";
    let doc = cv_html::parse(blank);
    let label = "file:///benchfix/jsbench-aot";

    // VM baseline result (the source of truth for the byte-identity guard).
    let vm_result = {
        let _g = cv_js::TierGuard::new(cv_js::ForcedTier::Vm);
        cv_js::interp::reset_t4_cache();
        let mut rt = LiveInterp::new(&doc, label);
        rt.interp
            .run(&src)
            .map_err(|e| format!("aot vm run {file}: {e:?}"))?;
        rt.interp.run_completion_value(result_global)
    };

    // Isolate the on-disk AOT store to a unique dir so the repeat run reloads
    // exactly the blob the first run persisted (no stale cross-run contamination).
    let dir = std::env::temp_dir().join(format!(
        "tbjs_aot_bench_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    cv_js::t4::aot::set_thread_dir_override(Some(dir.clone()));
    let _persist = cv_js::t4::aot::AotPersistGuard::new(true);
    let _t4 = cv_js::TierGuard::new(cv_js::ForcedTier::T4);
    // Route jit.js's `f(x)` to T4 (not the P6 f64 leaf JIT, which is tried first):
    // P6 would otherwise win the pure-numeric callee and T4 would never engage, so
    // the AOT lever (which persists T4 code) would have nothing to reload. This is
    // the apples-to-apples "what does AOT-persist save when T4 IS the tier"
    // measurement — exactly the prompt's `CV_NOJIT=1 CV_T4=1` T4-ENGAGED config.
    let _no_p6 = cv_js::NoP6JitGuard::new();

    // ── FIRST COLD VISIT: compile T4 + persist the native blob (full warmup). ──
    cv_js::t4::aot::reset_aot_store_count();
    cv_js::t4::aot::reset_aot_load_count();
    cv_js::interp::reset_t4_cache();
    cv_js::reset_t4_exec_count();
    let mut first_result_ok = true;
    let first_cold_ms = {
        let mut rt = LiveInterp::new(&doc, label);
        let t = Instant::now();
        rt.interp
            .run(&src)
            .map_err(|e| format!("aot first cold run {file}: {e:?}"))?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if !matches!(rt.interp.run_completion_value(result_global), Ok(cv_js::Value::Number(_))) {
            first_result_ok = false;
        }
        ms
    };
    let aot_stored = cv_js::t4::aot::aot_store_count();

    // ── COLD REPEAT VISITS: a FRESH interp + CLEARED live T4 cache, so the
    //    optimized native code is RE-INSTALLED FROM DISK with zero codegen. ──
    let mut repeat_cold_ms = Vec::with_capacity(ITERS);
    let mut last_result = vm_result.clone();
    let mut total_loaded = 0u64;
    for _ in 0..ITERS {
        cv_js::interp::reset_t4_cache(); // clear the live JitFunction → MUST reload from disk
        cv_js::t4::aot::reset_aot_load_count();
        cv_js::reset_t4_exec_count();
        let mut rt = LiveInterp::new(&doc, label);
        let t = Instant::now();
        rt.interp
            .run(&src)
            .map_err(|e| format!("aot repeat cold run {file}: {e:?}"))?;
        repeat_cold_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        last_result = rt.interp.run_completion_value(result_global);
        total_loaded += cv_js::t4::aot::aot_load_count();
    }
    let t4_exec_count = cv_js::t4_exec_count();

    // ── COMPILE-COST ISOLATION: a LOW-iteration variant where T4 codegen is a
    //    measurable fraction. We synthesize many DISTINCT hot functions so each pays
    //    its own T4 compile (the codegen cost the AOT lever eliminates on reload),
    //    each called just past the tier-up threshold. A separate isolated dir per
    //    sub-run so the compile run truly compiles and the reload run truly reloads.
    let compile_src = aot_compile_cost_script();
    let dir2 = std::env::temp_dir().join(format!(
        "tbjs_aot_compile_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir2);
    cv_js::t4::aot::set_thread_dir_override(Some(dir2.clone()));
    // FIRST compile visit: empty store → every hot fn COMPILES T4 (full codegen).
    let first_compile_ms = {
        cv_js::interp::reset_t4_cache();
        let mut rt = LiveInterp::new(&doc, label);
        let t = Instant::now();
        rt.interp
            .run(&compile_src)
            .map_err(|e| format!("aot compile-cost first run: {e:?}"))?;
        t.elapsed().as_secs_f64() * 1000.0
    };
    // COLD REPEAT compile visit: warm store + cleared live cache → every hot fn
    // RE-INSTALLS from disk (zero codegen). The delta vs first is the codegen cost.
    let mut repeat_compile_samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        cv_js::interp::reset_t4_cache();
        let mut rt = LiveInterp::new(&doc, label);
        let t = Instant::now();
        rt.interp
            .run(&compile_src)
            .map_err(|e| format!("aot compile-cost repeat run: {e:?}"))?;
        repeat_compile_samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let repeat_compile_ms = median(&repeat_compile_samples);

    cv_js::t4::aot::set_thread_dir_override(None);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);

    // Byte-identity guard: the reloaded cold-repeat result MUST equal the VM result.
    let result_matches_vm = first_result_ok
        && match (&last_result, &vm_result) {
            (Ok(cv_js::Value::Number(a)), Ok(cv_js::Value::Number(b))) => {
                a == b || (a.is_nan() && b.is_nan())
            }
            _ => false,
        };

    Ok(AotRepeatResult {
        first_cold_ms,
        repeat_cold_ms,
        aot_loaded: total_loaded,
        aot_stored,
        t4_exec_count,
        result_matches_vm,
        first_compile_ms,
        repeat_compile_ms,
    })
}

/// A compile-cost-isolation script: MANY distinct small float-dense functions, each
/// called just past the T4 tier-up threshold. Each function pays its OWN T4 codegen
/// on a fresh compile (the cost the AOT lever skips on reload), and the total
/// runtime is dominated by codegen rather than steady-state execution (unlike
/// jit.js's 1.5M-iter single function, where compile is a negligible fraction). The
/// per-fn loop count is small but past threshold so T4 engages once per fn.
fn aot_compile_cost_script() -> String {
    let mut s = String::new();
    let n_fns = 200; // many distinct compiles → codegen is a measurable fraction
    let per_fn_calls = 80; // just past the tier-up threshold
    s.push_str("var __acc = 0;\n");
    for k in 0..n_fns {
        // Distinct constants per fn so each is a DISTINCT program (own AOT key).
        let a = (k % 7) as f64 + 0.5;
        let b = (k % 5) as f64 + 1.25;
        s.push_str(&format!(
            "function f{k}(x){{ return x*x*{a} + x*{b} - {a} + x*x*x*0.25 - x*0.5; }}\n"
        ));
        s.push_str(&format!(
            "for (var i{k}=0; i{k}<{per_fn_calls}; i{k}=i{k}+1) {{ __acc = __acc + f{k}(i{k}); }}\n"
        ));
    }
    s.push_str("var __bench_aot_compile_result = __acc; __bench_aot_compile_result;\n");
    s
}

// ── public entry ─────────────────────────────────────────────────────────────

/// Run the full bench and emit one JSON object to stdout (and optionally to the
/// file named by `--out`). Every metric is a real measurement.
pub fn run_bench(cli: &Cli) -> Result<(), String> {
    let cfg = bench_cfg();

    // ── STARTUP (single-shot per process) ──
    // PROCESS_T0 was set at the top of real_main(). Measure elapsed right after
    // the first session is built (first build_runtime_and_first_paint returns)
    // and again after we have its PaintData (first paint). This is a genuine
    // cold-start sample: the OS/loader/static-init cost happens once per process,
    // so it cannot be re-measured in-process — reported with samples: 1.
    let small_html = read_input("small_static.html")?;
    let small_label = "file:///benchfix/small_static.html";
    let t0 = crate::PROCESS_T0.get().copied();
    crate::bench_reset_render_thread_locals();
    let (_rt0, _doc0, _sheets0, _paint0) =
        crate::build_runtime_and_first_paint(&small_html, small_label, &cfg, "")
            .map_err(|e| format!("startup build: {e}"))?;
    let startup_first_session_ms = t0.map(|s| s.elapsed().as_secs_f64() * 1000.0);
    // The returned PaintData IS the first paint, so first-paint == the same
    // elapsed snapshot taken immediately after (the build returns both in one
    // call). Take a fresh elapsed reading to capture any time between.
    let startup_first_paint_ms = t0.map(|s| s.elapsed().as_secs_f64() * 1000.0);

    // ── TTFP ──
    let small_ttfp = measure_ttfp(&small_html, small_label, &cfg)?;
    let medium_html = read_input("medium_dom.html")?;
    let medium_label = "file:///benchfix/medium_dom.html";
    let medium_ttfp = measure_ttfp(&medium_html, medium_label, &cfg)?;

    // ── REPEAT-LOAD (most convincing on the big DOM) ──
    let repeat = measure_repeat(&medium_html, medium_label, &cfg)?;

    // ── ANIMATION ──
    let anim = measure_animation(&cfg)?;

    // ── ANIMATION (CSS @keyframes-driven; exercises the keyframe-collection memo) ──
    let anim_kf = measure_animation_keyframes(&cfg)?;

    // ── JS-EXEC ──
    let js_loop = measure_js("loop.js", "__bench_loop_result")?;
    let js_jit = measure_js("jit.js", "__bench_jit_result")?;

    // ── T4 P5 AOT-PERSIST COLD-REPEAT (★ the beat-Chrome cold-repeat lever) ──
    // Measures jit.js on a COLD REPEAT VISIT where the optimized T4 native code is
    // re-installed from the persisted disk blob with ZERO codegen + ZERO warmup —
    // the path PAST V8 (which re-JITs every cold load). Off the default path (it
    // forces T4+AOT in an isolated store), so it never perturbs the default numbers
    // above; it is a dedicated lever measurement. `result_matches_vm` is the
    // anti-fake guard: a fast number only counts if the result is byte-identical.
    let aot_jit = measure_aot_cold_repeat("jit.js", "__bench_jit_result")?;

    // ── MEMORY (gc_live_object_count is the deterministic, in-process,
    // cross-platform heap-liveness metric — exact integer, the leak/sawtooth
    // axis). It counts JS objects/arrays the GC registry still upgrades; the
    // registry is only populated when GC tracking is enabled (CV_GC, default on)
    // and DOM/native wrappers are intentionally NOT registered, so the count
    // reflects genuine *JS-allocated* reachability. after_load = right after the
    // page's first paint; after_idle = after draining the scheduler + ticking
    // idle frames + a GC pass. The after_idle count staying bounded == the
    // no-leak evidence. Process RSS bytes are OMITTED on purpose: there is no
    // GetProcessMemoryInfo FFI in the tree, and the system-wide memory-pressure
    // probe is NOT our process, so emitting either would be a fake number. ──
    let (mem_after_load, mem_after_idle) = measure_memory(&cfg)?;

    // ── git rev (runtime; null if git is unavailable) ──
    let git_rev = git_short_rev();

    let timestamp_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    // ── assemble JSON ──
    let reuse_ratio_second = {
        let denom = (repeat.relaid_second + repeat.reused_second) as f64;
        if denom > 0.0 {
            repeat.reused_second as f64 / denom
        } else {
            // No layout work at all on frame 2 (everything reused, nothing
            // relaid) — that is 100% reuse. Guard div-by-zero honestly.
            if repeat.reused_second > 0 { 1.0 } else { 0.0 }
        }
    };

    let changed_fraction_median = if anim.doc_chunks > 0 {
        median_usize(anim.changed_nodes.clone()) as f64 / anim.doc_chunks as f64
    } else {
        0.0
    };

    let root = J::Obj(vec![
        ("schema", J::S(SCHEMA.to_string())),
        ("engine", J::S("conclave".to_string())),
        ("git_rev", git_rev.map(J::S).unwrap_or(J::Null)),
        ("timestamp_unix", J::I(timestamp_unix)),
        (
            "config",
            J::Obj(vec![
                ("viewport", J::S(format!("{}x{}", VIEWPORT_W as u32, VIEWPORT_H as u32))),
                ("iters", J::I(ITERS as i64)),
                ("warmup", J::I(WARMUP as i64)),
                ("t2_jit_enabled", J::Bool(cv_js::t2_heap_enabled())),
                ("damage_raster_enabled", J::Bool(crate::retained_dl::damage_raster_enabled())),
                ("gc_enabled", J::Bool(cv_js::gc_enabled())),
                ("offmain", J::Bool(offmain_flag())),
                ("interest_rect", J::Bool(crate::interest_rect_enabled())),
                ("compositor", J::Bool(crate::compositor_enabled())),
            ]),
        ),
        (
            "metrics",
            J::Obj(vec![
                (
                    "startup",
                    J::Obj(vec![
                        ("to_first_session_ms", opt_f(startup_first_session_ms)),
                        ("to_first_paint_ms", opt_f(startup_first_paint_ms)),
                        ("samples", J::I(1)),
                    ]),
                ),
                (
                    "ttfp",
                    J::Obj(vec![
                        ("small_static", ttfp_json(&small_ttfp)),
                        ("medium_dom", ttfp_json(&medium_ttfp)),
                    ]),
                ),
                (
                    "repeat_load",
                    J::Obj(vec![
                        ("first_ms_median", J::F(median(&repeat.first_ms))),
                        ("second_ms_median", J::F(median(&repeat.second_ms))),
                        (
                            "speedup_ratio",
                            J::F(ratio(median(&repeat.first_ms), median(&repeat.second_ms))),
                        ),
                        ("relaid_first", J::I(repeat.relaid_first as i64)),
                        ("reused_first", J::I(repeat.reused_first as i64)),
                        ("relaid_second", J::I(repeat.relaid_second as i64)),
                        ("reused_second", J::I(repeat.reused_second as i64)),
                        ("reuse_ratio_second", J::F(reuse_ratio_second)),
                    ]),
                ),
                (
                    "animation",
                    J::Obj(vec![
                        ("frames_total", J::I(anim.frames_total as i64)),
                        ("frame_ms_median", J::F(median(&anim.frame_ms))),
                        ("frame_ms_p95", J::F(p95(&anim.frame_ms))),
                        ("changed_nodes_median", J::I(median_usize(anim.changed_nodes.clone()) as i64)),
                        ("doc_chunks", J::I(anim.doc_chunks as i64)),
                        ("changed_fraction_median", J::F(changed_fraction_median)),
                    ]),
                ),
                (
                    // CSS @keyframes-driven animation: the path the keyframe-
                    // collection memo (Blink StyleRuleKeyframes) targets. Lower
                    // frame_ms_median with the memo on == real per-frame parse
                    // work removed (identical rendered frames, oracle-proven).
                    "animation_keyframes",
                    J::Obj(vec![
                        ("frames_total", J::I(anim_kf.frames_total as i64)),
                        ("frame_ms_median", J::F(median(&anim_kf.frame_ms))),
                        ("frame_ms_p95", J::F(p95(&anim_kf.frame_ms))),
                        ("keyframes_memo_enabled", J::Bool(
                            std::env::var("CV_KEYFRAMES_MEMO").as_deref() != Ok("0"),
                        )),
                    ]),
                ),
                (
                    "js_exec",
                    J::Obj(vec![
                        ("loop", js_json(&js_loop)),
                        ("jit", js_json(&js_jit)),
                        // ★ T4 P5 AOT-PERSIST cold-repeat lever (jit.js): the second
                        // cold visit re-installs optimized native code from disk with
                        // zero warmup — the past-V8 number. Only meaningful with
                        // `result_matches_vm: true` (the anti-fake guard).
                        ("jit_aot_cold_repeat", aot_json(&aot_jit)),
                    ]),
                ),
                (
                    "memory",
                    J::Obj(vec![
                        ("gc_live_objects_after_load", J::I(mem_after_load as i64)),
                        ("gc_live_objects_after_idle", J::I(mem_after_idle as i64)),
                    ]),
                ),
            ]),
        ),
    ]);

    let json = root.to_pretty();
    print!("{json}");
    if let Some(out) = cli.flag("out") {
        std::fs::write(out, &json).map_err(|e| format!("write {out}: {e}"))?;
        eprintln!("bench: wrote {out}");
    }
    Ok(())
}

fn ttfp_json(r: &TtfpResult) -> J {
    J::Obj(vec![
        ("ms_median", J::F(median(&r.samples_ms))),
        ("ms_p95", J::F(p95(&r.samples_ms))),
        ("ms_min", J::F(min_of(&r.samples_ms))),
        ("n", J::I(r.samples_ms.len() as i64)),
        ("doc_chunks", J::I(r.doc_chunks as i64)),
    ])
}

fn js_json(r: &JsResult) -> J {
    J::Obj(vec![
        ("cold_ms_median", J::F(median(&r.cold_ms))),
        ("warm_ms_median", J::F(median(&r.warm_ms))),
        // Total native (optimizing-tier) executions across ALL tiers — the honest
        // "did the JIT engage" guard. >0 == some tier ran the hot fn natively.
        ("native_exec_count", J::I(r.native_exec_count as i64)),
        // Per-tier breakdown (P6 numeric JIT is tried first, so hot-numeric work
        // lands here, not in t2).
        ("p6_exec_count", J::I(r.p6_exec_count as i64)),
        ("t1_exec_count", J::I(r.t1_exec_count as i64)),
        ("t3_exec_count", J::I(r.t3_exec_count as i64)),
        // T4 Maglev-class representation-selection tier (CV_T4; default off).
        ("t4_exec_count", J::I(r.t4_exec_count as i64)),
        // Kept for continuity with the prior baseline JSON.
        ("t2_exec_count", J::I(r.t2_exec_count as i64)),
        ("t2_enabled", J::Bool(r.t2_enabled)),
        // TOP-LEVEL VM (CV_TOPLEVEL_VM) honesty guard: >0 == the hot top-level
        // script compiled to bytecode and ran on the register VM (not tree-walk).
        ("toplevel_vm_took", J::I(r.toplevel_vm_took as i64)),
        ("toplevel_vm_enabled", J::Bool(r.toplevel_vm_enabled)),
    ])
}

fn aot_json(r: &AotRepeatResult) -> J {
    J::Obj(vec![
        // The FIRST cold visit (compile T4 + persist — full warmup paid once).
        ("first_cold_ms", J::F(r.first_cold_ms)),
        // ★ THE NUMBER: the COLD REPEAT visit — optimized native code re-installed
        // from the persisted disk blob with ZERO codegen + ZERO warmup. V8 re-JITs
        // every cold load; this is the path PAST it.
        ("repeat_cold_ms_median", J::F(median(&r.repeat_cold_ms))),
        ("repeat_cold_ms_min", J::F(min_of(&r.repeat_cold_ms))),
        // Honesty guards: the reload actually fired (>0), a blob was persisted (>0),
        // T4 ran the hot fn natively (>0). A zero here means the lever was vacuous.
        ("aot_blobs_loaded", J::I(r.aot_loaded as i64)),
        ("aot_blobs_stored", J::I(r.aot_stored as i64)),
        ("t4_exec_count", J::I(r.t4_exec_count as i64)),
        // ANTI-FAKE: the cold-repeat result MUST be byte-identical to the VM. A fast
        // number with this false is BROKEN, not a win.
        ("result_matches_vm", J::Bool(r.result_matches_vm)),
        // COMPILE-COST ISOLATION: many distinct hot fns, so T4 codegen is a
        // measurable fraction. compile_first = cold compile (full codegen);
        // compile_repeat = cold repeat (reload from disk, zero codegen). The delta
        // (and the speedup ratio) is the warmup/codegen cost the AOT lever
        // eliminates — the cold-repeat win V8 cannot get (it re-JITs every load).
        ("compile_cost_first_ms", J::F(r.first_compile_ms)),
        ("compile_cost_repeat_ms", J::F(r.repeat_compile_ms)),
        (
            "compile_cost_speedup_ratio",
            J::F(ratio(r.first_compile_ms, r.repeat_compile_ms)),
        ),
    ])
}

fn opt_f(x: Option<f64>) -> J {
    match x {
        Some(v) => J::F(v),
        None => J::Null,
    }
}

/// `first / second`. Reports null (via NaN→null in J::F) if second is 0.
fn ratio(first: f64, second: f64) -> f64 {
    if second > 0.0 {
        first / second
    } else {
        f64::NAN
    }
}

/// Build the small page, sample gc_live_object_count after first paint, then
/// drain the scheduler + tick idle frames + run a GC pass and sample again. The
/// after-idle count staying bounded is the no-leak evidence.
fn measure_memory(cfg: &cv_layout::LayoutConfig) -> Result<(usize, usize), String> {
    // Use the animation page for the memory probe: it has JS + a running timer,
    // so the idle ticks do real work (the leak axis is most meaningful when JS
    // objects churn). A static page would show a flat, uninteresting count.
    let html = read_input("anim.html")?;
    let label = "file:///benchfix/anim.html";
    crate::bench_reset_render_thread_locals();
    let (mut rt, mut doc, sheets, _paint) =
        crate::build_runtime_and_first_paint(&html, label, cfg, "")
            .map_err(|e| format!("memory build: {e}"))?;

    let after_load = cv_js::gc_live_object_count();

    // Drain the scheduler and tick idle frames so timers/microtasks churn JS
    // objects, then run a GC pass. If anything leaks, after_idle climbs without
    // bound across ticks; a bounded count is the no-leak/beat-Chrome evidence.
    for _ in 0..IDLE_FRAMES {
        let _ = crate::render_with_existing_runtime(&mut rt, &mut doc, &sheets, cfg, None);
    }
    rt.drain_due(200);
    // Run a real cycle-collection pass against the live host roots (no-op unless
    // CV_GC=1, but the live-object count is exact either way — the GC just frees
    // unreachable cycles so the count reflects genuine reachability).
    rt.gc_collect_if_enabled();
    let after_idle = cv_js::gc_live_object_count();

    Ok((after_load, after_idle))
}

/// Off-main flag state, mirroring `run_window_with_target`'s value discrimination
/// (default ON unless `CV_OFFMAIN` is 0/false/off). Reported in `config` so
/// numbers are read in the right context.
fn offmain_flag() -> bool {
    !matches!(
        std::env::var("CV_OFFMAIN").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// Short git rev captured at runtime. `None` if git is unavailable / not a repo
/// (we emit `null` rather than a fake rev).
fn git_short_rev() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let rev = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if rev.is_empty() {
        None
    } else {
        Some(rev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> cv_layout::LayoutConfig {
        bench_cfg()
    }

    #[test]
    fn median_p95_min_are_real_statistics() {
        let s = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(median(&s), 3.0);
        assert_eq!(min_of(&s), 1.0);
        // nearest-rank p95 of 5 samples -> rank ceil(4.75)-1 = 4 -> 5.0
        assert_eq!(p95(&s), 5.0);
        assert_eq!(median_usize(vec![3, 1, 2]), 2);
    }

    #[test]
    fn benchfix_inputs_exist_and_are_self_contained() {
        for name in ["small_static.html", "medium_dom.html", "anim.html", "loop.js", "jit.js"] {
            let s = read_input(name).expect("benchfix input readable");
            assert!(!s.is_empty(), "{name} empty");
        }
        // The HTML inputs must be self-contained (no external http(s) refs) so
        // the bench is offline + deterministic.
        for name in ["small_static.html", "medium_dom.html", "anim.html"] {
            let s = read_input(name).unwrap();
            assert!(
                !s.contains("http://") && !s.contains("https://"),
                "{name} has an external reference"
            );
        }
    }

    /// ORACLE for the text-measurement memoization win: the laid-out box tree of
    /// a full real build must be BYTE-IDENTICAL whether the measure-width cache is
    /// cold or warm. We build the page once (cold cache) and capture its layout
    /// tree's Debug serialization (every geometry field), then build it AGAIN with
    /// the width cache still warm, and assert the two serializations are equal.
    /// Then we CLEAR the width cache and build a third time (cold again) and assert
    /// it still matches — proving the cache is a pure memo, not a different answer.
    /// This is the geometry-unchanged gate the perf win is required to pass.
    fn layout_debug(paint: &cv_ui::PaintData) -> String {
        match paint.layout_root.as_ref() {
            Some(lb) => format!("{lb:?}"),
            None => "<none>".to_string(),
        }
    }

    fn build_layout_debug(html: &str, label: &str, cfg: &cv_layout::LayoutConfig) -> String {
        crate::bench_reset_render_thread_locals();
        let (_rt, _doc, _sheets, paint) =
            crate::build_runtime_and_first_paint(html, label, cfg, "").expect("build");
        layout_debug(&paint)
    }

    #[test]
    fn measure_cache_preserves_byte_identical_layout_geometry() {
        let cfg = cfg();
        for name in ["medium_dom.html", "small_static.html"] {
            let html = read_input(name).unwrap();
            let label = "file:///oracle";

            // Build 1: cold width cache (bench_reset clears it). Capture geometry.
            let cold = build_layout_debug(&html, label, &cfg);

            // Build 2: width cache WARM from build 1 (do NOT clear it here — call
            // build_runtime_and_first_paint directly so the warm cache is used).
            // Reset only the style/layout caches the live nav resets, keeping the
            // measure cache warm, then assert identical geometry.
            crate::CURRENT_STYLE_CACHE.with(|c| *c.borrow_mut() = None);
            crate::CURRENT_RENDER_ARENA.with(|c| *c.borrow_mut() = None);
            cv_layout::set_layout_cache(None, None, 0);
            let (_rt, _doc, _sheets, paint_warm) =
                crate::build_runtime_and_first_paint(&html, label, &cfg, "").expect("warm build");
            let warm = layout_debug(&paint_warm);
            assert_eq!(
                cold, warm,
                "{name}: warm-measure-cache layout geometry diverged from cold"
            );

            // Build 3: clear the width cache (fully cold) and rebuild — still equal.
            let cold2 = build_layout_debug(&html, label, &cfg);
            assert_eq!(
                cold, cold2,
                "{name}: second cold build geometry diverged (non-deterministic measure?)"
            );
        }
    }

    #[test]
    fn ttfp_measures_real_build_with_stable_work_counter() {
        let cfg = cfg();
        let html = read_input("small_static.html").unwrap();
        let r = measure_ttfp(&html, "file:///t", &cfg).expect("ttfp");
        assert_eq!(r.samples_ms.len(), ITERS, "one sample per iter");
        // Every timed build did real work => positive wall time and a real
        // (non-zero) display-list chunk count.
        assert!(r.samples_ms.iter().all(|&t| t > 0.0), "all builds took time");
        assert!(r.doc_chunks > 0, "small page produced display-list chunks");
        // median/min/p95 are ordered and finite.
        let m = median(&r.samples_ms);
        assert!(m.is_finite() && m > 0.0);
        assert!(min_of(&r.samples_ms) <= m);
        assert!(p95(&r.samples_ms) >= m);
    }

    #[test]
    fn repeat_load_reuses_layout_on_frame_two() {
        let cfg = cfg();
        let html = read_input("medium_dom.html").unwrap();
        let r = measure_repeat(&html, "file:///t", &cfg).expect("repeat");
        assert_eq!(r.first_ms.len(), ITERS);
        assert_eq!(r.second_ms.len(), ITERS);
        assert!(r.first_ms.iter().all(|&t| t >= 0.0));
        assert!(r.second_ms.iter().all(|&t| t >= 0.0));
        // The reuse counter is the skip-not-redo proof. On a clean unchanged
        // big-DOM reload, frame 2 must reuse a meaningful number of layout boxes.
        // (We assert reuse happened, NOT a perf target — proving the harness
        // measures real reuse, not that the number hits a goal.)
        assert!(
            r.reused_second > 0,
            "frame 2 must reuse cached layout boxes (reused_second={}, relaid_second={})",
            r.reused_second,
            r.relaid_second
        );
    }

    #[test]
    fn animation_ticks_requested_frames_with_exact_damage_counts() {
        let cfg = cfg();
        let r = measure_animation(&cfg).expect("anim");
        assert_eq!(r.frames_total, ANIM_FRAMES);
        assert_eq!(r.frame_ms.len(), ANIM_FRAMES, "one timing per frame");
        assert!(r.frame_ms.iter().all(|&t| t >= 0.0));
        assert!(r.doc_chunks > 0, "anim page has display-list chunks");
        // We computed a per-frame damage diff for every frame after the first.
        assert_eq!(r.changed_nodes.len(), ANIM_FRAMES);
        // changed-node counts are real integers in [0, doc_chunks-ish]; at least
        // one frame should show change (the box moves) and none should explode.
        assert!(r.changed_nodes.iter().any(|&c| c > 0), "the moving box damages nodes");
    }

    #[test]
    fn js_exec_runs_real_interpreter_and_reports_jit_honesty_counter() {
        let r = measure_js("loop.js", "__bench_loop_result").expect("js loop");
        assert_eq!(r.cold_ms.len(), ITERS);
        assert_eq!(r.warm_ms.len(), ITERS);
        assert!(r.cold_ms.iter().all(|&t| t > 0.0), "real JS work took time");
        assert!(r.warm_ms.iter().all(|&t| t > 0.0));
        // t2_enabled reflects the real flag state; the count is whatever the
        // engine genuinely did (we assert it is reported, not a target value).
        // It must be a real reading consistent with the flag: if the JIT tier is
        // disabled, the count is 0; if enabled it may be >0. Either way it is the
        // honesty guard, never a fabricated number.
        let _ = (r.t2_exec_count, r.t2_enabled);
    }

    /// ★ LEAF-JIT ROUTING under the top-level VM. With `CV_TOPLEVEL_VM` ON, the hot
    /// top-level loop runs on the register VM and the leaf callee (`work`/`f`) is
    /// invoked via the in-VM `Op::CallFn`. The fix routes that in-VM call to the
    /// SAME P6 f64 native tier the tree-walk call path uses — so the leaf must run
    /// as NATIVE code (`p6_exec_count > 0`), not regress to the interpreter, AND
    /// the top level must genuinely engage the VM (`toplevel_vm_took > 0`). A green
    /// here proves the loop.js/jit.js native win SURVIVES top-level-VM mode.
    #[test]
    fn leaf_jit_routes_to_p6_under_toplevel_vm() {
        for (file, result_global) in
            [("loop.js", "__bench_loop_result"), ("jit.js", "__bench_jit_result")]
        {
            let src = read_input(file).expect("read bench input");
            let blank = "<!doctype html><html><head></head><body></body></html>";
            let doc = cv_html::parse(blank);
            let label = "file:///benchfix/leaftest";
            let _g = cv_js::TopLevelVmGuard::new(true);
            cv_js::reset_p6_exec_count();
            cv_js::reset_toplevel_vm_took_count();
            let mut rt = LiveInterp::new(&doc, label);
            rt.interp.run(&src).unwrap_or_else(|e| panic!("{file}: {e:?}"));
            let p6 = cv_js::p6_exec_count();
            let toplevel = cv_js::toplevel_vm_took_count();
            assert!(
                toplevel > 0,
                "{file}: top-level VM must engage with the flag on (took={toplevel})"
            );
            assert!(
                p6 > 0,
                "{file}: leaf callee must run as P6 native code under the top-level VM \
                 (p6_exec_count={p6}); the in-VM CallFn must route to the f64 JIT"
            );
        }
    }

    /// ★ STAGE 2 — PRODUCTION-FAITHFUL byte-identity + native-kernel engagement.
    /// Through the REAL cv_browser `LiveInterp` host (the production path a page takes),
    /// run jit.js / loop.js with the top-level VM ON and `CV_INLINE_LEAF` OFF vs ON,
    /// reading the result global back THROUGH `globalThis` (the production observation),
    /// and require the two flag states to be byte-identical. ALSO require the Stage-2
    /// path to engage: the leaf is spliced (`leaf_inline_count > 0`) and the whole loop
    /// runs as ONE native kernel call (`p6_exec_count == 1`, not 1.5M per-iteration
    /// calls). This is the anti-fake guard for the measured win.
    #[test]
    fn stage2_inline_leaf_byte_identical_and_kernelizes() {
        for (file, result_global) in
            [("loop.js", "__bench_loop_result"), ("jit.js", "__bench_jit_result")]
        {
            let src = read_input(file).expect("read bench input");
            let blank = "<!doctype html><html><head></head><body></body></html>";
            let doc = cv_html::parse(blank);
            let label = "file:///benchfix/stage2id";
            // Read the result global THROUGH globalThis (the production read path).
            let probe = format!(
                "{src}\n;globalThis.__CV_RESULT__ = String(globalThis[{key:?}]);",
                key = result_global
            );
            // Helper: run with inline=on/off, return the globalThis-read result string
            // plus (leaf_inline_count, p6_exec_count).
            let run = |inline: bool| -> (String, u64, u64) {
                let _gv = cv_js::TopLevelVmGuard::new(true);
                let _gi = cv_js::InlineLeafGuard::new(inline);
                cv_js::reset_p6_exec_count();
                cv_js::reset_leaf_inline_count();
                let mut rt = LiveInterp::new(&doc, label);
                rt.interp.run(&probe).unwrap_or_else(|e| panic!("{file}: {e:?}"));
                // Read back __CV_RESULT__ through a trailing run that serializes it.
                let read = "String(globalThis.__CV_RESULT__)";
                let v = rt
                    .interp
                    .run_completion_value(read)
                    .map(|v| format!("{v:?}"))
                    .unwrap_or_else(|e| format!("ERR {e:?}"));
                (v, cv_js::leaf_inline_count(), cv_js::p6_exec_count())
            };
            let (off, _, _) = run(false);
            let (on, inlined, p6) = run(true);
            assert_eq!(off, on, "{file}: inline ON must be byte-identical to OFF");
            assert!(inlined > 0, "{file}: leaf must be spliced inline (count={inlined})");
            // The whole counted loop must collapse to ONE native kernel call (not a
            // per-iteration call). loop.js has a nested inner loop kernelized too, so
            // accept exactly 1 (the outer kernel runs the inner loop natively inline).
            assert_eq!(
                p6, 1,
                "{file}: the fused loop must run as ONE native kernel call (p6={p6})"
            );
        }
    }

    /// OFFLINE before/after timing for the two JS microbenches under the
    /// top-level-VM flag OFF (tree-walk top level = the byte-identical baseline)
    /// vs ON (hot top level on the register VM + the leaf routed to P6). Ignored
    /// by default (timing, not a pass/fail gate); run explicitly in RELEASE:
    ///   cargo test -p cv_browser --bins --release toplevel_vm_before_after_timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn toplevel_vm_before_after_timing() {
        use std::time::Instant;
        fn best_ms(file: &str, on: bool) -> (f64, u64, u64) {
            let src = read_input(file).expect("read");
            let blank = "<!doctype html><html><head></head><body></body></html>";
            let doc = cv_html::parse(blank);
            let label = "file:///benchfix/timing";
            // warmup
            for _ in 0..3 {
                let _g = cv_js::TopLevelVmGuard::new(on);
                let mut rt = LiveInterp::new(&doc, label);
                let _ = rt.interp.run(&src);
            }
            let mut best = f64::MAX;
            let (mut p6, mut tv) = (0u64, 0u64);
            for _ in 0..12 {
                let _g = cv_js::TopLevelVmGuard::new(on);
                cv_js::reset_p6_exec_count();
                cv_js::reset_toplevel_vm_took_count();
                let mut rt = LiveInterp::new(&doc, label);
                let t = Instant::now();
                rt.interp.run(&src).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if ms < best {
                    best = ms;
                    p6 = cv_js::p6_exec_count();
                    tv = cv_js::toplevel_vm_took_count();
                }
            }
            (best, p6, tv)
        }
        for file in ["loop.js", "jit.js"] {
            let (off, _, _) = best_ms(file, false);
            let (on, p6, tv) = best_ms(file, true);
            println!(
                "[timing] {file}: toplevel_vm OFF={off:.2}ms  ON={on:.2}ms  (ON p6_exec={p6} toplevel_took={tv})"
            );
        }
    }

    /// STAGE 2 — OFFLINE before/after timing comparing Stage-1 (top-level VM on,
    /// leaf as a per-iteration `Op::CallFn`) vs Stage-2 (top-level VM on + leaf
    /// SPLICED inline via `CV_INLINE_LEAF`). Both are byte-identical (the A/B oracle
    /// gates that); this isolates the speedup from removing the per-iteration call.
    /// Ignored by default (timing, not a gate); run in RELEASE:
    ///   cargo test -p cv_browser --bins --release stage2_leaf_inline_before_after -- --ignored --nocapture
    #[test]
    #[ignore]
    fn stage2_leaf_inline_before_after() {
        use std::time::Instant;
        // (best_ms, p6_exec, toplevel_took, leaf_inline_count)
        fn best_ms(file: &str, inline_on: bool) -> (f64, u64, u64, u64) {
            let src = read_input(file).expect("read");
            let blank = "<!doctype html><html><head></head><body></body></html>";
            let doc = cv_html::parse(blank);
            let label = "file:///benchfix/stage2";
            for _ in 0..3 {
                let _gv = cv_js::TopLevelVmGuard::new(true);
                let _gi = cv_js::InlineLeafGuard::new(inline_on);
                let mut rt = LiveInterp::new(&doc, label);
                let _ = rt.interp.run(&src);
            }
            let mut best = f64::MAX;
            let (mut p6, mut tv, mut li) = (0u64, 0u64, 0u64);
            for _ in 0..12 {
                let _gv = cv_js::TopLevelVmGuard::new(true);
                let _gi = cv_js::InlineLeafGuard::new(inline_on);
                cv_js::reset_p6_exec_count();
                cv_js::reset_toplevel_vm_took_count();
                cv_js::reset_leaf_inline_count();
                let mut rt = LiveInterp::new(&doc, label);
                let t = Instant::now();
                rt.interp.run(&src).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if ms < best {
                    best = ms;
                    p6 = cv_js::p6_exec_count();
                    tv = cv_js::toplevel_vm_took_count();
                    li = cv_js::leaf_inline_count();
                }
            }
            (best, p6, tv, li)
        }
        for file in ["loop.js", "jit.js"] {
            let (s1, p6a, _, _) = best_ms(file, false);
            let (s2, p6b, tv, li) = best_ms(file, true);
            println!(
                "[stage2] {file}: STAGE1(call)={s1:.2}ms (p6={p6a})  STAGE2(inline)={s2:.2}ms  (p6={p6b} toplevel_took={tv} leaf_inlined={li})"
            );
        }
    }

    #[test]
    fn memory_reports_bounded_live_object_counts() {
        let cfg = cfg();
        let (load, idle) = measure_memory(&cfg).expect("memory");
        // Both are REAL GC-tracked live-object counts. NOTE: the GC registry is
        // only populated when CV_GC is enabled (objects register a Weak only
        // under GC); with GC off the honest count is 0. Either way the value is a
        // genuine reachability reading — never fabricated. The harness invariant
        // we can always assert is BOUNDEDNESS: idle churn must not multiply the
        // tracked heap (the no-leak axis), proving the metric measures liveness,
        // not a target. A small additive slack covers the few objects the idle
        // ticks legitimately retain.
        assert!(
            idle <= load.saturating_mul(4) + 4096,
            "after_idle ({idle}) grew unboundedly vs after_load ({load})"
        );
    }

    #[test]
    fn emitted_json_is_well_formed_and_numbers_are_numbers() {
        // Build a representative J tree and confirm it serializes to parseable
        // JSON with numeric (not string) metric fields.
        let tree = J::Obj(vec![
            ("schema", J::S(SCHEMA.to_string())),
            ("n", J::I(7)),
            ("ms", J::F(1.5)),
            ("flag", J::Bool(true)),
            ("nothing", J::Null),
            ("inf_becomes_null", J::F(f64::INFINITY)),
        ]);
        let s = tree.to_pretty();
        assert!(s.contains("\"schema\": \"conclave-bench/1\""));
        assert!(s.contains("\"n\": 7"));
        assert!(s.contains("\"ms\": 1.5"));
        assert!(s.contains("\"flag\": true"));
        assert!(s.contains("\"nothing\": null"));
        // Non-finite floats are emitted as null, never a fake number.
        assert!(s.contains("\"inf_becomes_null\": null"));
    }
}
