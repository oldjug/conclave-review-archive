//! No-stubs guard — fails `cargo test` when stub/fakery signatures appear in
//! the workspace source. This is the MECHANICAL backstop for the project's
//! standing rule "no stubs, ever — implement for real or leave it honestly
//! `undefined`; a fake value is worse than absence." It does not rely on
//! anyone remembering to be careful: a new stub fails the build.
//!
//! Two tiers, because not every match is a real stub:
//!
//!   TIER 1 — HARD FAIL (zero tolerance). `todo!()`, `unimplemented!()`, and
//!     `panic!("… not implemented …")` are literal "not done" markers. The
//!     baseline is 0; ANY occurrence in shipped (non-test) code fails. These
//!     are unambiguous — there is no legitimate use in production paths.
//!
//!   TIER 2 — RATCHET. Softer signatures (suspicious comment phrases, native
//!     functions whose whole body returns Undefined/a constant) have many
//!     LEGITIMATE instances today (an event-listener no-op genuinely returns
//!     undefined; a comment that says "NOT a stub" is fine; an assembler
//!     "placeholder" jump target is a real term). Grepping them naively would
//!     drown in false positives. So instead we COUNT them and compare to a
//!     committed baseline (`stub_baseline.txt`). The test fails if the count
//!     GROWS — forcing any NEW stub to be either made real or, if it's a false
//!     positive, the baseline is updated WITH A JUSTIFICATION in the commit.
//!     The campaign is actively driving these counts DOWN; the ratchet makes
//!     sure they never go back up.
//!
//! To update the baseline (only when you've legitimately changed the count):
//!   run with NO_STUBS_GUARD_WRITE_BASELINE=1 to rewrite stub_baseline.txt,
//!   then review the diff and commit it with a note on WHY it changed.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Crates/dirs to scan: everything under `crates/*/src`. Tools are excluded
/// (this guard lives in tools and shouldn't scan itself or generators).
fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../tools/no_stubs_guard ; root is two levels up.
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Recursively collect every `.rs` file under `crates/*/src`.
fn collect_src_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let crates = root.join("crates");
    let Ok(entries) = fs::read_dir(&crates) else {
        return out;
    };
    for cr in entries.flatten() {
        let src = cr.path().join("src");
        if src.is_dir() {
            walk_rs(&src, &mut out);
        }
    }
    out.sort();
    out
}

fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_rs(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

/// A line is "test code" if it's inside an obvious test region. We approximate
/// per-line (cheap, no full parse): lines after a `#[cfg(test)]`/`mod tests`
/// marker in the same file are not reliably detectable per-line, so for TIER 1
/// we instead skip files whose path screams test, and skip lines that are
/// clearly test attributes/macros. Production `todo!()` is what we care about.
fn is_testish_path(p: &Path) -> bool {
    let s = p.to_string_lossy();
    s.contains("/tests/") || s.contains("\\tests\\")
}

/// Strip a `// …` line comment's body so TIER-1 regexes don't match a macro
/// name written inside a comment (e.g. "// we never call todo!()").
fn code_part(line: &str) -> &str {
    // naive: cut at the first `//` not inside a string. Good enough — a `//`
    // inside a string literal on a line that also contains todo!() in real
    // code is vanishingly rare, and a false negative here is safe (we'd just
    // not flag a comment, which is correct).
    if let Some(idx) = line.find("//") {
        &line[..idx]
    } else {
        line
    }
}

#[test]
fn tier1_no_todo_unimplemented_in_production() {
    let root = workspace_root();
    let files = collect_src_files(&root);
    let mut hits: Vec<String> = Vec::new();

    for f in &files {
        if is_testish_path(f) {
            continue;
        }
        let Ok(text) = fs::read_to_string(f) else {
            continue;
        };
        let mut in_test_mod = false;
        let mut brace_depth_at_test: i32 = -1;
        let mut depth: i32 = 0;
        for (lineno, raw) in text.lines().enumerate() {
            // Track a `#[cfg(test)] mod tests { … }` region crudely so we don't
            // flag todo!() inside unit-test modules.
            if raw.contains("#[cfg(test)]") {
                in_test_mod = true;
                brace_depth_at_test = depth;
            }
            let code = code_part(raw);
            depth += code.matches('{').count() as i32;
            depth -= code.matches('}').count() as i32;
            if in_test_mod && depth <= brace_depth_at_test {
                in_test_mod = false;
            }
            if in_test_mod {
                continue;
            }
            let has_todo = code.contains("todo!(") || code.contains("unimplemented!(");
            let has_panic_ni = code.contains("panic!(")
                && {
                    let lc = code.to_ascii_lowercase();
                    lc.contains("not implemented") || lc.contains("not yet implemented")
                };
            if has_todo || has_panic_ni {
                hits.push(format!(
                    "{}:{}: {}",
                    f.strip_prefix(&root).unwrap_or(f).display(),
                    lineno + 1,
                    raw.trim()
                ));
            }
        }
    }

    assert!(
        hits.is_empty(),
        "\n\nNO-STUBS GUARD (tier 1) FAILED — found {} todo!()/unimplemented!()/panic!(\"not implemented\") in production code.\n\
         These are literal 'not done' markers and the rule is: implement it for real, or leave the value honestly absent (return undefined / Option::None / a real Err), never a stub.\n\
         Offenders:\n  {}\n",
        hits.len(),
        hits.join("\n  ")
    );
}

/// TIER-2 soft signatures. Each returns the per-file count. The grand totals
/// are compared to the baseline file.
fn tier2_counts(root: &Path, files: &[PathBuf]) -> BTreeMap<&'static str, usize> {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    counts.insert("comment_stub_phrase", 0);
    counts.insert("native_fn_undefined_body", 0);

    // Comment phrases that, in a comment, strongly indicate fakery. We
    // deliberately EXCLUDE the word "stub" alone because the codebase has many
    // "NOT a stub" / "instead of a stub" comments documenting real work.
    let stub_phrases = [
        "no-op stub",
        "fake value",
        "returns fake",
        "return fake",
        "for now we just",
        "for now, just",
        "simplified to a",
        "placeholder value",
        "dummy value",
        "not a real implementation",
        "always returns false to keep",
        "to keep feature-detection happy",
    ];

    for f in files {
        if is_testish_path(f) {
            continue;
        }
        let _ = root;
        let Ok(text) = fs::read_to_string(f) else {
            continue;
        };
        let lc = text.to_ascii_lowercase();
        for phrase in &stub_phrases {
            *counts.get_mut("comment_stub_phrase").unwrap() +=
                lc.matches(phrase).count();
        }
        // native_fn(... |_| Ok(... Value::Undefined)) on a single line — the
        // classic JS-surface no-op. Counts only the trivially-empty ones.
        for line in text.lines() {
            let l = line.replace(' ', "");
            if l.contains("native_fn(")
                && l.contains("|_|Ok(")
                && (l.contains("Value::Undefined))") || l.contains("Value::Undefined)})"))
            {
                *counts.get_mut("native_fn_undefined_body").unwrap() += 1;
            }
        }
    }
    counts
}

fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stub_baseline.txt")
}

fn read_baseline() -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    if let Ok(txt) = fs::read_to_string(baseline_path()) {
        for line in txt.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                if let Ok(n) = v.trim().parse::<usize>() {
                    m.insert(k.trim().to_string(), n);
                }
            }
        }
    }
    m
}

#[test]
fn tier2_soft_signatures_do_not_grow() {
    let root = workspace_root();
    let files = collect_src_files(&root);
    let counts = tier2_counts(&root, &files);

    // Optional baseline rewrite (only for an intentional, justified change).
    if std::env::var("NO_STUBS_GUARD_WRITE_BASELINE").is_ok() {
        let mut out = String::from(
            "# no-stubs guard tier-2 baseline. The guard FAILS if any count exceeds these.\n\
             # Lower these as stubs are removed; only RAISE with a justification in the commit.\n\
             # Regenerate with NO_STUBS_GUARD_WRITE_BASELINE=1 cargo test -p no_stubs_guard.\n",
        );
        for (k, v) in &counts {
            out.push_str(&format!("{k} = {v}\n"));
        }
        fs::write(baseline_path(), out).expect("write baseline");
        eprintln!("no_stubs_guard: baseline rewritten to {:?}", baseline_path());
        return;
    }

    let baseline = read_baseline();
    let mut regressions = Vec::new();
    for (k, &now) in &counts {
        let allowed = baseline.get(*k).copied().unwrap_or(usize::MAX);
        if now > allowed {
            regressions.push(format!("  {k}: now {now}, baseline {allowed} (+{})", now - allowed));
        }
    }

    assert!(
        regressions.is_empty(),
        "\n\nNO-STUBS GUARD (tier 2) FAILED — stub signature count GREW above baseline.\n\
         A new stub/fakery was introduced. Either implement it for real, or — if this is a\n\
         false positive — update tools/no_stubs_guard/stub_baseline.txt (run with\n\
         NO_STUBS_GUARD_WRITE_BASELINE=1) AND justify why in your commit message.\n\
         Regressions:\n{}\n",
        regressions.join("\n")
    );
}
