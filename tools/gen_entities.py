#!/usr/bin/env python3
"""Generate crates/cv_html/src/entities_table.rs from the canonical
WHATWG named-character-reference table (html.spec.whatwg.org/entities.json).

The HTML Standard (§13.5 Named character references) ships entities.json as
the canonical machine-readable list. Each JSON key is the reference *with*
the leading ampersand and *with or without* the trailing semicolon (the
historical semicolon-optional forms are listed as separate keys). We emit a
Rust static slice sorted by name (name = key minus the leading '&'), values
being the decoded UTF-8 string (1 or 2 code points).

Usage:
    python tools/gen_entities.py path/to/entities.json
"""
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Canonical source data, fetched from html.spec.whatwg.org/entities.json and
# kept alongside this generator so the table is fully reproducible.
DEFAULT_JSON = os.path.join(HERE, "entities_whatwg.json")
OUT = os.path.join(REPO, "crates", "cv_html", "src", "entities_table.rs")


def rs_escape_value(s):
    # Emit every code point as \u{...} so the source file is pure ASCII and
    # there is zero ambiguity about which code points an entry maps to.
    return "".join("\\u{%X}" % ord(ch) for ch in s)


def rs_escape_name(n):
    # Names are ASCII per spec, but be defensive about quote/backslash.
    return n.replace("\\", "\\\\").replace('"', '\\"')


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    with open(src, "r", encoding="utf-8") as f:
        d = json.load(f)

    entries = []
    for k, v in d.items():
        assert k.startswith("&"), k
        name = k[1:]  # strip leading '&', keep trailing ';' if present
        entries.append((name, v["characters"]))
    entries.sort(key=lambda e: e[0])
    maxlen = max(len(n) for n, _ in entries)

    out = []
    out.append("//! WHATWG named character reference table -- the FULL canonical set.")
    out.append("//!")
    out.append("//! Generated from html.spec.whatwg.org/entities.json (the canonical")
    out.append("//! machine-readable table referenced by the HTML Standard, section 13.5).")
    out.append("//! Do not hand-edit: regenerate with `tools/gen_entities.py`.")
    out.append("//!")
    out.append("//! Each entry key is the reference name WITHOUT the leading ampersand")
    out.append("//! but WITH the trailing semicolon where the spec lists one, so the")
    out.append("//! historical semicolon-optional forms (e.g. `amp` and `amp;`) are")
    out.append("//! distinct entries. Values are the decoded UTF-8 string (1-2 code")
    out.append("//! points). The slice is sorted by key, enabling binary search and the")
    out.append("//! longest-prefix match required by the Named character reference state")
    out.append("//! (HTML Standard section 13.2.5.73).")
    out.append("")
    out.append("/// Number of entries in the table (sanity check / docs).")
    out.append("pub const ENTITY_COUNT: usize = %d;" % len(entries))
    out.append("")
    out.append("/// Length in bytes of the longest reference name (without leading `&`,")
    out.append("/// with trailing `;` where present). Bounds the longest-match scan.")
    out.append("pub const MAX_NAME_LEN: usize = %d;" % maxlen)
    out.append("")
    out.append("/// The full WHATWG named-character-reference table, sorted by name.")
    out.append("/// `(name, decoded)` where `name` omits the leading ampersand.")
    out.append("pub static ENTITIES: &[(&str, &str)] = &[")
    for n, c in entries:
        out.append('    ("%s", "%s"),' % (rs_escape_name(n), rs_escape_value(c)))
    out.append("];")
    out.append("")

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    print("wrote %s (%d entries, max name len %d)" % (OUT, len(entries), maxlen))


if __name__ == "__main__":
    main()
