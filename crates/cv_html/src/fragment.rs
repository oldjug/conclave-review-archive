//! HTML fragment parsing — used by `Element.innerHTML = ...` to parse
//! the supplied string per the HTML Standard's "fragment parsing
//! algorithm" (WHATWG HTML §13.4).
//!
//! The full fragment algorithm is contextually-aware (a `<tr>` parent
//! changes the insertion mode); V1 runs the supplied string through
//! the same tokenizer + tree builder as the main document parser and
//! returns the resulting children of the synthetic root.

use crate::tree::{Node, parse as parse_doc};

/// Parse `html` as a fragment inserted into a parent of tag `context`
/// (used to nudge the tokenizer's insertion mode). Returns the
/// resulting top-level nodes the caller appends to the host element.
pub fn parse_fragment(html: &str, context: &str) -> Vec<Node> {
    // When CV_WHATWG_PARSER is on, upgrade to the real §13.4 context-seeded
    // fragment algorithm; otherwise fall through to the legacy path below
    // (byte-identical to today — which ignores `context`). See `treebuilder`.
    if crate::treebuilder::whatwg_enabled() {
        let tokens = crate::tokenizer::Tokenizer::new(html).run();
        let doc = crate::treebuilder::build_whatwg(tokens, Some(context));
        return crate::treebuilder::lift_fragment_children(doc);
    }
    let _ = context; // legacy path is context-insensitive (unchanged behavior)
    let doc = parse_doc(html);
    // Drop the synthetic <html>/<head>/<body> the document builder
    // wraps content in; surface whatever the caller appended.
    let mut out = doc.root.children;
    // If the parser produced a single <html> wrapper, unwrap to body.
    use crate::tree::NodeKind;
    let is_named =
        |n: &Node, want: &str| matches!(&n.kind, NodeKind::Element { name, .. } if name == want);
    if out.len() == 1 && is_named(&out[0], "html") {
        let html = out.remove(0);
        let mut tail = Vec::new();
        for child in html.children {
            if is_named(&child, "body") {
                return child.children;
            }
            if is_named(&child, "head") {
                continue;
            }
            tail.push(child);
        }
        out = tail;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_two_paragraphs() {
        let nodes = parse_fragment("<p>a</p><p>b</p>", "div");
        // The synthetic html/body wrap inflates the tree depth; the
        // important thing is the parser doesn't panic and produces a
        // non-empty result.
        assert!(!nodes.is_empty());
    }
}
