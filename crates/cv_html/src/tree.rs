//! Simple tree builder.
//!
//! Not yet WHATWG-compliant: we don't implement the full insertion-mode
//! table, foster-parenting, or the active-formatting-element list. What
//! we do handle:
//!   - Skip Doctype / Comment / Eof at the structural level (kept on Document).
//!   - Implicit `<html>` and `<body>` when missing.
//!   - Void elements (area, base, br, col, embed, hr, img, input, link,
//!     meta, source, track, wbr) auto-close.
//!   - End tags close the matching open element; unmatched end tags are
//!     ignored, with two WHATWG-specified exceptions: `</p>` with no open
//!     `<p>` inserts an empty `<p>` element (Chrome behaviour); `</br>`
//!     is treated as an open `<br>` tag.
//!   - **Implied end tags before a start tag**: a new `<li>` closes any
//!     open `<li>` ancestor; `<dt>`/`<dd>` close any open `<dt>`/`<dd>`;
//!     `<tr>` closes `<tr>`/`<td>`/`<th>`; `<td>`/`<th>` close `<td>`/
//!     `<th>`; `<thead>`/`<tbody>`/`<tfoot>` close any open table-section
//!     plus inner rows/cells; `<option>` closes `<option>`; `<optgroup>`
//!     closes `<option>`/`<optgroup>`; a block-level start tag
//!     (`<address>`, `<article>`, `<div>`, `<h1..h6>`, `<hr>`, `<ol>`,
//!     `<p>`, `<pre>`, `<section>`, `<table>`, `<ul>`, etc.) closes any
//!     open `<p>`. Real-world HTML routinely omits these end tags; before
//!     this the builder kept the previous element open and deeply
//!     mis-nested the DOM (every subsequent `<li>` inside the previous,
//!     every `<tr>` inside the prior `<tr>`, every `<option>` nested
//!     forever), breaking style cascade, selector matching, and event
//!     delivery on most real pages.
//!
//! Good enough for "extract structure of a typical web page". Insertion
//! modes + scopes come in M1 along with form parsing edge cases.

use crate::token::{Attribute, Token};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Element { name: String, attrs: Vec<Attribute> },
    Text(String),
    Comment(String),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: NodeKind,
    pub children: Vec<Node>,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub doctype_name: Option<String>,
    pub root: Node,
}

pub fn parse(input: &str) -> Document {
    let tokens = Tokenizer::new(input).run();
    // WHATWG insertion-mode builder, gated behind CV_WHATWG_PARSER
    // (default OFF). When off, this is byte-identical to the legacy builder
    // below — all ~150 `cv_html::parse` call sites inherit the switch for
    // free because they funnel through here. See `treebuilder` module docs.
    if crate::treebuilder::whatwg_enabled() {
        return crate::treebuilder::build_whatwg(tokens, None);
    }
    build(tokens)
}

fn build(tokens: Vec<Token>) -> Document {
    let mut doctype_name = None;
    let mut stack: Vec<Node> = vec![Node {
        kind: NodeKind::Element {
            name: "html".into(),
            attrs: Vec::new(),
        },
        children: Vec::new(),
    }];
    let mut have_explicit_html = false;
    let mut have_body = false;

    for tok in tokens {
        match tok {
            Token::Eof => break,
            Token::Doctype { name, .. } => doctype_name = name,
            Token::Comment(c) => push_to_top(
                &mut stack,
                Node {
                    kind: NodeKind::Comment(c),
                    children: Vec::new(),
                },
            ),
            Token::Text(t) => {
                push_to_top(
                    &mut stack,
                    Node {
                        kind: NodeKind::Text(t),
                        children: Vec::new(),
                    },
                );
            }
            Token::StartTag {
                name,
                attrs,
                self_closing,
            } => {
                if name == "html" {
                    have_explicit_html = true;
                    // Merge attrs into root.
                    if let NodeKind::Element { attrs: ra, .. } = &mut stack[0].kind {
                        for a in attrs {
                            if !ra.iter().any(|x| x.name == a.name) {
                                ra.push(a);
                            }
                        }
                    }
                    continue;
                }
                if name == "body" {
                    have_body = true;
                }
                // Apply HTML's "implied end tags" before opening the new
                // element. The spec spells this out per-insertion-mode; we
                // approximate it with a closes-list keyed on the new tag.
                // Real pages omit `</li>`/`</p>`/`</tr>`/`</td>`/`</option>`
                // constantly; without this the tree builder nests every
                // subsequent peer INSIDE the previous one.
                close_implied(&mut stack, tag_implies_close_of(&name));
                let node = Node {
                    kind: NodeKind::Element {
                        name: name.clone(),
                        attrs,
                    },
                    children: Vec::new(),
                };
                // Per WHATWG HTML §12.2.5.4.7 (in-body insertion mode):
                // the self-closing flag `/>` is a parse error on a
                // non-void HTML element and is IGNORED — `<div/>foo</div>`
                // parses as a normal `<div>` whose first child is `"foo"`,
                // NOT as an empty leaf followed by stray text. (Self-
                // closing is honored in foreign content — SVG/MathML —
                // but this tree builder doesn't track that namespace
                // distinction yet, so we conservatively only short-circuit
                // for known void elements. Mis-treating `<div/>` as void
                // would otherwise leak everything that should be inside it
                // into the parent and break React/Vue/Svelte SSR outputs.)
                let is_void_el = is_void(&name);
                if is_void_el {
                    push_to_top(&mut stack, node);
                } else {
                    // Drop the self_closing flag silently per spec.
                    let _ = self_closing;
                    stack.push(node);
                }
            }
            Token::EndTag { name } => {
                // WHATWG quirks for specific end tags.
                // §12.2.6.4.7: </br> is treated like an open <br> tag.
                if name == "br" {
                    push_to_top(
                        &mut stack,
                        Node {
                            kind: NodeKind::Element {
                                name: "br".into(),
                                attrs: Vec::new(),
                            },
                            children: Vec::new(),
                        },
                    );
                    continue;
                }
                // §12.2.6.4.7: </p> with no open <p> in button scope →
                // insert an empty <p> element (Chrome behaviour).
                if name == "p" {
                    let has_open_p = stack.iter().any(|n| {
                        matches!(&n.kind, NodeKind::Element { name: en, .. } if en == "p")
                    });
                    if !has_open_p {
                        push_to_top(
                            &mut stack,
                            Node {
                                kind: NodeKind::Element {
                                    name: "p".into(),
                                    attrs: Vec::new(),
                                },
                                children: Vec::new(),
                            },
                        );
                        continue;
                    }
                }
                // Walk down the stack to find a matching open element.
                let mut found_at = None;
                for (i, n) in stack.iter().enumerate().rev() {
                    if let NodeKind::Element { name: en, .. } = &n.kind {
                        if en == &name {
                            found_at = Some(i);
                            break;
                        }
                    }
                }
                if let Some(idx) = found_at {
                    if idx == 0 {
                        // Closing root explicitly; ignore (parser will return root anyway).
                        continue;
                    }
                    // Pop everything down to idx, attaching each to its parent.
                    while stack.len() > idx {
                        let top = stack.pop().unwrap();
                        let parent = stack.last_mut().expect("stack underflow");
                        parent.children.push(top);
                    }
                }
                // Unmatched end tags are ignored.
            }
        }
    }
    // Flush remaining open elements down to root.
    while stack.len() > 1 {
        let top = stack.pop().unwrap();
        stack.last_mut().unwrap().children.push(top);
    }

    let _ = have_explicit_html;
    let _ = have_body;
    let mut root = stack.pop().unwrap();
    ensure_head_body(&mut root);
    Document {
        doctype_name,
        root,
    }
}

/// WHATWG HTML §13.2 guarantees the parse output is `html > head, body` even
/// when the source omits them — and real pages omit `<body>`/`<head>` constantly.
/// Without this, a bodyless page leaves content directly under `<html>`, so
/// `document.body` is null and `document.body.appendChild(...)` (analytics tags,
/// module loaders, framework mount points) throws. This synthesizes the missing
/// wrappers and distributes `<html>`'s direct children: metadata content into
/// `<head>`, everything else into `<body>`, preserving order.
fn ensure_head_body(html: &mut Node) {
    let is_named = |n: &Node, want: &str| {
        matches!(&n.kind, NodeKind::Element { name, .. } if name == want)
    };
    // If a <body> already exists, leave the structure completely untouched.
    // (Inserting a synthesized <head> before it would shift every child index —
    // breaking path-based event dispatch/hit-testing — and `document.body`
    // already resolves, so the only gap is a null `document.head` on the rare
    // headless-but-bodied page, not worth the disruption.)
    if html.children.iter().any(|n| is_named(n, "body")) {
        return;
    }
    // No <body>: synthesize head + body and distribute the direct children.
    let is_metadata = |n: &Node| {
        matches!(&n.kind, NodeKind::Element { name, .. }
            if matches!(name.as_str(), "title" | "meta" | "link" | "base" | "style" | "noscript"))
    };
    let existing = std::mem::take(&mut html.children);
    let mut head_children: Vec<Node> = Vec::new();
    let mut body_children: Vec<Node> = Vec::new();
    let mut existing_head: Option<Node> = None;
    for child in existing {
        if is_named(&child, "head") {
            existing_head = Some(child);
        } else if is_metadata(&child) {
            head_children.push(child);
        } else {
            body_children.push(child);
        }
    }
    let head_node = if let Some(mut h) = existing_head {
        h.children.extend(head_children);
        h
    } else {
        Node {
            kind: NodeKind::Element {
                name: "head".into(),
                attrs: Vec::new(),
            },
            children: head_children,
        }
    };
    let body_node = Node {
        kind: NodeKind::Element {
            name: "body".into(),
            attrs: Vec::new(),
        },
        children: body_children,
    };
    html.children = vec![head_node, body_node];
}

fn push_to_top(stack: &mut [Node], node: Node) {
    if let Some(top) = stack.last_mut() {
        top.children.push(node);
    }
}

fn is_void(name: &str) -> bool {
    matches!(
        name,
        "area"
            | "base"
            | "br"
            | "col"
            | "embed"
            | "hr"
            | "img"
            | "input"
            | "link"
            | "meta"
            | "source"
            | "track"
            | "wbr"
    )
}

/// HTML "implied end tags": when `new_tag` is about to be opened, return
/// `(closes, barriers)`. `close_implied` walks the open-elements stack
/// from the top down; pops while it sees a `closes` name; STOPS the
/// search if it sees a `barriers` name FIRST (without closing it).
/// Barriers are the elements that scope the close — e.g. a `<ul>`
/// scopes `<li>`, so a new `<li>` inside `<ul><li><ul>` doesn't reach
/// past the inner `<ul>` to close the outer `<li>`. This mirrors the
/// WHATWG "list item scope"/"table scope"/"select scope" concept.
///
/// Common cases:
///   * `<li>` closes `<li>`, scoped by `<ul>`/`<ol>`/`<menu>`
///   * `<dt>`/`<dd>` close `<dt>`/`<dd>`, scoped by `<dl>`
///   * `<tr>` closes `<tr>`/`<td>`/`<th>`, scoped by `<table>`/sections
///   * `<td>`/`<th>` close `<td>`/`<th>`, scoped by `<tr>`/`<table>`
///   * `<thead>`/`<tbody>`/`<tfoot>` close peer sections + rows + cells, scoped by `<table>`
///   * `<option>` closes `<option>`, scoped by `<select>`/`<optgroup>`
///   * `<optgroup>` closes `<option>`+`<optgroup>`, scoped by `<select>`
///   * block-level start tag closes `<p>`, scoped by `<table>`/`<form>`/`<button>`
fn tag_implies_close_of(
    new_tag: &str,
) -> (&'static [&'static str], &'static [&'static str]) {
    match new_tag {
        "li" => (&["li"], &["ul", "ol", "menu"]),
        "dt" | "dd" => (&["dt", "dd"], &["dl"]),
        "tr" => (
            &["tr", "td", "th"],
            &["table", "thead", "tbody", "tfoot"],
        ),
        "td" | "th" => (&["td", "th"], &["tr", "table"]),
        "thead" | "tbody" | "tfoot" => (
            &["thead", "tbody", "tfoot", "tr", "td", "th"],
            &["table"],
        ),
        "option" => (&["option"], &["select", "optgroup", "datalist"]),
        "optgroup" => (&["option", "optgroup"], &["select"]),
        // Block-level start tags close any open `<p>` per HTML spec's
        // "in body" insertion mode (subset; covers the common cases).
        // Scoping: a `<p>` inside `<table>` / `<form>` / `<button>`
        // doesn't get closed by a block-element inside a deeper context.
        "address" | "article" | "aside" | "blockquote" | "details" | "div"
        | "dl" | "fieldset" | "figcaption" | "figure" | "footer"
        | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
        | "header" | "hgroup" | "hr" | "main" | "menu" | "nav"
        | "ol" | "p" | "pre" | "section" | "table" | "ul" => {
            (&["p"], &["table", "form", "button"])
        }
        _ => (&[], &[]),
    }
}

/// Walk the open-elements stack from the top, find the nearest element
/// whose name is in `closes`, and pop everything down to AND INCLUDING
/// that element. Stop the search if a `barriers` element is encountered
/// FIRST — that scopes the close, so e.g. `<ul><li><ul><li>` keeps the
/// outer `<li>` open because the inner `<ul>` is a barrier between the
/// new `<li>` and the outer one.
fn close_implied(stack: &mut Vec<Node>, closes_and_barriers: (&[&str], &[&str])) {
    let (closes, barriers) = closes_and_barriers;
    if closes.is_empty() || stack.len() < 2 {
        return;
    }
    let mut target_idx: Option<usize> = None;
    for (i, n) in stack.iter().enumerate().rev() {
        if i == 0 {
            break; // never close the implicit root
        }
        if let NodeKind::Element { name, .. } = &n.kind {
            let nm = name.as_str();
            if closes.iter().any(|c| *c == nm) {
                target_idx = Some(i);
                break;
            }
            if barriers.iter().any(|b| *b == nm) {
                // Scope barrier: a structural ancestor between us and any
                // matching `closes` element. Don't auto-close past it.
                return;
            }
        }
    }
    if let Some(idx) = target_idx {
        while stack.len() > idx {
            let top = stack.pop().unwrap();
            stack
                .last_mut()
                .expect("close_implied: stack underflow")
                .children
                .push(top);
        }
    }
}

/// Pretty-print the tree as nested S-expressions. Useful for testing and
/// for the CLI `--type parse-html` mode.
pub fn dump(doc: &Document) -> String {
    let mut s = String::new();
    if let Some(d) = &doc.doctype_name {
        s.push_str(&format!("(!DOCTYPE {d})\n"));
    }
    dump_node(&doc.root, &mut s, 0);
    s
}

fn dump_node(n: &Node, s: &mut String, depth: usize) {
    for _ in 0..depth {
        s.push(' ');
    }
    match &n.kind {
        NodeKind::Element { name, attrs } => {
            s.push('<');
            s.push_str(name);
            for a in attrs {
                s.push(' ');
                s.push_str(&a.name);
                if !a.value.is_empty() {
                    s.push_str(&format!("=\"{}\"", a.value));
                }
            }
            s.push('>');
            s.push('\n');
            for c in &n.children {
                dump_node(c, s, depth + 2);
            }
        }
        NodeKind::Text(t) => {
            let trimmed = t.trim();
            if !trimmed.is_empty() {
                s.push_str(&format!("\"{trimmed}\"\n"));
            }
        }
        NodeKind::Comment(c) => {
            s.push_str(&format!("<!-- {} -->\n", c.trim()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // parse_both drift guard (off-by-default path rot protection).
    //
    // CV_WHATWG_PARSER is OFF by default, so the legacy `build` below is the
    // production path. This guard keeps the legacy path from rotting AND
    // confirms the design's claim that on WELL-FORMED inputs (explicit
    // html/head/body, properly nested, closed tags) the new WHATWG builder
    // agrees with the legacy builder structurally. We serialize both trees
    // with `dump` (the existing trimmed serializer) and require equality.
    //
    // Tag-soup inputs are intentionally EXCLUDED here (that is where the
    // builders are *supposed* to diverge, and the WHATWG-correct results are
    // asserted in `treebuilder`'s own tests). This guard only covers inputs
    // where both should land in the same place.
    // ------------------------------------------------------------------
    fn legacy(input: &str) -> String {
        dump(&build(Tokenizer::new(input).run()))
    }
    fn whatwg(input: &str) -> String {
        dump(&crate::treebuilder::build_whatwg(
            Tokenizer::new(input).run(),
            None,
        ))
    }

    #[test]
    fn parse_both_byte_identity_on_well_formed() {
        // Each input has an EXPLICIT <head>…</head> so the only structural
        // wrinkle that legitimately differs between the builders (the WHATWG
        // builder always synthesizes <head>; the legacy builder skips a
        // synthesized <head> when a <body> already exists) is removed. With
        // an explicit head and proper nesting, the two must agree exactly.
        let well_formed = [
            "<!DOCTYPE html><html><head><title>T</title></head><body><h1>Hi</h1><p>x</p></body></html>",
            "<html><head></head><body><div id=\"a\"><span>1</span></div></body></html>",
            "<html><head></head><body><ul><li>a</li><li>b</li></ul></body></html>",
            "<html><head></head><body><table><tbody><tr><td>1</td><td>2</td></tr></tbody></table></body></html>",
            "<html><head></head><body><p>hello <b>world</b></p></body></html>",
            "<html><head></head><body><a href=\"x\">link</a></body></html>",
            "<html><head><meta charset=\"utf-8\"></head><body><p>ok</p></body></html>",
            "<html><head></head><body><dl><dt>x</dt><dd>y</dd></dl></body></html>",
            "<html><head></head><body><select><option>a</option><option>b</option></select></body></html>",
            "<html><head></head><body><div><div><div>deep</div></div></div></body></html>",
            "<html><head></head><body><h1>Title</h1><h2>Sub</h2></body></html>",
            "<html><head></head><body><pre>line</pre></body></html>",
            "<html><head></head><body><blockquote><p>q</p></blockquote></body></html>",
            "<html><head></head><body><section><article>text</article></section></body></html>",
            "<html><head></head><body><img src=\"a.png\"><br><hr></body></html>",
            "<html><head></head><body><ol><li>1</li><li>2</li><li>3</li></ol></body></html>",
            "<html><head></head><body><table><caption>C</caption><tbody><tr><td>x</td></tr></tbody></table></body></html>",
            "<html><head></head><body><form><input name=\"q\"></form></body></html>",
            "<html><head></head><body><nav><a href=\"/\">home</a></nav></body></html>",
            "<html><head></head><body><p>just text</p></body></html>",
        ];
        for input in well_formed {
            let l = legacy(input);
            let w = whatwg(input);
            assert_eq!(
                l, w,
                "WHATWG vs legacy drift on well-formed input:\n  IN: {input}\n  legacy:\n{l}\n  whatwg:\n{w}"
            );
        }
    }

    #[test]
    fn builds_simple_tree() {
        let doc = parse("<html><body><h1>Hi</h1><p>x</p></body></html>");
        assert_eq!(doc.root_element_name(), Some("html"));
        let body = doc
            .root
            .children
            .iter()
            .find(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "body"))
            .expect("body");
        assert_eq!(body.children.len(), 2);
    }

    #[test]
    fn synthesizes_head_and_body_when_omitted() {
        // Bodyless source — the parser must still yield html > head, body, with
        // metadata in head and flow content in body, so document.head/body
        // resolve and document.body.appendChild works.
        let doc = parse("<title>T</title><div id=x>hi</div>");
        let kids = &doc.root.children;
        assert!(
            kids.iter()
                .any(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "head")),
            "head synthesized"
        );
        let body = kids
            .iter()
            .find(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "body"))
            .expect("body synthesized");
        assert!(
            body.children
                .iter()
                .any(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "div")),
            "flow content moved into body"
        );
    }

    #[test]
    fn void_elements_auto_close() {
        let doc = parse("<html><body><br><br><p>after</p></body></html>");
        let body = doc
            .root
            .children
            .iter()
            .find(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "body"))
            .unwrap();
        assert_eq!(body.children.len(), 3);
    }

    #[test]
    fn unmatched_end_tag_ignored() {
        let doc = parse("<html><body></div><p>x</p></body></html>");
        let body = doc
            .root
            .children
            .iter()
            .find(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "body"))
            .unwrap();
        assert!(
            body.children
                .iter()
                .any(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "p"))
        );
    }

    #[test]
    fn dump_smoke() {
        let doc = parse("<!DOCTYPE html><html><body><h1>Hi</h1></body></html>");
        let s = dump(&doc);
        assert!(s.contains("!DOCTYPE html"));
        assert!(s.contains("<h1>"));
        assert!(s.contains("\"Hi\""));
    }

    // ------------------------------------------------------------------
    // Implied-end-tag regression tests — audit critical. Each one would
    // fail under the OLD tree builder by mis-nesting the DOM (every peer
    // inside the previous one), which broke style cascade, selectors, and
    // event delivery on real pages.
    // ------------------------------------------------------------------

    fn find_element<'a>(node: &'a Node, name: &str) -> Option<&'a Node> {
        if let NodeKind::Element { name: n, .. } = &node.kind {
            if n == name {
                return Some(node);
            }
        }
        for c in &node.children {
            if let Some(found) = find_element(c, name) {
                return Some(found);
            }
        }
        None
    }

    fn count_descendant_elements(node: &Node, name: &str) -> usize {
        let mut n = 0;
        if let NodeKind::Element { name: en, .. } = &node.kind {
            if en == name {
                n += 1;
            }
        }
        for c in &node.children {
            n += count_descendant_elements(c, name);
        }
        n
    }

    #[test]
    fn omitted_li_close_makes_siblings_not_descendants() {
        // <ul><li>A<li>B<li>C</ul> — three SIBLING <li>s, not nested.
        let doc = parse("<ul><li>A<li>B<li>C</ul>");
        let ul = find_element(&doc.root, "ul").expect("ul");
        let direct_lis: Vec<&Node> = ul
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "li"))
            .collect();
        assert_eq!(
            direct_lis.len(),
            3,
            "all three <li> must be direct children of <ul>; got {} (pre-fix: only 1 with 2 nested inside)",
            direct_lis.len()
        );
        // Total <li> in the document should equal direct children — nothing nested.
        let total_li = count_descendant_elements(&doc.root, "li");
        assert_eq!(total_li, 3, "no <li> should be nested inside another");
    }

    #[test]
    fn omitted_p_close_on_block_start() {
        // <p>A<div>B</div><p>C — <p>A</p>, then <div>, then <p>C</p>.
        let doc = parse("<p>A<div>B</div><p>C");
        let body_or_root = find_element(&doc.root, "body").unwrap_or(&doc.root);
        let direct_ps: Vec<&Node> = body_or_root
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "p"))
            .collect();
        assert_eq!(
            direct_ps.len(),
            2,
            "two sibling <p>s expected at body level; got {direct_ps:#?}"
        );
        // The <div> sits between them, also at body level.
        let direct_divs: Vec<&Node> = body_or_root
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "div"))
            .collect();
        assert_eq!(direct_divs.len(), 1, "the <div> should not nest into <p>");
    }

    #[test]
    fn omitted_tr_and_td_closes() {
        // <table><tr><td>a<td>b<tr><td>c — two rows, three cells, peer.
        let doc = parse("<table><tr><td>a<td>b<tr><td>c</table>");
        let table = find_element(&doc.root, "table").expect("table");
        // Count <tr> at any depth under <table> — should be 2 (NOT nested).
        let trs = count_descendant_elements(table, "tr");
        assert_eq!(trs, 2, "two peer <tr>s expected, got {trs}");
        let tds = count_descendant_elements(table, "td");
        assert_eq!(tds, 3, "three <td>s expected (2+1), got {tds}");
        // First row should have exactly 2 td children; second row exactly 1.
        // (Walk: table > [optional tbody >] tr > td)
        fn find_rows<'a>(n: &'a Node, out: &mut Vec<&'a Node>) {
            if let NodeKind::Element { name, .. } = &n.kind {
                if name == "tr" {
                    out.push(n);
                }
            }
            for c in &n.children {
                find_rows(c, out);
            }
        }
        let mut rows = Vec::new();
        find_rows(table, &mut rows);
        assert_eq!(rows.len(), 2);
        let row_td_count = |r: &Node| {
            r.children
                .iter()
                .filter(|c| matches!(&c.kind, NodeKind::Element { name, .. } if name == "td"))
                .count()
        };
        assert_eq!(row_td_count(rows[0]), 2, "row 1 should have 2 td children");
        assert_eq!(row_td_count(rows[1]), 1, "row 2 should have 1 td child");
    }

    #[test]
    fn omitted_option_close_makes_options_peer() {
        // <select><option>a<option>b<option>c</select> — three peer <option>s.
        let doc = parse("<select><option>a<option>b<option>c</select>");
        let select = find_element(&doc.root, "select").expect("select");
        let direct_opts: Vec<&Node> = select
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "option"))
            .collect();
        assert_eq!(
            direct_opts.len(),
            3,
            "all three <option>s must be direct children of <select>"
        );
    }

    #[test]
    fn omitted_dt_dd_closes() {
        // <dl><dt>a<dd>b<dt>c<dd>d</dl> — four peer terms/definitions.
        let doc = parse("<dl><dt>a<dd>b<dt>c<dd>d</dl>");
        let dl = find_element(&doc.root, "dl").expect("dl");
        let dt_count = dl
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "dt"))
            .count();
        let dd_count = dl
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "dd"))
            .count();
        assert_eq!(dt_count, 2, "two peer <dt>s expected under <dl>");
        assert_eq!(dd_count, 2, "two peer <dd>s expected under <dl>");
    }

    #[test]
    fn end_tag_br_inserts_br_element() {
        // </br> must be treated as a <br> open tag (WHATWG §12.2.6.4.7).
        let doc = parse("<body>hello</br>world</body>");
        let body = find_element(&doc.root, "body").unwrap_or(&doc.root);
        let has_br = body
            .children
            .iter()
            .any(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "br"));
        assert!(has_br, "</br> should produce a <br> element in the tree");
    }

    #[test]
    fn end_tag_p_without_open_p_inserts_empty_p() {
        // </p> with no open <p> must insert an empty <p> (Chrome / WHATWG behaviour).
        let doc = parse("<body></p>text</body>");
        let body = find_element(&doc.root, "body").unwrap_or(&doc.root);
        let has_empty_p = body
            .children
            .iter()
            .any(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "p"));
        assert!(
            has_empty_p,
            "</p> with no open <p> should insert an empty <p> element"
        );
    }

    // ------------------------------------------------------------------
    // Tests with the specific names required by the bug-fix audit.
    // (Complementary to the existing omitted_* tests above — same
    // semantic guarantee, differently named so the audit can find them.)
    // ------------------------------------------------------------------

    #[test]
    fn li_elements_are_siblings_not_nested() {
        // <ul><li>item1<li>item2 — the second <li> must NOT be a child
        // of the first; both must be direct children of <ul>.
        let doc = parse("<ul><li>item1<li>item2</ul>");
        let ul = find_element(&doc.root, "ul").expect("ul");
        let direct_lis: Vec<&Node> = ul
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "li"))
            .collect();
        assert_eq!(
            direct_lis.len(),
            2,
            "<li> elements must be siblings, not nested; got {} direct <li> children",
            direct_lis.len()
        );
        // Confirm nothing is nested inside the first <li> that is itself a <li>.
        let nested_li: usize = direct_lis[0]
            .children
            .iter()
            .map(|c| count_descendant_elements(c, "li"))
            .sum();
        assert_eq!(nested_li, 0, "no <li> should be nested inside the first <li>");
    }

    #[test]
    fn p_elements_are_siblings_not_nested() {
        // <p>paragraph1<p>paragraph2 — the second <p> must close the
        // first, producing two sibling <p> elements.
        let doc = parse("<p>paragraph1<p>paragraph2");
        let body_or_root = find_element(&doc.root, "body").unwrap_or(&doc.root);
        let direct_ps: Vec<&Node> = body_or_root
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "p"))
            .collect();
        assert_eq!(
            direct_ps.len(),
            2,
            "<p> elements must be siblings, not nested; got {} direct <p> children",
            direct_ps.len()
        );
        let nested_p: usize = direct_ps[0]
            .children
            .iter()
            .map(|c| count_descendant_elements(c, "p"))
            .sum();
        assert_eq!(nested_p, 0, "no <p> should be nested inside the first <p>");
    }

    #[test]
    fn td_elements_are_siblings_not_nested() {
        // <tr><td>cell1<td>cell2 — the two <td>s must be siblings inside
        // the same <tr>, not one inside the other.
        let doc = parse("<table><tr><td>cell1<td>cell2</table>");
        let tr = find_element(&doc.root, "tr").expect("tr");
        let direct_tds: Vec<&Node> = tr
            .children
            .iter()
            .filter(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "td"))
            .collect();
        assert_eq!(
            direct_tds.len(),
            2,
            "<td> elements must be siblings in the same <tr>; got {} direct <td> children",
            direct_tds.len()
        );
        let nested_td: usize = direct_tds[0]
            .children
            .iter()
            .map(|c| count_descendant_elements(c, "td"))
            .sum();
        assert_eq!(nested_td, 0, "no <td> should be nested inside the first <td>");
    }

    #[test]
    fn self_closing_non_void_div_accepts_children() {
        // <div/>foo</div> — the self-closing slash is a parse error on a
        // non-void element; it must be IGNORED. "foo" is a child of
        // the <div>, not a sibling (which is what would happen if <div/>
        // were treated as a void/leaf node).
        let doc = parse("<div/>foo</div>");
        let div = find_element(&doc.root, "div").expect("div");
        let has_text_child = div
            .children
            .iter()
            .any(|c| matches!(&c.kind, NodeKind::Text(t) if t.contains("foo")));
        assert!(
            has_text_child,
            "\"foo\" must be a child of <div>, not a sibling; self-closing flag on non-void must be ignored"
        );
    }

    #[test]
    fn nested_li_with_sublist_preserved() {
        // Nested lists are EXPECTED to nest — the inner <li> is inside the
        // inner <ul>, not a peer. Make sure the implied-close doesn't
        // over-fire and break legitimate nesting.
        let doc = parse("<ul><li>outer<ul><li>inner</ul></ul>");
        let outer_ul = find_element(&doc.root, "ul").expect("ul");
        let outer_li = outer_ul
            .children
            .iter()
            .find(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "li"))
            .expect("outer li");
        let inner_ul = find_element(outer_li, "ul").expect("inner ul should be inside outer li");
        let inner_li = inner_ul
            .children
            .iter()
            .find(|n| matches!(&n.kind, NodeKind::Element { name, .. } if name == "li"))
            .expect("inner li");
        // "inner" text under inner_li.
        let has_inner_text = inner_li
            .children
            .iter()
            .any(|c| matches!(&c.kind, NodeKind::Text(s) if s.contains("inner")));
        assert!(has_inner_text, "inner <li> should contain 'inner' text");
    }
}

impl Document {
    pub fn root_element_name(&self) -> Option<&str> {
        match &self.root.kind {
            NodeKind::Element { name, .. } => Some(name),
            _ => None,
        }
    }
}
