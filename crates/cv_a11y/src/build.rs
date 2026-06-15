//! DOM → accessibility tree builder.
//!
//! Walks a [`cv_dom::Document`] and produces an [`AxTree`] mirroring the visible
//! DOM hierarchy, with:
//!
//!   * a computed **role** from tag semantics + an explicit ARIA `role`
//!     (ARIA-in-HTML §3 default implicit semantics; an explicit valid `role`
//!     wins);
//!   * an **accessible name** computed per the W3C accname-1.2 algorithm
//!     (§4.3.2): `aria-labelledby` > `aria-label` > native host-language label
//!     (`<label for>` / `alt` / `title`) > name-from-content > `title`;
//!   * **states**: disabled, required, checked (tri-state), expanded, focused,
//!     password; and a **value** for inputs/textareas;
//!   * **pruning** of presentational nodes (`role=presentation|none`,
//!     `<img alt="">`, non-element/structural nodes) — their accessible
//!     children are reparented to the nearest exposed ancestor, exactly like
//!     Blink's "ignored" AXObjects that are excluded from the platform tree.
//!
//! This is the Chrome `AXObjectCache` analogue (DOM-driven, layout box bounds
//! are filled in separately by the caller). The produced tree is what the UIA
//! provider ([`crate::uia`]) serves to Narrator / screen readers.

use crate::{AxNode, AxRole, AxTree, CheckedState, ExpandedState};
use cv_dom::{Document, NodeId, NodeKind};

/// Build an accessibility tree from `doc`. `focus` is the DOM node that
/// currently has keyboard focus (its AX node gets `focused = true` and the tree
/// `focus` pointer set), or `None`.
///
/// The document root becomes an `AxRole::Document` node; visible descendant
/// elements become AX nodes; text nodes contribute to names but are not their
/// own AX nodes (matching how screen readers read inline text as the parent's
/// name, not as separate objects — Blink folds anonymous text similarly).
pub fn build_ax_tree(doc: &Document, focus: Option<NodeId>) -> AxTree {
    let mut tree = AxTree::new();
    let root_dom = doc.root();
    // The document root is always exposed as the fragment root.
    let mut root = AxNode::new(AxRole::Document);
    root.name = document_title(doc).unwrap_or_default();
    root.dom_node = root_dom.to_bits();
    let root_ax = tree.add_node(None, root);
    for child in doc.children(root_dom) {
        build_subtree(doc, child, root_ax, focus, &mut tree);
    }
    tree
}

/// Build the AX subtree rooted at DOM node `dom`, attaching exposed nodes under
/// AX parent `ax_parent`. If `dom` is itself ignored/presentational, its
/// children are attached directly to `ax_parent` (reparenting).
fn build_subtree(
    doc: &Document,
    dom: NodeId,
    ax_parent: u32,
    focus: Option<NodeId>,
    tree: &mut AxTree,
) {
    let tag = match doc.kind(dom) {
        Some(NodeKind::Element { .. }) => doc.tag_raw(dom).unwrap_or("").to_ascii_lowercase(),
        // Non-element nodes (text/comment/doctype) are never their own AX node.
        // Their text is consumed by the parent's accessible-name computation.
        _ => return,
    };

    // Structural HTML elements that carry no semantics and no own AX node:
    // head and its metadata children are not part of the accessibility tree.
    if matches!(
        tag.as_str(),
        "head" | "title" | "meta" | "link" | "style" | "script" | "template" | "base"
    ) {
        return;
    }

    let role = compute_role(doc, dom, &tag);

    // Presentational / ignored nodes: don't create an AX node, but recurse so
    // their meaningful descendants still appear (reparented to ax_parent).
    if role == AxRole::Presentation {
        for child in doc.children(dom) {
            build_subtree(doc, child, ax_parent, focus, tree);
        }
        return;
    }

    let mut node = AxNode::new(role);
    node.dom_node = dom.to_bits();
    node.name = compute_accessible_name(doc, dom, role);
    node.description = compute_description(doc, dom);
    fill_states(doc, dom, &tag, role, &mut node);
    if Some(dom) == focus {
        node.focused = true;
    }

    let ax_id = tree.add_node(Some(ax_parent), node);
    for child in doc.children(dom) {
        build_subtree(doc, child, ax_id, focus, tree);
    }
}

// ---------------------------------------------------------------------------
// Role computation (ARIA-in-HTML default semantics + explicit role override).
// ---------------------------------------------------------------------------

fn compute_role(doc: &Document, dom: NodeId, tag: &str) -> AxRole {
    // An explicit, valid ARIA `role` wins over the host-language role
    // (WAI-ARIA; an unrecognized token is ignored — fall through to native).
    if let Some(r) = doc.attr_raw(dom, "role") {
        // role may be a space-separated list; the first valid token wins.
        for token in r.split_whitespace() {
            if let Some(role) = AxRole::from_aria(&token.to_ascii_lowercase()) {
                return role;
            }
        }
    }
    native_role(doc, dom, tag)
}

/// HTML element → implicit ARIA role, per the ARIA-in-HTML mapping table.
fn native_role(doc: &Document, dom: NodeId, tag: &str) -> AxRole {
    match tag {
        "a" | "area" => {
            // <a> is a link only when it has an href; otherwise generic.
            if doc.has_attribute(dom, "href") {
                AxRole::Link
            } else {
                AxRole::Generic
            }
        }
        "button" | "summary" => AxRole::Button,
        "input" => input_role(doc, dom),
        "select" => AxRole::Combobox,
        "textarea" => AxRole::Textbox,
        "img" => {
            // <img alt=""> is presentational (deliberately hidden). <img> with a
            // non-empty alt or no alt at all is an image.
            match doc.attr_raw(dom, "alt") {
                Some("") => AxRole::Presentation,
                _ => AxRole::Image,
            }
        }
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => AxRole::Heading,
        "ul" | "ol" | "menu" => AxRole::List,
        "li" => AxRole::ListItem,
        "nav" => AxRole::Navigation,
        "main" => AxRole::Main,
        "header" => AxRole::Banner,
        "footer" => AxRole::Contentinfo,
        "aside" => AxRole::Complementary,
        "article" => AxRole::Article,
        "section" => AxRole::Region,
        "form" => AxRole::Form,
        "search" => AxRole::Search,
        "p" => AxRole::Paragraph,
        "table" => AxRole::Table,
        "details" => AxRole::Group,
        "dialog" => AxRole::Group,
        "fieldset" => AxRole::Group,
        "progress" => AxRole::Spinbutton,
        // Generic containers (div/span/etc.) carry no role of their own. We
        // expose them as Generic so the hierarchy is preserved for navigation;
        // a name-less Generic with no children gets pruned by the caller if
        // desired. (Blink keeps a "GenericContainer" here.)
        _ => AxRole::Generic,
    }
}

/// `<input>` role from its `type` (ARIA-in-HTML input-type table). The default
/// type is `text`.
fn input_role(doc: &Document, dom: NodeId) -> AxRole {
    let ty = doc
        .attr_raw(dom, "type")
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "text".to_string());
    match ty.as_str() {
        "checkbox" => AxRole::Checkbox,
        "radio" => AxRole::Radio,
        "button" | "submit" | "reset" | "image" => AxRole::Button,
        "range" => AxRole::Slider,
        "number" => AxRole::Spinbutton,
        "search" => AxRole::Searchbox,
        "email" | "tel" | "url" | "text" | "" => AxRole::Textbox,
        // password / hidden / file / color / date have no mapped textbox role
        // in ARIA-in-HTML, but a password field IS exposed as a textbox by
        // every screen reader (with IsPassword). hidden inputs are not rendered.
        "password" => AxRole::Textbox,
        "hidden" => AxRole::Presentation,
        _ => AxRole::Textbox,
    }
}

// ---------------------------------------------------------------------------
// Accessible name (accname-1.2 §4.3.2).
// ---------------------------------------------------------------------------

/// Compute the accessible name for `dom` (role-aware). Precedence:
///   1. `aria-labelledby` → concatenated text of referenced elements
///   2. `aria-label`
///   3. native host-language label: associated `<label>`, `alt`, control value
///   4. name from content (only for roles that permit it)
///   5. `title`
pub fn compute_accessible_name(doc: &Document, dom: NodeId, role: AxRole) -> String {
    // 1. aria-labelledby.
    if let Some(ids) = doc.attr_raw(dom, "aria-labelledby") {
        let mut parts = Vec::new();
        for id in ids.split_whitespace() {
            if let Some(target) = doc.get_element_by_id(id) {
                let t = labelledby_text(doc, target);
                if !t.is_empty() {
                    parts.push(t);
                }
            }
        }
        let joined = parts.join(" ");
        if !joined.trim().is_empty() {
            return normalize_ws(&joined);
        }
    }

    // 2. aria-label.
    if let Some(l) = doc.attr_raw(dom, "aria-label") {
        let l = normalize_ws(l);
        if !l.is_empty() {
            return l;
        }
    }

    // 3. native host-language labeling.
    if let Some(n) = native_label(doc, dom) {
        let n = normalize_ws(&n);
        if !n.is_empty() {
            return n;
        }
    }

    // 4. name from content (button/link/heading/listitem/... per the role).
    if role.name_from_content() {
        let t = normalize_ws(&doc.text_content(dom));
        if !t.is_empty() {
            return t;
        }
    }

    // 5. title fallback.
    if let Some(t) = doc.attr_raw(dom, "title") {
        let t = normalize_ws(t);
        if !t.is_empty() {
            return t;
        }
    }

    String::new()
}

/// Text contributed by an element referenced from `aria-labelledby`. Per
/// accname §4.3.2 step 2.1, when the referenced node is itself a control its
/// value is used; otherwise its accessible name / text content is used. We use
/// a value-aware text gather (form-control value, else text content).
fn labelledby_text(doc: &Document, dom: NodeId) -> String {
    // If it has its own aria-label, use that.
    if let Some(l) = doc.attr_raw(dom, "aria-label") {
        let l = normalize_ws(l);
        if !l.is_empty() {
            return l;
        }
    }
    let tag = doc.tag_raw(dom).unwrap_or("").to_ascii_lowercase();
    if tag == "input" || tag == "textarea" || tag == "select" {
        if let Some(v) = doc.attr_raw(dom, "value") {
            let v = normalize_ws(v);
            if !v.is_empty() {
                return v;
            }
        }
    }
    normalize_ws(&doc.text_content(dom))
}

/// Native host-language label sources:
///   * for a form control: an associated `<label>` (via `for=` or wrapping),
///     else `alt` (image inputs), else the `value` of a button-type input;
///   * for `<img>`: the `alt` attribute;
///   * for `<input type=submit/reset>` with no value: the default button text.
fn native_label(doc: &Document, dom: NodeId) -> Option<String> {
    let tag = doc.tag_raw(dom)?.to_ascii_lowercase();
    match tag.as_str() {
        "img" | "area" => doc.attr_raw(dom, "alt").map(str::to_string),
        "input" => {
            let ty = doc
                .attr_raw(dom, "type")
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_else(|| "text".into());
            match ty.as_str() {
                // button-like inputs: the name is the value attribute, or the
                // browser's default label for submit/reset.
                "submit" => Some(
                    doc.attr_raw(dom, "value")
                        .map(str::to_string)
                        .unwrap_or_else(|| "Submit".into()),
                ),
                "reset" => Some(
                    doc.attr_raw(dom, "value")
                        .map(str::to_string)
                        .unwrap_or_else(|| "Reset".into()),
                ),
                "button" => doc.attr_raw(dom, "value").map(str::to_string),
                "image" => doc.attr_raw(dom, "alt").map(str::to_string),
                // text-like inputs: an associated <label> names them.
                _ => associated_label_text(doc, dom),
            }
        }
        "textarea" | "select" => associated_label_text(doc, dom),
        _ => None,
    }
}

/// Find the text of a `<label>` associated with the control `dom`:
///   * an explicit `<label for=ID>` whose `for` equals the control's `id`, or
///   * a `<label>` ancestor that wraps the control.
fn associated_label_text(doc: &Document, dom: NodeId) -> Option<String> {
    // Explicit association via for=.
    if let Some(id) = doc.attr_raw(dom, "id") {
        if let Some(lbl) = find_label_for(doc, doc.root(), id) {
            let t = doc.text_content(lbl);
            if !t.trim().is_empty() {
                return Some(t);
            }
        }
    }
    // Implicit association: a <label> ancestor.
    let mut cur = doc.parent(dom);
    while let Some(p) = cur {
        if doc.tag_raw(p).map(|t| t.eq_ignore_ascii_case("label")) == Some(true) {
            let t = doc.text_content(p);
            if !t.trim().is_empty() {
                return Some(t);
            }
        }
        cur = doc.parent(p);
    }
    None
}

/// Depth-first search for `<label for=target_id>`.
fn find_label_for(doc: &Document, start: NodeId, target_id: &str) -> Option<NodeId> {
    if doc.tag_raw(start).map(|t| t.eq_ignore_ascii_case("label")) == Some(true)
        && doc.attr_raw(start, "for") == Some(target_id)
    {
        return Some(start);
    }
    for c in doc.children(start) {
        if matches!(doc.kind(c), Some(NodeKind::Element { .. })) {
            if let Some(found) = find_label_for(doc, c, target_id) {
                return Some(found);
            }
        }
    }
    None
}

/// `aria-describedby` / `title` → UIA HelpText. `title` is NOT reused here if it
/// was already consumed as the name (the caller computes the name first; we
/// only surface description-specific sources). We follow Chrome: describedby
/// always contributes a description; title contributes a description only when
/// it didn't become the name. Since we can't cheaply know that here, we expose
/// describedby; title is left to the name path.
fn compute_description(doc: &Document, dom: NodeId) -> String {
    if let Some(ids) = doc.attr_raw(dom, "aria-describedby") {
        let mut parts = Vec::new();
        for id in ids.split_whitespace() {
            if let Some(target) = doc.get_element_by_id(id) {
                let t = normalize_ws(&doc.text_content(target));
                if !t.is_empty() {
                    parts.push(t);
                }
            }
        }
        return parts.join(" ");
    }
    String::new()
}

// ---------------------------------------------------------------------------
// States + value.
// ---------------------------------------------------------------------------

fn fill_states(doc: &Document, dom: NodeId, tag: &str, role: AxRole, node: &mut AxNode) {
    // disabled: native `disabled` attr or aria-disabled="true".
    node.disabled = doc.has_attribute(dom, "disabled")
        || doc.attr_raw(dom, "aria-disabled") == Some("true");

    // required: native `required` attr or aria-required="true".
    node.required = doc.has_attribute(dom, "required")
        || doc.attr_raw(dom, "aria-required") == Some("true");

    // password.
    if tag == "input" {
        node.password = doc.attr_raw(dom, "type") == Some("password");
    }

    // heading level (aria-level overrides the h-tag number).
    if role == AxRole::Heading {
        node.level = doc
            .attr_raw(dom, "aria-level")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or_else(|| heading_level_from_tag(tag));
    }

    // checked tri-state for checkbox / radio / switch.
    if matches!(role, AxRole::Checkbox | AxRole::Radio) {
        node.checked = checked_state(doc, dom, tag);
    }

    // expanded state for disclosure widgets.
    node.expanded = expanded_state(doc, dom, tag);

    // value for form controls.
    node.value = control_value(doc, dom, tag, role);
}

fn heading_level_from_tag(tag: &str) -> u32 {
    match tag {
        "h1" => 1,
        "h2" => 2,
        "h3" => 3,
        "h4" => 4,
        "h5" => 5,
        "h6" => 6,
        _ => 0,
    }
}

fn checked_state(doc: &Document, dom: NodeId, tag: &str) -> CheckedState {
    // aria-checked wins when present.
    match doc.attr_raw(dom, "aria-checked") {
        Some("true") => return CheckedState::Checked,
        Some("false") => return CheckedState::Unchecked,
        Some("mixed") => return CheckedState::Mixed,
        _ => {}
    }
    if tag == "input" {
        // Native indeterminate is a JS property (not reflected to an attribute);
        // the parsed DOM exposes only `checked`. `indeterminate` may appear as
        // an attribute if the page set it explicitly.
        if doc.has_attribute(dom, "indeterminate") {
            return CheckedState::Mixed;
        }
        return if doc.has_attribute(dom, "checked") {
            CheckedState::Checked
        } else {
            CheckedState::Unchecked
        };
    }
    // ARIA checkbox/radio on a non-input default to unchecked.
    CheckedState::Unchecked
}

fn expanded_state(doc: &Document, dom: NodeId, tag: &str) -> ExpandedState {
    match doc.attr_raw(dom, "aria-expanded") {
        Some("true") => return ExpandedState::Expanded,
        Some("false") => return ExpandedState::Collapsed,
        _ => {}
    }
    if tag == "details" {
        return if doc.has_attribute(dom, "open") {
            ExpandedState::Expanded
        } else {
            ExpandedState::Collapsed
        };
    }
    ExpandedState::Undefined
}

fn control_value(doc: &Document, dom: NodeId, tag: &str, role: AxRole) -> String {
    match tag {
        "input" => {
            // checkbox/radio/button have no "value" in the AX value sense.
            if matches!(role, AxRole::Checkbox | AxRole::Radio | AxRole::Button) {
                String::new()
            } else {
                doc.attr_raw(dom, "value").map(str::to_string).unwrap_or_default()
            }
        }
        "textarea" => doc.text_content(dom),
        "select" => selected_option_text(doc, dom),
        "progress" | "meter" => doc
            .attr_raw(dom, "value")
            .map(str::to_string)
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Text of the `<option selected>` (or the first option) of a `<select>`.
fn selected_option_text(doc: &Document, select: NodeId) -> String {
    let mut first: Option<NodeId> = None;
    let mut stack = doc.children(select);
    while let Some(c) = stack.pop() {
        if doc.tag_raw(c).map(|t| t.eq_ignore_ascii_case("option")) == Some(true) {
            if first.is_none() {
                first = Some(c);
            }
            if doc.has_attribute(c, "selected") {
                return normalize_ws(&doc.text_content(c));
            }
        }
        for g in doc.children(c) {
            stack.push(g);
        }
    }
    first.map(|f| normalize_ws(&doc.text_content(f))).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// The document `<title>` text, if present.
fn document_title(doc: &Document) -> Option<String> {
    let titles = doc.get_elements_by_tag_name("title");
    let t = titles.first()?;
    let s = normalize_ws(&doc.text_content(*t));
    if s.is_empty() { None } else { Some(s) }
}

/// Collapse runs of ASCII whitespace to single spaces and trim — accname
/// requires "flat string" normalization of the gathered text.
fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_dom::Document;

    /// Build a tiny document and return (doc, body) for tests.
    fn doc_with_body() -> (Document, NodeId) {
        let mut d = Document::new();
        let html = d.create_element("html");
        let body = d.create_element("body");
        d.append_child(d.root(), html).unwrap();
        d.append_child(html, body).unwrap();
        (d, body)
    }

    fn text(d: &mut Document, parent: NodeId, s: &str) {
        let t = d.create_text_node(s);
        d.append_child(parent, t).unwrap();
    }

    #[test]
    fn button_has_role_button_and_name_from_content() {
        let (mut d, body) = doc_with_body();
        let btn = d.create_element("button");
        d.append_child(body, btn).unwrap();
        text(&mut d, btn, "OK");
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_name("OK").expect("button node");
        assert_eq!(n.role, AxRole::Button);
        assert_eq!(n.name, "OK");
    }

    #[test]
    fn input_aria_label_names_it() {
        let (mut d, body) = doc_with_body();
        let inp = d.create_element("input");
        d.set_attribute(inp, "aria-label", "Search");
        d.append_child(body, inp).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_name("Search").expect("input node");
        assert_eq!(n.role, AxRole::Textbox);
        assert_eq!(n.name, "Search");
    }

    #[test]
    fn checked_checkbox_exposes_checked_state() {
        let (mut d, body) = doc_with_body();
        let cb = d.create_element("input");
        d.set_attribute(cb, "type", "checkbox");
        d.set_attribute(cb, "checked", "");
        d.append_child(body, cb).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree
            .find_by_dom(cb.to_bits())
            .expect("checkbox node");
        assert_eq!(n.role, AxRole::Checkbox);
        assert_eq!(n.checked, CheckedState::Checked);
        // An unchecked one is Unchecked, not None.
        let cb2 = d.create_element("input");
        d.set_attribute(cb2, "type", "checkbox");
        d.append_child(body, cb2).unwrap();
        let tree = build_ax_tree(&d, None);
        let n2 = tree.find_by_dom(cb2.to_bits()).unwrap();
        assert_eq!(n2.checked, CheckedState::Unchecked);
    }

    #[test]
    fn aria_labelledby_resolves_to_referenced_text() {
        let (mut d, body) = doc_with_body();
        let label = d.create_element("span");
        d.set_attribute(label, "id", "lbl");
        d.append_child(body, label).unwrap();
        text(&mut d, label, "Full name");
        let inp = d.create_element("input");
        d.set_attribute(inp, "aria-labelledby", "lbl");
        d.append_child(body, inp).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(inp.to_bits()).expect("input node");
        assert_eq!(n.name, "Full name");
    }

    #[test]
    fn associated_label_for_names_input() {
        let (mut d, body) = doc_with_body();
        let label = d.create_element("label");
        d.set_attribute(label, "for", "email");
        d.append_child(body, label).unwrap();
        text(&mut d, label, "Email address");
        let inp = d.create_element("input");
        d.set_attribute(inp, "id", "email");
        d.set_attribute(inp, "type", "email");
        d.append_child(body, inp).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(inp.to_bits()).unwrap();
        assert_eq!(n.name, "Email address");
        assert_eq!(n.role, AxRole::Textbox);
    }

    #[test]
    fn labelledby_beats_label_beats_arialabel_precedence() {
        // aria-labelledby must win over aria-label AND over an associated label.
        let (mut d, body) = doc_with_body();
        let lbl = d.create_element("span");
        d.set_attribute(lbl, "id", "L");
        d.append_child(body, lbl).unwrap();
        text(&mut d, lbl, "By labelledby");
        let assoc = d.create_element("label");
        d.set_attribute(assoc, "for", "ctl");
        d.append_child(body, assoc).unwrap();
        text(&mut d, assoc, "By label");
        let inp = d.create_element("input");
        d.set_attribute(inp, "id", "ctl");
        d.set_attribute(inp, "aria-label", "By aria-label");
        d.set_attribute(inp, "aria-labelledby", "L");
        d.append_child(body, inp).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(inp.to_bits()).unwrap();
        assert_eq!(n.name, "By labelledby");
    }

    #[test]
    fn arialabel_beats_label_when_no_labelledby() {
        let (mut d, body) = doc_with_body();
        let assoc = d.create_element("label");
        d.set_attribute(assoc, "for", "ctl");
        d.append_child(body, assoc).unwrap();
        text(&mut d, assoc, "By label");
        let inp = d.create_element("input");
        d.set_attribute(inp, "id", "ctl");
        d.set_attribute(inp, "aria-label", "By aria-label");
        d.append_child(body, inp).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(inp.to_bits()).unwrap();
        assert_eq!(n.name, "By aria-label");
    }

    #[test]
    fn explicit_role_overrides_native_tag() {
        let (mut d, body) = doc_with_body();
        let div = d.create_element("div");
        d.set_attribute(div, "role", "button");
        d.append_child(body, div).unwrap();
        text(&mut d, div, "Click");
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(div.to_bits()).unwrap();
        assert_eq!(n.role, AxRole::Button);
        assert_eq!(n.name, "Click");
    }

    #[test]
    fn a_without_href_is_generic_with_href_is_link() {
        let (mut d, body) = doc_with_body();
        let plain = d.create_element("a");
        d.append_child(body, plain).unwrap();
        let linked = d.create_element("a");
        d.set_attribute(linked, "href", "/x");
        d.append_child(body, linked).unwrap();
        text(&mut d, linked, "Home");
        let tree = build_ax_tree(&d, None);
        assert_eq!(tree.find_by_dom(plain.to_bits()).unwrap().role, AxRole::Generic);
        let l = tree.find_by_dom(linked.to_bits()).unwrap();
        assert_eq!(l.role, AxRole::Link);
        assert_eq!(l.name, "Home");
    }

    #[test]
    fn img_empty_alt_is_pruned() {
        let (mut d, body) = doc_with_body();
        let img = d.create_element("img");
        d.set_attribute(img, "alt", "");
        d.set_attribute(img, "src", "spacer.gif");
        d.append_child(body, img).unwrap();
        let img2 = d.create_element("img");
        d.set_attribute(img2, "alt", "A cat");
        d.append_child(body, img2).unwrap();
        let tree = build_ax_tree(&d, None);
        // The empty-alt image produced no AX node.
        assert!(tree.find_by_dom(img.to_bits()).is_none());
        // The real image is present, named from alt, role Image.
        let n = tree.find_by_dom(img2.to_bits()).unwrap();
        assert_eq!(n.role, AxRole::Image);
        assert_eq!(n.name, "A cat");
    }

    #[test]
    fn presentation_role_reparents_children() {
        // <ul role=presentation><li>X</li></ul>: the ul is pruned, the li
        // attaches to the body's AX node.
        let (mut d, body) = doc_with_body();
        let ul = d.create_element("ul");
        d.set_attribute(ul, "role", "presentation");
        d.append_child(body, ul).unwrap();
        let li = d.create_element("li");
        d.append_child(ul, li).unwrap();
        text(&mut d, li, "Item");
        let tree = build_ax_tree(&d, None);
        assert!(tree.find_by_dom(ul.to_bits()).is_none(), "ul pruned");
        let li_node = tree.find_by_dom(li.to_bits()).expect("li present");
        // Its parent is the body's AX node (the ul was skipped).
        let parent_id = li_node.parent.unwrap();
        let parent = tree.get(parent_id).unwrap();
        assert_eq!(parent.dom_node, body.to_bits());
    }

    #[test]
    fn tree_mirrors_dom_hierarchy() {
        let (mut d, body) = doc_with_body();
        let nav = d.create_element("nav");
        d.append_child(body, nav).unwrap();
        let ul = d.create_element("ul");
        d.append_child(nav, ul).unwrap();
        let li = d.create_element("li");
        d.append_child(ul, li).unwrap();
        let a = d.create_element("a");
        d.set_attribute(a, "href", "/");
        d.append_child(li, a).unwrap();
        text(&mut d, a, "Home");
        let tree = build_ax_tree(&d, None);
        let nav_n = tree.find_by_dom(nav.to_bits()).unwrap();
        assert_eq!(nav_n.role, AxRole::Navigation);
        let ul_id = nav_n.children[0];
        assert_eq!(tree.get(ul_id).unwrap().role, AxRole::List);
        let li_id = tree.get(ul_id).unwrap().children[0];
        assert_eq!(tree.get(li_id).unwrap().role, AxRole::ListItem);
        let a_id = tree.get(li_id).unwrap().children[0];
        let a_n = tree.get(a_id).unwrap();
        assert_eq!(a_n.role, AxRole::Link);
        assert_eq!(a_n.name, "Home");
    }

    #[test]
    fn heading_level_from_tag_and_arialevel() {
        let (mut d, body) = doc_with_body();
        let h3 = d.create_element("h3");
        d.append_child(body, h3).unwrap();
        text(&mut d, h3, "Section");
        let custom = d.create_element("div");
        d.set_attribute(custom, "role", "heading");
        d.set_attribute(custom, "aria-level", "2");
        d.append_child(body, custom).unwrap();
        text(&mut d, custom, "Custom");
        let tree = build_ax_tree(&d, None);
        assert_eq!(tree.find_by_dom(h3.to_bits()).unwrap().level, 3);
        assert_eq!(tree.find_by_dom(custom.to_bits()).unwrap().level, 2);
    }

    #[test]
    fn disabled_and_required_states() {
        let (mut d, body) = doc_with_body();
        let inp = d.create_element("input");
        d.set_attribute(inp, "disabled", "");
        d.set_attribute(inp, "required", "");
        d.append_child(body, inp).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(inp.to_bits()).unwrap();
        assert!(n.disabled);
        assert!(n.required);
    }

    #[test]
    fn password_input_marked_password_with_value() {
        let (mut d, body) = doc_with_body();
        let inp = d.create_element("input");
        d.set_attribute(inp, "type", "password");
        d.set_attribute(inp, "value", "hunter2");
        d.append_child(body, inp).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(inp.to_bits()).unwrap();
        assert!(n.password);
        assert_eq!(n.role, AxRole::Textbox);
    }

    #[test]
    fn details_expanded_state_from_open() {
        let (mut d, body) = doc_with_body();
        let det = d.create_element("details");
        d.set_attribute(det, "open", "");
        d.append_child(body, det).unwrap();
        let det2 = d.create_element("details");
        d.append_child(body, det2).unwrap();
        let tree = build_ax_tree(&d, None);
        assert_eq!(
            tree.find_by_dom(det.to_bits()).unwrap().expanded,
            ExpandedState::Expanded
        );
        assert_eq!(
            tree.find_by_dom(det2.to_bits()).unwrap().expanded,
            ExpandedState::Collapsed
        );
    }

    #[test]
    fn select_value_is_selected_option() {
        let (mut d, body) = doc_with_body();
        let sel = d.create_element("select");
        d.append_child(body, sel).unwrap();
        let o1 = d.create_element("option");
        d.append_child(sel, o1).unwrap();
        text(&mut d, o1, "Red");
        let o2 = d.create_element("option");
        d.set_attribute(o2, "selected", "");
        d.append_child(sel, o2).unwrap();
        text(&mut d, o2, "Green");
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(sel.to_bits()).unwrap();
        assert_eq!(n.role, AxRole::Combobox);
        assert_eq!(n.value, "Green");
    }

    #[test]
    fn focus_flag_set_on_focused_node() {
        let (mut d, body) = doc_with_body();
        let btn = d.create_element("button");
        d.append_child(body, btn).unwrap();
        text(&mut d, btn, "Go");
        let tree = build_ax_tree(&d, Some(btn));
        let n = tree.find_by_dom(btn.to_bits()).unwrap();
        assert!(n.focused);
        assert_eq!(tree.focus, Some(n.id));
    }

    #[test]
    fn input_type_variants_map_to_roles() {
        let cases = [
            ("checkbox", AxRole::Checkbox),
            ("radio", AxRole::Radio),
            ("submit", AxRole::Button),
            ("range", AxRole::Slider),
            ("number", AxRole::Spinbutton),
            ("search", AxRole::Searchbox),
            ("email", AxRole::Textbox),
            ("text", AxRole::Textbox),
        ];
        for (ty, want) in cases {
            let (mut d, body) = doc_with_body();
            let inp = d.create_element("input");
            d.set_attribute(inp, "type", ty);
            d.append_child(body, inp).unwrap();
            let tree = build_ax_tree(&d, None);
            let n = tree.find_by_dom(inp.to_bits()).unwrap();
            assert_eq!(n.role, want, "input type={ty}");
        }
    }

    #[test]
    fn whitespace_in_name_is_normalized() {
        let (mut d, body) = doc_with_body();
        let btn = d.create_element("button");
        d.append_child(body, btn).unwrap();
        text(&mut d, btn, "  Save\n   changes  ");
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(btn.to_bits()).unwrap();
        assert_eq!(n.name, "Save changes");
    }

    #[test]
    fn aria_describedby_becomes_description() {
        let (mut d, body) = doc_with_body();
        let help = d.create_element("span");
        d.set_attribute(help, "id", "h");
        d.append_child(body, help).unwrap();
        text(&mut d, help, "Must be 8+ chars");
        let inp = d.create_element("input");
        d.set_attribute(inp, "aria-label", "Password");
        d.set_attribute(inp, "aria-describedby", "h");
        d.append_child(body, inp).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(inp.to_bits()).unwrap();
        assert_eq!(n.name, "Password");
        assert_eq!(n.description, "Must be 8+ chars");
    }

    #[test]
    fn document_root_uses_title() {
        let mut d = Document::new();
        let html = d.create_element("html");
        d.append_child(d.root(), html).unwrap();
        let head = d.create_element("head");
        d.append_child(html, head).unwrap();
        let title = d.create_element("title");
        d.append_child(head, title).unwrap();
        text(&mut d, title, "My Page");
        let body = d.create_element("body");
        d.append_child(html, body).unwrap();
        let tree = build_ax_tree(&d, None);
        let roots = tree.roots();
        assert_eq!(roots.len(), 1);
        let root = tree.get(roots[0]).unwrap();
        assert_eq!(root.role, AxRole::Document);
        assert_eq!(root.name, "My Page");
        // <head>/<title> produced no AX nodes.
        assert!(tree.find_by_dom(head.to_bits()).is_none());
        assert!(tree.find_by_dom(title.to_bits()).is_none());
    }

    #[test]
    fn aria_checked_mixed_is_indeterminate() {
        let (mut d, body) = doc_with_body();
        let cb = d.create_element("div");
        d.set_attribute(cb, "role", "checkbox");
        d.set_attribute(cb, "aria-checked", "mixed");
        d.append_child(body, cb).unwrap();
        let tree = build_ax_tree(&d, None);
        let n = tree.find_by_dom(cb.to_bits()).unwrap();
        assert_eq!(n.checked, CheckedState::Mixed);
    }
}
