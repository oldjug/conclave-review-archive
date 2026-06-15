//! Process-wide published UIA provider snapshot — the bridge between the
//! renderer (which builds the [`crate::AxTree`]) and the UI thread (which fields
//! `WM_GETOBJECT` from the OS accessibility runtime / Narrator).
//!
//! ## Why a published snapshot
//!
//! The page graph (DOM, AX tree) lives on the renderer thread and is `!Send`.
//! `WM_GETOBJECT` arrives on the UI thread inside the window procedure. We
//! therefore cannot hand the OS a pointer into the renderer's AX tree. Instead
//! the renderer **publishes** a flat, `Send + Sync` snapshot here after each
//! layout; the UI thread reads it to answer UIA property / navigation requests.
//! This mirrors Chrome's design exactly: the renderer computes the AXTree and
//! serializes `AXNodeData` updates across the process boundary to the browser
//! process, which hosts the platform (UIA/IA2) objects.
//!
//! ## The COM bridge
//!
//! Win32 delivers accessibility through `WM_GETOBJECT`. When `lParam ==
//! UiaRootObjectId` (-25) the window must return, via
//! `UiaReturnRawElementProvider`, an `IRawElementProviderSimple` for the
//! window's content root. That provider's `GetPropertyValue`, and the
//! `IRawElementProviderFragment` navigation, are served from this snapshot:
//! each [`PublishedNode`] already holds the marshaled [`UiaProvider`] (its
//! property variants come from [`UiaProvider::property`]) plus parent/child
//! links for `NavigateDirection`.
//!
//! Instantiating the COM object whose vtable forwards into this snapshot is the
//! one remaining platform step; it is gated behind `CV_A11Y_UIA` (default OFF)
//! so the UIAutomationCore provider is constructed only when an assistive
//! technology is actually present, and the default browser path is never
//! perturbed by COM/UIA initialization. See [`a11y_uia_enabled`].

use crate::uia::{NavigateDirection, UiaProvider, VariantValue};
use crate::{AxTree, UiaPropertyId};
use std::sync::{Mutex, OnceLock};

/// One node of the published snapshot: its marshaled provider plus the tree
/// links the UIA fragment-navigation methods need.
#[derive(Debug, Clone)]
pub struct PublishedNode {
    pub provider: UiaProvider,
    /// AX id (== `provider.runtime_id[1]`), stable within a snapshot.
    pub ax_id: u32,
    pub parent: Option<u32>,
    pub children: Vec<u32>,
}

/// A flat, `Send + Sync` snapshot of an AX tree ready for UIA serving.
#[derive(Debug, Clone, Default)]
pub struct PublishedTree {
    /// Index 0 is the root; lookups are by `ax_id` through [`Self::find`].
    pub nodes: Vec<PublishedNode>,
    pub root: Option<u32>,
}

impl PublishedTree {
    /// Serialize an [`AxTree`] into a flat published snapshot. Every AX node
    /// becomes a [`PublishedNode`] carrying its [`UiaProvider`] (built via
    /// [`UiaProvider::from_ax`]) and parent/child links.
    pub fn from_ax(tree: &AxTree) -> Self {
        let roots = tree.roots();
        let root = roots.first().copied();
        let mut nodes = Vec::new();
        // Breadth-first from the root(s) so index 0 is the root.
        let mut queue: std::collections::VecDeque<u32> = roots.iter().copied().collect();
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = queue.pop_front() {
            if !seen.insert(id) {
                continue;
            }
            if let Some(n) = tree.get(id) {
                nodes.push(PublishedNode {
                    provider: UiaProvider::from_ax(n),
                    ax_id: n.id,
                    parent: n.parent,
                    children: n.children.clone(),
                });
                for &c in &n.children {
                    queue.push_back(c);
                }
            }
        }
        PublishedTree { nodes, root }
    }

    /// Find a node by AX id.
    pub fn find(&self, ax_id: u32) -> Option<&PublishedNode> {
        self.nodes.iter().find(|n| n.ax_id == ax_id)
    }

    /// The root node, if any.
    pub fn root_node(&self) -> Option<&PublishedNode> {
        self.root.and_then(|r| self.find(r))
    }

    /// Serve `IRawElementProviderSimple::GetPropertyValue` for `ax_id`.
    pub fn property(&self, ax_id: u32, prop: UiaPropertyId) -> VariantValue {
        match self.find(ax_id) {
            Some(n) => n.provider.property(prop),
            None => VariantValue::Empty,
        }
    }

    /// Serve `IRawElementProviderFragment::Navigate(direction)` for `ax_id`,
    /// returning the destination AX id (or `None` at a tree edge). This is the
    /// snapshot analogue of [`crate::uia::navigate`].
    pub fn navigate(&self, ax_id: u32, dir: NavigateDirection) -> Option<u32> {
        let n = self.find(ax_id)?;
        match dir {
            NavigateDirection::Parent => n.parent,
            NavigateDirection::FirstChild => n.children.first().copied(),
            NavigateDirection::LastChild => n.children.last().copied(),
            NavigateDirection::NextSibling => {
                let p = self.find(n.parent?)?;
                let i = p.children.iter().position(|&c| c == ax_id)?;
                p.children.get(i + 1).copied()
            }
            NavigateDirection::PreviousSibling => {
                let p = self.find(n.parent?)?;
                let i = p.children.iter().position(|&c| c == ax_id)?;
                if i == 0 {
                    None
                } else {
                    p.children.get(i - 1).copied()
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// The single published snapshot for the foreground page. The renderer writes it
/// after layout; the UI thread reads it on `WM_GETOBJECT`.
static PUBLISHED: OnceLock<Mutex<PublishedTree>> = OnceLock::new();

fn cell() -> &'static Mutex<PublishedTree> {
    PUBLISHED.get_or_init(|| Mutex::new(PublishedTree::default()))
}

/// Renderer side: publish the current page's AX tree for the UI thread to serve.
/// Called after the AX tree is (re)built. Cheap — a flat clone, no COM.
pub fn publish(tree: &AxTree) {
    let snap = PublishedTree::from_ax(tree);
    if let Ok(mut g) = cell().lock() {
        *g = snap;
    }
}

/// UI side: run `f` against the current published snapshot. Returns `None` if
/// nothing has been published yet. Used by the `WM_GETOBJECT` handler to read
/// properties / navigate without crossing the thread boundary into the renderer.
pub fn with_published<R>(f: impl FnOnce(&PublishedTree) -> R) -> Option<R> {
    cell().lock().ok().map(|g| f(&g))
}

/// True when live UIA COM registration is enabled (`CV_A11Y_UIA=1`). Default
/// OFF: the AX tree is always built + published (in-process queryable, tested),
/// but the OS-facing `IRawElementProviderSimple` is only constructed when an AT
/// is present and the operator opts in — keeping COM/UIA platform init out of
/// the default fast path.
pub fn a11y_uia_enabled() -> bool {
    std::env::var("CV_A11Y_UIA").map(|v| v == "1").unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_ax_tree, AxRole};
    use cv_dom::Document;

    fn page() -> Document {
        let mut d = Document::new();
        let html = d.create_element("html");
        let body = d.create_element("body");
        d.append_child(d.root(), html).unwrap();
        d.append_child(html, body).unwrap();
        let btn = d.create_element("button");
        d.append_child(body, btn).unwrap();
        let t = d.create_text_node("Save");
        d.append_child(btn, t).unwrap();
        d
    }

    #[test]
    fn published_tree_serializes_providers_and_links() {
        let d = page();
        let ax = build_ax_tree(&d, None);
        let pt = PublishedTree::from_ax(&ax);
        assert!(!pt.is_empty());
        // Root is the document.
        let root = pt.root_node().unwrap();
        assert_eq!(
            root.provider.control_type,
            crate::UiaControlType::Document
        );
        // The button provider is reachable and named.
        let btn = pt
            .nodes
            .iter()
            .find(|n| n.provider.control_type == crate::UiaControlType::Button)
            .unwrap();
        assert_eq!(btn.provider.name, "Save");
    }

    #[test]
    fn navigate_walks_published_links() {
        let d = page();
        let ax = build_ax_tree(&d, None);
        let pt = PublishedTree::from_ax(&ax);
        let root = pt.root.unwrap();
        // Descend root → body → button via FirstChild repeatedly until a button.
        let mut cur = root;
        let mut found_button = false;
        for _ in 0..8 {
            match pt.navigate(cur, NavigateDirection::FirstChild) {
                Some(next) => {
                    cur = next;
                    if pt.find(cur).unwrap().provider.control_type
                        == crate::UiaControlType::Button
                    {
                        found_button = true;
                        break;
                    }
                }
                None => break,
            }
        }
        assert!(found_button, "navigated down to the button");
        // Parent of the button is non-None.
        assert!(pt.navigate(cur, NavigateDirection::Parent).is_some());
    }

    #[test]
    fn property_served_from_snapshot() {
        let d = page();
        let ax = build_ax_tree(&d, None);
        let pt = PublishedTree::from_ax(&ax);
        let btn = pt
            .nodes
            .iter()
            .find(|n| n.provider.control_type == crate::UiaControlType::Button)
            .unwrap();
        assert_eq!(
            pt.property(btn.ax_id, UiaPropertyId::Name),
            VariantValue::BStr("Save".into())
        );
        // ControlType is the Button id (50000).
        assert_eq!(
            pt.property(btn.ax_id, UiaPropertyId::ControlType),
            VariantValue::I4(50000)
        );
    }

    #[test]
    fn publish_then_read_roundtrips_across_the_static() {
        let d = page();
        let ax = build_ax_tree(&d, None);
        publish(&ax);
        let name = with_published(|pt| {
            pt.nodes
                .iter()
                .find(|n| n.provider.control_type == crate::UiaControlType::Button)
                .map(|n| n.provider.name.clone())
        })
        .flatten();
        assert_eq!(name.as_deref(), Some("Save"));
    }

    #[test]
    fn role_states_survive_serialization() {
        let mut d = Document::new();
        let body = d.create_element("body");
        d.append_child(d.root(), body).unwrap();
        let cb = d.create_element("input");
        d.set_attribute(cb, "type", "checkbox");
        d.set_attribute(cb, "checked", "");
        d.set_attribute(cb, "aria-label", "Subscribe");
        d.append_child(body, cb).unwrap();
        let ax = build_ax_tree(&d, None);
        assert_eq!(
            ax.find_by_name("Subscribe").unwrap().role,
            AxRole::Checkbox
        );
        let pt = PublishedTree::from_ax(&ax);
        let n = pt
            .nodes
            .iter()
            .find(|n| n.provider.name == "Subscribe")
            .unwrap();
        // Toggle pattern is supported and reports On.
        use crate::uia::{ToggleState, UiaPatternId};
        assert!(n.provider.supports_pattern(UiaPatternId::Toggle));
        assert_eq!(n.provider.toggle_state, ToggleState::On);
    }
}
