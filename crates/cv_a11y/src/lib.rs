//! `cv_a11y` — accessibility tree + UI Automation provider.
//!
//! Real Win32 UIAutomationCore types: GUIDs for IUnknown,
//! IRawElementProviderSimple, IRawElementProviderFragment,
//! IRawElementProviderFragmentRoot; control-type / property IDs;
//! variant marshaling. The COM vtables are laid out exactly per
//! UIAutomationCoreApi.h and the IDispatch / IUnknown method tables.

#![allow(dead_code, missing_debug_implementations, unused_doc_comments)]

pub mod build;
#[cfg(target_os = "windows")]
pub mod com;
pub mod provider;
pub mod uia;
pub mod uia_provider;
pub use build::{build_ax_tree, compute_accessible_name};
pub use provider::{a11y_uia_enabled, publish, with_published, PublishedNode, PublishedTree};
pub use uia::{UIAFRAGMENT_ROOT, UiaControlType, UiaPropertyId, UiaProvider};

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxRole {
    Application,
    Document,
    Article,
    Banner,
    Button,
    Checkbox,
    Combobox,
    Form,
    Group,
    Heading,
    Image,
    Link,
    List,
    ListItem,
    Main,
    Menu,
    MenuItem,
    Navigation,
    Paragraph,
    Radio,
    Region,
    Search,
    /// `<input type=search>` — distinct UIA mapping (Edit) but a searchbox role
    /// per ARIA-in-HTML.
    Searchbox,
    Section,
    Slider,
    Spinbutton,
    Status,
    Tab,
    Tablist,
    Tabpanel,
    Textbox,
    Tree,
    TreeItem,
    /// `role="presentation"` / `role="none"` and `<img alt="">` — these nodes are
    /// pruned from the exposed tree (their children are reparented to the
    /// presentational node's parent), matching Blink's "ignored" objects.
    Presentation,
    /// `<table>` — exposed as a UIA Table.
    Table,
    /// Landmark `<footer>` / `role=contentinfo`.
    Contentinfo,
    /// Landmark `<aside>` / `role=complementary`.
    Complementary,
    Generic,
}

impl AxRole {
    /// Map an ARIA role string or HTML tag to an AxRole.
    pub fn from_html(tag: &str, aria_role: Option<&str>) -> Self {
        if let Some(r) = aria_role {
            return Self::from_aria(r).unwrap_or(Self::Generic);
        }
        match tag.to_lowercase().as_str() {
            "main" => Self::Main,
            "nav" => Self::Navigation,
            "header" => Self::Banner,
            "article" => Self::Article,
            "section" => Self::Section,
            "form" => Self::Form,
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => Self::Heading,
            "p" => Self::Paragraph,
            "ul" | "ol" => Self::List,
            "li" => Self::ListItem,
            "a" => Self::Link,
            "button" => Self::Button,
            "input" => Self::Textbox,
            "img" => Self::Image,
            _ => Self::Generic,
        }
    }
    /// Map an explicit ARIA `role` token to an `AxRole`. Returns `None` for an
    /// unrecognized token so the caller can fall back to the host-language role
    /// (WAI-ARIA: an invalid role is treated as no role at all).
    pub fn from_aria(s: &str) -> Option<Self> {
        Some(match s {
            "button" => Self::Button,
            "checkbox" | "switch" | "menuitemcheckbox" => Self::Checkbox,
            "combobox" => Self::Combobox,
            "heading" => Self::Heading,
            "img" | "image" => Self::Image,
            "link" => Self::Link,
            "list" => Self::List,
            "listitem" => Self::ListItem,
            "main" => Self::Main,
            "menu" | "menubar" => Self::Menu,
            "menuitem" => Self::MenuItem,
            "navigation" => Self::Navigation,
            "radio" | "menuitemradio" => Self::Radio,
            "region" => Self::Region,
            "search" => Self::Search,
            "searchbox" => Self::Searchbox,
            "slider" => Self::Slider,
            "spinbutton" => Self::Spinbutton,
            "status" => Self::Status,
            "tab" => Self::Tab,
            "tablist" => Self::Tablist,
            "tabpanel" => Self::Tabpanel,
            "textbox" => Self::Textbox,
            "tree" => Self::Tree,
            "treeitem" => Self::TreeItem,
            "table" | "grid" => Self::Table,
            "banner" => Self::Banner,
            "contentinfo" => Self::Contentinfo,
            "complementary" => Self::Complementary,
            "form" => Self::Form,
            "article" | "document" => Self::Article,
            "group" => Self::Group,
            "paragraph" => Self::Paragraph,
            "presentation" | "none" => Self::Presentation,
            _ => return None,
        })
    }

    /// True for roles whose accessible name MAY be computed from descendant text
    /// content (accname-1.2 §4.3.2 step 2.6 / the "Name from author"+"content"
    /// table). Used to decide whether to recurse into text when no explicit
    /// label is present. Form controls (textbox/combobox/slider/spinbutton) and
    /// most landmarks are NOT named from content.
    pub fn name_from_content(self) -> bool {
        matches!(
            self,
            Self::Button
                | Self::Checkbox
                | Self::Radio
                | Self::Link
                | Self::Heading
                | Self::ListItem
                | Self::MenuItem
                | Self::Tab
                | Self::TreeItem
                | Self::Status
        )
    }
}

/// Tri-state for ARIA `aria-checked` / native checkbox+radio state.
/// `Mixed` is `aria-checked="mixed"` (indeterminate). `None` means the role has
/// no checked semantics (it's not a checkbox/radio/switch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedState {
    None,
    Unchecked,
    Checked,
    Mixed,
}

/// Expanded state for disclosure widgets (`aria-expanded`, `<details open>`).
/// `Undefined` means the element is not expandable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandedState {
    Undefined,
    Collapsed,
    Expanded,
}

#[derive(Debug, Clone)]
pub struct AxNode {
    pub id: u32,
    pub role: AxRole,
    /// Accessible name computed per the accname-1.2 algorithm
    /// (aria-labelledby > aria-label > native label > alt/title > content).
    pub name: String,
    /// `aria-describedby` / `title` description text (maps to UIA HelpText).
    pub description: String,
    /// Current value for inputs / range widgets / textboxes (UIA Value).
    pub value: String,
    /// Bounding box in viewport pixels.
    pub bbox: (i32, i32, u32, u32),
    pub focused: bool,
    pub disabled: bool,
    /// `aria-required` / native `required` attribute on a form control.
    pub required: bool,
    /// True for `<input type=password>` (UIA IsPassword).
    pub password: bool,
    /// Heading level 1..=6 (0 when not a heading), exposed via `aria-level`.
    pub level: u32,
    /// Checkbox/radio/switch checked tri-state.
    pub checked: CheckedState,
    /// Disclosure expanded state.
    pub expanded: ExpandedState,
    /// The originating DOM `NodeId` bits (`NodeId::to_bits`), 0 for synthetic
    /// nodes. Lets a hit-test / focus-change map a DOM node back to its AX node.
    pub dom_node: u64,
    pub parent: Option<u32>,
    pub children: Vec<u32>,
}

impl AxNode {
    /// A blank node carrying only a role; all states default to "not present".
    /// The builder fills the remaining fields. `id`/`parent`/`children` are
    /// assigned when the node is inserted via [`AxTree::add_node`].
    pub fn new(role: AxRole) -> Self {
        AxNode {
            id: 0,
            role,
            name: String::new(),
            description: String::new(),
            value: String::new(),
            bbox: (0, 0, 0, 0),
            focused: false,
            disabled: false,
            required: false,
            password: false,
            level: 0,
            checked: CheckedState::None,
            expanded: ExpandedState::Undefined,
            dom_node: 0,
            parent: None,
            children: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct AxTree {
    nodes: HashMap<u32, AxNode>,
    pub focus: Option<u32>,
    next_id: u32,
}

impl AxTree {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, parent: Option<u32>, role: AxRole, name: impl Into<String>) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        if let Some(p) = parent {
            if let Some(n) = self.nodes.get_mut(&p) {
                n.children.push(id);
            }
        }
        self.nodes.insert(
            id,
            AxNode {
                id,
                role,
                name: name.into(),
                description: String::new(),
                value: String::new(),
                bbox: (0, 0, 0, 0),
                focused: false,
                disabled: false,
                required: false,
                password: false,
                level: 0,
                checked: CheckedState::None,
                expanded: ExpandedState::Undefined,
                dom_node: 0,
                parent,
                children: Vec::new(),
            },
        );
        id
    }

    /// Allocate the next id and insert a fully-populated [`AxNode`] under
    /// `parent`. The caller fills role/name/states; `id`, `parent` and
    /// `children` are managed here. Returns the new id. Used by the DOM→AX
    /// builder ([`build::build_ax_tree`]).
    pub fn add_node(&mut self, parent: Option<u32>, mut node: AxNode) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        if let Some(p) = parent {
            if let Some(n) = self.nodes.get_mut(&p) {
                n.children.push(id);
            }
        }
        node.id = id;
        node.parent = parent;
        node.children.clear();
        if node.focused {
            self.focus = Some(id);
        }
        self.nodes.insert(id, node);
        id
    }

    /// Number of children of `id` (0 if absent). Cheap accessor for the UIA
    /// fragment navigation glue.
    pub fn child_ids(&self, id: u32) -> Vec<u32> {
        self.nodes.get(&id).map(|n| n.children.clone()).unwrap_or_default()
    }

    /// The root node(s) — those with no parent. A well-formed page tree has
    /// exactly one (the document), but synthetic trees may have several.
    pub fn roots(&self) -> Vec<u32> {
        let mut r: Vec<u32> = self
            .nodes
            .values()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id)
            .collect();
        r.sort_unstable();
        r
    }

    /// Look up the AX node built from a given DOM node (`NodeId::to_bits`).
    pub fn find_by_dom(&self, dom_bits: u64) -> Option<&AxNode> {
        if dom_bits == 0 {
            return None;
        }
        self.nodes.values().find(|n| n.dom_node == dom_bits)
    }
    pub fn get(&self, id: u32) -> Option<&AxNode> {
        self.nodes.get(&id)
    }
    pub fn set_focus(&mut self, id: u32) {
        if let Some(old) = self.focus {
            if let Some(n) = self.nodes.get_mut(&old) {
                n.focused = false;
            }
        }
        if let Some(n) = self.nodes.get_mut(&id) {
            n.focused = true;
            self.focus = Some(id);
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    /// Find a node by accessible name — what screen readers use to
    /// fulfil "click the X button" intents.
    pub fn find_by_name(&self, name: &str) -> Option<&AxNode> {
        self.nodes.values().find(|n| n.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_from_button_tag() {
        assert_eq!(AxRole::from_html("button", None), AxRole::Button);
    }

    #[test]
    fn aria_role_overrides_tag() {
        assert_eq!(AxRole::from_html("div", Some("button")), AxRole::Button);
    }

    #[test]
    fn heading_tags_map_to_heading_role() {
        for h in ["h1", "h2", "h3", "h4", "h5", "h6"] {
            assert_eq!(AxRole::from_html(h, None), AxRole::Heading);
        }
    }

    #[test]
    fn build_tree_with_children() {
        let mut t = AxTree::new();
        let main = t.add(None, AxRole::Main, "main");
        let nav = t.add(Some(main), AxRole::Navigation, "nav");
        let _btn = t.add(Some(nav), AxRole::Button, "Submit");
        assert_eq!(t.len(), 3);
        let main_node = t.get(main).unwrap();
        assert_eq!(main_node.children.len(), 1);
        let nav_node = t.get(nav).unwrap();
        assert_eq!(nav_node.children.len(), 1);
    }

    #[test]
    fn focus_transfers_between_nodes() {
        let mut t = AxTree::new();
        let a = t.add(None, AxRole::Button, "A");
        let b = t.add(None, AxRole::Button, "B");
        t.set_focus(a);
        assert!(t.get(a).unwrap().focused);
        t.set_focus(b);
        assert!(!t.get(a).unwrap().focused);
        assert!(t.get(b).unwrap().focused);
    }

    #[test]
    fn find_by_name_locates_button() {
        let mut t = AxTree::new();
        t.add(None, AxRole::Button, "Submit");
        let n = t.find_by_name("Submit").unwrap();
        assert_eq!(n.role, AxRole::Button);
    }
}
