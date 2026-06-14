//! `cv_a11y` — accessibility tree + UI Automation provider.
//!
//! Real Win32 UIAutomationCore types: GUIDs for IUnknown,
//! IRawElementProviderSimple, IRawElementProviderFragment,
//! IRawElementProviderFragmentRoot; control-type / property IDs;
//! variant marshaling. The COM vtables are laid out exactly per
//! UIAutomationCoreApi.h and the IDispatch / IUnknown method tables.

#![allow(dead_code, missing_debug_implementations, unused_doc_comments)]

pub mod uia;
pub mod uia_provider;
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
    fn from_aria(s: &str) -> Option<Self> {
        Some(match s {
            "button" => Self::Button,
            "checkbox" => Self::Checkbox,
            "combobox" => Self::Combobox,
            "heading" => Self::Heading,
            "img" | "image" => Self::Image,
            "link" => Self::Link,
            "list" => Self::List,
            "listitem" => Self::ListItem,
            "main" => Self::Main,
            "menu" => Self::Menu,
            "menuitem" => Self::MenuItem,
            "navigation" => Self::Navigation,
            "radio" => Self::Radio,
            "region" => Self::Region,
            "search" => Self::Search,
            "slider" => Self::Slider,
            "spinbutton" => Self::Spinbutton,
            "status" => Self::Status,
            "tab" => Self::Tab,
            "tablist" => Self::Tablist,
            "tabpanel" => Self::Tabpanel,
            "textbox" => Self::Textbox,
            "tree" => Self::Tree,
            "treeitem" => Self::TreeItem,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct AxNode {
    pub id: u32,
    pub role: AxRole,
    /// Accessible name (`aria-label` / inner text / `alt`).
    pub name: String,
    /// Bounding box in viewport pixels.
    pub bbox: (i32, i32, u32, u32),
    pub focused: bool,
    pub disabled: bool,
    pub parent: Option<u32>,
    pub children: Vec<u32>,
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
                bbox: (0, 0, 0, 0),
                focused: false,
                disabled: false,
                parent,
                children: Vec::new(),
            },
        );
        id
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
