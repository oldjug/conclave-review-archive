//! NodeIterator / TreeWalker — real implementation.

use crate::{Document, NodeId, NodeKind};

/// Per-`NodeFilter` accept value (mirrors NodeFilter constants).
pub mod accept {
    pub const ACCEPT: u32 = 1;
    pub const REJECT: u32 = 2;
    pub const SKIP: u32 = 3;
}

#[derive(Debug, Clone, Copy)]
pub struct WhatToShow(pub u32);

impl WhatToShow {
    pub const ELEMENT: u32 = 0x1;
    pub const TEXT: u32 = 0x4;
    pub const COMMENT: u32 = 0x80;
    pub const DOCUMENT: u32 = 0x100;
    pub const DOCUMENT_FRAGMENT: u32 = 0x400;
    pub const ALL: u32 = 0xFFFF_FFFF;

    pub fn matches(self, kind: &NodeKind) -> bool {
        let bit = match kind {
            NodeKind::Element { .. } => Self::ELEMENT,
            NodeKind::Text(_) | NodeKind::CDataSection(_) => Self::TEXT,
            NodeKind::Comment(_) => Self::COMMENT,
            NodeKind::Document => Self::DOCUMENT,
            NodeKind::DocumentFragment => Self::DOCUMENT_FRAGMENT,
            _ => 0,
        };
        (self.0 & bit) != 0
    }
}

/// `TreeWalker` — pre-order walk filtered by `what_to_show` + an
/// optional user filter (we only support the bitmask + a Rust
/// callback for V1; JS-side NodeFilter wraps this).
pub struct TreeWalker<'a> {
    doc: &'a Document,
    root: NodeId,
    current: NodeId,
    what_to_show: WhatToShow,
    filter: Box<dyn Fn(&NodeKind) -> u32 + 'a>,
}

impl<'a> TreeWalker<'a> {
    pub fn new(doc: &'a Document, root: NodeId, what_to_show: u32) -> Self {
        Self {
            doc,
            root,
            current: root,
            what_to_show: WhatToShow(what_to_show),
            filter: Box::new(|_| accept::ACCEPT),
        }
    }

    pub fn with_filter<F>(mut self, f: F) -> Self
    where
        F: Fn(&NodeKind) -> u32 + 'a,
    {
        self.filter = Box::new(f);
        self
    }

    pub fn current(&self) -> NodeId {
        self.current
    }

    fn accept_node(&self, id: NodeId) -> u32 {
        let k = match self.doc.kind(id) {
            Some(k) => k,
            None => return accept::REJECT,
        };
        if !self.what_to_show.matches(k) {
            return accept::SKIP;
        }
        (self.filter)(k)
    }

    /// Move to next pre-order accepted node, return it.
    pub fn next_node(&mut self) -> Option<NodeId> {
        loop {
            let next = self.next_descendant(self.current).or_else(|| {
                let mut cur = self.current;
                loop {
                    if cur == self.root {
                        return None;
                    }
                    if let Some(s) = self.doc.next_sibling(cur) {
                        return Some(s);
                    }
                    let p = self.doc.parent(cur)?;
                    cur = p;
                }
            })?;
            self.current = next;
            match self.accept_node(next) {
                accept::ACCEPT => return Some(next),
                accept::REJECT => continue,
                _ => continue,
            }
        }
    }

    fn next_descendant(&self, id: NodeId) -> Option<NodeId> {
        self.doc.first_child(id)
    }

    pub fn previous_sibling(&mut self) -> Option<NodeId> {
        let s = self.doc.previous_sibling(self.current)?;
        self.current = s;
        Some(s)
    }
    pub fn next_sibling(&mut self) -> Option<NodeId> {
        let s = self.doc.next_sibling(self.current)?;
        self.current = s;
        Some(s)
    }
    pub fn parent_node(&mut self) -> Option<NodeId> {
        if self.current == self.root {
            return None;
        }
        let p = self.doc.parent(self.current)?;
        self.current = p;
        Some(p)
    }
    pub fn first_child(&mut self) -> Option<NodeId> {
        let c = self.doc.first_child(self.current)?;
        self.current = c;
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    #[test]
    fn walks_elements_only() {
        let mut doc = Document::new();
        let html = doc.create_element("html");
        let body = doc.create_element("body");
        let t = doc.create_text_node("ignored");
        let p = doc.create_element("p");
        doc.append_child(doc.root(), html).unwrap();
        doc.append_child(html, body).unwrap();
        doc.append_child(body, t).unwrap();
        doc.append_child(body, p).unwrap();

        let mut tw = TreeWalker::new(&doc, doc.root(), WhatToShow::ELEMENT);
        let mut seen = vec![tw.current()];
        while let Some(n) = tw.next_node() {
            seen.push(n);
        }
        // root is Document — doesn't match ELEMENT filter, but it's the
        // walker's starting position so we observe it once.
        assert!(seen.contains(&html));
        assert!(seen.contains(&body));
        assert!(seen.contains(&p));
        assert!(!seen.contains(&t));
    }

    #[test]
    fn parent_node_walks_up() {
        let mut doc = Document::new();
        let html = doc.create_element("html");
        let body = doc.create_element("body");
        doc.append_child(doc.root(), html).unwrap();
        doc.append_child(html, body).unwrap();
        let mut tw = TreeWalker::new(&doc, doc.root(), WhatToShow::ALL);
        let _ = tw.next_node(); // html
        let _ = tw.next_node(); // body
        assert_eq!(tw.parent_node(), Some(html));
    }
}
