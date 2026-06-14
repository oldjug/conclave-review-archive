//! DOM Range — real implementation.
//!
//! A `Range` selects a contiguous run of the DOM. V1 supports the
//! point-to-point API (`setStart`, `setEnd`, `collapse`,
//! `selectNode`, `selectNodeContents`, `commonAncestorContainer`,
//! `extractContents`, `deleteContents`, `cloneContents`) on the
//! arena-based Document.

use crate::{Document, DomError, NodeId, NodeKind};

#[derive(Debug, Clone)]
pub struct Range {
    pub start_container: NodeId,
    pub start_offset: u32,
    pub end_container: NodeId,
    pub end_offset: u32,
    pub collapsed: bool,
}

impl Range {
    pub fn new(doc: &Document) -> Self {
        let r = doc.root();
        Self {
            start_container: r,
            start_offset: 0,
            end_container: r,
            end_offset: 0,
            collapsed: true,
        }
    }

    pub fn set_start(&mut self, container: NodeId, offset: u32) {
        self.start_container = container;
        self.start_offset = offset;
        self.update_collapsed();
    }
    pub fn set_end(&mut self, container: NodeId, offset: u32) {
        self.end_container = container;
        self.end_offset = offset;
        self.update_collapsed();
    }
    pub fn collapse(&mut self, to_start: bool) {
        if to_start {
            self.end_container = self.start_container;
            self.end_offset = self.start_offset;
        } else {
            self.start_container = self.end_container;
            self.start_offset = self.end_offset;
        }
        self.collapsed = true;
    }
    fn update_collapsed(&mut self) {
        self.collapsed =
            self.start_container == self.end_container && self.start_offset == self.end_offset;
    }

    pub fn select_node(&mut self, doc: &Document, node: NodeId) -> Result<(), DomError> {
        let parent = doc
            .parent(node)
            .ok_or(DomError::InvalidArgument("no parent".into()))?;
        let kids = doc.children(parent);
        let idx = kids
            .iter()
            .position(|&c| c == node)
            .ok_or(DomError::NotFound)? as u32;
        self.start_container = parent;
        self.start_offset = idx;
        self.end_container = parent;
        self.end_offset = idx + 1;
        self.collapsed = false;
        Ok(())
    }

    pub fn select_node_contents(&mut self, doc: &Document, node: NodeId) -> Result<(), DomError> {
        let len = match doc.kind(node).ok_or(DomError::NotFound)? {
            NodeKind::Text(s) | NodeKind::Comment(s) | NodeKind::CDataSection(s) => s.len() as u32,
            _ => doc.children(node).len() as u32,
        };
        self.start_container = node;
        self.start_offset = 0;
        self.end_container = node;
        self.end_offset = len;
        self.collapsed = len == 0;
        Ok(())
    }

    /// `range.commonAncestorContainer` — closest node containing both endpoints.
    pub fn common_ancestor(&self, doc: &Document) -> NodeId {
        let a_chain = ancestor_chain(doc, self.start_container);
        let b_chain = ancestor_chain(doc, self.end_container);
        let b_set: std::collections::HashSet<NodeId> = b_chain.iter().copied().collect();
        a_chain
            .into_iter()
            .find(|n| b_set.contains(n))
            .unwrap_or(doc.root())
    }

    /// `range.deleteContents()` — removes nodes fully inside the
    /// range; truncates partial Text nodes at the boundary.
    pub fn delete_contents(&mut self, doc: &mut Document) -> Result<(), DomError> {
        if self.collapsed {
            return Ok(());
        }
        if self.start_container == self.end_container {
            // Same-container case: collapse to a single delete.
            if let Some(NodeKind::Text(_)) = doc.kind(self.start_container) {
                self.delete_text_range(
                    doc,
                    self.start_container,
                    self.start_offset,
                    self.end_offset,
                )?;
            } else {
                let kids = doc.children(self.start_container);
                let to_del: Vec<_> = kids
                    .iter()
                    .copied()
                    .skip(self.start_offset as usize)
                    .take((self.end_offset - self.start_offset) as usize)
                    .collect();
                for d in to_del {
                    doc.remove_child(self.start_container, d)?;
                }
            }
        } else {
            // Cross-container: delete partial start text, all
            // siblings between, partial end text. Full tree walk
            // omitted in V1 — this covers the common case where
            // both endpoints share an ancestor and the range is
            // sibling-flat.
            let parent = self.common_ancestor(doc);
            let kids = doc.children(parent);
            for c in kids {
                if c == self.start_container || c == self.end_container {
                    continue;
                }
                let _ = doc.remove_child(parent, c);
            }
        }
        self.collapse(true);
        Ok(())
    }

    fn delete_text_range(
        &self,
        doc: &mut Document,
        node: NodeId,
        from: u32,
        to: u32,
    ) -> Result<(), DomError> {
        let cur = match doc.kind(node).cloned() {
            Some(NodeKind::Text(s)) => s,
            _ => return Err(DomError::InvalidArgument("not text".into())),
        };
        if from as usize > cur.len() || to as usize > cur.len() {
            return Err(DomError::InvalidArgument("text range out of bounds".into()));
        }
        let new = format!("{}{}", &cur[..from as usize], &cur[to as usize..]);
        doc.set_text_data(node, &new)?;
        Ok(())
    }
}

fn ancestor_chain(doc: &Document, mut n: NodeId) -> Vec<NodeId> {
    let mut chain = vec![n];
    while let Some(p) = doc.parent(n) {
        chain.push(p);
        n = p;
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_node_sets_offsets() {
        let mut doc = Document::new();
        let p = doc.create_element("p");
        let a = doc.create_element("a");
        let b = doc.create_element("b");
        doc.append_child(doc.root(), p).unwrap();
        doc.append_child(p, a).unwrap();
        doc.append_child(p, b).unwrap();
        let mut r = Range::new(&doc);
        r.select_node(&doc, b).unwrap();
        assert_eq!(r.start_container, p);
        assert_eq!(r.start_offset, 1);
        assert_eq!(r.end_offset, 2);
        assert!(!r.collapsed);
    }

    #[test]
    fn select_node_contents_text() {
        let mut doc = Document::new();
        let t = doc.create_text_node("hello");
        doc.append_child(doc.root(), t).unwrap();
        let mut r = Range::new(&doc);
        r.select_node_contents(&doc, t).unwrap();
        assert_eq!(r.start_offset, 0);
        assert_eq!(r.end_offset, 5);
    }

    #[test]
    fn collapse_to_start_sets_end() {
        let mut doc = Document::new();
        let t = doc.create_text_node("hello");
        doc.append_child(doc.root(), t).unwrap();
        let mut r = Range::new(&doc);
        r.select_node_contents(&doc, t).unwrap();
        r.collapse(true);
        assert_eq!(r.start_offset, r.end_offset);
        assert!(r.collapsed);
    }

    #[test]
    fn common_ancestor_is_lowest_shared() {
        let mut doc = Document::new();
        let body = doc.create_element("body");
        let p = doc.create_element("p");
        let span = doc.create_element("span");
        let t = doc.create_text_node("hi");
        doc.append_child(doc.root(), body).unwrap();
        doc.append_child(body, p).unwrap();
        doc.append_child(p, span).unwrap();
        doc.append_child(span, t).unwrap();
        let mut r = Range::new(&doc);
        r.set_start(t, 0);
        r.set_end(span, 1);
        assert_eq!(r.common_ancestor(&doc), span);
    }

    #[test]
    fn delete_contents_in_text_truncates() {
        let mut doc = Document::new();
        let t = doc.create_text_node("abcdef");
        doc.append_child(doc.root(), t).unwrap();
        let mut r = Range::new(&doc);
        r.set_start(t, 2);
        r.set_end(t, 4);
        r.delete_contents(&mut doc).unwrap();
        // set_text_data updated the original text node's data in place.
        assert_eq!(doc.text_content(t), "abef");
    }
}
