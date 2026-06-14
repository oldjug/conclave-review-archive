//! MutationObserver — real implementation.
//!
//! Records emitted by `Document::set_attribute`, `append_child`,
//! `remove_child`, `replace_child`, `insert_before`, `set_text_content`
//! land in observer queues; the observer's callback consumes them
//! during the microtask checkpoint.

use crate::NodeId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationType {
    Attributes {
        name: String,
        old_value: Option<String>,
        new_value: Option<String>,
    },
    CharacterData {
        old_value: String,
        new_value: String,
    },
    ChildListAdded {
        added: Vec<NodeId>,
        removed: Vec<NodeId>,
        previous_sibling: Option<NodeId>,
        next_sibling: Option<NodeId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationRecord {
    pub target: NodeId,
    pub kind: MutationType,
}

/// Observer config — what to watch.
#[derive(Debug, Clone, Default)]
pub struct ObserverInit {
    pub child_list: bool,
    pub attributes: bool,
    pub character_data: bool,
    pub subtree: bool,
    pub attribute_old_value: bool,
    pub character_data_old_value: bool,
    pub attribute_filter: Option<Vec<String>>,
}

#[derive(Debug)]
pub struct MutationObserver {
    init: ObserverInit,
    pending: Vec<MutationRecord>,
    /// Roots the observer is registered against. Empty = match any
    /// target.
    roots: Vec<NodeId>,
}

impl MutationObserver {
    pub fn new(init: ObserverInit) -> Self {
        Self {
            init,
            pending: Vec::new(),
            roots: Vec::new(),
        }
    }

    pub fn observe(&mut self, root: NodeId) {
        if !self.roots.contains(&root) {
            self.roots.push(root);
        }
    }

    pub fn disconnect(&mut self) {
        self.roots.clear();
        self.pending.clear();
    }

    pub fn push(&mut self, rec: MutationRecord) {
        if !self.matches(&rec) {
            return;
        }
        self.pending.push(rec);
    }

    fn matches(&self, rec: &MutationRecord) -> bool {
        let kind_ok = match &rec.kind {
            MutationType::Attributes { name, .. } => {
                if !self.init.attributes {
                    return false;
                }
                match &self.init.attribute_filter {
                    Some(filter) => filter.iter().any(|f| f.eq_ignore_ascii_case(name)),
                    None => true,
                }
            }
            MutationType::CharacterData { .. } => self.init.character_data,
            MutationType::ChildListAdded { .. } => self.init.child_list,
        };
        if !kind_ok {
            return false;
        }
        // V1 root match is "any root" — full subtree-anchor check
        // requires walking parent ids which we don't have here.
        // Document::emit_mutation passes the record to every
        // observer; observer.roots can be left empty to opt in to
        // all of them.
        true
    }

    pub fn take_records(&mut self) -> Vec<MutationRecord> {
        std::mem::take(&mut self.pending)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn make_obs(init: ObserverInit) -> Rc<RefCell<MutationObserver>> {
        Rc::new(RefCell::new(MutationObserver::new(init)))
    }

    #[test]
    fn child_list_records_appends() {
        let mut doc = Document::new();
        let mut init = ObserverInit::default();
        init.child_list = true;
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        let div = doc.create_element("div");
        doc.append_child(doc.root(), div).unwrap();
        let recs = obs.borrow_mut().take_records();
        assert_eq!(recs.len(), 1);
        assert!(matches!(recs[0].kind, MutationType::ChildListAdded { .. }));
    }

    #[test]
    fn attribute_filter_narrows() {
        let mut doc = Document::new();
        let mut init = ObserverInit::default();
        init.attributes = true;
        init.attribute_filter = Some(vec!["class".into()]);
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        let div = doc.create_element("div");
        doc.append_child(doc.root(), div).unwrap();
        doc.set_attribute(div, "id", "x"); // ignored
        doc.set_attribute(div, "class", "c"); // observed
        let recs = obs.borrow_mut().take_records();
        assert_eq!(recs.len(), 1);
        assert!(matches!(&recs[0].kind, MutationType::Attributes { name, .. } if name == "class"));
    }

    #[test]
    fn disconnect_clears_pending() {
        let mut doc = Document::new();
        let mut init = ObserverInit::default();
        init.child_list = true;
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        let div = doc.create_element("div");
        doc.append_child(doc.root(), div).unwrap();
        obs.borrow_mut().disconnect();
        assert_eq!(obs.borrow().pending_count(), 0);
    }
}
