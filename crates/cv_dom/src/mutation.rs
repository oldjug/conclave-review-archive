//! MutationObserver — real, spec-shaped implementation.
//!
//! Records emitted by `Document::set_attribute`, `append_child`,
//! `remove_child`, `replace_child`, `insert_before`, `set_text_content`
//! land in observer queues; the observer's callback consumes them when the
//! document performs a microtask checkpoint
//! ([`Document::notify_mutation_observers`]).
//!
//! Matches WHATWG DOM §4.3 "Interface MutationObserver":
//!   * "queue a mutation record" (§4.3.4) — a record is delivered to an
//!     observer only when the observer's registered node is the target OR
//!     (with `subtree`) an inclusive ancestor of the target, the relevant
//!     option is set, and (for attributes) the `attributeFilter` admits the
//!     attribute. `oldValue` is included only when `attributeOldValue` /
//!     `characterDataOldValue` was requested — mirroring Blink
//!     `MutationObserverRegistration::ShouldReceiveMutationFrom`
//!     (third_party/blink/renderer/core/dom/mutation_observer_registration.cc).
//!   * "notify mutation observers" (§4.3) drains each observer's queue and
//!     invokes its callback once with `(records, observer)`.
//!
//! The ancestor walk that decides "is this observer interested in `target`?"
//! lives in [`crate::Document::emit_mutation`] — only the document knows the
//! tree topology. This module owns the option-filtering + record buffering.

use crate::NodeId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationType {
    Attributes {
        name: String,
        namespace: Option<String>,
        old_value: Option<String>,
        new_value: Option<String>,
    },
    CharacterData {
        old_value: String,
        new_value: String,
    },
    ChildList {
        added: Vec<NodeId>,
        removed: Vec<NodeId>,
        previous_sibling: Option<NodeId>,
        next_sibling: Option<NodeId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationRecord {
    /// `type` per WHATWG: "attributes" / "characterData" / "childList".
    pub target: NodeId,
    pub kind: MutationType,
}

impl MutationRecord {
    /// WHATWG `type` string for this record.
    pub fn type_str(&self) -> &'static str {
        match &self.kind {
            MutationType::Attributes { .. } => "attributes",
            MutationType::CharacterData { .. } => "characterData",
            MutationType::ChildList { .. } => "childList",
        }
    }
}

/// Observer config — what to watch. Mirrors `MutationObserverInit`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObserverInit {
    pub child_list: bool,
    pub attributes: bool,
    pub character_data: bool,
    pub subtree: bool,
    pub attribute_old_value: bool,
    pub character_data_old_value: bool,
    pub attribute_filter: Option<Vec<String>>,
}

/// One registered (node, options) pair. WHATWG: a single MutationObserver may
/// `observe()` several nodes with different options; each is its own
/// registration.
#[derive(Debug, Clone)]
struct Registration {
    root: NodeId,
    init: ObserverInit,
}

#[derive(Debug, Default)]
pub struct MutationObserver {
    /// (node, options) registrations. WHATWG: re-`observe()`-ing the same node
    /// REPLACES its prior options (we dedupe by root below).
    registrations: Vec<Registration>,
    pending: Vec<MutationRecord>,
    /// Options baked at `new(init)` time so the single-arg [`Self::observe`]
    /// (used by call sites that pre-configure the observer) keeps working.
    default_init: ObserverInit,
}

impl MutationObserver {
    pub fn new(init: ObserverInit) -> Self {
        // A bare `new(init)` with options pre-baked keeps the old single-config
        // ergonomics: the first `observe(root)` adopts these options.
        Self {
            registrations: Vec::new(),
            pending: Vec::new(),
            default_init: init,
        }
    }

    /// Observe `root` with `init`. Re-observing a node replaces its options
    /// (WHATWG step 7: "remove all registered observers ... whose ... node is
    /// node"). The legacy single-arg [`Self::observe`] adopts `default_init`.
    pub fn observe_with(&mut self, root: NodeId, init: ObserverInit) {
        self.registrations.retain(|r| r.root != root);
        self.registrations.push(Registration { root, init });
    }

    /// Legacy single-arg observe — adopts the options passed to `new()`.
    pub fn observe(&mut self, root: NodeId) {
        let init = self.default_init.clone();
        self.observe_with(root, init);
    }

    pub fn disconnect(&mut self) {
        self.registrations.clear();
        self.pending.clear();
    }

    /// All distinct observed roots (used by [`crate::Document::emit_mutation`]).
    pub(crate) fn roots(&self) -> Vec<NodeId> {
        self.registrations.iter().map(|r| r.root).collect()
    }

    /// For a record whose target is reachable from `root` at `distance` edges
    /// (0 = the root itself), decide whether this observer's registration for
    /// `root` admits it and, if so, return the option-trimmed record to queue.
    ///
    /// `distance > 0` requires `subtree`. Option/filter gating + `oldValue`
    /// trimming happen here so the queued record is exactly what Blink would
    /// deliver.
    pub(crate) fn record_for(
        &self,
        root: NodeId,
        distance: usize,
        rec: &MutationRecord,
    ) -> Option<MutationRecord> {
        let reg = self.registrations.iter().find(|r| r.root == root)?;
        let init = &reg.init;
        // Subtree gating: a non-root target requires subtree:true.
        if distance > 0 && !init.subtree {
            return None;
        }
        match &rec.kind {
            MutationType::Attributes {
                name,
                namespace,
                old_value,
                new_value,
            } => {
                if !init.attributes {
                    return None;
                }
                if let Some(filter) = &init.attribute_filter {
                    if !filter.iter().any(|f| f.eq_ignore_ascii_case(name)) {
                        return None;
                    }
                }
                Some(MutationRecord {
                    target: rec.target,
                    kind: MutationType::Attributes {
                        name: name.clone(),
                        namespace: namespace.clone(),
                        // oldValue ONLY when requested (WHATWG step "if ...
                        // attributeOldValue is true": otherwise null).
                        old_value: if init.attribute_old_value {
                            old_value.clone()
                        } else {
                            None
                        },
                        new_value: new_value.clone(),
                    },
                })
            }
            MutationType::CharacterData {
                old_value,
                new_value,
            } => {
                if !init.character_data {
                    return None;
                }
                Some(MutationRecord {
                    target: rec.target,
                    kind: MutationType::CharacterData {
                        old_value: if init.character_data_old_value {
                            old_value.clone()
                        } else {
                            String::new()
                        },
                        new_value: new_value.clone(),
                    },
                })
            }
            MutationType::ChildList { .. } => {
                if !init.child_list {
                    return None;
                }
                Some(rec.clone())
            }
        }
    }

    /// Push a fully option-trimmed record onto this observer's queue. Callers
    /// (the document) have already done matching/trimming via [`Self::record_for`].
    pub(crate) fn enqueue(&mut self, rec: MutationRecord) {
        self.pending.push(rec);
    }

    /// WHATWG `takeRecords()`: synchronously drain the queue WITHOUT invoking
    /// the callback.
    pub fn take_records(&mut self) -> Vec<MutationRecord> {
        std::mem::take(&mut self.pending)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
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
        obs.borrow_mut().observe(doc.root());
        let div = doc.create_element("div");
        doc.append_child(doc.root(), div).unwrap();
        let recs = obs.borrow_mut().take_records();
        assert_eq!(recs.len(), 1);
        match &recs[0].kind {
            MutationType::ChildList { added, .. } => assert_eq!(added, &vec![div]),
            other => panic!("expected childList, got {other:?}"),
        }
    }

    #[test]
    fn attribute_old_value_gating() {
        let mut doc = Document::new();
        let div = doc.create_element("div");
        doc.append_child(doc.root(), div).unwrap();
        doc.set_attribute(div, "data-x", "first");

        // Observer WITHOUT attributeOldValue → oldValue is None.
        let mut init = ObserverInit::default();
        init.attributes = true;
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        obs.borrow_mut().observe(div);
        doc.set_attribute(div, "data-x", "second");
        let recs = obs.borrow_mut().take_records();
        assert_eq!(recs.len(), 1);
        match &recs[0].kind {
            MutationType::Attributes {
                name, old_value, ..
            } => {
                assert_eq!(name, "data-x");
                assert_eq!(*old_value, None, "no oldValue without attributeOldValue");
            }
            other => panic!("expected attributes, got {other:?}"),
        }

        // Observer WITH attributeOldValue → oldValue carries the prior value.
        let mut init2 = ObserverInit::default();
        init2.attributes = true;
        init2.attribute_old_value = true;
        let obs2 = make_obs(init2);
        doc.add_observer(obs2.clone());
        obs2.borrow_mut().observe(div);
        doc.set_attribute(div, "data-x", "third");
        let recs2 = obs2.borrow_mut().take_records();
        assert_eq!(recs2.len(), 1);
        match &recs2[0].kind {
            MutationType::Attributes { old_value, .. } => {
                assert_eq!(old_value.as_deref(), Some("second"));
            }
            other => panic!("expected attributes, got {other:?}"),
        }
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
        obs.borrow_mut().observe(div);
        doc.set_attribute(div, "id", "x"); // ignored
        doc.set_attribute(div, "class", "c"); // observed
        let recs = obs.borrow_mut().take_records();
        assert_eq!(recs.len(), 1);
        assert!(matches!(&recs[0].kind, MutationType::Attributes { name, .. } if name == "class"));
    }

    #[test]
    fn subtree_true_catches_grandchild() {
        let mut doc = Document::new();
        let parent = doc.create_element("div");
        doc.append_child(doc.root(), parent).unwrap();
        let mid = doc.create_element("section");
        doc.append_child(parent, mid).unwrap();

        let mut init = ObserverInit::default();
        init.child_list = true;
        init.subtree = true;
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        obs.borrow_mut().observe(parent);

        // Append a grandchild under `mid` (target = mid, a descendant of parent).
        let gc = doc.create_element("span");
        doc.append_child(mid, gc).unwrap();
        let recs = obs.borrow_mut().take_records();
        assert_eq!(recs.len(), 1, "subtree:true catches descendant mutation");
        assert_eq!(recs[0].target, mid);
    }

    #[test]
    fn subtree_false_ignores_grandchild() {
        let mut doc = Document::new();
        let parent = doc.create_element("div");
        doc.append_child(doc.root(), parent).unwrap();
        let mid = doc.create_element("section");
        doc.append_child(parent, mid).unwrap();

        let mut init = ObserverInit::default();
        init.child_list = true;
        init.subtree = false;
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        obs.borrow_mut().observe(parent);

        let gc = doc.create_element("span");
        doc.append_child(mid, gc).unwrap();
        assert_eq!(
            obs.borrow().pending_count(),
            0,
            "subtree:false must NOT see a grandchild mutation"
        );
    }

    #[test]
    fn character_data_old_value() {
        let mut doc = Document::new();
        let t = doc.create_text_node("hello");
        doc.append_child(doc.root(), t).unwrap();
        let mut init = ObserverInit::default();
        init.character_data = true;
        init.character_data_old_value = true;
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        obs.borrow_mut().observe(t);
        doc.set_text_data(t, "world").unwrap();
        let recs = obs.borrow_mut().take_records();
        assert_eq!(recs.len(), 1);
        match &recs[0].kind {
            MutationType::CharacterData {
                old_value,
                new_value,
            } => {
                assert_eq!(old_value, "hello");
                assert_eq!(new_value, "world");
            }
            other => panic!("expected characterData, got {other:?}"),
        }
    }

    #[test]
    fn take_records_drains_and_callback_then_silent() {
        let mut doc = Document::new();
        let mut init = ObserverInit::default();
        init.child_list = true;
        let fired: Rc<RefCell<usize>> = Rc::new(RefCell::new(0));
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        obs.borrow_mut().observe(doc.root());
        let div = doc.create_element("div");
        doc.append_child(doc.root(), div).unwrap();
        // takeRecords drains synchronously.
        let recs = obs.borrow_mut().take_records();
        assert_eq!(recs.len(), 1);
        assert_eq!(obs.borrow().pending_count(), 0, "queue empty after take");
        // A subsequent notify must NOT invoke the callback (nothing pending).
        let f = fired.clone();
        doc.notify_mutation_observers(|_records| {
            *f.borrow_mut() += 1;
        });
        assert_eq!(*fired.borrow(), 0, "callback not called when queue is empty");
    }

    #[test]
    fn disconnect_clears_pending() {
        let mut doc = Document::new();
        let mut init = ObserverInit::default();
        init.child_list = true;
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        obs.borrow_mut().observe(doc.root());
        let div = doc.create_element("div");
        doc.append_child(doc.root(), div).unwrap();
        obs.borrow_mut().disconnect();
        assert_eq!(obs.borrow().pending_count(), 0);
    }

    #[test]
    fn notify_invokes_callback_once_per_observer() {
        let mut doc = Document::new();
        let mut init = ObserverInit::default();
        init.child_list = true;
        let obs = make_obs(init);
        doc.add_observer(obs.clone());
        obs.borrow_mut().observe(doc.root());
        let a = doc.create_element("a");
        let b = doc.create_element("b");
        doc.append_child(doc.root(), a).unwrap();
        doc.append_child(doc.root(), b).unwrap();
        // Two appends → two records, ONE callback invocation with both.
        let calls: Rc<RefCell<Vec<usize>>> = Rc::new(RefCell::new(Vec::new()));
        let c = calls.clone();
        doc.notify_mutation_observers(move |records| {
            c.borrow_mut().push(records.len());
        });
        assert_eq!(*calls.borrow(), vec![2], "one call, both records");
        assert_eq!(obs.borrow().pending_count(), 0, "queue drained by notify");
    }
}
