//! Mark-and-sweep garbage collector with cycle collection.
//!
//! Existing `cv_js::interp::Value` uses Rc<RefCell<…>> for object
//! storage, which leaks cycles (`a.x = b; b.x = a`). This module
//! introduces a generational arena `Heap` and a `Handle<T>` that is
//! a `u32` slot index — cycles between arena objects are reclaimable
//! because the mark phase walks the *root set* and only entries
//! reachable from a root survive a sweep.
//!
//! V1 is a stop-the-world collector — the interpreter calls
//! `Heap::collect(&roots)` between bytecode dispatches. A concurrent
//! incremental collector lands in a follow-up; the data structures
//! here are designed so the marker can become a tri-color reader
//! later (the per-object `mark` field is already there).
//!
//! Stand-alone: no dependency on `interp::Value` so the GC can be
//! tested in isolation. The interpreter wires its object pointers
//! through `Handle` once we land the second slice.

use std::collections::HashSet;

/// One arena slot. `mark` tracks the tri-color state; `outgoing` is
/// the slot indices this object references (filled in by the
/// concrete type's `Trace` impl when the interp wires its objects
/// through here).
#[derive(Debug, Clone)]
struct Slot<T> {
    payload: Option<T>,
    /// Slot indices reachable from this one. Populated by the trace
    /// callback the user passes to `collect`.
    outgoing: Vec<u32>,
    /// 0 = white (unmarked), 1 = grey (in worklist), 2 = black (live).
    mark: u8,
    /// Generation counter — bumped on free so stale handles can be
    /// detected if reused.
    generation: u32,
}

/// Opaque pointer into a `Heap`. The pair `(slot, generation)` is
/// validated on every deref so a freed-then-reused slot doesn't
/// alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle {
    slot: u32,
    generation: u32,
}

impl Handle {
    pub fn slot(self) -> u32 {
        self.slot
    }
}

/// The arena.
#[derive(Debug)]
pub struct Heap<T> {
    slots: Vec<Slot<T>>,
    free_list: Vec<u32>,
    live_count: usize,
}

impl<T> Default for Heap<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            live_count: 0,
        }
    }
}

impl<T> Heap<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn live_count(&self) -> usize {
        self.live_count
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Allocate. Reuses a freed slot when one is available.
    pub fn alloc(&mut self, payload: T) -> Handle {
        self.live_count += 1;
        if let Some(idx) = self.free_list.pop() {
            let s = &mut self.slots[idx as usize];
            s.payload = Some(payload);
            s.outgoing.clear();
            s.mark = 0;
            return Handle {
                slot: idx,
                generation: s.generation,
            };
        }
        let idx = self.slots.len() as u32;
        self.slots.push(Slot {
            payload: Some(payload),
            outgoing: Vec::new(),
            mark: 0,
            generation: 0,
        });
        Handle {
            slot: idx,
            generation: 0,
        }
    }

    /// Look up a handle. Returns `None` if the handle has been freed.
    pub fn get(&self, h: Handle) -> Option<&T> {
        let s = self.slots.get(h.slot as usize)?;
        if s.generation != h.generation {
            return None;
        }
        s.payload.as_ref()
    }

    pub fn get_mut(&mut self, h: Handle) -> Option<&mut T> {
        let s = self.slots.get_mut(h.slot as usize)?;
        if s.generation != h.generation {
            return None;
        }
        s.payload.as_mut()
    }

    /// Set the outgoing-edges list on a slot. The trace callback in
    /// `collect` is supposed to do this for live objects — but the
    /// interpreter can also call it directly when it knows the edges
    /// at allocation time.
    pub fn set_outgoing(&mut self, h: Handle, edges: Vec<u32>) {
        if let Some(s) = self.slots.get_mut(h.slot as usize) {
            if s.generation == h.generation {
                s.outgoing = edges;
            }
        }
    }

    /// Run a stop-the-world mark-and-sweep collection.
    ///
    /// `roots` is the live set the program is currently using —
    /// stack handles, register banks, persisted globals.
    ///
    /// `trace` is called for each grey object so the caller can
    /// recompute its outgoing edges. If your object's outgoing set
    /// doesn't change after allocation, `set_outgoing` once and pass
    /// a no-op trace.
    ///
    /// Returns the number of slots reclaimed.
    pub fn collect<F>(&mut self, roots: &[Handle], mut trace: F) -> usize
    where
        F: FnMut(&T) -> Vec<u32>,
    {
        // White everyone.
        for s in self.slots.iter_mut() {
            s.mark = 0;
        }
        // Push roots → grey.
        let mut worklist: Vec<u32> = Vec::with_capacity(roots.len());
        let mut seen: HashSet<u32> = HashSet::with_capacity(roots.len());
        for &r in roots {
            if let Some(s) = self.slots.get_mut(r.slot as usize) {
                if s.generation == r.generation && s.payload.is_some() && seen.insert(r.slot) {
                    s.mark = 1;
                    worklist.push(r.slot);
                }
            }
        }
        // Drain worklist.
        while let Some(idx) = worklist.pop() {
            let outgoing = {
                let s = &mut self.slots[idx as usize];
                if s.mark == 2 {
                    continue;
                }
                s.mark = 2;
                // Refresh outgoing edges from the payload via trace.
                let edges = trace(s.payload.as_ref().expect("black with no payload"));
                s.outgoing = edges;
                s.outgoing.clone()
            };
            for o in outgoing {
                if let Some(child) = self.slots.get_mut(o as usize) {
                    if child.payload.is_some() && child.mark == 0 {
                        child.mark = 1;
                        worklist.push(o);
                    }
                }
            }
        }
        // Sweep.
        let mut reclaimed = 0;
        for (i, s) in self.slots.iter_mut().enumerate() {
            if s.payload.is_some() && s.mark == 0 {
                s.payload = None;
                s.outgoing.clear();
                s.generation = s.generation.wrapping_add(1);
                self.free_list.push(i as u32);
                reclaimed += 1;
            }
        }
        self.live_count -= reclaimed;
        reclaimed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test payload: a tagged object with a name and a vector of
    /// outgoing handle slots that will be returned by trace.
    #[derive(Debug)]
    struct Obj {
        name: &'static str,
        edges: Vec<u32>,
    }

    fn trace(o: &Obj) -> Vec<u32> {
        o.edges.clone()
    }

    #[test]
    fn alloc_increments_live_count() {
        let mut h: Heap<Obj> = Heap::new();
        let _a = h.alloc(Obj {
            name: "a",
            edges: vec![],
        });
        let _b = h.alloc(Obj {
            name: "b",
            edges: vec![],
        });
        assert_eq!(h.live_count(), 2);
    }

    #[test]
    fn get_returns_payload() {
        let mut h: Heap<Obj> = Heap::new();
        let a = h.alloc(Obj {
            name: "x",
            edges: vec![],
        });
        assert_eq!(h.get(a).unwrap().name, "x");
    }

    #[test]
    fn collect_reclaims_unreached_objects() {
        let mut h: Heap<Obj> = Heap::new();
        let _a = h.alloc(Obj {
            name: "a",
            edges: vec![],
        });
        let _b = h.alloc(Obj {
            name: "b",
            edges: vec![],
        });
        assert_eq!(h.live_count(), 2);
        // No roots → everything goes.
        let reclaimed = h.collect(&[], trace);
        assert_eq!(reclaimed, 2);
        assert_eq!(h.live_count(), 0);
    }

    #[test]
    fn collect_keeps_reachable_via_root() {
        let mut h: Heap<Obj> = Heap::new();
        let a = h.alloc(Obj {
            name: "a",
            edges: vec![],
        });
        let _b = h.alloc(Obj {
            name: "b",
            edges: vec![],
        });
        let reclaimed = h.collect(&[a], trace);
        assert_eq!(reclaimed, 1); // b dies, a survives
        assert!(h.get(a).is_some());
    }

    #[test]
    fn collect_breaks_cycles() {
        // The whole reason for a real GC: refcount can't free a↔b.
        let mut h: Heap<Obj> = Heap::new();
        let a = h.alloc(Obj {
            name: "a",
            edges: vec![],
        });
        let b = h.alloc(Obj {
            name: "b",
            edges: vec![],
        });
        // a → b → a cycle.
        h.get_mut(a).unwrap().edges = vec![b.slot()];
        h.get_mut(b).unwrap().edges = vec![a.slot()];
        // No roots → both must die.
        let reclaimed = h.collect(&[], trace);
        assert_eq!(reclaimed, 2);
        assert_eq!(h.live_count(), 0);
    }

    #[test]
    fn collect_traces_transitively() {
        let mut h: Heap<Obj> = Heap::new();
        let a = h.alloc(Obj {
            name: "a",
            edges: vec![],
        });
        let b = h.alloc(Obj {
            name: "b",
            edges: vec![],
        });
        let c = h.alloc(Obj {
            name: "c",
            edges: vec![],
        });
        h.get_mut(a).unwrap().edges = vec![b.slot()];
        h.get_mut(b).unwrap().edges = vec![c.slot()];
        let reclaimed = h.collect(&[a], trace);
        assert_eq!(reclaimed, 0); // a→b→c all reachable
    }

    #[test]
    fn freed_handle_reads_as_none_then_reused() {
        let mut h: Heap<Obj> = Heap::new();
        let a = h.alloc(Obj {
            name: "a",
            edges: vec![],
        });
        h.collect(&[], trace);
        assert!(h.get(a).is_none()); // freed
        // Reuse the slot.
        let b = h.alloc(Obj {
            name: "b",
            edges: vec![],
        });
        assert_eq!(a.slot(), b.slot()); // same slot
        assert!(h.get(a).is_none()); // stale handle still invalid
        assert!(h.get(b).is_some());
    }
}
