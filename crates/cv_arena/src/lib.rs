//! `cv_arena` — the single allocation discipline for the engine.
//!
//! Milestone 1.1 of the master design (`MASTER_DESIGN_BEAT_CHROME.md`). Stable,
//! generational identity is the dependency cited across every skip-not-redo
//! subsystem (incremental render, fragment cache, paint-chunk keying, retained
//! DOM, JS object storage). Two primitives:
//!
//! - [`Slab<T>`] — a generational slab allocator. `insert` returns a `Handle`
//!   (an `index, generation` pair, 8 bytes, `Copy`). A handle to a removed slot
//!   does NOT alias the next occupant — the generation makes a stale handle
//!   detectably invalid (`get` returns `None`). No dangling indices, no ABA.
//!   This is the data-oriented replacement for pointer-rich object graphs:
//!   contiguous storage, `Handle` references instead of pointers.
//! - [`Arena<T>`] — a typed bump arena for transient per-phase data that is
//!   freed wholesale (`reset` keeps capacity). No generations, no per-item
//!   removal — cheaper than a `Slab` when the whole phase is dropped at once.
//!
//! Pure safe Rust, std only (no third-party crates, per workspace policy).

#![forbid(unsafe_code)]

use core::marker::PhantomData;
use core::num::NonZeroU32;

/// A stable, generational reference into a [`Slab<T>`]. `Copy`, 8 bytes, typed
/// by `T` so handles into different slabs cannot be confused. The `fn() -> T`
/// marker keeps the handle `Send`/`Sync` and variance correct without borrowing
/// `T`'s auto-traits.
pub struct Handle<T> {
    index: u32,
    generation: NonZeroU32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Handle<T> {}
impl<T> core::hash::Hash for Handle<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}
impl<T> core::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Handle({}, gen {})", self.index, self.generation.get())
    }
}

impl<T> Handle<T> {
    /// The slot index this handle points at.
    pub fn index(self) -> u32 {
        self.index
    }
    /// The generation stamp (≥1) that must still match the slot to be valid.
    pub fn generation(self) -> u32 {
        self.generation.get()
    }
    /// Pack into a `u64` (`index << 32 | generation`) for storage in `SoA` columns
    /// (e.g. the DOM wrapper-identity slot). Never zero, so `0` is a free niche.
    pub fn to_bits(self) -> u64 {
        (u64::from(self.index) << 32) | u64::from(self.generation.get())
    }
    /// Inverse of [`Self::to_bits`]. `None` if the low 32 bits (generation) are 0.
    pub fn from_bits(bits: u64) -> Option<Self> {
        let generation = NonZeroU32::new((bits & 0xFFFF_FFFF) as u32)?;
        Some(Self {
            index: (bits >> 32) as u32,
            generation,
            _marker: PhantomData,
        })
    }
}

enum Slot<T> {
    Occupied {
        generation: NonZeroU32,
        value: T,
    },
    Vacant {
        /// Generation the NEXT occupant of this slot will receive (already
        /// bumped past the last occupant, so old handles never re-match).
        generation: NonZeroU32,
        /// Next slot in the free list, or `None` if this is the tail.
        next_free: Option<u32>,
    },
}

/// A generational slab allocator: contiguous storage with stable, ABA-safe
/// `Handle` references. Reuses freed slots; a stale handle to a reused slot is
/// rejected by the generation check.
pub struct Slab<T> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
    len: usize,
}

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> core::fmt::Debug for Slab<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Slab")
            .field("len", &self.len)
            .field("capacity", &self.slots.len())
            .finish_non_exhaustive()
    }
}

impl<T> Slab<T> {
    /// An empty slab.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
        }
    }

    /// An empty slab with room for `capacity` slots before reallocating.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            len: 0,
        }
    }

    /// Number of live (occupied) slots.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no live slots.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total allocated slots (live + vacant). Memory footprint, not live count.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Insert a value, returning a stable handle. Reuses a freed slot if one is
    /// available (the new occupant gets that slot's bumped generation), else
    /// appends a fresh slot at generation 1.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        if let Some(index) = self.free_head {
            let slot = &mut self.slots[index as usize];
            let (generation, next_free) = match slot {
                Slot::Vacant {
                    generation,
                    next_free,
                } => (*generation, *next_free),
                Slot::Occupied { .. } => unreachable!("free list pointed at an occupied slot"),
            };
            *slot = Slot::Occupied { generation, value };
            self.free_head = next_free;
            self.len += 1;
            Handle {
                index,
                generation,
                _marker: PhantomData,
            }
        } else {
            let index = u32::try_from(self.slots.len())
                .expect("cv_arena::Slab exceeded 2^32 slots (>4 billion live objects)");
            let generation = NonZeroU32::MIN; // == 1
            self.slots.push(Slot::Occupied { generation, value });
            self.len += 1;
            Handle {
                index,
                generation,
                _marker: PhantomData,
            }
        }
    }

    /// Borrow the value behind a handle, or `None` if the handle is stale
    /// (the slot was removed/reused) or out of range.
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        match self.slots.get(handle.index as usize) {
            Some(Slot::Occupied { generation, value }) if *generation == handle.generation => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Mutably borrow the value behind a handle, or `None` if stale/out of range.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        match self.slots.get_mut(handle.index as usize) {
            Some(Slot::Occupied { generation, value }) if *generation == handle.generation => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Whether the handle currently refers to a live value.
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Remove and return the value behind a handle, freeing the slot for reuse.
    /// `None` if the handle is stale/out of range (idempotent double-remove).
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        let current = match slot {
            Slot::Occupied { generation, .. } if *generation == handle.generation => *generation,
            _ => return None,
        };
        // Bump the generation for the next occupant. On overflow (2^32 reuses of
        // one slot — astronomically unlikely) retire the slot rather than risk a
        // generation wraparound that could alias a stale handle.
        let next_gen = current.get().checked_add(1).and_then(NonZeroU32::new);
        let (new_slot, reusable) = match next_gen {
            Some(generation) => (
                Slot::Vacant {
                    generation,
                    next_free: self.free_head,
                },
                true,
            ),
            None => (
                Slot::Vacant {
                    generation: current,
                    next_free: None,
                },
                false,
            ),
        };
        let old = core::mem::replace(slot, new_slot);
        if reusable {
            self.free_head = Some(handle.index);
        }
        self.len -= 1;
        match old {
            Slot::Occupied { value, .. } => Some(value),
            Slot::Vacant { .. } => unreachable!("matched occupied above"),
        }
    }

    /// Drop all values and reset to empty, keeping allocated capacity.
    ///
    /// NOTE: handles obtained before `clear` MUST NOT be used afterward — clear
    /// resets generations, so a pre-clear handle could collide with a new
    /// occupant. Use this only at a hard lifetime boundary (e.g. navigation),
    /// never to free individual entries — that is [`Self::remove`].
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = None;
        self.len = 0;
    }

    /// Iterate live `(handle, &value)` pairs in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { generation, value } => Some((
                    Handle {
                        index: i as u32,
                        generation: *generation,
                        _marker: PhantomData,
                    },
                    value,
                )),
                Slot::Vacant { .. } => None,
            })
    }

    /// Iterate `&mut` references to every live value in slot order (handles not
    /// surfaced — borrow rules forbid a handle alongside the `&mut`). For bulk
    /// in-place updates of all occupants.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut().filter_map(|slot| match slot {
            Slot::Occupied { value, .. } => Some(value),
            Slot::Vacant { .. } => None,
        })
    }
}

/// A typed bump arena for transient, phase-scoped data freed all at once.
/// Items are addressed by a plain `u32` index (no generation — the whole arena
/// is dropped/reset between phases, so stale-handle safety isn't needed and we
/// pay nothing for it). `reset` keeps capacity, so steady-state reuse is
/// allocation-free.
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> core::fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Arena")
            .field("len", &self.items.len())
            .field("capacity", &self.items.capacity())
            .finish_non_exhaustive()
    }
}

impl<T> Arena<T> {
    /// An empty arena.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// An empty arena with preallocated room for `capacity` items.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    /// Allocate a value, returning its index.
    pub fn alloc(&mut self, value: T) -> u32 {
        let index = u32::try_from(self.items.len())
            .expect("cv_arena::Arena exceeded 2^32 items in one phase");
        self.items.push(value);
        index
    }

    /// Borrow an item by index.
    pub fn get(&self, index: u32) -> Option<&T> {
        self.items.get(index as usize)
    }

    /// Mutably borrow an item by index.
    pub fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        self.items.get_mut(index as usize)
    }

    /// Number of allocated items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Drop all items but keep capacity — the bump-reset that makes steady-state
    /// per-phase use allocation-free.
    pub fn reset(&mut self) {
        self.items.clear();
    }

    /// Iterate all items in allocation order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut slab: Slab<&str> = Slab::new();
        let a = slab.insert("a");
        let b = slab.insert("b");
        assert_eq!(slab.len(), 2);
        assert_eq!(slab.get(a), Some(&"a"));
        assert_eq!(slab.get(b), Some(&"b"));
        assert_eq!(slab.remove(a), Some("a"));
        assert_eq!(slab.len(), 1);
        assert_eq!(slab.get(a), None, "removed handle is stale");
        assert_eq!(slab.remove(a), None, "double-remove is a no-op");
        assert_eq!(slab.get(b), Some(&"b"));
    }

    #[test]
    fn stale_handle_does_not_alias_reused_slot() {
        let mut slab: Slab<u32> = Slab::new();
        let a = slab.insert(10);
        let idx = a.index();
        assert_eq!(slab.remove(a), Some(10));
        let b = slab.insert(20); // reuses the same slot index, bumped generation
        assert_eq!(b.index(), idx, "freed slot is reused");
        assert_ne!(a.generation(), b.generation(), "generation bumped on reuse");
        assert_eq!(slab.get(a), None, "old handle must NOT see the new occupant");
        assert_eq!(slab.get(b), Some(&20));
    }

    #[test]
    fn freelist_lifo_reuse() {
        let mut slab: Slab<u32> = Slab::new();
        let a = slab.insert(1);
        let b = slab.insert(2);
        let c = slab.insert(3);
        assert_eq!(slab.capacity(), 3);
        slab.remove(b);
        slab.remove(a);
        // Two slots free; two inserts reuse them, no growth.
        slab.insert(4);
        slab.insert(5);
        assert_eq!(slab.capacity(), 3, "freed slots reused, no new allocation");
        assert_eq!(slab.get(c), Some(&3), "untouched handle still valid");
    }

    #[test]
    fn iter_visits_only_live() {
        let mut slab: Slab<u32> = Slab::new();
        let a = slab.insert(1);
        let _b = slab.insert(2);
        let c = slab.insert(3);
        slab.remove(a);
        slab.remove(c);
        let live: Vec<u32> = slab.iter().map(|(_, v)| *v).collect();
        assert_eq!(live, vec![2]);
    }

    #[test]
    fn handle_bits_roundtrip() {
        let mut slab: Slab<u8> = Slab::new();
        let h = slab.insert(7);
        let bits = h.to_bits();
        assert_ne!(bits, 0);
        assert_eq!(Handle::<u8>::from_bits(bits), Some(h));
        assert_eq!(Handle::<u8>::from_bits(bits & 0xFFFF_FFFF_0000_0000), None, "zero generation is invalid");
    }

    #[test]
    fn clear_resets() {
        let mut slab: Slab<u32> = Slab::new();
        slab.insert(1);
        slab.insert(2);
        slab.clear();
        assert!(slab.is_empty());
        assert_eq!(slab.capacity(), 0);
        let h = slab.insert(9);
        assert_eq!(slab.get(h), Some(&9));
    }

    #[test]
    fn arena_alloc_get_reset() {
        let mut arena: Arena<u32> = Arena::new();
        let i = arena.alloc(100);
        let j = arena.alloc(200);
        assert_eq!(arena.get(i), Some(&100));
        assert_eq!(arena.get(j), Some(&200));
        *arena.get_mut(i).unwrap() = 101;
        assert_eq!(arena.get(i), Some(&101));
        assert_eq!(arena.len(), 2);
        arena.reset();
        assert!(arena.is_empty());
        // After reset, indexing restarts from 0 with capacity retained.
        assert_eq!(arena.alloc(1), 0);
    }
}
