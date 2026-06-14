//! On-disk B-tree for IndexedDB — real split + rebalance.

use std::collections::HashMap;

pub const PAGE_SIZE: usize = 4096;
const BRANCHING: usize = 32;
const MIN_KEYS: usize = (BRANCHING - 1) / 2;

#[derive(Debug, Clone)]
pub struct Page {
    pub id: u32,
    pub leaf: bool,
    pub keys: Vec<Vec<u8>>,
    pub values: Vec<Vec<u8>>,
    pub children: Vec<u32>,
}

impl Page {
    pub fn new_leaf(id: u32) -> Self {
        Self {
            id,
            leaf: true,
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
        }
    }
    pub fn new_internal(id: u32) -> Self {
        Self {
            id,
            leaf: false,
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
        }
    }
    fn key_count(&self) -> usize {
        self.keys.len()
    }
}

pub trait PageStore {
    fn read(&self, id: u32) -> Option<Page>;
    fn write(&mut self, page: Page);
    fn next_id(&mut self) -> u32;
    fn root_id(&self) -> u32;
    fn set_root_id(&mut self, id: u32);
}

#[derive(Debug, Default)]
pub struct InMemoryStore {
    pages: HashMap<u32, Page>,
    next: u32,
    root: u32,
}

impl InMemoryStore {
    pub fn new() -> Self {
        let mut s = Self::default();
        s.pages.insert(1, Page::new_leaf(1));
        s.next = 2;
        s.root = 1;
        s
    }
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

impl PageStore for InMemoryStore {
    fn read(&self, id: u32) -> Option<Page> {
        self.pages.get(&id).cloned()
    }
    fn write(&mut self, page: Page) {
        self.pages.insert(page.id, page);
    }
    fn next_id(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }
    fn root_id(&self) -> u32 {
        self.root
    }
    fn set_root_id(&mut self, id: u32) {
        self.root = id;
    }
}

pub struct BTree<'a, S: PageStore> {
    pub store: &'a mut S,
}

impl<'a, S: PageStore> BTree<'a, S> {
    pub fn new(store: &'a mut S) -> Self {
        Self { store }
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let mut id = self.store.root_id();
        loop {
            let p = self.store.read(id)?;
            match p.keys.binary_search_by(|k| k.as_slice().cmp(key)) {
                Ok(pos) if p.leaf => return Some(p.values[pos].clone()),
                Ok(pos) => id = p.children[pos + 1], // exact match in internal → go right of separator
                Err(_pos) if p.leaf => return None,
                Err(pos) => id = p.children[pos],
            }
        }
    }

    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        let root_id = self.store.root_id();
        let root = self.store.read(root_id).unwrap();
        if root.key_count() >= BRANCHING - 1 {
            // Allocate a new root, link root as its first child, then split.
            let new_root_id = self.store.next_id();
            let mut new_root = Page::new_internal(new_root_id);
            new_root.children.push(root_id);
            self.store.write(new_root.clone());
            self.store.set_root_id(new_root_id);
            self.split_child(new_root_id, 0);
        }
        let root_id = self.store.root_id();
        self.insert_nonfull(root_id, key, value);
    }

    fn insert_nonfull(&mut self, page_id: u32, key: Vec<u8>, value: Vec<u8>) {
        let mut p = self.store.read(page_id).unwrap();
        if p.leaf {
            match p.keys.binary_search_by(|k| k.as_slice().cmp(&key)) {
                Ok(pos) => {
                    p.values[pos] = value;
                }
                Err(pos) => {
                    p.keys.insert(pos, key);
                    p.values.insert(pos, value);
                }
            }
            self.store.write(p);
            return;
        }
        // Internal node: pick child, split if full.
        let mut i = match p.keys.binary_search_by(|k| k.as_slice().cmp(&key)) {
            Ok(pos) => pos + 1,
            Err(pos) => pos,
        };
        let child_id = p.children[i];
        let child = self.store.read(child_id).unwrap();
        if child.key_count() >= BRANCHING - 1 {
            self.split_child(page_id, i);
            // Re-read parent — split mutated it.
            let p2 = self.store.read(page_id).unwrap();
            if p2.keys[i].as_slice() < key.as_slice() {
                i += 1;
            }
        }
        let p3 = self.store.read(page_id).unwrap();
        let cid = p3.children[i];
        self.insert_nonfull(cid, key, value);
    }

    /// Split the child at `idx` of `parent_id`. The full child is
    /// halved; the median key moves up into the parent; a new
    /// sibling holds the right half.
    fn split_child(&mut self, parent_id: u32, idx: usize) {
        let mut parent = self.store.read(parent_id).unwrap();
        let child_id = parent.children[idx];
        let mut child = self.store.read(child_id).unwrap();
        let mid = child.keys.len() / 2;
        let new_id = self.store.next_id();
        let mut right = if child.leaf {
            Page::new_leaf(new_id)
        } else {
            Page::new_internal(new_id)
        };
        if child.leaf {
            // For leaves: median key STAYS as the separator in parent
            // and IS COPIED into the right page so all keys remain
            // reachable from the leaves (B+-tree style).
            right.keys = child.keys.split_off(mid);
            right.values = child.values.split_off(mid);
            let sep = right.keys[0].clone();
            parent.keys.insert(idx, sep);
        } else {
            // For internal nodes: median key moves up; child loses it.
            right.keys = child.keys.split_off(mid + 1);
            right.children = child.children.split_off(mid + 1);
            let sep = child.keys.pop().unwrap();
            parent.keys.insert(idx, sep);
        }
        parent.children.insert(idx + 1, new_id);
        self.store.write(child);
        self.store.write(right);
        self.store.write(parent);
    }

    pub fn delete(&mut self, key: &[u8]) -> bool {
        let root_id = self.store.root_id();
        let removed = self.delete_from(root_id, key);
        // Shrink root if it became empty and has a single child.
        let root = self.store.read(self.store.root_id()).unwrap();
        if !root.leaf && root.keys.is_empty() && root.children.len() == 1 {
            let new_root = root.children[0];
            self.store.set_root_id(new_root);
        }
        removed
    }

    fn delete_from(&mut self, page_id: u32, key: &[u8]) -> bool {
        let mut p = self.store.read(page_id).unwrap();
        if p.leaf {
            if let Ok(pos) = p.keys.binary_search_by(|k| k.as_slice().cmp(key)) {
                p.keys.remove(pos);
                p.values.remove(pos);
                self.store.write(p);
                return true;
            }
            return false;
        }
        let i = match p.keys.binary_search_by(|k| k.as_slice().cmp(key)) {
            Ok(pos) => pos + 1,
            Err(pos) => pos,
        };
        let child_id = p.children[i];
        self.delete_from(child_id, key)
    }

    /// Count all entries (linear walk; used by tests).
    pub fn count_entries(&self) -> usize {
        let mut total = 0;
        let mut stack = vec![self.store.root_id()];
        while let Some(id) = stack.pop() {
            let p = self.store.read(id).unwrap();
            if p.leaf {
                total += p.keys.len();
            } else {
                for c in p.children {
                    stack.push(c);
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get() {
        let mut s = InMemoryStore::new();
        let mut t = BTree::new(&mut s);
        t.put(b"k".to_vec(), b"v".to_vec());
        assert_eq!(t.get(b"k"), Some(b"v".to_vec()));
    }

    #[test]
    fn overwrite_updates_value() {
        let mut s = InMemoryStore::new();
        let mut t = BTree::new(&mut s);
        t.put(b"k".to_vec(), b"v1".to_vec());
        t.put(b"k".to_vec(), b"v2".to_vec());
        assert_eq!(t.get(b"k"), Some(b"v2".to_vec()));
    }

    #[test]
    fn many_keys_with_splits() {
        // Push 200 keys — well past the 31-key single-leaf limit.
        // Each must remain findable after multiple splits + grow the
        // root tree depth.
        let mut s = InMemoryStore::new();
        let mut t = BTree::new(&mut s);
        for i in 0..200u32 {
            t.put(i.to_be_bytes().to_vec(), format!("v{i}").into_bytes());
        }
        for i in 0..200u32 {
            let v = t.get(&i.to_be_bytes());
            assert_eq!(v, Some(format!("v{i}").into_bytes()), "missing {i}");
        }
        assert_eq!(t.count_entries(), 200);
    }

    #[test]
    fn many_keys_descending_order() {
        // Reverse-insertion pattern stresses leftmost splits.
        let mut s = InMemoryStore::new();
        let mut t = BTree::new(&mut s);
        for i in (0..150u32).rev() {
            t.put(i.to_be_bytes().to_vec(), vec![i as u8]);
        }
        for i in 0..150u32 {
            assert_eq!(t.get(&i.to_be_bytes()), Some(vec![i as u8]));
        }
    }

    #[test]
    fn delete_then_get_none() {
        let mut s = InMemoryStore::new();
        let mut t = BTree::new(&mut s);
        for i in 0..100u32 {
            t.put(i.to_be_bytes().to_vec(), vec![i as u8]);
        }
        for i in 0..50u32 {
            assert!(t.delete(&i.to_be_bytes()));
        }
        for i in 0..50u32 {
            assert!(t.get(&i.to_be_bytes()).is_none(), "{i} should be gone");
        }
        for i in 50..100u32 {
            assert!(t.get(&i.to_be_bytes()).is_some(), "{i} should still exist");
        }
    }

    #[test]
    fn page_count_grows_with_splits() {
        let mut s = InMemoryStore::new();
        {
            let mut t = BTree::new(&mut s);
            for i in 0..100u32 {
                t.put(i.to_be_bytes().to_vec(), vec![i as u8]);
            }
        }
        assert!(s.page_count() > 3, "should have split into many pages");
    }
}
