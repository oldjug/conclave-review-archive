//! SharedArrayBuffer + Atomics — V1 single-threaded model.
//!
//! Real cross-thread sharing routes through `cv_base` thread pool
//! once we land Worker scheduling; for now Atomics are correct on a
//! single thread and the API is structurally identical, so JS code
//! using SAB compiles + runs.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

#[derive(Debug, Clone)]
pub struct SharedArrayBuffer {
    inner: Arc<Vec<AtomicI32>>, // i32 view; other widths are reinterpretations
    byte_length: usize,
}

impl SharedArrayBuffer {
    pub fn new(byte_length: usize) -> Self {
        let n_i32 = (byte_length + 3) / 4;
        let mut v = Vec::with_capacity(n_i32);
        for _ in 0..n_i32 {
            v.push(AtomicI32::new(0));
        }
        Self {
            inner: Arc::new(v),
            byte_length,
        }
    }
    pub fn byte_length(&self) -> usize {
        self.byte_length
    }
}

pub struct AtomicsView {
    sab: SharedArrayBuffer,
}

impl std::fmt::Debug for AtomicsView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicsView")
            .field("byte_length", &self.sab.byte_length)
            .finish()
    }
}

impl AtomicsView {
    pub fn new(sab: SharedArrayBuffer) -> Self {
        Self { sab }
    }
    pub fn load(&self, index: usize) -> i32 {
        self.sab.inner[index].load(Ordering::SeqCst)
    }
    pub fn store(&self, index: usize, v: i32) {
        self.sab.inner[index].store(v, Ordering::SeqCst);
    }
    pub fn add(&self, index: usize, v: i32) -> i32 {
        self.sab.inner[index].fetch_add(v, Ordering::SeqCst)
    }
    pub fn sub(&self, index: usize, v: i32) -> i32 {
        self.sab.inner[index].fetch_sub(v, Ordering::SeqCst)
    }
    pub fn or(&self, index: usize, v: i32) -> i32 {
        self.sab.inner[index].fetch_or(v, Ordering::SeqCst)
    }
    pub fn and(&self, index: usize, v: i32) -> i32 {
        self.sab.inner[index].fetch_and(v, Ordering::SeqCst)
    }
    pub fn xor(&self, index: usize, v: i32) -> i32 {
        self.sab.inner[index].fetch_xor(v, Ordering::SeqCst)
    }
    pub fn exchange(&self, index: usize, v: i32) -> i32 {
        self.sab.inner[index].swap(v, Ordering::SeqCst)
    }
    pub fn compare_exchange(&self, index: usize, expected: i32, new_value: i32) -> i32 {
        match self.sab.inner[index].compare_exchange(
            expected,
            new_value,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(v) | Err(v) => v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sab_zeroes_initial_contents() {
        let sab = SharedArrayBuffer::new(16);
        let view = AtomicsView::new(sab);
        for i in 0..4 {
            assert_eq!(view.load(i), 0);
        }
    }

    #[test]
    fn add_returns_previous_value() {
        let sab = SharedArrayBuffer::new(8);
        let view = AtomicsView::new(sab);
        view.store(0, 10);
        let prev = view.add(0, 5);
        assert_eq!(prev, 10);
        assert_eq!(view.load(0), 15);
    }

    #[test]
    fn bitwise_ops_round_trip() {
        let sab = SharedArrayBuffer::new(8);
        let view = AtomicsView::new(sab);
        view.store(0, 0b1010);
        view.or(0, 0b0101);
        assert_eq!(view.load(0), 0b1111);
        view.and(0, 0b1100);
        assert_eq!(view.load(0), 0b1100);
        view.xor(0, 0b1111);
        assert_eq!(view.load(0), 0b0011);
    }

    #[test]
    fn exchange_swaps_value() {
        let sab = SharedArrayBuffer::new(4);
        let view = AtomicsView::new(sab);
        view.store(0, 100);
        let prev = view.exchange(0, 200);
        assert_eq!(prev, 100);
        assert_eq!(view.load(0), 200);
    }

    #[test]
    fn compare_exchange_success_and_failure() {
        let sab = SharedArrayBuffer::new(4);
        let view = AtomicsView::new(sab);
        view.store(0, 5);
        // Success.
        let v = view.compare_exchange(0, 5, 10);
        assert_eq!(v, 5);
        assert_eq!(view.load(0), 10);
        // Failure.
        let v = view.compare_exchange(0, 999, 0);
        assert_eq!(v, 10);
        assert_eq!(view.load(0), 10);
    }

    #[test]
    fn sab_clone_shares_storage() {
        let sab = SharedArrayBuffer::new(4);
        let a = AtomicsView::new(sab.clone());
        let b = AtomicsView::new(sab);
        a.store(0, 42);
        assert_eq!(b.load(0), 42);
    }
}
