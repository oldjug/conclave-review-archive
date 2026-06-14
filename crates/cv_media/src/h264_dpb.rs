//! H.264 Decoded Picture Buffer (DPB) + bi-prediction helpers.
//!
//! Holds reconstructed reference frames keyed by `frame_num` /
//! `pic_order_cnt`. The decoder feeds each new decoded frame via
//! `insert`; the display path drains in POC (presentation) order
//! via `output_next`. Sliding-window reference management evicts
//! the oldest short-term ref when the buffer overflows
//! `max_num_ref_frames`.
//!
//! Also provides `bi_predict_avg` for B-frames: per-sample average
//! of two motion-compensated prediction blocks.

use crate::h264_mb_loop::Frame;

#[derive(Debug, Clone)]
pub struct DpbEntry {
    pub frame_num: u32,
    pub pic_order_cnt: i32,
    pub frame: Frame,
    /// True for short-term refs (the typical case). The full
    /// memory_management_control_operation set marks long-term refs;
    /// those land in a follow-up.
    pub short_term: bool,
}

#[derive(Debug)]
pub struct Dpb {
    entries: Vec<DpbEntry>,
    max_num_ref_frames: usize,
    /// POC of the last frame the consumer fetched via `output_next`.
    last_output_poc: i32,
}

impl Dpb {
    pub fn new(max_num_ref_frames: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_num_ref_frames,
            last_output_poc: i32::MIN,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a decoded picture. Evicts the sliding-window short-term
    /// ref if the cache is at capacity.
    pub fn insert(&mut self, entry: DpbEntry) {
        if self.entries.len() >= self.max_num_ref_frames {
            // Sliding window: evict the short-term ref with the lowest
            // frame_num.
            if let Some((idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.short_term)
                .min_by_key(|(_, e)| e.frame_num)
            {
                self.entries.remove(idx);
            } else {
                // All long-term — drop the oldest by frame_num anyway
                // so we don't grow without bound.
                let idx = self
                    .entries
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, e)| e.frame_num)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.entries.remove(idx);
            }
        }
        self.entries.push(entry);
    }

    /// Find an entry by `pic_order_cnt`. Used by B-frame ref list
    /// construction to grab the forward/backward refs.
    pub fn find_by_poc(&self, poc: i32) -> Option<&DpbEntry> {
        self.entries.iter().find(|e| e.pic_order_cnt == poc)
    }

    /// Output the next frame in display order (lowest POC > last
    /// output). Returns None if no buffered frame is ready.
    pub fn output_next(&mut self) -> Option<DpbEntry> {
        let idx = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.pic_order_cnt > self.last_output_poc)
            .min_by_key(|(_, e)| e.pic_order_cnt)
            .map(|(i, _)| i)?;
        let entry = self.entries.remove(idx);
        self.last_output_poc = entry.pic_order_cnt;
        Some(entry)
    }
}

/// B-frame bi-prediction: per-sample average of two prediction
/// blocks, rounded half-up (spec §8.4.2.3.1 eq. 8-262).
pub fn bi_predict_avg(p0: &[u8], p1: &[u8]) -> Vec<u8> {
    assert_eq!(p0.len(), p1.len());
    let mut out = Vec::with_capacity(p0.len());
    for i in 0..p0.len() {
        let v = (p0[i] as i32 + p1[i] as i32 + 1) >> 1;
        out.push(v.clamp(0, 255) as u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(frame_num: u32, poc: i32) -> DpbEntry {
        DpbEntry {
            frame_num,
            pic_order_cnt: poc,
            frame: Frame::new(16, 16),
            short_term: true,
        }
    }

    #[test]
    fn insert_grows_within_capacity() {
        let mut dpb = Dpb::new(4);
        dpb.insert(make_entry(0, 0));
        dpb.insert(make_entry(1, 2));
        assert_eq!(dpb.len(), 2);
    }

    #[test]
    fn insert_at_capacity_evicts_oldest_short_term() {
        let mut dpb = Dpb::new(2);
        dpb.insert(make_entry(0, 0));
        dpb.insert(make_entry(1, 2));
        dpb.insert(make_entry(2, 4));
        assert_eq!(dpb.len(), 2);
        // Oldest (frame_num=0) should be gone.
        assert!(dpb.find_by_poc(0).is_none());
        assert!(dpb.find_by_poc(2).is_some());
        assert!(dpb.find_by_poc(4).is_some());
    }

    #[test]
    fn output_drains_in_poc_order_even_when_inserted_out_of_order() {
        let mut dpb = Dpb::new(8);
        // Real B-frame pattern: I0, P3, B1, B2 — decode order ≠
        // display order.
        dpb.insert(make_entry(0, 0)); // I
        dpb.insert(make_entry(1, 6)); // P
        dpb.insert(make_entry(2, 2)); // B
        dpb.insert(make_entry(3, 4)); // B
        // Display order should drain POC 0 → 2 → 4 → 6.
        let mut display = Vec::new();
        while let Some(e) = dpb.output_next() {
            display.push(e.pic_order_cnt);
        }
        assert_eq!(display, vec![0, 2, 4, 6]);
    }

    #[test]
    fn bi_predict_averages_samples_with_rounding() {
        let p0 = vec![100u8; 16];
        let p1 = vec![120u8; 16];
        let out = bi_predict_avg(&p0, &p1);
        for &v in &out {
            // (100 + 120 + 1) >> 1 = 110
            assert_eq!(v, 110);
        }
    }

    #[test]
    fn bi_predict_handles_uneven_average() {
        let p0 = vec![50u8; 4];
        let p1 = vec![51u8; 4];
        let out = bi_predict_avg(&p0, &p1);
        for &v in &out {
            // (50 + 51 + 1) >> 1 = 51 (rounded up)
            assert_eq!(v, 51);
        }
    }
}
