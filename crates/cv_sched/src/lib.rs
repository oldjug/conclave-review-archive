//! `cv_sched` — the engine's compute work pool (Milestone 1.3 of the master
//! design). The shared substrate for parallel networking, image decode,
//! rasterization, style/layout, and JIT compilation: submit a job with a
//! priority, a fixed set of worker threads run it. No per-task OS-thread spawn
//! (the cost the design targets — image decode currently spawns a thread per
//! image).
//!
//! This is **distinct** from the HTML event-loop task scheduler (timers / rAF /
//! microtasks, which is single-threaded and ordering-sensitive). `cv_sched` is
//! for `Send` compute work that may run on any thread.
//!
//! v1 is a correct `Condvar`-guarded shared priority queue. The Chase-Lev
//! per-worker work-stealing deque (the design's named target) is a later
//! scalability refinement — it lowers queue contention at high core counts but
//! is not required to meet the Milestone 1.3 exit criterion (net + image decode
//! run on the pool with no per-task thread spawn). Bounded backpressure is
//! likewise a follow-on; v1's queue is unbounded.
//!
//! Pure safe Rust, std only (no third-party crates, per workspace policy).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Priority band for a submitted job. Higher bands drain fully before lower —
/// an interactive raster job is not starved behind a pile of background decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Priority {
    /// Latency-critical, on the path to a frame (e.g. visible-tile raster).
    High,
    /// Normal foreground work.
    Normal,
    /// Background / speculative (e.g. offscreen image decode, prefetch parse).
    Low,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct Inner {
    high: VecDeque<Job>,
    normal: VecDeque<Job>,
    low: VecDeque<Job>,
    shutdown: bool,
}

impl Inner {
    fn push(&mut self, priority: Priority, job: Job) {
        match priority {
            Priority::High => self.high.push_back(job),
            Priority::Normal => self.normal.push_back(job),
            Priority::Low => self.low.push_back(job),
        }
    }
    fn pop(&mut self) -> Option<Job> {
        self.high
            .pop_front()
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.low.pop_front())
    }
    fn pending(&self) -> usize {
        self.high.len() + self.normal.len() + self.low.len()
    }
}

struct Shared {
    inner: Mutex<Inner>,
    cv: Condvar,
}

/// A fixed-size pool of worker threads draining a priority queue. Dropping the
/// pool drains all queued jobs, then joins the workers (no work is lost).
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("threads", &self.workers.len())
            .field("pending", &self.pending())
            .finish_non_exhaustive()
    }
}

fn worker(shared: &Arc<Shared>) {
    loop {
        let job = {
            let mut inner = shared.inner.lock().unwrap();
            loop {
                if let Some(job) = inner.pop() {
                    break Some(job);
                }
                if inner.shutdown {
                    break None; // queue drained AND shutting down → exit
                }
                inner = shared.cv.wait(inner).unwrap();
            }
        };
        match job {
            Some(job) => job(), // run OUTSIDE the lock so other workers proceed
            None => return,
        }
    }
}

impl Pool {
    /// Create a pool with `threads` workers (clamped to ≥1).
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner::default()),
            cv: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(threads);
        for i in 0..threads {
            let s = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("cv_sched-{i}"))
                .spawn(move || worker(&s))
                .expect("cv_sched: failed to spawn worker thread");
            workers.push(handle);
        }
        Self { shared, workers }
    }

    /// Create a pool sized to the machine's parallelism (minus a little headroom
    /// for the UI + renderer threads), the way the design wants it provisioned.
    pub fn with_available_parallelism() -> Self {
        let n = thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
        // Leave 2 cores for the UI + renderer-main threads (design: those are
        // sacred for input latency); never go below 1 worker.
        Self::new(n.saturating_sub(2).max(1))
    }

    /// Submit a job at the given priority. Returns immediately; the job runs on
    /// a worker thread.
    pub fn submit(&self, priority: Priority, job: impl FnOnce() + Send + 'static) {
        {
            let mut inner = self.shared.inner.lock().unwrap();
            inner.push(priority, Box::new(job));
        }
        self.shared.cv.notify_one();
    }

    /// Submit at [`Priority::Normal`].
    pub fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        self.submit(Priority::Normal, job);
    }

    /// Number of worker threads.
    pub fn thread_count(&self) -> usize {
        self.workers.len()
    }

    /// Jobs queued but not yet started (a snapshot; racy by nature).
    pub fn pending(&self) -> usize {
        self.shared.inner.lock().unwrap().pending()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut inner = self.shared.inner.lock().unwrap();
            inner.shutdown = true;
        }
        self.shared.cv.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    #[test]
    fn all_submitted_jobs_run() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let pool = Pool::new(4);
            for _ in 0..1000 {
                let c = Arc::clone(&counter);
                pool.submit(Priority::Normal, move || {
                    c.fetch_add(1, Ordering::Relaxed);
                });
            }
            // Drop drains the queue + joins — no work is lost.
        }
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn higher_priority_runs_first() {
        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let pool = Pool::new(1); // single worker → deterministic ordering
        let (tx, rx) = mpsc::channel::<()>();
        // A blocker holds the one worker until we release it, so the next three
        // jobs queue and are then drained strictly in priority order.
        pool.submit(Priority::Normal, move || {
            rx.recv().unwrap();
        });
        for (p, name) in [
            (Priority::Low, "low"),
            (Priority::Normal, "normal"),
            (Priority::High, "high"),
        ] {
            let o = Arc::clone(&order);
            pool.submit(p, move || o.lock().unwrap().push(name));
        }
        tx.send(()).unwrap(); // release the blocker
        drop(pool); // drain + join
        assert_eq!(*order.lock().unwrap(), vec!["high", "normal", "low"]);
    }

    #[test]
    fn empty_pool_shuts_down_cleanly() {
        let pool = Pool::new(8);
        assert_eq!(pool.thread_count(), 8);
        assert_eq!(pool.pending(), 0);
        drop(pool); // must not hang
    }

    #[test]
    fn zero_threads_clamps_to_one() {
        let pool = Pool::new(0);
        assert_eq!(pool.thread_count(), 1);
        let done = Arc::new(AtomicUsize::new(0));
        let d = Arc::clone(&done);
        pool.submit(Priority::High, move || {
            d.fetch_add(1, Ordering::Relaxed);
        });
        drop(pool);
        assert_eq!(done.load(Ordering::Relaxed), 1);
    }
}
