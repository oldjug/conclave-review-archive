//! Sequenced task runner. Like Chromium's `SequencedTaskRunner`: tasks posted
//! to the same sequence run in FIFO order on a single worker at a time.
//! Free-floating tasks run on any worker.
//!
//! Bare minimum to drive the event loop. Priorities, deadlines, and
//! frame-aligned scheduling come in M2 with the compositor.

use crate::time::{Duration, Instant};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

type Task = Box<dyn FnOnce() + Send + 'static>;

struct DelayedTask {
    at: Instant,
    seq: u64,
    task: Task,
}

impl PartialEq for DelayedTask {
    fn eq(&self, other: &Self) -> bool {
        self.at == other.at && self.seq == other.seq
    }
}
impl Eq for DelayedTask {}
impl Ord for DelayedTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is max-heap; we want earliest first.
        other.at.cmp(&self.at).then(other.seq.cmp(&self.seq))
    }
}
impl PartialOrd for DelayedTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct PoolInner {
    queue: VecDeque<Task>,
    delayed: BinaryHeap<DelayedTask>,
    running_sequences: Vec<u64>,
    sequence_queues: HashMap<u64, VecDeque<Task>>,
    shutdown: bool,
}

pub struct TaskRunner {
    inner: Arc<(Mutex<PoolInner>, Condvar)>,
    next_seq: AtomicU64,
    next_sequence_id: AtomicU64,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for TaskRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskRunner").finish_non_exhaustive()
    }
}

impl TaskRunner {
    pub fn new(workers: usize) -> Arc<Self> {
        let inner = Arc::new((
            Mutex::new(PoolInner {
                queue: VecDeque::new(),
                delayed: BinaryHeap::new(),
                running_sequences: Vec::new(),
                sequence_queues: HashMap::new(),
                shutdown: false,
            }),
            Condvar::new(),
        ));

        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let cv = Arc::clone(&inner);
            let h = thread::Builder::new()
                .name(format!("tb-pool-{i}"))
                .spawn(move || worker_loop(&cv))
                .expect("spawn worker");
            handles.push(h);
        }

        Arc::new(Self {
            inner,
            next_seq: AtomicU64::new(0),
            next_sequence_id: AtomicU64::new(1),
            workers: Mutex::new(handles),
        })
    }

    pub fn post<F: FnOnce() + Send + 'static>(&self, f: F) {
        let (lock, cv) = &*self.inner;
        lock.lock().unwrap().queue.push_back(Box::new(f));
        cv.notify_one();
    }

    pub fn post_delayed<F: FnOnce() + Send + 'static>(&self, delay: Duration, f: F) {
        let at = Instant::now() + delay;
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let (lock, cv) = &*self.inner;
        lock.lock().unwrap().delayed.push(DelayedTask {
            at,
            seq,
            task: Box::new(f),
        });
        cv.notify_all();
    }

    /// Allocate a sequence ID. Tasks posted with this ID run in FIFO order.
    pub fn new_sequence(&self) -> u64 {
        self.next_sequence_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn post_on_sequence<F: FnOnce() + Send + 'static>(&self, sequence_id: u64, f: F) {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        g.sequence_queues
            .entry(sequence_id)
            .or_default()
            .push_back(Box::new(f));
        cv.notify_one();
    }

    pub fn shutdown(&self) {
        let (lock, cv) = &*self.inner;
        lock.lock().unwrap().shutdown = true;
        cv.notify_all();
        let handles: Vec<_> = self.workers.lock().unwrap().drain(..).collect();
        for h in handles {
            let _ = h.join();
        }
    }
}

enum PickedWork {
    Free(Task),
    Sequenced { sequence: u64, task: Task },
}

fn pick_one(g: &mut PoolInner) -> Option<PickedWork> {
    // Promote ripe delayed tasks first.
    let now = Instant::now();
    while g.delayed.peek().is_some_and(|t| t.at <= now) {
        let t = g.delayed.pop().unwrap();
        g.queue.push_back(t.task);
    }
    if let Some(t) = g.queue.pop_front() {
        return Some(PickedWork::Free(t));
    }
    let keys: Vec<u64> = g.sequence_queues.keys().copied().collect();
    for k in keys {
        if g.running_sequences.contains(&k) {
            continue;
        }
        if let Some(t) = g.sequence_queues.get_mut(&k).and_then(VecDeque::pop_front) {
            g.running_sequences.push(k);
            return Some(PickedWork::Sequenced {
                sequence: k,
                task: t,
            });
        }
    }
    None
}

fn worker_loop(inner: &Arc<(Mutex<PoolInner>, Condvar)>) {
    let (lock, cv) = &**inner;
    loop {
        let work = {
            let mut g = lock.lock().unwrap();
            loop {
                if g.shutdown
                    && g.queue.is_empty()
                    && g.delayed.is_empty()
                    && g.sequence_queues.values().all(VecDeque::is_empty)
                {
                    return;
                }
                if let Some(w) = pick_one(&mut g) {
                    break w;
                }
                let next = g.delayed.peek().map(|t| t.at);
                if let Some(at) = next {
                    let dur_micros = (at - Instant::now()).as_micros().max(0) as u64;
                    let std_dur = std::time::Duration::from_micros(dur_micros);
                    let (ng, _) = cv.wait_timeout(g, std_dur).unwrap();
                    g = ng;
                } else {
                    g = cv.wait(g).unwrap();
                }
            }
        };
        match work {
            PickedWork::Free(t) => t(),
            PickedWork::Sequenced { sequence, task } => {
                task();
                let mut g = lock.lock().unwrap();
                g.running_sequences.retain(|&x| x != sequence);
                cv.notify_one();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn posts_and_runs() {
        let r = TaskRunner::new(4);
        let n = Arc::new(AtomicUsize::new(0));
        for _ in 0..200 {
            let n = Arc::clone(&n);
            r.post(move || {
                n.fetch_add(1, Ordering::Relaxed);
            });
        }
        for _ in 0..400 {
            if n.load(Ordering::Relaxed) == 200 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(n.load(Ordering::Relaxed), 200);
    }

    #[test]
    fn sequence_is_serial() {
        let r = TaskRunner::new(4);
        let seq = r.new_sequence();
        let log = Arc::new(Mutex::new(Vec::new()));
        for i in 0..20 {
            let log = Arc::clone(&log);
            r.post_on_sequence(seq, move || {
                log.lock().unwrap().push(i);
            });
        }
        for _ in 0..400 {
            if log.lock().unwrap().len() == 20 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let got = log.lock().unwrap().clone();
        assert_eq!(
            got,
            (0..20).collect::<Vec<_>>(),
            "sequence ran out of order"
        );
    }
}
