//! JS event-loop scheduler — backs `setTimeout`, `setInterval`,
//! `clearTimeout`, `clearInterval`, and `requestAnimationFrame`.
//!
//! All times are absolute milliseconds since process start, read from
//! `Instant::now()`. The browser's frame ticker drains the queue,
//! pops any due tasks, invokes their callbacks via the persistent
//! `cv_js::Interp`, and re-renders if the DOM changed.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use cv_js::Value;

/// A pending setTimeout / setInterval / RAF callback.
#[derive(Clone)]
pub(crate) struct Task {
    pub id: u32,
    /// Wall-clock fire deadline (milliseconds since `Scheduler::base`).
    pub fire_at_ms: u64,
    /// `Some(period_ms)` for `setInterval`; `None` for `setTimeout` and RAF.
    pub repeat_ms: Option<u64>,
    /// JS callable (`Value::Function` or `Value::NativeFunction`).
    pub callable: Value,
    /// Origin label for debugging.
    pub kind: TaskKind,
    /// Extra positional arguments forwarded to the callback per HTML spec
    /// (e.g. `setTimeout(fn, 100, a, b)` → `fn(a, b)`).
    pub extra_args: Vec<Value>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum TaskKind {
    Timeout,
    Interval,
    AnimationFrame,
}

pub(crate) type SchedRef = Rc<RefCell<Scheduler>>;

pub(crate) struct Scheduler {
    base: Instant,
    next_id: u32,
    queue: Vec<Task>,
}

impl Scheduler {
    pub(crate) fn new() -> Self {
        Self {
            base: Instant::now(),
            next_id: 1,
            queue: Vec::new(),
        }
    }

    pub(crate) fn new_ref() -> SchedRef {
        Rc::new(RefCell::new(Self::new()))
    }

    /// All pending task callables — GC roots. Timer/interval/rAF callbacks hold
    /// page objects (via their closures) that are NOT reachable from the global
    /// object graph, so the cycle collector must treat them as live.
    pub(crate) fn task_callables(&self) -> Vec<Value> {
        self.queue.iter().map(|t| t.callable.clone()).collect()
    }

    /// Milliseconds elapsed since the scheduler started — our monotonic
    /// clock for deadline math.
    pub(crate) fn now_ms(&self) -> u64 {
        self.base.elapsed().as_millis() as u64
    }

    /// Diagnostic: count pending tasks by kind (timeouts, intervals, rafs,
    /// other). A non-zero interval count after a page settles is direct
    /// evidence that `useEffect`-registered `setInterval` polling fired.
    pub(crate) fn diag_counts(&self) -> (usize, usize, usize, usize) {
        let mut t = 0;
        let mut i = 0;
        let mut r = 0;
        let mut o = 0;
        for task in &self.queue {
            match task.kind {
                TaskKind::Timeout => t += 1,
                TaskKind::Interval => i += 1,
                TaskKind::AnimationFrame => r += 1,
                _ => o += 1,
            }
        }
        (t, i, r, o)
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    pub(crate) fn schedule_timeout(&mut self, callable: Value, delay_ms: u64) -> u32 {
        self.schedule_timeout_with_args(callable, delay_ms, Vec::new())
    }

    pub(crate) fn schedule_timeout_with_args(
        &mut self,
        callable: Value,
        delay_ms: u64,
        extra_args: Vec<Value>,
    ) -> u32 {
        let id = self.alloc_id();
        self.queue.push(Task {
            id,
            fire_at_ms: self.now_ms() + delay_ms,
            repeat_ms: None,
            callable,
            kind: TaskKind::Timeout,
            extra_args,
        });
        id
    }

    pub(crate) fn schedule_interval(&mut self, callable: Value, period_ms: u64) -> u32 {
        self.schedule_interval_with_args(callable, period_ms, Vec::new())
    }

    pub(crate) fn schedule_interval_with_args(
        &mut self,
        callable: Value,
        period_ms: u64,
        extra_args: Vec<Value>,
    ) -> u32 {
        let id = self.alloc_id();
        // Per the HTML spec, the first interval fire happens after the
        // period — not immediately.
        let p = period_ms.max(1);
        self.queue.push(Task {
            id,
            fire_at_ms: self.now_ms() + p,
            repeat_ms: Some(p),
            callable,
            kind: TaskKind::Interval,
            extra_args,
        });
        id
    }

    pub(crate) fn schedule_raf(&mut self, callable: Value) -> u32 {
        let id = self.alloc_id();
        // RAF fires on the *next* tick.  Use `now_ms() + 1` so that a rAF
        // callback that immediately re-arms itself (the particles.js pattern:
        // `function animate() { …; requestAnimationFrame(animate); }`) always
        // gets a `fire_at_ms` that is STRICTLY GREATER than the `raf_cutoff`
        // snapshot taken at the start of the current drain.  Without the +1,
        // if the callback finishes within the same millisecond the new rAF's
        // `fire_at_ms == raf_cutoff` satisfies the `<= raf_cutoff` drain
        // predicate, causing the re-armed rAF to fire again in the same drain
        // pass (tight busy-loop, delta = 0, animation freezes).
        self.queue.push(Task {
            id,
            fire_at_ms: self.now_ms() + 1,
            repeat_ms: None,
            callable,
            kind: TaskKind::AnimationFrame,
            extra_args: Vec::new(),
        });
        id
    }

    pub(crate) fn cancel(&mut self, id: u32) {
        self.queue.retain(|t| t.id != id);
    }

    /// Push an already-popped Task back onto the queue verbatim. Used by the
    /// drain's cooperative deadline yield: when a frame's time slice runs out
    /// mid-batch, the un-run tasks are re-queued so they fire on the NEXT tick
    /// instead of all running in one >12ms burst that locks the UI (the
    /// particles.js + 41-CSS-animation case where one batch was ~380ms).
    pub(crate) fn requeue(&mut self, task: Task) {
        self.queue.push(task);
    }

    /// Pop all tasks whose deadline is `<= cutoff_ms`. Intervals are
    /// re-queued with their next deadline before being returned, so the
    /// caller only has to invoke the callable.
    ///
    /// The caller passes a snapshot `cutoff_ms` (typically the value of
    /// `now_ms()` at the start of the drain cycle) rather than us
    /// reading the clock here. That snapshot is the key behaviour for
    /// `requestAnimationFrame`: when a RAF callback re-schedules itself
    /// via `requestAnimationFrame(tick)` inside its own invocation, the
    /// new task's `fire_at_ms` reads from the live `now_ms()` (which is
    /// slightly later than `cutoff_ms`), so the new RAF is deferred to
    /// the *next* drain cycle instead of firing 50 times in a row with
    /// the same effective timestamp. Without this, JS-driven
    /// animations (canvas particles, counter increments, orbital
    /// spins) see `delta = t - last = 0` between consecutive RAF calls
    /// and never advance.
    pub(crate) fn drain_due(&mut self, cutoff_ms: u64) -> Vec<Task> {
        self.drain_due_split(cutoff_ms, cutoff_ms)
    }

    /// Like `drain_due`, but `requestAnimationFrame` tasks use a SEPARATE
    /// `raf_cutoff_ms` — a snapshot of `now_ms()` taken ONCE at the start of the
    /// host's drain cycle. Timers/microtasks use the live, re-read `cutoff_ms`
    /// so React's 0-delay `MessageChannel` continuations complete in one drain;
    /// but a RAF callback that re-arms itself schedules the next RAF with
    /// `fire_at = now > raf_cutoff_ms`, so it is deferred to the NEXT drain
    /// cycle instead of re-firing immediately. Without this split a
    /// self-sustaining RAF loop (every canvas animation) busy-loops thousands
    /// of times per host tick with the SAME timestamp — burning the frame and
    /// making `delta = 0`, so nothing visibly animates.
    pub(crate) fn drain_due_split(&mut self, cutoff_ms: u64, raf_cutoff_ms: u64) -> Vec<Task> {
        let mut out = Vec::new();
        // Iterate from the end so swap_remove doesn't shift live items.
        let mut i = 0;
        while i < self.queue.len() {
            let task_cutoff = if self.queue[i].kind == TaskKind::AnimationFrame {
                raf_cutoff_ms
            } else {
                cutoff_ms
            };
            if self.queue[i].fire_at_ms <= task_cutoff {
                let task = self.queue.swap_remove(i);
                // Re-queue intervals at next deadline first so the
                // produced clone-callable is the only thing the host
                // needs to fire.
                if let Some(p) = task.repeat_ms {
                    self.queue.push(Task {
                        id: task.id,
                        fire_at_ms: cutoff_ms + p,
                        repeat_ms: Some(p),
                        callable: task.callable.clone(),
                        kind: TaskKind::Interval,
                        extra_args: task.extra_args.clone(),
                    });
                }
                out.push(task);
            } else {
                i += 1;
            }
        }
        // Preserve FIFO among same-deadline tasks for spec-ish behavior.
        out.sort_by_key(|t| (t.fire_at_ms, t.id));
        out
    }

    /// Drain only due ONE-SHOT timeouts (`setTimeout`/`queueMicrotask`/
    /// `requestIdleCallback`), leaving intervals and animation frames in the
    /// queue. Used by the `await` host-pump: core-js's Promise job queue runs
    /// on 0-delay timeouts, so this lets `await import()` settle a core-js
    /// promise WITHOUT firing self-re-arming intervals/RAF mid-await (which
    /// would spin the pump forever and re-enter React renders).
    pub(crate) fn drain_due_timeouts(&mut self, cutoff_ms: u64) -> Vec<Task> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.queue.len() {
            if self.queue[i].fire_at_ms <= cutoff_ms
                && matches!(self.queue[i].kind, TaskKind::Timeout)
            {
                out.push(self.queue.swap_remove(i));
            } else {
                i += 1;
            }
        }
        out.sort_by_key(|t| (t.fire_at_ms, t.id));
        out
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.queue.len()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_id_unique_and_cancellable() {
        let mut s = Scheduler::new();
        let a = s.schedule_timeout(Value::Number(1.0), 100);
        let b = s.schedule_timeout(Value::Number(2.0), 200);
        assert_ne!(a, b);
        s.cancel(a);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn raf_fires_on_next_tick() {
        let mut s = Scheduler::new();
        s.schedule_raf(Value::Number(0.0));
        // schedule_raf sets fire_at = now() + 1 so a rAF re-armed from within
        // its own callback is deferred to the next tick (fire_at > raf_cutoff).
        // Immediate drain (same ms) must NOT fire it.
        let snap = s.now_ms();
        assert!(
            s.drain_due(snap).is_empty(),
            "rAF scheduled with now()+1 must not fire in the same drain as the snapshot"
        );
        // After at least 1ms has elapsed the drain must fire it.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let due = s.drain_due(s.now_ms());
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].kind, TaskKind::AnimationFrame);
    }

    #[test]
    fn interval_requeues_itself() {
        let mut s = Scheduler::new();
        s.schedule_interval(Value::Number(7.0), 1);
        std::thread::sleep(std::time::Duration::from_millis(3));
        let first = s.drain_due(s.now_ms());
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].kind, TaskKind::Interval);
        // Should still be queued for next fire.
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn future_timeout_not_drained_yet() {
        let mut s = Scheduler::new();
        s.schedule_timeout(Value::Number(0.0), 60_000);
        assert!(s.drain_due(s.now_ms()).is_empty());
        assert_eq!(s.len(), 1);
    }
}
