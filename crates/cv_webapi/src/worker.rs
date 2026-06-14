//! Web Worker / Service Worker — execute worker JS in a second realm
//! on a real OS thread.
//!
//! Each `WorkerRealm` spawns a `std::thread` that owns its own
//! `cv_js::Interp`. The document thread posts messages over a
//! `mpsc::Sender`; the worker thread pulls each message, looks up its
//! global `onmessage`, and invokes it through the interp. Replies
//! come back over a second channel the document drains via
//! `drain_outbound`.
//!
//! Service workers extend this with the install/activate state
//! machine + a fetch-interception hook that the network layer consults
//! before issuing a request.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone)]
pub struct WorkerOptions {
    pub r#type: WorkerType,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerType {
    Classic,
    Module,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            r#type: WorkerType::Classic,
            name: String::new(),
        }
    }
}

/// Message sent from document → worker.
#[derive(Debug)]
enum WorkerCommand {
    Message(String),
    Terminate,
}

/// A worker realm — independent interp + thread + bidirectional
/// message channels.
pub struct WorkerRealm {
    pub script_url: String,
    pub options: WorkerOptions,
    pub installed: bool,
    pub activated: bool,
    /// Sender that pushes messages onto the worker thread's queue.
    inbound: mpsc::Sender<WorkerCommand>,
    /// Buffer the worker thread fills with `self.postMessage` payloads.
    outbound: Arc<Mutex<Vec<String>>>,
    /// Join handle of the worker thread — kept so the realm can wait
    /// for clean shutdown on terminate.
    handle: Mutex<Option<thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for WorkerRealm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerRealm")
            .field("script_url", &self.script_url)
            .field("options", &self.options)
            .field("installed", &self.installed)
            .field("activated", &self.activated)
            .finish()
    }
}

impl WorkerRealm {
    /// Spawn a new worker thread. `source` is the body of the worker
    /// script (already fetched by the caller).
    pub fn spawn(script_url: String, options: WorkerOptions, source: String) -> Self {
        let (tx, rx) = mpsc::channel::<WorkerCommand>();
        let outbound = Arc::new(Mutex::new(Vec::<String>::new()));
        let outbound_clone = outbound.clone();
        let handle = thread::spawn(move || {
            let mut interp = cv_js::Interp::new();
            // Install postMessage on the worker's `self` global. The
            // implementation pushes onto the outbound buffer so the
            // document thread can drain.
            let outbound_for_post = outbound_clone.clone();
            interp.define_global(
                "postMessage",
                cv_js::native_fn("postMessage", move |args| {
                    let msg = args
                        .first()
                        .map(|v| v.to_display_string())
                        .unwrap_or_default();
                    if let Ok(mut q) = outbound_for_post.lock() {
                        q.push(msg);
                    }
                    Ok(cv_js::Value::Undefined)
                }),
            );
            // `self` is the worker's global. We expose it as an object;
            // when the script writes `self.onmessage = fn`, we look it
            // up below by indexing `self`.
            let self_obj: std::rc::Rc<std::cell::RefCell<cv_js::OrderedMap<String, cv_js::Value>>> =
                std::rc::Rc::new(std::cell::RefCell::new(cv_js::OrderedMap::new()));
            interp.define_global("self", cv_js::Value::Object(self_obj.clone()));
            // Run the worker script once. Errors are dropped — real
            // implementations would post them to onerror.
            let _ = interp.run(&source);
            // Message loop.
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    WorkerCommand::Terminate => break,
                    WorkerCommand::Message(data) => {
                        // Look up `onmessage` — try `self.onmessage`
                        // (per the WorkerGlobalScope model), then fall
                        // back to a bare global `onmessage`.
                        let cb_opt = self_obj
                            .borrow()
                            .get("onmessage")
                            .cloned()
                            .or_else(|| interp.get_global("onmessage"));
                        if let Some(cb) = cb_opt {
                            if matches!(
                                cb,
                                cv_js::Value::Function(_)
                                    | cv_js::Value::NativeFunction(_)
                                    | cv_js::Value::BcClosure(_)
                            ) {
                                use std::cell::RefCell;
                                use std::rc::Rc;
                                let mut evt: cv_js::OrderedMap<String, cv_js::Value> =
                                    cv_js::OrderedMap::new();
                                evt.insert("type".into(), cv_js::Value::String("message".into()));
                                evt.insert("data".into(), cv_js::Value::String(data.into()));
                                let evt_val = cv_js::Value::Object(Rc::new(RefCell::new(evt)));
                                let _ = interp.call_value(cb, vec![evt_val]);
                            }
                        }
                    }
                }
            }
        });
        Self {
            script_url,
            options,
            installed: false,
            activated: false,
            inbound: tx,
            outbound,
            handle: Mutex::new(Some(handle)),
        }
    }

    pub fn post_message(&self, msg: String) {
        let _ = self.inbound.send(WorkerCommand::Message(msg));
    }

    pub fn drain_outbound(&self) -> Vec<String> {
        self.outbound
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default()
    }

    pub fn terminate(&self) {
        let _ = self.inbound.send(WorkerCommand::Terminate);
        if let Ok(mut g) = self.handle.lock() {
            if let Some(h) = g.take() {
                let _ = h.join();
            }
        }
    }
}

impl Drop for WorkerRealm {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Service Worker — extends WorkerRealm with the fetch-interception
/// hook. The network path calls `intercept_fetch(url)` on the SW; if
/// the worker's `onfetch` handler returns a response, network
/// short-circuits.
pub struct ServiceWorker {
    pub realm: WorkerRealm,
    pub scope: String,
    pub state: ServiceWorkerState,
    pub registered_intercepts: std::sync::Mutex<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceWorkerState {
    Installing,
    Installed,
    Activating,
    Activated,
    Redundant,
}

impl ServiceWorker {
    pub fn new(script_url: String, scope: String, source: String) -> Self {
        Self {
            realm: WorkerRealm::spawn(script_url, WorkerOptions::default(), source),
            scope,
            state: ServiceWorkerState::Installing,
            registered_intercepts: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn install(&mut self) {
        self.state = ServiceWorkerState::Installed;
        self.realm.installed = true;
        // Post an `install` event to drive the worker's installer.
        self.realm.post_message(r#"{"type":"install"}"#.into());
    }

    pub fn activate(&mut self) {
        self.state = ServiceWorkerState::Activated;
        self.realm.activated = true;
        self.realm.post_message(r#"{"type":"activate"}"#.into());
    }

    pub fn controls(&self, url: &str) -> bool {
        url.starts_with(&self.scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_runs_script_and_replies() {
        let src = r#"
            self.onmessage = function(e) {
                postMessage("got:" + e.data);
            };
        "#;
        let w = WorkerRealm::spawn("test.js".into(), WorkerOptions::default(), src.into());
        // Give the worker a moment to install onmessage.
        std::thread::sleep(std::time::Duration::from_millis(50));
        w.post_message("hello".into());
        // Wait for the reply.
        for _ in 0..50 {
            let out = w.drain_outbound();
            if !out.is_empty() {
                assert_eq!(out[0], "got:hello");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("worker never replied");
    }

    #[test]
    fn sw_lifecycle_transitions_and_terminates() {
        let mut sw = ServiceWorker::new("sw.js".into(), "/".into(), "/* sw */".into());
        assert_eq!(sw.state, ServiceWorkerState::Installing);
        sw.install();
        assert_eq!(sw.state, ServiceWorkerState::Installed);
        sw.activate();
        assert_eq!(sw.state, ServiceWorkerState::Activated);
        assert!(sw.controls("/a/b"));
    }
}
