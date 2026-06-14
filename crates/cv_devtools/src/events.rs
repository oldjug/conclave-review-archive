//! DevTools CDP — real event emission.
//!
//! Tracks the CDP event surface for Page / DOM / CSS / Runtime /
//! Debugger / Network. Each domain holds a queue the renderer pushes
//! to from its lifecycle hooks; `drain()` is what the CDP socket
//! pumps out over the ws transport.

use std::collections::VecDeque;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct CdpEvent {
    pub method: String,
    pub params_json: String,
}

#[derive(Default)]
pub struct EventQueue {
    events: Mutex<VecDeque<CdpEvent>>,
}

impl EventQueue {
    pub fn push(&self, method: &str, params_json: impl Into<String>) {
        if let Ok(mut q) = self.events.lock() {
            q.push_back(CdpEvent {
                method: method.to_string(),
                params_json: params_json.into(),
            });
        }
    }
    pub fn drain(&self) -> Vec<CdpEvent> {
        self.events
            .lock()
            .map(|mut q| q.drain(..).collect())
            .unwrap_or_default()
    }
}

/// Builders that produce the canonical CDP JSON payload for each
/// event the renderer fires. Keeping them here makes the lifecycle
/// hook sites a single line.
pub mod build {
    pub fn page_load_event_fired(timestamp: f64) -> String {
        format!(r#"{{"timestamp":{timestamp}}}"#)
    }
    pub fn page_dom_content_event_fired(timestamp: f64) -> String {
        format!(r#"{{"timestamp":{timestamp}}}"#)
    }
    pub fn page_frame_started_loading(frame_id: &str) -> String {
        format!(r#"{{"frameId":"{}"}}"#, frame_id)
    }
    pub fn page_frame_navigated(frame_id: &str, url: &str) -> String {
        format!(r#"{{"frame":{{"id":"{}","url":"{}"}}}}"#, frame_id, url)
    }
    pub fn network_request_will_be_sent(
        request_id: &str,
        url: &str,
        method: &str,
        ts: f64,
    ) -> String {
        format!(
            r#"{{"requestId":"{}","request":{{"url":"{}","method":"{}"}},"timestamp":{}}}"#,
            request_id, url, method, ts
        )
    }
    pub fn network_response_received(request_id: &str, url: &str, status: u16, ts: f64) -> String {
        format!(
            r#"{{"requestId":"{}","response":{{"url":"{}","status":{}}},"timestamp":{}}}"#,
            request_id, url, status, ts
        )
    }
    pub fn runtime_console_api_called(level: &str, text: &str, ts: f64) -> String {
        format!(
            r#"{{"type":"{}","args":[{{"type":"string","value":"{}"}}],"timestamp":{}}}"#,
            level,
            escape(text),
            ts
        )
    }
    pub fn dom_attribute_modified(node_id: u32, name: &str, value: &str) -> String {
        format!(
            r#"{{"nodeId":{},"name":"{}","value":"{}"}}"#,
            node_id,
            name,
            escape(value)
        )
    }
    pub fn debugger_paused(reason: &str, hit_breakpoints: &[String]) -> String {
        let bps = hit_breakpoints
            .iter()
            .map(|b| format!("\"{}\"", b))
            .collect::<Vec<_>>()
            .join(",");
        format!(r#"{{"reason":"{}","hitBreakpoints":[{}]}}"#, reason, bps)
    }
    fn escape(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_drains_in_order() {
        let q = EventQueue::default();
        q.push("Page.loadEventFired", "{}");
        q.push("Network.requestWillBeSent", "{}");
        let out = q.drain();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].method, "Page.loadEventFired");
        assert!(q.drain().is_empty());
    }
}
