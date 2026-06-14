//! In-process `MessagePipe` — two endpoints that can send and receive
//! typed messages. The wire format is identical to the cross-process
//! transport so an Encode + frame call here is byte-for-byte what a
//! named-pipe write will send.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::codec::{Decode, Encode, Reader, Writer};
use crate::{MAGIC, Queue, RecvError, SendError};

/// Build a pair of connected endpoints. `(a, b)` — anything `a` sends
/// is delivered to `b` and vice-versa.
pub fn channel() -> (Endpoint, Endpoint) {
    let a_to_b: Queue = Rc::new(RefCell::new(VecDeque::new()));
    let b_to_a: Queue = Rc::new(RefCell::new(VecDeque::new()));
    let a = Endpoint {
        send: Rc::clone(&a_to_b),
        recv: Rc::clone(&b_to_a),
    };
    let b = Endpoint {
        send: Rc::clone(&b_to_a),
        recv: Rc::clone(&a_to_b),
    };
    (a, b)
}

/// One half of a `MessagePipe`. Owns one outbound and one inbound
/// queue. Cloning an Endpoint creates a second handle to the same
/// queues — useful when the same process wants to read from one task
/// and write from another.
#[derive(Clone)]
pub struct Endpoint {
    pub(crate) send: Queue,
    pub(crate) recv: Queue,
}

impl std::fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Endpoint")
            .field("send_pending", &self.send.borrow().len())
            .field("recv_pending", &self.recv.borrow().len())
            .finish()
    }
}

impl Endpoint {
    /// Send a typed message identified by `message_id`. The payload is
    /// encoded via `T::encode` and framed with the wire header.
    pub fn send<T: Encode>(&self, message_id: u32, payload: &T) -> Result<(), SendError> {
        // For now there's no way to drop only one half of a pipe
        // independently of cloning; in-process the send queue is alive
        // as long as either endpoint exists.
        let mut payload_w = Writer::new();
        payload.encode(&mut payload_w);
        let payload_bytes = payload_w.into_bytes();
        let mut frame = Writer::with_capacity(12 + payload_bytes.len());
        frame.write_u32(MAGIC);
        frame.write_u32(message_id);
        frame.write_u32(payload_bytes.len() as u32);
        frame.bytes.extend_from_slice(&payload_bytes);
        self.send.borrow_mut().push_back(frame.into_bytes());
        Ok(())
    }

    /// Non-blocking receive. Returns Empty when no frame is queued.
    pub fn try_recv<T: Decode>(&self) -> Result<(u32, T), RecvError> {
        let frame = self.recv.borrow_mut().pop_front().ok_or(RecvError::Empty)?;
        Self::decode_frame::<T>(&frame)
    }

    /// Number of frames waiting in the inbound queue.
    pub fn pending(&self) -> usize {
        self.recv.borrow().len()
    }

    fn decode_frame<T: Decode>(frame: &[u8]) -> Result<(u32, T), RecvError> {
        let mut r = Reader::new(frame);
        let magic = r.read_u32().map_err(|_| RecvError::BadFrame)?;
        if magic != MAGIC {
            return Err(RecvError::BadFrame);
        }
        let id = r.read_u32().map_err(|_| RecvError::BadFrame)?;
        let payload_len = r.read_u32().map_err(|_| RecvError::BadFrame)? as usize;
        if r.remaining() < payload_len {
            return Err(RecvError::BadFrame);
        }
        // Decode against just the payload section. We don't strictly
        // need the slice — Reader stops at remaining — but bounding it
        // lets a decoder catch over-reads when payload_len lies.
        let start = frame.len() - r.remaining();
        let payload = &frame[start..start + payload_len];
        let mut pr = Reader::new(payload);
        let v = T::decode(&mut pr).map_err(|_| RecvError::BadMessage)?;
        Ok((id, v))
    }
}

/// A factory facade so the public API reads better than `channel()`.
/// `MessagePipe::new()` returns the same `(Endpoint, Endpoint)` pair.
pub struct MessagePipe;

impl MessagePipe {
    /// Build a fresh in-process pipe with two connected endpoints.
    #[must_use]
    pub fn new() -> (Endpoint, Endpoint) {
        channel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_simple_message() {
        let (a, b) = MessagePipe::new();
        a.send(7, &"hello".to_string()).unwrap();
        let (id, msg): (u32, String) = b.try_recv().unwrap();
        assert_eq!(id, 7);
        assert_eq!(msg, "hello");
    }

    #[test]
    fn bidirectional_send() {
        let (a, b) = MessagePipe::new();
        a.send(1, &42u32).unwrap();
        b.send(2, &"pong".to_string()).unwrap();
        let (id_b, n_b): (u32, u32) = b.try_recv().unwrap();
        assert_eq!(id_b, 1);
        assert_eq!(n_b, 42);
        let (id_a, s_a): (u32, String) = a.try_recv().unwrap();
        assert_eq!(id_a, 2);
        assert_eq!(s_a, "pong");
    }

    #[test]
    fn empty_receive_returns_empty() {
        let (a, _b) = MessagePipe::new();
        let r: Result<(u32, u32), _> = a.try_recv();
        assert_eq!(r, Err(RecvError::Empty));
    }

    #[test]
    fn pending_counts_inbound() {
        let (a, b) = MessagePipe::new();
        for i in 0..5u32 {
            a.send(i, &i).unwrap();
        }
        assert_eq!(b.pending(), 5);
        assert_eq!(a.pending(), 0);
        let _: (u32, u32) = b.try_recv().unwrap();
        assert_eq!(b.pending(), 4);
    }

    #[test]
    fn messages_arrive_in_send_order() {
        let (a, b) = MessagePipe::new();
        let names = ["alpha", "beta", "gamma", "delta"];
        for (i, n) in names.iter().enumerate() {
            a.send(i as u32, &n.to_string()).unwrap();
        }
        for (i, expected) in names.iter().enumerate() {
            let (id, s): (u32, String) = b.try_recv().unwrap();
            assert_eq!(id, i as u32);
            assert_eq!(s, *expected);
        }
    }

    #[test]
    fn cross_message_types() {
        // Sender doesn't have to commit to a single type — `id`
        // distinguishes messages and the receiver dispatches.
        let (a, b) = MessagePipe::new();
        a.send(100, &"first".to_string()).unwrap();
        a.send(200, &vec![1u32, 2, 3]).unwrap();
        let (id1, s): (u32, String) = b.try_recv().unwrap();
        assert_eq!(id1, 100);
        assert_eq!(s, "first");
        let (id2, v): (u32, Vec<u32>) = b.try_recv().unwrap();
        assert_eq!(id2, 200);
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn bad_magic_rejected() {
        let (a, b) = MessagePipe::new();
        // Manually inject a bad frame.
        a.send
            .borrow_mut()
            .push_back(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let r: Result<(u32, u32), _> = b.try_recv();
        assert_eq!(r, Err(RecvError::BadFrame));
    }

    #[test]
    fn decode_rejection_is_bad_message() {
        let (a, b) = MessagePipe::new();
        a.send(7, &vec![0xFFu8]).unwrap();
        let r: Result<(u32, String), _> = b.try_recv();
        assert_eq!(r, Err(RecvError::BadMessage));
    }
}
