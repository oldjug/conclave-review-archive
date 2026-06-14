//! `cv_ipc` — Mojo-equivalent IPC for `Conclave`.
//!
//! V1 scope: in-process `MessagePipe` with two endpoints. Each endpoint
//! can both send and receive typed messages. The wire format is
//! identical to what the cross-process named-pipe transport will use,
//! so encoders written today work tomorrow unchanged.
//!
//! Wire format (per message):
//! ```text
//!   u32 magic        = 0x544252_4D  ("TBRM" — Toasty Browser ReMote)
//!   u32 message_id   — caller-defined, used for routing / typing
//!   u32 payload_len  — bytes that follow
//!   u8  payload[payload_len]
//! ```
//!
//! Primitives encode little-endian. Strings are u32 length followed by
//! UTF-8 bytes. Vec<T> is u32 count followed by T encoded N times.

#![allow(clippy::module_name_repetitions)]
#![allow(missing_debug_implementations, unreachable_pub)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

mod codec;
mod pipe;

#[cfg(target_os = "windows")]
mod named_pipe;

#[cfg(target_os = "windows")]
mod spawn;

#[cfg(target_os = "windows")]
mod sandbox;

#[cfg(target_os = "windows")]
mod sandboxed_child;

pub mod renderer_host;
pub mod renderer_proto;
pub use renderer_proto::{
    Msg, MsgDirection, MsgKind, PROTOCOL_VERSION, decode_message, encode_message,
    protocol_compatible,
};

pub use codec::{Decode, DecodeError, Encode, Reader, Writer};
pub use pipe::{Endpoint, MessagePipe};

#[cfg(target_os = "windows")]
pub use named_pipe::{NamedPipeEndpoint, PipeHandle, TransportError};

#[cfg(target_os = "windows")]
pub use spawn::{AppliedMitigation, ChildProcess, LowIntegrityToken, SpawnError};

#[cfg(target_os = "windows")]
pub use sandbox::{JobObject, JobObjectBuilder, SandboxError};

#[cfg(target_os = "windows")]
pub use sandboxed_child::{
    AppliedTier, ChildKind, SandboxedChild, SandboxedSpawnError, SpawnOptions,
    options_from_channel_policy,
};

/// Re-export the cv_sandbox policy authority so callers can build
/// `SpawnOptions` from a channel policy without taking a direct
/// dependency on cv_sandbox. `MitigationPolicies` is re-exported too so a
/// caller can compose a bespoke renderer profile (e.g. the persistent
/// renderer keeps win32k enabled for in-process GDI text bake) without
/// pulling in cv_sandbox directly.
pub use cv_sandbox::{ChannelPolicy, MitigationPolicies};

/// Wire-format magic word. Every framed message begins with it.
pub const MAGIC: u32 = 0x5442_524D;

/// Internal: shared queue between two endpoints. One direction per
/// `MessagePipe` half; the pipe owns two of them, swapped per endpoint.
pub(crate) type Queue = Rc<RefCell<VecDeque<Vec<u8>>>>;

/// Errors a receive operation can hit.
#[derive(Debug, PartialEq, Eq)]
pub enum RecvError {
    /// Both endpoints dropped — no more messages will ever arrive.
    Closed,
    /// No message available right now (non-blocking receive).
    Empty,
    /// Frame header was malformed (bad magic or length).
    BadFrame,
    /// Frame was well-formed but the payload violated protocol rules.
    BadMessage,
}

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => f.write_str("pipe closed"),
            Self::Empty => f.write_str("no message available"),
            Self::BadFrame => f.write_str("bad frame"),
            Self::BadMessage => f.write_str("bad message"),
        }
    }
}

impl std::error::Error for RecvError {}

/// Errors a send operation can hit.
#[derive(Debug, PartialEq, Eq)]
pub enum SendError {
    /// Peer endpoint dropped.
    PeerClosed,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("pipe peer closed")
    }
}

impl std::error::Error for SendError {}
