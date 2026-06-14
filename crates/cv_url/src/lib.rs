//! `cv_url` — URL parsing per the WHATWG URL Standard.
//!
//! Scope today: absolute URLs with the schemes the engine actually drives
//! (`http`, `https`, `ws`, `wss`, `file`, `data`, `about`, `blob`). The
//! parser is structured as the spec state machine so adding the remaining
//! special-scheme cases and relative-URL resolution is incremental.
//!
//! Reference: <https://url.spec.whatwg.org/>.

pub mod origin;
pub mod parse;
pub mod percent;
pub mod scheme;

pub use origin::Origin;
pub use parse::{Url, UrlError};
pub use scheme::Scheme;
