//! no_stubs_guard — mechanical enforcement of the project's "no stubs, ever"
//! rule. All logic lives in `tests/no_stubs.rs`; this lib is intentionally
//! empty so the crate exists in the workspace and its test runs under
//! `cargo test`. See that test for the scan rules + rationale.
