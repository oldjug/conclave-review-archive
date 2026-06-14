//! `cv_profile` — user profile state: history, bookmarks, downloads,
//! autofill, password vault.

pub mod autofill;
pub mod bookmarks;
pub mod disk;
pub mod downloads;
pub mod history;
pub mod passwords;

pub use disk::{ProfileRoot, export_profile_json, import_profile_json};
