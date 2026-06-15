//! `cv_audio` — audio decoders + WASAPI output backend.

#![allow(missing_debug_implementations, non_upper_case_globals)]

pub mod aac;
pub mod decode;
pub mod graph;
pub mod imdct;
pub mod mp3;
pub mod opus;
pub mod output;
pub mod wasapi;
