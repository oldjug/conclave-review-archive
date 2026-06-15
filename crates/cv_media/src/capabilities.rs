//! Media capability probing — the engine behind
//! `HTMLMediaElement.canPlayType()` and
//! `MediaSource.isTypeSupported()`.
//!
//! Returns one of the three spec strings, mirroring Blink's
//! `MIMETypeRegistry::SupportsMediaMIMEType` →
//! `HTMLMediaElement::canPlayType` mapping (HTML §4.8.12.3 "Ability to
//! play media resources"):
//!
//!   * `""`         — the type is *not* supported.
//!   * `"maybe"`    — the container is supported but no codecs string
//!                    was supplied, so support can't be confirmed until
//!                    playback is attempted.
//!   * `"probably"` — the container *and* every listed codec are
//!                    supported.
//!
//! Reference: <https://html.spec.whatwg.org/multipage/media.html#dom-navigator-canplaytype>
//! and Chromium `html_media_element.cc` `CanPlayType()` which maps
//! `kIsNotSupported`→"", `kMayBeSupported`→"maybe",
//! `kIsSupported`→"probably".
//!
//! "Supported" here is grounded in what `cv_media` can actually demux
//! and decode — not a hard-coded yes. If we add/remove a codec, this
//! probe changes with it.

/// The three canPlayType results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    No,
    Maybe,
    Probably,
}

impl Support {
    pub fn as_str(self) -> &'static str {
        match self {
            Support::No => "",
            Support::Maybe => "maybe",
            Support::Probably => "probably",
        }
    }
}

/// Containers `cv_media` can demux. Tied to the real demuxers present
/// in this crate (ebml/webm, mp4) so it tracks actual capability.
fn container_supported(mime: &str) -> bool {
    matches!(
        mime,
        // WebM/Matroska — real demuxer in `webm.rs`.
        "video/webm" | "audio/webm"
        // ISO-BMFF / MP4 — header walker in `mp4.rs`.
        | "video/mp4" | "audio/mp4"
        // Bare codec containers we parse directly.
        | "audio/flac" | "audio/x-flac"
        | "audio/ogg" | "video/ogg" | "application/ogg"
        | "audio/mpeg" | "audio/mp3"
        | "audio/aac" | "audio/wav" | "audio/x-wav"
    )
}

/// Is a single RFC 6381 codec short-name decodable by `cv_media`?
/// Compared case-insensitively against the family prefix (the profile
/// suffix after the first '.' is ignored — we match the codec family,
/// which is what determines which decoder we dispatch to).
fn codec_supported(codec: &str) -> bool {
    let c = codec.trim().to_ascii_lowercase();
    // H.264 / AVC — `avc1.*`, `avc3.*` (cv_media::h264).
    if c.starts_with("avc1") || c.starts_with("avc3") || c == "h264" {
        return true;
    }
    // H.265 / HEVC — header parser only (cv_media::h265).
    if c.starts_with("hev1") || c.starts_with("hvc1") || c == "h265" || c == "hevc" {
        return true;
    }
    // AV1 — `av01.*` (cv_media::av1).
    if c.starts_with("av01") || c == "av1" {
        return true;
    }
    // VP8 / VP9 — `vp8`, `vp9`, `vp09.*` (cv_media::vp9).
    if c == "vp8" || c == "vp9" || c.starts_with("vp09") || c.starts_with("vp08") {
        return true;
    }
    // Audio codecs.
    if c.starts_with("mp4a") || c == "aac" {
        return true; // AAC (cv_audio::aac)
    }
    if c == "opus" {
        return true; // (cv_media::* / cv_audio::opus)
    }
    if c == "vorbis" {
        return true; // (cv_media::vorbis)
    }
    if c == "flac" {
        return true; // (cv_media::flac)
    }
    if c == "mp3" || c.starts_with("mp4a.6b") || c.starts_with("mp4a.69") {
        return true; // MP3 (cv_audio::mp3)
    }
    false
}

/// Split a content-type into the bare MIME and an optional `codecs="…"`
/// parameter. Per RFC 6381 the codecs list is comma-separated and may
/// be quoted.
fn split_type(content_type: &str) -> (String, Option<Vec<String>>) {
    let mut mime = String::new();
    let mut codecs: Option<Vec<String>> = None;
    for (i, part) in content_type.split(';').enumerate() {
        let part = part.trim();
        if i == 0 {
            mime = part.to_ascii_lowercase();
            continue;
        }
        if let Some(rest) = part.strip_prefix("codecs") {
            let rest = rest.trim_start();
            if let Some(val) = rest.strip_prefix('=') {
                let val = val.trim().trim_matches('"').trim_matches('\'');
                let list: Vec<String> =
                    val.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                if !list.is_empty() {
                    codecs = Some(list);
                }
            }
        }
    }
    (mime, codecs)
}

/// Implements `HTMLMediaElement.canPlayType(type)`.
pub fn can_play_type(content_type: &str) -> Support {
    let ct = content_type.trim();
    if ct.is_empty() {
        return Support::No;
    }
    let (mime, codecs) = split_type(ct);
    if !container_supported(&mime) {
        return Support::No;
    }
    match codecs {
        // No codecs string: container alone is supported ⇒ "maybe".
        None => Support::Maybe,
        Some(list) => {
            // Every listed codec must be decodable ⇒ "probably".
            if list.iter().all(|c| codec_supported(c)) {
                Support::Probably
            } else {
                Support::No
            }
        }
    }
}

/// `MediaSource.isTypeSupported(type)` is stricter: it requires the
/// container be MSE-compatible (fragmented MP4 / WebM) and, if codecs
/// are listed, that they all be supported. We return a bool.
pub fn mse_is_type_supported(content_type: &str) -> bool {
    let (mime, codecs) = split_type(content_type.trim());
    let mse_container = matches!(
        mime.as_str(),
        "video/mp4" | "audio/mp4" | "video/webm" | "audio/webm"
    );
    if !mse_container {
        return false;
    }
    match codecs {
        // MSE requires codecs to be specified for a confident answer,
        // but Blink will still accept a bare supported container.
        None => true,
        Some(list) => list.iter().all(|c| codec_supported(c)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_container_is_empty_string() {
        assert_eq!(can_play_type("video/x-msvideo"), Support::No);
        assert_eq!(can_play_type("application/json"), Support::No);
        assert_eq!(can_play_type("").as_str(), "");
    }

    #[test]
    fn supported_container_without_codecs_is_maybe() {
        assert_eq!(can_play_type("video/mp4"), Support::Maybe);
        assert_eq!(can_play_type("video/webm"), Support::Maybe);
        assert_eq!(can_play_type("audio/mp4").as_str(), "maybe");
    }

    #[test]
    fn supported_container_and_codecs_is_probably() {
        assert_eq!(
            can_play_type("video/mp4; codecs=\"avc1.42E01E\""),
            Support::Probably
        );
        assert_eq!(
            can_play_type("video/webm; codecs=\"vp9\""),
            Support::Probably
        );
        assert_eq!(
            can_play_type("video/webm; codecs=\"vp8, vorbis\"").as_str(),
            "probably"
        );
        assert_eq!(
            can_play_type("video/mp4; codecs=\"av01.0.05M.08\"").as_str(),
            "probably"
        );
        assert_eq!(
            can_play_type("audio/mp4; codecs=\"mp4a.40.2\"").as_str(),
            "probably"
        );
    }

    #[test]
    fn supported_container_with_unsupported_codec_is_empty() {
        // Container is fine, but the codec family is not one we decode.
        assert_eq!(
            can_play_type("video/mp4; codecs=\"theora\""),
            Support::No
        );
        assert_eq!(
            can_play_type("video/webm; codecs=\"vp9, theora\""),
            Support::No
        );
    }

    #[test]
    fn mse_is_type_supported_tracks_real_capability() {
        assert!(mse_is_type_supported("video/mp4; codecs=\"avc1.42E01E\""));
        assert!(mse_is_type_supported("video/webm; codecs=\"vp9, opus\""));
        assert!(!mse_is_type_supported("audio/flac")); // not an MSE container
        assert!(!mse_is_type_supported("video/mp4; codecs=\"theora\""));
    }
}
