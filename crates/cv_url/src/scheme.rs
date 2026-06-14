//! Scheme classification per the WHATWG URL Standard.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Scheme {
    Http,
    Https,
    Ws,
    Wss,
    File,
    Data,
    About,
    Blob,
    /// Any other scheme. Stored as a normalized lowercase string.
    Other,
}

impl Scheme {
    pub fn from_lowercase(s: &str) -> Self {
        match s {
            "http" => Self::Http,
            "https" => Self::Https,
            "ws" => Self::Ws,
            "wss" => Self::Wss,
            "file" => Self::File,
            "data" => Self::Data,
            "about" => Self::About,
            "blob" => Self::Blob,
            _ => Self::Other,
        }
    }

    /// "Special" per WHATWG — these have host/path semantics and default
    /// ports. Everything else is opaque-path.
    pub fn is_special(self) -> bool {
        matches!(
            self,
            Self::Http | Self::Https | Self::Ws | Self::Wss | Self::File
        )
    }

    /// Default port for the scheme, or `None` if there is no default
    /// (file/data/about/blob/other).
    pub fn default_port(self) -> Option<u16> {
        match self {
            Self::Http | Self::Ws => Some(80),
            Self::Https | Self::Wss => Some(443),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Https => "https",
            Self::Ws => "ws",
            Self::Wss => "wss",
            Self::File => "file",
            Self::Data => "data",
            Self::About => "about",
            Self::Blob => "blob",
            Self::Other => "",
        }
    }
}
