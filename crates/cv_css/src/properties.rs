//! Computed-style value types — a small starter set.
//!
//! Grows as layout/paint demand more. We don't aim for full Color level 4
//! or full font shorthand parsing yet.

use crate::tokenizer::CssToken;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    /// Sentinel for `currentColor`. CSS Color 4 §4.4: `currentColor`
    /// resolves at used-value time to the element's computed `color`.
    /// We can't resolve it at parse time (we don't yet know the
    /// element's color in the cascade order), so we carry this
    /// distinct fully-transparent marker through to `lower_style`,
    /// which swaps in the element's resolved text color. Chrome's
    /// `StyleColor` carries an `is_current_color_` flag for the same
    /// reason. Distinct from `TRANSPARENT` (0,0,0,0) by the green
    /// channel so the two never collide; fully transparent so an
    /// accidental literal match is visually inert.
    pub const CURRENT: Self = Self {
        r: 0,
        g: 1,
        b: 0,
        a: 0,
    };

    /// True if this is the `currentColor` sentinel (see `CURRENT`).
    pub fn is_current_color(self) -> bool {
        self == Self::CURRENT
    }

    pub fn from_name(name: &str) -> Option<Self> {
        let lc = name.to_ascii_lowercase();
        // `currentColor` keyword — defer to the element's `color` at
        // used-value time (resolved in `lower_style`).
        if lc == "currentcolor" {
            return Some(Self::CURRENT);
        }
        // CSS Color Module Level 4 — full named-color keyword set, plus
        // `transparent` and the British `grey` spellings. Comprehensive
        // so real production CSS with names like `cornflowerblue`,
        // `rebeccapurple`, etc. renders correctly instead of falling back.
        let rgb = match lc.as_str() {
            "transparent" => return Some(Self::TRANSPARENT),
            "aliceblue" => (240, 248, 255),
            "antiquewhite" => (250, 235, 215),
            "aqua" | "cyan" => (0, 255, 255),
            "aquamarine" => (127, 255, 212),
            "azure" => (240, 255, 255),
            "beige" => (245, 245, 220),
            "bisque" => (255, 228, 196),
            "black" => (0, 0, 0),
            "blanchedalmond" => (255, 235, 205),
            "blue" => (0, 0, 255),
            "blueviolet" => (138, 43, 226),
            "brown" => (165, 42, 42),
            "burlywood" => (222, 184, 135),
            "cadetblue" => (95, 158, 160),
            "chartreuse" => (127, 255, 0),
            "chocolate" => (210, 105, 30),
            "coral" => (255, 127, 80),
            "cornflowerblue" => (100, 149, 237),
            "cornsilk" => (255, 248, 220),
            "crimson" => (220, 20, 60),
            "darkblue" => (0, 0, 139),
            "darkcyan" => (0, 139, 139),
            "darkgoldenrod" => (184, 134, 11),
            "darkgray" | "darkgrey" => (169, 169, 169),
            "darkgreen" => (0, 100, 0),
            "darkkhaki" => (189, 183, 107),
            "darkmagenta" => (139, 0, 139),
            "darkolivegreen" => (85, 107, 47),
            "darkorange" => (255, 140, 0),
            "darkorchid" => (153, 50, 204),
            "darkred" => (139, 0, 0),
            "darksalmon" => (233, 150, 122),
            "darkseagreen" => (143, 188, 143),
            "darkslateblue" => (72, 61, 139),
            "darkslategray" | "darkslategrey" => (47, 79, 79),
            "darkturquoise" => (0, 206, 209),
            "darkviolet" => (148, 0, 211),
            "deeppink" => (255, 20, 147),
            "deepskyblue" => (0, 191, 255),
            "dimgray" | "dimgrey" => (105, 105, 105),
            "dodgerblue" => (30, 144, 255),
            "firebrick" => (178, 34, 34),
            "floralwhite" => (255, 250, 240),
            "forestgreen" => (34, 139, 34),
            "fuchsia" | "magenta" => (255, 0, 255),
            "gainsboro" => (220, 220, 220),
            "ghostwhite" => (248, 248, 255),
            "gold" => (255, 215, 0),
            "goldenrod" => (218, 165, 32),
            "gray" | "grey" => (128, 128, 128),
            "green" => (0, 128, 0),
            "greenyellow" => (173, 255, 47),
            "honeydew" => (240, 255, 240),
            "hotpink" => (255, 105, 180),
            "indianred" => (205, 92, 92),
            "indigo" => (75, 0, 130),
            "ivory" => (255, 255, 240),
            "khaki" => (240, 230, 140),
            "lavender" => (230, 230, 250),
            "lavenderblush" => (255, 240, 245),
            "lawngreen" => (124, 252, 0),
            "lemonchiffon" => (255, 250, 205),
            "lightblue" => (173, 216, 230),
            "lightcoral" => (240, 128, 128),
            "lightcyan" => (224, 255, 255),
            "lightgoldenrodyellow" => (250, 250, 210),
            "lightgray" | "lightgrey" => (211, 211, 211),
            "lightgreen" => (144, 238, 144),
            "lightpink" => (255, 182, 193),
            "lightsalmon" => (255, 160, 122),
            "lightseagreen" => (32, 178, 170),
            "lightskyblue" => (135, 206, 250),
            "lightslategray" | "lightslategrey" => (119, 136, 153),
            "lightsteelblue" => (176, 196, 222),
            "lightyellow" => (255, 255, 224),
            "lime" => (0, 255, 0),
            "limegreen" => (50, 205, 50),
            "linen" => (250, 240, 230),
            "maroon" => (128, 0, 0),
            "mediumaquamarine" => (102, 205, 170),
            "mediumblue" => (0, 0, 205),
            "mediumorchid" => (186, 85, 211),
            "mediumpurple" => (147, 112, 219),
            "mediumseagreen" => (60, 179, 113),
            "mediumslateblue" => (123, 104, 238),
            "mediumspringgreen" => (0, 250, 154),
            "mediumturquoise" => (72, 209, 204),
            "mediumvioletred" => (199, 21, 133),
            "midnightblue" => (25, 25, 112),
            "mintcream" => (245, 255, 250),
            "mistyrose" => (255, 228, 225),
            "moccasin" => (255, 228, 181),
            "navajowhite" => (255, 222, 173),
            "navy" => (0, 0, 128),
            "oldlace" => (253, 245, 230),
            "olive" => (128, 128, 0),
            "olivedrab" => (107, 142, 35),
            "orange" => (255, 165, 0),
            "orangered" => (255, 69, 0),
            "orchid" => (218, 112, 214),
            "palegoldenrod" => (238, 232, 170),
            "palegreen" => (152, 251, 152),
            "paleturquoise" => (175, 238, 238),
            "palevioletred" => (219, 112, 147),
            "papayawhip" => (255, 239, 213),
            "peachpuff" => (255, 218, 185),
            "peru" => (205, 133, 63),
            "pink" => (255, 192, 203),
            "plum" => (221, 160, 221),
            "powderblue" => (176, 224, 230),
            "purple" => (128, 0, 128),
            "rebeccapurple" => (102, 51, 153),
            "red" => (255, 0, 0),
            "rosybrown" => (188, 143, 143),
            "royalblue" => (65, 105, 225),
            "saddlebrown" => (139, 69, 19),
            "salmon" => (250, 128, 114),
            "sandybrown" => (244, 164, 96),
            "seagreen" => (46, 139, 87),
            "seashell" => (255, 245, 238),
            "sienna" => (160, 82, 45),
            "silver" => (192, 192, 192),
            "skyblue" => (135, 206, 235),
            "slateblue" => (106, 90, 205),
            "slategray" | "slategrey" => (112, 128, 144),
            "snow" => (255, 250, 250),
            "springgreen" => (0, 255, 127),
            "steelblue" => (70, 130, 180),
            "tan" => (210, 180, 140),
            "teal" => (0, 128, 128),
            "thistle" => (216, 191, 216),
            "tomato" => (255, 99, 71),
            "turquoise" => (64, 224, 208),
            "violet" => (238, 130, 238),
            "wheat" => (245, 222, 179),
            "white" => (255, 255, 255),
            "whitesmoke" => (245, 245, 245),
            "yellow" => (255, 255, 0),
            "yellowgreen" => (154, 205, 50),
            // CSS Color L4 system colors — mapped to a light-mode
            // palette by default. A future dark-mode pass can swap the
            // table when prefers-color-scheme reports dark.
            "canvas" | "canvastext" => match name.to_ascii_lowercase().as_str() {
                "canvas" => (255, 255, 255),
                "canvastext" => (0, 0, 0),
                _ => unreachable!(),
            },
            "linktext" => (0, 0, 238),
            "visitedtext" => (85, 26, 139),
            "activetext" => (255, 0, 0),
            "buttonface" => (240, 240, 240),
            "buttontext" => (0, 0, 0),
            "buttonborder" => (118, 118, 118),
            "field" => (255, 255, 255),
            "fieldtext" => (0, 0, 0),
            "graytext" | "greytext" => (128, 128, 128),
            "highlight" => (0, 120, 215),
            "highlighttext" => (255, 255, 255),
            "selecteditem" => (0, 120, 215),
            "selecteditemtext" => (255, 255, 255),
            "mark" => (255, 255, 0),
            "marktext" => (0, 0, 0),
            "accentcolor" => (0, 120, 215),
            "accentcolortext" => (255, 255, 255),
            _ => return None,
        };
        Some(Self {
            r: rgb.0,
            g: rgb.1,
            b: rgb.2,
            a: 255,
        })
    }

    pub fn from_hash(hex: &str) -> Option<Self> {
        let h = hex.as_bytes();
        let pull = |i: usize, n: usize| -> Option<u8> {
            let s = std::str::from_utf8(&h[i..i + n]).ok()?;
            let v = u32::from_str_radix(s, 16).ok()?;
            Some(if n == 1 {
                ((v << 4) | v) as u8
            } else {
                v as u8
            })
        };
        match h.len() {
            3 => Some(Self {
                r: pull(0, 1)?,
                g: pull(1, 1)?,
                b: pull(2, 1)?,
                a: 255,
            }),
            4 => Some(Self {
                r: pull(0, 1)?,
                g: pull(1, 1)?,
                b: pull(2, 1)?,
                a: pull(3, 1)?,
            }),
            6 => Some(Self {
                r: pull(0, 2)?,
                g: pull(2, 2)?,
                b: pull(4, 2)?,
                a: 255,
            }),
            8 => Some(Self {
                r: pull(0, 2)?,
                g: pull(2, 2)?,
                b: pull(4, 2)?,
                a: pull(6, 2)?,
            }),
            _ => None,
        }
    }

    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        let mut i = 0;
        while i < toks.len() {
            match &toks[i] {
                CssToken::Ident(name) => {
                    if let Some(c) = Self::from_name(name) {
                        return Some(c);
                    }
                }
                CssToken::Hash(h) => {
                    if let Some(c) = Self::from_hash(h) {
                        return Some(c);
                    }
                }
                CssToken::Function(name) => {
                    let end = find_matching_paren(toks, i + 1)?;
                    let args = &toks[i + 1..end];
                    let parsed = match name.to_ascii_lowercase().as_str() {
                        "rgb" | "rgba" => Self::from_rgb_args(args),
                        "hsl" | "hsla" => Self::from_hsl_args(args),
                        "hwb" => Self::from_hwb_args(args),
                        "lab" => Self::from_lab_args(args),
                        "lch" => Self::from_lch_args(args),
                        "oklab" => Self::from_oklab_args(args),
                        "oklch" => Self::from_oklch_args(args),
                        "color-mix" => Self::from_color_mix_args(args),
                        // `light-dark(<light>, <dark>)` — V1 chooses the
                        // light branch unless `prefers-color-scheme:
                        // dark` is on, but we don't have that signal at
                        // value-parse time. Pick the first colour.
                        "light-dark" => {
                            let parts = split_color_args(args)?;
                            if parts.is_empty() {
                                None
                            } else {
                                Self::from_tokens(&parts[0])
                            }
                        }
                        // `color(<space> r g b / a)` — V1 ignores the
                        // colour space and treats the three numbers as
                        // sRGB in [0, 1].
                        "color" => Self::from_color_function_args(args),
                        _ => None,
                    };
                    if let Some(c) = parsed {
                        return Some(c);
                    }
                    i = end;
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    fn from_rgb_args(args: &[CssToken]) -> Option<Self> {
        let parts = split_color_args(args)?;
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        // Each of R/G/B may be a number 0..255 or a percent 0..100.
        let r = parse_channel_byte(&parts[0])?;
        let g = parse_channel_byte(&parts[1])?;
        let b = parse_channel_byte(&parts[2])?;
        let a = if parts.len() == 4 {
            parse_alpha_byte(&parts[3])?
        } else {
            255
        };
        Some(Self { r, g, b, a })
    }

    fn from_hsl_args(args: &[CssToken]) -> Option<Self> {
        let parts = split_color_args(args)?;
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        let h = parse_hue_deg(&parts[0])?;
        let s = parse_percent_unit(&parts[1])? / 100.0;
        let l = parse_percent_unit(&parts[2])? / 100.0;
        let a = if parts.len() == 4 {
            parse_alpha_byte(&parts[3])?
        } else {
            255
        };
        let (r, g, b) = hsl_to_rgb(h, s, l);
        Some(Self { r, g, b, a })
    }

    /// CSS Color L4 §6.6 `hwb(H W B [/ A])`. H is a `<hue>`, W and B
    /// are percentages.  Equivalent to HSL with a particular substitution.
    fn from_hwb_args(args: &[CssToken]) -> Option<Self> {
        let parts = split_color_args(args)?;
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        let h = parse_hue_deg(&parts[0])?;
        let mut w = parse_percent_unit(&parts[1])? / 100.0;
        let mut bk = parse_percent_unit(&parts[2])? / 100.0;
        if w + bk >= 1.0 {
            let gray = (w / (w + bk) * 255.0).round() as u8;
            let a = if parts.len() == 4 {
                parse_alpha_byte(&parts[3])?
            } else {
                255
            };
            return Some(Self {
                r: gray,
                g: gray,
                b: gray,
                a,
            });
        }
        // hwb(H, W, B) = hsl(H, 100%, 50%) blended to white by W and black by B.
        let (mut r, mut g, mut bl) = hsl_to_rgb(h, 1.0, 0.5);
        // Convert to [0,1] floats so we can apply tint/shade arithmetic.
        let mut rf = r as f32 / 255.0;
        let mut gf = g as f32 / 255.0;
        let mut bf = bl as f32 / 255.0;
        rf = rf * (1.0 - w - bk) + w;
        gf = gf * (1.0 - w - bk) + w;
        bf = bf * (1.0 - w - bk) + w;
        // bk subtraction already covered by (1 - w - bk) above; clamp.
        let _ = (&mut w, &mut bk, &mut r, &mut g, &mut bl);
        let a = if parts.len() == 4 {
            parse_alpha_byte(&parts[3])?
        } else {
            255
        };
        Some(Self {
            r: (rf.clamp(0.0, 1.0) * 255.0).round() as u8,
            g: (gf.clamp(0.0, 1.0) * 255.0).round() as u8,
            b: (bf.clamp(0.0, 1.0) * 255.0).round() as u8,
            a,
        })
    }

    /// CSS Color L4 §10 `lab(L a b [/ A])`. L is 0..100 (or 0..100%),
    /// a/b are signed numbers (typically -125..125 per spec).
    fn from_lab_args(args: &[CssToken]) -> Option<Self> {
        let parts = split_color_args(args)?;
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        let l = parse_number_or_percent(&parts[0], 100.0)?;
        let a = parse_number_or_percent(&parts[1], 125.0)?;
        let b = parse_number_or_percent(&parts[2], 125.0)?;
        let alpha = if parts.len() == 4 {
            parse_alpha_byte(&parts[3])?
        } else {
            255
        };
        let (r, g, bl) = lab_to_srgb(l, a, b);
        Some(Self {
            r,
            g,
            b: bl,
            a: alpha,
        })
    }

    /// CSS Color L4 §10 `lch(L C H [/ A])`.
    fn from_lch_args(args: &[CssToken]) -> Option<Self> {
        let parts = split_color_args(args)?;
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        let l = parse_number_or_percent(&parts[0], 100.0)?;
        let c = parse_number_or_percent(&parts[1], 150.0)?;
        let h = parse_hue_deg(&parts[2])?;
        let alpha = if parts.len() == 4 {
            parse_alpha_byte(&parts[3])?
        } else {
            255
        };
        let h_rad = h.to_radians();
        let a = c * h_rad.cos();
        let b = c * h_rad.sin();
        let (r, g, bl) = lab_to_srgb(l, a, b);
        Some(Self {
            r,
            g,
            b: bl,
            a: alpha,
        })
    }

    /// CSS Color L4 §10 `oklab(L a b [/ A])`. L is 0..1 (or 0..100%).
    fn from_oklab_args(args: &[CssToken]) -> Option<Self> {
        let parts = split_color_args(args)?;
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        let l = parse_number_or_percent(&parts[0], 1.0)?;
        let a = parse_number_or_percent(&parts[1], 0.4)?;
        let b = parse_number_or_percent(&parts[2], 0.4)?;
        let alpha = if parts.len() == 4 {
            parse_alpha_byte(&parts[3])?
        } else {
            255
        };
        let (r, g, bl) = oklab_to_srgb(l, a, b);
        Some(Self {
            r,
            g,
            b: bl,
            a: alpha,
        })
    }

    /// CSS Color L4 §10 `oklch(L C H [/ A])`.
    fn from_oklch_args(args: &[CssToken]) -> Option<Self> {
        let parts = split_color_args(args)?;
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        let l = parse_number_or_percent(&parts[0], 1.0)?;
        let c = parse_number_or_percent(&parts[1], 0.4)?;
        let h = parse_hue_deg(&parts[2])?;
        let alpha = if parts.len() == 4 {
            parse_alpha_byte(&parts[3])?
        } else {
            255
        };
        let h_rad = h.to_radians();
        let a = c * h_rad.cos();
        let b = c * h_rad.sin();
        let (r, g, bl) = oklab_to_srgb(l, a, b);
        Some(Self {
            r,
            g,
            b: bl,
            a: alpha,
        })
    }

    /// CSS Color L4 §12 `color(<space> <r> <g> <b> [/ <alpha>])`. The
    /// colour space is parsed and dropped; the three numeric channels
    /// are interpreted as sRGB in [0, 1].
    fn from_color_function_args(args: &[CssToken]) -> Option<Self> {
        // Split off the colour-space ident (first non-whitespace token)
        // then read up to 3 number/percent values for r,g,b and an
        // optional `/ alpha`.
        let mut nums: Vec<f32> = Vec::new();
        let mut after_slash: Option<f32> = None;
        let mut past_first_ident = false;
        let mut slash_seen = false;
        for t in args {
            match t {
                CssToken::Whitespace => continue,
                CssToken::Ident(_) if !past_first_ident => {
                    past_first_ident = true;
                }
                CssToken::Delim('/') => {
                    slash_seen = true;
                }
                CssToken::Number(n) => {
                    if slash_seen {
                        after_slash = Some(*n as f32);
                    } else if nums.len() < 3 {
                        nums.push(*n as f32);
                    }
                }
                CssToken::Percent(p) => {
                    let v = (*p as f32) / 100.0;
                    if slash_seen {
                        after_slash = Some(v);
                    } else if nums.len() < 3 {
                        nums.push(v);
                    }
                }
                _ => {}
            }
        }
        if nums.len() < 3 {
            return None;
        }
        let to_byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let alpha = after_slash.map(|v| to_byte(v)).unwrap_or(255);
        Some(Self {
            r: to_byte(nums[0]),
            g: to_byte(nums[1]),
            b: to_byte(nums[2]),
            a: alpha,
        })
    }

    /// CSS Color L4 §12 `color-mix(in <space>, c1 [<pct>], c2 [<pct>])`.
    /// Interpolates in gamma-encoded sRGB by default (the `in srgb` space),
    /// and in linearised sRGB only when `in srgb-linear` is specified.
    /// `color-mix(in srgb, white 50%, black)` → rgb(128,128,128), not ~182.
    fn from_color_mix_args(args: &[CssToken]) -> Option<Self> {
        // Top-level comma split, then inspect the leading "in <space>" chunk.
        let parts = split_color_args(args)?;
        if parts.len() < 3 {
            return None;
        }
        // First chunk must contain `in`. Detect whether the space is
        // `srgb-linear` so we know which interpolation path to use.
        let first = &parts[0];
        let mut has_in = false;
        let mut use_linear = false;
        {
            let idents: Vec<&str> = first
                .iter()
                .filter_map(|t| {
                    if let CssToken::Ident(s) = t {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            for (i, &id) in idents.iter().enumerate() {
                if id.eq_ignore_ascii_case("in") {
                    has_in = true;
                    // The token after "in" names the colour space.
                    if let Some(&space) = idents.get(i + 1) {
                        if space.eq_ignore_ascii_case("srgb-linear") {
                            use_linear = true;
                        }
                    }
                    break;
                }
            }
        }
        if !has_in {
            return None;
        }
        // Parse colour and optional weight from a chunk like `white 60%`.
        fn parse_color_and_weight(toks: &[CssToken]) -> Option<(Color, Option<f32>)> {
            let mut pct: Option<f32> = None;
            let mut end = toks.len();
            for (i, t) in toks.iter().enumerate().rev() {
                if matches!(t, CssToken::Whitespace) {
                    continue;
                }
                if let CssToken::Percent(v) = t {
                    pct = Some(*v as f32);
                    end = i;
                    break;
                }
                break;
            }
            let colour = Color::from_tokens(&toks[..end])?;
            Some((colour, pct))
        }
        let (c1, w1_opt) = parse_color_and_weight(&parts[1])?;
        let (c2, w2_opt) = parse_color_and_weight(&parts[2])?;
        // Determine effective weights per CSS Color L4 §12.1.
        // When both weights are present and their sum is < 100%, the
        // result alpha is scaled by (p1+p2)/100 — captured via w1+w2 < 1.
        let (w1, w2) = match (w1_opt, w2_opt) {
            (Some(p1), Some(p2)) => {
                let sum = p1 + p2;
                if sum == 0.0 {
                    return None;
                }
                if sum > 100.0 {
                    (p1 / sum, p2 / sum)
                } else {
                    (p1 / 100.0, p2 / 100.0)
                }
            }
            (Some(p1), None) => (p1 / 100.0, 1.0 - p1 / 100.0),
            (None, Some(p2)) => (1.0 - p2 / 100.0, p2 / 100.0),
            (None, None) => (0.5, 0.5),
        };
        // Normalised interpolation weights (sum to 1.0) for the RGB channels.
        // The alpha is scaled separately by the raw weight sum.
        let weight_sum = w1 + w2;
        let (nw1, nw2) = if weight_sum > 0.0 {
            (w1 / weight_sum, w2 / weight_sum)
        } else {
            (0.5, 0.5)
        };
        let mix_channel = if use_linear {
            // `in srgb-linear`: linearise → mix → gamma-encode back.
            let to_lin = |c: u8| -> f32 {
                let s = c as f32 / 255.0;
                if s <= 0.04045 {
                    s / 12.92
                } else {
                    ((s + 0.055) / 1.055).powf(2.4)
                }
            };
            let to_srgb = |l: f32| -> u8 {
                let v = if l <= 0.0031308 {
                    12.92 * l
                } else {
                    1.055 * l.powf(1.0 / 2.4) - 0.055
                };
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            };
            [
                to_srgb(to_lin(c1.r) * nw1 + to_lin(c2.r) * nw2),
                to_srgb(to_lin(c1.g) * nw1 + to_lin(c2.g) * nw2),
                to_srgb(to_lin(c1.b) * nw1 + to_lin(c2.b) * nw2),
            ]
        } else {
            // `in srgb` (default) and all other spaces: mix in gamma-encoded
            // sRGB directly — channel values are already gamma-encoded byte
            // values, so `r = r1*p + r2*(1-p)` is the correct sRGB mix.
            [
                (c1.r as f32 * nw1 + c2.r as f32 * nw2).round() as u8,
                (c1.g as f32 * nw1 + c2.g as f32 * nw2).round() as u8,
                (c1.b as f32 * nw1 + c2.b as f32 * nw2).round() as u8,
            ]
        };
        // Alpha scales by the raw weight sum (< 1 when sum of input pcts < 100%).
        let a = ((c1.a as f32 * w1 + c2.a as f32 * w2).clamp(0.0, 255.0)).round() as u8;
        Some(Self {
            r: mix_channel[0],
            g: mix_channel[1],
            b: mix_channel[2],
            a,
        })
    }
}

/// Parse a `<number> | <percentage>` token, normalising the percent
/// against `pct_scale` (e.g. percent_scale = 100 means `50%` → 50).
fn parse_number_or_percent(toks: &[CssToken], pct_scale: f32) -> Option<f32> {
    for t in toks {
        match t {
            CssToken::Whitespace => continue,
            CssToken::Number(n) => return Some(*n as f32),
            CssToken::Percent(p) => return Some(*p as f32 / 100.0 * pct_scale),
            _ => break,
        }
    }
    None
}

/// CIELab D65 → linear-sRGB → sRGB byte triple.
fn lab_to_srgb(l: f32, a: f32, b: f32) -> (u8, u8, u8) {
    // Lab → XYZ (D65 white).
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let eps: f32 = 216.0 / 24389.0;
    let kappa: f32 = 24389.0 / 27.0;
    let xn = if fx.powi(3) > eps {
        fx.powi(3)
    } else {
        (116.0 * fx - 16.0) / kappa
    };
    let yn = if l > kappa * eps {
        ((l + 16.0) / 116.0).powi(3)
    } else {
        l / kappa
    };
    let zn = if fz.powi(3) > eps {
        fz.powi(3)
    } else {
        (116.0 * fz - 16.0) / kappa
    };
    let x = xn * 0.95047;
    let y = yn * 1.0;
    let z = zn * 1.08883;
    // XYZ → linear sRGB.
    let r = 3.2404542 * x + -1.5371385 * y + -0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x + -0.2040259 * y + 1.0572252 * z;
    let to_srgb = |l: f32| {
        let v = if l <= 0.0031308 {
            12.92 * l
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    (to_srgb(r), to_srgb(g), to_srgb(bl))
}

/// OKLab → sRGB byte triple. Björn Ottosson's reference matrix.
fn oklab_to_srgb(l: f32, a: f32, b: f32) -> (u8, u8, u8) {
    // OKLab → LMS (cube each).
    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;
    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;
    // LMS → linear sRGB.
    let r = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
    let g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
    let b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3;
    let to_srgb = |l: f32| {
        let v = if l <= 0.0031308 {
            12.92 * l
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    (to_srgb(r), to_srgb(g), to_srgb(b_lin))
}

fn find_matching_paren(toks: &[CssToken], start: usize) -> Option<usize> {
    let mut depth = 1;
    let mut i = start;
    while i < toks.len() {
        match &toks[i] {
            CssToken::Function(_) | CssToken::LeftParen => depth += 1,
            CssToken::RightParen => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Split the body of a color function into 3 or 4 channel argument lists.
/// Accepts both the legacy comma form (`rgb(r, g, b, a)`) and the modern
/// whitespace + `/` form (`rgb(r g b / a)`). Note: callers strip
/// `Whitespace` tokens before invoking us, so in modern syntax we have to
/// fall back to one-value-token-per-arg splitting.
fn split_color_args(args: &[CssToken]) -> Option<Vec<Vec<CssToken>>> {
    // Decide which mode by scanning for commas. Comma form: split at
    // commas. Modern form: each individual Number / Percent / Dimension
    // becomes its own arg, with `/` toggling into the alpha slot.
    let has_comma = args.iter().any(|t| matches!(t, CssToken::Comma));
    let mut parts: Vec<Vec<CssToken>> = Vec::new();
    let mut cur: Vec<CssToken> = Vec::new();
    if has_comma {
        for t in args {
            match t {
                CssToken::Comma => parts.push(std::mem::take(&mut cur)),
                CssToken::Whitespace => {}
                _ => cur.push(t.clone()),
            }
        }
        if !cur.is_empty() {
            parts.push(cur);
        }
    } else {
        for t in args {
            match t {
                CssToken::Whitespace => {}
                CssToken::Delim('/') => {
                    // separator before alpha — nothing to emit; next value
                    // token lands in its own part.
                }
                CssToken::Number(_) | CssToken::Percent(_) | CssToken::Dimension { .. } => {
                    parts.push(vec![t.clone()]);
                }
                _ => {
                    // Ignore other tokens (ident keywords like "none").
                }
            }
        }
    }
    parts.retain(|p| !p.is_empty());
    if parts.is_empty() {
        return None;
    }
    Some(parts)
}

fn parse_channel_byte(toks: &[CssToken]) -> Option<u8> {
    for t in toks {
        match t {
            CssToken::Number(n) => return Some(clamp_byte(*n)),
            CssToken::Percent(p) => return Some(clamp_byte(*p * 2.55)),
            _ => {}
        }
    }
    None
}

fn parse_alpha_byte(toks: &[CssToken]) -> Option<u8> {
    for t in toks {
        match t {
            CssToken::Number(n) => return Some(clamp_byte(*n * 255.0)),
            CssToken::Percent(p) => return Some(clamp_byte(*p * 2.55)),
            _ => {}
        }
    }
    None
}

fn parse_hue_deg(toks: &[CssToken]) -> Option<f32> {
    for t in toks {
        match t {
            CssToken::Number(n) => return Some(*n as f32),
            CssToken::Dimension { value, unit } => {
                let v = *value as f32;
                return Some(match unit.to_ascii_lowercase().as_str() {
                    "deg" | "" => v,
                    "grad" => v * 360.0 / 400.0,
                    "rad" => v * 180.0 / core::f32::consts::PI,
                    "turn" => v * 360.0,
                    _ => v,
                });
            }
            _ => {}
        }
    }
    None
}

fn parse_percent_unit(toks: &[CssToken]) -> Option<f32> {
    for t in toks {
        if let CssToken::Percent(p) = t {
            return Some(*p as f32);
        }
    }
    None
}

fn clamp_byte(v: f64) -> u8 {
    if v.is_nan() {
        0
    } else if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v.round() as u8
    }
}

/// HSL → RGB per CSS Color 4 §7. h in degrees, s/l in 0..=1.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let h = ((h % 360.0) + 360.0) % 360.0;
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);
    let f = |n: f32| -> f32 {
        let k = (n + h / 30.0) % 12.0;
        let a = s * l.min(1.0 - l);
        l - a * (-1.0_f32).max((k - 3.0).min((9.0 - k).min(1.0)))
    };
    (
        clamp_byte(f(0.0) as f64 * 255.0),
        clamp_byte(f(8.0) as f64 * 255.0),
        clamp_byte(f(4.0) as f64 * 255.0),
    )
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Length {
    Px(f32),
    Em(f32),
    Rem(f32),
    /// Viewport-width percent (`vw`). 1vw = 1% of viewport width.
    Vw(f32),
    /// Viewport-height percent (`vh`). 1vh = 1% of viewport height.
    Vh(f32),
    /// CSS `pt` (1pt = 1.333…px at 96dpi). Common in print stylesheets.
    Pt(f32),
    Percent(f32),
    Auto,
    Zero,
    /// `calc(a + b - c)` — stored as a linear combination of all the
    /// primitive unit families so we stay `Copy`. Only `+` and `-` are
    /// supported in V1; `*` / `/` against a unitless number reduce to
    /// scaled-coefficient terms, but `*` against another length errors
    /// out at parse time (CSS specifies that's invalid anyway).
    Calc(Calc),
    /// `clamp(min, preferred, max)` — each arm is stored in the same
    /// linear unit space as `calc()` so resolution can stay `Copy`.
    Clamp(ClampExpr),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClampExpr {
    pub min: Calc,
    pub preferred: Calc,
    pub max: Calc,
}

/// Linear combination of the primitive length units, used by `Length::Calc`.
/// At resolve time each component is multiplied by its environment value
/// and the results summed: `px + em*em + rem*rem + vw*vw_px/100 + vh*vh_px/100
/// + pt*96/72 + percent*parent/100`.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Calc {
    pub px: f32,
    pub em: f32,
    pub rem: f32,
    pub vw: f32,
    pub vh: f32,
    pub pt: f32,
    pub percent: f32,
}

impl Calc {
    fn add_unit(&mut self, sign: f32, value: f32, unit_lc: &str) {
        let v = sign * value;
        match unit_lc {
            "px" | "" => self.px += v,
            "em" => self.em += v,
            "rem" => self.rem += v,
            "vw" => self.vw += v,
            "vh" => self.vh += v,
            "pt" => self.pt += v,
            // Unknown units fall through to px so we still get sensible
            // numbers (cm/mm/in could be added properly later).
            _ => self.px += v,
        }
    }

    fn from_length(length: Length) -> Option<Self> {
        let mut calc = Calc::default();
        match length {
            Length::Px(v) => calc.px = v,
            Length::Em(v) => calc.em = v,
            Length::Rem(v) => calc.rem = v,
            Length::Vw(v) => calc.vw = v,
            Length::Vh(v) => calc.vh = v,
            Length::Pt(v) => calc.pt = v,
            Length::Percent(v) => calc.percent = v,
            Length::Zero => {}
            Length::Calc(c) => calc = c,
            Length::Auto | Length::Clamp(_) => return None,
        }
        Some(calc)
    }

    fn resolve_px_with_viewport(
        self,
        em: f32,
        rem: f32,
        parent: f32,
        vw_px: f32,
        vh_px: f32,
    ) -> f32 {
        self.px
            + self.em * em
            + self.rem * rem
            + self.vw * vw_px / 100.0
            + self.vh * vh_px / 100.0
            + self.pt * 96.0 / 72.0
            + self.percent * parent / 100.0
    }
}

impl Length {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        let mut i = 0;
        while i < toks.len() {
            let t = &toks[i];
            match t {
                CssToken::Function(name) if name.eq_ignore_ascii_case("calc") => {
                    let end = find_matching_paren(toks, i + 1)?;
                    let args = &toks[i + 1..end];
                    if let Some(c) = Calc::parse(args) {
                        return Some(Self::Calc(c));
                    }
                    i = end + 1;
                    continue;
                }
                // `env(<name>, <fallback>)` — environment variable
                // substitution. V1 only knows the safe-area-inset-*
                // family; other names fall back to zero (or the
                // supplied fallback if it parses).
                CssToken::Function(name) if name.eq_ignore_ascii_case("env") => {
                    let end = find_matching_paren(toks, i + 1)?;
                    let args = &toks[i + 1..end];
                    let parts = split_top_level_commas(args);
                    let env_name: String = parts
                        .first()
                        .map(|toks| {
                            toks.iter()
                                .find_map(|t| match t {
                                    CssToken::Ident(s) => Some(s.clone()),
                                    _ => None,
                                })
                                .unwrap_or_default()
                        })
                        .unwrap_or_default();
                    let known_zero = matches!(
                        env_name.to_ascii_lowercase().as_str(),
                        "safe-area-inset-top"
                            | "safe-area-inset-right"
                            | "safe-area-inset-bottom"
                            | "safe-area-inset-left"
                            | "keyboard-inset-top"
                            | "keyboard-inset-right"
                            | "keyboard-inset-bottom"
                            | "keyboard-inset-left"
                            | "viewport-segment-width"
                            | "viewport-segment-height"
                    );
                    if known_zero {
                        return Some(Self::Px(0.0));
                    }
                    if let Some(fb) = parts.get(1) {
                        return Self::from_tokens(fb);
                    }
                    return Some(Self::Px(0.0));
                }
                // `attr(<name> <type>?, <fallback>?)` — typed attribute
                // substitution. We can't read DOM attrs at value-parse
                // time, so return the fallback if present, otherwise
                // None.
                CssToken::Function(name) if name.eq_ignore_ascii_case("attr") => {
                    let end = find_matching_paren(toks, i + 1)?;
                    let args = &toks[i + 1..end];
                    let parts = split_top_level_commas(args);
                    if let Some(fb) = parts.get(1) {
                        return Self::from_tokens(fb);
                    }
                    i = end + 1;
                    continue;
                }
                CssToken::Function(name) if name.eq_ignore_ascii_case("clamp") => {
                    let end = find_matching_paren(toks, i + 1)?;
                    let args = &toks[i + 1..end];
                    let parts = split_top_level_commas(args);
                    if parts.len() == 3 {
                        let min = Self::from_tokens(parts[0]).and_then(Calc::from_length)?;
                        let preferred = Self::from_tokens(parts[1]).and_then(Calc::from_length)?;
                        let max = Self::from_tokens(parts[2]).and_then(Calc::from_length)?;
                        return Some(Self::Clamp(ClampExpr {
                            min,
                            preferred,
                            max,
                        }));
                    }
                    i = end + 1;
                    continue;
                }
                // CSS Values L4 math functions. We fold these into a
                // Length::Px when every argument resolves to a unitless
                // number or a px value at parse time. Mixed-unit cases
                // (e.g. `abs(50% - 20px)`) fall through to None — they
                // would need a viewport-aware evaluator we don't have.
                CssToken::Function(name)
                    if matches!(
                        name.to_ascii_lowercase().as_str(),
                        "min"
                            | "max"
                            | "round"
                            | "mod"
                            | "rem"
                            | "abs"
                            | "sign"
                            | "pow"
                            | "sqrt"
                            | "hypot"
                            | "log"
                            | "exp"
                            | "sin"
                            | "cos"
                            | "tan"
                            | "asin"
                            | "acos"
                            | "atan"
                            | "atan2"
                    ) =>
                {
                    let end = find_matching_paren(toks, i + 1)?;
                    let args = &toks[i + 1..end];
                    let parts = split_top_level_commas(args);
                    // Percent-bearing `min()`/`max()` can't fold to a px
                    // constant at parse time — they must resolve against
                    // the container at layout time. Synthesize a Clamp
                    // (which IS layout-deferred): for the canonical
                    // 2-arg forms,
                    //   `min(a, b)` ≡ clamp(0, a, b)        = max(0, min(a,b))
                    //   `max(a, b)` ≡ clamp(a, b, BIG)      = max(a, min(b,BIG))
                    // This makes `min(100%, 600px)` (Tailwind `max-w-*`,
                    // responsive widths) resolve correctly instead of
                    // dropping the whole declaration. Only the 2-arg
                    // min/max case is diverted; other math funcs and
                    // 3+-arg forms keep the numeric fold below.
                    {
                        let fname = name.to_ascii_lowercase();
                        if (fname == "min" || fname == "max") && parts.len() == 2 {
                            let a = Self::from_tokens(parts[0]).and_then(Calc::from_length);
                            let b = Self::from_tokens(parts[1]).and_then(Calc::from_length);
                            if let (Some(a), Some(b)) = (a, b) {
                                let has_pct = a.percent != 0.0 || b.percent != 0.0;
                                if has_pct {
                                    let clamp = if fname == "min" {
                                        ClampExpr {
                                            min: Calc::default(),
                                            preferred: a,
                                            max: b,
                                        }
                                    } else {
                                        ClampExpr {
                                            min: a,
                                            preferred: b,
                                            max: Calc {
                                                px: 1.0e9,
                                                ..Calc::default()
                                            },
                                        }
                                    };
                                    return Some(Self::Clamp(clamp));
                                }
                            }
                        }
                    }
                    let mut nums: Vec<f32> = Vec::new();
                    let mut bail = false;
                    for p in &parts {
                        let mut got: Option<f32> = None;
                        for t in p.iter() {
                            match t {
                                CssToken::Whitespace => continue,
                                CssToken::Number(n) => {
                                    got = Some(*n as f32);
                                    break;
                                }
                                CssToken::Dimension { value, unit } => {
                                    if unit.eq_ignore_ascii_case("px") {
                                        got = Some(*value as f32);
                                    }
                                    break;
                                }
                                CssToken::Percent(_) => {
                                    bail = true;
                                    break;
                                }
                                CssToken::Function(_) | CssToken::LeftParen => {
                                    if let Some(l) = Self::from_tokens(p) {
                                        if let Self::Px(v) = l {
                                            got = Some(v);
                                        }
                                    }
                                    break;
                                }
                                _ => {}
                            }
                        }
                        match got {
                            Some(v) => nums.push(v),
                            None => {
                                bail = true;
                                break;
                            }
                        }
                    }
                    if bail || nums.is_empty() {
                        i = end + 1;
                        continue;
                    }
                    let result = match name.to_ascii_lowercase().as_str() {
                        "min" => nums.iter().cloned().fold(f32::INFINITY, f32::min),
                        "max" => nums.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                        "abs" => nums[0].abs(),
                        "sign" => nums[0].signum(),
                        "sqrt" => nums[0].sqrt(),
                        "log" => {
                            if nums.len() == 2 {
                                nums[0].log(nums[1])
                            } else {
                                nums[0].ln()
                            }
                        }
                        "exp" => nums[0].exp(),
                        "sin" => nums[0].sin(),
                        "cos" => nums[0].cos(),
                        "tan" => nums[0].tan(),
                        "asin" => nums[0].asin(),
                        "acos" => nums[0].acos(),
                        "atan" => nums[0].atan(),
                        "atan2" if nums.len() == 2 => nums[0].atan2(nums[1]),
                        "pow" if nums.len() == 2 => nums[0].powf(nums[1]),
                        "hypot" => nums.iter().fold(0.0_f32, |a, &b| (a * a + b * b).sqrt()),
                        "mod" if nums.len() == 2 => {
                            // CSS mod: result has sign of divisor.
                            let r = nums[0].rem_euclid(nums[1].abs());
                            if nums[1] < 0.0 { r - nums[1].abs() } else { r }
                        }
                        "rem" if nums.len() == 2 => nums[0] % nums[1],
                        "round" => {
                            // round(strategy?, value, step?) — V1 supports
                            // round(value) and round(value, step). The
                            // strategy keyword (nearest/up/down/to-zero)
                            // is parsed when present but only "nearest"
                            // is honoured.
                            let (v, step) = if nums.len() == 1 {
                                (nums[0], 1.0)
                            } else {
                                (nums[0], nums[1].abs().max(f32::EPSILON))
                            };
                            (v / step).round() * step
                        }
                        _ => {
                            i = end + 1;
                            continue;
                        }
                    };
                    return Some(Self::Px(result));
                }
                CssToken::Ident(s) if s.eq_ignore_ascii_case("auto") => return Some(Self::Auto),
                CssToken::Number(v) if *v == 0.0 => return Some(Self::Zero),
                CssToken::Percent(v) => return Some(Self::Percent(*v as f32)),
                CssToken::Dimension { value, unit } => {
                    let v = *value as f32;
                    return Some(match unit.to_ascii_lowercase().as_str() {
                        "px" => Self::Px(v),
                        "em" => Self::Em(v),
                        "rem" => Self::Rem(v),
                        "vw" => Self::Vw(v),
                        "vh" => Self::Vh(v),
                        "pt" => Self::Pt(v),
                        // Treat unknown dimensional units (cm, mm, in,
                        // pc, ch, ex, …) as pixel so the layout still
                        // gives a reasonable approximation instead of
                        // collapsing to None.
                        _ => Self::Px(v),
                    });
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    pub fn resolve_px(self, em: f32, rem: f32, parent: f32) -> Option<f32> {
        // Approximation: until we plumb the live viewport in, vw/vh
        // default to a 1024x768 window. Browser code can override by
        // calling `resolve_px_with_viewport`.
        self.resolve_px_with_viewport(em, rem, parent, 1024.0, 768.0)
    }

    /// Same as `resolve_px` but takes the real viewport so `vw`/`vh`
    /// resolve correctly. Use this in code that knows the layout config.
    pub fn resolve_px_with_viewport(
        self,
        em: f32,
        rem: f32,
        parent: f32,
        vw_px: f32,
        vh_px: f32,
    ) -> Option<f32> {
        Some(match self {
            Self::Px(v) => v,
            Self::Em(v) => v * em,
            Self::Rem(v) => v * rem,
            Self::Vw(v) => vw_px * v / 100.0,
            Self::Vh(v) => vh_px * v / 100.0,
            Self::Pt(v) => v * 96.0 / 72.0,
            Self::Percent(p) => parent * p / 100.0,
            Self::Zero => 0.0,
            Self::Auto => return None,
            Self::Calc(c) => c.resolve_px_with_viewport(em, rem, parent, vw_px, vh_px),
            Self::Clamp(expr) => {
                let min = expr
                    .min
                    .resolve_px_with_viewport(em, rem, parent, vw_px, vh_px);
                let preferred = expr
                    .preferred
                    .resolve_px_with_viewport(em, rem, parent, vw_px, vh_px);
                let max = expr
                    .max
                    .resolve_px_with_viewport(em, rem, parent, vw_px, vh_px);
                preferred.max(min).min(max.max(min))
            }
        })
    }
}

impl Calc {
    /// Parse `calc()` body tokens (the slice between `calc(` and `)`).
    ///
    /// Implements the CSS Values & Units §8 expression grammar:
    ///   <sum>     = [sign] <product> [ ['+' | '-'] <product> ]*
    ///   <product> = <number> ['*' | '/'] <atom>
    ///             | <atom> [['*' | '/'] <number>]*
    ///   <atom>    = <dimension> | <percent> | <number> | '(' <sum> ')'
    ///             | 'calc(' <sum> ')'
    ///
    /// `*`/`/` bind tighter than `+`/`-`, so `calc(10px + 20px * 2)` = 50px.
    /// Number-first multiplication `calc(2 * 50px)` is also supported.
    pub fn parse(toks: &[CssToken]) -> Option<Self> {
        let (result, mut i) = Self::parse_sum(toks, 0)?;
        // Allow trailing whitespace; any other trailing token is a parse
        // failure (the calc expression was syntactically invalid).
        while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
            i += 1;
        }
        if i < toks.len() {
            return None;
        }
        Some(result)
    }

    /// Sum level: `[sign] product ([+-] product)*`.
    /// Returns (value, position_past_last_consumed_token).
    fn parse_sum(toks: &[CssToken], start: usize) -> Option<(Calc, usize)> {
        let mut i = start;
        while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
            i += 1;
        }
        // Optional leading sign.  Our tokenizer produces Ident("-") for a
        // lone `-` (since `-` is an ident-start byte) and Delim('+') for
        // a lone `+`, so we accept both spellings.
        let leading_sign: f32 = match toks.get(i) {
            Some(CssToken::Delim('+')) => { i += 1; 1.0 }
            Some(CssToken::Ident(s)) if s == "+" => { i += 1; 1.0 }
            Some(CssToken::Delim('-')) => { i += 1; -1.0 }
            Some(CssToken::Ident(s)) if s == "-" => { i += 1; -1.0 }
            _ => 1.0,
        };
        while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
            i += 1;
        }

        let (first, new_i) = Self::parse_product(toks, i)?;
        i = new_i;
        let mut acc = Calc {
            px: leading_sign * first.px,
            em: leading_sign * first.em,
            rem: leading_sign * first.rem,
            vw: leading_sign * first.vw,
            vh: leading_sign * first.vh,
            pt: leading_sign * first.pt,
            percent: leading_sign * first.percent,
        };

        loop {
            let save = i;
            while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
                i += 1;
            }
            let sign: f32 = match toks.get(i) {
                Some(CssToken::Delim('+')) => 1.0,
                Some(CssToken::Delim('-')) => -1.0,
                Some(CssToken::Ident(s)) if s == "-" => -1.0,
                Some(CssToken::Ident(s)) if s == "+" => 1.0,
                _ => {
                    i = save;
                    break;
                }
            };
            i += 1;
            while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
                i += 1;
            }
            match Self::parse_product(toks, i) {
                Some((term, new_i)) => {
                    i = new_i;
                    acc.px += sign * term.px;
                    acc.em += sign * term.em;
                    acc.rem += sign * term.rem;
                    acc.vw += sign * term.vw;
                    acc.vh += sign * term.vh;
                    acc.pt += sign * term.pt;
                    acc.percent += sign * term.percent;
                }
                None => {
                    i = save;
                    break;
                }
            }
        }
        Some((acc, i))
    }

    /// Product level: `number ['*'|'/'] atom` | `atom [['*'|'/'] number]*`.
    /// Returns (value, position_past_last_consumed_token).
    fn parse_product(toks: &[CssToken], start: usize) -> Option<(Calc, usize)> {
        let mut i = start;
        while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
            i += 1;
        }

        // Number-first form: `2 * 50px` or `2 / 50px`.
        // Peek ahead past the number to see if `*` or `/` follows.
        let mut acc = if let Some(CssToken::Number(n)) = toks.get(i) {
            let n = *n as f32;
            let mut j = i + 1;
            while j < toks.len() && matches!(toks[j], CssToken::Whitespace) {
                j += 1;
            }
            if matches!(toks.get(j), Some(CssToken::Delim('*') | CssToken::Delim('/'))) {
                let is_div = matches!(toks[j], CssToken::Delim('/'));
                if is_div && n == 0.0 {
                    return None;
                }
                let factor = if is_div { 1.0 / n } else { n };
                let mut k = j + 1;
                while k < toks.len() && matches!(toks[k], CssToken::Whitespace) {
                    k += 1;
                }
                let (base, new_k) = Self::parse_atom(toks, k)?;
                i = new_k;
                Calc {
                    px: base.px * factor,
                    em: base.em * factor,
                    rem: base.rem * factor,
                    vw: base.vw * factor,
                    vh: base.vh * factor,
                    pt: base.pt * factor,
                    percent: base.percent * factor,
                }
            } else {
                // Bare number with no operator ahead — treat as px.
                i += 1;
                let mut a = Calc::default();
                a.px = n;
                a
            }
        } else {
            let (base, new_i) = Self::parse_atom(toks, i)?;
            i = new_i;
            base
        };

        // Consume trailing `* number` / `/ number` chains.
        loop {
            let save = i;
            while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
                i += 1;
            }
            match toks.get(i) {
                Some(CssToken::Delim('*')) => {
                    i += 1;
                    while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
                        i += 1;
                    }
                    match toks.get(i) {
                        Some(CssToken::Number(n)) => {
                            let mul = *n as f32;
                            acc.px *= mul;
                            acc.em *= mul;
                            acc.rem *= mul;
                            acc.vw *= mul;
                            acc.vh *= mul;
                            acc.pt *= mul;
                            acc.percent *= mul;
                            i += 1;
                        }
                        _ => {
                            i = save;
                            break;
                        }
                    }
                }
                Some(CssToken::Delim('/')) => {
                    i += 1;
                    while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
                        i += 1;
                    }
                    match toks.get(i) {
                        Some(CssToken::Number(n)) if *n != 0.0 => {
                            let div = *n as f32;
                            acc.px /= div;
                            acc.em /= div;
                            acc.rem /= div;
                            acc.vw /= div;
                            acc.vh /= div;
                            acc.pt /= div;
                            acc.percent /= div;
                            i += 1;
                        }
                        _ => {
                            i = save;
                            break;
                        }
                    }
                }
                _ => {
                    i = save;
                    break;
                }
            }
        }
        Some((acc, i))
    }

    /// Atom level: a single value or a parenthesised sub-expression.
    /// Returns (value, position_past_last_consumed_token).
    fn parse_atom(toks: &[CssToken], start: usize) -> Option<(Calc, usize)> {
        let mut i = start;
        while i < toks.len() && matches!(toks[i], CssToken::Whitespace) {
            i += 1;
        }
        match toks.get(i)? {
            CssToken::Dimension { value, unit } => {
                let mut acc = Calc::default();
                acc.add_unit(1.0, *value as f32, &unit.to_ascii_lowercase());
                Some((acc, i + 1))
            }
            CssToken::Percent(p) => {
                let mut acc = Calc::default();
                acc.percent = *p as f32;
                Some((acc, i + 1))
            }
            CssToken::Number(n) => {
                // Bare number — technically invalid in a length calc, but
                // accept as px so lenient parsing doesn't silently drop it.
                let mut acc = Calc::default();
                acc.px = *n as f32;
                Some((acc, i + 1))
            }
            CssToken::LeftParen => {
                let end = find_matching_paren(toks, i + 1)?;
                let (inner, _) = Self::parse_sum(&toks[i + 1..end], 0)?;
                Some((inner, end + 1))
            }
            CssToken::Function(name) if name.eq_ignore_ascii_case("calc") => {
                let end = find_matching_paren(toks, i + 1)?;
                let (inner, _) = Self::parse_sum(&toks[i + 1..end], 0)?;
                Some((inner, end + 1))
            }
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Display {
    Inline,
    Block,
    InlineBlock,
    Flex,
    /// `display: inline-flex` — inline-level flex container (shrink-wraps to
    /// content width; lays out children as flex items).
    InlineFlex,
    Grid,
    /// `display: inline-grid` — inline-level grid container (shrink-wraps to
    /// content width; lays out children as grid items).
    InlineGrid,
    /// `display: table` — generates a table-level box.
    Table,
    /// `display: inline-table` — inline-level table container (shrinks to
    /// content width; lays out children as table items).
    InlineTable,
    /// `display: table-row` — a row of table cells.
    TableRow,
    /// `display: table-cell` — a single cell.
    TableCell,
    /// `display: table-row-group` (`<tbody>`) — fully transparent to
    /// our layout: just collapses into the table flow.
    TableRowGroup,
    None,
}

/// A single track in `grid-template-columns` / `grid-template-rows`.
/// `Fr` carries the fractional weight (`1fr` → `Fr(1.0)`). `Auto` sizes
/// to its content; for V1 we treat it like `Fr(1.0)` so the row still
/// adds up to the available width.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AutoRepeatMode {
    Fit,
    Fill,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AutoRepeatTrack {
    Px(f32),
    Pct(f32),
    Fr(f32),
    Auto,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutoRepeat {
    pub mode: AutoRepeatMode,
    pub min_px: f32,
    pub tracks: Vec<AutoRepeatTrack>,
}

/// Bound variant inside `minmax(<min>, <max>)`. Mirrors
/// `cv_layout::MinMaxBound`. See `cv_layout::GridTrack::MinMax` for
/// why this exists separately from the full `GridTrack` enum.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MinMaxBound {
    Px(f32),
    Pct(f32),
    Fr(f32),
    Auto,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GridTrack {
    Px(f32),
    Pct(f32),
    Fr(f32),
    Auto,
    AutoRepeat(AutoRepeat),
    /// `subgrid` keyword — track sizing inherits from parent grid.
    /// Layout treats subgrid children as participating in the ancestor
    /// grid's track sizing; the track entry is a placeholder.
    Subgrid,
    /// `minmax(<min>, <max>)`. See `cv_layout::GridTrack::MinMax`.
    MinMax {
        min: MinMaxBound,
        max: MinMaxBound,
    },
}

impl GridTrack {
    /// Parse a `grid-template-columns` / `-rows` value. Accepts:
    /// - `100px 1fr 200px`
    /// - `repeat(3, 1fr)`
    /// - `repeat(2, 100px)`
    /// - `auto auto auto`
    /// - mixes thereof. Unknown idents become `Auto`.
    pub fn parse_track_list(toks: &[CssToken]) -> Vec<GridTrack> {
        let mut out: Vec<GridTrack> = Vec::new();
        let mut i = 0;
        while i < toks.len() {
            match &toks[i] {
                CssToken::Whitespace | CssToken::Comma => {
                    i += 1;
                }
                CssToken::Function(name) if name.eq_ignore_ascii_case("repeat") => {
                    // Find the matching close paren. Tokens between are
                    // (count, tracks…). Our tokenizer flattens, so we
                    // walk until ')'.
                    i += 1;
                    let start = i;
                    let mut depth = 1;
                    while i < toks.len() && depth > 0 {
                        match &toks[i] {
                            CssToken::Function(_) | CssToken::LeftParen => depth += 1,
                            CssToken::RightParen => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    let inner = &toks[start..i];
                    // Skip the closing ')'.
                    if i < toks.len() {
                        i += 1;
                    }
                    // inner = count , tracks…
                    let mut count_mode: Option<AutoRepeatMode> = None;
                    let mut j = 0;
                    let mut count: u32 = 1;
                    while j < inner.len() {
                        match &inner[j] {
                            CssToken::Whitespace => j += 1,
                            CssToken::Ident(ident) if ident.eq_ignore_ascii_case("auto-fit") => {
                                count_mode = Some(AutoRepeatMode::Fit);
                                j += 1;
                            }
                            CssToken::Ident(ident) if ident.eq_ignore_ascii_case("auto-fill") => {
                                count_mode = Some(AutoRepeatMode::Fill);
                                j += 1;
                            }
                            CssToken::Number(n) => {
                                count = (*n as u32).max(1);
                                j += 1;
                            }
                            CssToken::Comma => {
                                j += 1;
                                break;
                            }
                            _ => {
                                j += 1;
                            }
                        }
                    }
                    if let Some(mode) = count_mode {
                        if let Some(track) = parse_auto_repeat_track(&inner[j..], mode) {
                            out.push(track);
                            continue;
                        }
                    }
                    let sub = GridTrack::parse_track_list(&inner[j..]);
                    for _ in 0..count {
                        for t in &sub {
                            out.push(t.clone());
                        }
                    }
                }
                CssToken::Function(name) if name.eq_ignore_ascii_case("minmax") => {
                    i += 1;
                    let start = i;
                    let mut depth = 1;
                    while i < toks.len() && depth > 0 {
                        match &toks[i] {
                            CssToken::Function(_) | CssToken::LeftParen => depth += 1,
                            CssToken::RightParen => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    let inner = &toks[start..i];
                    if i < toks.len() {
                        i += 1;
                    }
                    let parts = split_top_level_commas(inner);
                    // Emit the full minmax structure so layout can
                    // apply the min as a floor. Reference: CSS Grid 2
                    // §7.3 — `minmax(<min>, <max>)`. The old V1 path
                    // collapsed to just the max and missed Tailwind's
                    // `minmax(0, 1fr)` floor=0 semantics — `1fr` alone
                    // has implicit min=auto = min-content, which lets
                    // a long-text column blow past its 1fr share.
                    let track_from_part = |part: &[CssToken]| -> GridTrack {
                        GridTrack::parse_track_list(part)
                            .into_iter()
                            .next()
                            .unwrap_or(GridTrack::Auto)
                    };
                    let to_bound = |t: GridTrack| -> Option<GridTrack> {
                        // Either valid as a minmax bound or fall back
                        // to the original. We re-emit as the more
                        // specific MinMax shape only when both parts
                        // map cleanly to MinMaxBound shapes; otherwise
                        // collapse to just the max (V1 behaviour).
                        match t {
                            GridTrack::Px(_)
                            | GridTrack::Pct(_)
                            | GridTrack::Fr(_)
                            | GridTrack::Auto => Some(t),
                            _ => None,
                        }
                    };
                    match (parts.first(), parts.get(1)) {
                        (Some(min_part), Some(max_part)) => {
                            let min_t = to_bound(track_from_part(min_part));
                            let max_t = to_bound(track_from_part(max_part));
                            match (min_t, max_t) {
                                (Some(min_g), Some(max_g)) => {
                                    let to_b = |g: GridTrack| match g {
                                        GridTrack::Px(v) => MinMaxBound::Px(v),
                                        GridTrack::Pct(p) => MinMaxBound::Pct(p),
                                        GridTrack::Fr(f) => MinMaxBound::Fr(f),
                                        _ => MinMaxBound::Auto,
                                    };
                                    out.push(GridTrack::MinMax {
                                        min: to_b(min_g),
                                        max: to_b(max_g),
                                    });
                                }
                                (_, Some(max_g)) => out.push(max_g),
                                _ => out.push(GridTrack::Auto),
                            }
                        }
                        (Some(only), None) => {
                            out.push(track_from_part(only));
                        }
                        _ => out.push(GridTrack::Auto),
                    }
                }
                CssToken::Number(n) if *n == 0.0 => {
                    out.push(GridTrack::Px(0.0));
                    i += 1;
                }
                CssToken::Percent(p) => {
                    out.push(GridTrack::Pct(*p as f32));
                    i += 1;
                }
                CssToken::Dimension { value, unit } => {
                    let v = *value as f32;
                    match unit.to_ascii_lowercase().as_str() {
                        "fr" => out.push(GridTrack::Fr(v.max(0.0))),
                        "px" => out.push(GridTrack::Px(v)),
                        // em/rem at parse time → defer with px approximation
                        // (cascade doesn't know font-size here). Browser
                        // can refine later if needed.
                        "em" | "rem" => out.push(GridTrack::Px(v * 16.0)),
                        "pt" => out.push(GridTrack::Px(v * 96.0 / 72.0)),
                        _ => out.push(GridTrack::Px(v)),
                    }
                    i += 1;
                }
                CssToken::Ident(s) => {
                    let s = s.to_ascii_lowercase();
                    if s == "auto" || s == "min-content" || s == "max-content" {
                        out.push(GridTrack::Auto);
                    } else if s == "subgrid" {
                        out.push(GridTrack::Subgrid);
                    }
                    // minmax(...) and other complex idents skipped for V1.
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }
        out
    }
}

fn parse_auto_repeat_track(toks: &[CssToken], mode: AutoRepeatMode) -> Option<GridTrack> {
    let trimmed = trim_css_whitespace(toks);
    if trimmed.is_empty() {
        return None;
    }
    if let [CssToken::Function(name), rest @ ..] = trimmed {
        if name.eq_ignore_ascii_case("minmax") {
            let end = rest
                .iter()
                .position(|tok| matches!(tok, CssToken::RightParen))?;
            let inner = &rest[..end];
            let parts = split_top_level_commas(inner);
            let min_px = parts
                .first()
                .and_then(|part| Length::from_tokens(part))
                .and_then(|length| length.resolve_px(16.0, 16.0, 0.0))
                .unwrap_or(0.0)
                .max(0.0);
            let preferred = parts
                .get(1)
                .or_else(|| parts.first())
                .and_then(|part| parse_simple_grid_tracks(part))?;
            return Some(GridTrack::AutoRepeat(AutoRepeat {
                mode,
                min_px,
                tracks: preferred,
            }));
        }
    }
    let preferred = parse_simple_grid_tracks(trimmed)?;
    let min_px = preferred
        .iter()
        .map(|track| match track {
            AutoRepeatTrack::Px(v) => (*v).max(0.0),
            _ => 0.0,
        })
        .sum::<f32>();
    Some(GridTrack::AutoRepeat(AutoRepeat {
        mode,
        min_px,
        tracks: preferred,
    }))
}

fn parse_simple_grid_tracks(toks: &[CssToken]) -> Option<Vec<AutoRepeatTrack>> {
    let tracks = GridTrack::parse_track_list(toks);
    if tracks.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(tracks.len());
    for track in tracks {
        match track {
            GridTrack::Px(v) => out.push(AutoRepeatTrack::Px(v)),
            GridTrack::Pct(v) => out.push(AutoRepeatTrack::Pct(v)),
            GridTrack::Fr(v) => out.push(AutoRepeatTrack::Fr(v)),
            GridTrack::Auto => out.push(AutoRepeatTrack::Auto),
            GridTrack::AutoRepeat(_) => return None,
            GridTrack::Subgrid => out.push(AutoRepeatTrack::Auto),
            GridTrack::MinMax { max, .. } => {
                // Inside `repeat(auto-fit/fill, minmax(...))` the max
                // bound drives the per-track sizing. The min bound is
                // used by the auto-fit/fill expansion algorithm
                // (CSS Grid 2 §7.2.2.1) to compute how many tracks
                // fit; we already extract the min separately in
                // `parse_auto_repeat_track` for that purpose.
                match max {
                    MinMaxBound::Px(v) => out.push(AutoRepeatTrack::Px(v)),
                    MinMaxBound::Pct(v) => out.push(AutoRepeatTrack::Pct(v)),
                    MinMaxBound::Fr(v) => out.push(AutoRepeatTrack::Fr(v)),
                    MinMaxBound::Auto => out.push(AutoRepeatTrack::Auto),
                }
            }
        }
    }
    Some(out)
}

fn trim_css_whitespace(toks: &[CssToken]) -> &[CssToken] {
    let start = toks
        .iter()
        .position(|tok| !matches!(tok, CssToken::Whitespace))
        .unwrap_or(toks.len());
    let end = toks
        .iter()
        .rposition(|tok| !matches!(tok, CssToken::Whitespace))
        .map(|idx| idx + 1)
        .unwrap_or(start);
    &toks[start..end]
}

fn split_top_level_commas(toks: &[CssToken]) -> Vec<&[CssToken]> {
    let mut out: Vec<&[CssToken]> = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    for (i, tok) in toks.iter().enumerate() {
        match tok {
            CssToken::Function(_) | CssToken::LeftParen => depth += 1,
            CssToken::RightParen => depth = depth.saturating_sub(1),
            CssToken::Comma if depth == 0 => {
                out.push(&toks[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start <= toks.len() {
        out.push(&toks[start..]);
    }
    out
}

/// CSS `overflow` (and the `-x`/`-y` longhands) — CSS Overflow 3 §3.1.
/// `Visible` is the initial value: content is not clipped and may render
/// outside the box. `Hidden`/`Clip` clip to the padding box (no UA scroll
/// mechanism). `Scroll`/`Auto` establish a *scroll container*: content
/// that overflows the padding box is clipped AND the box gets an
/// independent scroll offset the user (or `element.scrollTop`) can drive.
/// We don't paint a UA scrollbar gutter, so `Scroll` and `Auto` behave
/// identically for layout/paint purposes (both are scrollable).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Overflow {
    #[default]
    Visible,
    Hidden,
    Clip,
    Scroll,
    Auto,
}

impl Overflow {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "visible" => Self::Visible,
                    "hidden" => Self::Hidden,
                    "clip" => Self::Clip,
                    "scroll" => Self::Scroll,
                    "auto" => Self::Auto,
                    _ => return None,
                });
            }
        }
        None
    }

    /// True when this value clips overflow to the padding box.
    /// Per CSS Overflow 3, anything other than `visible` clips.
    pub fn clips(self) -> bool {
        !matches!(self, Self::Visible)
    }

    /// True when this value establishes a *scroll container* (an
    /// independently scrollable region). `scroll` and `auto` do;
    /// `hidden`/`clip`/`visible` do not (hidden can still be scrolled
    /// programmatically in Chrome, but it offers no user scroll
    /// mechanism — we treat `hidden` as non-user-scrollable here, which
    /// matches the common case and keeps wheel routing correct).
    pub fn is_scrollable(self) -> bool {
        matches!(self, Self::Scroll | Self::Auto)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Position {
    #[default]
    Static,
    Relative,
    Absolute,
    Fixed,
    Sticky,
}

impl Position {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "static" => Self::Static,
                    "relative" => Self::Relative,
                    "absolute" => Self::Absolute,
                    "fixed" => Self::Fixed,
                    "sticky" => Self::Sticky,
                    _ => return None,
                });
            }
        }
        None
    }
}

/// `float: left | right | none` — CSS 2.1 §9.5. We don't model `inline-start`
/// or `inline-end` (logical floats); the physical pair is what every site
/// in the wild actually uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FloatSide {
    #[default]
    None,
    Left,
    Right,
}

impl FloatSide {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "left" => Self::Left,
                    "right" => Self::Right,
                    "none" => Self::None,
                    _ => return None,
                });
            }
        }
        None
    }
}

/// `vertical-align` — only the discrete keyword set, not the length /
/// percent form. Used for sup/sub and inline-block alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerticalAlign {
    #[default]
    Baseline,
    Sub,
    Super,
    Top,
    Middle,
    Bottom,
    TextTop,
    TextBottom,
}

impl VerticalAlign {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "baseline" => Self::Baseline,
                    "sub" => Self::Sub,
                    "super" => Self::Super,
                    "top" => Self::Top,
                    "middle" => Self::Middle,
                    "bottom" => Self::Bottom,
                    "text-top" => Self::TextTop,
                    "text-bottom" => Self::TextBottom,
                    _ => return None,
                });
            }
        }
        None
    }
}

/// `clear: left | right | both | none` — CSS 2.1 §9.5.2. A `clear` value
/// of `left` jumps the cleared box past every active left-floated box;
/// `both` jumps past both sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClearMode {
    #[default]
    None,
    Left,
    Right,
    Both,
}

impl ClearMode {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "none" => Self::None,
                    "left" => Self::Left,
                    "right" => Self::Right,
                    "both" => Self::Both,
                    _ => return None,
                });
            }
        }
        None
    }
}

impl Display {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "inline" => Self::Inline,
                    "block" => Self::Block,
                    "inline-block" => Self::InlineBlock,
                    "flex" => Self::Flex,
                    "inline-flex" => Self::InlineFlex,
                    "grid" => Self::Grid,
                    "inline-grid" => Self::InlineGrid,
                    "table" => Self::Table,
                    "inline-table" => Self::InlineTable,
                    "table-row" => Self::TableRow,
                    "table-cell" => Self::TableCell,
                    "table-row-group" | "table-header-group" | "table-footer-group" => {
                        Self::TableRowGroup
                    }
                    // `flow-root` generates a block box with a new BFC.
                    // We don't track BFCs separately yet, so the cheapest
                    // correct behaviour for layout is `Block` — at least
                    // the box is participating in block flow (previously
                    // the parser returned None, leaving display=None and
                    // hiding the box entirely, which is the WORST
                    // outcome for sites that use the modern `flow-root`
                    // clearfix idiom).
                    "flow-root" => Self::Block,
                    // `list-item` generates a block box plus a marker.
                    // Markers come from `::marker` styling separately;
                    // mapping the box itself to `Block` keeps layout
                    // correct even before marker generation lands.
                    "list-item" => Self::Block,
                    // `contents` makes the box disappear and promotes
                    // its children to the parent. Without a real
                    // promotion pass we approximate by treating the box
                    // as Inline so it doesn't disrupt block flow; this
                    // is closer to the spec intent than dropping the
                    // declaration. Real `contents` semantics needs a
                    // separate layout-time transform.
                    "contents" => Self::Inline,
                    "none" => Self::None,
                    _ => return None,
                });
            }
        }
        None
    }
}

/// CSS `visibility` — see CSS 2.2 §11.2.
///
/// Unlike `display: none`, `visibility: hidden` keeps the box in the
/// flow and reserves its size; it just doesn't paint itself or its
/// descendants.  `collapse` is special for table parts; for normal
/// boxes it behaves like `hidden`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Visibility {
    Visible,
    Hidden,
    Collapse,
}

impl Visibility {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "visible" => Self::Visible,
                    "hidden" => Self::Hidden,
                    "collapse" => Self::Collapse,
                    _ => return None,
                });
            }
        }
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FlexDirection {
    Row,
    Column,
    RowReverse,
    ColumnReverse,
}

impl FlexDirection {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "row" => Self::Row,
                    "column" => Self::Column,
                    "row-reverse" => Self::RowReverse,
                    "column-reverse" => Self::ColumnReverse,
                    _ => return None,
                });
            }
        }
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FlexWrap {
    NoWrap,
    Wrap,
    /// `flex-wrap: wrap-reverse` — wrap lines are placed in reverse
    /// cross-axis order (last line at the start, first line at the end).
    WrapReverse,
}

impl FlexWrap {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "nowrap" => Self::NoWrap,
                    "wrap" => Self::Wrap,
                    "wrap-reverse" => Self::WrapReverse,
                    _ => return None,
                });
            }
        }
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JustifyContent {
    Start,
    End,
    Center,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

impl JustifyContent {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        for t in toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "flex-start" | "start" | "left" => Self::Start,
                    "flex-end" | "end" | "right" => Self::End,
                    "center" => Self::Center,
                    "space-between" => Self::SpaceBetween,
                    "space-around" => Self::SpaceAround,
                    "space-evenly" => Self::SpaceEvenly,
                    _ => return None,
                });
            }
        }
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AlignItems {
    Stretch,
    Start,
    End,
    Center,
    /// `align-items: baseline` — align items so their first text
    /// baselines are on the same horizontal line. In layout this uses
    /// the item's `baseline_y` offset when available; items without a
    /// computable baseline fall back to `Start`.
    Baseline,
}

impl AlignItems {
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        // Strip leading/trailing whitespace for the scan.
        let toks: Vec<&CssToken> = toks
            .iter()
            .filter(|t| !matches!(t, CssToken::Whitespace))
            .collect();
        // Two-word forms: `first baseline` / `last baseline`.
        if toks.len() >= 2 {
            if let (CssToken::Ident(a), CssToken::Ident(b)) = (toks[0], toks[1]) {
                let a = a.to_ascii_lowercase();
                let b = b.to_ascii_lowercase();
                match (a.as_str(), b.as_str()) {
                    ("first" | "last", "baseline") => return Some(Self::Baseline),
                    _ => {}
                }
            }
        }
        // Single-keyword forms.
        for t in &toks {
            if let CssToken::Ident(s) = t {
                return Some(match s.to_ascii_lowercase().as_str() {
                    "stretch" => Self::Stretch,
                    "flex-start" | "start" | "self-start" => Self::Start,
                    "flex-end" | "end" | "self-end" => Self::End,
                    "center" => Self::Center,
                    // `baseline`: align on the first text baseline.
                    // The layout engine uses LayoutBox::baseline_y when
                    // present and falls back to Start otherwise.
                    // Two-word forms (`first baseline`, `last baseline`)
                    // are handled above.
                    "baseline" => Self::Baseline,
                    // `normal` is the Flexbox L1 initial value (= stretch
                    // for flex items per spec); recognising it stops
                    // author resets like `align-items: normal` from
                    // silently failing.
                    "normal" => Self::Stretch,
                    _ => return None,
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn vals(s: &str) -> Vec<CssToken> {
        tokenize(s)
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace | CssToken::Eof))
            .collect()
    }

    #[test]
    fn color_names() {
        assert_eq!(
            Color::from_name("red"),
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        // Full CSS Color 4 named-color set including rebeccapurple.
        assert_eq!(
            Color::from_name("rebeccapurple"),
            Some(Color {
                r: 102,
                g: 51,
                b: 153,
                a: 255
            })
        );
        assert_eq!(
            Color::from_name("cornflowerblue"),
            Some(Color {
                r: 100,
                g: 149,
                b: 237,
                a: 255
            })
        );
        assert_eq!(
            Color::from_name("MediumSpringGreen"),
            Some(Color {
                r: 0,
                g: 250,
                b: 154,
                a: 255
            })
        );
        // Unknown ident still returns None.
        assert_eq!(Color::from_name("notacolor"), None);
    }

    #[test]
    fn color_rgb_function() {
        let toks = vals("rgb(255, 128, 0)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(
            c,
            Color {
                r: 255,
                g: 128,
                b: 0,
                a: 255
            }
        );
    }

    #[test]
    fn color_rgba_function_alpha_zero_to_one() {
        let toks = vals("rgba(0, 0, 0, 0.5)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(c.a, 128);
    }

    #[test]
    fn color_oklch_white_is_white() {
        // oklch(1 0 0) = pure white.
        let toks = vals("oklch(1 0 0)");
        let c = Color::from_tokens(&toks).unwrap();
        assert!(
            c.r >= 250 && c.g >= 250 && c.b >= 250,
            "expected ~white, got {c:?}"
        );
    }

    #[test]
    fn color_oklch_black_is_black() {
        let toks = vals("oklch(0 0 0)");
        let c = Color::from_tokens(&toks).unwrap();
        assert!(
            c.r <= 5 && c.g <= 5 && c.b <= 5,
            "expected ~black, got {c:?}"
        );
    }

    #[test]
    fn light_dark_picks_light_branch_by_default() {
        // light-dark(red, blue) → red because we treat the parse-time
        // preference as "light".
        let toks = vals("light-dark(red, blue)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(
            c,
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }
        );
    }

    #[test]
    fn color_function_srgb_components() {
        // color(srgb 0.5 0 0) → mid-red.
        let toks = vals("color(srgb 0.5 0 0)");
        let c = Color::from_tokens(&toks).unwrap();
        assert!(c.r > 120 && c.r < 140, "r={}", c.r);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn color_function_alpha_after_slash() {
        let toks = vals("color(srgb 1 0 0 / 0.5)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(c.r, 255);
        assert_eq!(c.a, 128);
    }

    #[test]
    fn system_color_canvas_is_white() {
        // System colours map to the light-mode palette by default.
        assert_eq!(
            Color::from_name("Canvas"),
            Some(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255
            })
        );
        assert_eq!(
            Color::from_name("CanvasText"),
            Some(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert_eq!(
            Color::from_name("LinkText"),
            Some(Color {
                r: 0,
                g: 0,
                b: 238,
                a: 255
            })
        );
    }

    #[test]
    fn calc_math_min_max_round() {
        // Standalone math functions resolve to Px when args are unitless.
        let l = Length::from_tokens(&vals("min(10, 20)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v - 10.0).abs() < 0.01));
        let l = Length::from_tokens(&vals("max(10, 20)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v - 20.0).abs() < 0.01));
        // round(7, 5) = 5; round(8, 5) = 10.
        let l = Length::from_tokens(&vals("round(7, 5)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v - 5.0).abs() < 0.01));
        let l = Length::from_tokens(&vals("round(8, 5)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v - 10.0).abs() < 0.01));
        // sqrt(16) = 4, hypot(3,4) = 5, abs(-7) = 7, sign(-3) = -1.
        let l = Length::from_tokens(&vals("sqrt(16)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v - 4.0).abs() < 0.01));
        let l = Length::from_tokens(&vals("hypot(3, 4)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v - 5.0).abs() < 0.01));
        let l = Length::from_tokens(&vals("abs(-7)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v - 7.0).abs() < 0.01));
        let l = Length::from_tokens(&vals("sign(-3)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v + 1.0).abs() < 0.01));
        // pow(2, 8) = 256.
        let l = Length::from_tokens(&vals("pow(2, 8)")).unwrap();
        assert!(matches!(l, Length::Px(v) if (v - 256.0).abs() < 0.01));
    }

    #[test]
    fn color_mix_in_srgb_white_blue_50_50() {
        // color-mix(in srgb, white, blue) — equal-weight gamma-sRGB mix.
        // white=(255,255,255) + blue=(0,0,255) at 50/50:
        //   r = 128, g = 128, b = 255.  Chrome reports #8080ff.
        let toks = vals("color-mix(in srgb, white, blue)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(c.r, 128, "r={}", c.r);
        assert_eq!(c.g, 128, "g={}", c.g);
        assert!(c.b > 250, "b={}", c.b);
    }

    #[test]
    fn color_mix_weighted_pulls_toward_first() {
        // 80% red, 20% blue — should bias hard toward red.
        let toks = vals("color-mix(in srgb, red 80%, blue 20%)");
        let c = Color::from_tokens(&toks).unwrap();
        assert!(c.r > c.b, "red {} should beat blue {}", c.r, c.b);
    }

    #[test]
    fn color_hwb_white_keeps_white() {
        let toks = vals("hwb(0 100% 0%)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!((c.r, c.g, c.b), (255, 255, 255));
    }

    #[test]
    fn color_rgb_percent_channels() {
        let toks = vals("rgb(100%, 0%, 0%)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(
            c,
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }
        );
    }

    #[test]
    fn color_hsl_red() {
        let toks = vals("hsl(0, 100%, 50%)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(
            c,
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }
        );
    }

    #[test]
    fn color_hsl_green() {
        let toks = vals("hsl(120, 100%, 50%)");
        let c = Color::from_tokens(&toks).unwrap();
        // Pure green at L=50% / S=100%
        assert_eq!(
            c,
            Color {
                r: 0,
                g: 255,
                b: 0,
                a: 255
            }
        );
    }

    #[test]
    fn color_rgb_modern_slash_alpha() {
        // Vec<CssToken> from `rgb(255 128 0 / 0.5)`
        let toks = vals("rgb(255 128 0 / 0.5)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 128);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 128);
    }

    #[test]
    fn color_hex() {
        assert_eq!(
            Color::from_hash("ff0000"),
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert_eq!(
            Color::from_hash("f00"),
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert_eq!(
            Color::from_hash("80808080"),
            Some(Color {
                r: 128,
                g: 128,
                b: 128,
                a: 128
            })
        );
    }

    #[test]
    fn length_parses() {
        assert_eq!(Length::from_tokens(&vals("12px")), Some(Length::Px(12.0)));
        assert_eq!(Length::from_tokens(&vals("0.5em")), Some(Length::Em(0.5)));
        assert_eq!(
            Length::from_tokens(&vals("50%")),
            Some(Length::Percent(50.0))
        );
        assert_eq!(Length::from_tokens(&vals("auto")), Some(Length::Auto));
    }

    #[test]
    fn length_viewport_units() {
        assert_eq!(Length::from_tokens(&vals("50vw")), Some(Length::Vw(50.0)));
        assert_eq!(Length::from_tokens(&vals("100vh")), Some(Length::Vh(100.0)));
        // 50vw of 1024px viewport = 512px.
        let l = Length::Vw(50.0);
        assert_eq!(
            l.resolve_px_with_viewport(16.0, 16.0, 0.0, 1024.0, 768.0),
            Some(512.0)
        );
        // 100vh of 768px viewport = 768px.
        let l = Length::Vh(100.0);
        assert_eq!(
            l.resolve_px_with_viewport(16.0, 16.0, 0.0, 1024.0, 768.0),
            Some(768.0)
        );
    }

    #[test]
    fn length_pt_to_px() {
        // 12pt = 12 * 96/72 = 16px.
        let l = Length::from_tokens(&vals("12pt")).unwrap();
        assert_eq!(l, Length::Pt(12.0));
        assert_eq!(l.resolve_px(16.0, 16.0, 0.0), Some(16.0));
    }

    #[test]
    fn position_parses() {
        assert_eq!(
            Position::from_tokens(&vals("absolute")),
            Some(Position::Absolute)
        );
        assert_eq!(Position::from_tokens(&vals("FIXED")), Some(Position::Fixed));
        assert_eq!(
            Position::from_tokens(&vals("sticky")),
            Some(Position::Sticky)
        );
        assert_eq!(Position::from_tokens(&vals("flarp")), None);
    }

    #[test]
    fn length_unit_case_insensitive() {
        assert_eq!(Length::from_tokens(&vals("12PX")), Some(Length::Px(12.0)));
        assert_eq!(Length::from_tokens(&vals("3REM")), Some(Length::Rem(3.0)));
    }

    #[test]
    fn calc_percent_minus_px() {
        let l = Length::from_tokens(&vals("calc(100% - 20px)")).unwrap();
        // Resolve against 800px parent: 800 - 20 = 780.
        assert_eq!(l.resolve_px(16.0, 16.0, 800.0), Some(780.0));
    }

    #[test]
    fn calc_em_plus_px() {
        let l = Length::from_tokens(&vals("calc(1em + 8px)")).unwrap();
        // em=16 → 16 + 8 = 24.
        assert_eq!(l.resolve_px(16.0, 16.0, 0.0), Some(24.0));
    }

    #[test]
    fn calc_division_by_number() {
        let l = Length::from_tokens(&vals("calc(100% / 3)")).unwrap();
        // 900px / 3 = 300px — float-tolerant.
        let v = l.resolve_px(16.0, 16.0, 900.0).unwrap();
        assert!((v - 300.0).abs() < 0.001, "got {v}");
    }

    #[test]
    fn calc_multiplication_by_number() {
        let l = Length::from_tokens(&vals("calc(50px * 2)")).unwrap();
        assert_eq!(l.resolve_px(16.0, 16.0, 0.0), Some(100.0));
    }

    #[test]
    fn calc_nested() {
        // calc((100% - 40px) / 2) ⇒ percent half minus 20.
        let l = Length::from_tokens(&vals("calc((100% - 40px) / 2)")).unwrap();
        // 600 parent: (600 - 40)/2 = 280.
        assert_eq!(l.resolve_px(16.0, 16.0, 600.0), Some(280.0));
    }

    #[test]
    fn calc_mixed_add_mul() {
        // calc(10px + 20px * 2) — `*` must bind tighter than `+`.
        // Old flat-loop code gave 60 (scaled the whole acc); correct is 50.
        let l = Length::from_tokens(&vals("calc(10px + 20px * 2)")).unwrap();
        assert_eq!(l.resolve_px(16.0, 16.0, 0.0), Some(50.0));
    }

    #[test]
    fn calc_number_first_mul() {
        // calc(2 * 50px) — number-first multiplication.
        let l = Length::from_tokens(&vals("calc(2 * 50px)")).unwrap();
        assert_eq!(l.resolve_px(16.0, 16.0, 0.0), Some(100.0));
    }

    #[test]
    fn calc_clamp_fn() {
        // clamp(resolves differently) tested elsewhere; here verify that
        // calc with three-operand precedence still works:
        // calc(100% - 10px - 5px) = 600 - 10 - 5 = 585.
        let l = Length::from_tokens(&vals("calc(100% - 10px - 5px)")).unwrap();
        assert_eq!(l.resolve_px(16.0, 16.0, 600.0), Some(585.0));
    }

    #[test]
    fn clamp_resolves_preferred_value_within_bounds() {
        let l = Length::from_tokens(&vals("clamp(200px, 50%, 500px)")).unwrap();
        assert_eq!(l.resolve_px(16.0, 16.0, 600.0), Some(300.0));
    }

    #[test]
    fn clamp_resolves_to_min_and_max_bounds() {
        let low = Length::from_tokens(&vals("clamp(200px, 10%, 500px)")).unwrap();
        assert_eq!(low.resolve_px(16.0, 16.0, 600.0), Some(200.0));

        let high = Length::from_tokens(&vals("clamp(200px, 90%, 500px)")).unwrap();
        assert_eq!(high.resolve_px(16.0, 16.0, 600.0), Some(500.0));
    }

    #[test]
    fn grid_track_list_parses_repeat_fr_tracks() {
        let got = GridTrack::parse_track_list(&vals("repeat(3, 1fr)"));
        assert_eq!(
            got,
            vec![GridTrack::Fr(1.0), GridTrack::Fr(1.0), GridTrack::Fr(1.0)]
        );
    }

    #[test]
    fn grid_track_list_parses_minmax_with_both_bounds() {
        // Per Chrome-divergence #5 fix: `minmax(0, 1fr)` is the
        // canonical Tailwind `grid-cols-N` track — the 0 min sets a
        // hard floor at 0 so the column can shrink to its 1fr share
        // (without it, `1fr` alone has implicit min=auto = min-content
        // and a column with long text grows past its share).
        let got = GridTrack::parse_track_list(&vals("minmax(0, 1fr) 200px"));
        assert_eq!(
            got,
            vec![
                GridTrack::MinMax {
                    min: MinMaxBound::Px(0.0),
                    max: MinMaxBound::Fr(1.0)
                },
                GridTrack::Px(200.0)
            ]
        );
    }

    #[test]
    fn grid_track_list_parses_repeat_minmax_tracks() {
        let got = GridTrack::parse_track_list(&vals("repeat(3, minmax(0, 1fr))"));
        let one = GridTrack::MinMax {
            min: MinMaxBound::Px(0.0),
            max: MinMaxBound::Fr(1.0),
        };
        assert_eq!(got, vec![one.clone(), one.clone(), one]);
    }

    #[test]
    fn grid_track_list_parses_auto_repeat_minmax_tracks() {
        let got = GridTrack::parse_track_list(&vals("repeat(auto-fit, minmax(200px, 1fr))"));
        assert_eq!(
            got,
            vec![GridTrack::AutoRepeat(AutoRepeat {
                mode: AutoRepeatMode::Fit,
                min_px: 200.0,
                tracks: vec![AutoRepeatTrack::Fr(1.0)],
            })]
        );
    }

    #[test]
    fn grid_track_list_parses_auto_repeat_multi_track_patterns() {
        let got = GridTrack::parse_track_list(&vals("repeat(auto-fill, 100px 50px)"));
        assert_eq!(
            got,
            vec![GridTrack::AutoRepeat(AutoRepeat {
                mode: AutoRepeatMode::Fill,
                min_px: 150.0,
                tracks: vec![AutoRepeatTrack::Px(100.0), AutoRepeatTrack::Px(50.0)],
            })]
        );
    }

    #[test]
    fn display_parses() {
        assert_eq!(Display::from_tokens(&vals("block")), Some(Display::Block));
        assert_eq!(
            Display::from_tokens(&vals("inline-block")),
            Some(Display::InlineBlock)
        );
    }

    #[test]
    fn display_inline_flex_and_inline_grid_parse_correctly() {
        // Bug 1: inline-flex must map to InlineFlex, not Flex.
        assert_eq!(
            Display::from_tokens(&vals("inline-flex")),
            Some(Display::InlineFlex)
        );
        // Bug 1: inline-grid must map to InlineGrid, not Grid.
        assert_eq!(
            Display::from_tokens(&vals("inline-grid")),
            Some(Display::InlineGrid)
        );
        // Sanity: block counterparts still correct.
        assert_eq!(Display::from_tokens(&vals("flex")), Some(Display::Flex));
        assert_eq!(Display::from_tokens(&vals("grid")), Some(Display::Grid));
    }

    #[test]
    fn color_mix_srgb_interpolates_in_gamma_space() {
        // Bug 2: color-mix(in srgb, white 50%, black) must be rgb(128,128,128),
        // NOT ~rgb(182,182,182) which is the linearised-sRGB result.
        let toks = vals("color-mix(in srgb, white 50%, black)");
        let c = Color::from_tokens(&toks).unwrap();
        assert_eq!(c.r, 128, "red channel should be 128, got {}", c.r);
        assert_eq!(c.g, 128, "green channel should be 128, got {}", c.g);
        assert_eq!(c.b, 128, "blue channel should be 128, got {}", c.b);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn color_mix_srgb_linear_interpolates_in_linear_space() {
        // In linear-sRGB, white 50% + black → ~rgb(182,182,182) (gamma ~0.7146).
        let toks = vals("color-mix(in srgb-linear, white 50%, black)");
        let c = Color::from_tokens(&toks).unwrap();
        // linear 0.5 gamma-encoded ≈ 186 (1.055*0.5^(1/2.4)-0.055 ≈ 0.7297).
        assert!(
            c.r >= 180 && c.r <= 190,
            "linear-sRGB 50% mix should be ~184, got {}",
            c.r
        );
    }

    #[test]
    fn color_mix_srgb_no_explicit_pct() {
        // Without explicit percentages, each color gets 50%.
        let toks = vals("color-mix(in srgb, red, blue)");
        let c = Color::from_tokens(&toks).unwrap();
        // red=(255,0,0) + blue=(0,0,255) at 50/50 gamma → (128,0,128).
        assert_eq!(c.r, 128);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 128);
    }

    /// Regression: `calc(10px + 20px * 2)` must be 50px, not 60px.
    ///
    /// The old flat-accumulator loop applied `*` to the whole accumulated
    /// value (30px * 2 = 60px) instead of only to the `20px` term.
    /// The two-level parser (parse_sum / parse_product) fixes precedence.
    #[test]
    fn calc_mul_binds_tighter_than_add() {
        // 10 + (20 * 2) = 50, NOT (10 + 20) * 2 = 60
        let l = Length::from_tokens(&vals("calc(10px + 20px * 2)")).unwrap();
        assert_eq!(
            l.resolve_px(16.0, 16.0, 0.0),
            Some(50.0),
            "calc(10px + 20px * 2) must be 50px — * binds tighter than +"
        );
    }

    /// Regression: `calc(2 * 50%)` must be 100%.
    ///
    /// Number-first multiplication: `number * length` is legal CSS and
    /// produces the length scaled by the number.
    #[test]
    fn calc_number_first_mul_percent() {
        // 2 * 50% = 100%
        let l = Length::from_tokens(&vals("calc(2 * 50%)")).unwrap();
        // Resolve against 400px parent: 100% of 400px = 400px.
        assert_eq!(
            l.resolve_px(16.0, 16.0, 400.0),
            Some(400.0),
            "calc(2 * 50%) must be 100% → 400px when parent=400px"
        );
    }

    /// Named regression: `*` must bind tighter than `+` in calc().
    ///
    /// `calc(10px + 20px * 2)` = 10 + (20 * 2) = 50px, NOT (10+20)*2 = 60px.
    #[test]
    fn calc_multiply_precedence_correct() {
        let l = Length::from_tokens(&vals("calc(10px + 20px * 2)")).unwrap();
        assert_eq!(
            l.resolve_px(16.0, 16.0, 0.0),
            Some(50.0),
            "calc(10px + 20px * 2) must be 50px — * binds tighter than +"
        );
    }

    /// Named regression: number-first form `calc(2 * 50px)` must parse and
    /// produce 100px.
    #[test]
    fn calc_number_first_multiply() {
        let l = Length::from_tokens(&vals("calc(2 * 50px)")).unwrap();
        assert_eq!(
            l.resolve_px(16.0, 16.0, 0.0),
            Some(100.0),
            "calc(2 * 50px) must be 100px"
        );
    }

    /// `align-items: baseline` must parse successfully (not silently drop
    /// the declaration and leave `stretch` as the fallback).
    #[test]
    fn align_items_baseline_parses() {
        let toks: Vec<CssToken> = tokenize("baseline")
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace | CssToken::Eof))
            .collect();
        assert_eq!(
            AlignItems::from_tokens(&toks),
            Some(AlignItems::Baseline),
            "align-items: baseline must parse to AlignItems::Baseline"
        );

        // Aliases that map to Baseline:
        let toks2: Vec<CssToken> = tokenize("first baseline")
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace | CssToken::Eof))
            .collect();
        assert_eq!(
            AlignItems::from_tokens(&toks2),
            Some(AlignItems::Baseline),
            "align-items: first baseline must also parse to AlignItems::Baseline"
        );
    }

    /// `flex-wrap: wrap-reverse` must parse successfully (not silently drop
    /// the declaration and leave `nowrap` as the fallback).
    #[test]
    fn flex_wrap_wrap_reverse_parses() {
        let toks: Vec<CssToken> = tokenize("wrap-reverse")
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Whitespace | CssToken::Eof))
            .collect();
        assert_eq!(
            FlexWrap::from_tokens(&toks),
            Some(FlexWrap::WrapReverse),
            "flex-wrap: wrap-reverse must parse to FlexWrap::WrapReverse"
        );
    }
}
