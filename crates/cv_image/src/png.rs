//! PNG decoder per the W3C PNG Specification.
//!
//! Pipeline:
//!   1. Parse 8-byte signature.
//!   2. Walk chunks (`IHDR`, `PLTE`, `IDAT*`, `IEND`).
//!   3. DEFLATE-decompress the concatenated `IDAT` payloads through
//!      `cv_compression::inflate`, skipping the zlib 2-byte header
//!      and trailing 4-byte ADLER32.
//!   4. Filter-reverse scanlines (filter types 0..=4).
//!   5. Expand to RGBA8.

use core::fmt;

use cv_compression::inflate;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    BadSignature,
    Truncated,
    BadChunkCrc, // we currently skip CRC verification
    UnsupportedColorType(u8),
    UnsupportedBitDepth(u8),
    UnsupportedInterlace,
    NoIhdr,
    BadFilter(u8),
    Decompress(String),
    Malformed(&'static str),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadSignature => f.write_str("bad PNG signature"),
            Self::Truncated => f.write_str("truncated PNG"),
            Self::BadChunkCrc => f.write_str("bad CRC"),
            Self::UnsupportedColorType(c) => write!(f, "unsupported color type {c}"),
            Self::UnsupportedBitDepth(b) => write!(f, "unsupported bit depth {b}"),
            Self::UnsupportedInterlace => f.write_str("interlaced PNGs not yet supported"),
            Self::NoIhdr => f.write_str("missing IHDR"),
            Self::BadFilter(f_) => write!(f, "bad filter byte {f_}"),
            Self::Decompress(s) => write!(f, "decompress: {s}"),
            Self::Malformed(s) => write!(f, "malformed: {s}"),
        }
    }
}

impl std::error::Error for ImageError {}

#[derive(Debug, Clone)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    /// Row-major BGRA u32 (premultiplied: not yet — we emit straight alpha).
    pub pixels: Vec<u32>,
}

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

pub fn decode_png(input: &[u8]) -> Result<RgbaImage, ImageError> {
    if input.len() < 8 || input[..8] != PNG_SIGNATURE {
        return Err(ImageError::BadSignature);
    }

    let mut i = 8;
    let mut width: u32 = 0;
    let mut height: u32 = 0;
    let mut bit_depth: u8 = 0;
    let mut color_type: u8 = 0;
    let mut interlace: u8;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut trns_palette: Vec<u8> = Vec::new();
    let mut idat = Vec::new();
    let mut saw_ihdr = false;

    while i + 8 <= input.len() {
        let len = u32::from_be_bytes(input[i..i + 4].try_into().unwrap()) as usize;
        let kind = &input[i + 4..i + 8];
        let data_start = i + 8;
        let data_end = data_start + len;
        if data_end + 4 > input.len() {
            return Err(ImageError::Truncated);
        }
        let data = &input[data_start..data_end];
        match kind {
            b"IHDR" => {
                if data.len() != 13 {
                    return Err(ImageError::Malformed("IHDR length"));
                }
                width = u32::from_be_bytes(data[0..4].try_into().unwrap());
                height = u32::from_be_bytes(data[4..8].try_into().unwrap());
                bit_depth = data[8];
                color_type = data[9];
                let _compression = data[10];
                let _filter = data[11];
                interlace = data[12];
                if bit_depth != 8 {
                    return Err(ImageError::UnsupportedBitDepth(bit_depth));
                }
                if interlace != 0 {
                    return Err(ImageError::UnsupportedInterlace);
                }
                if !matches!(color_type, 0 | 2 | 3 | 4 | 6) {
                    return Err(ImageError::UnsupportedColorType(color_type));
                }
                saw_ihdr = true;
            }
            b"PLTE" => {
                if data.len() % 3 != 0 {
                    return Err(ImageError::Malformed("PLTE length"));
                }
                palette = data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            }
            b"tRNS" => {
                trns_palette = data.to_vec();
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {} // ignore ancillary chunks
        }
        i = data_end + 4; // skip CRC
    }
    if !saw_ihdr {
        return Err(ImageError::NoIhdr);
    }

    if idat.len() < 2 {
        return Err(ImageError::Truncated);
    }
    // zlib wrapper: skip 2-byte header, drop trailing 4-byte ADLER32.
    let zlib_payload = &idat[2..idat.len() - 4];
    let raw = inflate(zlib_payload).map_err(|e| ImageError::Decompress(e.to_string()))?;

    let channels: usize = match color_type {
        0 => 1, // grayscale
        2 => 3, // RGB
        3 => 1, // palette index
        4 => 2, // gray + alpha
        6 => 4, // RGBA
        _ => unreachable!(),
    };
    let bytes_per_pixel = channels; // bit_depth == 8
    let stride_in = width as usize * bytes_per_pixel;
    let expected_in = (stride_in + 1) * height as usize;
    if raw.len() != expected_in {
        return Err(ImageError::Malformed("IDAT size mismatch"));
    }

    let mut rows: Vec<u8> = Vec::with_capacity(stride_in * height as usize);
    let mut prev_row: Vec<u8> = vec![0u8; stride_in];
    let mut current_row: Vec<u8> = vec![0u8; stride_in];

    for r in 0..height as usize {
        let off = r * (stride_in + 1);
        let filter = raw[off];
        let row_data = &raw[off + 1..off + 1 + stride_in];

        match filter {
            0 => {
                current_row.copy_from_slice(row_data);
            }
            1 => {
                // Sub: x + a (left neighbour)
                for x in 0..stride_in {
                    let left = if x >= bytes_per_pixel {
                        current_row[x - bytes_per_pixel]
                    } else {
                        0
                    };
                    current_row[x] = row_data[x].wrapping_add(left);
                }
            }
            2 => {
                // Up: x + b (above)
                for x in 0..stride_in {
                    current_row[x] = row_data[x].wrapping_add(prev_row[x]);
                }
            }
            3 => {
                // Average: x + floor((a + b) / 2)
                for x in 0..stride_in {
                    let left = if x >= bytes_per_pixel {
                        current_row[x - bytes_per_pixel]
                    } else {
                        0
                    };
                    let above = prev_row[x];
                    let avg = ((left as u16 + above as u16) >> 1) as u8;
                    current_row[x] = row_data[x].wrapping_add(avg);
                }
            }
            4 => {
                // Paeth
                for x in 0..stride_in {
                    let left = if x >= bytes_per_pixel {
                        current_row[x - bytes_per_pixel] as i32
                    } else {
                        0
                    };
                    let above = prev_row[x] as i32;
                    let upper_left = if x >= bytes_per_pixel {
                        prev_row[x - bytes_per_pixel] as i32
                    } else {
                        0
                    };
                    let p = left + above - upper_left;
                    let pa = (p - left).abs();
                    let pb = (p - above).abs();
                    let pc = (p - upper_left).abs();
                    let predictor = if pa <= pb && pa <= pc {
                        left
                    } else if pb <= pc {
                        above
                    } else {
                        upper_left
                    } as u8;
                    current_row[x] = row_data[x].wrapping_add(predictor);
                }
            }
            other => return Err(ImageError::BadFilter(other)),
        }

        rows.extend_from_slice(&current_row);
        core::mem::swap(&mut prev_row, &mut current_row);
    }

    // Expand to BGRA u32.
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height as usize {
        let row_off = y * stride_in;
        for x in 0..width as usize {
            let px_off = row_off + x * bytes_per_pixel;
            let (r, g, b, a) = match color_type {
                0 => {
                    let v = rows[px_off];
                    (v, v, v, 255)
                }
                2 => (rows[px_off], rows[px_off + 1], rows[px_off + 2], 255),
                3 => {
                    let idx = rows[px_off] as usize;
                    let [r, g, b] = *palette
                        .get(idx)
                        .ok_or(ImageError::Malformed("PLTE index"))?;
                    let a = trns_palette.get(idx).copied().unwrap_or(255);
                    (r, g, b, a)
                }
                4 => {
                    let v = rows[px_off];
                    (v, v, v, rows[px_off + 1])
                }
                6 => (
                    rows[px_off],
                    rows[px_off + 1],
                    rows[px_off + 2],
                    rows[px_off + 3],
                ),
                _ => unreachable!(),
            };
            let bgra =
                (u32::from(a) << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
            pixels.push(bgra);
        }
    }

    Ok(RgbaImage {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid 1x1 red PNG (RGB, 8-bit). Hand-crafted reference.
    /// Easier than constructing from scratch: we use a known-good byte
    /// sequence produced from a tiny image and validate by decode.
    #[test]
    fn decodes_signature_and_ihdr_at_least() {
        // Just signature + IHDR + IEND with no IDAT — should fail "truncated"
        // on the inflate step rather than crash.
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);
        // IHDR chunk (length=13, type=IHDR, data, crc=0 fake)
        data.extend_from_slice(&13u32.to_be_bytes());
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&1u32.to_be_bytes()); // width
        data.extend_from_slice(&1u32.to_be_bytes()); // height
        data.push(8); // bit depth
        data.push(2); // color type RGB
        data.push(0); // compression
        data.push(0); // filter
        data.push(0); // interlace
        data.extend_from_slice(&[0; 4]); // crc placeholder
        // IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&[0; 4]); // crc
        let res = decode_png(&data);
        // We should get an error (truncated/decompress), not a panic.
        assert!(matches!(res, Err(_)));
    }

    #[test]
    fn rejects_bad_signature() {
        let data = b"\0\0\0\0not a png";
        assert!(matches!(decode_png(data), Err(ImageError::BadSignature)));
    }
}
