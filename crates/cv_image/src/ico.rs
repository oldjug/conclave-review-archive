//! ICO (Windows icon) decoder.
//!
//! Picks the largest sub-image in the file and decodes it as either an
//! embedded PNG (modern Vista+ icons) or a headerless BMP (with the
//! 1-bit AND-mask appended after the colour bits, like legacy icons).
//! Cursor (`.cur`) files share the same container so we accept them too.

use crate::bmp::decode_bmp;
use crate::png::{ImageError, RgbaImage, decode_png};

pub fn decode_ico(input: &[u8]) -> Result<RgbaImage, ImageError> {
    if input.len() < 6 {
        return Err(ImageError::BadSignature);
    }
    let reserved = u16::from_le_bytes([input[0], input[1]]);
    let ty = u16::from_le_bytes([input[2], input[3]]);
    let count = u16::from_le_bytes([input[4], input[5]]) as usize;
    if reserved != 0 || (ty != 1 && ty != 2) || count == 0 {
        return Err(ImageError::BadSignature);
    }
    if input.len() < 6 + count * 16 {
        return Err(ImageError::Truncated);
    }
    // Pick the entry with the largest pixel count.
    let mut best: Option<(u32, usize, usize)> = None; // (pixels, offset, size)
    for k in 0..count {
        let e = &input[6 + k * 16..6 + k * 16 + 16];
        let w = if e[0] == 0 { 256u32 } else { e[0] as u32 };
        let h = if e[1] == 0 { 256u32 } else { e[1] as u32 };
        let size = u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
        let off = u32::from_le_bytes(e[12..16].try_into().unwrap()) as usize;
        if off + size > input.len() {
            continue;
        }
        let pixels = w * h;
        if best.map(|b| b.0 < pixels).unwrap_or(true) {
            best = Some((pixels, off, size));
        }
    }
    let (_pixels, off, size) = best.ok_or(ImageError::Truncated)?;
    let slice = &input[off..off + size];

    // PNG sub-image is the easy case.
    if slice.len() >= 8 && slice[..8] == [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'] {
        return decode_png(slice);
    }

    // Headerless BMP. The first 40 bytes are the BITMAPINFOHEADER. The
    // declared height is double the real height (it counts the AND-mask
    // rows). We synthesize a BITMAPFILEHEADER and let `decode_bmp`
    // handle the rest. The AND-mask is then applied as alpha for icons
    // whose colour bits don't carry alpha.
    if slice.len() < 40 {
        return Err(ImageError::Truncated);
    }
    let dib_size = u32::from_le_bytes(slice[0..4].try_into().unwrap()) as usize;
    let declared_h = i32::from_le_bytes(slice[8..12].try_into().unwrap());
    let bpp = u16::from_le_bytes(slice[14..16].try_into().unwrap()) as u32;
    let real_h = (declared_h.abs() / 2) as i32;
    let width = i32::from_le_bytes(slice[4..8].try_into().unwrap());
    if width <= 0 || real_h <= 0 {
        return Err(ImageError::Malformed("ICO: bad dims"));
    }
    // Build a fake file with corrected height.
    let mut fake = Vec::with_capacity(slice.len() + 14);
    let pix_off: u32 = 14
        + dib_size as u32
        + match bpp {
            1 => 8,
            4 => 64,
            8 => 1024,
            _ => 0,
        };
    fake.extend_from_slice(b"BM");
    let size_u32 = (slice.len() + 14) as u32;
    fake.extend_from_slice(&size_u32.to_le_bytes());
    fake.extend_from_slice(&0u16.to_le_bytes());
    fake.extend_from_slice(&0u16.to_le_bytes());
    fake.extend_from_slice(&pix_off.to_le_bytes());
    // DIB header with corrected height.
    fake.extend_from_slice(&slice[..8]);
    fake.extend_from_slice(&real_h.to_le_bytes());
    fake.extend_from_slice(&slice[12..]);

    let mut img = decode_bmp(&fake)?;

    // Apply 1-bit AND-mask as alpha when bpp < 32. The mask sits right
    // after the colour rows; each row is width bits padded to 32-bit.
    if bpp != 32 {
        let stride = ((width as usize * bpp as usize + 31) / 32) * 4;
        let colour_bytes = stride * real_h as usize;
        let palette_bytes = match bpp {
            1 => 8,
            4 => 64,
            8 => 1024,
            _ => 0,
        };
        let mask_off = dib_size + palette_bytes + colour_bytes;
        let mask_stride = ((width as usize + 31) / 32) * 4;
        if mask_off + mask_stride * real_h as usize <= slice.len() {
            for y in 0..real_h as usize {
                let src_y = real_h as usize - 1 - y; // ICO is bottom-up
                let src_row = mask_off + src_y * mask_stride;
                for x in 0..width as usize {
                    let bit_off = x;
                    let b = slice[src_row + bit_off / 8];
                    let bit = (b >> (7 - (bit_off & 7))) & 1;
                    if bit == 1 {
                        let p = &mut img.pixels[y * width as usize + x];
                        *p &= 0x00FF_FFFF; // zero alpha channel
                    }
                }
            }
        }
    }

    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_ico() {
        assert!(matches!(decode_ico(b"NOPE"), Err(ImageError::BadSignature)));
    }

    #[test]
    fn rejects_empty_directory() {
        let v = [0u8; 6];
        assert!(matches!(decode_ico(&v), Err(ImageError::BadSignature)));
    }
}
