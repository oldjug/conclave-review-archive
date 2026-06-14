//! BMP / DIB decoder.
//!
//! Supports the common Windows BMP formats: 1/4/8/24/32 bpp, BI_RGB
//! (uncompressed) and BI_BITFIELDS for 16/32-bit. Top-down (negative
//! height) and bottom-up rows. RLE compression and OS/2 v1 headers are
//! not handled.

use crate::png::{ImageError, RgbaImage};

const BI_RGB: u32 = 0;
const BI_BITFIELDS: u32 = 3;

pub fn decode_bmp(input: &[u8]) -> Result<RgbaImage, ImageError> {
    if input.len() < 14 {
        return Err(ImageError::BadSignature);
    }
    if &input[..2] != b"BM" {
        return Err(ImageError::BadSignature);
    }
    let pixel_offset = u32::from_le_bytes(input[10..14].try_into().unwrap()) as usize;
    if input.len() < 14 + 4 {
        return Err(ImageError::Truncated);
    }
    let dib_size = u32::from_le_bytes(input[14..18].try_into().unwrap()) as usize;
    if dib_size < 40 || input.len() < 14 + dib_size {
        return Err(ImageError::Malformed("BMP: unsupported DIB header"));
    }
    let dib = &input[14..14 + dib_size];
    let width = i32::from_le_bytes(dib[4..8].try_into().unwrap());
    let height_signed = i32::from_le_bytes(dib[8..12].try_into().unwrap());
    let planes = u16::from_le_bytes(dib[12..14].try_into().unwrap());
    let bpp = u16::from_le_bytes(dib[14..16].try_into().unwrap());
    let compression = u32::from_le_bytes(dib[16..20].try_into().unwrap());
    let colors_used = u32::from_le_bytes(dib[32..36].try_into().unwrap()) as usize;
    if planes != 1 {
        return Err(ImageError::Malformed("BMP: planes != 1"));
    }
    if width <= 0 {
        return Err(ImageError::Malformed("BMP: bad width"));
    }
    let top_down = height_signed < 0;
    let height = height_signed.unsigned_abs();
    if !(bpp == 1 || bpp == 4 || bpp == 8 || bpp == 16 || bpp == 24 || bpp == 32) {
        return Err(ImageError::Malformed("BMP: unsupported bpp"));
    }
    if compression != BI_RGB && compression != BI_BITFIELDS {
        return Err(ImageError::Malformed("BMP: unsupported compression"));
    }

    // Palette comes after the DIB header (and any bit-field masks).
    let palette_start = 14 + dib_size;
    let palette_entries = if bpp <= 8 {
        let max = 1usize << bpp;
        if colors_used == 0 {
            max
        } else {
            colors_used.min(max)
        }
    } else {
        0
    };

    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(palette_entries);
    if palette_entries > 0 {
        if palette_start + palette_entries * 4 > input.len() {
            return Err(ImageError::Truncated);
        }
        for k in 0..palette_entries {
            let off = palette_start + k * 4;
            // BMP palette is BGR0.
            palette.push([input[off + 2], input[off + 1], input[off], 255]);
        }
    }

    // Optional bit-field masks for compression == BI_BITFIELDS. For 16/32
    // bpp we read 3 (or 4 with alpha) u32 masks immediately after the
    // DIB header.
    let (rm, gm, bm, am) = if compression == BI_BITFIELDS {
        let m_start = 14 + dib_size;
        if m_start + 12 > input.len() {
            return Err(ImageError::Truncated);
        }
        let rm = u32::from_le_bytes(input[m_start..m_start + 4].try_into().unwrap());
        let gm = u32::from_le_bytes(input[m_start + 4..m_start + 8].try_into().unwrap());
        let bm = u32::from_le_bytes(input[m_start + 8..m_start + 12].try_into().unwrap());
        let am = if m_start + 16 <= input.len() {
            u32::from_le_bytes(input[m_start + 12..m_start + 16].try_into().unwrap())
        } else {
            0
        };
        (rm, gm, bm, am)
    } else if bpp == 16 {
        // Default 16-bit: 5/5/5/x (xRGB1555).
        (0x7C00, 0x03E0, 0x001F, 0)
    } else {
        (0x00FF_0000, 0x0000_FF00, 0x0000_00FF, 0xFF00_0000)
    };

    let stride = ((width as usize * bpp as usize + 31) / 32) * 4;
    let pix_start = pixel_offset.max(palette_start + palette_entries * 4);
    if pix_start + stride * height as usize > input.len() {
        return Err(ImageError::Truncated);
    }
    let w = width as u32;
    let h = height;
    let mut pixels = vec![0u32; (w * h) as usize];

    let extract = |v: u32, mask: u32| -> u32 {
        if mask == 0 {
            return 0;
        }
        let shift = mask.trailing_zeros();
        let bits = 32 - mask.leading_zeros() - shift;
        let raw = (v & mask) >> shift;
        if bits >= 8 {
            raw >> (bits - 8)
        } else if bits == 0 {
            0
        } else {
            // Replicate to fill 8 bits.
            let mut x = raw << (8 - bits);
            x |= x >> bits;
            x & 0xFF
        }
    };

    for y in 0..h {
        let row_src = pix_start + (y as usize) * stride;
        let dst_y = if top_down { y } else { h - 1 - y };
        let dst_row = (dst_y * w) as usize;
        match bpp {
            1 => {
                for x in 0..w {
                    let byte = input[row_src + (x as usize) / 8];
                    let bit = 7 - ((x as usize) & 7);
                    let idx = ((byte >> bit) & 1) as usize;
                    let p = palette.get(idx).copied().unwrap_or([0, 0, 0, 255]);
                    pixels[dst_row + x as usize] = pack_bgra(p);
                }
            }
            4 => {
                for x in 0..w {
                    let byte = input[row_src + (x as usize) / 2];
                    let nyb = if x & 1 == 0 { byte >> 4 } else { byte & 0x0F } as usize;
                    let p = palette.get(nyb).copied().unwrap_or([0, 0, 0, 255]);
                    pixels[dst_row + x as usize] = pack_bgra(p);
                }
            }
            8 => {
                for x in 0..w {
                    let idx = input[row_src + x as usize] as usize;
                    let p = palette.get(idx).copied().unwrap_or([0, 0, 0, 255]);
                    pixels[dst_row + x as usize] = pack_bgra(p);
                }
            }
            16 => {
                for x in 0..w {
                    let off = row_src + (x as usize) * 2;
                    let v = u16::from_le_bytes([input[off], input[off + 1]]) as u32;
                    let r = extract(v, rm) as u8;
                    let g = extract(v, gm) as u8;
                    let b = extract(v, bm) as u8;
                    let a = if am != 0 { extract(v, am) as u8 } else { 255 };
                    pixels[dst_row + x as usize] = pack_bgra([r, g, b, a]);
                }
            }
            24 => {
                for x in 0..w {
                    let off = row_src + (x as usize) * 3;
                    let b = input[off];
                    let g = input[off + 1];
                    let r = input[off + 2];
                    pixels[dst_row + x as usize] = pack_bgra([r, g, b, 255]);
                }
            }
            32 => {
                for x in 0..w {
                    let off = row_src + (x as usize) * 4;
                    let v = u32::from_le_bytes([
                        input[off],
                        input[off + 1],
                        input[off + 2],
                        input[off + 3],
                    ]);
                    let r = extract(v, rm) as u8;
                    let g = extract(v, gm) as u8;
                    let b = extract(v, bm) as u8;
                    let a = if am != 0 { extract(v, am) as u8 } else { 255 };
                    pixels[dst_row + x as usize] = pack_bgra([r, g, b, a]);
                }
            }
            _ => unreachable!(),
        }
    }

    Ok(RgbaImage {
        width: w,
        height: h,
        pixels,
    })
}

fn pack_bgra(p: [u8; 4]) -> u32 {
    (u32::from(p[3]) << 24) | (u32::from(p[0]) << 16) | (u32::from(p[1]) << 8) | u32::from(p[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_24bpp_2x2(b00: [u8; 3], b01: [u8; 3], b10: [u8; 3], b11: [u8; 3]) -> Vec<u8> {
        // 24bpp BMP, 2×2. Each row = 2 pixels × 3 bytes = 6 bytes, padded
        // to 8 (4-byte boundary). Pixel array stored bottom-up.
        let mut v = Vec::new();
        // File header: BM, size, 0, 0, offset.
        let pix_off: u32 = 14 + 40;
        let pix_size: u32 = 8 * 2;
        let file_size = pix_off + pix_size;
        v.extend_from_slice(b"BM");
        v.extend_from_slice(&file_size.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&pix_off.to_le_bytes());
        // DIB header (BITMAPINFOHEADER): size=40.
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&2i32.to_le_bytes()); // width
        v.extend_from_slice(&2i32.to_le_bytes()); // height (positive = bottom-up)
        v.extend_from_slice(&1u16.to_le_bytes()); // planes
        v.extend_from_slice(&24u16.to_le_bytes()); // bpp
        v.extend_from_slice(&BI_RGB.to_le_bytes()); // compression
        v.extend_from_slice(&pix_size.to_le_bytes()); // sizeImage
        v.extend_from_slice(&2835u32.to_le_bytes()); // xPelsPerMeter
        v.extend_from_slice(&2835u32.to_le_bytes()); // yPelsPerMeter
        v.extend_from_slice(&0u32.to_le_bytes()); // colorsUsed
        v.extend_from_slice(&0u32.to_le_bytes()); // importantColors
        // Pixel rows, bottom-up. Each row 24bpp: BGR per pixel; pad to 4 bytes.
        // Bottom row: (0,0), (1,0) → b00, b01... wait, BMP is bottom-up so
        // the FIRST stored row is the BOTTOM row in image coordinates.
        // Image rows top-to-bottom: row 0 = [b00, b01], row 1 = [b10, b11].
        // Stored order: row 1 first, then row 0.
        let push_row = |v: &mut Vec<u8>, l: [u8; 3], r: [u8; 3]| {
            v.extend_from_slice(&[l[2], l[1], l[0], r[2], r[1], r[0], 0, 0]);
        };
        push_row(&mut v, b10, b11);
        push_row(&mut v, b00, b01);
        v
    }

    #[test]
    fn rejects_bad_signature() {
        assert!(matches!(
            decode_bmp(b"NOTBMP"),
            Err(ImageError::BadSignature)
        ));
    }

    #[test]
    fn decodes_24bpp_2x2() {
        // 4 distinct colors.
        let bmp = build_24bpp_2x2(
            [255, 0, 0],   // r
            [0, 255, 0],   // g
            [0, 0, 255],   // b
            [255, 255, 0], // y
        );
        let img = decode_bmp(&bmp).expect("decode");
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        let bgra = |r: u8, g: u8, b: u8| {
            (255u32 << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
        };
        assert_eq!(img.pixels[0], bgra(255, 0, 0), "tl");
        assert_eq!(img.pixels[1], bgra(0, 255, 0), "tr");
        assert_eq!(img.pixels[2], bgra(0, 0, 255), "bl");
        assert_eq!(img.pixels[3], bgra(255, 255, 0), "br");
    }
}
