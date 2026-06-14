//! Integration test that decodes a real JPEG produced by .NET
//! `System.Drawing`. The 16x16 fixture has four colored quadrants:
//! red, green, blue, white.

use cv_image::decode_jpeg;

const JPEG_BYTES: &[u8] = include_bytes!("test16.jpg");

#[test]
fn decodes_real_jpeg() {
    let img = decode_jpeg(JPEG_BYTES).expect("decode jpeg");
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.pixels.len(), 16 * 16);

    // JPEG is lossy and chroma subsampled, so individual pixels can drift
    // significantly. Average each quadrant and check that the dominant
    // channel is roughly right.
    let extract = |x0: u32, y0: u32| -> (u32, u32, u32) {
        let mut sr = 0u32;
        let mut sg = 0u32;
        let mut sb = 0u32;
        for y in y0..y0 + 8 {
            for x in x0..x0 + 8 {
                let p = img.pixels[(y * 16 + x) as usize];
                sr += (p >> 16) & 0xFF;
                sg += (p >> 8) & 0xFF;
                sb += p & 0xFF;
            }
        }
        (sr / 64, sg / 64, sb / 64)
    };

    let (r, g, b) = extract(0, 0); // red quadrant
    assert!(
        r > g + 30 && r > b + 30,
        "top-left should be reddish, got ({r}, {g}, {b})"
    );

    let (r, g, b) = extract(8, 0); // green
    assert!(
        g > r + 30 && g > b + 30,
        "top-right should be greenish, got ({r}, {g}, {b})"
    );

    let (r, g, b) = extract(0, 8); // blue
    assert!(
        b > r + 30 && b > g + 30,
        "bottom-left should be bluish, got ({r}, {g}, {b})"
    );

    let (r, g, b) = extract(8, 8); // white
    assert!(
        r > 200 && g > 200 && b > 200,
        "bottom-right should be white-ish, got ({r}, {g}, {b})"
    );
}
