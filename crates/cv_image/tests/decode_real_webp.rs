//! Decodes a real lossy (VP8) WebP produced by ffmpeg/libwebp. The 64x64
//! fixture has four colored quadrants: red (TL), green (TR), blue (BL),
//! white (BR). VP8 is lossy + chroma-subsampled so we average each quadrant
//! and assert the dominant channel, exactly like the JPEG fixture test.

use cv_image::decode_webp;

const WEBP_BYTES: &[u8] = include_bytes!("test_lossy_quad.webp");

#[test]
fn decodes_real_lossy_webp() {
    let img = decode_webp(WEBP_BYTES).expect("decode lossy webp");
    assert_eq!(img.width, 64, "width from VP8 header");
    assert_eq!(img.height, 64, "height from VP8 header");
    assert_eq!(img.pixels.len(), 64 * 64);

    // Not all-one-color (would indicate a flat placeholder).
    let first = img.pixels[0];
    assert!(
        img.pixels.iter().any(|&p| p != first),
        "decoded image is a flat color — looks like a placeholder, not a real decode"
    );

    let avg = |x0: u32, y0: u32| -> (u32, u32, u32) {
        let (mut sr, mut sg, mut sb) = (0u32, 0u32, 0u32);
        for y in y0 + 4..y0 + 28 {
            for x in x0 + 4..x0 + 28 {
                let p = img.pixels[(y * 64 + x) as usize];
                sr += (p >> 16) & 0xFF;
                sg += (p >> 8) & 0xFF;
                sb += p & 0xFF;
            }
        }
        let n = 24 * 24;
        (sr / n, sg / n, sb / n)
    };

    let (r, g, b) = avg(0, 0);
    assert!(r > g + 40 && r > b + 40, "TL should be red: ({r},{g},{b})");
    let (r, g, b) = avg(32, 0);
    assert!(g > r + 40 && g > b + 40, "TR should be green: ({r},{g},{b})");
    let (r, g, b) = avg(0, 32);
    assert!(b > r + 40 && b > g + 40, "BL should be blue: ({r},{g},{b})");
    let (r, g, b) = avg(32, 32);
    assert!(r > 180 && g > 180 && b > 180, "BR should be white: ({r},{g},{b})");
}
