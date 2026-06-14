//! Integration test that decodes a real PNG produced by .NET
//! `System.Drawing` (see `tests/build_fixture.ps1` or the PowerShell
//! invocation that generated `test4x3.png`). The fixture is a 4×3 image
//! with row 0 = [red, green, blue, white] and rows 1–2 filled with
//! (64, 128, 192).

use cv_image::decode_png;

const PNG_BYTES: &[u8] = include_bytes!("test4x3.png");

#[test]
fn decodes_real_png() {
    let img = decode_png(PNG_BYTES).expect("decode");
    assert_eq!(img.width, 4);
    assert_eq!(img.height, 3);
    assert_eq!(img.pixels.len(), 12);

    // BGRA u32 layout (low byte = blue).
    let bgra = |r: u8, g: u8, b: u8, a: u8| -> u32 {
        (u32::from(a) << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
    };
    assert_eq!(img.pixels[0], bgra(255, 0, 0, 255), "row0 col0 = red");
    assert_eq!(img.pixels[1], bgra(0, 255, 0, 255), "row0 col1 = green");
    assert_eq!(img.pixels[2], bgra(0, 0, 255, 255), "row0 col2 = blue");
    assert_eq!(img.pixels[3], bgra(255, 255, 255, 255), "row0 col3 = white");
    for i in 4..12 {
        assert_eq!(
            img.pixels[i],
            bgra(64, 128, 192, 255),
            "row >=1 col {} = (64,128,192)",
            i % 4
        );
    }
}
