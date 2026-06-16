// Per-quad pixel shader for the cv_gpu GPU rasterization path.
//
// Reproduces cv_gfx's CPU rasterizer (the project's oracle + fallback)
// BYTE-FOR-BYTE so the GPU draw is golden-diff exact:
//   * straight-alpha (non-premultiplied) Porter-Duff source-over, matching
//     cv_gfx::blend_bgra: out_a = sa + da*(1-sa); rgb = round((src*sa +
//     dst*da*(1-sa)) / out_a); a = round(out_a*255). Chrome's viz draws
//     DrawQuads with SkBlendMode::kSrcOver (the same straight-alpha
//     source-over); we do the blend IN-SHADER (not fixed-function OM blend)
//     because cv_gfx un-premultiplies+rounds and the OM premultiplied blend
//     would drift by +/-1 LSB.
//   * an OPAQUE source (a==255) is written VERBATIM (no blend math), matching
//     cv_gfx::fill_rect / blend_pixel's a==255 hard-write fast path.
//   * NO sRGB conversion (the RTV is _UNORM not _SRGB), NO gamma, NO premul.
//
// The backdrop is sampled from a separate SRV (t1) at the pixel's device
// screen position — viz's backdrop/readback technique — so the composited
// result equals "draw this quad over the existing framebuffer" exactly.
//
// kind (cbuffer params.x): 0 = solid, 1 = linear gradient, 2 = image.

Texture2D srcTex   : register(t0); // image source (kind==2)
Texture2D backdrop : register(t1); // pre-quad framebuffer contents
SamplerState samp0 : register(s0); // POINT, CLAMP

cbuffer QuadPS : register(b0) {
    float4 solid;     // straight-alpha RGBA in [0,255] (kind==0)
    float4 grad_from; // RGBA [0,255] at t=0   (kind==1)
    float4 grad_to;   // RGBA [0,255] at t=1   (kind==1)
    float4 grad_axis; // dx, dy, t_min, denom  (kind==1)
    float4 params;    // kind, rect_w, rect_h, _
    float4 vp2;       // vp_w, vp_h, rect_x, rect_y
};

struct VSOut {
    float4 pos    : SV_POSITION;
    float2 uv     : TEXCOORD0;
    float2 screen : TEXCOORD1;
};

// Quantize a [0,255] float channel the way cv_gfx stores it (round), then
// return the [0,1] _UNORM value. The RTV requantization (round(v*255)) is a
// no-op on an already-integer v, so this is bit-exact with the CPU u8 store.
float q255(float v) { return round(clamp(v, 0.0, 255.0)) / 255.0; }

// Straight-alpha source-over of src (RGBA 0..255, a straight) over the
// backdrop dst (RGBA 0..255). Byte-exact reproduction of cv_gfx::blend_bgra,
// including the a==255 hard-write fast path used by fill_rect/blend_pixel.
float4 src_over(float4 src255, float4 dst255) {
    if (src255.a >= 255.0) {
        return float4(q255(src255.r), q255(src255.g), q255(src255.b), 1.0);
    }
    float sa = src255.a / 255.0;
    float da = dst255.a / 255.0;
    float inv = 1.0 - sa;
    float out_a = sa + da * inv;
    if (out_a <= 0.0) return float4(0.0, 0.0, 0.0, 0.0);
    float r = (src255.r * sa + dst255.r * da * inv) / out_a;
    float g = (src255.g * sa + dst255.g * da * inv) / out_a;
    float b = (src255.b * sa + dst255.b * da * inv) / out_a;
    float a = out_a * 255.0;
    return float4(q255(r), q255(g), q255(b), q255(a));
}

float4 PSMain(VSOut i) : SV_TARGET {
    int kind = (int)(params.x + 0.5);

    // Backdrop at this device pixel. i.screen is already the pixel CENTER
    // (rect.xy + uv*rect.zw, uv interpolated at the center), so dividing by
    // the viewport size lands on the matching texel center for POINT sampling.
    // Adding another +0.5 here would shift a half-texel and read the neighbor.
    float2 buv = i.screen / vp2.xy;
    float4 dst255 = backdrop.Sample(samp0, buv) * 255.0;

    if (kind == 0) {
        return src_over(solid, dst255);
    } else if (kind == 1) {
        // Linear gradient. cv_gfx computes the axis projection at the LOCAL
        // pixel center px = (xx - rect_x) + 0.5, py = (yy - rect_y) + 0.5, with
        // rect origin in vp2.zw = (rect_x, rect_y).
        //
        // Two changes drive the GPU toward the CPU's exact fp32 result (the
        // 1-LSB gradient gap shrank from ~40 px to ~11 px on real hardware):
        //   1. Use SV_POSITION.xy (i.pos), which D3D guarantees IS the exact
        //      device pixel center (xx+0.5, yy+0.5) — no interpolation. The old
        //      `i.uv.x * rect_w` form ran through the rasterizer's barycentric
        //      interpolation, which landed a ULP off and flipped floor().
        //   2. `precise` forbids the compiler from contracting a*b+c into a
        //      fused mad and from reassociating, so each op runs in the SAME
        //      order at the SAME fp32 rounding as the CPU oracle.
        // A residual <=1-LSB drift remains at exact floor() boundaries (GPU
        // fp32 division differs from CPU by a ULP there) — the same tolerance
        // Chrome's pixel tests allow for GPU gradients; the gradient is kept
        // OFF the byte-exact GPU path because of it.
        precise float lx = i.pos.x - vp2.z;
        precise float ly = i.pos.y - vp2.w;
        precise float projx = lx * grad_axis.x;
        precise float projy = ly * grad_axis.y;
        precise float num = (projx + projy) - grad_axis.z;
        precise float t = num / grad_axis.w;
        t = clamp(t, 0.0, 1.0);
        // cv_gfx lerps then TRUNCATES each channel (`as u8`), not round.
        precise float r = floor(grad_from.r * (1.0 - t) + grad_to.r * t);
        precise float g = floor(grad_from.g * (1.0 - t) + grad_to.g * t);
        precise float b = floor(grad_from.b * (1.0 - t) + grad_to.b * t);
        precise float a = floor(grad_from.a * (1.0 - t) + grad_to.a * t);
        return src_over(float4(r, g, b, a), dst255);
    } else {
        // Image (textured) quad: sample source 1:1 and source-over.
        float4 s255 = srcTex.Sample(samp0, i.uv) * 255.0;
        return src_over(s255, dst255);
    }
}
