// Per-quad vertex shader for the cv_gpu GPU rasterization path (Chrome
// cc/viz DrawQuad model: a quad's rect lives in target/device-pixel space
// and the vertex shader transforms it to clip space).
//
// NO vertex/index buffer: a 4-vertex SV_VertexID triangle strip. The quad's
// device-pixel rect + the viewport size come from a constant buffer. We emit
// BOTH a [0,1] local uv (across the quad, for gradient/image sampling) and
// the absolute device-pixel screen position (for backdrop source-over).
//
// Clip-space mapping matches D3D's top-left-origin pixel convention used by
// present_bgra / cv_gfx::Bitmap: device y grows DOWN, clip y grows UP, so
// clip.y = 1 - 2*(py/vp_h). DO NOT "fix" the V flip — it is correct.

cbuffer QuadCB : register(b0) {
    float4 rect;     // x, y, w, h   (device pixels)
    float4 viewport; // vp_w, vp_h, _, _ (device pixels)
};

struct VSOut {
    float4 pos    : SV_POSITION;
    float2 uv     : TEXCOORD0; // [0,1] across the quad
    float2 screen : TEXCOORD1; // absolute device-pixel position
};

VSOut VSMain(uint vid : SV_VertexID) {
    VSOut o;
    float2 uv = float2((vid & 1) ? 1.0 : 0.0, (vid & 2) ? 1.0 : 0.0);
    o.uv = uv;
    float2 px = rect.xy + uv * rect.zw; // device-pixel corner
    o.screen = px;
    float2 ndc = float2(px.x / viewport.x * 2.0 - 1.0,
                        1.0 - px.y / viewport.y * 2.0);
    o.pos = float4(ndc, 0.0, 1.0);
    return o;
}
