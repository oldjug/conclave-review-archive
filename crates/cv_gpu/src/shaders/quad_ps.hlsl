Texture2D tex0 : register(t0);
SamplerState samp0 : register(s0);
struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
float4 PSMain(VSOut i) : SV_TARGET {
    return tex0.Sample(samp0, i.uv);
}
float4 PSMain_solid(VSOut i) : SV_TARGET { return float4(1.0, 0.0, 0.0, 1.0); }
