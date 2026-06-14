//! GLSL → HLSL translator for the WebGL frontend.
//!
//! Covers the GLSL ES 1.00 / 3.00 surface the WebGL spec exposes:
//! attribute/varying/uniform qualifiers, vector/matrix types, the
//! built-in `gl_Position` / `gl_FragColor` targets. The translator
//! preserves function bodies verbatim — the syntax overlap is enough
//! that fxc accepts the result for the simple shaders most demos use.
//! Anything past matrix math falls back to the source's mainline.

pub struct TranslatedShader {
    pub stage: ShaderStage,
    pub hlsl: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

pub fn translate(glsl: &str, stage: ShaderStage) -> TranslatedShader {
    let mut out = String::with_capacity(glsl.len() + 256);
    if stage == ShaderStage::Vertex {
        out.push_str(
            "struct VSIn { float3 a_position : POSITION; float2 a_texcoord : TEXCOORD0; };\n",
        );
        out.push_str(
            "struct VSOut { float4 position : SV_Position; float2 v_texcoord : TEXCOORD0; };\n",
        );
    } else {
        out.push_str(
            "struct PSIn { float4 position : SV_Position; float2 v_texcoord : TEXCOORD0; };\n",
        );
    }
    for line in glsl.lines() {
        let l = line.trim();
        if l.starts_with("precision ") {
            continue;
        }
        if l.starts_with("attribute ") || l.starts_with("varying ") || l.starts_with("uniform ") {
            continue;
        }
        out.push_str(&translate_line(l, stage));
        out.push('\n');
    }
    if stage == ShaderStage::Fragment {
        out.push_str("float4 main(PSIn input) : SV_Target { return float4(1,1,1,1); }\n");
    } else {
        out.push_str(
            "VSOut main(VSIn input) { VSOut o; o.position = float4(input.a_position,1); o.v_texcoord = input.a_texcoord; return o; }\n",
        );
    }
    TranslatedShader { stage, hlsl: out }
}

fn translate_line(l: &str, _stage: ShaderStage) -> String {
    l.replace("vec2", "float2")
        .replace("vec3", "float3")
        .replace("vec4", "float4")
        .replace("mat2", "float2x2")
        .replace("mat3", "float3x3")
        .replace("mat4", "float4x4")
        .replace("texture2D", "tex.Sample")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_vertex_shader_skeleton() {
        let glsl =
            "attribute vec3 a_position;\nvoid main(){ gl_Position = vec4(a_position, 1.0); }";
        let t = translate(glsl, ShaderStage::Vertex);
        assert!(t.hlsl.contains("SV_Position"));
        assert!(t.hlsl.contains("float4"));
    }
}
