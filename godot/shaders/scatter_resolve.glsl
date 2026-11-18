#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D color0;
layout(rgba32f, set = 0, binding = 1) uniform readonly image2D color1;
layout(rgba32f, set = 0, binding = 2) uniform readonly image2D aux0;
layout(rgba32f, set = 0, binding = 3) uniform readonly image2D aux1;
layout(rgba32f, set = 0, binding = 4) uniform writeonly image2D result;
layout(rgba32f, set = 0, binding = 5) uniform writeonly image2D validity;
layout(push_constant, std430) uniform Params {
    float near_z;
    float far_z;
    float fraction;
    float depth_tolerance;
    float color_tolerance;
    float radius;
    float policy;
    float pad;
}
params;
layout(r32ui, set = 0, binding = 6) uniform readonly uimage2D owner0;
layout(r32ui, set = 0, binding = 7) uniform readonly uimage2D owner1;
bool candidate(ivec2 p, bool second, out vec4 color, out float depth) {
    uint id = second ? imageLoad(owner1, p).x : imageLoad(owner0, p).x;
    if (id == 0xffffffffu) {
        depth = 0;
        color = vec4(0);
        return false;
    }
    ivec2 size = imageSize(result);
    ivec2 q = ivec2(int(id) % size.x, int(id) / size.x);
    vec4 a = second ? imageLoad(aux1, q) : imageLoad(aux0, q);
    color = second ? imageLoad(color1, q) : imageLoad(color0, q);
    depth = params.near_z * params.far_z / (params.near_z + a.x * (params.far_z - params.near_z));
    return true;
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(result))))
        return;
    vec4 c0, c1;
    float d0, d1;
    bool v0 = candidate(p, false, c0, d0);
    bool v1 = candidate(p, true, c1, d1);
    bool agree =
        v0 && v1 && abs(d0 - d1) <= params.depth_tolerance * max(d0, d1) &&
        max(max(abs(c0.r - c1.r), abs(c0.g - c1.g)), abs(c0.b - c1.b)) <= params.color_tolerance;
    vec4 value = agree ? mix(c0, c1, params.fraction) : imageLoad(color0, p);
    float code = agree ? 1 : 0;
    if (params.policy == 1.0 && v0 != v1) {
        value = v0 ? c0 : c1;
        code = 2;
    }
    if (params.policy == 3.0) {
        value = imageLoad(color0, p);
        code = 0;
    }
    if (params.policy == 4.0) {
        value = mix(imageLoad(color0, p), imageLoad(color1, p), params.fraction);
        code = 0;
    }
    imageStore(result, p, value);
    imageStore(validity, p, vec4(code, v0 ? 1 : 0, v1 ? 1 : 0, 1));
}
