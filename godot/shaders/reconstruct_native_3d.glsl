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
bool candidate(ivec2 p, bool second, out vec4 color, out float depth) {
    ivec2 size = imageSize(result);
    float best = -1;
    bool found = false;
    int radius = int(params.radius);
    for (int y = -radius; y <= radius; y++)
        for (int x = -radius; x <= radius; x++) {
            ivec2 q = p + ivec2(x, y);
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size)))
                continue;
            vec4 a = second ? imageLoad(aux1, q) : imageLoad(aux0, q);
            ivec2 destination =
                ivec2(floor(vec2(q) + params.fraction * a.yz * vec2(size) + vec2(.5)));
            if (all(equal(destination, p)) && a.x > 0 && a.x > best) {
                found = true;
                best = a.x;
                color = second ? imageLoad(color1, q) : imageLoad(color0, q);
            }
        }
    depth = found ? params.near_z * params.far_z /
                        (params.near_z + best * (params.far_z - params.near_z))
                  : 0;
    return found;
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
    vec4 value = agree ? .5 * (c0 + c1) : imageLoad(color0, p);
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
        value = .5 * (imageLoad(color0, p) + imageLoad(color1, p));
        code = 0;
    }
    imageStore(result, p, value);
    imageStore(validity, p, vec4(code, v0 ? 1 : 0, v1 ? 1 : 0, 1));
}
