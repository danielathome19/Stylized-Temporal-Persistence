#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D color0;
layout(rgba32f, set = 0, binding = 1) uniform readonly image2D color1;
layout(rgba32f, set = 0, binding = 2) uniform readonly image2D aux0;
layout(rgba32f, set = 0, binding = 3) uniform readonly image2D aux1;
layout(rgba32f, set = 0, binding = 4) uniform writeonly image2D result;
layout(rgba32f, set = 0, binding = 5) uniform writeonly image2D validity;
// Single translating object in 2D. aux=(vx,vy,linear depth,id).
layout(push_constant, std430) uniform Params {
    float dt0;
    float dt1;
    float vx0;
    float vy0;
    float vx1;
    float vy1;
    float policy;
    float disagreement_px;
}
params;
bool candidate(ivec2 p, bool second, float dt, vec2 velocity, out vec4 c, out vec4 a) {
    ivec2 size = imageSize(result);
    vec4 here = second ? imageLoad(aux1, p) : imageLoad(aux0, p);
    bool found = here.w != 1.0;
    a = here;
    c = second ? imageLoad(color1, p) : imageLoad(color0, p);
    ivec2 q = ivec2(floor(vec2(p) - dt * velocity + vec2(0.5)));
    if (all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, size))) {
        vec4 moving = second ? imageLoad(aux1, q) : imageLoad(aux0, q);
        if (moving.w == 1.0 && length(moving.xy - velocity) < 0.001 && (!found || moving.z < a.z)) {
            a = moving;
            c = second ? imageLoad(color1, q) : imageLoad(color0, q);
            found = true;
        }
    }
    return found;
}
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(result))))
        return;
    vec4 c0, c1, a0, a1;
    bool v0 = candidate(p, false, params.dt0, vec2(params.vx0, params.vy0), c0, a0);
    bool v1 = candidate(p, true, params.dt1, vec2(params.vx1, params.vy1), c1, a1);
    // Conservative two-endpoint agreement. Never use target-time masks or depth.
    bool accept = v0 && v1 && a0.w == a1.w && abs(a0.z - a1.z) < 0.001 &&
                  max(max(abs(c0.r - c1.r), abs(c0.g - c1.g)), abs(c0.b - c1.b)) < 0.001;
    float fraction = params.dt0 / (params.dt0 - params.dt1);
    vec4 value = accept ? mix(c0, c1, fraction) : imageLoad(color0, p);
    float provenance = accept ? 1.0 : 0.0;
    if (params.policy >= 1.0 && (v0 != v1)) {
        value = v0 ? c0 : c1;
        provenance = 2.0;
    }
    // Motion disagreement is a bound proxy, not a calibrated probability.
    float uncertainty = length(vec2(params.vx0, params.vy0) - vec2(params.vx1, params.vy1)) *
                        (params.dt0 - params.dt1);
    if (params.policy == 2.0 && uncertainty > params.disagreement_px) {
        value = mix(imageLoad(color0, p), imageLoad(color1, p), fraction);
        provenance = 3.0;
    }
    if (params.policy == 3.0) {
        value = imageLoad(color0, p);
        provenance = 0.0;
    }
    if (params.policy == 4.0) {
        value = mix(imageLoad(color0, p), imageLoad(color1, p), fraction);
        provenance = 0.0;
    }
    imageStore(result, p, value);
    imageStore(validity, p, vec4(provenance, v0 ? 1.0 : 0.0, v1 ? 1.0 : 0.0, 1.0));
}
