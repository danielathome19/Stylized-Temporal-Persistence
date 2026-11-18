#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D color0;
layout(rgba32f, set = 0, binding = 1) uniform readonly image2D color1;
layout(rgba32f, set = 0, binding = 2) uniform readonly image2D aux0;
layout(rgba32f, set = 0, binding = 3) uniform readonly image2D aux1;
layout(rgba32f, set = 0, binding = 4) uniform writeonly image2D result;
layout(rgba32f, set = 0, binding = 5) uniform writeonly image2D validity;
// Single horizontally translating object calibration. aux=(vx,0,linear depth,id).
layout(push_constant, std430) uniform Params {
    float dt0;
    float dt1;
    float velocity0;
    float velocity1;
}
params;
bool candidate(ivec2 p, bool second, float dt, float velocity, out vec4 c, out vec4 a) {
    ivec2 size = imageSize(result);
    vec4 here = second ? imageLoad(aux1, p) : imageLoad(aux0, p);
    bool found = here.w != 1.0;
    a = here;
    c = second ? imageLoad(color1, p) : imageLoad(color0, p);
    ivec2 q = ivec2(int(floor(float(p.x) - dt * velocity + 0.5)), p.y);
    if (q.x >= 0 && q.x < size.x) {
        vec4 moving = second ? imageLoad(aux1, q) : imageLoad(aux0, q);
        if (moving.w == 1.0 && abs(moving.x - velocity) < 0.001 && (!found || moving.z < a.z)) {
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
    bool v0 = candidate(p, false, params.dt0, params.velocity0, c0, a0);
    bool v1 = candidate(p, true, params.dt1, params.velocity1, c1, a1);
    // Conservative two-endpoint agreement. Never use target-time masks or depth.
    bool accept = v0 && v1 && a0.w == a1.w && abs(a0.z - a1.z) < 0.001 &&
                  max(max(abs(c0.r - c1.r), abs(c0.g - c1.g)), abs(c0.b - c1.b)) < 0.001;
    float fraction = params.dt0 / (params.dt0 - params.dt1);
    imageStore(result, p, accept ? mix(c0, c1, fraction) : imageLoad(color0, p));
    imageStore(validity, p, vec4(accept ? 1.0 : 0.0, v0 ? 1.0 : 0.0, v1 ? 1.0 : 0.0, 1.0));
}
