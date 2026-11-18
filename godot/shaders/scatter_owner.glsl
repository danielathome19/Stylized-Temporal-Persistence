#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D aux0;
layout(rgba32f, set = 0, binding = 1) uniform readonly image2D aux1;
layout(r32ui, set = 0, binding = 2) uniform readonly uimage2D depth0;
layout(r32ui, set = 0, binding = 3) uniform readonly uimage2D depth1;
layout(r32ui, set = 0, binding = 4) uniform uimage2D owner0;
layout(r32ui, set = 0, binding = 5) uniform uimage2D owner1;
layout(push_constant, std430) uniform Timing {
    float left_scale;
    float right_scale;
    float radius;
    float pad;
}
timing;
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(aux0);
    if (any(greaterThanEqual(q, size)))
        return;
    for (int side = 0; side < 2; side++) {
        vec4 a = side == 0 ? imageLoad(aux0, q) : imageLoad(aux1, q);
        ivec2 p = ivec2(floor(
            vec2(q) + (side == 0 ? timing.left_scale : timing.right_scale) * a.yz * vec2(size) +
            vec2(.5)));
        if (a.x <= 0 || any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size)) ||
            any(greaterThan(abs(p - q), ivec2(timing.radius))))
            continue;
        uint depth = side == 0 ? imageLoad(depth0, p).x : imageLoad(depth1, p).x;
        if (depth != floatBitsToUint(a.x))
            continue;
        uint id = uint(q.y * size.x + q.x);
        if (side == 0)
            imageAtomicMin(owner0, p, id);
        else
            imageAtomicMin(owner1, p, id);
    }
}
