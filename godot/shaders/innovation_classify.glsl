#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
// INNOVATION_COMMON
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= p.width || q.y >= p.height)
        return;
    int pixel = q.y * p.width + q.x;
    work[heads() + pixel] = INVALID;
    float score = -1;
    if (p.sequence > 0 && p.history > 1) {
        vec4 d = imageLoad(scene, q) - imageLoad(previous, q);
        if (any(notEqual(d, vec4(0))))
            score = energy(0) * dot(d.rgb, d.rgb);
    }
    work[incoming() + pixel] = floatBitsToUint(score);
}
