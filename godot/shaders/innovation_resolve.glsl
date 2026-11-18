#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
// INNOVATION_COMMON
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= p.width || q.y >= p.height)
        return;
    vec4 current = imageLoad(scene, q), value = current;
    uint s = work[heads() + q.y * p.width + q.x];
    while (s != INVALID) {
        value -= coefficient(p.sequence - int(meta[2 * s + 1]), 0) * delta[s];
        s = work[nextbase() + int(s)];
    }
    imageStore(result, q, value);
    imageStore(previous, q, current);
}
