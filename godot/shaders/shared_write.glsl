#[compute]
#version 450
layout(local_size_x = 1) in;
// SHARED_COMMON
void main() {
    int t = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x);
    if (t >= tiles())
        return;
    uint physical = meta[info(t) + 4];
    if (physical == INVALID)
        return;
    for (int y = 0; y < p.tile; y++)
        for (int x = 0; x < p.tile; x++) {
            ivec2 q = ivec2(t % nx(), t / nx()) * p.tile + ivec2(x, y);
            pixels[int(physical) * p.tile * p.tile + y * p.tile + x] =
                (q.x < p.width && q.y < p.height) ? imageLoad(scene, q) : vec4(0);
        }
}
