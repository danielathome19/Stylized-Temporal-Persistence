#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
// SHARED_COMMON
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= p.width || q.y >= p.height)
        return;
    int t = (q.y / p.tile) * nx() + q.x / p.tile, offset = (q.y % p.tile) * p.tile + q.x % p.tile;
    vec4 value = vec4(0);
    for (int v = 0; v < int(meta[info(t)]); v++) {
        float w = data[t * p.history + v];
        if (w > 0)
            value += w * pixels[int(meta[versionbase(t) + v]) * p.tile * p.tile + offset];
    }
    imageStore(scene, q, value);
}
