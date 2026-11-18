#[compute]
#version 450
layout(local_size_x = 64) in;
// SHARED_COMMON
void main() {
    int t = int(gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x);
    if (t >= tiles())
        return;
    for (int v = 0; v < int(meta[info(t)]); v++)
        if (meta[alivebase(t) + v] != 0u && meta[newbase(t) + v] != 0u)
            for (int offset = int(gl_LocalInvocationIndex); offset < p.tile * p.tile;
                 offset += 64) {
                ivec2 q =
                    ivec2(t % nx(), t / nx()) * p.tile + ivec2(offset % p.tile, offset / p.tile);
                if (q.x >= p.width || q.y >= p.height)
                    continue;
                store_pixel(int(meta[versionbase(t) + v]) * p.tile * p.tile + offset,
                            imageLoad(scene, q));
            }
}
