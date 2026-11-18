#[compute]
#version 450
layout(local_size_x = 64) in;
// SHARED_COMMON
void main() {
    int t = int(gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x);
    if (t >= tiles())
        return;
    int u = int(meta[info(t)]);
    for (int offset = int(gl_LocalInvocationIndex); offset < p.tile * p.tile; offset += 64) {
        ivec2 q = ivec2(t % nx(), t / nx()) * p.tile + ivec2(offset % p.tile, offset / p.tile);
        if (q.x >= p.width || q.y >= p.height)
            continue;
        // Each invocation owns one pixel offset: gather all sources before writing any old destination.
        vec4 original[32];
        for (int c = 0; c < u; c++) {
            uint slot = meta[sourcebase(t) + c];
            original[c] = slot == uint(p.capacity)
                              ? imageLoad(scene, q)
                              : load_pixel(int(slot) * p.tile * p.tile + offset);
        }
        for (int v = 0; v < u; v++)
            if (meta[alivebase(t) + v] != 0u && meta[newbase(t) + v] == 0u) {
                vec4 color = vec4(0);
                for (int c = 0; c < u; c++)
                    color += data[coeffbase(t) + v * p.history + c] * original[c];
                store_pixel(int(meta[versionbase(t) + v]) * p.tile * p.tile + offset, color);
            }
    }
}
