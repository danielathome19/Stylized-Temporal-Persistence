#[compute]
#version 450
layout(local_size_x = 64) in;
// SHARED_COMMON
shared int tile_versions;
shared int incoming_version;
shared uint matched_version;
void main() {
    int t = int(gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x);
    if (t >= tiles())
        return;
    int lane = int(gl_LocalInvocationIndex);
    if (lane == 0) {
        int slot = p.sequence % p.history, u = 0;
        uint old[32];
        uint ids[32];
        if (p.sequence == 0) {
            data[ledgerbase() + t] = 0;
            meta[info(t) + 6] = 0;
            meta[info(t) + 7] = 0;
        }
        for (int h = 0; h < p.history; h++)
            old[h] = p.sequence == 0 ? INVALID : meta[mapbase(t) + h];
        old[slot] = uint(p.capacity);
        for (int h = 0; h < p.history; h++) {
            uint value = old[h];
            if (value == INVALID) {
                meta[mapbase(t) + h] = INVALID;
                continue;
            }
            int v = 0;
            while (v < u && ids[v] != value)
                v++;
            if (v == u) {
                ids[u] = value;
                u++;
            }
            meta[mapbase(t) + h] = uint(v);
        }
        int incoming = -1;
        for (int v = 0; v < p.history; v++) {
            meta[alivebase(t) + v] = v < u ? 1u : 0u;
            if (v < u) {
                meta[versionbase(t) + v] = ids[v];
                if (ids[v] == uint(p.capacity))
                    incoming = v;
            }
        }
        meta[info(t)] = uint(u);
        meta[info(t) + 1] = uint(u);
        meta[info(t) + 4] = INVALID;
        meta[info(t) + 5] = uint(incoming);
        data[lossbase() + t] = 0;
        tile_versions = u;
        incoming_version = incoming;
        matched_version = INVALID;
    }
    memoryBarrierBuffer();
    barrier();
    int u = tile_versions, incoming = incoming_version;
    for (int pair = lane; pair < u * u; pair += 64) {
        int a = pair / u, b = pair % u;
        if (b < a)
            continue;
        float distance = 0, maxerror = 0;
        for (int y = 0; y < p.tile; y++)
            for (int x = 0; x < p.tile; x++) {
                ivec2 q = ivec2(t % nx(), t / nx()) * p.tile + ivec2(x, y);
                if (q.x >= p.width || q.y >= p.height)
                    continue;
                vec4 d =
                    abs(read_version(t, a, y * p.tile + x) - read_version(t, b, y * p.tile + x));
                distance += dot(d.rgb, d.rgb);
                maxerror = max(maxerror, max(max(d.r, d.g), max(d.b, d.a)));
            }
        distance /= 3.0 * area(t);
        data[distancebase(t) + a * p.history + b] = distance;
        data[distancebase(t) + b * p.history + a] = distance;
        if (a != b && maxerror == 0.0 && (a == incoming || b == incoming)) {
            uint candidate = uint(a == incoming ? b : a);
            atomicMin(matched_version, candidate);
        }
    }
    memoryBarrierBuffer();
    barrier();
    if (lane == 0) {
        int match = matched_version == INVALID ? -1 : int(matched_version);
        if (match >= 0) {
            for (int h = 0; h < p.history; h++)
                if (meta[mapbase(t) + h] == uint(incoming))
                    meta[mapbase(t) + h] = uint(match);
            meta[alivebase(t) + incoming] = 0;
            meta[info(t) + 1]--;
        }
        propose(t);
    }
}
