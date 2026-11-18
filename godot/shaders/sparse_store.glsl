#[compute]
#version 450
layout(local_size_x = 1) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D source;
layout(std430, set = 0, binding = 1) buffer Pool {
    vec4 pixels[];
};
layout(std430, set = 0, binding = 2) buffer Meta {
    uint maps[];
};
layout(std430, set = 0, binding = 3) buffer Mass {
    float mass[];
};
layout(push_constant, std430) uniform Params {
    int width;
    int height;
    int tile;
    int versions;
    int history;
    int sequence;
    float epsilon;
    float alpha;
}
p;
void main() {
    int nx = (p.width + p.tile - 1) / p.tile, ny = (p.height + p.tile - 1) / p.tile;
    int id = int(gl_GlobalInvocationID.x);
    if (id >= nx * ny)
        return;
    int base = id * p.history, area = p.tile * p.tile, root = id * p.versions * area;
    int count = min(p.sequence + 1, p.history), slot = p.sequence % p.history;
    uint invalid = 0xffffffffu;
    uint previous = p.sequence == 0 ? invalid : maps[base + (p.sequence - 1) % p.history];
    if (p.sequence == 0)
        for (int h = 0; h < p.history; h++)
            maps[base + h] = invalid;
    maps[base + slot] = invalid;
    bool same = previous != invalid;
    // Compare to retained data, not the preceding raw frame: no tolerance drift.
    for (int y = 0; y < p.tile; y++)
        for (int x = 0; x < p.tile; x++) {
            ivec2 q = ivec2(id % nx, id / nx) * p.tile + ivec2(x, y);
            if (q.x >= p.width || q.y >= p.height)
                continue;
            if (previous != invalid) {
                vec4 d = abs(imageLoad(source, q) -
                             pixels[root + int(previous) * area + y * p.tile + x]);
                if (any(greaterThan(d, vec4(p.epsilon))))
                    same = false;
            }
        }
    // Reappearing states can reference any still-live version, not only the latest.
    if (!same)
        for (int k = 0; k < p.versions; k++) {
            bool live = false;
            for (int h = 0; h < p.history; h++)
                if (maps[base + h] == uint(k))
                    live = true;
            if (!live)
                continue;
            bool match = true;
            for (int y = 0; y < p.tile; y++)
                for (int x = 0; x < p.tile; x++) {
                    ivec2 q = ivec2(id % nx, id / nx) * p.tile + ivec2(x, y);
                    if (q.x >= p.width || q.y >= p.height)
                        continue;
                    if (any(greaterThan(
                            abs(imageLoad(source, q) - pixels[root + k * area + y * p.tile + x]),
                            vec4(p.epsilon))))
                        match = false;
                }
            if (match) {
                previous = uint(k);
                same = true;
                break;
            }
        }
    uint chosen = previous;
    bool copied = false, overflow = false;
    if (!same) {
        chosen = invalid;
        for (int k = 0; k < p.versions; k++) {
            bool used = false;
            for (int h = 0; h < p.history; h++)
                if (maps[base + h] == uint(k))
                    used = true;
            if (!used) {
                chosen = uint(k);
                break;
            }
        }
        if (chosen == invalid) {
            chosen = previous;
            overflow = true;
        } else {
            copied = true;
            for (int y = 0; y < p.tile; y++)
                for (int x = 0; x < p.tile; x++) {
                    ivec2 q = ivec2(id % nx, id / nx) * p.tile + ivec2(x, y);
                    pixels[root + int(chosen) * area + y * p.tile + x] =
                        (q.x < p.width && q.y < p.height) ? imageLoad(source, q) : vec4(0);
                }
        }
    }
    maps[base + slot] = chosen;
    // Collapse historical kernel weights onto shared versions before resolve.
    float total = 0;
    for (int k = 0; k < p.versions; k++)
        mass[id * p.versions + k] = 0;
    for (int age = 0; age < count; age++) {
        int h = (p.sequence - age) % p.history;
        float w = pow(p.alpha, float(age));
        mass[id * p.versions + int(maps[base + h])] += w;
        total += w;
    }
    for (int k = 0; k < p.versions; k++)
        mass[id * p.versions + k] /= total;
    // Three counters per tile, cumulative since initialization.
    int stats = nx * ny * p.history + id * 3;
    if (p.sequence == 0) {
        maps[stats] = 0;
        maps[stats + 1] = 0;
        maps[stats + 2] = 0;
    }
    if (copied)
        maps[stats]++;
    else if (overflow)
        maps[stats + 2]++;
    else
        maps[stats + 1]++;
}
