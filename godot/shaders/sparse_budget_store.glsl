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
    int diagnostic = nx * ny * p.versions + id * 4;
    mass[diagnostic] = -1;
    mass[diagnostic + 1] = -1;
    mass[diagnostic + 2] = 0;
    if (p.sequence == 0)
        mass[diagnostic + 3] = 0;
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
            overflow = true;
            // K denotes the incoming image, whose age is zero. Evaluate every directed
            // merge over the remaining lifetime of the represented timestamps.
            float best = 3.402823466e38;
            int remove = -1, keep = -1;
            for (int a = 0; a <= p.versions; a++) {
                float exposure = 0;
                for (int future = 0; future < p.history; future++) {
                    float m = 0, normalizer = 0;
                    for (int age = 0; age < min(count + future, p.history); age++)
                        normalizer += pow(p.alpha, float(age));
                    if (a == p.versions)
                        m = pow(p.alpha, float(future));
                    else
                        for (int age = 1; age < count; age++) {
                            int h = (p.sequence - age) % p.history;
                            if (age + future < p.history && maps[base + h] == uint(a))
                                m += pow(p.alpha, float(age + future));
                        }
                    exposure += (m / normalizer) * (m / normalizer);
                }
                for (int b = 0; b <= p.versions; b++) {
                    if (a == b)
                        continue;
                    float distance = 0;
                    int valid = 0;
                    for (int y = 0; y < p.tile; y++)
                        for (int x = 0; x < p.tile; x++) {
                            ivec2 q = ivec2(id % nx, id / nx) * p.tile + ivec2(x, y);
                            if (q.x >= p.width || q.y >= p.height)
                                continue;
                            vec3 ca = a == p.versions
                                          ? imageLoad(source, q).rgb
                                          : pixels[root + a * area + y * p.tile + x].rgb;
                            vec3 cb = b == p.versions
                                          ? imageLoad(source, q).rgb
                                          : pixels[root + b * area + y * p.tile + x].rgb;
                            vec3 d = ca - cb;
                            distance += dot(d, d);
                            valid++;
                        }
                    float score = exposure * distance / float(3 * valid);
                    if (score < best) {
                        best = score;
                        remove = a;
                        keep = b;
                    }
                }
            }
            // Keep complete sampled tiles; never average distinct silhouette geometry.
            // A directed merge redirects timestamp references, retaining their ages.
            if (remove == p.versions) {
                chosen = uint(keep);
            } else {
                for (int h = 0; h < p.history; h++)
                    if (maps[base + h] == uint(remove))
                        maps[base + h] = uint(keep == p.versions ? remove : keep);
                chosen = uint(remove);
                copied = true;
                for (int y = 0; y < p.tile; y++)
                    for (int x = 0; x < p.tile; x++) {
                        ivec2 q = ivec2(id % nx, id / nx) * p.tile + ivec2(x, y);
                        pixels[root + remove * area + y * p.tile + x] =
                            (q.x < p.width && q.y < p.height) ? imageLoad(source, q) : vec4(0);
                    }
            }
            int diagnostic = nx * ny * p.versions + id * 4;
            mass[diagnostic] = float(remove);
            mass[diagnostic + 1] = float(keep);
            mass[diagnostic + 2] =
                best; // Triangle inequality: accumulated norms bound total compression error
            // energy over the capture timeline (epsilon=0, fixed kernel/schedule).
            mass[diagnostic + 3] += sqrt(best);
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
    else
        maps[stats + 1]++;
    if (overflow)
        maps[stats + 2]++;
}
