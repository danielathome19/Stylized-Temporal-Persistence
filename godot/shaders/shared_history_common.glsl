layout(rgba32f, set = 0, binding = 0) uniform image2D scene;
layout(std430, set = 0, binding = 1) buffer Pool {
    vec4 pixels[];
};
layout(std430, set = 0, binding = 2) buffer Meta {
    uint meta[];
};
layout(std430, set = 0, binding = 3) buffer Data {
    float data[];
};
layout(push_constant, std430) uniform Params {
    int width;
    int height;
    int tile;
    int capacity;
    int history;
    int sequence;
    float epsilon;
    float alpha;
}
p;
const uint INVALID = 0xffffffffu;
const float HUGE = 3.402823466e38;
int nx() {
    return (p.width + p.tile - 1) / p.tile;
}
int tiles() {
    return nx() * ((p.height + p.tile - 1) / p.tile);
}
int mapbase(int t) {
    return t * p.history;
}
int versionbase(int t) {
    return tiles() * p.history + t * p.history;
}
int alivebase(int t) {
    return 2 * tiles() * p.history + t * p.history;
}
int info(int t) {
    return 3 * tiles() * p.history + t * 8;
}
int heapbase() {
    return 3 * tiles() * p.history + tiles() * 8;
}
int usedbase() {
    return heapbase() + tiles();
}
int control() {
    return usedbase() + p.capacity;
}
int distancebase(int t) {
    return tiles() * p.history + t * p.history * p.history;
}
int prioritybase() {
    return tiles() * p.history * (p.history + 1);
}
int ledgerbase() {
    return prioritybase() + tiles();
}
int lossbase() {
    return ledgerbase() + tiles();
}
float weight(int age) {
    return age == 0 ? 1.0 : pow(p.alpha, float(age));
}
float area(int t) {
    return float(min(p.tile, p.width - (t % nx()) * p.tile) *
                 min(p.tile, p.height - (t / nx()) * p.tile));
}
float image_fraction(int t) {
    return area(t) / float(p.width * p.height);
}
vec4 read_version(int t, int v, int offset) {
    uint slot = meta[versionbase(t) + v];
    ivec2 q = ivec2(t % nx(), t / nx()) * p.tile + ivec2(offset % p.tile, offset / p.tile);
    return slot == uint(p.capacity) ? imageLoad(scene, q)
                                    : pixels[int(slot) * p.tile * p.tile + offset];
}
void propose(int t) {
    int count = min(p.sequence + 1, p.history), u = int(meta[info(t)]);
    float best = HUGE;
    uint remove = INVALID, keep = INVALID;
    for (int a = 0; a < u; a++) {
        if (meta[alivebase(t) + a] == 0u)
            continue;
        float exposure = 0;
        for (int r = 0; r < p.history; r++) {
            float m = 0, z = 0;
            for (int age = 0; age < min(count + r, p.history); age++)
                z += weight(age);
            for (int age = 0; age < count; age++)
                if (age + r < p.history &&
                    meta[mapbase(t) + (p.sequence - age) % p.history] == uint(a))
                    m += weight(age + r);
            exposure += (m / z) * (m / z);
        }
        for (int b = 0; b < u; b++) {
            if (a == b || meta[alivebase(t) + b] == 0u)
                continue;
            float score = exposure * data[distancebase(t) + a * p.history + b];
            if (score < best) {
                best = score;
                remove = uint(a);
                keep = uint(b);
            }
        }
    }
    meta[info(t) + 2] = remove;
    meta[info(t) + 3] = keep;
    data[prioritybase() + t] = best == HUGE ? HUGE : best * image_fraction(t);
}
