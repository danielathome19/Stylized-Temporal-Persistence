layout(rgba32f, set = 0, binding = 0) uniform readonly image2D scene;
layout(rgba32f, set = 0, binding = 1) uniform image2D previous;
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D result;
layout(std430, set = 0, binding = 3) buffer Pool {
    vec4 delta[];
};
layout(std430, set = 0, binding = 4) buffer Meta {
    uint meta[];
};
layout(std430, set = 0, binding = 5) buffer Workspace {
    uint work[];
};
layout(push_constant, std430) uniform Params {
    int width;
    int height;
    int capacity;
    int history;
    int sequence;
    float alpha;
}
p;
const uint INVALID = 0xffffffffu;
int pixels() {
    return p.width * p.height;
}
int nextbase() {
    return p.capacity;
}
int heads() {
    return 2 * p.capacity;
}
int controls() {
    return 2 * p.capacity + pixels();
}
int priorities() {
    return controls() + 4;
}
int incoming() {
    return priorities() + p.capacity;
}
float weight(int i) {
    return i == 0 ? 1 : pow(p.alpha, float(i));
}
float coefficient(int age, int r) {
    int count = min(p.sequence + 1 + r, p.history);
    float z = 0, b = 0;
    for (int i = 0; i < count; i++) {
        float w = weight(i);
        z += w;
        if (i > age + r)
            b += w;
    }
    return b / z;
}
float energy(int age) {
    float e = 0;
    for (int r = 0; r < p.history; r++) {
        float b = coefficient(age, r);
        e += b * b;
    }
    return e;
}
float priority(int slot) {
    return uintBitsToFloat(work[priorities() + slot]);
}
bool less_slot(uint a, uint b) {
    float x = priority(int(a)), y = priority(int(b));
    if (x != y)
        return x < y;
    if (meta[2 * a] != meta[2 * b])
        return meta[2 * a] < meta[2 * b];
    return meta[2 * a + 1] < meta[2 * b + 1];
}
void sift(int pos, int count) {
    while (2 * pos + 1 < count) {
        int child = 2 * pos + 1;
        if (child + 1 < count && less_slot(work[child + 1], work[child]))
            child++;
        if (!less_slot(work[child], work[pos]))
            break;
        uint v = work[pos];
        work[pos] = work[child];
        work[child] = v;
        pos = child;
    }
}
