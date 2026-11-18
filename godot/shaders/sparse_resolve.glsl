#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform writeonly image2D result;
layout(std430, set = 0, binding = 1) readonly buffer Pool {
    vec4 pixels[];
};
layout(std430, set = 0, binding = 2) readonly buffer Meta {
    uint maps[];
};
layout(std430, set = 0, binding = 3) readonly buffer Mass {
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
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= p.width || q.y >= p.height)
        return;
    int nx = (p.width + p.tile - 1) / p.tile, id = (q.y / p.tile) * nx + q.x / p.tile;
    int area = p.tile * p.tile, offset = (q.y % p.tile) * p.tile + q.x % p.tile;
    vec4 value = vec4(0);
    for (int k = 0; k < p.versions; k++) {
        float w = mass[id * p.versions + k];
        if (w > 0)
            value += w * pixels[(id * p.versions + k) * area + offset];
    }
    imageStore(result, q, value);
}
