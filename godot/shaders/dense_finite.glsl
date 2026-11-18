#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D source;
layout(rgba32f, set = 0, binding = 1) uniform writeonly image2D result;
#ifdef HALF_RING
layout(std430, set = 0, binding = 2) buffer Ring {
    uvec2 pixels[];
};
vec4 load_color(int i) {
    return vec4(unpackHalf2x16(pixels[i].x), unpackHalf2x16(pixels[i].y));
}
void store_color(int i, vec4 v) {
    pixels[i] = uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#elif defined(UNORM_RING)
layout(std430, set = 0, binding = 2) buffer Ring {
    uint pixels[];
};
vec4 load_color(int i) {
    return unpackUnorm4x8(pixels[i]);
}
void store_color(int i, vec4 v) {
    pixels[i] = packUnorm4x8(v);
}
#else
layout(std430, set = 0, binding = 2) buffer Ring {
    vec4 pixels[];
};
vec4 load_color(int i) {
    return pixels[i];
}
void store_color(int i, vec4 v) {
    pixels[i] = v;
}
#endif
layout(push_constant, std430) uniform Params {
    int width;
    int height;
    int history;
    int sequence;
    float alpha;
    float pad0;
    float pad1;
    float pad2;
}
p;
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= p.width || q.y >= p.height)
        return;
    int n = p.width * p.height, index = q.y * p.width + q.x;
    store_color((p.sequence % p.history) * n + index, imageLoad(source, q));
    vec4 value = vec4(0);
    float z = 0;
    for (int age = 0; age < min(p.sequence + 1, p.history); age++) {
        float w = age == 0 ? 1.0 : pow(p.alpha, float(age));
        value += w * load_color(((p.sequence - age) % p.history) * n + index);
        z += w;
    }
    imageStore(result, q, value / z);
}
