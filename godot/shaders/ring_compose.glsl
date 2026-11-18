#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D current_image;
layout(rgba32f, set = 0, binding = 1) uniform readonly image2DArray history_ring;
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D output_image;
layout(std430, set = 0, binding = 3) readonly buffer Weights {
    float weights[];
};
layout(push_constant, std430) uniform Params {
    float current_weight;
    float capacity;
    float p0;
    float p1;
}
params;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(output_image))))
        return;
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int i = 0; i < int(params.capacity); ++i) {
        if (weights[i] > 0.0) {
            sum += weights[i] * imageLoad(history_ring, ivec3(p, i));
            total += weights[i];
        }
    }
    vec4 value = imageLoad(current_image, p);
    if (total > 0.0)
        value = params.current_weight * value + (1.0 - params.current_weight) * sum / total;
    imageStore(output_image, p, value);
}
