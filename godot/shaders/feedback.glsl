#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D current_image;
layout(rgba32f, set = 0, binding = 1) uniform readonly image2D previous_history;
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D next_history;
layout(push_constant, std430) uniform Params {
    float retention;
    float pad0;
    float pad1;
    float pad2;
}
params;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(next_history))))
        return;
    vec4 value = imageLoad(current_image, p);
    if (params.retention > 0.0)
        value =
            (1.0 - params.retention) * value + params.retention * imageLoad(previous_history, p);
    imageStore(next_history, p, value);
}
