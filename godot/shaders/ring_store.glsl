#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, set = 0, binding = 0) uniform readonly image2D source_image;
layout(rgba32f, set = 0, binding = 1) uniform writeonly image2DArray history_ring;
layout(push_constant, std430) uniform Params {
    int layer;
    int p0;
    int p1;
    int p2;
}
params;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(source_image))))
        return;
    imageStore(history_ring, ivec3(p, params.layer), imageLoad(source_image, p));
}
