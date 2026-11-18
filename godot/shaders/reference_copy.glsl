#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(rgba16f, set = 0, binding = 0) uniform readonly image2D source_image;
layout(rgba16f, set = 0, binding = 1) uniform writeonly image2D destination;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(destination))))
        return;
    imageStore(destination, p, imageLoad(source_image, p));
}
