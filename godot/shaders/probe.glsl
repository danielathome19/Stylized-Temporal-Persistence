#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(rgba32f, set = 0, binding = 0) uniform writeonly image2D destination;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(destination))))
        return;
    imageStore(destination, p, vec4(float(p.x), float(p.y), 0.25, 1.0));
}
