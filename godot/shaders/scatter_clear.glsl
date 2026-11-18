#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(r32ui, set = 0, binding = 0) uniform writeonly uimage2D depth0;
layout(r32ui, set = 0, binding = 1) uniform writeonly uimage2D depth1;
layout(r32ui, set = 0, binding = 2) uniform writeonly uimage2D owner0;
layout(r32ui, set = 0, binding = 3) uniform writeonly uimage2D owner1;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(depth0))))
        return;
    imageStore(depth0, p, uvec4(0));
    imageStore(depth1, p, uvec4(0));
    imageStore(owner0, p, uvec4(0xffffffffu));
    imageStore(owner1, p, uvec4(0xffffffffu));
}
