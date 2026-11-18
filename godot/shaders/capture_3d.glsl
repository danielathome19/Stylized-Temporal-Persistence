#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D scene_color;
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
layout(set = 0, binding = 2) uniform sampler2D scene_motion;
layout(rgba32f, set = 0, binding = 3) uniform writeonly image2D color_out;
layout(rgba32f, set = 0, binding = 4) uniform writeonly image2D aux_out;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(color_out))))
        return;
    imageStore(color_out, p, texelFetch(scene_color, p, 0));
    imageStore(aux_out, p,
               vec4(texelFetch(scene_depth, p, 0).r, texelFetch(scene_motion, p, 0).rg, 1));
}
