extends CompositorEffect

signal captured
var destination := ""
var target_time := 0.0
var staging := RID()
var shader := RID()
var pipeline := RID()
var staging_size := Vector2i.ZERO


func release() -> void:
    var rd := RenderingServer.get_rendering_device()
    for rid in [staging, pipeline, shader]:
        if rid.is_valid():
            rd.free_rid(rid)
    staging = RID()
    pipeline = RID()
    shader = RID()
    captured.emit.call_deferred()


func _init() -> void:
    effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
    access_resolved_color = true


# Invoked on the render thread after scene transforms have settled.
func request(path: String, timestamp: float) -> void:
    destination = path
    target_time = timestamp


func _render_callback(_type: int, render_data: RenderData) -> void:
    if destination.is_empty():
        return
    var rd := RenderingServer.get_rendering_device()
    var buffers := render_data.get_render_scene_buffers() as RenderSceneBuffersRD
    var texture := buffers.get_color_layer(0)
    var format := rd.texture_get_format(texture)
    if not shader.is_valid():
        var source := RDShaderSource.new()
        source.source_compute = (
            FileAccess
            . get_file_as_string("res://shaders/reference_copy.glsl")
            . replace("#[compute]", "")
        )
        var spirv := rd.shader_compile_spirv_from_source(source)
        if not spirv.compile_error_compute.is_empty():
            push_error(spirv.compile_error_compute)
            return
        shader = rd.shader_create_from_spirv(spirv)
        pipeline = rd.compute_pipeline_create(shader)
    if staging_size != Vector2i(format.width, format.height):
        if staging.is_valid():
            rd.free_rid(staging)
        var staging_format := RDTextureFormat.new()
        staging_format.width = format.width
        staging_format.height = format.height
        staging_format.format = format.format
        staging_format.usage_bits = (
            RenderingDevice.TEXTURE_USAGE_STORAGE_BIT
            | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
        )
        staging = rd.texture_create(staging_format, RDTextureView.new())
        staging_size = Vector2i(format.width, format.height)
    var uniforms: Array[RDUniform] = []
    for i in range(2):
        var uniform := RDUniform.new()
        uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
        uniform.binding = i
        uniform.add_id(texture if i == 0 else staging)
        uniforms.append(uniform)
    var uniform_set := rd.uniform_set_create(uniforms, shader, 0)
    var list := rd.compute_list_begin()
    rd.compute_list_bind_compute_pipeline(list, pipeline)
    rd.compute_list_bind_uniform_set(list, uniform_set, 0)
    rd.compute_list_dispatch(list, ceili(format.width / 8.0), ceili(format.height / 8.0), 1)
    rd.compute_list_end()
    var data := rd.texture_get_data(staging, 0)
    rd.free_rid(uniform_set)
    var file := FileAccess.open(destination + ".bin", FileAccess.WRITE)
    file.store_buffer(data)
    file.close()
    var metadata := {
        "width": format.width,
        "height": format.height,
        "rd_format": format.format,
        "rgba16f": format.format == RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT,
        "byte_count": data.size(),
        "target_time": target_time,
        "callback": "POST_TRANSPARENT",
        "render_frame": Engine.get_frames_drawn(),
        "wall_ticks_usec": Time.get_ticks_usec(),
        "gpu": rd.get_device_name()
    }
    file = FileAccess.open(destination + ".json", FileAccess.WRITE)
    file.store_string(JSON.stringify(metadata, "\t"))
    file.close()
    destination = ""
    captured.emit.call_deferred()
