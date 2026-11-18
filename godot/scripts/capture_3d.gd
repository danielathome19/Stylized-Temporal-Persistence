extends CompositorEffect
signal captured
var request_data: Dictionary = {}
var core
var sampler := RID()
var color_out := RID()
var aux_out := RID()


func _init() -> void:
    effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
    access_resolved_color = true
    access_resolved_depth = true
    needs_motion_vectors = true


func request(data: Dictionary) -> void:
    request_data = data


func release() -> void:
    if core != null:
        core.rd.free_rid(sampler)
        core.release()
        core = null
    captured.emit.call_deferred()


func _render_callback(_type: int, data: RenderData) -> void:
    if request_data.is_empty():
        return
    var rd := RenderingServer.get_rendering_device()
    var buffers := data.get_render_scene_buffers() as RenderSceneBuffersRD
    var size := buffers.get_internal_size()
    if core == null:
        core = load("res://scripts/temporal_baselines.gd").new()
        core.setup(
            rd, size, {"method": "M1", "alpha": 0.8, "period_us": 50000, "support_us": 500000}
        )
        color_out = core.make_texture()
        aux_out = core.make_texture()
        var source := RDShaderSource.new()
        source.source_compute = (
            FileAccess.get_file_as_string("res://shaders/capture_3d.glsl").replace("#[compute]", "")
        )
        var spirv := rd.shader_compile_spirv_from_source(source)
        assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
        core.shaders["capture"] = rd.shader_create_from_spirv(spirv)
        core.pipelines["capture"] = rd.compute_pipeline_create(core.shaders["capture"])
        sampler = rd.sampler_create(RDSamplerState.new())
    var textures := [
        buffers.get_color_layer(0),
        buffers.get_depth_layer(0),
        buffers.get_velocity_layer(0),
        color_out,
        aux_out
    ]
    var uniforms: Array[RDUniform] = []
    for i in range(5):
        var u := RDUniform.new()
        u.binding = i
        u.uniform_type = (
            RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
            if i < 3
            else RenderingDevice.UNIFORM_TYPE_IMAGE
        )
        if i < 3:
            u.add_id(sampler)
        u.add_id(textures[i])
        uniforms.append(u)
    var set := rd.uniform_set_create(uniforms, core.shaders["capture"], 0)
    var list := rd.compute_list_begin()
    rd.compute_list_bind_compute_pipeline(list, core.pipelines["capture"])
    rd.compute_list_bind_uniform_set(list, set, 0)
    rd.compute_list_dispatch(list, ceili(size.x / 8.), ceili(size.y / 8.), 1)
    rd.compute_list_end()
    for entry in [["color", color_out], ["aux", aux_out]]:
        var file := FileAccess.open(
            request_data["path"] + "_" + entry[0] + ".bin", FileAccess.WRITE
        )
        file.store_buffer(rd.texture_get_data(entry[1], 0))
        file.close()
    rd.free_rid(set)
    request_data["width"] = size.x
    request_data["height"] = size.y
    request_data["frame"] = Engine.get_frames_drawn()
    request_data["wall_us"] = Time.get_ticks_usec()
    var file := FileAccess.open(request_data["path"] + ".json", FileAccess.WRITE)
    file.store_string(JSON.stringify(request_data, "\t"))
    file.close()
    request_data = {}
    captured.emit.call_deferred()
