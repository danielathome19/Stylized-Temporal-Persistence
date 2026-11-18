extends "res://scripts/persistence_effect.gd"
# Perspective reverse-Z; held scheduled composite, available at right endpoint.
var sampler := RID()
var colors: Array = []
var auxiliaries: Array = []
var scratch: Array = []
var sample := RID()
var provenance := RID()
var endpoint := 0
var previous_time := -1
var previous_interval := 0
var target_time := 0
var paused_frame := false
var reconstructed := 0
var startup_holds := 0


func _init() -> void:
    super()
    access_resolved_depth = true
    needs_motion_vectors = true


func dispose() -> void:
    if core != null and sampler.is_valid():
        core.rd.free_rid(sampler)
        sampler = RID()
    super()


func _render_callback(_type: int, data: RenderData) -> void:
    if disposed:
        return
    var rd := RenderingServer.get_rendering_device()
    var buffers := data.get_render_scene_buffers() as RenderSceneBuffersRD
    var size := buffers.get_internal_size()
    if size.x == 0 or size.y == 0:
        return
    if core == null or dimensions != size:
        if core != null:
            rd.free_rid(sampler)
            core.release()
        core = load("res://scripts/temporal_baselines.gd").new()
        core.setup(rd, size, config)
        colors = [core.make_texture(), core.make_texture()]
        auxiliaries = [core.make_texture(), core.make_texture()]
        sample = core.make_texture()
        provenance = core.make_texture()
        scratch = []
        for i in range(4):
            var format := RDTextureFormat.new()
            format.width = size.x
            format.height = size.y
            format.format = RenderingDevice.DATA_FORMAT_R32_UINT
            format.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT
            var texture := rd.texture_create(format, RDTextureView.new())
            scratch.append(texture)
            core.textures.append(texture)
        for name in [
            "capture_3d",
            "scatter_clear",
            "scatter_depth",
            "scatter_owner",
            "scatter_resolve",
            "float_to_scene"
        ]:
            var shader_source := RDShaderSource.new()
            shader_source.source_compute = (
                FileAccess
                . get_file_as_string("res://shaders/" + name + ".glsl")
                . replace("#[compute]", "")
            )
            var spirv := rd.shader_compile_spirv_from_source(shader_source)
            assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
            core.shaders[name] = rd.shader_create_from_spirv(spirv)
            core.pipelines[name] = rd.compute_pipeline_create(core.shaders[name])
        sampler = rd.sampler_create(RDSamplerState.new())
        dimensions = size
        previous_time = -1
        _setup_history(rd, size)
    var color := buffers.get_color_layer(0)
    if cut_pending or time_us < previous_time:
        previous_time = -1
    cut_pending = false
    if previous_time != time_us:
        var current := 1 - endpoint
        var textures := [
            color,
            buffers.get_depth_layer(0),
            buffers.get_velocity_layer(0),
            colors[current],
            auxiliaries[current]
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
        var uniform_set := rd.uniform_set_create(uniforms, core.shaders["capture_3d"], 0)
        var list := rd.compute_list_begin()
        rd.compute_list_bind_compute_pipeline(list, core.pipelines["capture_3d"])
        rd.compute_list_bind_uniform_set(list, uniform_set, 0)
        rd.compute_list_dispatch(list, ceili(size.x / 8.), ceili(size.y / 8.), 1)
        rd.compute_list_end()
        rd.free_rid(uniform_set)
        if previous_time < 0:
            _store_history(colors[current], true)
            target_time = time_us + int(config["period_us"])
            previous_interval = 0
        else:
            var interval := time_us - previous_time
            while target_time <= time_us:
                if target_time == time_us:
                    _store_history(colors[current])
                elif previous_interval <= 0:
                    _store_history(colors[endpoint])
                    startup_holds += 1
                else:
                    var fraction := float(target_time - previous_time) / interval
                    var scales := (
                        PackedFloat32Array(
                            [
                                -float(target_time - previous_time) / previous_interval,
                                1.0 - fraction,
                                16,
                                0
                            ]
                        )
                        . to_byte_array()
                    )
                    core.dispatch("scatter_clear", scratch, PackedByteArray())
                    core.dispatch(
                        "scatter_depth",
                        [auxiliaries[endpoint], auxiliaries[current], scratch[0], scratch[1]],
                        scales
                    )
                    core.dispatch(
                        "scatter_owner",
                        [auxiliaries[endpoint], auxiliaries[current]] + scratch,
                        scales
                    )
                    core.dispatch(
                        "scatter_resolve",
                        [
                            colors[endpoint],
                            colors[current],
                            auxiliaries[endpoint],
                            auxiliaries[current],
                            sample,
                            provenance,
                            scratch[2],
                            scratch[3]
                        ],
                        PackedFloat32Array([0.1, 100, fraction, .02, .05, 16, 1, 0]).to_byte_array()
                    )
                    _store_history(sample)
                    reconstructed += 1
                target_time += int(config["period_us"])
            # A held presentation makes the next native motion span ambiguous. Hold
            # reconstruction through the following interval, then resume valid pairs.
            previous_interval = 0 if paused_frame else interval
        previous_time = time_us
        endpoint = current
        paused_frame = false
    else:
        paused_frame = true
    var stale: Array = []
    for key in core.sets:
        if key.begins_with("float_to_scene"):
            stale.append(key)
    for key in stale:
        if rd.uniform_set_is_valid(core.sets[key]):
            rd.free_rid(core.sets[key])
        core.sets.erase(key)
    core.dispatch("float_to_scene", [_history_result(), color], PackedByteArray())
    if report_once:
        print("LIVE_M5_ACTIVE ", size)
        report_once = false


# Storage hooks let sparse history consume the same reconstructed GPU samples.
func _setup_history(_rd: RenderingDevice, _size: Vector2i) -> void:
    pass


func _store_history(source_image: RID, reset_history: bool = false) -> void:
    core.feedback(source_image, 0.0 if reset_history else config["alpha"])


func _history_result() -> RID:
    return core.histories[core.ping]
