extends CompositorEffect

var config := {
    "method": "M3",
    "alpha": pow(0.5, 0.05 / 0.25),
    "period_us": 50000,
    "support_us": 500000,
    "kernel": "exponential",
    "current_weight": 0.25,
    "scale_s": 0.25 / log(2.0),
    "record_events": false
}
var core
var source := RID()
var dimensions := Vector2i.ZERO
var time_us := 0
var cut_pending := false
var disposed := false
var report_once := true


func _init() -> void:
    effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
    access_resolved_color = true


# Caller invokes on render thread after updating scene state.
func set_time(timestamp_us: int, cut: bool = false) -> void:
    time_us = timestamp_us
    cut_pending = cut_pending or cut


func dispose() -> void:
    disposed = true
    if core != null:
        core.release()
        core = null


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
            core.release()
        core = load("res://scripts/temporal_baselines.gd").new()
        core.setup(rd, size, config)
        source = core.make_texture()
        for name in ["scene_to_float", "float_to_scene"]:
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
        dimensions = size
    var color := buffers.get_color_layer(0)
    assert(rd.texture_get_format(color).format == RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT)
    # Engine textures can be recreated independently of dimensions; transient sets avoid stale RIDs.
    for name in ["scene_to_float", "float_to_scene"]:
        var stale: Array = []
        for key in core.sets:
            if key.begins_with(name):
                stale.append(key)
        for key in stale:
            rd.free_rid(core.sets[key])
            core.sets.erase(key)
    core.dispatch("scene_to_float", [color, source], PackedByteArray())
    var result: RID = core.process(source, time_us, cut_pending)
    cut_pending = false
    core.dispatch("float_to_scene", [result, color], PackedByteArray())
    if report_once:
        print("LIVE_PERSISTENCE_ACTIVE ", config["method"], " ", size)
        report_once = false
