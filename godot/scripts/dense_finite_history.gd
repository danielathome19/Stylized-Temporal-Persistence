extends "res://scripts/sparse_history.gd"
var storage_format := "rgba32f"


func setup(device: RenderingDevice, dimensions: Vector2i, config: Dictionary = {}) -> void:
    rd = device
    size = dimensions
    history = int(config.get("history", 10))
    alpha = float(config.get("alpha", alpha))
    sequence = 0
    assert(history > 0 and alpha >= 0 and alpha < 1)
    storage_format = config.get("storage_format", "rgba32f")
    assert(storage_format in ["rgba32f", "rgba16f", "rgba8"])
    var pixel_bytes := (
        16 if storage_format == "rgba32f" else (8 if storage_format == "rgba16f" else 4)
    )
    payload_bytes = size.x * size.y * history * pixel_bytes
    pool = rd.storage_buffer_create(payload_bytes)
    output = make_texture()
    compile_shader("dense_finite")


func compile_shader(name: String) -> void:
    var code := FileAccess.get_file_as_string("res://shaders/" + name + ".glsl").replace(
        "#[compute]", ""
    )
    if storage_format != "rgba32f":
        code = code.replace(
            "#version 450",
            (
                "#version 450\n#define "
                + ("HALF_RING" if storage_format == "rgba16f" else "UNORM_RING")
            )
        )
    var source := RDShaderSource.new()
    source.source_compute = code
    var spirv := rd.shader_compile_spirv_from_source(source)
    assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
    shaders[name] = rd.shader_create_from_spirv(spirv)
    pipelines[name] = rd.compute_pipeline_create(shaders[name])


func capture(source: RID) -> RID:
    var params := PackedInt32Array([size.x, size.y, history, sequence]).to_byte_array()
    params.append_array(PackedFloat32Array([alpha, 0, 0, 0]).to_byte_array())
    dispatch("dense_finite", [source, output], params, [pool])
    sequence += 1
    return output


func release() -> void:
    for rid in sets.values():
        if rd.uniform_set_is_valid(rid):
            rd.free_rid(rid)
    for rid in textures:
        rd.free_rid(rid)
    if pool.is_valid():
        rd.free_rid(pool)
    for rid in pipelines.values():
        rd.free_rid(rid)
    for rid in shaders.values():
        rd.free_rid(rid)
    sets.clear()
    textures.clear()
    pipelines.clear()
    shaders.clear()
