extends "res://scripts/sparse_history.gd"
# Fixed pool plus explicit incoming scores, minheap and linked per-pixel lookup.
var capacity := 0
var previous := RID()


func setup(device: RenderingDevice, dimensions: Vector2i, config: Dictionary = {}) -> void:
    rd = device
    size = dimensions
    history = int(config.get("history", 10))
    alpha = float(config.get("alpha", alpha))
    capacity = int(config.get("pool_slots", 763))
    assert(capacity > 0 and history > 0 and history <= 32 and alpha >= 0 and alpha < 1)
    pool = rd.storage_buffer_create(16 * capacity)
    metadata = rd.storage_buffer_create(8 * capacity)
    mass = rd.storage_buffer_create(4 * (3 * capacity + 2 * size.x * size.y + 4))
    previous = make_texture()
    output = make_texture()
    payload_bytes = 24 * size.x * size.y + 36 * capacity + 16
    for name in ["innovation_classify", "innovation_allocate", "innovation_resolve"]:
        compile_shader(name)


func compile_shader(name: String) -> void:
    var code := FileAccess.get_file_as_string("res://shaders/" + name + ".glsl").replace(
        "#[compute]", ""
    )
    code = code.replace(
        "// INNOVATION_COMMON",
        FileAccess.get_file_as_string("res://shaders/innovation_common.glsl")
    )
    var source := RDShaderSource.new()
    source.source_compute = code
    var spirv := rd.shader_compile_spirv_from_source(source)
    assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
    shaders[name] = rd.shader_create_from_spirv(spirv)
    pipelines[name] = rd.compute_pipeline_create(shaders[name])


func capture(source: RID) -> RID:
    var params := PackedInt32Array([size.x, size.y, capacity, history, sequence]).to_byte_array()
    params.append_array(PackedFloat32Array([alpha]).to_byte_array())
    dispatch("innovation_classify", [source, previous, output], params, [pool, metadata, mass])
    dispatch(
        "innovation_allocate",
        [source, previous, output],
        params,
        [pool, metadata, mass],
        Vector3i(1, 1, 1)
    )
    dispatch("innovation_resolve", [source, previous, output], params, [pool, metadata, mass])
    sequence += 1
    return output
