extends "res://scripts/sparse_history.gd"
# One globally shared physical pool. Timestamp maps, heap and free-slot scan stay GPU resident.
var prepare_shader := "shared_prepare"
var capacity := 0
var score_policy := "lifetime"
var metadata_words := 0
var control_offset := 0


func setup(device: RenderingDevice, dimensions: Vector2i, config: Dictionary = {}) -> void:
    rd = device
    size = dimensions
    prepare_shader = (
        "shared_prepare" if config.get("parallel_prepare", true) else "shared_prepare_serial"
    )
    score_policy = config.get("score_policy", "lifetime")
    assert(score_policy in ["lifetime", "current", "distance"])
    tile = int(config.get("tile", 16))
    history = int(config.get("history", 10))
    alpha = float(config.get("alpha", alpha))
    epsilon = 0.0
    sequence = 0
    assert(tile > 0 and history > 0 and history <= 32 and alpha >= 0 and alpha < 1)
    tiles = ceili(size.x / float(tile)) * ceili(size.y / float(tile))
    capacity = int(
        config.get(
            "pool_slots",
            ceili(tiles * float(config.get("slot_multiplier", config.get("versions", 3))))
        )
    )
    assert(
        capacity >= tiles and capacity <= tiles * history,
        "Pool needs at least one slot per tile and at most dense history capacity"
    )
    versions = capacity
    control_offset = 3 * tiles * history + 9 * tiles + capacity
    metadata_words = control_offset + 16
    var floats := tiles * history * (history + 1) + 3 * tiles
    pool = rd.storage_buffer_create(capacity * tile * tile * 16)
    metadata = rd.storage_buffer_create(metadata_words * 4)
    mass = rd.storage_buffer_create(floats * 4)
    payload_bytes = capacity * tile * tile * 16 + metadata_words * 4 + floats * 4
    output = make_texture()
    for name in [prepare_shader, "shared_allocate", "shared_write", "shared_resolve"]:
        compile_shader(name)


func compile_shader(name: String) -> void:
    var code := FileAccess.get_file_as_string("res://shaders/" + name + ".glsl").replace(
        "#[compute]", ""
    )
    if code.contains("// SHARED_COMMON"):
        code = code.replace(
            "// SHARED_COMMON",
            FileAccess.get_file_as_string("res://shaders/shared_history_common.glsl")
        )
    if score_policy == "current":
        code = code.replace("r<p.history;r++", "r<1;r++")
    if score_policy == "distance":
        code = code.replace(
            "float score=exposure*data[distancebase(t)+a*p.history+b];",
            "float score=data[distancebase(t)+a*p.history+b];"
        )
    var source := RDShaderSource.new()
    source.source_compute = code
    var spirv := rd.shader_compile_spirv_from_source(source)
    assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
    shaders[name] = rd.shader_create_from_spirv(spirv)
    pipelines[name] = rd.compute_pipeline_create(shaders[name])


func capture(source: RID) -> RID:
    var params := (
        PackedInt32Array([size.x, size.y, tile, capacity, history, sequence]).to_byte_array()
    )
    params.append_array(PackedFloat32Array([0, alpha]).to_byte_array())
    var buffers := [pool, metadata, mass]
    var groups := Vector3i(mini(tiles, 65535), ceili(tiles / 65535.0), 1)
    dispatch(prepare_shader, [source], params, buffers, groups)
    dispatch("shared_allocate", [source], params, buffers, Vector3i(1, 1, 1))
    dispatch("shared_write", [source], params, buffers, groups)
    dispatch("shared_resolve", [output], params, buffers)
    sequence += 1
    return output
