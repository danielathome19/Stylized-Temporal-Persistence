extends RefCounted
# Fixed-capacity partitioned GPU tile pool. No dense historical texture allocation.
var rd: RenderingDevice
var size: Vector2i
var tile := 16
var versions := 3
var history := 10
var epsilon := 0.0
var alpha := pow(0.5, .05 / .25)
var sequence := 0
var tiles := 0
var pool := RID()
var metadata := RID()
var mass := RID()
var output := RID()
var textures: Array[RID] = []
var shaders: Dictionary = {}
var pipelines: Dictionary = {}
var sets: Dictionary = {}
var payload_bytes := 0
var store_shader := "sparse_store"


func setup(device: RenderingDevice, dimensions: Vector2i, config: Dictionary = {}) -> void:
    rd = device
    size = dimensions
    tile = int(config.get("tile", 16))
    versions = int(config.get("versions", 3))
    history = int(config.get("history", 10))
    epsilon = float(config.get("epsilon", 0))
    alpha = float(config.get("alpha", alpha))
    var policy: String = config.get("pressure_policy", "hold")
    assert(policy in ["hold", "horizon_merge"])
    store_shader = "sparse_budget_store" if policy == "horizon_merge" else "sparse_store"
    assert(
        (
            tile > 0
            and versions >= 1
            and versions <= history
            and history > 0
            and epsilon >= 0
            and alpha >= 0
            and alpha < 1
        )
    )
    tiles = ceili(size.x / float(tile)) * ceili(size.y / float(tile))
    var pool_bytes := tiles * versions * tile * tile * 16
    var meta_bytes := tiles * (history + 3) * 4
    var mass_bytes := (
        tiles * versions * 4 + (tiles * 16 if store_shader == "sparse_budget_store" else 0)
    )
    pool = rd.storage_buffer_create(pool_bytes)
    metadata = rd.storage_buffer_create(meta_bytes)
    mass = rd.storage_buffer_create(mass_bytes)
    payload_bytes = pool_bytes + meta_bytes + mass_bytes
    output = make_texture()
    for name in [store_shader, "sparse_resolve"]:
        compile_shader(name)


func compile_shader(name: String) -> void:
    var source := RDShaderSource.new()
    source.source_compute = (
        FileAccess.get_file_as_string("res://shaders/" + name + ".glsl").replace("#[compute]", "")
    )
    var spirv := rd.shader_compile_spirv_from_source(source)
    assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
    shaders[name] = rd.shader_create_from_spirv(spirv)
    pipelines[name] = rd.compute_pipeline_create(shaders[name])


func make_texture() -> RID:
    var format := RDTextureFormat.new()
    format.width = size.x
    format.height = size.y
    format.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
    format.usage_bits = (
        RenderingDevice.TEXTURE_USAGE_STORAGE_BIT
        | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
        | RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
        | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
    )
    var texture := rd.texture_create(format, RDTextureView.new())
    textures.append(texture)
    return texture


func dispatch(
    name: String,
    images: Array,
    constants: PackedByteArray,
    storage: Array = [],
    groups: Vector3i = Vector3i.ZERO
) -> void:
    var key := name + str(images) + str(storage)
    if not sets.has(key):
        var uniforms: Array[RDUniform] = []
        for i in range(images.size() + storage.size()):
            var u := RDUniform.new()
            u.binding = i
            u.uniform_type = (
                RenderingDevice.UNIFORM_TYPE_IMAGE
                if i < images.size()
                else RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
            )
            u.add_id(images[i] if i < images.size() else storage[i - images.size()])
            uniforms.append(u)
        sets[key] = rd.uniform_set_create(uniforms, shaders[name], 0)
    var list := rd.compute_list_begin()
    rd.compute_list_bind_compute_pipeline(list, pipelines[name])
    rd.compute_list_bind_uniform_set(list, sets[key], 0)
    if not constants.is_empty():
        rd.compute_list_set_push_constant(list, constants, constants.size())
    if groups == Vector3i.ZERO:
        groups = Vector3i(ceili(size.x / 8.), ceili(size.y / 8.), 1)
    rd.compute_list_dispatch(list, groups.x, groups.y, groups.z)
    rd.compute_list_end()


func capture(source: RID) -> RID:
    var params := (
        PackedInt32Array([size.x, size.y, tile, versions, history, sequence]).to_byte_array()
    )
    params.append_array(PackedFloat32Array([epsilon, alpha]).to_byte_array())
    dispatch(store_shader, [source], params, [pool, metadata, mass], Vector3i(tiles, 1, 1))
    dispatch("sparse_resolve", [output], params, [pool, metadata, mass])
    sequence += 1
    return output


func reset() -> void:
    sequence = 0


func release() -> void:
    for rid in sets.values():
        if rd.uniform_set_is_valid(rid):
            rd.free_rid(rid)
    for rid in textures:
        rd.free_rid(rid)
    for rid in [pool, metadata, mass]:
        rd.free_rid(rid)
    for rid in pipelines.values():
        rd.free_rid(rid)
    for rid in shaders.values():
        rd.free_rid(rid)
    sets.clear()
    textures.clear()
    pipelines.clear()
    shaders.clear()
