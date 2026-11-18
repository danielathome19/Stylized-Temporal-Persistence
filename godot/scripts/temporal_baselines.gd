extends RefCounted

# GPU core for controlled-input correctness. No readback or submit/sync here.
# Caller owns input and command submission. Times use integer microseconds.
var rd: RenderingDevice
var size := Vector2i.ZERO
var textures: Array[RID] = []
var shaders: Dictionary = {}
var pipelines: Dictionary = {}
var sets: Dictionary = {}
var histories: Array[RID] = []
var previous := RID()
var ring := RID()
var output := RID()
var weights_buffer := RID()
var method := "M1"
var alpha := 0.8
var period_us := 50000
var support_us := 250000
var current_weight := 0.25
var kernel := "exponential"
var scale_s := 0.25 / log(2.0)
var capacity := 0
var slot := 0
var ring_times: Array[int] = []
var ping := 0
var last_us := -1
var next_tick := 0
var events: Array[Dictionary] = []
var record_events := true


func setup(device: RenderingDevice, dimensions: Vector2i, config: Dictionary) -> void:
    if not textures.is_empty():
        release()
    reset()
    rd = device
    size = dimensions
    method = config["method"]
    record_events = bool(config.get("record_events", true))
    assert(method in ["M1", "M2", "M3", "M4"])
    alpha = float(config["alpha"])
    period_us = int(config["period_us"])
    support_us = int(config["support_us"])
    kernel = config.get("kernel", "exponential")
    current_weight = float(config.get("current_weight", 0.25))
    scale_s = float(config.get("scale_s", 0.25 / log(2.0)))
    assert(alpha >= 0 and alpha < 1 and period_us > 0 and support_us > 0)
    assert(current_weight >= 0 and current_weight <= 1 and scale_s > 0)
    for name in ["feedback", "ring_store", "ring_compose"]:
        var source := RDShaderSource.new()
        source.source_compute = (
            FileAccess
            . get_file_as_string("res://shaders/" + name + ".glsl")
            . replace("#[compute]", "")
        )
        var spirv := rd.shader_compile_spirv_from_source(source)
        assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
        shaders[name] = rd.shader_create_from_spirv(spirv)
        pipelines[name] = rd.compute_pipeline_create(shaders[name])
    histories = [make_texture(), make_texture()]
    previous = make_texture()
    output = make_texture()
    if method == "M4":
        capacity = ceili(float(support_us) / period_us) + 2
        ring = make_texture(capacity)
        ring_times.resize(capacity)
        ring_times.fill(-1)
        var zeros := PackedFloat32Array()
        zeros.resize(capacity)
        weights_buffer = rd.storage_buffer_create(capacity * 4, zeros.to_byte_array())


func make_texture(layers: int = 1) -> RID:
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
    if layers > 1:
        format.texture_type = RenderingDevice.TEXTURE_TYPE_2D_ARRAY
        format.array_layers = layers
    var texture := rd.texture_create(format, RDTextureView.new())
    textures.append(texture)
    return texture


func dispatch(name: String, images: Array, constants: PackedByteArray, buffer: RID = RID()) -> void:
    var key := name + str(images) + str(buffer)
    if not sets.has(key):
        var uniforms: Array[RDUniform] = []
        for i in range(images.size()):
            var uniform := RDUniform.new()
            uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
            uniform.binding = i
            uniform.add_id(images[i])
            uniforms.append(uniform)
        if buffer.is_valid():
            var uniform := RDUniform.new()
            uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
            uniform.binding = images.size()
            uniform.add_id(buffer)
            uniforms.append(uniform)
        sets[key] = rd.uniform_set_create(uniforms, shaders[name], 0)
    var list := rd.compute_list_begin()
    rd.compute_list_bind_compute_pipeline(list, pipelines[name])
    rd.compute_list_bind_uniform_set(list, sets[key], 0)
    if not constants.is_empty():
        rd.compute_list_set_push_constant(list, constants, constants.size())
    rd.compute_list_dispatch(list, ceili(size.x / 8.0), ceili(size.y / 8.0), 1)
    rd.compute_list_end()


func feedback(source: RID, a: float) -> void:
    dispatch(
        "feedback",
        [source, histories[ping], histories[1 - ping]],
        PackedFloat32Array([a, 0, 0, 0]).to_byte_array()
    )
    ping = 1 - ping


func store_sample(source: RID, tick: int) -> void:
    dispatch("ring_store", [source, ring], PackedInt32Array([slot, 0, 0, 0]).to_byte_array())
    ring_times[slot] = tick
    slot = (slot + 1) % capacity


func reset() -> void:
    # Stale pixels are never read: next call seeds history; ring metadata invalidated.
    last_us = -1
    ping = 0
    slot = 0
    ring_times.fill(-1)


func process(input: RID, now_us: int, cut: bool = false) -> RID:
    assert(now_us >= 0, "Simulation time must be nonnegative")
    if cut or (last_us >= 0 and now_us < last_us):
        reset()
    if last_us < 0:
        feedback(input, 0)
        if method == "M4":
            store_sample(input, now_us)
        next_tick = now_us + period_us
    elif now_us == last_us:
        return output if method == "M4" else histories[ping]
    elif method == "M1":
        feedback(input, alpha)
    elif method == "M2":
        feedback(input, pow(alpha, float(now_us - last_us) / period_us))
    elif method == "M3" or method == "M4":
        while next_tick <= now_us:
            var exact := next_tick == now_us
            var source := input if exact else previous
            if record_events:
                events.append(
                    {
                        "target_us": next_tick,
                        "source_us": now_us if exact else last_us,
                        "delivered_us": now_us,
                        "missed": not exact
                    }
                )
            if method == "M3":
                feedback(source, alpha)
            else:
                store_sample(source, next_tick)
            next_tick += period_us
    if method == "M4":
        var weights := PackedFloat32Array()
        weights.resize(capacity)
        for i in range(capacity):
            var age_us := now_us - ring_times[i]
            if ring_times[i] < 0 or age_us <= 0 or age_us > support_us:
                continue
            var age_s := age_us / 1000000.0
            match kernel:
                "exponential":
                    weights[i] = exp(-age_s / scale_s)
                "linear":
                    weights[i] = maxf(0, 1 - float(age_us) / support_us)
                "gaussian":
                    weights[i] = exp(-0.5 * pow(age_s / scale_s, 2))
                "hyperbolic":
                    weights[i] = 1 / (1 + age_s / scale_s)
                _:
                    assert(false, "Unknown kernel")
        rd.buffer_update(weights_buffer, 0, weights.size() * 4, weights.to_byte_array())
        dispatch(
            "ring_compose",
            [input, ring, output],
            PackedFloat32Array([current_weight, capacity, 0, 0]).to_byte_array(),
            weights_buffer
        )
    rd.texture_copy(
        input, previous, Vector3.ZERO, Vector3.ZERO, Vector3(size.x, size.y, 1), 0, 0, 0, 0
    )
    last_us = now_us
    return output if method == "M4" else histories[ping]


func release() -> void:
    for rid in sets.values():
        if rd.uniform_set_is_valid(rid):
            rd.free_rid(rid)
    sets.clear()
    for rid in textures:
        rd.free_rid(rid)
    textures.clear()
    if weights_buffer.is_valid():
        rd.free_rid(weights_buffer)
    weights_buffer = RID()
    for rid in pipelines.values():
        rd.free_rid(rid)
    for rid in shaders.values():
        rd.free_rid(rid)
    pipelines.clear()
    shaders.clear()
    reset()
