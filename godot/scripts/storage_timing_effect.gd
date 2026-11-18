extends CompositorEffect
signal finished
var job: Dictionary
var core
var inputs: Array[RID] = []
var records: Array[Dictionary] = []
var submissions: Dictionary = {}
var index := 0
var last_query_frame := -1
var first_wall := 0
var measurement_start := 0
var first_measured := -1
var measured_count := 0
var stop_index := -1
var drain := 0
var disposed := false
var last_wall := 0


func _init() -> void:
    effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT


func collect(rd: RenderingDevice) -> void:
    var frame := rd.get_captured_timestamps_frame()
    if frame == last_query_frame:
        return
    last_query_frame = frame
    var pairs: Dictionary = {}
    for i in range(rd.get_captured_timestamps_count()):
        var name := rd.get_captured_timestamp_name(i)
        if not name.begins_with("mb_"):
            continue
        var parts := name.split("_")
        var id := int(parts[1])
        if not pairs.has(id):
            pairs[id] = {}
        pairs[id][parts[2]] = rd.get_captured_timestamp_gpu_time(i)
    for id in pairs:
        if pairs[id].has("begin") and pairs[id].has("end") and submissions.has(id):
            var row: Dictionary = submissions[id]
            row["gpu_ns"] = pairs[id]["end"] - pairs[id]["begin"]
            row["query_frame"] = frame
            row["query_delay_callbacks"] = index - int(id)
            records.append(row)
            submissions.erase(id)


func _render_callback(_type: int, _data: RenderData) -> void:
    if disposed:
        return
    var rd := RenderingServer.get_rendering_device()
    collect(rd)
    if inputs.is_empty():
        if job["mode"] != "none":
            core = (
                load(
                    (
                        "res://scripts/dense_finite_history.gd"
                        if job["mode"] == "dense"
                        else "res://scripts/shared_history.gd"
                    )
                )
                . new()
            )
            if job.get("synthesis", false):
                core = load("res://scripts/synthesis_history.gd").new()
            if job.get("innovation", false):
                core = load("res://scripts/innovation_history.gd").new()
            core.setup(rd, Vector2i(job["width"], job["height"]), job)
        var format := RDTextureFormat.new()
        format.width = job["width"]
        format.height = job["height"]
        format.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
        format.usage_bits = (
            RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
        )
        for path in job["inputs"]:
            var texture := rd.texture_create(format, RDTextureView.new())
            rd.texture_update(texture, 0, FileAccess.get_file_as_bytes(path))
            inputs.append(texture)
        first_wall = Time.get_ticks_usec()
    if stop_index >= 0:
        drain += 1
        if drain >= 16:
            var file := FileAccess.open(job["output"], FileAccess.WRITE)
            file.store_string(
                JSON.stringify(
                    {
                        "records": records,
                        "unresolved": submissions.values(),
                        "submitted": index,
                        "first_measured": first_measured,
                        "history_payload_bytes": core.payload_bytes if core != null else 0,
                        "input_cache_bytes":
                        inputs.size() * int(job["width"]) * int(job["height"]) * 16
                    },
                    "\t"
                )
            )
            file.close()
            if core != null:
                core.release()
            for texture in inputs:
                rd.free_rid(texture)
            disposed = true
            finished.emit.call_deferred()
        return
    var wall := Time.get_ticks_usec()
    if (
        first_measured < 0
        and index >= int(job["warm_up"])
        and wall - first_wall >= int(job["warm_up_us"])
        and index % inputs.size() == 0
    ):
        first_measured = index
        measurement_start = wall
    var measured := first_measured >= 0
    if measured:
        measured_count += 1
    var label := "mb_" + str(index)
    rd.capture_timestamp(label + "_begin")
    var cpu_start := Time.get_ticks_usec()
    if core != null:
        if index % inputs.size() == 0:
            core.reset()
        core.capture(inputs[index % inputs.size()])
    var cpu_end := Time.get_ticks_usec()
    rd.capture_timestamp(label + "_end")
    submissions[index] = {
        "index": index,
        "measured": measured,
        "source_index": index % inputs.size(),
        "wall_us": wall,
        "callback_interval_us": wall - last_wall if last_wall else 0,
        "cpu_submission_us": cpu_end - cpu_start,
        "gpu_ns": null
    }
    last_wall = wall
    index += 1
    if (
        measured_count >= int(job["minimum_frames"])
        and wall - measurement_start >= int(job["duration_us"])
        and index % inputs.size() == 0
    ):
        stop_index = index
