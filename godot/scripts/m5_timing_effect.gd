extends "res://scripts/m5_live_effect.gd"

signal finished
var job: Dictionary
var records: Array[Dictionary] = []
var submissions: Dictionary = {}
var index := 0
var last_query_frame := -1
var last_wall := 0
var measurement_start := 0
var stop_index := -1
var drain_frames := 0
var calibration_core
var calibration_image := RID()
var first_wall := 0
var first_measured := -1
var measured_count := 0


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


func _render_callback(type: int, data: RenderData) -> void:
    if disposed:
        return
    var rd := RenderingServer.get_rendering_device()
    var instrumented: bool = job.get("instrumented", true)
    if instrumented:
        collect(rd)
    if stop_index >= 0:
        drain_frames += 1
        if drain_frames >= 12:
            var file := FileAccess.open(job["output"], FileAccess.WRITE)
            file.store_string(
                JSON.stringify(
                    {
                        "records": records,
                        "unresolved": submissions.values(),
                        "submitted": index,
                        "first_measured": first_measured
                    },
                    "\t"
                )
            )
            file.close()
            if calibration_core != null:
                calibration_core.release()
            dispose()
            finished.emit.call_deferred()
        return
    var wall := Time.get_ticks_usec()
    if first_wall == 0:
        first_wall = wall
    var mode: String = job["mode"]
    if mode.begins_with("cal") and calibration_core == null:
        calibration_core = load("res://scripts/temporal_baselines.gd").new()
        calibration_core.setup(rd, Vector2i(job["width"], job["height"]), config)
        calibration_image = calibration_core.make_texture()
        var source_code := RDShaderSource.new()
        source_code.source_compute = (
            FileAccess.get_file_as_string("res://shaders/probe.glsl").replace("#[compute]", "")
        )
        calibration_core.shaders["probe"] = rd.shader_create_from_spirv(
            rd.shader_compile_spirv_from_source(source_code)
        )
        calibration_core.pipelines["probe"] = rd.compute_pipeline_create(
            calibration_core.shaders["probe"]
        )
    var measured := (
        index >= int(job["warm_up"]) and wall - first_wall >= int(job.get("warm_up_us", 0))
    )
    if measured and measurement_start == 0:
        measurement_start = wall
        first_measured = index
    if measured:
        measured_count += 1
    var label := "mb_" + str(index)
    var old_tick: int = target_time if core != null else -1
    if instrumented:
        rd.capture_timestamp(label + "_begin")
    var cpu_start := Time.get_ticks_usec()
    if mode.begins_with("cal"):
        for k in range(int(mode.trim_prefix("cal"))):
            calibration_core.dispatch("probe", [calibration_image], PackedByteArray())
    elif mode != "none":
        super._render_callback(type, data)
    var cpu_end := Time.get_ticks_usec()
    if instrumented:
        rd.capture_timestamp(label + "_end")
    var row := {
        "index": index,
        "measured": measured,
        "wall_us": wall,
        "simulation_us": time_us,
        "callback_interval_us": wall - last_wall if last_wall else 0,
        "cpu_submission_us": cpu_end - cpu_start,
        "gpu_ns": null,
        "scheduled_updates":
        (
            int((target_time - old_tick) / 50000)
            if core != null and old_tick >= 0 and mode == "M5"
            else 0
        )
    }
    if instrumented:
        submissions[index] = row
    else:
        records.append(row)
    last_wall = wall
    index += 1
    if (
        measured
        and measured_count >= int(job["minimum_frames"])
        and wall - measurement_start >= int(job["duration_us"])
    ):
        stop_index = index
