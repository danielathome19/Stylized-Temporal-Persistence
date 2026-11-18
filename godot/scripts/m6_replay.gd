extends SceneTree


func _initialize() -> void:
    var job: Dictionary = JSON.parse_string(
        FileAccess.get_file_as_string(OS.get_cmdline_user_args()[0])
    )
    var rd := RenderingServer.create_local_rendering_device()
    for case in job["cases"]:
        var core = (
            load(
                (
                    "res://scripts/dense_finite_history.gd"
                    if case.get("dense", false)
                    else (
                        "res://scripts/shared_history.gd"
                        if case.get("shared", false)
                        else "res://scripts/sparse_history.gd"
                    )
                )
            )
            . new()
        )
        if case.get("synthesis", false):
            core = load("res://scripts/synthesis_history.gd").new()
        if case.get("innovation", false):
            core = load("res://scripts/innovation_history.gd").new()
        core.setup(rd, Vector2i(case["width"], case["height"]), case)
        var source: RID = core.make_texture()
        for frame in case["frames"]:
            if frame.get("reset", false):
                core.reset()
            rd.texture_update(source, 0, FileAccess.get_file_as_bytes(frame["input"]))
            core.capture(source)
            rd.submit()
            rd.sync()
            var file := FileAccess.open(frame["output"], FileAccess.WRITE)
            file.store_buffer(rd.texture_get_data(core.output, 0))
            file.close()
            if frame.has("state"):
                file = FileAccess.open(frame["state"], FileAccess.WRITE)
                file.store_buffer(rd.buffer_get_data(core.metadata))
                file.close()
            if frame.has("diagnostics"):
                file = FileAccess.open(frame["diagnostics"], FileAccess.WRITE)
                file.store_buffer(rd.buffer_get_data(core.mass))
                file.close()
        if not case.get("dense", false):
            var file := FileAccess.open(case["metadata"], FileAccess.WRITE)
            file.store_buffer(rd.buffer_get_data(core.metadata))
            file.close()
        print("SPARSE_PAYLOAD ", case["name"], " ", core.payload_bytes)
        core.release()
    rd.free()
    print("M6_REPLAY_COMPLETE")
    quit()
