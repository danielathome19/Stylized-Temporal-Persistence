extends SceneTree


func _initialize() -> void:
    var args := OS.get_cmdline_user_args()
    if args.size() != 1:
        quit(2)
        return
    var job: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
    var rd := RenderingServer.create_local_rendering_device()
    if rd == null:
        quit(3)
        return
    for test in job["cases"]:
        var core = load("res://scripts/temporal_baselines.gd").new()
        core.setup(rd, Vector2i(test["width"], test["height"]), test)
        var input: RID = core.make_texture()
        for frame in test["frames"]:
            if frame.has("width"):
                core.setup(rd, Vector2i(frame["width"], frame["height"]), test)
                input = core.make_texture()
            var bytes := FileAccess.get_file_as_bytes(frame["input"])
            assert(bytes.size() == core.size.x * core.size.y * 16)
            rd.texture_update(input, 0, bytes)
            var result: RID = core.process(input, int(frame["time_us"]), frame.get("cut", false))
            rd.submit()
            rd.sync()  # Accuracy replay only. Never interpreted as GPU timing.
            var file := FileAccess.open(frame["output"], FileAccess.WRITE)
            file.store_buffer(rd.texture_get_data(result, 0))
            file.close()
        var file := FileAccess.open(test["events"], FileAccess.WRITE)
        file.store_string(JSON.stringify(core.events, "\t"))
        file.close()
        core.release()
        print("BASELINE_CASE_COMPLETE ", test["name"])
    rd.free()
    print("BASELINE_REPLAY_COMPLETE")
    quit()
