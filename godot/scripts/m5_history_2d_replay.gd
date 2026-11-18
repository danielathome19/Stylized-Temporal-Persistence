extends SceneTree


func _initialize() -> void:
    var job: Dictionary = JSON.parse_string(
        FileAccess.get_file_as_string(OS.get_cmdline_user_args()[0])
    )
    var rd := RenderingServer.create_local_rendering_device()
    for sequence in job["sequences"]:
        var core = load("res://scripts/temporal_baselines.gd").new()
        core.setup(
            rd,
            Vector2i(sequence["width"], sequence["height"]),
            {"method": "M1", "alpha": sequence["alpha"], "period_us": 50000, "support_us": 500000}
        )
        var source := RDShaderSource.new()
        source.source_compute = (
            FileAccess
            . get_file_as_string("res://shaders/reconstruct_visibility_2d.glsl")
            . replace("#[compute]", "")
        )
        var spirv := rd.shader_compile_spirv_from_source(source)
        assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
        core.shaders["reconstruct_visibility_2d"] = rd.shader_create_from_spirv(spirv)
        core.pipelines["reconstruct_visibility_2d"] = rd.compute_pipeline_create(
            core.shaders["reconstruct_visibility_2d"]
        )
        var images: Array = []
        for i in range(6):
            images.append(core.make_texture())
        var initialized := false
        for frame in sequence["frames"]:
            for i in range(4):
                rd.texture_update(
                    images[i],
                    0,
                    FileAccess.get_file_as_bytes(frame[["color0", "color1", "aux0", "aux1"][i]])
                )
            if not initialized:
                core.feedback(images[0], 0)
                initialized = true
            core.dispatch(
                "reconstruct_visibility_2d",
                images,
                PackedFloat32Array(frame["params"] + [sequence["policy"], 0.5]).to_byte_array()
            )
            core.feedback(images[4], sequence["alpha"])
            core.feedback(images[1], sequence["alpha"])
            rd.submit()
            rd.sync()  # Accuracy replay. No GPU timing claim.
            for entry in [
                ["sample", images[4]],
                ["provenance", images[5]],
                ["history", core.histories[core.ping]]
            ]:
                var file := FileAccess.open(
                    frame["output"] + "_" + entry[0] + ".bin", FileAccess.WRITE
                )
                file.store_buffer(rd.texture_get_data(entry[1], 0))
                file.close()
        core.release()
    rd.free()
    print("M5_HISTORY_COMPLETE")
    quit()
