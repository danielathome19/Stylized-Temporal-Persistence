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
        for name in ["scatter_clear", "scatter_depth", "scatter_owner", "scatter_resolve"]:
            var source := RDShaderSource.new()
            source.source_compute = (
                FileAccess
                . get_file_as_string("res://shaders/" + name + ".glsl")
                . replace("#[compute]", "")
            )
            var spirv := rd.shader_compile_spirv_from_source(source)
            assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
            core.shaders[name] = rd.shader_create_from_spirv(spirv)
            core.pipelines[name] = rd.compute_pipeline_create(core.shaders[name])
        var scratch: Array = []
        for i in range(4):
            var format := RDTextureFormat.new()
            format.width = sequence["width"]
            format.height = sequence["height"]
            format.format = RenderingDevice.DATA_FORMAT_R32_UINT
            format.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT
            var texture := rd.texture_create(format, RDTextureView.new())
            core.textures.append(texture)
            scratch.append(texture)
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
            core.dispatch("scatter_clear", scratch, PackedByteArray())
            core.dispatch(
                "scatter_depth",
                [images[2], images[3], scratch[0], scratch[1]],
                PackedFloat32Array(frame.get("scales", [-0.5, 0.5, 16, 0])).to_byte_array()
            )
            core.dispatch(
                "scatter_owner",
                [images[2], images[3], scratch[0], scratch[1], scratch[2], scratch[3]],
                PackedFloat32Array(frame.get("scales", [-0.5, 0.5, 16, 0])).to_byte_array()
            )
            core.dispatch(
                "scatter_resolve",
                images + [scratch[2], scratch[3]],
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
