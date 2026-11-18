extends SceneTree


func _initialize() -> void:
    var job: Dictionary = JSON.parse_string(
        FileAccess.get_file_as_string(OS.get_cmdline_user_args()[0])
    )
    var rd := RenderingServer.create_local_rendering_device()
    for test in job["cases"]:
        var core = load("res://scripts/temporal_baselines.gd").new()
        core.setup(
            rd,
            Vector2i(test["width"], test["height"]),
            {"method": "M1", "alpha": 0.8, "period_us": 50000, "support_us": 500000}
        )
        var source := RDShaderSource.new()
        source.source_compute = (
            FileAccess
            . get_file_as_string("res://shaders/reconstruct_translation.glsl")
            . replace("#[compute]", "")
        )
        var spirv := rd.shader_compile_spirv_from_source(source)
        assert(spirv.compile_error_compute.is_empty(), spirv.compile_error_compute)
        core.shaders["reconstruct_translation"] = rd.shader_create_from_spirv(spirv)
        core.pipelines["reconstruct_translation"] = rd.compute_pipeline_create(
            core.shaders["reconstruct_translation"]
        )
        var images: Array = []
        for name in ["color0", "color1", "aux0", "aux1", "result", "validity"]:
            var texture: RID = core.make_texture()
            images.append(texture)
            if test.has(name):
                rd.texture_update(texture, 0, FileAccess.get_file_as_bytes(test[name]))
        core.dispatch(
            "reconstruct_translation", images, PackedFloat32Array(test["params"]).to_byte_array()
        )
        rd.submit()
        rd.sync()  # Correctness only, not timing.
        for i in [4, 5]:
            var file := FileAccess.open(
                test["output"] + (".bin" if i == 4 else "_valid.bin"), FileAccess.WRITE
            )
            file.store_buffer(rd.texture_get_data(images[i], 0))
            file.close()
        core.release()
    rd.free()
    print("RECONSTRUCTION_COMPLETE")
    quit()
