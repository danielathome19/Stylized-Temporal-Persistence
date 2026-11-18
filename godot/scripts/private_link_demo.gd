extends Node3D
# Optional private asset; no model or textures are embedded in this script/scene.
var model_path := "res://assets/private/MM_Young_Link_rigged.glb"
var posed_bones: Array = []
var actor: Node3D
var effect
var frame := 0
var origin := 0
var folder := ""
var mode := "M6"
var mesh_count := 0
var material_count := 0
var skeleton_count := 0
var bounds := AABB()
var has_bounds := false
var lifecycle := false


func prepare(node: Node) -> void:
    if node is Skeleton3D:
        skeleton_count += 1
        for bone_name in ["limb_13", "limb_16", "limb_10"]:
            var bone: int = node.find_bone(bone_name)
            if bone >= 0:
                posed_bones.append(
                    {"skeleton": node, "bone": bone, "base": node.get_bone_pose_rotation(bone)}
                )
    if node is MeshInstance3D and node.mesh != null:
        mesh_count += 1
        var box: AABB = node.global_transform * node.get_aabb()
        bounds = bounds.merge(box) if has_bounds else box
        has_bounds = true
        for surface in range(node.mesh.get_surface_count()):
            var original: Material = node.get_active_material(surface)
            if original is BaseMaterial3D:
                var material: BaseMaterial3D = original.duplicate()
                material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
                material.metallic = 0.0
                material.roughness = 1.0
                material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR
                node.set_surface_override_material(surface, material)
                material_count += 1
    for child in node.get_children():
        prepare(child)


func _ready() -> void:
    var args := OS.get_cmdline_user_args()
    if args.size() > 0:
        folder = args[0]
    if args.size() > 1:
        mode = args[1]
    if args.size() > 2:
        lifecycle = args[2] == "lifecycle"
    if not FileAccess.file_exists(model_path):
        model_path = "res://assets/private/MM_Young_Link.glb"
    if not FileAccess.file_exists(model_path):
        push_error("Private Link model missing; see assets/private/ATTRIBUTION.md")
        get_tree().quit(2)
        return
    get_tree().root.size = Vector2i(640, 360)
    var document := GLTFDocument.new()
    var state := GLTFState.new()
    var error := document.append_from_file(model_path, state)
    if error != OK:
        push_error("GLB import failed: " + str(error))
        get_tree().quit(2)
        return
    actor = Node3D.new()
    add_child(actor)
    var centered := Node3D.new()
    actor.add_child(centered)
    var model := document.generate_scene(state)
    centered.add_child(model)
    prepare(model)
    assert(has_bounds)
    centered.position = -bounds.get_center()
    actor.scale = (
        Vector3.ONE * (2.4 / maxf(maxf(bounds.size.x, bounds.size.y), maxf(bounds.size.z, .001)))
    )
    var camera := Camera3D.new()
    camera.fov = 60
    camera.near = .1
    camera.far = 100
    add_child(camera)
    camera.position = Vector3(0, 0, 6)
    var environment := Environment.new()
    environment.background_mode = Environment.BG_COLOR
    environment.background_color = Color(.045, .055, .075)
    environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
    camera.environment = environment
    if mode != "none":
        effect = load("res://scripts/m6_motion_effect.gd").new()
        if lifecycle:
            effect.storage_config["slot_multiplier"] = 1.25
        var compositor := Compositor.new()
        compositor.compositor_effects = [effect]
        camera.compositor = compositor
    var layer := CanvasLayer.new()
    add_child(layer)
    var label := Label.new()
    label.position = Vector2(10, 10)
    label.text = (
        "Private Link demo / "
        + mode
        + " / "
        + ("procedural bone motion" if not posed_bones.is_empty() else "rigid motion")
        + "\nNintendo: original model | spikey-vi: ripper"
    )
    layer.add_child(label)
    print(
        "PRIVATE_LINK_LOADED meshes=",
        mesh_count,
        " materials=",
        material_count,
        " skeletons=",
        skeleton_count,
        " posed_bones=",
        posed_bones.size(),
        " bounds=",
        bounds
    )
    origin = Time.get_ticks_usec()


func _process(_delta: float) -> void:
    if actor == null:
        return
    var timestamp := (
        frame * 16667 + (frame % 3) * 1000
        if not folder.is_empty()
        else Time.get_ticks_usec() - origin
    )
    if lifecycle:
        if frame >= 25 and frame <= 29:
            timestamp = 24 * 16667
        if frame == 40:
            get_tree().root.size = Vector2i(641, 359)
        if frame == 50:
            get_tree().root.size = Vector2i(640, 360)
        if frame == 60:
            timestamp = 0
    var t := timestamp / 1000000.0
    actor.position = Vector3(sin(t * 2) * 1.1, 0, 0)
    actor.rotation.y = -PI / 2 + .6 * sin(t * 1.5)
    for i in range(posed_bones.size()):
        var entry: Dictionary = posed_bones[i]
        entry["skeleton"].set_bone_pose_rotation(
            entry["bone"], entry["base"] * Quaternion(Vector3.RIGHT, .55 * sin(t * 4 + i * PI))
        )
    if effect != null:
        RenderingServer.call_on_render_thread(
            effect.set_time.bind(timestamp, lifecycle and frame == 90)
        )
    frame += 1
    if not folder.is_empty() and frame >= 120:
        set_process(false)
        await RenderingServer.frame_post_draw
        get_viewport().get_texture().get_image().save_png(folder + "/preview.png")
        print(
            "PRIVATE_LINK_COMPLETE mode=",
            mode,
            " reconstructed=",
            effect.reconstructed if effect != null else 0
        )
        if effect != null:
            RenderingServer.call_on_render_thread(
                effect.save_diagnostics.bind(folder + "/allocator.json")
            )
            await effect.diagnostics_ready
            RenderingServer.call_on_render_thread(effect.dispose)
        await RenderingServer.frame_post_draw
        get_tree().quit()


func _exit_tree() -> void:
    if effect != null:
        RenderingServer.call_on_render_thread(effect.dispose)
