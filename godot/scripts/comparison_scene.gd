extends "res://scripts/private_link_demo.gd"
var job: Dictionary
var camera3d: Camera3D


func box(size: Vector3, color: Color, where: Vector3) -> MeshInstance3D:
    var node := MeshInstance3D.new()
    var mesh := BoxMesh.new()
    mesh.size = size
    node.mesh = mesh
    var material := StandardMaterial3D.new()
    material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
    material.albedo_color = color
    node.material_override = material
    add_child(node)
    node.position = where
    return node


func _ready() -> void:
    job = JSON.parse_string(FileAccess.get_file_as_string(OS.get_cmdline_user_args()[0]))
    get_tree().root.size = Vector2i(640, 360)
    get_tree().root.content_scale_size = Vector2i(job["width"], job["height"])
    get_tree().root.content_scale_mode = Window.CONTENT_SCALE_MODE_VIEWPORT
    if job["scene"] == "link":
        var document := GLTFDocument.new()
        var state := GLTFState.new()
        assert(document.append_from_file(model_path, state) == OK)
        actor = Node3D.new()
        add_child(actor)
        var centered := Node3D.new()
        actor.add_child(centered)
        var model := document.generate_scene(state)
        centered.add_child(model)
        prepare(model)
        centered.position = -bounds.get_center()
        actor.scale = Vector3.ONE * (2.4 / maxf(maxf(bounds.size.x, bounds.size.y), bounds.size.z))
    else:
        actor = box(Vector3(1.2, .8, .5), Color(1, .6, .2), Vector3.ZERO)
    camera3d = Camera3D.new()
    camera3d.fov = 60
    camera3d.near = .1
    camera3d.far = 100
    add_child(camera3d)
    camera3d.position = Vector3(0, 0, 6)
    var env := Environment.new()
    env.background_mode = Environment.BG_COLOR
    env.background_color = Color(.03, .04, .06)
    env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
    camera3d.environment = env
    if job["scene"] == "occlusion":
        box(Vector3(.4, 2.7, .3), Color(.1, .7, .3), Vector3(0, 0, 1))
    if job["scene"] == "camera":
        for x in range(-4, 5):
            for y in range(-2, 3):
                box(
                    Vector3(.9, .9, .1),
                    Color(.12 + .1 * ((x + y + 8) % 2), .18, .3),
                    Vector3(x, y, -1)
                )
    effect = load("res://scripts/m5_sample_capture.gd").new()
    effect.folder = job["folder"]
    var compositor := Compositor.new()
    compositor.compositor_effects = [effect]
    camera3d.compositor = compositor


func _process(_delta: float) -> void:
    if actor == null:
        return
    var timestamp := frame * 16667 + (frame % 3) * 1000
    var t := timestamp / 1000000.0
    actor.position = Vector3(sin(t * 2) * 1.2, .25 * sin(t), 0)
    actor.rotation = (
        Vector3(.3 * sin(t), t, .2 * cos(t))
        if job["scene"] != "link"
        else Vector3(0, -PI / 2 + .6 * sin(t * 1.5), 0)
    )
    for i in range(posed_bones.size()):
        var entry: Dictionary = posed_bones[i]
        entry["skeleton"].set_bone_pose_rotation(
            entry["bone"], entry["base"] * Quaternion(Vector3.RIGHT, .55 * sin(t * 4 + i * PI))
        )
    if job["scene"] == "camera":
        camera3d.position.x = .8 * sin(t)
    RenderingServer.call_on_render_thread(effect.set_time.bind(timestamp))
    frame += 1
    if frame >= 120:
        set_process(false)
        await RenderingServer.frame_post_draw
        RenderingServer.call_on_render_thread(effect.save_manifest)
        await effect.saved
        print("M5_SAMPLE_CAPTURE_COMPLETE")
        get_tree().quit()
