extends Node3D

var moving: Array[Node3D] = []
var scene_id := "A"
var capture: CompositorEffect
var reference_persistence: CompositorEffect


func box(parent: Node3D, size: Vector3, color: Color, position_value: Vector3) -> MeshInstance3D:
    var instance := MeshInstance3D.new()
    var mesh := BoxMesh.new()
    mesh.size = size
    instance.mesh = mesh
    var material := StandardMaterial3D.new()
    material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
    material.albedo_color = color
    instance.material_override = material
    parent.add_child(instance)
    instance.position = position_value
    return instance


func build_scene() -> void:
    var camera := Camera3D.new()
    camera.projection = Camera3D.PROJECTION_ORTHOGONAL
    camera.size = 6.0
    camera.position.z = 10.0
    add_child(camera)
    var environment := Environment.new()
    environment.background_mode = Environment.BG_COLOR
    environment.background_color = Color.BLACK
    environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
    camera.environment = environment
    capture = load("res://scripts/reference_capture.gd").new()
    var compositor := Compositor.new()
    compositor.compositor_effects = [capture]
    camera.compositor = compositor
    if scene_id == "A":
        moving.append(box(self, Vector3(0.6, 0.8, 0.2), Color.WHITE, Vector3.ZERO))
    elif scene_id == "B":
        var pivot := Node3D.new()
        add_child(pivot)
        box(pivot, Vector3(0.3, 1.4, 0.2), Color.WHITE, Vector3(-0.35, 0, 0))
        box(pivot, Vector3(1.0, 0.3, 0.2), Color.WHITE, Vector3(0, -0.55, 0))
        moving.append(pivot)
    elif scene_id == "C":
        moving.append(box(self, Vector3(0.55, 0.7, 0.2), Color(1, 0, 0), Vector3.ZERO))
        moving.append(box(self, Vector3(0.7, 0.45, 0.2), Color(0, 1, 0), Vector3.ZERO))
        box(self, Vector3(0.45, 2.2, 0.2), Color(0, 0, 1), Vector3(0, 0, 1))
    else:
        push_error("Unknown scene")
        get_tree().quit(2)


func evaluate(timestamp: float) -> void:
    if scene_id == "A":
        moving[0].position = Vector3(-2.0 + 2.0 * timestamp, 0, 0)
    elif scene_id == "B":
        moving[0].rotation = Vector3(0, 0, 2.0 * timestamp)
    else:
        moving[0].position = Vector3(1.8 * sin(2.0 * timestamp), 0.4 * cos(3.0 * timestamp), 0)
        moving[1].position = Vector3(-1.8 * sin(2.0 * timestamp), -0.4 * cos(3.0 * timestamp), -0.4)


func _ready() -> void:
    var args := OS.get_cmdline_user_args()
    if args.size() != 1:
        push_error("Expected one capture job JSON path after --")
        get_tree().quit(2)
        return
    var job: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
    scene_id = job["scene"]
    get_tree().root.size = Vector2i(job["width"], job["height"])
    build_scene()
    if job.has("persistence"):
        reference_persistence = load("res://scripts/persistence_effect.gd").new()
        reference_persistence.config = job["persistence"]
        for child in get_children():
            if child is Camera3D:
                child.compositor.compositor_effects = [reference_persistence, capture]
    for frame in job["frames"]:
        evaluate(float(frame["time"]))
        if reference_persistence != null:
            RenderingServer.call_on_render_thread(
                reference_persistence.set_time.bind(
                    roundi(float(frame["time"]) * 1000000), frame.get("cut", false)
                )
            )
        # Hold the exact state while queued engine work completes. No delta integration.
        await RenderingServer.frame_post_draw
        await RenderingServer.frame_post_draw
        RenderingServer.call_on_render_thread(
            capture.request.bind(frame["path"], float(frame["time"]))
        )
        await capture.captured
    print("REFERENCE_CAPTURE_COMPLETE ", scene_id, " ", job["frames"].size())
    if reference_persistence != null:
        RenderingServer.call_on_render_thread(reference_persistence.dispose)
    RenderingServer.call_on_render_thread(capture.release)
    await capture.captured
    get_tree().quit()
