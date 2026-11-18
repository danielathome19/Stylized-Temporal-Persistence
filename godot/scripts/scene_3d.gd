extends "res://scripts/reference_scene.gd"
var camera3d: Camera3D
var object3d: MeshInstance3D
var effect3d
var kind := "translation"


func state_at(t: float) -> void:
    object3d.position = Vector3(-0.6 + 2 * t, t, 0)
    object3d.rotation = Vector3.ZERO
    camera3d.position = Vector3(0, 0, 6)
    if kind == "rotation":
        object3d.rotation = Vector3(.5 * t, 2 * t, .3 * t)
    if kind == "camera":
        camera3d.position.x = t * 1.5


func _ready() -> void:
    var job: Dictionary = JSON.parse_string(
        FileAccess.get_file_as_string(OS.get_cmdline_user_args()[0])
    )
    kind = job["scene"]
    get_tree().root.size = Vector2i(320, 180)
    camera3d = Camera3D.new()
    camera3d.fov = 60
    camera3d.near = .1
    camera3d.far = 100
    add_child(camera3d)
    var env := Environment.new()
    env.background_mode = Environment.BG_COLOR
    env.background_color = Color.BLACK
    env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
    camera3d.environment = env
    object3d = box(self, Vector3(1, .8, .5), Color.WHITE, Vector3.ZERO)
    box(self, Vector3(10, 7, .1), Color(.1, .2, .6), Vector3(0, 0, -1))
    if kind == "occlusion":
        box(self, Vector3(.35, 2.5, .2), Color(1, 0, 0), Vector3(0, 0, 1))
    effect3d = load("res://scripts/capture_3d.gd").new()
    var compositor := Compositor.new()
    compositor.compositor_effects = [effect3d]
    camera3d.compositor = compositor
    for frame in job["frames"]:
        await get_tree().process_frame
        state_at(frame["time"])
        var screen := camera3d.unproject_position(object3d.position)
        var request: Dictionary = frame.duplicate()
        request["projected_center"] = [screen.x, screen.y]
        request["near"] = camera3d.near
        request["far"] = camera3d.far
        RenderingServer.call_on_render_thread(effect3d.request.bind(request))
        await effect3d.captured
    RenderingServer.call_on_render_thread(effect3d.release)
    await effect3d.captured
    print("CAPTURE_3D_COMPLETE")
    get_tree().quit()
