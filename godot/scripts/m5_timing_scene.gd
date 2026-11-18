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
    get_tree().root.size = Vector2i(job["width"], job["height"])
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
    effect3d = (
        load(
            (
                "res://scripts/m5_timing_effect.gd"
                if job["mode"] == "M5"
                else "res://scripts/timing_effect.gd"
            )
        )
        . new()
    )
    effect3d.job = job
    effect3d.config["method"] = "M3"
    # Equal native input requirements even for no-effect and M3 controls.
    effect3d.needs_motion_vectors = true
    effect3d.access_resolved_depth = true
    effect3d.finished.connect(done)
    var compositor := Compositor.new()
    compositor.compositor_effects = [effect3d]
    camera3d.compositor = compositor
    origin = Time.get_ticks_usec()


var origin := 0


func _process(_delta: float) -> void:
    var timestamp := Time.get_ticks_usec() - origin
    var t := timestamp / 1000000.0
    state_at(t)
    object3d.position = Vector3(sin(t * 2) * 1.5, sin(t) * .5, 0)
    object3d.rotation = Vector3(.2 * t, t, .3 * t)
    RenderingServer.call_on_render_thread(effect3d.set_time.bind(timestamp))


func done() -> void:
    set_process(false)
    print("TIMING_PILOT_COMPLETE")
    get_tree().quit()
