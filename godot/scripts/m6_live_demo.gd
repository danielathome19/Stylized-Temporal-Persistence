extends "res://scripts/reference_scene.gd"
var camera3d: Camera3D
var object3d: MeshInstance3D
var effect3d
var kind := "translation"
var history_settings: Dictionary = {}


func state_at(t: float) -> void:
    object3d.position = Vector3(1.5 * sin(t * 2), .5 * sin(t), 0)
    object3d.rotation = Vector3(.25 * sin(t), t, .15 * cos(t))
    camera3d.position = Vector3(0, 0, 6)
    if kind == "rotation":
        object3d.rotation = Vector3(.5 * t, 2 * t, .3 * t)
    if kind == "camera":
        camera3d.position.x = t * 1.5


func _ready() -> void:
    get_tree().root.size = Vector2i(640, 360)
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
    effect3d = load("res://scripts/m6_motion_effect.gd").new()
    effect3d.storage_config.merge(history_settings, true)
    var compositor := Compositor.new()
    compositor.compositor_effects = [effect3d]
    camera3d.compositor = compositor
    var layer := CanvasLayer.new()
    add_child(layer)
    var label := Label.new()
    label.position = Vector2(10, 10)
    label.text = "Shared temporal history / procedural 3D\nEsc: close | R: reset history"
    if history_settings.get("synthesis", false):
        label.text = "Temporal-profile synthesis / packed RGBA8\nEsc: close | R: reset history"
    layer.add_child(label)
    origin = Time.get_ticks_usec()


var origin := 0
var count := 0


func _process(_delta: float) -> void:
    var timestamp := Time.get_ticks_usec() - origin
    state_at(timestamp / 1000000.0)
    RenderingServer.call_on_render_thread(effect3d.set_time.bind(timestamp))
    count += 1
    if count == 240 and not OS.get_cmdline_user_args().is_empty():
        set_process(false)
        await RenderingServer.frame_post_draw
        var args := OS.get_cmdline_user_args()
        if not args.is_empty():
            get_viewport().get_texture().get_image().save_png(args[0])
        print("LIVE_M6_COMPLETE history_payload_bytes=", effect3d.sparse.payload_bytes)
        RenderingServer.call_on_render_thread(effect3d.dispose)
        await RenderingServer.frame_post_draw
        get_tree().quit()


func _unhandled_key_input(event: InputEvent) -> void:
    if event is InputEventKey and event.pressed and not event.echo:
        if event.keycode == KEY_ESCAPE:
            get_tree().quit()
        if event.keycode == KEY_R:
            origin = Time.get_ticks_usec()
            RenderingServer.call_on_render_thread(effect3d.set_time.bind(0, true))


func _exit_tree() -> void:
    if effect3d != null:
        RenderingServer.call_on_render_thread(effect3d.dispose)
