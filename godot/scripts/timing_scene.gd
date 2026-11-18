extends "res://scripts/reference_scene.gd"
var effect
var origin := 0


func _ready() -> void:
    var job: Dictionary = JSON.parse_string(
        FileAccess.get_file_as_string(OS.get_cmdline_user_args()[0])
    )
    scene_id = "C"
    get_tree().root.size = Vector2i(job["width"], job["height"])
    build_scene()
    effect = load("res://scripts/timing_effect.gd").new()
    effect.job = job
    effect.config["method"] = job["mode"] if job["mode"].begins_with("M") else "M1"
    effect.finished.connect(done)
    for child in get_children():
        if child is Camera3D:
            child.compositor.compositor_effects = [effect]
    origin = Time.get_ticks_usec()


func _process(_delta: float) -> void:
    if effect == null:
        return
    var t := Time.get_ticks_usec() - origin
    evaluate(t / 1000000.0)
    RenderingServer.call_on_render_thread(effect.set_time.bind(t))


func done() -> void:
    set_process(false)
    print("TIMING_PILOT_COMPLETE")
    get_tree().quit()
