extends Node3D


func _ready() -> void:
    var job: Dictionary = JSON.parse_string(
        FileAccess.get_file_as_string(OS.get_cmdline_user_args()[0])
    )
    get_tree().root.size = Vector2i(640, 360)
    get_tree().root.content_scale_size = Vector2i(job["width"], job["height"])
    get_tree().root.content_scale_mode = Window.CONTENT_SCALE_MODE_VIEWPORT
    var camera := Camera3D.new()
    add_child(camera)
    var effect = load("res://scripts/storage_timing_effect.gd").new()
    effect.job = job
    effect.finished.connect(done)
    var compositor := Compositor.new()
    compositor.compositor_effects = [effect]
    camera.compositor = compositor


func done() -> void:
    print("STORAGE_TIMING_COMPLETE")
    get_tree().quit()
