extends "res://scripts/comparison_scene.gd"


func _process(_delta: float) -> void:
    pass


func _ready() -> void:
    super._ready()
    var capture = load("res://scripts/reference_capture.gd").new()
    var compositor := Compositor.new()
    compositor.compositor_effects = [capture]
    camera3d.compositor = compositor
    # No reconstructed samples: independently render each exact scheduled time.
    for i in range(16):
        var t := i * .05
        actor.position = Vector3(1.2 * sin(t * 2), .3 * cos(t * 3), 0)
        actor.rotation = Vector3(.5 * t, 2 * t, .3 * t)
        if job["scene"] == "camera":
            camera3d.position.x = .8 * sin(t * 2)
        await RenderingServer.frame_post_draw
        await RenderingServer.frame_post_draw
        RenderingServer.call_on_render_thread(
            capture.request.bind(job["folder"] + "/native_" + str(i), t)
        )
        await capture.captured
    RenderingServer.call_on_render_thread(capture.release)
    await capture.captured
    print("PAPER_CAPTURE_COMPLETE")
    get_tree().quit()
