extends "res://scripts/comparison_scene.gd"
var objects: Array[MeshInstance3D] = []


func _ready() -> void:
    super._ready()
    var rng := RandomNumberGenerator.new()
    rng.seed = 21717
    for i in range(9):
        objects.append(
            box(
                Vector3(.2 + rng.randf() * .5, .3 + rng.randf() * .7, .2),
                Color.from_hsv(rng.randf(), .8, .9),
                Vector3.ZERO
            )
        )
    for x in range(-5, 6):
        for y in range(-3, 4):
            box(
                Vector3(.85, .85, .1),
                Color(.07 + .12 * ((x + y + 10) % 2), .13, .2),
                Vector3(x, y, -1.5)
            )
    box(Vector3(.3, 3, .2), Color(.6, .1, .4), Vector3(.4, 0, 1.1))


func _process(_delta: float) -> void:
    if actor == null:
        return
    var timestamp := (
        frame * 16667
        + (frame % 5) * 700
        + (120000 if frame >= 50 else 0)
        + (90000 if frame >= 95 else 0)
    )
    var t := timestamp / 1000000.0
    actor.position = Vector3(1.7 * sin(3.1 * t), .4 * cos(2.3 * t), .4 * sin(t))
    actor.rotation = Vector3(t * .9, t * 1.7, t * .6)
    for i in range(objects.size()):
        objects[i].position = Vector3(
            2.5 * sin(t * (.7 + i * .17) + i),
            1.2 * cos(t * (1.3 + i * .13) + i * .7),
            .6 * sin(t + i)
        )
        objects[i].rotation = Vector3(t * (i + 1) * .12, t * .8, i * .3)
    if job["scene"] == "camera_turn":
        camera3d.position = Vector3(1.8 * sin(t * .9), .5 * sin(t * 1.3), 5.5 + .8 * cos(t))
        camera3d.look_at(Vector3.ZERO)
    RenderingServer.call_on_render_thread(effect.set_time.bind(timestamp))
    frame += 1
    if frame >= 180:
        set_process(false)
        await RenderingServer.frame_post_draw
        RenderingServer.call_on_render_thread(effect.save_manifest)
        await effect.saved
        print("M5_SAMPLE_CAPTURE_COMPLETE")
        get_tree().quit()
