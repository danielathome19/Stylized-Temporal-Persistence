extends "res://scripts/m6_live_demo.gd"


func _ready() -> void:
    history_settings = {
        "synthesis": true, "synthesis_policy": "horizon", "packed": true, "slot_multiplier": 1.25
    }
    super()
