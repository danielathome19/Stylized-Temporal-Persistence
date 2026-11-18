extends "res://scripts/m5_live_effect.gd"
signal saved
var folder := ""
var samples: Array = []


func _store_history(source_image: RID, reset_history: bool = false) -> void:
    var path := folder + "/sample_" + str(samples.size()) + ".bin"
    var file := FileAccess.open(path, FileAccess.WRITE)
    file.store_buffer(core.rd.texture_get_data(source_image, 0))
    file.close()
    samples.append(
        {
            "path": path,
            "target_us": time_us if reset_history else target_time,
            "delivered_us": time_us,
            "width": dimensions.x,
            "height": dimensions.y
        }
    )
    super._store_history(source_image, reset_history)


func save_manifest() -> void:
    var file := FileAccess.open(folder + "/samples.json", FileAccess.WRITE)
    file.store_string(
        JSON.stringify(
            {"samples": samples, "reconstructed": reconstructed, "startup_holds": startup_holds},
            "\t"
        )
    )
    file.close()
    dispose()
    saved.emit.call_deferred()
