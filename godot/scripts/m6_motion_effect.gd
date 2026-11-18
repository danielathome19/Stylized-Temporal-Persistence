extends "res://scripts/m5_live_effect.gd"
var sparse
var storage_config := {"tile": 16, "versions": 3, "history": 10, "epsilon": 0.0}


func _setup_history(rd: RenderingDevice, size: Vector2i) -> void:
    if sparse != null:
        sparse.release()
    sparse = (
        load(
            (
                "res://scripts/synthesis_history.gd"
                if storage_config.get("synthesis", false)
                else "res://scripts/shared_history.gd"
            )
        )
        . new()
    )
    var settings := storage_config.duplicate()
    settings["alpha"] = config["alpha"]
    sparse.setup(rd, size, settings)
    # M5's generic baseline core allocated four images we no longer need.
    for texture in core.histories + [core.previous, core.output]:
        rd.free_rid(texture)
        core.textures.erase(texture)
    core.histories.clear()
    core.previous = RID()
    core.output = RID()


func _store_history(source_image: RID, reset_history: bool = false) -> void:
    if reset_history:
        sparse.reset()
    sparse.capture(source_image)


func _history_result() -> RID:
    return sparse.output


func dispose() -> void:
    super()
    if sparse != null:
        sparse.release()
        sparse = null


# Optional end-of-run diagnostics only; never called in a timing path.
signal diagnostics_ready


func save_diagnostics(path: String) -> void:
    if sparse != null:
        var bytes: PackedByteArray = sparse.rd.buffer_get_data(sparse.metadata)
        var controls: Array = []
        for i in range(16):
            controls.append(bytes.decode_u32((sparse.control_offset + i) * 4))
        var maximum_versions := 0
        for t in range(sparse.tiles):
            maximum_versions = maxi(
                maximum_versions,
                bytes.decode_u32((3 * sparse.tiles * sparse.history + t * 8 + 1) * 4)
            )
        var file := FileAccess.open(path, FileAccess.WRITE)
        file.store_string(
            JSON.stringify(
                {
                    "capacity": sparse.capacity,
                    "tiles": sparse.tiles,
                    "used": controls[0],
                    "peak_used": controls[1],
                    "last_merges": controls[2],
                    "last_stores": controls[3],
                    "free_slots": controls[4],
                    "allocation_errors": controls[5],
                    "last_requested": controls[6],
                    "sequence": controls[7],
                    "peak_requested": controls[8],
                    "cumulative_merges_since_reset": controls[9],
                    "cumulative_stores_since_reset": controls[10],
                    "max_versions_per_tile": maximum_versions,
                    "history_payload_bytes": sparse.payload_bytes,
                    "reconstructed": reconstructed,
                    "startup_holds": startup_holds
                },
                "\t"
            )
        )
        file.close()
    diagnostics_ready.emit.call_deferred()
