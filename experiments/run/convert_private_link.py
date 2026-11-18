"""Run with Blender --background --disable-autoexec --python this_file.
Reads the supplied original Collada rip; writes a separate private rigged GLB.
The installed Blender 2.91 cannot safely read the supplied 3.3.6 blend.
"""

import bpy
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
private = root / "godot/assets/private"
source = private / "mm_young_link/DAE/MM Young Link.dae"
output = private / "MM_Young_Link_rigged.glb"
if output.exists():
    raise RuntimeError("Refusing to overwrite existing rigged export")
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.collada_import(filepath=str(source))
report = {
    "blender": bpy.app.version_string,
    "source": str(source.relative_to(root)),
    "output": str(output.relative_to(root)),
    "objects": [],
}
for o in bpy.data.objects:
    report["objects"].append(
        {
            "name": o.name,
            "type": o.type,
            "groups": len(o.vertex_groups) if o.type == "MESH" else 0,
            "bones": len(o.data.bones) if o.type == "ARMATURE" else 0,
            "modifiers": [m.type for m in o.modifiers],
        }
    )
assert any(
    o.type == "MESH" and len(o.vertex_groups) > 0 for o in bpy.data.objects
), "No skin weights imported"
for im in bpy.data.images:
    if im.source == "FILE":
        candidate = source.parent / Path(im.filepath.replace("\\", "/")).name
        if candidate.exists():
            im.filepath = str(candidate)
            im.reload()
# Blender 2.91 defaults to GLB; omitting the property avoids its headless UI callback.
bpy.ops.export_scene.gltf(
    filepath=str(output), export_skins=True, export_animations=True, export_apply=False
)
(private / "rigged_conversion.json").write_text(json.dumps(report, indent=2))
print("LINK_RIGGED_EXPORT " + json.dumps(report))
