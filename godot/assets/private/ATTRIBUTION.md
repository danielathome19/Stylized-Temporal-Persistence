# Private model attribution and setup

**Asset:** Young Link from *The Legend of Zelda: Majora's Mask*.

- **Original model creator:** Nintendo.
- **Model ripper:** spikey-vi.
- **Source:** [Majora's Mask Link 3d Model Rigged (Download), spikey-vi on DeviantArt](https://www.deviantart.com/spikey-vi/art/Majora%27s-Mask-Link-3d-Model-Rigged-%28Download%29-897266876).

We utilized the Young Link model from *The Legend of Zelda: Majora's Mask*, originally created by Nintendo and ripped by spikey-vi, for local temporal-rendering experiments. Our local asset collection includes `MM_Young_Link.glb` and the `mm_young_link/` folder containing textures and Blender, Collada and FBX files.

We retain the original creators' ownership and attribution. The model and its derived assets are excluded from the project source distribution; this project does not grant permission to redistribute them. We have not independently verified a redistribution license.

## Obtaining and converting the model

1. Visit the source page and download the ripped rig through its download link, subject to its access conditions and your permission to use the asset. This project does not mirror the download.
2. Extract the archive and open its `.blend` file in Blender. Preserve the mesh's armature modifier, vertex groups, armature and textures.
3. Select the model meshes and armature if exporting selected objects. Use **File > Export > glTF 2.0**, choose **glTF Binary (.glb)**, and export to `godot/assets/private/MM_Young_Link_rigged.glb`, preserving any existing export.
4. Enable skinning/armature export; include animations if animation tests are intended. The precise option labels depend on Blender version. Ensure embedded materials/textures are exported. Reopen the GLB and verify moving a bone deforms the mesh before using it as a skeletal test case.
5. Open the Godot project or run `godot/scenes/private_link_demo.tscn`. The demo loads the private GLB at runtime and is optional; the public procedural scenes do not depend on it.

Our initial `MM_Young_Link.glb` contains one mesh, 42 materials and 17 embedded images, plus limb-named nodes, but **zero glTF skins and zero animations**. Its mesh has no JOINTS_0/WEIGHTS_0 attributes. It supports rigid motion tests, not a verified skeletal-deformation experiment. We use the separate rigged export below for skeletal motion.

## Godot materials

Use Shading Mode **Unshaded**, Metallic **0**, Roughness **1**, Texture Filter **Linear**. The demo duplicates each imported surface material before applying these settings, preserving embedded textures and avoiding changes to the original GLB:

```gdscript
material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
material.metallic = 0.0
material.roughness = 1.0
material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR
```

We use these settings for an unlit appearance. Linear filtering does not reproduce the N64 texture filter exactly.


## Rig-preserving conversion completed locally

The source `mm_young_link/MM Young Link.blend` reports Blender 3.3.6, while the installed Blender is 2.91.0. We preserved that file and used a compatible source format instead: `experiments/run/convert_private_link.py` imported the original `mm_young_link/DAE/MM Young Link.dae` with its accompanying textures and exported a new `MM_Young_Link_rigged.glb`. We preserved the original files and initial GLB.

The new GLB contains one skin, 29 joints, JOINTS_0/WEIGHTS_0 attributes and 14 embedded images. Blender imported 18 weighted vertex groups; Godot loaded one Skeleton3D and 42 materials. There are no original animation clips. The optional demo applies deterministic procedural rotations to limb_13, limb_16 and limb_10, plus whole-model translation/rotation. These are research motions, not Nintendo animations. It prefers the rigged GLB and falls back to the initial rigid export if absent.

Reproduce the Collada fallback with the installed Blender (script refuses to overwrite an existing rigged export):

```powershell
& 'C:/Program Files/Blender Foundation/Blender 2.91/blender.exe' --background --disable-autoexec --python experiments/run/convert_private_link.py
```

For direct `.blend` conversion, use a compatible Blender version and follow the export instructions above. We used Collada for compatibility with Blender 2.91; direct conversion of the newer `.blend` requires a compatible Blender version.

Runtime loading follows the official [Godot GLTFDocument API](https://docs.godotengine.org/en/stable/classes/class_gltfdocument.html). Run `.\Godot_v4.8-beta3_win64.exe --path godot res://scenes/private_link_demo.tscn` for the interactive private demo, or `python experiments/run/private_link.py` for a 120-frame local smoke capture. Captures remain under ignored `results/raw/`.
