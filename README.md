# Finite-history temporal persistence experiments

Godot compute implementations and Python experiment/reference code. Used in the following paper:
 * "Temporal Persistence for Stylized Rendering: Historical Image Storage and Reconstruction" (DOI: [TBD](https://doi.org/10.48550/arXiv.2611.00000)) 


## Layout

- `godot/`: runnable procedural demo, dense histories, shared/synthesized tile histories, sparse pixel innovations, and capture/replay/timing resources.
- `experiments/run/`: exact-time capture, accuracy comparisons and GPU timing.
- `experiments/configs/`: experiment parameters and pinned portable-engine hashes.
- `analysis/`: independent arithmetic, metrics and result generation.
- `tests/`: relevant mathematical, image-reference, scheduling, allocation and metric checks.

The experiments used Python 3.13.2; Python package versions are pinned in `requirements.txt`. Native follow-up experiments use Godot 4.7.2 stable; original rendering and development experiments used 4.8 beta3. `python experiments/run/paper_setup.py` verifies/downloads portable official binaries into the workspace. It does not install system software.

```powershell
python -m pip install -r requirements.txt
python experiments/run/paper_setup.py
New-Item -ItemType Directory -Force results/raw, results/processed, results/figures
python -m unittest discover -s tests
python experiments/run/paper_capture.py
python experiments/run/paper_precision.py <input-run-directory>
python experiments/run/paper_timing.py <input-run-directory>
python experiments/run/paper_format_edges.py
python analysis/paper_storage_report.py <precision-run-directory> <timing-run-directory>
```

To regenerate the manuscript figures from the recorded evidence, run `python analysis/publication_figures.py <evidence-root> <output-figures-directory>`. The evidence root must contain `results/processed/paper_storage.json` and its referenced raw precision and crossing datasets. This command does not perform new experiments.

Use the new input directory printed by the capture command for the following runs. Every run writes a new directory under `results/raw`. Existing frozen datasets may also be passed by absolute path. GPU timing requires Forward+ Vulkan and currently uses `nvidia-smi` for telemetry. GPU timings exclude input preload and image readback; accuracy runs deliberately read back images. CPU prototype durations are not GPU timing evidence.

The default Godot project launches an asset-free procedural shared-history demonstration. `godot/scenes/synthesis_live_demo.tscn` selects packed synthesis. No Nintendo model is required. Optional private-model setup and attribution are in `godot/assets/private/ATTRIBUTION.md`; the model, textures and private captures are not distributed here.

The core target is a normalized ten-sample exponential history at 20 Hz with 250 ms half-life. This experimental rate is not a claim about every version or effect in Majora's Mask. Recorded resource totals count explicit history/workspace buffers; physical residency and driver overhead are unmeasured. The paper describes the scene, source-precision and single-GPU limits of the results.

Raw experiment outputs are excluded from this source directory and are generated under `results/` by the experiment runners. Recorded datasets are maintained separately; this README does not specify a public data archive. Small default-run identifiers in report scripts refer to the recorded development evidence; regenerate the corresponding run or use the parameterized native-paper runners above. 

## License

Project source code is licensed under the [MIT License](LICENSE.md). Third-party dependencies retain their own licenses. Nintendo models, textures, reference screenshots and other third-party assets are not covered by this license; private model assets and captures are excluded from this repository. Attribution and optional local setup are documented in `godot/assets/private/ATTRIBUTION.md`. The source-code license does not assign a license to the manuscript or its figures.

