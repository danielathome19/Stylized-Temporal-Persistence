"""Audit native 3D captures/reconstruction and generate scene comparison."""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
p = Path(sys.argv[1])
meta = json.loads((p / "metadata.json").read_text())
ref = Path(meta["source_reference"])
for folder in [p, ref]:
    for rel, h in json.loads((folder / "sha256.json").read_text()).items():
        assert hashlib.sha256((folder / rel).read_bytes()).hexdigest() == h, rel
summary = []
canvas = Image.new("RGB", (1600, 4 * 210), (24, 24, 26))
draw = ImageDraw.Draw(canvas)
for y, scene in enumerate(["translation", "rotation", "occlusion", "camera"]):
    row = {"scene": scene}
    for x, policy in enumerate(["target", "hold", "crossfade", "strict", "one_sided"]):
        path = (
            ref / scene / "target_color.bin"
            if policy == "target"
            else p / (scene + "_" + policy) / "result_sample.bin"
        )
        im = np.fromfile(path, dtype="<f4").reshape(180, 320, 4)[..., :3]
        rgb = np.where(
            im <= 0.0031308, 12.92 * im, 1.055 * np.maximum(im, 0) ** (1 / 2.4) - 0.055
        )
        image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5))
        canvas.paste(image, (x * 320, y * 210 + 28))
        draw.text((x * 320 + 4, y * 210 + 6), scene + " / " + policy, fill="white")
        if policy != "target":
            result = next(
                r
                for r in meta["results"]
                if r["scene"] == scene and r["policy"] == policy
            )
            row[policy] = result["sample"]["mse_rgb"]
    row["one_sided_reduction_vs_crossfade"] = 1 - row["one_sided"] / row["crossfade"]
    summary.append(row)
canvas.save(ROOT / "results/figures/native_3d_comparison.png")
(ROOT / "results/processed/native_3d_summary.json").write_text(
    json.dumps(
        {"run": p.name, "calibration": meta["calibration"], "results": summary},
        indent=2,
    )
)
print(json.dumps(summary, indent=2))
