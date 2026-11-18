"""Verify raw reconstruction evidence and create a target/baseline/failure preview."""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
p = Path(sys.argv[1])
for rel, h in json.loads((p / "sha256.json").read_text()).items():
    assert hashlib.sha256((p / rel).read_bytes()).hexdigest() == h, rel
jobs = {j["name"]: j for j in json.loads((p / "job.json").read_text())["cases"]}
names = ["A_50000", "translation", "occlusion", "hidden_midpoint", "acceleration"]
canvas = Image.new("RGB", (1280, 5 * 160), (24, 24, 26))
draw = ImageDraw.Draw(canvas)
for row, name in enumerate(names):
    j = jobs[name]
    shape = (j["height"], j["width"], 4)
    folder = p / name
    c0 = np.fromfile(folder / "color0.bin", dtype="<f4").reshape(shape)
    c1 = np.fromfile(folder / "color1.bin", dtype="<f4").reshape(shape)
    target = np.fromfile(folder / "target.bin", dtype="<f4").reshape(shape)
    out = np.fromfile(folder / "output.bin", dtype="<f4").reshape(shape)
    images = [target, c0, (c0 + c1) / 2, out]
    for col, (label, im) in enumerate(
        zip(["Exact target", "Hold", "Crossfade", "Reconstruction"], images)
    ):
        rgb = im[..., :3]
        rgb = np.where(
            rgb <= 0.0031308,
            12.92 * rgb,
            1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055,
        )
        image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5))
        image.thumbnail((320, 130), Image.Resampling.NEAREST)
        if name != "A_50000":
            image = image.resize((260, 130), Image.Resampling.NEAREST)
        canvas.paste(image, (col * 320, row * 160 + 28))
        draw.text((col * 320 + 4, row * 160 + 6), name + " / " + label, fill="white")
canvas.save(ROOT / "results/figures/reconstruction_pilot.png")
meta = json.loads((p / "metadata.json").read_text())
(ROOT / "results/processed/reconstruction_summary.json").write_text(
    json.dumps(
        {
            "run": p.name,
            "results": meta["results"],
            "trail_endpoint": meta["trail_endpoint"],
        },
        indent=2,
    )
)
print("Verified", len(json.loads((p / "sha256.json").read_text())), "files")
