"""Rebuild scene-A endpoint preview and descriptive pilot table from raw evidence."""

import sys
import json
import math
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.temporal_reference import recursive_oracle

p = Path(sys.argv[1])
rows = json.loads((p / "endpoints.json").read_text())
for rel, h in json.loads((p / "sha256.json").read_text()).items():
    assert hashlib.sha256((p / rel).read_bytes()).hexdigest() == h, rel
summary = []
for target in ["sampled_recursive", "live_finite"]:
    for schedule in ["20hz", "10hz", "irregular"]:
        record = {"target": target, "schedule": schedule}
        for m in ["M1", "M2", "M3", "M4"]:
            record[m] = float(
                np.mean(
                    [
                        r["motion_region_mse"]
                        for r in rows
                        if r["target"] == target
                        and r["schedule"] == schedule
                        and r["method"] == m
                    ]
                )
            )
        summary.append(record)
(ROOT / "results/processed/fidelity_summary.json").write_text(
    json.dumps(
        {
            "run": p.name,
            "aggregation": "equal scene mean, endpoint only, descriptive",
            "results": summary,
        },
        indent=2,
    )
)
canvas = Image.new("RGB", (1600, 630), (24, 24, 26))
draw = ImageDraw.Draw(canvas)
a = 2 ** (-0.05 / 0.25)
hist = [
    np.fromfile(p / "A" / f"{t}.bin", dtype="<f4").reshape(180, 320, 4)
    for t in range(0, 1000001, 50000)
]
target = recursive_oracle(hist, [a] * 20)
for y, s in enumerate(["20hz", "10hz", "irregular"]):
    for x, m in enumerate(["Exact sampled target", "M1", "M2", "M3", "M4"]):
        im = (
            target
            if x == 0
            else np.fromfile(p / "A" / f"{s}_{m}" / "1000000.bin", dtype="<f4").reshape(
                180, 320, 4
            )
        )
        rgb = im[..., :3]
        rgb = np.where(
            rgb <= 0.0031308,
            12.92 * rgb,
            1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055,
        )
        canvas.paste(
            Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5)),
            (x * 320, y * 210 + 30),
        )
        draw.text((x * 320 + 6, y * 210 + 8), s + " / " + m, fill="white")
canvas.save(ROOT / "results/figures/fidelity_pilot.png")
print(json.dumps(summary, indent=2))
