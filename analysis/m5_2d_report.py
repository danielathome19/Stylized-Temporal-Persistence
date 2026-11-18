"""Audit frozen 2D policy tests, retain all comparisons and render failures."""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

p = Path(sys.argv[1])
root = Path(__file__).resolve().parents[1]
for rel, h in json.loads((p / "sha256.json").read_text()).items():
    assert hashlib.sha256((p / rel).read_bytes()).hexdigest() == h, rel
meta = json.loads((p / "metadata.json").read_text())
rows = meta["results"]
groups = []
for group in ["integer_translation", "subpixel_translation", "accelerated"]:
    groups.append(
        {
            "group": group,
            "cases": 12,
            "sample_mse": {
                policy: float(
                    np.mean(
                        [
                            r["sample"]["mse_rgb"]
                            for r in rows
                            if r["group"] == group and r["policy"] == policy
                        ]
                    )
                )
                for policy in ["hold", "crossfade", "strict", "one_sided", "gated"]
            },
        }
    )
comparisons = []
for base in ["hold", "crossfade", "strict", "one_sided"]:
    differences = [
        {
            "case": r["case"],
            "delta_mse": r["sample"]["mse_rgb"]
            - next(
                b["sample"]["mse_rgb"]
                for b in rows
                if b["case"] == r["case"] and b["policy"] == base
            ),
        }
        for r in rows
        if r["policy"] == "gated"
    ]
    comparisons.append(
        {
            "candidate": "gated",
            "baseline": base,
            "wins": sum(d["delta_mse"] < -1e-12 for d in differences),
            "ties": sum(abs(d["delta_mse"]) <= 1e-12 for d in differences),
            "losses": sum(d["delta_mse"] > 1e-12 for d in differences),
            "differences": differences,
        }
    )
summary = {
    "run": p.name,
    "groups": groups,
    "comparisons": comparisons,
    "max_oracle_error": max(r["oracle_max_abs"] for r in rows),
}
(root / "results/processed/m5_2d_summary.json").write_text(
    json.dumps(summary, indent=2)
)
# Deterministic selection: largest remaining one-sided error in each category,
# plus the worst gated-versus-one-sided regression, not handpicked successes.
names = [
    max(
        [r for r in rows if r["policy"] == "one_sided" and r["group"] == g],
        key=lambda r: r["sample"]["mse_rgb"],
    )["case"]
    for g in ["integer_translation", "subpixel_translation"]
]
names.append(
    max(
        next(c for c in comparisons if c["baseline"] == "one_sided")["differences"],
        key=lambda d: d["delta_mse"],
    )["case"]
)
canvas = Image.new("RGB", (1200, 600), (24, 24, 26))
draw = ImageDraw.Draw(canvas)
for y, name in enumerate(names):
    for x, label in enumerate(["target", "crossfade", "one_sided", "gated"]):
        filename = "target.bin" if label == "target" else label + "_sample.bin"
        im = np.fromfile(p / name / filename, dtype="<f4").reshape(65, 97, 4)[..., :3]
        rgb = np.where(
            im <= 0.0031308, 12.92 * im, 1.055 * np.maximum(im, 0) ** (1 / 2.4) - 0.055
        )
        image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5)).resize(
            (250, 168), Image.Resampling.NEAREST
        )
        canvas.paste(image, (x * 300, y * 200 + 28))
        draw.text((x * 300 + 3, y * 200 + 7), name + " / " + label, fill="white")
canvas.save(root / "results/figures/m5_2d_failures.png")
print(
    json.dumps(
        {
            "groups": groups,
            "max_oracle_error": summary["max_oracle_error"],
            "figure_cases": names,
        },
        indent=2,
    )
)
