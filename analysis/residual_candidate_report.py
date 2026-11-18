"""Summarize analytical candidate evidence without omitting extra-slot controls."""

import sys
import json
import hashlib
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = Path(sys.argv[1])
    for name, digest in json.loads((path / "sha256.json").read_text()).items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest, name
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["outcome"] == "passed"
    lines = [
        "# Residual candidate: analytical CPU results",
        "",
        f'Source: `{path.name}`. Projection dimensions: {meta["projection_dimensions"]}.',
        "",
        "| Scene | Lifetime75 MSE | Residual75 MSE | Lifetime76 MSE | Lifetime77 MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene in ["camera", "rotation", "occlusion", "link"]:
        group = {r["policy"]: r for r in meta["results"] if r["scene"] == scene}
        values = [
            group[p]["motion_region_mse"]
            for p in [
                "lifetime75",
                "residual75",
                "lifetime_equal_bytes",
                "lifetime_next_slot",
            ]
        ]
        lines.append(
            "| " + scene + " | " + " | ".join(f"{v:.7g}" for v in values) + " |"
        )
    lines += [
        "",
        "Candidate projected payload352144bytes;76slots348144bytes;77slots352244bytes. Float64 CPU fidelity, not measured GPU allocation. The77-slot control is100bytes above candidate budget. No robust memory–fidelity advantage established.",
    ]
    (ROOT / "results/processed/residual_candidate.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), layout="constrained")
    # Same pre-existing counterexample timestamp, not selected to favor the new candidate.
    for ax, policy, title in zip(
        axes,
        ["lifetime75", "residual75", "lifetime_equal_bytes", "lifetime_next_slot"],
        ["Lifetime75", "Residual75", "Lifetime76", "Lifetime77 (+100bytes)"],
    ):
        im = np.fromfile(path / f"camera_{policy}/34.bin", "<f4").reshape(90, 160, 4)
        ax.imshow(np.clip(im[..., :3], 0, 1) ** (1 / 2.2))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Saved camera failure, capture34; display gamma2.2")
    fig.savefig(ROOT / "results/figures/residual_camera.png", dpi=160)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
