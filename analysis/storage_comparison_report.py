"""Regenerate matched storage tables and figures from immutable pilot outputs."""

import sys
import json
import hashlib
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def verified(path):
    for name, digest in json.loads((path / "sha256.json").read_text()).items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest, name
    data = json.loads((path / "metadata.json").read_text())
    assert data["outcome"] == "passed"
    return data


def main():
    source = Path(sys.argv[1])
    data = verified(source)
    rows = data["results"]
    scenes = ["rotation", "occlusion", "camera", "link"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    table = [
        "| Scene | Slots/tile | Lifetime MSE | Current MSE | Distance MSE | Lifetime vs current |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ax, ratio in zip(axes, [1.25, 2]):
        for i, policy in enumerate(["lifetime", "current", "distance"]):
            values = [
                next(
                    r["motion_region_mse"]
                    for r in rows
                    if r["scene"] == s and r["policy"] == policy and r["ratio"] == ratio
                )
                for s in scenes
            ]
            ax.bar(
                np.arange(4) + (i - 1) * 0.25,
                np.maximum(values, 1e-15),
                width=0.25,
                label=policy,
            )
        ax.set(
            yscale="log",
            xticks=np.arange(4),
            xticklabels=scenes,
            ylabel="Motion-region MSE (RGB)",
            title=f"{ratio:g} slots per tile (global budget)",
        )
        ax.set_ylim(1e-16, 1e-2)
        ax.legend()
        for scene in scenes:
            subset = {
                r["policy"]: r
                for r in rows
                if r["scene"] == scene and r["ratio"] == ratio
            }
            a, b, c = [
                subset[p]["motion_region_mse"]
                for p in ["lifetime", "current", "distance"]
            ]
            change = f"{100*(a/b-1):+.2f}%" if b > 1e-12 else "roundoff only"
            table.append(
                f"| {scene} | {ratio} | {a:.6g} | {b:.6g} | {c:.6g} | {change} |"
            )
    fig.suptitle(
        "Compression error against float64 dense composite; values below 1e-15 plotted at 1e-15"
    )
    dest = ROOT / "results/figures"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / "storage_fidelity.png", dpi=160)
    plt.close(fig)
    report = [
        "# Matched storage pilot",
        "",
        f"Fidelity source: `{source.name}`.",
        "",
        *table,
        "",
        "Negative percentages mean lower error. These deterministic sequences are not independent statistical replicates.",
    ]
    summary = {
        "fidelity_run": source.name,
        "fidelity": [
            {k: v for k, v in r.items() if not k.startswith("per_capture")}
            for r in rows
        ],
    }
    # Inspect the synthetic camera failure at the largest lifetime-minus-distance full-frame MSE.
    a = next(
        r
        for r in rows
        if r["scene"] == "camera" and r["policy"] == "lifetime" and r["ratio"] == 1.25
    )
    b = next(
        r
        for r in rows
        if r["scene"] == "camera" and r["policy"] == "distance" and r["ratio"] == 1.25
    )
    index = int(np.argmax(np.array(a["per_capture_mse"]) - b["per_capture_mse"]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
    for ax, policy, ratio in zip(
        axes, ["dense", "lifetime", "distance"], [10, 1.25, 1.25]
    ):
        im = np.fromfile(
            source / f"camera_{policy}_{ratio}" / f"{index}.bin", "<f4"
        ).reshape(90, 160, 4)
        ax.imshow(np.clip(im[..., :3], 0, 1) ** (1 / 2.2))
        ax.set_title(policy)
        ax.axis("off")
    fig.suptitle(f"Camera counterexample, capture {index}; display gamma 2.2 only")
    fig.savefig(dest / "storage_camera_counterexample.png", dpi=160)
    plt.close(fig)
    summary["counterexample_capture"] = index
    if len(sys.argv) > 2:
        timing_path = Path(sys.argv[2])
        timing = verified(timing_path)
        stats = {}
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        report += [
            "",
            "Timing source: `"
            + timing_path.name
            + "`. Five independent processes per method; same camera stream, 1.25 slots/tile.",
            "",
            "| Method | Median of run medians (ms) | Run median range (ms) | Median run p95 (ms) |",
            "|---|---:|---:|---:|",
        ]
        for i, mode in enumerate(["none", "dense", "lifetime", "current", "distance"]):
            runs = [r for r in timing["results"] if r["mode"] == mode]
            assert len(runs) == 5
            med = np.array([r["gpu_ms_median_p95_p99"][0] for r in runs])
            p95 = np.median([r["gpu_ms_median_p95_p99"][1] for r in runs])
            stats[mode] = {
                "median_ms": float(np.median(med)),
                "range_ms": [float(med.min()), float(med.max())],
                "median_p95_ms": float(p95),
            }
            ax.scatter([i] * 5, med, label=mode)
            ax.plot([i - 0.2, i + 0.2], [np.median(med)] * 2, color="black")
            report.append(
                f"| {mode} | {np.median(med):.6f} | {med.min():.6f}–{med.max():.6f} | {p95:.6f} |"
            )
        ax.set(
            xticks=range(5),
            xticklabels=list(stats),
            yscale="symlog",
            ylabel="GPU ms per capture (symlog scale)",
            title="Each dot is one process median; lines are medians of five runs",
        )
        fig.savefig(dest / "storage_gpu_timing.png", dpi=160)
        plt.close(fig)
        summary.update(timing_run=timing_path.name, timing=stats)
    (ROOT / "results/processed/storage_comparison.json").write_text(
        json.dumps(summary, indent=2)
    )
    (ROOT / "results/processed/storage_comparison.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
