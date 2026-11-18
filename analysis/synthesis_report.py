"""Regenerate synthesis accuracy, byte-budget controls and run-level timing report."""

import sys
import json
import hashlib
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"


def checked(name):
    path = RAW / name
    for f, h in json.loads((path / "sha256.json").read_text()).items():
        assert hashlib.sha256((path / f).read_bytes()).hexdigest() == h, f
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["outcome"] == "passed"
    return path, meta


def main():
    quant_path, quant = checked("20261117T063015Z-dense-quantization-79a01ecf")
    rows = []
    visual = None
    pairs = [
        (
            "20261117T063243Z-synthesis-gpu-22a92294",
            "20261117T061916Z-synthesis-candidate-765511fd",
        ),
        (
            "20261117T063217Z-synthesis-gpu-d21f12f2",
            "20261117T062342Z-synthesis-candidate-1238859d",
        ),
    ]
    for gpu, cpu in pairs:
        path, meta = checked(gpu)
        _, base = checked(cpu)
        source = RAW / meta["source"]
        job = json.loads((path / "job.json").read_text())
        for case in job["cases"]:
            if case["name"].startswith("odd_reset"):
                continue
            scene, policy = case["name"].rsplit("_", 1)
            images = np.array(
                [
                    np.fromfile(f["input"], "<f4")
                    .reshape(90, 160, 4)[..., :3]
                    .astype(float)
                    for f in case["frames"]
                ]
            )
            region = np.any(np.max(abs(images - images[0]), axis=0) > 1e-6, axis=2)
            full = []
            motion = []
            for i, f in enumerate(case["frames"]):
                start = max(0, i - 9)
                w = 2 ** (-0.05 / 0.25 * np.arange(i - start, -1, -1))
                w /= w.sum()
                truth = np.einsum("n,nhwc->hwc", w, images[start : i + 1])
                actual = np.fromfile(f["output"], "<f4").reshape(90, 160, 4)[..., :3]
                err = (actual - truth) ** 2
                full.append(float(err.mean()))
                motion.append(float(err[region].mean()))
                if scene == "crossing" and policy == "horizon" and i == 48:
                    controls = []
                    for levels in [np.array([7, 7, 3]), np.array([31, 63, 31])]:
                        controls.append(
                            np.einsum(
                                "n,nhwc->hwc",
                                w,
                                np.rint(np.clip(images[start : i + 1], 0, 1) * levels)
                                / levels,
                            )
                        )
                    visual = [truth, actual, *controls]
            result = {
                "scene": scene,
                "method": "packed_" + policy,
                "payload_bytes": 142684,
                "full_mse": float(np.mean(full)),
                "motion_region_mse": float(np.mean(motion)),
                "per_capture_region_mse": motion,
            }
            rows.append(result)
        rows.extend(
            dict(
                scene=r["scene"],
                method=r["policy"],
                payload_bytes=(
                    base["candidate_payload_bytes"]
                    if r["policy"] in ["horizon", "independent"]
                    else (
                        base["baseline_payload_bytes"]
                        if r["policy"] == "lifetime_equal_bytes"
                        else None
                    )
                ),
                full_mse=r["full_mse"],
                motion_region_mse=r["motion_region_mse"],
            )
            for r in base["results"]
        )
    rows.extend(
        dict(
            scene=r["scene"],
            method="RGB" + "".join(map(str, r["bits"])),
            payload_bytes=r["history_payload_bytes"],
            full_mse=r["full_mse"],
            motion_region_mse=r["motion_region_mse"],
        )
        for r in quant["results"]
    )
    _, timing = checked("20261117T061844Z-storage-timing-e622bf99")
    times = {}
    for mode in ["horizon", "independent", "lifetime"]:
        runs = sorted(
            (r for r in timing["results"] if r["mode"] == mode),
            key=lambda r: r["block"],
        )
        times[mode] = {
            "run_quantiles_ms": [r["gpu_ms_median_p95_p99"] for r in runs],
            "median_run_quantiles_ms": np.median(
                [r["gpu_ms_median_p95_p99"] for r in runs], axis=0
            ).tolist(),
        }
    report = {
        "scope": "160x90 frozen post-M5 compression accuracy; exact finite-history composition of supplied inputs, not scene oracle; packed GPU outputs, float CPU controls, dense quantization CPU controls; no packed timing",
        "rows": rows,
        "float_GPU_timing": times,
        "sources": [g for pair in pairs for g in pair]
        + [quant_path.name, timing["run_id"]],
    }
    (ROOT / "results/processed/synthesis.json").write_text(json.dumps(report, indent=2))
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    methods = [
        "horizon",
        "lifetime_equal_bytes",
        "packed_horizon",
        "packed_independent",
        "RGB332",
        "RGB565",
        "RGB888",
    ]
    lines = [
        "# Temporal-profile synthesis results",
        "",
        report["scope"],
        "",
        "Mean motion-region RGB MSE. Development: first four scenes; new scenes: last two.",
        "",
        "| Scene | " + " | ".join(methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for scene in scenes:
        lines.append(
            "| "
            + scene
            + " | "
            + " | ".join(
                f"{next(r['motion_region_mse'] for r in rows if r['scene']==scene and r['method']==m):.6g}"
                for m in methods
            )
            + " |"
        )
    lines += [
        "",
        "Payload: float horizon 373084; endpoint equal-budget 372744; packed synthesis 142684; RGB332 144000; RGB565 288000; RGB888 432000 bytes. Dense controls count packed RGB payload only, favoring the controls; all omit common images/driver padding.",
        "",
        "Float GPU median / p95 / p99 (median of five process quantiles):",
    ]
    for m, t in times.items():
        lines.append(f"- {m}: {t['median_run_quantiles_ms']} ms")
    lines += [
        "",
        "The new scenes are already exposed after this experiment; future tuning cannot treat them as unseen. No perceptual, broad speed or universal novelty claim.",
        "",
        "Sources: " + ", ".join(report["sources"]),
    ]
    (ROOT / "results/processed/synthesis.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for m in methods:
        ax.plot(
            scenes,
            [
                next(
                    r["motion_region_mse"]
                    for r in rows
                    if r["scene"] == s and r["method"] == m
                )
                for s in scenes
            ],
            "o-",
            label=m,
        )
    ax.set(
        yscale="log",
        ylabel="Mean motion-region RGB MSE",
        title="Finite-history compression: same frozen inputs",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(ROOT / "results/figures/synthesis_accuracy.png", dpi=160)
    if visual:
        fig, axes = plt.subplots(2, 4, figsize=(12, 5), layout="constrained")
        for i, (im, title) in enumerate(
            zip(
                visual,
                [
                    "Exact input composite",
                    "Packed synthesis",
                    "Dense RGB332",
                    "Dense RGB565",
                ],
            )
        ):
            axes[0, i].imshow(np.clip(im, 0, 1), interpolation="nearest")
            axes[0, i].set_title(title)
            axes[0, i].axis("off")
            axes[1, i].imshow(
                np.sqrt(np.mean((im - visual[0]) ** 2, axis=2)),
                vmin=0,
                vmax=0.08,
                cmap="magma",
                interpolation="nearest",
            )
            axes[1, i].axis("off")
        fig.suptitle(
            "Crossing scene, capture 48; lower row RGB RMS error, fixed 0–0.08 scale"
        )
        fig.savefig(ROOT / "results/figures/synthesis_crossing.png", dpi=160)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
