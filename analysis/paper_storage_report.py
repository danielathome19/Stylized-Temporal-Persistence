"""Audited native-resolution tables, run-level uncertainty and visual evidence."""

import sys
import json
import hashlib
import matplotlib
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.artifact_paths import recorded_path
from analysis.timing_report import validate_samples


def checked(path):
    path = Path(path)
    for f, h in json.loads((path / "sha256.json").read_text()).items():
        assert hashlib.sha256((path / f).read_bytes()).hexdigest() == h, f
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["outcome"] == "passed"
    return meta


def luminance_ssim(a, b):
    a = np.asarray(a[..., :3], float) @ np.array([0.2126, 0.7152, 0.0722])
    b = np.asarray(b[..., :3], float) @ np.array([0.2126, 0.7152, 0.0722])
    filt = lambda x: gaussian_filter(x, 1.5, truncate=3.5, mode="reflect")
    ma, mb = filt(a), filt(b)
    va = np.maximum(0, filt(a * a) - ma * ma)
    vb = np.maximum(0, filt(b * b) - mb * mb)
    cov = filt(a * b) - ma * mb
    value = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return float(value[5:-5, 5:-5].mean())


def main():
    precision_path = Path(sys.argv[1])
    timing_path = Path(sys.argv[2])
    precision = checked(precision_path)
    timing = checked(timing_path)
    groups = []
    rng = np.random.default_rng(915)
    if timing.get("parent_run"):
        parent = recorded_path(timing["parent_run"])
        for f, h in json.loads((parent / "sha256.json").read_text()).items():
            assert hashlib.sha256((parent / f).read_bytes()).hexdigest() == h, f
    for width in [1280, 1920]:
        for mode in ["none", "rgba32f", "rgba16f", "rgba8"]:
            rows = sorted(
                (
                    r
                    for r in timing["results"]
                    if r["width"] == width and r["mode"] == mode and not r["pilot"]
                ),
                key=lambda r: r["block"],
            )
            assert len(rows) == 5
            for r in rows:
                folder = timing_path / r["folder"]
                job = json.loads((folder / "job.json").read_text())
                data = json.loads((folder / "samples.json").read_text())
                samples = validate_samples(data, 120)
                assert (
                    len(samples) >= 300
                    and data["first_measured"] >= 120
                    and samples[-1]["wall_us"] - samples[0]["wall_us"] >= 10000000
                )
                assert len(samples) % len(job["inputs"]) == 0 and all(
                    s["gpu_ns"] is not None for s in samples
                )
                np.testing.assert_allclose(
                    np.percentile([s["gpu_ns"] / 1e6 for s in samples], [50, 95, 99]),
                    r["gpu_ms_quantiles"],
                )
            values = np.array([r["gpu_ms_quantiles"] for r in rows])
            groups.append(
                {
                    "width": width,
                    "mode": mode,
                    "run_quantiles_ms": values.tolist(),
                    "median_quantiles_ms": np.median(values, axis=0).tolist(),
                    "median_range_ms": [
                        float(values[:, 0].min()),
                        float(values[:, 0].max()),
                    ],
                    "history_bytes": rows[0]["history_workspace_bytes"],
                    "cpu_submission_medians_ms": [
                        r["cpu_submission_ms_median"] for r in rows
                    ],
                    "delivery_quantiles_ms": [r["delivery_ms_quantiles"] for r in rows],
                }
            )
    ratios = []
    for width in [1280, 1920]:
        base = np.array(
            next(g for g in groups if g["width"] == width and g["mode"] == "rgba32f")[
                "run_quantiles_ms"
            ]
        )[:, 0]
        for mode in ["rgba16f", "rgba8"]:
            vals = np.array(
                next(g for g in groups if g["width"] == width and g["mode"] == mode)[
                    "run_quantiles_ms"
                ]
            )[:, 0]
            paired = base / vals
            bootstrap = np.median(paired[rng.integers(0, 5, (10000, 5))], axis=1)
            ratios.append(
                {
                    "width": width,
                    "mode": mode,
                    "paired_speedup": paired.tolist(),
                    "median": float(np.median(paired)),
                    "descriptive_bootstrap95": np.percentile(
                        bootstrap, [2.5, 97.5]
                    ).tolist(),
                }
            )
    pilots = [r for r in timing["results"] if r["pilot"]]
    for r in pilots:
        data = json.loads((timing_path / r["folder"] / "samples.json").read_text())
        samples = validate_samples(data, 0)
        assert len(samples) == 16
        assert not r["gate_passed"] and r["gpu_ms_quantiles"][0] > 50
    ssim = []
    job = json.loads((precision_path / "job.json").read_text())
    visual = []
    for case in job["cases"]:
        w, h = case["width"], case["height"]
        i = 15
        weights = case["alpha"] ** np.arange(9, -1, -1)
        weights /= weights.sum()
        truth = sum(
            a
            * np.fromfile(recorded_path(f["input"]), "<f4")
            .reshape(h, w, 4)
            .astype(float)
            for a, f in zip(weights, case["frames"][6:16])
        )
        actual = np.fromfile(recorded_path(case["frames"][i]["output"]), "<f4").reshape(
            h, w, 4
        )
        ssim.append(
            {
                "case": case["name"],
                "capture": 15,
                "linear_luminance_ssim": luminance_ssim(truth, actual),
            }
        )
        if case["name"].startswith("camera_1280"):
            visual.append((case["storage_format"], truth, actual))
    result = {
        "precision_source": precision_path.name,
        "timing_source": timing_path.name,
        "timing_parent": timing.get("parent_run"),
        "precision": precision["results"],
        "timing": groups,
        "paired_ratios": ratios,
        "sparse_feasibility_pilots": pilots,
        "ssim": ssim,
        "ssim_definition": "selected final capture only; linearRGB Rec709 luminance; Gaussian sigma1.5 radius5; population covariance; data_range1; K1.01,K2.03; valid5pixel border excluded; not perceptual validation",
        "uncertainty": "median/range over5processes; paired block bootstrap10000seed915 descriptive only, oneGPU, no equivalence hypothesis or significance test",
    }
    dest = ROOT / "results/processed"
    (dest / "paper_storage.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Native-resolution storage evaluation",
        "",
        f"Precision: {precision_path.name}. Timing: {timing_path.name}. Godot4.7.2stable, RTX3060Ti, Vulkan. Input renderer4.8beta3; all storage methods receive identical exact-time frozen procedural inputs.",
        "",
        "## GPU cost and explicit history bytes",
        "",
        "Median of five process medians; bracket is observed range. Tails are medians of process quantiles. Common output16Pbytes and input cache are additional.",
        "",
        "|Resolution|Format|GPU median ms [range]|p95 ms|p99 ms|History MiB|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for g in groups:
        q = g["median_quantiles_ms"]
        lo, hi = g["median_range_ms"]
        lines.append(
            f"|{g['width']}|{g['mode']}|{q[0]:.6f} [{lo:.6f},{hi:.6f}]|{q[1]:.6f}|{q[2]:.6f}|{g['history_bytes']/1048576:.3f}|"
        )
    lines += [
        "",
        "No-effect zero ticks are below query resolution, not literal zero cost. Measurement duration>=10s and>=300captures after>=120captures/2swarmup; no timed image readback. Four-method rotation overfiveblocks is not perfectly position-balanced. Initial720pblocks overlapped CPU-only precision analysis; no concurrent GPU experiment. CPU submission and delivered callback distributions remain in JSON; these are not whole-effect FPS.",
        "",
        "## Paired speed ratios vsRGBA32F",
        "",
        "|Resolution|Format|Median ratio|Descriptive bootstrap95 interval|",
        "|---|---|---:|---|",
    ]
    for r in ratios:
        lines.append(
            f"|{r['width']}|{r['mode']}|{r['median']:.3f}|{r['descriptive_bootstrap95']}|"
        )
    lines += [
        "",
        "Five-run bootstrap intervals describe this machine and protocol; no population-level or perceptual equivalence inference.",
    ]
    if timing.get("parent_run"):
        lines += [
            "",
            f"Interrupted campaign preserved: {Path(timing['parent_run']).name}. One1080pRGBA8process exited before its completion marker; remaining repetitions resumed in a later session. Valid completed runs are retained, the incomplete process is excluded as failed, and all referenced parent hashes are verified. Logical block pairing is not uninterrupted temporal adjacency. The cause of exit remains unknown.",
        ]
    lines += [
        "",
        "## Native storage accuracy",
        "",
        "Float64oracle independently composites exact scheduled scene inputs. Scene sources areRGBA16F;RGBA16Fhistory therefore loses no additional source precision. Alphaexcluded from image metrics; PSNRdata_range1.",
        "",
        "|Scene|Width|Format|Full MSE|Motion-region MSE|Full PSNR dB|",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in precision["results"]:
        lines.append(
            f"|{r['scene']}|{r['width']}|{r['format']}|{r['full_mse']:.7g}|{r['motion_mse']:.7g}|{r['psnr_full_range1']:.4g}|"
        )
    lines += [
        "",
        "Selected-final-capture linear-luminance SSIM values and definition are in JSON. No human-perception claim.",
        "",
        "## Sparse feasibility pilots",
        "",
        "One16capture sequence per resolution; no warmup or independent repetitions. Declared50ms gate exceeded, so long repeated-throughput runs skipped.",
        "",
        "|Resolution|Capacity|History/workspace MiB|Pilot median GPU ms|",
        "|---|---:|---:|---:|",
    ]
    for r in pilots:
        lines.append(
            f"|{r['width']}|{r['capacity']}|{r['history_workspace_bytes']/1048576:.3f}|{r['gpu_ms_quantiles'][0]:.3f}|"
        )
    lines += [
        "",
        "These pilot values are feasibility observations, not five-run comparable throughput estimates. They support failure of this implementation to meet its20Hzcapture budget, not impossibility of sparse approaches.",
    ]
    (dest / "paper_storage.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, width in zip(axes, [1280, 1920]):
        for i, mode in enumerate(["rgba32f", "rgba16f", "rgba8"]):
            vals = np.array(
                next(g for g in groups if g["width"] == width and g["mode"] == mode)[
                    "run_quantiles_ms"
                ]
            )[:, 0]
            ax.scatter([i] * 5, vals)
        ax.set(
            xticks=[0, 1, 2],
            xticklabels=["RGBA32F", "RGBA16F", "RGBA8"],
            ylabel="Per-capture GPU ms, process median",
            title=f"{width}x{720 if width==1280 else 1080}",
            ylim=(0, None),
        )
    fig.savefig(ROOT / "results/figures/paper_native_timing.png", dpi=180)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), layout="constrained")
    truth = visual[0][1]
    images = [truth] + [v[2] for v in visual]
    for i, (im, name) in enumerate(
        zip(images, ["Oracle", "RGBA32F", "RGBA16F", "RGBA8"])
    ):
        axes[0, i].imshow(np.clip(im[..., :3], 0, 1))
        axes[0, i].set_title(name)
        axes[1, i].imshow(
            np.sqrt(np.mean((im[..., :3] - truth[..., :3]) ** 2, axis=2)),
            cmap="magma",
            vmin=0,
            vmax=0.004,
        )
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle(
        "720p moving-camera final composite; lower row: RGB RMS error, fixed 0–0.004 scale",
        fontsize=12,
    )
    fig.colorbar(
        axes[1, -1].images[0],
        ax=axes[1, :].tolist(),
        label="Linear RGB RMS error",
        fraction=0.025,
        pad=0.015,
    )
    fig.savefig(ROOT / "results/figures/paper_native_precision.png", dpi=180)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
