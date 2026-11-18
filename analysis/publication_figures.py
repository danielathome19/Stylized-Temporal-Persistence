"""Render publication figures from recorded measurements and image outputs."""

import sys
import json
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def main():
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / "results/processed/paper_storage.json").read_text())
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), layout="constrained")
    for ax, w in zip(axes, [1280, 1920]):
        for i, fmt in enumerate(["rgba32f", "rgba16f", "rgba8"]):
            row = next(
                r for r in data["timing"] if r["width"] == w and r["mode"] == fmt
            )
            v = np.array(row["run_quantiles_ms"])[:, 0]
            ax.scatter(i + np.linspace(-0.05, 0.05, 5), v, s=18)
        ax.set(
            xticks=range(3),
            xticklabels=["32-bit float", "16-bit float", "8-bit UNORM"],
            ylabel="Storage GPU time (ms)",
            title=f"{w} x {720 if w==1280 else 1080}",
            ylim=(0, None),
        )
        ax.tick_params(axis="x", labelsize=7)
    fig.savefig(out / "native_timing.pdf")
    plt.close(fig)
    precision = root / "results/raw" / data["precision_source"]
    job = json.loads((precision / "job.json").read_text())

    def path(value):
        text = str(value).replace("\\", "/")
        return root / "results/raw" / text.split("/results/raw/", 1)[1]

    cases = [
        next(c for c in job["cases"] if c["name"] == "camera_1280_" + fmt)
        for fmt in ["rgba16f", "rgba8"]
    ]
    weights = cases[0]["alpha"] ** np.arange(9, -1, -1)
    weights /= weights.sum()
    truth = sum(
        a
        * np.fromfile(path(f["input"]), "<f4").reshape(720, 1280, 4).astype(np.float64)
        for a, f in zip(weights, cases[0]["frames"][6:16])
    )
    images = [truth] + [
        np.fromfile(path(c["frames"][15]["output"]), "<f4").reshape(720, 1280, 4)
        for c in cases
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 1.45), layout="constrained")
    for i, (im, title) in enumerate(
        zip(images, ["Reference composite", "Dense RGBA16F", "Dense RGBA8"])
    ):
        axes[i].imshow(srgb(im[..., :3]))
        axes[i].set_title(title, fontsize=8)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    half_error = images[1][..., :3] - truth[..., :3]
    half_rms = float(np.sqrt(np.mean(half_error**2)))
    axes[1].set_xlabel(f"RGB RMS error: {half_rms:.2e}", fontsize=7)
    plot = axes[3].imshow(
        np.sqrt(np.mean((images[2][..., :3] - truth[..., :3]) ** 2, axis=2)),
        vmin=0,
        vmax=0.004,
        cmap="magma",
    )
    axes[3].set_title("RGBA8 error", fontsize=8)
    axes[3].axis("off")
    fig.colorbar(plot, ax=axes[3], fraction=0.035, pad=0.025, ticks=[0, 0.002, 0.004])
    axes[3].text(
        0.5,
        -0.17,
        "Linear RGB RMS error",
        transform=axes[3].transAxes,
        ha="center",
        fontsize=7,
    )
    fig.savefig(out / "precision_teaser.png", dpi=300)
    plt.close(fig)
    raw = root / "results/raw/20261117T071338Z-innovation-gpu-295375be"
    job = json.loads((raw / "job.json").read_text())
    case = next(c for c in job["cases"] if c["name"] == "crossing")
    i = 48
    weights = case["alpha"] ** np.arange(9, -1, -1)
    weights /= weights.sum()
    truth = sum(
        a * np.fromfile(path(f["input"]), "<f4").reshape(90, 160, 4).astype(np.float64)
        for a, f in zip(weights, case["frames"][i - 9 : i + 1])
    )
    actual = np.fromfile(path(case["frames"][i]["output"]), "<f4").reshape(90, 160, 4)
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.7), layout="constrained")
    axes[0].imshow(srgb(truth[..., :3]), interpolation="nearest")
    axes[0].set_title("Reference")
    axes[1].imshow(srgb(actual[..., :3]), interpolation="nearest")
    axes[1].set_title("Sparse, 763 events")
    err = axes[2].imshow(
        np.sqrt(np.mean((actual[..., :3] - truth[..., :3]) ** 2, axis=2)),
        vmin=0,
        vmax=0.15,
        cmap="magma",
        interpolation="nearest",
    )
    axes[2].set_title("Linear RGB RMS error")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(err, ax=axes[2], fraction=0.035, pad=0.02)
    fig.savefig(out / "sparse_failure.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
