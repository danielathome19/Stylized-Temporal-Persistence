"""Run-level GPU timing report; verifies queries and immutable raw evidence."""

import sys
import json
import hashlib
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.timing_report import validate_samples


def main():
    path = Path(sys.argv[1])
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["outcome"] == "passed"
    for name, digest in json.loads((path / "sha256.json").read_text()).items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest, name
    groups = {}
    for mode in ["none", "dense", "innovation"]:
        runs = sorted(
            (r for r in meta["results"] if r["mode"] == mode), key=lambda r: r["block"]
        )
        assert len(runs) == 5
        intervals = []
        cpu = []
        for r in runs:
            folder = path / r["folder"]
            job = json.loads((folder / "job.json").read_text())
            data = json.loads((folder / "samples.json").read_text())
            rows = validate_samples(data, job["warm_up"])
            assert (
                len(rows) >= 300
                and rows[-1]["wall_us"] - rows[0]["wall_us"] >= 10000000
            )
            assert data["first_measured"] >= 120 and len(rows) % len(job["inputs"]) == 0
            assert all(r["gpu_ns"] is not None for r in rows)
            observed = np.percentile([r["gpu_ns"] / 1e6 for r in rows], [50, 95, 99])
            np.testing.assert_allclose(observed, r["gpu_ms_median_p95_p99"])
            intervals.append(
                np.percentile(
                    [r["callback_interval_us"] / 1000 for r in rows], [50, 95, 99]
                ).tolist()
            )
            cpu.append(r["cpu_submission_ms_median"])
        quant = [r["gpu_ms_median_p95_p99"] for r in runs]
        groups[mode] = {
            "run_gpu_quantiles_ms": quant,
            "median_run_gpu_quantiles_ms": np.median(quant, axis=0).tolist(),
            "run_cpu_submission_medians_ms": cpu,
            "run_delivery_quantiles_ms": intervals,
            "history_workspace_bytes": runs[0]["history_payload_bytes"],
            "common_output_bytes": 0 if mode == "none" else 230400,
        }
    ratios = (
        np.array(groups["innovation"]["run_gpu_quantiles_ms"])[:, 0]
        / np.array(groups["dense"]["run_gpu_quantiles_ms"])[:, 0]
    )
    result = {
        "source": path.name,
        "scene": meta["scene"],
        "scope": "isolated GPU storage work per scheduled capture,160x90,H10,RGBA32F,one RTX3060Ti; excludes scene/M5 reconstruction, common preloaded input cache and presentation; no-effect is query/submission control",
        "methods": groups,
        "paired_innovation_dense_median_ratios": ratios.tolist(),
        "median_paired_ratio": float(np.median(ratios)),
        "history_workspace_reduction_percent": 100
        * (
            1
            - groups["innovation"]["history_workspace_bytes"]
            / groups["dense"]["history_workspace_bytes"]
        ),
        "no_equivalent_fidelity_claim": "four of six earlier accuracy streams lose to float synthesis at this event budget; dense RGBA32F target is exact up to arithmetic",
    }
    (ROOT / "results/processed/innovation_timing.json").write_text(
        json.dumps(result, indent=2)
    )
    lines = [
        "# Sparse-event GPU timing",
        "",
        result["scope"],
        "",
        "Five independent processes per method. At least120captures/2seconds warmup and300captures/10seconds measurement; whole40sample cycles, reset each cycle. Asynchronous timestamp pairs validated, no image readback in timing. Three-method order rotates across five blocks; positions are not perfectly balanced.",
        "",
        "Median across five process quantiles:",
        "",
        "|Method|Median GPU ms|p95 GPU ms|p99 GPU ms|History/workspace bytes|",
        "|---|---:|---:|---:|---:|",
    ]
    for m, g in groups.items():
        lines.append(
            "|"
            + m
            + "|"
            + "|".join(f"{v:.6f}" for v in g["median_run_gpu_quantiles_ms"])
            + f"|{g['history_workspace_bytes']}|"
        )
    lines += [
        "",
        f"Median paired innovation/dense GPU-duration ratio: {result['median_paired_ratio']:.2f}x. History/workspace payload reduction: {result['history_workspace_reduction_percent']:.2f}%. Common output230400bytes per active method is additional; each run preloads9216000input-cachebytes outside the timed scope.",
        "",
        "The current sparse implementation trades memory for much greater GPU work and reduced fidelity under pressure. It does not match dense-history speed. No-effect duration is not subtracted: tiny timestamp scopes and query overhead limit precision. No claim of overall scene/render FPS, high-resolution performance or speedup over published methods. Dense and sparse have different accuracy; this is a cost comparison, not matched-fidelity evidence.",
        "",
        f"Raw source: {path.name}",
        "",
        "Reproduce: python analysis/innovation_timing_report.py <raw-timing-run>",
    ]
    (ROOT / "results/processed/innovation_timing.md").write_text(
        "\n".join(lines) + "\n"
    )
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for i, m in enumerate(groups):
        values = [q[0] for q in groups[m]["run_gpu_quantiles_ms"]]
        if all(v == 0 for v in values):
            ax.annotate(
                "below timestamp resolution",
                (i, 0.008),
                ha="center",
                rotation=90,
                fontsize=8,
            )
        else:
            ax.scatter([i] * len(values), values, label=m)
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=list(groups),
        xlim=(-0.45, 2.45),
        yscale="log",
        ylabel="Per-capture GPU ms, process median",
        title="Five processes per method;160x90 frozen camera inputs",
    )
    fig.savefig(ROOT / "results/figures/innovation_timing.png", dpi=160)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
