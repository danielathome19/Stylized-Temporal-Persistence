"""Run-level paired serial/parallel GPU summary, with immutable evidence checks."""

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
    for mode in ["serial", "parallel"]:
        runs = sorted(
            [r for r in meta["results"] if r["mode"] == mode], key=lambda r: r["block"]
        )
        assert len(runs) == 5
        for r in runs:
            data = json.loads((path / r["folder"] / "samples.json").read_text())
            rows = validate_samples(data, 120)
            assert (
                len(rows) >= 300
                and rows[-1]["wall_us"] - rows[0]["wall_us"] >= 10000000
            )
        med = [r["gpu_ms_median_p95_p99"][0] for r in runs]
        groups[mode] = {
            "run_medians_ms": med,
            "median_ms": float(np.median(med)),
            "range_ms": [min(med), max(med)],
            "median_p95_ms": float(
                np.median([r["gpu_ms_median_p95_p99"][1] for r in runs])
            ),
        }
    ratios = (
        np.array(groups["serial"]["run_medians_ms"])
        / groups["parallel"]["run_medians_ms"]
    )
    result = {
        "source": path.name,
        "methods": groups,
        "paired_speedup_ratios": ratios.tolist(),
        "median_paired_speedup": float(np.median(ratios)),
        "median_paired_gpu_reduction_percent": float(np.median(100 * (1 - 1 / ratios))),
        "scope": "per-capture storage only; deterministic160x90 camera sequence; singleGPU; no quality objective change",
    }
    (ROOT / "results/processed/prepare_timing.json").write_text(
        json.dumps(result, indent=2)
    )
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for i in range(5):
        ax.plot(
            [0, 1],
            [groups[m]["run_medians_ms"][i] for m in ["serial", "parallel"]],
            "o-",
            alpha=0.65,
            label=f"block{i+1}",
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Serial pairs", "Parallel pairs"],
        ylabel="Per-capture GPU ms (process median)",
        title="Same lifetime merge rule and storage budget",
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.savefig(ROOT / "results/figures/prepare_timing.png", dpi=160)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
