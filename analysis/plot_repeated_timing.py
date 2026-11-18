"""Standalone figure from process-level timing summaries; no pooled-frame error bars."""

import json
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results/processed/repeated_timing_summary.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
rows = [r for r in d["summary"] if r["mode"] != "none"]
for i, r in enumerate(rows):
    vals = r["gpu_run_median_ms"]["runs"]
    axes[0].scatter(
        np.full(5, i) + np.linspace(-0.12, 0.12, 5), vals, color="#2563eb", s=32
    )
    axes[0].hlines(np.median(vals), i - 0.25, i + 0.25, color="black", linewidth=2)
axes[0].set_xticks(range(4), [r["mode"] for r in rows])
axes[0].set_ylabel("Effect GPU interval (ms)")
axes[0].set_title("Five process-level medians per method")
axes[0].set_ylim(bottom=0)
axes[0].grid(axis="y", alpha=0.2)
for i, r in enumerate(d["overhead"]):
    vals = r["paired_mean_callback_interval_change_percent"]["runs"]
    axes[1].scatter(
        np.full(5, i) + np.linspace(-0.12, 0.12, 5), vals, color="#b45309", s=32
    )
    axes[1].hlines(np.median(vals), i - 0.25, i + 0.25, color="black", linewidth=2)
axes[1].axhline(0, color="gray", linewidth=1)
axes[1].set_xticks([0, 1], ["No effect", "M4"])
axes[1].set_ylabel("Query-on vs query-off interval change (%)")
axes[1].set_title("Five adjacent process pairs per control")
axes[1].grid(axis="y", alpha=0.2)
fig.suptitle(
    "720p scene C: exploratory timing, one GPU\nEach dot is a process summary or pair; black line is median",
    fontsize=11,
)
fig.savefig(ROOT / "results/figures/repeated_timing.png", dpi=160)
fig.savefig(ROOT / "results/figures/repeated_timing.pdf")
(ROOT / "results/processed/timing_plot_environment.json").write_text(
    json.dumps(
        {"matplotlib": matplotlib.__version__, "numpy": np.__version__}, indent=2
    )
)
