"""Audit repeated timing and summarize independent process runs, never pooled frames."""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.timing_report import validate_samples


def spread(values):
    a = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "runs": a.tolist(),
    }


def main():
    path = Path(sys.argv[1])
    meta = json.loads((path / "metadata.json").read_text())
    if meta["outcome"] != "passed":
        raise ValueError("Incomplete campaign")
    for rel, h in json.loads((path / "sha256.json").read_text()).items():
        if hashlib.sha256((path / rel).read_bytes()).hexdigest() != h:
            raise ValueError(rel)
    results = meta["results"]
    details = []
    for result in results:
        folder = path / result["folder"]
        job = json.loads((folder / "job.json").read_text())
        data = json.loads((folder / "samples.json").read_text())
        rows = validate_samples(data, job["warm_up"])
        if len(rows) < 300 or rows[-1]["wall_us"] - rows[0]["wall_us"] < 10000000:
            raise ValueError("Short window")
        if data["first_measured"] < 120:
            raise ValueError("Short warmup")
        # First resolved frame may be index 1; permit its interval when checking elapsed warmup.
        first = min(data["records"], key=lambda r: r["index"])
        origin = (
            first["wall_us"] - first["callback_interval_us"]
            if first["index"] == 1
            else first["wall_us"]
        )
        if rows[0]["wall_us"] - origin < 2000000:
            raise ValueError("Short time warmup")
        if not result["instrumented"] and any(r["gpu_ns"] is not None for r in rows):
            raise ValueError("Invented disabled GPU timing")
        if result["instrumented"] and any(r["gpu_ns"] is None for r in rows):
            raise ValueError("Missing GPU query")
        tag = {}
        if result["instrumented"] and result["mode"] in ["M3", "M4"]:
            for name, predicate in [
                ("update", lambda r: r["scheduled_updates"] > 0),
                ("hold", lambda r: r["scheduled_updates"] == 0),
            ]:
                selected = [r for r in rows if predicate(r)]
                tag[name] = {
                    "frames": len(selected),
                    "gpu_median_ms": float(
                        np.median([r["gpu_ns"] / 1e6 for r in selected])
                    ),
                }
        details.append(
            {
                **result,
                "tagged": tag,
                "measured_seconds": (rows[-1]["wall_us"] - rows[0]["wall_us"]) / 1e6,
            }
        )
    summary = []
    for mode in ["none", "M1", "M2", "M3", "M4"]:
        group = [r for r in details if r["mode"] == mode and r["instrumented"]]
        if len(group) != 5:
            raise ValueError("Need five process runs")
        row = {
            "mode": mode,
            "gpu_run_median_ms": spread([r["gpu_ms_median_p95_p99"][0] for r in group]),
            "gpu_run_p95_ms": spread([r["gpu_ms_median_p95_p99"][1] for r in group]),
            "cpu_run_median_ms": spread(
                [r["cpu_submission_ms_median_p95_p99"][0] for r in group]
            ),
        }
        if mode in ["M3", "M4"]:
            row["update_gpu_run_median_ms"] = spread(
                [r["tagged"]["update"]["gpu_median_ms"] for r in group]
            )
            row["hold_gpu_run_median_ms"] = spread(
                [r["tagged"]["hold"]["gpu_median_ms"] for r in group]
            )
        summary.append(row)
    overhead = []
    for mode in ["none", "M4"]:
        pairs = []
        for block in range(5):
            on = next(
                r
                for r in details
                if r["mode"] == mode and r["block"] == block and r["instrumented"]
            )
            off = next(
                r
                for r in details
                if r["mode"] == mode and r["block"] == block and not r["instrumented"]
            )
            pairs.append(
                100
                * (1 / on["callback_fps"] - 1 / off["callback_fps"])
                / (1 / off["callback_fps"])
            )
        overhead.append(
            {
                "mode": mode,
                "paired_mean_callback_interval_change_percent": spread(pairs),
            }
        )
    output = {
        "run": path.name,
        "summary": summary,
        "overhead": overhead,
        "details": details,
        "inference": "Descriptive median and full range of five independent process summaries; no frame-level inference or equivalence claim.",
    }
    (ROOT / "results/processed/repeated_timing_summary.json").write_text(
        json.dumps(output, indent=2)
    )
    print(
        json.dumps(
            {"run": path.name, "summary": summary, "overhead": overhead}, indent=2
        )
    )


if __name__ == "__main__":
    main()
