"""Verify immutable timestamp evidence and summarize at run/configuration level."""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path


def validate_samples(data, warm_up):
    rows = [r for r in data["records"] if r["measured"]]
    ids = [r["index"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(
        range(data.get("first_measured", warm_up), data["submitted"])
    ):
        raise ValueError("Missing or duplicated measured frame")
    if any(r["measured"] for r in data["unresolved"]):
        raise ValueError("Unresolved measured query")
    if any(
        (r["gpu_ns"] is not None and r["gpu_ns"] < 0) or r["cpu_submission_us"] < 0
        for r in rows
    ):
        raise ValueError("Negative duration")
    return rows


def main():
    summary = []
    for name in sys.argv[1:]:
        p = Path(name)
        for rel, h in json.loads((p / "sha256.json").read_text()).items():
            if hashlib.sha256((p / rel).read_bytes()).hexdigest() != h:
                raise ValueError("Hash mismatch " + rel)
        meta = json.loads((p / "metadata.json").read_text())
        for result in meta["results"]:
            folder = p / result.get("folder", result["mode"])
            job = json.loads((folder / "job.json").read_text())
            rows = validate_samples(
                json.loads((folder / "samples.json").read_text()), job["warm_up"]
            )
            elapsed = (rows[-1]["wall_us"] - rows[0]["wall_us"]) / 1e6
            if elapsed < job["duration_us"] / 1e6:
                raise ValueError("Short measurement")
            summary.append(
                {
                    "run": p.name,
                    "stage": meta["stage"],
                    **result,
                    "measured_seconds": elapsed,
                }
            )
    dest = Path(__file__).resolve().parents[1] / "results/processed/timing_summary.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
