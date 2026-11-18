"""Focused candidate implementation check, not a benchmark campaign."""

import datetime
import hashlib
import json
import subprocess
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.horizon_merge import step, scores


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-horizon-merge-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    cases = []
    inputs = []
    horizon = 6
    k = 2
    alpha = 2 ** (-0.05 / 0.25)
    for name in ["moving_silhouette", "returning_silhouette"]:
        folder = out / name
        folder.mkdir()
        frames = []
        case = {
            "name": name,
            "width": 8,
            "height": 8,
            "tile": 8,
            "versions": k,
            "history": horizon,
            "alpha": alpha,
            "epsilon": 0,
            "pressure_policy": "horizon_merge",
            "metadata": str(folder / "stats.bin"),
            "frames": [],
        }
        for i in range(12):
            im = np.zeros((8, 8, 4), np.float32)
            im[..., 3] = 1
            x = (
                i % 6
                if name == "moving_silhouette"
                else [0, 0, 2, 4, 4, 0, 0, 3, 5, 5, 1, 0][i]
            )
            im[2:6, x : x + 2, :3] = [1, 0.6, 0.2]
            path = folder / (str(i) + "_input.bin")
            im.tofile(path)
            frames.append(im.astype(float))
            case["frames"].append(
                {
                    "input": str(path),
                    "output": str(folder / (str(i) + "_output.bin")),
                    "diagnostics": str(folder / (str(i) + "_decision.bin")),
                }
            )
        cases.append(case)
        inputs.append(frames)
    job = out / "job.json"
    job.write_text(json.dumps({"cases": cases}, indent=2))
    cmd = [
        str(ROOT / "Godot_v4.8-beta3_win64.exe"),
        "--path",
        str(ROOT / "godot"),
        "--script",
        "res://scripts/m6_replay.gd",
        "--rendering-method",
        "forward_plus",
        "--rendering-driver",
        "vulkan",
        "--audio-driver",
        "Dummy",
        "--log-file",
        str(out / "engine.log"),
        "--",
        str(job),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=90)
    (out / "stdout.log").write_bytes(proc.stdout)
    (out / "stderr.log").write_bytes(proc.stderr)
    meta = {
        "run_id": out.name,
        "stage": "candidate_rule_correctness",
        "method": "finite-lifetime contribution-weighted merge",
        "outcome": "started",
        "timing_available": False,
        "configuration_hash": hashlib.sha256(job.read_bytes()).hexdigest(),
        "results": [],
    }
    try:
        log = (out / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0 and "M6_REPLAY_COMPLETE" in log and "ERROR:" not in log
        )
        for case, frames in zip(cases, inputs):
            pool = [None] * k
            queue = []
            max_error = 0
            decisions = []
            mse = []
            norm_bound = 0.0
            for i, (frame, im) in enumerate(zip(case["frames"], frames)):
                diag = np.fromfile(frame["diagnostics"], "<f4")[k:]
                decision = (int(diag[0]), int(diag[1])) if diag[0] >= 0 else None
                expected, candidates = step(pool, queue, im, horizon, alpha, decision)
                if candidates:
                    best = min(c[0] for c in candidates)
                    chosen = next(c[0] for c in candidates if c[1:] == decision)
                    assert chosen <= best + 2e-6, (decision, chosen, best)
                    assert abs(float(diag[2]) - chosen) < 2e-6
                    norm_bound += np.sqrt(chosen)
                    assert abs(float(diag[3]) - norm_bound) < 2e-6
                    decisions.append(
                        {
                            "frame": i,
                            "remove": decision[0],
                            "keep": decision[1],
                            "predicted_cost": chosen,
                        }
                    )
                actual = np.fromfile(frame["output"], "<f4").reshape(8, 8, 4)
                max_error = max(max_error, float(abs(actual - expected).max()))
                originals = frames[max(0, i - horizon + 1) : i + 1]
                w = alpha ** np.arange(len(originals) - 1, -1, -1)
                w /= w.sum()
                truth = sum(weight * image for weight, image in zip(w, originals))
                mse.append(float(np.mean((actual[..., :3] - truth[..., :3]) ** 2)))
            assert max_error < 2e-6, max_error
            assert sum(mse) <= norm_bound**2 + 2e-6
            meta["results"].append(
                {
                    "case": case["name"],
                    "arithmetic_max_abs": max_error,
                    "decisions": decisions,
                    "dense_target_mse_per_capture": mse,
                    "compression_energy_bound": norm_bound**2,
                    "observed_compression_energy": sum(mse),
                }
            )
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "analysis/horizon_merge.py",
            ROOT / "godot/shaders/sparse_budget_store.glsl",
            ROOT / "godot/scripts/sparse_history.gd",
        ]
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "sha256.json").write_text(
        json.dumps(
            {
                p.relative_to(out)
                .as_posix(): hashlib.sha256(p.read_bytes())
                .hexdigest()
                for p in out.rglob("*")
                if p.is_file()
            },
            indent=2,
        )
    )
    print(out)
    print(json.dumps(meta, indent=2))
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
