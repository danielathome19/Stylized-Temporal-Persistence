"""M5 visibility-policy ablations with persistent GPU recursive history."""

import datetime
import hashlib
import json
import math
import platform
import subprocess
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.reconstruction_reference import reconstruct_visibility
from analysis.temporal_reference import recursive_oracle
from experiments.run.reconstruction import metrics


def main():
    reference = ROOT / "results/raw/20261115T045147Z-reconstruction-d13b3141"
    for rel, h in json.loads((reference / "sha256.json").read_text()).items():
        assert hashlib.sha256((reference / rel).read_bytes()).hexdigest() == h
    source = json.loads((reference / "job.json").read_text())["cases"]
    groups = {"A": [c for c in source if c["name"].startswith("A_")]}
    groups.update({c["name"]: [c] for c in source if not c["name"].startswith("A_")})
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-m5-history-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    policies = {"strict": 0, "one_sided": 1, "gated": 2, "hold": 3, "crossfade": 4}
    sequences = []
    for name, frames in groups.items():
        for label, policy in policies.items():
            folder = out / (name + "_" + label)
            folder.mkdir()
            sequence = {
                "name": name,
                "label": label,
                "policy": policy,
                "width": frames[0]["width"],
                "height": frames[0]["height"],
                "alpha": 2 ** (-frames[0]["params"][0] / 0.25),
                "frames": [],
            }
            for frame in frames:
                sequence["frames"].append(
                    {**frame, "output": str(folder / frame["name"])}
                )
            sequences.append(sequence)
    path = out / "job.json"
    path.write_text(json.dumps({"sequences": sequences}, indent=2))
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
    meta = {
        "run_id": out.name,
        "stage": "M5_visibility_ablation_GPU_history",
        "git_commit": None,
        "engine": "4.8.beta3.official.c51ff06e8",
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "os": platform.platform(),
        "backend": "vulkan",
        "format": "RGBA32F",
        "resolution": "per sequence in job",
        "seed": 0,
        "scene": list(groups),
        "method": policies,
        "warm_up": 0,
        "history_schedule": "midpoint and right endpoint, explicit times in jobs",
        "target_frame_schedule": "paired source timestamps in job",
        "actual_frame_schedule": "synchronized accuracy replay",
        "latency_policy": "all policies wait for right endpoint, same delivery budget",
        "kernel_parameters": {"half_life_s": 0.25, "seed": "first source"},
        "timing_availability": "not_measured",
        "source_reference": str(reference),
        "source_manifest_sha256": hashlib.sha256(
            (reference / "sha256.json").read_bytes()
        ).hexdigest(),
        "configuration_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "outcome": "started",
        "tolerance_max_abs": 2e-6,
        "candidate_scope": "uniform horizontal motion and source-derived auxiliary data; uncertainty gate is a heuristic, not calibrated probability",
    }
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "analysis/reconstruction_reference.py",
            ROOT / "analysis/temporal_reference.py",
            ROOT / "godot/scripts/m5_history_replay.gd",
            ROOT / "godot/shaders/reconstruct_visibility.glsl",
            ROOT / "godot/scripts/temporal_baselines.gd",
            ROOT / "godot/shaders/feedback.glsl",
        ]
    }
    try:
        meta["gpu_driver_report"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        proc = subprocess.run(
            [
                str(engine),
                "--path",
                str(ROOT / "godot"),
                "--rendering-method",
                "forward_plus",
                "--rendering-driver",
                "vulkan",
                "--audio-driver",
                "Dummy",
                "--log-file",
                str(out / "engine.log"),
                "--script",
                "res://scripts/m5_history_replay.gd",
                "--",
                str(path),
            ],
            capture_output=True,
            timeout=120,
        )
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        if proc.returncode or "M5_HISTORY_COMPLETE" not in log or "ERROR:" in log:
            raise RuntimeError("GPU replay failed")
        rows = []
        for sequence in sequences:
            oracle_history = []
            exact_history = []
            maxerr = 0
            sample_rows = []
            for frame in sequence["frames"]:
                shape = (sequence["height"], sequence["width"], 4)
                c0, c1, a0, a1 = [
                    np.fromfile(frame[k], dtype="<f4").reshape(shape)
                    for k in ["color0", "color1", "aux0", "aux1"]
                ]
                exact = np.fromfile(
                    reference / frame["name"] / "target.bin", dtype="<f4"
                ).reshape(shape)
                expected, provenance = reconstruct_visibility(
                    c0, c1, a0, a1, *frame["params"], sequence["policy"]
                )
                gpu = np.fromfile(frame["output"] + "_sample.bin", dtype="<f4").reshape(
                    shape
                )
                codes = np.fromfile(
                    frame["output"] + "_provenance.bin", dtype="<f4"
                ).reshape(shape)[..., 0]
                history = np.fromfile(
                    frame["output"] + "_history.bin", dtype="<f4"
                ).reshape(shape)
                if not oracle_history:
                    oracle_history = [c0]
                    exact_history = [c0]
                oracle_history.extend([expected, c1])
                exact_history.extend([exact, c1])
                oracle = recursive_oracle(
                    oracle_history, [sequence["alpha"]] * (len(oracle_history) - 1)
                )
                err = max(
                    float(np.abs(gpu - expected).max()),
                    float(np.abs(history - oracle).max()),
                )
                maxerr = max(maxerr, err)
                if (
                    not np.isfinite(history).all()
                    or not np.isfinite(gpu).all()
                    or err > 2e-6
                    or not np.array_equal(codes, provenance)
                ):
                    raise RuntimeError(
                        "Oracle mismatch " + sequence["name"] + " " + sequence["label"]
                    )
                mask = np.any(
                    (c0[..., :3] != 0) | (c1[..., :3] != 0) | (exact[..., :3] != 0),
                    axis=2,
                )
                sample_rows.append(
                    {
                        "frame": frame["name"],
                        "latency_s": frame["latency_s"],
                        **metrics(gpu, exact, mask),
                        "provenance_fraction": {
                            str(k): float(np.mean(codes == k)) for k in range(4)
                        },
                    }
                )
            exact_composite = recursive_oracle(
                exact_history, [sequence["alpha"]] * (len(exact_history) - 1)
            )
            (
                np.save(out / (sequence["name"] + "_exact.npy"), exact_composite)
                if sequence["label"] == "strict"
                else None
            )
            rows.append(
                {
                    "scene": sequence["name"],
                    "policy": sequence["label"],
                    "oracle_max_abs": maxerr,
                    "samples": sample_rows,
                    "final_history": metrics(
                        history,
                        exact_composite,
                        np.any(exact_composite[..., :3] != 0, axis=2),
                    ),
                }
            )
            print(sequence["name"], sequence["label"], "passed", flush=True)
        meta.update(
            outcome="passed",
            results=rows,
            sequences=len(sequences),
            reconstructed_samples=sum(len(s["frames"]) for s in sequences),
        )
    except Exception as e:
        meta.update(outcome="failed", error=str(e))
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
    print(
        json.dumps(
            {"run": str(out), "outcome": meta["outcome"], "error": meta.get("error")}
        )
    )
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
