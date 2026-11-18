"""Frozen-policy new-case validation of 2D M5 reconstruction plus GPU history."""

import datetime
import hashlib
import json
import platform
import subprocess
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.reconstruction_2d import render, reconstruct, specifications
from analysis.temporal_reference import recursive_oracle
from experiments.run.reconstruction import metrics


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-m5-2d-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    specs = specifications()
    (out / "specifications.json").write_text(json.dumps(specs, indent=2))
    sequences = []
    inputs = {}
    alpha = 2 ** (-0.05 / 0.25)
    for spec in specs:
        folder = out / spec["name"]
        folder.mkdir()
        c0, a0, v0 = render(spec, 0)
        c1, a1, v1 = render(spec, 0.1)
        target, _, _ = render(spec, 0.05)
        inputs[spec["name"]] = (c0, c1, a0, a1, v0, v1, target)
        paths = {}
        for name, im in [
            ("color0", c0),
            ("color1", c1),
            ("aux0", a0),
            ("aux1", a1),
            ("target", target),
        ]:
            p = folder / (name + ".bin")
            im.astype("<f4").tofile(p)
            paths[name] = str(p)
        for label, policy in [
            ("strict", 0),
            ("one_sided", 1),
            ("gated", 2),
            ("hold", 3),
            ("crossfade", 4),
        ]:
            frame = {k: paths[k] for k in ["color0", "color1", "aux0", "aux1"]}
            frame.update(
                params=[0.05, -0.05, *v0.tolist(), *v1.tolist()],
                output=str(folder / label),
            )
            sequences.append(
                {
                    "name": spec["name"],
                    "group": spec["group"],
                    "label": label,
                    "policy": policy,
                    "width": 97,
                    "height": 65,
                    "alpha": alpha,
                    "frames": [frame],
                }
            )
    job = out / "job.json"
    job.write_text(json.dumps({"sequences": sequences}, indent=2))
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
    meta = {
        "run_id": out.name,
        "stage": "2D_M5_frozen_policy_new_cases",
        "git_commit": None,
        "engine": "4.8.beta3.official.c51ff06e8",
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "os": platform.platform(),
        "backend": "vulkan",
        "resolution": [97, 65],
        "format": "RGBA32F",
        "seed": 20261115,
        "scene": "36 analytical 2D fixtures",
        "method": "strict/one_sided/gated/hold/crossfade",
        "target_frame_schedule": [0, 0.05, 0.1],
        "actual_frame_schedule": "synchronized accuracy replay",
        "history_schedule": "midpoint and right endpoint",
        "latency_policy": "50 ms midpoint delay for every policy",
        "kernel_parameters": {"half_life_s": 0.25, "alpha": alpha},
        "warm_up": 0,
        "timing_availability": "not_measured",
        "configuration_hash": hashlib.sha256(job.read_bytes()).hexdigest(),
        "policy": "unchanged .5px global motion disagreement threshold; no tuning on new-case outputs",
        "motion_provenance": "known 2D per-object velocity, source IDs/depth; not engine buffers",
        "outcome": "started",
    }
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "analysis/reconstruction_2d.py",
            ROOT / "analysis/temporal_reference.py",
            ROOT / "godot/shaders/reconstruct_visibility_2d.glsl",
            ROOT / "godot/scripts/m5_history_2d_replay.gd",
            ROOT / "godot/scripts/temporal_baselines.gd",
            ROOT / "godot/shaders/feedback.glsl",
        ]
    }
    try:
        meta["gpu_driver_report"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        process = subprocess.run(
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
                "res://scripts/m5_history_2d_replay.gd",
                "--",
                str(job),
            ],
            capture_output=True,
            timeout=180,
        )
        (out / "stdout.log").write_bytes(process.stdout)
        (out / "stderr.log").write_bytes(process.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        if process.returncode or "M5_HISTORY_COMPLETE" not in log or "ERROR:" in log:
            raise RuntimeError("GPU replay failed")
        rows = []
        for seq in sequences:
            c0, c1, a0, a1, v0, v1, target = inputs[seq["name"]]
            frame = seq["frames"][0]
            expected, codes = reconstruct(
                c0, c1, a0, a1, 0.05, -0.05, v0, v1, seq["policy"]
            )
            gpu = np.fromfile(frame["output"] + "_sample.bin", dtype="<f4").reshape(
                c0.shape
            )
            prov = np.fromfile(
                frame["output"] + "_provenance.bin", dtype="<f4"
            ).reshape(c0.shape)[..., 0]
            history = np.fromfile(
                frame["output"] + "_history.bin", dtype="<f4"
            ).reshape(c0.shape)
            oracle = recursive_oracle([c0, expected, c1], [alpha, alpha])
            truth = recursive_oracle([c0, target, c1], [alpha, alpha])
            err = max(
                float(np.abs(gpu - expected).max()),
                float(np.abs(history - oracle).max()),
            )
            if (
                not np.isfinite(gpu).all()
                or not np.isfinite(history).all()
                or err > 2e-6
                or not np.array_equal(prov, codes)
            ):
                raise RuntimeError(
                    "GPU oracle mismatch " + seq["name"] + " " + seq["label"]
                )
            region = np.any(
                (c0[..., :3] != 0) | (c1[..., :3] != 0) | (target[..., :3] != 0), axis=2
            )
            rows.append(
                {
                    "case": seq["name"],
                    "group": seq["group"],
                    "policy": seq["label"],
                    "oracle_max_abs": err,
                    "sample": metrics(gpu, target, region),
                    "history": metrics(history, truth, region),
                    "provenance_fraction": {
                        str(k): float(np.mean(codes == k)) for k in range(4)
                    },
                }
            )
        meta.update(outcome="passed", results=rows)
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
