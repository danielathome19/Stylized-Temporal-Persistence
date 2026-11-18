"""Small common-target GPU fidelity pilot. Synchronized replay, never GPU timing."""

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
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.temporal_reference import recursive_oracle, live_composite
from experiments.run.baselines import expected


def metrics(actual, target, mask):
    d = np.asarray(actual, dtype=np.float64)[..., :3] - target[..., :3]
    mse = float(np.mean(d * d))
    return {
        "mse_rgb": mse,
        "psnr_db": 10 * math.log10(1 / mse) if mse else None,
        "psnr_infinite": mse == 0,
        "motion_region_mse": float(np.mean(d[mask] ** 2)) if mask.any() else None,
    }


def main():
    reference = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else ROOT / "results/raw/20261115T033527Z-reference-bd9de00e"
    )
    for rel, h in json.loads((reference / "sha256.json").read_text()).items():
        assert hashlib.sha256((reference / rel).read_bytes()).hexdigest() == h, rel
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-fidelity-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    schedules = {
        "20hz": list(range(0, 1000001, 50000)),
        "10hz": list(range(0, 1000001, 100000)),
        "irregular": [
            0,
            33333,
            116667,
            200000,
            350000,
            500000,
            550000,
            800000,
            966667,
            1000000,
        ],
    }
    a = 2 ** (-0.05 / 0.25)
    specification = {
        "schedules_us": schedules,
        "alpha": a,
        "half_life_s": 0.25,
        "support_s": 1.0,
        "current_weight": 1 - a,
        "phase_us": 0,
        "seed_policy": "current at zero",
        "targets": ["sampled_recursive", "live_finite"],
        "primary_summary": "endpoint at 1 second, no independent statistical replicates",
        "motion_region": "union of nonblack RGB source pixels over all exact history and presentation captures",
        "data_range": 1,
        "psnr_zero_error": "null plus psnr_infinite=true",
        "warm_up": 0,
    }
    (out / "specification.json").write_text(json.dumps(specification, indent=2))
    cases = []
    scenes = {}
    targets = {}
    masks = {}
    for scene in "ABC":
        folder = out / scene
        folder.mkdir()
        job = json.loads((reference / scene / "job.json").read_text())
        images = {
            round(f["time"] * 1e6): np.fromfile(f["path"] + ".bin", dtype="<f2")
            .reshape(180, 320, 4)
            .astype("<f4")
            for f in job["frames"][:24]
        }
        scenes[scene] = images
        masks[scene] = np.any(
            np.stack([im[..., :3] != 0 for im in images.values()]), axis=(0, 3)
        )
        for t, im in images.items():
            im.tofile(folder / f"{t}.bin")
        for t in sorted(set(sum(schedules.values(), []))):
            ticks = list(range(0, t + 1, 50000))
            hist = [images[k] for k in ticks]
            targets[scene, t, "sampled_recursive"] = recursive_oracle(
                hist, [a] * (len(hist) - 1)
            )
            targets[scene, t, "live_finite"] = live_composite(
                images[t], hist, [k / 1e6 for k in ticks], t / 1e6, 1 - a, support=1.0
            )
        for schedule, times in schedules.items():
            for method in ["M1", "M2", "M3", "M4"]:
                dest = folder / (schedule + "_" + method)
                dest.mkdir()
                cases.append(
                    {
                        "name": scene + "_" + schedule + "_" + method,
                        "scene": scene,
                        "schedule": schedule,
                        "method": method,
                        "kernel": "exponential",
                        "width": 320,
                        "height": 180,
                        "alpha": a,
                        "period_us": 50000,
                        "support_us": 1000000,
                        "scale_s": 0.25 / math.log(2),
                        "current_weight": 1 - a,
                        "events": str(dest / "events.json"),
                        "frames": [
                            {
                                "time_us": t,
                                "input": str(folder / f"{t}.bin"),
                                "output": str(dest / f"{t}.bin"),
                            }
                            for t in times
                        ],
                    }
                )
    job = out / "job.json"
    job.write_text(json.dumps({"cases": cases}, indent=2))
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
    metadata = {
        "run_id": out.name,
        "stage": "common_target_fidelity_pilot_not_timing",
        "git_commit": None,
        "os": platform.platform(),
        "engine": "4.8.beta3.official.c51ff06e8",
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "backend": "vulkan",
        "resolution": [320, 180],
        "format": "RGBA16F independent inputs / RGBA32F history",
        "seed": 0,
        "warm_up": 0,
        "timing_availability": "not_measured",
        "latency_policy": "causal hold, no future input",
        "reference_run": str(reference),
        "history_schedule": "50,000 us",
        "target_frame_schedule": schedules,
        "actual_frame_schedule": "synthetic replay timestamps in job; hardware frame delivery not measured",
        "configuration_hash": hashlib.sha256(
            (out / "specification.json").read_bytes()
        ).hexdigest(),
        "kernel_parameters": specification,
        "outcome": "started",
    }
    metadata["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in list((ROOT / "godot/scripts").glob("*.gd"))
        + list((ROOT / "godot/shaders").glob("*.glsl"))
        + [
            Path(__file__),
            ROOT / "analysis/temporal_reference.py",
            ROOT / "experiments/run/baselines.py",
        ]
    }
    try:
        metadata["gpu_driver_report"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        cmd = [
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
            "res://scripts/baseline_replay.gd",
            "--",
            str(job),
        ]
        process = subprocess.run(cmd, capture_output=True, timeout=240)
        (out / "stdout.log").write_bytes(process.stdout)
        (out / "stderr.log").write_bytes(process.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        if (
            process.returncode
            or "BASELINE_REPLAY_COMPLETE" not in log
            or "ERROR:" in log
        ):
            raise RuntimeError("GPU replay failed; inspect log")
        rows = []
        endpoints = []
        for case in cases:
            scene = case["scene"]
            obs = []
            for frame in case["frames"]:
                t = frame["time_us"]
                obs.append((t, scenes[scene][t].astype(np.float64)))
                gpu = np.fromfile(frame["output"], dtype="<f4").reshape(180, 320, 4)
                if not np.isfinite(gpu).all():
                    raise RuntimeError("Nonfinite output")
                for target in specification["targets"]:
                    row = {
                        "scene": scene,
                        "schedule": case["schedule"],
                        "method": case["method"],
                        "time_us": t,
                        "target": target,
                        **metrics(gpu, targets[scene, t, target], masks[scene]),
                    }
                    rows.append(row)
                    if t == 1000000:
                        endpoints.append(row)
            err = float(np.abs(gpu - expected(obs, case)).max())
            if err > 2e-6:
                raise RuntimeError("Intended endpoint mismatch: " + case["name"])
        (out / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        (out / "endpoints.json").write_text(
            json.dumps(endpoints, indent=2, allow_nan=False)
        )
        metadata.update(
            outcome="passed",
            cases=len(cases),
            outputs=sum(len(c["frames"]) for c in cases),
        )
    except Exception as exc:
        metadata.update(outcome="failed", error=str(exc))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
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
            {
                "run": str(out),
                "outcome": metadata["outcome"],
                "error": metadata.get("error"),
            },
            indent=2,
        )
    )
    return int(metadata["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
