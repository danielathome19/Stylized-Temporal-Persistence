"""Restricted M5 GPU reconstruction pilot: known horizontal velocity, delayed endpoints."""

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
from analysis.reconstruction_reference import reconstruct, synthetic
from analysis.temporal_reference import recursive_oracle


def metrics(im, target, mask):
    diff = im[..., :3].astype(float) - target[..., :3]
    a = im[..., 0] > 0.5
    b = target[..., 0] > 0.5
    union = np.count_nonzero(a | b)
    return {
        "mse_rgb": float(np.mean(diff * diff)),
        "region_mse": float(np.mean(diff[mask] ** 2)) if mask.any() else None,
        "object_iou": float(np.count_nonzero(a & b) / union) if union else 1.0,
    }


def main():
    ref = ROOT / "results/raw/20261115T033527Z-reference-bd9de00e"
    for rel, h in json.loads((ref / "sha256.json").read_text()).items():
        assert hashlib.sha256((ref / rel).read_bytes()).hexdigest() == h
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-reconstruction-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    jobs = []
    truth = {}
    cases = {}

    def add(name, c0, c1, a0, a1, target, t0, t, t1, v0, v1):
        folder = out / name
        folder.mkdir()
        h, w, _ = c0.shape
        job = {
            "name": name,
            "width": w,
            "height": h,
            "params": [t - t0, t - t1, v0, v1],
            "output": str(folder / "output"),
            "source_times": [t0, t1],
            "target_time": t,
            "delivery_time": t1,
            "latency_s": t1 - t,
        }
        for key, im in [
            ("color0", c0),
            ("color1", c1),
            ("aux0", a0),
            ("aux1", a1),
            ("target", target),
        ]:
            path = folder / (key + ".bin")
            im.astype("<f4").tofile(path)
            if key != "target":
                job[key] = str(path)
        jobs.append(job)
        truth[name] = target
        cases[name] = (c0, c1, a0, a1)

    job = json.loads((ref / "A/job.json").read_text())
    images = {
        round(f["time"] * 1e6): np.fromfile(f["path"] + ".bin", dtype="<f2")
        .reshape(180, 320, 4)
        .astype("<f4")
        for f in job["frames"][:24]
    }

    def aux(im):
        a = np.zeros_like(im)
        a[..., 2] = 10
        moving = im[..., 0] > 0.5
        a[moving] = [60, 0, 2, 1]
        return a

    for tick in range(50000, 1000000, 100000):
        c0 = images[tick - 50000]
        c1 = images[tick + 50000]
        add(
            "A_" + str(tick),
            c0,
            c1,
            aux(c0),
            aux(c1),
            images[tick],
            (tick - 50000) / 1e6,
            tick / 1e6,
            (tick + 50000) / 1e6,
            60,
            60,
        )
    for kind in ["translation", "occlusion", "hidden_midpoint", "acceleration"]:
        c0, a0, v0 = synthetic(0, kind)
        c1, a1, v1 = synthetic(1, kind)
        target, _, _ = synthetic(0.5, kind)
        add(kind, c0, c1, a0, a1, target, 0, 0.5, 1, v0, v1)
    # Empty/static and zero-displacement verify normalization and no invented motion.
    for name, value in [("empty", 0.0), ("static", 0.4)]:
        c = np.full((19, 17, 4), value, np.float32)
        c[..., 3] = 1
        a = np.zeros_like(c)
        a[..., 2] = 10
        add(name, c, c, a, a, c, 0, 0.05, 0.1, 0, 0)
    path = out / "job.json"
    path.write_text(json.dumps({"cases": jobs}, indent=2))
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
    meta = {
        "run_id": out.name,
        "git_commit": None,
        "stage": "restricted_M5_reconstruction_accuracy",
        "engine": "4.8.beta3.official.c51ff06e8",
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "os": platform.platform(),
        "backend": "vulkan",
        "format": "RGBA32F",
        "seed": 0,
        "resolution": "per-case width/height in job",
        "scene": "Godot A plus analytical odd-size fixtures",
        "method": "two-endpoint uniform-horizontal-motion gather with depth/id agreement and causal-hold fallback",
        "target_frame_schedule": "per-case source/target/delivery times in job",
        "actual_frame_schedule": "synchronized replay, not delivery benchmark",
        "history_schedule": "20 Hz target, 10 Hz source for A",
        "kernel_parameters": {"half_life_s": 0.25},
        "latency_policy": "wait until right endpoint; hold/crossfade evaluated at same delivery",
        "warm_up": 0,
        "timing_availability": "not_measured",
        "motion_provenance": "analytical per-object velocity; source-only IDs/depth; not engine motion vectors",
        "configuration_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "outcome": "started",
        "tolerance_max_abs": 2e-6,
    }
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "analysis/reconstruction_reference.py",
            ROOT / "godot/scripts/reconstruction_replay.gd",
            ROOT / "godot/shaders/reconstruct_translation.glsl",
            ROOT / "godot/scripts/temporal_baselines.gd",
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
                "res://scripts/reconstruction_replay.gd",
                "--",
                str(path),
            ],
            capture_output=True,
            timeout=120,
        )
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        if proc.returncode or "RECONSTRUCTION_COMPLETE" not in log or "ERROR:" in log:
            raise RuntimeError("Engine error")
        results = []
        reconstructed = {}
        for job in jobs:
            name = job["name"]
            c0, c1, a0, a1 = cases[name]
            target = truth[name]
            shape = c0.shape
            gpu = np.fromfile(job["output"] + ".bin", dtype="<f4").reshape(shape)
            valid = (
                np.fromfile(job["output"] + "_valid.bin", dtype="<f4").reshape(shape)[
                    ..., 0
                ]
                > 0.5
            )
            oracle, expected_valid = reconstruct(c0, c1, a0, a1, *job["params"])
            err = float(np.max(np.abs(gpu - oracle)))
            if (
                not np.isfinite(gpu).all()
                or err > 2e-6
                or not np.array_equal(valid, expected_valid)
            ):
                raise RuntimeError("GPU oracle mismatch " + name)
            mask = np.any(
                (c0[..., :3] != 0) | (c1[..., :3] != 0) | (target[..., :3] != 0), axis=2
            )
            fraction = job["params"][0] / (job["params"][0] - job["params"][1])
            cross = (1 - fraction) * c0 + fraction * c1
            results.append(
                {
                    "case": name,
                    "oracle_max_abs": err,
                    "fallback_fraction": float(np.mean(~valid)),
                    "latency_s": job["latency_s"],
                    "hold": metrics(c0, target, mask),
                    "crossfade": metrics(cross, target, mask),
                    "reconstruct": metrics(gpu, target, mask),
                }
            )
            reconstructed[name] = gpu
        alpha = 2 ** (-0.05 / 0.25)
        exact = [images[t] for t in range(0, 1000001, 50000)]
        held = []
        recon = []
        for t in range(0, 1000001, 50000):
            held.append(images[t - t % 100000])
            recon.append(reconstructed["A_" + str(t)] if t % 100000 else images[t])
        truth_trail = recursive_oracle(exact, [alpha] * 20)
        meta["trail_endpoint"] = {}
        for label, frames in [("hold", held), ("reconstruct", recon)]:
            composite = recursive_oracle(frames, [alpha] * 20)
            meta["trail_endpoint"][label] = metrics(
                composite, truth_trail, np.any(truth_trail[..., :3] > 0, axis=2)
            )
            np.save(out / (label + "_trail.npy"), composite)
        np.save(out / "exact_trail.npy", truth_trail)
        meta.update(outcome="passed", results=results)
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
            {
                "run": str(out),
                "outcome": meta["outcome"],
                "error": meta.get("error"),
                "results": meta.get("results"),
                "trail_endpoint": meta.get("trail_endpoint"),
            },
            indent=2,
        )
    )
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
