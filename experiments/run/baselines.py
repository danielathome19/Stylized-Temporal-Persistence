"""GPU M1-M4 correctness replay with independently expanded float64 oracles.

This uploads inputs and reads every result: never a throughput benchmark.
"""

import argparse
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
from analysis.temporal_reference import recursive_oracle, kernel_weights


def expected(observations, config):
    """Rebuild intended result from all eligible samples, not previous output."""
    times = [t for t, _ in observations]
    images = [im for _, im in observations]
    now = times[-1]
    a = config["alpha"]
    method = config["method"]
    period = config["period_us"]
    if method in ("M1", "M2"):
        alphas = [
            a if method == "M1" else a ** ((r - l) / period)
            for l, r in zip(times, times[1:])
        ]
        return recursive_oracle(images, alphas)
    targets = list(range(times[0], now + 1, period))
    sources = [max(i for i, t in enumerate(times) if t <= tick) for tick in targets]
    sampled = [images[i] for i in sources]
    if method == "M3":
        return recursive_oracle(sampled, [a] * (len(sampled) - 1))
    eligible = [i for i, t in enumerate(targets) if 0 < now - t <= config["support_us"]]
    if not eligible:
        return images[-1].copy()
    weights = kernel_weights(
        [(now - targets[i]) / 1e6 for i in eligible],
        config["kernel"],
        config["support_us"] / 1e6,
        config["scale_s"],
    )
    if not weights.sum():
        return images[-1].copy()
    return config["current_weight"] * images[-1] + (
        1 - config["current_weight"]
    ) * np.tensordot(weights, np.asarray([sampled[i] for i in eligible]), axes=(0, 0))


def fixture(index, width=17, height=19):
    im = np.zeros((height, width, 4), dtype="<f4")
    im[:, :, 3] = 1
    x = (index * 3) % (width - 4)
    im[3:12, x : x + 4, 0] = 1
    im[8:15, (width - 5 - x // 2) : (width - 1 - x // 2), 1] = 0.5
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "results/raw/20261115T033527Z-reference-bd9de00e",
    )
    args = parser.parse_args()
    run_id = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-baselines-"
        + uuid.uuid4().hex[:8]
    )
    out = ROOT / "results/raw" / run_id
    out.mkdir(parents=True, exist_ok=False)
    methods = [("M1", "exponential"), ("M2", "exponential"), ("M3", "exponential")] + [
        ("M4", k) for k in ("exponential", "linear", "gaussian", "hyperbolic")
    ]
    schedules = {
        "30hz": [round(i * 1e6 / 30) for i in range(31)],
        "jitter_drops": [
            0,
            17000,
            72000,
            73000,
            190000,
            410000,
            425000,
            600000,
            900000,
            1000000,
        ],
        "pause_cut_seek": [
            0,
            50000,
            100000,
            100000,
            150000,
            200000,
            250000,
            10000,
            60000,
        ],
    }
    cases = []
    oracle = {}
    expected_events = {}
    for schedule, times in schedules.items():
        for method, kernel in methods:
            name = f"{schedule}_{method}_{kernel}"
            folder = out / name
            folder.mkdir()
            config = {
                "name": name,
                "method": method,
                "kernel": kernel,
                "width": 17,
                "height": 19,
                "alpha": math.exp(-math.log(2) * 0.05 / 0.25),
                "period_us": 50000,
                "support_us": 250000,
                "scale_s": 0.25 / math.log(2),
                "current_weight": 0.25,
                "events": str(folder / "events.json"),
                "frames": [],
            }
            obs = []
            prev_output = None
            events = []
            for i, t in enumerate(times):
                # Static/zero-retention covered in additional cases below.
                im = fixture(i)
                path = folder / f"input_{i:03d}.bin"
                im.tofile(path)
                cut = schedule == "pause_cut_seek" and i == 5
                if cut or (obs and t < obs[-1][0]):
                    obs = []
                paused = bool(obs and t == obs[-1][0])
                if not paused:
                    if obs and method in ("M3", "M4"):
                        first = (
                            obs[0][0] + ((obs[-1][0] - obs[0][0]) // 50000 + 1) * 50000
                        )
                        for tick in range(first, t + 1, 50000):
                            events.append(
                                {
                                    "target_us": tick,
                                    "source_us": t if tick == t else obs[-1][0],
                                    "delivered_us": t,
                                    "missed": tick != t,
                                }
                            )
                    obs.append((t, im.astype(np.float64)))
                    prev_output = expected(obs, config)
                output = folder / f"output_{i:03d}.bin"
                config["frames"].append(
                    {
                        "time_us": t,
                        "cut": cut,
                        "input": str(path),
                        "output": str(output),
                    }
                )
                oracle[str(output)] = prev_output.copy()
            expected_events[name] = events
            cases.append(config)
    # Empty/static/zero retention exercise separate allocations and different odd sizes.
    for mode in ["static", "empty", "zero"]:
        for method, kernel in methods:
            name = f"{mode}_{method}_{kernel}"
            folder = out / name
            folder.mkdir()
            cfg = {
                "name": name,
                "method": method,
                "kernel": kernel,
                "width": 9,
                "height": 11,
                "alpha": 0 if mode == "zero" else 0.8,
                "period_us": 50000,
                "support_us": 250000,
                "scale_s": 0.25 / math.log(2),
                "current_weight": 0.25,
                "events": str(folder / "events.json"),
                "frames": [],
            }
            obs = []
            for i, t in enumerate([0, 50000, 100000, 400000]):
                im = fixture(i if mode == "zero" else 0, 9, 11)
                if mode == "empty":
                    im[:, :, :3] = 0
                path = folder / f"input_{i:03d}.bin"
                im.tofile(path)
                dest = folder / f"output_{i:03d}.bin"
                cfg["frames"].append(
                    {"time_us": t, "input": str(path), "output": str(dest)}
                )
                obs.append((t, im.astype(np.float64)))
                oracle[str(dest)] = expected(obs, cfg)
            cases.append(cfg)
    # Replay real no-effect Godot scene captures, not synthetic expected images.
    # Recreate dimensions on the same core, then verify freshly seeded history.
    for method, kernel in methods:
        name = f"resize_{method}_{kernel}"
        folder = out / name
        folder.mkdir()
        cfg = {
            "name": name,
            "method": method,
            "kernel": kernel,
            "width": 17,
            "height": 19,
            "alpha": 0.8,
            "period_us": 50000,
            "support_us": 250000,
            "scale_s": 0.25 / math.log(2),
            "current_weight": 0.25,
            "events": str(folder / "events.json"),
            "frames": [],
        }
        obs = []
        for i, (w, h) in enumerate([(17, 19), (17, 19), (9, 11), (9, 11), (17, 19)]):
            changed = i in (2, 4)
            if changed:
                obs = []
            im = fixture(i, w, h)
            path = folder / f"input_{i:03d}.bin"
            im.tofile(path)
            dest = folder / f"output_{i:03d}.bin"
            t = i * 50000
            frame = {
                "time_us": t,
                "input": str(path),
                "output": str(dest),
                "output_width": w,
                "output_height": h,
            }
            if changed:
                frame.update(width=w, height=h)
            cfg["frames"].append(frame)
            obs.append((t, im.astype(np.float64)))
            oracle[str(dest)] = expected(obs, cfg)
        cases.append(cfg)
    reference = args.reference.resolve()
    if not reference.exists():
        raise RuntimeError(
            "Run reference.py first and set the reference path in this script."
        )
    for rel, digest in json.loads((reference / "sha256.json").read_text()).items():
        if hashlib.sha256((reference / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Reference hash mismatch: " + rel)
    for scene in ["A", "B", "C"]:
        job = json.loads((reference / scene / "job.json").read_text())
        selected = [
            f
            for f in job["frames"][:24]
            if f["rational_time"] in ["0", "1/20", "1/10", "1/2", "1"]
        ]
        for method, kernel in methods:
            name = f"scene{scene}_{method}_{kernel}"
            folder = out / name
            folder.mkdir()
            cfg = {
                "name": name,
                "method": method,
                "kernel": kernel,
                "width": 320,
                "height": 180,
                "alpha": 0.8,
                "period_us": 50000,
                "support_us": 250000,
                "scale_s": 0.25 / math.log(2),
                "current_weight": 0.25,
                "events": str(folder / "events.json"),
                "frames": [],
            }
            obs = []
            for i, frame in enumerate(selected):
                im = (
                    np.fromfile(frame["path"] + ".bin", dtype="<f2")
                    .reshape(180, 320, 4)
                    .astype("<f4")
                )
                path = folder / f"input_{i:03d}.bin"
                im.tofile(path)
                dest = folder / f"output_{i:03d}.bin"
                t = round(frame["time"] * 1e6)
                cfg["frames"].append(
                    {"time_us": t, "input": str(path), "output": str(dest)}
                )
                obs.append((t, im.astype(np.float64)))
                oracle[str(dest)] = expected(obs, cfg)
            cases.append(cfg)
    job_path = out / "job.json"
    job_path.write_text(json.dumps({"cases": cases}, indent=2))
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
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
        str(job_path),
    ]
    metadata = {
        "run_id": run_id,
        "stage": "GPU_correctness_replay_not_benchmark",
        "engine": "4.8.beta3.official.c51ff06e8",
        "os": platform.platform(),
        "backend": "vulkan",
        "format": "RGBA32F",
        "seed": 0,
        "history_schedule": "50,000 us, reset-anchored phase",
        "warm_up": 0,
        "latency_policy": "causal",
        "timing_availability": "not_measured",
        "git_commit": None,
        "outcome": "started",
        "configuration_hash": hashlib.sha256(job_path.read_bytes()).hexdigest(),
        "cases": len(cases),
        "tolerance_max_abs": 2e-6,
    }
    metadata["reference_run"] = str(reference)
    metadata["engine_sha256"] = hashlib.sha256(engine.read_bytes()).hexdigest()
    metadata["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in list((ROOT / "godot/shaders").glob("*.glsl"))
        + list((ROOT / "godot/scripts").glob("*.gd"))
        + [Path(__file__), ROOT / "analysis/temporal_reference.py"]
    }
    try:
        metadata["gpu_driver_report"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        metadata["gpu_driver_report"] = None
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=180)
        (out / "stdout.log").write_bytes(result.stdout)
        (out / "stderr.log").write_bytes(result.stderr)
        log = (out / "engine.log").read_text(encoding="utf-8", errors="replace")
        if result.returncode or "BASELINE_REPLAY_COMPLETE" not in log:
            raise RuntimeError("GPU replay incomplete; inspect engine.log")
        rows = []
        for case in cases:
            error = 0.0
            mse = 0.0
            for f in case["frames"]:
                gpu = (
                    np.fromfile(f["output"], dtype="<f4")
                    .reshape(
                        f.get("output_height", case["height"]),
                        f.get("output_width", case["width"]),
                        4,
                    )
                    .astype(np.float64)
                )
                diff = gpu - oracle[f["output"]]
                if not np.isfinite(gpu).all():
                    raise RuntimeError("Nonfinite output")
                error = max(error, float(np.abs(diff).max()))
                mse = max(mse, float(np.mean(diff * diff)))
            if case["name"] in expected_events:
                if (
                    json.loads(Path(case["events"]).read_text())
                    != expected_events[case["name"]]
                ):
                    raise RuntimeError("Schedule event mismatch: " + case["name"])
            rows.append(
                {
                    "case": case["name"],
                    "frames": len(case["frames"]),
                    "max_abs_error": error,
                    "max_frame_mse": mse,
                    "pass": error <= 2e-6,
                }
            )
        metadata["results"] = rows
        metadata["outcome"] = "passed" if all(r["pass"] for r in rows) else "failed"
        metadata["engine_errors_present"] = "ERROR:" in log
        metadata["outputs_compared"] = sum(r["frames"] for r in rows)
        metadata["worst_abs_error"] = max(r["max_abs_error"] for r in rows)
    except Exception as exc:
        metadata["outcome"] = "failed"
        metadata["error"] = str(exc)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    manifest = {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.rglob("*")
        if p.is_file()
    }
    (out / "sha256.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                k: metadata.get(k)
                for k in [
                    "run_id",
                    "outcome",
                    "error",
                    "cases",
                    "outputs_compared",
                    "worst_abs_error",
                    "engine_errors_present",
                ]
            },
            indent=2,
        )
    )
    return 0 if metadata["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
