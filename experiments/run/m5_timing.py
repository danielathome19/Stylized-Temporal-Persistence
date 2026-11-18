"""Asynchronous main-device timestamp calibration and exploratory effect pilot."""

import datetime
import hashlib
import json
import platform
import random
import subprocess
import sys
import uuid
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.timing_report import validate_samples

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-m5-timing-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    calibration = "--calibration" in sys.argv
    repeated = "--repeated" in sys.argv
    smoke = "--smoke" in sys.argv
    modes = ["none", "cal1", "cal16"] if calibration else ["none", "M3", "M5"]
    random.Random(19).shuffle(modes)
    plan = [{"mode": m, "instrumented": True, "block": 0} for m in modes]
    if repeated:
        # Latin cyclic order: each method occurs at each position once across five blocks.
        plan = []
        for block in range(5):
            order = modes[block:] + modes[:block]
            for m in order:
                variants = [True, False] if m in ["none", "M4"] else [True]
                if block % 2:
                    variants.reverse()
                plan.extend(
                    {"mode": m, "instrumented": v, "block": block} for v in variants
                )
    if smoke:
        plan = [
            {"mode": m, "instrumented": v, "block": 0}
            for m in ["none", "M4"]
            for v in [True, False]
        ]
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
    metadata = {
        "run_id": out.name,
        "stage": (
            "timestamp_calibration"
            if calibration
            else ("repeated_timing_pilot" if repeated else "exploratory_timing_pilot")
        ),
        "git_commit": None,
        "engine": "4.8.beta3.official.c51ff06e8",
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "os": platform.platform(),
        "backend": "vulkan",
        "resolution": [1280, 720],
        "format": "RGBA32F history, RGBA16F scene",
        "scene": "native perspective oscillating rotating box",
        "seed": 19,
        "method_order": modes,
        "plan": plan,
        "repeated": repeated,
        "smoke": smoke,
        "history_schedule": "20 Hz, M3 causal hold / M5 reconstructed at right endpoint",
        "kernel_parameters": {
            "half_life_s": 0.25,
            "support_s": 0.5,
            "current_weight": 0.25,
        },
        "latency_policy": "causal",
        "target_frame_schedule": "uncapped, vsync disabled",
        "actual_frame_schedule": "callback wall_us and simulation_us in samples",
        "warm_up": {"minimum_frames": 120, "minimum_seconds": 2 if repeated else 0},
        "timing_availability": "main RenderingDevice asynchronous GPU nanoseconds and separate CPU microseconds",
        "power_thermal_policy": "no controls changed; before/after telemetry; overlays uncontrolled",
        "outcome": "started",
        "results": [],
    }
    metadata["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in list((ROOT / "godot/scripts").glob("*.gd"))
        + list((ROOT / "godot/shaders").glob("*.glsl"))
        + [Path(__file__)]
    }

    def telemetry():
        try:
            return subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.gr,utilization.gpu",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        except Exception as e:
            return str(e)

    metadata["gpu_before"] = telemetry()
    try:
        jobs = []
        for ordinal, item in enumerate(plan):
            mode = item["mode"]
            folder = out / (
                str(ordinal).zfill(2)
                + "_"
                + mode
                + ("_on" if item["instrumented"] else "_off")
            )
            folder.mkdir()
            job = {
                "mode": mode,
                "width": 1280,
                "height": 720,
                "warm_up": 10 if smoke else 120,
                "warm_up_us": 200000 if smoke else (2000000 if repeated else 0),
                "instrumented": item["instrumented"],
                "minimum_frames": 30 if smoke else 300,
                "duration_us": (
                    300000 if smoke else (2000000 if calibration else 10000000)
                ),
                "output": str(folder / "samples.json"),
            }
            jobs.append(job)
            path = folder / "job.json"
            path.write_text(json.dumps(job, indent=2))
            (out / "planned_order.json").write_text(json.dumps(plan, indent=2))
            cmd = [
                str(engine),
                "--path",
                str(ROOT / "godot"),
                "--scene",
                "res://scenes/m5_timing.tscn",
                "--rendering-method",
                "forward_plus",
                "--rendering-driver",
                "vulkan",
                "--disable-vsync",
                "--audio-driver",
                "Dummy",
                "--log-file",
                str(folder / "engine.log"),
                "--",
                str(path),
            ]
            gpu_before = telemetry()
            process = subprocess.run(cmd, capture_output=True, timeout=90)
            gpu_after = telemetry()
            (folder / "stdout.log").write_bytes(process.stdout)
            (folder / "stderr.log").write_bytes(process.stderr)
            log = (folder / "engine.log").read_text(errors="replace")
            if (
                process.returncode
                or "TIMING_PILOT_COMPLETE" not in log
                or "ERROR:" in log
            ):
                raise RuntimeError("Engine failed: " + mode)
            data = json.loads((folder / "samples.json").read_text())
            rows = validate_samples(data, job["warm_up"])
            if (
                len(rows) < job["minimum_frames"]
                or rows[-1]["wall_us"] - rows[0]["wall_us"] < job["duration_us"]
            ):
                raise RuntimeError("Missing/invalid timing samples: " + mode)
            result = {
                "mode": mode,
                **item,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "folder": folder.name,
                "frames": len(rows),
                "unresolved_warmup_frames": len(data["unresolved"]),
                "gpu_ms_median_p95_p99": (
                    np.percentile(
                        [r["gpu_ns"] / 1e6 for r in rows], [50, 95, 99]
                    ).tolist()
                    if item["instrumented"]
                    else None
                ),
                "cpu_submission_ms_median_p95_p99": np.percentile(
                    [r["cpu_submission_us"] / 1e3 for r in rows], [50, 95, 99]
                ).tolist(),
                "callback_interval_ms_median_p95_p99": np.percentile(
                    [r["callback_interval_us"] / 1e3 for r in rows], [50, 95, 99]
                ).tolist(),
                "callback_fps": (len(rows) - 1)
                * 1e6
                / (rows[-1]["wall_us"] - rows[0]["wall_us"]),
                "query_delay_callbacks_range": (
                    [
                        min(r["query_delay_callbacks"] for r in rows),
                        max(r["query_delay_callbacks"] for r in rows),
                    ]
                    if item["instrumented"]
                    else None
                ),
                "gpu_update_median_ms": (
                    float(
                        np.median(
                            [
                                r["gpu_ns"] / 1e6
                                for r in rows
                                if r["scheduled_updates"] > 0
                            ]
                        )
                    )
                    if item["instrumented"] and mode in ["M3", "M5"]
                    else None
                ),
                "scheduled_update_frames": sum(
                    r["scheduled_updates"] > 0 for r in rows
                ),
            }
            metadata["results"].append(result)
            print(json.dumps(result), flush=True)
        metadata["configuration_hash"] = hashlib.sha256(
            json.dumps(jobs, sort_keys=True).encode()
        ).hexdigest()
        metadata["outcome"] = "passed"
    except Exception as e:
        metadata.update(outcome="failed", error=str(e))
    metadata["gpu_after"] = telemetry()
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
            }
        ),
        flush=True,
    )
    return int(metadata["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
