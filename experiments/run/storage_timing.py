"""Counterbalanced isolated per-capture GPU timing on preloaded identical samples."""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.timing_report import validate_samples


def main():
    source = Path(sys.argv[1]).resolve()
    scene = sys.argv[2] if len(sys.argv) > 2 else "camera"
    manifest = json.loads((source / "sha256.json").read_text())
    samples = json.loads((source / ("input_" + scene) / "samples.json").read_text())[
        "samples"
    ]
    for frame in samples:
        p = Path(frame["path"])
        assert (
            hashlib.sha256(p.read_bytes()).hexdigest()
            == manifest[p.relative_to(source).as_posix()]
        )
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-storage-timing-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    comparison = len(sys.argv) > 3 and sys.argv[3] == "prepare"
    synthesis = len(sys.argv) > 3 and sys.argv[3] == "synthesis"
    innovation = len(sys.argv) > 3 and sys.argv[3] == "innovation"
    order = (
        ["none", "dense", "innovation"]
        if innovation
        else (
            ["horizon", "independent", "lifetime"]
            if synthesis
            else (
                ["serial", "parallel"]
                if comparison
                else ["none", "dense", "lifetime", "current", "distance"]
            )
        )
    )
    random.Random(31).shuffle(order)
    plan = [
        {"block": b, "mode": m}
        for b in range(5)
        for m in order[b % len(order) :] + order[: b % len(order)]
    ]
    meta = {
        "run_id": out.name,
        "stage": "five_run_isolated_GPU_storage_pilot",
        "source": str(source),
        "scene": scene,
        "seed": 31,
        "plan": plan,
        "git_commit": None,
        "engine": "4.8.beta3.official.c51ff06e8",
        "os": platform.platform(),
        "backend": "Vulkan Forward+",
        "resolution": [160, 90],
        "format": "RGBA32F",
        "warm_up": "at least120captures and2seconds, aligned to40-capture sequence",
        "measurement": "at least300captures and10seconds, whole40-capture cycles",
        "timing": "main-device asynchronous GPU timestamps; CPU submission reported separately",
        "policy": "identical input stream preloaded outside query scope; history reset at each cycle; no image readback during timing",
        "outcome": "started",
        "results": [],
    }

    def telemetry():
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.gr",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()

    try:
        meta["gpu_before"] = telemetry()
        for ordinal, item in enumerate(plan):
            folder = out / (str(ordinal).zfill(2) + "_" + item["mode"])
            folder.mkdir()
            job = {
                "mode": item["mode"],
                "parallel_prepare": item["mode"] != "serial",
                "synthesis": item["mode"] in ["horizon", "independent"],
                "synthesis_policy": (
                    item["mode"]
                    if item["mode"] in ["horizon", "independent"]
                    else "horizon"
                ),
                "width": 160,
                "height": 90,
                "tile": 16,
                "history": 10,
                "slot_multiplier": 1.25,
                "score_policy": (
                    item["mode"]
                    if item["mode"] in ["lifetime", "current", "distance"]
                    else "lifetime"
                ),
                "alpha": 2 ** (-0.05 / 0.25),
                "inputs": [f["path"] for f in samples],
                "warm_up": 120,
                "warm_up_us": 2000000,
                "minimum_frames": 300,
                "duration_us": 10000000,
                "output": str(folder / "samples.json"),
            }
            if item["mode"] == "innovation":
                job.update(innovation=True, pool_slots=763)
            path = folder / "job.json"
            path.write_text(json.dumps(job, indent=2))
            before = telemetry()
            cmd = [
                str(ROOT / "Godot_v4.8-beta3_win64.exe"),
                "--path",
                str(ROOT / "godot"),
                "res://scenes/storage_timing.tscn",
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
            proc = subprocess.run(cmd, capture_output=True, timeout=120)
            (folder / "stdout.log").write_bytes(proc.stdout)
            (folder / "stderr.log").write_bytes(proc.stderr)
            log = (folder / "engine.log").read_text(errors="replace")
            assert (
                proc.returncode == 0
                and "STORAGE_TIMING_COMPLETE" in log
                and "ERROR:" not in log
            ), log[:1500]
            data = json.loads((folder / "samples.json").read_text())
            rows = validate_samples(data, 120)
            assert (
                len(rows) >= 300
                and rows[-1]["wall_us"] - rows[0]["wall_us"] >= 10000000
            )
            assert len(rows) % 40 == 0
            row = {
                **item,
                "folder": folder.name,
                "captures": len(rows),
                "history_payload_bytes": data["history_payload_bytes"],
                "input_cache_bytes": data["input_cache_bytes"],
                "gpu_ms_median_p95_p99": np.percentile(
                    [r["gpu_ns"] / 1e6 for r in rows], [50, 95, 99]
                ).tolist(),
                "cpu_submission_ms_median": float(
                    np.median([r["cpu_submission_us"] / 1000 for r in rows])
                ),
                "gpu_before": before,
                "gpu_after": telemetry(),
            }
            meta["results"].append(row)
            print(json.dumps(row), flush=True)
            (out / "progress.json").write_text(json.dumps(meta, indent=2))
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "godot/scripts/storage_timing_effect.gd",
            ROOT / "godot/scripts/shared_history.gd",
            ROOT / "godot/scripts/dense_finite_history.gd",
            ROOT / "godot/scripts/synthesis_history.gd",
            *ROOT.glob("godot/shaders/synthesis*.glsl"),
            *ROOT.glob("godot/shaders/shared*.glsl"),
            ROOT / "godot/shaders/dense_finite.glsl",
        ]
    }
    for p in [
        ROOT / "godot/scripts/innovation_history.gd",
        *ROOT.glob("godot/shaders/innovation*.glsl"),
    ]:
        meta["implementation_hashes"][p.relative_to(ROOT).as_posix()] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
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
    print(out, meta["outcome"], meta.get("error"), flush=True)
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
