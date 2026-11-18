"""Capture renderer-native color/depth/motion from perspective Godot scenes."""

import datetime
import hashlib
import json
import platform
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-3d-inputs-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
    meta = {
        "run_id": out.name,
        "git_commit": None,
        "stage": "native_3D_inputs",
        "engine": "4.8.beta3.official.c51ff06e8",
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "os": platform.platform(),
        "backend": "vulkan",
        "resolution": [320, 180],
        "format": "RGBA32F staging of native scene buffers",
        "method": "POST_TRANSPARENT resolved capture",
        "scene": ["translation", "rotation", "occlusion", "camera"],
        "seed": 0,
        "target_frame_schedule": [0.2, 0.2, 0.2, 0, 0.2, 0.1],
        "actual_frame_schedule": "render frame and wall_us per capture",
        "history_schedule": "none; endpoint interval .2s, midpoint .1s",
        "kernel_parameters": None,
        "latency_policy": "offline capture, bidirectional endpoint motion",
        "warm_up": "three repeated right-endpoint states",
        "timing_availability": "not_measured",
        "outcome": "started",
    }
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "godot/scripts/capture_3d.gd",
            ROOT / "godot/scripts/scene_3d.gd",
            ROOT / "godot/shaders/capture_3d.glsl",
            ROOT / "godot/scripts/reference_scene.gd",
        ]
    }
    jobs = []
    try:
        meta["gpu_driver_report"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        for scene in meta["scene"]:
            folder = out / scene
            folder.mkdir()
            job = {
                "scene": scene,
                "frames": [
                    {"time": t, "path": str(folder / name)}
                    for name, t in [
                        ("warm0", 0.2),
                        ("warm1", 0.2),
                        ("warm2", 0.2),
                        ("left", 0),
                        ("right", 0.2),
                        ("target", 0.1),
                    ]
                ],
            }
            path = folder / "job.json"
            path.write_text(json.dumps(job, indent=2))
            jobs.append(job)
            proc = subprocess.run(
                [
                    str(engine),
                    "--path",
                    str(ROOT / "godot"),
                    "--scene",
                    "res://scenes/capture_3d.tscn",
                    "--rendering-method",
                    "forward_plus",
                    "--rendering-driver",
                    "vulkan",
                    "--audio-driver",
                    "Dummy",
                    "--log-file",
                    str(folder / "engine.log"),
                    "--",
                    str(path),
                ],
                capture_output=True,
                timeout=90,
            )
            (folder / "stdout.log").write_bytes(proc.stdout)
            (folder / "stderr.log").write_bytes(proc.stderr)
            log = (folder / "engine.log").read_text(errors="replace")
            if proc.returncode or "CAPTURE_3D_COMPLETE" not in log or "ERROR:" in log:
                raise RuntimeError("Engine failed " + scene)
        meta["outcome"] = "passed"
    except Exception as e:
        meta.update(outcome="failed", error=str(e))
    meta["configuration_hash"] = hashlib.sha256(
        json.dumps(jobs, sort_keys=True).encode()
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
    print(
        json.dumps(
            {"run": str(out), "outcome": meta["outcome"], "error": meta.get("error")}
        )
    )
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
