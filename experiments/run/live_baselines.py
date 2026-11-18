"""End-to-end compositor accuracy validation against independent no-effect captures."""

import datetime
import hashlib
import json
import math
import subprocess
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.run.baselines import expected


def main():
    reference = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else ROOT / "results/raw/20261115T033527Z-reference-bd9de00e"
    )
    for rel, digest in json.loads((reference / "sha256.json").read_text()).items():
        assert hashlib.sha256((reference / rel).read_bytes()).hexdigest() == digest, rel
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-live-baselines-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    status = "passed"
    error = None
    try:
        for scene in ["A", "B", "C"]:
            original = json.loads((reference / scene / "job.json").read_text())
            selected = [
                f
                for f in original["frames"][:24]
                if f["rational_time"] in ["0", "1/20", "1/10", "1/2", "1"]
            ]
            for method in ["M1", "M2", "M3", "M4"]:
                folder = out / f"{scene}_{method}"
                folder.mkdir()
                cfg = {
                    "method": method,
                    "alpha": 0.8,
                    "period_us": 50000,
                    "support_us": 250000,
                    "current_weight": 0.25,
                    "kernel": "exponential",
                    "scale_s": 0.25 / math.log(2),
                }
                frames = [
                    {"time": f["time"], "path": str(folder / f"{i:03d}")}
                    for i, f in enumerate(selected)
                ]
                job = {
                    "scene": scene,
                    "width": 320,
                    "height": 180,
                    "persistence": cfg,
                    "frames": frames,
                }
                path = folder / "job.json"
                path.write_text(json.dumps(job, indent=2))
                cmd = [
                    str(ROOT / "Godot_v4.8-beta3_win64.exe"),
                    "--path",
                    str(ROOT / "godot"),
                    "--scene",
                    "res://scenes/reference.tscn",
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
                ]
                process = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=90)
                (folder / "stdout.log").write_bytes(process.stdout)
                (folder / "stderr.log").write_bytes(process.stderr)
                log = (folder / "engine.log").read_text(
                    encoding="utf-8", errors="replace"
                )
                if (
                    process.returncode
                    or "REFERENCE_CAPTURE_COMPLETE" not in log
                    or "ERROR:" in log
                ):
                    raise RuntimeError("Engine error: " + folder.name)
                obs = []
                worst = 0
                for i, source in enumerate(selected):
                    current = (
                        np.fromfile(source["path"] + ".bin", dtype="<f2")
                        .reshape(180, 320, 4)
                        .astype(np.float64)
                    )
                    obs.append((round(source["time"] * 1e6), current))
                    target = expected(obs, cfg)
                    actual = (
                        np.fromfile(folder / f"{i:03d}.bin", dtype="<f2")
                        .reshape(180, 320, 4)
                        .astype(np.float64)
                    )
                    if not np.isfinite(actual).all():
                        raise RuntimeError("Nonfinite output")
                    worst = max(worst, float(np.max(np.abs(actual - target))))
                    if i == len(selected) - 1:
                        from PIL import Image

                        rgb = actual[:, :, :3]
                        encoded = np.where(
                            rgb <= 0.0031308,
                            12.92 * rgb,
                            1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055,
                        )
                        Image.fromarray(
                            np.uint8(np.clip(encoded, 0, 1) * 255 + 0.5)
                        ).save(folder / "final.png")
                rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "frames": len(frames),
                        "max_abs_error": worst,
                        "pass": worst <= 0.001,
                    }
                )
        if not all(r["pass"] for r in rows):
            status = "failed"
    except Exception as exc:
        status = "failed"
        error = str(exc)
    metadata = {
        "outcome": status,
        "error": error,
        "stage": "end_to_end_compositor_accuracy_not_benchmark",
        "reference_run": str(reference),
        "engine": "4.8.beta3.official.c51ff06e8",
        "backend": "vulkan",
        "resolution": [320, 180],
        "format": "RGBA32F history / RGBA16F presentation",
        "timing_availability": "not_measured",
        "warm_up": "two held frames per timestamp",
        "latency_policy": "causal",
        "seed": 0,
        "tolerance_max_abs": 0.001,
        "results": rows,
    }
    metadata["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in list((ROOT / "godot/shaders").glob("*.glsl"))
        + list((ROOT / "godot/scripts").glob("*.gd"))
        + [Path(__file__)]
    }
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
                "run": str(out),
                "outcome": status,
                "error": error,
                "cases": len(rows),
                "worst_abs_error": max(
                    (r["max_abs_error"] for r in rows), default=None
                ),
            },
            indent=2,
        )
    )
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
