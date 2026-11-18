"""Capture exact scene states and compose offline references. Accuracy only."""

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


def main():
    run_id = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-reference-"
        + uuid.uuid4().hex[:8]
    )
    out = ROOT / "results/raw" / run_id
    out.mkdir(parents=True, exist_ok=False)
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
    times = sorted(
        set(
            [Fraction(k, 20) for k in range(21)]
            + [Fraction(1, 30), Fraction(7, 60), Fraction(29, 30)]
        )
    )
    # Revisit states non-monotonically: detect hidden dependence on prior animation state.
    requested = times + [Fraction(1), Fraction(1, 2), Fraction(0)]
    metadata = {
        "run_id": run_id,
        "stage": "exact_time_reference_validation",
        "engine": "4.8.beta3.official.c51ff06e8",
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "os": platform.platform(),
        "git_commit": None,
        "backend": "vulkan",
        "resolution": [320, 180],
        "format": "RGBA16F source / float64 oracle",
        "method": "no_effect_exact_capture",
        "seed": 0,
        "target_frame_schedule": [str(t) for t in requested],
        "actual_frame_schedule": "per-capture render_frame/wall_ticks_usec in JSON",
        "history_schedule": "k/20 seconds",
        "kernel_parameters": {"half_life_s": 0.25, "support_s": 2},
        "latency_policy": "offline exact-time oracle",
        "warm_up": "two rendered frames per held state",
        "timing_availability": "not_measured",
        "outcome": "started",
        "scenes": ["A", "B", "C"],
    }
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    try:
        metadata["gpu_driver_report"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        metadata["gpu_driver_report"] = None
    metadata["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT / "godot/scripts/reference_capture.gd",
            ROOT / "godot/scripts/reference_scene.gd",
            ROOT / "godot/shaders/reference_copy.glsl",
            ROOT / "godot/project.godot",
            ROOT / "analysis/temporal_reference.py",
            Path(__file__),
        ]
    }
    summaries = {}
    try:
        for scene in metadata["scenes"]:
            folder = out / scene
            folder.mkdir()
            job = {
                "scene": scene,
                "width": 320,
                "height": 180,
                "frames": [
                    {
                        "time": float(t),
                        "rational_time": str(t),
                        "path": str(folder / f"{i:03d}"),
                    }
                    for i, t in enumerate(requested)
                ],
            }
            job_file = folder / "job.json"
            job_file.write_text(json.dumps(job, indent=2))
            cmd = [
                str(engine),
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
                str(job_file),
            ]
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=120)
            (folder / "stdout.log").write_bytes(result.stdout)
            (folder / "stderr.log").write_bytes(result.stderr)
            log = (folder / "engine.log").read_text(encoding="utf-8", errors="replace")
            if (
                result.returncode
                or "REFERENCE_CAPTURE_COMPLETE" not in log
                or "SCRIPT ERROR" in log
            ):
                raise RuntimeError(f"Scene {scene} failed; inspect its logs")
            images = []
            for i, t in enumerate(requested):
                record = json.loads((folder / f"{i:03d}.json").read_text())
                if not record["rgba16f"] or (record["width"], record["height"]) != (
                    320,
                    180,
                ):
                    raise RuntimeError("Unexpected source format/size")
                data = (
                    np.fromfile(folder / f"{i:03d}.bin", dtype="<f2")
                    .reshape(180, 320, 4)
                    .astype(np.float64)
                )
                if not np.isfinite(data).all():
                    raise RuntimeError("Nonfinite capture")
                images.append(data[..., :3])
            differences = [
                float(np.max(np.abs(images[len(times) + j] - images[times.index(t)])))
                for j, t in enumerate(requested[len(times) :])
            ]
            if max(differences) > 0:
                raise RuntimeError(f"Non-deterministic capture {scene}: {differences}")
            a = math.exp(-math.log(2) * 0.05 / 0.25)
            history_times = [Fraction(k, 20) for k in range(21)]
            history = [images[times.index(t)] for t in history_times]
            sampled = recursive_oracle(history, [a] * 20)
            live = live_composite(
                images[times.index(Fraction(1))],
                history,
                history_times,
                Fraction(1),
                1 - a,
            )
            np.save(folder / "oracle_recursive_t1.npy", sampled)
            np.save(folder / "oracle_live_t1.npy", live)
            # Simple PNG previews are derivative artifacts, never oracle inputs.
            from PIL import Image

            for name, data in [
                ("scene_t0", images[0]),
                ("scene_t1", history[-1]),
                ("oracle_recursive_t1", sampled),
                ("oracle_live_t1", live),
            ]:
                srgb = np.where(
                    data <= 0.0031308,
                    12.92 * data,
                    1.055 * np.maximum(data, 0) ** (1 / 2.4) - 0.055,
                )
                Image.fromarray(np.uint8(np.clip(srgb, 0, 1) * 255 + 0.5)).save(
                    folder / (name + ".png")
                )
            summaries[scene] = {
                "revisited_max_abs_errors": differences,
                "captures": len(requested),
                "source_max": float(np.max(images)),
                "oracle_dtype": str(sampled.dtype),
                "engine_errors_present": "ERROR:" in log,
            }
            if scene == "A":
                centroids = []
                for data in (images[0], history[-1]):
                    mask = data[..., 0] > 0.5
                    centroids.append(float(np.where(mask)[1].mean()))
                expected_displacement = 2 * 180 / 6
                error = abs(centroids[1] - centroids[0] - expected_displacement)
                summaries[scene].update(
                    centroids_x=centroids,
                    expected_displacement_px=expected_displacement,
                    displacement_error_px=error,
                )
                if error > 0.51:
                    raise RuntimeError("Analytical centroid displacement mismatch")
                # Pixel-center rectangle oracle; boundaries chosen away from pixel centers.
                yy, xx = np.mgrid[:180, :320]
                x_world = (xx + 0.5 - 160) / 30
                y_world = (90 - yy - 0.5) / 30
                mask = (np.abs(x_world + 2) < 0.3) & (np.abs(y_world) < 0.4)
                mismatch = int(np.count_nonzero(mask != (images[0][..., 0] > 0.5)))
                summaries[scene]["analytical_mask_mismatch_pixels"] = mismatch
                if mismatch:
                    raise RuntimeError("Analytical rectangle mask mismatch")
        metadata["outcome"] = "passed"
    except Exception as exc:
        metadata["outcome"] = "failed"
        metadata["error"] = str(exc)
    metadata["summaries"] = summaries
    metadata["configuration_hash"] = hashlib.sha256(
        json.dumps(
            {
                "times": [str(t) for t in requested],
                "resolution": [320, 180],
                "scenes": ["A", "B", "C"],
                "half_life": 0.25,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
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
                "path": str(out),
                "outcome": metadata["outcome"],
                "error": metadata.get("error"),
                "summaries": summaries,
            },
            indent=2,
        )
    )
    return 0 if metadata["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
