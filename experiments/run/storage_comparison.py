"""Matched GPU storage-fidelity pilot on frozen post-M5 sample streams."""

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


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-storage-comparison-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    meta = {
        "run_id": out.name,
        "stage": "matched_storage_fidelity_pilot",
        "outcome": "started",
        "git_commit": None,
        "engine": "4.8.beta3.official.c51ff06e8",
        "os": platform.platform(),
        "backend": "Vulkan Forward+",
        "resolution": [160, 90],
        "format": "RGBA32F",
        "seed": 0,
        "history": 10,
        "alpha": 2 ** (-0.05 / 0.25),
        "sampling": "20Hz targets from identical native M5 reconstructed streams; jittered presentations",
        "latency": "same captured right-endpoint delivery for every storage method",
        "timing_available": False,
        "results": [],
    }

    def run(scene, folder, args, script=False):
        cmd = (
            [
                str(ROOT / "Godot_v4.8-beta3_win64.exe"),
                "--path",
                str(ROOT / "godot"),
                "--rendering-method",
                "forward_plus",
                "--rendering-driver",
                "vulkan",
                "--audio-driver",
                "Dummy",
                "--log-file",
                str(folder / "engine.log"),
            ]
            + (["--script", scene] if script else [scene])
            + ["--"]
            + args
        )
        process = subprocess.run(cmd, capture_output=True, timeout=150)
        (folder / "stdout.log").write_bytes(process.stdout)
        (folder / "stderr.log").write_bytes(process.stderr)
        log = (folder / "engine.log").read_text(errors="replace")
        assert process.returncode == 0 and "ERROR:" not in log, log
        return log

    try:
        streams = {}
        cases = []
        for scene in ["rotation", "occlusion", "camera", "link"]:
            folder = out / ("input_" + scene)
            folder.mkdir()
            job = folder / "job.json"
            job.write_text(
                json.dumps(
                    {"width": 160, "height": 90, "scene": scene, "folder": str(folder)},
                    indent=2,
                )
            )
            assert "M5_SAMPLE_CAPTURE_COMPLETE" in run(
                "res://scenes/comparison_capture.tscn", folder, [str(job)]
            )
            stream = json.loads((folder / "samples.json").read_text())
            assert len(stream["samples"]) == 40
            assert all(
                f["width"] == 160 and f["height"] == 90 for f in stream["samples"]
            ), [(f["width"], f["height"]) for f in stream["samples"]]
            assert [f["target_us"] for f in stream["samples"]] == list(
                range(0, 2000000, 50000)
            )
            streams[scene] = stream
            for policy, ratio in [
                ("dense", 10),
                ("lifetime", 1.25),
                ("current", 1.25),
                ("distance", 1.25),
                ("lifetime", 2),
                ("current", 2),
                ("distance", 2),
            ]:
                target = out / (scene + "_" + policy + "_" + str(ratio))
                target.mkdir()
                case = {
                    "name": scene + "_" + policy + "_" + str(ratio),
                    "scene": scene,
                    "policy": policy,
                    "ratio": ratio,
                    "dense": policy == "dense",
                    "shared": policy != "dense",
                    "width": 160,
                    "height": 90,
                    "tile": 16,
                    "history": 10,
                    "slot_multiplier": ratio,
                    "score_policy": policy if policy != "dense" else "lifetime",
                    "alpha": meta["alpha"],
                    "metadata": str(target / "state.bin"),
                    "frames": [],
                }
                for i, frame in enumerate(stream["samples"]):
                    case["frames"].append(
                        {
                            "input": frame["path"],
                            "output": str(target / (str(i) + ".bin")),
                        }
                    )
                cases.append(case)
            print("Captured", scene, flush=True)
        job = out / "replay.json"
        job.write_text(json.dumps({"cases": cases}, indent=2))
        meta["configuration_hash"] = hashlib.sha256(job.read_bytes()).hexdigest()
        assert "M6_REPLAY_COMPLETE" in run(
            "res://scripts/m6_replay.gd", out, [str(job)], True
        )
        for scene, stream in streams.items():
            images = [
                np.fromfile(f["path"], "<f4").reshape(90, 160, 4).astype(float)
                for f in stream["samples"]
            ]
            region = np.any(
                np.max(np.abs(np.stack(images)[..., :3] - images[0][..., :3]), axis=0)
                > 1e-6,
                axis=2,
            )
            truth = []
            for i in range(len(images)):
                subset = images[max(0, i - 9) : i + 1]
                w = meta["alpha"] ** np.arange(len(subset) - 1, -1, -1)
                w /= w.sum()
                truth.append(sum(weight * im for weight, im in zip(w, subset)))
            for case in [c for c in cases if c["scene"] == scene]:
                errors = []
                region_errors = []
                maximum = 0
                previous = None
                temporal = []
                for i, frame in enumerate(case["frames"]):
                    actual = (
                        np.fromfile(frame["output"], "<f4")
                        .reshape(90, 160, 4)
                        .astype(float)
                    )
                    assert np.isfinite(actual).all()
                    diff = actual[..., :3] - truth[i][..., :3]
                    maximum = max(maximum, float(abs(actual - truth[i]).max()))
                    errors.append(float(np.mean(diff**2)))
                    region_errors.append(
                        float(np.mean(diff[region] ** 2)) if region.any() else 0.0
                    )
                    if previous is not None:
                        temporal.append(float(np.mean((diff - previous) ** 2)))
                    previous = diff
                n = 60
                h = 10
                capacity = int(np.ceil(n * case["ratio"]))
                if case["dense"]:
                    assert maximum < 3e-6, maximum
                    memory = 160 * 90 * 10 * 16
                    state = None
                else:
                    memory = (
                        capacity * 256 * 16
                        + 4 * (3 * n * h + 9 * n + capacity + 16)
                        + 4 * n * (h * h + h + 3)
                    )
                    values = np.fromfile(case["metadata"], "<u4")
                    offset = 3 * n * h + 9 * n + capacity
                    state = values[offset : offset + 16].tolist()
                    assert state[5] == 0 and state[1] <= capacity
                row = {
                    "scene": scene,
                    "policy": case["policy"],
                    "ratio": case["ratio"],
                    "history_payload_bytes": memory,
                    "full_mse": float(np.mean(errors)),
                    "motion_region_mse": float(np.mean(region_errors)),
                    "temporal_residual_mse": float(np.mean(temporal)),
                    "max_abs": maximum,
                    "per_capture_mse": errors,
                    "per_capture_region_mse": region_errors,
                    "motion_region_pixels": int(region.sum()),
                    "allocator_control": state,
                }
                meta["results"].append(row)
                print(
                    scene,
                    case["policy"],
                    case["ratio"],
                    row["motion_region_mse"],
                    flush=True,
                )
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "godot/scripts/comparison_scene.gd",
            ROOT / "godot/scripts/m5_sample_capture.gd",
            ROOT / "godot/scripts/shared_history.gd",
            ROOT / "godot/scripts/dense_finite_history.gd",
            *ROOT.glob("godot/shaders/shared*.glsl"),
            ROOT / "godot/shaders/dense_finite.glsl",
        ]
    }
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
    print(out, meta["outcome"], str(meta.get("error"))[:500], flush=True)
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
