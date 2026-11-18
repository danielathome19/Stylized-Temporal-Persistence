"""Check actual GPU pixel-event layout and independent CPU fidelity, accuracy only."""

import datetime
import hashlib
import json
import subprocess
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.innovation_history_reference import InnovationHistory
from analysis.innovation_memory import allocation, capacity_for_budget


def main():
    source = Path(sys.argv[1]).resolve()
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-innovation-gpu-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    manifest = json.loads((source / "sha256.json").read_text())
    cases = []
    references = []
    scenes = ["odd_reset"] + json.loads((source / "metadata.json").read_text()).get(
        "scenes", ["camera", "rotation", "occlusion", "link"]
    )
    for scene in scenes:
        if scene == "odd_reset":
            w, h, H, cap = 17, 9, 4, 31
            inputs = []
            rng = np.random.default_rng(305)
            for i in range(20):
                im = rng.random((h, w, 4)).astype("<f4")
                im[..., 3] = 1
                p = out / f"input_{i}.bin"
                im.tofile(p)
                inputs.append(p)
        else:
            w, h, H = 160, 90, 10
            cap = capacity_for_budget(w, h, 373084)
            inputs = [
                Path(f["path"])
                for f in json.loads(
                    (source / f"input_{scene}/samples.json").read_text()
                )["samples"]
            ]
            for p in inputs:
                assert (
                    hashlib.sha256(p.read_bytes()).hexdigest()
                    == manifest[p.relative_to(source).as_posix()]
                )
        folder = out / scene
        folder.mkdir()
        case = {
            "name": scene,
            "innovation": True,
            "width": w,
            "height": h,
            "history": H,
            "pool_slots": cap,
            "alpha": 2 ** (-0.05 / 0.25),
            "metadata": str(folder / "metadata.bin"),
            "frames": [],
        }
        model = InnovationHistory(w, h, tile=1, history=H, capacity=cap)
        expected = []
        images = []
        for i, p in enumerate(inputs):
            reset = scene == "odd_reset" and i == 10
            if reset:
                model = InnovationHistory(w, h, tile=1, history=H, capacity=cap)
            im = np.fromfile(p, "<f4").reshape(h, w, 4)
            actual, stats = model.capture(im)
            expected.append((actual, stats))
            images.append(im.astype(float))
            case["frames"].append(
                {
                    "input": str(p),
                    "output": str(folder / f"{i}.bin"),
                    "diagnostics": str(folder / f"{i}_workspace.bin"),
                    "state": str(folder / f"{i}_state.bin"),
                    "reset": reset,
                }
            )
        cases.append(case)
        references.append((expected, images))
    path = out / "job.json"
    path.write_text(json.dumps({"cases": cases}, indent=2))
    cmd = [
        str(ROOT / "Godot_v4.8-beta3_win64.exe"),
        "--path",
        str(ROOT / "godot"),
        "--script",
        "res://scripts/m6_replay.gd",
        "--rendering-method",
        "forward_plus",
        "--rendering-driver",
        "vulkan",
        "--audio-driver",
        "Dummy",
        "--log-file",
        str(out / "engine.log"),
        "--",
        str(path),
    ]
    meta = {
        "outcome": "started",
        "scope": "GPU correctness and requested-resource accounting; readback outside any timing experiment; no GPU speed/residency claim",
        "source": source.name,
        "git_commit": None,
        "engine": "4.8.beta3.c51ff06e8",
        "backend": "Forward+ Vulkan",
        "results": [],
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0 and "M6_REPLAY_COMPLETE" in log and "ERROR:" not in log
        ), log[:3000]
        for case, (expected, images) in zip(cases, references):
            w, h, H, cap = [
                case[k] for k in ["width", "height", "history", "pool_slots"]
            ]
            maximum = 0
            motion = []
            full = []
            counts = []
            region = np.any(
                np.max(abs(np.array(images)[..., :3] - images[0][..., :3]), axis=0)
                > 1e-6,
                axis=2,
            )
            for i, (frame, (truth, stats)) in enumerate(zip(case["frames"], expected)):
                gpu = np.fromfile(frame["output"], "<f4").reshape(h, w, 4)
                assert np.isfinite(gpu).all()
                maximum = max(maximum, float(abs(gpu - truth).max()))
                work = np.fromfile(frame["diagnostics"], "<u4")
                state = np.fromfile(frame["state"], "<u4")
                control = work[2 * cap + w * h : 2 * cap + w * h + 4]
                assert control[0] <= cap
                seen = set()
                for pixel in range(w * h):
                    slot = int(work[2 * cap + pixel])
                    chain = set()
                    while slot != 0xFFFFFFFF:
                        assert slot < cap and slot not in chain and slot not in seen
                        chain.add(slot)
                        seen.add(slot)
                        assert state[2 * slot] == pixel
                        slot = int(work[cap + slot])
                    assert len(chain) <= H - 1
                assert len(seen) == control[0]
                counts.append(control.tolist())
                if case["name"] != "odd_reset":
                    subset = images[max(0, i - H + 1) : i + 1]
                    weights = case["alpha"] ** np.arange(len(subset) - 1, -1, -1)
                    weights /= weights.sum()
                    oracle = sum(a * b for a, b in zip(weights, subset))
                    error = (gpu[..., :3] - oracle[..., :3]) ** 2
                    full.append(float(error.mean()))
                    motion.append(float(error[region].mean()))
            layout = allocation(w, h, cap)
            payload = sum(layout.values())
            assert f"SPARSE_PAYLOAD {case['name']} {payload}" in log
            row = {
                "scene": case["name"],
                "capacity": cap,
                "allocation_bytes": layout,
                "payload_bytes": payload,
                "common_output_bytes": 16 * w * h,
                "total_owned_resource_bytes": payload + 16 * w * h,
                "max_gpu_cpu_abs": maximum,
                "motion_region_mse": float(np.mean(motion)) if motion else None,
                "full_mse": float(np.mean(full)) if full else None,
                "counters": counts,
            }
            meta["results"].append(row)
            print(case["name"], maximum, row["motion_region_mse"], payload, flush=True)
        assert (
            max(r["max_gpu_cpu_abs"] for r in meta["results"]) < 3e-5
        ), "CPU/GPU selection or arithmetic mismatch"
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "analysis/innovation_history_reference.py",
            ROOT / "analysis/innovation_memory.py",
            ROOT / "godot/scripts/innovation_history.gd",
            ROOT / "godot/scripts/m6_replay.gd",
            *ROOT.glob("godot/shaders/innovation*.glsl"),
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
    print(out, meta["outcome"], meta.get("error"))
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
