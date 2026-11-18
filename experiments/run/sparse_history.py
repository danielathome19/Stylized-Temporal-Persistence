"""Focused M6 allocation/resolve check; no performance comparisons."""

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


def oracle(frames, tile, k, h, epsilon, alpha):
    height, width, _ = frames[0].shape
    ny = (height + tile - 1) // tile
    nx = (width + tile - 1) // tile
    queues = [[] for _ in range(nx * ny)]
    versions = [{} for _ in queues]
    stats = np.zeros((len(queues), 3), np.uint32)
    outputs = []
    for image in frames:
        result = np.zeros_like(image, dtype=np.float64)
        for id in range(len(queues)):
            y, x = divmod(id, nx)
            sl = np.s_[
                y * tile : min((y + 1) * tile, height),
                x * tile : min((x + 1) * tile, width),
            ]
            q = queues[id]
            pool = versions[id]
            previous = q[-1] if q else None
            same = (
                previous is not None
                and np.max(abs(image[sl] - pool[previous])) <= epsilon
            )
            if len(q) == h:
                q.pop(0)
            if not same:
                match = next(
                    (
                        v
                        for v in sorted(set(q))
                        if np.max(abs(image[sl] - pool[v])) <= epsilon
                    ),
                    None,
                )
                if match is not None:
                    previous = match
                    same = True
            if same:
                chosen = previous
                stats[id, 1] += 1
            else:
                free = next((i for i in range(k) if i not in q), None)
                if free is None:
                    chosen = previous
                    stats[id, 2] += 1
                else:
                    chosen = free
                    pool[chosen] = image[sl].copy()
                    stats[id, 0] += 1
            q.append(chosen)
            weights = alpha ** np.arange(len(q) - 1, -1, -1, dtype=float)
            weights /= weights.sum()
            result[sl] = sum(w * pool[v] for w, v in zip(weights, q))
        outputs.append(result)
    return outputs, stats


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-sparse-history-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    alpha = 2 ** (-0.05 / 0.25)
    cases = []
    expected = []
    for name, k, epsilon in [
        ("sharing", 3, 0),
        ("overflow", 1, 0),
        ("tolerance", 3, 0.03),
    ]:
        frames = []
        for i in range(12):
            im = np.zeros((17, 33, 4), np.float32)
            im[..., 3] = 1
            if name == "tolerance":
                im[2:6, 2:6, 0] = i * 0.01
            else:
                im[2:6, 2:6, 0] = float((i // 2) % 2)
            frames.append(im)
        folder = out / name
        folder.mkdir()
        case = {
            "name": name,
            "width": 33,
            "height": 17,
            "tile": 16,
            "versions": k,
            "history": 6,
            "epsilon": epsilon,
            "alpha": alpha,
            "frames": [],
            "metadata": str(folder / "metadata.bin"),
        }
        for i, im in enumerate(frames):
            path = folder / (str(i) + "_input.bin")
            im.tofile(path)
            case["frames"].append(
                {"input": str(path), "output": str(folder / (str(i) + "_output.bin"))}
            )
        cases.append(case)
        expected.append(oracle(frames, 16, k, 6, epsilon, alpha))
    job = out / "job.json"
    job.write_text(json.dumps({"cases": cases}, indent=2))
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
        str(job),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=90)
    (out / "stdout.log").write_bytes(proc.stdout)
    (out / "stderr.log").write_bytes(proc.stderr)
    meta = {
        "run_id": out.name,
        "git_commit": None,
        "engine": "4.8.beta3.official.c51ff06e8",
        "os": platform.platform(),
        "backend": "Vulkan Forward+",
        "method": "M6 partitioned sparse history",
        "seed": 0,
        "resolution": [33, 17],
        "format": "RGBA32F",
        "warm_up": 0,
        "timing_available": False,
        "target_schedule": "12 exact scheduled samples at 20 Hz",
        "latency": "scheduled held composite",
        "configuration_hash": hashlib.sha256(job.read_bytes()).hexdigest(),
        "outcome": "started",
        "results": [],
    }
    try:
        log = (out / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0 and "M6_REPLAY_COMPLETE" in log and "ERROR:" not in log
        )
        for case, (images, stats) in zip(cases, expected):
            error = 0
            for frame, im in zip(case["frames"], images):
                actual = np.fromfile(frame["output"], "<f4").reshape(im.shape)
                error = max(error, float(abs(actual - im).max()))
            assert error < 2e-6, (case["name"], error)
            actual_stats = np.fromfile(case["metadata"], "<u4")[6 * 6 :].reshape(6, 3)
            np.testing.assert_array_equal(actual_stats, stats)
            meta["results"].append(
                {
                    "case": case["name"],
                    "max_abs_error": error,
                    "stored_reused_overflow": stats.sum(axis=0).tolist(),
                    "history_payload_bytes": 6 * case["versions"] * 256 * 16
                    + 6 * (6 + 3) * 4
                    + 6 * case["versions"] * 4,
                    "dense_history_payload_bytes": 33 * 17 * 6 * 16,
                }
            )
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "godot/scripts/sparse_history.gd",
            ROOT / "godot/scripts/m6_replay.gd",
            *ROOT.glob("godot/shaders/sparse*.glsl"),
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
    print(out)
    print(json.dumps(meta, indent=2))
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
