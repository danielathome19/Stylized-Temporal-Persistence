"""Global allocator correctness and pressure exercise; no throughput claims."""

import datetime
import hashlib
import json
import subprocess
import sys
import uuid
import platform
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.shared_history_reference import SharedReference


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-shared-history-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    cases = []
    oracles = []
    rng = np.random.default_rng(819)
    for name, capacity in [("borrowing", 9), ("global_pressure", 8), ("reset", 8)]:
        folder = out / name
        folder.mkdir()
        case = {
            "name": name,
            "shared": True,
            "width": 17,
            "height": 9,
            "tile": 8,
            "history": 4,
            "pool_slots": capacity,
            "alpha": 0.8,
            "frames": [],
            "metadata": str(folder / "metadata.bin"),
        }
        ref = SharedReference(17, 9, 8, 4, capacity, 0.8)
        expected = []
        for i in range(10):
            im = np.zeros((9, 17, 4), np.float32)
            im[..., 3] = 1
            for t, sl in enumerate(ref.slices):
                im[sl][..., :3] = (
                    rng.random(3).astype(np.float32)
                    if name != "borrowing" or t == 0
                    else np.array([0.1, 0.2, 0.3], np.float32)
                )
            reset = name == "reset" and i == 5
            if reset:
                ref.reset()
            expected.append(ref.capture(im))
            path = folder / (str(i) + "_input.bin")
            im.tofile(path)
            case["frames"].append(
                {
                    "input": str(path),
                    "output": str(folder / (str(i) + "_output.bin")),
                    "diagnostics": str(folder / (str(i) + "_diagnostics.bin")),
                    "state": str(folder / (str(i) + "_state.bin")),
                    "reset": reset,
                }
            )
        cases.append(case)
        oracles.append(expected)
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
        "stage": "shared_allocator_correctness",
        "git_commit": None,
        "engine": "4.8.beta3.official.c51ff06e8",
        "os": platform.platform(),
        "backend": "Vulkan Forward+",
        "resolution": [17, 9],
        "format": "RGBA32F",
        "seed": 819,
        "method": "global lifetime-aware merge",
        "timing_available": False,
        "configuration_hash": hashlib.sha256(job.read_bytes()).hexdigest(),
        "outcome": "started",
        "results": [],
    }
    try:
        log = (out / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0 and "M6_REPLAY_COMPLETE" in log and "ERROR:" not in log
        )
        for case, expected in zip(cases, oracles):
            error = 0
            rows = []
            n = 6
            h = 4
            capacity = case["pool_slots"]
            control = 3 * n * h + 9 * n + capacity
            for frame, (truth, stats) in zip(case["frames"], expected):
                actual = np.fromfile(frame["output"], "<f4").reshape(9, 17, 4)
                error = max(error, float(abs(actual - truth).max()))
                m = np.fromfile(frame["state"], "<u4")
                c = m[control : control + 16]
                assert c[5] == 0, c.tolist()
                assert (
                    c[0] <= capacity
                    and c[0] == stats["used"]
                    and c[2] == stats["merges"]
                    and c[6] == stats["requested"]
                )
                owners = {}
                for t in range(n):
                    for physical in m[t * h : (t + 1) * h]:
                        if physical == 0xFFFFFFFF:
                            continue
                        assert physical < capacity
                        assert int(physical) not in owners or owners[int(physical)] == t
                        owners[int(physical)] = t
                assert len(owners) == c[0]
                rows.append(
                    {
                        "used": int(c[0]),
                        "merges": int(c[2]),
                        "stores": int(c[3]),
                        "versions_per_tile": stats["versions_per_tile"],
                    }
                )
            assert error < 3e-6, (case["name"], error)
            meta["results"].append(
                {"case": case["name"], "max_abs_error": error, "captures": rows}
            )
        assert (
            max(r["versions_per_tile"][0] for r in meta["results"][0]["captures"]) == 4
        )
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "analysis/shared_history_reference.py",
            ROOT / "godot/scripts/shared_history.gd",
            *ROOT.glob("godot/shaders/shared*.glsl"),
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
