"""Replay frozen streams with parallel preparation; compare every output and final map."""

import datetime
import hashlib
import json
import subprocess
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = Path(sys.argv[1]).resolve()
    for name, digest in json.loads((source / "sha256.json").read_text()).items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-prepare-equivalence-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    job = json.loads((source / "replay.json").read_text())
    original = json.loads((source / "replay.json").read_text())
    for case in job["cases"]:
        dest = out / case["name"]
        dest.mkdir()
        case["parallel_prepare"] = True
        case["metadata"] = str(dest / "state.bin")
        for i, frame in enumerate(case["frames"]):
            frame["output"] = str(dest / f"{i}.bin")
    path = out / "job.json"
    path.write_text(json.dumps(job, indent=2))
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
        "source": source.name,
        "run_id": out.name,
        "outcome": "started",
        "comparison": "frozen serial outputs versus parallel pair evaluation",
        "results": [],
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=150)
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0 and "M6_REPLAY_COMPLETE" in log and "ERROR:" not in log
        )
        for before, after in zip(original["cases"], job["cases"]):
            maximum = 0.0
            bitwise = True
            for a, b in zip(before["frames"], after["frames"]):
                x = np.fromfile(a["output"], "<f4")
                y = np.fromfile(b["output"], "<f4")
                assert np.isfinite(y).all()
                maximum = max(maximum, float(abs(x - y).max()))
                bitwise &= (
                    Path(a["output"]).read_bytes() == Path(b["output"]).read_bytes()
                )
            assert bitwise, (after["name"], maximum)
            state_equal = (
                True
                if after.get("dense")
                else Path(before["metadata"]).read_bytes()
                == Path(after["metadata"]).read_bytes()
            )
            assert state_equal, after["name"]
            meta["results"].append(
                {
                    "case": after["name"],
                    "outputs": len(after["frames"]),
                    "bitwise_equal": bitwise,
                    "final_metadata_equal": state_equal,
                    "max_abs": maximum,
                }
            )
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
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
    print(out, meta["outcome"], meta.get("error"))
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
