"""Smoke the default public demo without requiring private assets."""

import subprocess
import datetime
import uuid
import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-shared-demo-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    cmd = [
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
        str(out / "engine.log"),
        "--",
        str(out / "preview.png"),
    ]
    if "--synthesis" in sys.argv:
        cmd.insert(3, "res://scenes/synthesis_live_demo.tscn")
    meta = {
        "stage": "public_default_demo_smoke",
        "outcome": "started",
        "timing_available": False,
    }
    meta["scene"] = "synthesis_live_demo" if "--synthesis" in sys.argv else "default"
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        meta["outcome"] = (
            "passed"
            if proc.returncode == 0
            and "LIVE_M6_COMPLETE" in log
            and "ERROR:" not in log
            else "failed"
        )
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT / "godot/project.godot",
            ROOT / "godot/scripts/m6_live_demo.gd",
            ROOT / "godot/scripts/shared_history.gd",
            *ROOT.glob("godot/shaders/shared*.glsl"),
        ]
    }
    for p in [
        Path(__file__),
        ROOT / "godot/scripts/m6_motion_effect.gd",
        ROOT / "godot/scripts/synthesis_live_demo.gd",
        ROOT / "godot/scripts/synthesis_history.gd",
        *ROOT.glob("godot/shaders/synthesis*.glsl"),
    ]:
        meta["implementation_hashes"][p.relative_to(ROOT).as_posix()] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "sha256.json").write_text(
        json.dumps(
            {
                str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in out.rglob("*")
                if f.is_file()
            },
            indent=2,
        )
    )
    print(out)
    print(meta["outcome"])
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
