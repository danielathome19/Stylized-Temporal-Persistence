"""Local-only private model smoke run, never a benchmark or asset distribution."""

import datetime
import hashlib
import json
import platform
import subprocess
import uuid
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-private-link-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    model = ROOT / "godot/assets/private/MM_Young_Link_rigged.glb"
    if not model.exists():
        model = ROOT / "godot/assets/private/MM_Young_Link.glb"
    if not model.exists():
        raise SystemExit("See godot/assets/private/ATTRIBUTION.md for local setup")
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
        "res://scenes/private_link_demo.tscn",
        "--",
        str(out),
        "M6",
    ]
    if "--lifecycle" in sys.argv:
        cmd.append("lifecycle")
    proc = subprocess.run(cmd, capture_output=True, timeout=90)
    (out / "stdout.log").write_bytes(proc.stdout)
    (out / "stderr.log").write_bytes(proc.stderr)
    log = (out / "engine.log").read_text(errors="replace")
    meta = {
        "run_id": out.name,
        "stage": "private_asset_live_smoke_not_benchmark",
        "outcome": (
            "passed"
            if proc.returncode == 0
            and "PRIVATE_LINK_COMPLETE" in log
            and "ERROR:" not in log
            else "failed"
        ),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "model_redistributable": False,
        "os": platform.platform(),
        "engine": "4.8.beta3.official.c51ff06e8",
        "backend": "Vulkan Forward+",
        "resolution": [640, 360],
        "method": "M5 reconstruction -> globally shared lifetime-aware M6 storage",
        "lifecycle_checks": "--lifecycle" in sys.argv,
        "target_schedule": "120 presentations: frame*16667+(frame%3)*1000 us; 20 Hz history",
        "animation": (
            "procedural limb_13/limb_16/limb_10 bone rotations plus rigid motion; no original game animation clips"
            if "rigged" in model.name
            else "rigid motion only"
        ),
        "timing_available": False,
        "implementation_hashes": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "godot/scripts/private_link_demo.gd",
                ROOT / "godot/scripts/m5_live_effect.gd",
                ROOT / "godot/scripts/m6_motion_effect.gd",
                ROOT / "godot/scripts/sparse_history.gd",
                ROOT / "godot/scripts/shared_history.gd",
                *ROOT.glob("godot/shaders/shared*.glsl"),
            ]
        },
    }
    if (out / "allocator.json").exists():
        meta["allocator"] = json.loads((out / "allocator.json").read_text())
        if (
            meta["allocator"]["allocation_errors"]
            or meta["allocator"]["peak_used"] > meta["allocator"]["capacity"]
        ):
            meta["outcome"] = "failed"
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
    print(log)
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
