"""Capture new asset-free native3D scenes without changing the frozen merge rule."""

import datetime
import hashlib
import json
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
            + "-synthesis-holdout-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    meta = {
        "run_id": out.name,
        "scenes": ["crossing", "camera_turn"],
        "seed": 21717,
        "resolution": [160, 90],
        "format": "RGBA32F",
        "source": "native3D M5 sample capture",
        "history_hz": 20,
        "presentation": "180 nominal60Hz frames, deterministic jitter,120ms and90ms permanent timestamp jumps",
        "selection": "first new scene test after synthesis development; rule frozen before capture; not independent study preregistration",
        "outcome": "started",
    }
    try:
        for scene in meta["scenes"]:
            folder = out / f"input_{scene}"
            folder.mkdir()
            job = {"scene": scene, "width": 160, "height": 90, "folder": str(folder)}
            path = folder / "job.json"
            path.write_text(json.dumps(job, indent=2))
            cmd = [
                str(ROOT / "Godot_v4.8-beta3_win64.exe"),
                "--path",
                str(ROOT / "godot"),
                "res://scenes/synthesis_holdout.tscn",
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
            proc = subprocess.run(cmd, capture_output=True, timeout=120)
            (folder / "stdout.log").write_bytes(proc.stdout)
            (folder / "stderr.log").write_bytes(proc.stderr)
            log = (folder / "engine.log").read_text(errors="replace")
            assert (
                proc.returncode == 0
                and "M5_SAMPLE_CAPTURE_COMPLETE" in log
                and "ERROR:" not in log
            ), log[:2000]
            samples = json.loads((folder / "samples.json").read_text())["samples"]
            assert all(f["width"] == 160 and f["height"] == 90 for f in samples)
            assert [f["target_us"] for f in samples] == list(
                range(0, len(samples) * 50000, 50000)
            )
            print(scene, len(samples), flush=True)
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "godot/scripts/synthesis_holdout_scene.gd",
            ROOT / "godot/scripts/comparison_scene.gd",
            ROOT / "godot/scripts/m5_sample_capture.gd",
            ROOT / "godot/scripts/synthesis_history.gd",
            *ROOT.glob("godot/shaders/synthesis*.glsl"),
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
