"""Exact native-resolution scene inputs, independently captured before temporal storage."""

import datetime
import hashlib
import json
import subprocess
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
            + "-paper-inputs-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    protocol = json.loads(
        (ROOT / "experiments/configs/paper_storage_protocol.json").read_text()
    )
    meta = {
        "outcome": "started",
        "sources": [],
        "scope": "native exact-time procedural3D scene inputs; noM5; unshaded linear pre-tonemap color; half-float scene expanded losslessly tofloat32",
        "engine": "4.8.beta3.c51ff06e8",
        "git_commit": None,
        "protocol": protocol,
    }
    try:
        for w, h in protocol["native_resolutions"]:
            for scene in protocol["native_scenes"]:
                folder = out / f"{scene}_{w}"
                folder.mkdir()
                p = folder / "job.json"
                p.write_text(
                    json.dumps(
                        {"width": w, "height": h, "scene": scene, "folder": str(folder)}
                    )
                )
                cmd = [
                    str(ROOT / "Godot_v4.8-beta3_win64.exe"),
                    "--path",
                    str(ROOT / "godot"),
                    "res://scenes/paper_capture.tscn",
                    "--rendering-method",
                    "forward_plus",
                    "--rendering-driver",
                    "vulkan",
                    "--audio-driver",
                    "Dummy",
                    "--log-file",
                    str(folder / "engine.log"),
                    "--",
                    str(p),
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=180)
                (folder / "stdout.log").write_bytes(proc.stdout)
                (folder / "stderr.log").write_bytes(proc.stderr)
                log = (folder / "engine.log").read_text(errors="replace")
                assert (
                    proc.returncode == 0
                    and "PAPER_CAPTURE_COMPLETE" in log
                    and "ERROR:" not in log
                ), log[:2500]
                samples = []
                for i in range(16):
                    native = folder / f"native_{i}.bin"
                    info = json.loads((folder / f"native_{i}.json").read_text())
                    assert info["width"] == w and info["height"] == h
                    im = np.fromfile(native, "<f2").reshape(h, w, 4).astype("<f4")
                    assert np.isfinite(im).all()
                    dest = folder / f"{i}.bin"
                    im.tofile(dest)
                    samples.append(
                        {
                            "path": str(dest),
                            "width": w,
                            "height": h,
                            "target_us": i * 50000,
                            "delivered_us": i * 50000,
                        }
                    )
                (folder / "samples.json").write_text(
                    json.dumps({"samples": samples}, indent=2)
                )
                meta["sources"].append(
                    {"scene": scene, "width": w, "height": h, "folder": folder.name}
                )
                print(scene, w, h, flush=True)
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "godot/scripts/paper_capture.gd",
            ROOT / "godot/scripts/reference_capture.gd",
            ROOT / "godot/scripts/comparison_scene.gd",
            ROOT / "experiments/configs/paper_storage_protocol.json",
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
