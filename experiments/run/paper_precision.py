"""Stable-engine GPU dense-format accuracy at exact native target timestamps."""

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
from analysis.artifact_paths import recorded_path

ENGINE = ROOT / "experiments/tools/Godot_v4.7.2-stable_win64.exe"


def main():
    source = Path(sys.argv[1]).resolve()
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-paper-precision-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    meta = {
        "outcome": "started",
        "engine": subprocess.check_output(
            [str(ENGINE), "--version"], text=True
        ).strip(),
        "source": source.name,
        "git_commit": None,
        "scope": "GPU precision replay; exact finite float64 target of independently rendered inputs; no timed readbacks",
        "results": [],
    }
    cases = []
    streams = []
    manifest = json.loads((source / "sha256.json").read_text())
    for entry in json.loads((source / "metadata.json").read_text())["sources"]:
        w, h = entry["width"], entry["height"]
        samples = json.loads((source / entry["folder"] / "samples.json").read_text())[
            "samples"
        ]
        for row in samples:
            p = recorded_path(row["path"])
            assert (
                hashlib.sha256(p.read_bytes()).hexdigest()
                == manifest[p.relative_to(source).as_posix()]
            )
            row["path"] = str(p)
        for fmt in ["rgba32f", "rgba16f", "rgba8"]:
            name = entry["folder"] + "_" + fmt
            folder = out / name
            folder.mkdir()
            case = {
                "name": name,
                "dense": True,
                "storage_format": fmt,
                "width": w,
                "height": h,
                "history": 10,
                "alpha": 2 ** (-0.05 / 0.25),
                "frames": [
                    {"input": f["path"], "output": str(folder / f"{i}.bin")}
                    for i, f in enumerate(samples)
                ],
            }
            cases.append(case)
            streams.append(entry)
    path = out / "job.json"
    path.write_text(json.dumps({"cases": cases}, indent=2))
    try:
        cmd = [
            str(ENGINE),
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
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0 and "M6_REPLAY_COMPLETE" in log and "ERROR:" not in log
        ), log[:2000]
        for case, entry in zip(cases, streams):
            w, h = case["width"], case["height"]
            images = [
                np.fromfile(f["input"], "<f4").reshape(h, w, 4).astype(float)
                for f in case["frames"]
            ]
            region = np.zeros((h, w), bool)
            for im in images:
                region |= np.max(abs(im[..., :3] - images[0][..., :3]), axis=2) > 1e-6
            full = []
            motion = []
            maxerror = 0
            formaterror = 0
            for i, f in enumerate(case["frames"]):
                subset = images[max(0, i - 9) : i + 1]
                weights = case["alpha"] ** np.arange(len(subset) - 1, -1, -1)
                weights /= weights.sum()
                truth = sum(a * b for a, b in zip(weights, subset))
                if case["storage_format"] == "rgba16f":
                    quant = [im.astype(np.float16).astype(float) for im in subset]
                elif case["storage_format"] == "rgba8":
                    quant = [
                        np.floor(np.clip(im, 0, 1) * 255 + 0.5) / 255 for im in subset
                    ]
                else:
                    quant = subset
                intended = sum(a * b for a, b in zip(weights, quant))
                gpu = np.fromfile(f["output"], "<f4").reshape(h, w, 4)
                assert np.isfinite(gpu).all()
                formaterror = max(formaterror, float(abs(gpu - intended).max()))
                maxerror = max(maxerror, float(abs(gpu - truth).max()))
                err = (gpu[..., :3] - truth[..., :3]) ** 2
                full.append(float(err.mean()))
                motion.append(float(err[region].mean()))
            assert formaterror < 3e-6, formaterror
            row = {
                **entry,
                "format": case["storage_format"],
                "history_bytes": w
                * h
                * 10
                * {"rgba32f": 16, "rgba16f": 8, "rgba8": 4}[case["storage_format"]],
                "full_mse": float(np.mean(full)),
                "motion_mse": float(np.mean(motion)),
                "max_abs": maxerror,
                "max_format_reference_abs": formaterror,
                "per_capture_mse": full,
                "per_capture_motion_mse": motion,
                "motion_pixels": int(region.sum()),
                "psnr_full_range1": (
                    float(-10 * np.log10(np.mean(full)))
                    if np.mean(full) > 0
                    else "infinity"
                ),
            }
            meta["results"].append(row)
            print(
                entry["folder"], case["storage_format"], row["motion_mse"], flush=True
            )
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "godot/scripts/dense_finite_history.gd",
            ROOT / "godot/shaders/dense_finite.glsl",
            ROOT / "godot/scripts/m6_replay.gd",
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
