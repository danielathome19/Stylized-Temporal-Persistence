"""Small stable-GPU format regression: HDR, odd sizes, reset and H1."""

import datetime
import hashlib
import json
import subprocess
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "experiments/tools/Godot_v4.7.2-stable_win64.exe"


def main():
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-paper-format-edges-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    cases = []
    truths = []
    rng = np.random.default_rng(914)
    for history in [1, 10]:
        images = [rng.uniform(-2, 8, (9, 17, 4)).astype("<f4") for _ in range(15)]
        images[0][:] = 0
        images[1][:] = 1
        for fmt in ["rgba32f", "rgba16f", "rgba8"]:
            folder = out / f"{history}_{fmt}"
            folder.mkdir()
            frames = []
            for i, im in enumerate(images):
                p = folder / f"input_{i}.bin"
                im.tofile(p)
                frames.append(
                    {
                        "input": str(p),
                        "output": str(folder / f"{i}.bin"),
                        "reset": i == 8,
                    }
                )
            cases.append(
                {
                    "name": folder.name,
                    "dense": True,
                    "storage_format": fmt,
                    "width": 17,
                    "height": 9,
                    "history": history,
                    "alpha": 2 ** (-0.05 / 0.25),
                    "frames": frames,
                }
            )
            truths.append(images)
    job = out / "job.json"
    job.write_text(json.dumps({"cases": cases}, indent=2))
    meta = {
        "outcome": "started",
        "scope": "stableGPU format handling; randomHDR,zeros/ones,oddsize,H1/reset and cross-case recreation",
        "seed": 914,
        "results": [],
    }
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
            str(job),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=90)
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0 and "M6_REPLAY_COMPLETE" in log and "ERROR:" not in log
        ), log[:2000]
        for case, images in zip(cases, truths):
            maximum = 0
            start = 0
            for i, f in enumerate(case["frames"]):
                if f["reset"]:
                    start = i
                subset = images[max(start, i - case["history"] + 1) : i + 1]
                if case["storage_format"] == "rgba16f":
                    subset = [im.astype(np.float16).astype(float) for im in subset]
                elif case["storage_format"] == "rgba8":
                    subset = [
                        np.floor(np.clip(im.astype(float), 0, 1) * 255 + 0.5) / 255
                        for im in subset
                    ]
                weights = case["alpha"] ** np.arange(len(subset) - 1, -1, -1)
                weights /= weights.sum()
                oracle = sum(a * b for a, b in zip(weights, subset))
                actual = np.fromfile(f["output"], "<f4").reshape(9, 17, 4)
                maximum = max(maximum, float(abs(actual - oracle).max()))
            assert maximum < 5e-6
            meta["results"].append({"case": case["name"], "max_abs": maximum})
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["engine"] = subprocess.check_output(
        [str(ENGINE), "--version"], text=True
    ).strip()
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "godot/scripts/dense_finite_history.gd",
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
    print(out, meta["outcome"], meta.get("error"))
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
