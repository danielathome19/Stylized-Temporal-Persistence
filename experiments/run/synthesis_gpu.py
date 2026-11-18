"""Independent numerical check for bounded GPU temporal-profile synthesis."""

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
from analysis.synthesis_history_reference import SynthesisHistory


def main():
    source = Path(sys.argv[1]).resolve()
    manifest = json.loads((source / "sha256.json").read_text())
    packed = "--packed" in sys.argv[2:]
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-synthesis-gpu-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    cases = []
    oracles = []
    rng = np.random.default_rng(582)
    for scene in ["odd_reset"] + json.loads((source / "metadata.json").read_text()).get(
        "scenes", ["camera", "rotation", "occlusion", "link"]
    ):
        if scene == "odd_reset":
            w, h, tile, H, cap = 17, 9, 8, 4, 8
            inputs = []
            for i in range(20):
                im = rng.random((h, w, 4)).astype("<f4")
                im[..., 3] = 1
                path = out / f"input_{i}.bin"
                im.tofile(path)
                inputs.append(path)
        else:
            w, h, tile, H, cap = 160, 90, 16, 10, 75
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
        for policy in ["horizon", "independent"]:
            folder = out / f"{scene}_{policy}"
            folder.mkdir()
            case = {
                "name": folder.name,
                "synthesis": True,
                "width": w,
                "height": h,
                "tile": tile,
                "history": H,
                "pool_slots": cap,
                "alpha": 2 ** (-0.05 / 0.25),
                "synthesis_policy": policy,
                "metadata": str(folder / "metadata.bin"),
                "frames": [],
            }
            case["packed"] = packed
            model = SynthesisHistory(
                w, h, tile, H, cap, case["alpha"], policy, packed=packed
            )
            truth = []
            for i, p in enumerate(inputs):
                reset = scene == "odd_reset" and i == 10
                if reset:
                    model = SynthesisHistory(
                        w, h, tile, H, cap, case["alpha"], policy, packed=packed
                    )
                im = np.fromfile(p, "<f4").reshape(h, w, 4)
                expected, stats = model.capture(im)
                truth.append((expected, stats))
                case["frames"].append(
                    {
                        "input": str(p),
                        "output": str(folder / f"{i}.bin"),
                        "state": str(folder / f"{i}_state.bin"),
                        "reset": reset,
                    }
                )
            cases.append(case)
            oracles.append(truth)
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
        "run_id": out.name,
        "source": source.name,
        "stage": "GPU_synthesis_correctness",
        "git_commit": None,
        "engine": "4.8.beta3.c51ff06e8",
        "backend": "Forward+ Vulkan",
        "seed": 582,
        "timing_available": False,
        "outcome": "started",
        "results": [],
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=150)
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0 and "M6_REPLAY_COMPLETE" in log and "ERROR:" not in log
        ), log[:2000]
        for case, truth in zip(cases, oracles):
            w, h, H, cap = [
                case[k] for k in ["width", "height", "history", "pool_slots"]
            ]
            n = ((w + case["tile"] - 1) // case["tile"]) * (
                (h + case["tile"] - 1) // case["tile"]
            )
            offset = 3 * n * H + 9 * n + cap
            maximum = 0
            per = []
            for f, (expected, stats) in zip(case["frames"], truth):
                actual = np.fromfile(f["output"], "<f4").reshape(h, w, 4)
                assert np.isfinite(actual).all()
                error = float(abs(actual - expected).max())
                maximum = max(maximum, error)
                per.append(error)
                state = np.fromfile(f["state"], "<u4")
                c = state[offset : offset + 16]
                assert c[5] == 0 and c[0] <= cap
                owners = {}
                for t in range(n):
                    for physical in state[t * H : (t + 1) * H]:
                        if physical == 0xFFFFFFFF:
                            continue
                        assert physical < cap and (
                            physical not in owners or owners[physical] == t
                        )
                        owners[physical] = t
                assert len(owners) == c[0]
            payload = (
                cap * case["tile"] ** 2 * (4 if packed else 16)
                + 4 * (3 * n * H + 9 * n + cap + 16 + 2 * n * H)
                + 4 * (n * H * (H + 1) + 3 * n + n * H * H + n)
            )
            meta["results"].append(
                {
                    "case": case["name"],
                    "max_abs": maximum,
                    "per_capture_max_abs": per,
                    "payload_bytes": payload,
                }
            )
            print(case["name"], maximum, payload, flush=True)
        assert (
            max(r["max_abs"] for r in meta["results"]) < 3e-5
        ), "GPU/reference discrepancy: inspect per-capture errors"
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "analysis/synthesis_history_reference.py",
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
