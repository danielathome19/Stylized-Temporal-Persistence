"""Analytical fidelity screen of derivative-history storage; no GPU timings."""

import datetime
import hashlib
import json
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.innovation_history_reference import InnovationHistory


def main():
    pixel = "--pixel" in sys.argv
    sources = [s for s in sys.argv[1:] if s != "--pixel"]
    tile = 1 if pixel else 16
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-innovation-candidate-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    meta = {
        "outcome": "started",
        "stage": "CPU_analytical_not_GPU",
        "resolution": [160, 90],
        "tile": tile,
        "history": 10,
        "alpha": 2 ** (-0.05 / 0.25),
        "seed": 293,
        "sources": sources,
        "results": [],
        "scope": "frozen post-M5 compression only; signed RGBA32F payload projection; previous full image included; no GPU allocation/timing evidence",
        "git_commit": None,
    }
    for name in sources:
        source = Path(name).resolve()
        manifest = json.loads((source / "sha256.json").read_text())
        for scene in json.loads((source / "metadata.json").read_text()).get(
            "scenes", ["camera", "rotation", "occlusion", "link"]
        ):
            frames = []
            for row in json.loads((source / f"input_{scene}/samples.json").read_text())[
                "samples"
            ]:
                p = Path(row["path"])
                assert (
                    hashlib.sha256(p.read_bytes()).hexdigest()
                    == manifest[p.relative_to(source).as_posix()]
                )
                frames.append(np.fromfile(p, "<f4").reshape(90, 160, 4).astype(float))
            region = np.any(
                np.max(abs(np.stack(frames)[..., :3] - frames[0][..., :3]), axis=0)
                > 1e-6,
                axis=2,
            )
            for slots in ([5944] if pixel else [34, 75]):
                model = InnovationHistory(160, 90, tile=tile, capacity=slots)
                full = []
                motion = []
                stats = []
                for i, im in enumerate(frames):
                    actual, counters = model.capture(im)
                    subset = frames[max(0, i - 9) : i + 1]
                    w = meta["alpha"] ** np.arange(len(subset) - 1, -1, -1)
                    w /= w.sum()
                    error = (
                        actual[..., :3] - sum(a * b for a, b in zip(w, subset))[..., :3]
                    ) ** 2
                    full.append(float(error.mean()))
                    motion.append(float(error[region].mean()))
                    stats.append(counters)
                r = {
                    "scene": scene,
                    "slots": slots,
                    "projected_payload_bytes": 160 * 90 * 16
                    + slots * (tile * tile * 16 + 8)
                    + 16,
                    "full_mse": float(np.mean(full)),
                    "motion_region_mse": float(np.mean(motion)),
                    "per_capture_region_mse": motion,
                    "counters": stats,
                }
                meta["results"].append(r)
                print(scene, slots, r["motion_region_mse"], flush=True)
    meta["outcome"] = "passed"
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "analysis/innovation_history_reference.py"]
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "sha256.json").write_text(
        json.dumps(
            {
                "metadata.json": hashlib.sha256(
                    (out / "metadata.json").read_bytes()
                ).hexdigest()
            },
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
