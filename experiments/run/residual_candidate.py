"""CPU analytical fidelity experiment; no GPU performance or allocation claims."""

import datetime
import hashlib
import json
import platform
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.residual_history_reference import ResidualHistory


def payload(slots):
    return (
        slots * 4096 + 4 * (3 * 60 * 10 + 9 * 60 + slots + 16) + 4 * 60 * (100 + 10 + 3)
    )


def main():
    source = Path(sys.argv[1]).resolve()
    manifest = json.loads((source / "sha256.json").read_text())
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-residual-candidate-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dimensions = 3 * grid * grid
    extra = (60 * 10 + 75) * dimensions * 4
    budget = payload(75) + extra
    matched = max(s for s in range(60, 601) if payload(s) <= budget)
    meta = {
        "run_id": out.name,
        "source": source.name,
        "stage": "CPU_analytical_candidate_not_GPU",
        "outcome": "started",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": None,
        "resolution": [160, 90],
        "tile": 16,
        "history": 10,
        "grid": grid,
        "projection_dimensions": dimensions,
        "projection_selection": "4x4initial;1x1minimalstatefollowup after equal-budget failure",
        "budget_basis": "prospective float32 GPU payload; prototype arithmetic float64; not measured allocation",
        "candidate_payload_bytes": budget,
        "extra_projection_bytes": extra,
        "equal_budget_baseline_slots": matched,
        "baseline_payload_bytes": payload(matched),
        "timing_available": False,
        "results": [],
    }
    try:
        for scene in ["camera", "rotation", "occlusion", "link"]:
            stream = json.loads((source / f"input_{scene}/samples.json").read_text())[
                "samples"
            ]
            images = []
            for row in stream:
                p = Path(row["path"])
                assert (
                    hashlib.sha256(p.read_bytes()).hexdigest()
                    == manifest[p.relative_to(source).as_posix()]
                )
                images.append(np.fromfile(p, "<f4").reshape(90, 160, 4).astype(float))
            region = np.any(
                np.max(abs(np.stack(images)[..., :3] - images[0][..., :3]), axis=0)
                > 1e-6,
                axis=2,
            )
            truths = []
            for i in range(len(images)):
                subset = images[max(0, i - 9) : i + 1]
                w = (2 ** (-0.05 / 0.25)) ** np.arange(len(subset) - 1, -1, -1)
                w /= w.sum()
                truths.append(sum(a * b for a, b in zip(w, subset)))
            for policy, slots, correct in [
                ("lifetime75", 75, False),
                ("residual75", 75, True),
                ("lifetime_equal_bytes", matched, False),
                ("lifetime_next_slot", matched + 1, False),
            ]:
                model = ResidualHistory(
                    160, 90, capacity=slots, correct=correct, grid=grid
                )
                folder = out / f"{scene}_{policy}"
                folder.mkdir()
                full = []
                motion = []
                counters = []
                for i, (im, truth) in enumerate(zip(images, truths)):
                    result, stats = model.capture(im)
                    assert np.isfinite(result).all()
                    diff = result[..., :3] - truth[..., :3]
                    full.append(float(np.mean(diff**2)))
                    motion.append(float(np.mean(diff[region] ** 2)))
                    counters.append(stats)
                    result.astype("<f4").tofile(folder / f"{i}.bin")
                row = {
                    "scene": scene,
                    "policy": policy,
                    "slots": slots,
                    "full_mse": float(np.mean(full)),
                    "motion_region_mse": float(np.mean(motion)),
                    "per_capture_full_mse": full,
                    "per_capture_region_mse": motion,
                    "counters": counters,
                }
                meta["results"].append(row)
                print(scene, policy, row["motion_region_mse"], flush=True)
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "analysis/residual_history_reference.py"]
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
    print(out, meta["outcome"], meta.get("error"), flush=True)
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
