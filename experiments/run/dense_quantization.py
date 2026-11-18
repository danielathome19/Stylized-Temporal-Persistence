"""Accuracy-only fixed-rate dense RGB quantization control; no GPU timing claim."""

import datetime
import hashlib
import json
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    sources = [Path(s).resolve() for s in sys.argv[1:]]
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-dense-quantization-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    meta = {
        "outcome": "started",
        "scope": "CPU accuracy of rounded uniform RGB codes; alpha excluded, packed payload estimates, no GPU timings or residency",
        "results": [],
        "sources": [s.name for s in sources],
    }
    for source in sources:
        manifest = json.loads((source / "sha256.json").read_text())
        for scene in json.loads((source / "metadata.json").read_text()).get(
            "scenes", ["camera", "rotation", "occlusion", "link"]
        ):
            images = []
            for row in json.loads((source / f"input_{scene}/samples.json").read_text())[
                "samples"
            ]:
                p = Path(row["path"])
                assert (
                    hashlib.sha256(p.read_bytes()).hexdigest()
                    == manifest[p.relative_to(source).as_posix()]
                )
                images.append(
                    np.fromfile(p, "<f4")
                    .reshape(row.get("height", 90), row.get("width", 160), 4)[..., :3]
                    .astype(float)
                )
            images = np.array(images)
            region = np.any(np.max(abs(images - images[0]), axis=0) > 1e-6, axis=2)
            for bits in [(3, 3, 2), (5, 6, 5), (8, 8, 8)]:
                levels = 2 ** np.array(bits) - 1
                quant = np.rint(np.clip(images, 0, 1) * levels) / levels
                full = []
                motion = []
                for i in range(len(images)):
                    start = max(0, i - 9)
                    w = 2 ** (-0.05 / 0.25 * np.arange(i - start, -1, -1))
                    w /= w.sum()
                    error = np.einsum(
                        "n,nhwc->hwc", w, quant[start : i + 1] - images[start : i + 1]
                    )
                    full.append(float(np.mean(error**2)))
                    motion.append(float(np.mean(error[region] ** 2)))
                result = {
                    "scene": scene,
                    "bits": bits,
                    "history_payload_bytes": 160 * 90 * 10 * sum(bits) // 8,
                    "full_mse": float(np.mean(full)),
                    "motion_region_mse": float(np.mean(motion)),
                    "per_capture_region_mse": motion,
                    "source_range": [float(images.min()), float(images.max())],
                }
                meta["results"].append(result)
                print(scene, bits, result["motion_region_mse"], flush=True)
    meta.update(
        outcome="passed",
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy=np.__version__,
    )
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
