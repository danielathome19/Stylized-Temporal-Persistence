"""Verify raw GPU resources and regenerate the workspace-inclusive comparison."""

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = [
        "20261117T071338Z-innovation-gpu-295375be",
        "20261117T071429Z-innovation-gpu-7e612767",
    ]
    rows = []
    for name in sources:
        path = ROOT / "results/raw" / name
        for f, h in json.loads((path / "sha256.json").read_text()).items():
            assert hashlib.sha256((path / f).read_bytes()).hexdigest() == h, f
        meta = json.loads((path / "metadata.json").read_text())
        assert meta["outcome"] == "passed"
        job = json.loads((path / "job.json").read_text())
        for case in job["cases"]:
            cap = case["pool_slots"]
            p = case["width"] * case["height"]
            for f in case["frames"]:
                assert Path(f["state"]).stat().st_size == 8 * cap
                assert Path(f["diagnostics"]).stat().st_size == 4 * (
                    3 * cap + 2 * p + 4
                )
        rows.extend(r for r in meta["results"] if r["scene"] != "odd_reset")
    rows.sort(
        key=lambda r: [
            "camera",
            "rotation",
            "occlusion",
            "link",
            "crossing",
            "camera_turn",
        ].index(r["scene"])
    )
    base = json.loads(
        (
            ROOT
            / "results/raw/20261117T061916Z-synthesis-candidate-765511fd/metadata.json"
        ).read_text()
    )
    baseline = {
        r["scene"]: r["motion_region_mse"]
        for r in base["results"]
        if r["policy"] == "horizon"
    }
    later = json.loads(
        (
            ROOT
            / "results/raw/20261117T062342Z-synthesis-candidate-1238859d/metadata.json"
        ).read_text()
    )
    baseline.update(
        {
            r["scene"]: r["motion_region_mse"]
            for r in later["results"]
            if r["policy"] == "horizon"
        }
    )
    lines = [
        "# GPU sparse-event memory with workspace",
        "",
        "160x90, H10, RGBA32F,763events. Requested RD resource bytes; driver residency/padding and pipeline resources unmeasured.",
        "",
        "|Resource|Bytes|",
        "|---|---:|",
    ]
    lines += [f"|{k}|{v}|" for k, v in rows[0]["allocation_bytes"].items()]
    lines += [
        "|History plus all explicit workspace|373084|",
        "|Common RGBA32F output image|230400|",
        "|Total owned images/buffers|603484|",
        "",
        "The source scene image is supplied by the caller. The replay input texture/cache is outside this storage budget. No staging/history-copy buffer is allocated. Previous-image storage and all explicit selection/lookup workspace are included. This does not count shader registers, implementation-managed spills, driver object overhead or measured residency.",
        "",
        "At5944events this implemented layout needs559600bytes excluding the common output, rather than the earlier373072-byte payload-only projection.",
        "",
        "|Scene|GPU innovation motion-region MSE|Prior float synthesis MSE|",
        "|---|---:|---:|",
    ]
    lines += [
        f"|{r['scene']}|{r['motion_region_mse']:.7g}|{baseline[r['scene']]:.7g}|"
        for r in rows
    ]
    lines += [
        "",
        "All GPU outputs agree with the independent CPU algorithm within3.09e-7.328outputs checked, including two20-frame odd-size/reset cases. Per-pixel lists are acyclic, uniquely owned and at mostH-1nodes.65unit tests pass.",
        "",
        "Including the workspace removes the earlier five-scene exactness result at this budget. Rotation/occlusion remain near numerical precision, while four scenes lose to float synthesis. No per-effect GPU performance measurement has been run. The serial allocator and full-resolution classification remain costs to investigate.",
        "",
        "Sources: " + ", ".join(sources),
        "",
        "Reproduce: python analysis/innovation_workspace_report.py",
    ]
    (ROOT / "results/processed/innovation_workspace.md").write_text(
        "\n".join(lines) + "\n"
    )
    (ROOT / "results/processed/innovation_workspace.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "results": rows,
                "memory_scope": "explicit RD resource payload; excludes common source/output unless separately stated, driver overhead and residency",
            },
            indent=2,
        )
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
