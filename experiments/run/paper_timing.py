"""Native-resolution stable-engine precision controls and sparse feasibility gate."""

import datetime
import hashlib
import json
import platform
import random
import subprocess
import sys
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.timing_report import validate_samples
from analysis.innovation_memory import capacity_for_budget
from analysis.artifact_paths import recorded_path

ENGINE = ROOT / "experiments/tools/Godot_v4.7.2-stable_win64.exe"


def main():
    source = Path(sys.argv[1]).resolve()
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-paper-timing-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()
    protocol = json.loads(
        (ROOT / "experiments/configs/paper_storage_protocol.json").read_text()
    )
    inputs = json.loads((source / "metadata.json").read_text())["sources"]
    manifest = json.loads((source / "sha256.json").read_text())
    meta = {
        "outcome": "started",
        "engine": subprocess.check_output(
            [str(ENGINE), "--version"], text=True
        ).strip(),
        "source": source.name,
        "git_commit": None,
        "os": platform.platform(),
        "scope": "native exact-time camera inputs; isolated storageGPUqueries; no screenshots/readback in timed scope",
        "protocol": protocol,
        "seed": 913,
        "results": [],
    }
    if "--resume" in sys.argv:
        parent = Path(sys.argv[sys.argv.index("--resume") + 1]).resolve()
        previous = json.loads((parent / "metadata.json").read_text())
        assert previous["source"] == source.name and previous["protocol"] == protocol
        for f, h in json.loads((parent / "sha256.json").read_text()).items():
            assert hashlib.sha256((parent / f).read_bytes()).hexdigest() == h, f
        meta["parent_run"] = str(parent)
        meta["parent_outcome"] = previous["outcome"]
        meta["parent_error"] = previous.get("error")
        meta["results"] = [
            dict(r, folder=str(Path("..") / parent.name / r["folder"]))
            for r in previous["results"]
        ]

    def telemetry():
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.gr",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()

    def execute(entry, mode, block, pilot=False):
        done = next(
            (
                r
                for r in meta["results"]
                if r["width"] == entry["width"]
                and r["mode"] == mode
                and r["block"] == block
                and r["pilot"] == pilot
            ),
            None,
        )
        if done:
            return done
        w, h = entry["width"], entry["height"]
        folder = out / f"{len(meta['results']):02d}_{w}_{mode}"
        folder.mkdir()
        samples = json.loads((source / entry["folder"] / "samples.json").read_text())[
            "samples"
        ]
        for f in samples:
            p = recorded_path(f["path"])
            assert (
                hashlib.sha256(p.read_bytes()).hexdigest()
                == manifest[p.relative_to(source).as_posix()]
            )
            f["path"] = str(p)
        budget = round(373084 * w * h / (160 * 90))
        cap = capacity_for_budget(w, h, budget)
        job = {
            "mode": "none" if mode == "none" else "dense",
            "innovation": mode == "innovation",
            "pool_slots": cap,
            "storage_format": (
                mode if mode in ["rgba32f", "rgba16f", "rgba8"] else "rgba32f"
            ),
            "width": w,
            "height": h,
            "history": 10,
            "alpha": 2 ** (-0.05 / 0.25),
            "inputs": [f["path"] for f in samples],
            "warm_up": 0 if pilot else 120,
            "warm_up_us": 0 if pilot else 2000000,
            "minimum_frames": len(samples) if pilot else 300,
            "duration_us": 0 if pilot else 10000000,
            "output": str(folder / "samples.json"),
        }
        p = folder / "job.json"
        p.write_text(json.dumps(job, indent=2))
        before = telemetry()
        cmd = [
            str(ENGINE),
            "--path",
            str(ROOT / "godot"),
            "res://scenes/storage_timing.tscn",
            "--rendering-method",
            "forward_plus",
            "--rendering-driver",
            "vulkan",
            "--disable-vsync",
            "--audio-driver",
            "Dummy",
            "--log-file",
            str(folder / "engine.log"),
            "--",
            str(p),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=240)
        (folder / "stdout.log").write_bytes(proc.stdout)
        (folder / "stderr.log").write_bytes(proc.stderr)
        log = (folder / "engine.log").read_text(errors="replace")
        assert (
            proc.returncode == 0
            and "STORAGE_TIMING_COMPLETE" in log
            and "ERROR:" not in log
        ), {
            "returncode": proc.returncode,
            "completed": "STORAGE_TIMING_COMPLETE" in log,
            "log": log[-2000:],
            "stderr": proc.stderr.decode(errors="replace")[-2000:],
        }
        data = json.loads((folder / "samples.json").read_text())
        rows = validate_samples(data, job["warm_up"])
        assert len(rows) >= job["minimum_frames"]
        assert rows[-1]["wall_us"] - rows[0]["wall_us"] >= job["duration_us"]
        assert len(rows) % len(samples) == 0
        result = {
            "width": w,
            "height": h,
            "mode": mode,
            "block": block,
            "pilot": pilot,
            "folder": folder.name,
            "captures": len(rows),
            "gpu_ms_quantiles": np.percentile(
                [r["gpu_ns"] / 1e6 for r in rows], [50, 95, 99]
            ).tolist(),
            "cpu_submission_ms_median": float(
                np.median([r["cpu_submission_us"] / 1000 for r in rows])
            ),
            "delivery_ms_quantiles": np.percentile(
                [
                    r["callback_interval_us"] / 1000
                    for r in rows
                    if r["callback_interval_us"] > 0
                ],
                [50, 95, 99],
            ).tolist(),
            "history_workspace_bytes": data["history_payload_bytes"],
            "input_cache_bytes": data["input_cache_bytes"],
            "gpu_before": before,
            "gpu_after": telemetry(),
            "capacity": cap if mode == "innovation" else None,
        }
        meta["results"].append(result)
        (out / "progress.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(result), flush=True)
        return result

    try:
        for entry in [e for e in inputs if e["scene"] == "camera"]:
            pilot = execute(entry, "innovation", -1, True)
            sparse_feasible = (
                pilot["gpu_ms_quantiles"][0]
                <= protocol["sparse_native_pilot_max_capture_ms"]
            )
            pilot["gate_passed"] = sparse_feasible
            order = ["none", "rgba32f", "rgba16f", "rgba8"] + (
                ["innovation"] if sparse_feasible else []
            )
            random.Random(meta["seed"]).shuffle(order)
            for block in range(5):
                for mode in order[block % len(order) :] + order[: block % len(order)]:
                    execute(entry, mode, block)
        meta["outcome"] = "passed"
    except Exception as exc:
        meta.update(outcome="failed", error=str(exc))
    meta["implementation_hashes"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "experiments/configs/paper_storage_protocol.json",
            ROOT / "godot/scripts/storage_timing_effect.gd",
            ROOT / "godot/scripts/dense_finite_history.gd",
            ROOT / "godot/scripts/innovation_history.gd",
            ROOT / "godot/shaders/dense_finite.glsl",
            *ROOT.glob("godot/shaders/innovation*.glsl"),
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
