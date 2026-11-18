"""Renderer-native perspective motion/depth calibration and GPU reconstruction."""

import datetime
import hashlib
import json
import sys
import subprocess
import uuid
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.native_3d_reference import reconstruct
from analysis.temporal_reference import recursive_oracle
from experiments.run.reconstruction import metrics


def main():
    ref = ROOT / "results/raw/20261115T213510Z-3d-forward-inputs-09d971ac"
    for rel, h in json.loads((ref / "sha256.json").read_text()).items():
        assert hashlib.sha256((ref / rel).read_bytes()).hexdigest() == h
    out = (
        ROOT
        / "results/raw"
        / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-scatter-3d-"
            + uuid.uuid4().hex[:8]
        )
    )
    out.mkdir()

    def read(scene, name, kind):
        return np.fromfile(
            ref / scene / (name + "_" + kind + ".bin"), dtype="<f4"
        ).reshape(180, 320, 4)

    c = read("translation", "right", "color")
    a = read("translation", "right", "aux")
    mask = np.all(c[..., :3] > 0.9, axis=2)
    observed = np.median(a[mask, 1:3], axis=0) * [320, 180]
    f = 90 / np.tan(np.pi / 6)
    expected = np.array([-0.4 * f / 5.75, 0.2 * f / 5.75])
    expected_depth = 0.1 * 100 / (100 - 0.1) / 5.75 - 0.1 / (100 - 0.1)
    calibration = {
        "expected_motion_pixels": expected.tolist(),
        "observed_motion_pixels": observed.tolist(),
        "max_motion_error_px": float(abs(observed - expected).max()),
        "expected_reverse_depth": expected_depth,
        "observed_reverse_depth": float(np.median(a[mask, 0])),
        "static_motion_max": float(
            np.max(abs(read("translation", "warm2", "aux")[..., 1:3]))
        ),
    }
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
    assert calibration["max_motion_error_px"] < 0.02
    assert abs(calibration["observed_reverse_depth"] - expected_depth) < 1e-6
    assert calibration["static_motion_max"] == 0
    seqs = []
    alpha = 2 ** (-0.1 / 0.25)
    for scene in ["translation", "rotation", "occlusion", "camera"]:
        for label, policy in [
            ("strict", 0),
            ("one_sided", 1),
            ("hold", 3),
            ("crossfade", 4),
        ]:
            folder = out / (scene + "_" + label)
            folder.mkdir()
            frame = {
                "color0": str(ref / scene / "left_color.bin"),
                "color1": str(ref / scene / "right_color.bin"),
                "aux0": str(ref / scene / "left_aux.bin"),
                "aux1": str(ref / scene / "right_aux.bin"),
                "params": [0.1, 100, 0.5, 0.02, 0.05, 16],
                "output": str(folder / "result"),
            }
            seqs.append(
                {
                    "name": scene,
                    "label": label,
                    "policy": policy,
                    "width": 320,
                    "height": 180,
                    "alpha": alpha,
                    "frames": [frame],
                }
            )
    job = out / "job.json"
    job.write_text(json.dumps({"sequences": seqs}, indent=2))
    engine = ROOT / "Godot_v4.8-beta3_win64.exe"
    meta = json.loads((ref / "metadata.json").read_text())
    meta.update(
        run_id=out.name,
        stage="forward_scatter_3D_reconstruction_accuracy",
        source_reference=str(ref),
        method="past-motion left extrapolation plus current-motion right interpolation with one-sided evidence",
        warm_up=0,
        history_schedule="seed 0s, midpoint .1s, right endpoint .2s",
        target_frame_schedule=[0, 0.1, 0.2],
        actual_frame_schedule="synchronized GPU reconstruction replay; native capture frame indices in source reference",
        latency_policy="100ms midpoint delay, chronological previous/left/right motion only; accuracy replay",
        kernel_parameters={"half_life_s": 0.25, "alpha": alpha},
        configuration_hash=hashlib.sha256(job.read_bytes()).hexdigest(),
        outcome="started",
        calibration=calibration,
    )
    meta["implementation_hashes"].update(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "analysis/native_3d_reference.py",
                ROOT / "godot/scripts/m5_scatter_replay.gd",
                *[
                    ROOT / "godot/shaders" / name
                    for name in [
                        "scatter_clear.glsl",
                        "scatter_depth.glsl",
                        "scatter_owner.glsl",
                        "scatter_resolve.glsl",
                    ]
                ],
            ]
        }
    )
    try:
        proc = subprocess.run(
            [
                str(engine),
                "--path",
                str(ROOT / "godot"),
                "--rendering-method",
                "forward_plus",
                "--rendering-driver",
                "vulkan",
                "--audio-driver",
                "Dummy",
                "--log-file",
                str(out / "engine.log"),
                "--script",
                "res://scripts/m5_scatter_replay.gd",
                "--",
                str(job),
            ],
            capture_output=True,
            timeout=180,
        )
        (out / "stdout.log").write_bytes(proc.stdout)
        (out / "stderr.log").write_bytes(proc.stderr)
        log = (out / "engine.log").read_text(errors="replace")
        if proc.returncode or "M5_HISTORY_COMPLETE" not in log or "ERROR:" in log:
            raise RuntimeError("GPU reconstruction failed")
        rows = []
        for seq in seqs:
            scene = seq["name"]
            c0 = read(scene, "left", "color")
            c1 = read(scene, "right", "color")
            a0 = read(scene, "left", "aux")
            a1 = read(scene, "right", "aux")
            target = read(scene, "target", "color")
            expected, codes = reconstruct(
                c0, c1, a0, a1, seq["policy"], left_direction=-1
            )
            prefix = seq["frames"][0]["output"]
            actual = np.fromfile(prefix + "_sample.bin", dtype="<f4").reshape(
                180, 320, 4
            )
            provenance = np.fromfile(prefix + "_provenance.bin", dtype="<f4").reshape(
                180, 320, 4
            )[..., 0]
            history = np.fromfile(prefix + "_history.bin", dtype="<f4").reshape(
                180, 320, 4
            )
            oracle = recursive_oracle([c0, expected, c1], [alpha, alpha])
            truth = recursive_oracle([c0, target, c1], [alpha, alpha])
            error = max(
                float(abs(actual - expected).max()), float(abs(history - oracle).max())
            )
            if (
                not np.isfinite(actual).all()
                or not np.isfinite(history).all()
                or error > 2e-6
                or not np.array_equal(provenance, codes)
            ):
                raise RuntimeError(
                    "GPU oracle mismatch "
                    + scene
                    + " "
                    + seq["label"]
                    + " "
                    + str(error)
                )
            region = np.any(abs(c0[..., :3] - c1[..., :3]) > 1e-5, axis=2) | np.any(
                abs(target[..., :3] - c0[..., :3]) > 1e-5, axis=2
            )
            rows.append(
                {
                    "scene": scene,
                    "policy": seq["label"],
                    "oracle_max_abs": error,
                    "sample": metrics(actual, target, region),
                    "history": metrics(history, truth, region),
                    "one_sided_fraction": float(np.mean(codes == 2)),
                    "fallback_fraction": float(np.mean(codes == 0)),
                }
            )
        meta.update(outcome="passed", results=rows)
    except Exception as e:
        meta.update(outcome="failed", error=str(e))
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
    print(
        json.dumps(
            {
                "run": str(out),
                "outcome": meta["outcome"],
                "error": meta.get("error"),
                "calibration": calibration,
                "results": meta.get("results"),
            },
            indent=2,
        )
    )
    return int(meta["outcome"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
