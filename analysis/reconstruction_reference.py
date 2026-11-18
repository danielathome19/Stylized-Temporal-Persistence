"""Independent scalar candidate oracle for restricted translation reconstruction."""

import numpy as np


def reconstruct(c0, c1, a0, a1, dt0, dt1, v0, v1, policy=0):
    h, w, _ = c0.shape
    out = np.empty_like(c0)
    valid = np.zeros((h, w), bool)
    for y in range(h):
        for x in range(w):
            candidates = []
            for c, a, dt, v in [(c0, a0, dt0, v0), (c1, a1, dt1, v1)]:
                choices = []
                if a[y, x, 3] != 1:
                    choices.append((a[y, x], c[y, x]))
                q = int(np.floor(x - dt * v + 0.5))
                if 0 <= q < w and a[y, q, 3] == 1 and abs(a[y, q, 0] - v) < 0.001:
                    choices.append((a[y, q], c[y, q]))
                candidates.append(
                    min(choices, key=lambda z: z[0][2]) if choices else None
                )
            left, right = candidates
            accept = (
                left is not None
                and right is not None
                and left[0][3] == right[0][3]
                and abs(left[0][2] - right[0][2]) < 0.001
                and np.max(np.abs(left[1][:3] - right[1][:3])) < 0.001
            )
            valid[y, x] = accept
            out[y, x] = (
                (1 - dt0 / (dt0 - dt1)) * left[1] + dt0 / (dt0 - dt1) * right[1]
                if accept
                else c0[y, x]
            )
    return out, valid


def reconstruct_visibility(c0, c1, a0, a1, dt0, dt1, v0, v1, policy):
    # Candidate oracle derived from per-endpoint visible source geometry.
    out, agree = reconstruct(c0, c1, a0, a1, dt0, dt1, v0, v1)
    provenance = agree.astype(np.float32)
    fraction = dt0 / (dt0 - dt1)
    if policy in (3, 4):
        return (
            c0.copy() if policy == 3 else (1 - fraction) * c0 + fraction * c1
        ), np.zeros(agree.shape, np.float32)
    if policy == 2 and abs(v0 - v1) * (dt0 - dt1) > 0.5:
        return (1 - fraction) * c0 + fraction * c1, np.full(agree.shape, 3, np.float32)
    if policy >= 1:
        h, w = agree.shape
        for y in range(h):
            for x in range(w):
                candidates = []
                for color, aux, dt, v in [(c0, a0, dt0, v0), (c1, a1, dt1, v1)]:
                    choices = []
                    if aux[y, x, 3] != 1:
                        choices.append((aux[y, x, 2], color[y, x]))
                    q = int(np.floor(x - dt * v + 0.5))
                    if (
                        0 <= q < w
                        and aux[y, q, 3] == 1
                        and abs(aux[y, q, 0] - v) < 0.001
                    ):
                        choices.append((aux[y, q, 2], color[y, q]))
                    candidates.append(
                        min(choices, key=lambda z: z[0])[1] if choices else None
                    )
                if (candidates[0] is None) != (candidates[1] is None):
                    out[y, x] = (
                        candidates[0] if candidates[0] is not None else candidates[1]
                    )
                    provenance[y, x] = 2
    return out, provenance


def synthetic(t, kind, w=65, h=33):
    color = np.zeros((h, w, 4), dtype=np.float32)
    color[..., 3] = 1
    aux = np.zeros_like(color)
    aux[..., 2] = 10
    velocity = 20.0 if kind != "acceleration" else 80 * t
    center = 12 + 20 * t if kind != "acceleration" else 12 + 40 * t * t
    yy, xx = np.mgrid[:h, :w]
    mask = (abs(xx - center) < 4) & (abs(yy - 16) < 5)
    color[mask, :3] = [1, 0.3, 0]
    aux[mask] = [velocity, 0, 2, 1]
    if kind == "occlusion":
        cover = (xx >= 21) & (xx <= 27)
        color[cover, :3] = [0, 0, 1]
        aux[cover] = [0, 0, 1, 2]
    if kind == "hidden_midpoint":
        cover = (xx < 20) | (xx > 25)
        color[cover, :3] = [0, 0, 1]
        aux[cover] = [0, 0, 1, 2]
    return color, aux, velocity
