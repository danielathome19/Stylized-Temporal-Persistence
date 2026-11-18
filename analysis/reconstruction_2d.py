"""2D analytical fixtures and vectorized source-only candidate reference."""

import numpy as np


def render(spec, t):
    w, h = spec["size"]
    yy, xx = np.mgrid[:h, :w]
    pos = (
        np.asarray(spec["position"])
        + np.asarray(spec["velocity"]) * t
        + 0.5 * np.asarray(spec["acceleration"]) * t * t
    )
    velocity = np.asarray(spec["velocity"]) + np.asarray(spec["acceleration"]) * t
    dx = xx - pos[0]
    dy = yy - pos[1]
    sx, sy = spec["halfsize"]
    if spec["shape"] == "ellipse":
        mask = (dx / sx) ** 2 + (dy / sy) ** 2 < 1
    elif spec["shape"] == "L":
        mask = (abs(dx) < sx) & (abs(dy) < sy) & ((dx < 0) | (dy > 0))
    else:
        mask = (abs(dx) < sx) & (abs(dy) < sy)
    color = np.zeros((h, w, 4), np.float32)
    color[..., 3] = 1
    aux = np.zeros_like(color)
    aux[..., 2] = 10
    color[mask, :3] = [1, 0.3, 0]
    aux[mask] = [*velocity, 2, 1]
    for x0, y0, x1, y1 in spec["occluders"]:
        cover = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        color[cover, :3] = [0, 0, 1]
        aux[cover] = [0, 0, 1, 2]
    return color, aux, velocity


def reconstruct(c0, c1, a0, a1, dt0, dt1, v0, v1, policy):
    h, w, _ = c0.shape
    yy, xx = np.mgrid[:h, :w]
    candidates = []
    for color, aux, dt, v in [(c0, a0, dt0, v0), (c1, a1, dt1, v1)]:
        qx = np.floor(xx - dt * v[0] + 0.5).astype(int)
        qy = np.floor(yy - dt * v[1] + 0.5).astype(int)
        inside = (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
        qx = np.clip(qx, 0, w - 1)
        qy = np.clip(qy, 0, h - 1)
        sampled = aux[qy, qx]
        found = aux[..., 3] != 1
        take = (
            inside
            & (sampled[..., 3] == 1)
            & (np.linalg.norm(sampled[..., :2] - v, axis=2) < 0.001)
            & ((~found) | (sampled[..., 2] < aux[..., 2]))
        )
        chosen_aux = np.where(take[..., None], sampled, aux)
        chosen_color = np.where(take[..., None], color[qy, qx], color)
        candidates.append((chosen_color, chosen_aux, found | take))
    left, la, lv = candidates[0]
    right, ra, rv = candidates[1]
    agree = (
        lv
        & rv
        & (la[..., 3] == ra[..., 3])
        & (abs(la[..., 2] - ra[..., 2]) < 0.001)
        & (np.max(abs(left[..., :3] - right[..., :3]), axis=2) < 0.001)
    )
    f = dt0 / (dt0 - dt1)
    out = np.where(agree[..., None], (1 - f) * left + f * right, c0)
    codes = agree.astype(np.float32)
    if policy >= 1:
        one = lv ^ rv
        out = np.where(one[..., None], np.where(lv[..., None], left, right), out)
        codes[one] = 2
    if policy == 2 and np.linalg.norm(np.asarray(v0) - v1) * (dt0 - dt1) > 0.5:
        out = (1 - f) * c0 + f * c1
        codes[:] = 3
    if policy == 3:
        out = c0.copy()
        codes[:] = 0
    if policy == 4:
        out = (1 - f) * c0 + f * c1
        codes[:] = 0
    return out, codes


def specifications():
    rng = np.random.default_rng(20261115)
    cases = []
    for i in range(36):
        group = ["integer_translation", "subpixel_translation", "accelerated"][i // 12]
        velocity = rng.integers(-10, 11, size=2) * 20
        if group == "subpixel_translation":
            velocity = velocity + np.array([7.0, -11.0])
        acceleration = (
            rng.integers(-3, 4, size=2) * 400 if group == "accelerated" else np.zeros(2)
        )
        position = rng.integers([8, 8], [88, 56]).astype(float)
        if group == "subpixel_translation":
            position += 0.23
        covers = (
            []
            if i % 3 == 0
            else [[int(rng.integers(25, 60)), 0, int(rng.integers(61, 85)), 65]]
        )
        cases.append(
            {
                "name": f"{group}_{i:02d}",
                "group": group,
                "size": [97, 65],
                "position": position.tolist(),
                "velocity": velocity.tolist(),
                "acceleration": acceleration.tolist(),
                "halfsize": rng.integers(3, 10, size=2).tolist(),
                "shape": ["box", "ellipse", "L"][i % 3],
                "occluders": covers,
            }
        )
    return cases
