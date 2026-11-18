"""Independent source-forward scatter oracle for bounded GPU inverse gathering."""

import numpy as np


def candidates(color, aux, direction=1):
    h, w, _ = color.shape
    y, x = np.mgrid[:h, :w]
    dx = np.floor(x + direction * 0.5 * aux[..., 1] * w + 0.5).astype(int)
    dy = np.floor(y + direction * 0.5 * aux[..., 2] * h + 0.5).astype(int)
    valid = (
        (dx >= 0)
        & (dx < w)
        & (dy >= 0)
        & (dy < h)
        & (abs(dx - x) <= 16)
        & (abs(dy - y) <= 16)
        & (aux[..., 0] > 0)
    )
    src = np.flatnonzero(valid)
    dest = (dy * w + dx).ravel()[src]
    depth = aux[..., 0].ravel()[src]
    # Depth descending, then source index ascending: matches strict greater-depth winner.
    order = np.lexsort((src, -depth, dest))
    ordered_dest = dest[order]
    first = (
        np.r_[True, ordered_dest[1:] != ordered_dest[:-1]]
        if len(order)
        else np.zeros(0, bool)
    )
    pick = order[first]
    target = dest[pick]
    source = src[pick]
    out = np.zeros_like(color)
    z = np.zeros(h * w)
    seen = np.zeros(h * w, bool)
    out.reshape(-1, 4)[target] = color.reshape(-1, 4)[source]
    z[target] = depth[pick]
    seen[target] = True
    linear = 0.1 * 100 / (0.1 + z * (100 - 0.1))
    return out, linear.reshape(h, w), seen.reshape(h, w)


def reconstruct(c0, c1, a0, a1, policy, left_direction=1, *, scales=None, fraction=0.5):
    left, right = (
        (left_direction, 1) if scales is None else (2 * scales[0], 2 * scales[1])
    )
    l, d0, v0 = candidates(c0, a0, left)
    r, d1, v1 = candidates(c1, a1, right)
    agree = (
        v0
        & v1
        & (abs(d0 - d1) <= 0.02 * np.maximum(d0, d1))
        & (np.max(abs(l[..., :3] - r[..., :3]), axis=2) <= 0.05)
    )
    out = np.where(agree[..., None], (1 - fraction) * l + fraction * r, c0)
    code = agree.astype(np.float32)
    if policy == 1:
        one = v0 ^ v1
        out = np.where(one[..., None], np.where(v0[..., None], l, r), out)
        code[one] = 2
    if policy == 3:
        out = c0.copy()
        code[:] = 0
    if policy == 4:
        out = (1 - fraction) * c0 + fraction * c1
        code[:] = 0
    return out, code
