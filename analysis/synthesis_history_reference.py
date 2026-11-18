"""Temporal-profile least-squares representative synthesis; analytical CPU model."""

import numpy as np
from analysis.residual_history_reference import future_weights


def pair_parameters(mass, a, b, policy="horizon"):
    x = mass[:, a]
    y = mass[:, b]
    aa = float(x @ x)
    bb = float(y @ y)
    ab = float(x @ y)
    if policy == "independent":
        ab = 0.0
    denominator = aa + bb + 2 * ab
    if denominator == 0:
        return 0.0, 0.5
    theta = (aa + ab) / denominator
    # Norm of theta*(x+y)-x avoids cancellation in aa*bb-ab*ab.
    if policy == "horizon":
        cost = float(np.sum((theta * (x + y) - x) ** 2))
    else:
        cost = aa * bb / denominator
    return cost, theta


class SynthesisHistory:
    def __init__(
        self,
        width,
        height,
        tile=16,
        history=10,
        capacity=75,
        alpha=2 ** (-0.05 / 0.25),
        policy="horizon",
        packed=False,
    ):
        self.packed = packed
        self.w = width
        self.h = height
        self.tile = tile
        self.H = history
        self.capacity = capacity
        self.alpha = alpha
        self.policy = policy
        self.slices = [
            np.s_[y : min(y + tile, height), x : min(x + tile, width)]
            for y in range(0, height, tile)
            for x in range(0, width, tile)
        ]
        self.maps = [[None] * history for _ in self.slices]
        self.images = {}
        self.sequence = 0

    def capture(self, image):
        seq = self.sequence
        count = min(seq + 1, self.H)
        slot = seq % self.H
        ages = [(seq - age) % self.H for age in range(count)]
        weights = future_weights(count, self.H, self.alpha)
        for t, sl in enumerate(self.slices):
            token = (t, seq)
            im = image[sl].astype(float).copy()
            self.images[token] = im
            self.maps[t][slot] = token
            ids = list(dict.fromkeys(v for v in self.maps[t] if v is not None))
            match = next(
                (v for v in ids if v != token and np.array_equal(self.images[v], im)),
                None,
            )
            if match is not None:
                self.maps[t] = [match if v == token else v for v in self.maps[t]]

        def proposal(t):
            ids = list(dict.fromkeys(v for v in self.maps[t] if v is not None))
            u = len(ids)
            if u < 2:
                return None
            assignment = np.array([ids.index(self.maps[t][i]) for i in ages])
            mass = weights @ np.eye(u)[assignment]
            best = None
            for a in range(u):
                for b in range(a + 1, u):
                    coefficient, theta = pair_parameters(mass, a, b, self.policy)
                    loss = coefficient * np.mean(
                        (self.images[ids[a]][..., :3] - self.images[ids[b]][..., :3])
                        ** 2
                    )
                    area = self.images[ids[a]].shape[0] * self.images[ids[a]].shape[1]
                    candidate = (
                        float(loss) * area / (self.w * self.h),
                        t,
                        a,
                        b,
                        theta,
                        ids[a],
                        ids[b],
                    )
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
            return best

        total = sum(len(set(m) - {None}) for m in self.maps)
        requested = total
        merges = 0
        proposals = [proposal(t) for t in range(len(self.slices))]
        while total > self.capacity:
            score, t, _, _, theta, a, b = min(p for p in proposals if p is not None)
            self.images[b] = theta * self.images[a] + (1 - theta) * self.images[b]
            self.maps[t] = [b if v == a else v for v in self.maps[t]]
            total -= 1
            merges += 1
            proposals[t] = proposal(t)
        if self.packed:
            alive = {v for m in self.maps for v in m if v is not None}
            for v in alive:
                self.images[v] = (
                    np.floor(np.clip(self.images[v], 0, 1) * 255 + 0.5) / 255
                )
        out = np.zeros_like(image, dtype=float)
        for t, sl in enumerate(self.slices):
            out[sl] = sum(
                weights[0, age] * self.images[self.maps[t][i]]
                for age, i in enumerate(ages)
            )
        alive = {v for m in self.maps for v in m if v is not None}
        self.images = {v: self.images[v] for v in alive}
        assert len(self.images) == total <= self.capacity
        self.sequence += 1
        return out, {"used": total, "merges": merges, "requested": requested}
