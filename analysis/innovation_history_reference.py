"""Bounded temporal-derivative history; CPU analytical candidate, no GPU claim.

The live input is exact. Stored signed innovations reconstruct only historical
departures from that input. Dropped innovations stop affecting the output when
their finite support expires, so omission cannot create permanent drift.
"""

import numpy as np


def innovation_weights(count, history, alpha):
    """Rows=future captures; columns=present innovation ages (0 is newest).

    An innovation of age j separates sample ages j and j+1. Its coefficient
    is the normalized weight of samples older than it, with a negative sign.
    """
    weights = np.zeros((history, max(count - 1, 0)))
    for r in range(history):
        z = sum(alpha**i for i in range(min(count + r, history)))
        for j in range(count - 1):
            weights[r, j] = (
                sum(alpha**i for i in range(j + r + 1, min(count + r, history))) / z
            )
    return weights


class InnovationHistory:
    def __init__(
        self, width, height, tile=16, history=10, capacity=75, alpha=2 ** (-0.05 / 0.25)
    ):
        assert history > 0 and 0 <= alpha < 1 and capacity >= 0
        self.w = width
        self.h = height
        self.H = history
        self.alpha = alpha
        self.capacity = capacity
        self.slices = [
            np.s_[y : min(y + tile, height), x : min(x + tile, width)]
            for y in range(0, height, tile)
            for x in range(0, width, tile)
        ]
        self.previous = None
        self.events = {}
        self.sequence = 0

    def capture(self, image):
        image = np.asarray(image, dtype=float)
        n = self.sequence
        count = min(n + 1, self.H)
        self.events = {
            key: value for key, value in self.events.items() if n - key[1] < self.H - 1
        }
        if self.previous is not None and self.H > 1:
            for t, sl in enumerate(self.slices):
                delta = image[sl] - self.previous[sl]
                if np.any(delta):
                    self.events[t, n] = delta.copy()
        weights = innovation_weights(count, self.H, self.alpha)
        requested = len(self.events)
        # Incremental omission energy, not total original-error optimum.
        if len(self.events) > self.capacity:
            energy = np.sum(weights**2, axis=0)
            order = sorted(
                self.events,
                key=lambda k: (
                    energy[n - k[1]]
                    * np.sum(self.events[k][..., :3] ** 2)
                    / (self.w * self.h * 3),
                    k,
                ),
            )
            for key in order[: len(self.events) - self.capacity]:
                del self.events[key]
        output = image.copy()
        for (t, birth), delta in self.events.items():
            output[self.slices[t]] -= weights[0, n - birth] * delta
        self.previous = image.copy()
        self.sequence += 1
        return output, {
            "used": len(self.events),
            "requested": requested,
            "discarded": requested - len(self.events),
        }
