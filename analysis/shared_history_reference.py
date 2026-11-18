"""Small independent global-budget model using image objects, not GPU slots."""

import numpy as np


class SharedReference:
    def __init__(self, width, height, tile, history, capacity, alpha):
        self.w = width
        self.h = height
        self.tile = tile
        self.H = history
        self.capacity = capacity
        self.alpha = alpha
        self.slices = [
            np.s_[y : min(y + tile, height), x : min(x + tile, width)]
            for y in range(0, height, tile)
            for x in range(0, width, tile)
        ]
        self.reset()

    def reset(self):
        self.sequence = 0
        self.maps = [[None] * self.H for _ in self.slices]
        self.images = {}
        self.ledger = np.zeros(len(self.slices))

    def capture(self, image):
        sequence = self.sequence
        count = min(sequence + 1, self.H)
        slot = sequence % self.H
        versions = []
        for t, sl in enumerate(self.slices):
            token = (t, sequence)
            self.images[token] = image[sl].astype(float).copy()
            self.maps[t][slot] = token
            ids = list(dict.fromkeys(v for v in self.maps[t] if v is not None))
            versions.append(ids)
            match = next(
                (
                    v
                    for v in ids
                    if v != token and np.array_equal(self.images[v], self.images[token])
                ),
                None,
            )
            if match is not None:
                self.maps[t] = [match if v == token else v for v in self.maps[t]]

        def proposal(t):
            refs = self.maps[t]
            alive = set(refs) - {None}
            ids = versions[t]
            best = None
            for ai, a in enumerate(ids):
                if a not in alive:
                    continue
                exposure = 0.0
                for r in range(self.H):
                    z = sum(self.alpha**age for age in range(min(count + r, self.H)))
                    mass = sum(
                        self.alpha ** (age + r)
                        for age in range(count)
                        if age + r < self.H and refs[(sequence - age) % self.H] == a
                    )
                    exposure += (mass / z) ** 2
                for bi, b in enumerate(ids):
                    if a == b or b not in alive:
                        continue
                    loss = exposure * np.mean(
                        (self.images[a][..., :3] - self.images[b][..., :3]) ** 2
                    )
                    area = self.images[a].shape[0] * self.images[a].shape[1]
                    candidate = (loss * area / (self.w * self.h), t, ai, bi, loss)
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
            return best

        total = sum(len(set(refs) - {None}) for refs in self.maps)
        requested = total
        merges = 0
        proposals = [proposal(t) for t in range(len(self.slices))]
        while total > self.capacity:
            item = min(p for p in proposals if p is not None)
            _, t, ai, bi, loss = item
            a, b = versions[t][ai], versions[t][bi]
            self.maps[t] = [b if v == a else v for v in self.maps[t]]
            self.ledger[t] += np.sqrt(loss)
            total -= 1
            merges += 1
            proposals[t] = proposal(t)
        output = np.zeros_like(image, dtype=float)
        weights = self.alpha ** np.arange(count, dtype=float)
        weights /= sum(weights)
        for t, sl in enumerate(self.slices):
            output[sl] = sum(
                w * self.images[self.maps[t][(sequence - age) % self.H]]
                for age, w in enumerate(weights)
            )
        self.sequence += 1
        return output, {
            "used": total,
            "requested": requested,
            "merges": merges,
            "versions_per_tile": [len(set(refs) - {None}) for refs in self.maps],
        }
