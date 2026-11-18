"""Bounded-image analytical prototype: original-sample sketches correct merge cross terms.

This is CPU research arithmetic, not a GPU implementation or timing result.
"""

import numpy as np


def block_projection(image, grid=4):
    """Orthonormal disjoint cell means, scaled so squared norm uses RGB MSE units."""
    h, w, _ = image.shape
    values = []
    for ys in np.array_split(np.arange(h), grid):
        for xs in np.array_split(np.arange(w), grid):
            cell = image[np.ix_(ys, xs)][..., :3]
            values.extend(
                cell.sum(axis=(0, 1)) / np.sqrt(len(ys) * len(xs) * 3 * h * w)
                if len(ys) * len(xs)
                else np.zeros(3)
            )
    return np.array(values)


def future_weights(count, history, alpha):
    weights = np.zeros((history, count))
    for r in range(history):
        z = sum(alpha**j for j in range(min(history, count + r)))
        for age in range(count):
            if age + r < history:
                weights[r, age] = alpha ** (age + r) / z
    return weights


def corrected_scores(images, projections, originals, assignment, weights, correct=True):
    """Rows remove a, columns retain b. Originals/assignment are ordered by age."""
    u = len(images)
    distances = np.array([np.mean((images - images[a]) ** 2, axis=1) for a in range(u)])
    mass = weights @ np.eye(u)[assignment]
    scores = (mass * mass).sum(axis=0)[:, None] * distances
    if correct:
        residual = weights @ (projections[assignment] - originals)
        cross = mass.T @ residual
        scores += 2 * (
            cross @ projections.T - np.sum(cross * projections, axis=1)[:, None]
        )
    np.fill_diagonal(scores, np.inf)
    return scores


class ResidualHistory:
    def __init__(
        self,
        width,
        height,
        tile=16,
        history=10,
        capacity=75,
        alpha=2 ** (-0.05 / 0.25),
        correct=True,
        grid=4,
    ):
        self.w = width
        self.h = height
        self.tile = tile
        self.H = history
        self.capacity = capacity
        self.alpha = alpha
        self.correct = correct
        self.grid = grid
        self.slices = [
            np.s_[y : min(y + tile, height), x : min(x + tile, width)]
            for y in range(0, height, tile)
            for x in range(0, width, tile)
        ]
        self.maps = [[None] * history for _ in self.slices]
        self.images = {}
        self.q = {}
        self.original = [np.zeros((history, 3 * grid * grid)) for _ in self.slices]
        self.sequence = 0

    def capture(self, image):
        sequence = self.sequence
        count = min(sequence + 1, self.H)
        slot = sequence % self.H
        ages = [(sequence - age) % self.H for age in range(count)]
        weights = future_weights(count, self.H, self.alpha)
        ids = []
        for t, sl in enumerate(self.slices):
            token = (t, sequence)
            im = image[sl].astype(float).copy()
            self.images[token] = im
            self.q[token] = block_projection(im, self.grid)
            self.original[t][slot] = self.q[token]
            self.maps[t][slot] = token
            versions = list(dict.fromkeys(v for v in self.maps[t] if v is not None))
            ids.append(versions)
            match = next(
                (
                    v
                    for v in versions
                    if v != token and np.array_equal(self.images[v], im)
                ),
                None,
            )
            if match is not None:
                self.maps[t] = [match if v == token else v for v in self.maps[t]]

        def proposal(t):
            versions = [v for v in ids[t] if v in self.maps[t]]
            if len(versions) < 2:
                return None
            ims = np.stack([self.images[v][..., :3].ravel() for v in versions])
            qs = np.stack([self.q[v] for v in versions])
            assignment = np.array([versions.index(self.maps[t][i]) for i in ages])
            scores = corrected_scores(
                ims, qs, self.original[t][ages], assignment, weights, self.correct
            )
            a, b = np.unravel_index(np.argmin(scores), scores.shape)
            area = self.images[versions[0]].shape[0] * self.images[versions[0]].shape[1]
            return (
                float(scores[a, b]) * area / (self.w * self.h),
                t,
                versions[a],
                versions[b],
            )

        total = sum(len(set(m) - {None}) for m in self.maps)
        requested = total
        merges = 0
        negative = 0
        proposals = [proposal(t) for t in range(len(self.slices))]
        while total > self.capacity:
            score, t, a, b = min(p for p in proposals if p is not None)
            self.maps[t] = [b if v == a else v for v in self.maps[t]]
            total -= 1
            merges += 1
            negative += int(score < 0)
            proposals[t] = proposal(t)
        output = np.zeros_like(image, dtype=float)
        for t, sl in enumerate(self.slices):
            output[sl] = sum(
                weights[0, age] * self.images[self.maps[t][i]]
                for age, i in enumerate(ages)
            )
        # No discarded full images survive capture; only original projections retain their age.
        alive = {v for m in self.maps for v in m if v is not None}
        self.images = {v: self.images[v] for v in alive}
        self.q = {v: self.q[v] for v in alive}
        assert len(self.images) == total <= self.capacity
        self.sequence += 1
        return output, {
            "used": total,
            "requested": requested,
            "merges": merges,
            "negative_scores": negative,
        }
