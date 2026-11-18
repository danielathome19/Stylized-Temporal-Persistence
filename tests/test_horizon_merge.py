import unittest
import numpy as np
from analysis.horizon_merge import scores


class HorizonMergeTests(unittest.TestCase):
    def test_score_equals_explicit_future_composite_perturbation(self):
        # A retained version owns two timestamps; their weights add before squaring.
        pool = [np.full((2, 2, 4), 0.2), np.full((2, 2, 4), 0.8)]
        incoming = np.full((2, 2, 4), 0.5)
        queue = [0, 1, 0]
        horizon = 5
        alpha = 0.8
        for score, a, b in scores(pool, queue, incoming, horizon, alpha):
            images = pool + [incoming]
            energy = 0.0
            originals = queue + [2]
            merged = [b if v == a else v for v in originals]
            for future in range(horizon):
                retained = originals[max(0, len(originals) + future - horizon) :]
                altered = merged[max(0, len(merged) + future - horizon) :]
                weights = alpha ** np.arange(len(retained) + future - 1, future - 1, -1)
                denominator = sum(
                    alpha**age for age in range(min(len(originals) + future, horizon))
                )
                delta = (
                    sum(
                        w * (images[y] - images[x])
                        for w, x, y in zip(weights, retained, altered)
                    )
                    / denominator
                    if retained
                    else np.zeros_like(incoming)
                )
                energy += np.mean(delta[..., :3] ** 2)
            self.assertAlmostEqual(score, energy, places=12)
