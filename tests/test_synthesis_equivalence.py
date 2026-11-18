"""Falsification checks for overbroad originality claims and exactness claims."""

import unittest
import numpy as np
from analysis.residual_history_reference import future_weights
from analysis.synthesis_history_reference import pair_parameters


class EquivalenceTests(unittest.TestCase):
    def test_expiring_kernel_determinant_and_infinite_exponential_rank(self):
        rng = np.random.default_rng(911)
        for h in [2, 4, 7]:
            kernel = rng.uniform(0.1, 1, h)
            schedule = np.zeros((h, h))
            z = kernel.sum()
            for r in range(h):
                for age in range(h - r):
                    schedule[r, age] = kernel[age + r] / z
            expected = (-1) ** (h * (h - 1) // 2) * (kernel[-1] / z) ** h
            self.assertAlmostEqual(np.linalg.det(schedule), expected, places=14)
            infinite = 0.8 ** (np.arange(h)[:, None] + np.arange(h)[None, :])
            self.assertEqual(np.linalg.matrix_rank(infinite), 1)

    def test_pair_synthesis_is_standard_fixed_design_least_squares(self):
        rng = np.random.default_rng(910)
        for count in [2, 4, 10]:
            profiles = future_weights(count, 10, 0.8)
            colors = rng.random((count, 21))
            for a in range(count):
                for b in range(a + 1, count):
                    _, theta = pair_parameters(profiles, a, b)
                    target = (
                        profiles[:, a, None] * colors[a]
                        + profiles[:, b, None] * colors[b]
                    )
                    design = (profiles[:, a] + profiles[:, b])[:, None]
                    solved = np.linalg.lstsq(design, target, rcond=None)[0][0]
                    np.testing.assert_allclose(
                        solved, theta * colors[a] + (1 - theta) * colors[b], atol=1e-14
                    )

    def test_overlap_objective_is_not_sum_of_independent_cluster_quadrics(self):
        # Additive independent energies omit the cross term 2<x,y>(X-A)(X-B).
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 0.0])
        a = 0.0
        b = 1.0
        candidate = 0.4
        true = np.sum(((x + y) * candidate - x * a - y * b) ** 2)
        additive = (
            np.dot(x, x) * (candidate - a) ** 2 + np.dot(y, y) * (candidate - b) ** 2
        )
        self.assertAlmostEqual(
            true - additive, 2 * np.dot(x, y) * (candidate - a) * (candidate - b)
        )
        self.assertGreater(abs(true - additive), 0.4)

    def test_disjoint_expiring_profiles_cannot_merge_exactly_when_images_differ(self):
        # Positive finite kernel -> full-column-rank future schedule. Disjoint nonempty
        # timestamp groups are independent; their Gram determinant is strictly positive.
        for h in [2, 4, 10]:
            w = future_weights(h, h, 0.8)
            assignment = np.arange(h) % 2
            mass = w @ np.eye(2)[assignment]
            cost, _ = pair_parameters(mass, 0, 1)
            self.assertGreater(cost, 0)
            self.assertEqual(np.linalg.matrix_rank(mass), 2)


if __name__ == "__main__":
    unittest.main()
