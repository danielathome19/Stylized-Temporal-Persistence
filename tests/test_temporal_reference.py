import math
import unittest
import numpy as np
from fractions import Fraction as F
from analysis.temporal_reference import (
    retention,
    target_ticks,
    recursive_oracle,
    kernel_weights,
    live_composite,
)


class OracleTests(unittest.TestCase):
    def test_impulse_seed_and_new_frame(self):
        np.testing.assert_allclose(recursive_oracle([1.0, 0.0, 0.0], [0.7, 0.7]), 0.49)
        np.testing.assert_allclose(recursive_oracle([0.0, 1.0, 0.0], [0.7, 0.7]), 0.21)

    def test_static_normalization_and_zero_retention(self):
        np.testing.assert_allclose(recursive_oracle(np.ones((8, 3, 5)), [0.91] * 7), 1)
        np.testing.assert_equal(recursive_oracle([2.0, 3.0, 7.0], [0.0, 0.0]), 7)

    def test_half_life_and_pause(self):
        a = math.exp(-math.log(2) * 0.05 / 0.25)
        self.assertAlmostEqual(retention(a, 0.25, 0.05), 0.5)
        self.assertEqual(retention(0, 0, 0.05), 1)
        self.assertEqual(retention(0, 0.1, 0.05), 0)
        with self.assertRaises(ValueError):
            retention(a, -0.1, 0.05)
        np.testing.assert_equal(recursive_oracle([0.3, 100.0], [1.0]), 0.3)
        with self.assertRaises(ValueError):
            kernel_weights([0.1], "exponential", 1, float("nan"))

    def test_phase_boundaries_and_multiple_missed_ticks(self):
        self.assertEqual(
            target_ticks(F(0), F(19, 100), 20), [F(1, 20), F(1, 10), F(3, 20)]
        )
        self.assertEqual(target_ticks(F(1, 20), F(1, 20), 20), [])
        self.assertEqual(
            target_ticks(F(0), F(1, 10), 20, F(1, 40)), [F(1, 40), F(3, 40)]
        )

    def test_kernels_support_and_empty(self):
        for kind in ["exponential", "linear", "gaussian", "hyperbolic"]:
            w = kernel_weights([0, 0.1, 0.2, 3], kind, 1, 0.25)
            self.assertAlmostEqual(w.sum(), 1)
            self.assertEqual(w[-1], 0)
            self.assertTrue(np.all(w >= 0))
            self.assertEqual(len(kernel_weights([], kind, 1, 0.25)), 0)

    def test_live_excludes_newest_and_initial_reset(self):
        np.testing.assert_allclose(
            live_composite(1, [0, 100], [F(0), F(1, 10)], F(1, 10), 0.3), 0.3
        )
        np.testing.assert_equal(live_composite(1, [], [], F(0), 0.3), 1)
        np.testing.assert_equal(live_composite(1, [0], [F(0)], F(0), 0.3), 1)

    def test_expansion_matches_independent_scalar_recurrence(self):
        values = [0.2, 0.9, 0.1, 0.5]
        alphas = [0.1, 0.8, 0.4]
        result = values[0]
        for f, a in zip(values[1:], alphas):
            result = (1 - a) * f + a * result
        self.assertAlmostEqual(float(recursive_oracle(values, alphas)), result)


if __name__ == "__main__":
    unittest.main()
