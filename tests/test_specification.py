"""Analytical feasibility checks, not tests of unimplemented M1-M6 shaders."""

import math
import unittest
from fractions import Fraction


class SpecificationChecks(unittest.TestCase):
    def test_half_life_partition_invariance(self):
        half_life = 0.25
        decay = math.log(2) / half_life
        for steps in ([0.25], [0.01] * 25, [0.03, 0.07, 0.15]):
            self.assertAlmostEqual(
                math.prod(math.exp(-decay * dt) for dt in steps), 0.5, places=14
            )

    def test_fixed_retention_changes_duration(self):
        a = math.exp(-math.log(2) * 0.05 / 0.25)
        self.assertAlmostEqual(a**5, 0.5)
        self.assertAlmostEqual(a**15, 0.125)

    def test_seed_and_recursive_weights_sum_to_one(self):
        for a in (0.0, 0.2, 0.8, 0.999):
            for n in (1, 2, 100):
                self.assertAlmostEqual(
                    a**n + sum((1 - a) * a**j for j in range(n)), 1.0
                )

    def test_missed_targets_are_causal(self):
        available = [Fraction(0), Fraction(7, 100), Fraction(19, 100)]
        targets = [Fraction(k, 20) for k in range(4)]
        sources = [max(t for t in available if t <= target) for target in targets]
        self.assertEqual(
            sources, [Fraction(0), Fraction(0), Fraction(7, 100), Fraction(7, 100)]
        )

    def test_live_current_excludes_equal_timestamp(self):
        now = Fraction(1, 10)
        targets = [Fraction(k, 20) for k in range(3)]
        self.assertEqual(
            [t for t in targets if t < now], [Fraction(0), Fraction(1, 20)]
        )


if __name__ == "__main__":
    unittest.main()
