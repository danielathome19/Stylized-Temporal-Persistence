import unittest
import numpy as np
from analysis.native_3d_reference import reconstruct


class FractionalScatter(unittest.TestCase):
    def test_unequal_motion_intervals(self):
        left = np.zeros((1, 9, 4), np.float32)
        right = left.copy()
        left[0, 2] = [1, 1, 1, 1]
        right[0, 6] = [1, 1, 1, 1]
        a = np.zeros_like(left)
        b = a.copy()
        # Previous interval 100 ms, current interval 200 ms, target 50 ms after left.
        a[0, 2] = [0.2, -2 / 9, 0, 1]
        b[0, 6] = [0.2, -4 / 9, 0, 1]
        result, codes = reconstruct(
            left, right, a, b, 1, scales=(-0.5, 0.75), fraction=0.25
        )
        np.testing.assert_array_equal(result[0, 3], [1, 1, 1, 1])
        self.assertEqual(codes[0, 3], 1)

    def test_crossfade_fraction(self):
        left = np.zeros((1, 1, 4), np.float32)
        right = np.ones_like(left)
        result, _ = reconstruct(left, right, left, left, 4, fraction=0.25)
        np.testing.assert_array_equal(result, np.full_like(left, 0.25))
