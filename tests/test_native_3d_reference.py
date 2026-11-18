import unittest
import numpy as np
from analysis.native_3d_reference import candidates, reconstruct


class Native3DReferenceTests(unittest.TestCase):
    def test_reverse_depth_front_surface_wins_collision(self):
        c = np.zeros((3, 5, 4), np.float32)
        c[..., 3] = 1
        a = np.zeros_like(c)
        c[1, 1, 0] = 1
        c[1, 3, 1] = 1
        a[1, 1] = [0.2, 0.4, 0, 1]
        a[1, 3] = [0.4, -0.4, 0, 1]
        color, z, seen = candidates(c, a)
        self.assertTrue(seen[1, 2])
        np.testing.assert_array_equal(color[1, 2], c[1, 3])

    def test_zero_depth_does_not_claim_geometry(self):
        c = np.zeros((3, 5, 4), np.float32)
        a = np.zeros_like(c)
        _, _, seen = candidates(c, a)
        self.assertFalse(seen.any())

    def test_static_geometry_preserved(self):
        c = np.full((3, 5, 4), 0.3, np.float32)
        a = np.zeros_like(c)
        a[..., 0] = 0.2
        result, code = reconstruct(c, c, a, a, 1)
        np.testing.assert_array_equal(result, c)
        self.assertTrue((code == 1).all())

    def test_past_motion_is_extrapolated_for_left_endpoint(self):
        c = np.zeros((3, 5, 4), np.float32)
        a = np.zeros_like(c)
        c[1, 2, 0] = 1
        a[1, 2] = [0.2, -0.4, 0, 1]
        result, _, seen = candidates(c, a, direction=-1)
        self.assertTrue(seen[1, 3])
        self.assertEqual(result[1, 3, 0], 1)

    def test_equal_depth_tie_uses_first_source_index(self):
        c = np.zeros((3, 5, 4), np.float32)
        a = np.zeros_like(c)
        c[1, 1, 0] = 1
        c[1, 3, 1] = 1
        a[1, 1] = [0.2, 0.4, 0, 1]
        a[1, 3] = [0.2, -0.4, 0, 1]
        result, _, _ = candidates(c, a)
        np.testing.assert_array_equal(result[1, 2], c[1, 1])
