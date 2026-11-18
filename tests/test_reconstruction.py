import unittest
import numpy as np
from analysis.reconstruction_reference import reconstruct, synthetic


class ReconstructionTests(unittest.TestCase):
    def test_static_preserves_color(self):
        c = np.ones((5, 7, 4), np.float32)
        a = np.zeros_like(c)
        a[..., 2] = 10
        out, valid = reconstruct(c, c, a, a, 0.05, -0.05, 0, 0)
        np.testing.assert_array_equal(c, out)
        self.assertTrue(valid.all())

    def test_hidden_midpoint_exposes_false_confidence(self):
        c0, a0, v0 = synthetic(0, "hidden_midpoint")
        c1, a1, v1 = synthetic(1, "hidden_midpoint")
        target, _, _ = synthetic(0.5, "hidden_midpoint")
        out, valid = reconstruct(c0, c1, a0, a1, 0.5, -0.5, v0, v1)
        self.assertTrue(valid.all())
        self.assertGreater(float(np.mean((out - target) ** 2)), 0)

    def test_fallback_is_previous_pixel(self):
        c0, a0, v0 = synthetic(0, "translation")
        c1, a1, v1 = synthetic(1, "translation")
        out, valid = reconstruct(c0, c1, a0, a1, 0.5, -0.5, v0, v1)
        self.assertTrue((~valid).any())
        np.testing.assert_array_equal(out[~valid], c0[~valid])

    def test_static_foreground_wins_depth(self):
        c0, a0, v0 = synthetic(0, "occlusion")
        c1, a1, v1 = synthetic(1, "occlusion")
        out, _ = reconstruct(c0, c1, a0, a1, 0.5, -0.5, v0, v1)
        np.testing.assert_array_equal(out[:, 21:28], c0[:, 21:28])
