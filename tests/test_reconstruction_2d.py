import unittest
import numpy as np
from analysis.reconstruction_2d import render, reconstruct, specifications


class Reconstruction2DTests(unittest.TestCase):
    def fixture(self):
        return {
            "size": [97, 65],
            "position": [40, 30],
            "velocity": [80, -40],
            "acceleration": [0, 0],
            "halfsize": [6, 8],
            "shape": "L",
            "occluders": [],
        }

    def test_integer_diagonal_matches_exact_target(self):
        s = self.fixture()
        c0, a0, v0 = render(s, 0)
        c1, a1, v1 = render(s, 0.1)
        target, _, _ = render(s, 0.05)
        result, _ = reconstruct(c0, c1, a0, a1, 0.05, -0.05, v0, v1, 2)
        np.testing.assert_array_equal(result, target)

    def test_axis_transpose_invariance(self):
        s = self.fixture()
        c0, a0, v0 = render(s, 0)
        c1, a1, v1 = render(s, 0.1)
        result, codes = reconstruct(c0, c1, a0, a1, 0.05, -0.05, v0, v1, 1)
        b0 = a0.transpose(1, 0, 2).copy()
        b1 = a1.transpose(1, 0, 2).copy()
        b0[..., :2] = b0[..., :2][..., ::-1]
        b1[..., :2] = b1[..., :2][..., ::-1]
        rotated, rcodes = reconstruct(
            c0.transpose(1, 0, 2),
            c1.transpose(1, 0, 2),
            b0,
            b1,
            0.05,
            -0.05,
            v0[::-1],
            v1[::-1],
            1,
        )
        np.testing.assert_array_equal(result.transpose(1, 0, 2), rotated)
        np.testing.assert_array_equal(codes.T, rcodes)

    def test_new_cases_are_deterministic(self):
        self.assertEqual(specifications(), specifications())
        self.assertEqual(len(specifications()), 36)
