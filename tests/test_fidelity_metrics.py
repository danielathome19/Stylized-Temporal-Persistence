import unittest
import numpy as np
from experiments.run.fidelity import metrics


class FidelityMetricsTests(unittest.TestCase):
    def test_background_does_not_dilute_region_error(self):
        target = np.zeros((2, 2, 4))
        actual = target.copy()
        actual[0, 0, :3] = 1
        mask = np.array([[True, False], [False, False]])
        result = metrics(actual, target, mask)
        self.assertEqual(result["mse_rgb"], 0.25)
        self.assertEqual(result["motion_region_mse"], 1)
        self.assertAlmostEqual(result["psnr_db"], 6.020599913)

    def test_empty_region_and_exact_equality(self):
        im = np.zeros((2, 2, 4))
        result = metrics(im, im, np.zeros((2, 2), dtype=bool))
        self.assertTrue(result["psnr_infinite"])
        self.assertIsNone(result["psnr_db"])
        self.assertIsNone(result["motion_region_mse"])

    def test_alpha_excluded(self):
        a = np.zeros((1, 1, 4))
        b = a.copy()
        b[..., 3] = 1
        self.assertEqual(metrics(a, b, np.ones((1, 1), dtype=bool))["mse_rgb"], 0)
