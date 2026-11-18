import unittest
import numpy as np
from analysis.paper_storage_report import luminance_ssim


class PaperMetricsTests(unittest.TestCase):
    def test_ssim_identity(self):
        im = np.random.default_rng(916).random((32, 37, 3))
        self.assertAlmostEqual(luminance_ssim(im, im), 1.0, places=13)

    def test_ssim_constant_luminance_has_analytic_value(self):
        a = np.full((32, 37, 3), 0.2)
        b = np.full((32, 37, 3), 0.4)
        expected = (2 * 0.2 * 0.4 + 0.01**2) / (0.2**2 + 0.4**2 + 0.01**2)
        self.assertAlmostEqual(luminance_ssim(a, b), expected, places=10)


if __name__ == "__main__":
    unittest.main()
