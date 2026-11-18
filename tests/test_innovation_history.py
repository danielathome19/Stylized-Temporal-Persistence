import unittest
import numpy as np
from analysis.innovation_history_reference import InnovationHistory, innovation_weights


class InnovationTests(unittest.TestCase):
    def test_unbounded_transform_matches_independent_dense_oracle(self):
        rng = np.random.default_rng(293)
        for h in [1, 2, 7]:
            for alpha in [0.0, 0.8]:
                model = InnovationHistory(
                    5, 3, tile=4, history=h, capacity=2 * h, alpha=alpha
                )
                frames = []
                for _ in range(20):
                    im = rng.random((3, 5, 4))
                    frames.append(im)
                    actual, _ = model.capture(im)
                    samples = frames[-h:]
                    w = alpha ** np.arange(len(samples) - 1, -1, -1)
                    w /= w.sum()
                    np.testing.assert_allclose(
                        actual, sum(v * f for v, f in zip(w, samples)), atol=2e-15
                    )

    def test_omitted_changes_recover_after_expiration_without_reset(self):
        model = InnovationHistory(2, 2, history=4, capacity=0)
        model.capture(np.zeros((2, 2, 4)))
        im = np.ones((2, 2, 4))
        for _ in range(8):
            actual, stats = model.capture(im)
        np.testing.assert_array_equal(actual, im)
        self.assertEqual(stats["used"], 0)

    def test_current_frame_is_exact_when_alpha_zero(self):
        rng = np.random.default_rng(294)
        model = InnovationHistory(3, 2, history=5, capacity=1, alpha=0)
        for _ in range(12):
            im = rng.random((2, 3, 4))
            actual, stats = model.capture(im)
            np.testing.assert_array_equal(actual, im)
            self.assertLessEqual(stats["used"], 1)

    def test_each_innovation_has_finite_future_support(self):
        w = innovation_weights(6, 6, 0.8)
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.all(w[-1] == 0))
        for j in range(5):
            self.assertTrue(np.all(w[5 - j :, j] == 0))

    def test_pixel_payloads_match_oracle_for_sparse_motion(self):
        model = InnovationHistory(9, 7, tile=1, history=5, capacity=40)
        frames = []
        for n in range(20):
            im = np.zeros((7, 9, 4))
            im[..., 3] = 1
            im[2, n % 9, :3] = 1
            frames.append(im)
            actual, stats = model.capture(im)
            subset = frames[-5:]
            w = model.alpha ** np.arange(len(subset) - 1, -1, -1)
            w /= w.sum()
            np.testing.assert_allclose(
                actual, sum(a * b for a, b in zip(w, subset)), atol=1e-15
            )
            self.assertEqual(stats["discarded"], 0)


if __name__ == "__main__":
    unittest.main()
