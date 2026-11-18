import unittest
import numpy as np
from analysis.residual_history_reference import (
    block_projection,
    corrected_scores,
    future_weights,
    ResidualHistory,
)


class ResidualHistoryTests(unittest.TestCase):
    def test_same_compressed_state_can_require_different_optimal_merges(self):
        images = np.array([[0.0], [0.5], [1.0]])
        assignment = np.array([0, 0, 1, 2])
        w = future_weights(4, 4, 0.8)
        choices = []
        for original in [
            np.array([[0.0], [0.0], [0.5], [1.0]]),
            np.array([[0.0], [0.8], [0.5], [1.0]]),
        ]:
            scores = corrected_scores(images, images, original, assignment, w)
            choices.append(np.unravel_index(np.argmin(scores), scores.shape))
        self.assertEqual(choices, [(2, 1), (0, 1)])

    def test_finite_expiring_history_has_full_rank(self):
        for history in [2, 4, 10]:
            for alpha in [0.2, 0.8, 0.95]:
                weights = future_weights(history, history, alpha)
                self.assertEqual(np.linalg.matrix_rank(weights), history)

    def test_full_projection_equals_explicit_total_error_change(self):
        rng = np.random.default_rng(192)
        images = rng.normal(size=(3, 12))
        original = rng.normal(size=(5, 12))
        assignment = np.array([2, 0, 1, 0, 2])
        w = future_weights(5, 7, 0.8)
        scores = corrected_scores(
            images, images / np.sqrt(12), original / np.sqrt(12), assignment, w
        )
        before = w @ (images[assignment] - original)
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                changed = assignment.copy()
                changed[changed == a] = b
                after = w @ (images[changed] - original)
                expected = np.sum(after * after - before * before) / 12
                self.assertAlmostEqual(scores[a, b], expected, places=12)

    def test_zero_residual_recovers_existing_score(self):
        rng = np.random.default_rng(193)
        images = rng.normal(size=(3, 12))
        assignment = np.array([0, 1, 2, 1])
        q = images / np.sqrt(12)
        w = future_weights(4, 6, 0.7)
        np.testing.assert_allclose(
            corrected_scores(images, q, q[assignment], assignment, w),
            corrected_scores(images, q, q[assignment], assignment, w, False),
            atol=1e-14,
        )

    def test_block_projection_energy_and_orthogonality(self):
        rng = np.random.default_rng(194)
        im = rng.normal(size=(9, 17, 3))
        q = block_projection(im)
        reconstruction = np.zeros_like(im)
        for ys in np.array_split(np.arange(9), 4):
            for xs in np.array_split(np.arange(17), 4):
                reconstruction[np.ix_(ys, xs)] = im[np.ix_(ys, xs)].mean(axis=(0, 1))
        self.assertAlmostEqual(np.sum(q * q), np.mean(reconstruction**2), places=12)
        self.assertAlmostEqual(
            np.mean(im**2),
            np.sum(q * q) + np.mean((im - reconstruction) ** 2),
            places=12,
        )

    def test_discarded_images_do_not_accumulate(self):
        rng = np.random.default_rng(195)
        model = ResidualHistory(5, 3, tile=4, history=4, capacity=3, grid=2)
        for _ in range(12):
            result, stats = model.capture(rng.random((3, 5, 4)))
            self.assertLessEqual(len(model.images), 3)
            self.assertEqual(len(model.q), len(model.images))
            self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
