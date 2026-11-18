import unittest
import numpy as np
from analysis.residual_history_reference import future_weights
from analysis.synthesis_history_reference import pair_parameters, SynthesisHistory


class SynthesisTests(unittest.TestCase):
    def test_no_fixed_scalar_centroid_weights_for_expiring_profiles(self):
        # Fixed per-cluster scalar weights require odds(a,b)*odds(b,c)=odds(a,c).
        mass = future_weights(3, 3, 0.8)
        theta = [pair_parameters(mass, a, b)[1] for a, b in [(0, 1), (1, 2), (0, 2)]]
        odds = [x / (1 - x) for x in theta]
        self.assertGreater(abs(odds[0] * odds[1] - odds[2]), 0.15)

    def test_closed_form_matches_explicit_future_composites(self):
        rng = np.random.default_rng(581)
        w = future_weights(6, 8, 0.8)
        assignment = np.array([0, 1, 0, 2, 1, 2])
        mass = w @ np.eye(3)[assignment]
        colors = rng.random((3, 15))
        for a in range(3):
            for b in range(a + 1, 3):
                cost, theta = pair_parameters(mass, a, b)
                c = theta * colors[a] + (1 - theta) * colors[b]
                diff = (
                    (mass[:, a] + mass[:, b])[:, None] * c
                    - mass[:, a, None] * colors[a]
                    - mass[:, b, None] * colors[b]
                )
                self.assertAlmostEqual(
                    np.sum(diff**2) / 15,
                    cost * np.mean((colors[a] - colors[b]) ** 2),
                    places=13,
                )
                for other in [0.0, 1.0, 0.2, 0.8, theta - 0.01, theta + 0.01]:
                    self.assertLessEqual(
                        cost,
                        np.sum((other * (mass[:, a] + mass[:, b]) - mass[:, a]) ** 2)
                        + 1e-14,
                    )

    def test_collinear_profiles_merge_without_perturbation(self):
        mass = np.array([[2.0, 3.0], [4.0, 6.0], [0.0, 0.0]])
        cost, theta = pair_parameters(mass, 0, 1)
        self.assertAlmostEqual(cost, 0)
        self.assertAlmostEqual(theta, 0.4)

    def test_orthogonal_profiles_reduce_to_weighted_centroid(self):
        mass = np.array([[2.0, 0.0], [0.0, 3.0]])
        cost, theta = pair_parameters(mass, 0, 1)
        self.assertAlmostEqual(theta, 4 / 13)
        self.assertAlmostEqual(cost, 36 / 13)

    def test_temporal_overlap_changes_representative(self):
        mass = np.array([[1.0, 1.0], [1.0, 0.0]])
        _, theta = pair_parameters(mass, 0, 1)
        _, independent = pair_parameters(mass, 0, 1, "independent")
        self.assertAlmostEqual(theta, 0.6)
        self.assertAlmostEqual(independent, 2 / 3)

    def test_capacity_and_constant_stream(self):
        model = SynthesisHistory(5, 3, tile=4, history=4, capacity=2)
        im = np.full((3, 5, 4), 0.4)
        for _ in range(12):
            result, stats = model.capture(im)
            np.testing.assert_allclose(result, im, atol=1e-15)
            self.assertEqual(stats["used"], 2)

    def test_packed_pool_clamps_rounds_and_does_not_drift_on_static_input(self):
        model = SynthesisHistory(3, 2, tile=4, history=4, capacity=1, packed=True)
        im = np.broadcast_to(np.array([-0.2, 0.4, 1.2, 1.0]), (2, 3, 4)).copy()
        expected = np.clip(im, 0, 1)
        for _ in range(10):
            result, _ = model.capture(im)
            np.testing.assert_allclose(result, expected, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
