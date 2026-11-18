import unittest
import numpy as np
from analysis.reconstruction_reference import reconstruct_visibility, synthetic


class VisibilityCandidateTests(unittest.TestCase):
    def data(self, kind):
        c0, a0, v0 = synthetic(0, kind)
        c1, a1, v1 = synthetic(1, kind)
        target, _, _ = synthetic(0.5, kind)
        return c0, c1, a0, a1, v0, v1, target

    def test_one_sided_removes_translation_ghost(self):
        c0, c1, a0, a1, v0, v1, target = self.data("translation")
        result, codes = reconstruct_visibility(c0, c1, a0, a1, 0.5, -0.5, v0, v1, 1)
        np.testing.assert_array_equal(result, target)
        self.assertTrue((codes == 2).any())

    def test_depth_preserves_occluder_with_one_side(self):
        c0, c1, a0, a1, v0, v1, target = self.data("occlusion")
        result, _ = reconstruct_visibility(c0, c1, a0, a1, 0.5, -0.5, v0, v1, 1)
        np.testing.assert_array_equal(result, target)

    def test_motion_disagreement_explicitly_uses_crossfade(self):
        c0, c1, a0, a1, v0, v1, target = self.data("acceleration")
        result, codes = reconstruct_visibility(c0, c1, a0, a1, 0.5, -0.5, v0, v1, 2)
        np.testing.assert_array_equal(result, (c0 + c1) / 2)
        self.assertTrue((codes == 3).all())

    def test_hidden_content_still_unrecoverable(self):
        c0, c1, a0, a1, v0, v1, target = self.data("hidden_midpoint")
        result, codes = reconstruct_visibility(c0, c1, a0, a1, 0.5, -0.5, v0, v1, 2)
        self.assertGreater(float(np.mean((result - target) ** 2)), 0)
        self.assertTrue((codes == 1).all())
