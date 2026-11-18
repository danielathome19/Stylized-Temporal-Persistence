import unittest
from analysis.artifact_paths import recorded_path, ROOT


class ArtifactPathTests(unittest.TestCase):
    def test_windows_recording_relocates_to_current_root(self):
        self.assertEqual(
            recorded_path("C:\\old machine\\project\\results\\raw\\run-1\\frame.bin"),
            ROOT / "results/raw/run-1/frame.bin",
        )

    def test_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            recorded_path("C:/old/results/raw/../secret")


if __name__ == "__main__":
    unittest.main()
