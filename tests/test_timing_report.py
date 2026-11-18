import unittest
from analysis.timing_report import validate_samples


class TimingAuditTests(unittest.TestCase):
    def data(self):
        return {
            "submitted": 3,
            "records": [
                {"index": i, "measured": True, "gpu_ns": 5, "cpu_submission_us": 1}
                for i in [1, 2]
            ],
            "unresolved": [{"index": 0, "measured": False}],
        }

    def test_missing_warmup_is_explicitly_allowed(self):
        self.assertEqual(len(validate_samples(self.data(), 1)), 2)

    def test_missing_measured_rejected(self):
        d = self.data()
        d["records"].pop()
        with self.assertRaises(ValueError):
            validate_samples(d, 1)

    def test_duplicate_rejected(self):
        d = self.data()
        d["records"].append(d["records"][0])
        with self.assertRaises(ValueError):
            validate_samples(d, 1)

    def test_negative_rejected(self):
        d = self.data()
        d["records"][0]["gpu_ns"] = -1
        with self.assertRaises(ValueError):
            validate_samples(d, 1)

    def test_time_based_warmup_uses_first_measured_index(self):
        d = self.data()
        d["first_measured"] = 2
        d["records"] = d["records"][1:]
        self.assertEqual(len(validate_samples(d, 1)), 1)

    def test_disabled_timestamps_are_null_not_zero(self):
        d = self.data()
        for r in d["records"]:
            r["gpu_ns"] = None
        self.assertEqual(len(validate_samples(d, 1)), 2)
