import unittest
from analysis.innovation_memory import allocation, capacity_for_budget


class InnovationMemoryTests(unittest.TestCase):
    def test_matched_budget_includes_workspace_and_maximal_whole_slot(self):
        budget = 373084
        cap = capacity_for_budget(160, 90, budget)
        self.assertEqual(cap, 763)
        self.assertEqual(sum(allocation(160, 90, cap).values()), budget)
        self.assertGreater(sum(allocation(160, 90, cap + 1).values()), budget)
        self.assertEqual(sum(allocation(160, 90, 5944).values()), 559600)

    def test_budget_below_fixed_buffers_cannot_hold_events(self):
        self.assertEqual(capacity_for_budget(160, 90, 345615), 0)

    def test_odd_dimensions_no_implicit_tile_padding(self):
        self.assertEqual(sum(allocation(17, 9, 31).values()), 4804)


if __name__ == "__main__":
    unittest.main()
