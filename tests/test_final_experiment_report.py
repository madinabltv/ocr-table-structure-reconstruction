import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_final_experiment_report import tied_ranks  # noqa: E402


class FinalExperimentReportTests(unittest.TestCase):
    def test_competition_ranking_preserves_ties(self):
        self.assertEqual(tied_ranks([0.972, 0.972, 0.972, 0.965]), [1, 1, 1, 4])

    def test_ranking_orders_distinct_values(self):
        self.assertEqual(tied_ranks([0.8, 1.0, 0.9]), [3, 1, 2])


if __name__ == "__main__":
    unittest.main()
