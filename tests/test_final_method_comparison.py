import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_final_method_comparison import (  # noqa: E402
    comparison_row,
    markdown_table,
    parse_methods,
)


class FinalMethodComparisonTests(unittest.TestCase):
    def test_parse_rules_and_model(self):
        methods = parse_methods(["rules=RULES", "ordered=model.joblib"])
        self.assertEqual(methods[0], ("rules", None))
        self.assertEqual(methods[1], ("ordered", Path("model.joblib")))

    def test_conditional_span_average(self):
        report = {
            "macro_average": {key: 0.5 for key in (
                "relation_accuracy", "relation_macro_f1", "relation_same_cell_f1",
                "exact_cell_f1", "same_cell_pair_f1", "coordinate_accuracy",
                "span_accuracy", "spanning_cell_f1",
            )},
            "documents": [
                {"document_id": "span_a", "spanning_cell_f1": 1.0},
                {"document_id": "plain", "spanning_cell_f1": 0.0},
                {"document_id": "span_b", "spanning_cell_f1": 0.5},
            ],
        }
        row = comparison_row("m", None, report, {"span_a", "span_b"}, 2.0)
        self.assertEqual(row["spanning_cell_f1_on_span_documents"], 0.75)

    def test_markdown_contains_method(self):
        row = {
            "method": "ordered",
            "relation_macro_f1": 0.5,
            "relation_same_cell_f1": 0.7,
            "exact_cell_f1": 0.9,
            "same_cell_pair_f1": 0.9,
            "coordinate_accuracy": 1.0,
            "span_accuracy": 1.0,
            "spanning_cell_f1_on_span_documents": 0.8,
            "elapsed_seconds": 3.0,
        }
        self.assertIn("ordered", markdown_table([row]))


if __name__ == "__main__":
    unittest.main()
