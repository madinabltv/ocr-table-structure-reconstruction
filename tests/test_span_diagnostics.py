import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.analyze_span_errors import analyse_spans, render_preview


class SpanDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.ground_truth = {
            "document_id": "fixture",
            "cells": [
                {
                    "id": "gt_a",
                    "row_start": 0,
                    "row_end": 1,
                    "column_start": 0,
                    "column_end": 0,
                    "fragment_ids": [1],
                    "text": "vertical",
                },
                {
                    "id": "gt_b",
                    "row_start": 0,
                    "row_end": 0,
                    "column_start": 1,
                    "column_end": 2,
                    "fragment_ids": [2],
                    "text": "horizontal",
                },
                {
                    "id": "gt_c",
                    "row_start": 1,
                    "row_end": 1,
                    "column_start": 1,
                    "column_end": 1,
                    "fragment_ids": [3],
                    "text": "regular",
                },
            ],
        }
        self.prediction = {
            "document_id": "fixture",
            "row_count": 2,
            "column_count": 3,
            "cells": [
                {
                    "id": "pred_a",
                    "row_start": 0,
                    "row_end": 1,
                    "column_start": 0,
                    "column_end": 0,
                    "fragment_ids": [1],
                    "text": "vertical",
                },
                {
                    "id": "pred_extra",
                    "row_start": 0,
                    "row_end": 1,
                    "column_start": 2,
                    "column_end": 2,
                    "fragment_ids": [4],
                    "text": "extra",
                },
            ],
        }

    def test_reports_exact_missed_and_extra_spans(self):
        report = analyse_spans(self.prediction, self.ground_truth)
        self.assertEqual(report["status_counts"], {"exact": 1, "missed": 1, "extra": 1})
        self.assertEqual(report["evaluator_compatible_metrics"]["f1"], 0.5)

    def test_renders_preview(self):
        report = analyse_spans(self.prediction, self.ground_truth)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.png"
            render_preview(report, output, 60, 36)
            self.assertTrue(output.exists())
            with Image.open(output) as image:
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)

    def test_ignores_ocr_empty_spanning_cells_like_structure_evaluator(self):
        self.ground_truth["cells"].append(
            {
                "id": "empty_span",
                "row_start": 0,
                "row_end": 0,
                "column_start": 0,
                "column_end": 2,
                "fragment_ids": [],
                "text": "",
            }
        )
        report = analyse_spans(self.prediction, self.ground_truth)
        self.assertEqual(report["evaluator_compatible_metrics"]["actual_positive"], 2)


if __name__ == "__main__":
    unittest.main()
