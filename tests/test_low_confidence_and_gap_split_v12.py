import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reconstruct_table_structure_hybrid import (
    coalesce_identical_logical_cells,
    split_components_on_horizontal_gaps,
)


def fragment(identifier, text, x1, x2, y1=10, y2=26):
    return {
        "id": identifier,
        "text": text,
        "bbox": [x1, y1, x2, y2],
        "confidence": 1.0,
    }


class LowConfidenceAndGapSplitTests(unittest.TestCase):
    def test_splits_classifier_merge_at_outlying_column_gap(self):
        fragments = [
            fragment(4, "Регистрационный", 264, 451),
            fragment(5, "номер", 459, 523),
            fragment(6, "выпуска", 530, 617),
            fragment(7, "Эмитент", 654, 742),
        ]
        component = {
            "id": "cell_1",
            "fragment_ids": [4, 5, 6, 7],
            "text": "Регистрационный номер выпуска Эмитент",
            "bbox": [264, 10, 742, 26],
            "center": [503, 18],
        }
        parts, trace = split_components_on_horizontal_gaps([component], fragments)
        self.assertEqual([part["fragment_ids"] for part in parts], [[4, 5, 6], [7]])
        self.assertEqual(len(trace), 1)

    def test_does_not_split_normal_word_spacing(self):
        fragments = [
            fragment(1, "Цена", 100, 150),
            fragment(2, "ценной", 158, 220),
            fragment(3, "бумаги", 228, 290),
        ]
        component = {
            "id": "cell",
            "fragment_ids": [1, 2, 3],
            "text": "Цена ценной бумаги",
            "bbox": [100, 10, 290, 26],
            "center": [195, 18],
        }
        parts, trace = split_components_on_horizontal_gaps([component], fragments)
        self.assertEqual(len(parts), 1)
        self.assertEqual(trace, [])

    def test_does_not_split_multiline_component(self):
        fragments = [
            fragment(1, "Первая", 100, 150, 10, 26),
            fragment(2, "строка", 158, 220, 10, 26),
            fragment(3, "вторая", 100, 160, 34, 50),
        ]
        component = {
            "id": "cell",
            "fragment_ids": [1, 2, 3],
            "text": "Первая строка вторая",
            "bbox": [100, 10, 220, 50],
            "center": [160, 30],
        }
        parts, trace = split_components_on_horizontal_gaps([component], fragments)
        self.assertEqual(len(parts), 1)
        self.assertEqual(trace, [])

    def test_coalesces_low_confidence_fragment_in_same_logical_cell(self):
        fragments = [
            fragment(18, "Механизм", 100, 180),
            {**fragment(19, "~~", 186, 200), "confidence": 0.43},
        ]
        common = {
            "row_start": 0,
            "row_end": 1,
            "column_start": 8,
            "column_end": 8,
            "rowspan": 2,
            "colspan": 1,
        }
        components = [
            {
                **common,
                "id": "cell_18",
                "fragment_ids": [18],
                "text": "Механизм",
                "bbox": [100, 10, 180, 26],
                "center": [140, 18],
            },
            {
                **common,
                "id": "cell_19",
                "fragment_ids": [19],
                "text": "~~",
                "bbox": [186, 10, 200, 26],
                "center": [193, 18],
            },
        ]
        merged, trace = coalesce_identical_logical_cells(components, fragments)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["fragment_ids"], [18, 19])
        self.assertEqual(len(trace), 1)


if __name__ == "__main__":
    unittest.main()
