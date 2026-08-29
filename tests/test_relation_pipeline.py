import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_relation_baseline import classify, pair_features  # noqa: E402
from label_relations_from_cells import ground_truth_relation  # noqa: E402
from convert_scitsr import (  # noqa: E402
    label_pair,
    load_chunks,
    load_relations,
    split_text,
    synthetic_same_cell_examples,
)
from relation_features import text_features  # noqa: E402
from relation_features import semantic_features  # noqa: E402
from reconstruct_table_structure import UnionFind, join_text  # noqa: E402
from evaluate_structure import pair_set, prf  # noqa: E402
from reconstruct_table_structure_hybrid import (  # noqa: E402
    aligned_row_bands,
    boundary_rows,
    constrain_same_cell_edges,
    detect_data_top,
    infer_header_spans_from_occupancy,
    infer_logical_header_depth,
    inferred_header_children,
    inferred_missing_header_cells,
    merge_header_components,
    ordered_kmeans,
)
from export_reconstructed_html import build_html  # noqa: E402
from build_benchmark_annotations import BUILDERS  # noqa: E402
from build_heldout_annotations import BUILDERS as HELDOUT_BUILDERS  # noqa: E402
from run_structure_benchmark import mean  # noqa: E402
from detect_table_grid_light import (  # noqa: E402
    complete_boundaries,
    group_positions as group_line_positions,
)
from reconstruct_table_from_grid import (  # noqa: E402
    infer_grid_slot_groups,
    interval_index,
    reconstruct as reconstruct_from_grid,
)
from auto_reconstruct_table import (  # noqa: E402
    assess_grid,
    augment_partial_header_rows,
    estimate_ocr_row_bands,
    select_automatic_mode,
)
from relation_model import TwoStageExtraTreesClassifier  # noqa: E402
from build_domain_relation_dataset import examples_for_document  # noqa: E402
import numpy as np
import json
from PIL import Image, ImageDraw


class RelationPipelineTests(unittest.TestCase):
    def test_two_stage_classifier_probability_shape(self):
        x = np.asarray(
            [[0.0, 0.0], [0.1, 0.1], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]] * 4
        )
        y = np.asarray(
            ["SAME_CELL", "SAME_CELL", "RIGHT", "BELOW", "NO_RELATION"] * 4
        )
        model = TwoStageExtraTreesClassifier(n_estimators=10, random_state=1)
        model.fit(x, y)
        probabilities = model.predict_proba(x)
        self.assertEqual(probabilities.shape, (20, 4))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_domain_examples_respect_ignored_fragments(self):
        ocr = {
            "image_size": {"width": 100, "height": 100},
            "fragments": [
                {"id": 1, "text": "A", "bbox": [0, 0, 10, 10]},
                {"id": 2, "text": "B", "bbox": [12, 0, 22, 10]},
                {"id": 3, "text": "noise", "bbox": [50, 50, 60, 60]},
            ],
        }
        annotation = {
            "ignored_fragment_ids": [3],
            "cells": [
                {"id": "a", "row_start": 0, "row_end": 0, "column_start": 0, "column_end": 0, "fragment_ids": [1, 2]},
            ],
        }
        examples = examples_for_document("demo", ocr, annotation, 4, 1.0)
        self.assertTrue(examples)
        self.assertTrue(all(3 not in (item["source_fragment_id"], item["target_fragment_id"]) for item in examples))
        self.assertEqual(examples[0]["label"], "SAME_CELL")
    def test_close_fragments_are_same_cell(self):
        left = {"id": 1, "text": "ПОС", "bbox": [10, 10, 70, 40]}
        right = {"id": 2, "text": "61", "bbox": [78, 10, 108, 40]}
        prediction, _ = classify(pair_features(left, right))
        self.assertEqual(prediction, "SAME_CELL")
        features = pair_features(left, right)
        self.assertIn("center_distance_norm", features)

    def test_fragment_below(self):
        upper = {"id": 1, "text": "183", "bbox": [100, 10, 150, 40]}
        lower = {"id": 2, "text": "238", "bbox": [100, 80, 150, 110]}
        prediction, _ = classify(pair_features(upper, lower))
        self.assertEqual(prediction, "BELOW")

    def test_ground_truth_right(self):
        left = {
            "id": "a",
            "row_start": 1,
            "row_end": 1,
            "column_start": 0,
            "column_end": 0,
        }
        right = {
            "id": "b",
            "row_start": 1,
            "row_end": 1,
            "column_start": 1,
            "column_end": 1,
        }
        self.assertEqual(ground_truth_relation(left, right), "RIGHT")

    def test_spanning_cell_is_above_child_cell(self):
        spanning = {
            "id": "header",
            "row_start": 0,
            "row_end": 0,
            "column_start": 2,
            "column_end": 3,
        }
        child = {
            "id": "child",
            "row_start": 1,
            "row_end": 1,
            "column_start": 3,
            "column_end": 3,
        }
        self.assertEqual(ground_truth_relation(spanning, child), "BELOW")

    def test_scitsr_conversion(self):
        fixture = PROJECT_ROOT / "tests" / "fixtures" / "scitsr"
        chunks = load_chunks(fixture / "chunk" / "demo.chunk")
        relations = load_relations(fixture / "rel" / "demo.rel")
        self.assertEqual(chunks[0]["bbox"], [10.0, -120.0, 60.0, -100.0])
        self.assertEqual(label_pair(0, 1, relations), "RIGHT")
        self.assertEqual(label_pair(0, 2, relations), "BELOW")
        self.assertEqual(label_pair(1, 2, relations), "NO_RELATION")

    def test_synthetic_same_cell_is_reproducible(self):
        chunks = [{"id": 0, "text": "Probability", "bbox": [10, 10, 110, 30]}]
        first = list(synthetic_same_cell_examples("demo", chunks, 1.0))
        second = list(synthetic_same_cell_examples("demo", chunks, 1.0))
        self.assertEqual(first, second)
        self.assertEqual(first[0]["label"], "SAME_CELL")

    def test_split_text(self):
        self.assertEqual(split_text("Generated Text"), ("Generated", "Text"))
        self.assertEqual(split_text("Probability"), ("Proba", "bility"))

    def test_text_features_for_decimal_fragments(self):
        features = text_features("42,", "1")
        self.assertEqual(features["source_ends_decimal_separator"], 1.0)
        self.assertEqual(features["concatenated_numeric"], 1.0)

    def test_text_features_for_hyphenated_word(self):
        features = text_features("Плот-", "ность")
        self.assertEqual(features["source_ends_hyphen"], 1.0)
        self.assertEqual(features["hyphen_join_is_alpha"], 1.0)

    def test_semantic_features(self):
        same = semantic_features(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        opposite = semantic_features(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
        self.assertAlmostEqual(same["embedding_cosine_similarity"], 1.0)
        self.assertAlmostEqual(same["embedding_l2_distance"], 0.0)
        self.assertAlmostEqual(opposite["embedding_cosine_similarity"], -1.0)

    def test_union_find(self):
        groups = UnionFind([1, 2, 3])
        groups.union(1, 2)
        self.assertEqual(groups.find(1), groups.find(2))
        self.assertNotEqual(groups.find(1), groups.find(3))

    def test_join_decimal_fragments(self):
        fragments = [
            {"id": 1, "text": "42,", "bbox": [10, 10, 40, 30]},
            {"id": 2, "text": "1", "bbox": [45, 10, 55, 30]},
        ]
        self.assertEqual(join_text(fragments), "42,1")

    def test_pairwise_cell_membership(self):
        pairs = pair_set([frozenset((1, 2, 3)), frozenset((4,))])
        self.assertEqual(len(pairs), 3)

    def test_precision_recall_f1(self):
        metrics = prf(true_positive=2, predicted_positive=4, actual_positive=5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.4)
        self.assertAlmostEqual(metrics["f1"], 0.444444)

    def test_detect_data_top(self):
        fragments = [
            {"bbox": [0, 10, 10, 20]},
            {"bbox": [0, 30, 10, 40]},
            {"bbox": [0, 100, 10, 110]},
            {"bbox": [0, 120, 10, 130]},
        ]
        self.assertGreater(detect_data_top(fragments), 40)
        self.assertLess(detect_data_top(fragments), 100)

    def test_high_confidence_vertical_same_cell_is_preserved(self):
        fragments = [
            {"id": 1, "bbox": [10, 100, 50, 120]},
            {"id": 2, "bbox": [10, 125, 50, 145]},
        ]
        relation = {
            "source_fragment_id": 1,
            "target_fragment_id": 2,
            "prediction": "SAME_CELL",
            "probabilities": {"SAME_CELL": 0.9},
        }
        filtered, rejected = constrain_same_cell_edges([relation], fragments, 50, 0.65)
        self.assertEqual(filtered[0]["prediction"], "SAME_CELL")
        self.assertEqual(rejected, [])

    def test_low_confidence_vertical_same_cell_is_rejected(self):
        fragments = [
            {"id": 1, "bbox": [10, 100, 50, 120]},
            {"id": 2, "bbox": [10, 125, 50, 145]},
        ]
        relation = {
            "source_fragment_id": 1,
            "target_fragment_id": 2,
            "prediction": "SAME_CELL",
            "probabilities": {"SAME_CELL": 0.55},
        }
        filtered, rejected = constrain_same_cell_edges([relation], fragments, 50, 0.65)
        self.assertEqual(filtered[0]["prediction"], "NO_RELATION")
        self.assertEqual(len(rejected), 1)

    def test_geometry_supports_medium_confidence_vertical_same_cell(self):
        fragments = [
            {"id": 1, "bbox": [10, 100, 50, 120]},
            {"id": 2, "bbox": [11, 125, 51, 145]},
        ]
        relation = {
            "source_fragment_id": 1,
            "target_fragment_id": 2,
            "prediction": "SAME_CELL",
            "probabilities": {"SAME_CELL": 0.55},
        }
        filtered, rejected = constrain_same_cell_edges(
            [relation], fragments, 50, 0.65, 0.45, 0.55, 1.5, []
        )
        self.assertEqual(filtered[0]["prediction"], "SAME_CELL")
        self.assertEqual(
            filtered[0]["vertical_same_cell_decision"],
            "geometry_supported_soft_probability",
        )
        self.assertEqual(rejected, [])

    def test_row_boundary_blocks_soft_vertical_same_cell(self):
        fragments = [
            {"id": 1, "bbox": [10, 100, 50, 120]},
            {"id": 2, "bbox": [11, 125, 51, 145]},
        ]
        relation = {
            "source_fragment_id": 1,
            "target_fragment_id": 2,
            "prediction": "SAME_CELL",
            "probabilities": {"SAME_CELL": 0.55},
        }
        filtered, rejected = constrain_same_cell_edges(
            [relation], fragments, 50, 0.65, 0.45, 0.55, 1.5, [122.0]
        )
        self.assertEqual(filtered[0]["prediction"], "NO_RELATION")
        self.assertEqual(rejected[0]["reason"], "row_boundary_between")

    def test_row_boundary_blocks_high_confidence_vertical_same_cell(self):
        fragments = [
            {"id": 1, "text": "A", "bbox": [0, 0, 20, 10]},
            {"id": 2, "text": "B", "bbox": [0, 30, 20, 40]},
        ]
        relations = [{
            "source_fragment_id": 1,
            "target_fragment_id": 2,
            "prediction": "SAME_CELL",
            "probabilities": {"SAME_CELL": 0.99},
        }]
        filtered, rejected = constrain_same_cell_edges(
            relations,
            fragments,
            data_top=-1.0,
            row_boundaries=[20.0],
        )
        self.assertEqual(filtered[0]["prediction"], "NO_RELATION")
        self.assertEqual(rejected[0]["reason"], "row_boundary_between")

    def test_horizontal_misalignment_blocks_soft_vertical_same_cell(self):
        fragments = [
            {"id": 1, "bbox": [10, 100, 50, 120]},
            {"id": 2, "bbox": [80, 125, 120, 145]},
        ]
        relation = {
            "source_fragment_id": 1,
            "target_fragment_id": 2,
            "prediction": "SAME_CELL",
            "probabilities": {"SAME_CELL": 0.55},
        }
        filtered, rejected = constrain_same_cell_edges(
            [relation], fragments, 50, 0.65, 0.45, 0.55, 1.5, []
        )
        self.assertEqual(filtered[0]["prediction"], "NO_RELATION")
        self.assertEqual(rejected[0]["reason"], "insufficient_x_overlap")

    def test_ordered_kmeans(self):
        labels, centers = ordered_kmeans([101, 99, 301, 299], 2, 42)
        self.assertEqual(labels, [0, 0, 1, 1])
        self.assertLess(centers[0], centers[1])

    def test_merge_multiline_header_components(self):
        fragments = [
            {"id": 1, "text": "Плот-", "bbox": [100, 10, 150, 30]},
            {"id": 2, "text": "ность", "bbox": [100, 35, 150, 55]},
            {"id": 3, "text": "8,5", "bbox": [100, 100, 140, 120]},
        ]
        components = [
            {"id": "a", "fragment_ids": [1], "bbox": fragments[0]["bbox"], "center": [125, 20]},
            {"id": "b", "fragment_ids": [2], "bbox": fragments[1]["bbox"], "center": [125, 45]},
            {"id": "c", "fragment_ids": [3], "bbox": fragments[2]["bbox"], "center": [120, 110]},
        ]
        merged, trace = merge_header_components(components, [0, 1], [4, 4, 4], fragments)
        header = next(item for item in merged if item.get("header_line_merge"))
        self.assertEqual(header["fragment_ids"], [1, 2])
        self.assertEqual(len(trace), 1)

    def test_infer_two_level_missing_header(self):
        self.assertEqual(infer_logical_header_depth({0, 1, 4, 5}, 6, 3), 2)
        cells = inferred_missing_header_cells({0, 1, 4, 5}, 6, 2)
        self.assertEqual(len(cells), 3)
        self.assertEqual(cells[0]["colspan"], 2)
        self.assertEqual(infer_logical_header_depth({0, 2}, 5, 1), 2)
        children = inferred_header_children([1, 2, 3, 4], 2)
        self.assertEqual(len(children), 4)
        self.assertTrue(all(cell["row_start"] == 1 for cell in children))

    def test_aligned_row_bands_ignores_sparse_multiline_noise(self):
        components = []
        for center_y in (100, 200, 300):
            for center_x in (10, 110, 210, 310):
                components.append(
                    {"center": [center_x, center_y], "bbox": [center_x, center_y - 10, center_x + 40, center_y + 10]}
                )
        components.append({"center": [10, 145], "bbox": [10, 135, 50, 155]})
        labels, centers = aligned_row_bands(components, list(range(len(components))))
        self.assertEqual(len(centers), 3)
        self.assertEqual(labels[-1], 0)

    def test_html_export_escapes_text(self):
        structure = {
            "method": "test",
            "source_relations": "test.json",
            "row_count": 1,
            "column_count": 1,
            "warnings": [],
            "cells": [
                {
                    "id": "cell_0",
                    "text": "A < B",
                    "fragment_ids": [1],
                    "row_start": 0,
                    "column_start": 0,
                    "rowspan": 1,
                    "colspan": 1,
                }
            ],
        }
        rendered = build_html(structure, "Test", True)
        self.assertIn("A &lt; B", rendered)
        self.assertNotIn("A < B", rendered)

    def test_benchmark_annotations_cover_all_fragments(self):
        expected_cells = {
            "table_03_payment_schedule": 52,
            "table_05_impact_strength": 14,
            "table_06_element_tolerances": 10,
            "table_08_composition_long": 116,
            "table_09_tax_elements": 36,
        }
        for document_id, builder in BUILDERS.items():
            path = PROJECT_ROOT / "data" / "ocr" / f"{document_id}.json"
            ocr = json.loads(path.read_text(encoding="utf-8"))
            result = builder(ocr)
            self.assertEqual(len(result["cells"]), expected_cells[document_id])

    def test_heldout_annotations_cover_all_fragments(self):
        expected_cells = {
            "table_01_solders_composition": 43,
            "table_05_hardness": 73,
            "table_06_mechanical_properties": 124,
            "table_07_heat_treatment": 26,
        }
        for document_id, builder in HELDOUT_BUILDERS.items():
            path = PROJECT_ROOT / "data" / "ocr" / f"{document_id}.json"
            ocr = json.loads(path.read_text(encoding="utf-8"))
            result = builder(ocr)
            self.assertEqual(len(result["cells"]), expected_cells[document_id])

    def test_benchmark_mean(self):
        self.assertEqual(mean([{"score": 0.5}, {"score": 1.0}], "score"), 0.75)

    def test_light_grid_position_grouping(self):
        self.assertEqual(group_line_positions([9, 10, 11, 100, 101]), [10, 100])

    def test_grid_cell_assignment(self):
        ocr = {
            "source_image": "demo.png",
            "fragments": [
                {"id": 1, "text": "A", "bbox": [10, 10, 20, 20], "confidence": 1.0},
                {"id": 2, "text": "B", "bbox": [60, 60, 70, 70], "confidence": 1.0},
                {"id": 3, "text": "|", "bbox": [49, 0, 51, 100], "confidence": 1.0},
            ],
        }
        ocr["image_size"] = {"width": 100, "height": 100}
        grid = {
            "source_image": "demo.png", "image_size": {"width": 100, "height": 100},
            "row_boundaries": [0, 50, 100], "column_boundaries": [0, 50, 100]
        }
        result = reconstruct_from_grid(ocr, grid, 0.0)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["column_count"], 2)
        self.assertEqual(result["cells"][0]["fragment_ids"], [1])
        self.assertEqual(interval_index(75, [0, 50, 100]), 1)
        self.assertIn(3, result["ignored_fragment_ids"])

    def test_grid_missing_border_creates_rowspan(self):
        image = Image.new("RGB", (101, 101), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 100, 100), outline="black", width=3)
        draw.line((50, 0, 50, 100), fill="black", width=3)
        draw.line((50, 50, 100, 50), fill="black", width=3)
        ocr = {
            "source_image": "span.png",
            "image_size": {"width": 101, "height": 101},
            "fragments": [
                {"id": 1, "text": "A", "bbox": [10, 20, 20, 30], "confidence": 1.0},
                {"id": 2, "text": "B", "bbox": [70, 20, 80, 30], "confidence": 1.0},
                {"id": 3, "text": "C", "bbox": [70, 70, 80, 80], "confidence": 1.0},
            ],
        }
        grid = {
            "source_image": "span.png",
            "image_size": {"width": 101, "height": 101},
            "row_boundaries": [0, 50, 100],
            "column_boundaries": [0, 50, 100],
        }
        result = reconstruct_from_grid(ocr, grid, 0.0, image=image)
        spanning = next(cell for cell in result["cells"] if cell["fragment_ids"] == [1])
        self.assertEqual(spanning["rowspan"], 2)
        self.assertEqual(spanning["colspan"], 1)
        self.assertEqual(result["logical_cell_count"], 3)

    def test_partial_grid_grouped_header_spans(self):
        components = [
            {"id": "left"}, {"id": "group"}, {"id": "right"},
            {"id": "a"}, {"id": "b"}, {"id": "c"},
            *({"id": f"body_{index}"} for index in range(6)),
        ]
        rows = [0, 0, 0, 1, 1, 1, *([2] * 6)]
        columns = [0, 3, 5, 2, 3, 4, *range(6)]
        rowspans = [1] * len(components)
        colspans = [1] * len(components)
        trace = infer_header_spans_from_occupancy(
            components, rows, columns, rowspans, colspans, 3, 6
        )
        self.assertEqual(columns[1], 2)
        self.assertEqual(colspans[1], 3)
        self.assertEqual(rowspans[0], 2)
        self.assertEqual(rowspans[2], 2)
        self.assertTrue(trace)

    def test_auto_selector_accepts_complete_grid(self):
        result = assess_grid(
            width=1000, height=800,
            vertical_lines=[0, 300, 700, 999],
            horizontal_lines=[0, 150, 300, 500, 799],
            column_boundaries=[0, 300, 700, 999],
            row_boundaries=[0, 150, 300, 500, 799],
            minimum_rows=3, minimum_columns=2,
            minimum_axis_span=0.45, maximum_band_ratio=4.0,
        )
        self.assertEqual(result["selected_mode"], "grid")

    def test_multiline_ocr_does_not_override_complete_grid(self):
        assessment = assess_grid(
            width=1710, height=1426,
            vertical_lines=[4, 360, 592, 754, 909, 1087, 1709],
            horizontal_lines=[5, 179, 390, 564, 701, 912, 1418],
            column_boundaries=[4, 360, 592, 754, 909, 1087, 1709],
            row_boundaries=[5, 179, 390, 564, 701, 912, 1418],
            minimum_rows=3, minimum_columns=2,
            minimum_axis_span=0.45, maximum_band_ratio=4.0,
        )
        mode, _ = select_automatic_mode(
            assessment, estimated_ocr_rows=34, minimum_partial_grid_columns=8
        )
        self.assertEqual(mode, "grid")

    def test_auto_selector_rejects_header_only_lines(self):
        result = assess_grid(
            width=1000, height=1000,
            vertical_lines=[0, 300, 700, 999],
            horizontal_lines=[0, 80, 160],
            column_boundaries=[0, 300, 700, 999],
            row_boundaries=[0, 80, 160, 999],
            minimum_rows=3, minimum_columns=2,
            minimum_axis_span=0.45, maximum_band_ratio=4.0,
        )
        self.assertEqual(result["selected_mode"], "hybrid")

    def test_auto_selector_accepts_complete_grid_with_unequal_row_heights(self):
        assessment = assess_grid(
            width=2000, height=1200,
            vertical_lines=[8, 250, 1990],
            horizontal_lines=[6, 120, 350, 395, 475, 1180],
            column_boundaries=[8, 250, 1990],
            row_boundaries=[6, 120, 350, 395, 475, 1180],
            minimum_rows=3, minimum_columns=2,
            minimum_axis_span=0.45, maximum_band_ratio=4.0,
        )
        self.assertFalse(assessment["checks"]["distributed_rows"])
        self.assertEqual(assessment["selected_mode"], "grid")

    def test_auto_selector_uses_horizontal_partial_grid(self):
        assessment = assess_grid(
            width=2000, height=1000,
            vertical_lines=[],
            horizontal_lines=[100, 200, 300, 400, 900],
            column_boundaries=[0, 1999],
            row_boundaries=[0, 100, 200, 300, 400, 900, 999],
            minimum_rows=3, minimum_columns=2,
            minimum_axis_span=0.45, maximum_band_ratio=4.0,
        )
        mode, _ = select_automatic_mode(
            assessment, estimated_ocr_rows=8, minimum_partial_grid_columns=2
        )
        self.assertEqual(mode, "partial_grid")

    def test_complete_boundaries_scales_edge_margin(self):
        self.assertEqual(
            complete_boundaries([14, 644, 1042, 1490], 1506),
            [14, 644, 1042, 1490],
        )
        self.assertEqual(complete_boundaries([14, 244, 564], 574), [14, 244, 564])

    def test_partial_header_row_augmentation(self):
        document = {
            "fragments": [
                {"text": "Group", "bbox": [0, 10, 20, 30]},
                {"text": "A", "bbox": [0, 60, 20, 80]},
                {"text": "B", "bbox": [30, 61, 50, 81]},
            ]
        }
        self.assertEqual(
            augment_partial_header_rows(document, [0, 100, 200], 0.0),
            [0, 45, 100, 200],
        )

    def test_boundary_rows(self):
        components = [
            {"center": [10.0, 20.0]},
            {"center": [10.0, 75.0]},
        ]
        self.assertEqual(boundary_rows(components, [0.0, 50.0, 100.0]), [0, 1])

    def test_estimate_ocr_row_bands(self):
        document = {
            "fragments": [
                {"id": 1, "text": "A", "bbox": [0, 10, 10, 20]},
                {"id": 2, "text": "B", "bbox": [20, 11, 30, 21]},
                {"id": 3, "text": "C", "bbox": [0, 50, 10, 60]},
                {"id": 4, "text": "|", "bbox": [40, 0, 42, 100]},
            ]
        }
        self.assertEqual(estimate_ocr_row_bands(document, 0.0), 2)


if __name__ == "__main__":
    unittest.main()
