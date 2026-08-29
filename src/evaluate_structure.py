from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed table structure")
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def prf(true_positive: int, predicted_positive: int, actual_positive: int) -> dict[str, float | int]:
    precision = divide(true_positive, predicted_positive)
    recall = divide(true_positive, actual_positive)
    f1 = divide(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positive": true_positive,
        "predicted_positive": predicted_positive,
        "actual_positive": actual_positive,
    }


def fragment_set(cell: dict[str, Any], ignored: set[Any]) -> frozenset[Any]:
    return frozenset(fragment_id for fragment_id in cell.get("fragment_ids", []) if fragment_id not in ignored)


def pair_set(cells: list[frozenset[Any]]) -> set[frozenset[Any]]:
    result = set()
    for cell in cells:
        for left, right in itertools.combinations(sorted(cell, key=str), 2):
            result.add(frozenset((left, right)))
    return result


def main() -> None:
    args = parse_args()
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    ignored = set(ground_truth.get("ignored_fragment_ids", []))

    predicted_cells = [
        (cell, fragment_set(cell, ignored))
        for cell in prediction["cells"]
        if fragment_set(cell, ignored)
    ]
    actual_cells = [
        (cell, fragment_set(cell, ignored))
        for cell in ground_truth["cells"]
        if fragment_set(cell, ignored)
    ]
    predicted_sets = {fragments for _, fragments in predicted_cells}
    actual_sets = {fragments for _, fragments in actual_cells}
    exact_sets = predicted_sets & actual_sets
    cell_metrics = prf(len(exact_sets), len(predicted_sets), len(actual_sets))

    predicted_pairs = pair_set([fragments for _, fragments in predicted_cells])
    actual_pairs = pair_set([fragments for _, fragments in actual_cells])
    pairwise_metrics = prf(
        len(predicted_pairs & actual_pairs), len(predicted_pairs), len(actual_pairs)
    )

    predicted_by_set = {fragments: cell for cell, fragments in predicted_cells}
    actual_by_set = {fragments: cell for cell, fragments in actual_cells}
    coordinate_correct = 0
    span_correct = 0
    spanning_exact_correct = 0
    exact_details = []
    for fragments in sorted(exact_sets, key=lambda value: sorted(map(str, value))):
        predicted_cell = predicted_by_set[fragments]
        actual_cell = actual_by_set[fragments]
        coordinate_match = all(
            int(predicted_cell[key]) == int(actual_cell[key])
            for key in ("row_start", "row_end", "column_start", "column_end")
        )
        predicted_span = (
            int(predicted_cell.get("rowspan", predicted_cell["row_end"] - predicted_cell["row_start"] + 1)),
            int(predicted_cell.get("colspan", predicted_cell["column_end"] - predicted_cell["column_start"] + 1)),
        )
        actual_span = (
            int(actual_cell["row_end"] - actual_cell["row_start"] + 1),
            int(actual_cell["column_end"] - actual_cell["column_start"] + 1),
        )
        span_match = predicted_span == actual_span
        coordinate_correct += int(coordinate_match)
        span_correct += int(span_match)
        if actual_span != (1, 1) and span_match:
            spanning_exact_correct += 1
        exact_details.append(
            {
                "fragment_ids": sorted(fragments, key=str),
                "text": actual_cell.get("text", ""),
                "coordinate_match": coordinate_match,
                "span_match": span_match,
            }
        )

    overmerged = []
    for predicted_cell, predicted_fragments in predicted_cells:
        overlaps = [
            actual_cell["id"]
            for actual_cell, actual_fragments in actual_cells
            if predicted_fragments & actual_fragments
        ]
        if len(overlaps) > 1:
            overmerged.append(
                {
                    "predicted_cell_id": predicted_cell["id"],
                    "text": predicted_cell.get("text", ""),
                    "fragment_ids": sorted(predicted_fragments, key=str),
                    "ground_truth_cell_ids": overlaps,
                }
            )

    split_cells = []
    for actual_cell, actual_fragments in actual_cells:
        overlaps = [
            predicted_cell["id"]
            for predicted_cell, predicted_fragments in predicted_cells
            if predicted_fragments & actual_fragments
        ]
        if len(overlaps) > 1:
            split_cells.append(
                {
                    "ground_truth_cell_id": actual_cell["id"],
                    "text": actual_cell.get("text", ""),
                    "fragment_ids": sorted(actual_fragments, key=str),
                    "predicted_cell_ids": overlaps,
                }
            )

    gt_row_count = max(cell["row_end"] for cell in ground_truth["cells"]) + 1
    gt_column_count = max(cell["column_end"] for cell in ground_truth["cells"]) + 1
    predicted_spanning_count = sum(
        1
        for cell, _ in predicted_cells
        if int(cell.get("rowspan", 1)) > 1 or int(cell.get("colspan", 1)) > 1
    )
    actual_spanning_count = sum(
        1
        for cell, _ in actual_cells
        if cell["row_end"] > cell["row_start"] or cell["column_end"] > cell["column_start"]
    )
    report = {
        "schema_version": "1.0",
        "task": "table_structure_evaluation",
        "prediction": args.prediction.name,
        "ground_truth": args.ground_truth.name,
        "predicted_grid": {
            "rows": prediction["row_count"],
            "columns": prediction["column_count"],
        },
        "ground_truth_grid": {"rows": gt_row_count, "columns": gt_column_count},
        "row_count_error": abs(prediction["row_count"] - gt_row_count),
        "column_count_error": abs(prediction["column_count"] - gt_column_count),
        "predicted_nonempty_cell_count": len(predicted_cells),
        "ground_truth_nonempty_cell_count": len(actual_cells),
        "exact_cell_metrics": cell_metrics,
        "same_cell_pair_metrics": pairwise_metrics,
        "coordinate_accuracy_on_exact_cells": round(
            divide(coordinate_correct, len(exact_sets)), 6
        ),
        "span_accuracy_on_exact_cells": round(divide(span_correct, len(exact_sets)), 6),
        "spanning_cell_metrics": prf(
            spanning_exact_correct, predicted_spanning_count, actual_spanning_count
        ),
        "overmerged_cell_count": len(overmerged),
        "split_ground_truth_cell_count": len(split_cells),
        "overmerged_cells": overmerged,
        "split_ground_truth_cells": split_cells,
        "exact_cell_details": exact_details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Predicted grid: {prediction['row_count']}x{prediction['column_count']}")
    print(f"Ground-truth grid: {gt_row_count}x{gt_column_count}")
    print(
        f"Exact Cell F1: {cell_metrics['f1']:.3f} "
        f"(P={cell_metrics['precision']:.3f}, R={cell_metrics['recall']:.3f})"
    )
    print(
        f"SAME_CELL pair F1: {pairwise_metrics['f1']:.3f} "
        f"(P={pairwise_metrics['precision']:.3f}, R={pairwise_metrics['recall']:.3f})"
    )
    print(f"Coordinate accuracy: {report['coordinate_accuracy_on_exact_cells']:.3f}")
    print(f"Span accuracy: {report['span_accuracy_on_exact_cells']:.3f}")
    print(
        f"Spanning Cell F1: {report['spanning_cell_metrics']['f1']:.3f} "
        f"(P={report['spanning_cell_metrics']['precision']:.3f}, "
        f"R={report['spanning_cell_metrics']['recall']:.3f})"
    )
    print(f"Overmerged cells: {len(overmerged)}")
    print(f"Split ground-truth cells: {len(split_cells)}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
