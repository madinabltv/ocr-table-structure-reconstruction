from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse rowspan/colspan errors")
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--cell-width", type=int, default=86)
    parser.add_argument("--cell-height", type=int, default=46)
    return parser.parse_args()


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def prf(true_positive: int, predicted_positive: int, actual_positive: int) -> dict[str, Any]:
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


def coordinates(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    row_start = int(cell["row_start"])
    column_start = int(cell["column_start"])
    row_end = int(cell.get("row_end", row_start + int(cell.get("rowspan", 1)) - 1))
    column_end = int(
        cell.get("column_end", column_start + int(cell.get("colspan", 1)) - 1)
    )
    return row_start, row_end, column_start, column_end


def span(cell: dict[str, Any]) -> tuple[int, int]:
    row_start, row_end, column_start, column_end = coordinates(cell)
    return row_end - row_start + 1, column_end - column_start + 1


def fragment_set(cell: dict[str, Any], ignored: set[Any]) -> frozenset[Any]:
    return frozenset(
        fragment_id
        for fragment_id in cell.get("fragment_ids", [])
        if fragment_id not in ignored
    )


def is_spanning(cell: dict[str, Any]) -> bool:
    return span(cell) != (1, 1)


def cell_record(cell: dict[str, Any], ignored: set[Any]) -> dict[str, Any]:
    row_start, row_end, column_start, column_end = coordinates(cell)
    return {
        "cell_id": cell.get("id"),
        "text": cell.get("text", ""),
        "fragment_ids": sorted(fragment_set(cell, ignored), key=str),
        "row_start": row_start,
        "row_end": row_end,
        "column_start": column_start,
        "column_end": column_end,
        "rowspan": row_end - row_start + 1,
        "colspan": column_end - column_start + 1,
    }


def jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    return divide(len(left & right), len(left | right))


def analyse_spans(prediction: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    ignored = set(ground_truth.get("ignored_fragment_ids", []))

    actual = [
        cell
        for cell in ground_truth["cells"]
        if is_spanning(cell) and fragment_set(cell, ignored)
    ]
    predicted = [
        cell
        for cell in prediction["cells"]
        if is_spanning(cell) and fragment_set(cell, ignored)
    ]
    predicted_fragments = [fragment_set(cell, ignored) for cell in predicted]
    used_predictions: set[int] = set()
    comparisons: list[dict[str, Any]] = []
    evaluator_true_positive = 0
    coordinate_true_positive = 0

    for actual_cell in actual:
        actual_fragments = fragment_set(actual_cell, ignored)
        exact_fragment_indices = [
            index
            for index, fragments in enumerate(predicted_fragments)
            if index not in used_predictions and fragments == actual_fragments
        ]
        match_index: int | None = exact_fragment_indices[0] if exact_fragment_indices else None
        overlap = 1.0 if match_index is not None else 0.0

        if match_index is None and actual_fragments:
            candidates = [
                (jaccard(actual_fragments, fragments), index)
                for index, fragments in enumerate(predicted_fragments)
                if index not in used_predictions and fragments
            ]
            if candidates:
                overlap, candidate_index = max(candidates)
                if overlap > 0.0:
                    match_index = candidate_index

        actual_info = cell_record(actual_cell, ignored)
        if match_index is None:
            comparisons.append(
                {
                    "status": "missed",
                    "ground_truth": actual_info,
                    "prediction": None,
                    "fragment_jaccard": 0.0,
                }
            )
            continue

        used_predictions.add(match_index)
        predicted_cell = predicted[match_index]
        predicted_info = cell_record(predicted_cell, ignored)
        same_fragments = actual_fragments == predicted_fragments[match_index]
        same_span = span(actual_cell) == span(predicted_cell)
        same_coordinates = coordinates(actual_cell) == coordinates(predicted_cell)

        if same_fragments and same_coordinates:
            status = "exact"
            coordinate_true_positive += 1
        elif same_fragments and same_span:
            status = "shifted"
        elif same_fragments:
            status = "wrong_extent"
        else:
            status = "fragment_mismatch"
        if same_fragments and same_span:
            evaluator_true_positive += 1

        comparisons.append(
            {
                "status": status,
                "ground_truth": actual_info,
                "prediction": predicted_info,
                "fragment_jaccard": round(overlap, 6),
            }
        )

    extras = [
        {
            "status": "extra",
            "ground_truth": None,
            "prediction": cell_record(cell, ignored),
            "fragment_jaccard": 0.0,
        }
        for index, cell in enumerate(predicted)
        if index not in used_predictions
    ]
    comparisons.extend(extras)

    status_counts: dict[str, int] = {}
    for comparison in comparisons:
        status = comparison["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    ground_truth_rows = max(coordinates(cell)[1] for cell in ground_truth["cells"]) + 1
    ground_truth_columns = max(coordinates(cell)[3] for cell in ground_truth["cells"]) + 1
    return {
        "schema_version": "1.0",
        "task": "table_span_error_analysis",
        "prediction": prediction.get("document_id"),
        "ground_truth": ground_truth.get("document_id"),
        "predicted_grid": {
            "rows": int(prediction.get("row_count", 0)),
            "columns": int(prediction.get("column_count", 0)),
        },
        "ground_truth_grid": {
            "rows": ground_truth_rows,
            "columns": ground_truth_columns,
        },
        "evaluator_compatible_metrics": prf(
            evaluator_true_positive, len(predicted), len(actual)
        ),
        "coordinate_exact_metrics": prf(
            coordinate_true_positive, len(predicted), len(actual)
        ),
        "status_counts": status_counts,
        "comparisons": comparisons,
    }


def _label(cell: dict[str, Any]) -> str:
    text = " ".join(str(cell.get("text", "")).split())
    if len(text) > 17:
        text = text[:16] + "…"
    return f"{cell.get('cell_id') or '?'} {cell['rowspan']}x{cell['colspan']} {text}".strip()


def render_preview(report: dict[str, Any], output: Path, cell_width: int, cell_height: int) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:  # pragma: no cover - depends on runtime environment
        raise SystemExit("Pillow is required for --preview: pip install pillow") from error

    rows = max(report["predicted_grid"]["rows"], report["ground_truth_grid"]["rows"])
    columns = max(
        report["predicted_grid"]["columns"], report["ground_truth_grid"]["columns"]
    )
    margin = 24
    title_height = 72
    legend_height = 54
    panel_width = columns * cell_width
    gap = 42
    width = margin * 2 + panel_width * 2 + gap
    height = title_height + rows * cell_height + legend_height + margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((margin, 16), "Ground truth spans", fill="#1f2937", font=font)
    right_x = margin + panel_width + gap
    draw.text((right_x, 16), "Predicted spans", fill="#1f2937", font=font)
    draw.text(
        (margin, 38),
        "green=exact  orange=wrong/partial  red=missed or extra",
        fill="#4b5563",
        font=font,
    )

    def draw_grid(origin_x: int) -> None:
        origin_y = title_height
        for row in range(rows + 1):
            y = origin_y + row * cell_height
            draw.line((origin_x, y, origin_x + panel_width, y), fill="#cbd5e1", width=1)
        for column in range(columns + 1):
            x = origin_x + column * cell_width
            draw.line((x, origin_y, x, origin_y + rows * cell_height), fill="#cbd5e1", width=1)

    draw_grid(margin)
    draw_grid(right_x)

    colors = {
        "exact": ("#dcfce7", "#16a34a"),
        "shifted": ("#ffedd5", "#ea580c"),
        "wrong_extent": ("#ffedd5", "#ea580c"),
        "fragment_mismatch": ("#fef3c7", "#d97706"),
        "missed": ("#fee2e2", "#dc2626"),
        "extra": ("#fee2e2", "#dc2626"),
    }

    def draw_cell(origin_x: int, cell: dict[str, Any], status: str) -> None:
        origin_y = title_height
        x0 = origin_x + cell["column_start"] * cell_width + 2
        y0 = origin_y + cell["row_start"] * cell_height + 2
        x1 = origin_x + (cell["column_end"] + 1) * cell_width - 2
        y1 = origin_y + (cell["row_end"] + 1) * cell_height - 2
        fill, outline = colors[status]
        draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=3)
        draw.text((x0 + 4, y0 + 4), _label(cell), fill="#111827", font=font)

    for comparison in report["comparisons"]:
        status = comparison["status"]
        if comparison["ground_truth"] is not None:
            draw_cell(margin, comparison["ground_truth"], status)
        if comparison["prediction"] is not None:
            draw_cell(right_x, comparison["prediction"], status)

    counts = ", ".join(
        f"{name}={count}" for name, count in sorted(report["status_counts"].items())
    )
    draw.text(
        (margin, title_height + rows * cell_height + 18),
        counts or "No spanning cells",
        fill="#374151",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    args = parse_args()
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    report = analyse_spans(prediction, ground_truth)
    report["prediction_file"] = str(args.prediction)
    report["ground_truth_file"] = str(args.ground_truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.preview:
        render_preview(report, args.preview, args.cell_width, args.cell_height)

    metrics = report["evaluator_compatible_metrics"]
    print(f"Ground-truth spanning cells: {metrics['actual_positive']}")
    print(f"Predicted spanning cells: {metrics['predicted_positive']}")
    print(
        f"Spanning Cell F1: {metrics['f1']:.3f} "
        f"(P={metrics['precision']:.3f}, R={metrics['recall']:.3f})"
    )
    print("Statuses: " + json.dumps(report["status_counts"], ensure_ascii=False))
    print(f"JSON: {args.output}")
    if args.preview:
        print(f"Preview: {args.preview}")


if __name__ == "__main__":
    main()
