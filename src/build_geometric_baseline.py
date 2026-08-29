from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group OCR fragments into table rows using their vertical centers."
    )
    parser.add_argument("--input", required=True, type=Path, help="OCR JSON")
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument(
        "--data-top",
        required=True,
        type=float,
        help="Ignore fragments whose vertical center is above this y coordinate",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--row-tolerance",
        type=float,
        default=20.0,
        help="Maximum center-y distance for fragments in one row",
    )
    parser.add_argument("--expected-columns", type=int)
    return parser.parse_args()


def center(fragment: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = fragment["bbox"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def group_rows(fragments: list[dict], tolerance: float) -> list[list[dict]]:
    rows: list[dict] = []
    for fragment in sorted(fragments, key=lambda item: (center(item)[1], center(item)[0])):
        _, center_y = center(fragment)
        candidates = [
            row for row in rows if abs(center_y - row["center_y"]) <= tolerance
        ]
        if candidates:
            row = min(candidates, key=lambda item: abs(center_y - item["center_y"]))
            row["fragments"].append(fragment)
            centers = [center(item)[1] for item in row["fragments"]]
            row["center_y"] = sum(centers) / len(centers)
        else:
            rows.append({"center_y": center_y, "fragments": [fragment]})

    rows.sort(key=lambda item: item["center_y"])
    return [
        sorted(row["fragments"], key=lambda item: center(item)[0]) for row in rows
    ]


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    selected = []
    rejected = []
    for fragment in source["fragments"]:
        _, center_y = center(fragment)
        reasons = []
        if center_y < args.data_top:
            reasons.append("above_data_top")
        if fragment["confidence"] < args.min_confidence:
            reasons.append("low_confidence")
        if reasons:
            rejected.append({"fragment_id": fragment["id"], "reasons": reasons})
        else:
            selected.append(fragment)

    grouped = group_rows(selected, args.row_tolerance)
    output_rows = []
    relations = []
    warnings = []
    for row_index, fragments in enumerate(grouped):
        cells = []
        for column_index, fragment in enumerate(fragments):
            cells.append(
                {
                    "row": row_index,
                    "column": column_index,
                    "text": fragment["text"],
                    "fragment_ids": [fragment["id"]],
                    "bbox": fragment["bbox"],
                    "confidence": fragment["confidence"],
                    "rowspan": 1,
                    "colspan": 1,
                }
            )
        for left, right in zip(fragments, fragments[1:]):
            relations.append(
                {
                    "source_fragment_id": left["id"],
                    "target_fragment_id": right["id"],
                    "relation": "RIGHT",
                }
            )
        if args.expected_columns and len(cells) != args.expected_columns:
            warnings.append(
                f"row {row_index}: expected {args.expected_columns} columns, got {len(cells)}"
            )
        output_rows.append({"row": row_index, "cells": cells})

    for upper, lower in zip(grouped, grouped[1:]):
        for upper_fragment in upper:
            upper_x, _ = center(upper_fragment)
            lower_fragment = min(
                lower, key=lambda item: abs(center(item)[0] - upper_x)
            )
            relations.append(
                {
                    "source_fragment_id": upper_fragment["id"],
                    "target_fragment_id": lower_fragment["id"],
                    "relation": "BELOW",
                }
            )

    result = {
        "schema_version": "0.1",
        "method": "geometric_row_grouping_baseline",
        "source_ocr": args.input.name,
        "parameters": {
            "data_top": args.data_top,
            "min_confidence": args.min_confidence,
            "row_tolerance": args.row_tolerance,
            "expected_columns": args.expected_columns,
        },
        "selected_fragment_count": len(selected),
        "rejected_fragments": rejected,
        "row_count": len(output_rows),
        "warnings": warnings,
        "rows": output_rows,
        "relations": relations,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    max_columns = max((len(row["cells"]) for row in output_rows), default=0)
    with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"column_{index}" for index in range(max_columns)])
        for row in output_rows:
            values = [cell["text"] for cell in row["cells"]]
            writer.writerow(values + [""] * (max_columns - len(values)))

    print(f"Selected fragments: {len(selected)}")
    print(f"Rows: {len(output_rows)}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"JSON: {args.json_output}")
    print(f"CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
