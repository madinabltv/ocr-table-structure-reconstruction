from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--data-top", required=True, type=float)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--row-tolerance", type=float, default=20.0)
    parser.add_argument(
        "--column-centers",
        required=True,
        help="Comma-separated x coordinates, for example 220,600,917",
    )
    return parser.parse_args()


def center(fragment: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = fragment["bbox"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def group_rows(fragments: list[dict], tolerance: float) -> list[list[dict]]:
    rows: list[dict] = []
    for fragment in sorted(fragments, key=lambda item: (center(item)[1], center(item)[0])):
        cy = center(fragment)[1]
        candidates = [row for row in rows if abs(cy - row["cy"]) <= tolerance]
        if candidates:
            row = min(candidates, key=lambda item: abs(cy - item["cy"]))
            row["fragments"].append(fragment)
            row["cy"] = sum(center(item)[1] for item in row["fragments"]) / len(
                row["fragments"]
            )
        else:
            rows.append({"cy": cy, "fragments": [fragment]})
    rows.sort(key=lambda item: item["cy"])
    return [row["fragments"] for row in rows]


def join_fragments(fragments: list[dict]) -> str:
    ordered = sorted(fragments, key=lambda item: center(item)[0])
    result = ""
    for fragment in ordered:
        token = fragment["text"]
        if not result:
            result = token
        elif result.endswith((",", ".")) and token[:1].isdigit():
            result += token
        else:
            result += " " + token
    return result


def main() -> None:
    args = parse_args()
    column_centers = [float(value) for value in args.column_centers.split(",")]
    if len(column_centers) < 2:
        raise ValueError("At least two column centers are required")
    if column_centers != sorted(column_centers):
        raise ValueError("Column centers must be sorted from left to right")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    selected = [
        fragment
        for fragment in source["fragments"]
        if center(fragment)[1] >= args.data_top
        and fragment["confidence"] >= args.min_confidence
    ]
    grouped_rows = group_rows(selected, args.row_tolerance)

    rows = []
    relations = []
    for row_index, fragments in enumerate(grouped_rows):
        columns: list[list[dict]] = [[] for _ in column_centers]
        for fragment in fragments:
            cx = center(fragment)[0]
            column_index = min(
                range(len(column_centers)),
                key=lambda index: abs(cx - column_centers[index]),
            )
            columns[column_index].append(fragment)

        cells = []
        for column_index, cell_fragments in enumerate(columns):
            ordered = sorted(cell_fragments, key=lambda item: center(item)[0])
            for left, right in zip(ordered, ordered[1:]):
                relations.append(
                    {
                        "source_fragment_id": left["id"],
                        "target_fragment_id": right["id"],
                        "relation": "SAME_CELL",
                    }
                )
            cells.append(
                {
                    "row": row_index,
                    "column": column_index,
                    "text": join_fragments(ordered),
                    "fragment_ids": [item["id"] for item in ordered],
                    "rowspan": 1,
                    "colspan": 1,
                    "missing": not ordered,
                }
            )

        nonempty = [cell for cell in cells if not cell["missing"]]
        for left, right in zip(nonempty, nonempty[1:]):
            relations.append(
                {
                    "source_cell": [left["row"], left["column"]],
                    "target_cell": [right["row"], right["column"]],
                    "relation": "RIGHT",
                }
            )
        rows.append({"row": row_index, "cells": cells})

    for column_index in range(len(column_centers)):
        nonempty = [
            row["cells"][column_index]
            for row in rows
            if not row["cells"][column_index]["missing"]
        ]
        for upper, lower in zip(nonempty, nonempty[1:]):
            relations.append(
                {
                    "source_cell": [upper["row"], upper["column"]],
                    "target_cell": [lower["row"], lower["column"]],
                    "relation": "BELOW",
                }
            )

    missing_cells = [
        [cell["row"], cell["column"]]
        for row in rows
        for cell in row["cells"]
        if cell["missing"]
    ]
    result = {
        "schema_version": "0.2",
        "method": "fixed_column_geometric_baseline",
        "source_ocr": args.input.name,
        "parameters": {
            "data_top": args.data_top,
            "min_confidence": args.min_confidence,
            "row_tolerance": args.row_tolerance,
            "column_centers": column_centers,
        },
        "row_count": len(rows),
        "column_count": len(column_centers),
        "missing_cells": missing_cells,
        "rows": rows,
        "relations": relations,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"column_{index}" for index in range(len(column_centers))])
        for row in rows:
            writer.writerow([cell["text"] for cell in row["cells"]])

    print(f"Selected fragments: {len(selected)}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(column_centers)}")
    print(f"Missing cells: {missing_cells}")
    print(f"JSON: {args.json_output}")
    print(f"CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
