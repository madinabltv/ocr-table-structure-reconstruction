from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reconstruct_table_structure import join_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark ground-truth JSON files")
    parser.add_argument("--ocr-dir", type=Path, default=Path("data/ocr"))
    parser.add_argument("--output-dir", type=Path, default=Path("annotations"))
    return parser.parse_args()


def make_cell(
    cell_id: str,
    row_start: int,
    row_end: int,
    column_start: int,
    column_end: int,
    fragment_ids: list[int],
    fragments_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    fragments = [fragments_by_id[fragment_id] for fragment_id in fragment_ids]
    cell = {
        "id": cell_id,
        "row_start": row_start,
        "row_end": row_end,
        "column_start": column_start,
        "column_end": column_end,
        "fragment_ids": fragment_ids,
        "text": join_text(fragments) if fragments else "",
    }
    if not fragments:
        cell["missing_in_ocr"] = True
    return cell


def annotation(
    document_id: str,
    ocr: dict[str, Any],
    cells_spec: list[tuple[str, int, int, int, int, list[int]]],
    ignored: list[int] | None = None,
) -> dict[str, Any]:
    ignored = ignored or []
    fragments_by_id = {int(item["id"]): item for item in ocr["fragments"]}
    cells = [
        make_cell(cell_id, r0, r1, c0, c1, ids, fragments_by_id)
        for cell_id, r0, r1, c0, c1, ids in cells_spec
    ]
    assigned = [fragment_id for cell in cells for fragment_id in cell["fragment_ids"]]
    if len(assigned) != len(set(assigned)):
        raise ValueError(f"{document_id}: a fragment is assigned to more than one cell")
    known = set(fragments_by_id)
    accounted = set(assigned) | set(ignored)
    if accounted != known:
        missing = sorted(known - accounted)
        unknown = sorted(accounted - known)
        raise ValueError(f"{document_id}: unassigned={missing}, unknown={unknown}")
    return {
        "schema_version": "1.0",
        "document_id": document_id,
        "description": "Reviewed logical cell annotations for independent evaluation",
        "annotation_method": "manual grid transcription from OCR preview and source image",
        "source_ocr": ocr.get("source_image"),
        "ignored_fragment_ids": ignored,
        "cells": cells,
    }


def table_03(ocr: dict[str, Any]) -> dict[str, Any]:
    specs = [
        ("h00", 0, 0, 0, 0, [0]),
        ("h01", 0, 0, 1, 1, [1, 2]),
        ("h02", 0, 0, 2, 2, [3]),
        ("h03", 0, 0, 3, 3, [4, 5]),
    ]
    for row in range(12):
        base = 6 + row * 12
        output_row = row + 1
        specs.extend(
            [
                (f"c{output_row}0", output_row, output_row, 0, 0, list(range(base, base + 4))),
                (f"c{output_row}1", output_row, output_row, 1, 1, list(range(base + 4, base + 6))),
                (f"c{output_row}2", output_row, output_row, 2, 2, list(range(base + 6, base + 10))),
                (f"c{output_row}3", output_row, output_row, 3, 3, list(range(base + 10, base + 12))),
            ]
        )
    return annotation("table_03_payment_schedule", ocr, specs)


def table_05(ocr: dict[str, Any]) -> dict[str, Any]:
    specs = [
        ("h00", 0, 0, 0, 0, [4, 5]),
        ("h01", 0, 0, 1, 1, [0, 1, 2, 3, 6, 7, 8]),
    ]
    for row, left in enumerate(range(9, 21, 2), start=1):
        specs.extend(
            [
                (f"c{row}0", row, row, 0, 0, [left]),
                (f"c{row}1", row, row, 1, 1, [left + 1]),
            ]
        )
    return annotation("table_05_impact_strength", ocr, specs)


def table_06(ocr: dict[str, Any]) -> dict[str, Any]:
    specs = [
        ("h00", 0, 0, 0, 0, [0, 1]),
        ("h01", 0, 0, 1, 1, [2, 3, 4]),
        ("c10", 1, 1, 0, 0, [5]),
        ("c11", 1, 1, 1, 1, [6]),
        ("c20", 2, 2, 0, 0, [7, 8, 9, 10]),
        ("c21", 2, 2, 1, 1, [11]),
        ("c30", 3, 3, 0, 0, [12]),
        ("c31", 3, 3, 1, 1, [13]),
        ("c40", 4, 4, 0, 0, [14]),
        ("c41", 4, 4, 1, 1, [15]),
    ]
    return annotation("table_06_element_tolerances", ocr, specs)


def cluster_rows(fragments: list[dict[str, Any]], tolerance: float = 18.0) -> list[list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for fragment in sorted(fragments, key=lambda item: (sum(item["bbox"][1::2]) / 2, item["bbox"][0])):
        center = (fragment["bbox"][1] + fragment["bbox"][3]) / 2
        candidates = [row for row in rows if abs(center - row["center"]) <= tolerance]
        if candidates:
            row = min(candidates, key=lambda item: abs(center - item["center"]))
            row["items"].append(fragment)
            row["center"] = sum((x["bbox"][1] + x["bbox"][3]) / 2 for x in row["items"]) / len(row["items"])
        else:
            rows.append({"center": center, "items": [fragment]})
    return [row["items"] for row in sorted(rows, key=lambda item: item["center"])]


def table_08(ocr: dict[str, Any]) -> dict[str, Any]:
    specs = [
        ("h00", 0, 1, 0, 0, [4, 5]),
        ("h01", 0, 0, 1, 4, [0, 1, 2, 3]),
        ("h11", 1, 1, 1, 1, []),
        ("h12", 1, 1, 2, 2, []),
        ("h13", 1, 1, 3, 3, []),
        ("h14", 1, 1, 4, 4, []),
    ]
    body = [item for item in ocr["fragments"] if int(item["id"]) >= 6]
    rows = cluster_rows(body)
    boundaries = [0, 350, 850, 1400, 1950, float("inf")]
    for row_offset, row_fragments in enumerate(rows, start=2):
        grouped = [[] for _ in range(5)]
        for fragment in row_fragments:
            center_x = (fragment["bbox"][0] + fragment["bbox"][2]) / 2
            column = next(i for i in range(5) if boundaries[i] <= center_x < boundaries[i + 1])
            grouped[column].append(int(fragment["id"]))
        for column, ids in enumerate(grouped):
            specs.append((f"c{row_offset}{column}", row_offset, row_offset, column, column, ids))
    return annotation("table_08_composition_long", ocr, specs)


def table_09(ocr: dict[str, Any]) -> dict[str, Any]:
    ignored = [2, 4, 6, 14, 18, 82, 105]
    fragments = [item for item in ocr["fragments"] if int(item["id"]) not in ignored]
    boundaries = [0, 360, 592, 754, 909, 1088, float("inf")]
    row_boundaries = [0, 180, 390, 565, 701, 912, float("inf")]
    grouped: dict[tuple[int, int], list[int]] = {
        (row, column): [] for row in range(6) for column in range(6)
    }
    for fragment in fragments:
        center_x = (fragment["bbox"][0] + fragment["bbox"][2]) / 2
        center_y = (fragment["bbox"][1] + fragment["bbox"][3]) / 2
        column = next(i for i in range(6) if boundaries[i] <= center_x < boundaries[i + 1])
        row = next(i for i in range(6) if row_boundaries[i] <= center_y < row_boundaries[i + 1])
        grouped[(row, column)].append(int(fragment["id"]))
    specs = [
        (f"c{row}{column}", row, row, column, column, grouped[(row, column)])
        for row in range(6)
        for column in range(6)
    ]
    return annotation("table_09_tax_elements", ocr, specs, ignored)


BUILDERS = {
    "table_03_payment_schedule": table_03,
    "table_05_impact_strength": table_05,
    "table_06_element_tolerances": table_06,
    "table_08_composition_long": table_08,
    "table_09_tax_elements": table_09,
}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for document_id, builder in BUILDERS.items():
        input_path = args.ocr_dir / f"{document_id}.json"
        ocr = json.loads(input_path.read_text(encoding="utf-8"))
        result = builder(ocr)
        output_path = args.output_dir / f"{document_id}_cells_ground_truth.json"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        nonempty = sum(bool(cell["fragment_ids"]) for cell in result["cells"])
        print(f"{document_id}: cells={len(result['cells'])}, nonempty={nonempty}, ignored={len(result['ignored_fragment_ids'])}")
        print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()
