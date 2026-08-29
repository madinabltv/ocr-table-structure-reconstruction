
"""
    python src/build_generalization_v8_annotations.py \
        --ocr-dir data/ocr/generalization_v8/adaptive \
        --output-dir annotations/generalization_v8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from reconstruct_table_structure import join_text


@dataclass(frozen=True)
class CellSpec:
    row_start: int
    row_end: int
    column_start: int
    column_end: int


@dataclass(frozen=True)
class TableSpec:
    x_lines: tuple[float, ...]
    y_lines: tuple[float, ...]
    spans: tuple[CellSpec, ...] = ()
    description: str = "Russian table from the generalization_v8 test set"


def rectangular_grid(
    row_count: int,
    column_count: int,
    spans: Sequence[CellSpec] = (),
) -> list[CellSpec]:
    occupied: dict[tuple[int, int], CellSpec] = {}
    for span in spans:
        if not (0 <= span.row_start <= span.row_end < row_count):
            raise ValueError(f"Invalid row span: {span}")
        if not (0 <= span.column_start <= span.column_end < column_count):
            raise ValueError(f"Invalid column span: {span}")
        for row in range(span.row_start, span.row_end + 1):
            for column in range(span.column_start, span.column_end + 1):
                key = (row, column)
                if key in occupied:
                    raise ValueError(f"Overlapping spans at {key}: {occupied[key]} and {span}")
                occupied[key] = span

    cells: list[CellSpec] = list(spans)
    for row in range(row_count):
        for column in range(column_count):
            if (row, column) not in occupied:
                cells.append(CellSpec(row, row, column, column))
    return sorted(cells, key=lambda cell: (cell.row_start, cell.column_start))


TABLE_SPECS: dict[int, TableSpec] = {
    1: TableSpec(
        x_lines=(8.5, 168.5, 244.5, 716.5, 1284.5, 1890.5),
        y_lines=(
            6.5, 73.5, 191.5, 258.5, 325.5, 392.5, 458.5,
            525.5, 592.5, 659.5, 725.5, 844.5, 911.5, 978.5,
            1044.5, 1111.5, 1230.5, 1349.5, 1467.5,
        ),
        description="Wikipedia filmography table with five columns",
    ),
    2: TableSpec(
        x_lines=(4.5, 672.5, 818.5, 959.5, 1070.5, 1684.5),
        y_lines=(13.5, 184.0, 244.5, 362.5, 429.5, 496.5, 563.5),
        spans=(
            CellSpec(0, 1, 0, 0),
            CellSpec(0, 1, 1, 1),
            CellSpec(0, 0, 2, 3),
            CellSpec(0, 1, 4, 4),
        ),
        description="Music chart table with a two-level merged header",
    ),
    4: TableSpec(
        x_lines=(8.5, 246.5, 2061.5),
        y_lines=(
            5.5, 121.5, 351.5, 396.5, 474.5, 551.5, 591.5,
            707.5, 746.5, 786.5, 825.5, 903.5, 942.5, 982.5,
            1021.5, 1061.5, 1176.5,
        ),
        description="Two-column educational requirements table",
    ),
    9: TableSpec(
        x_lines=(0.0, 250.0, 640.0, 1340.0, 1550.0, 2180.0, 2280.0, 2400.0, 2520.0, 2756.0),
        y_lines=(
            0.0, 106.0, 160.0, 244.0, 326.0, 408.0, 490.0,
            572.0, 654.0, 736.0, 818.0, 900.0, 982.0, 1064.0,
            1146.0, 1242.0,
        ),
        spans=(
            CellSpec(0, 1, 0, 0),
            CellSpec(0, 1, 1, 1),
            CellSpec(0, 1, 2, 2),
            CellSpec(0, 1, 3, 3),
            CellSpec(0, 1, 4, 4),
            CellSpec(0, 0, 5, 7),
            CellSpec(0, 1, 8, 8),
        ),
        description="Nine-column bond table with a grouped discount header",
    ),
    12: TableSpec(
        x_lines=(15.0, 644.0, 1042.0, 1490.0),
        y_lines=(
            11.0, 47.0, 124.0, 178.0, 248.0, 319.0, 460.0,
            495.0, 530.0, 566.0, 601.0, 636.0, 671.0, 707.0,
            742.0, 777.0, 1094.0, 1305.0,
        ),
        spans=(
            CellSpec(2, 3, 2, 2),
            CellSpec(4, 5, 2, 2),
        ),
        description="Bank regulatory indicators with vertical merged cells",
    ),
    13: TableSpec(
        x_lines=(5.0, 107.0, 813.0, 977.0, 1295.0, 1649.0),
        y_lines=(
            5.0, 109.0, 144.0, 178.0, 213.0, 283.0, 317.0,
            352.0, 422.0, 491.0, 561.0, 665.0, 771.0, 805.0,
            840.0, 875.0, 944.0, 979.0, 1014.0, 1048.5,
        ),
        spans=(CellSpec(2, 2, 0, 4),),
        description="Bank report page with a full-width section heading",
    ),
    14: TableSpec(
        x_lines=(5.0, 162.0, 1250.0, 1502.0, 1993.0, 2539.0),
        y_lines=(4.0, 111.0, 218.0, 272.0, 379.0, 486.0, 593.0, 700.0, 807.0, 908.0, 1015.0, 1069.0),
        description="Continuation page of a five-column bank report",
    ),
    15: TableSpec(
        x_lines=(4.5, 134.5, 858.5, 1087.5, 1455.5, 1823.5, 2273.5),
        y_lines=(5.5, 94.5, 169.5, 245.5, 320.5, 396.5, 471.5, 546.5, 680.5, 813.5, 946.5),
        description="Continuation page of a six-column bank capital table",
    ),
}


def fragment_center(fragment: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = fragment["bbox"]
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def physical_box(cell: CellSpec, spec: TableSpec) -> tuple[float, float, float, float]:
    return (
        spec.x_lines[cell.column_start],
        spec.y_lines[cell.row_start],
        spec.x_lines[cell.column_end + 1],
        spec.y_lines[cell.row_end + 1],
    )


def contains(box: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x < x2 and y1 <= y < y2


def make_annotation(table_number: int, ocr_path: Path) -> dict[str, Any]:
    spec = TABLE_SPECS[table_number]
    with ocr_path.open("r", encoding="utf-8") as stream:
        ocr = json.load(stream)

    fragments = ocr.get("fragments", [])
    fragments_by_id = {fragment["id"]: fragment for fragment in fragments}
    if len(fragments_by_id) != len(fragments):
        raise ValueError(f"Duplicate fragment ids in {ocr_path}")

    row_count = len(spec.y_lines) - 1
    column_count = len(spec.x_lines) - 1
    cell_specs = rectangular_grid(row_count, column_count, spec.spans)
    assignments: dict[CellSpec, list[int]] = {cell: [] for cell in cell_specs}
    ignored: list[int] = []

    for fragment in fragments:
        point = fragment_center(fragment)
        matches = [cell for cell in cell_specs if contains(physical_box(cell, spec), point)]
        if len(matches) == 1:
            assignments[matches[0]].append(fragment["id"])
        elif not matches:
            ignored.append(fragment["id"])
        else:
            raise ValueError(
                f"Fragment {fragment['id']} at {point} matches several cells: {matches}"
            )

    cells: list[dict[str, Any]] = []
    for index, cell in enumerate(cell_specs):
        ids = assignments[cell]
        item: dict[str, Any] = {
            "id": f"cell_{index}",
            "row_start": cell.row_start,
            "row_end": cell.row_end,
            "column_start": cell.column_start,
            "column_end": cell.column_end,
            "fragment_ids": ids,
            "text": (
                join_text([fragments_by_id[fragment_id] for fragment_id in ids])
                if ids
                else ""
            ),
        }
        if not ids:
            item["missing_in_ocr"] = True
        cells.append(item)

    validate_annotation(fragments_by_id, cells, ignored, table_number)
    return {
        "schema_version": "1.0",
        "document_id": f"table_{table_number}",
        "description": spec.description,
        "annotation_method": "manual_grid_boundaries_with_fragment_center_assignment",
        "source_ocr": str(ocr_path),
        "grid": {"rows": row_count, "columns": column_count},
        "ignored_fragment_ids": sorted(ignored),
        "cells": cells,
    }


def validate_annotation(
    fragments_by_id: dict[Any, dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    ignored: Sequence[Any],
    table_number: int,
) -> None:
    assigned: list[Any] = [fragment_id for cell in cells for fragment_id in cell["fragment_ids"]]
    duplicates = sorted({fragment_id for fragment_id in assigned if assigned.count(fragment_id) > 1})
    if duplicates:
        raise ValueError(f"table_{table_number}: fragments assigned more than once: {duplicates}")

    assigned_set = set(assigned)
    ignored_set = set(ignored)
    if assigned_set & ignored_set:
        raise ValueError(f"table_{table_number}: assigned fragments are also ignored")
    unknown = (assigned_set | ignored_set) - set(fragments_by_id)
    if unknown:
        raise ValueError(f"table_{table_number}: unknown fragment ids: {sorted(unknown)}")
    missing = set(fragments_by_id) - assigned_set - ignored_set
    if missing:
        raise ValueError(f"table_{table_number}: unaccounted fragments: {sorted(missing)}")

    occupied: dict[tuple[int, int], str] = {}
    for cell in cells:
        for row in range(cell["row_start"], cell["row_end"] + 1):
            for column in range(cell["column_start"], cell["column_end"] + 1):
                key = (row, column)
                if key in occupied:
                    raise ValueError(
                        f"table_{table_number}: logical cell collision at {key}: "
                        f"{occupied[key]} and {cell['id']}"
                    )
                occupied[key] = cell["id"]


def parse_tables(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return sorted(TABLE_SPECS)
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    unknown = sorted(set(result) - set(TABLE_SPECS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unsupported table numbers: {unknown}")
    return result


def build_all(ocr_dir: Path, output_dir: Path, tables: Iterable[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_number in tables:
        ocr_path = ocr_dir / f"table_{table_number}.json"
        if not ocr_path.exists():
            raise FileNotFoundError(f"OCR JSON not found: {ocr_path}")
        annotation = make_annotation(table_number, ocr_path)
        output_path = output_dir / f"table_{table_number}_cells_ground_truth.json"
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(annotation, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        fragment_count = sum(len(cell["fragment_ids"]) for cell in annotation["cells"])
        empty_count = sum(not cell["fragment_ids"] for cell in annotation["cells"])
        spanning_count = sum(
            cell["row_start"] != cell["row_end"]
            or cell["column_start"] != cell["column_end"]
            for cell in annotation["cells"]
        )
        print(f"table_{table_number}:")
        print(f"  grid: {annotation['grid']['rows']}x{annotation['grid']['columns']}")
        print(f"  logical cells: {len(annotation['cells'])}")
        print(f"  assigned fragments: {fragment_count}")
        print(f"  ignored fragments: {len(annotation['ignored_fragment_ids'])}")
        print(f"  empty OCR cells: {empty_count}")
        print(f"  spanning cells: {spanning_count}")
        print(f"  JSON: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cell-level annotations for the generalization_v8 tables."
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=Path("data/ocr/generalization_v8/adaptive"),
        help="Directory containing table_N.json adaptive-OCR files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotations/generalization_v8"),
        help="Directory for generated ground-truth JSON files.",
    )
    parser.add_argument(
        "--tables",
        type=parse_tables,
        default=sorted(TABLE_SPECS),
        help="Comma-separated table numbers or 'all' (default: all).",
    )
    args = parser.parse_args()
    build_all(args.ocr_dir, args.output_dir, args.tables)


if __name__ == "__main__":
    main()
