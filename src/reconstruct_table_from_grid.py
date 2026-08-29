from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from build_relation_baseline import load_document
from reconstruct_table_structure import UnionFind, bbox_center, bbox_union, join_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct table cells from detected lines")
    parser.add_argument("--ocr", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--span-dark-threshold", type=int, default=230)
    parser.add_argument("--minimum-border-support", type=float, default=0.60)
    return parser.parse_args()


def interval_index(value: float, boundaries: list[int]) -> int | None:
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start <= value < end or (index == len(boundaries) - 2 and value == end):
            return index
    return None


def segment_border_support(
    dark: np.ndarray,
    *,
    orientation: str,
    boundary: int,
    start: int,
    end: int,
    thickness: int = 2,
) -> float:
    """Measure how much of a shared cell border contains a dark rule."""
    length = max(1, end - start)
    inset = min(max(3, round(length * 0.04)), max(3, length // 4))
    start += inset
    end -= inset
    if end <= start:
        return 1.0
    if orientation == "vertical":
        band = dark[
            max(0, start):min(dark.shape[0], end + 1),
            max(0, boundary - thickness):min(dark.shape[1], boundary + thickness + 1),
        ]
        return float(np.mean(np.any(band, axis=1))) if band.size else 1.0
    band = dark[
        max(0, boundary - thickness):min(dark.shape[0], boundary + thickness + 1),
        max(0, start):min(dark.shape[1], end + 1),
    ]
    return float(np.mean(np.any(band, axis=0))) if band.size else 1.0


def infer_grid_slot_groups(
    dark: np.ndarray,
    rows: list[int],
    columns: list[int],
    minimum_border_support: float,
) -> tuple[list[list[tuple[int, int]]], list[dict[str, Any]]]:
    """Join adjacent grid slots when their shared rule segment is absent."""
    slots = [
        (row, column)
        for row in range(len(rows) - 1)
        for column in range(len(columns) - 1)
    ]
    union_find = UnionFind(slots)
    trace: list[dict[str, Any]] = []

    def members(root: tuple[int, int]) -> set[tuple[int, int]]:
        return {slot for slot in slots if union_find.find(slot) == root}

    def rectangular_union(left: tuple[int, int], right: tuple[int, int]) -> bool:
        joined = members(union_find.find(left)) | members(union_find.find(right))
        row_values = [item[0] for item in joined]
        column_values = [item[1] for item in joined]
        area = (
            (max(row_values) - min(row_values) + 1)
            * (max(column_values) - min(column_values) + 1)
        )
        return len(joined) == area

    for row, (top, bottom) in enumerate(zip(rows, rows[1:])):
        for column, boundary in enumerate(columns[1:-1]):
            support = segment_border_support(
                dark,
                orientation="vertical",
                boundary=boundary,
                start=top,
                end=bottom,
            )
            left, right = (row, column), (row, column + 1)
            merged = support < minimum_border_support and rectangular_union(left, right)
            if merged:
                union_find.union(left, right)
            if support < minimum_border_support:
                trace.append(
                    {
                        "orientation": "vertical",
                        "row": row,
                        "between_columns": [column, column + 1],
                        "border_support": round(support, 6),
                        "merged": merged,
                    }
                )

    for row, boundary in enumerate(rows[1:-1]):
        for column, (left, right) in enumerate(zip(columns, columns[1:])):
            support = segment_border_support(
                dark,
                orientation="horizontal",
                boundary=boundary,
                start=left,
                end=right,
            )
            upper, lower = (row, column), (row + 1, column)
            merged = support < minimum_border_support and rectangular_union(upper, lower)
            if merged:
                union_find.union(upper, lower)
            if support < minimum_border_support:
                trace.append(
                    {
                        "orientation": "horizontal",
                        "between_rows": [row, row + 1],
                        "column": column,
                        "border_support": round(support, 6),
                        "merged": merged,
                    }
                )

    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for slot in slots:
        grouped.setdefault(union_find.find(slot), []).append(slot)
    return list(grouped.values()), trace


def reconstruct(
    ocr: dict[str, Any],
    grid: dict[str, Any],
    min_confidence: float,
    image: Image.Image | None = None,
    span_dark_threshold: int = 230,
    minimum_border_support: float = 0.60,
) -> dict[str, Any]:
    ocr_size = ocr.get("image_size", {})
    grid_size = grid.get("image_size", {})
    scale_x = float(ocr_size.get("width", grid_size.get("width", 1))) / max(
        1.0, float(grid_size.get("width", ocr_size.get("width", 1)))
    )
    scale_y = float(ocr_size.get("height", grid_size.get("height", 1))) / max(
        1.0, float(grid_size.get("height", ocr_size.get("height", 1)))
    )
    rows = [round(float(value) * scale_y) for value in grid["row_boundaries"]]
    columns = [round(float(value) * scale_x) for value in grid["column_boundaries"]]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {
        (row, column): []
        for row in range(len(rows) - 1)
        for column in range(len(columns) - 1)
    }
    ignored = []
    outside = []
    for fragment in ocr["fragments"]:
        if float(fragment.get("confidence", 1.0)) < min_confidence:
            ignored.append(fragment["id"])
            continue
        if str(fragment.get("text", "")).strip() in {"|", "│"}:
            ignored.append(fragment["id"])
            continue
        center_x, center_y = bbox_center(fragment["bbox"])
        row = interval_index(center_y, rows)
        column = interval_index(center_x, columns)
        if row is None or column is None:
            outside.append(fragment["id"])
            continue
        grouped[(row, column)].append(fragment)

    slot_groups = [[slot] for slot in grouped]
    span_trace: list[dict[str, Any]] = []
    if image is not None:
        grid_image = image.convert("L")
        if grid_image.size != (
            int(grid_size.get("width", image.width)),
            int(grid_size.get("height", image.height)),
        ):
            grid_image = grid_image.resize(
                (
                    int(grid_size.get("width", image.width)),
                    int(grid_size.get("height", image.height)),
                )
            )
        dark = np.asarray(grid_image) < span_dark_threshold
        grid_rows = [round(value / max(scale_y, 1e-9)) for value in rows]
        grid_columns = [round(value / max(scale_x, 1e-9)) for value in columns]
        slot_groups, span_trace = infer_grid_slot_groups(
            dark, grid_rows, grid_columns, minimum_border_support
        )

    cells = []
    for cell_index, slots in enumerate(slot_groups):
        row_values = [slot[0] for slot in slots]
        column_values = [slot[1] for slot in slots]
        row_start, row_end = min(row_values), max(row_values)
        column_start, column_end = min(column_values), max(column_values)
        fragments = [fragment for slot in slots for fragment in grouped[slot]]
        cell = {
            "id": f"cell_{cell_index}",
            "fragment_ids": [fragment["id"] for fragment in fragments],
            "text": join_text(fragments) if fragments else "",
            "bbox": bbox_union(fragments) if fragments else [],
            "row_start": row_start, "row_end": row_end,
            "column_start": column_start, "column_end": column_end,
            "rowspan": row_end - row_start + 1,
            "colspan": column_end - column_start + 1,
        }
        if not fragments:
            cell["empty_cell"] = True
        cells.append(cell)
    return {
        "schema_version": "1.0",
        "method": "ruled_grid_geometry_reconstruction_v1",
        "source_ocr": ocr.get("source_image"),
        "source_grid": grid.get("source_image"),
        "parameters": {
            "min_confidence": min_confidence,
            "grid_to_ocr_scale_x": scale_x,
            "grid_to_ocr_scale_y": scale_y,
            "span_dark_threshold": span_dark_threshold,
            "minimum_border_support": minimum_border_support,
        },
        "fragment_count": len(ocr["fragments"]),
        "logical_cell_count": len(cells),
        "row_count": len(rows) - 1,
        "column_count": len(columns) - 1,
        "row_boundaries": rows,
        "column_boundaries": columns,
        "ignored_fragment_ids": ignored,
        "outside_fragment_ids": outside,
        "missing_border_merges": span_trace,
        "warnings": [f"fragments outside detected grid: {outside}"] if outside else [],
        "cells": cells,
    }


def main() -> None:
    args = parse_args()
    ocr = load_document(args.ocr)
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    image = Image.open(args.image).convert("RGB") if args.image else None
    result = reconstruct(
        ocr,
        grid,
        args.min_confidence,
        image=image,
        span_dark_threshold=args.span_dark_threshold,
        minimum_border_support=args.minimum_border_support,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rows: {result['row_count']}")
    print(f"Columns: {result['column_count']}")
    print(f"Logical cells: {result['logical_cell_count']}")
    print(f"Ignored line artifacts: {len(result['ignored_fragment_ids'])}")
    print(f"Outside fragments: {len(result['outside_fragment_ids'])}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
