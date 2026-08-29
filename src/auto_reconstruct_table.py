from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from build_relation_baseline import load_document
from detect_table_grid_light import complete_boundaries, detect_lines, longest_run
from reconstruct_table_from_grid import reconstruct as reconstruct_from_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically reconstruct an OCR table")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--ocr", required=True, type=Path)
    parser.add_argument("--relations", type=Path, help="Required when hybrid mode is selected")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument("--grid-output", type=Path)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument(
        "--mode", choices=("auto", "grid", "partial_grid", "hybrid"), default="auto"
    )
    parser.add_argument("--dark-threshold", type=int, default=230)
    parser.add_argument("--minimum-line-ratio", type=float, default=0.35)
    parser.add_argument("--minimum-grid-rows", type=int, default=3)
    parser.add_argument("--minimum-grid-columns", type=int, default=2)
    parser.add_argument(
        "--minimum-partial-grid-columns",
        type=int,
        default=2,
        help="Minimum detected columns for a reliable partial vertical grid",
    )
    parser.add_argument(
        "--relaxed-dark-threshold",
        type=int,
        default=245,
        help="Retry a missing grid axis with this threshold for light gray rules",
    )
    parser.add_argument("--minimum-axis-span", type=float, default=0.45)
    parser.add_argument("--maximum-band-ratio", type=float, default=4.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--assignment-min-confidence",
        type=float,
        default=0.0,
        help=(
            "Minimum OCR confidence used while assigning fragments to reconstructed "
            "cells. Kept separate from --min-confidence so low-confidence text can "
            "be retained after relation prediction (default: 0.0)"
        ),
    )
    parser.add_argument("--same-cell-threshold", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--vertical-same-cell-probability", type=float, default=0.65)
    parser.add_argument("--vertical-same-cell-soft-probability", type=float, default=0.45)
    parser.add_argument("--vertical-same-cell-min-x-overlap", type=float, default=0.55)
    parser.add_argument("--vertical-same-cell-max-gap-ratio", type=float, default=1.5)
    parser.add_argument("--span-dark-threshold", type=int, default=230)
    parser.add_argument("--minimum-border-support", type=float, default=0.60)
    return parser.parse_args()


def axis_span(lines: list[int], limit: int) -> float:
    if len(lines) < 2 or limit <= 1:
        return 0.0
    return (max(lines) - min(lines)) / float(limit - 1)


def band_ratio(boundaries: list[int]) -> float:
    gaps = sorted(
        end - start for start, end in zip(boundaries, boundaries[1:]) if end > start
    )
    if not gaps:
        return float("inf")
    median = gaps[len(gaps) // 2]
    return max(gaps) / max(1.0, float(median))


def assess_grid(
    *,
    width: int,
    height: int,
    vertical_lines: list[int],
    horizontal_lines: list[int],
    column_boundaries: list[int],
    row_boundaries: list[int],
    minimum_rows: int,
    minimum_columns: int,
    minimum_axis_span: float,
    maximum_band_ratio: float,
) -> dict[str, Any]:
    row_count = max(0, len(row_boundaries) - 1)
    column_count = max(0, len(column_boundaries) - 1)
    horizontal_span = axis_span(horizontal_lines, height)
    vertical_span = axis_span(vertical_lines, width)
    row_band_ratio = band_ratio(row_boundaries)
    checks = {
        "enough_rows": row_count >= minimum_rows,
        "enough_columns": column_count >= minimum_columns,
        "horizontal_span": horizontal_span >= minimum_axis_span,
        "vertical_span": vertical_span >= minimum_axis_span,
        "distributed_rows": row_band_ratio <= maximum_band_ratio,
    }

    use_grid = all(
        checks[name]
        for name in (
            "enough_rows",
            "enough_columns",
            "horizontal_span",
            "vertical_span",
        )
    )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "selected_mode": "grid" if use_grid else "hybrid",
        "reason": (
            "long lines form a sufficiently complete two-dimensional grid"
            if use_grid
            else "grid checks failed: " + ", ".join(failed)
        ),
        "checks": checks,
        "raw_vertical_line_count": len(vertical_lines),
        "raw_horizontal_line_count": len(horizontal_lines),
        "detected_row_count": row_count,
        "detected_column_count": column_count,
        "horizontal_axis_span": horizontal_span,
        "vertical_axis_span": vertical_span,
        "maximum_to_median_row_band_ratio": row_band_ratio,
    }


def save_preview(
    image: Image.Image, columns: list[int], rows: list[int], output: Path
) -> None:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for x in columns:
        draw.line((x, 0, x, image.height - 1), fill=(230, 20, 20), width=3)
    for y in rows:
        draw.line((0, y, image.width - 1, y), fill=(20, 80, 230), width=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output)


def estimate_ocr_row_bands(document: dict[str, Any], min_confidence: float) -> int:
    fragments = [
        fragment
        for fragment in document["fragments"]
        if float(fragment.get("confidence", 1.0)) >= min_confidence
        and str(fragment.get("text", "")).strip() not in {"|", "│"}
    ]
    if not fragments:
        return 0
    heights = sorted(max(1.0, item["bbox"][3] - item["bbox"][1]) for item in fragments)
    tolerance = max(3.0, heights[len(heights) // 2] * 0.6)
    bands: list[list[float]] = []
    centers = sorted((item["bbox"][1] + item["bbox"][3]) / 2.0 for item in fragments)
    for center in centers:
        if bands and abs(center - sum(bands[-1]) / len(bands[-1])) <= tolerance:
            bands[-1].append(center)
        else:
            bands.append([center])
    return len(bands)


def augment_partial_header_rows(
    document: dict[str, Any],
    row_boundaries: list[int],
    min_confidence: float,
    image: Image.Image | None = None,
    dark_threshold: int = 230,
    minimum_partial_rule_ratio: float = 0.08,
) -> list[int]:

    if len(row_boundaries) < 2:
        return row_boundaries
    top, bottom = row_boundaries[0], row_boundaries[1]
    fragments = [
        item
        for item in document["fragments"]
        if float(item.get("confidence", 1.0)) >= min_confidence
        and str(item.get("text", "")).strip() not in {"|", "│"}
        and top <= (item["bbox"][1] + item["bbox"][3]) / 2.0 < bottom
    ]
    if len(fragments) < 3:
        return row_boundaries
    heights = sorted(max(1.0, item["bbox"][3] - item["bbox"][1]) for item in fragments)
    median_height = heights[len(heights) // 2]
    tolerance = max(3.0, median_height * 0.6)
    bands: list[list[float]] = []
    for center in sorted((item["bbox"][1] + item["bbox"][3]) / 2.0 for item in fragments):
        if bands and abs(center - sum(bands[-1]) / len(bands[-1])) <= tolerance:
            bands[-1].append(center)
        else:
            bands.append([center])
    centers = [sum(band) / len(band) for band in bands]
    if len(centers) < 2:
        return row_boundaries
    gap, index = max(
        (right - left, index)
        for index, (left, right) in enumerate(zip(centers, centers[1:]))
    )
    if gap < max(8.0, median_height * 0.8):
        return row_boundaries
    boundary = round((centers[index] + centers[index + 1]) / 2.0)
    if boundary - top < median_height or bottom - boundary < median_height:
        return row_boundaries
    if image is not None:
        gray = np.asarray(image.convert("L"))
        strongest_run = max(
            longest_run(gray[y, :] < dark_threshold)
            for y in range(max(0, boundary - 3), min(gray.shape[0], boundary + 4))
        )
        if strongest_run < image.width * minimum_partial_rule_ratio:
            return row_boundaries
    return sorted(set([*row_boundaries, boundary]))


def select_automatic_mode(
    assessment: dict[str, Any],
    estimated_ocr_rows: int,
    minimum_partial_grid_columns: int,
) -> tuple[str, str]:

    reliable_vertical_axis = (
        assessment["checks"]["enough_columns"]
        and assessment["checks"]["vertical_span"]
    )
    enough_partial_columns = (
        int(assessment["detected_column_count"]) >= minimum_partial_grid_columns
    )
    reliable_horizontal_axis = (
        assessment["checks"]["enough_rows"]
        and assessment["checks"]["horizontal_span"]
    )
    initially_full_grid = assessment["selected_mode"] == "grid"

    if initially_full_grid:
        return "grid", "long lines form a sufficiently complete two-dimensional grid"
    if reliable_vertical_axis and enough_partial_columns:
        return (
            "partial_grid",
            "vertical grid is reliable; OCR geometry is used for incomplete row boundaries",
        )
    if reliable_horizontal_axis:
        return (
            "partial_grid",
            "horizontal grid is reliable; OCR geometry is used for missing column boundaries",
        )
    return "hybrid", assessment["reason"]


def run_hybrid(
    args: argparse.Namespace,
    grid: dict[str, Any],
    use_detected_columns: bool,
    use_detected_rows: bool,
) -> dict[str, Any]:
    if args.relations is None:
        raise ValueError("--relations is required because hybrid mode was selected")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "reconstruct_table_structure_hybrid.py"),
        "--ocr", str(args.ocr),
        "--relations", str(args.relations),
        "--output", str(args.output),
        "--min-confidence", str(args.assignment_min_confidence),
        "--same-cell-threshold", str(args.same_cell_threshold),
        "--edge-threshold", str(args.edge_threshold),
        "--vertical-same-cell-probability", (
            str(args.vertical_same_cell_probability)
            if (use_detected_columns or use_detected_rows)
            else "1.01"
        ),
        "--vertical-same-cell-soft-probability", (
            str(args.vertical_same_cell_soft_probability)
            if (use_detected_columns or use_detected_rows)
            else "1.01"
        ),
        "--vertical-same-cell-min-x-overlap", str(args.vertical_same_cell_min_x_overlap),
        "--vertical-same-cell-max-gap-ratio", str(args.vertical_same_cell_max_gap_ratio),
        "--row-banding", "aligned",
        "--merge-header-lines",
        "--infer-missing-header-cells",
    ]
    if use_detected_columns:
        command.extend(
            [
                "--column-boundaries-json", json.dumps(grid["column_boundaries"]),
                "--grid-image-width", str(grid["image_size"]["width"]),
            ]
        )
    if use_detected_rows:
        command.extend(
            [
                "--row-boundaries-json", json.dumps(grid["row_boundaries"]),
                "--grid-image-height", str(grid["image_size"]["height"]),
            ]
        )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return json.loads(args.output.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    vertical, horizontal, _ = detect_lines(
        image, args.dark_threshold, args.minimum_line_ratio
    )
    relaxed_vertical: list[int] = []
    relaxed_horizontal: list[int] = []
    if (not vertical or not horizontal) and args.relaxed_dark_threshold > args.dark_threshold:
        relaxed_vertical, relaxed_horizontal, _ = detect_lines(
            image, args.relaxed_dark_threshold, args.minimum_line_ratio
        )
        if not vertical:
            vertical = relaxed_vertical
        if not horizontal:
            horizontal = relaxed_horizontal
    columns = complete_boundaries(vertical, image.width)
    rows = complete_boundaries(horizontal, image.height)
    ocr_document = load_document(args.ocr)
    assessment = assess_grid(
        width=image.width,
        height=image.height,
        vertical_lines=vertical,
        horizontal_lines=horizontal,
        column_boundaries=columns,
        row_boundaries=rows,
        minimum_rows=args.minimum_grid_rows,
        minimum_columns=args.minimum_grid_columns,
        minimum_axis_span=args.minimum_axis_span,
        maximum_band_ratio=args.maximum_band_ratio,
    )
    estimated_ocr_rows = estimate_ocr_row_bands(ocr_document, args.min_confidence)
    assessment["estimated_ocr_text_row_bands"] = estimated_ocr_rows
    assessment["primary_dark_threshold"] = args.dark_threshold
    assessment["relaxed_dark_threshold"] = args.relaxed_dark_threshold
    assessment["used_relaxed_vertical_axis"] = bool(relaxed_vertical)
    assessment["used_relaxed_horizontal_axis"] = bool(relaxed_horizontal)
    if args.mode == "auto":
        selected_mode, selection_reason = select_automatic_mode(
            assessment, estimated_ocr_rows, args.minimum_partial_grid_columns
        )
        assessment["reason"] = selection_reason
    else:
        selected_mode = args.mode
    assessment["requested_mode"] = args.mode
    assessment["selected_mode"] = selected_mode
    assessment["minimum_partial_grid_columns"] = args.minimum_partial_grid_columns
    if args.mode != "auto":
        assessment["reason"] = f"mode forced by --mode {args.mode}"

    if (
        selected_mode in {"grid", "partial_grid"}
        and assessment["checks"]["horizontal_span"]
    ):
        original_rows = list(rows)
        rows = augment_partial_header_rows(
            ocr_document,
            rows,
            args.min_confidence,
            # A full grid should only gain a header boundary when a matching
            # partial rule is visible.  Borderless partial grids (for example
            # web tables) may express the same hierarchy only through OCR
            # text bands, so retain the geometry-only fallback there.
            image=image if selected_mode == "grid" else None,
            dark_threshold=args.span_dark_threshold,
        )
        assessment["partial_header_boundaries_added"] = len(rows) - len(original_rows)
        assessment["detected_row_count_after_augmentation"] = len(rows) - 1

    grid = {
        "schema_version": "1.0",
        "method": "pillow_numpy_long_line_projection",
        "source_image": args.image.name,
        "image_size": {"width": image.width, "height": image.height},
        "parameters": {
            "dark_threshold": args.dark_threshold,
            "minimum_line_ratio": args.minimum_line_ratio,
        },
        "raw_vertical_lines": vertical,
        "raw_horizontal_lines": horizontal,
        "column_boundaries": columns,
        "row_boundaries": rows,
        "column_count": len(columns) - 1,
        "row_count": len(rows) - 1,
    }
    if args.grid_output:
        args.grid_output.parent.mkdir(parents=True, exist_ok=True)
        args.grid_output.write_text(
            json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.preview_output:
        save_preview(image, columns, rows, args.preview_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if selected_mode == "grid":
        result = reconstruct_from_grid(
            ocr_document,
            grid,
            args.assignment_min_confidence,
            image=image,
            span_dark_threshold=args.span_dark_threshold,
            minimum_border_support=args.minimum_border_support,
        )
    else:
        use_partial_grid = selected_mode == "partial_grid"
        use_detected_columns = use_partial_grid and (
            assessment["checks"]["enough_columns"]
            and assessment["checks"]["vertical_span"]
        )
        use_detected_rows = use_partial_grid and (
            assessment["checks"]["enough_rows"]
            and assessment["checks"]["horizontal_span"]
        )
        assessment["partial_grid_uses_columns"] = use_detected_columns
        assessment["partial_grid_uses_rows"] = use_detected_rows
        result = run_hybrid(
            args,
            grid,
            use_detected_columns=use_detected_columns,
            use_detected_rows=use_detected_rows,
        )
    result["automatic_mode_selection"] = assessment
    result.setdefault("parameters", {})["relation_min_confidence"] = args.min_confidence
    result["parameters"]["assignment_min_confidence"] = args.assignment_min_confidence
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    diagnostics = {
        "schema_version": "1.0",
        "image": args.image.name,
        "ocr": args.ocr.name,
        "selection": assessment,
        "result": {
            "rows": result["row_count"],
            "columns": result["column_count"],
            "logical_cells": result["logical_cell_count"],
            "warnings": len(result.get("warnings", [])),
        },
    }
    if args.diagnostics_output:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Selected mode: {selected_mode}")
    print(f"Reason: {assessment['reason']}")
    print(f"Detected grid: {len(rows) - 1}x{len(columns) - 1}")
    print(f"Reconstructed grid: {result['row_count']}x{result['column_count']}")
    print(f"Logical cells: {result['logical_cell_count']}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
