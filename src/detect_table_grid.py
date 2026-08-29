from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--preview-output", required=True, type=Path)
    parser.add_argument("--expected-columns", type=int)
    parser.add_argument("--vertical-kernel-ratio", type=float, default=0.12)
    parser.add_argument("--horizontal-kernel-ratio", type=float, default=0.05)
    return parser.parse_args()


def group_positions(indices: np.ndarray, max_gap: int = 2) -> list[int]:
    if len(indices) == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        value = int(value)
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group)) for group in groups]


def deduplicate(values: list[int], tolerance: int = 8) -> list[int]:
    result: list[int] = []
    for value in sorted(values):
        if not result or value - result[-1] > tolerance:
            result.append(value)
        else:
            result[-1] = round((result[-1] + value) / 2)
    return result


def extract_segments(
    mask: np.ndarray, orientation: str, minimum_length: int
) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if orientation == "vertical" and height >= minimum_length:
            segments.append(
                {
                    "x": round(x + width / 2),
                    "y1": y,
                    "y2": y + height - 1,
                    "length": height,
                }
            )
        elif orientation == "horizontal" and width >= minimum_length:
            segments.append(
                {
                    "y": round(y + height / 2),
                    "x1": x,
                    "x2": x + width - 1,
                    "length": width,
                }
            )
    if orientation == "vertical":
        segments.sort(key=lambda item: (item["x"], item["y1"]))
    else:
        segments.sort(key=lambda item: (item["y"], item["x1"]))
    return segments


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    vertical_length = max(15, round(height * args.vertical_kernel_ratio))
    horizontal_length = max(30, round(width * args.horizontal_kernel_ratio))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_length, 1)
    )
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_projection = np.count_nonzero(vertical_mask, axis=0)
    horizontal_projection = np.count_nonzero(horizontal_mask, axis=1)
    vertical_threshold = max(10, round(float(vertical_projection.max()) * 0.25))
    horizontal_threshold = max(30, round(float(horizontal_projection.max()) * 0.25))

    vertical_lines = group_positions(
        np.flatnonzero(vertical_projection >= vertical_threshold)
    )
    horizontal_lines = group_positions(
        np.flatnonzero(horizontal_projection >= horizontal_threshold)
    )
    vertical_segments = extract_segments(
        vertical_mask, "vertical", vertical_length
    )
    horizontal_segments = extract_segments(
        horizontal_mask, "horizontal", horizontal_length
    )

    horizontal_points = np.argwhere(horizontal_mask > 0)
    if horizontal_points.size == 0:
        raise RuntimeError("No horizontal table lines detected")
    left_edge = int(horizontal_points[:, 1].min())
    right_edge = int(horizontal_points[:, 1].max())

    internal_verticals = [
        x for x in vertical_lines if left_edge + 8 < x < right_edge - 8
    ]
    boundaries = deduplicate([left_edge, *internal_verticals, right_edge])
    column_centers = [
        round((left + right) / 2, 1)
        for left, right in zip(boundaries, boundaries[1:])
    ]
    warnings = []
    if args.expected_columns and len(column_centers) != args.expected_columns:
        warnings.append(
            f"expected {args.expected_columns} columns, detected {len(column_centers)}"
        )

    result = {
        "schema_version": "0.1",
        "source_image": args.input.name,
        "image_size": {"width": width, "height": height},
        "parameters": {
            "vertical_kernel_length": vertical_length,
            "horizontal_kernel_length": horizontal_length,
            "vertical_threshold": vertical_threshold,
            "horizontal_threshold": horizontal_threshold,
        },
        "vertical_lines": vertical_lines,
        "horizontal_lines": horizontal_lines,
        "vertical_segments": vertical_segments,
        "horizontal_segments": horizontal_segments,
        "table_bounds": {
            "left": left_edge,
            "right": right_edge,
            "top": min(horizontal_lines) if horizontal_lines else None,
            "bottom": max(horizontal_lines) if horizontal_lines else None,
        },
        "column_boundaries": boundaries,
        "column_centers": column_centers,
        "warnings": warnings,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.preview_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    preview = image.copy()
    for segment in horizontal_segments:
        cv2.line(
            preview,
            (segment["x1"], segment["y"]),
            (segment["x2"], segment["y"]),
            (255, 80, 0),
            2,
        )
    for segment in vertical_segments:
        cv2.line(
            preview,
            (segment["x"], segment["y1"]),
            (segment["x"], segment["y2"]),
            (0, 0, 255),
            2,
        )
    for index, center_x in enumerate(column_centers):
        cv2.putText(
            preview,
            f"C{index}",
            (round(center_x) - 15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 140, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(args.preview_output), preview)

    print(f"Vertical lines: {vertical_lines}")
    print(f"Horizontal lines: {horizontal_lines}")
    print(f"Vertical segments: {len(vertical_segments)}")
    print(f"Horizontal segments: {len(horizontal_segments)}")
    print(f"Column boundaries: {boundaries}")
    print(f"Column centers: {column_centers}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"JSON: {args.json_output}")
    print(f"Preview: {args.preview_output}")


if __name__ == "__main__":
    main()
