from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect ruled-table lines")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--preview-output", required=True, type=Path)
    parser.add_argument("--dark-threshold", type=int, default=180)
    parser.add_argument("--minimum-line-ratio", type=float, default=0.35)
    return parser.parse_args()


def longest_run(values: np.ndarray) -> int:
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return int(np.max(changes[1::2] - changes[::2])) if len(changes) else 0


def group_positions(values: list[int], max_gap: int = 2) -> list[int]:
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group)) for group in groups]


def detect_lines(
    image: Image.Image, dark_threshold: int, minimum_line_ratio: float
) -> tuple[list[int], list[int], np.ndarray]:
    gray = np.asarray(image.convert("L"))
    dark = gray < dark_threshold
    height, width = dark.shape
    vertical = [
        x for x in range(width) if longest_run(dark[:, x]) >= height * minimum_line_ratio
    ]
    horizontal = [
        y for y in range(height) if longest_run(dark[y, :]) >= width * minimum_line_ratio
    ]
    return group_positions(vertical), group_positions(horizontal), dark


def complete_boundaries(
    lines: list[int], limit: int, margin: int | None = None
) -> list[int]:
    # Scans and screenshots often leave a narrow white frame around a ruled
    # table.  A fixed 12-pixel tolerance created duplicate edge columns on
    # larger images, so scale it mildly with the corresponding image axis.
    if margin is None:
        # A 12--15 px white frame is common in screenshots.  Treat it as the
        # table edge instead of inserting an extra zero-width logical band.
        margin = max(15, round(limit * 0.015))
    result = sorted(lines)
    if not result or result[0] > margin:
        result.insert(0, 0)
    if result[-1] < limit - 1 - margin:
        result.append(limit - 1)
    return result


def main() -> None:
    args = parse_args()
    image = Image.open(args.input).convert("RGB")
    vertical, horizontal, _ = detect_lines(
        image, args.dark_threshold, args.minimum_line_ratio
    )
    columns = complete_boundaries(vertical, image.width)
    rows = complete_boundaries(horizontal, image.height)
    if len(columns) < 2 or len(rows) < 2:
        raise RuntimeError("Not enough long grid lines were detected")
    result = {
        "schema_version": "1.0",
        "method": "pillow_numpy_long_line_projection",
        "source_image": args.input.name,
        "image_size": {"width": image.width, "height": image.height},
        "parameters": {
            "dark_threshold": args.dark_threshold,
            "minimum_line_ratio": args.minimum_line_ratio,
        },
        "column_boundaries": columns,
        "row_boundaries": rows,
        "column_count": len(columns) - 1,
        "row_count": len(rows) - 1,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.preview_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for x in columns:
        draw.line((x, 0, x, image.height - 1), fill=(230, 20, 20), width=3)
    for y in rows:
        draw.line((0, y, image.width - 1, y), fill=(20, 80, 230), width=3)
    preview.save(args.preview_output)
    print(f"Rows: {result['row_count']}")
    print(f"Columns: {result['column_count']}")
    print(f"Row boundaries: {rows}")
    print(f"Column boundaries: {columns}")
    print(f"JSON: {args.json_output}")
    print(f"Preview: {args.preview_output}")


if __name__ == "__main__":
    main()
