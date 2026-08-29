from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from build_relation_baseline import load_document
from reconstruct_table_structure import (
    bbox_center,
    bbox_union,
    build_component_edges,
    build_components,
    infer_spans,
    join_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid relation/geometry reconstruction")
    parser.add_argument("--ocr", required=True, type=Path)
    parser.add_argument("--relations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--same-cell-threshold", type=float, default=0.5)
    parser.add_argument(
        "--vertical-same-cell-probability",
        type=float,
        default=0.65,
        help="Always keep vertical body merges above this probability",
    )
    parser.add_argument(
        "--vertical-same-cell-soft-probability",
        type=float,
        default=0.45,
        help=(
            "Consider geometrically compatible vertical merges above this lower "
            "probability (default: 0.45)"
        ),
    )
    parser.add_argument(
        "--vertical-same-cell-min-x-overlap",
        type=float,
        default=0.55,
        help="Minimum horizontal overlap for a soft vertical merge",
    )
    parser.add_argument(
        "--vertical-same-cell-max-gap-ratio",
        type=float,
        default=1.5,
        help="Maximum vertical gap divided by median OCR-fragment height",
    )
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--expected-columns", type=int)
    parser.add_argument(
        "--column-boundaries-json",
        help="Detected image-space column boundaries, encoded as a JSON array",
    )
    parser.add_argument(
        "--grid-image-width",
        type=float,
        help="Image width used for --column-boundaries-json",
    )
    parser.add_argument(
        "--row-boundaries-json",
        help="Detected image-space row boundaries, encoded as a JSON array",
    )
    parser.add_argument(
        "--grid-image-height",
        type=float,
        help="Image height used for --row-boundaries-json",
    )
    parser.add_argument("--header-rows", type=int)
    parser.add_argument("--merge-header-lines", action="store_true")
    parser.add_argument("--logical-header-rows", type=int)
    parser.add_argument("--infer-missing-header-cells", action="store_true")
    parser.add_argument(
        "--row-banding",
        choices=("kmeans", "aligned"),
        default="kmeans",
        help="Body row detection strategy (default: kmeans for reproducibility)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def center_y(fragment: dict[str, Any]) -> float:
    return (fragment["bbox"][1] + fragment["bbox"][3]) / 2.0


def y_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    a1, a2 = left["bbox"][1], left["bbox"][3]
    b1, b2 = right["bbox"][1], right["bbox"][3]
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    return overlap / max(1e-9, min(a2 - a1, b2 - b1))


def x_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    a1, a2 = left["bbox"][0], left["bbox"][2]
    b1, b2 = right["bbox"][0], right["bbox"][2]
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    return overlap / max(1e-9, min(a2 - a1, b2 - b1))


def vertical_gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    upper, lower = sorted((left, right), key=center_y)
    return max(0.0, float(lower["bbox"][1]) - float(upper["bbox"][3]))


def crosses_row_boundary(
    left: dict[str, Any], right: dict[str, Any], row_boundaries: list[float]
) -> bool:
    """Return True when a detected horizontal rule separates two fragments."""
    low, high = sorted((center_y(left), center_y(right)))
    return any(low < boundary < high for boundary in row_boundaries)


def detect_data_top(fragments: list[dict[str, Any]]) -> float:
    centers = sorted(center_y(fragment) for fragment in fragments)
    if len(centers) < 4:
        return -math.inf
    gaps = [(centers[index + 1] - centers[index], index) for index in range(len(centers) - 1)]
    positive = sorted(gap for gap, _ in gaps if gap > 0)
    median_gap = positive[len(positive) // 2] if positive else 0.0
    candidates = [
        (gap, index)
        for gap, index in gaps
        if index + 1 < len(centers) - 1 and gap >= max(15.0, median_gap * 2.5)
    ]
    if not candidates:
        return -math.inf
    _, split_index = max(candidates)
    return (centers[split_index] + centers[split_index + 1]) / 2.0


def constrain_same_cell_edges(
    relations: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    data_top: float,
    vertical_probability_threshold: float = 0.65,
    vertical_soft_probability_threshold: float | None = None,
    minimum_x_overlap: float = 0.55,
    maximum_gap_ratio: float = 1.5,
    row_boundaries: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if vertical_soft_probability_threshold is None:
        # Preserve the historical behaviour for direct callers that only pass
        # the original hard threshold.
        vertical_soft_probability_threshold = vertical_probability_threshold
    row_boundaries = row_boundaries or []
    by_id = {fragment["id"]: fragment for fragment in fragments}
    heights = sorted(
        max(1.0, float(fragment["bbox"][3]) - float(fragment["bbox"][1]))
        for fragment in fragments
    )
    median_height = heights[len(heights) // 2] if heights else 1.0
    filtered = []
    rejected = []
    for original in relations:
        item = dict(original)
        if item.get("prediction") == "SAME_CELL":
            left = by_id.get(item["source_fragment_id"])
            right = by_id.get(item["target_fragment_id"])
            if left is not None and right is not None:
                left_body = center_y(left) >= data_top
                right_body = center_y(right) >= data_top
                crosses_region = left_body != right_body
                vertical_body_merge = left_body and right_body and y_overlap(left, right) < 0.5
                same_probability = float(
                    item.get("probabilities", {}).get("SAME_CELL", 0.0)
                )
                boundary_between = crosses_row_boundary(left, right, row_boundaries)
                hard_probability_merge = (
                    vertical_body_merge
                    and same_probability >= vertical_probability_threshold
                    and not boundary_between
                )
                overlap = x_overlap(left, right)
                gap_ratio = vertical_gap(left, right) / max(1.0, median_height)
                geometric_soft_merge = (
                    vertical_body_merge
                    and same_probability >= vertical_soft_probability_threshold
                    and overlap >= minimum_x_overlap
                    and gap_ratio <= maximum_gap_ratio
                    and not boundary_between
                )
                trusted_vertical_merge = hard_probability_merge or geometric_soft_merge
                if vertical_body_merge and trusted_vertical_merge:
                    item["vertical_same_cell_decision"] = (
                        "hard_probability"
                        if hard_probability_merge
                        else "geometry_supported_soft_probability"
                    )
                    item["vertical_same_cell_diagnostics"] = {
                        "probability": same_probability,
                        "x_overlap": overlap,
                        "gap_ratio": gap_ratio,
                        "row_boundary_between": boundary_between,
                    }
                if (
                    crosses_region
                    or boundary_between
                    or (vertical_body_merge and not trusted_vertical_merge)
                ):
                    if crosses_region:
                        reason = "cross_region"
                    elif boundary_between:
                        reason = "row_boundary_between"
                    elif same_probability < vertical_soft_probability_threshold:
                        reason = "low_same_cell_probability"
                    elif overlap < minimum_x_overlap:
                        reason = "insufficient_x_overlap"
                    else:
                        reason = "vertical_gap_too_large"
                    rejected.append(
                        {
                            "source_fragment_id": left["id"],
                            "target_fragment_id": right["id"],
                            "reason": reason,
                            "same_cell_probability": same_probability,
                            "x_overlap": overlap,
                            "gap_ratio": gap_ratio,
                            "row_boundary_between": boundary_between,
                        }
                    )
                    item["prediction"] = "NO_RELATION"
        filtered.append(item)
    return filtered, rejected


def ordered_kmeans(values: list[float], clusters: int, random_state: int) -> tuple[list[int], list[float]]:
    if clusters < 1 or clusters > len(values):
        raise ValueError(f"invalid cluster count {clusters} for {len(values)} values")
    if clusters == 1:
        return [0] * len(values), [float(np.mean(values))]
    data = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    model = KMeans(n_clusters=clusters, random_state=random_state, n_init=10)
    raw_labels = model.fit_predict(data)
    raw_centers = model.cluster_centers_.ravel()
    order = np.argsort(raw_centers)
    mapping = {int(raw): rank for rank, raw in enumerate(order)}
    labels = [mapping[int(label)] for label in raw_labels]
    centers = [float(raw_centers[raw]) for raw in order]
    return labels, centers


def choose_cluster_count(values: list[float], maximum: int, random_state: int) -> int:
    if len(values) <= 2:
        return 1
    data = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    best_count = 1
    best_score = -math.inf
    distinct_count = len(set(values))
    upper = min(maximum, len(values) - 1, distinct_count)
    for clusters in range(2, upper + 1):
        labels = KMeans(n_clusters=clusters, random_state=random_state, n_init=10).fit_predict(data)
        if len(set(labels)) < 2:
            continue
        # Penalise excessive thin bands produced by individual text fragments.
        score = float(silhouette_score(data, labels)) - 0.04 * clusters
        if score > best_score:
            best_score = score
            best_count = clusters
    return best_count


def nearest_columns(
    components: list[dict[str, Any]], column_centers: list[float]
) -> list[int]:
    return [
        min(
            range(len(column_centers)),
            key=lambda column: abs(component["bbox"][0] - column_centers[column]),
        )
        for component in components
    ]


def boundary_columns(
    components: list[dict[str, Any]], boundaries: list[float]
) -> list[int]:
    """Assign components to detected column intervals by their horizontal centre."""
    result = []
    for component in components:
        center = float(component["center"][0])
        matches = [
            index
            for index, (left, right) in enumerate(zip(boundaries, boundaries[1:]))
            if left <= center < right
            or (index == len(boundaries) - 2 and center == right)
        ]
        if matches:
            result.append(matches[0])
        else:
            result.append(
                min(
                    range(len(boundaries) - 1),
                    key=lambda index: abs(center - (boundaries[index] + boundaries[index + 1]) / 2),
                )
            )
    return result


def boundary_rows(
    components: list[dict[str, Any]], boundaries: list[float]
) -> list[int]:
    """Assign components to detected row intervals by their vertical centre."""
    result = []
    for component in components:
        center = float(component["center"][1])
        matches = [
            index
            for index, (top, bottom) in enumerate(zip(boundaries, boundaries[1:]))
            if top <= center < bottom
            or (index == len(boundaries) - 2 and center == bottom)
        ]
        if matches:
            result.append(matches[0])
        else:
            result.append(
                min(
                    range(len(boundaries) - 1),
                    key=lambda index: abs(
                        center - (boundaries[index] + boundaries[index + 1]) / 2
                    ),
                )
            )
    return result


def split_components_on_horizontal_gaps(
    components: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    *,
    minimum_fragments: int = 3,
    minimum_absolute_gap: float = 18.0,
    height_ratio: float = 1.25,
    local_gap_ratio: float = 3.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Undo obvious cross-column SAME_CELL merges in horizontal partial grids.

    A relation classifier can join two neighbouring header cells because their
    words share a text line.  We only split components containing at least
    three nearly collinear fragments and only at a gap that is simultaneously
    large in pixels, relative to text height, and relative to the other word
    gaps in that component.  This deliberately leaves multiline components and
    ordinary two-fragment cells unchanged.
    """
    fragments_by_id = {fragment["id"]: fragment for fragment in fragments}
    output: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for component in components:
        items = [
            fragments_by_id[fragment_id]
            for fragment_id in component.get("fragment_ids", [])
            if fragment_id in fragments_by_id
        ]
        if len(items) < minimum_fragments:
            output.append(component)
            continue
        heights = sorted(max(1.0, item["bbox"][3] - item["bbox"][1]) for item in items)
        median_height = heights[len(heights) // 2]
        centers_y = [bbox_center(item["bbox"])[1] for item in items]
        if max(centers_y) - min(centers_y) > median_height * 0.75:
            output.append(component)
            continue

        ordered = sorted(items, key=lambda item: (item["bbox"][0], item["bbox"][1]))
        gaps = [
            max(0.0, float(right["bbox"][0]) - float(left["bbox"][2]))
            for left, right in zip(ordered, ordered[1:])
        ]
        if not gaps:
            output.append(component)
            continue
        sorted_gaps = sorted(gaps)
        baseline_gaps = sorted_gaps[:-1] or sorted_gaps
        local_baseline = baseline_gaps[len(baseline_gaps) // 2]
        threshold = max(
            minimum_absolute_gap,
            median_height * height_ratio,
            local_baseline * local_gap_ratio,
        )
        split_after = {index for index, gap in enumerate(gaps) if gap > threshold}
        if not split_after:
            output.append(component)
            continue

        groups: list[list[dict[str, Any]]] = [[]]
        for index, fragment in enumerate(ordered):
            groups[-1].append(fragment)
            if index in split_after:
                groups.append([])
        groups = [group for group in groups if group]
        if len(groups) < 2:
            output.append(component)
            continue

        part_ids = []
        for part_index, group in enumerate(groups):
            bbox = bbox_union(group)
            part = dict(component)
            part["id"] = f"{component['id']}_gap_part_{part_index}"
            part["fragment_ids"] = [fragment["id"] for fragment in group]
            part["text"] = join_text(group)
            part["bbox"] = bbox
            part["center"] = list(bbox_center(bbox))
            part["split_from_component_id"] = component["id"]
            part["horizontal_gap_split"] = True
            output.append(part)
            part_ids.append(part["id"])
        trace.append(
            {
                "source_component_id": component["id"],
                "source_fragment_ids": component.get("fragment_ids", []),
                "part_ids": part_ids,
                "gaps": [round(value, 3) for value in gaps],
                "threshold": round(threshold, 3),
            }
        )
    return sorted(output, key=lambda item: (item["center"][1], item["center"][0])), trace


def coalesce_identical_logical_cells(
    components: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge components assigned to exactly the same logical cell rectangle.

    Low-confidence punctuation is intentionally retained during reconstruction,
    even when it was absent from relation prediction.  It therefore starts as a
    singleton component.  Once grid coordinates are known, a singleton and the
    neighbouring high-confidence text can occupy the exact same logical cell;
    keeping both would create an invalid collision and an artificial fragment
    mismatch.  Components with different spans are left untouched.
    """
    fragments_by_id = {fragment["id"]: fragment for fragment in fragments}
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for component in components:
        key = (
            int(component["row_start"]),
            int(component["row_end"]),
            int(component["column_start"]),
            int(component["column_end"]),
        )
        groups.setdefault(key, []).append(component)

    output: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for key, group in groups.items():
        if len(group) == 1:
            output.append(group[0])
            continue
        fragment_ids = list(
            dict.fromkeys(
                fragment_id
                for component in group
                for fragment_id in component.get("fragment_ids", [])
            )
        )
        items = [
            fragments_by_id[fragment_id]
            for fragment_id in fragment_ids
            if fragment_id in fragments_by_id
        ]
        if not items:
            output.extend(group)
            continue
        bbox = bbox_union(items)
        merged = dict(group[0])
        merged["fragment_ids"] = [
            fragment["id"]
            for fragment in sorted(
                items,
                key=lambda item: (item["bbox"][1], item["bbox"][0]),
            )
        ]
        merged["text"] = join_text(items)
        merged["bbox"] = bbox
        merged["center"] = list(bbox_center(bbox))
        merged["coalesced_component_ids"] = [item["id"] for item in group]
        output.append(merged)
        trace.append(
            {
                "logical_rectangle": list(key),
                "component_ids": [item["id"] for item in group],
                "fragment_ids": merged["fragment_ids"],
            }
        )
    return sorted(
        output,
        key=lambda item: (
            item["row_start"],
            item["column_start"],
            item["row_end"],
            item["column_end"],
        ),
    ), trace


def infer_header_spans_from_occupancy(
    components: list[dict[str, Any]],
    rows: list[int],
    columns: list[int],
    rowspans: list[int],
    colspans: list[int],
    row_count: int,
    column_count: int,
) -> list[dict[str, Any]]:
    """Infer a two-level grouped header from complementary row occupancy.

    This is used for partial grids such as web tables that expose horizontal
    rules but no vertical rules. A sparse second header row supplies child
    columns; its parent is expanded over their contiguous run, while the other
    first-row labels extend vertically through that second header row.
    """
    if row_count < 3 or column_count < 2:
        return []
    indices_by_row = {
        row: [index for index, value in enumerate(rows) if value == row]
        for row in range(row_count)
    }
    occupied_by_row = {
        row: {columns[index] for index in indices}
        for row, indices in indices_by_row.items()
    }
    body_counts = sorted(
        len(value) for row, value in occupied_by_row.items() if row >= 2 and value
    )
    if not body_counts:
        return []
    typical_body_count = body_counts[len(body_counts) // 2]
    first_columns = occupied_by_row.get(0, set())
    child_columns = occupied_by_row.get(1, set())
    if (
        not first_columns
        or not child_columns
        or len(child_columns) >= max(2, round(typical_body_count * 0.75))
        or len(first_columns | child_columns)
        < max(2, round(typical_body_count * 0.75))
    ):
        return []

    trace: list[dict[str, Any]] = []
    runs: list[list[int]] = []
    for column in sorted(child_columns):
        if runs and column == runs[-1][-1] + 1:
            runs[-1].append(column)
        else:
            runs.append([column])
    first_indices = indices_by_row[0]
    representatives: dict[int, int] = {}
    for index in first_indices:
        column = columns[index]
        current = representatives.get(column)
        if current is None or len(components[index].get("fragment_ids", [])) > len(
            components[current].get("fragment_ids", [])
        ):
            representatives[column] = index
    for run in runs:
        if len(run) < 2:
            continue
        candidates = [index for column, index in representatives.items() if column in run]
        if not candidates:
            continue
        center = sum(run) / len(run)
        parent = min(candidates, key=lambda index: abs(columns[index] - center))
        columns[parent] = run[0]
        colspans[parent] = len(run)
        trace.append(
            {
                "cell_id": components[parent]["id"],
                "kind": "grouped_header_colspan",
                "columns": [run[0], run[-1]],
            }
        )

    for index in representatives.values():
        start = columns[index]
        covered = set(range(start, start + max(1, colspans[index])))
        if not (covered & child_columns):
            rowspans[index] = max(rowspans[index], 2)
            trace.append(
                {
                    "cell_id": components[index]["id"],
                    "kind": "header_rowspan",
                    "rows": [0, 1],
                }
            )
    return trace


def merge_header_components(
    components: list[dict[str, Any]],
    header_indices: list[int],
    columns: list[int],
    fragments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge OCR lines assigned to the same column inside the header region."""
    header_set = set(header_indices)
    by_column: dict[int, list[dict[str, Any]]] = {}
    for index in header_indices:
        by_column.setdefault(columns[index], []).append(components[index])

    fragments_by_id = {fragment["id"]: fragment for fragment in fragments}
    merged: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for column, group in sorted(by_column.items()):
        fragment_ids = [
            fragment_id
            for component in group
            for fragment_id in component.get("fragment_ids", [])
        ]
        group_fragments = [fragments_by_id[fragment_id] for fragment_id in fragment_ids]
        bbox = bbox_union(group_fragments)
        merged.append(
            {
                "id": f"header_column_{column}",
                "fragment_ids": fragment_ids,
                "text": join_text(group_fragments),
                "bbox": bbox,
                "center": list(bbox_center(bbox)),
                "header_line_merge": True,
                "merged_component_ids": [component["id"] for component in group],
                "assigned_header_column": column,
            }
        )
        if len(group) > 1:
            trace.append(
                {
                    "column": column,
                    "component_ids": [component["id"] for component in group],
                    "fragment_ids": fragment_ids,
                }
            )

    body = [component for index, component in enumerate(components) if index not in header_set]
    return sorted(merged + body, key=lambda item: (item["center"][1], item["center"][0])), trace


def infer_logical_header_depth(
    occupied_columns: set[int], column_count: int, physical_row_count: int
) -> int:
    missing_interior = [
        column
        for column in range(1, max(1, column_count - 1))
        if column not in occupied_columns
    ]
    grouped_remainder = (
        column_count >= 3
        and len(occupied_columns) == 2
        and 0 in occupied_columns
        and any(column > 0 for column in occupied_columns)
    )
    if len(missing_interior) >= 2 and (physical_row_count > 1 or grouped_remainder):
        return 2
    return 1


def inferred_header_children(columns: list[int], header_depth: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"inferred_header_child_{column}",
            "fragment_ids": [], "text": "", "bbox": [],
            "row_start": 1, "row_end": header_depth - 1,
            "column_start": column, "column_end": column,
            "rowspan": header_depth - 1, "colspan": 1,
            "missing_in_ocr": True, "inferred_structure": True,
            "inference_reason": "child label under an observed grouped header is absent from OCR",
        }
        for column in columns
    ]


def inferred_missing_header_cells(
    occupied_columns: set[int], column_count: int, header_depth: int
) -> list[dict[str, Any]]:
    missing = [column for column in range(column_count) if column not in occupied_columns]
    runs: list[list[int]] = []
    for column in missing:
        if runs and column == runs[-1][-1] + 1:
            runs[-1].append(column)
        else:
            runs.append([column])

    cells: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        if header_depth > 1 and len(run) > 1:
            cells.append(
                {
                    "id": f"inferred_header_group_{run_index}",
                    "fragment_ids": [], "text": "", "bbox": [],
                    "row_start": 0, "row_end": 0,
                    "column_start": run[0], "column_end": run[-1],
                    "rowspan": 1, "colspan": len(run),
                    "missing_in_ocr": True, "inferred_structure": True,
                    "inference_reason": "consecutive header columns absent from OCR",
                }
            )
            for column in run:
                cells.append(
                    {
                        "id": f"inferred_header_child_{column}",
                        "fragment_ids": [], "text": "", "bbox": [],
                        "row_start": 1, "row_end": header_depth - 1,
                        "column_start": column, "column_end": column,
                        "rowspan": header_depth - 1, "colspan": 1,
                        "missing_in_ocr": True, "inferred_structure": True,
                        "inference_reason": "header column absent from OCR",
                    }
                )
        else:
            for column in run:
                cells.append(
                    {
                        "id": f"inferred_header_{column}",
                        "fragment_ids": [], "text": "", "bbox": [],
                        "row_start": 0, "row_end": header_depth - 1,
                        "column_start": column, "column_end": column,
                        "rowspan": header_depth, "colspan": 1,
                        "missing_in_ocr": True, "inferred_structure": True,
                        "inference_reason": "header column absent from OCR",
                    }
                )
    return cells


def aligned_row_bands(
    components: list[dict[str, Any]], indices: list[int]
) -> tuple[list[int], list[float]]:
    """Find logical row anchors supported by several horizontally aligned cells.

    Dense tables contain many equally spaced rows, for which silhouette-based model
    selection strongly underestimates the cluster count. Multiline cells, however,
    add sparse intermediate text lines. We therefore retain bands whose support is
    comparable to the densest band and attach sparse lines to the nearest anchor.
    """
    if not indices:
        return [], []
    heights = sorted(
        max(1.0, components[index]["bbox"][3] - components[index]["bbox"][1])
        for index in indices
    )
    median_height = heights[len(heights) // 2]
    tolerance = max(3.0, median_height * 0.55)
    raw_bands: list[dict[str, Any]] = []
    for index in sorted(indices, key=lambda value: components[value]["center"][1]):
        center = float(components[index]["center"][1])
        candidates = [band for band in raw_bands if abs(center - band["center"]) <= tolerance]
        if candidates:
            band = min(candidates, key=lambda value: abs(center - value["center"]))
            band["indices"].append(index)
            band["center"] = float(
                np.mean([components[item]["center"][1] for item in band["indices"]])
            )
        else:
            raw_bands.append({"center": center, "indices": [index]})

    maximum_support = max(len(band["indices"]) for band in raw_bands)
    minimum_support = max(2, math.ceil(maximum_support * 0.6))
    anchors = [band for band in raw_bands if len(band["indices"]) >= minimum_support]
    if len(anchors) < 2:
        anchors = raw_bands
    anchors.sort(key=lambda band: band["center"])

    labels = []
    assigned: list[list[int]] = [[] for _ in anchors]
    for index in indices:
        label = min(
            range(len(anchors)),
            key=lambda value: abs(components[index]["center"][1] - anchors[value]["center"]),
        )
        labels.append(label)
        assigned[label].append(index)
    centers = [
        float(np.mean([components[index]["center"][1] for index in group]))
        for group in assigned
    ]
    return labels, centers


def main() -> None:
    args = parse_args()
    ocr = load_document(args.ocr)
    prediction = json.loads(args.relations.read_text(encoding="utf-8"))
    ignored_line_artifacts = [
        fragment["id"]
        for fragment in ocr["fragments"]
        if str(fragment.get("text", "")).strip() in {"|", "│"}
    ]
    ignored_line_artifact_set = set(ignored_line_artifacts)
    fragments = [
        fragment
        for fragment in ocr["fragments"]
        if float(fragment.get("confidence", 1.0)) >= args.min_confidence
        and fragment["id"] not in ignored_line_artifact_set
    ]
    data_top = detect_data_top(fragments)
    detected_row_boundaries: list[float] = []
    if args.row_boundaries_json:
        raw_row_boundaries = [
            float(value) for value in json.loads(args.row_boundaries_json)
        ]
        ocr_height = float(
            ocr.get("image_size", {}).get("height", raw_row_boundaries[-1])
        )
        grid_height = float(args.grid_image_height or raw_row_boundaries[-1] or 1.0)
        detected_row_boundaries = [
            value * ocr_height / grid_height for value in raw_row_boundaries
        ]
    constrained_relations, rejected_merges = constrain_same_cell_edges(
        prediction["relations"],
        fragments,
        data_top,
        args.vertical_same_cell_probability,
        args.vertical_same_cell_soft_probability,
        args.vertical_same_cell_min_x_overlap,
        args.vertical_same_cell_max_gap_ratio,
        detected_row_boundaries,
    )
    components, fragment_to_component, same_cell_edges = build_components(
        fragments, constrained_relations, args.same_cell_threshold
    )
    horizontal_gap_splits: list[dict[str, Any]] = []
    if detected_row_boundaries and not args.column_boundaries_json:
        components, horizontal_gap_splits = split_components_on_horizontal_gaps(
            components, fragments
        )
        fragment_to_component = {
            fragment_id: index
            for index, component in enumerate(components)
            for fragment_id in component["fragment_ids"]
        }
    body_indices = [
        index for index, component in enumerate(components) if component["center"][1] >= data_top
    ]
    header_indices = [index for index in range(len(components)) if index not in set(body_indices)]
    if not body_indices:
        body_indices = list(range(len(components)))
        header_indices = []

    body_y = [components[index]["center"][1] for index in body_indices]
    if args.row_banding == "aligned":
        body_row_labels, body_row_centers = aligned_row_bands(components, body_indices)
        body_row_count = len(body_row_centers)
    else:
        body_row_count = choose_cluster_count(body_y, maximum=20, random_state=args.random_state)
        body_row_labels, body_row_centers = ordered_kmeans(
            body_y, body_row_count, args.random_state
        )

    if header_indices:
        header_y = [components[index]["center"][1] for index in header_indices]
        header_row_count = args.header_rows or choose_cluster_count(
            header_y, maximum=6, random_state=args.random_state
        )
        header_row_count = min(header_row_count, len(header_indices))
        header_row_labels, header_row_centers = ordered_kmeans(
            header_y, header_row_count, args.random_state
        )
    else:
        header_row_count = 0
        header_row_labels = []
        header_row_centers = []

    detected_column_boundaries: list[float] | None = None
    if args.column_boundaries_json:
        raw_boundaries = [float(value) for value in json.loads(args.column_boundaries_json)]
        if len(raw_boundaries) < 2:
            raise ValueError("--column-boundaries-json must contain at least two values")
        ocr_width = float(ocr.get("image_size", {}).get("width", raw_boundaries[-1]))
        grid_width = float(args.grid_image_width or raw_boundaries[-1] or 1.0)
        detected_column_boundaries = [value * ocr_width / grid_width for value in raw_boundaries]
        column_count = len(detected_column_boundaries) - 1
        column_centers = [
            (left + right) / 2.0
            for left, right in zip(detected_column_boundaries, detected_column_boundaries[1:])
        ]
    else:
        column_source_indices = body_indices
        inferred_from_partial_rows = None
        if detected_row_boundaries:
            preliminary_rows = boundary_rows(components, detected_row_boundaries)
            row_counts = [
                sum(label == row for label in preliminary_rows)
                for row in range(len(detected_row_boundaries) - 1)
            ]
            nonempty_counts = sorted(count for count in row_counts if count > 0)
            if nonempty_counts:
                inferred_from_partial_rows = nonempty_counts[len(nonempty_counts) // 2]
                column_source_indices = list(range(len(components)))
        body_x = [components[index]["bbox"][0] for index in column_source_indices]
        column_count = (
            args.expected_columns
            or inferred_from_partial_rows
            or choose_cluster_count(body_x, maximum=12, random_state=args.random_state)
        )
        column_count = min(12, column_count, len(column_source_indices))
        _, column_centers = ordered_kmeans(body_x, column_count, args.random_state)

    def assign_component_columns() -> list[int]:
        if detected_column_boundaries is not None:
            return boundary_columns(components, detected_column_boundaries)
        return nearest_columns(components, column_centers)

    physical_header_row_count = header_row_count
    header_merge_trace: list[dict[str, Any]] = []
    if args.merge_header_lines and header_indices and not detected_row_boundaries:
        preliminary_columns = assign_component_columns()
        components, header_merge_trace = merge_header_components(
            components, header_indices, preliminary_columns, fragments
        )
        fragment_to_component = {
            fragment_id: index
            for index, component in enumerate(components)
            for fragment_id in component["fragment_ids"]
        }
        body_indices = [
            index
            for index, component in enumerate(components)
            if component["center"][1] >= data_top
        ]
        header_indices = [
            index for index in range(len(components)) if index not in set(body_indices)
        ]
        body_y = [components[index]["center"][1] for index in body_indices]
        if args.row_banding == "aligned":
            body_row_labels, body_row_centers = aligned_row_bands(components, body_indices)
            body_row_count = len(body_row_centers)
        else:
            body_row_count = choose_cluster_count(body_y, maximum=20, random_state=args.random_state)
            body_row_labels, body_row_centers = ordered_kmeans(
                body_y, body_row_count, args.random_state
            )
        occupied_header_columns = {
            int(components[index]["assigned_header_column"])
            for index in header_indices
        }
        header_row_count = args.logical_header_rows or infer_logical_header_depth(
            occupied_header_columns, column_count, physical_header_row_count
        )
        header_row_labels = [0] * len(header_indices)
        mean_header_y = float(
            np.mean([components[index]["center"][1] for index in header_indices])
        )
        header_row_centers = [mean_header_y] + [None] * (header_row_count - 1)

    edges = build_component_edges(
        constrained_relations, components, fragment_to_component, args.edge_threshold
    )

    if detected_row_boundaries:
        rows = boundary_rows(components, detected_row_boundaries)
        reconstructed_row_count = len(detected_row_boundaries) - 1
        output_row_centers = [
            (top + bottom) / 2.0
            for top, bottom in zip(
                detected_row_boundaries, detected_row_boundaries[1:]
            )
        ]
    else:
        rows = [0] * len(components)
        for index, label in zip(header_indices, header_row_labels):
            rows[index] = label
        for index, label in zip(body_indices, body_row_labels):
            rows[index] = header_row_count + label
        reconstructed_row_count = header_row_count + body_row_count
        output_row_centers = header_row_centers + body_row_centers
    columns = assign_component_columns()
    rowspans, colspans = infer_spans(components, edges, rows, columns)
    occupancy_span_trace: list[dict[str, Any]] = []
    if detected_row_boundaries and not detected_column_boundaries:
        occupancy_span_trace = infer_header_spans_from_occupancy(
            components,
            rows,
            columns,
            rowspans,
            colspans,
            reconstructed_row_count,
            column_count,
        )
    grouped_header_columns: list[int] = []
    if args.merge_header_lines and not detected_row_boundaries:
        for index in header_indices:
            rows[index] = 0
            rowspans[index] = header_row_count
        nonzero_header_indices = [index for index in header_indices if columns[index] > 0]
        zero_header_indices = [index for index in header_indices if columns[index] == 0]
        if (
            header_row_count > 1
            and column_count >= 3
            and len(header_indices) == 2
            and len(zero_header_indices) == 1
            and len(nonzero_header_indices) == 1
        ):
            group_index = nonzero_header_indices[0]
            columns[group_index] = 1
            colspans[group_index] = column_count - 1
            rowspans[group_index] = 1
            grouped_header_columns = list(range(1, column_count))

    for index, component in enumerate(components):
        component.update(
            {
                "row_start": rows[index],
                "row_end": rows[index] + rowspans[index] - 1,
                "column_start": columns[index],
                "column_end": columns[index] + colspans[index] - 1,
                "rowspan": rowspans[index],
                "colspan": colspans[index],
            }
        )
    components, same_slot_merge_trace = coalesce_identical_logical_cells(
        components, fragments
    )
    occupancy: dict[tuple[int, int], list[str]] = {}
    for component in components:
        occupancy.setdefault(
            (component["row_start"], component["column_start"]), []
        ).append(component["id"])
    warnings = [
        f"cell collision at {list(position)}: {ids}"
        for position, ids in sorted(occupancy.items())
        if len(ids) > 1
    ]

    inferred_cells: list[dict[str, Any]] = []
    if args.merge_header_lines and args.infer_missing_header_cells:
        if grouped_header_columns:
            inferred_cells = inferred_header_children(grouped_header_columns, header_row_count)
        else:
            occupied_header_columns = {
                int(component["column_start"])
                for component in components
                if int(component["row_start"]) < max(1, header_row_count)
            }
            inferred_cells = inferred_missing_header_cells(
                occupied_header_columns, column_count, header_row_count
            )
        components.extend(inferred_cells)

    result = {
        "schema_version": "1.0",
        "method": "hybrid_relation_geometry_reconstruction_v2_adaptive_vertical_merge",
        "source_ocr": args.ocr.name,
        "source_relations": args.relations.name,
        "parameters": {
            "min_confidence": args.min_confidence,
            "same_cell_threshold": args.same_cell_threshold,
            "vertical_same_cell_probability": args.vertical_same_cell_probability,
            "vertical_same_cell_soft_probability": (
                args.vertical_same_cell_soft_probability
            ),
            "vertical_same_cell_min_x_overlap": args.vertical_same_cell_min_x_overlap,
            "vertical_same_cell_max_gap_ratio": args.vertical_same_cell_max_gap_ratio,
            "edge_threshold": args.edge_threshold,
            "expected_columns": args.expected_columns,
            "detected_column_boundaries": detected_column_boundaries,
            "detected_row_boundaries": detected_row_boundaries,
            "header_rows": args.header_rows,
            "merge_header_lines": args.merge_header_lines,
            "logical_header_rows": args.logical_header_rows,
            "infer_missing_header_cells": args.infer_missing_header_cells,
            "row_banding": args.row_banding,
            "random_state": args.random_state,
        },
        "detected_data_top": data_top,
        "row_centers": output_row_centers,
        "column_anchors": column_centers,
        "fragment_count": len(fragments),
        "ignored_line_artifact_ids": ignored_line_artifacts,
        "logical_cell_count": len(components),
        "observed_logical_cell_count": len(components) - len(inferred_cells),
        "inferred_cell_count": len(inferred_cells),
        "row_count": reconstructed_row_count,
        "column_count": column_count,
        "rejected_same_cell_edges": rejected_merges,
        "geometry_supported_vertical_same_cell_edges": [
            {
                "source_fragment_id": item["source_fragment_id"],
                "target_fragment_id": item["target_fragment_id"],
                "decision": item["vertical_same_cell_decision"],
                "diagnostics": item["vertical_same_cell_diagnostics"],
            }
            for item in constrained_relations
            if item.get("vertical_same_cell_decision")
            == "geometry_supported_soft_probability"
        ],
        "same_cell_edges": same_cell_edges,
        "horizontal_gap_component_splits": horizontal_gap_splits,
        "same_slot_component_merges": same_slot_merge_trace,
        "header_line_merges": header_merge_trace,
        "component_edges": edges,
        "occupancy_inferred_header_spans": occupancy_span_trace,
        "warnings": warnings,
        "cells": components,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detected data top: {data_top:.1f}")
    print(f"Rejected vertical SAME_CELL edges: {len(rejected_merges)}")
    print(
        "Geometry-supported soft vertical SAME_CELL edges: "
        f"{len(result['geometry_supported_vertical_same_cell_edges'])}"
    )
    print(f"Logical cells: {len(components)}")
    if inferred_cells:
        print(f"Inferred empty header cells: {len(inferred_cells)}")
    print(f"Rows: {result['row_count']}")
    print(f"Columns: {result['column_count']}")
    print(f"Column anchors: {[round(value, 1) for value in column_centers]}")
    print(f"Warnings: {len(warnings)}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
