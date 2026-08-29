from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_relation_baseline import load_document


class UnionFind:
    def __init__(self, values: list[Any]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: Any) -> Any:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: Any, right: Any) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a table relation graph")
    parser.add_argument("--ocr", required=True, type=Path)
    parser.add_argument("--relations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--same-cell-threshold", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    return parser.parse_args()


def relation_score(item: dict[str, Any], label: str) -> float:
    probabilities = item.get("probabilities")
    if isinstance(probabilities, dict) and label in probabilities:
        return float(probabilities[label])
    return 1.0 if item.get("prediction") == label else 0.0


def bbox_union(fragments: list[dict[str, Any]]) -> list[float]:
    return [
        min(fragment["bbox"][0] for fragment in fragments),
        min(fragment["bbox"][1] for fragment in fragments),
        max(fragment["bbox"][2] for fragment in fragments),
        max(fragment["bbox"][3] for fragment in fragments),
    ]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def join_text(fragments: list[dict[str, Any]]) -> str:
    heights = sorted(fragment["bbox"][3] - fragment["bbox"][1] for fragment in fragments)
    median_height = heights[len(heights) // 2]
    tolerance = max(1.0, median_height * 0.6)
    lines: list[dict[str, Any]] = []
    for fragment in sorted(
        fragments,
        key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2.0, item["bbox"][0]),
    ):
        center_y = (fragment["bbox"][1] + fragment["bbox"][3]) / 2.0
        candidates = [line for line in lines if abs(center_y - line["center_y"]) <= tolerance]
        if candidates:
            line = min(candidates, key=lambda item: abs(center_y - item["center_y"]))
            line["fragments"].append(fragment)
            centers = [
                (item["bbox"][1] + item["bbox"][3]) / 2.0 for item in line["fragments"]
            ]
            line["center_y"] = sum(centers) / len(centers)
        else:
            lines.append({"center_y": center_y, "fragments": [fragment]})
    ordered = []
    for line in sorted(lines, key=lambda item: item["center_y"]):
        ordered.extend(sorted(line["fragments"], key=lambda item: item["bbox"][0]))
    result = ""
    for fragment in ordered:
        value = str(fragment["text"]).strip()
        if not value:
            continue
        if not result:
            result = value
        elif result.endswith((",", ".")) and value[0].isdigit():
            result += value
        else:
            result += " " + value
    return result


def build_components(
    fragments: list[dict[str, Any]], relations: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], dict[Any, int], list[dict[str, Any]]]:
    ids = [fragment["id"] for fragment in fragments]
    valid_ids = set(ids)
    union_find = UnionFind(ids)
    accepted = []
    for item in relations:
        left = item["source_fragment_id"]
        right = item["target_fragment_id"]
        if left not in valid_ids or right not in valid_ids:
            continue
        score = relation_score(item, "SAME_CELL")
        if item.get("prediction") == "SAME_CELL" and score >= threshold:
            union_find.union(left, right)
            accepted.append(
                {"source_fragment_id": left, "target_fragment_id": right, "score": score}
            )

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for fragment in fragments:
        grouped[union_find.find(fragment["id"])].append(fragment)
    groups = sorted(
        grouped.values(),
        key=lambda group: (bbox_center(bbox_union(group))[1], bbox_center(bbox_union(group))[0]),
    )
    components = []
    fragment_to_component = {}
    for index, group in enumerate(groups):
        bbox = bbox_union(group)
        component = {
            "id": f"cell_{index}",
            "fragment_ids": [fragment["id"] for fragment in group],
            "text": join_text(group),
            "bbox": bbox,
            "center": list(bbox_center(bbox)),
        }
        components.append(component)
        for fragment in group:
            fragment_to_component[fragment["id"]] = index
    return components, fragment_to_component, accepted


def build_component_edges(
    relations: list[dict[str, Any]],
    components: list[dict[str, Any]],
    fragment_to_component: dict[Any, int],
    threshold: float,
) -> list[dict[str, Any]]:
    votes: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for item in relations:
        label = item.get("prediction")
        if label not in ("RIGHT", "BELOW"):
            continue
        left_id = item["source_fragment_id"]
        right_id = item["target_fragment_id"]
        if left_id not in fragment_to_component or right_id not in fragment_to_component:
            continue
        source = fragment_to_component[left_id]
        target = fragment_to_component[right_id]
        if source == target:
            continue
        score = relation_score(item, label)
        if score < threshold:
            continue
        source_center = components[source]["center"]
        target_center = components[target]["center"]
        if label == "RIGHT" and source_center[0] > target_center[0]:
            source, target = target, source
        if label == "BELOW" and source_center[1] > target_center[1]:
            source, target = target, source
        votes[(source, target, label)].append(score)

    edges = []
    for (source, target, label), scores in sorted(votes.items()):
        edges.append(
            {
                "source": components[source]["id"],
                "target": components[target]["id"],
                "source_index": source,
                "target_index": target,
                "relation": label,
                "score": round(max(scores), 6),
                "votes": len(scores),
            }
        )
    return edges


def directional_ranks(
    components: list[dict[str, Any]], edges: list[dict[str, Any]], relation: str
) -> list[int]:
    axis = 0 if relation == "RIGHT" else 1
    order = sorted(range(len(components)), key=lambda index: components[index]["center"][axis])
    predecessors: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        if edge["relation"] == relation:
            predecessors[edge["target_index"]].add(edge["source_index"])
    ranks = [0] * len(components)
    for index in order:
        earlier = [source for source in predecessors[index] if components[source]["center"][axis] <= components[index]["center"][axis]]
        if earlier:
            ranks[index] = max(ranks[source] + 1 for source in earlier)
    return ranks


def infer_spans(
    components: list[dict[str, Any]], edges: list[dict[str, Any]], rows: list[int], columns: list[int]
) -> tuple[list[int], list[int]]:
    rowspans = [1] * len(components)
    colspans = [1] * len(components)
    below_targets: dict[int, set[int]] = defaultdict(set)
    right_targets: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        if edge["relation"] == "BELOW":
            below_targets[edge["source_index"]].add(edge["target_index"])
        elif edge["relation"] == "RIGHT":
            right_targets[edge["source_index"]].add(edge["target_index"])
    for source, targets in below_targets.items():
        target_columns = sorted({columns[target] for target in targets})
        if len(target_columns) >= 2:
            colspans[source] = max(1, target_columns[-1] - target_columns[0] + 1)
            columns[source] = min(columns[source], target_columns[0])
    for source, targets in right_targets.items():
        target_rows = sorted({rows[target] for target in targets})
        if len(target_rows) >= 2:
            rowspans[source] = max(1, target_rows[-1] - target_rows[0] + 1)
            rows[source] = min(rows[source], target_rows[0])
    return rowspans, colspans


def main() -> None:
    args = parse_args()
    ocr = load_document(args.ocr)
    prediction = json.loads(args.relations.read_text(encoding="utf-8"))
    fragments = [
        fragment
        for fragment in ocr["fragments"]
        if float(fragment.get("confidence", 1.0)) >= args.min_confidence
    ]
    components, fragment_to_component, same_cell_edges = build_components(
        fragments, prediction["relations"], args.same_cell_threshold
    )
    edges = build_component_edges(
        prediction["relations"], components, fragment_to_component, args.edge_threshold
    )
    columns = directional_ranks(components, edges, "RIGHT")
    rows = directional_ranks(components, edges, "BELOW")
    rowspans, colspans = infer_spans(components, edges, rows, columns)

    warnings = []
    occupancy: dict[tuple[int, int], list[str]] = defaultdict(list)
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
        occupancy[(rows[index], columns[index])].append(component["id"])
    for position, ids in sorted(occupancy.items()):
        if len(ids) > 1:
            warnings.append(f"cell collision at {list(position)}: {ids}")

    result = {
        "schema_version": "1.0",
        "method": "relation_graph_reconstruction_v1",
        "source_ocr": args.ocr.name,
        "source_relations": args.relations.name,
        "parameters": {
            "min_confidence": args.min_confidence,
            "same_cell_threshold": args.same_cell_threshold,
            "edge_threshold": args.edge_threshold,
        },
        "fragment_count": len(fragments),
        "logical_cell_count": len(components),
        "row_count": max((component["row_end"] for component in components), default=-1) + 1,
        "column_count": max((component["column_end"] for component in components), default=-1) + 1,
        "same_cell_edges": same_cell_edges,
        "component_edges": edges,
        "warnings": warnings,
        "cells": components,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fragments: {result['fragment_count']}")
    print(f"Logical cells: {result['logical_cell_count']}")
    print(f"Rows: {result['row_count']}")
    print(f"Columns: {result['column_count']}")
    print(f"SAME_CELL edges: {len(same_cell_edges)}")
    print(f"RIGHT/BELOW edges: {len(edges)}")
    print(f"Warnings: {len(warnings)}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
