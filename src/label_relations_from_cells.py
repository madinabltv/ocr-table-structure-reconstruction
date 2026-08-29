from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_relation_baseline import candidate_pairs, image_size, load_document, pair_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create relation-training examples")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-neighbors", type=int, default=6)
    parser.add_argument("--max-distance-ratio", type=float, default=0.4)
    return parser.parse_args()


def cell_index(document: dict[str, Any]) -> tuple[dict[Any, dict[str, Any]], dict[Any, Any]]:
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise ValueError("annotated input must contain a cells array")
    by_id = {}
    fragment_to_cell = {}
    for cell in cells:
        for key in ("id", "row_start", "row_end", "column_start", "column_end", "fragment_ids"):
            if key not in cell:
                raise ValueError(f"cell is missing {key!r}: {cell!r}")
        by_id[cell["id"]] = cell
        for fragment_id in cell["fragment_ids"]:
            if fragment_id in fragment_to_cell:
                raise ValueError(f"fragment {fragment_id!r} belongs to multiple cells")
            fragment_to_cell[fragment_id] = cell["id"]
    return by_id, fragment_to_cell


def ground_truth_relation(a: dict[str, Any], b: dict[str, Any]) -> str:
    if a["id"] == b["id"]:
        return "SAME_CELL"
    rows_overlap = max(a["row_start"], b["row_start"]) <= min(a["row_end"], b["row_end"])
    if rows_overlap and a["column_end"] + 1 == b["column_start"]:
        return "RIGHT"
    columns_overlap = max(a["column_start"], b["column_start"]) <= min(
        a["column_end"], b["column_end"]
    )
    if columns_overlap and a["row_end"] + 1 == b["row_start"]:
        return "BELOW"
    return "NO_RELATION"


def main() -> None:
    args = parse_args()
    document = load_document(args.input)
    cells, fragment_to_cell = cell_index(document)
    fragments = document["fragments"]
    missing = [fragment["id"] for fragment in fragments if fragment["id"] not in fragment_to_cell]
    if missing:
        raise ValueError(f"fragments without a cell annotation: {missing}")

    width, height = image_size(document, fragments)
    pairs = candidate_pairs(
        fragments,
        max_neighbors=args.max_neighbors,
        max_distance=args.max_distance_ratio * (width**2 + height**2) ** 0.5,
    )
    examples = []
    counts = {name: 0 for name in ("SAME_CELL", "RIGHT", "BELOW", "NO_RELATION")}
    for source, target in pairs:
        source_cell = cells[fragment_to_cell[source["id"]]]
        target_cell = cells[fragment_to_cell[target["id"]]]
        label = ground_truth_relation(source_cell, target_cell)
        counts[label] += 1
        examples.append(
            {
                "source_fragment_id": source["id"],
                "target_fragment_id": target["id"],
                "source_text": source["text"],
                "target_text": target["text"],
                "label": label,
                "features": pair_features(source, target),
            }
        )

    result = {
        "schema_version": "1.0",
        "task": "ocr_fragment_relation_classification",
        "source": args.input.name,
        "class_counts": counts,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training examples: {len(examples)}")
    for label, count in counts.items():
        print(f"{label}: {count}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
