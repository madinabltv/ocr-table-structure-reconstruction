from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from build_relation_baseline import candidate_pairs, image_size, load_document, pair_features
from label_relations_from_cells import cell_index, ground_truth_relation


CLASSES = ("SAME_CELL", "RIGHT", "BELOW", "NO_RELATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real-OCR relation training JSONL")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--max-distance-ratio", type=float, default=0.4)
    return parser.parse_args()


def examples_for_document(
    document_id: str,
    ocr: dict[str, Any],
    annotation: dict[str, Any],
    max_neighbors: int,
    max_distance_ratio: float,
) -> list[dict[str, Any]]:
    cells, fragment_to_cell = cell_index(annotation)
    ignored = set(annotation.get("ignored_fragment_ids", []))
    fragments = [
        fragment
        for fragment in ocr["fragments"]
        if fragment["id"] not in ignored and fragment["id"] in fragment_to_cell
    ]
    width, height = image_size(ocr, fragments)
    pairs = candidate_pairs(
        fragments,
        max_neighbors=max_neighbors,
        max_distance=max_distance_ratio * math.hypot(width, height),
    )
    result = []
    for source, target in pairs:
        source_cell = cells[fragment_to_cell[source["id"]]]
        target_cell = cells[fragment_to_cell[target["id"]]]
        result.append(
            {
                "document_id": document_id,
                "source_fragment_id": source["id"],
                "target_fragment_id": target["id"],
                "source_text": source["text"],
                "target_text": target["text"],
                "label": ground_truth_relation(source_cell, target_cell),
                "features": pair_features(source, target),
                "domain": "russian_ocr",
            }
        )
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent.parent
    examples: list[dict[str, Any]] = []
    table_counts: dict[str, int] = {}
    for item in manifest["documents"]:
        ocr = load_document(root / item["ocr"])
        annotation = json.loads((root / item["ground_truth"]).read_text(encoding="utf-8"))
        current = examples_for_document(
            item["id"], ocr, annotation, args.max_neighbors, args.max_distance_ratio
        )
        examples.extend(current)
        table_counts[item["id"]] = len(current)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    counts = Counter(example["label"] for example in examples)
    print(f"Tables: {len(table_counts)}")
    print(f"Examples: {len(examples)}")
    for label in CLASSES:
        print(f"{label}: {counts[label]}")
    print(f"JSONL: {args.output}")


if __name__ == "__main__":
    main()
