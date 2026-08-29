from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RELATIONS = ("SAME_CELL", "RIGHT", "BELOW", "NO_RELATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build geometric pair features and rule-based relations."
    )
    parser.add_argument("--input", required=True, type=Path, help="OCR JSON")
    parser.add_argument("--output", required=True, type=Path, help="Result JSON")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=4,
        help="Maximum nearest candidate fragments retained per fragment",
    )
    parser.add_argument(
        "--max-distance-ratio",
        type=float,
        default=0.35,
        help="Maximum pair distance divided by image diagonal",
    )
    return parser.parse_args()


def validate_bbox(bbox: Any) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"bbox must be [x1, y1, x2, y2], got {bbox!r}")
    values = [float(value) for value in bbox]
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"bbox has non-positive size: {bbox!r}")
    return values


def load_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if "fragments" not in document or not isinstance(document["fragments"], list):
        raise ValueError("input JSON must contain a fragments array")

    seen: set[Any] = set()
    for fragment in document["fragments"]:
        for key in ("id", "text", "bbox"):
            if key not in fragment:
                raise ValueError(f"fragment is missing {key!r}: {fragment!r}")
        if fragment["id"] in seen:
            raise ValueError(f"duplicate fragment id: {fragment['id']!r}")
        seen.add(fragment["id"])
        fragment["bbox"] = validate_bbox(fragment["bbox"])
        fragment.setdefault("confidence", 1.0)
    return document


def geometry(fragment: dict[str, Any]) -> dict[str, float]:
    x1, y1, x2, y2 = fragment["bbox"]
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
    }


def overlap_ratio(a1: float, a2: float, b1: float, b2: float) -> float:
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    denominator = max(1e-9, min(a2 - a1, b2 - b1))
    return overlap / denominator


def pair_features(source: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    a = geometry(source)
    b = geometry(target)
    mean_width = max(1e-9, (a["width"] + b["width"]) / 2.0)
    mean_height = max(1e-9, (a["height"] + b["height"]) / 2.0)
    horizontal_gap = max(0.0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
    vertical_gap = max(0.0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))
    dx = b["cx"] - a["cx"]
    dy = b["cy"] - a["cy"]
    return {
        "dx": dx,
        "dy": dy,
        "abs_dx": abs(dx),
        "abs_dy": abs(dy),
        "center_distance": math.hypot(dx, dy),
        "horizontal_gap": horizontal_gap,
        "vertical_gap": vertical_gap,
        "horizontal_gap_norm": horizontal_gap / mean_height,
        "vertical_gap_norm": vertical_gap / mean_height,
        "x_overlap": overlap_ratio(a["x1"], a["x2"], b["x1"], b["x2"]),
        "y_overlap": overlap_ratio(a["y1"], a["y2"], b["y1"], b["y2"]),
        "width_ratio": min(a["width"], b["width"]) / max(a["width"], b["width"]),
        "height_ratio": min(a["height"], b["height"]) / max(a["height"], b["height"]),
        "dx_norm": dx / mean_width,
        "dy_norm": dy / mean_height,
        "abs_dx_norm": abs(dx) / mean_width,
        "abs_dy_norm": abs(dy) / mean_height,
        "center_distance_norm": math.hypot(dx / mean_width, dy / mean_height),
    }


def canonical_order(a: dict[str, Any], b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ga, gb = geometry(a), geometry(b)
    same_visual_row = abs(ga["cy"] - gb["cy"]) <= 0.6 * max(ga["height"], gb["height"])
    if same_visual_row:
        return (a, b) if ga["cx"] <= gb["cx"] else (b, a)
    return (a, b) if ga["cy"] <= gb["cy"] else (b, a)


def candidate_pairs(
    fragments: list[dict[str, Any]], max_neighbors: int, max_distance: float
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for fragment in fragments:
        distances = []
        center_a = geometry(fragment)
        for other in fragments:
            if other["id"] == fragment["id"]:
                continue
            center_b = geometry(other)
            distance = math.hypot(
                center_b["cx"] - center_a["cx"], center_b["cy"] - center_a["cy"]
            )
            if distance <= max_distance:
                distances.append((distance, str(other["id"]), other))
        for _, _, other in sorted(distances)[:max_neighbors]:
            source, target = canonical_order(fragment, other)
            key = tuple(sorted((str(source["id"]), str(target["id"]))))
            pairs[key] = (source, target)
    return [pairs[key] for key in sorted(pairs)]


def classify(features: dict[str, float]) -> tuple[str, str]:
    if (
        features["y_overlap"] >= 0.55
        and features["horizontal_gap_norm"] <= 1.1
        and features["abs_dy"] <= features["abs_dx"]
    ):
        return "SAME_CELL", "same text line and small horizontal gap"

    if features["y_overlap"] >= 0.55 and features["dx"] > 0:
        return "RIGHT", "strong vertical overlap and target is to the right"

    if features["x_overlap"] >= 0.25 and features["dy"] > 0:
        return "BELOW", "horizontal overlap and target is below"

    return "NO_RELATION", "no baseline rule matched"


def image_size(document: dict[str, Any], fragments: list[dict[str, Any]]) -> tuple[float, float]:
    supplied = document.get("image_size", {})
    width = float(supplied.get("width", 0))
    height = float(supplied.get("height", 0))
    if width <= 0:
        width = max((fragment["bbox"][2] for fragment in fragments), default=1.0)
    if height <= 0:
        height = max((fragment["bbox"][3] for fragment in fragments), default=1.0)
    return width, height


def main() -> None:
    args = parse_args()
    if args.max_neighbors < 1:
        raise ValueError("--max-neighbors must be at least 1")
    document = load_document(args.input)
    fragments = [
        fragment
        for fragment in document["fragments"]
        if float(fragment.get("confidence", 1.0)) >= args.min_confidence
    ]
    width, height = image_size(document, fragments)
    diagonal = math.hypot(width, height)
    pairs = candidate_pairs(
        fragments,
        max_neighbors=args.max_neighbors,
        max_distance=args.max_distance_ratio * diagonal,
    )

    relations = []
    counts = {relation: 0 for relation in RELATIONS}
    for source, target in pairs:
        features = pair_features(source, target)
        relation, reason = classify(features)
        counts[relation] += 1
        relations.append(
            {
                "source_fragment_id": source["id"],
                "target_fragment_id": target["id"],
                "source_text": source["text"],
                "target_text": target["text"],
                "prediction": relation,
                "reason": reason,
                "features": {key: round(value, 6) for key, value in features.items()},
            }
        )

    result = {
        "schema_version": "1.0",
        "task": "ocr_fragment_relation_classification",
        "method": "geometric_rules_v1",
        "source": args.input.name,
        "relation_classes": list(RELATIONS),
        "parameters": {
            "min_confidence": args.min_confidence,
            "max_neighbors": args.max_neighbors,
            "max_distance_ratio": args.max_distance_ratio,
        },
        "image_size": {"width": width, "height": height},
        "fragment_count": len(fragments),
        "candidate_pair_count": len(relations),
        "prediction_counts": counts,
        "relations": relations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Fragments: {len(fragments)}")
    print(f"Candidate pairs: {len(relations)}")
    for relation in RELATIONS:
        print(f"{relation}: {counts[relation]}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
