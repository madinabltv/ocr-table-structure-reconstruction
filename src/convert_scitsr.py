from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from build_relation_baseline import candidate_pairs, pair_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a SciTSR split to JSONL")
    parser.add_argument("--chunk-dir", required=True, type=Path)
    parser.add_argument("--relation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument(
        "--same-cell-rate",
        type=float,
        default=0.0,
        help="Fraction of eligible chunks split into synthetic SAME_CELL pairs",
    )
    parser.add_argument("--limit", type=int, help="Convert only the first N tables")
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for index, chunk in enumerate(source["chunks"]):
        # Official SciTSR order is [x1, x2, y1, y2] in PDF coordinates, whose
        # origin is bottom-left.  Negating and swapping Y converts it to the
        # image/OCR convention where Y grows downwards.
        x1, x2, y1, y2 = (float(value) for value in chunk["pos"])
        result.append(
            {
                "id": index,
                "text": chunk.get("text", ""),
                "bbox": [x1, -y2, x2, -y1],
                "confidence": 1.0,
            }
        )
    return result


def load_relations(path: Path) -> dict[frozenset[int], tuple[int, int]]:
    relations = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            left, right, encoded = line.split()
            relation_id, blank_count = encoded.split(":")
            relations[frozenset((int(left), int(right)))] = (
                int(relation_id),
                int(blank_count),
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid relation at {path}:{line_number}: {line!r}") from error
    return relations


def label_pair(
    source_id: int,
    target_id: int,
    relations: dict[frozenset[int], tuple[int, int]],
) -> str:
    relation = relations.get(frozenset((source_id, target_id)))
    if relation is None:
        return "NO_RELATION"
    relation_id, blank_count = relation
    if blank_count != 0:
        return "NO_RELATION"
    if relation_id == 1:
        return "RIGHT"
    if relation_id == 2:
        return "BELOW"
    raise ValueError(f"unsupported SciTSR relation id: {relation_id}")


def deterministic_fraction(document_id: str, fragment_id: int) -> float:
    digest = hashlib.sha256(f"{document_id}:{fragment_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def deterministic_value(document_id: str, fragment_id: int, salt: str) -> float:
    digest = hashlib.sha256(
        f"{document_id}:{fragment_id}:{salt}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_text(text: str) -> tuple[str, str] | None:
    cleaned = text.strip()
    if len(cleaned) < 4:
        return None
    words = cleaned.split()
    if len(words) >= 2:
        cut = max(1, len(words) // 2)
        return " ".join(words[:cut]), " ".join(words[cut:])
    cut = len(cleaned) // 2
    if cut < 2 or len(cleaned) - cut < 2:
        return None
    return cleaned[:cut], cleaned[cut:]


def synthetic_same_cell_examples(
    table_id: str, chunks: list[dict[str, Any]], rate: float
) -> Iterator[dict[str, Any]]:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("--same-cell-rate must be between 0 and 1")
    for chunk in chunks:
        parts = split_text(chunk["text"])
        if parts is None or deterministic_fraction(table_id, chunk["id"]) >= rate:
            continue
        left_text, right_text = parts
        x1, y1, x2, y2 = chunk["bbox"]
        width = x2 - x1
        height = y2 - y1
        text_total = max(1, len(left_text) + len(right_text))
        variant = deterministic_value(table_id, chunk["id"], "orientation")
        if variant < 0.65:
            boundary = x1 + width * len(left_text) / text_total
            gap_factor = 0.15 + 0.70 * deterministic_value(
                table_id, chunk["id"], "horizontal-gap"
            )
            gap = min(width * 0.25, max(0.2, height * gap_factor))
            left_bbox = [x1, y1, max(x1 + 0.1, boundary - gap / 2), y2]
            right_bbox = [min(x2 - 0.1, boundary + gap / 2), y1, x2, y2]
            synthetic_variant = "horizontal"
        else:
            gap_factor = 0.10 + 0.35 * deterministic_value(
                table_id, chunk["id"], "vertical-gap"
            )
            gap = height * gap_factor
            band_height = max(0.1, (height - gap) / 2)
            shift = width * 0.25 * (
                deterministic_value(table_id, chunk["id"], "x-shift") - 0.5
            )
            left_bbox = [x1, y1, x2, y1 + band_height]
            right_bbox = [
                max(x1, x1 + shift),
                y2 - band_height,
                min(x2, x2 + shift),
                y2,
            ]
            synthetic_variant = "multiline"
        left = {"id": f"{chunk['id']}:a", "text": left_text, "bbox": left_bbox}
        right = {"id": f"{chunk['id']}:b", "text": right_text, "bbox": right_bbox}
        yield {
            "document_id": table_id,
            "source_fragment_id": left["id"],
            "target_fragment_id": right["id"],
            "source_text": left_text,
            "target_text": right_text,
            "label": "SAME_CELL",
            "synthetic": True,
            "synthetic_variant": synthetic_variant,
            "original_fragment_id": chunk["id"],
            "features": pair_features(left, right),
        }


def table_examples(
    table_id: str,
    chunks: list[dict[str, Any]],
    relations: dict[frozenset[int], tuple[int, int]],
    max_neighbors: int,
) -> Iterator[dict[str, Any]]:
    if not chunks:
        return
    width = max(chunk["bbox"][2] for chunk in chunks) - min(
        chunk["bbox"][0] for chunk in chunks
    )
    height = max(chunk["bbox"][3] for chunk in chunks) - min(
        chunk["bbox"][1] for chunk in chunks
    )
    max_distance = (width**2 + height**2) ** 0.5
    for source, target in candidate_pairs(chunks, max_neighbors, max_distance):
        yield {
            "document_id": table_id,
            "source_fragment_id": source["id"],
            "target_fragment_id": target["id"],
            "source_text": source["text"],
            "target_text": target["text"],
            "label": label_pair(source["id"], target["id"], relations),
            "features": pair_features(source, target),
        }


def main() -> None:
    args = parse_args()
    chunk_files = sorted(args.chunk_dir.glob("*.chunk"))
    if args.limit is not None:
        chunk_files = chunk_files[: args.limit]
    if not chunk_files:
        raise FileNotFoundError(f"no .chunk files found in {args.chunk_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_count = 0
    example_count = 0
    class_counts = {"SAME_CELL": 0, "RIGHT": 0, "BELOW": 0, "NO_RELATION": 0}
    with args.output.open("w", encoding="utf-8") as stream:
        for chunk_path in chunk_files:
            relation_path = args.relation_dir / f"{chunk_path.stem}.rel"
            if not relation_path.exists():
                raise FileNotFoundError(f"missing relation file: {relation_path}")
            chunks = load_chunks(chunk_path)
            relations = load_relations(relation_path)
            for example in table_examples(
                chunk_path.stem, chunks, relations, args.max_neighbors
            ):
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
                class_counts[example["label"]] += 1
                example_count += 1
            for example in synthetic_same_cell_examples(
                chunk_path.stem, chunks, args.same_cell_rate
            ):
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
                class_counts[example["label"]] += 1
                example_count += 1
            table_count += 1

    print(f"Tables: {table_count}")
    print(f"Examples: {example_count}")
    for label, count in class_counts.items():
        print(f"{label}: {count}")
    print(f"JSONL: {args.output}")


if __name__ == "__main__":
    main()
