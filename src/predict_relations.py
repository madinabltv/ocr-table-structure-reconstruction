from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from build_relation_baseline import (
    candidate_pairs,
    image_size,
    load_document,
    pair_features,
)
from relation_features import combined_features, normalized_text
from relation_features import feature_names as relation_feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict OCR-fragment relations")
    parser.add_argument("--input", required=True, type=Path, help="OCR JSON")
    parser.add_argument("--model", required=True, type=Path, help="Joblib model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--max-distance-ratio", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    feature_set = bundle.get("feature_set", "geometry")
    document = load_document(args.input)
    fragments = [
        fragment
        for fragment in document["fragments"]
        if float(fragment.get("confidence", 1.0)) >= args.min_confidence
    ]
    width, height = image_size(document, fragments)
    pairs = candidate_pairs(
        fragments,
        max_neighbors=args.max_neighbors,
        max_distance=args.max_distance_ratio * math.hypot(width, height),
    )
    geometry_records = [pair_features(source, target) for source, target in pairs]
    embedding_lookup = None
    if feature_set == "geometry_text_semantic":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "semantic model requires sentence-transformers; install requirements.txt"
            ) from error
        texts = sorted(
            {
                normalized_text(fragment["text"])
                for pair in pairs
                for fragment in pair
            }
        )
        embedding_model = bundle["embedding_model"]
        print(f"Embedding model: {embedding_model}")
        print(f"Unique OCR texts to encode: {len(texts)}")
        encoder = SentenceTransformer(embedding_model, device="cpu")
        vectors = encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding_lookup = {text: vector for text, vector in zip(texts, vectors)}
    feature_records = []
    if feature_set == "geometry_text_ordered_pair":
        from ordered_pair_embeddings import (
            build_ordered_embedding_lookup,
            ordered_pair_key,
            projected_feature_names,
        )

        base_names = tuple(bundle.get("base_feature_names", relation_feature_names("geometry_text")))
        for (source, target), geometry in zip(pairs, geometry_records):
            feature_records.append(
                combined_features(
                    geometry,
                    source["text"],
                    target["text"],
                    "geometry_text",
                )
            )
        pair_lookup = build_ordered_embedding_lookup(
            ((source["text"], target["text"]) for source, target in pairs),
            model_name=bundle["embedding_model"],
            max_length=int(bundle.get("ordered_pair_max_length", 128)),
            batch_size=64,
            device="cpu",
            show_progress=False,
        )
        raw_embeddings = np.stack(
            [
                pair_lookup[ordered_pair_key(source["text"], target["text"])]
                for source, target in pairs
            ]
        )
        projected = bundle["ordered_pair_projector"].transform(raw_embeddings)
        component_names = projected_feature_names(projected.shape[1])
        for row, features in zip(projected, feature_records):
            features.update(
                {name: float(value) for name, value in zip(component_names, row)}
            )
        x = np.hstack(
            [
                np.asarray(
                    [[float(features[name]) for name in base_names] for features in feature_records],
                    dtype=np.float64,
                ),
                projected,
            ]
        )
    else:
        for (source, target), geometry in zip(pairs, geometry_records):
            source_text = normalized_text(source["text"])
            target_text = normalized_text(target["text"])
            feature_records.append(
                combined_features(
                    geometry,
                    source_text,
                    target_text,
                    feature_set,
                    embedding_lookup[source_text] if embedding_lookup is not None else None,
                    embedding_lookup[target_text] if embedding_lookup is not None else None,
                )
            )
        x = np.asarray(
            [[float(features[name]) for name in feature_names] for features in feature_records],
            dtype=np.float64,
        )
    predictions = model.predict(x)
    probabilities = model.predict_proba(x) if hasattr(model, "predict_proba") else None
    model_classes = list(model.classes_) if hasattr(model, "classes_") else bundle["classes"]

    relations = []
    for index, ((source, target), features, prediction) in enumerate(
        zip(pairs, feature_records, predictions)
    ):
        item = {
            "source_fragment_id": source["id"],
            "target_fragment_id": target["id"],
            "source_text": source["text"],
            "target_text": target["text"],
            "prediction": str(prediction),
            "features": {key: round(value, 6) for key, value in geometry_records[index].items()},
            "model_features": {
                key: round(value, 6)
                for key, value in features.items()
                if not key.startswith("ordered_pair_component_")
            },
        }
        if probabilities is not None:
            item["probabilities"] = {
                str(label): round(float(probabilities[index, class_index]), 6)
                for class_index, label in enumerate(model_classes)
            }
            item["confidence"] = max(item["probabilities"].values())
        relations.append(item)

    prediction_counts = Counter(item["prediction"] for item in relations)
    result = {
        "schema_version": "1.0",
        "task": "ocr_fragment_relation_classification",
        "method": "trained_geometric_classifier",
        "source": args.input.name,
        "model": args.model.name,
        "relation_classes": bundle["classes"],
        "feature_names": feature_names,
        "feature_set": feature_set,
        "parameters": {
            "min_confidence": args.min_confidence,
            "max_neighbors": args.max_neighbors,
            "max_distance_ratio": args.max_distance_ratio,
        },
        "image_size": {"width": width, "height": height},
        "fragment_count": len(fragments),
        "candidate_pair_count": len(relations),
        "prediction_counts": {
            label: prediction_counts.get(label, 0) for label in bundle["classes"]
        },
        "relations": relations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Fragments: {len(fragments)}")
    print(f"Candidate pairs: {len(relations)}")
    for label in bundle["classes"]:
        print(f"{label}: {prediction_counts.get(label, 0)}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
