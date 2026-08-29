 from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from relation_features import combined_features, feature_names, normalized_text
from relation_model import TwoStageExtraTreesClassifier


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


CLASSES = ("SAME_CELL", "RIGHT", "BELOW", "NO_RELATION")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OCR relation classifier")
    parser.add_argument("--input", required=True, type=Path, help="JSONL examples")
    parser.add_argument(
        "--extra-training-input",
        action="append",
        default=[],
        type=Path,
        help="Additional JSONL used only for training; may be repeated",
    )
    parser.add_argument(
        "--extra-training-weight",
        type=int,
        default=1,
        help="Repeat every extra-training example this many times",
    )
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        help="Maximum NO_RELATION/positive ratio in training; omit to keep all",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--feature-set",
        choices=("geometry", "geometry_text", "geometry_text_semantic"),
        default="geometry",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--classifier",
        choices=("logreg", "two_stage_extra_trees"),
        default="logreg",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--same-cell-threshold", type=float, default=0.45)
    return parser.parse_args()


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        if item.get("label") not in CLASSES:
            raise ValueError(f"unsupported label at line {line_number}: {item.get('label')!r}")
        geometry = item.get("features", {})
        missing = [name for name in feature_names("geometry") if name not in geometry]
        if missing:
            raise ValueError(f"missing features at line {line_number}: {missing}")
        examples.append(item)
    if not examples:
        raise ValueError("input contains no examples")
    return examples


def build_embedding_lookup(
    examples: list[dict[str, Any]], model_name: str, cache_path: Path | None, batch_size: int
) -> dict[str, np.ndarray]:
    texts = sorted(
        {
            normalized_text(item.get(field, ""))
            for item in examples
            for field in ("source_text", "target_text")
        }
    )
    if cache_path and cache_path.exists():
        cached = joblib.load(cache_path)
        if cached.get("model_name") != model_name:
            raise ValueError("embedding cache was created with a different model")
        lookup = {
            text: vector for text, vector in zip(cached["texts"], cached["embeddings"])
        }
        missing = [text for text in texts if text not in lookup]
        if not missing:
            print(f"Embedding cache: loaded {len(lookup)} texts")
            return lookup
        print(f"Embedding cache is missing {len(missing)} texts; rebuilding")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "semantic features require sentence-transformers; install requirements.txt"
        ) from error
    print(f"Embedding model: {model_name}")
    print(f"Unique texts to encode: {len(texts)}")
    encoder = SentenceTransformer(model_name, device="cpu")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model_name": model_name, "texts": texts, "embeddings": embeddings},
            cache_path,
        )
        print(f"Embedding cache saved: {cache_path}")
    return {text: vector for text, vector in zip(texts, embeddings)}


def attach_model_features(
    examples: list[dict[str, Any]], feature_set: str, embedding_lookup: dict[str, np.ndarray] | None
) -> None:
    for item in examples:
        source_text = normalized_text(item.get("source_text", ""))
        target_text = normalized_text(item.get("target_text", ""))
        source_embedding = embedding_lookup[source_text] if embedding_lookup is not None else None
        target_embedding = embedding_lookup[target_text] if embedding_lookup is not None else None
        item["model_features"] = combined_features(
            item["features"],
            source_text,
            target_text,
            feature_set,
            source_embedding,
            target_embedding,
        )


def split_by_document(
    examples: list[dict[str, Any]], validation_ratio: float, random_state: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1")
    document_ids = sorted({str(item["document_id"]) for item in examples})
    if len(document_ids) < 2:
        raise ValueError("at least two document_id values are required")
    rng = random.Random(random_state)
    rng.shuffle(document_ids)
    validation_count = max(1, round(len(document_ids) * validation_ratio))
    validation_ids = set(document_ids[:validation_count])
    train = [item for item in examples if str(item["document_id"]) not in validation_ids]
    validation = [item for item in examples if str(item["document_id"]) in validation_ids]
    return train, validation, sorted(set(document_ids) - validation_ids), sorted(validation_ids)


def downsample_negatives(
    examples: list[dict[str, Any]], ratio: float | None, random_state: int
) -> list[dict[str, Any]]:
    if ratio is None:
        return examples
    if ratio <= 0:
        raise ValueError("--negative-ratio must be positive")
    positive = [item for item in examples if item["label"] != "NO_RELATION"]
    negative = [item for item in examples if item["label"] == "NO_RELATION"]
    keep = min(len(negative), round(len(positive) * ratio))
    rng = random.Random(random_state)
    selected_negative = rng.sample(negative, keep)
    result = positive + selected_negative
    rng.shuffle(result)
    return result


def arrays(
    examples: list[dict[str, Any]], selected_features: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(item["model_features"][name]) for name in selected_features] for item in examples],
        dtype=np.float64,
    )
    y = np.asarray([item["label"] for item in examples])
    return x, y


def counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(item["label"] for item in examples)
    return {label: counter[label] for label in CLASSES}


def main() -> None:
    args = parse_args()
    if args.extra_training_weight < 1:
        raise ValueError("--extra-training-weight must be at least 1")
    selected_features = feature_names(args.feature_set)
    examples = load_examples(args.input)
    extra_examples = [
        item
        for path in args.extra_training_input
        for item in load_examples(path)
    ]
    all_feature_examples = examples + extra_examples
    embedding_lookup = None
    if args.feature_set == "geometry_text_semantic":
        embedding_lookup = build_embedding_lookup(
            all_feature_examples,
            args.embedding_model,
            args.embedding_cache,
            args.embedding_batch_size,
        )
    attach_model_features(all_feature_examples, args.feature_set, embedding_lookup)
    train_all, validation, train_ids, validation_ids = split_by_document(
        examples, args.validation_ratio, args.random_state
    )
    weighted_extra = [
        item
        for item in extra_examples
        for _ in range(args.extra_training_weight)
    ]
    train_all = train_all + weighted_extra
    train = downsample_negatives(train_all, args.negative_ratio, args.random_state)
    train_labels = {item["label"] for item in train}
    missing_classes = set(CLASSES) - train_labels
    if missing_classes:
        raise ValueError(f"training split is missing classes: {sorted(missing_classes)}")

    x_train, y_train = arrays(train, selected_features)
    x_validation, y_validation = arrays(validation, selected_features)
    if args.classifier == "two_stage_extra_trees":
        model = TwoStageExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            same_cell_threshold=args.same_cell_threshold,
        )
    else:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=args.max_iter,
                        random_state=args.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    model.fit(x_train, y_train)
    predicted = model.predict(x_validation)

    matrix = confusion_matrix(y_validation, predicted, labels=CLASSES)
    class_report = classification_report(
        y_validation,
        predicted,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    report = {
        "schema_version": "1.0",
        "method": (
            f"two_stage_extra_trees_{args.feature_set}_v5"
            if args.classifier == "two_stage_extra_trees"
            else (
            "normalized_geometry_text_semantic_logistic_regression_v4"
            if args.feature_set == "geometry_text_semantic"
            else (
                "normalized_geometry_text_logistic_regression_v3"
                if args.feature_set == "geometry_text"
                else "normalized_geometry_logistic_regression_v2"
            )
            )
        ),
        "input": args.input.name,
        "feature_set": args.feature_set,
        "feature_names": list(selected_features),
        "classes": list(CLASSES),
        "parameters": {
            "validation_ratio": args.validation_ratio,
            "random_state": args.random_state,
            "negative_ratio": args.negative_ratio,
            "class_weight": "balanced",
            "max_iter": args.max_iter,
            "feature_set": args.feature_set,
            "embedding_model": (
                args.embedding_model if args.feature_set == "geometry_text_semantic" else None
            ),
            "classifier": args.classifier,
            "n_estimators": args.n_estimators if args.classifier == "two_stage_extra_trees" else None,
            "max_depth": args.max_depth if args.classifier == "two_stage_extra_trees" else None,
            "min_samples_leaf": (
                args.min_samples_leaf if args.classifier == "two_stage_extra_trees" else None
            ),
            "same_cell_threshold": (
                args.same_cell_threshold if args.classifier == "two_stage_extra_trees" else None
            ),
            "extra_training_inputs": [path.name for path in args.extra_training_input],
            "extra_training_weight": args.extra_training_weight,
        },
        "train_document_count": len(train_ids),
        "validation_document_count": len(validation_ids),
        "train_example_count_before_sampling": len(train_all),
        "train_example_count": len(train),
        "validation_example_count": len(validation),
        "extra_training_example_count_before_weighting": len(extra_examples),
        "extra_training_example_count_after_weighting": len(weighted_extra),
        "train_class_counts": counts(train),
        "validation_class_counts": counts(validation),
        "accuracy": float(accuracy_score(y_validation, predicted)),
        "macro_f1": float(class_report["macro avg"]["f1-score"]),
        "weighted_f1": float(class_report["weighted avg"]["f1-score"]),
        "classification_report": class_report,
        "confusion_matrix": {
            actual: {predicted_label: int(matrix[i, j]) for j, predicted_label in enumerate(CLASSES)}
            for i, actual in enumerate(CLASSES)
        },
        "train_document_ids": train_ids,
        "validation_document_ids": validation_ids,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_set": args.feature_set,
            "feature_names": list(selected_features),
            "classes": list(CLASSES),
            "embedding_model": (
                args.embedding_model if args.feature_set == "geometry_text_semantic" else None
            ),
        },
        args.model_output,
    )
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Train documents: {len(train_ids)}")
    print(f"Validation documents: {len(validation_ids)}")
    print(f"Train examples: {len(train)} (before sampling: {len(train_all)})")
    print(f"Validation examples: {len(validation)}")
    print(f"Train classes: {counts(train)}")
    print(f"Validation classes: {counts(validation)}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Macro F1: {report['macro_f1']:.3f}")
    for label in CLASSES:
        values = class_report[label]
        print(
            f"{label}: precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} f1={values['f1-score']:.3f} "
            f"support={int(values['support'])}"
        )
    print(f"Model: {args.model_output}")
    print(f"Report: {args.report_output}")


if __name__ == "__main__":
    main()
