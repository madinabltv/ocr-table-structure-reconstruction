from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

from relation_features import feature_names
from relation_model import TwoStageExtraTreesClassifier
from train_geometric_classifier import (
    DEFAULT_EMBEDDING_MODEL,
    arrays,
    attach_model_features,
    build_embedding_lookup,
    downsample_negatives,
    load_examples,
)


CLASSES = ("SAME_CELL", "RIGHT", "BELOW", "NO_RELATION")


@dataclass(frozen=True)
class Configuration:
    russian_weight: int
    negative_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune V5 on Russian OCR tables with leave-one-document-out CV"
    )
    parser.add_argument(
        "--source-input",
        required=True,
        type=Path,
        help="SciTSR JSONL used in every training fold",
    )
    parser.add_argument(
        "--russian-input",
        required=True,
        type=Path,
        help="Russian OCR JSONL; every document_id becomes one validation fold",
    )
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument(
        "--feature-set",
        choices=("geometry", "geometry_text", "geometry_text_semantic"),
        default="geometry_text",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--russian-weights",
        nargs="+",
        type=int,
        default=[3, 6, 10],
        help="Candidate integer repetition weights (default: 3 6 10)",
    )
    parser.add_argument(
        "--negative-ratios",
        nargs="+",
        type=float,
        default=[1.5, 2.0, 3.0],
        help="Candidate maximum NO_RELATION/positive ratios",
    )
    parser.add_argument(
        "--same-cell-thresholds",
        nargs="+",
        type=float,
        default=[0.20, 0.30, 0.40, 0.45],
        help="Candidate SAME_CELL probability thresholds",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(weight < 1 for weight in args.russian_weights):
        raise ValueError("all --russian-weights must be at least 1")
    if any(ratio <= 0 for ratio in args.negative_ratios):
        raise ValueError("all --negative-ratios must be positive")
    if any(not 0.0 < threshold < 1.0 for threshold in args.same_cell_thresholds):
        raise ValueError("all --same-cell-thresholds must be between 0 and 1")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be positive")
    if args.min_samples_leaf < 1:
        raise ValueError("--min-samples-leaf must be positive")


def group_by_document(
    examples: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[str(example["document_id"])].append(example)
    return dict(sorted(grouped.items()))


def repeat_examples(
    examples: Iterable[dict[str, Any]], weight: int
) -> list[dict[str, Any]]:
    return [example for example in examples for _ in range(weight)]


def predictions_at_threshold(
    probabilities: np.ndarray, threshold: float
) -> np.ndarray:
    """Apply a SAME_CELL threshold to TwoStageExtraTrees probabilities."""
    p_same = probabilities[:, 0]
    other_indices = np.argmax(probabilities[:, 1:], axis=1) + 1
    indices = np.where(p_same >= threshold, 0, other_indices)
    return np.asarray(CLASSES)[indices]


def empty_matrix() -> np.ndarray:
    return np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)


def matrix_to_dict(matrix: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        actual: {
            predicted: int(matrix[row_index, column_index])
            for column_index, predicted in enumerate(CLASSES)
        }
        for row_index, actual in enumerate(CLASSES)
    }


def metrics_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    supported_f1_values = []
    for index, label in enumerate(CLASSES):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        if support:
            supported_f1_values.append(f1)
    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "supported_macro_f1": float(np.mean(supported_f1_values)),
        "same_cell_f1": float(per_class["SAME_CELL"]["f1"]),
        "per_class": per_class,
        "example_count": total,
        "confusion_matrix": matrix_to_dict(matrix),
    }


def class_counts(examples: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(example["label"] for example in examples)
    return {label: counter[label] for label in CLASSES}


def make_model(args: argparse.Namespace, threshold: float) -> TwoStageExtraTreesClassifier:
    return TwoStageExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        same_cell_threshold=threshold,
    )


def config_seed(base_seed: int, configuration_index: int, fold_index: int) -> int:
    return base_seed + configuration_index * 10_000 + fold_index


def tune(
    source_examples: list[dict[str, Any]],
    russian_by_document: dict[str, list[dict[str, Any]]],
    selected_features: tuple[str, ...],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    document_ids = list(russian_by_document)
    configurations = [
        Configuration(weight, ratio)
        for weight in args.russian_weights
        for ratio in args.negative_ratios
    ]
    results = []
    for configuration_index, configuration in enumerate(configurations):
        matrices = {threshold: empty_matrix() for threshold in args.same_cell_thresholds}
        fold_records: dict[float, list[dict[str, Any]]] = {
            threshold: [] for threshold in args.same_cell_thresholds
        }
        print(
            f"\nConfiguration {configuration_index + 1}/{len(configurations)}: "
            f"Russian weight={configuration.russian_weight}, "
            f"negative ratio={configuration.negative_ratio:g}"
        )
        for fold_index, validation_id in enumerate(document_ids):
            russian_train = [
                example
                for document_id, examples in russian_by_document.items()
                if document_id != validation_id
                for example in examples
            ]
            validation = russian_by_document[validation_id]
            weighted_russian = repeat_examples(
                russian_train, configuration.russian_weight
            )
            train_before_sampling = source_examples + weighted_russian
            seed = config_seed(args.random_state, configuration_index, fold_index)
            train = downsample_negatives(
                train_before_sampling, configuration.negative_ratio, seed
            )
            missing = set(CLASSES) - {example["label"] for example in train}
            if missing:
                raise ValueError(
                    f"fold {validation_id!r} is missing training classes: {sorted(missing)}"
                )
            x_train, y_train = arrays(train, selected_features)
            x_validation, y_validation = arrays(validation, selected_features)
            # The threshold does not affect fitting, so one model evaluates all
            # threshold candidates in this fold.
            model = make_model(args, args.same_cell_thresholds[0])
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)
            for threshold in args.same_cell_thresholds:
                predicted = predictions_at_threshold(probabilities, threshold)
                fold_matrix = confusion_matrix(
                    y_validation, predicted, labels=CLASSES
                ).astype(np.int64)
                matrices[threshold] += fold_matrix
                fold_metrics = metrics_from_matrix(fold_matrix)
                fold_records[threshold].append(
                    {
                        "validation_document_id": validation_id,
                        "train_example_count_before_sampling": len(train_before_sampling),
                        "train_example_count": len(train),
                        "validation_example_count": len(validation),
                        "metrics": fold_metrics,
                    }
                )
            print(
                f"  fold {fold_index + 1}/{len(document_ids)}: {validation_id} "
                f"({len(validation)} validation pairs)"
            )
        for threshold in args.same_cell_thresholds:
            pooled = metrics_from_matrix(matrices[threshold])
            mean_document_macro_f1 = float(
                np.mean(
                    [record["metrics"]["macro_f1"] for record in fold_records[threshold]]
                )
            )
            result = {
                "russian_weight": configuration.russian_weight,
                "negative_ratio": configuration.negative_ratio,
                "same_cell_threshold": threshold,
                "pooled": pooled,
                "mean_document_macro_f1": mean_document_macro_f1,
                "folds": fold_records[threshold],
            }
            results.append(result)
            print(
                f"    threshold={threshold:.2f}: "
                f"pooled Macro F1={pooled['macro_f1']:.3f}, "
                f"SAME_CELL F1={pooled['same_cell_f1']:.3f}"
            )
    return results


def selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    """Prefer pooled relation quality, then grouping quality and stability."""
    pooled = result["pooled"]
    return (
        pooled["macro_f1"],
        pooled["same_cell_f1"],
        result["mean_document_macro_f1"],
        pooled["accuracy"],
    )


def fit_final_model(
    source_examples: list[dict[str, Any]],
    russian_examples: list[dict[str, Any]],
    selected_features: tuple[str, ...],
    best: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[TwoStageExtraTreesClassifier, list[dict[str, Any]], int]:
    weighted_russian = repeat_examples(russian_examples, best["russian_weight"])
    before_sampling = source_examples + weighted_russian
    train = downsample_negatives(
        before_sampling, best["negative_ratio"], args.random_state
    )
    x_train, y_train = arrays(train, selected_features)
    model = make_model(args, best["same_cell_threshold"])
    model.fit(x_train, y_train)
    return model, train, len(before_sampling)


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in sorted(results, key=selection_key, reverse=True):
        pooled = result["pooled"]
        rows.append(
            {
                "russian_weight": result["russian_weight"],
                "negative_ratio": result["negative_ratio"],
                "same_cell_threshold": result["same_cell_threshold"],
                "pooled_accuracy": pooled["accuracy"],
                "pooled_macro_f1": pooled["macro_f1"],
                "pooled_supported_macro_f1": pooled["supported_macro_f1"],
                "pooled_same_cell_f1": pooled["same_cell_f1"],
                "mean_document_macro_f1": result["mean_document_macro_f1"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    selected_features = feature_names(args.feature_set)
    print("Loading source-domain examples...")
    source_examples = load_examples(args.source_input)
    print("Loading Russian OCR development examples...")
    russian_examples = load_examples(args.russian_input)
    russian_by_document = group_by_document(russian_examples)
    if len(russian_by_document) < 2:
        raise ValueError("Russian input must contain at least two document_id values")
    overlap = {str(item["document_id"]) for item in source_examples} & set(
        russian_by_document
    )
    if overlap:
        raise ValueError(f"source and Russian document ids overlap: {sorted(overlap)}")

    all_examples = source_examples + russian_examples
    embedding_lookup = None
    if args.feature_set == "geometry_text_semantic":
        embedding_lookup = build_embedding_lookup(
            all_examples,
            args.embedding_model,
            args.embedding_cache,
            args.embedding_batch_size,
        )
    attach_model_features(all_examples, args.feature_set, embedding_lookup)

    print(f"Source examples: {len(source_examples)}")
    print(f"Russian examples: {len(russian_examples)}")
    print(f"Russian folds: {len(russian_by_document)}")
    print(f"Russian documents: {list(russian_by_document)}")
    results = tune(
        source_examples, russian_by_document, selected_features, args
    )
    best = max(results, key=selection_key)
    print("\nFitting final model with all Russian development tables...")
    model, final_train, before_sampling = fit_final_model(
        source_examples, russian_examples, selected_features, best, args
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_set": args.feature_set,
            "feature_names": list(selected_features),
            "classes": list(CLASSES),
            "embedding_model": (
                args.embedding_model
                if args.feature_set == "geometry_text_semantic"
                else None
            ),
            "tuning": {
                "protocol": "leave_one_russian_document_out",
                "best_parameters": {
                    "russian_weight": best["russian_weight"],
                    "negative_ratio": best["negative_ratio"],
                    "same_cell_threshold": best["same_cell_threshold"],
                },
            },
        },
        args.model_output,
    )
    report = {
        "schema_version": "1.0",
        "method": "v5_two_stage_extra_trees_russian_domain_tuning",
        "protocol": "leave_one_russian_document_out",
        "selection_metric": [
            "pooled_macro_f1",
            "pooled_same_cell_f1",
            "mean_document_macro_f1",
            "pooled_accuracy",
        ],
        "source_input": str(args.source_input),
        "russian_input": str(args.russian_input),
        "feature_set": args.feature_set,
        "feature_names": list(selected_features),
        "classes": list(CLASSES),
        "parameters": {
            "russian_weights": args.russian_weights,
            "negative_ratios": args.negative_ratios,
            "same_cell_thresholds": args.same_cell_thresholds,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs,
            "embedding_model": (
                args.embedding_model
                if args.feature_set == "geometry_text_semantic"
                else None
            ),
        },
        "source_example_count": len(source_examples),
        "source_class_counts": class_counts(source_examples),
        "russian_example_count": len(russian_examples),
        "russian_class_counts": class_counts(russian_examples),
        "russian_document_ids": list(russian_by_document),
        "best_parameters": {
            "russian_weight": best["russian_weight"],
            "negative_ratio": best["negative_ratio"],
            "same_cell_threshold": best["same_cell_threshold"],
        },
        "best_cross_validation": {
            "pooled": best["pooled"],
            "mean_document_macro_f1": best["mean_document_macro_f1"],
            "folds": best["folds"],
        },
        "final_train_example_count_before_sampling": before_sampling,
        "final_train_example_count": len(final_train),
        "final_train_class_counts": class_counts(final_train),
        "all_results": sorted(results, key=selection_key, reverse=True),
    }
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.csv_output, results)

    print("\nBest parameters:")
    print(f"  Russian weight: {best['russian_weight']}")
    print(f"  Negative ratio: {best['negative_ratio']:g}")
    print(f"  SAME_CELL threshold: {best['same_cell_threshold']:.2f}")
    print(f"  Pooled accuracy: {best['pooled']['accuracy']:.3f}")
    print(f"  Pooled Macro F1: {best['pooled']['macro_f1']:.3f}")
    print(f"  Pooled SAME_CELL F1: {best['pooled']['same_cell_f1']:.3f}")
    print(f"Model: {args.model_output}")
    print(f"Report: {args.report_output}")
    print(f"CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
