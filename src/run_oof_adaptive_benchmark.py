from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from relation_features import feature_names
from relation_model import TwoStageExtraTreesClassifier
from run_structure_benchmark import mean, report_row
from train_geometric_classifier import (
    arrays,
    attach_model_features,
    downsample_negatives,
    load_examples,
)
from tune_russian_domain import group_by_document, repeat_examples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("SAME_CELL", "RIGHT", "BELOW", "NO_RELATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run honest out-of-fold Russian adaptive-reconstruction benchmark"
    )
    parser.add_argument("--source-input", required=True, type=Path)
    parser.add_argument("--russian-input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method-name", default="v7_adaptive_vertical_merge_oof")
    parser.add_argument("--russian-weight", type=int, default=3)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--same-cell-threshold", type=float, default=0.45)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def make_model(args: argparse.Namespace, fold_index: int) -> TwoStageExtraTreesClassifier:
    return TwoStageExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state + fold_index,
        n_jobs=args.n_jobs,
        same_cell_threshold=args.same_cell_threshold,
    )


def relation_document(
    document_id: str,
    examples: list[dict[str, Any]],
    model: TwoStageExtraTreesClassifier,
    selected_features: tuple[str, ...],
) -> dict[str, Any]:
    x, _ = arrays(examples, selected_features)
    probabilities = model.predict_proba(x)
    predictions = model.predict(x)
    counts = Counter(str(value) for value in predictions)
    relations = []
    for index, (example, prediction) in enumerate(zip(examples, predictions)):
        relations.append(
            {
                "source_fragment_id": example["source_fragment_id"],
                "target_fragment_id": example["target_fragment_id"],
                "source_text": example.get("source_text", ""),
                "target_text": example.get("target_text", ""),
                "prediction": str(prediction),
                "confidence": float(np.max(probabilities[index])),
                "probabilities": {
                    label: float(probabilities[index, class_index])
                    for class_index, label in enumerate(CLASSES)
                },
                "features": {
                    name: float(value)
                    for name, value in example["features"].items()
                },
                "model_features": {
                    name: float(example["model_features"][name])
                    for name in selected_features
                },
            }
        )
    return {
        "schema_version": "1.0",
        "task": "ocr_fragment_relation_classification",
        "method": "v5_two_stage_extra_trees_out_of_fold",
        "source": document_id,
        "relation_classes": list(CLASSES),
        "feature_names": list(selected_features),
        "feature_set": "geometry_text",
        "candidate_pair_count": len(relations),
        "prediction_counts": {label: counts[label] for label in CLASSES},
        "relations": relations,
    }


def main() -> None:
    args = parse_args()
    if args.russian_weight < 1:
        raise ValueError("--russian-weight must be at least 1")
    if args.negative_ratio <= 0:
        raise ValueError("--negative-ratio must be positive")
    selected_features = feature_names("geometry_text")
    source_examples = load_examples(resolve(args.source_input))
    russian_examples = load_examples(resolve(args.russian_input))
    attach_model_features(source_examples + russian_examples, "geometry_text", None)
    by_document = group_by_document(russian_examples)
    manifest = json.loads(resolve(args.manifest).read_text(encoding="utf-8"))
    manifest_by_id = {item["id"]: item for item in manifest["documents"]}
    if set(by_document) != set(manifest_by_id):
        raise ValueError(
            "Russian JSONL and manifest document ids differ: "
            f"JSONL-only={sorted(set(by_document) - set(manifest_by_id))}, "
            f"manifest-only={sorted(set(manifest_by_id) - set(by_document))}"
        )

    output_dir = resolve(args.output_dir)
    directories = {
        name: output_dir / name
        for name in ("relations", "structures", "evaluation", "grid", "previews")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold_index, validation_id in enumerate(by_document):
        validation = by_document[validation_id]
        russian_train = [
            example
            for document_id, examples in by_document.items()
            if document_id != validation_id
            for example in examples
        ]
        before_sampling = source_examples + repeat_examples(
            russian_train, args.russian_weight
        )
        train = downsample_negatives(
            before_sampling, args.negative_ratio, args.random_state + fold_index
        )
        x_train, y_train = arrays(train, selected_features)
        model = make_model(args, fold_index)
        model.fit(x_train, y_train)

        document = manifest_by_id[validation_id]
        relations_path = directories["relations"] / f"{validation_id}.json"
        structure_path = directories["structures"] / f"{validation_id}.json"
        relation_metrics_path = (
            directories["evaluation"] / f"{validation_id}_relations.json"
        )
        structure_metrics_path = (
            directories["evaluation"] / f"{validation_id}_structure.json"
        )
        relations_path.write_text(
            json.dumps(
                relation_document(
                    validation_id, validation, model, selected_features
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        image = resolve(
            Path(document.get("image", f"data/images/{validation_id}.png"))
        )
        ocr = resolve(Path(document["ocr"]))
        truth = resolve(Path(document["ground_truth"]))
        run(
            [
                sys.executable,
                "src/evaluate_relations.py",
                "--prediction",
                str(relations_path),
                "--ground-truth",
                str(truth),
                "--output",
                str(relation_metrics_path),
            ]
        )
        run(
            [
                sys.executable,
                "src/auto_reconstruct_table.py",
                "--image",
                str(image),
                "--ocr",
                str(ocr),
                "--relations",
                str(relations_path),
                "--output",
                str(structure_path),
                "--diagnostics-output",
                str(directories["evaluation"] / f"{validation_id}_selection.json"),
                "--grid-output",
                str(directories["grid"] / f"{validation_id}.json"),
                "--preview-output",
                str(directories["previews"] / f"{validation_id}.png"),
                "--min-confidence",
                str(args.min_confidence),
                "--same-cell-threshold",
                str(args.same_cell_threshold),
            ]
        )
        run(
            [
                sys.executable,
                "src/evaluate_structure.py",
                "--prediction",
                str(structure_path),
                "--ground-truth",
                str(truth),
                "--output",
                str(structure_metrics_path),
            ]
        )
        relation_report = json.loads(relation_metrics_path.read_text(encoding="utf-8"))
        structure_report = json.loads(
            structure_metrics_path.read_text(encoding="utf-8")
        )
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        row = report_row(document, relation_report, structure_report)
        selection = structure["automatic_mode_selection"]
        row["selected_mode"] = selection["selected_mode"]
        row["selection_reason"] = selection["reason"]
        rows.append(row)
        print(
            f"Completed fold {fold_index + 1}/{len(by_document)}: {validation_id}"
        )

    mean_keys = [
        "relation_accuracy",
        "relation_macro_f1",
        "relation_same_cell_f1",
        "exact_cell_f1",
        "same_cell_pair_f1",
        "coordinate_accuracy",
        "span_accuracy",
        "spanning_cell_f1",
    ]
    summary = {
        "schema_version": "1.0",
        "benchmark": manifest["name"],
        "method": args.method_name,
        "evaluation_mode": "leave_one_russian_document_out",
        "feature_set": "geometry_text",
        "parameters": {
            "russian_weight": args.russian_weight,
            "negative_ratio": args.negative_ratio,
            "same_cell_threshold": args.same_cell_threshold,
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
        },
        "document_count": len(rows),
        "macro_average": {key: mean(rows, key) for key in mean_keys},
        "documents": rows,
    }
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    for key, value in summary["macro_average"].items():
        print(f"Mean {key}: {value:.3f}")


if __name__ == "__main__":
    main()
