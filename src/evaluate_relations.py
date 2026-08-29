from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from label_relations_from_cells import cell_index, ground_truth_relation


CLASSES = ("SAME_CELL", "RIGHT", "BELOW", "NO_RELATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate relation predictions")
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metrics(confusion: dict[str, dict[str, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    f1_values = []
    total_correct = 0
    total = 0
    for label in CLASSES:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in CLASSES if actual != label)
        fn = sum(confusion[label][predicted] for predicted in CLASSES if predicted != label)
        support = sum(confusion[label].values())
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        total_correct += tp
        total += support
        result[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
    result["accuracy"] = round(safe_divide(total_correct, total), 6)
    result["macro_f1"] = round(sum(f1_values) / len(f1_values), 6)
    result["evaluated_pairs"] = total
    return result


def main() -> None:
    args = parse_args()
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    cells, fragment_to_cell = cell_index(ground_truth)
    ignored = set(ground_truth.get("ignored_fragment_ids", []))

    confusion = {
        actual: {predicted: 0 for predicted in CLASSES} for actual in CLASSES
    }
    errors = []
    skipped = []
    for item in prediction["relations"]:
        source_id = item["source_fragment_id"]
        target_id = item["target_fragment_id"]
        if source_id in ignored or target_id in ignored:
            skipped.append({"source_fragment_id": source_id, "target_fragment_id": target_id})
            continue
        if source_id not in fragment_to_cell or target_id not in fragment_to_cell:
            raise ValueError(
                f"pair contains an unannotated fragment: {source_id!r}, {target_id!r}"
            )
        source_cell = cells[fragment_to_cell[source_id]]
        target_cell = cells[fragment_to_cell[target_id]]
        actual = ground_truth_relation(source_cell, target_cell)
        predicted = item["prediction"]
        if predicted not in CLASSES:
            raise ValueError(f"unknown predicted class: {predicted!r}")
        confusion[actual][predicted] += 1
        if actual != predicted:
            errors.append(
                {
                    "source_fragment_id": source_id,
                    "target_fragment_id": target_id,
                    "source_text": item.get("source_text", ""),
                    "target_text": item.get("target_text", ""),
                    "actual": actual,
                    "predicted": predicted,
                }
            )

    report = {
        "schema_version": "1.0",
        "task": "relation_classification_evaluation",
        "prediction": args.prediction.name,
        "ground_truth": args.ground_truth.name,
        "classes": list(CLASSES),
        "confusion_matrix": confusion,
        "metrics": metrics(confusion),
        "error_count": len(errors),
        "errors": errors,
        "skipped_pairs": skipped,
        "evaluation_scope": "candidate pairs present in the prediction file",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluated pairs: {report['metrics']['evaluated_pairs']}")
    print(f"Accuracy: {report['metrics']['accuracy']:.3f}")
    print(f"Macro F1: {report['metrics']['macro_f1']:.3f}")
    for label in CLASSES:
        values = report["metrics"][label]
        print(
            f"{label}: precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} f1={values['f1']:.3f} "
            f"support={values['support']}"
        )
    print(f"Errors: {len(errors)}")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
