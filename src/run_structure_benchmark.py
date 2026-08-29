from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the table structure benchmark")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--model", type=Path, help="Omit to run the rule baseline")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--max-distance-ratio", type=float, default=0.35)
    parser.add_argument("--row-banding", choices=("kmeans", "aligned"), default="kmeans")
    parser.add_argument(
        "--use-expected-shape",
        action="store_true",
        help="Pass annotated column/header counts; report this as an oracle-shape run",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def report_row(
    document: dict[str, Any], relation_report: dict[str, Any], structure_report: dict[str, Any]
) -> dict[str, Any]:
    relation = relation_report["metrics"]
    return {
        "document_id": document["id"],
        "difficulty": document.get("difficulty", ""),
        "relation_accuracy": relation["accuracy"],
        "relation_macro_f1": relation["macro_f1"],
        "relation_same_cell_f1": relation["SAME_CELL"]["f1"],
        "relation_right_f1": relation["RIGHT"]["f1"],
        "relation_below_f1": relation["BELOW"]["f1"],
        "exact_cell_f1": structure_report["exact_cell_metrics"]["f1"],
        "same_cell_pair_f1": structure_report["same_cell_pair_metrics"]["f1"],
        "coordinate_accuracy": structure_report["coordinate_accuracy_on_exact_cells"],
        "span_accuracy": structure_report["span_accuracy_on_exact_cells"],
        "spanning_cell_f1": structure_report["spanning_cell_metrics"]["f1"],
        "predicted_rows": structure_report["predicted_grid"]["rows"],
        "expected_rows": structure_report["ground_truth_grid"]["rows"],
        "predicted_columns": structure_report["predicted_grid"]["columns"],
        "expected_columns": structure_report["ground_truth_grid"]["columns"],
        "overmerged_cells": structure_report["overmerged_cell_count"],
        "split_cells": structure_report["split_ground_truth_cell_count"],
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row[key]) for row in rows) / len(rows), 6) if rows else 0.0


def main() -> None:
    args = parse_args()
    manifest = json.loads(resolve(str(args.manifest)).read_text(encoding="utf-8"))
    output_dir = resolve(str(args.output_dir))
    relations_dir = output_dir / "relations"
    structures_dir = output_dir / "structures"
    evaluations_dir = output_dir / "evaluation"
    for path in (relations_dir, structures_dir, evaluations_dir):
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    for document in manifest["documents"]:
        document_id = document["id"]
        ocr = resolve(document["ocr"])
        ground_truth = resolve(document["ground_truth"])
        relations = relations_dir / f"{document_id}_{args.method_name}.json"
        structure = structures_dir / f"{document_id}_{args.method_name}.json"
        relation_metrics = evaluations_dir / f"{document_id}_{args.method_name}_relations.json"
        structure_metrics = evaluations_dir / f"{document_id}_{args.method_name}_structure.json"

        if args.model:
            command = [
                sys.executable, "src/predict_relations.py", "--input", str(ocr),
                "--model", str(resolve(str(args.model))), "--output", str(relations),
            ]
        else:
            command = [
                sys.executable, "src/build_relation_baseline.py", "--input", str(ocr),
                "--output", str(relations),
            ]
        command.extend(
            [
                "--min-confidence", str(args.min_confidence),
                "--max-neighbors", str(args.max_neighbors),
                "--max-distance-ratio", str(args.max_distance_ratio),
            ]
        )
        run(command)
        run(
            [
                sys.executable, "src/evaluate_relations.py", "--prediction", str(relations),
                "--ground-truth", str(ground_truth), "--output", str(relation_metrics),
            ]
        )

        reconstruction = [
            sys.executable, "src/reconstruct_table_structure_hybrid.py",
            "--ocr", str(ocr), "--relations", str(relations), "--output", str(structure),
            "--min-confidence", str(args.min_confidence),
            "--merge-header-lines", "--infer-missing-header-cells",
            "--row-banding", args.row_banding,
        ]
        if args.use_expected_shape:
            reconstruction.extend(
                [
                    "--expected-columns", str(document["expected_columns"]),
                    "--logical-header-rows", str(document["logical_header_rows"]),
                ]
            )
        run(reconstruction)
        run(
            [
                sys.executable, "src/evaluate_structure.py", "--prediction", str(structure),
                "--ground-truth", str(ground_truth), "--output", str(structure_metrics),
            ]
        )
        rows.append(
            report_row(
                document,
                json.loads(relation_metrics.read_text(encoding="utf-8")),
                json.loads(structure_metrics.read_text(encoding="utf-8")),
            )
        )

    mean_keys = [
        "relation_accuracy", "relation_macro_f1", "relation_same_cell_f1",
        "exact_cell_f1", "same_cell_pair_f1", "coordinate_accuracy",
        "span_accuracy", "spanning_cell_f1",
    ]
    summary = {
        "schema_version": "1.0",
        "benchmark": manifest["name"],
        "method": args.method_name,
        "model": str(args.model) if args.model else None,
        "evaluation_mode": "oracle_shape" if args.use_expected_shape else "fully_automatic",
        "row_banding": args.row_banding,
        "document_count": len(rows),
        "macro_average": {key: mean(rows, key) for key in mean_keys},
        "documents": rows,
    }
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["document_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    for key, value in summary["macro_average"].items():
        print(f"Mean {key}: {value:.3f}")


if __name__ == "__main__":
    main()
