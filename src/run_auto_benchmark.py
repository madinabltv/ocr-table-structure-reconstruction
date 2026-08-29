from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from run_structure_benchmark import mean, report_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic table reconstruction benchmark")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--max-distance-ratio", type=float, default=0.35)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    manifest = json.loads(resolve(args.manifest).read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir)
    directories = {
        name: output_dir / name
        for name in ("relations", "structures", "evaluation", "grid", "previews")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for document in manifest["documents"]:
        name = document["id"]
        image = resolve(document["image"])
        ocr = resolve(document["ocr"])
        truth = resolve(document["ground_truth"])
        relations = directories["relations"] / f"{name}.json"
        structure = directories["structures"] / f"{name}.json"
        relation_metrics = directories["evaluation"] / f"{name}_relations.json"
        structure_metrics = directories["evaluation"] / f"{name}_structure.json"

        if args.model:
            prediction = [
                sys.executable, "src/predict_relations.py", "--input", str(ocr),
                "--model", str(resolve(args.model)), "--output", str(relations),
            ]
        else:
            prediction = [
                sys.executable, "src/build_relation_baseline.py", "--input", str(ocr),
                "--output", str(relations),
            ]
        prediction.extend([
            "--min-confidence", str(args.min_confidence),
            "--max-neighbors", str(args.max_neighbors),
            "--max-distance-ratio", str(args.max_distance_ratio),
        ])
        run(prediction)
        run([
            sys.executable, "src/evaluate_relations.py", "--prediction", str(relations),
            "--ground-truth", str(truth), "--output", str(relation_metrics),
        ])
        run([
            sys.executable, "src/auto_reconstruct_table.py",
            "--image", str(image), "--ocr", str(ocr), "--relations", str(relations),
            "--output", str(structure),
            "--diagnostics-output", str(directories["evaluation"] / f"{name}_selection.json"),
            "--grid-output", str(directories["grid"] / f"{name}.json"),
            "--preview-output", str(directories["previews"] / f"{name}.png"),
            "--min-confidence", str(args.min_confidence),
        ])
        run([
            sys.executable, "src/evaluate_structure.py", "--prediction", str(structure),
            "--ground-truth", str(truth), "--output", str(structure_metrics),
        ])
        relation_report = json.loads(relation_metrics.read_text(encoding="utf-8"))
        structure_report = json.loads(structure_metrics.read_text(encoding="utf-8"))
        structure_data = json.loads(structure.read_text(encoding="utf-8"))
        row = report_row(document, relation_report, structure_report)
        selection = structure_data["automatic_mode_selection"]
        row["selected_mode"] = selection["selected_mode"]
        row["selection_reason"] = selection["reason"]
        rows.append(row)

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
        "evaluation_mode": "heldout_automatic_mode_selection",
        "row_banding": "aligned",
        "document_count": len(rows),
        "macro_average": {key: mean(rows, key) for key in mean_keys},
        "documents": rows,
    }
    json_output = output_dir / "summary.json"
    csv_output = output_dir / "summary.csv"
    json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary JSON: {json_output}")
    print(f"Summary CSV: {csv_output}")
    for key, value in summary["macro_average"].items():
        print(f"Mean {key}: {value:.3f}")


if __name__ == "__main__":
    main()
