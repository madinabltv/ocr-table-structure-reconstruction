from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = [
    ("rules", None),
    ("geometry_only_v2", Path("outputs/models/geometric_logreg_v2.joblib")),
    ("geometry_text_v5", Path("outputs/models/geometric_text_v5_tuned.joblib")),
    ("ordered_pair_v6", Path("outputs/models/ordered_pair_minilm_v6.joblib")),
]
METRICS = [
    "relation_accuracy",
    "relation_macro_f1",
    "relation_same_cell_f1",
    "exact_cell_f1",
    "same_cell_pair_f1",
    "coordinate_accuracy",
    "span_accuracy",
    "spanning_cell_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and compare all final table-reconstruction methods"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("experiments/generalization_v8.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark/generalization_v8/final_comparison_v12"),
    )
    parser.add_argument(
        "--method",
        action="append",
        metavar="LABEL=MODEL|RULES",
        help=(
            "Override default methods; repeat for every method. Example: "
            "--method rules=RULES --method ordered=outputs/models/model.joblib"
        ),
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--max-distance-ratio", type=float, default=0.35)
    parser.add_argument(
        "--force", action="store_true", help="Re-run methods with an existing summary"
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_methods(values: list[str] | None) -> list[tuple[str, Path | None]]:
    if not values:
        return DEFAULT_METHODS.copy()
    methods: list[tuple[str, Path | None]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=MODEL or LABEL=RULES, got: {value}")
        label, raw_model = value.split("=", 1)
        label = label.strip()
        if not label or label in labels:
            raise ValueError(f"Empty or duplicate method label: {label!r}")
        labels.add(label)
        model = None if raw_model.strip().upper() == "RULES" else Path(raw_model.strip())
        methods.append((label, model))
    return methods


def evaluable_span_documents(manifest: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for document in manifest["documents"]:
        truth = json.loads(resolve(document["ground_truth"]).read_text(encoding="utf-8"))
        cells = truth.get("cells", truth.get("logical_cells", []))
        if any(
            cell.get("fragment_ids")
            and (
                int(cell.get("row_end", cell.get("row_start", 0)))
                > int(cell.get("row_start", 0))
                or int(cell.get("column_end", cell.get("column_start", 0)))
                > int(cell.get("column_start", 0))
            )
            for cell in cells
        ):
            result.add(document["id"])
    return result


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def comparison_row(
    label: str,
    model: Path | None,
    report: dict[str, Any],
    span_documents: set[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": label,
        "model": str(model) if model else "geometric_rules",
    }
    row.update({metric: report["macro_average"].get(metric, 0.0) for metric in METRICS})
    conditional = [
        float(document.get("spanning_cell_f1", 0.0))
        for document in report.get("documents", [])
        if document.get("document_id") in span_documents
    ]
    row["spanning_cell_f1_on_span_documents"] = mean(conditional)
    row["elapsed_seconds"] = round(elapsed_seconds, 3)
    return row


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("method", "Method"),
        ("relation_macro_f1", "Relation Macro F1"),
        ("relation_same_cell_f1", "SAME_CELL F1"),
        ("exact_cell_f1", "Exact Cell F1"),
        ("same_cell_pair_f1", "Cell-pair F1"),
        ("coordinate_accuracy", "Coordinate acc."),
        ("span_accuracy", "Span acc."),
        ("spanning_cell_f1_on_span_documents", "Spanning F1*"),
        ("elapsed_seconds", "Time, s"),
    ]
    lines = [
        "# Final method comparison (generalization_v8, reconstruction v12)",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "*Spanning F1 is averaged only over documents containing evaluable "
            "non-empty spanning cells. All methods use the same manifest, candidate "
            "parameters, automatic mode selection, and v12 reconstruction.*",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    manifest_path = resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [(label, resolve(model)) for label, model in methods if model and not resolve(model).is_file()]
    if missing:
        details = "\n".join(f"  {label}: {path}" for label, path in missing)
        raise FileNotFoundError(
            "Required model files are missing:\n"
            + details
            + "\nCopy/train them first, or provide only available methods with --method."
        )

    span_documents = evaluable_span_documents(manifest)
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for index, (label, model) in enumerate(methods, start=1):
        method_dir = output_dir / label
        summary_path = method_dir / "summary.json"
        print(f"\nMethod {index}/{len(methods)}: {label}")
        started = time.perf_counter()
        if args.force or not summary_path.is_file():
            command = [
                sys.executable,
                "src/run_auto_benchmark.py",
                "--manifest",
                str(manifest_path),
                "--method-name",
                label,
                "--output-dir",
                str(method_dir),
                "--min-confidence",
                str(args.min_confidence),
                "--max-neighbors",
                str(args.max_neighbors),
                "--max-distance-ratio",
                str(args.max_distance_ratio),
            ]
            if model:
                command.extend(["--model", str(resolve(model))])
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        else:
            print(f"Reusing: {summary_path}")
        elapsed = time.perf_counter() - started
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        reports[label] = report
        rows.append(comparison_row(label, model, report, span_documents, elapsed))

    rows.sort(key=lambda row: row["exact_cell_f1"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["exact_cell_rank"] = rank
    result = {
        "schema_version": "1.0",
        "benchmark": manifest.get("name", manifest_path.stem),
        "reconstruction_version": "v12",
        "span_documents": sorted(span_documents),
        "methods": rows,
        "best_exact_cell_method": rows[0]["method"],
    }

    json_path = output_dir / "final_comparison.json"
    csv_path = output_dir / "final_comparison.csv"
    markdown_path = output_dir / "final_comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown_table(rows), encoding="utf-8")

    print("\nFinal comparison")
    for row in rows:
        print(
            f"  {row['exact_cell_rank']}. {row['method']}: "
            f"Relation Macro F1={row['relation_macro_f1']:.3f}, "
            f"Exact Cell F1={row['exact_cell_f1']:.3f}, "
            f"Spanning F1*={row['spanning_cell_f1_on_span_documents']:.3f}"
        )
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
