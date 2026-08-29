from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELATION_FIELDS = [
    "relation_macro_f1",
    "relation_same_cell_f1",
    "relation_right_f1",
    "relation_below_f1",
]
STRUCTURE_FIELDS = [
    "exact_cell_f1",
    "same_cell_pair_f1",
    "coordinate_accuracy",
    "span_accuracy",
    "spanning_cell_f1_on_span_documents",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final experiment report")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("outputs/benchmark/generalization_v8/final_comparison_v12"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/final_experiment_v14"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def tied_ranks(values: list[float], tolerance: float = 1e-12) -> list[int]:
    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    result = [0] * len(values)
    previous_value: float | None = None
    previous_rank = 0
    for position, (original_index, value) in enumerate(ordered, start=1):
        if previous_value is None or abs(value - previous_value) > tolerance:
            previous_rank = position
            previous_value = value
        result[original_index] = previous_rank
    return result


def load_reports(comparison_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    comparison = json.loads(
        (comparison_dir / "final_comparison.json").read_text(encoding="utf-8")
    )
    reports: dict[str, dict[str, Any]] = {}
    missing = []
    for method in comparison["methods"]:
        label = method["method"]
        path = comparison_dir / label / "summary.json"
        if path.is_file():
            reports[label] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing per-method summary files:\n" + "\n".join(str(path) for path in missing)
        )
    return comparison, reports


def build_rows(
    comparison: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    span_documents = set(comparison.get("span_documents", []))
    method_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    for original in comparison["methods"]:
        label = original["method"]
        report = reports[label]
        by_document = report.get("documents", [])
        row = dict(original)
        row["relation_right_f1"] = mean(
            [float(item.get("relation_right_f1", 0.0)) for item in by_document]
        )
        row["relation_below_f1"] = mean(
            [float(item.get("relation_below_f1", 0.0)) for item in by_document]
        )
        row["spanning_cell_f1_on_span_documents"] = mean(
            [
                float(item.get("spanning_cell_f1", 0.0))
                for item in by_document
                if item.get("document_id") in span_documents
            ]
        )
        method_rows.append(row)
        for item in by_document:
            document_rows.append(
                {
                    "method": label,
                    "document_id": item["document_id"],
                    "difficulty": item.get("difficulty", ""),
                    "selected_mode": item.get("selected_mode", ""),
                    "relation_macro_f1": item.get("relation_macro_f1", 0.0),
                    "relation_same_cell_f1": item.get("relation_same_cell_f1", 0.0),
                    "relation_right_f1": item.get("relation_right_f1", 0.0),
                    "relation_below_f1": item.get("relation_below_f1", 0.0),
                    "exact_cell_f1": item.get("exact_cell_f1", 0.0),
                    "same_cell_pair_f1": item.get("same_cell_pair_f1", 0.0),
                    "coordinate_accuracy": item.get("coordinate_accuracy", 0.0),
                    "span_accuracy": item.get("span_accuracy", 0.0),
                    "spanning_cell_f1": item.get("spanning_cell_f1", 0.0),
                }
            )
    ranks = tied_ranks([float(row["exact_cell_f1"]) for row in method_rows])
    for row, rank in zip(method_rows, ranks):
        row["exact_cell_rank"] = rank
    method_rows.sort(key=lambda row: (row["exact_cell_rank"], -row["relation_macro_f1"]))
    return method_rows, document_rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_bar(
    rows: list[dict[str, Any]],
    fields: list[str],
    labels: list[str],
    title: str,
    output: Path,
) -> None:
    methods = [row["method"] for row in rows]
    x = np.arange(len(methods))
    width = 0.8 / len(fields)
    fig, axis = plt.subplots(figsize=(11, 6.5))
    for index, (field, label) in enumerate(zip(fields, labels)):
        offset = (index - (len(fields) - 1) / 2) * width
        bars = axis.bar(
            x + offset,
            [float(row[field]) for row in rows],
            width,
            label=label,
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)
    axis.set_title(title)
    axis.set_ylabel("F1 / accuracy")
    axis.set_ylim(0.0, 1.12)
    axis.set_xticks(x, methods)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def speed_quality_plot(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    for row in rows:
        x = float(row["elapsed_seconds"])
        y = float(row["exact_cell_f1"])
        axis.scatter(x, y, s=90)
        axis.annotate(row["method"], (x, y), xytext=(6, 7), textcoords="offset points")
    axis.set_title("Скорость и качество восстановления структуры")
    axis.set_xlabel("Время обработки benchmark, с")
    axis.set_ylabel("Exact Cell F1")
    axis.set_ylim(min(float(row["exact_cell_f1"]) for row in rows) - 0.01, 1.0)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def document_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    documents = list(dict.fromkeys(row["document_id"] for row in rows))
    lookup = {(row["method"], row["document_id"]): row for row in rows}
    values = np.array(
        [[float(lookup[(method, document)]["exact_cell_f1"]) for document in documents] for method in methods]
    )
    fig, axis = plt.subplots(figsize=(11, 4.5))
    image = axis.imshow(values, vmin=0.85, vmax=1.0, cmap="YlGn")
    axis.set_xticks(range(len(documents)), documents, rotation=35, ha="right")
    axis.set_yticks(range(len(methods)), methods)
    axis.set_title("Exact Cell F1 по отдельным таблицам")
    for row_index in range(len(methods)):
        for column_index in range(len(documents)):
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, label="Exact Cell F1")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def conclusions(rows: list[dict[str, Any]]) -> str:
    best_exact = max(float(row["exact_cell_f1"]) for row in rows)
    tied = [row["method"] for row in rows if abs(float(row["exact_cell_f1"]) - best_exact) < 1e-12]
    best_relation = max(rows, key=lambda row: float(row["relation_macro_f1"]))
    best_same = max(rows, key=lambda row: float(row["relation_same_cell_f1"]))
    fastest = min(rows, key=lambda row: float(row["elapsed_seconds"]))
    geometry = next((row for row in rows if row["method"] == "geometry_only_v2"), None)
    ordered = next((row for row in rows if row["method"] == "ordered_pair_v6"), None)
    lines = [
        "# Выводы итогового эксперимента",
        "",
        f"- Лучшее итоговое качество структуры (`Exact Cell F1 = {best_exact:.3f}`) "
        f"совместно показали методы: {', '.join(tied)}.",
        f"- Лучший `Relation Macro F1 = {float(best_relation['relation_macro_f1']):.3f}` "
        f"получен методом `{best_relation['method']}`.",
        f"- Лучший `SAME_CELL F1 = {float(best_same['relation_same_cell_f1']):.3f}` "
        f"получен методом `{best_same['method']}`.",
        f"- Самым быстрым оказался `{fastest['method']}`: "
        f"{float(fastest['elapsed_seconds']):.1f} с на полный benchmark.",
    ]
    if geometry and ordered:
        lines.append(
            "- По сравнению с geometry-only последовательная модель повышает "
            f"Relation Macro F1 на {float(ordered['relation_macro_f1']) - float(geometry['relation_macro_f1']):.3f}, "
            f"а SAME_CELL F1 — на {float(ordered['relation_same_cell_f1']) - float(geometry['relation_same_cell_f1']):.3f}."
        )
    lines.extend(
        [
            "- Совпадение структурных результатов нескольких методов показывает, "
            "что автоматический выбор режима и геометрический постпроцессор v12 "
            "компенсируют часть ошибок классификации отношений.",
            "- Поэтому итоговая система интерпретируется как гибридная: модель "
            "отношений особенно полезна для SAME_CELL, а линии и геометрия обеспечивают "
            "устойчивое восстановление координат и объединённых ячеек.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    comparison_dir = resolve(args.comparison_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison, reports = load_reports(comparison_dir)
    method_rows, document_rows = build_rows(comparison, reports)

    summary = {
        "schema_version": "1.0",
        "source_comparison": str(comparison_dir),
        "ranking_rule": "competition ranking with ties",
        "methods": method_rows,
    }
    (output_dir / "method_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_csv(output_dir / "method_summary.csv", method_rows)
    save_csv(output_dir / "per_document_metrics.csv", document_rows)
    (output_dir / "experiment_conclusions.md").write_text(
        conclusions(method_rows), encoding="utf-8"
    )

    grouped_bar(
        method_rows,
        RELATION_FIELDS,
        ["Macro F1", "SAME_CELL", "RIGHT", "BELOW"],
        "Качество классификации структурных отношений",
        output_dir / "relation_f1_by_method.png",
    )
    grouped_bar(
        method_rows,
        ["exact_cell_f1", "same_cell_pair_f1", "spanning_cell_f1_on_span_documents"],
        ["Exact Cell F1", "Cell-pair F1", "Spanning Cell F1*"],
        "Качество восстановления структуры таблиц",
        output_dir / "structure_f1_by_method.png",
    )
    speed_quality_plot(method_rows, output_dir / "speed_vs_quality.png")
    document_heatmap(document_rows, output_dir / "exact_cell_f1_by_document.png")

    print(f"Methods: {len(method_rows)}")
    print(f"Documents: {len({row['document_id'] for row in document_rows})}")
    print(f"Output directory: {output_dir}")
    for row in method_rows:
        print(
            f"rank={row['exact_cell_rank']} {row['method']}: "
            f"macro={row['relation_macro_f1']:.3f}, "
            f"RIGHT={row['relation_right_f1']:.3f}, "
            f"BELOW={row['relation_below_f1']:.3f}, "
            f"exact={row['exact_cell_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
