from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


TABLES: dict[int, dict[str, Any]] = {
    1: {
        "expected_rows": 18,
        "expected_columns": 5,
        "logical_header_rows": 1,
        "difficulty": "web_grid_multiline_cells",
        "evaluation_group": "primary",
        "ocr_psm": 6,
    },
    2: {
        "expected_rows": 6,
        "expected_columns": 5,
        "logical_header_rows": 2,
        "difficulty": "two_level_header",
        "evaluation_group": "primary",
        "ocr_psm": 6,
    },
    4: {
        "expected_rows": 16,
        "expected_columns": 2,
        "logical_header_rows": 1,
        "difficulty": "dense_multiline_cells",
        "evaluation_group": "primary",
        "ocr_psm": 3,
    },
    9: {
        "expected_rows": 15,
        "expected_columns": 9,
        "logical_header_rows": 2,
        "difficulty": "wide_table_grouped_header",
        "evaluation_group": "primary",
        "ocr_psm": 6,
    },
    12: {
        "expected_rows": 17,
        "expected_columns": 3,
        "logical_header_rows": 1,
        "difficulty": "vertical_spans_and_ocr_omissions",
        "evaluation_group": "primary",
        "ocr_psm": 3,
    },
    13: {
        "expected_rows": 19,
        "expected_columns": 5,
        "logical_header_rows": 3,
        "difficulty": "degraded_scan_with_missing_fragments",
        "evaluation_group": "ocr_stress",
        "ocr_psm": 3,
    },
    14: {
        "expected_rows": 11,
        "expected_columns": 5,
        "logical_header_rows": 0,
        "difficulty": "continuation_page_with_missing_fragments",
        "evaluation_group": "ocr_stress",
        "ocr_psm": 3,
    },
    15: {
        "expected_rows": 10,
        "expected_columns": 6,
        "logical_header_rows": 0,
        "difficulty": "continuation_page_with_missing_fragments",
        "evaluation_group": "ocr_stress",
        "ocr_psm": 3,
    },
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build experiments/generalization_v8.json and validate its files."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/images/generalization_v8"),
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=Path("data/ocr/generalization_v8/adaptive"),
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("annotations/generalization_v8"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generalization_v8.json"),
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Write the manifest even if referenced files are not present yet.",
    )
    return parser.parse_args()


def relative_to_project(path: Path) -> str:
    absolute = resolve(path).resolve()
    try:
        return str(absolute.relative_to(PROJECT_ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"Manifest path must be inside the project: {absolute}") from error


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_document(document: dict[str, Any]) -> None:
    paths = {
        field: resolve(Path(document[field]))
        for field in ("image", "ocr", "ground_truth")
    }
    missing = [f"{field}: {path}" for field, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{document['id']}: missing referenced files:\n  " + "\n  ".join(missing)
        )

    ocr = load_json(paths["ocr"])
    truth = load_json(paths["ground_truth"])
    fragment_ids = {fragment["id"] for fragment in ocr.get("fragments", [])}
    assigned = [
        fragment_id
        for cell in truth.get("cells", [])
        for fragment_id in cell.get("fragment_ids", [])
    ]
    ignored = set(truth.get("ignored_fragment_ids", []))

    if len(assigned) != len(set(assigned)):
        raise ValueError(f"{document['id']}: a fragment is assigned to several cells")
    accounted = set(assigned) | ignored
    if accounted != fragment_ids:
        absent = sorted(fragment_ids - accounted)
        unknown = sorted(accounted - fragment_ids)
        raise ValueError(
            f"{document['id']}: fragment coverage mismatch; "
            f"unaccounted={absent}, unknown={unknown}"
        )

    grid = truth.get("grid", {})
    if int(grid.get("rows", -1)) != document["expected_rows"]:
        raise ValueError(
            f"{document['id']}: expected_rows={document['expected_rows']}, "
            f"annotation rows={grid.get('rows')}"
        )
    if int(grid.get("columns", -1)) != document["expected_columns"]:
        raise ValueError(
            f"{document['id']}: expected_columns={document['expected_columns']}, "
            f"annotation columns={grid.get('columns')}"
        )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    for number, metadata in TABLES.items():
        document = {
            "id": f"table_{number}",
            "image": relative_to_project(args.image_dir / f"table_{number}.png"),
            "ocr": relative_to_project(args.ocr_dir / f"table_{number}.json"),
            "ground_truth": relative_to_project(
                args.annotation_dir / f"table_{number}_cells_ground_truth.json"
            ),
            **metadata,
        }
        if not args.skip_file_check:
            validate_document(document)
        documents.append(document)

    return {
        "schema_version": "1.0",
        "name": "russian_ocr_table_structure_generalization_v8",
        "description": (
            "Independent Russian-language tables not used for model training or tuning. "
            "The primary group measures generalization; ocr_stress isolates degraded OCR."
        ),
        "selection_policy": "held_out_before_final_evaluation",
        "groups": {
            "primary": [document["id"] for document in documents if document["evaluation_group"] == "primary"],
            "ocr_stress": [document["id"] for document in documents if document["evaluation_group"] == "ocr_stress"],
        },
        "documents": documents,
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    primary_count = len(manifest["groups"]["primary"])
    stress_count = len(manifest["groups"]["ocr_stress"])
    print(f"Documents: {len(manifest['documents'])}")
    print(f"Primary: {primary_count}")
    print(f"OCR stress-test: {stress_count}")
    print(f"File validation: {'skipped' if args.skip_file_check else 'passed'}")
    print(f"JSON: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
