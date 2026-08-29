
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark summaries")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for every method",
    )
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for value in args.input:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got: {value}")
        label, raw_path = value.split("=", 1)
        report = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        row = {
            "method": label,
            "evaluation_mode": report.get("evaluation_mode", ""),
            "row_banding": report.get("row_banding", ""),
        }
        row.update(report["macro_average"])
        rows.append(row)

    result = {"schema_version": "1.0", "methods": rows}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Methods: {len(rows)}")
    print(f"JSON: {args.json_output}")
    print(f"CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
