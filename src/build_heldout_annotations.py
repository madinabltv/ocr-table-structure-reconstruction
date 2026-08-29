from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_benchmark_annotations import annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build held-out table annotations")
    parser.add_argument("--ocr-dir", type=Path, default=Path("data/ocr"))
    parser.add_argument("--output-dir", type=Path, default=Path("annotations/heldout"))
    return parser.parse_args()


def table_01(ocr: dict[str, Any]) -> dict[str, Any]:
    specs = [
        ("h00", 0, 1, 0, 0, []),
        ("h01", 0, 0, 1, 4, []),
        ("h05", 0, 1, 5, 5, []),
        ("h11", 1, 1, 1, 1, []),
        ("h12", 1, 1, 2, 2, []),
        ("h13", 1, 1, 3, 3, []),
        ("h14", 1, 1, 4, 4, []),
    ]
    body = [
        [[6], [7], [8], [9], [10], [11]],
        [[12], [13], [14], [15], [], [16]],
        [[17], [18], [19], [20], [21], [22]],
        [[23], [24], [25], [26], [27], [28]],
        [[29, 30], [31], [32], [], [], [33]],
        [[34, 35], [36], [38], [39], [40], [41]],
    ]
    for row_offset, row in enumerate(body, start=2):
        for column, ids in enumerate(row):
            specs.append(
                (f"c{row_offset}_{column}", row_offset, row_offset, column, column, ids)
            )
    return annotation(
        "table_01_solders_composition", ocr, specs, ignored=[0, 1, 2, 3, 4, 5, 37, 42]
    )


def table_05(ocr: dict[str, Any]) -> dict[str, Any]:
    specs = [
        ("h00", 0, 2, 0, 0, [11, 12]),
        ("h01", 0, 0, 1, 4, [0, 1, 2, 3, 4]),
        ("h11", 1, 1, 1, 2, [13]),
        ("h13", 1, 1, 3, 4, [5, 6, 7, 8, 9, 10, 14, 15]),
        ("h21", 2, 2, 1, 1, [16, 17, 23]),
        ("h22", 2, 2, 2, 2, [18, 19, 20, 24, 25]),
        ("h23", 2, 2, 3, 3, [26]),
        ("h24", 2, 2, 4, 4, [21, 22, 27, 28]),
    ]
    for body_row in range(13):
        output_row = body_row + 3
        start = 29 + body_row * 5
        for column in range(5):
            specs.append(
                (f"c{output_row}_{column}", output_row, output_row, column, column, [start + column])
            )
    return annotation("table_05_hardness", ocr, specs)


def index_for(value: float, boundaries: list[float]) -> int:
    return next(
        index
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
        if start <= value < end
    )


def table_06(ocr: dict[str, Any]) -> dict[str, Any]:
    x_boundaries = [0, 75, 210, 376, 510, 646, 780, 937, 1072, 1206, 1340, 1496, 1630, 1779]
    body_y_boundaries = [496, 573, 646, 719, 792, 865, 938, 1010, 1096]
    ignored = [int(item["id"]) for item in ocr["fragments"] if str(item["text"]).strip() == "|"]
    usable = [item for item in ocr["fragments"] if int(item["id"]) not in set(ignored)]
    header = [item for item in usable if (item["bbox"][1] + item["bbox"][3]) / 2 < 496]
    body = [item for item in usable if item not in header]

    specs: list[tuple[str, int, int, int, int, list[int]]] = []
    h00 = [int(item["id"]) for item in header if (item["bbox"][0] + item["bbox"][2]) / 2 < 75]
    h01 = [
        int(item["id"]) for item in header
        if (item["bbox"][0] + item["bbox"][2]) / 2 >= 75
        and (item["bbox"][1] + item["bbox"][3]) / 2 < 103
    ]
    specs.extend([
        ("h00", 0, 3, 0, 0, h00),
        ("h01", 0, 0, 1, 12, h01),
        ("h11", 1, 1, 1, 4, []),
        ("h15", 1, 1, 5, 8, []),
        ("h19", 1, 1, 9, 12, []),
    ])

    fixed_columns = {1, 2, 5, 6, 9, 10}
    pair_starts = {3, 7, 11}
    header_by_column: dict[int, list[int]] = {column: [] for column in range(1, 13)}
    for item in header:
        center_x = (item["bbox"][0] + item["bbox"][2]) / 2
        center_y = (item["bbox"][1] + item["bbox"][3]) / 2
        if center_x < 75 or center_y < 179:
            continue
        header_by_column[index_for(center_x, x_boundaries)].append(int(item["id"]))
    for column in range(1, 13):
        if column in fixed_columns:
            specs.append((f"h2_{column}", 2, 3, column, column, header_by_column[column]))
        elif column not in {4, 8, 12}:
            specs.append((f"h2_{column}", 2, 2, column, column, header_by_column[column]))
    for start in sorted(pair_starts):
        second = start + 1
        specs.append((f"h2_{second}", 2, 2, second, second, header_by_column[second]))
        specs.append((f"h3_{start}", 3, 3, start, second, []))

    grouped: dict[tuple[int, int], list[int]] = {
        (row, column): [] for row in range(4, 12) for column in range(13)
    }
    for item in body:
        center_x = (item["bbox"][0] + item["bbox"][2]) / 2
        center_y = (item["bbox"][1] + item["bbox"][3]) / 2
        column = index_for(center_x, x_boundaries)
        row = index_for(center_y, body_y_boundaries) + 4
        grouped[(row, column)].append(int(item["id"]))
    for row in range(4, 12):
        for column in range(13):
            specs.append((f"c{row}_{column}", row, row, column, column, grouped[(row, column)]))
    return annotation("table_06_mechanical_properties", ocr, specs, ignored=ignored)


def table_07(ocr: dict[str, Any]) -> dict[str, Any]:
    specs = [
        ("h00", 0, 1, 0, 0, [1, 2]),
        ("h01", 0, 0, 1, 2, []),
        ("h03", 0, 0, 3, 3, [0]),
        ("h11", 1, 1, 1, 1, []),
        ("h12", 1, 1, 2, 2, []),
        ("h13", 1, 1, 3, 3, [3, 4, 5]),
    ]
    left = [[6], [9], [11], [17], [22], [24], [29], [31]]
    temperatures = [[7], [10], [12], [18], [23], [25], [30], [32]]
    for offset, (left_ids, temperature_ids) in enumerate(zip(left, temperatures), start=2):
        specs.append((f"c{offset}_0", offset, offset, 0, 0, left_ids))
        specs.append((f"c{offset}_1", offset, offset, 1, 1, temperature_ids))
    specs.extend([
        ("c2_2", 2, 3, 2, 2, [8]),
        ("c4_2", 4, 6, 2, 2, [13, 14, 15]),
        ("c7_2", 7, 9, 2, 2, [26, 27, 28]),
        ("c2_3", 2, 9, 3, 3, [16, 19, 20, 21]),
    ])
    return annotation("table_07_heat_treatment", ocr, specs)


BUILDERS = {
    "table_01_solders_composition": table_01,
    "table_05_hardness": table_05,
    "table_06_mechanical_properties": table_06,
    "table_07_heat_treatment": table_07,
}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for document_id, builder in BUILDERS.items():
        ocr_path = args.ocr_dir / f"{document_id}.json"
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        result = builder(ocr)
        output = args.output_dir / f"{document_id}_cells_ground_truth.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{document_id}: cells={len(result['cells'])}, "
            f"ignored={len(result['ignored_fragment_ids'])}"
        )
        print(f"JSON: {output}")


if __name__ == "__main__":
    main()
