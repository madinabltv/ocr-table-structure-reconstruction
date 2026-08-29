from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reconstructed structure to HTML")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Reconstructed table")
    parser.add_argument(
        "--show-coordinates",
        action="store_true",
        help="Show predicted row/column/span metadata inside cells",
    )
    return parser.parse_args()


def cell_text(cell: dict[str, Any]) -> str:
    value = str(cell.get("text", "")).strip()
    return value if value else "[пусто / OCR-пропуск]"


def render_group(cells: list[dict[str, Any]], show_coordinates: bool) -> str:
    parts = []
    for cell in cells:
        text = html.escape(cell_text(cell))
        fragments = ", ".join(html.escape(str(value)) for value in cell.get("fragment_ids", []))
        metadata = ""
        if show_coordinates:
            metadata = (
                '<div class="meta">'
                f"r{cell['row_start']} c{cell['column_start']} · "
                f"{cell.get('rowspan', 1)}×{cell.get('colspan', 1)} · "
                f"fragments: [{fragments}]"
                "</div>"
            )
        parts.append(f'<div class="cell-part"><div>{text}</div>{metadata}</div>')
    return '<div class="collision-separator">+</div>'.join(parts)


def build_html(structure: dict[str, Any], title: str, show_coordinates: bool) -> str:
    row_count = int(structure["row_count"])
    column_count = int(structure["column_count"])
    starts: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in structure["cells"]:
        starts[(int(cell["row_start"]), int(cell["column_start"]))].append(cell)

    occupied: set[tuple[int, int]] = set()
    rows = []
    for row in range(row_count):
        columns = []
        for column in range(column_count):
            if (row, column) in occupied:
                continue
            group = starts.get((row, column), [])
            if not group:
                columns.append('<td class="missing">[нет предсказанной ячейки]</td>')
                continue
            rowspan = max(int(cell.get("rowspan", 1)) for cell in group)
            colspan = max(int(cell.get("colspan", 1)) for cell in group)
            rowspan = max(1, min(rowspan, row_count - row))
            colspan = max(1, min(colspan, column_count - column))
            for covered_row in range(row, row + rowspan):
                for covered_column in range(column, column + colspan):
                    if (covered_row, covered_column) != (row, column):
                        occupied.add((covered_row, covered_column))
            css_class = "collision" if len(group) > 1 else "predicted"
            attributes = ""
            if rowspan > 1:
                attributes += f' rowspan="{rowspan}"'
            if colspan > 1:
                attributes += f' colspan="{colspan}"'
            columns.append(
                f'<td class="{css_class}"{attributes}>{render_group(group, show_coordinates)}</td>'
            )
        rows.append("<tr>" + "".join(columns) + "</tr>")

    warning_items = "".join(
        f"<li>{html.escape(str(warning))}</li>" for warning in structure.get("warnings", [])
    ) or "<li>Нет предупреждений</li>"
    escaped_title = html.escape(title)
    source = html.escape(str(structure.get("source_relations", "")))
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2937; }}
    h1 {{ margin-bottom: 6px; }}
    .summary {{ color: #4b5563; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    td {{ border: 1px solid #64748b; padding: 10px; vertical-align: top; overflow-wrap: anywhere; }}
    td.predicted {{ background: #f8fafc; }}
    td.missing {{ background: #fff7ed; color: #9a3412; font-style: italic; }}
    td.collision {{ background: #fef2f2; border: 2px solid #dc2626; }}
    .cell-part {{ padding: 3px 0; }}
    .collision-separator {{ color: #dc2626; font-weight: 700; border-top: 1px dashed #dc2626; margin: 5px 0; }}
    .meta {{ color: #64748b; font-family: ui-monospace, monospace; font-size: 11px; margin-top: 5px; }}
    .legend {{ display: flex; gap: 18px; margin: 16px 0; font-size: 13px; }}
    .swatch {{ display: inline-block; width: 13px; height: 13px; margin-right: 5px; vertical-align: -2px; border: 1px solid #64748b; }}
    .warnings {{ margin-top: 24px; padding: 14px 18px; background: #f8fafc; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <div class="summary">Метод: {html.escape(str(structure.get('method', '')))} · источник: {source} · сетка: {row_count}×{column_count}</div>
  <div class="legend">
    <span><i class="swatch" style="background:#f8fafc"></i>предсказанная ячейка</span>
    <span><i class="swatch" style="background:#fff7ed"></i>пустая позиция</span>
    <span><i class="swatch" style="background:#fef2f2;border-color:#dc2626"></i>коллизия</span>
  </div>
  <table>{''.join(rows)}</table>
  <section class="warnings"><strong>Предупреждения</strong><ul>{warning_items}</ul></section>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    structure = json.loads(args.input.read_text(encoding="utf-8"))
    result = build_html(structure, args.title, args.show_coordinates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    collision_count = sum(1 for warning in structure.get("warnings", []) if "collision" in warning)
    print(f"Grid: {structure['row_count']}x{structure['column_count']}")
    print(f"Logical cells: {structure['logical_cell_count']}")
    print(f"Collisions: {collision_count}")
    print(f"HTML: {args.output}")


if __name__ == "__main__":
    main()
