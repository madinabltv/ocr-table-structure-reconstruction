from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


OCR_LIST_KEYS = ("fragments", "ocr_fragments", "words", "items")
CELL_LIST_KEYS = ("cells", "predicted_cells", "reconstructed_cells", "items")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-ready image showing the source table, "
            "OCR boxes, and reconstructed cells."
        )
    )
    parser.add_argument("--image", required=True, type=Path,
                        help="Source table image (PNG/JPG/TIFF).")
    parser.add_argument("--ocr", required=True, type=Path,
                        help="JSON file with OCR fragments and bbox values.")
    parser.add_argument("--structure", required=True, type=Path,
                        help="JSON file with reconstructed cells.")
    parser.add_argument("--output", type=Path,
                        default=Path("figures/reconstruction_example.png"),
                        help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output resolution in DPI (default: 300).")
    parser.add_argument("--show-text", action="store_true",
                        help="Show shortened OCR text next to fragment IDs.")
    parser.add_argument("--open", action="store_true", dest="open_result",
                        help="Open the generated image in the default viewer.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def extract_list(data: Any, keys: Iterable[str], source_name: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(
        f"Cannot find a list of objects in {source_name}. "
        f"Expected a JSON list or one of the keys: {', '.join(keys)}"
    )


def parse_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = item.get("bbox", item.get("box"))
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x1, y1, x2, y2 = map(float, value)
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    if isinstance(value, dict):
        try:
            x1 = float(value.get("x1", value.get("left")))
            y1 = float(value.get("y1", value.get("top")))
            x2 = float(value.get("x2", x1 + float(value["width"])))
            y2 = float(value.get("y2", y1 + float(value["height"])))
        except (KeyError, TypeError, ValueError):
            return None
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    return None


def fragment_id(item: dict[str, Any], fallback: int) -> str:
    return str(item.get("id", item.get("fragment_id", fallback)))


def union_bbox(
    fragment_ids: Iterable[Any],
    fragment_boxes: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    boxes = [fragment_boxes[str(item_id)] for item_id in fragment_ids
             if str(item_id) in fragment_boxes]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def setup_axis(axis: Any, image: Image.Image, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title, fontsize=11, pad=8)
    axis.set_xlim(0, image.width)
    axis.set_ylim(image.height, 0)
    axis.axis("off")


def draw_box(
    axis: Any,
    bbox: tuple[float, float, float, float],
    color: str,
    linewidth: float,
    label: str | None = None,
) -> None:
    x1, y1, x2, y2 = bbox
    axis.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=linewidth,
        )
    )
    if label:
        axis.text(
            x1 + 2, max(2, y1 - 3), label,
            color="white", fontsize=6,
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85,
                  "boxstyle": "round,pad=0.15"},
        )


def open_file(path: Path) -> None:
    system = platform.system()
    if system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    command = "open" if system == "Darwin" else "xdg-open"
    if not shutil.which(command):
        print(f"The image was saved to {path}, but {command!r} is unavailable.")
        return
    subprocess.Popen(
        [command, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    fragments = extract_list(load_json(args.ocr), OCR_LIST_KEYS, str(args.ocr))
    cells = extract_list(load_json(args.structure), CELL_LIST_KEYS, str(args.structure))

    fragment_boxes: dict[str, tuple[float, float, float, float]] = {}
    for index, fragment in enumerate(fragments):
        bbox = parse_bbox(fragment)
        if bbox is not None:
            fragment_boxes[fragment_id(fragment, index)] = bbox

    figure, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    setup_axis(axes[0], image, "а) Исходное изображение")
    setup_axis(axes[1], image, "б) OCR-фрагменты")
    setup_axis(axes[2], image, "в) Восстановленные ячейки")

    for index, fragment in enumerate(fragments):
        bbox = parse_bbox(fragment)
        if bbox is None:
            continue
        item_id = fragment_id(fragment, index)
        label = item_id
        if args.show_text:
            text = " ".join(str(fragment.get("text", "")).split())
            if text:
                label = f"{item_id}: {text[:14]}" + ("…" if len(text) > 14 else "")
        draw_box(axes[1], bbox, color="#1565C0", linewidth=1.2, label=label)

    for index, cell in enumerate(cells):
        bbox = parse_bbox(cell)
        if bbox is None:
            ids = cell.get("fragment_ids", cell.get("fragments", []))
            bbox = union_bbox(ids if isinstance(ids, list) else [], fragment_boxes)
        if bbox is None:
            continue

        rowspan = int(cell.get("rowspan", 1) or 1)
        colspan = int(cell.get("colspan", 1) or 1)
        is_spanning = rowspan > 1 or colspan > 1
        color = "#D32F2F" if is_spanning else "#2E7D32"

        row_start = cell.get("row_start", "?")
        row_end = cell.get("row_end", row_start)
        col_start = cell.get("column_start", cell.get("col_start", "?"))
        col_end = cell.get("column_end", cell.get("col_end", col_start))
        cell_id = str(cell.get("id", f"cell_{index}"))
        label = f"{cell_id}  r:{row_start}-{row_end}, c:{col_start}-{col_end}"
        if is_spanning:
            label += f"  span:{rowspan}x{colspan}"
        draw_box(axes[2], bbox, color=color, linewidth=2.0, label=label)

    figure.text(
        0.5, 0.01,
        "Синий — OCR-фрагменты; зелёный — обычные ячейки; "
        "красный — объединённые ячейки.",
        ha="center", fontsize=9,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output = args.output.resolve()
    print(f"Saved: {output}")
    if args.open_result:
        open_file(output)


if __name__ == "__main__":
    main()
