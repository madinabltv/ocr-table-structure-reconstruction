from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize text fragments in a table image with Tesseract."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input PNG/JPG")
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--preview-output", required=True, type=Path)
    parser.add_argument("--lang", default="rus+eng")
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Discard words below this confidence, from 0 to 100",
    )
    return parser.parse_args()


def load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if not 0 <= args.min_confidence <= 100:
        raise ValueError("--min-confidence must be between 0 and 100")

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.preview_output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.input).convert("RGB")
    config = f"--oem 1 --psm {args.psm}"
    data = pytesseract.image_to_data(
        image,
        lang=args.lang,
        config=config,
        output_type=Output.DICT,
    )

    fragments: list[dict] = []
    for index, raw_text in enumerate(data["text"]):
        text = " ".join(str(raw_text).split())
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if not text or confidence < args.min_confidence:
            continue

        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])
        fragments.append(
            {
                "id": len(fragments),
                "text": text,
                "bbox": [left, top, left + width, top + height],
                "confidence": round(confidence / 100.0, 4),
                "tesseract": {
                    "block": int(data["block_num"][index]),
                    "paragraph": int(data["par_num"][index]),
                    "line": int(data["line_num"][index]),
                    "word": int(data["word_num"][index]),
                },
            }
        )

    result = {
        "schema_version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_image": args.input.name,
        "image_size": {"width": image.width, "height": image.height},
        "ocr": {"engine": "tesseract", "lang": args.lang, "psm": args.psm},
        "fragment_count": len(fragments),
        "fragments": fragments,
    }
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    font = load_font()
    for fragment in fragments:
        x1, y1, x2, y2 = fragment["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(220, 30, 30), width=2)
        label = str(fragment["id"])
        label_box = draw.textbbox((x1, y1), label, font=font)
        label_width = label_box[2] - label_box[0] + 6
        label_height = label_box[3] - label_box[1] + 4
        label_top = max(0, y1 - label_height)
        draw.rectangle(
            (x1, label_top, x1 + label_width, label_top + label_height),
            fill=(220, 30, 30),
        )
        draw.text((x1 + 3, label_top + 1), label, fill="white", font=font)
    preview.save(args.preview_output)

    print(f"Recognized fragments: {len(fragments)}")
    print(f"JSON: {args.json_output}")
    print(f"Preview: {args.preview_output}")


if __name__ == "__main__":
    main()
