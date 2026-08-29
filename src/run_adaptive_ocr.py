from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output


@dataclass(frozen=True)
class Candidate:
    psm: int
    fragments: list[dict[str, Any]]
    metrics: dict[str, float | int]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive Tesseract OCR and select the best PSM mode."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input PNG/JPG")
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--preview-output", required=True, type=Path)
    parser.add_argument("--lang", default="rus+eng")
    parser.add_argument(
        "--psm-candidates",
        type=int,
        nargs="+",
        default=[3, 6],
        help="Tesseract PSM modes to compare (default: 3 6)",
    )
    parser.add_argument(
        "--prefer-psm-on-tie",
        type=int,
        default=6,
        help="Mode preferred when candidate scores differ by at most tie margin",
    )
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=0.03,
        help="Relative score difference treated as a tie (default: 0.03)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Discard words below this confidence, from 0 to 100",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="Optional directory for JSON and preview files of every candidate",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return " ".join(str(value).split())


def recognize(
    image: Image.Image,
    *,
    lang: str,
    psm: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--oem 1 --psm {psm}",
        output_type=Output.DICT,
    )

    fragments: list[dict[str, Any]] = []
    for index, raw_text in enumerate(data["text"]):
        text = normalize_text(raw_text)
        try:
            confidence_percent = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence_percent = -1.0
        if not text or confidence_percent < min_confidence:
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
                "confidence": round(confidence_percent / 100.0, 4),
                "tesseract": {
                    "block": int(data["block_num"][index]),
                    "paragraph": int(data["par_num"][index]),
                    "line": int(data["line_num"][index]),
                    "word": int(data["word_num"][index]),
                },
            }
        )
    return fragments


def visible_character_count(text: str) -> int:
    return sum(not character.isspace() for character in text)


def spatial_coverage(
    fragments: Iterable[dict[str, Any]],
    *,
    width: int,
    height: int,
    grid_size: int = 10,
) -> float:
    """Return the fraction of coarse image regions containing OCR fragments."""
    occupied: set[tuple[int, int]] = set()
    for fragment in fragments:
        x1, y1, x2, y2 = fragment["bbox"]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        column = min(grid_size - 1, max(0, int(center_x / width * grid_size)))
        row = min(grid_size - 1, max(0, int(center_y / height * grid_size)))
        occupied.add((row, column))
    return len(occupied) / float(grid_size * grid_size)


def evaluate_candidate(
    *,
    psm: int,
    fragments: list[dict[str, Any]],
    width: int,
    height: int,
) -> Candidate:
    character_count = 0
    weighted_characters = 0.0
    high_confidence_characters = 0
    low_confidence_fragments = 0
    confidence_sum = 0.0
    confidence_character_weight = 0

    for fragment in fragments:
        characters = visible_character_count(fragment["text"])
        confidence = min(1.0, max(0.0, float(fragment["confidence"])))
        character_count += characters
        weighted_characters += characters * confidence
        confidence_sum += confidence * max(1, characters)
        confidence_character_weight += max(1, characters)
        if confidence >= 0.50:
            high_confidence_characters += characters
        if confidence < 0.30:
            low_confidence_fragments += 1

    mean_confidence = (
        confidence_sum / confidence_character_weight
        if confidence_character_weight
        else 0.0
    )
    low_confidence_ratio = (
        low_confidence_fragments / len(fragments) if fragments else 1.0
    )
    coverage = spatial_coverage(
        fragments, width=width, height=height
    ) if fragments else 0.0

    # Text evidence is deliberately the dominant component.  Confidence keeps
    # garbage tokens from winning, while coverage rewards candidates that do
    # not recognize only one small part of a table.
    coverage_factor = 0.85 + 0.30 * coverage
    low_confidence_penalty = 1.0 - 0.25 * low_confidence_ratio
    score = weighted_characters * coverage_factor * low_confidence_penalty

    metrics: dict[str, float | int] = {
        "fragment_count": len(fragments),
        "character_count": character_count,
        "confidence_weighted_characters": round(weighted_characters, 4),
        "high_confidence_characters": high_confidence_characters,
        "mean_character_confidence": round(mean_confidence, 4),
        "low_confidence_fragments": low_confidence_fragments,
        "low_confidence_ratio": round(low_confidence_ratio, 4),
        "spatial_coverage": round(coverage, 4),
    }
    return Candidate(psm=psm, fragments=fragments, metrics=metrics, score=score)


def select_candidate(
    candidates: list[Candidate],
    *,
    preferred_psm: int,
    tie_margin: float,
) -> Candidate:
    if not candidates:
        raise ValueError("No OCR candidates were produced")
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    best = ranked[0]
    if best.score <= 0:
        return best

    tied = [
        candidate
        for candidate in ranked
        if (best.score - candidate.score) / best.score <= tie_margin
    ]
    return next(
        (candidate for candidate in tied if candidate.psm == preferred_psm),
        best,
    )


def load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_preview(
    image: Image.Image,
    fragments: list[dict[str, Any]],
    *,
    selected_psm: int,
) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    font = load_font()
    banner_font = load_font(24)

    banner_text = f"adaptive OCR: selected PSM {selected_psm}"
    banner_box = draw.textbbox((8, 8), banner_text, font=banner_font)
    draw.rectangle(
        (4, 4, banner_box[2] + 12, banner_box[3] + 12),
        fill=(25, 80, 150),
    )
    draw.text((8, 8), banner_text, fill="white", font=banner_font)

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
    return preview


def candidate_payload(
    candidate: Candidate,
    *,
    input_path: Path,
    image: Image.Image,
    lang: str,
) -> dict[str, Any]:
    return {
        "schema_version": "0.2",
        "source_image": input_path.name,
        "image_size": {"width": image.width, "height": image.height},
        "ocr": {"engine": "tesseract", "lang": lang, "psm": candidate.psm},
        "selection_score": round(candidate.score, 4),
        "selection_metrics": candidate.metrics,
        "fragment_count": len(candidate.fragments),
        "fragments": candidate.fragments,
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if not 0 <= args.min_confidence <= 100:
        raise ValueError("--min-confidence must be between 0 and 100")
    if not 0 <= args.tie_margin < 1:
        raise ValueError("--tie-margin must be in the range [0, 1)")
    psm_modes = list(dict.fromkeys(args.psm_candidates))
    if any(mode < 0 or mode > 13 for mode in psm_modes):
        raise ValueError("Tesseract PSM modes must be between 0 and 13")

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.preview_output.parent.mkdir(parents=True, exist_ok=True)
    if args.candidate_dir:
        args.candidate_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.input).convert("RGB")
    candidates: list[Candidate] = []
    for psm in psm_modes:
        fragments = recognize(
            image,
            lang=args.lang,
            psm=psm,
            min_confidence=args.min_confidence,
        )
        candidate = evaluate_candidate(
            psm=psm,
            fragments=fragments,
            width=image.width,
            height=image.height,
        )
        candidates.append(candidate)

        if args.candidate_dir:
            stem = f"{args.input.stem}_psm{psm}"
            payload = candidate_payload(
                candidate,
                input_path=args.input,
                image=image,
                lang=args.lang,
            )
            (args.candidate_dir / f"{stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            draw_preview(
                image, candidate.fragments, selected_psm=psm
            ).save(args.candidate_dir / f"{stem}_boxes.png")

    selected = select_candidate(
        candidates,
        preferred_psm=args.prefer_psm_on_tie,
        tie_margin=args.tie_margin,
    )
    diagnostics = {
        str(candidate.psm): {
            "score": round(candidate.score, 4),
            **candidate.metrics,
        }
        for candidate in candidates
    }
    result = {
        "schema_version": "0.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_image": args.input.name,
        "image_size": {"width": image.width, "height": image.height},
        "ocr": {
            "engine": "tesseract",
            "lang": args.lang,
            "psm": selected.psm,
            "selection": "adaptive_psm",
        },
        "adaptive_ocr": {
            "selected_psm": selected.psm,
            "psm_candidates": psm_modes,
            "preferred_psm_on_tie": args.prefer_psm_on_tie,
            "tie_margin": args.tie_margin,
            "candidate_diagnostics": diagnostics,
        },
        "fragment_count": len(selected.fragments),
        "fragments": selected.fragments,
    }
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_preview(
        image, selected.fragments, selected_psm=selected.psm
    ).save(args.preview_output)

    print(f"Input: {args.input}")
    for candidate in candidates:
        print(
            f"PSM {candidate.psm}: score={candidate.score:.2f}, "
            f"fragments={len(candidate.fragments)}, "
            f"characters={candidate.metrics['character_count']}, "
            f"mean_confidence={candidate.metrics['mean_character_confidence']:.3f}, "
            f"coverage={candidate.metrics['spatial_coverage']:.3f}"
        )
    print(f"Selected PSM: {selected.psm}")
    print(f"Recognized fragments: {len(selected.fragments)}")
    print(f"JSON: {args.json_output}")
    print(f"Preview: {args.preview_output}")


if __name__ == "__main__":
    main()
