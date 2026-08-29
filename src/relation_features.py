from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import numpy as np


GEOMETRY_FEATURE_NAMES = (
    "horizontal_gap_norm",
    "vertical_gap_norm",
    "x_overlap",
    "y_overlap",
    "width_ratio",
    "height_ratio",
    "dx_norm",
    "dy_norm",
    "abs_dx_norm",
    "abs_dy_norm",
    "center_distance_norm",
)

TEXT_FEATURE_NAMES = (
    "source_length_log",
    "target_length_log",
    "length_ratio",
    "source_digit_ratio",
    "target_digit_ratio",
    "source_alpha_ratio",
    "target_alpha_ratio",
    "source_ends_hyphen",
    "source_ends_decimal_separator",
    "target_starts_digit",
    "both_numeric",
    "concatenated_numeric",
    "hyphen_join_is_alpha",
    "same_text_type",
    "mixed_alpha_numeric_pair",
)

SEMANTIC_FEATURE_NAMES = (
    "embedding_cosine_similarity",
    "embedding_l2_distance",
    "embedding_mean_abs_difference",
    "embedding_max_abs_difference",
)


def normalized_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def character_ratio(text: str, predicate) -> float:
    compact = "".join(character for character in text if not character.isspace())
    if not compact:
        return 0.0
    return sum(1 for character in compact if predicate(character)) / len(compact)


def numeric_text(text: str) -> bool:
    compact = normalized_text(text).replace(" ", "")
    return bool(re.fullmatch(r"[+−-]?\d+(?:[.,]\d+)?%?", compact))


def text_type(text: str) -> str:
    value = normalized_text(text)
    has_alpha = any(character.isalpha() for character in value)
    has_digit = any(character.isdigit() for character in value)
    if has_alpha and has_digit:
        return "mixed"
    if has_alpha:
        return "alpha"
    if has_digit:
        return "numeric"
    return "other"


def text_features(source_text: str, target_text: str) -> dict[str, float]:
    source = normalized_text(source_text)
    target = normalized_text(target_text)
    source_length = max(1, len(source))
    target_length = max(1, len(target))
    joined = source + target
    dehyphenated = source.rstrip("-‐‑‒–—") + target
    source_type = text_type(source)
    target_type = text_type(target)
    return {
        "source_length_log": math.log1p(len(source)),
        "target_length_log": math.log1p(len(target)),
        "length_ratio": min(source_length, target_length) / max(source_length, target_length),
        "source_digit_ratio": character_ratio(source, str.isdigit),
        "target_digit_ratio": character_ratio(target, str.isdigit),
        "source_alpha_ratio": character_ratio(source, str.isalpha),
        "target_alpha_ratio": character_ratio(target, str.isalpha),
        "source_ends_hyphen": float(source.endswith(("-", "‐", "‑", "‒", "–", "—"))),
        "source_ends_decimal_separator": float(source.endswith((",", "."))),
        "target_starts_digit": float(bool(target) and target[0].isdigit()),
        "both_numeric": float(numeric_text(source) and numeric_text(target)),
        "concatenated_numeric": float(numeric_text(joined)),
        "hyphen_join_is_alpha": float(
            source.endswith(("-", "‐", "‑", "‒", "–", "—"))
            and bool(dehyphenated)
            and all(character.isalpha() for character in dehyphenated)
        ),
        "same_text_type": float(source_type == target_type),
        "mixed_alpha_numeric_pair": float(
            {source_type, target_type} == {"alpha", "numeric"}
        ),
    }


def feature_names(feature_set: str) -> tuple[str, ...]:
    if feature_set == "geometry":
        return GEOMETRY_FEATURE_NAMES
    if feature_set == "geometry_text":
        return GEOMETRY_FEATURE_NAMES + TEXT_FEATURE_NAMES
    if feature_set == "geometry_text_semantic":
        return GEOMETRY_FEATURE_NAMES + TEXT_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES
    raise ValueError(f"unknown feature set: {feature_set!r}")


def combined_features(
    geometry: dict[str, Any],
    source_text: str,
    target_text: str,
    feature_set: str,
    source_embedding: np.ndarray | None = None,
    target_embedding: np.ndarray | None = None,
) -> dict[str, float]:
    result = {name: float(geometry[name]) for name in GEOMETRY_FEATURE_NAMES}
    if feature_set in ("geometry_text", "geometry_text_semantic"):
        result.update(text_features(source_text, target_text))
    if feature_set == "geometry_text_semantic":
        if source_embedding is None or target_embedding is None:
            raise ValueError("semantic feature set requires both embeddings")
        result.update(semantic_features(source_embedding, target_embedding))
    return result


def semantic_features(source_embedding: np.ndarray, target_embedding: np.ndarray) -> dict[str, float]:
    source = np.asarray(source_embedding, dtype=np.float64)
    target = np.asarray(target_embedding, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 1:
        raise ValueError("embeddings must be one-dimensional and have equal shapes")
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    denominator = max(1e-12, source_norm * target_norm)
    difference = source - target
    return {
        "embedding_cosine_similarity": float(np.dot(source, target) / denominator),
        "embedding_l2_distance": float(np.linalg.norm(difference)),
        "embedding_mean_abs_difference": float(np.mean(np.abs(difference))),
        "embedding_max_abs_difference": float(np.max(np.abs(difference))),
    }
