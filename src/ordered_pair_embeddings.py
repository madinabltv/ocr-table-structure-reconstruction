from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from relation_features import normalized_text


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def ordered_pair_key(source_text: str, target_text: str) -> tuple[str, str]:
    """Return a normalized cache key while preserving source/target order."""

    return normalized_text(source_text), normalized_text(target_text)


def projected_feature_names(dimension: int) -> tuple[str, ...]:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return tuple(f"ordered_pair_component_{index:03d}" for index in range(dimension))


class OrderedPairEncoder:
    """Encode source and target jointly with a frozen transformer."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 128,
        device: str = "cpu",
    ) -> None:
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "ordered-pair embeddings require torch and transformers; "
                "install requirements.txt"
            ) from error

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dimension = int(self.model.config.hidden_size)

    def encode(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not pairs:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        torch = self.torch
        batches = range(0, len(pairs), batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                batches = tqdm(batches, total=(len(pairs) + batch_size - 1) // batch_size)
            except ImportError:
                pass

        encoded_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in batches:
                batch = pairs[start : start + batch_size]
                inputs = self.tokenizer(
                    [source for source, _ in batch],
                    [target for _, target in batch],
                    padding=True,
                    truncation="longest_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                output = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                encoded_batches.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(encoded_batches)


def build_ordered_embedding_lookup(
    pairs: Iterable[tuple[str, str]],
    model_name: str = DEFAULT_MODEL,
    cache_path: Path | None = None,
    batch_size: int = 64,
    max_length: int = 128,
    device: str = "cpu",
    show_progress: bool = True,
) -> dict[tuple[str, str], np.ndarray]:
    """Load or calculate embeddings for unique ordered pairs."""

    unique_pairs = sorted({ordered_pair_key(source, target) for source, target in pairs})
    lookup: dict[tuple[str, str], np.ndarray] = {}
    if cache_path and cache_path.exists():
        cached = joblib.load(cache_path)
        expected = {
            "model_name": model_name,
            "max_length": max_length,
            "pooling": "mean",
        }
        actual = {name: cached.get(name) for name in expected}
        if actual != expected:
            raise ValueError(
                "ordered-pair cache settings do not match the requested model/max_length"
            )
        lookup = {
            tuple(pair): np.asarray(vector, dtype=np.float32)
            for pair, vector in zip(cached["pairs"], cached["embeddings"])
        }
        print(f"Ordered-pair cache: loaded {len(lookup)} pairs")

    missing = [pair for pair in unique_pairs if pair not in lookup]
    if missing:
        print(f"Embedding model: {model_name}")
        print(f"Unique ordered pairs to encode: {len(unique_pairs)}")
        print(f"Missing ordered pairs: {len(missing)}")
        encoder = OrderedPairEncoder(model_name, max_length=max_length, device=device)
        embeddings = encoder.encode(missing, batch_size=batch_size, show_progress=show_progress)
        lookup.update(zip(missing, embeddings))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached_pairs = sorted(lookup)
        cached_embeddings = np.stack([lookup[pair] for pair in cached_pairs]).astype(np.float32)
        joblib.dump(
            {
                "schema_version": "1.0",
                "model_name": model_name,
                "max_length": max_length,
                "pooling": "mean",
                "pairs": cached_pairs,
                "embeddings": cached_embeddings,
            },
            cache_path,
        )
        print(f"Ordered-pair cache saved: {cache_path}")
    return {pair: lookup[pair] for pair in unique_pairs}
