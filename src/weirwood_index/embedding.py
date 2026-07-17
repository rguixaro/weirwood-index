from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from weirwood_index.models import WeirwoodError

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MODEL_REVISIONS = {
    DEFAULT_MODEL: DEFAULT_REVISION,
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
}
MODEL_QUERY_INSTRUCTIONS = {
    DEFAULT_MODEL: QUERY_INSTRUCTION,
    "BAAI/bge-base-en-v1.5": QUERY_INSTRUCTION,
    "BAAI/bge-m3": "",
}


@runtime_checkable
class Encoder(Protocol):
    model_id: str
    revision: str
    max_tokens: int
    query_instruction: str

    def token_count(self, text: str) -> int: ...

    def encode_passages(self, passages: list[str]) -> np.ndarray: ...

    def encode_queries(self, queries: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """CPU-only BGE encoder. The retrieval instruction is applied only to queries."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str = DEFAULT_REVISION,
        *,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise WeirwoodError(
                "sentence-transformers is not installed; run `uv sync --dev`"
            ) from exc

        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.query_instruction = MODEL_QUERY_INSTRUCTIONS.get(model_id, "")
        local_root = Path(os.environ.get("WEIRWOOD_MODEL_CACHE", "models"))
        local_model = local_root / model_id.rsplit("/", maxsplit=1)[-1]
        try:
            if local_model.is_dir():
                self._model = SentenceTransformer(
                    str(local_model), device="cpu", local_files_only=True
                )
            else:
                self._model = SentenceTransformer(model_id, revision=revision, device="cpu")
        except Exception as exc:
            raise WeirwoodError(
                f"could not load embedding model {model_id}@{revision}; "
                "download it while online or check the local model cache"
            ) from exc
        self.max_tokens = min(int(self._model.max_seq_length), 512)

    def token_count(self, text: str) -> int:
        encoded = self._model.tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self._encode(passages)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([self.query_instruction + query for query in queries])


def resolve_model_revision(model_id: str, revision: str | None) -> str:
    if revision:
        return revision
    known = MODEL_REVISIONS.get(model_id)
    if known is None:
        raise WeirwoodError(
            f"--revision is required for unregistered model {model_id!r}"
        )
    return known


def create_encoder(
    model_id: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    *,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Encoder:
    return SentenceTransformerEncoder(
        model_id=model_id,
        revision=revision,
        batch_size=batch_size,
        show_progress=show_progress,
    )
