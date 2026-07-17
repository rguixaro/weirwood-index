from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from weirwood_index.models import WeirwoodError

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
DEFAULT_COLBERT_MODEL = "BAAI/bge-m3"
DEFAULT_COLBERT_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_RERANK_CANDIDATES = 20
DEFAULT_RERANKER_MAX_TOKENS = 512
DEFAULT_COLBERT_QUERY_TOKENS = 64
DEFAULT_COLBERT_PASSAGE_TOKENS = 256
RERANKER_KINDS = ("cross-encoder", "bge-m3-colbert")


@runtime_checkable
class Reranker(Protocol):
    model_id: str
    revision: str

    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    """CPU-only cross-encoder used after semantic candidate retrieval."""

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_MODEL,
        revision: str = DEFAULT_RERANKER_REVISION,
        *,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise WeirwoodError(
                "sentence-transformers is not installed; run `uv sync --dev`"
            ) from exc

        self.model_id = model_id
        self.revision = revision
        self.kind = "cross-encoder"
        self.batch_size = batch_size
        self.show_progress = show_progress
        local_root = Path(os.environ.get("WEIRWOOD_MODEL_CACHE", "models"))
        local_model = local_root / model_id.rsplit("/", maxsplit=1)[-1]
        try:
            if local_model.is_dir():
                self._model = CrossEncoder(
                    str(local_model),
                    device="cpu",
                    local_files_only=True,
                    max_length=DEFAULT_RERANKER_MAX_TOKENS,
                )
            else:
                self._model = CrossEncoder(
                    model_id,
                    revision=revision,
                    device="cpu",
                    max_length=DEFAULT_RERANKER_MAX_TOKENS,
                )
        except Exception as exc:
            raise WeirwoodError(
                f"could not load reranker {model_id}@{revision}; "
                "download it while online or check the local model cache"
            ) from exc

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty(0, dtype=np.float32)
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


class BgeM3ColbertReranker:
    """Local BGE-M3 token-level MaxSim reranker for retrieved passages."""

    def __init__(
        self,
        model_id: str = DEFAULT_COLBERT_MODEL,
        revision: str = DEFAULT_COLBERT_REVISION,
        *,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> None:
        del show_progress  # FlagEmbedding currently controls progress internally.
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise WeirwoodError(
                "BGE-M3 ColBERT reranking requires `uv sync --extra late-interaction`"
            ) from exc

        self.model_id = f"{model_id}#colbert"
        self.revision = revision
        self.kind = "bge-m3-colbert"
        self.batch_size = batch_size
        local_root = Path(os.environ.get("WEIRWOOD_MODEL_CACHE", "models"))
        local_model = local_root / model_id.rsplit("/", maxsplit=1)[-1]
        model_source = str(local_model) if local_model.is_dir() else model_id
        try:
            self._model = BGEM3FlagModel(
                model_source,
                use_fp16=False,
                devices="cpu",
                batch_size=batch_size,
                query_max_length=DEFAULT_COLBERT_QUERY_TOKENS,
                passage_max_length=DEFAULT_COLBERT_PASSAGE_TOKENS,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
                revision=revision,
            )
        except Exception as exc:
            raise WeirwoodError(
                f"could not load BGE-M3 ColBERT model {model_id}@{revision}; "
                "download it while online or check the local model cache"
            ) from exc

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty(0, dtype=np.float32)
        query_output = self._model.encode(
            [query],
            batch_size=1,
            max_length=DEFAULT_COLBERT_QUERY_TOKENS,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        passage_output = self._model.encode(
            passages,
            batch_size=self.batch_size,
            max_length=DEFAULT_COLBERT_PASSAGE_TOKENS,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        query_vectors = query_output["colbert_vecs"][0]
        passage_vectors = passage_output["colbert_vecs"]
        return colbert_maxsim_scores(query_vectors, passage_vectors)


def colbert_maxsim_scores(
    query_vectors: np.ndarray, passage_vectors: list[np.ndarray]
) -> np.ndarray:
    if query_vectors.ndim != 2:
        raise WeirwoodError("ColBERT query vectors must be a 2D array")
    scores = np.empty(len(passage_vectors), dtype=np.float32)
    for position, vectors in enumerate(passage_vectors):
        if vectors.ndim != 2 or vectors.shape[1] != query_vectors.shape[1]:
            raise WeirwoodError("ColBERT passage vectors have incompatible dimensions")
        token_scores = query_vectors @ vectors.T
        scores[position] = float(token_scores.max(axis=1).mean())
    return scores


def create_reranker(
    model_id: str | None = None,
    revision: str | None = None,
    *,
    kind: str = "cross-encoder",
    batch_size: int = 8,
    show_progress: bool = True,
) -> Reranker:
    if kind == "cross-encoder":
        return CrossEncoderReranker(
            model_id=model_id or DEFAULT_RERANKER_MODEL,
            revision=revision or DEFAULT_RERANKER_REVISION,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    if kind == "bge-m3-colbert":
        return BgeM3ColbertReranker(
            model_id=model_id or DEFAULT_COLBERT_MODEL,
            revision=revision or DEFAULT_COLBERT_REVISION,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    raise WeirwoodError(
        f"unknown reranker kind {kind!r}; choose one of: {', '.join(RERANKER_KINDS)}"
    )
