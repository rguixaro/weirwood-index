from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weirwood_index.chunking import ChunkProfile, chunk_corpus
from weirwood_index.corpus import parse_corpus, source_sha256
from weirwood_index.embedding import Encoder
from weirwood_index.events import EVENT_INDEX_VERSION, EventRecord, build_event_records
from weirwood_index.models import Chunk, IndexValidationError, WeirwoodError
from weirwood_index.narrative import (
    MAX_SENTENCE_VIEWS,
    NARRATIVE_VIEW_NAMES,
    NARRATIVE_VIEW_VERSION,
    NarrativeView,
    build_narrative_view,
)
from weirwood_index.scenes import (
    DEFAULT_ENTITY_SCOPE_WORDS,
    DEFAULT_SCENE_WINDOW_OVERLAP,
    DEFAULT_SCENE_WINDOW_WORDS,
    SCENE_WINDOW_VERSION,
    SceneWindow,
    build_scene_windows,
    map_chunks_to_scene_windows,
)

INDEX_FORMAT_VERSION = 2
SUPPORTED_INDEX_FORMATS = {1, INDEX_FORMAT_VERSION}
REQUIRED_ARTIFACTS = ("chunks.jsonl", "embeddings.npy", "manifest.json")
NARRATIVE_ARTIFACTS = ("narrative_views.jsonl", "narrative_embeddings.npz")
SCENE_WINDOW_ARTIFACTS = ("scene_windows.jsonl", "scene_embeddings.npy")
EVENT_ARTIFACTS = ("event_records.jsonl",)


@dataclass(frozen=True)
class BuiltIndex:
    path: Path
    chunk_count: int
    vector_dimensions: int
    embedding_seconds: float
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SceneEnrichment:
    path: Path
    window_count: int
    embedding_seconds: float
    manifest: dict[str, Any]


@dataclass(frozen=True)
class EventEnrichment:
    path: Path
    event_count: int
    extraction_seconds: float
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LoadedIndex:
    path: Path
    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray
    manifest: dict[str, Any]
    narrative_views: tuple[NarrativeView, ...] = ()
    narrative_embeddings: dict[str, np.ndarray] | None = None
    narrative_masks: dict[str, np.ndarray] | None = None
    sentence_embeddings: np.ndarray | None = None
    sentence_mask: np.ndarray | None = None
    context_embeddings: np.ndarray | None = None
    scene_windows: tuple[SceneWindow, ...] = ()
    scene_embeddings: np.ndarray | None = None
    chunk_scene_positions: tuple[tuple[int, ...], ...] = ()
    event_records: tuple[EventRecord, ...] = ()
    chunk_event_positions: tuple[tuple[int, ...], ...] = ()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _safe_model_name(model_id: str) -> str:
    return model_id.rsplit("/", maxsplit=1)[-1].replace("/", "-")


def _validate_vectors(vectors: np.ndarray, expected_count: int) -> None:
    if vectors.dtype != np.float32:
        raise IndexValidationError(f"embeddings must be float32, found {vectors.dtype}")
    if vectors.ndim != 2:
        raise IndexValidationError(f"embeddings must be a 2D array, found shape {vectors.shape}")
    if vectors.shape[0] != expected_count:
        raise IndexValidationError(
            f"embedding count {vectors.shape[0]} does not match chunk count {expected_count}"
        )
    if vectors.shape[1] < 1:
        raise IndexValidationError("embedding vectors have zero dimensions")
    if not np.isfinite(vectors).all():
        raise IndexValidationError("embedding array contains non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise IndexValidationError("embedding vectors are not unit-normalized")


def build_context_embeddings(
    chunks: Sequence[Chunk], vectors: np.ndarray
) -> np.ndarray:
    """Pool each passage with its immediate same-chapter neighbours."""
    _validate_vectors(vectors, len(chunks))
    position_by_ordinal = {
        (chunk.chapter_id, chunk.chunk_ordinal): position
        for position, chunk in enumerate(chunks)
    }
    contexts = np.empty_like(vectors)
    for position, chunk in enumerate(chunks):
        neighbor_positions = [
            position_by_ordinal.get((chunk.chapter_id, ordinal))
            for ordinal in range(chunk.chunk_ordinal - 1, chunk.chunk_ordinal + 2)
        ]
        pooled = vectors[
            [item for item in neighbor_positions if item is not None]
        ].sum(axis=0)
        norm = np.linalg.norm(pooled)
        if not np.isfinite(norm) or norm == 0.0:
            raise IndexValidationError(
                f"cannot derive a context vector for chunk {chunk.id}"
            )
        contexts[position] = pooled / norm
    return contexts


def _encode_optional_texts(
    texts: Sequence[str],
    encoder: Encoder,
    *,
    dimensions: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions = [position for position, text in enumerate(texts) if text.strip()]
    mask = np.zeros(len(texts), dtype=np.bool_)
    vectors = np.zeros((len(texts), dimensions), dtype=np.float32)
    if not positions:
        return vectors, mask
    selected = [texts[position] for position in positions]
    for position, text in zip(positions, selected, strict=True):
        tokens = encoder.token_count(text)
        if tokens > encoder.max_tokens or tokens > 512:
            raise WeirwoodError(
                f"narrative {label} view at chunk position {position} contains {tokens} "
                f"model tokens and would be truncated"
            )
    encoded = np.asarray(encoder.encode_passages(selected), dtype=np.float32)
    _validate_vectors(encoded, len(selected))
    if encoded.shape[1] != dimensions:
        raise WeirwoodError(
            f"narrative {label} view dimensions {encoded.shape[1]} do not match "
            f"raw passage dimensions {dimensions}"
        )
    vectors[positions] = encoded
    mask[positions] = True
    return vectors, mask


def _encode_sentence_views(
    views: Sequence[NarrativeView],
    encoder: Encoder,
    *,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.zeros(
        (len(views), MAX_SENTENCE_VIEWS, dimensions), dtype=np.float32
    )
    mask = np.zeros((len(views), MAX_SENTENCE_VIEWS), dtype=np.bool_)
    texts: list[str] = []
    positions: list[tuple[int, int]] = []
    for view_position, view in enumerate(views):
        for sentence_position, sentence in enumerate(view.sentences):
            tokens = encoder.token_count(sentence)
            if tokens > encoder.max_tokens or tokens > 512:
                raise WeirwoodError(
                    f"sentence narrative view {view.chunk_id}:{sentence_position} "
                    f"contains {tokens} model tokens and would be truncated"
                )
            texts.append(sentence)
            positions.append((view_position, sentence_position))
    if not texts:
        return vectors, mask
    encoded = np.asarray(encoder.encode_passages(texts), dtype=np.float32)
    _validate_vectors(encoded, len(texts))
    if encoded.shape[1] != dimensions:
        raise WeirwoodError(
            "sentence narrative view dimensions do not match raw passage dimensions"
        )
    for position, vector in zip(positions, encoded, strict=True):
        vectors[position] = vector
        mask[position] = True
    return vectors, mask


def _load_narrative_views(
    index_path: Path,
    chunks: Sequence[Chunk],
    config: Any,
) -> tuple[NarrativeView, ...]:
    if not isinstance(config, dict) or config.get("version") != NARRATIVE_VIEW_VERSION:
        raise IndexValidationError(
            "narrative view metadata is invalid or unsupported. Rebuild the index."
        )
    missing = [name for name in NARRATIVE_ARTIFACTS if not (index_path / name).is_file()]
    if missing:
        raise IndexValidationError(
            f"narrative index is incomplete; missing {', '.join(missing)}. Rebuild the index."
        )
    loaded: list[NarrativeView] = []
    try:
        with (index_path / "narrative_views.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IndexValidationError(
                        f"empty narrative view record at line {line_number}"
                    )
                loaded.append(NarrativeView.from_dict(json.loads(line)))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IndexValidationError(
            f"invalid narrative_views.jsonl: {exc}. Rebuild the index."
        ) from exc
    if len(loaded) != len(chunks):
        raise IndexValidationError(
            "narrative view count does not match chunk count. Rebuild the index."
        )
    if any(view.chunk_id != chunk.id for view, chunk in zip(loaded, chunks, strict=True)):
        raise IndexValidationError(
            "narrative view ordering does not match chunks. Rebuild the index."
        )
    return tuple(loaded)


def _load_narrative_embeddings(
    index_path: Path,
    *,
    chunk_count: int,
    dimensions: int,
    config: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    view_names = config.get("view_names")
    if view_names != list(NARRATIVE_VIEW_NAMES):
        raise IndexValidationError(
            "narrative view names do not match this version. Rebuild the index."
        )
    try:
        archive = np.load(index_path / "narrative_embeddings.npz", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise IndexValidationError(
            f"cannot read narrative embeddings: {exc}. Rebuild the index."
        ) from exc
    embeddings: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name in NARRATIVE_VIEW_NAMES:
        vectors = archive[name]
        mask = archive[f"{name}_mask"]
        if vectors.shape != (chunk_count, dimensions) or vectors.dtype != np.float32:
            raise IndexValidationError(
                f"narrative {name} embeddings are invalid. Rebuild the index."
            )
        if mask.shape != (chunk_count,) or mask.dtype != np.bool_:
            raise IndexValidationError(
                f"narrative {name} mask is invalid. Rebuild the index."
            )
        if not np.isfinite(vectors).all():
            raise IndexValidationError(
                f"narrative {name} embeddings contain non-finite values"
            )
        if mask.any():
            norms = np.linalg.norm(vectors[mask], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                raise IndexValidationError(
                    f"narrative {name} embeddings are not normalized"
                )
        embeddings[name] = vectors
        masks[name] = mask
    sentence_vectors = archive["sentences"]
    sentence_mask = archive["sentences_mask"]
    expected_sentence_shape = (chunk_count, MAX_SENTENCE_VIEWS, dimensions)
    if sentence_vectors.shape != expected_sentence_shape or sentence_vectors.dtype != np.float32:
        raise IndexValidationError(
            "sentence narrative embeddings are invalid. Rebuild the index."
        )
    if sentence_mask.shape != expected_sentence_shape[:2] or sentence_mask.dtype != np.bool_:
        raise IndexValidationError(
            "sentence narrative mask is invalid. Rebuild the index."
        )
    if not np.isfinite(sentence_vectors).all():
        raise IndexValidationError("sentence narrative embeddings contain non-finite values")
    if sentence_mask.any():
        norms = np.linalg.norm(sentence_vectors[sentence_mask], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise IndexValidationError("sentence narrative embeddings are not normalized")
    return embeddings, masks, sentence_vectors, sentence_mask


def _load_scene_windows(
    index_path: Path,
    chunks: tuple[Chunk, ...],
    dimensions: int,
    config: Any,
) -> tuple[tuple[SceneWindow, ...], np.ndarray, tuple[tuple[int, ...], ...]]:
    if not isinstance(config, dict) or config.get("version") != SCENE_WINDOW_VERSION:
        raise IndexValidationError(
            "scene window metadata is invalid or unsupported. Rebuild the scene windows."
        )
    missing = [name for name in SCENE_WINDOW_ARTIFACTS if not (index_path / name).is_file()]
    if missing:
        raise IndexValidationError(
            f"scene window index is incomplete; missing {', '.join(missing)}. "
            "Rebuild the scene windows."
        )
    windows: list[SceneWindow] = []
    try:
        with (index_path / "scene_windows.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IndexValidationError(
                        f"empty scene window record at line {line_number}"
                    )
                windows.append(SceneWindow.from_dict(json.loads(line)))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IndexValidationError(
            f"invalid scene_windows.jsonl: {exc}. Rebuild the scene windows."
        ) from exc
    if len(windows) != config.get("window_count"):
        raise IndexValidationError(
            "scene window count does not match the manifest. Rebuild the scene windows."
        )
    try:
        vectors = np.load(index_path / "scene_embeddings.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise IndexValidationError(
            f"cannot read scene_embeddings.npy: {exc}. Rebuild the scene windows."
        ) from exc
    _validate_vectors(vectors, len(windows))
    if vectors.shape[1] != dimensions:
        raise IndexValidationError(
            "scene vector dimensions do not match passage vectors. Rebuild the scene windows."
        )
    loaded_windows = tuple(windows)
    try:
        mappings = map_chunks_to_scene_windows(chunks, loaded_windows)
    except WeirwoodError as exc:
        raise IndexValidationError(
            f"scene window mapping is invalid: {exc}. Rebuild the scene windows."
        ) from exc
    return loaded_windows, vectors, mappings


def enrich_scene_windows(
    index: LoadedIndex,
    encoder: Encoder,
    *,
    window_words: int = DEFAULT_SCENE_WINDOW_WORDS,
    overlap_words: int = DEFAULT_SCENE_WINDOW_OVERLAP,
    entity_scope_words: int = DEFAULT_ENTITY_SCOPE_WORDS,
    force: bool = False,
) -> SceneEnrichment:
    """Add independently embedded scene-window artifacts to an existing index."""
    model = index.manifest["model"]
    if encoder.model_id != model["id"] or encoder.revision != model["revision"]:
        raise WeirwoodError(
            "scene windows must use the same pinned model and revision as the passage index"
        )
    if "scene_windows" in index.manifest and not force:
        raise WeirwoodError(
            f"scene windows already exist in {index.path}; pass --force to rebuild them"
        )

    windows = build_scene_windows(
        index.chunks,
        window_words=window_words,
        overlap_words=overlap_words,
        entity_scope_words=entity_scope_words,
    )
    texts = [window.embedding_text for window in windows]
    for window, text in zip(windows, texts, strict=True):
        tokens = encoder.token_count(text)
        if tokens > encoder.max_tokens or tokens > 512:
            raise WeirwoodError(
                f"scene window {window.id} contains {tokens} model tokens and would be "
                f"truncated; reduce --window-words"
            )

    started = time.perf_counter()
    vectors = np.asarray(encoder.encode_passages(texts), dtype=np.float32)
    elapsed = time.perf_counter() - started
    _validate_vectors(vectors, len(windows))
    if vectors.shape[1] != index.embeddings.shape[1]:
        raise WeirwoodError(
            "scene embedding dimensions do not match the existing passage index"
        )

    manifest = json.loads(json.dumps(index.manifest))
    manifest["scene_windows"] = {
        "version": SCENE_WINDOW_VERSION,
        "window_words": window_words,
        "overlap_words": overlap_words,
        "entity_scope_words": entity_scope_words,
        "window_count": len(windows),
        "embedding_seconds": round(elapsed, 6),
        "artifacts": list(SCENE_WINDOW_ARTIFACTS),
        "entity_propagation": "local-scope-aliases-v1",
    }
    suffix = f".tmp-{os.getpid()}"
    windows_temp = index.path / f"scene_windows.jsonl{suffix}"
    vectors_temp = index.path / f"scene_embeddings.npy{suffix}"
    manifest_temp = index.path / f"manifest.json{suffix}"
    try:
        with windows_temp.open("w", encoding="utf-8", newline="\n") as handle:
            for window in windows:
                handle.write(
                    json.dumps(window.to_dict(), ensure_ascii=True, sort_keys=True) + "\n"
                )
        with vectors_temp.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        manifest_temp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(windows_temp, index.path / "scene_windows.jsonl")
        os.replace(vectors_temp, index.path / "scene_embeddings.npy")
        os.replace(manifest_temp, index.path / "manifest.json")
    except Exception:
        for path in (windows_temp, vectors_temp, manifest_temp):
            path.unlink(missing_ok=True)
        raise
    return SceneEnrichment(
        path=index.path,
        window_count=len(windows),
        embedding_seconds=elapsed,
        manifest=manifest,
    )


def _load_event_records(
    index_path: Path,
    chunks: tuple[Chunk, ...],
    windows: tuple[SceneWindow, ...],
    chunk_scene_positions: tuple[tuple[int, ...], ...],
    config: Any,
) -> tuple[tuple[EventRecord, ...], tuple[tuple[int, ...], ...]]:
    if not isinstance(config, dict) or config.get("version") != EVENT_INDEX_VERSION:
        raise IndexValidationError(
            "event index metadata is invalid or unsupported. Rebuild the event index."
        )
    if not windows or len(windows) != config.get("scene_window_count"):
        raise IndexValidationError(
            "event index does not match the scene windows. Rebuild the event index."
        )
    path = index_path / "event_records.jsonl"
    if not path.is_file():
        raise IndexValidationError(
            "event index is incomplete; missing event_records.jsonl. Rebuild the event index."
        )
    records: list[EventRecord] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IndexValidationError(
                        f"empty event record at line {line_number}"
                    )
                records.append(EventRecord.from_dict(json.loads(line)))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IndexValidationError(
            f"invalid event_records.jsonl: {exc}. Rebuild the event index."
        ) from exc
    if len(records) != config.get("event_count"):
        raise IndexValidationError(
            "event record count does not match the manifest. Rebuild the event index."
        )
    window_positions = {window.id: position for position, window in enumerate(windows)}
    events_by_window: dict[int, list[int]] = {}
    for event_position, record in enumerate(records):
        scene_position = window_positions.get(record.scene_window_id)
        if scene_position is None:
            raise IndexValidationError(
                f"event {record.id} references an unknown scene window"
            )
        events_by_window.setdefault(scene_position, []).append(event_position)
    mappings = tuple(
        tuple(
            event_position
            for scene_position in scene_positions
            for event_position in events_by_window.get(scene_position, ())
        )
        for scene_positions in chunk_scene_positions
    )
    if len(mappings) != len(chunks) or any(not positions for positions in mappings):
        raise IndexValidationError(
            "event index does not cover every passage. Rebuild the event index."
        )
    return tuple(records), mappings


def enrich_event_records(
    index: LoadedIndex,
    nlp: Any,
    *,
    batch_size: int = 32,
    force: bool = False,
) -> EventEnrichment:
    """Parse scene windows into auditable subject/action/object event records."""
    if not index.scene_windows:
        raise WeirwoodError(
            "event extraction requires `weirwood index enrich-scenes` first"
        )
    if "event_index" in index.manifest and not force:
        raise WeirwoodError(
            f"an event index already exists in {index.path}; pass --force to rebuild it"
        )
    if batch_size < 1:
        raise WeirwoodError("event parser batch size must be positive")
    started = time.perf_counter()
    records = build_event_records(index.scene_windows, nlp, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    if not records:
        raise WeirwoodError("event extraction produced no records")
    covered_windows = {record.scene_window_id for record in records}
    missing = [window.id for window in index.scene_windows if window.id not in covered_windows]
    if missing:
        raise WeirwoodError(
            f"event extraction produced no event for {len(missing)} scene windows; "
            f"first missing window: {missing[0]}"
        )

    manifest = json.loads(json.dumps(index.manifest))
    manifest["event_index"] = {
        "version": EVENT_INDEX_VERSION,
        "parser_model": nlp.meta.get("name", "unknown"),
        "parser_version": nlp.meta.get("version", "unknown"),
        "scene_window_count": len(index.scene_windows),
        "event_count": len(records),
        "extraction_seconds": round(elapsed, 6),
        "artifacts": list(EVENT_ARTIFACTS),
        "structure": "dependency-svo-modality-v1",
    }
    suffix = f".tmp-{os.getpid()}"
    events_temp = index.path / f"event_records.jsonl{suffix}"
    manifest_temp = index.path / f"manifest.json{suffix}"
    try:
        with events_temp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True) + "\n"
                )
        manifest_temp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(events_temp, index.path / "event_records.jsonl")
        os.replace(manifest_temp, index.path / "manifest.json")
    except Exception:
        events_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
        raise
    return EventEnrichment(
        path=index.path,
        event_count=len(records),
        extraction_seconds=elapsed,
        manifest=manifest,
    )


def build_index(
    *,
    source: str | Path | Sequence[str | Path],
    profile: ChunkProfile,
    encoder: Encoder,
    output_root: str | Path = "data/indexes",
    force: bool = False,
    narrative_views: bool = False,
) -> BuiltIndex:
    corpus = parse_corpus(source)
    chunks = chunk_corpus(corpus, profile)
    if not chunks:
        raise WeirwoodError("chunking produced no passages")

    for chunk in chunks:
        tokens = encoder.token_count(chunk.text)
        if tokens > encoder.max_tokens or tokens > 512:
            raise WeirwoodError(
                f"chunk {chunk.id} contains {tokens} model tokens and would be truncated; "
                f"limit is {min(encoder.max_tokens, 512)}"
            )

    started = time.perf_counter()
    vectors = np.asarray(
        encoder.encode_passages([chunk.text for chunk in chunks]), dtype=np.float32
    )
    _validate_vectors(vectors, len(chunks))
    built_views: tuple[NarrativeView, ...] = ()
    view_embeddings: dict[str, np.ndarray] = {}
    view_masks: dict[str, np.ndarray] = {}
    sentence_embeddings: np.ndarray | None = None
    sentence_mask: np.ndarray | None = None
    if narrative_views:
        built_views = tuple(build_narrative_view(chunk) for chunk in chunks)
        dimensions = vectors.shape[1]
        for view_name in NARRATIVE_VIEW_NAMES:
            texts = [getattr(view, view_name) for view in built_views]
            view_vectors, mask = _encode_optional_texts(
                texts,
                encoder,
                dimensions=dimensions,
                label=view_name,
            )
            view_embeddings[view_name] = view_vectors
            view_masks[view_name] = mask
        sentence_embeddings, sentence_mask = _encode_sentence_views(
            built_views,
            encoder,
            dimensions=dimensions,
        )
    elapsed = time.perf_counter() - started

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    feature_suffix = "-narrative" if narrative_views else ""
    index_name = (
        f"{_safe_model_name(encoder.model_id)}-{profile.name}{feature_suffix}-"
        f"{corpus.source_sha256[:12]}"
    )
    destination = output_root / index_name
    if destination.exists() and not force:
        raise WeirwoodError(
            f"index already exists: {destination}; pass --force to rebuild it"
        )

    temp = output_root / f".{index_name}.tmp-{os.getpid()}"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=False)
    try:
        with (temp / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
        np.save(temp / "embeddings.npy", vectors, allow_pickle=False)
        if narrative_views:
            with (temp / "narrative_views.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                for view in built_views:
                    handle.write(
                        json.dumps(view.to_dict(), ensure_ascii=True, sort_keys=True) + "\n"
                    )
            assert sentence_embeddings is not None
            assert sentence_mask is not None
            np.savez(
                temp / "narrative_embeddings.npz",
                **view_embeddings,
                **{f"{name}_mask": mask for name, mask in view_masks.items()},
                sentences=sentence_embeddings,
                sentences_mask=sentence_mask,
            )
        corpus_sources = corpus.sources
        source_records = [
            {
                "book_id": item.book_id,
                "book_title": item.book_title,
                "book_sequence": item.book_sequence,
                "path": os.path.relpath(item.path, destination),
                "path_base": "index",
                "sha256": item.sha256,
            }
            for item in corpus_sources
        ]
        manifest: dict[str, Any] = {
            "format_version": INDEX_FORMAT_VERSION,
            "normalization_version": corpus.normalization_version,
            "source": {
                # Resolve relative to the index directory so the complete project can move.
                "path": os.path.relpath(corpus.source_path, destination),
                "path_base": "index",
                "sha256": corpus.source_sha256,
                "sources": source_records,
            },
            "model": {
                "id": encoder.model_id,
                "revision": encoder.revision,
                "query_instruction": getattr(encoder, "query_instruction", "unknown"),
            },
            "packages": {
                "numpy": _package_version("numpy"),
                "sentence-transformers": _package_version("sentence-transformers"),
                "transformers": _package_version("transformers"),
                "weirwood-index": _package_version("weirwood-index"),
            },
            "chunk_profile": {
                "name": profile.name,
                "words": profile.words,
                "overlap": profile.overlap,
                "strategy": profile.strategy,
                "min_words": profile.min_words,
            },
            "vector_dimensions": int(vectors.shape[1]),
            "chunk_count": len(chunks),
            "embedding_seconds": round(elapsed, 6),
        }
        if narrative_views:
            manifest["narrative_views"] = {
                "version": NARRATIVE_VIEW_VERSION,
                "view_names": list(NARRATIVE_VIEW_NAMES),
                "max_sentence_views": MAX_SENTENCE_VIEWS,
                "artifacts": list(NARRATIVE_ARTIFACTS),
            }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        temp.replace(destination)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    return BuiltIndex(
        path=destination,
        chunk_count=len(chunks),
        vector_dimensions=int(vectors.shape[1]),
        embedding_seconds=elapsed,
        manifest=manifest,
    )


def load_index(path: str | Path, *, verify_source: bool = True) -> LoadedIndex:
    index_path = Path(path).expanduser().resolve()
    if not index_path.is_dir():
        raise IndexValidationError(f"index directory does not exist: {index_path}")
    missing = [name for name in REQUIRED_ARTIFACTS if not (index_path / name).is_file()]
    if missing:
        raise IndexValidationError(
            f"index is incomplete; missing {', '.join(missing)}. Rebuild the index."
        )

    try:
        manifest = json.loads((index_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexValidationError(f"cannot read manifest.json: {exc}. Rebuild the index.") from exc

    if manifest.get("format_version") not in SUPPORTED_INDEX_FORMATS:
        raise IndexValidationError(
            f"unsupported index format {manifest.get('format_version')!r}; "
            f"expected one of {sorted(SUPPORTED_INDEX_FORMATS)}. Rebuild the index."
        )
    required_keys = {
        "normalization_version",
        "source",
        "model",
        "chunk_profile",
        "vector_dimensions",
        "chunk_count",
    }
    absent = sorted(required_keys - manifest.keys())
    if absent:
        raise IndexValidationError(
            f"manifest is missing {', '.join(absent)}. Rebuild the index."
        )

    chunks: list[Chunk] = []
    try:
        with (index_path / "chunks.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IndexValidationError(f"empty chunk record at line {line_number}")
                chunks.append(Chunk.from_dict(json.loads(line)))
    except json.JSONDecodeError as exc:
        raise IndexValidationError(f"invalid chunks.jsonl: {exc}. Rebuild the index.") from exc

    expected_count = manifest["chunk_count"]
    if not isinstance(expected_count, int) or expected_count != len(chunks):
        raise IndexValidationError(
            f"manifest chunk count {expected_count!r} does not match {len(chunks)} records"
        )
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise IndexValidationError("chunk IDs are not unique. Rebuild the index.")

    try:
        vectors = np.load(index_path / "embeddings.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise IndexValidationError(
            f"cannot read embeddings.npy: {exc}. Rebuild the index."
        ) from exc
    _validate_vectors(vectors, len(chunks))
    if vectors.shape[1] != manifest["vector_dimensions"]:
        raise IndexValidationError(
            "vector dimensions do not match the manifest. Rebuild the index."
        )

    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise IndexValidationError("manifest source metadata is invalid. Rebuild the index.")
    if verify_source:
        records = source.get("sources")
        if records is None:
            records = [source]
        if not isinstance(records, list) or not records:
            raise IndexValidationError(
                "manifest source list is invalid. Rebuild the index."
            )
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise IndexValidationError(
                    "manifest source record is invalid. Rebuild the index."
                )
            source_path = Path(record["path"])
            if not source_path.is_absolute():
                source_path = (index_path / source_path).resolve()
            if not source_path.is_file():
                raise IndexValidationError(
                    f"raw source is missing at {source_path}; restore it or rebuild from a new path"
                )
            current_hash = source_sha256(source_path)
            if current_hash != record.get("sha256"):
                raise IndexValidationError(
                    "raw source hash no longer matches this index. Rebuild the index."
                )

    narrative_views: tuple[NarrativeView, ...] = ()
    narrative_embeddings: dict[str, np.ndarray] | None = None
    narrative_masks: dict[str, np.ndarray] | None = None
    sentence_embeddings: np.ndarray | None = None
    sentence_mask: np.ndarray | None = None
    narrative_config = manifest.get("narrative_views")
    if narrative_config is not None:
        narrative_views = _load_narrative_views(index_path, chunks, narrative_config)
        (
            narrative_embeddings,
            narrative_masks,
            sentence_embeddings,
            sentence_mask,
        ) = _load_narrative_embeddings(
            index_path,
            chunk_count=len(chunks),
            dimensions=vectors.shape[1],
            config=narrative_config,
        )

    scene_windows: tuple[SceneWindow, ...] = ()
    scene_embeddings: np.ndarray | None = None
    chunk_scene_positions: tuple[tuple[int, ...], ...] = ()
    scene_config = manifest.get("scene_windows")
    loaded_chunks = tuple(chunks)
    if scene_config is not None:
        (
            scene_windows,
            scene_embeddings,
            chunk_scene_positions,
        ) = _load_scene_windows(
            index_path,
            loaded_chunks,
            vectors.shape[1],
            scene_config,
        )

    event_records: tuple[EventRecord, ...] = ()
    chunk_event_positions: tuple[tuple[int, ...], ...] = ()
    event_config = manifest.get("event_index")
    if event_config is not None:
        event_records, chunk_event_positions = _load_event_records(
            index_path,
            loaded_chunks,
            scene_windows,
            chunk_scene_positions,
            event_config,
        )

    return LoadedIndex(
        path=index_path,
        chunks=loaded_chunks,
        embeddings=vectors,
        manifest=manifest,
        narrative_views=narrative_views,
        narrative_embeddings=narrative_embeddings,
        narrative_masks=narrative_masks,
        sentence_embeddings=sentence_embeddings,
        sentence_mask=sentence_mask,
        context_embeddings=build_context_embeddings(loaded_chunks, vectors),
        scene_windows=scene_windows,
        scene_embeddings=scene_embeddings,
        chunk_scene_positions=chunk_scene_positions,
        event_records=event_records,
        chunk_event_positions=chunk_event_positions,
    )
