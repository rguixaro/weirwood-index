from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from weirwood_index.embedding import Encoder
from weirwood_index.events import load_event_parser, structured_event_scores
from weirwood_index.indexing import LoadedIndex
from weirwood_index.lexical import BM25Index, LexicalEvidenceScores, tokenize
from weirwood_index.models import Chunk, WeirwoodError
from weirwood_index.narrative import expand_narrative_query
from weirwood_index.reranking import DEFAULT_RERANK_CANDIDATES, Reranker
from weirwood_index.scenes import expand_scene_query

RETRIEVAL_MODES = ("semantic", "lexical", "hybrid")
RETENTION_MODES = ("per-chapter", "global")
DEFAULT_CANDIDATE_POOL = 50
DEFAULT_CHAPTER_CANDIDATES = 20
DEFAULT_PASSAGES_PER_CHAPTER = 8
DEFAULT_PASSAGE_CANDIDATE_POOL = 200
DEFAULT_CHAPTER_EVIDENCE_PASSAGES = 3
DEFAULT_CHAPTER_WEIGHT = 0.25
DEFAULT_NEIGHBOR_WEIGHT = 0.10
DEFAULT_CONTEXT_VECTOR_WEIGHT = 0.0
DEFAULT_SCENE_WINDOW_WEIGHT = 0.0
DEFAULT_SCENE_LEXICAL_WEIGHT = 0.0
DEFAULT_EVENT_WEIGHT = 0.0
DEFAULT_LEXICAL_EVIDENCE_WEIGHT = 0.0
DEFAULT_RETENTION_MODE = "per-chapter"
RRF_CONSTANT = 60
QUOTATION_QUERY_TERMS = frozenset(
    {
        "asked",
        "asks",
        "called",
        "phrase",
        "quote",
        "repeats",
        "said",
        "says",
        "speaks",
        "teaches",
        "tells",
        "told",
        "words",
    }
)
IMAGERY_QUERY_TERMS = frozenset(
    {"appears", "color", "dream", "glowing", "image", "looks", "vision"}
)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk: Chunk
    retrieval: dict[str, Any] | None = None
    context_text: str | None = None
    context_word_start: int | None = None
    context_word_end: int | None = None

    def to_dict(self, *, excerpt_chars: int = 300) -> dict[str, Any]:
        data = asdict(self)
        data["chunk"].pop("text")
        retrieval = data.pop("retrieval")
        data.pop("context_text")
        context_start = data.pop("context_word_start")
        context_end = data.pop("context_word_end")
        if retrieval is not None:
            data["retrieval"] = retrieval
        if context_start is not None and context_end is not None:
            data["context_word_start"] = context_start
            data["context_word_end"] = context_end
        # Keep the preview centered on the scored focus passage. The context offsets
        # identify the wider adjacent-chunk window that a caller can expand on demand.
        data["excerpt"] = bounded_excerpt(self.chunk.text, excerpt_chars)
        return data


@dataclass(frozen=True)
class RetrievalTrace:
    chapter_ranking: tuple[str, ...]
    candidate_chunk_ids: frozenset[str]
    retained_chunk_ids: frozenset[str]
    within_chapter_ranks: dict[str, int]


@dataclass(frozen=True)
class SearchRun:
    results: list[SearchResult]
    trace: RetrievalTrace


def bounded_excerpt(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", maxsplit=1)[0]
    return clipped.rstrip(" ,;:") + "…"


def _normalize_query(query: str) -> str:
    normalized = " ".join(query.split())
    if not normalized:
        raise WeirwoodError("query must not be empty")
    return normalized


def _validate_top(top: int) -> None:
    if not 1 <= top <= 100:
        raise WeirwoodError("--top must be between 1 and 100")


def _normalized_pov(index: LoadedIndex, pov: str | None) -> str | None:
    if pov is None:
        return None
    normalized = pov.upper()
    available = {chunk.pov for chunk in index.chunks}
    if normalized not in available:
        raise WeirwoodError(
            f"unknown POV {normalized!r}; choose one of: {', '.join(sorted(available))}"
        )
    return normalized


def _query_vector(index: LoadedIndex, query: str, encoder: Encoder) -> np.ndarray:
    vectors = np.asarray(encoder.encode_queries([query]), dtype=np.float32)
    if vectors.shape != (1, index.embeddings.shape[1]):
        raise WeirwoodError(
            f"encoder returned shape {vectors.shape}; expected (1, {index.embeddings.shape[1]})"
        )
    norm = np.linalg.norm(vectors[0])
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-4):
        raise WeirwoodError("query encoder did not return a normalized finite vector")
    return vectors[0]


def _narrative_lexical_index(index: LoadedIndex) -> BM25Index:
    if not index.narrative_views:
        raise WeirwoodError(
            "narrative retrieval requires an index built with --narrative-views"
        )
    return BM25Index.from_texts([view.lexical_text for view in index.narrative_views])


def _scene_lexical_index(index: LoadedIndex) -> BM25Index:
    if not index.scene_windows:
        raise WeirwoodError(
            "scene-window retrieval requires `weirwood index enrich-scenes` first"
        )
    return BM25Index.from_texts([window.lexical_text for window in index.scene_windows])


def _event_lexical_index(index: LoadedIndex) -> BM25Index:
    if not index.event_records:
        raise WeirwoodError(
            "event retrieval requires `weirwood index enrich-events` first"
        )
    return BM25Index.from_texts([record.lexical_text for record in index.event_records])


def _scene_scores_for_chunks(
    index: LoadedIndex, scene_scores: np.ndarray
) -> np.ndarray:
    if scene_scores.shape != (len(index.scene_windows),):
        raise WeirwoodError("scene score count does not match the loaded scene windows")
    if len(index.chunk_scene_positions) != len(index.chunks):
        raise WeirwoodError("scene windows are not mapped to every passage")
    return np.asarray(
        [
            max(float(scene_scores[position]) for position in positions)
            for positions in index.chunk_scene_positions
        ],
        dtype=np.float32,
    )


def _event_scores_for_chunks(
    index: LoadedIndex,
    query: str,
    *,
    event_lexical_index: BM25Index | None,
    event_parser: Any | None,
) -> np.ndarray:
    if not index.event_records or len(index.chunk_event_positions) != len(index.chunks):
        raise WeirwoodError(
            "event retrieval requires `weirwood index enrich-events` first"
        )
    event_lexical_index = event_lexical_index or _event_lexical_index(index)
    event_parser = event_parser or load_event_parser()
    lexical_scores = event_lexical_index.scores(query)
    lexical_maximum = float(np.max(lexical_scores, initial=0.0))
    if lexical_maximum > 0.0:
        lexical_scores = lexical_scores / lexical_maximum
    structure_scores = structured_event_scores(
        index.event_records, query, event_parser
    )
    event_scores = 0.40 * lexical_scores + 0.60 * structure_scores
    return np.asarray(
        [
            max(float(event_scores[position]) for position in positions)
            for positions in index.chunk_event_positions
        ],
        dtype=np.float32,
    )


def _combined_lexical_scores(
    index: LoadedIndex,
    query: str,
    lexical_index: BM25Index,
    *,
    scene_lexical_weight: float,
    scene_lexical_index: BM25Index | None,
) -> np.ndarray:
    if not 0.0 <= scene_lexical_weight <= 1.0:
        raise WeirwoodError("--scene-lexical-weight must be between 0 and 1")
    passage_scores = lexical_index.scores(query)
    if not scene_lexical_weight:
        return passage_scores
    scene_lexical_index = scene_lexical_index or _scene_lexical_index(index)
    window_scores = scene_lexical_index.scores(expand_scene_query(query))
    scene_scores = _scene_scores_for_chunks(index, window_scores)

    def normalize(scores: np.ndarray) -> np.ndarray:
        maximum = float(np.max(scores, initial=0.0))
        return scores / maximum if maximum > 0.0 else scores

    return (
        (1.0 - scene_lexical_weight) * normalize(passage_scores)
        + scene_lexical_weight * normalize(scene_scores)
    )


def _semantic_scores(
    index: LoadedIndex,
    query: str,
    encoder: Encoder,
    *,
    narrative: bool,
    late_interaction: bool,
    context_vector_weight: float,
    scene_window_weight: float,
) -> np.ndarray:
    if not 0.0 <= context_vector_weight <= 1.0:
        raise WeirwoodError("--context-vector-weight must be between 0 and 1")
    if not 0.0 <= scene_window_weight <= 1.0:
        raise WeirwoodError("--scene-window-weight must be between 0 and 1")
    if narrative:
        query = expand_narrative_query(query)
    query_vector = _query_vector(index, query, encoder)
    raw_scores = index.embeddings @ query_vector
    if context_vector_weight:
        if index.context_embeddings is None:
            raise WeirwoodError("this loaded index has no derived context vectors")
        context_scores = index.context_embeddings @ query_vector
        raw_scores = raw_scores + context_vector_weight * np.maximum(
            context_scores - raw_scores,
            0.0,
        )
    if scene_window_weight:
        if index.scene_embeddings is None or not index.scene_windows:
            raise WeirwoodError(
                "scene-window retrieval requires `weirwood index enrich-scenes` first"
            )
        scene_query_vector = _query_vector(
            index, expand_scene_query(query), encoder
        )
        window_scores = index.scene_embeddings @ scene_query_vector
        scene_scores = _scene_scores_for_chunks(index, window_scores)
        raw_scores = raw_scores + scene_window_weight * np.maximum(
            scene_scores - raw_scores,
            0.0,
        )
    if not narrative and not late_interaction:
        return raw_scores
    if not narrative:
        raise WeirwoodError("--late-interaction requires --narrative")
    if index.narrative_embeddings is None or index.narrative_masks is None:
        raise WeirwoodError(
            "narrative retrieval requires an index built with --narrative-views"
        )

    query_terms = set(tokenize(query))
    quotation_query = bool(query_terms & QUOTATION_QUERY_TERMS) or '"' in query
    imagery_query = bool(query_terms & IMAGERY_QUERY_TERMS)
    weights = {
        "raw": 0.30,
        "context": 0.20,
        "summary": 0.15,
        "dialogue": 0.05,
        "events": 0.20,
        "entities": 0.10,
    }
    if quotation_query:
        weights.update({"raw": 0.25, "dialogue": 0.30, "events": 0.15})
    elif imagery_query:
        weights.update({"raw": 0.25, "summary": 0.25, "events": 0.15})

    fused = np.zeros(len(index.chunks), dtype=np.float32)

    def add_ranked(scores: np.ndarray, mask: np.ndarray, weight: float) -> None:
        eligible = np.flatnonzero(mask).tolist()
        order = _rank_positions(scores, eligible)
        for rank, position in enumerate(order, start=1):
            fused[position] += weight / (RRF_CONSTANT + rank)

    add_ranked(raw_scores, np.ones(len(index.chunks), dtype=np.bool_), weights["raw"])
    for view_name in ("context", "summary", "dialogue", "events", "entities"):
        view_scores = index.narrative_embeddings[view_name] @ query_vector
        add_ranked(view_scores, index.narrative_masks[view_name], weights[view_name])

    if late_interaction:
        if index.sentence_embeddings is None or index.sentence_mask is None:
            raise WeirwoodError(
                "late interaction requires sentence embeddings in the narrative index"
            )
        sentence_scores = np.einsum(
            "nsd,d->ns", index.sentence_embeddings, query_vector
        )
        sentence_scores = np.where(index.sentence_mask, sentence_scores, -np.inf)
        max_sentence_scores = sentence_scores.max(axis=1)
        sentence_mask = index.sentence_mask.any(axis=1)
        add_ranked(max_sentence_scores, sentence_mask, 0.25)
    return fused


def _normalized_book(index: LoadedIndex, book: str | None) -> str | None:
    if book is None:
        return None
    normalized = book.casefold()
    available = {chunk.book_id for chunk in index.chunks}
    if normalized not in available:
        raise WeirwoodError(
            f"unknown book {normalized!r}; choose one of: {', '.join(sorted(available))}"
        )
    return normalized


def _eligible_positions(
    index: LoadedIndex, pov: str | None, book: str | None = None
) -> list[int]:
    return [
        position
        for position, chunk in enumerate(index.chunks)
        if (pov is None or chunk.pov == pov)
        and (book is None or chunk.book_id == book)
    ]


def _rank_positions(
    scores: np.ndarray,
    eligible: list[int],
    *,
    limit: int | None = None,
    positive_only: bool = False,
) -> list[int]:
    ordered = sorted(eligible, key=lambda position: (-float(scores[position]), position))
    if positive_only:
        ordered = [position for position in ordered if scores[position] > 0]
    return ordered if limit is None else ordered[:limit]


def _select_non_overlapping_passages(
    index: LoadedIndex,
    ordered: list[int],
    *,
    limit: int | None,
) -> list[int]:
    selected: list[int] = []

    def overlaps_selected(position: int) -> bool:
        chunk = index.chunks[position]
        return any(
            index.chunks[prior].chapter_id == chunk.chapter_id
            and index.chunks[prior].word_start < chunk.word_end
            and chunk.word_start < index.chunks[prior].word_end
            for prior in selected
        )

    # The ranking is query-specific and overlap filtering already prevents near
    # duplicate windows. Keep filling by relevance: reserving extra slots for
    # distant passages reduced target retention in the two-volume benchmark.
    for position in ordered:
        if overlaps_selected(position):
            continue
        selected.append(position)
        if limit is not None and len(selected) == limit:
            break
    return selected


def _empty_lexical_evidence(document_count: int) -> LexicalEvidenceScores:
    empty = np.zeros(document_count, dtype=np.float32)
    return LexicalEvidenceScores(
        score=empty,
        coverage=empty.copy(),
        proximity=empty.copy(),
        phrase=empty.copy(),
    )


def _candidate_results(
    index: LoadedIndex,
    scores: np.ndarray,
    *,
    top: int,
    pov: str | None,
    book: str | None = None,
    excluded_ids: set[str] | None = None,
    source_chunk: Chunk | None = None,
    positive_only: bool = False,
    deduplicate_chapters: bool = False,
    retrieval_details: dict[int, dict[str, Any]] | None = None,
) -> list[SearchResult]:
    _validate_top(top)
    if scores.shape != (len(index.chunks),):
        raise WeirwoodError(
            f"score vector shape {scores.shape} does not match {len(index.chunks)} chunks"
        )
    pov = _normalized_pov(index, pov)
    book = _normalized_book(index, book)
    order = _rank_positions(
        scores,
        _eligible_positions(index, pov, book),
        positive_only=positive_only,
    )
    selected: list[tuple[int, Chunk, float]] = []
    excluded_ids = excluded_ids or set()
    for position in order:
        score = float(scores[position])
        if not np.isfinite(score):
            continue
        chunk = index.chunks[position]
        if chunk.id in excluded_ids:
            continue
        if source_chunk is not None and chunk.chapter_id == source_chunk.chapter_id:
            overlaps = (
                chunk.word_start < source_chunk.word_end
                and source_chunk.word_start < chunk.word_end
            )
            adjacent = abs(chunk.chunk_ordinal - source_chunk.chunk_ordinal) <= 1
            if overlaps or adjacent or chunk.text == source_chunk.text:
                continue
        if deduplicate_chapters and any(
            prior.chapter_id == chunk.chapter_id for _, prior, _ in selected
        ):
            continue
        if any(
            prior.chapter_id == chunk.chapter_id
            and prior.word_start < chunk.word_end
            and chunk.word_start < prior.word_end
            for _, prior, _ in selected
        ):
            continue
        selected.append((position, chunk, score))
        if len(selected) == top:
            break
    return [
        SearchResult(
            rank=rank,
            score=score,
            chunk=chunk,
            retrieval=(retrieval_details or {}).get(position),
        )
        for rank, (position, chunk, score) in enumerate(selected, 1)
    ]


def semantic_search(
    index: LoadedIndex,
    query: str,
    encoder: Encoder,
    *,
    top: int = 10,
    pov: str | None = None,
    book: str | None = None,
    deduplicate_chapters: bool = False,
    reranker: Reranker | None = None,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    rerank_context_words: int = 0,
    narrative: bool = False,
    late_interaction: bool = False,
    context_vector_weight: float = DEFAULT_CONTEXT_VECTOR_WEIGHT,
    scene_window_weight: float = DEFAULT_SCENE_WINDOW_WEIGHT,
) -> list[SearchResult]:
    query = _normalize_query(query)
    scores = _semantic_scores(
        index,
        query,
        encoder,
        narrative=narrative,
        late_interaction=late_interaction,
        context_vector_weight=context_vector_weight,
        scene_window_weight=scene_window_weight,
    )
    if reranker is not None:
        return _reranked_semantic_results(
            index,
            query,
            scores,
            reranker,
            top=top,
            pov=pov,
            book=book,
            deduplicate_chapters=deduplicate_chapters,
            rerank_candidates=rerank_candidates,
            rerank_context_words=rerank_context_words,
        )
    return _candidate_results(
        index,
        scores,
        top=top,
        pov=pov,
        book=book,
        deduplicate_chapters=deduplicate_chapters,
    )


def _reranked_semantic_results(
    index: LoadedIndex,
    query: str,
    semantic_scores: np.ndarray,
    reranker: Reranker,
    *,
    top: int,
    pov: str | None,
    book: str | None,
    deduplicate_chapters: bool,
    rerank_candidates: int,
    rerank_context_words: int,
) -> list[SearchResult]:
    _validate_top(top)
    if not 1 <= rerank_candidates <= 100:
        raise WeirwoodError("--rerank-candidates must be between 1 and 100")
    if not 0 <= rerank_context_words <= 500:
        raise WeirwoodError("--rerank-context-words must be between 0 and 500")

    # Candidate generation remains purely semantic. Chapter deduplication happens
    # after reranking so two passages from one chapter can compete on relevance.
    pool_size = max(top, rerank_candidates)
    semantic_pool = _candidate_results(
        index,
        semantic_scores,
        top=pool_size,
        pov=pov,
        book=book,
        deduplicate_chapters=False,
    )
    rerank_count = min(rerank_candidates, len(semantic_pool))
    rerank_pool = semantic_pool[:rerank_count]
    passages = _reranker_passages(
        index,
        [result.chunk for result in rerank_pool],
        context_words=rerank_context_words,
    )
    reranker_scores = np.asarray(reranker.score(query, passages), dtype=np.float32)
    if reranker_scores.shape != (rerank_count,):
        raise WeirwoodError(
            f"reranker returned shape {reranker_scores.shape}; expected ({rerank_count},)"
        )
    if not np.isfinite(reranker_scores).all():
        raise WeirwoodError("reranker returned non-finite scores")

    reranked = sorted(
        zip(rerank_pool, reranker_scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].rank),
    )
    ordered: list[SearchResult] = []
    for candidate, reranker_score in reranked:
        ordered.append(
            SearchResult(
                rank=0,
                score=float(reranker_score),
                chunk=candidate.chunk,
                retrieval={
                    "mode": "semantic-rerank",
                    "semantic_rank": candidate.rank,
                    "semantic_score": candidate.score,
                    "reranked": True,
                    "reranker_model": reranker.model_id,
                    "reranker_score": float(reranker_score),
                    "context_words": rerank_context_words,
                },
            )
        )
    for candidate in semantic_pool[rerank_count:]:
        ordered.append(
            SearchResult(
                rank=0,
                score=candidate.score,
                chunk=candidate.chunk,
                retrieval={
                    "mode": "semantic-rerank",
                    "semantic_rank": candidate.rank,
                    "semantic_score": candidate.score,
                    "reranked": False,
                    "reranker_model": reranker.model_id,
                    "reranker_score": None,
                    "context_words": rerank_context_words,
                },
            )
        )

    if deduplicate_chapters:
        # Ensure a request for (for example) 50 chapter-deduplicated results can
        # still be filled when the raw semantic pool contains repeated chapters.
        unique_semantic = _candidate_results(
            index,
            semantic_scores,
            top=top,
            pov=pov,
            book=book,
            deduplicate_chapters=True,
        )
        present_ids = {result.chunk.id for result in ordered}
        for candidate in unique_semantic:
            if candidate.chunk.id in present_ids:
                continue
            ordered.append(
                SearchResult(
                    rank=0,
                    score=candidate.score,
                    chunk=candidate.chunk,
                    retrieval={
                        "mode": "semantic-rerank",
                        "semantic_rank": candidate.rank,
                        "semantic_score": candidate.score,
                        "reranked": False,
                        "reranker_model": reranker.model_id,
                        "reranker_score": None,
                        "context_words": rerank_context_words,
                    },
                )
            )
            present_ids.add(candidate.chunk.id)

    selected: list[SearchResult] = []
    seen_chapters: set[str] = set()
    for result in ordered:
        if deduplicate_chapters and result.chunk.chapter_id in seen_chapters:
            continue
        selected.append(result)
        seen_chapters.add(result.chunk.chapter_id)
        if len(selected) == top:
            break
    return [
        SearchResult(rank=rank, score=result.score, chunk=result.chunk, retrieval=result.retrieval)
        for rank, result in enumerate(selected, start=1)
    ]


def _reranker_passages(
    index: LoadedIndex,
    chunks: list[Chunk],
    *,
    context_words: int,
) -> list[str]:
    if context_words <= 0:
        return [chunk.text for chunk in chunks]

    chapter_cache: dict[str, list[str]] = {}
    passages: list[str] = []
    for chunk in chunks:
        chapter_words = chapter_cache.get(chunk.chapter_id)
        if chapter_words is None:
            chapter_words = _reconstruct_chapter_words(index, chunk.chapter_id)
            chapter_cache[chunk.chapter_id] = chapter_words
        target_words = max(context_words, chunk.word_end - chunk.word_start)
        extra = target_words - (chunk.word_end - chunk.word_start)
        start = max(0, chunk.word_start - extra // 2)
        end = min(len(chapter_words), start + target_words)
        start = max(0, end - target_words)
        passages.append(" ".join(chapter_words[start:end]))
    return passages


def _reconstruct_chapter_words(index: LoadedIndex, chapter_id: str) -> list[str]:
    chapter_chunks = [chunk for chunk in index.chunks if chunk.chapter_id == chapter_id]
    if not chapter_chunks:
        raise WeirwoodError(f"cannot reconstruct missing chapter {chapter_id}")
    word_count = max(chunk.word_end for chunk in chapter_chunks)
    words: list[str | None] = [None] * word_count
    for chunk in chapter_chunks:
        chunk_words = chunk.text.split()
        if len(chunk_words) != chunk.word_end - chunk.word_start:
            raise WeirwoodError(f"chunk offsets do not match text for {chunk.id}")
        for position, word in enumerate(chunk_words, start=chunk.word_start):
            existing = words[position]
            if existing is not None and existing != word:
                raise WeirwoodError(f"overlapping chunk text disagrees in chapter {chapter_id}")
            words[position] = word
    if any(word is None for word in words):
        raise WeirwoodError(f"chunk coverage has gaps in chapter {chapter_id}")
    return [word for word in words if word is not None]


def lexical_search(
    index: LoadedIndex,
    query: str,
    *,
    top: int = 10,
    pov: str | None = None,
    book: str | None = None,
    deduplicate_chapters: bool = False,
    lexical_index: BM25Index | None = None,
    narrative: bool = False,
    scene_lexical_weight: float = DEFAULT_SCENE_LEXICAL_WEIGHT,
    scene_lexical_index: BM25Index | None = None,
) -> list[SearchResult]:
    query = _normalize_query(query)
    if narrative:
        query = expand_narrative_query(query)
    lexical_index = lexical_index or (
        _narrative_lexical_index(index)
        if narrative
        else BM25Index.from_chunks(index.chunks)
    )
    scores = _combined_lexical_scores(
        index,
        query,
        lexical_index,
        scene_lexical_weight=scene_lexical_weight,
        scene_lexical_index=scene_lexical_index,
    )
    pov = _normalized_pov(index, pov)
    book = _normalized_book(index, book)
    order = _rank_positions(
        scores,
        _eligible_positions(index, pov, book),
        positive_only=True,
    )
    details = {
        position: {
            "mode": "lexical",
            "lexical_rank": rank,
            "lexical_score": float(scores[position]),
        }
        for rank, position in enumerate(order, 1)
    }
    return _candidate_results(
        index,
        scores,
        top=top,
        pov=pov,
        book=book,
        positive_only=True,
        deduplicate_chapters=deduplicate_chapters,
        retrieval_details=details,
    )


def hybrid_search(
    index: LoadedIndex,
    query: str,
    encoder: Encoder,
    *,
    top: int = 10,
    pov: str | None = None,
    book: str | None = None,
    semantic_weight: float = 0.5,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    deduplicate_chapters: bool = False,
    lexical_index: BM25Index | None = None,
    narrative: bool = False,
    late_interaction: bool = False,
    context_vector_weight: float = DEFAULT_CONTEXT_VECTOR_WEIGHT,
    scene_window_weight: float = DEFAULT_SCENE_WINDOW_WEIGHT,
    scene_lexical_weight: float = DEFAULT_SCENE_LEXICAL_WEIGHT,
    scene_lexical_index: BM25Index | None = None,
    lexical_evidence_weight: float = DEFAULT_LEXICAL_EVIDENCE_WEIGHT,
    event_weight: float = DEFAULT_EVENT_WEIGHT,
    event_lexical_index: BM25Index | None = None,
    event_parser: Any | None = None,
) -> list[SearchResult]:
    query = _normalize_query(query)
    if narrative:
        query = expand_narrative_query(query)
    _validate_top(top)
    if not 0.0 <= semantic_weight <= 1.0:
        raise WeirwoodError("--semantic-weight must be between 0 and 1")
    if not 0.0 <= event_weight <= 1.0:
        raise WeirwoodError("--event-weight must be between 0 and 1")
    if not 0.0 <= lexical_evidence_weight <= 1.0:
        raise WeirwoodError("--lexical-evidence-weight must be between 0 and 1")
    if not top <= candidate_pool <= 500:
        raise WeirwoodError("--candidate-pool must be between --top and 500")

    pov = _normalized_pov(index, pov)
    book = _normalized_book(index, book)
    eligible = _eligible_positions(index, pov, book)
    semantic_scores = _semantic_scores(
        index,
        query,
        encoder,
        narrative=narrative,
        late_interaction=late_interaction,
        context_vector_weight=context_vector_weight,
        scene_window_weight=scene_window_weight,
    )
    lexical_index = lexical_index or (
        _narrative_lexical_index(index)
        if narrative
        else BM25Index.from_chunks(index.chunks)
    )
    lexical_scores = _combined_lexical_scores(
        index,
        query,
        lexical_index,
        scene_lexical_weight=scene_lexical_weight,
        scene_lexical_index=scene_lexical_index,
    )
    semantic_order = _rank_positions(
        semantic_scores, eligible, limit=candidate_pool
    )
    lexical_order = _rank_positions(
        lexical_scores,
        eligible,
        limit=candidate_pool,
        positive_only=True,
    )
    semantic_ranks = {position: rank for rank, position in enumerate(semantic_order, 1)}
    lexical_ranks = {position: rank for rank, position in enumerate(lexical_order, 1)}
    lexical_evidence = _empty_lexical_evidence(len(index.chunks))
    lexical_evidence_ranks: dict[int, int] = {}
    if lexical_evidence_weight:
        lexical_evidence = lexical_index.evidence_scores(query, positions=eligible)
        lexical_evidence_order = _rank_positions(
            lexical_evidence.score,
            eligible,
            limit=candidate_pool,
            positive_only=True,
        )
        lexical_evidence_ranks = {
            position: rank
            for rank, position in enumerate(lexical_evidence_order, 1)
        }
    event_scores = None
    event_ranks: dict[int, int] = {}
    if event_weight:
        event_scores = _event_scores_for_chunks(
            index,
            query,
            event_lexical_index=event_lexical_index,
            event_parser=event_parser,
        )
        event_order = _rank_positions(
            event_scores, eligible, limit=candidate_pool, positive_only=True
        )
        event_ranks = {
            position: rank for rank, position in enumerate(event_order, 1)
        }

    lexical_weight = 1.0 - semantic_weight
    fused_scores = np.full(len(index.chunks), -np.inf, dtype=np.float32)
    details: dict[int, dict[str, Any]] = {}
    fused_positions = (
        semantic_ranks.keys()
        | lexical_ranks.keys()
        | lexical_evidence_ranks.keys()
        | event_ranks.keys()
    )
    for position in fused_positions:
        semantic_rank = semantic_ranks.get(position)
        lexical_rank = lexical_ranks.get(position)
        fused = 0.0
        if semantic_rank is not None:
            fused += semantic_weight / (RRF_CONSTANT + semantic_rank)
        if lexical_rank is not None:
            fused += lexical_weight / (RRF_CONSTANT + lexical_rank)
        lexical_evidence_rank = lexical_evidence_ranks.get(position)
        if lexical_evidence_rank is not None:
            fused += lexical_evidence_weight / (
                RRF_CONSTANT + lexical_evidence_rank
            )
        event_rank = event_ranks.get(position)
        if event_rank is not None:
            fused += event_weight / (RRF_CONSTANT + event_rank)
        fused_scores[position] = fused
        details[position] = {
            "mode": "hybrid",
            "semantic_weight": semantic_weight,
            "semantic_rank": semantic_rank,
            "semantic_score": float(semantic_scores[position]),
            "lexical_weight": lexical_weight,
            "lexical_rank": lexical_rank,
            "lexical_score": float(lexical_scores[position]),
            "lexical_evidence_weight": lexical_evidence_weight,
            "lexical_evidence_rank": lexical_evidence_rank,
            "lexical_evidence_score": float(lexical_evidence.score[position]),
            "term_coverage_score": float(lexical_evidence.coverage[position]),
            "term_proximity_score": float(lexical_evidence.proximity[position]),
            "phrase_score": float(lexical_evidence.phrase[position]),
            "event_weight": event_weight,
            "event_rank": event_rank,
            "event_score": (
                float(event_scores[position]) if event_scores is not None else None
            ),
        }

    return _candidate_results(
        index,
        fused_scores,
        top=top,
        pov=pov,
        book=book,
        deduplicate_chapters=deduplicate_chapters,
        retrieval_details=details,
    )


def hierarchical_hybrid_search(
    index: LoadedIndex,
    query: str,
    encoder: Encoder,
    *,
    top: int = 10,
    pov: str | None = None,
    book: str | None = None,
    semantic_weight: float = 0.5,
    chapter_candidates: int = DEFAULT_CHAPTER_CANDIDATES,
    passages_per_chapter: int = DEFAULT_PASSAGES_PER_CHAPTER,
    passage_candidate_pool: int = DEFAULT_PASSAGE_CANDIDATE_POOL,
    chapter_evidence_passages: int = DEFAULT_CHAPTER_EVIDENCE_PASSAGES,
    chapter_weight: float = DEFAULT_CHAPTER_WEIGHT,
    neighbor_weight: float = DEFAULT_NEIGHBOR_WEIGHT,
    lexical_index: BM25Index | None = None,
    reranker: Reranker | None = None,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    rerank_context_words: int = 0,
    rerank_fusion_weight: float = 1.0,
    narrative: bool = False,
    late_interaction: bool = False,
    context_vector_weight: float = DEFAULT_CONTEXT_VECTOR_WEIGHT,
    scene_window_weight: float = DEFAULT_SCENE_WINDOW_WEIGHT,
    scene_lexical_weight: float = DEFAULT_SCENE_LEXICAL_WEIGHT,
    scene_lexical_index: BM25Index | None = None,
    lexical_evidence_weight: float = DEFAULT_LEXICAL_EVIDENCE_WEIGHT,
    retention_mode: str = DEFAULT_RETENTION_MODE,
    event_weight: float = DEFAULT_EVENT_WEIGHT,
    event_lexical_index: BM25Index | None = None,
    event_parser: Any | None = None,
) -> list[SearchResult]:
    return hierarchical_hybrid_search_run(
        index,
        query,
        encoder,
        top=top,
        pov=pov,
        book=book,
        semantic_weight=semantic_weight,
        chapter_candidates=chapter_candidates,
        passages_per_chapter=passages_per_chapter,
        passage_candidate_pool=passage_candidate_pool,
        chapter_evidence_passages=chapter_evidence_passages,
        chapter_weight=chapter_weight,
        neighbor_weight=neighbor_weight,
        lexical_index=lexical_index,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
        rerank_context_words=rerank_context_words,
        rerank_fusion_weight=rerank_fusion_weight,
        narrative=narrative,
        late_interaction=late_interaction,
        context_vector_weight=context_vector_weight,
        scene_window_weight=scene_window_weight,
        scene_lexical_weight=scene_lexical_weight,
        scene_lexical_index=scene_lexical_index,
        lexical_evidence_weight=lexical_evidence_weight,
        retention_mode=retention_mode,
        event_weight=event_weight,
        event_lexical_index=event_lexical_index,
        event_parser=event_parser,
    ).results


def hierarchical_hybrid_search_run(
    index: LoadedIndex,
    query: str,
    encoder: Encoder,
    *,
    top: int = 10,
    pov: str | None = None,
    book: str | None = None,
    semantic_weight: float = 0.5,
    chapter_candidates: int = DEFAULT_CHAPTER_CANDIDATES,
    passages_per_chapter: int = DEFAULT_PASSAGES_PER_CHAPTER,
    passage_candidate_pool: int = DEFAULT_PASSAGE_CANDIDATE_POOL,
    chapter_evidence_passages: int = DEFAULT_CHAPTER_EVIDENCE_PASSAGES,
    chapter_weight: float = DEFAULT_CHAPTER_WEIGHT,
    neighbor_weight: float = DEFAULT_NEIGHBOR_WEIGHT,
    lexical_index: BM25Index | None = None,
    reranker: Reranker | None = None,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    rerank_context_words: int = 0,
    rerank_fusion_weight: float = 1.0,
    narrative: bool = False,
    late_interaction: bool = False,
    context_vector_weight: float = DEFAULT_CONTEXT_VECTOR_WEIGHT,
    scene_window_weight: float = DEFAULT_SCENE_WINDOW_WEIGHT,
    scene_lexical_weight: float = DEFAULT_SCENE_LEXICAL_WEIGHT,
    scene_lexical_index: BM25Index | None = None,
    lexical_evidence_weight: float = DEFAULT_LEXICAL_EVIDENCE_WEIGHT,
    retention_mode: str = DEFAULT_RETENTION_MODE,
    event_weight: float = DEFAULT_EVENT_WEIGHT,
    event_lexical_index: BM25Index | None = None,
    event_parser: Any | None = None,
) -> SearchRun:
    query = _normalize_query(query)
    if narrative:
        query = expand_narrative_query(query)
    _validate_top(top)
    if not 0.0 <= semantic_weight <= 1.0:
        raise WeirwoodError("--semantic-weight must be between 0 and 1")
    if not 0.0 <= event_weight <= 1.0:
        raise WeirwoodError("--event-weight must be between 0 and 1")
    if not 0.0 <= lexical_evidence_weight <= 1.0:
        raise WeirwoodError("--lexical-evidence-weight must be between 0 and 1")
    if retention_mode not in RETENTION_MODES:
        raise WeirwoodError(
            f"--retention-mode must be one of: {', '.join(RETENTION_MODES)}"
        )
    if not 1 <= chapter_candidates <= 100:
        raise WeirwoodError("--chapter-candidates must be between 1 and 100")
    if not 1 <= passages_per_chapter <= 20:
        raise WeirwoodError("--passages-per-chapter must be between 1 and 20")
    if not chapter_candidates <= passage_candidate_pool <= 2000:
        raise WeirwoodError(
            "--passage-candidate-pool must be between --chapter-candidates and 2000"
        )
    if not 1 <= chapter_evidence_passages <= 10:
        raise WeirwoodError("--chapter-evidence-passages must be between 1 and 10")
    if chapter_weight < 0.0 or neighbor_weight < 0.0:
        raise WeirwoodError("hierarchical score weights must be non-negative")
    if reranker is not None and not 1 <= rerank_candidates <= 100:
        raise WeirwoodError("--rerank-candidates must be between 1 and 100")
    if not 0.0 <= rerank_fusion_weight <= 1.0:
        raise WeirwoodError("--rerank-fusion-weight must be between 0 and 1")
    if not 0 <= rerank_context_words <= 500:
        raise WeirwoodError("--rerank-context-words must be between 0 and 500")

    pov = _normalized_pov(index, pov)
    book = _normalized_book(index, book)
    eligible = _eligible_positions(index, pov, book)
    semantic_scores = _semantic_scores(
        index,
        query,
        encoder,
        narrative=narrative,
        late_interaction=late_interaction,
        context_vector_weight=context_vector_weight,
        scene_window_weight=scene_window_weight,
    )
    lexical_index = lexical_index or (
        _narrative_lexical_index(index)
        if narrative
        else BM25Index.from_chunks(index.chunks)
    )
    lexical_scores = _combined_lexical_scores(
        index,
        query,
        lexical_index,
        scene_lexical_weight=scene_lexical_weight,
        scene_lexical_index=scene_lexical_index,
    )
    semantic_order = _rank_positions(
        semantic_scores, eligible, limit=passage_candidate_pool
    )
    lexical_order = _rank_positions(
        lexical_scores,
        eligible,
        limit=passage_candidate_pool,
        positive_only=True,
    )
    semantic_ranks = {position: rank for rank, position in enumerate(semantic_order, 1)}
    lexical_ranks = {position: rank for rank, position in enumerate(lexical_order, 1)}
    lexical_evidence = _empty_lexical_evidence(len(index.chunks))
    lexical_evidence_ranks: dict[int, int] = {}
    if lexical_evidence_weight:
        lexical_evidence = lexical_index.evidence_scores(query, positions=eligible)
        lexical_evidence_order = _rank_positions(
            lexical_evidence.score,
            eligible,
            limit=passage_candidate_pool,
            positive_only=True,
        )
        lexical_evidence_ranks = {
            position: rank
            for rank, position in enumerate(lexical_evidence_order, 1)
        }
    event_scores = None
    event_ranks: dict[int, int] = {}
    if event_weight:
        event_scores = _event_scores_for_chunks(
            index,
            query,
            event_lexical_index=event_lexical_index,
            event_parser=event_parser,
        )
        event_order = _rank_positions(
            event_scores,
            eligible,
            limit=passage_candidate_pool,
            positive_only=True,
        )
        event_ranks = {
            position: rank for rank, position in enumerate(event_order, 1)
        }
    lexical_weight = 1.0 - semantic_weight

    global_fused: dict[int, float] = {}
    globally_ranked_positions = (
        semantic_ranks.keys()
        | lexical_ranks.keys()
        | lexical_evidence_ranks.keys()
        | event_ranks.keys()
    )
    for position in globally_ranked_positions:
        score = 0.0
        if position in semantic_ranks:
            score += semantic_weight / (RRF_CONSTANT + semantic_ranks[position])
        if position in lexical_ranks:
            score += lexical_weight / (RRF_CONSTANT + lexical_ranks[position])
        if position in lexical_evidence_ranks:
            score += lexical_evidence_weight / (
                RRF_CONSTANT + lexical_evidence_ranks[position]
            )
        if position in event_ranks:
            score += event_weight / (RRF_CONSTANT + event_ranks[position])
        global_fused[position] = score

    chapter_evidence: defaultdict[str, list[float]] = defaultdict(list)
    for position, score in global_fused.items():
        chapter_evidence[index.chunks[position].chapter_id].append(score)
    chapter_scores = {
        chapter_id: sum(sorted(scores, reverse=True)[:chapter_evidence_passages])
        for chapter_id, scores in chapter_evidence.items()
    }
    chapter_sort_keys = {
        chunk.chapter_id: (chunk.book_sequence, chunk.chapter_sequence)
        for chunk in index.chunks
    }
    chapter_order = tuple(
        sorted(
            chapter_scores,
            key=lambda chapter_id: (
                -chapter_scores[chapter_id],
                *chapter_sort_keys[chapter_id],
            ),
        )
    )
    shortlisted = set(chapter_order[:chapter_candidates])
    chapter_ranks = {
        chapter_id: rank for rank, chapter_id in enumerate(chapter_order, start=1)
    }
    passage_positions = [
        position
        for position in eligible
        if index.chunks[position].chapter_id in shortlisted
    ]
    local_semantic_order = _rank_positions(semantic_scores, passage_positions)
    local_lexical_order = _rank_positions(
        lexical_scores, passage_positions, positive_only=True
    )
    local_semantic_ranks = {
        position: rank for rank, position in enumerate(local_semantic_order, start=1)
    }
    local_lexical_ranks = {
        position: rank for rank, position in enumerate(local_lexical_order, start=1)
    }
    local_lexical_evidence_order = _rank_positions(
        lexical_evidence.score, passage_positions, positive_only=True
    )
    local_lexical_evidence_ranks = {
        position: rank
        for rank, position in enumerate(local_lexical_evidence_order, start=1)
    }
    local_event_ranks: dict[int, int] = {}
    if event_scores is not None:
        local_event_order = _rank_positions(
            event_scores, passage_positions, positive_only=True
        )
        local_event_ranks = {
            position: rank
            for rank, position in enumerate(local_event_order, start=1)
        }
    passage_scores: dict[int, float] = {}
    agreement_scores: dict[int, float] = {}
    for position in passage_positions:
        semantic_rank = local_semantic_ranks[position]
        lexical_rank = local_lexical_ranks.get(position)
        score = semantic_weight / (RRF_CONSTANT + semantic_rank)
        if lexical_rank is not None:
            score += lexical_weight / (RRF_CONSTANT + lexical_rank)
            agreement_scores[position] = 1.0 / (
                RRF_CONSTANT + max(semantic_rank, lexical_rank)
            )
        else:
            agreement_scores[position] = 0.0
        lexical_evidence_rank = local_lexical_evidence_ranks.get(position)
        if lexical_evidence_rank is not None:
            score += lexical_evidence_weight / (
                RRF_CONSTANT + lexical_evidence_rank
            )
        event_rank = local_event_ranks.get(position)
        if event_rank is not None:
            score += event_weight / (RRF_CONSTANT + event_rank)
        passage_scores[position] = score

    position_by_ordinal = {
        (chunk.chapter_id, chunk.chunk_ordinal): position
        for position, chunk in enumerate(index.chunks)
    }
    final_scores: dict[int, float] = {}
    neighbor_scores: dict[int, float] = {}
    for position in passage_positions:
        chunk = index.chunks[position]
        neighbors = [
            position_by_ordinal.get((chunk.chapter_id, chunk.chunk_ordinal - 1)),
            position_by_ordinal.get((chunk.chapter_id, chunk.chunk_ordinal + 1)),
        ]
        neighbor_score = max(
            (passage_scores.get(neighbor, 0.0) for neighbor in neighbors),
            default=0.0,
        )
        neighbor_scores[position] = neighbor_score
        chapter_score = 1.0 / (RRF_CONSTANT + chapter_ranks[chunk.chapter_id])
        final_scores[position] = (
            passage_scores[position]
            + chapter_weight * chapter_score
            + neighbor_weight * neighbor_score
            + 0.10 * agreement_scores[position]
        )

    positions_by_chapter: defaultdict[str, list[int]] = defaultdict(list)
    for position in passage_positions:
        positions_by_chapter[index.chunks[position].chapter_id].append(position)
    retained: list[int] = []
    within_chapter_ranks: dict[str, int] = {}
    for chapter_id in chapter_order[:chapter_candidates]:
        ordered = sorted(
            positions_by_chapter[chapter_id],
            key=lambda position: (-final_scores[position], position),
        )
        for rank, position in enumerate(ordered, start=1):
            within_chapter_ranks[index.chunks[position].id] = rank
        if retention_mode == "per-chapter":
            retained.extend(
                _select_non_overlapping_passages(
                    index,
                    ordered,
                    limit=passages_per_chapter,
                )
            )

    if retention_mode == "global":
        globally_ordered = sorted(
            passage_positions,
            key=lambda position: (-final_scores[position], position),
        )
        retained = _select_non_overlapping_passages(
            index,
            globally_ordered,
            limit=None,
        )

    ordered_retained = sorted(
        retained, key=lambda position: (-final_scores[position], position)
    )
    reranker_scores_by_position: dict[int, float] = {}
    reranker_fusion_scores: dict[int, float] = {}
    if reranker is not None:
        rerank_count = min(rerank_candidates, len(ordered_retained))
        rerank_positions = ordered_retained[:rerank_count]
        passages = _reranker_passages(
            index,
            [index.chunks[position] for position in rerank_positions],
            context_words=rerank_context_words,
        )
        scores = np.asarray(reranker.score(query, passages), dtype=np.float32)
        if scores.shape != (rerank_count,):
            raise WeirwoodError(
                f"reranker returned shape {scores.shape}; expected ({rerank_count},)"
            )
        if not np.isfinite(scores).all():
            raise WeirwoodError("reranker returned non-finite scores")
        reranker_scores_by_position = dict(
            zip(rerank_positions, (float(score) for score in scores), strict=True)
        )
        reranked_prefix = sorted(
            rerank_positions,
            key=lambda position: (
                -reranker_scores_by_position[position],
                -final_scores[position],
                position,
            ),
        )
        if rerank_fusion_weight < 1.0:
            baseline_ranks = {
                position: rank
                for rank, position in enumerate(rerank_positions, start=1)
            }
            reranker_ranks = {
                position: rank
                for rank, position in enumerate(reranked_prefix, start=1)
            }
            reranker_fusion_scores = {
                position: (
                    (1.0 - rerank_fusion_weight)
                    / (RRF_CONSTANT + baseline_ranks[position])
                    + rerank_fusion_weight
                    / (RRF_CONSTANT + reranker_ranks[position])
                )
                for position in rerank_positions
            }
            reranked_prefix = sorted(
                rerank_positions,
                key=lambda position: (
                    -reranker_fusion_scores[position],
                    -final_scores[position],
                    position,
                ),
            )
        ordered_retained = reranked_prefix + ordered_retained[rerank_count:]
    ordered_retained = ordered_retained[:top]
    chapter_word_cache: dict[str, list[str]] = {}
    results: list[SearchResult] = []
    for rank, position in enumerate(ordered_retained, start=1):
        chunk = index.chunks[position]
        context_start, context_end, context_text = _adjacent_context_window(
            index, chunk, position_by_ordinal, chapter_word_cache
        )
        results.append(
            SearchResult(
                rank=rank,
                score=reranker_fusion_scores.get(
                    position,
                    reranker_scores_by_position.get(position, final_scores[position]),
                ),
                chunk=chunk,
                retrieval={
                    "mode": (
                        "hierarchical-hybrid-rerank"
                        if position in reranker_scores_by_position
                        else "hierarchical-hybrid"
                    ),
                    "semantic_weight": semantic_weight,
                    "lexical_weight": lexical_weight,
                    "chapter_rank": chapter_ranks[chunk.chapter_id],
                    "chapter_score": chapter_scores[chunk.chapter_id],
                    "within_chapter_rank": within_chapter_ranks[chunk.id],
                    "semantic_rank": local_semantic_ranks[position],
                    "semantic_score": float(semantic_scores[position]),
                    "lexical_rank": local_lexical_ranks.get(position),
                    "lexical_score": float(lexical_scores[position]),
                    "lexical_evidence_weight": lexical_evidence_weight,
                    "lexical_evidence_rank": local_lexical_evidence_ranks.get(
                        position
                    ),
                    "lexical_evidence_score": float(
                        lexical_evidence.score[position]
                    ),
                    "term_coverage_score": float(
                        lexical_evidence.coverage[position]
                    ),
                    "term_proximity_score": float(
                        lexical_evidence.proximity[position]
                    ),
                    "phrase_score": float(lexical_evidence.phrase[position]),
                    "retention_mode": retention_mode,
                    "event_weight": event_weight,
                    "event_rank": local_event_ranks.get(position),
                    "event_score": (
                        float(event_scores[position])
                        if event_scores is not None
                        else None
                    ),
                    "passage_score": passage_scores[position],
                    "neighbor_score": neighbor_scores[position],
                    "agreement_score": agreement_scores[position],
                    "hierarchical_score": final_scores[position],
                    "reranker_score": reranker_scores_by_position.get(position),
                    "reranker_fusion_weight": rerank_fusion_weight,
                    "reranker_fusion_score": reranker_fusion_scores.get(position),
                },
                context_text=context_text,
                context_word_start=context_start,
                context_word_end=context_end,
            )
        )
    return SearchRun(
        results=results,
        trace=RetrievalTrace(
            chapter_ranking=chapter_order,
            candidate_chunk_ids=frozenset(
                index.chunks[position].id for position in passage_positions
            ),
            retained_chunk_ids=frozenset(index.chunks[position].id for position in retained),
            within_chapter_ranks=within_chapter_ranks,
        ),
    )


def _adjacent_context_window(
    index: LoadedIndex,
    chunk: Chunk,
    position_by_ordinal: dict[tuple[str, int], int],
    chapter_word_cache: dict[str, list[str]],
) -> tuple[int, int, str]:
    positions = [
        position_by_ordinal.get((chunk.chapter_id, ordinal))
        for ordinal in range(chunk.chunk_ordinal - 1, chunk.chunk_ordinal + 2)
    ]
    context_chunks = [index.chunks[position] for position in positions if position is not None]
    start = min(item.word_start for item in context_chunks)
    end = max(item.word_end for item in context_chunks)
    words = chapter_word_cache.get(chunk.chapter_id)
    if words is None:
        words = _reconstruct_chapter_words(index, chunk.chapter_id)
        chapter_word_cache[chunk.chapter_id] = words
    return start, end, " ".join(words[start:end])


def search_index(
    index: LoadedIndex,
    query: str,
    encoder: Encoder | None,
    *,
    mode: str = "semantic",
    top: int = 10,
    pov: str | None = None,
    book: str | None = None,
    semantic_weight: float = 0.5,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    deduplicate_chapters: bool = False,
    lexical_index: BM25Index | None = None,
    reranker: Reranker | None = None,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    rerank_context_words: int = 0,
    rerank_fusion_weight: float = 1.0,
    hierarchical: bool = False,
    chapter_candidates: int = DEFAULT_CHAPTER_CANDIDATES,
    passages_per_chapter: int = DEFAULT_PASSAGES_PER_CHAPTER,
    passage_candidate_pool: int = DEFAULT_PASSAGE_CANDIDATE_POOL,
    chapter_evidence_passages: int = DEFAULT_CHAPTER_EVIDENCE_PASSAGES,
    chapter_weight: float = DEFAULT_CHAPTER_WEIGHT,
    neighbor_weight: float = DEFAULT_NEIGHBOR_WEIGHT,
    narrative: bool = False,
    late_interaction: bool = False,
    context_vector_weight: float = DEFAULT_CONTEXT_VECTOR_WEIGHT,
    scene_window_weight: float = DEFAULT_SCENE_WINDOW_WEIGHT,
    scene_lexical_weight: float = DEFAULT_SCENE_LEXICAL_WEIGHT,
    scene_lexical_index: BM25Index | None = None,
    lexical_evidence_weight: float = DEFAULT_LEXICAL_EVIDENCE_WEIGHT,
    retention_mode: str = DEFAULT_RETENTION_MODE,
    event_weight: float = DEFAULT_EVENT_WEIGHT,
    event_lexical_index: BM25Index | None = None,
    event_parser: Any | None = None,
) -> list[SearchResult]:
    if mode not in RETRIEVAL_MODES:
        raise WeirwoodError(
            f"unknown retrieval mode {mode!r}; choose one of: {', '.join(RETRIEVAL_MODES)}"
        )
    if event_weight and mode != "hybrid":
        raise WeirwoodError("--event-weight currently requires --mode hybrid")
    if lexical_evidence_weight and mode != "hybrid":
        raise WeirwoodError(
            "--lexical-evidence-weight currently requires --mode hybrid"
        )
    if hierarchical:
        if mode != "hybrid":
            raise WeirwoodError("--hierarchical currently requires --mode hybrid")
        if deduplicate_chapters:
            raise WeirwoodError(
                "--hierarchical retains multiple passages and cannot deduplicate chapters"
            )
        if encoder is None:
            raise WeirwoodError("hierarchical retrieval requires an embedding encoder")
        return hierarchical_hybrid_search(
            index,
            query,
            encoder,
            top=top,
            pov=pov,
            book=book,
            semantic_weight=semantic_weight,
            chapter_candidates=chapter_candidates,
            passages_per_chapter=passages_per_chapter,
            passage_candidate_pool=passage_candidate_pool,
            chapter_evidence_passages=chapter_evidence_passages,
            chapter_weight=chapter_weight,
            neighbor_weight=neighbor_weight,
            lexical_index=lexical_index,
            reranker=reranker,
            rerank_candidates=rerank_candidates,
            rerank_context_words=rerank_context_words,
            rerank_fusion_weight=rerank_fusion_weight,
            narrative=narrative,
            late_interaction=late_interaction,
            context_vector_weight=context_vector_weight,
            scene_window_weight=scene_window_weight,
            scene_lexical_weight=scene_lexical_weight,
            scene_lexical_index=scene_lexical_index,
            lexical_evidence_weight=lexical_evidence_weight,
            retention_mode=retention_mode,
            event_weight=event_weight,
            event_lexical_index=event_lexical_index,
            event_parser=event_parser,
        )
    if mode == "lexical":
        if reranker is not None:
            raise WeirwoodError("reranking currently supports semantic mode only")
        return lexical_search(
            index,
            query,
            top=top,
            pov=pov,
            book=book,
            deduplicate_chapters=deduplicate_chapters,
            lexical_index=lexical_index,
            narrative=narrative,
            scene_lexical_weight=scene_lexical_weight,
            scene_lexical_index=scene_lexical_index,
        )
    if encoder is None:
        raise WeirwoodError(f"{mode} retrieval requires an embedding encoder")
    if mode == "semantic":
        return semantic_search(
            index,
            query,
            encoder,
            top=top,
            pov=pov,
            book=book,
            deduplicate_chapters=deduplicate_chapters,
            reranker=reranker,
            rerank_candidates=rerank_candidates,
            rerank_context_words=rerank_context_words,
            narrative=narrative,
            late_interaction=late_interaction,
            context_vector_weight=context_vector_weight,
            scene_window_weight=scene_window_weight,
        )
    if reranker is not None:
        raise WeirwoodError("reranking currently supports semantic mode only")
    return hybrid_search(
        index,
        query,
        encoder,
        top=top,
        pov=pov,
        book=book,
        semantic_weight=semantic_weight,
        candidate_pool=candidate_pool,
        deduplicate_chapters=deduplicate_chapters,
        lexical_index=lexical_index,
        narrative=narrative,
        late_interaction=late_interaction,
        context_vector_weight=context_vector_weight,
        scene_window_weight=scene_window_weight,
        scene_lexical_weight=scene_lexical_weight,
        scene_lexical_index=scene_lexical_index,
        lexical_evidence_weight=lexical_evidence_weight,
        event_weight=event_weight,
        event_lexical_index=event_lexical_index,
        event_parser=event_parser,
    )


def similar_chunks(
    index: LoadedIndex,
    chunk_id: str,
    *,
    top: int = 10,
    pov: str | None = None,
    book: str | None = None,
) -> list[SearchResult]:
    positions = {chunk.id: position for position, chunk in enumerate(index.chunks)}
    if chunk_id not in positions:
        raise WeirwoodError(f"chunk ID not found in index: {chunk_id}")
    position = positions[chunk_id]
    source = index.chunks[position]
    return _candidate_results(
        index,
        index.embeddings @ index.embeddings[position],
        top=top,
        pov=pov,
        book=book,
        excluded_ids={chunk_id},
        source_chunk=source,
    )
