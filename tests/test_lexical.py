from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weirwood_index.indexing import LoadedIndex
from weirwood_index.lexical import BM25Index, best_query_span, tokenize
from weirwood_index.models import Chunk, WeirwoodError
from weirwood_index.retrieval import (
    hierarchical_hybrid_search_run,
    hybrid_search,
    lexical_search,
    lexical_search_page,
    semantic_search,
)


class FixedEncoder:
    model_id = "test/fixed"
    revision = "test"
    max_tokens = 512

    def token_count(self, text: str) -> int:
        return len(text.split())

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        raise AssertionError("not used")

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in queries], dtype=np.float32)


def _chunk(
    chunk_id: str,
    chapter_id: str,
    chapter_sequence: int,
    text: str,
    ordinal: int = 1,
    *,
    book_id: str = "agot",
    book_title: str = "A Game of Thrones",
    book_sequence: int = 1,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        chapter_id=chapter_id,
        chapter_title=chapter_id.upper(),
        chapter_sequence=chapter_sequence,
        pov="TEST",
        pov_ordinal=chapter_sequence,
        chunk_ordinal=ordinal,
        word_start=(ordinal - 1) * 100,
        word_end=(ordinal - 1) * 100 + len(text.split()),
        text=text,
        book_id=book_id,
        book_title=book_title,
        book_sequence=book_sequence,
    )


@pytest.fixture
def retrieval_index() -> LoadedIndex:
    chunks = (
        _chunk("gold-1", "gold", 1, "Viserys receives a crown of molten gold"),
        _chunk("arya-1", "arya", 2, "Arya escapes the guards with courage"),
        _chunk("fire-1", "fire", 3, "Daenerys enters a funeral pyre of fire"),
        _chunk("gold-2", "gold", 1, "The golden king wears a crown", ordinal=2),
    )
    embeddings = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]], dtype=np.float32)
    return LoadedIndex(Path("index"), chunks, embeddings, {})


def test_tokenize_normalizes_case_curly_apostrophes_and_possessives() -> None:
    assert tokenize("The KING’S crown") == ("the", "king", "crown")


def test_best_query_span_prefers_the_contiguous_query_terms() -> None:
    text = "dance lessons came first, before he said Dance with me then."

    span = best_query_span(text, "dance with me then")

    assert span is not None
    assert text[span.start : span.end] == "Dance with me then"


def test_exact_phrase_matching_normalizes_case_and_punctuation() -> None:
    lexical = BM25Index.from_texts(
        [
            "Ser Waymar said, “Dance with me then.”",
            "dance first, then come with me",
        ]
    )

    assert lexical.exact_phrase_matches("DANCE WITH ME THEN").tolist() == [
        True,
        False,
    ]


def test_bm25_ranks_distinctive_terms_first(retrieval_index) -> None:
    lexical = BM25Index.from_chunks(retrieval_index.chunks)

    results = lexical_search(retrieval_index, "Viserys molten gold", lexical_index=lexical, top=3)

    assert results[0].chunk.id == "gold-1"
    assert results[0].retrieval["mode"] == "lexical"
    assert results[0].retrieval["lexical_rank"] == 1
    assert results[0].retrieval["lexical_score"] == results[0].score
    assert results[0].retrieval["exact_phrase_match"] is False


def test_lexical_search_prioritizes_exact_phrase_then_compact_coverage() -> None:
    chunks = (
        _chunk(
            "exact",
            "prologue",
            1,
            "Ser Waymar met him bravely. “Dance with me then.”",
        ),
        _chunk(
            "compact",
            "training",
            2,
            "Dance with me now and then leave quietly.",
        ),
        _chunk(
            "repeated",
            "lessons",
            3,
            "Dance dance dance dance while you watch me.",
        ),
        _chunk(
            "second-book",
            "feast",
            1,
            "Then dance again with swords and call for me to dance.",
            book_id="acok",
            book_title="A Clash of Kings",
            book_sequence=2,
        ),
    )
    index = LoadedIndex(
        Path("index"),
        chunks,
        np.asarray(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        {},
    )

    all_books = lexical_search(index, "dance with me then", top=4)
    agot_only = lexical_search(index, "dance with me then", top=3, book="agot")

    assert [result.chunk.id for result in all_books[:2]] == ["exact", "compact"]
    assert agot_only[0].chunk.id == "exact"
    assert all_books[0].retrieval["exact_phrase_match"] is True
    assert all_books[1].retrieval["exact_phrase_match"] is False


def test_exact_phrase_overrides_semantic_ranking_in_hybrid_modes() -> None:
    chunks = (
        _chunk(
            "exact",
            "prologue",
            1,
            "Ser Waymar met him bravely. “Dance with me then.”",
        ),
        _chunk("semantic-1", "first", 2, "Dance lessons continued all day."),
        _chunk("semantic-2", "second", 3, "Come with me after the dance."),
    )
    embeddings = np.asarray([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    index = LoadedIndex(Path("index"), chunks, embeddings, {})

    hybrid = hybrid_search(
        index,
        "dance with me then",
        FixedEncoder(),
        top=3,
        candidate_pool=3,
    )
    hierarchical = hierarchical_hybrid_search_run(
        index,
        "dance with me then",
        FixedEncoder(),
        top=1,
        chapter_candidates=1,
        passages_per_chapter=1,
        passage_candidate_pool=3,
    )

    assert hybrid[0].chunk.id == "exact"
    assert hybrid[0].retrieval["exact_phrase_match"] is True
    assert hierarchical.results[0].chunk.id == "exact"
    assert hierarchical.results[0].retrieval["exact_phrase_match"] is True


def test_lexical_search_page_returns_global_ranks_and_total(retrieval_index) -> None:
    first = lexical_search_page(
        retrieval_index,
        "gold crown king viserys molten",
        page=1,
        page_size=1,
    )
    second = lexical_search_page(
        retrieval_index,
        "gold crown king viserys molten",
        page=2,
        page_size=1,
    )

    assert first.total_results == second.total_results == 2
    assert first.book_counts == second.book_counts == {"agot": 2}
    assert first.results[0].rank == 1
    assert second.results[0].rank == 2
    assert first.results[0].chunk.id != second.results[0].chunk.id


def test_lexical_evidence_rewards_complete_compact_term_matches() -> None:
    lexical = BM25Index.from_texts(
        [
            "blue flower growing from the Wall",
            "blue banners hung above a distant flower garden near the Wall",
            "a blue wall",
        ]
    )

    evidence = lexical.evidence_scores("blue flower growing from the Wall")

    assert evidence.score[0] > evidence.score[1] > evidence.score[2]
    assert evidence.coverage[0] == pytest.approx(1.0)
    assert evidence.proximity[0] > evidence.proximity[1]
    assert evidence.phrase[0] == pytest.approx(2 / 3)


def test_hybrid_fusion_can_rescue_lexical_match_from_semantic_ranking(
    retrieval_index,
) -> None:
    results = hybrid_search(
        retrieval_index,
        "Viserys molten gold",
        FixedEncoder(),
        top=3,
        candidate_pool=4,
        semantic_weight=0.5,
    )

    assert results[0].chunk.id == "gold-1"
    assert results[0].retrieval["lexical_rank"] == 1
    assert results[0].retrieval["semantic_rank"] == 3


def test_chapter_deduplication_returns_only_one_chunk_per_chapter(
    retrieval_index,
) -> None:
    results = lexical_search(
        retrieval_index,
        "gold crown king",
        top=4,
        deduplicate_chapters=True,
    )

    assert len({result.chunk.chapter_id for result in results}) == len(results)


def test_hybrid_validates_weight_and_candidate_pool(retrieval_index) -> None:
    with pytest.raises(WeirwoodError, match="semantic-weight"):
        hybrid_search(
            retrieval_index,
            "query",
            FixedEncoder(),
            semantic_weight=1.5,
        )
    with pytest.raises(WeirwoodError, match="candidate-pool"):
        hybrid_search(
            retrieval_index,
            "query",
            FixedEncoder(),
            top=10,
            candidate_pool=5,
        )


def test_semantic_search_validates_context_vector_weight(retrieval_index) -> None:
    with pytest.raises(WeirwoodError, match="context-vector-weight"):
        semantic_search(
            retrieval_index,
            "query",
            FixedEncoder(),
            context_vector_weight=1.5,
        )
    with pytest.raises(WeirwoodError, match="no derived context vectors"):
        semantic_search(
            retrieval_index,
            "query",
            FixedEncoder(),
            context_vector_weight=0.5,
        )
    with pytest.raises(WeirwoodError, match="enrich-scenes"):
        semantic_search(
            retrieval_index,
            "query",
            FixedEncoder(),
            scene_window_weight=0.5,
        )


def test_context_vector_promotes_without_diluting_focus_scores(retrieval_index) -> None:
    contexts = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]], dtype=np.float32)
    index = LoadedIndex(
        Path("index"),
        retrieval_index.chunks,
        retrieval_index.embeddings,
        {},
        context_embeddings=contexts,
    )

    results = semantic_search(
        index,
        "query",
        FixedEncoder(),
        top=4,
        context_vector_weight=1.0,
    )

    assert [result.chunk.id for result in results[:2]] == ["gold-1", "arya-1"]
    assert results[1].score == pytest.approx(1.0)


def test_hierarchical_search_shortlists_chapters_and_returns_context() -> None:
    chunks = tuple(
        _chunk(
            f"{chapter}-{ordinal}",
            chapter,
            chapter_number,
            text,
            ordinal=ordinal,
        )
        for chapter, chapter_number, texts in (
            ("first", 1, ("semantic lead", "ordinary middle", "quiet ending")),
            ("second", 2, ("distant opening", "hidden needle", "closing scene")),
        )
        for ordinal, text in enumerate(texts, start=1)
    )
    # Make the synthetic chunks contiguous so context reconstruction is valid.
    chunks = tuple(
        Chunk(
            **{
                **chunk.to_dict(),
                "word_start": (chunk.chunk_ordinal - 1) * 2,
                "word_end": chunk.chunk_ordinal * 2,
            }
        )
        for chunk in chunks
    )
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.43589], [0.8, 0.6], [0.6, 0.8], [0.7, 0.71414], [0.0, 1.0]],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = LoadedIndex(Path("index"), chunks, embeddings, {})

    run = hierarchical_hybrid_search_run(
        index,
        "hidden needle",
        FixedEncoder(),
        top=4,
        semantic_weight=0.25,
        chapter_candidates=2,
        passages_per_chapter=2,
        passage_candidate_pool=6,
    )

    assert run.results[0].chunk.id == "second-2"
    assert run.results[0].retrieval["mode"] == "hierarchical-hybrid"
    assert run.results[0].context_word_start == 0
    assert run.results[0].context_word_end == 6
    assert len(run.trace.candidate_chunk_ids) == 6
    assert run.trace.within_chapter_ranks["second-2"] == 1

    class ReverseReranker:
        model_id = "test/reverse"
        revision = "test"

        def score(self, query: str, passages: list[str]) -> np.ndarray:
            del query
            return np.arange(len(passages), dtype=np.float32)

    reranked = hierarchical_hybrid_search_run(
        index,
        "hidden needle",
        FixedEncoder(),
        top=4,
        semantic_weight=0.25,
        chapter_candidates=2,
        passages_per_chapter=2,
        passage_candidate_pool=6,
        reranker=ReverseReranker(),
        rerank_candidates=4,
    )

    assert reranked.results[0].retrieval["mode"] == "hierarchical-hybrid-rerank"
    assert reranked.results[0].chunk.id == "second-2"
    assert reranked.results[0].retrieval["exact_phrase_match"] is True

    baseline_fused = hierarchical_hybrid_search_run(
        index,
        "hidden needle",
        FixedEncoder(),
        top=4,
        semantic_weight=0.25,
        chapter_candidates=2,
        passages_per_chapter=2,
        passage_candidate_pool=6,
        reranker=ReverseReranker(),
        rerank_candidates=4,
        rerank_fusion_weight=0.0,
    )

    assert [result.chunk.id for result in baseline_fused.results] == [
        result.chunk.id for result in run.results
    ]

    per_chapter = hierarchical_hybrid_search_run(
        index,
        "hidden needle",
        FixedEncoder(),
        top=4,
        semantic_weight=0.25,
        chapter_candidates=2,
        passages_per_chapter=1,
        passage_candidate_pool=6,
        retention_mode="per-chapter",
    )
    global_retention = hierarchical_hybrid_search_run(
        index,
        "hidden needle",
        FixedEncoder(),
        top=4,
        semantic_weight=0.25,
        chapter_candidates=2,
        passages_per_chapter=1,
        passage_candidate_pool=6,
        retention_mode="global",
    )

    assert len(per_chapter.trace.retained_chunk_ids) == 2
    assert len(global_retention.trace.retained_chunk_ids) == 6
