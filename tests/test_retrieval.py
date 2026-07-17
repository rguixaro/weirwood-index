from __future__ import annotations

import pytest

from weirwood_index.chunking import PROFILES
from weirwood_index.indexing import build_index, load_index
from weirwood_index.models import WeirwoodError
from weirwood_index.retrieval import search_index, semantic_search, similar_chunks

from .helpers import FakeEncoder, FakeReranker, write_valid_source


@pytest.fixture
def local_index(tmp_path):
    source = write_valid_source(tmp_path, words_per_chapter=400)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    return load_index(built.path)


def test_search_returns_ranked_results_and_supports_pov_filter(local_index) -> None:
    results = semantic_search(local_index, "forgotten scene", FakeEncoder(), top=4, pov="BRAN")

    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert all(result.chunk.pov == "BRAN" for result in results)
    assert all(results[index].score >= results[index + 1].score for index in range(3))


def test_search_rejects_empty_query_and_invalid_filter(local_index) -> None:
    with pytest.raises(WeirwoodError, match="must not be empty"):
        semantic_search(local_index, "   ", FakeEncoder())
    with pytest.raises(WeirwoodError, match="unknown POV"):
        semantic_search(local_index, "scene", FakeEncoder(), pov="nobody")


def test_similar_excludes_source_and_adjacent_overlapping_chunks(local_index) -> None:
    source = local_index.chunks[1]

    results = similar_chunks(local_index, source.id, top=10)

    assert all(result.chunk.id != source.id for result in results)
    assert all(
        result.chunk.chapter_id != source.chapter_id
        or abs(result.chunk.chunk_ordinal - source.chunk_ordinal) > 1
        for result in results
    )


def test_semantic_reranker_reorders_only_the_candidate_prefix(local_index) -> None:
    results = semantic_search(
        local_index,
        "forgotten scene",
        FakeEncoder(),
        top=6,
        reranker=FakeReranker(),
        rerank_candidates=4,
    )

    assert [result.retrieval["semantic_rank"] for result in results[:4]] == [4, 3, 2, 1]
    assert all(result.retrieval["reranked"] for result in results[:4])
    assert [result.retrieval["semantic_rank"] for result in results[4:]] == [5, 6]
    assert all(not result.retrieval["reranked"] for result in results[4:])


def test_reranker_can_use_expanded_context_and_deduplicate_after_scoring(
    local_index,
) -> None:
    class RecordingReranker(FakeReranker):
        def __init__(self) -> None:
            self.passages: list[str] = []

        def score(self, query: str, passages: list[str]):
            self.passages = passages
            return super().score(query, passages)

    reranker = RecordingReranker()
    results = semantic_search(
        local_index,
        "forgotten scene",
        FakeEncoder(),
        top=10,
        deduplicate_chapters=True,
        reranker=reranker,
        rerank_candidates=20,
        rerank_context_words=320,
    )

    assert len({result.chunk.chapter_id for result in results}) == len(results) == 10
    assert reranker.passages
    assert all(len(passage.split()) >= 320 for passage in reranker.passages)


def test_narrative_and_sentence_late_interaction_require_and_use_view_index(
    tmp_path,
    local_index,
) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=220)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
        narrative_views=True,
    )
    narrative_index = load_index(built.path)

    results = search_index(
        narrative_index,
        "a ranger said a remembered phrase",
        FakeEncoder(),
        mode="hybrid",
        top=5,
        narrative=True,
        late_interaction=True,
    )

    assert len(results) == 5
    assert [result.rank for result in results] == [1, 2, 3, 4, 5]

    with pytest.raises(WeirwoodError, match="built with --narrative-views"):
        semantic_search(local_index, "scene", FakeEncoder(), narrative=True)
