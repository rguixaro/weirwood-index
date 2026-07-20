from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weirwood_api.config import ApiSettings
from weirwood_api.schemas import SearchRequest
from weirwood_api.service import (
    CONTEXT_WORDS,
    CatalogBook,
    SearchCatalog,
    WeirwoodSearchRuntime,
    catalog_from_chunks,
    neighboring_context,
    paragraph_context,
    passage_preview,
    validate_search_filters,
)
from weirwood_index.indexing import LoadedIndex
from weirwood_index.lexical import BM25Index
from weirwood_index.models import Chunk, Paragraph, SearchValidationError
from weirwood_index.retrieval import SearchResult


def chunk(ordinal: int, start: int, end: int) -> Chunk:
    return Chunk(
        id=f"agot-001-prologue-c{ordinal:03d}",
        chapter_id="agot-001-prologue",
        chapter_title="PROLOGUE",
        chapter_sequence=1,
        pov="PROLOGUE",
        pov_ordinal=1,
        chunk_ordinal=ordinal,
        word_start=start,
        word_end=end,
        text=" ".join(f"word{position}" for position in range(start, end)),
    )


def test_catalog_groups_povs_by_book_and_sequence() -> None:
    base = chunk(1, 0, 10).to_dict()
    chunks = tuple(
        Chunk.from_dict({**base, **overrides})
        for overrides in (
            {
                "id": "adwd-001-reek-c001",
                "book_id": "adwd",
                "book_title": "A Dance with Dragons",
                "book_sequence": 5,
                "pov": "THEON",
            },
            {
                "id": "affc-001-prophet-c001",
                "book_id": "affc",
                "book_title": "A Feast for Crows",
                "book_sequence": 4,
                "pov": "AERON",
            },
            {
                "id": "adwd-002-queens-hand-c001",
                "book_id": "adwd",
                "book_title": "A Dance with Dragons",
                "book_sequence": 5,
                "pov": "BARRISTAN",
            },
        )
    )

    catalog = catalog_from_chunks(chunks)

    assert [book.book_id for book in catalog.books] == ["affc", "adwd"]
    assert catalog.books[1].povs == ("BARRISTAN", "THEON")


def test_search_filters_are_validated_against_the_active_catalog() -> None:
    catalog = SearchCatalog(
        books=(
            CatalogBook("agot", "A Game of Thrones", 1, ("ARYA", "EDDARD")),
            CatalogBook("acok", "A Clash of Kings", 2, ("ARYA", "TYRION")),
        )
    )

    validate_search_filters(
        catalog,
        SearchRequest(query="blue flower", book="agot", povs=["arya", "eddard"]),
    )
    with pytest.raises(SearchValidationError, match="unknown book 'missing'"):
        validate_search_filters(
            catalog,
            SearchRequest(query="blue flower", book="missing"),
        )
    with pytest.raises(SearchValidationError, match="unknown POV 'TYRION'"):
        validate_search_filters(
            catalog,
            SearchRequest(query="blue flower", book="agot", pov="tyrion"),
        )


def test_neighboring_context_excludes_overlapping_chunk_words() -> None:
    previous = chunk(1, 0, 180)
    focus = chunk(2, 135, 315)
    following = chunk(3, 270, 450)
    locations = {
        (item.chapter_id, item.chunk_ordinal): item
        for item in (previous, focus, following)
    }

    before, after = neighboring_context(focus, locations)

    assert before == " ".join(
        f"word{position}" for position in range(135 - CONTEXT_WORDS, 135)
    )
    assert after == " ".join(
        f"word{position}" for position in range(315, 315 + CONTEXT_WORDS)
    )


def test_neighboring_context_stops_at_chapter_boundaries() -> None:
    focus = chunk(1, 0, 180)
    other_chapter_data = chunk(2, 135, 315).to_dict()
    other_chapter_data.update(
        id="agot-002-bran-c002",
        chapter_id="agot-002-bran",
    )
    other_chapter = Chunk.from_dict(other_chapter_data)

    before, after = neighboring_context(
        focus,
        {
            (focus.chapter_id, focus.chunk_ordinal): focus,
            (other_chapter.chapter_id, other_chapter.chunk_ordinal): other_chapter,
        },
    )

    assert before is None
    assert after is None


def test_paragraph_context_preserves_boundaries_and_splits_regions() -> None:
    focus = chunk(1, 4, 9)
    paragraphs = (
        Paragraph(
            id="agot-001-prologue-p0001",
            chapter_id=focus.chapter_id,
            ordinal=1,
            word_start=0,
            word_end=5,
            text="word0 word1 word2 word3 word4",
        ),
        Paragraph(
            id="agot-001-prologue-p0002",
            chapter_id=focus.chapter_id,
            ordinal=2,
            word_start=5,
            word_end=10,
            text="word5 word6 word7 word8 word9",
        ),
    )

    rendered = paragraph_context(focus, paragraphs, excerpt_chars=11)

    assert [paragraph["id"] for paragraph in rendered] == [
        "agot-001-prologue-p0001",
        "agot-001-prologue-p0002",
    ]
    assert rendered[0]["fragments"] == [
        {"region": "before", "text": "word0 word1 word2 word3"},
        {"region": "focus", "text": "word4"},
    ]
    assert rendered[1]["fragments"] == [
        {"region": "focus", "text": "word5"},
        {"region": "after", "text": "word6 word7 word8 word9"},
    ]


def test_passage_preview_centers_an_exact_match_beyond_the_default_excerpt() -> None:
    before = [f"before{position}" for position in range(55)]
    quote = ["Dance", "with", "me", "then."]
    after = [f"after{position}" for position in range(40)]
    text = " ".join(before + quote + after)
    data = chunk(1, 0, len(before + quote + after)).to_dict()
    data["text"] = text
    target = Chunk.from_dict(data)
    assert text.casefold().index("dance with me then") > 360

    preview = passage_preview(
        target,
        "dance with me then",
        excerpt_chars=160,
    )

    assert preview is not None
    assert "Dance with me then." in preview.text
    assert len(preview.text) <= 160
    assert preview.word_start > target.word_start
    assert preview.before is not None
    assert preview.after is not None


def test_passage_preview_centers_scattered_meaningful_terms() -> None:
    before = [f"before{position}" for position in range(55)]
    relevant = [
        "You",
        "have",
        "danced",
        "the",
        "dance",
        "before",
        "my",
        "friend",
        "and",
        "he",
        "moved",
        "lightly",
        "on",
        "his",
        "feet",
        "as",
        "a",
        "water",
        "dancer",
        "might",
    ]
    after = [f"after{position}" for position in range(30)]
    text = " ".join(before + relevant + after)
    data = chunk(1, 0, len(before + relevant + after)).to_dict()
    data["text"] = text
    target = Chunk.from_dict(data)

    preview = passage_preview(target, "water dance", excerpt_chars=180)

    assert preview is not None
    assert "water" in preview.text.casefold()
    assert "dance" in preview.text.casefold().split()
    assert len(preview.text) <= 180
    assert preview.word_start > target.word_start


def test_passage_preview_centers_the_chunk_without_a_literal_match() -> None:
    words = [f"word{position}" for position in range(101)]
    data = chunk(1, 0, len(words)).to_dict()
    data["text"] = " ".join(words)
    target = Chunk.from_dict(data)

    preview = passage_preview(target, "forgotten promise", excerpt_chars=80)

    assert preview is not None
    preview_positions = [int(word.removeprefix("word")) for word in preview.text.split()]
    assert 35 < preview_positions[0] < 65
    assert 35 < preview_positions[-1] < 70
    assert len(preview.text) <= 80
    assert preview.before is not None
    assert preview.after is not None


def test_hybrid_payload_uses_the_match_centered_preview(monkeypatch) -> None:
    before = [f"before{position}" for position in range(55)]
    relevant = [
        "You",
        "have",
        "danced",
        "the",
        "dance",
        "before",
        "moving",
        "as",
        "a",
        "water",
        "dancer",
        "might",
    ]
    after = [f"after{position}" for position in range(30)]
    text = " ".join(before + relevant + after)
    data = chunk(1, 0, len(before + relevant + after)).to_dict()
    data["text"] = text
    target = Chunk.from_dict(data)
    index = LoadedIndex(
        Path("index"),
        (target,),
        np.zeros((1, 2), dtype=np.float32),
        {},
    )
    runtime = WeirwoodSearchRuntime(
        ApiSettings(index_path=Path("index"), origin_token=None, excerpt_chars=180)
    )
    runtime._index = index
    runtime._encoder = object()
    runtime._lexical_index = BM25Index.from_chunks(index.chunks)
    runtime._catalog = catalog_from_chunks(index.chunks)

    monkeypatch.setattr(
        "weirwood_api.service.search_index",
        lambda *args, **kwargs: [SearchResult(1, 1.0, target)],
    )

    payloads, pagination, _book_counts = runtime._search_sync(
        SearchRequest(query="water dance", mode="hybrid", top=1)
    )

    assert pagination is None
    assert "water" in payloads[0]["excerpt"].casefold()
    assert "dance" in payloads[0]["excerpt"].casefold().split()
    assert "water" not in (payloads[0]["context_after"] or "").casefold()


def test_paragraph_context_accepts_a_centered_lexical_focus() -> None:
    focus = chunk(1, 0, 10)
    paragraphs = (
        Paragraph(
            id="agot-001-prologue-p0001",
            chapter_id=focus.chapter_id,
            ordinal=1,
            word_start=0,
            word_end=5,
            text="word0 word1 word2 word3 word4",
        ),
        Paragraph(
            id="agot-001-prologue-p0002",
            chapter_id=focus.chapter_id,
            ordinal=2,
            word_start=5,
            word_end=10,
            text="word5 word6 word7 word8 word9",
        ),
    )

    rendered = paragraph_context(
        focus,
        paragraphs,
        excerpt_chars=100,
        focus_word_start=5,
        focus_word_end=7,
    )

    assert rendered[0]["fragments"] == [
        {"region": "before", "text": "word0 word1 word2 word3 word4"}
    ]
    assert rendered[1]["fragments"] == [
        {"region": "focus", "text": "word5 word6"},
        {"region": "after", "text": "word7 word8 word9"},
    ]
