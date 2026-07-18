from __future__ import annotations

import pytest

from weirwood_api.schemas import SearchRequest
from weirwood_api.service import (
    CONTEXT_WORDS,
    CatalogBook,
    SearchCatalog,
    lexical_preview,
    neighboring_context,
    paragraph_context,
    validate_search_filters,
)
from weirwood_index.models import Chunk, Paragraph, SearchValidationError


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


def test_lexical_preview_centers_a_match_beyond_the_default_excerpt() -> None:
    before = [f"before{position}" for position in range(55)]
    quote = ["Dance", "with", "me", "then."]
    after = [f"after{position}" for position in range(40)]
    text = " ".join(before + quote + after)
    data = chunk(1, 0, len(before + quote + after)).to_dict()
    data["text"] = text
    target = Chunk.from_dict(data)
    assert text.casefold().index("dance with me then") > 360

    preview = lexical_preview(
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
