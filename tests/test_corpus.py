from __future__ import annotations

from collections import Counter

import pytest

from weirwood_index.corpus import (
    ADWD_EXPECTED_POV_COUNTS,
    ADWD_HEADING_ALIASES,
    AFFC_EXPECTED_POV_COUNTS,
    AFFC_HEADING_ALIASES,
    ASOS_EXPECTED_POV_COUNTS,
    EXPECTED_POV_COUNTS,
    parse_corpus,
)
from weirwood_index.models import CorpusValidationError

from .helpers import (
    write_valid_acok_source,
    write_valid_epub_source,
    write_valid_source,
)


def test_parser_cleans_and_validates_expected_structure(tmp_path) -> None:
    source = write_valid_source(tmp_path)

    corpus = parse_corpus(source)

    assert len(corpus.chapters) == 73
    assert Counter(chapter.pov for chapter in corpus.chapters) == EXPECTED_POV_COUNTS
    assert corpus.chapters[0].id == "agot-001-prologue"
    assert corpus.chapters[1].id == "agot-002-bran-1"
    assert corpus.chapters[-1].id == "agot-073-daenerys-10"
    assert corpus.cleaning_counts["page_markers_removed"] == 1
    assert corpus.cleaning_counts["ocr_headers_removed"] == 1
    assert corpus.cleaning_counts["dafnerys_headings_corrected"] == 1
    assert "Page 1" not in corpus.chapters[0].text
    assert "THRONES 109" not in corpus.chapters[0].text
    assert "\n\n" in corpus.chapters[0].text


def test_parser_does_not_modify_raw_bytes(tmp_path) -> None:
    source = write_valid_source(tmp_path)
    before = source.read_bytes()

    parse_corpus(source)

    assert source.read_bytes() == before


def test_parser_rejects_wrong_chapter_count(tmp_path) -> None:
    source = write_valid_source(tmp_path)
    source.write_bytes(source.read_bytes().replace(b"BRAN \r\n", b"BRANN \r\n", 1))

    with pytest.raises(CorpusValidationError, match="expected 73 chapters"):
        parse_corpus(source)


def test_parser_rejects_non_ascii(tmp_path) -> None:
    source = write_valid_source(tmp_path)
    source.write_bytes(source.read_bytes() + b"\xff")

    with pytest.raises(CorpusValidationError, match="must be ASCII"):
        parse_corpus(source)


def test_parser_combines_supported_books_and_excludes_appendix(tmp_path) -> None:
    agot = write_valid_source(tmp_path)
    acok = write_valid_acok_source(tmp_path)

    corpus = parse_corpus([acok, agot])

    assert len(corpus.sources) == 2
    assert [source.book_id for source in corpus.sources] == ["agot", "acok"]
    assert len(corpus.chapters) == 143
    assert corpus.chapters[73].id == "acok-001-prologue"
    assert corpus.chapters[-1].id == "acok-070-bran-7"
    assert corpus.cleaning_counts["chapter_markers_removed"] == 69
    assert corpus.cleaning_counts["appendix_sections_removed"] == 1
    assert all("appendix material" not in chapter.text for chapter in corpus.chapters)


def test_parser_preserves_epub_paragraphs_and_stable_offsets(tmp_path) -> None:
    agot = write_valid_epub_source(tmp_path, book_id="agot")
    acok = write_valid_epub_source(tmp_path, book_id="acok")

    corpus = parse_corpus([acok, agot])

    assert corpus.normalization_version == "asoiaf-epub-paragraphs-v1"
    assert len(corpus.chapters) == 143
    assert len(corpus.paragraphs) == 1430
    assert all(source.source_format == "epub" for source in corpus.sources)
    first = corpus.paragraphs[0]
    second = corpus.paragraphs[1]
    assert first.id == "agot-001-prologue-p0001"
    assert first.word_start == 0
    assert first.word_end == second.word_start == 5
    assert corpus.chapters[0].text == "\n\n".join(
        paragraph.text
        for paragraph in corpus.paragraphs
        if paragraph.chapter_id == corpus.chapters[0].id
    )


def _synthetic_headings(
    counts: dict[str, int], aliases: dict[str, str]
) -> list[str]:
    headings = list(aliases)
    represented = Counter(aliases.values())
    for pov, count in counts.items():
        headings.extend([pov] * (count - represented[pov]))
    return headings


def test_parser_combines_all_five_main_novels_and_maps_alias_povs(tmp_path) -> None:
    sources = [
        write_valid_epub_source(tmp_path, book_id="agot"),
        write_valid_epub_source(tmp_path, book_id="acok"),
        write_valid_epub_source(
            tmp_path,
            book_id="asos",
            title_override="A Storm of Swords",
            headings_override=_synthetic_headings(ASOS_EXPECTED_POV_COUNTS, {}),
            required_marker="merrett frey",
        ),
        write_valid_epub_source(
            tmp_path,
            book_id="affc",
            title_override="A Feast for Crows",
            headings_override=_synthetic_headings(
                AFFC_EXPECTED_POV_COUNTS, AFFC_HEADING_ALIASES
            ),
            required_marker="glass candle",
        ),
        write_valid_epub_source(
            tmp_path,
            book_id="adwd",
            title_override="A Dance with Dragons",
            headings_override=_synthetic_headings(
                ADWD_EXPECTED_POV_COUNTS, ADWD_HEADING_ALIASES
            ),
            required_marker="dragons plant no trees",
        ),
    ]

    corpus = parse_corpus(reversed(sources))

    assert [source.book_id for source in corpus.sources] == [
        "agot",
        "acok",
        "asos",
        "affc",
        "adwd",
    ]
    assert len(corpus.chapters) == 344
    assert len(corpus.paragraphs) == 3440
    assert Counter(
        chapter.pov for chapter in corpus.chapters if chapter.book_id == "asos"
    ) == ASOS_EXPECTED_POV_COUNTS
    assert Counter(
        chapter.pov for chapter in corpus.chapters if chapter.book_id == "affc"
    ) == AFFC_EXPECTED_POV_COUNTS
    assert Counter(
        chapter.pov for chapter in corpus.chapters if chapter.book_id == "adwd"
    ) == ADWD_EXPECTED_POV_COUNTS

    prophet = next(chapter for chapter in corpus.chapters if chapter.title == "THE PROPHET")
    reek = next(chapter for chapter in corpus.chapters if chapter.title == "REEK")
    queens_hand = next(
        chapter for chapter in corpus.chapters if chapter.title == "THE QUEEN’S HAND"
    )
    assert (prophet.book_id, prophet.pov) == ("affc", "AERON")
    assert (reek.book_id, reek.pov) == ("adwd", "THEON")
    assert (queens_hand.book_id, queens_hand.pov) == ("adwd", "BARRISTAN")
    assert any(
        chapter.id.endswith("jon-connington-1") and chapter.pov == "JON CONNINGTON"
        for chapter in corpus.chapters
    )
    assert any(chapter.title == "EPILOGUE" for chapter in corpus.chapters)


def test_parser_rejects_epub_with_collapsed_chapter_paragraphs(tmp_path) -> None:
    source = write_valid_epub_source(
        tmp_path,
        book_id="agot",
        collapsed_paragraphs=True,
    )

    with pytest.raises(CorpusValidationError, match="use a better EPUB source"):
        parse_corpus(source)


def test_parser_rejects_structurally_valid_but_incomplete_epub(tmp_path) -> None:
    source = write_valid_epub_source(
        tmp_path,
        book_id="acok",
        include_required_content=False,
    )

    with pytest.raises(CorpusValidationError, match="EPUB content appears incomplete"):
        parse_corpus(source)


def test_parser_rejects_incomplete_new_novel_epub(tmp_path) -> None:
    source = write_valid_epub_source(
        tmp_path,
        book_id="affc",
        title_override="A Feast for Crows",
        headings_override=_synthetic_headings(
            AFFC_EXPECTED_POV_COUNTS, AFFC_HEADING_ALIASES
        ),
        required_marker="glass candle",
        include_required_content=False,
    )

    with pytest.raises(CorpusValidationError, match="EPUB content appears incomplete"):
        parse_corpus(source)
