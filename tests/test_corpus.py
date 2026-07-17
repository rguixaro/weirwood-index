from __future__ import annotations

from collections import Counter

import pytest

from weirwood_index.corpus import EXPECTED_POV_COUNTS, parse_corpus
from weirwood_index.models import CorpusValidationError

from .helpers import write_valid_acok_source, write_valid_source


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
