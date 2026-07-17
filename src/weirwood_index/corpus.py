from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from weirwood_index.models import (
    Chapter,
    Corpus,
    CorpusSource,
    CorpusValidationError,
)

NORMALIZATION_VERSION = "asoiaf-ascii-v3"
AGOT_EXPECTED_POV_COUNTS = {
    "PROLOGUE": 1,
    "BRAN": 7,
    "CATELYN": 11,
    "DAENERYS": 10,
    "EDDARD": 15,
    "JON": 9,
    "TYRION": 9,
    "ARYA": 5,
    "SANSA": 6,
}
# Preserve the original public name for callers that only know about AGOT.
EXPECTED_POV_COUNTS = AGOT_EXPECTED_POV_COUNTS
ACOK_EXPECTED_POV_COUNTS = {
    "PROLOGUE": 1,
    "ARYA": 10,
    "SANSA": 8,
    "TYRION": 15,
    "BRAN": 7,
    "JON": 8,
    "CATELYN": 7,
    "DAVOS": 3,
    "THEON": 6,
    "DAENERYS": 5,
}
PAGE_MARKER = re.compile(r"^Page\s+\d+$")
CHAPTER_MARKER = re.compile(r"^CHAPTER\s+\d+$")
# Sixteen observed scanner headers are corrupt variants of "A GAME OF THRONES <page>".
OCR_PAGE_HEADER = re.compile(
    r"^A (?:GAME,?|GAML|CAME) OF (?:THRONES|THRONLS|THRONFS|TFIRONES|'FHRONES) \d+$"
)


@dataclass(frozen=True)
class BookSpec:
    id: str
    title: str
    sequence: int
    metadata: tuple[str, str, str]
    expected_pov_counts: dict[str, int]
    heading_corrections: dict[str, str]
    stop_heading: str | None = None

    @property
    def expected_chapters(self) -> int:
        return sum(self.expected_pov_counts.values())


BOOK_SPECS = {
    "A Game Of Thrones": BookSpec(
        id="agot",
        title="A Game of Thrones",
        sequence=1,
        metadata=(
            "A Game Of Thrones",
            "Book One of A Song of Ice and Fire",
            "By George R. R. Martin",
        ),
        expected_pov_counts=AGOT_EXPECTED_POV_COUNTS,
        heading_corrections={"DAFNERYS": "DAENERYS"},
    ),
    "A Clash of Kings": BookSpec(
        id="acok",
        title="A Clash of Kings",
        sequence=2,
        metadata=(
            "A Clash of Kings",
            "Book Two of A song of Ice and Fire",
            "By George R. R. Martin",
        ),
        expected_pov_counts=ACOK_EXPECTED_POV_COUNTS,
        heading_corrections={"CALTELYN": "CATELYN", "CATIELYN": "CATELYN"},
        stop_heading="APPENDIX",
    ),
}


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _roman(number: int) -> str:
    values = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    result: list[str] = []
    for value, numeral in values:
        while number >= value:
            result.append(numeral)
            number -= value
    return "".join(result)


def _chapter_title(pov: str, ordinal: int) -> str:
    return "PROLOGUE" if pov == "PROLOGUE" else f"{pov} {_roman(ordinal)}"


def _source_paths(source: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    values = (source,) if isinstance(source, (str, Path)) else tuple(source)
    if not values:
        raise CorpusValidationError("at least one source file is required")
    paths = tuple(Path(value).expanduser().resolve() for value in values)
    duplicates = [str(path) for path, count in Counter(paths).items() if count > 1]
    if duplicates:
        raise CorpusValidationError(f"duplicate source files: {', '.join(duplicates)}")
    return paths


def _combined_source_hash(sources: Sequence[CorpusSource]) -> str:
    if len(sources) == 1:
        return sources[0].sha256
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda item: item.book_sequence):
        digest.update(f"{source.book_id}:{source.sha256}\n".encode())
    return digest.hexdigest()


def parse_corpus(
    source: str | Path | Sequence[str | Path], *, validate: bool = True
) -> Corpus:
    parsed = [_parse_book(path, validate=validate) for path in _source_paths(source)]
    book_ids = [item[0].book_id for item in parsed]
    if len(set(book_ids)) != len(book_ids):
        raise CorpusValidationError(
            f"only one source per book is allowed; found: {', '.join(book_ids)}"
        )
    parsed.sort(key=lambda item: item[0].book_sequence)
    sources = tuple(item[0] for item in parsed)
    chapters = tuple(chapter for _, book_chapters, _ in parsed for chapter in book_chapters)
    counts: Counter[str] = Counter()
    for _, _, book_counts in parsed:
        counts.update(book_counts)
    return Corpus(
        source_path=sources[0].path,
        source_sha256=_combined_source_hash(sources),
        normalization_version=NORMALIZATION_VERSION,
        cleaning_counts=dict(sorted(counts.items())),
        chapters=chapters,
        sources=sources,
    )


def _parse_book(
    source_path: Path, *, validate: bool
) -> tuple[CorpusSource, tuple[Chapter, ...], dict[str, int]]:
    if not source_path.is_file():
        raise CorpusValidationError(f"source file does not exist: {source_path}")
    raw = source_path.read_bytes()
    try:
        decoded = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CorpusValidationError(
            f"source must be ASCII; invalid byte at offset {exc.start}"
        ) from exc
    lines = decoded.splitlines()
    if len(lines) < 4:
        raise CorpusValidationError("source is too short to contain the expected book")

    metadata = tuple(line.strip() for line in lines[:3])
    spec = BOOK_SPECS.get(metadata[0])
    if spec is None:
        raise CorpusValidationError(f"unsupported book title: {metadata[0]!r}")
    if validate and metadata != spec.metadata:
        raise CorpusValidationError(
            f"the title/author lines do not match the expected {spec.id} source"
        )

    counts = Counter[str]()
    counts["title_author_lines_removed"] = 3
    headings = set(spec.expected_pov_counts) | set(spec.heading_corrections)
    chapter_parts: list[tuple[str, list[str]]] = []
    active_pov: str | None = None
    active_paragraphs: list[list[str]] = []
    active_lines: list[str] = []

    def finish_paragraph() -> None:
        nonlocal active_lines
        if active_lines:
            active_paragraphs.append(active_lines)
            active_lines = []

    def finish_chapter() -> None:
        if active_pov is not None:
            finish_paragraph()
            chapter_parts.append(
                (active_pov, [" ".join(paragraph) for paragraph in active_paragraphs])
            )

    for original_line in lines[3:]:
        line = original_line.strip()
        if spec.stop_heading is not None and line == spec.stop_heading:
            finish_chapter()
            counts["appendix_sections_removed"] += 1
            break
        if not line:
            counts["blank_lines_removed"] += 1
            finish_paragraph()
            continue
        if PAGE_MARKER.fullmatch(line):
            counts["page_markers_removed"] += 1
            continue
        if CHAPTER_MARKER.fullmatch(line):
            counts["chapter_markers_removed"] += 1
            continue
        if OCR_PAGE_HEADER.fullmatch(line):
            counts["ocr_headers_removed"] += 1
            continue
        if line in headings:
            finish_chapter()
            corrected = spec.heading_corrections.get(line, line)
            if corrected != line:
                counts[f"{line.casefold()}_headings_corrected"] += 1
            active_pov = corrected
            active_paragraphs = []
            active_lines = []
            continue
        if active_pov is None:
            raise CorpusValidationError(
                f"found prose before the first chapter heading: {line[:60]!r}"
            )
        active_lines.append(line)
    else:
        finish_chapter()

    pov_ordinals: Counter[str] = Counter()
    chapters: list[Chapter] = []
    for sequence, (pov, prose_lines) in enumerate(chapter_parts, start=1):
        if not prose_lines:
            raise CorpusValidationError(
                f"{spec.id} chapter {sequence} ({pov}) contains no prose"
            )
        pov_ordinals[pov] += 1
        ordinal = pov_ordinals[pov]
        slug = "prologue" if pov == "PROLOGUE" else f"{pov.lower()}-{ordinal}"
        chapters.append(
            Chapter(
                id=f"{spec.id}-{sequence:03d}-{slug}",
                sequence=sequence,
                pov=pov,
                pov_ordinal=ordinal,
                title=_chapter_title(pov, ordinal),
                text="\n\n".join(
                    " ".join(paragraph.split()) for paragraph in prose_lines
                ),
                book_id=spec.id,
                book_title=spec.title,
                book_sequence=spec.sequence,
            )
        )

    if validate:
        errors: list[str] = []
        if len(chapters) != spec.expected_chapters:
            errors.append(
                f"expected {spec.expected_chapters} chapters for {spec.id}, "
                f"found {len(chapters)}"
            )
        actual_counts = Counter(chapter.pov for chapter in chapters)
        for pov, expected in spec.expected_pov_counts.items():
            actual = actual_counts.get(pov, 0)
            if actual != expected:
                errors.append(f"expected {expected} {pov} chapters, found {actual}")
        unexpected = sorted(set(actual_counts) - set(spec.expected_pov_counts))
        if unexpected:
            errors.append(f"unexpected chapter headings: {', '.join(unexpected)}")
        for original in spec.heading_corrections:
            key = f"{original.casefold()}_headings_corrected"
            if counts[key] != 1:
                errors.append(
                    f"expected exactly one {original} heading correction, "
                    f"found {counts[key]}"
                )
        if errors:
            raise CorpusValidationError("; ".join(errors))

    source = CorpusSource(
        book_id=spec.id,
        book_title=spec.title,
        book_sequence=spec.sequence,
        path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    return source, tuple(chapters), dict(sorted(counts.items()))


def corpus_summary(corpus: Corpus) -> dict[str, object]:
    sources = corpus.sources or (
        CorpusSource("agot", "A Game of Thrones", 1, corpus.source_path, corpus.source_sha256),
    )
    return {
        "source": str(corpus.source_path),
        "sources": [
            {
                "book_id": source.book_id,
                "book_title": source.book_title,
                "book_sequence": source.book_sequence,
                "path": str(source.path),
                "sha256": source.sha256,
            }
            for source in sources
        ],
        "source_sha256": corpus.source_sha256,
        "normalization_version": corpus.normalization_version,
        "cleaning_counts": corpus.cleaning_counts,
        "book_count": len(sources),
        "chapter_count": len(corpus.chapters),
        "chapter_counts_by_book": dict(
            sorted(Counter(chapter.book_id for chapter in corpus.chapters).items())
        ),
        "pov_counts": dict(
            sorted(Counter(chapter.pov for chapter in corpus.chapters).items())
        ),
        "word_count": sum(chapter.word_count for chapter in corpus.chapters),
    }
