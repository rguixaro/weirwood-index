from __future__ import annotations

import hashlib
import posixpath
import re
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from weirwood_index.models import (
    Chapter,
    Corpus,
    CorpusSource,
    CorpusValidationError,
    Paragraph,
)

TEXT_NORMALIZATION_VERSION = "asoiaf-ascii-v3"
EPUB_NORMALIZATION_VERSION = "asoiaf-epub-paragraphs-v1"
MIXED_NORMALIZATION_VERSION = "asoiaf-mixed-sources-v1"
NORMALIZATION_VERSION = TEXT_NORMALIZATION_VERSION
MIN_EPUB_PARAGRAPHS_PER_CHAPTER = 10
MAX_EPUB_PARAGRAPH_WORDS = 1000
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
ASOS_EXPECTED_POV_COUNTS = {
    "PROLOGUE": 1,
    "JAIME": 9,
    "CATELYN": 7,
    "ARYA": 13,
    "TYRION": 11,
    "DAVOS": 6,
    "SANSA": 7,
    "JON": 12,
    "DAENERYS": 6,
    "BRAN": 4,
    "SAMWELL": 5,
    "EPILOGUE": 1,
}
AFFC_EXPECTED_POV_COUNTS = {
    "PROLOGUE": 1,
    "AERON": 2,
    "AREO": 1,
    "CERSEI": 10,
    "BRIENNE": 8,
    "SAMWELL": 5,
    "ARYA": 3,
    "JAIME": 7,
    "SANSA": 3,
    "ASHA": 1,
    "ARYS": 1,
    "VICTARION": 2,
    "ARIANNE": 2,
}
AFFC_HEADING_ALIASES = {
    "THE PROPHET": "AERON",
    "THE CAPTAIN OF GUARDS": "AREO",
    "THE KRAKEN’S DAUGHTER": "ASHA",
    "THE SOILED KNIGHT": "ARYS",
    "THE IRON CAPTAIN": "VICTARION",
    "THE DROWNED MAN": "AERON",
    "THE QUEENMAKER": "ARIANNE",
    "ALAYNE": "SANSA",
    "THE REAVER": "VICTARION",
    "CAT OF THE CANALS": "ARYA",
    "THE PRINCESS IN THE TOWER": "ARIANNE",
}
ADWD_EXPECTED_POV_COUNTS = {
    "PROLOGUE": 1,
    "TYRION": 12,
    "DAENERYS": 10,
    "JON": 13,
    "BRAN": 3,
    "QUENTYN": 4,
    "DAVOS": 4,
    "THEON": 7,
    "JON CONNINGTON": 2,
    "ASHA": 3,
    "MELISANDRE": 1,
    "AREO": 1,
    "ARYA": 2,
    "JAIME": 1,
    "CERSEI": 2,
    "BARRISTAN": 4,
    "VICTARION": 2,
    "EPILOGUE": 1,
}
ADWD_HEADING_ALIASES = {
    "THE MERCHANT’S MAN": "QUENTYN",
    "REEK": "THEON",
    "THE LOST LORD": "JON CONNINGTON",
    "THE WINDBLOWN": "QUENTYN",
    "THE WAYWARD BRIDE": "ASHA",
    "THE PRINCE OF WINTERFELL": "THEON",
    "THE WATCHER": "AREO",
    "THE TURNCLOAK": "THEON",
    "THE KING’S PRIZE": "ASHA",
    "THE BLIND GIRL": "ARYA",
    "A GHOST IN WINTERFELL": "THEON",
    "THE QUEENSGUARD": "BARRISTAN",
    "THE IRON SUITOR": "VICTARION",
    "THE DISCARDED KNIGHT": "BARRISTAN",
    "THE SPURNED SUITOR": "QUENTYN",
    "THE GRIFFIN REBORN": "JON CONNINGTON",
    "THE SACRIFICE": "ASHA",
    "THE UGLY LITTLE GIRL": "ARYA",
    "THE KINGBREAKER": "BARRISTAN",
    "THE DRAGONTAMER": "QUENTYN",
    "THE QUEEN’S HAND": "BARRISTAN",
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
    epub_title_term: str = ""
    epub_content_markers: tuple[str, ...] = ()
    heading_aliases: dict[str, str] = field(default_factory=dict)

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
        epub_title_term="game of thrones",
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
        epub_title_term="clash of kings",
        epub_content_markers=(
            "a blue flower grew from a chink in a wall of ice",
        ),
    ),
    "A Storm Of Swords": BookSpec(
        id="asos",
        title="A Storm of Swords",
        sequence=3,
        metadata=(
            "A Storm Of Swords",
            "Book Three of A Song of Ice and Fire",
            "By George R.R. Martin",
        ),
        expected_pov_counts=ASOS_EXPECTED_POV_COUNTS,
        heading_corrections={},
        stop_heading="APPENDIX",
        epub_title_term="storm of swords",
        epub_content_markers=("merrett frey",),
    ),
    "A Feast for Crows": BookSpec(
        id="affc",
        title="A Feast for Crows",
        sequence=4,
        metadata=(
            "A Feast for Crows",
            "Book Four of A song of Ice and Fire",
            "By George R. R. Martin",
        ),
        expected_pov_counts=AFFC_EXPECTED_POV_COUNTS,
        heading_corrections={},
        stop_heading="APPENDIX",
        epub_title_term="feast for crows",
        epub_content_markers=("glass candle",),
        heading_aliases=AFFC_HEADING_ALIASES,
    ),
    "A Dance with Dragons": BookSpec(
        id="adwd",
        title="A Dance with Dragons",
        sequence=5,
        metadata=(
            "A Dance with Dragons",
            "Book Five of A song of Ice and Fire",
            "By George R. R. Martin",
        ),
        expected_pov_counts=ADWD_EXPECTED_POV_COUNTS,
        heading_corrections={},
        stop_heading="APPENDIX",
        epub_title_term="dance with dragons",
        epub_content_markers=("dragons plant no trees",),
        heading_aliases=ADWD_HEADING_ALIASES,
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
    return pov if pov in {"PROLOGUE", "EPILOGUE"} else f"{pov} {_roman(ordinal)}"


def _paragraph_records(
    chapter: Chapter, paragraph_texts: Sequence[str]
) -> tuple[Paragraph, ...]:
    records: list[Paragraph] = []
    word_start = 0
    for ordinal, original in enumerate(paragraph_texts, start=1):
        text = " ".join(original.split())
        if not text:
            continue
        word_end = word_start + len(text.split())
        records.append(
            Paragraph(
                id=f"{chapter.id}-p{ordinal:04d}",
                chapter_id=chapter.id,
                ordinal=ordinal,
                word_start=word_start,
                word_end=word_end,
                text=text,
            )
        )
        word_start = word_end
    return tuple(records)


def _build_chapter(
    spec: BookSpec,
    *,
    sequence: int,
    pov: str,
    pov_ordinal: int,
    paragraphs: Sequence[str],
    title: str | None = None,
) -> tuple[Chapter, tuple[Paragraph, ...]]:
    pov_slug = re.sub(r"[^a-z0-9]+", "-", pov.casefold()).strip("-")
    slug = "prologue" if pov == "PROLOGUE" else f"{pov_slug}-{pov_ordinal}"
    chapter = Chapter(
        id=f"{spec.id}-{sequence:03d}-{slug}",
        sequence=sequence,
        pov=pov,
        pov_ordinal=pov_ordinal,
        title=title or _chapter_title(pov, pov_ordinal),
        text="\n\n".join(paragraphs),
        book_id=spec.id,
        book_title=spec.title,
        book_sequence=spec.sequence,
    )
    return chapter, _paragraph_records(chapter, paragraphs)


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
    chapters = tuple(
        chapter for _, book_chapters, _, _, _ in parsed for chapter in book_chapters
    )
    paragraphs = tuple(
        paragraph
        for _, _, book_paragraphs, _, _ in parsed
        for paragraph in book_paragraphs
    )
    counts: Counter[str] = Counter()
    for _, _, _, book_counts, _ in parsed:
        counts.update(book_counts)
    versions = {version for _, _, _, _, version in parsed}
    normalization_version = (
        versions.pop() if len(versions) == 1 else MIXED_NORMALIZATION_VERSION
    )
    return Corpus(
        source_path=sources[0].path,
        source_sha256=_combined_source_hash(sources),
        normalization_version=normalization_version,
        cleaning_counts=dict(sorted(counts.items())),
        chapters=chapters,
        sources=sources,
        paragraphs=paragraphs,
    )


def _parse_book(
    source_path: Path, *, validate: bool
) -> tuple[
    CorpusSource,
    tuple[Chapter, ...],
    tuple[Paragraph, ...],
    dict[str, int],
    str,
]:
    if source_path.suffix.casefold() == ".epub":
        return _parse_epub_book(source_path, validate=validate)
    return _parse_text_book(source_path, validate=validate)


def _parse_text_book(
    source_path: Path, *, validate: bool
) -> tuple[
    CorpusSource,
    tuple[Chapter, ...],
    tuple[Paragraph, ...],
    dict[str, int],
    str,
]:
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
    headings = (
        set(spec.expected_pov_counts)
        | set(spec.heading_corrections)
        | set(spec.heading_aliases)
    )
    chapter_parts: list[tuple[str, str | None, list[str]]] = []
    active_pov: str | None = None
    active_title: str | None = None
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
                (
                    active_pov,
                    active_title,
                    [" ".join(paragraph) for paragraph in active_paragraphs],
                )
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
            active_pov = spec.heading_aliases.get(corrected, corrected)
            active_title = corrected if corrected in spec.heading_aliases else None
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
    paragraphs: list[Paragraph] = []
    for sequence, (pov, title, prose_lines) in enumerate(chapter_parts, start=1):
        if not prose_lines:
            raise CorpusValidationError(
                f"{spec.id} chapter {sequence} ({pov}) contains no prose"
            )
        pov_ordinals[pov] += 1
        ordinal = pov_ordinals[pov]
        chapter, chapter_paragraphs = _build_chapter(
            spec,
            sequence=sequence,
            pov=pov,
            pov_ordinal=ordinal,
            paragraphs=[" ".join(paragraph.split()) for paragraph in prose_lines],
            title=title,
        )
        chapters.append(chapter)
        paragraphs.extend(chapter_paragraphs)

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
        source_format="txt",
    )
    return (
        source,
        tuple(chapters),
        tuple(paragraphs),
        dict(sorted(counts.items())),
        TEXT_NORMALIZATION_VERSION,
    )


def _xml_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _normalized_epub_text(element: ElementTree.Element) -> str:
    text = unicodedata.normalize("NFC", "".join(element.itertext()))
    return " ".join(text.replace("\u00ad", "").split())


def _epub_book_spec(package: ElementTree.Element) -> BookSpec:
    title = next(
        (
            _normalized_epub_text(element)
            for element in package.iter()
            if _xml_name(element.tag) == "title" and _normalized_epub_text(element)
        ),
        "",
    )
    matches = [
        spec
        for spec in BOOK_SPECS.values()
        if spec.epub_title_term and spec.epub_title_term in title.casefold()
    ]
    if len(matches) != 1:
        raise CorpusValidationError(f"unsupported EPUB title: {title!r}")
    return matches[0]


def _parse_epub_book(
    source_path: Path, *, validate: bool
) -> tuple[
    CorpusSource,
    tuple[Chapter, ...],
    tuple[Paragraph, ...],
    dict[str, int],
    str,
]:
    if not source_path.is_file():
        raise CorpusValidationError(f"source file does not exist: {source_path}")
    raw_hash = source_sha256(source_path)
    try:
        archive = zipfile.ZipFile(source_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise CorpusValidationError(f"cannot open EPUB: {exc}") from exc

    counts = Counter[str]()
    try:
        try:
            container = ElementTree.fromstring(
                archive.read("META-INF/container.xml")
            )
            package_path = next(
                element.attrib["full-path"]
                for element in container.iter()
                if _xml_name(element.tag) == "rootfile"
                and "full-path" in element.attrib
            )
            package = ElementTree.fromstring(archive.read(package_path))
        except (KeyError, StopIteration, ElementTree.ParseError) as exc:
            raise CorpusValidationError(
                "EPUB container or package metadata is invalid"
            ) from exc

        spec = _epub_book_spec(package)
        package_directory = posixpath.dirname(package_path)
        manifest = {
            element.attrib["id"]: (
                element.attrib.get("href", ""),
                element.attrib.get("media-type", ""),
            )
            for element in package.iter()
            if _xml_name(element.tag) == "item" and "id" in element.attrib
        }
        spine = [
            element.attrib["idref"]
            for element in package.iter()
            if _xml_name(element.tag) == "itemref" and "idref" in element.attrib
        ]
        if not spine:
            raise CorpusValidationError("EPUB package contains no reading-order spine")

        headings = set(spec.expected_pov_counts) | set(spec.heading_aliases)
        chapter_parts: list[tuple[str, str, list[str]]] = []
        for item_id in spine:
            href, media_type = manifest.get(item_id, ("", ""))
            if "html" not in media_type:
                continue
            document_path = posixpath.normpath(
                posixpath.join(package_directory, href.split("#", maxsplit=1)[0])
            )
            try:
                document = ElementTree.fromstring(archive.read(document_path))
            except (KeyError, ElementTree.ParseError) as exc:
                raise CorpusValidationError(
                    f"cannot parse EPUB spine document {document_path!r}"
                ) from exc
            body = next(
                (
                    element
                    for element in document.iter()
                    if _xml_name(element.tag) == "body"
                ),
                document,
            )
            chapter_heading = next(
                (
                    _normalized_epub_text(element).upper()
                    for element in body.iter()
                    if _xml_name(element.tag) in {"h1", "h2", "h3", "h4", "h5", "h6"}
                    and _normalized_epub_text(element).upper() in headings
                ),
                None,
            )
            if chapter_heading is None:
                counts["epub_spine_documents_skipped"] += 1
                continue
            prose = [
                text
                for element in body.iter()
                if _xml_name(element.tag) == "p"
                and (text := _normalized_epub_text(element))
                and text.upper() != chapter_heading
            ]
            if not prose:
                raise CorpusValidationError(
                    f"{spec.id} chapter {len(chapter_parts) + 1} contains no EPUB paragraphs"
                )
            if validate and len(prose) < MIN_EPUB_PARAGRAPHS_PER_CHAPTER:
                raise CorpusValidationError(
                    f"{spec.id} chapter {len(chapter_parts) + 1} preserves only "
                    f"{len(prose)} paragraph elements; use a better EPUB source"
                )
            largest_paragraph = max(len(paragraph.split()) for paragraph in prose)
            if validate and largest_paragraph > MAX_EPUB_PARAGRAPH_WORDS:
                raise CorpusValidationError(
                    f"{spec.id} chapter {len(chapter_parts) + 1} contains a "
                    f"{largest_paragraph}-word paragraph; paragraph boundaries appear lost"
                )
            chapter_parts.append(
                (
                    spec.heading_aliases.get(chapter_heading, chapter_heading),
                    chapter_heading,
                    prose,
                )
            )
            if len(chapter_parts) == spec.expected_chapters:
                break
    finally:
        archive.close()

    pov_ordinals: Counter[str] = Counter()
    chapters: list[Chapter] = []
    paragraphs: list[Paragraph] = []
    for sequence, (pov, source_heading, prose) in enumerate(chapter_parts, start=1):
        pov_ordinals[pov] += 1
        chapter, chapter_paragraphs = _build_chapter(
            spec,
            sequence=sequence,
            pov=pov,
            pov_ordinal=pov_ordinals[pov],
            paragraphs=prose,
            title=(
                source_heading if source_heading in spec.heading_aliases else None
            ),
        )
        chapters.append(chapter)
        paragraphs.extend(chapter_paragraphs)

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
        normalized_book_text = " ".join(chapter.text for chapter in chapters).casefold()
        for marker in spec.epub_content_markers:
            if marker.casefold() not in normalized_book_text:
                errors.append(
                    f"required passage {marker!r} is missing; EPUB content appears incomplete"
                )
        if errors:
            raise CorpusValidationError("; ".join(errors))

    counts["epub_chapter_documents_read"] = len(chapters)
    counts["epub_paragraphs_preserved"] = len(paragraphs)
    source = CorpusSource(
        book_id=spec.id,
        book_title=spec.title,
        book_sequence=spec.sequence,
        path=source_path,
        sha256=raw_hash,
        source_format="epub",
    )
    return (
        source,
        tuple(chapters),
        tuple(paragraphs),
        dict(sorted(counts.items())),
        EPUB_NORMALIZATION_VERSION,
    )


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
                "source_format": source.source_format,
            }
            for source in sources
        ],
        "source_sha256": corpus.source_sha256,
        "normalization_version": corpus.normalization_version,
        "cleaning_counts": corpus.cleaning_counts,
        "book_count": len(sources),
        "chapter_count": len(corpus.chapters),
        "paragraph_count": len(corpus.paragraphs),
        "chapter_counts_by_book": dict(
            sorted(Counter(chapter.book_id for chapter in corpus.chapters).items())
        ),
        "pov_counts": dict(
            sorted(Counter(chapter.pov for chapter in corpus.chapters).items())
        ),
        "word_count": sum(chapter.word_count for chapter in corpus.chapters),
    }
