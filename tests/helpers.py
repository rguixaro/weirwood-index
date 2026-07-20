from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

CHAPTER_SEQUENCE = [
    "PROLOGUE",
    "BRAN",
    "CATELYN",
    "DAENERYS",
    "EDDARD",
    "JON",
    "CATELYN",
    "ARYA",
    "BRAN",
    "TYRION",
    "JON",
    "DAENERYS",
    "EDDARD",
    "TYRION",
    "CATELYN",
    "SANSA",
    "EDDARD",
    "BRAN",
    "CATELYN",
    "JON",
    "EDDARD",
    "TYRION",
    "ARYA",
    "DAENERYS",
    "BRAN",
    "EDDARD",
    "JON",
    "EDDARD",
    "CATELYN",
    "SANSA",
    "EDDARD",
    "TYRION",
    "ARYA",
    "EDDARD",
    "CATELYN",
    "EDDARD",
    "DAENERYS",
    "BRAN",
    "TYRION",
    "EDDARD",
    "CATELYN",
    "JON",
    "TYRION",
    "EDDARD",
    "SANSA",
    "EDDARD",
    "DAENERYS",
    "EDDARD",
    "JON",
    "EDDARD",
    "ARYA",
    "SANSA",
    "JON",
    "BRAN",
    "DAENERYS",
    "CATELYN",
    "TYRION",
    "SANSA",
    "EDDARD",
    "CATELYN",
    "JON",
    "DAENERYS",
    "TYRION",
    "CATELYN",
    "DAENERYS",
    "ARYA",
    "BRAN",
    "SANSA",
    "DAENERYS",
    "TYRION",
    "JON",
    "CATELYN",
    "DAFNERYS",
]

ACOK_CHAPTER_SEQUENCE = [
    "PROLOGUE", "ARYA", "SANSA", "TYRION", "BRAN", "ARYA", "JON", "CATELYN",
    "TYRION", "ARYA", "DAVOS", "THEON", "DAENERYS", "JON", "ARYA", "TYRION",
    "BRAN", "TYRION", "SANSA", "ARYA", "TYRION", "BRAN", "CATELYN", "JON",
    "THEON", "TYRION", "ARYA", "DAENERYS", "BRAN", "TYRION", "ARYA", "CATELYN",
    "SANSA", "CATELYN", "JON", "BRAN", "TYRION", "THEON", "ARYA", "CATELYN",
    "DAENERYS", "TYRION", "DAVOS", "JON", "TYRION", "CATELYN", "BRAN", "ARYA",
    "DAENERYS", "TYRION", "THEON", "JON", "SANSA", "JON", "TYRION", "CATELYN",
    "THEON", "SANSA", "DAVOS", "TYRION", "SANSA", "TYRION", "SANSA", "DAENERYS",
    "ARYA", "SANSA", "THEON", "TYRION", "JON", "BRAN",
]


def write_valid_source(tmp_path: Path, *, words_per_chapter: int = 24) -> Path:
    lines = [
        "A Game Of Thrones ",
        "Book One of A Song of Ice and Fire ",
        "By George R. R. Martin ",
    ]
    for sequence, heading in enumerate(CHAPTER_SEQUENCE, start=1):
        lines.append(heading + " ")
        words = [f"chapter{sequence}"] + [f"word{number}" for number in range(words_per_chapter)]
        midpoint = len(words) // 2
        lines.append(" ".join(words[:midpoint]) + " ")
        if sequence == 1:
            lines.extend(["Page 1 ", "A GAML OF THRONES 109 ", ""])
        lines.append(" ".join(words[midpoint:]) + " ")
    path = tmp_path / "book.txt"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
    return path


def write_valid_acok_source(tmp_path: Path, *, words_per_chapter: int = 24) -> Path:
    lines = [
        "A Clash of Kings",
        "Book Two of A song of Ice and Fire",
        "By George R. R. Martin",
    ]
    catelyn_corrections = iter(("CALTELYN", "CATIELYN"))
    for sequence, heading in enumerate(ACOK_CHAPTER_SEQUENCE, start=1):
        if sequence > 1:
            lines.append(f"CHAPTER {sequence - 1}")
        if heading == "CATELYN" and sequence in {8, 23}:
            heading = next(catelyn_corrections)
        lines.append(heading)
        words = [f"acokchapter{sequence}"] + [
            f"word{number}" for number in range(words_per_chapter)
        ]
        midpoint = len(words) // 2
        lines.append(" ".join(words[:midpoint]))
        if sequence == 1:
            lines.append("Page 1")
        lines.append(" ".join(words[midpoint:]))
    lines.extend(["APPENDIX", "HOUSE STARK", "appendix material must not be indexed"])
    path = tmp_path / "acok.txt"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
    return path


def write_valid_epub_source(
    tmp_path: Path,
    *,
    book_id: str,
    collapsed_paragraphs: bool = False,
    include_required_content: bool = True,
    title_override: str | None = None,
    headings_override: Sequence[str] | None = None,
    required_marker: str | None = None,
) -> Path:
    if title_override is not None and headings_override is not None:
        title = title_override
        headings = list(headings_override)
    elif book_id == "agot":
        title = "A Game of Thrones"
        headings = [
            "DAENERYS" if heading == "DAFNERYS" else heading
            for heading in CHAPTER_SEQUENCE
        ]
    elif book_id == "acok":
        title = "A Clash of Kings"
        headings = ACOK_CHAPTER_SEQUENCE
    else:
        raise ValueError(f"unsupported synthetic EPUB book: {book_id}")

    manifest_items: list[str] = []
    spine_items: list[str] = []
    documents: dict[str, str] = {}
    for sequence, heading in enumerate(headings, start=1):
        item_id = f"chapter-{sequence:03d}"
        href = f"text/{item_id}.xhtml"
        manifest_items.append(
            f'<item id="{item_id}" href="{href}" media-type="application/xhtml+xml"/>'
        )
        spine_items.append(f'<itemref idref="{item_id}"/>')
        paragraphs = [
            " ".join(
                f"{book_id}chapter{sequence}p{paragraph}w{word}"
                for word in range(5)
            )
            for paragraph in range(1, 11)
        ]
        if book_id == "acok" and sequence == 49 and include_required_content:
            paragraphs[-1] += " A blue flower grew from a chink in a wall of ice."
        if (
            required_marker
            and sequence == len(headings)
            and include_required_content
        ):
            paragraphs[-1] += f" {required_marker}."
        if collapsed_paragraphs:
            paragraphs = [" ".join(paragraphs)]
        paragraph_markup = "".join(f"<p>{text}</p>" for text in paragraphs)
        documents[href] = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            f"<h1>{heading}</h1>{paragraph_markup}</body></html>"
        )

    package = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title></metadata>"
        f"<manifest>{''.join(manifest_items)}</manifest>"
        f"<spine>{''.join(spine_items)}</spine></package>"
    )
    container = (
        '<?xml version="1.0"?>'
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    path = tmp_path / f"{book_id}.epub"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", package)
        for name, content in documents.items():
            archive.writestr(name, content)
    return path


class FakeEncoder:
    model_id = "test/fake-encoder"
    revision = "frozen-test"
    max_tokens = 512

    def __init__(self, **_: object) -> None:
        pass

    def token_count(self, text: str) -> int:
        return len(text.split()) + 2

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode()).digest()
        vector = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) + 1.0
        return vector / np.linalg.norm(vector)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return np.stack([self._vector(passage) for passage in passages]).astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return np.stack([self._vector(query) for query in queries]).astype(np.float32)


class FakeReranker:
    model_id = "test/fake-reranker"
    revision = "frozen-test"

    def __init__(self, **_: object) -> None:
        pass

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        del query
        # Deterministically reverse the semantic candidate order in tests.
        return np.arange(len(passages), dtype=np.float32)
