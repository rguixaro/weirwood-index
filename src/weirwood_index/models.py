from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class WeirwoodError(Exception):
    """Base class for actionable command failures."""


class CorpusValidationError(WeirwoodError):
    """The source corpus does not match the expected edition structure."""


class IndexValidationError(WeirwoodError):
    """An on-disk index is missing, corrupt, stale, or incompatible."""


class BenchmarkValidationError(WeirwoodError):
    """A benchmark definition is malformed."""


@dataclass(frozen=True)
class Chapter:
    id: str
    sequence: int
    pov: str
    pov_ordinal: int
    title: str
    text: str
    book_id: str = "agot"
    book_title: str = "A Game of Thrones"
    book_sequence: int = 1

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class CorpusSource:
    book_id: str
    book_title: str
    book_sequence: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class Corpus:
    source_path: Path
    source_sha256: str
    normalization_version: str
    cleaning_counts: dict[str, int]
    chapters: tuple[Chapter, ...]
    sources: tuple[CorpusSource, ...] = ()


@dataclass(frozen=True)
class Chunk:
    id: str
    chapter_id: str
    chapter_title: str
    chapter_sequence: int
    pov: str
    pov_ordinal: int
    chunk_ordinal: int
    word_start: int
    word_end: int
    text: str
    book_id: str = "agot"
    book_title: str = "A Game of Thrones"
    book_sequence: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        try:
            return cls(**data)
        except (TypeError, KeyError) as exc:
            raise IndexValidationError(f"invalid chunk record: {exc}") from exc
