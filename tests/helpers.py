from __future__ import annotations

import hashlib
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
