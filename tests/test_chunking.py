from __future__ import annotations

from pathlib import Path

from weirwood_index.chunking import PROFILES, ChunkProfile, chunk_corpus
from weirwood_index.models import Chapter, Corpus


def _corpus() -> Corpus:
    chapters = (
        Chapter("agot-001-one-1", 1, "ONE", 1, "ONE I", " ".join(f"a{i}" for i in range(420))),
        Chapter("agot-002-two-1", 2, "TWO", 1, "TWO I", " ".join(f"b{i}" for i in range(200))),
    )
    return Corpus(Path("book.txt"), "hash", "test", {}, chapters)


def test_short_chunks_have_expected_overlap_and_boundaries() -> None:
    chunks = chunk_corpus(_corpus(), PROFILES["short"])
    first_chapter = [chunk for chunk in chunks if chunk.chapter_id == "agot-001-one-1"]

    assert [(chunk.word_start, chunk.word_end) for chunk in first_chapter] == [
        (0, 180),
        (135, 315),
        (270, 420),
    ]
    assert first_chapter[0].text.split()[-45:] == first_chapter[1].text.split()[:45]
    assert all(chunk.text.startswith("a") for chunk in first_chapter)


def test_chunking_is_deterministic_and_never_crosses_chapters() -> None:
    profile = ChunkProfile("custom", words=100, overlap=25)

    first = chunk_corpus(_corpus(), profile)
    second = chunk_corpus(_corpus(), profile)

    assert first == second
    assert len({chunk.id for chunk in first}) == len(first)
    assert all(
        all(word.startswith("a") for word in chunk.text.split())
        if chunk.chapter_id == "agot-001-one-1"
        else all(word.startswith("b") for word in chunk.text.split())
        for chunk in first
    )


def test_structured_chunks_prefer_paragraph_then_sentence_boundaries() -> None:
    def sentence(number: int) -> str:
        words = [f"s{number}w{word}" for word in range(10)]
        words[-1] += "."
        return " ".join(words)

    text = "\n\n".join(
        [
            " ".join(sentence(number) for number in range(14)),
            " ".join(sentence(number) for number in range(14, 28)),
            " ".join(sentence(number) for number in range(28, 30)),
        ]
    )
    chapter = Chapter("agot-001-one-1", 1, "ONE", 1, "ONE I", text)
    corpus = Corpus(Path("book.txt"), "hash", "test", {}, (chapter,))

    chunks = chunk_corpus(corpus, PROFILES["structured"])

    assert [(chunk.word_start, chunk.word_end) for chunk in chunks] == [
        (0, 140),
        (100, 260),
        (220, 300),
    ]
    assert chunks[0].text.split()[-1].endswith(".")
    assert chunks[1].text.split()[-1].endswith(".")
