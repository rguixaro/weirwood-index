from __future__ import annotations

import re
from dataclasses import dataclass

from weirwood_index.models import Chunk, Corpus, WeirwoodError


@dataclass(frozen=True)
class ChunkProfile:
    name: str
    words: int
    overlap: int
    strategy: str = "fixed"
    min_words: int | None = None

    @property
    def stride(self) -> int:
        return self.words - self.overlap


PROFILES = {
    "short": ChunkProfile("short", words=180, overlap=45),
    "medium": ChunkProfile("medium", words=300, overlap=60),
    "structured": ChunkProfile(
        "structured",
        words=160,
        overlap=40,
        strategy="structure",
        min_words=120,
    ),
}

SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


def get_profile(name: str) -> ChunkProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise WeirwoodError(f"unknown chunk profile {name!r}; choose one of: {choices}") from exc


def chunk_corpus(corpus: Corpus, profile: ChunkProfile) -> tuple[Chunk, ...]:
    if profile.words <= 0 or profile.overlap < 0 or profile.overlap >= profile.words:
        raise WeirwoodError(
            f"invalid chunk profile {profile.name}: words={profile.words}, "
            f"overlap={profile.overlap}"
        )

    if profile.strategy not in {"fixed", "structure"}:
        raise WeirwoodError(
            f"invalid chunk profile {profile.name}: unknown strategy {profile.strategy!r}"
        )
    if profile.strategy == "structure" and (
        profile.min_words is None or not 1 <= profile.min_words <= profile.words
    ):
        raise WeirwoodError(
            f"invalid chunk profile {profile.name}: min_words={profile.min_words}"
        )

    chunks: list[Chunk] = []
    for chapter in corpus.chapters:
        words = chapter.text.split()
        spans = (
            _fixed_spans(len(words), profile)
            if profile.strategy == "fixed"
            else _structured_spans(chapter.text, profile)
        )
        for ordinal, (start, end) in enumerate(spans, start=1):
            chunks.append(
                Chunk(
                    id=f"{chapter.id}-c{ordinal:03d}",
                    chapter_id=chapter.id,
                    chapter_title=chapter.title,
                    chapter_sequence=chapter.sequence,
                    pov=chapter.pov,
                    pov_ordinal=chapter.pov_ordinal,
                    chunk_ordinal=ordinal,
                    word_start=start,
                    word_end=end,
                    text=" ".join(words[start:end]),
                    book_id=chapter.book_id,
                    book_title=chapter.book_title,
                    book_sequence=chapter.book_sequence,
                )
            )
    return tuple(chunks)


def _fixed_spans(word_count: int, profile: ChunkProfile) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while start < word_count:
        end = min(start + profile.words, word_count)
        spans.append((start, end))
        if end == word_count:
            break
        start += profile.stride
    return spans


def _structured_spans(text: str, profile: ChunkProfile) -> list[tuple[int, int]]:
    words = text.split()
    if not words:
        return []

    paragraph_ends: set[int] = set()
    word_offset = 0
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph_words = paragraph.split()
        if not paragraph_words:
            continue
        word_offset += len(paragraph_words)
        paragraph_ends.add(word_offset)

    sentence_ends = {
        position
        for position, word in enumerate(words, start=1)
        if SENTENCE_END.search(word)
    }
    sentence_ends.update(paragraph_ends)
    sentence_ends.add(len(words))

    assert profile.min_words is not None
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(words):
        hard_end = min(start + profile.words, len(words))
        minimum_end = min(start + profile.min_words, len(words))
        if hard_end == len(words):
            end = len(words)
        else:
            paragraph_candidates = [
                boundary
                for boundary in paragraph_ends
                if minimum_end <= boundary <= hard_end
            ]
            sentence_candidates = [
                boundary
                for boundary in sentence_ends
                if minimum_end <= boundary <= hard_end
            ]
            if paragraph_candidates:
                end = max(paragraph_candidates)
            elif sentence_candidates:
                end = max(sentence_candidates)
            else:
                # A single sentence can exceed the target. Cutting it is preferable
                # to silently producing an overlong embedding input.
                end = hard_end
        spans.append((start, end))
        if end == len(words):
            break

        desired_start = end - profile.overlap
        start_candidates = [
            boundary
            for boundary in sentence_ends | {0}
            if start < boundary < end
        ]
        if start_candidates:
            next_start = min(
                start_candidates,
                key=lambda boundary: (abs(boundary - desired_start), -boundary),
            )
        else:
            next_start = max(start + 1, desired_start)
        start = next_start
    return spans
