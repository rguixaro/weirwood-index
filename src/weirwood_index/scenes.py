from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from weirwood_index.models import Chunk, WeirwoodError
from weirwood_index.narrative import extract_entities

SCENE_WINDOW_VERSION = 1
DEFAULT_SCENE_WINDOW_WORDS = 280
DEFAULT_SCENE_WINDOW_OVERLAP = 100
DEFAULT_ENTITY_SCOPE_WORDS = 1200

# Deliberately excludes ambiguous role-only aliases such as "the queen". Those are
# useful metadata when a named character occurs in the local scope, but unsafe as
# unconditional query expansions across a five-volume corpus.
CHARACTER_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Eddard Stark", ("Ned", "Lord Eddard", "Lord Stark")),
    ("Cersei Lannister", ("Cersei", "Queen Cersei")),
    ("Jaime Lannister", ("Jaime", "Kingslayer")),
    ("Tyrion Lannister", ("Tyrion", "the Imp")),
    ("Daenerys Targaryen", ("Daenerys", "Dany", "Khaleesi")),
    ("Viserys Targaryen", ("Viserys",)),
    ("Khal Drogo", ("Drogo", "Khal Drogo")),
    ("Jorah Mormont", ("Jorah", "Ser Jorah")),
    ("Jon Snow", ("Jon Snow", "Lord Snow")),
    ("Arya Stark", ("Arya", "Arry")),
    ("Sansa Stark", ("Sansa", "little bird")),
    ("Bran Stark", ("Bran", "Brandon Stark")),
    ("Robb Stark", ("Robb", "the Young Wolf")),
    ("Catelyn Stark", ("Catelyn", "Cat", "Lady Stark")),
    ("Robert Baratheon", ("Robert", "King Robert")),
    ("Joffrey Baratheon", ("Joffrey", "King Joffrey")),
    ("Stannis Baratheon", ("Stannis", "King Stannis")),
    ("Renly Baratheon", ("Renly", "King Renly")),
    ("Theon Greyjoy", ("Theon", "Reek")),
    ("Asha Greyjoy", ("Asha", "Esgred")),
    ("Sandor Clegane", ("Sandor", "the Hound")),
    ("Gregor Clegane", ("Gregor", "the Mountain")),
    ("Davos Seaworth", ("Davos", "the Onion Knight")),
    ("Melisandre", ("Melisandre", "the red woman", "red priestess")),
    ("Brienne of Tarth", ("Brienne", "Brienne the Beauty")),
    ("Qhorin Halfhand", ("Qhorin", "the Halfhand")),
    ("Samwell Tarly", ("Samwell", "Sam Tarly")),
    ("Petyr Baelish", ("Petyr", "Littlefinger")),
    ("Varys", ("Varys", "the Spider")),
    ("Barristan Selmy", ("Barristan", "Barristan the Bold")),
    ("Jaqen H'ghar", ("Jaqen", "Jaqen H'ghar")),
)


@dataclass(frozen=True)
class SceneWindow:
    id: str
    chapter_id: str
    chapter_title: str
    book_id: str
    book_title: str
    pov: str
    word_start: int
    word_end: int
    text: str
    entities: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entities"] = list(self.entities)
        data["aliases"] = list(self.aliases)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneWindow:
        values = dict(data)
        for key in ("entities", "aliases"):
            items = values.get(key, [])
            if not isinstance(items, list) or not all(
                isinstance(item, str) for item in items
            ):
                raise ValueError(f"scene window {key} must be a string array")
            values[key] = tuple(items)
        return cls(**values)

    @property
    def embedding_text(self) -> str:
        metadata = (
            f"Book: {self.book_title}. Chapter: {self.chapter_title}. "
            f"Point of view: {self.pov}."
        )
        if self.entities:
            metadata += " Entities in surrounding scene: " + ", ".join(self.entities) + "."
        if self.aliases:
            metadata += " Canonical character identities: " + ", ".join(self.aliases) + "."
        return f"{metadata} Scene passage: {self.text}"

    @property
    def lexical_text(self) -> str:
        return self.embedding_text


def expand_scene_query(query: str) -> str:
    normalized = " ".join(query.split())
    additions: list[str] = []
    for canonical, aliases in CHARACTER_ALIAS_GROUPS:
        terms = (canonical, *aliases)
        if any(_contains_phrase(normalized, term) for term in terms):
            expansion = " ".join(dict.fromkeys(terms))
            if expansion.casefold() not in normalized.casefold():
                additions.append(expansion)
    return " ".join((normalized, *additions))


def build_scene_windows(
    chunks: tuple[Chunk, ...],
    *,
    window_words: int = DEFAULT_SCENE_WINDOW_WORDS,
    overlap_words: int = DEFAULT_SCENE_WINDOW_OVERLAP,
    entity_scope_words: int = DEFAULT_ENTITY_SCOPE_WORDS,
) -> tuple[SceneWindow, ...]:
    if window_words < 100:
        raise WeirwoodError("scene window words must be at least 100")
    if not 0 <= overlap_words < window_words:
        raise WeirwoodError("scene window overlap must be smaller than its width")
    if entity_scope_words < window_words:
        raise WeirwoodError("entity scope must be at least as wide as a scene window")

    by_chapter: defaultdict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_chapter[chunk.chapter_id].append(chunk)

    windows: list[SceneWindow] = []
    stride = window_words - overlap_words
    for chapter_chunks in by_chapter.values():
        chapter_chunks.sort(key=lambda item: item.chunk_ordinal)
        chapter_words = _reconstruct_chapter_words(chapter_chunks)
        first = chapter_chunks[0]
        starts: list[int] = []
        start = 0
        while True:
            final_start = max(0, len(chapter_words) - window_words)
            if start + window_words >= len(chapter_words):
                if final_start not in starts:
                    starts.append(final_start)
                break
            starts.append(start)
            start += stride

        for ordinal, start in enumerate(starts, start=1):
            end = min(start + window_words, len(chapter_words))
            scope_start, scope_end = _centered_scope(
                start,
                end,
                len(chapter_words),
                entity_scope_words,
            )
            scope_text = " ".join(chapter_words[scope_start:scope_end])
            # Metadata must leave most of the 512-token budget to prose. Alias
            # identities carry the important coreference signal; extracted proper
            # nouns are capped to avoid scanner noise and long place-name lists.
            entities = extract_entities(scope_text)[:8]
            aliases = canonical_identities(scope_text)[:8]
            windows.append(
                SceneWindow(
                    id=f"{first.chapter_id}-s{ordinal:03d}",
                    chapter_id=first.chapter_id,
                    chapter_title=first.chapter_title,
                    book_id=first.book_id,
                    book_title=first.book_title,
                    pov=first.pov,
                    word_start=start,
                    word_end=end,
                    text=" ".join(chapter_words[start:end]),
                    entities=entities,
                    aliases=aliases,
                )
            )
    return tuple(windows)


def map_chunks_to_scene_windows(
    chunks: tuple[Chunk, ...], windows: tuple[SceneWindow, ...]
) -> tuple[tuple[int, ...], ...]:
    windows_by_chapter: defaultdict[str, list[tuple[int, SceneWindow]]] = defaultdict(list)
    for position, window in enumerate(windows):
        windows_by_chapter[window.chapter_id].append((position, window))
    mappings: list[tuple[int, ...]] = []
    for chunk in chunks:
        positions = tuple(
            position
            for position, window in windows_by_chapter[chunk.chapter_id]
            if window.word_start < chunk.word_end and chunk.word_start < window.word_end
        )
        if not positions:
            raise WeirwoodError(f"no scene window overlaps chunk {chunk.id}")
        mappings.append(positions)
    return tuple(mappings)


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(
        re.search(
            rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)",
            text.casefold(),
        )
    )


def canonical_identities(text: str) -> tuple[str, ...]:
    groups = []
    for canonical, aliases in CHARACTER_ALIAS_GROUPS:
        if any(_contains_phrase(text, term) for term in (canonical, *aliases)):
            # Queries expand an observed alias to this canonical identity. Repeating
            # every alias in every window wastes model tokens and adds lexical noise.
            groups.append(canonical)
    return tuple(groups[:12])


def _centered_scope(
    start: int, end: int, word_count: int, scope_words: int
) -> tuple[int, int]:
    extra = max(0, scope_words - (end - start))
    scope_start = max(0, start - extra // 2)
    scope_end = min(word_count, scope_start + scope_words)
    scope_start = max(0, scope_end - scope_words)
    return scope_start, scope_end


def _reconstruct_chapter_words(chunks: list[Chunk]) -> list[str]:
    word_count = max(chunk.word_end for chunk in chunks)
    words: list[str | None] = [None] * word_count
    for chunk in chunks:
        chunk_words = chunk.text.split()
        if len(chunk_words) != chunk.word_end - chunk.word_start:
            raise WeirwoodError(f"chunk offsets do not match text for {chunk.id}")
        for position, word in enumerate(chunk_words, start=chunk.word_start):
            existing = words[position]
            if existing is not None and existing != word:
                raise WeirwoodError(
                    f"overlapping chunk text disagrees in chapter {chunk.chapter_id}"
                )
            words[position] = word
    if any(word is None for word in words):
        raise WeirwoodError(f"chunk coverage has gaps in chapter {chunks[0].chapter_id}")
    return [word for word in words if word is not None]
