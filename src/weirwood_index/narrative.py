from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from weirwood_index.lexical import tokenize
from weirwood_index.models import Chunk

NARRATIVE_VIEW_VERSION = 1
NARRATIVE_VIEW_NAMES = ("context", "summary", "dialogue", "events", "entities")
MAX_SENTENCE_VIEWS = 6
QUERY_EXPANSION_VERSION = "asoiaf-aliases-v1"

SAGA_QUERY_EXPANSIONS = {
    "acting hand": "Tyrion Lannister the Imp Hand of the King dwarf",
    "chief minister": "Hand of the King",
    "city watch": "gold cloaks",
    "foul-smelling prisoner": "Reek Ramsay Bolton",
    "little bird": "Sansa Stark",
    "red priestess": "Melisandre red woman",
    "scarred guard": "Sandor Clegane the Hound",
    "scarred warrior": "Sandor Clegane the Hound",
    "silver-haired queen": "Daenerys Targaryen",
    "the bull": "Gendry horned helm",
    "the halfhand": "Qhorin Halfhand",
    "the hound": "Sandor Clegane scarred warrior",
    "the imp": "Tyrion Lannister dwarf",
    "the kingslayer": "Jaime Lannister",
    "the mountain": "Gregor Clegane",
    "the onion knight": "Davos Seaworth smuggler",
    "the red woman": "Melisandre red priestess",
    "the young wolf": "Robb Stark",
    "white knight": "Kingsguard",
}

SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"'])|(?<=[.!?][\"')\]])\s+(?=[A-Z\"'])"
)
QUOTATION_PATTERN = re.compile(r'"([^"\n]{3,})"')
ENTITY_PATTERN = re.compile(
    r"\b(?:(?:Grand\s+)?(?:Maester|Ser|Lord|Lady|King|Queen|Prince|Princess|Septa|Khal)\s+)?"
    r"[A-Z][a-z]+(?:\s+(?:of|the|[A-Z][a-z]+)){0,3}\b"
)

ROLE_TERMS = frozenset(
    {
        "assassin",
        "bastard",
        "boy",
        "brother",
        "captive",
        "child",
        "commander",
        "daughter",
        "dragon",
        "dwarf",
        "father",
        "fool",
        "girl",
        "guard",
        "heir",
        "king",
        "knight",
        "lord",
        "maester",
        "mother",
        "priestess",
        "prince",
        "princess",
        "prisoner",
        "queen",
        "ranger",
        "sailor",
        "singer",
        "sister",
        "smuggler",
        "soldier",
        "squire",
        "ward",
        "watchman",
        "widow",
        "wolf",
    }
)
EVENT_TERMS = frozenset(
    {
        "admitted",
        "attacked",
        "beat",
        "betrayed",
        "burned",
        "captured",
        "confessed",
        "died",
        "escaped",
        "fell",
        "fled",
        "fought",
        "gave",
        "killed",
        "kissed",
        "murdered",
        "offered",
        "ordered",
        "promised",
        "pushed",
        "refused",
        "released",
        "rescued",
        "returned",
        "rode",
        "said",
        "saw",
        "seized",
        "sent",
        "slew",
        "stabbed",
        "stole",
        "struck",
        "told",
        "watched",
        "wounded",
    }
)
ENTITY_STOPWORDS = frozenset(
    {
        "A",
        "An",
        "And",
        "As",
        "At",
        "But",
        "For",
        "He",
        "Her",
        "His",
        "I",
        "If",
        "In",
        "It",
        "My",
        "No",
        "Not",
        "She",
        "So",
        "That",
        "The",
        "Then",
        "There",
        "They",
        "This",
        "We",
        "When",
        "With",
        "You",
    }
)


@dataclass(frozen=True)
class NarrativeView:
    chunk_id: str
    context: str
    summary: str
    dialogue: str
    events: str
    entities: str
    sentences: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sentences"] = list(self.sentences)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrativeView:
        values = dict(data)
        sentences = values.get("sentences")
        if not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise ValueError("narrative view sentences must be a string array")
        values["sentences"] = tuple(sentences)
        return cls(**values)

    @property
    def lexical_text(self) -> str:
        return " ".join(
            value
            for value in (self.context, self.summary, self.dialogue, self.events, self.entities)
            if value
        )


def split_sentences(text: str) -> tuple[str, ...]:
    return tuple(
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(" ".join(text.split()))
        if sentence.strip()
    )


def expand_narrative_query(query: str) -> str:
    normalized = " ".join(query.split())
    lowered = normalized.casefold()
    additions = [
        expansion
        for phrase, expansion in SAGA_QUERY_EXPANSIONS.items()
        if phrase in lowered and expansion.casefold() not in lowered
    ]
    return " ".join([normalized, *additions])


def build_narrative_view(chunk: Chunk) -> NarrativeView:
    sentences = split_sentences(chunk.text)
    entities = extract_entities(chunk.text)
    roles = sorted(set(tokenize(chunk.text)) & ROLE_TERMS)
    dialogue_parts = [
        " ".join(match.group(1).split())
        for match in QUOTATION_PATTERN.finditer(chunk.text)
    ]
    event_sentences = _rank_event_sentences(sentences)
    summary_sentences = _extractive_summary(sentences, event_sentences)

    entity_parts = []
    if entities:
        entity_parts.append("Named characters and places: " + ", ".join(entities))
    if roles:
        entity_parts.append("Narrative roles: " + ", ".join(roles))
    entity_text = ". ".join(entity_parts)
    metadata = (
        f"Book: {chunk.book_title}. Chapter: {chunk.chapter_title}. "
        f"Point of view: {chunk.pov}."
    )
    if entity_text:
        metadata += " " + entity_text + "."

    return NarrativeView(
        chunk_id=chunk.id,
        context=f"{metadata} Passage: {chunk.text}",
        summary=" ".join(summary_sentences),
        dialogue=" Dialogue: ".join(dialogue_parts),
        events=" ".join(event_sentences),
        entities=entity_text,
        sentences=tuple(sentences[:MAX_SENTENCE_VIEWS]),
    )


def extract_entities(text: str) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    first_positions: dict[str, int] = {}
    for match in ENTITY_PATTERN.finditer(text):
        value = " ".join(match.group(0).split())
        if value in ENTITY_STOPWORDS:
            continue
        # Single capitalized words at sentence starts are mostly false positives.
        # Keep them only when repeated or when they carry a narrative title.
        counts[value] += 1
        first_positions.setdefault(value, match.start())
    ranked = [
        value
        for value, count in counts.most_common()
        if count > 1 or " " in value or value.split()[0] in {"Ser", "Lord", "Lady", "King", "Queen"}
    ]
    ranked.sort(key=lambda value: (first_positions[value], value))
    return tuple(ranked[:12])


def _rank_event_sentences(sentences: tuple[str, ...]) -> tuple[str, ...]:
    scored: list[tuple[int, int, str]] = []
    for position, sentence in enumerate(sentences):
        tokens = set(tokenize(sentence))
        event_score = len(tokens & EVENT_TERMS)
        role_score = len(tokens & ROLE_TERMS)
        quote_score = int('"' in sentence)
        score = event_score * 3 + role_score + quote_score
        if score:
            scored.append((-score, position, sentence))
    return tuple(sentence for _, _, sentence in sorted(scored)[:3])


def _extractive_summary(
    sentences: tuple[str, ...], event_sentences: tuple[str, ...]
) -> tuple[str, ...]:
    if not sentences:
        return ()
    selected = [sentences[0]]
    for sentence in event_sentences:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) == 2:
            break
    return tuple(selected)
