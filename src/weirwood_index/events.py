from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Span, Token

from weirwood_index.scenes import SceneWindow, canonical_identities

EVENT_INDEX_VERSION = 1
EVENT_PARSER_MODEL = "en_core_web_sm"
MAX_EVENTS_PER_WINDOW = 40

ACTION_GROUPS: dict[str, frozenset[str]] = {
    "attack": frozenset({"attack", "fight", "strike", "assault", "charge", "ambush"}),
    "betray": frozenset({"betray", "deceive", "lie", "trick"}),
    "burn": frozenset({"burn", "scorch", "set", "light"}),
    "capture": frozenset({"capture", "catch", "hold", "imprison", "seize", "take"}),
    "command": frozenset({"command", "demand", "forbid", "order", "warn"}),
    "crown": frozenset({"crown", "name", "proclaim"}),
    "die": frozenset({"die", "drown", "fall", "perish"}),
    "dream": frozenset({"dream", "remember", "recall", "see", "vision"}),
    "escape": frozenset({"escape", "flee", "fly", "leave", "run"}),
    "execute": frozenset({"behead", "execute", "hang"}),
    "give": frozenset({"bring", "give", "offer", "present"}),
    "kill": frozenset({"kill", "murder", "slay"}),
    "marry": frozenset({"betroth", "marry", "wed"}),
    "protect": frozenset({"defend", "guard", "protect", "save"}),
    "release": frozenset({"free", "pardon", "release", "spare"}),
    "rescue": frozenset({"help", "rescue", "save"}),
    "reveal": frozenset({"admit", "confess", "reveal", "show"}),
    "say": frozenset({"ask", "say", "speak", "tell", "warn"}),
    "stop": frozenset({"block", "halt", "prevent", "stop"}),
    "steal": frozenset({"rob", "steal", "take"}),
}
ACTION_BY_LEMMA = {
    lemma: action for action, lemmas in ACTION_GROUPS.items() for lemma in lemmas
}
ACTION_BY_LEMMA.update({"take": "capture", "warn": "command"})
MODAL_TERMS = frozenset({"could", "if", "may", "might", "should", "unless", "would"})
RECALL_LEMMAS = frozenset({"dream", "imagine", "recall", "remember", "think"})
SUBJECT_DEPS = frozenset({"csubj", "nsubj", "nsubjpass"})
OBJECT_DEPS = frozenset({"attr", "dative", "dobj", "obj", "oprd"})
EVENT_VERB_DEPS = frozenset({"ROOT", "advcl", "ccomp", "conj", "relcl", "xcomp"})
ENTITY_NOISE = frozenset(
    {
        "and",
        "as",
        "at",
        "behind",
        "bring",
        "despite",
        "each",
        "even",
        "finally",
        "from",
        "his",
        "many",
        "or",
        "the",
        "their",
        "toward",
        "tell",
        "keep",
        "what",
        "when",
        "where",
        "your",
    }
)


@dataclass(frozen=True)
class EventRecord:
    id: str
    scene_window_id: str
    chapter_id: str
    book_id: str
    chronology_word: int
    sentence_index: int
    sentence: str
    subject: str | None
    action: str
    action_lemma: str
    object: str | None
    context_entities: tuple[str, ...]
    negated: bool
    hypothetical: bool
    recalled: bool
    passive: bool
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["context_entities"] = list(self.context_entities)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventRecord:
        values = dict(data)
        entities = values.get("context_entities", [])
        if not isinstance(entities, list) or not all(
            isinstance(entity, str) for entity in entities
        ):
            raise ValueError("event context_entities must be a string array")
        values["context_entities"] = tuple(entities)
        return cls(**values)

    @property
    def lexical_text(self) -> str:
        parts = [self.sentence, f"action {self.action} {self.action_lemma}"]
        if self.subject:
            parts.append(f"subject {self.subject}")
        if self.object:
            parts.append(f"object {self.object}")
        if self.context_entities:
            parts.append("scene entities " + " ".join(self.context_entities))
        parts.append("hypothetical" if self.hypothetical else "asserted")
        if self.recalled:
            parts.append("recalled memory")
        if self.negated:
            parts.append("negated")
        return " ".join(parts)


@dataclass(frozen=True)
class QueryEventSignature:
    entities: frozenset[str]
    actions: frozenset[str]
    subject: str | None
    object: str | None
    hypothetical: bool
    negated: bool


def load_event_parser() -> Language:
    import spacy

    return spacy.load(EVENT_PARSER_MODEL, disable=["ner"])


def build_event_records(
    windows: tuple[SceneWindow, ...],
    nlp: Language,
    *,
    batch_size: int = 32,
) -> tuple[EventRecord, ...]:
    records: list[EventRecord] = []
    docs = nlp.pipe((window.text for window in windows), batch_size=batch_size)
    for window, doc in zip(windows, docs, strict=True):
        window_records = 0
        context_entities = _context_entities(window)
        for sentence_index, sentence in enumerate(doc.sents):
            for action_token in _action_tokens(sentence):
                subject_token = _event_subject(action_token)
                object_token = _object_token(action_token)
                subject = _canonical_phrase(subject_token)
                object_value = _canonical_phrase(object_token)
                action_lemma = action_token.lemma_.casefold()
                confidence = (
                    "high" if subject is not None and object_value is not None
                    else "medium" if subject is not None
                    else "low"
                )
                records.append(
                    EventRecord(
                        id=f"{window.id}-e{window_records + 1:02d}",
                        scene_window_id=window.id,
                        chapter_id=window.chapter_id,
                        book_id=window.book_id,
                        chronology_word=window.word_start,
                        sentence_index=sentence_index,
                        sentence=" ".join(sentence.text.split()),
                        subject=subject,
                        action=canonical_action(action_lemma),
                        action_lemma=action_lemma,
                        object=object_value,
                        context_entities=context_entities,
                        negated=_is_negated(action_token),
                        hypothetical=_is_hypothetical(action_token),
                        recalled=_is_recalled(action_token),
                        passive=any(
                            child.dep_ == "nsubjpass"
                            for child in action_token.children
                        ),
                        confidence=confidence,
                    )
                )
                window_records += 1
                if window_records == MAX_EVENTS_PER_WINDOW:
                    break
            if window_records == MAX_EVENTS_PER_WINDOW:
                break
    return tuple(records)


def query_event_signature(query: str, nlp: Language) -> QueryEventSignature:
    doc = nlp(query)
    entities = set(canonical_identities(query))
    entities.update(_proper_noun_phrases(doc))
    actions = {
        canonical_action(token.lemma_.casefold())
        for token in doc
        if token.pos_ in {"VERB", "AUX"}
    }
    root = next((token for token in doc if token.dep_ == "ROOT"), None)
    if root is not None and root.pos_ == "AUX":
        root = next(
            (child for child in root.children if child.pos_ == "VERB"), root
        )
    subject = _canonical_phrase(_dependent(root, SUBJECT_DEPS)) if root else None
    object_value = _canonical_phrase(_object_token(root)) if root else None
    terms = {token.lower_ for token in doc}
    return QueryEventSignature(
        entities=frozenset(entity.casefold() for entity in entities),
        actions=frozenset(actions),
        subject=subject,
        object=object_value,
        hypothetical=bool(terms & MODAL_TERMS),
        negated=bool(terms & {"never", "no", "not"}),
    )


def structured_event_scores(
    records: tuple[EventRecord, ...], query: str, nlp: Language
) -> np.ndarray:
    signature = query_event_signature(query, nlp)
    scores = np.zeros(len(records), dtype=np.float32)
    for position, record in enumerate(records):
        event_entities = {
            entity.casefold()
            for entity in (
                *record.context_entities,
                record.subject or "",
                record.object or "",
            )
            if entity
        }
        components: list[tuple[float, float]] = []
        if signature.entities:
            entity_overlap = len(signature.entities & event_entities) / len(
                signature.entities
            )
            components.append((0.45, entity_overlap))
        if signature.actions:
            components.append((0.35, float(record.action in signature.actions)))
        if signature.subject or signature.object:
            direction_parts = []
            if signature.subject:
                direction_parts.append(
                    float(_same_entity(signature.subject, record.subject))
                )
            if signature.object:
                direction_parts.append(
                    float(_same_entity(signature.object, record.object))
                )
            components.append((0.15, sum(direction_parts) / len(direction_parts)))
        assertion_match = float(
            signature.hypothetical == record.hypothetical
            and signature.negated == record.negated
        )
        components.append((0.05, assertion_match))
        weight_sum = sum(weight for weight, _ in components)
        score = sum(weight * value for weight, value in components) / weight_sum
        if not signature.hypothetical and record.recalled:
            score *= 0.85
        scores[position] = score
    return scores


def canonical_action(lemma: str) -> str:
    return ACTION_BY_LEMMA.get(lemma.casefold(), lemma.casefold())


def _action_tokens(sentence: Span) -> tuple[Token, ...]:
    verbs = tuple(
        token
        for token in sentence
        if token.pos_ == "VERB" and token.dep_ in EVENT_VERB_DEPS
    )
    if verbs:
        return verbs
    root = sentence.root
    if root.pos_ == "AUX":
        child = next((item for item in root.children if item.pos_ == "VERB"), None)
        return (child or root,)
    return ()


def _dependent(token: Token | None, dependencies: frozenset[str]) -> Token | None:
    if token is None:
        return None
    return next((child for child in token.children if child.dep_ in dependencies), None)


def _object_token(token: Token | None) -> Token | None:
    direct = _dependent(token, OBJECT_DEPS)
    if direct is not None or token is None:
        return direct
    for child in token.children:
        if child.dep_ == "prep":
            nested = _dependent(child, frozenset({"pobj"}))
            if nested is not None:
                return nested
    return None


def _event_subject(token: Token) -> Token | None:
    subject = _dependent(token, SUBJECT_DEPS)
    if subject is not None:
        return subject
    # In "Ned warned Cersei to leave", the object of "warned" is the
    # understood subject of the xcomp "leave".
    if token.dep_ == "xcomp":
        inherited_object = _object_token(token.head)
        if inherited_object is not None:
            return inherited_object
    ancestor = token.head
    for _ in range(3):
        if ancestor == token:
            break
        subject = _dependent(ancestor, SUBJECT_DEPS)
        if subject is not None:
            return subject
        if ancestor.dep_ == "ROOT":
            break
        ancestor = ancestor.head
    return None


def _is_negated(token: Token) -> bool:
    return any(child.dep_ == "neg" for child in token.children)


def _is_hypothetical(token: Token) -> bool:
    local_tokens = {child.lower_ for child in token.children}
    local_tokens.add(token.lower_)
    if local_tokens & MODAL_TERMS:
        return True
    return any(
        descendant.dep_ in {"aux", "mark"}
        and descendant.lower_ in MODAL_TERMS
        for descendant in token.subtree
    )


def _is_recalled(token: Token) -> bool:
    if token.lemma_.casefold() in RECALL_LEMMAS:
        return True
    ancestor = token.head
    for _ in range(2):
        if ancestor == token:
            break
        if ancestor.lemma_.casefold() in RECALL_LEMMAS:
            return True
        ancestor = ancestor.head
    return False


def _canonical_phrase(token: Token | None) -> str | None:
    if token is None:
        return None
    phrase = " ".join(
        item.text for item in sorted(token.subtree, key=lambda item: item.i) if not item.is_punct
    )
    identities = canonical_identities(phrase)
    if identities:
        return identities[0].casefold()
    normalized = re.sub(r"\s+", " ", phrase).strip().casefold()
    return normalized or None


def _proper_noun_phrases(doc: Any) -> tuple[str, ...]:
    phrases: list[str] = []
    active: list[str] = []
    for token in doc:
        if token.pos_ == "PROPN":
            active.append(token.text)
        elif active:
            phrases.append(_canonical_proper_phrase(active))
            active = []
    if active:
        phrases.append(_canonical_proper_phrase(active))
    return tuple(phrases)


def _canonical_proper_phrase(tokens: list[str]) -> str:
    phrase = " ".join(tokens)
    identities = canonical_identities(phrase)
    return (identities[0] if identities else phrase).casefold()


def _context_entities(window: SceneWindow) -> tuple[str, ...]:
    values: list[str] = []
    for value in (*window.aliases, *window.entities):
        normalized = value.casefold().strip()
        first = normalized.split(maxsplit=1)[0]
        if first in ENTITY_NOISE or normalized in ENTITY_NOISE:
            continue
        identities = canonical_identities(value)
        values.append((identities[0] if identities else value).casefold())
    return tuple(dict.fromkeys(values))


def _same_entity(left: str, right: str | None) -> bool:
    if right is None:
        return False
    return left.casefold() == right.casefold()
