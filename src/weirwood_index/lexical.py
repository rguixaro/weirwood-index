from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from weirwood_index.models import Chunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "for",
        "from",
        "he",
        "her",
        "his",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "to",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True)
class LexicalMatchSpan:
    start: int
    end: int


def _normalized_token(token: str) -> str:
    if token.endswith("'s") and len(token) > 2:
        return token[:-2]
    return token


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize for exact-term retrieval without stemming names into unrelated words."""
    normalized = text.casefold().replace("’", "'")
    return tuple(_normalized_token(token) for token in TOKEN_PATTERN.findall(normalized))


def best_query_span(text: str, query: str) -> LexicalMatchSpan | None:
    """Locate the tightest text span containing the query terms that are present."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return None
    normalized = text.casefold().replace("’", "'")
    matches = list(TOKEN_PATTERN.finditer(normalized))
    document_tokens = tuple(_normalized_token(match.group()) for match in matches)

    width = len(query_tokens)
    for start in range(len(document_tokens) - width + 1):
        if document_tokens[start : start + width] == query_tokens:
            return LexicalMatchSpan(matches[start].start(), matches[start + width - 1].end())

    meaningful_query_tokens = {
        token for token in query_tokens if token not in QUERY_STOPWORDS
    }
    matched_terms = meaningful_query_tokens.intersection(document_tokens)
    if not matched_terms:
        return None
    occurrences = [
        (position, token)
        for position, token in enumerate(document_tokens)
        if token in matched_terms
    ]
    counts: Counter[str] = Counter()
    covered = 0
    left = 0
    best: LexicalMatchSpan | None = None
    for right_position, right_term in occurrences:
        counts[right_term] += 1
        if counts[right_term] == 1:
            covered += 1
        while covered == len(matched_terms):
            left_position, left_term = occurrences[left]
            candidate = LexicalMatchSpan(
                matches[left_position].start(), matches[right_position].end()
            )
            if best is None or candidate.end - candidate.start < best.end - best.start:
                best = candidate
            counts[left_term] -= 1
            if counts[left_term] == 0:
                covered -= 1
            left += 1
    return best


@dataclass(frozen=True)
class BM25Index:
    document_count: int
    document_lengths: np.ndarray
    average_document_length: float
    postings: dict[str, tuple[tuple[int, int], ...]]
    inverse_document_frequencies: dict[str, float]
    document_tokens: tuple[tuple[str, ...], ...]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_chunks(cls, chunks: tuple[Chunk, ...]) -> BM25Index:
        return cls.from_texts([chunk.text for chunk in chunks])

    @classmethod
    def from_texts(cls, texts: list[str]) -> BM25Index:
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequencies: Counter[str] = Counter()
        lengths = np.zeros(len(texts), dtype=np.float32)
        documents: list[tuple[str, ...]] = []

        for position, text in enumerate(texts):
            tokens = tokenize(text)
            documents.append(tokens)
            frequencies = Counter(tokens)
            lengths[position] = sum(frequencies.values())
            for term, frequency in frequencies.items():
                postings[term].append((position, frequency))
                document_frequencies[term] += 1

        document_count = len(texts)
        average_length = float(lengths.mean()) if document_count else 0.0
        idf = {
            term: math.log1p((document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }
        return cls(
            document_count=document_count,
            document_lengths=lengths,
            average_document_length=average_length,
            postings={term: tuple(values) for term, values in postings.items()},
            inverse_document_frequencies=idf,
            document_tokens=tuple(documents),
        )

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.document_count, dtype=np.float32)
        if self.document_count == 0 or self.average_document_length == 0:
            return scores

        for term, query_frequency in Counter(tokenize(query)).items():
            idf = self.inverse_document_frequencies.get(term)
            if idf is None:
                continue
            for position, term_frequency in self.postings[term]:
                length_ratio = float(self.document_lengths[position]) / self.average_document_length
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                scores[position] += (
                    query_frequency * idf * (term_frequency * (self.k1 + 1.0) / denominator)
                )
        return scores

    def exact_phrase_matches(self, query: str) -> np.ndarray:
        """Return passages containing the complete normalized multi-token query."""
        matches = np.zeros(self.document_count, dtype=np.bool_)
        query_tokens = tokenize(query)
        if len(query_tokens) < 2:
            return matches

        width = len(query_tokens)
        for position in self.positions_containing_all(query):
            document = self.document_tokens[position]
            matches[position] = any(
                document[start : start + width] == query_tokens
                for start in range(len(document) - width + 1)
            )
        return matches

    def positions_containing_all(
        self, query: str, *, exclude_stopwords: bool = False
    ) -> list[int]:
        terms = tuple(
            dict.fromkeys(
                token
                for token in tokenize(query)
                if not exclude_stopwords or token not in QUERY_STOPWORDS
            )
        )
        if not terms or any(term not in self.postings for term in terms):
            return []
        positions = [
            {position for position, _ in self.postings[term]} for term in terms
        ]
        return sorted(set.intersection(*positions))

    def evidence_scores(
        self, query: str, *, positions: list[int] | None = None
    ) -> LexicalEvidenceScores:
        """Score exact query-term coverage and how tightly the terms occur.

        BM25 rewards individual rare terms but does not explicitly require several
        distinctive terms to occur in the same passage. These bounded features make
        that evidence visible without replacing BM25 or introducing corpus-specific
        rules.
        """
        scores = np.zeros(self.document_count, dtype=np.float32)
        coverage_scores = np.zeros(self.document_count, dtype=np.float32)
        proximity_scores = np.zeros(self.document_count, dtype=np.float32)
        phrase_scores = np.zeros(self.document_count, dtype=np.float32)
        query_sequence = tuple(
            token
            for token in tokenize(query)
            if token not in QUERY_STOPWORDS and token in self.inverse_document_frequencies
        )
        query_terms = tuple(dict.fromkeys(query_sequence))
        if not query_terms:
            return LexicalEvidenceScores(
                score=scores,
                coverage=coverage_scores,
                proximity=proximity_scores,
                phrase=phrase_scores,
            )

        term_weights = {term: self.inverse_document_frequencies[term] for term in query_terms}
        total_weight = sum(term_weights.values())
        query_term_set = set(query_terms)
        query_bigrams = set(zip(query_sequence, query_sequence[1:], strict=False))
        selected_positions = (
            positions if positions is not None else list(range(self.document_count))
        )

        for position in selected_positions:
            document = self.document_tokens[position]
            matched = query_term_set.intersection(document)
            if not matched:
                continue
            coverage = sum(term_weights[term] for term in matched) / total_weight
            coverage_scores[position] = coverage

            if len(matched) >= 2:
                proximity_scores[position] = coverage * _term_proximity(document, matched)
            if query_bigrams:
                document_bigrams = set(zip(document, document[1:], strict=False))
                phrase_scores[position] = len(query_bigrams & document_bigrams) / len(query_bigrams)
            scores[position] = (
                0.60 * coverage_scores[position]
                + 0.30 * proximity_scores[position]
                + 0.10 * phrase_scores[position]
            )

        return LexicalEvidenceScores(
            score=scores,
            coverage=coverage_scores,
            proximity=proximity_scores,
            phrase=phrase_scores,
        )


@dataclass(frozen=True)
class LexicalEvidenceScores:
    score: np.ndarray
    coverage: np.ndarray
    proximity: np.ndarray
    phrase: np.ndarray


def _term_proximity(document: tuple[str, ...], matched_terms: set[str]) -> float:
    """Return matched-term density in the smallest span containing every term."""
    occurrences = [
        (offset, token) for offset, token in enumerate(document) if token in matched_terms
    ]
    counts: Counter[str] = Counter()
    covered = 0
    left = 0
    best_width = len(document) + 1
    for right_offset, right_term in occurrences:
        counts[right_term] += 1
        if counts[right_term] == 1:
            covered += 1
        while covered == len(matched_terms):
            left_offset, left_term = occurrences[left]
            best_width = min(best_width, right_offset - left_offset + 1)
            counts[left_term] -= 1
            if counts[left_term] == 0:
                covered -= 1
            left += 1
    return len(matched_terms) / best_width if best_width <= len(document) else 0.0
