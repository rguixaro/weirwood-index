from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weirwood_index.embedding import Encoder
from weirwood_index.events import load_event_parser
from weirwood_index.indexing import LoadedIndex
from weirwood_index.lexical import BM25Index
from weirwood_index.models import BenchmarkValidationError, Chunk
from weirwood_index.narrative import QUERY_EXPANSION_VERSION
from weirwood_index.reranking import DEFAULT_RERANK_CANDIDATES, Reranker
from weirwood_index.retrieval import (
    DEFAULT_CHAPTER_CANDIDATES,
    DEFAULT_CHAPTER_EVIDENCE_PASSAGES,
    DEFAULT_CHAPTER_WEIGHT,
    DEFAULT_LEXICAL_EVIDENCE_WEIGHT,
    DEFAULT_NEIGHBOR_WEIGHT,
    DEFAULT_PASSAGE_CANDIDATE_POOL,
    DEFAULT_PASSAGES_PER_CHAPTER,
    DEFAULT_RETENTION_MODE,
    RetrievalTrace,
    SearchResult,
    hierarchical_hybrid_search_run,
    search_index,
)


@dataclass(frozen=True)
class PassageTarget:
    chapter_id: str
    word_start: int
    word_end: int
    relevance: int = 3

    def to_dict(self) -> dict[str, str | int]:
        return {
            "chapter_id": self.chapter_id,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "relevance": self.relevance,
        }


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    expected_chapter_ids: tuple[str, ...]
    category: str
    rationale: str
    expected_passages: tuple[PassageTarget, ...] = ()
    verification_status: str = "legacy"
    query_style: str = "unspecified"
    name_usage: str = "unspecified"
    scene_id: str | None = None


QUERY_STYLES = {
    "descriptive-search",
    "terse-search",
    "quotation-or-noisy",
}
NAME_USAGE_VALUES = {"named", "name-free"}
PASSAGE_MATCH_STRATEGIES = {"midpoint", "overlap"}


@dataclass(frozen=True)
class Benchmark(Sequence[BenchmarkCase]):
    cases: tuple[BenchmarkCase, ...]
    source_sha256: str | None = None
    source_hashes: tuple[tuple[str, str], ...] = ()
    name: str | None = None
    split: str = "development"
    status: str = "legacy"
    passage_match_strategy: str = "midpoint"
    minimum_overlap_words: int = 20

    def __getitem__(self, index: int) -> BenchmarkCase:
        return self.cases[index]

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[BenchmarkCase]:
        return iter(self.cases)


def load_benchmark(path: str | Path) -> Benchmark:
    benchmark_path = Path(path)
    if not benchmark_path.is_file():
        raise BenchmarkValidationError(f"benchmark file does not exist: {benchmark_path}")
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkValidationError(f"malformed benchmark JSON: {exc}") from exc
    source_sha256: str | None = None
    source_hashes: tuple[tuple[str, str], ...] = ()
    name: str | None = None
    split = "development"
    status = "legacy"
    schema_version = 1
    passage_match_strategy = "midpoint"
    minimum_overlap_words = 20
    if isinstance(payload, dict):
        schema_version = payload.get("schema_version")
        if schema_version not in {1, 2}:
            raise BenchmarkValidationError("benchmark schema_version must be 1 or 2")
        if schema_version == 2:
            name = payload.get("name")
            split = payload.get("split")
            status = payload.get("status")
            if not isinstance(name, str) or not name.strip():
                raise BenchmarkValidationError("schema 2 benchmark name must be non-empty")
            if split not in {"development", "acceptance"}:
                raise BenchmarkValidationError(
                    "schema 2 benchmark split must be development or acceptance"
                )
            if status not in {"draft", "sealed"}:
                raise BenchmarkValidationError(
                    "schema 2 benchmark status must be draft or sealed"
                )
        matching = payload.get("passage_matching")
        if matching is not None:
            if not isinstance(matching, dict):
                raise BenchmarkValidationError(
                    "benchmark passage_matching must be an object"
                )
            passage_match_strategy = matching.get("strategy")
            minimum_overlap_words = matching.get("minimum_overlap_words", 20)
            if passage_match_strategy not in PASSAGE_MATCH_STRATEGIES:
                raise BenchmarkValidationError(
                    "benchmark passage_matching strategy must be midpoint or overlap"
                )
            if (
                isinstance(minimum_overlap_words, bool)
                or not isinstance(minimum_overlap_words, int)
                or minimum_overlap_words < 1
            ):
                raise BenchmarkValidationError(
                    "benchmark passage_matching minimum_overlap_words must be positive"
                )
        corpus = payload.get("corpus")
        if not isinstance(corpus, dict):
            raise BenchmarkValidationError("benchmark corpus metadata must be an object")
        source_sha256 = corpus.get("source_sha256")
        source_payload = corpus.get("sources")
        if source_sha256 is not None:
            _validate_source_hash(source_sha256, "benchmark corpus source_sha256")
        if source_payload is not None:
            if not isinstance(source_payload, list) or not source_payload:
                raise BenchmarkValidationError(
                    "benchmark corpus sources must be a non-empty object array"
                )
            loaded_sources: list[tuple[str, str]] = []
            for number, item in enumerate(source_payload, start=1):
                if not isinstance(item, dict):
                    raise BenchmarkValidationError(
                        f"benchmark corpus source {number} must be an object"
                    )
                book_id = item.get("book_id")
                digest = item.get("source_sha256")
                if not isinstance(book_id, str) or not book_id:
                    raise BenchmarkValidationError(
                        f"benchmark corpus source {number} has an invalid book_id"
                    )
                _validate_source_hash(
                    digest, f"benchmark corpus source {number} source_sha256"
                )
                loaded_sources.append((book_id, digest))
            if len({book_id for book_id, _ in loaded_sources}) != len(loaded_sources):
                raise BenchmarkValidationError(
                    "benchmark corpus sources contain duplicate book_id values"
                )
            source_hashes = tuple(loaded_sources)
        if source_sha256 is None and not source_hashes:
            raise BenchmarkValidationError(
                "benchmark corpus must define source_sha256 or sources"
            )
        payload = payload.get("cases")
    if not isinstance(payload, list) or not payload:
        raise BenchmarkValidationError(
            "benchmark must be a non-empty JSON array or versioned object with cases"
        )

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    required = {"id", "query", "category", "rationale"}
    for number, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"case {number} must be an object")
        missing = sorted(required - item.keys())
        if missing:
            raise BenchmarkValidationError(
                f"case {number} is missing fields: {', '.join(missing)}"
            )
        case_id = item["id"]
        query = item["query"]
        expected = item.get("expected_chapter_ids")
        passage_payload = item.get("expected_passages")
        category = item["category"]
        rationale = item["rationale"]
        verification = item.get("verification")
        query_style = item.get("query_style", "unspecified")
        name_usage = item.get("name_usage", "unspecified")
        scene_id = item.get("scene_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise BenchmarkValidationError(f"case {number} has an invalid id")
        if case_id in seen_ids:
            raise BenchmarkValidationError(f"duplicate case id: {case_id}")
        if not isinstance(query, str) or not query.strip():
            raise BenchmarkValidationError(f"case {case_id} has an empty query")
        if expected is not None and (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(value, str) and value for value in expected)
        ):
            raise BenchmarkValidationError(
                f"case {case_id} expected_chapter_ids must be a non-empty string array"
            )
        passages = _load_passage_targets(case_id, passage_payload)
        if expected is None and not passages:
            raise BenchmarkValidationError(
                f"case {case_id} must define expected_chapter_ids or expected_passages"
            )
        passage_chapters = tuple(dict.fromkeys(target.chapter_id for target in passages))
        if expected is None:
            expected = list(passage_chapters)
        elif passages and set(expected) != set(passage_chapters):
            raise BenchmarkValidationError(
                f"case {case_id} expected_chapter_ids must match expected_passages chapters"
            )
        if not isinstance(category, str) or not category.strip():
            raise BenchmarkValidationError(f"case {case_id} has an invalid category")
        if not isinstance(rationale, str) or not rationale.strip():
            raise BenchmarkValidationError(f"case {case_id} has an invalid rationale")
        if query_style != "unspecified" and (
            not isinstance(query_style, str) or query_style not in QUERY_STYLES
        ):
            raise BenchmarkValidationError(
                f"case {case_id} has an invalid query_style: {query_style}"
            )
        if name_usage != "unspecified" and (
            not isinstance(name_usage, str) or name_usage not in NAME_USAGE_VALUES
        ):
            raise BenchmarkValidationError(
                f"case {case_id} has an invalid name_usage: {name_usage}"
            )
        if scene_id is not None and (
            not isinstance(scene_id, str) or not scene_id.strip()
        ):
            raise BenchmarkValidationError(f"case {case_id} has an invalid scene_id")
        has_query_form_metadata = (
            query_style != "unspecified"
            or name_usage != "unspecified"
            or scene_id is not None
        )
        if has_query_form_metadata and (
            query_style == "unspecified"
            or name_usage == "unspecified"
            or scene_id is None
        ):
            raise BenchmarkValidationError(
                f"case {case_id} must define query_style, name_usage, and scene_id together"
            )
        verification_status = "legacy"
        if schema_version == 2:
            if not isinstance(verification, dict):
                raise BenchmarkValidationError(
                    f"case {case_id} verification must be an object"
                )
            verification_status = verification.get("status")
            if verification_status not in {"draft", "verified"}:
                raise BenchmarkValidationError(
                    f"case {case_id} verification status must be draft or verified"
                )
            if status == "sealed" and verification_status != "verified":
                raise BenchmarkValidationError(
                    f"sealed benchmark case {case_id} must be verified"
                )
        seen_ids.add(case_id)
        cases.append(
            BenchmarkCase(
                id=case_id,
                query=" ".join(query.split()),
                expected_chapter_ids=tuple(expected),
                category=category,
                rationale=rationale,
                expected_passages=passages,
                verification_status=verification_status,
                query_style=query_style,
                name_usage=name_usage,
                scene_id=scene_id,
            )
        )
    passage_cases = sum(bool(case.expected_passages) for case in cases)
    if passage_cases not in {0, len(cases)}:
        raise BenchmarkValidationError(
            "benchmark must define expected_passages for every case or for none"
        )
    return Benchmark(
        tuple(cases),
        source_sha256,
        source_hashes,
        name=name,
        split=split,
        status=status,
        passage_match_strategy=passage_match_strategy,
        minimum_overlap_words=minimum_overlap_words,
    )


def _validate_source_hash(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BenchmarkValidationError(
            f"{label} must be 64 lowercase hex characters"
        )


def load_benchmarks(paths: Sequence[str | Path]) -> Benchmark:
    benchmarks = [load_benchmark(path) for path in paths]
    if not benchmarks:
        raise BenchmarkValidationError("at least one benchmark file is required")
    cases = tuple(case for benchmark in benchmarks for case in benchmark)
    case_ids = [case.id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        duplicates = sorted(
            case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
        )
        raise BenchmarkValidationError(
            f"duplicate case ids across benchmarks: {', '.join(duplicates)}"
        )
    legacy_hashes = {
        benchmark.source_sha256
        for benchmark in benchmarks
        if benchmark.source_sha256 is not None
    }
    if len(legacy_hashes) > 1:
        raise BenchmarkValidationError(
            "combined benchmarks must identify per-book hashes with corpus sources"
        )
    source_hashes: dict[str, str] = {}
    for benchmark in benchmarks:
        for book_id, digest in benchmark.source_hashes:
            prior = source_hashes.setdefault(book_id, digest)
            if prior != digest:
                raise BenchmarkValidationError(
                    f"conflicting source hashes for benchmark book {book_id}"
                )
    matching_configurations = {
        (benchmark.passage_match_strategy, benchmark.minimum_overlap_words)
        for benchmark in benchmarks
    }
    if len(matching_configurations) > 1:
        raise BenchmarkValidationError(
            "combined benchmarks must use the same passage_matching configuration"
        )
    passage_match_strategy, minimum_overlap_words = next(
        iter(matching_configurations)
    )
    return Benchmark(
        cases,
        next(iter(legacy_hashes), None),
        tuple(sorted(source_hashes.items())),
        name="combined",
        split=(
            "acceptance"
            if all(benchmark.split == "acceptance" for benchmark in benchmarks)
            else "development"
        ),
        status=(
            "sealed"
            if all(benchmark.status == "sealed" for benchmark in benchmarks)
            else "draft"
        ),
        passage_match_strategy=passage_match_strategy,
        minimum_overlap_words=minimum_overlap_words,
    )


def _load_passage_targets(
    case_id: str, payload: Any
) -> tuple[PassageTarget, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list) or not payload:
        raise BenchmarkValidationError(
            f"case {case_id} expected_passages must be a non-empty object array"
        )
    targets: list[PassageTarget] = []
    for number, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(
                f"case {case_id} passage {number} must be an object"
            )
        missing = {"chapter_id", "word_start", "word_end"} - item.keys()
        if missing:
            raise BenchmarkValidationError(
                f"case {case_id} passage {number} is missing fields: "
                f"{', '.join(sorted(missing))}"
            )
        chapter_id = item["chapter_id"]
        word_start = item["word_start"]
        word_end = item["word_end"]
        relevance = item.get("relevance", 3)
        if not isinstance(chapter_id, str) or not chapter_id:
            raise BenchmarkValidationError(
                f"case {case_id} passage {number} has an invalid chapter_id"
            )
        if (
            isinstance(word_start, bool)
            or not isinstance(word_start, int)
            or isinstance(word_end, bool)
            or not isinstance(word_end, int)
            or word_start < 0
            or word_end <= word_start
        ):
            raise BenchmarkValidationError(
                f"case {case_id} passage {number} must satisfy "
                "0 <= word_start < word_end"
            )
        if isinstance(relevance, bool) or not isinstance(relevance, int) or not 1 <= relevance <= 3:
            raise BenchmarkValidationError(
                f"case {case_id} passage {number} relevance must be 1, 2, or 3"
            )
        targets.append(PassageTarget(chapter_id, word_start, word_end, relevance))
    if len(set(targets)) != len(targets):
        raise BenchmarkValidationError(f"case {case_id} has duplicate expected_passages")
    return tuple(targets)


def _aggregate(
    rows: list[dict[str, Any]], *, rank_field: str = "rank"
) -> dict[str, Any]:
    total = len(rows)
    ranks = [row[rank_field] for row in rows]
    metrics: dict[str, Any] = {
        "cases": total,
        # Preserve the original cutoff so historical MRR values remain comparable.
        "mrr": sum(
            0.0 if rank is None or rank > 10 else 1.0 / rank for rank in ranks
        )
        / total,
        "mrr_at_50": sum(
            0.0 if rank is None or rank > 50 else 1.0 / rank for rank in ranks
        )
        / total,
    }
    for cutoff in (1, 3, 5, 10, 20, 50):
        successes = sum(rank is not None and rank <= cutoff for rank in ranks)
        hit_rate = successes / total
        lower, upper = _wilson_interval(successes, total)
        metrics[f"hit_at_{cutoff}"] = hit_rate
        # Backwards-compatible alias. For a single expected passage this metric
        # has always been Hit@K rather than classical multi-relevant recall.
        metrics[f"recall_at_{cutoff}"] = hit_rate
        metrics[f"hit_at_{cutoff}_ci95"] = {
            "lower": lower,
            "upper": upper,
        }
        target_key = f"target_recall_at_{cutoff}"
        if rows and target_key in rows[0]:
            metrics[target_key] = sum(float(row[target_key]) for row in rows) / total
    for cutoff in (5, 10):
        key = f"ndcg_at_{cutoff}"
        if rows and key in rows[0]:
            metrics[key] = sum(float(row[key]) for row in rows) / total
    return metrics


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def _discounted_gain(grades: Sequence[int], cutoff: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:cutoff], start=1)
    )


def _passage_relevance_metrics(
    results: Sequence[SearchResult],
    targets: Sequence[PassageTarget],
    *,
    match_strategy: str = "midpoint",
    minimum_overlap_words: int = 20,
) -> dict[str, Any]:
    target_ranks: list[int | None] = [None] * len(targets)
    credited: set[int] = set()
    result_grades: list[int] = []
    for result in results:
        matching = [
            position
            for position, target in enumerate(targets)
            if _result_matches_target(
                result,
                target,
                strategy=match_strategy,
                minimum_overlap_words=minimum_overlap_words,
            )
        ]
        for position in matching:
            if target_ranks[position] is None:
                target_ranks[position] = result.rank
        newly_credited = [position for position in matching if position not in credited]
        result_grades.append(
            max((targets[position].relevance for position in newly_credited), default=0)
        )
        credited.update(newly_credited)

    metrics: dict[str, Any] = {"target_ranks": target_ranks}
    for cutoff in (1, 3, 5, 10, 20, 50):
        metrics[f"target_recall_at_{cutoff}"] = (
            sum(rank is not None and rank <= cutoff for rank in target_ranks)
            / len(targets)
        )
    ideal_grades = sorted((target.relevance for target in targets), reverse=True)
    for cutoff in (5, 10):
        ideal = _discounted_gain(ideal_grades, cutoff)
        metrics[f"ndcg_at_{cutoff}"] = (
            _discounted_gain(result_grades, cutoff) / ideal if ideal else 0.0
        )
    return metrics


def evaluate_benchmark(
    index: LoadedIndex,
    cases: Sequence[BenchmarkCase],
    encoder: Encoder | None,
    *,
    mode: str = "semantic",
    semantic_weight: float = 0.5,
    candidate_pool: int = 50,
    deduplicate_chapters: bool = False,
    reranker: Reranker | None = None,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    rerank_context_words: int = 0,
    rerank_fusion_weight: float = 1.0,
    hierarchical: bool = False,
    chapter_candidates: int = DEFAULT_CHAPTER_CANDIDATES,
    passages_per_chapter: int = DEFAULT_PASSAGES_PER_CHAPTER,
    passage_candidate_pool: int = DEFAULT_PASSAGE_CANDIDATE_POOL,
    chapter_evidence_passages: int = DEFAULT_CHAPTER_EVIDENCE_PASSAGES,
    chapter_weight: float = DEFAULT_CHAPTER_WEIGHT,
    neighbor_weight: float = DEFAULT_NEIGHBOR_WEIGHT,
    narrative: bool = False,
    late_interaction: bool = False,
    context_vector_weight: float = 0.0,
    scene_window_weight: float = 0.0,
    scene_lexical_weight: float = 0.0,
    lexical_evidence_weight: float = DEFAULT_LEXICAL_EVIDENCE_WEIGHT,
    retention_mode: str = DEFAULT_RETENTION_MODE,
    event_weight: float = 0.0,
) -> dict[str, Any]:
    if not cases:
        raise BenchmarkValidationError("benchmark must contain at least one case")
    if hierarchical and mode != "hybrid":
        raise BenchmarkValidationError(
            "hierarchical evaluation currently requires hybrid mode"
        )
    if hierarchical and deduplicate_chapters:
        raise BenchmarkValidationError(
            "hierarchical evaluation cannot deduplicate chapters"
        )
    if narrative and not index.narrative_views:
        raise BenchmarkValidationError(
            "narrative evaluation requires an index built with narrative views"
        )
    if late_interaction and not narrative:
        raise BenchmarkValidationError(
            "late interaction evaluation requires narrative retrieval"
        )
    benchmark_source_sha256 = (
        cases.source_sha256 if isinstance(cases, Benchmark) else None
    )
    benchmark_source_hashes = (
        dict(cases.source_hashes) if isinstance(cases, Benchmark) else {}
    )
    benchmark_name = cases.name if isinstance(cases, Benchmark) else None
    benchmark_split = cases.split if isinstance(cases, Benchmark) else "development"
    benchmark_status = cases.status if isinstance(cases, Benchmark) else "legacy"
    passage_match_strategy = (
        cases.passage_match_strategy if isinstance(cases, Benchmark) else "midpoint"
    )
    minimum_overlap_words = (
        cases.minimum_overlap_words if isinstance(cases, Benchmark) else 20
    )
    index_source = index.manifest["source"]
    index_records = index_source.get("sources") or [index_source]
    index_hashes = {
        record.get("book_id"): record.get("sha256")
        for record in index_records
        if isinstance(record, dict) and record.get("book_id")
    }
    available_hashes = {
        record.get("sha256")
        for record in index_records
        if isinstance(record, dict) and record.get("sha256")
    }
    available_hashes.add(index_source.get("sha256"))
    if benchmark_source_sha256 is not None and benchmark_source_sha256 not in available_hashes:
        raise BenchmarkValidationError(
            "benchmark corpus hash does not match the index source hash"
        )
    for book_id, digest in benchmark_source_hashes.items():
        if index_hashes.get(book_id) != digest:
            raise BenchmarkValidationError(
                f"benchmark source hash for {book_id} does not match the index"
            )
    known_chapters = {chunk.chapter_id for chunk in index.chunks}
    chapter_word_counts: dict[str, int] = {}
    for chunk in index.chunks:
        chapter_word_counts[chunk.chapter_id] = max(
            chapter_word_counts.get(chunk.chapter_id, 0), chunk.word_end
        )
    for case in cases:
        unknown = sorted(set(case.expected_chapter_ids) - known_chapters)
        if unknown:
            raise BenchmarkValidationError(
                f"case {case.id} references chapters not in this index: {', '.join(unknown)}"
            )
        for target in case.expected_passages:
            if target.word_end > chapter_word_counts[target.chapter_id]:
                raise BenchmarkValidationError(
                    f"case {case.id} passage ends at word {target.word_end}, beyond "
                    f"{target.chapter_id}'s {chapter_word_counts[target.chapter_id]} words"
                )

    evaluation_level = "passage" if cases[0].expected_passages else "chapter"

    lexical_index = None
    if mode in {"lexical", "hybrid"}:
        lexical_index = (
            BM25Index.from_texts([view.lexical_text for view in index.narrative_views])
            if narrative and index.narrative_views
            else BM25Index.from_chunks(index.chunks)
        )
    scene_lexical_index = None
    if scene_lexical_weight:
        if not index.scene_windows:
            raise BenchmarkValidationError(
                "scene lexical evaluation requires an index enriched with scene windows"
            )
        scene_lexical_index = BM25Index.from_texts(
            [window.lexical_text for window in index.scene_windows]
        )
    event_lexical_index = None
    event_parser = None
    if event_weight:
        if not index.event_records:
            raise BenchmarkValidationError(
                "event evaluation requires an index enriched with structured events"
            )
        event_lexical_index = BM25Index.from_texts(
            [record.lexical_text for record in index.event_records]
        )
        event_parser = load_event_parser()
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        if hierarchical:
            if encoder is None:
                raise BenchmarkValidationError(
                    "hierarchical evaluation requires an embedding encoder"
                )
            run = hierarchical_hybrid_search_run(
                index,
                case.query,
                encoder,
                top=50,
                semantic_weight=semantic_weight,
                chapter_candidates=chapter_candidates,
                passages_per_chapter=passages_per_chapter,
                passage_candidate_pool=passage_candidate_pool,
                chapter_evidence_passages=chapter_evidence_passages,
                chapter_weight=chapter_weight,
                neighbor_weight=neighbor_weight,
                lexical_index=lexical_index,
                reranker=reranker,
                rerank_candidates=rerank_candidates,
                rerank_context_words=rerank_context_words,
                rerank_fusion_weight=rerank_fusion_weight,
                narrative=narrative,
                late_interaction=late_interaction,
                context_vector_weight=context_vector_weight,
                scene_window_weight=scene_window_weight,
                scene_lexical_weight=scene_lexical_weight,
                scene_lexical_index=scene_lexical_index,
                lexical_evidence_weight=lexical_evidence_weight,
                retention_mode=retention_mode,
                event_weight=event_weight,
                event_lexical_index=event_lexical_index,
                event_parser=event_parser,
            )
            results = run.results
            trace = run.trace
        else:
            results = search_index(
                index,
                case.query,
                encoder,
                mode=mode,
                top=50,
                semantic_weight=semantic_weight,
                candidate_pool=candidate_pool,
                deduplicate_chapters=deduplicate_chapters,
                lexical_index=lexical_index,
                reranker=reranker,
                rerank_candidates=rerank_candidates,
                rerank_context_words=rerank_context_words,
                rerank_fusion_weight=rerank_fusion_weight,
                narrative=narrative,
                late_interaction=late_interaction,
                context_vector_weight=context_vector_weight,
                scene_window_weight=scene_window_weight,
                scene_lexical_weight=scene_lexical_weight,
                scene_lexical_index=scene_lexical_index,
                lexical_evidence_weight=lexical_evidence_weight,
                retention_mode=retention_mode,
                event_weight=event_weight,
                event_lexical_index=event_lexical_index,
                event_parser=event_parser,
            )
            trace = _trace_from_results(results)
        latency_ms = (time.perf_counter() - started) * 1000.0
        chapter_rank = next(
            (
                result.rank
                for result in results
                if result.chunk.chapter_id in case.expected_chapter_ids
            ),
            None,
        )
        relevance_metrics = (
            _passage_relevance_metrics(
                results,
                case.expected_passages,
                match_strategy=passage_match_strategy,
                minimum_overlap_words=minimum_overlap_words,
            )
            if case.expected_passages
            else {}
        )
        target_ranks = relevance_metrics.get("target_ranks", [])
        passage_rank = min(
            (rank for rank in target_ranks if rank is not None),
            default=None,
        )
        rank = passage_rank if evaluation_level == "passage" else chapter_rank
        chapter_shortlist_rank = next(
            (
                rank
                for rank, chapter_id in enumerate(trace.chapter_ranking, start=1)
                if chapter_id in case.expected_chapter_ids
            ),
            None,
        )
        target_chunk_ids_by_target = [
            {
                chunk.id
                for chunk in index.chunks
                if chunk_matches_passage(
                    chunk,
                    target,
                    strategy=passage_match_strategy,
                    minimum_overlap_words=minimum_overlap_words,
                )
            }
            for target in case.expected_passages
        ]
        target_chunk_ids = set().union(*target_chunk_ids_by_target)
        candidate_covered = bool(target_chunk_ids & trace.candidate_chunk_ids)
        retained_covered = bool(target_chunk_ids & trace.retained_chunk_ids)
        candidate_target_recall = (
            sum(
                bool(chunk_ids & trace.candidate_chunk_ids)
                for chunk_ids in target_chunk_ids_by_target
            )
            / len(target_chunk_ids_by_target)
            if target_chunk_ids_by_target
            else 0.0
        )
        retained_target_recall = (
            sum(
                bool(chunk_ids & trace.retained_chunk_ids)
                for chunk_ids in target_chunk_ids_by_target
            )
            / len(target_chunk_ids_by_target)
            if target_chunk_ids_by_target
            else 0.0
        )
        within_chapter_rank = min(
            (
                trace.within_chapter_ranks[chunk_id]
                for chunk_id in target_chunk_ids
                if chunk_id in trace.within_chapter_ranks
            ),
            default=None,
        )
        passed_at_5 = rank is not None and rank <= 5
        if passed_at_5:
            failure_stage = "pass"
        elif not candidate_covered:
            failure_stage = (
                "wrong-chapter"
                if chapter_shortlist_rank is None
                or chapter_shortlist_rank > chapter_candidates
                else "target-absent"
            )
        else:
            failure_stage = "target-misranked"
        row = {
                "id": case.id,
                "query": case.query,
                "category": case.category,
                "query_style": case.query_style,
                "name_usage": case.name_usage,
                "scene_id": case.scene_id,
                "verification_status": case.verification_status,
                "expected_chapter_ids": list(case.expected_chapter_ids),
                "expected_passages": [
                    target.to_dict() for target in case.expected_passages
                ],
                "chapter_rank": chapter_rank,
                "passage_rank": passage_rank,
                "rank": rank,
                "chapter_shortlist_rank": chapter_shortlist_rank,
                "passage_candidate_covered": candidate_covered,
                "passage_retained_covered": retained_covered,
                "candidate_target_recall": candidate_target_recall,
                "retained_target_recall": retained_target_recall,
                "within_chapter_rank": within_chapter_rank,
                "failure_stage": failure_stage,
                "latency_ms": round(latency_ms, 3),
                "top_chapter_id": results[0].chunk.chapter_id if results else None,
                "top_chunk_id": results[0].chunk.id if results else None,
                "passed_at_5": passed_at_5,
            }
        row.update(relevance_metrics)
        rows.append(row)

    categories: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    query_styles: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    name_usage_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
        if row["query_style"] != "unspecified":
            query_styles[row["query_style"]].append(row)
        if row["name_usage"] != "unspecified":
            name_usage_groups[row["name_usage"]].append(row)
    selected = [
        row
        for row in rows
        if row["name_usage"] == "name-free"
        or row["category"] in {"name-free", "paraphrased-quotation"}
    ]
    failures = []
    for row in rows:
        if row["passed_at_5"]:
            continue
        failure = dict(row)
        failure["false_positive_review"] = {
            "thematically_related": None,
            "wrong_actor": None,
            "wrong_direction": None,
            "wrong_outcome": None,
            "notes": "",
        }
        failures.append(failure)
    return {
        "configuration": {
            "index": str(index.path),
            "embedding_model": index.manifest["model"],
            "chunk_profile": index.manifest["chunk_profile"],
            "mode": mode,
            "semantic_weight": semantic_weight if mode == "hybrid" else None,
            "lexical_weight": 1.0 - semantic_weight if mode == "hybrid" else None,
            "candidate_pool": candidate_pool if mode == "hybrid" else None,
            "deduplicate_chapters": deduplicate_chapters,
            "reranker": (
                {
                    "kind": getattr(reranker, "kind", "custom"),
                    "model": reranker.model_id,
                    "revision": reranker.revision,
                    "candidates": rerank_candidates,
                    "context_words": rerank_context_words,
                    "fusion_weight": rerank_fusion_weight,
                }
                if reranker is not None
                else None
            ),
            "evaluation_level": evaluation_level,
            "benchmark_source_sha256": benchmark_source_sha256,
            "benchmark_source_hashes": benchmark_source_hashes,
            "benchmark_name": benchmark_name,
            "benchmark_split": benchmark_split,
            "benchmark_status": benchmark_status,
            "passage_matching": {
                "strategy": passage_match_strategy,
                "minimum_overlap_words": minimum_overlap_words,
            },
            "hierarchical": hierarchical,
            "chapter_candidates": chapter_candidates if hierarchical else None,
            "passages_per_chapter": passages_per_chapter if hierarchical else None,
            "passage_candidate_pool": (
                passage_candidate_pool if hierarchical else None
            ),
            "chapter_evidence_passages": (
                chapter_evidence_passages if hierarchical else None
            ),
            "chapter_weight": chapter_weight if hierarchical else None,
            "neighbor_weight": neighbor_weight if hierarchical else None,
            "retention_mode": retention_mode if hierarchical else None,
            "narrative": narrative,
            "late_interaction": late_interaction,
            "context_vector_weight": context_vector_weight,
            "context_vector_fusion": (
                "weighted-max-uplift" if context_vector_weight else None
            ),
            "scene_window_weight": scene_window_weight,
            "scene_window_fusion": (
                "weighted-max-uplift" if scene_window_weight else None
            ),
            "scene_lexical_weight": scene_lexical_weight,
            "lexical_evidence_weight": lexical_evidence_weight,
            "event_weight": event_weight,
            "event_index": index.manifest.get("event_index") if event_weight else None,
            "query_expansion_version": QUERY_EXPANSION_VERSION if narrative else None,
        },
        "overall": _aggregate(rows),
        "chapter_overall": _aggregate(rows, rank_field="chapter_rank"),
        "passage_overall": (
            _aggregate(rows, rank_field="passage_rank")
            if evaluation_level == "passage"
            else None
        ),
        "stages": (
            _aggregate_stages(rows, chapter_candidates=chapter_candidates)
            if evaluation_level == "passage" and hierarchical
            else None
        ),
        "latency_ms": _aggregate_latency(rows),
        "by_category": {
            category: _aggregate(category_rows)
            for category, category_rows in sorted(categories.items())
        },
        "by_query_style": {
            query_style: _aggregate(style_rows)
            for query_style, style_rows in sorted(query_styles.items())
        },
        "by_name_usage": {
            name_usage: _aggregate(group_rows)
            for name_usage, group_rows in sorted(name_usage_groups.items())
        },
        "name_free_and_paraphrased": _aggregate(selected) if selected else None,
        "failures": failures,
        "cases": rows,
    }


def _aggregate_stages(
    rows: list[dict[str, Any]], *, chapter_candidates: int
) -> dict[str, float | int]:
    total = len(rows)
    shortlisted = [row for row in rows if row["passage_candidate_covered"]]
    within_ranks = [row["within_chapter_rank"] for row in shortlisted]
    return {
        "cases": total,
        "chapter_recall_at_1": sum(
            row["chapter_shortlist_rank"] is not None
            and row["chapter_shortlist_rank"] <= 1
            for row in rows
        )
        / total,
        "chapter_recall_at_5": sum(
            row["chapter_shortlist_rank"] is not None
            and row["chapter_shortlist_rank"] <= 5
            for row in rows
        )
        / total,
        "chapter_recall_at_10": sum(
            row["chapter_shortlist_rank"] is not None
            and row["chapter_shortlist_rank"] <= 10
            for row in rows
        )
        / total,
        "chapter_recall_at_20": sum(
            row["chapter_shortlist_rank"] is not None
            and row["chapter_shortlist_rank"] <= 20
            for row in rows
        )
        / total,
        "passage_candidate_coverage": len(shortlisted) / total,
        "candidate_target_recall": sum(
            float(row["candidate_target_recall"]) for row in rows
        )
        / total,
        "passage_retained_coverage": sum(
            row["passage_retained_covered"] for row in rows
        )
        / total,
        "retained_target_recall": sum(
            float(row["retained_target_recall"]) for row in rows
        )
        / total,
        "chapter_shortlist_oracle_hit_at_5": sum(
            row["chapter_shortlist_rank"] is not None
            and row["chapter_shortlist_rank"] <= chapter_candidates
            for row in rows
        )
        / total,
        "candidate_pool_oracle_hit_at_5": len(shortlisted) / total,
        "retained_pool_oracle_hit_at_5": sum(
            row["passage_retained_covered"] for row in rows
        )
        / total,
        "conditional_passage_recall_at_5": (
            sum(row["passage_rank"] is not None and row["passage_rank"] <= 5 for row in shortlisted)
            / len(shortlisted)
            if shortlisted
            else 0.0
        ),
        "within_chapter_recall_at_1": (
            sum(rank is not None and rank <= 1 for rank in within_ranks)
            / len(shortlisted)
            if shortlisted
            else 0.0
        ),
        "within_chapter_recall_at_3": (
            sum(rank is not None and rank <= 3 for rank in within_ranks)
            / len(shortlisted)
            if shortlisted
            else 0.0
        ),
        "within_chapter_recall_at_5": (
            sum(rank is not None and rank <= 5 for rank in within_ranks)
            / len(shortlisted)
            if shortlisted
            else 0.0
        ),
        "wrong_chapter_failures": sum(
            row["failure_stage"] == "wrong-chapter" for row in rows
        ),
        "target_absent_failures": sum(
            row["failure_stage"] == "target-absent" for row in rows
        ),
        "target_misranked_failures": sum(
            row["failure_stage"] == "target-misranked" for row in rows
        ),
    }


def _aggregate_latency(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = sorted(float(row["latency_ms"]) for row in rows)

    def percentile(fraction: float) -> float:
        index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
        return values[index]

    return {
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": values[-1],
    }


def _trace_from_results(results: list[SearchResult]) -> RetrievalTrace:
    chapter_ranking = tuple(dict.fromkeys(result.chunk.chapter_id for result in results))
    within_counts: defaultdict[str, int] = defaultdict(int)
    within_ranks: dict[str, int] = {}
    for result in results:
        within_counts[result.chunk.chapter_id] += 1
        within_ranks[result.chunk.id] = within_counts[result.chunk.chapter_id]
    chunk_ids = frozenset(result.chunk.id for result in results)
    return RetrievalTrace(chapter_ranking, chunk_ids, chunk_ids, within_ranks)


def _interval_matches_passage(
    start: int,
    end: int,
    target: PassageTarget,
    *,
    strategy: str,
    minimum_overlap_words: int,
) -> bool:
    if strategy == "midpoint":
        midpoint = (target.word_start + target.word_end - 1) // 2
        return start <= midpoint < end
    if strategy != "overlap":
        raise BenchmarkValidationError(f"unsupported passage match strategy: {strategy}")
    overlap = max(0, min(end, target.word_end) - max(start, target.word_start))
    required = min(minimum_overlap_words, target.word_end - target.word_start)
    return overlap >= required


def _result_matches_target(
    result: SearchResult,
    target: PassageTarget,
    *,
    strategy: str,
    minimum_overlap_words: int,
) -> bool:
    if result.chunk.chapter_id != target.chapter_id:
        return False
    start = (
        result.context_word_start
        if result.context_word_start is not None
        else result.chunk.word_start
    )
    end = (
        result.context_word_end
        if result.context_word_end is not None
        else result.chunk.word_end
    )
    return _interval_matches_passage(
        start,
        end,
        target,
        strategy=strategy,
        minimum_overlap_words=minimum_overlap_words,
    )


def _result_contains_target_midpoint(
    result: SearchResult, target: PassageTarget
) -> bool:
    return _result_matches_target(
        result,
        target,
        strategy="midpoint",
        minimum_overlap_words=20,
    )


def chunk_matches_passage(
    chunk: Chunk,
    target: PassageTarget,
    *,
    strategy: str = "midpoint",
    minimum_overlap_words: int = 20,
) -> bool:
    if chunk.chapter_id != target.chapter_id:
        return False
    return _interval_matches_passage(
        chunk.word_start,
        chunk.word_end,
        target,
        strategy=strategy,
        minimum_overlap_words=minimum_overlap_words,
    )


def _chunk_contains_target_midpoint(chunk: Chunk, target: PassageTarget) -> bool:
    return chunk_matches_passage(chunk, target)


def decision_gate(report: dict[str, Any]) -> str:
    overall = report["overall"]
    subset = report["name_free_and_paraphrased"]
    if (
        overall["recall_at_5"] >= 0.70
        and overall["recall_at_10"] >= 0.80
        and subset is not None
        and subset["recall_at_5"] >= 0.60
    ):
        return "proceed"
    if overall["recall_at_5"] >= 0.50:
        return "investigate-once"
    return "stop-or-rethink"
