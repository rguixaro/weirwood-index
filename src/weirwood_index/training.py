from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from weirwood_index.embedding import Encoder
from weirwood_index.evaluation import (
    Benchmark,
    BenchmarkCase,
    PassageTarget,
    chunk_matches_passage,
)
from weirwood_index.indexing import LoadedIndex
from weirwood_index.models import BenchmarkValidationError, Chunk
from weirwood_index.retrieval import search_index


def grouped_scene_folds(
    benchmark: Benchmark, *, folds: int = 5, seed: int = 17
) -> tuple[tuple[BenchmarkCase, ...], ...]:
    """Assign complete narrative scenes to deterministic, approximately stratified folds."""
    if folds < 2:
        raise BenchmarkValidationError("fold count must be at least 2")
    groups: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in benchmark:
        groups[case.scene_id or case.id].append(case)
    if len(groups) < folds:
        raise BenchmarkValidationError(
            "fold count cannot exceed the number of distinct scenes"
        )

    strata: defaultdict[tuple[Any, ...], list[tuple[str, list[BenchmarkCase]]]] = (
        defaultdict(list)
    )
    for scene_id, scene_cases in groups.items():
        books = tuple(
            sorted(
                {
                    chapter_id.split("-", maxsplit=1)[0]
                    for case in scene_cases
                    for chapter_id in case.expected_chapter_ids
                }
            )
        )
        styles = tuple(sorted({case.query_style for case in scene_cases}))
        name_usage = tuple(sorted({case.name_usage for case in scene_cases}))
        strata[(books, styles, name_usage)].append((scene_id, scene_cases))

    fold_groups: list[list[list[BenchmarkCase]]] = [[] for _ in range(folds)]
    fold_sizes = [0] * folds
    stratum_counts: defaultdict[tuple[Any, ...], list[int]] = defaultdict(
        lambda: [0] * folds
    )
    for stratum, scene_groups in sorted(strata.items(), key=lambda item: repr(item[0])):
        random.Random(f"{seed}:{stratum!r}").shuffle(scene_groups)
        for _, scene_cases in scene_groups:
            destination = min(
                range(folds),
                key=lambda fold: (
                    stratum_counts[stratum][fold],
                    fold_sizes[fold],
                    fold,
                ),
            )
            fold_groups[destination].append(scene_cases)
            fold_sizes[destination] += len(scene_cases)
            stratum_counts[stratum][destination] += 1

    original_order = {case.id: position for position, case in enumerate(benchmark)}
    return tuple(
        tuple(
            sorted(
                (case for scene_cases in scene_groups for case in scene_cases),
                key=lambda case: original_order[case.id],
            )
        )
        for scene_groups in fold_groups
    )


def benchmark_subset(
    benchmark: Benchmark,
    cases: Sequence[BenchmarkCase],
    *,
    name: str,
) -> Benchmark:
    return Benchmark(
        tuple(cases),
        source_sha256=benchmark.source_sha256,
        source_hashes=benchmark.source_hashes,
        name=name,
        split="development",
        status="draft",
        passage_match_strategy=benchmark.passage_match_strategy,
        minimum_overlap_words=benchmark.minimum_overlap_words,
    )


def mine_hard_negatives(
    index: LoadedIndex,
    benchmark: Benchmark,
    encoder: Encoder | None,
    *,
    negatives_per_case: int = 8,
    mode: str = "hybrid",
    semantic_weight: float = 0.5,
    context_vector_weight: float = 0.0,
    scene_window_weight: float = 0.0,
    scene_lexical_weight: float = 0.0,
    event_weight: float = 0.0,
    hierarchical: bool = True,
    narrative: bool = False,
    late_interaction: bool = False,
    chapter_candidates: int = 20,
    passages_per_chapter: int = 8,
    passage_candidate_pool: int = 200,
) -> dict[str, Any]:
    if benchmark.split == "acceptance":
        raise BenchmarkValidationError(
            "acceptance benchmarks must never be used for hard-negative mining"
        )
    if not 1 <= negatives_per_case <= 50:
        raise BenchmarkValidationError("negatives_per_case must be between 1 and 50")
    if not all(case.expected_passages for case in benchmark):
        raise BenchmarkValidationError(
            "hard-negative mining requires passage-level benchmark cases"
        )

    examples = [
        _mine_case(
            index,
            case,
            encoder,
            negatives_per_case=negatives_per_case,
            mode=mode,
            semantic_weight=semantic_weight,
            context_vector_weight=context_vector_weight,
            scene_window_weight=scene_window_weight,
            scene_lexical_weight=scene_lexical_weight,
            event_weight=event_weight,
            hierarchical=hierarchical,
            narrative=narrative,
            late_interaction=late_interaction,
            chapter_candidates=chapter_candidates,
            passages_per_chapter=passages_per_chapter,
            passage_candidate_pool=passage_candidate_pool,
            match_strategy=benchmark.passage_match_strategy,
            minimum_overlap_words=benchmark.minimum_overlap_words,
        )
        for case in benchmark
    ]
    return {
        "schema_version": 1,
        "purpose": "narrative-reranker-hard-negatives",
        "benchmark": {
            "name": benchmark.name,
            "split": benchmark.split,
            "status": benchmark.status,
            "source_sha256": benchmark.source_sha256,
            "source_hashes": dict(benchmark.source_hashes),
        },
        "index": str(index.path),
        "configuration": {
            "mode": mode,
            "semantic_weight": semantic_weight,
            "context_vector_weight": context_vector_weight,
            "scene_window_weight": scene_window_weight,
            "scene_lexical_weight": scene_lexical_weight,
            "event_weight": event_weight,
            "hierarchical": hierarchical,
            "narrative": narrative,
            "late_interaction": late_interaction,
            "negatives_per_case": negatives_per_case,
            "passage_matching": {
                "strategy": benchmark.passage_match_strategy,
                "minimum_overlap_words": benchmark.minimum_overlap_words,
            },
        },
        "examples": examples,
    }


def _mine_case(
    index: LoadedIndex,
    case: BenchmarkCase,
    encoder: Encoder | None,
    **options: Any,
) -> dict[str, Any]:
    negatives_per_case = int(options.pop("negatives_per_case"))
    mode = str(options.pop("mode"))
    semantic_weight = float(options.pop("semantic_weight"))
    context_vector_weight = float(options.pop("context_vector_weight"))
    scene_window_weight = float(options.pop("scene_window_weight"))
    scene_lexical_weight = float(options.pop("scene_lexical_weight"))
    event_weight = float(options.pop("event_weight"))
    hierarchical = bool(options.pop("hierarchical"))
    match_strategy = str(options.pop("match_strategy"))
    minimum_overlap_words = int(options.pop("minimum_overlap_words"))
    results = search_index(
        index,
        case.query,
        encoder,
        mode=mode,
        top=100,
        semantic_weight=semantic_weight,
        context_vector_weight=context_vector_weight,
        scene_window_weight=scene_window_weight,
        scene_lexical_weight=scene_lexical_weight,
        event_weight=event_weight,
        candidate_pool=100,
        hierarchical=hierarchical,
        **options,
    )
    positive_chunks = [
        _best_chunk_for_target(
            index.chunks,
            target,
            match_strategy=match_strategy,
            minimum_overlap_words=minimum_overlap_words,
        )
        for target in case.expected_passages
    ]
    positive_ids = {chunk.id for chunk in positive_chunks}
    negatives = [
        result
        for result in results
        if result.chunk.id not in positive_ids
        and not any(
            chunk_matches_passage(
                result.chunk,
                target,
                strategy=match_strategy,
                minimum_overlap_words=minimum_overlap_words,
            )
            for target in case.expected_passages
        )
    ]
    expected_chapters = set(case.expected_chapter_ids)
    expected_books = {chunk.book_id for chunk in positive_chunks}
    entity_terms = _query_entity_terms(case.query)
    same_chapter = [
        result for result in negatives if result.chunk.chapter_id in expected_chapters
    ]
    same_entities = [
        result
        for result in negatives
        if result.chunk.chapter_id not in expected_chapters
        and result.chunk.book_id in expected_books
        and _contains_entity_term(result.chunk.text, entity_terms)
    ]
    global_confusable = [
        result
        for result in negatives
        if result not in same_chapter and result not in same_entities
    ]
    selected: list[Any] = []
    selected_types: dict[str, str] = {}

    def take(bucket: list[Any], limit: int, negative_type: str) -> None:
        for result in bucket:
            if len(selected) >= negatives_per_case or limit <= 0:
                break
            if result in selected:
                continue
            selected.append(result)
            selected_types[result.chunk.id] = negative_type
            limit -= 1

    same_chapter_quota = max(1, negatives_per_case // 2)
    same_entity_quota = max(1, (negatives_per_case - same_chapter_quota) // 2)
    take(same_chapter, same_chapter_quota, "same-chapter")
    take(same_entities, same_entity_quota, "same-entity")
    take(global_confusable, negatives_per_case, "global")
    take(same_chapter, negatives_per_case, "same-chapter")
    take(same_entities, negatives_per_case, "same-entity")
    return {
        "id": case.id,
        "query": case.query,
        "category": case.category,
        "positive_chunk_ids": [chunk.id for chunk in positive_chunks],
        "negative_chunk_ids": [result.chunk.id for result in selected],
        "negatives": [
            {
                "chunk_id": result.chunk.id,
                "retrieval_rank": result.rank,
                "same_chapter": result.chunk.chapter_id in expected_chapters,
                "same_book": result.chunk.book_id in expected_books,
                "negative_type": selected_types[result.chunk.id],
            }
            for result in selected
        ],
    }


_GENERIC_CAPITALIZED_TERMS = {
    "a",
    "an",
    "and",
    "hand",
    "king",
    "kingsguard",
    "lord",
    "night",
    "queen",
    "red",
    "the",
    "wall",
    "watch",
}


def _query_entity_terms(query: str) -> frozenset[str]:
    return frozenset(
        token.casefold()
        for token in re.findall(r"\b[A-Z][A-Za-z'-]+\b", query)
        if token.casefold() not in _GENERIC_CAPITALIZED_TERMS
    )


def _contains_entity_term(text: str, entity_terms: frozenset[str]) -> bool:
    if not entity_terms:
        return False
    text_terms = {token.casefold() for token in re.findall(r"\b[A-Za-z'-]+\b", text)}
    return bool(text_terms & entity_terms)


def _best_chunk_for_target(
    chunks: Sequence[Chunk],
    target: PassageTarget,
    *,
    match_strategy: str = "midpoint",
    minimum_overlap_words: int = 20,
) -> Chunk:
    midpoint = (target.word_start + target.word_end - 1) / 2
    candidates = [
        chunk
        for chunk in chunks
        if chunk_matches_passage(
            chunk,
            target,
            strategy=match_strategy,
            minimum_overlap_words=minimum_overlap_words,
        )
    ]
    if not candidates:
        raise BenchmarkValidationError(
            f"no indexed chunk contains target in {target.chapter_id}"
        )
    return min(
        candidates,
        key=lambda chunk: (
            -max(
                0,
                min(chunk.word_end, target.word_end)
                - max(chunk.word_start, target.word_start),
            ),
            abs((chunk.word_start + chunk.word_end - 1) / 2 - midpoint),
            chunk.chunk_ordinal,
        ),
    )
