from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from weirwood_index.chunking import PROFILES
from weirwood_index.evaluation import (
    Benchmark,
    BenchmarkCase,
    PassageTarget,
    _chunk_contains_target_midpoint,
    chunk_matches_passage,
    evaluate_benchmark,
    load_benchmark,
    load_benchmarks,
)
from weirwood_index.indexing import build_index, load_index
from weirwood_index.models import BenchmarkValidationError, Chunk
from weirwood_index.retrieval import semantic_search

from .helpers import FakeEncoder, write_valid_source


def test_load_passage_benchmark_derives_chapter_ids(tmp_path) -> None:
    path = tmp_path / "passages.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus": {"source_sha256": "a" * 64},
                "cases": [
                    {
                        "id": "case-1",
                        "query": "a remembered event",
                        "expected_passages": [
                            {
                                "chapter_id": "agot-001-prologue",
                                "word_start": 10,
                                "word_end": 30,
                            }
                        ],
                        "category": "name-free",
                        "rationale": "Synthetic passage benchmark.",
                    }
                ],
            }
        )
    )

    cases = load_benchmark(path)

    assert cases[0].expected_chapter_ids == ("agot-001-prologue",)
    assert cases[0].expected_passages == (
        PassageTarget("agot-001-prologue", 10, 30),
    )
    assert cases.source_sha256 == "a" * 64


def test_load_benchmark_preserves_search_form_metadata(tmp_path) -> None:
    path = tmp_path / "natural.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": "natural-development",
                "split": "development",
                "status": "draft",
                "corpus": {"source_sha256": "a" * 64},
                "cases": [
                    {
                        "id": "natural-1",
                        "scene_id": "scene-1",
                        "query": "Arya throws the sword in the river",
                        "query_style": "descriptive-search",
                        "name_usage": "named",
                        "expected_passages": [
                            {
                                "chapter_id": "agot-001-prologue",
                                "word_start": 10,
                                "word_end": 30,
                            }
                        ],
                        "category": "representative",
                        "rationale": "Synthetic passage benchmark.",
                        "verification": {"status": "draft"},
                    }
                ],
            }
        )
    )

    benchmark = load_benchmark(path)

    assert benchmark[0].query_style == "descriptive-search"
    assert benchmark[0].name_usage == "named"
    assert benchmark[0].scene_id == "scene-1"


@pytest.mark.parametrize(
    "passage",
    [
        {"chapter_id": "agot-001-prologue", "word_start": -1, "word_end": 5},
        {"chapter_id": "agot-001-prologue", "word_start": 5, "word_end": 5},
        {"chapter_id": "agot-001-prologue", "word_start": 8, "word_end": 4},
    ],
)
def test_load_passage_benchmark_rejects_invalid_ranges(tmp_path, passage) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "query": "event",
                    "expected_passages": [passage],
                    "category": "name-free",
                    "rationale": "Synthetic passage benchmark.",
                }
            ]
        )
    )

    with pytest.raises(BenchmarkValidationError, match="word_start < word_end"):
        load_benchmark(path)


def test_passage_match_uses_target_midpoint() -> None:
    chunk = Chunk(
        id="chunk",
        chapter_id="chapter",
        chapter_title="CHAPTER",
        chapter_sequence=1,
        pov="POV",
        pov_ordinal=1,
        chunk_ordinal=1,
        word_start=100,
        word_end=200,
        text="passage",
    )

    assert _chunk_contains_target_midpoint(chunk, PassageTarget("chapter", 150, 170))
    assert not _chunk_contains_target_midpoint(
        chunk, PassageTarget("chapter", 190, 250)
    )


def test_passage_match_can_require_meaningful_scene_overlap() -> None:
    chunk = Chunk(
        id="chunk",
        chapter_id="chapter",
        chapter_title="CHAPTER",
        chapter_sequence=1,
        pov="POV",
        pov_ordinal=1,
        chunk_ordinal=1,
        word_start=100,
        word_end=200,
        text="passage",
    )
    target = PassageTarget("chapter", 190, 260)

    assert not chunk_matches_passage(
        chunk, target, strategy="overlap", minimum_overlap_words=20
    )
    assert chunk_matches_passage(
        chunk, target, strategy="overlap", minimum_overlap_words=10
    )


def test_passage_evaluation_distinguishes_right_chapter_from_right_span(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=500)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    index = load_index(built.path)
    query = "remembered scene"
    results = semantic_search(index, query, FakeEncoder(), top=50)
    first_by_chapter = {}
    later_result = None
    for result in results:
        first = first_by_chapter.setdefault(result.chunk.chapter_id, result)
        if first is not result:
            later_result = result
            break
    assert later_result is not None
    target = PassageTarget(
        later_result.chunk.chapter_id,
        later_result.chunk.word_start,
        later_result.chunk.word_end,
    )
    case = BenchmarkCase(
        id="case-1",
        query=query,
        expected_chapter_ids=(later_result.chunk.chapter_id,),
        category="name-free",
        rationale="Synthetic passage evaluation.",
        expected_passages=(target,),
    )

    report = evaluate_benchmark(index, (case,), FakeEncoder())
    row = report["cases"][0]

    assert report["configuration"]["evaluation_level"] == "passage"
    assert row["chapter_rank"] < row["passage_rank"]
    assert row["rank"] == row["passage_rank"]
    assert report["overall"] == report["passage_overall"]
    assert report["chapter_overall"]["recall_at_50"] == 1.0


def test_passage_evaluation_rejects_wrong_corpus_hash(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=500)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    index = load_index(built.path)
    case = BenchmarkCase(
        id="case-1",
        query="remembered scene",
        expected_chapter_ids=("agot-001-prologue",),
        category="name-free",
        rationale="Synthetic passage evaluation.",
        expected_passages=(PassageTarget("agot-001-prologue", 10, 30),),
    )

    with pytest.raises(BenchmarkValidationError, match="corpus hash"):
        evaluate_benchmark(index, Benchmark((case,), "0" * 64), FakeEncoder())


def test_evaluation_rejects_empty_case_sequence(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=500)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    index = load_index(built.path)

    with pytest.raises(BenchmarkValidationError, match="at least one case"):
        evaluate_benchmark(index, (), FakeEncoder())


def test_load_benchmarks_combines_legacy_and_per_book_hashes(tmp_path) -> None:
    legacy = tmp_path / "legacy.json"
    per_book = tmp_path / "per-book.json"
    base_case = {
        "query": "event",
        "expected_passages": [
            {"chapter_id": "agot-001-prologue", "word_start": 10, "word_end": 20}
        ],
        "category": "name-free",
        "rationale": "Synthetic benchmark.",
    }
    legacy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus": {"source_sha256": "a" * 64},
                "cases": [{**base_case, "id": "legacy"}],
            }
        )
    )
    per_book.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus": {
                    "sources": [
                        {"book_id": "acok", "source_sha256": "b" * 64}
                    ]
                },
                "cases": [{**base_case, "id": "per-book"}],
            }
        )
    )

    benchmark = load_benchmarks([legacy, per_book])

    assert len(benchmark) == 2
    assert benchmark.source_sha256 == "a" * 64
    assert dict(benchmark.source_hashes) == {"acok": "b" * 64}


def test_schema_two_supports_graded_targets_and_requires_verified_sealed_cases(
    tmp_path,
) -> None:
    path = tmp_path / "graded.json"
    payload = {
        "schema_version": 2,
        "name": "review-set",
        "split": "acceptance",
        "status": "draft",
        "corpus": {"source_sha256": "a" * 64},
        "cases": [
            {
                "id": "case-1",
                "query": "a remembered event",
                "expected_passages": [
                    {
                        "chapter_id": "agot-001-prologue",
                        "word_start": 10,
                        "word_end": 30,
                        "relevance": 2,
                    },
                    {
                        "chapter_id": "agot-002-bran-1",
                        "word_start": 40,
                        "word_end": 60,
                        "relevance": 3,
                    },
                ],
                "category": "name-free",
                "rationale": "Two valid scenes with different relevance.",
                "verification": {"status": "draft"},
            }
        ],
    }
    path.write_text(json.dumps(payload))

    benchmark = load_benchmark(path)

    assert benchmark.name == "review-set"
    assert benchmark.split == "acceptance"
    assert benchmark.status == "draft"
    assert [target.relevance for target in benchmark[0].expected_passages] == [2, 3]

    payload["status"] = "sealed"
    path.write_text(json.dumps(payload))
    with pytest.raises(BenchmarkValidationError, match="must be verified"):
        load_benchmark(path)


def test_multiple_passages_report_hit_target_recall_ndcg_and_confidence(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=500)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    index = load_index(built.path)
    results = semantic_search(index, "remembered scene", FakeEncoder(), top=50)
    first = results[0].chunk
    missing_chapter = next(
        chunk for chunk in index.chunks if chunk.chapter_id != first.chapter_id
    )
    case = BenchmarkCase(
        id="case-1",
        query="remembered scene",
        expected_chapter_ids=(first.chapter_id, missing_chapter.chapter_id),
        category="name-free",
        rationale="Synthetic graded evaluation.",
        expected_passages=(
            PassageTarget(first.chapter_id, first.word_start, first.word_end, 3),
            PassageTarget(
                missing_chapter.chapter_id,
                missing_chapter.word_start,
                missing_chapter.word_end,
                1,
            ),
        ),
    )

    report = evaluate_benchmark(index, (case,), FakeEncoder())
    overall = report["overall"]

    assert overall["hit_at_1"] == 1.0
    assert overall["recall_at_1"] == overall["hit_at_1"]
    assert overall["target_recall_at_1"] == 0.5
    assert 0.0 < overall["ndcg_at_5"] <= 1.0
    assert overall["hit_at_1_ci95"]["lower"] < 1.0


def test_evaluation_reports_query_style_and_name_usage(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=500)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    index = load_index(built.path)
    first = semantic_search(index, "Arya sword", FakeEncoder(), top=1)[0].chunk
    case = BenchmarkCase(
        id="natural-1",
        query="Arya sword",
        expected_chapter_ids=(first.chapter_id,),
        category="representative",
        rationale="Synthetic query-form reporting test.",
        expected_passages=(
            PassageTarget(first.chapter_id, first.word_start, first.word_end),
        ),
        query_style="terse-search",
        name_usage="named",
        scene_id="scene-1",
    )

    report = evaluate_benchmark(index, (case,), FakeEncoder())

    assert report["by_query_style"]["terse-search"]["cases"] == 1
    assert report["by_name_usage"]["named"]["cases"] == 1
    assert report["cases"][0]["query"] == "Arya sword"


def test_search_box_benchmarks_have_intended_query_form_mix() -> None:
    root = Path(__file__).parents[1]
    development = load_benchmark(
        root / "evaluation" / "representative-passage-dev.review.json"
    )
    acceptance = load_benchmark(root / "evaluation" / "acceptance-passage.review.json")

    assert Counter(case.query_style for case in development) == {
        "descriptive-search": 70,
        "terse-search": 20,
        "quotation-or-noisy": 10,
    }
    assert Counter(case.name_usage for case in development) == {
        "named": 60,
        "name-free": 40,
    }
    assert len({case.scene_id for case in development}) == 99
    assert Counter(case.scene_id for case in development)["bran-tower-push"] == 2
    assert development.passage_match_strategy == "overlap"
    assert development.minimum_overlap_words == 20
    prompt_prefixes = ("what ", "when ", "where ", "why ", "who ", "how ", "does ", "which ")
    for case in (*development, *acceptance):
        query = case.query.lower()
        assert not query.startswith(prompt_prefixes)
        assert not query.endswith("?")
        assert not query.startswith(("i remember ", "i'm trying ", "looking for ", "scene where "))
    assert acceptance.split == "acceptance"
    assert acceptance.status == "draft"
    assert all(case.verification_status == "draft" for case in acceptance)


def test_representative_targets_match_their_base_benchmarks() -> None:
    root = Path(__file__).parents[1]
    representative = load_benchmark(
        root / "evaluation" / "representative-passage-dev.review.json"
    )
    base = load_benchmarks(
        (
            root / "evaluation" / "agot-passage-dev.json",
            root / "evaluation" / "acok-passage-dev.json",
            root / "evaluation" / "acok-passage-dev-expansion.review.json",
        )
    )
    targets_by_id = {case.id: case.expected_passages for case in base}

    assert len(targets_by_id) == len(representative)
    for case in representative:
        base_id = case.id.removeprefix("representative-")
        assert case.expected_passages == targets_by_id[base_id]


def test_five_book_smoke_benchmark_loads() -> None:
    root = Path(__file__).parents[1]

    benchmark = load_benchmark(root / "evaluation" / "five-book-smoke.dev.json")

    assert benchmark.name == "asos-affc-adwd-chapter-smoke-development"
    assert benchmark.status == "draft"
    assert len(benchmark) == 9
    assert {book_id for book_id, _digest in benchmark.source_hashes} == {
        "asos",
        "affc",
        "adwd",
    }
