from __future__ import annotations

import pytest

from weirwood_index.chunking import PROFILES
from weirwood_index.evaluation import Benchmark, BenchmarkCase, PassageTarget
from weirwood_index.indexing import build_index, load_index
from weirwood_index.models import BenchmarkValidationError
from weirwood_index.retrieval import semantic_search
from weirwood_index.training import grouped_scene_folds, mine_hard_negatives

from .helpers import FakeEncoder, write_valid_source


def test_hard_negative_mining_keeps_ids_only_and_prioritizes_confusable_results(
    tmp_path,
) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=500)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    index = load_index(built.path)
    positive = semantic_search(index, "remembered scene", FakeEncoder(), top=1)[0].chunk
    case = BenchmarkCase(
        id="case-1",
        query="remembered scene",
        expected_chapter_ids=(positive.chapter_id,),
        category="name-free",
        rationale="Synthetic mining case.",
        expected_passages=(
            PassageTarget(positive.chapter_id, positive.word_start, positive.word_end),
        ),
    )

    payload = mine_hard_negatives(
        index,
        Benchmark((case,), source_sha256=built.manifest["source"]["sha256"]),
        FakeEncoder(),
        negatives_per_case=4,
        mode="semantic",
        hierarchical=False,
    )

    example = payload["examples"][0]
    assert example["positive_chunk_ids"]
    assert len(example["negative_chunk_ids"]) == 4
    assert not set(example["positive_chunk_ids"]) & set(example["negative_chunk_ids"])
    assert all(
        negative["negative_type"] in {"same-chapter", "same-entity", "global"}
        for negative in example["negatives"]
    )
    assert "text" not in example


def test_hard_negative_mining_rejects_acceptance_benchmarks(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=220)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    index = load_index(built.path)
    case = BenchmarkCase(
        id="case-1",
        query="event",
        expected_chapter_ids=("agot-001-prologue",),
        category="name-free",
        rationale="Synthetic holdout.",
        expected_passages=(PassageTarget("agot-001-prologue", 1, 10),),
    )

    with pytest.raises(BenchmarkValidationError, match="must never"):
        mine_hard_negatives(
            index,
            Benchmark((case,), split="acceptance", status="sealed"),
            FakeEncoder(),
        )


def test_grouped_scene_folds_are_balanced_reproducible_and_leak_free() -> None:
    cases = tuple(
        BenchmarkCase(
            id=f"case-{number}",
            query=f"Arya scene {number}",
            expected_chapter_ids=(f"agot-{number:03d}-arya-1",),
            category="narrative-situation",
            rationale="Synthetic grouped fold case.",
            expected_passages=(
                PassageTarget(f"agot-{number:03d}-arya-1", 10, 60),
            ),
            query_style="descriptive-search",
            name_usage="named" if number % 2 else "name-free",
            scene_id="shared-scene" if number in {1, 2} else f"scene-{number}",
        )
        for number in range(1, 13)
    )
    benchmark = Benchmark(cases, name="grouped", split="development", status="draft")

    first = grouped_scene_folds(benchmark, folds=5, seed=17)
    second = grouped_scene_folds(benchmark, folds=5, seed=17)

    assert [[case.id for case in fold] for fold in first] == [
        [case.id for case in fold] for fold in second
    ]
    assert max(map(len, first)) - min(map(len, first)) <= 2
    scene_folds = {
        case.scene_id: fold_number
        for fold_number, fold in enumerate(first)
        for case in fold
    }
    assert scene_folds["shared-scene"] == next(
        fold_number
        for fold_number, fold in enumerate(first)
        if {"case-1", "case-2"} <= {case.id for case in fold}
    )
    assert {case.id for fold in first for case in fold} == {
        case.id for case in benchmark
    }
