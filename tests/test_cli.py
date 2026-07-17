from __future__ import annotations

import json

from weirwood_index.cli import main, run

from .helpers import (
    FakeEncoder,
    FakeReranker,
    write_valid_acok_source,
    write_valid_source,
)


def _build(tmp_path):
    source = write_valid_source(tmp_path, words_per_chapter=220)
    code = run(
        [
            "index",
            "build",
            "--source",
            str(source),
            "--profile",
            "short",
            "--output-root",
            str(tmp_path / "indexes"),
            "--json",
        ],
        encoder_factory=FakeEncoder,
    )
    assert code == 0
    return source, next((tmp_path / "indexes").iterdir())


def test_corpus_inspect_human_and_json(tmp_path, capsys) -> None:
    source = write_valid_source(tmp_path)

    assert run(["corpus", "inspect", "--source", str(source)]) == 0
    assert "Chapters: 73" in capsys.readouterr().out

    assert run(["corpus", "inspect", "--source", str(source), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["chapter_count"] == 73

    acok = write_valid_acok_source(tmp_path)
    assert run(
        [
            "corpus",
            "inspect",
            "--source",
            str(source),
            "--source",
            str(acok),
            "--json",
        ]
    ) == 0
    combined = json.loads(capsys.readouterr().out)
    assert combined["book_count"] == 2
    assert combined["chapter_counts_by_book"] == {"acok": 70, "agot": 73}


def test_search_human_and_json_output(tmp_path, capsys) -> None:
    _, index = _build(tmp_path)
    capsys.readouterr()

    assert run(
        ["search", "a remembered scene", "--index", str(index), "--top", "3"],
        encoder_factory=FakeEncoder,
    ) == 0
    assert "POV=" in capsys.readouterr().out

    assert run(
        ["search", "a remembered scene", "--index", str(index), "--top", "2", "--json"],
        encoder_factory=FakeEncoder,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    assert set(payload[0]) == {"chunk", "excerpt", "rank", "score"}


def test_lexical_and_hybrid_search_modes(tmp_path, capsys) -> None:
    _, index = _build(tmp_path)
    capsys.readouterr()

    def encoder_must_not_load(**_):
        raise AssertionError("lexical mode must not load the embedding model")

    assert run(
        [
            "search",
            "chapter1",
            "--index",
            str(index),
            "--mode",
            "lexical",
            "--top",
            "2",
            "--json",
        ],
        encoder_factory=encoder_must_not_load,
    ) == 0
    lexical = json.loads(capsys.readouterr().out)
    assert lexical[0]["retrieval"]["mode"] == "lexical"

    assert run(
        [
            "search",
            "chapter1",
            "--index",
            str(index),
            "--mode",
            "hybrid",
            "--top",
            "2",
            "--deduplicate-chapters",
            "--json",
        ],
        encoder_factory=FakeEncoder,
    ) == 0
    hybrid = json.loads(capsys.readouterr().out)
    assert hybrid[0]["retrieval"]["mode"] == "hybrid"
    assert len({result["chunk"]["chapter_id"] for result in hybrid}) == len(hybrid)


def test_semantic_reranking_is_exposed_in_search_json(tmp_path, capsys) -> None:
    _, index = _build(tmp_path)
    capsys.readouterr()

    assert run(
        [
            "search",
            "chapter1",
            "--index",
            str(index),
            "--rerank",
            "--rerank-candidates",
            "4",
            "--top",
            "2",
            "--json",
        ],
        encoder_factory=FakeEncoder,
        reranker_factory=FakeReranker,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["retrieval"]["mode"] == "semantic-rerank"
    assert payload[0]["retrieval"]["semantic_rank"] == 4


def test_hierarchical_hybrid_search_is_exposed_in_json(tmp_path, capsys) -> None:
    _, index = _build(tmp_path)
    capsys.readouterr()

    assert run(
        [
            "search",
            "chapter1",
            "--index",
            str(index),
            "--mode",
            "hybrid",
            "--hierarchical",
            "--top",
            "2",
            "--json",
        ],
        encoder_factory=FakeEncoder,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["retrieval"]["mode"] == "hierarchical-hybrid"
    assert payload[0]["context_word_end"] > payload[0]["context_word_start"]


def test_narrative_build_and_late_interaction_search_are_exposed(tmp_path, capsys) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=220)
    output_root = tmp_path / "narrative-indexes"
    assert run(
        [
            "index",
            "build",
            "--source",
            str(source),
            "--profile",
            "short",
            "--output-root",
            str(output_root),
            "--narrative-views",
            "--json",
        ],
        encoder_factory=FakeEncoder,
    ) == 0
    assert json.loads(capsys.readouterr().out)["narrative_views"] is True
    index = next(output_root.iterdir())

    assert run(
        [
            "search",
            "a remembered phrase",
            "--index",
            str(index),
            "--mode",
            "hybrid",
            "--narrative",
            "--late-interaction",
            "--top",
            "2",
            "--json",
        ],
        encoder_factory=FakeEncoder,
    ) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_scene_window_enrichment_is_exposed(tmp_path, capsys) -> None:
    _, index = _build(tmp_path)
    capsys.readouterr()

    assert run(
        [
            "index",
            "enrich-scenes",
            "--index",
            str(index),
            "--window-words",
            "100",
            "--overlap-words",
            "20",
            "--entity-scope-words",
            "180",
            "--json",
        ],
        encoder_factory=FakeEncoder,
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["window_count"] > 0
    assert payload["window_words"] == 100

    assert run(
        [
            "search",
            "Ned warns Cersei",
            "--index",
            str(index),
            "--mode",
            "hybrid",
            "--scene-window-weight",
            "0.25",
            "--scene-lexical-weight",
            "0.25",
            "--top",
            "2",
            "--json",
        ],
        encoder_factory=FakeEncoder,
    ) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_evaluate_can_write_reviewable_json_report(tmp_path, capsys) -> None:
    _, index = _build(tmp_path)
    capsys.readouterr()
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "query": "remembered scene",
                    "expected_chapter_ids": ["agot-001-prologue"],
                    "category": "name-free",
                    "rationale": "Synthetic CLI test.",
                }
            ]
        )
    )
    report_path = tmp_path / "reports" / "result.json"

    assert run(
        [
            "evaluate",
            "--queries",
            str(benchmark),
            "--index",
            str(index),
            "--output",
            str(report_path),
        ],
        encoder_factory=FakeEncoder,
    ) == 0
    report = json.loads(report_path.read_text())
    assert report["overall"]["cases"] == 1
    assert {
        "recall_at_3",
        "recall_at_20",
        "recall_at_50",
        "mrr_at_50",
    } <= report["overall"].keys()
    assert report["threshold_status"] in {"proceed", "investigate-once", "stop-or-rethink"}
    assert str(report_path.resolve()) in capsys.readouterr().out


def test_cli_returns_nonzero_for_empty_query_invalid_filter_and_bad_benchmark(
    tmp_path, capsys
) -> None:
    _, index = _build(tmp_path)
    capsys.readouterr()

    assert main(
        ["search", "", "--index", str(index)], encoder_factory=FakeEncoder
    ) == 2
    assert "must not be empty" in capsys.readouterr().err

    assert main(
        ["search", "scene", "--index", str(index), "--pov", "HODOR"],
        encoder_factory=FakeEncoder,
    ) == 2
    assert "unknown POV" in capsys.readouterr().err

    malformed = tmp_path / "bad.json"
    malformed.write_text("not json")
    assert main(
        ["evaluate", "--queries", str(malformed), "--index", str(index)],
        encoder_factory=FakeEncoder,
    ) == 2
    assert "malformed benchmark" in capsys.readouterr().err
