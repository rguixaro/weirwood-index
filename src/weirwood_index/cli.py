from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from weirwood_index.chunking import PROFILES, get_profile
from weirwood_index.corpus import corpus_summary, parse_corpus
from weirwood_index.embedding import (
    DEFAULT_MODEL,
    Encoder,
    create_encoder,
    resolve_model_revision,
)
from weirwood_index.evaluation import decision_gate, evaluate_benchmark, load_benchmarks
from weirwood_index.events import load_event_parser
from weirwood_index.indexing import (
    build_index,
    enrich_event_records,
    enrich_scene_windows,
    load_index,
)
from weirwood_index.models import WeirwoodError
from weirwood_index.reranking import (
    DEFAULT_RERANK_CANDIDATES,
    RERANKER_KINDS,
    Reranker,
    create_reranker,
)
from weirwood_index.retrieval import (
    DEFAULT_CANDIDATE_POOL,
    DEFAULT_CHAPTER_CANDIDATES,
    DEFAULT_CHAPTER_EVIDENCE_PASSAGES,
    DEFAULT_CHAPTER_WEIGHT,
    DEFAULT_CONTEXT_VECTOR_WEIGHT,
    DEFAULT_EVENT_WEIGHT,
    DEFAULT_LEXICAL_EVIDENCE_WEIGHT,
    DEFAULT_NEIGHBOR_WEIGHT,
    DEFAULT_PASSAGE_CANDIDATE_POOL,
    DEFAULT_PASSAGES_PER_CHAPTER,
    DEFAULT_RETENTION_MODE,
    DEFAULT_SCENE_LEXICAL_WEIGHT,
    DEFAULT_SCENE_WINDOW_WEIGHT,
    RETENTION_MODES,
    RETRIEVAL_MODES,
    SearchResult,
    search_index,
    similar_chunks,
)
from weirwood_index.scenes import (
    DEFAULT_ENTITY_SCOPE_WORDS,
    DEFAULT_SCENE_WINDOW_OVERLAP,
    DEFAULT_SCENE_WINDOW_WORDS,
)
from weirwood_index.training import mine_hard_negatives

EncoderFactory = Callable[..., Encoder]
RerankerFactory = Callable[..., Reranker]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weirwood",
        description="Local passage retrieval for A Song of Ice and Fire",
    )
    groups = parser.add_subparsers(dest="group", required=True)

    corpus = groups.add_parser("corpus", help="inspect and validate source text")
    corpus_commands = corpus.add_subparsers(dest="command", required=True)
    inspect = corpus_commands.add_parser("inspect", help="parse without embedding")
    inspect.add_argument("--source", type=Path, action="append", required=True)
    inspect.add_argument("--json", action="store_true")

    index = groups.add_parser("index", help="build local vector indexes")
    index_commands = index.add_subparsers(dest="command", required=True)
    build = index_commands.add_parser("build", help="build a passage index")
    build.add_argument("--source", type=Path, action="append", required=True)
    build.add_argument("--profile", choices=tuple(PROFILES), required=True)
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--revision")
    build.add_argument("--output-root", type=Path, default=Path("data/indexes"))
    build.add_argument("--batch-size", type=int, default=32)
    build.add_argument("--narrative-views", action="store_true")
    build.add_argument("--force", action="store_true")
    build.add_argument("--json", action="store_true")
    enrich_scenes = index_commands.add_parser(
        "enrich-scenes", help="add independently embedded scene windows"
    )
    enrich_scenes.add_argument("--index", type=Path, required=True)
    enrich_scenes.add_argument("--batch-size", type=int, default=8)
    enrich_scenes.add_argument(
        "--window-words", type=int, default=DEFAULT_SCENE_WINDOW_WORDS
    )
    enrich_scenes.add_argument(
        "--overlap-words", type=int, default=DEFAULT_SCENE_WINDOW_OVERLAP
    )
    enrich_scenes.add_argument(
        "--entity-scope-words", type=int, default=DEFAULT_ENTITY_SCOPE_WORDS
    )
    enrich_scenes.add_argument("--force", action="store_true")
    enrich_scenes.add_argument("--json", action="store_true")
    enrich_events = index_commands.add_parser(
        "enrich-events", help="extract structured events from scene windows"
    )
    enrich_events.add_argument("--index", type=Path, required=True)
    enrich_events.add_argument("--batch-size", type=int, default=32)
    enrich_events.add_argument("--force", action="store_true")
    enrich_events.add_argument("--json", action="store_true")

    search = groups.add_parser("search", help="find passages by meaning")
    search.add_argument("query")
    _add_retrieval_options(search)
    _add_search_mode_options(search)

    similar = groups.add_parser("similar", help="find passages like an indexed passage")
    similar.add_argument("chunk_id")
    _add_retrieval_options(similar)

    evaluate = groups.add_parser("evaluate", help="run a frozen retrieval benchmark")
    evaluate.add_argument("--queries", type=Path, action="append", required=True)
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=32)
    _add_search_mode_options(evaluate)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--json", action="store_true")

    training = groups.add_parser("training", help="build narrative retrieval training data")
    training_commands = training.add_subparsers(dest="command", required=True)
    mine = training_commands.add_parser(
        "mine-negatives", help="mine hard negatives from development queries"
    )
    mine.add_argument("--queries", type=Path, action="append", required=True)
    mine.add_argument("--index", type=Path, required=True)
    mine.add_argument("--output", type=Path, required=True)
    mine.add_argument("--negatives-per-case", type=int, default=8)
    mine.add_argument("--batch-size", type=int, default=32)
    mine.add_argument("--json", action="store_true")
    _add_search_mode_options(mine)
    return parser


def _add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pov")
    parser.add_argument("--book")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--json", action="store_true")


def _add_search_mode_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="semantic")
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    parser.add_argument(
        "--context-vector-weight", type=float, default=DEFAULT_CONTEXT_VECTOR_WEIGHT
    )
    parser.add_argument(
        "--scene-window-weight", type=float, default=DEFAULT_SCENE_WINDOW_WEIGHT
    )
    parser.add_argument(
        "--scene-lexical-weight", type=float, default=DEFAULT_SCENE_LEXICAL_WEIGHT
    )
    parser.add_argument("--event-weight", type=float, default=DEFAULT_EVENT_WEIGHT)
    parser.add_argument(
        "--lexical-evidence-weight",
        type=float,
        default=DEFAULT_LEXICAL_EVIDENCE_WEIGHT,
    )
    parser.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--deduplicate-chapters", action="store_true")
    parser.add_argument("--narrative", action="store_true")
    parser.add_argument("--late-interaction", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--reranker-kind", choices=RERANKER_KINDS, default="cross-encoder"
    )
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-revision")
    parser.add_argument(
        "--rerank-candidates", type=int, default=DEFAULT_RERANK_CANDIDATES
    )
    parser.add_argument("--rerank-context-words", type=int, default=0)
    parser.add_argument("--rerank-fusion-weight", type=float, default=1.0)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument(
        "--chapter-candidates", type=int, default=DEFAULT_CHAPTER_CANDIDATES
    )
    parser.add_argument(
        "--passages-per-chapter", type=int, default=DEFAULT_PASSAGES_PER_CHAPTER
    )
    parser.add_argument(
        "--passage-candidate-pool", type=int, default=DEFAULT_PASSAGE_CANDIDATE_POOL
    )
    parser.add_argument(
        "--chapter-evidence-passages",
        type=int,
        default=DEFAULT_CHAPTER_EVIDENCE_PASSAGES,
    )
    parser.add_argument("--chapter-weight", type=float, default=DEFAULT_CHAPTER_WEIGHT)
    parser.add_argument("--neighbor-weight", type=float, default=DEFAULT_NEIGHBOR_WEIGHT)
    parser.add_argument(
        "--retention-mode", choices=RETENTION_MODES, default=DEFAULT_RETENTION_MODE
    )


def _encoder_for_manifest(
    manifest: dict[str, Any], encoder_factory: EncoderFactory, *, batch_size: int
) -> Encoder:
    model = manifest["model"]
    return encoder_factory(
        model_id=model["id"],
        revision=model["revision"],
        batch_size=batch_size,
        show_progress=False,
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _print_results(results: list[SearchResult]) -> None:
    if not results:
        print("No matching passages.")
        return
    for result in results:
        print(
            f"{result.rank:>2}. {result.score:.4f}  {result.chunk.chapter_id}  "
            f"{result.chunk.chapter_title}  POV={result.chunk.pov}  {result.chunk.id}"
        )
        if result.retrieval is not None and result.retrieval["mode"] == "hybrid":
            semantic_rank = result.retrieval["semantic_rank"] or "-"
            lexical_rank = result.retrieval["lexical_rank"] or "-"
            print(f"    semantic-rank={semantic_rank} lexical-rank={lexical_rank}")
        if result.retrieval is not None and result.retrieval["mode"] == "semantic-rerank":
            semantic_rank = result.retrieval["semantic_rank"]
            reranker_score = result.retrieval["reranker_score"]
            if reranker_score is None:
                print(f"    semantic-rank={semantic_rank} not-reranked")
            else:
                print(
                    f"    semantic-rank={semantic_rank} "
                    f"reranker-score={reranker_score:.4f}"
                )
        print(f"    {result.to_dict()['excerpt']}")


def _print_evaluation(report: dict[str, Any]) -> None:
    overall = report["overall"]
    configuration = report["configuration"]
    print(f"Mode:       {configuration['mode']}")
    print(f"Evaluation: {configuration['evaluation_level']}")
    print(f"Chapter deduplication: {configuration['deduplicate_chapters']}")
    if configuration["hierarchical"]:
        hierarchy = f"{configuration['chapter_candidates']} chapters"
        if configuration["retention_mode"] == "per-chapter":
            hierarchy += f", {configuration['passages_per_chapter']} passages/chapter"
        else:
            hierarchy += ", global passage retention"
        print(f"Hierarchy:  {hierarchy}")
    if configuration["mode"] == "hybrid":
        print(
            f"Weights:    semantic={configuration['semantic_weight']:.2f}, "
            f"lexical={configuration['lexical_weight']:.2f}"
        )
    if configuration["scene_window_weight"] or configuration["scene_lexical_weight"]:
        print(
            "Scenes:     "
            f"semantic={configuration['scene_window_weight']:.2f}, "
            f"lexical={configuration['scene_lexical_weight']:.2f}"
        )
    if configuration["event_weight"]:
        print(f"Events:     weight={configuration['event_weight']:.2f}")
    if configuration["lexical_evidence_weight"]:
        print(
            "Evidence:   "
            f"coverage/proximity weight={configuration['lexical_evidence_weight']:.2f}"
        )
    if configuration["reranker"] is not None:
        reranker = configuration["reranker"]
        print(
            f"Reranker:   {reranker['model']} over {reranker['candidates']} candidates; "
            f"context={reranker['context_words']} words; "
            f"fusion={reranker['fusion_weight']:.2f}"
        )
    print(f"Cases:      {overall['cases']}")
    print(f"Hit@1:      {overall['hit_at_1']:.1%}")
    print(f"Hit@3:      {overall['hit_at_3']:.1%}")
    print(f"Hit@5:      {overall['hit_at_5']:.1%}")
    print(f"Hit@10:     {overall['hit_at_10']:.1%}")
    print(f"Hit@20:     {overall['hit_at_20']:.1%}")
    print(f"Hit@50:     {overall['hit_at_50']:.1%}")
    hit_at_5_ci = overall["hit_at_5_ci95"]
    print(
        f"Hit@5 CI95: {hit_at_5_ci['lower']:.1%}–{hit_at_5_ci['upper']:.1%}"
    )
    print(f"MRR@10:     {overall['mrr']:.3f}")
    print(f"MRR@50:     {overall['mrr_at_50']:.3f}")
    print(
        f"Latency:    mean={report['latency_ms']['mean']:.1f}ms, "
        f"p95={report['latency_ms']['p95']:.1f}ms"
    )
    if configuration["evaluation_level"] == "passage":
        chapter = report["chapter_overall"]
        print(
            f"Chapter-only comparison: Hit@1={chapter['hit_at_1']:.1%}, "
            f"Hit@5={chapter['hit_at_5']:.1%}, Hit@20={chapter['hit_at_20']:.1%}"
        )
        if configuration["hierarchical"]:
            stages = report["stages"]
            print(
                f"Stages: chapter R@10={stages['chapter_recall_at_10']:.1%}, "
                f"candidate coverage={stages['passage_candidate_coverage']:.1%}, "
                f"within-chapter R@5={stages['within_chapter_recall_at_5']:.1%}"
            )
    print("Categories:")
    for category, metrics in report["by_category"].items():
        print(
            f"  {category}: Hit@5={metrics['hit_at_5']:.1%}, "
            f"Hit@10={metrics['hit_at_10']:.1%}, MRR@10={metrics['mrr']:.3f}"
        )
    if report["by_query_style"]:
        print("Query styles:")
        for query_style, metrics in report["by_query_style"].items():
            print(
                f"  {query_style}: Hit@5={metrics['hit_at_5']:.1%}, "
                f"Hit@10={metrics['hit_at_10']:.1%}, MRR@10={metrics['mrr']:.3f}"
            )
    if report["by_name_usage"]:
        print("Name usage:")
        for name_usage, metrics in report["by_name_usage"].items():
            print(
                f"  {name_usage}: Hit@5={metrics['hit_at_5']:.1%}, "
                f"Hit@10={metrics['hit_at_10']:.1%}, MRR@10={metrics['mrr']:.3f}"
            )
    if report["failures"]:
        print("Failures at Hit@5:")
        for failure in report["failures"]:
            print(
                f"  {failure['id']}: expected {', '.join(failure['expected_chapter_ids'])}; "
                f"rank={failure['rank']}; stage={failure['failure_stage']}; "
                f"top={failure['top_chapter_id']}"
            )
    print(
        f"Threshold comparison: {report['threshold_status']} "
        "(a final decision requires the unseen acceptance set)"
    )


def run(
    argv: Sequence[str] | None = None,
    *,
    encoder_factory: EncoderFactory = create_encoder,
    reranker_factory: RerankerFactory = create_reranker,
) -> int:
    args = _parser().parse_args(argv)

    if args.group == "corpus":
        summary = corpus_summary(parse_corpus(args.source))
        if args.json:
            _print_json(summary)
        else:
            for source in summary["sources"]:
                print(
                    f"Source: {source['book_id']} {source['path']} "
                    f"SHA-256={source['sha256']}"
                )
            print(f"SHA-256: {summary['source_sha256']}")
            print(f"Books: {summary['book_count']}")
            print(f"Chapters: {summary['chapter_count']}")
            print(f"Words: {summary['word_count']}")
            povs = ", ".join(
                f"{key}={value}" for key, value in summary["pov_counts"].items()
            )
            print("POVs: " + povs)
            print(
                "Cleaning: "
                + ", ".join(
                    f"{key}={value}" for key, value in summary["cleaning_counts"].items()
                )
            )
        return 0

    if args.group == "index":
        if args.batch_size < 1:
            raise WeirwoodError("--batch-size must be positive")
        if args.command == "enrich-events":
            loaded = load_index(args.index)
            enriched = enrich_event_records(
                loaded,
                load_event_parser(),
                batch_size=args.batch_size,
                force=args.force,
            )
            output = {
                "artifact_location": str(enriched.path),
                "event_count": enriched.event_count,
                "extraction_seconds": round(enriched.extraction_seconds, 3),
            }
            if args.json:
                _print_json(output)
            else:
                print(f"Structured events: {output['event_count']}")
                print(f"Extraction duration: {output['extraction_seconds']:.3f}s")
                print(f"Index: {output['artifact_location']}")
            return 0
        if args.command == "enrich-scenes":
            loaded = load_index(args.index)
            encoder = _encoder_for_manifest(
                loaded.manifest, encoder_factory, batch_size=args.batch_size
            )
            enriched = enrich_scene_windows(
                loaded,
                encoder,
                window_words=args.window_words,
                overlap_words=args.overlap_words,
                entity_scope_words=args.entity_scope_words,
                force=args.force,
            )
            output = {
                "artifact_location": str(enriched.path),
                "window_count": enriched.window_count,
                "embedding_seconds": round(enriched.embedding_seconds, 3),
                "window_words": args.window_words,
                "overlap_words": args.overlap_words,
                "entity_scope_words": args.entity_scope_words,
            }
            if args.json:
                _print_json(output)
            else:
                print(f"Scene windows: {output['window_count']}")
                print(f"Embedding duration: {output['embedding_seconds']:.3f}s")
                print(f"Index: {output['artifact_location']}")
            return 0
        encoder = encoder_factory(
            model_id=args.model,
            revision=resolve_model_revision(args.model, args.revision),
            batch_size=args.batch_size,
            show_progress=not args.json,
        )
        built = build_index(
            source=args.source,
            profile=get_profile(args.profile),
            encoder=encoder,
            output_root=args.output_root,
            force=args.force,
            narrative_views=args.narrative_views,
        )
        output = {
            "artifact_location": str(built.path),
            "model": built.manifest["model"],
            "profile": built.manifest["chunk_profile"],
            "chunk_count": built.chunk_count,
            "vector_dimensions": built.vector_dimensions,
            "embedding_seconds": round(built.embedding_seconds, 3),
            "narrative_views": "narrative_views" in built.manifest,
        }
        if args.json:
            _print_json(output)
        else:
            print(f"Model: {output['model']['id']}@{output['model']['revision']}")
            print(f"Profile: {output['profile']['name']}")
            print(f"Chunks: {output['chunk_count']}")
            print(f"Dimensions: {output['vector_dimensions']}")
            print(f"Embedding duration: {output['embedding_seconds']:.3f}s")
            print(f"Index: {output['artifact_location']}")
        return 0

    loaded = load_index(args.index)
    if getattr(args, "late_interaction", False) and not args.narrative:
        raise WeirwoodError("--late-interaction requires --narrative")
    if getattr(args, "narrative", False) and not loaded.narrative_views:
        raise WeirwoodError(
            "--narrative requires an index built with --narrative-views"
        )
    if args.group == "training":
        if args.batch_size < 1:
            raise WeirwoodError("--batch-size must be positive")
        benchmark = load_benchmarks(args.queries)
        encoder = None
        if args.mode != "lexical":
            encoder = _encoder_for_manifest(
                loaded.manifest, encoder_factory, batch_size=args.batch_size
            )
        payload = mine_hard_negatives(
            loaded,
            benchmark,
            encoder,
            negatives_per_case=args.negatives_per_case,
            mode=args.mode,
            semantic_weight=args.semantic_weight,
            context_vector_weight=args.context_vector_weight,
            scene_window_weight=args.scene_window_weight,
            scene_lexical_weight=args.scene_lexical_weight,
            event_weight=args.event_weight,
            hierarchical=args.hierarchical,
            narrative=args.narrative,
            late_interaction=args.late_interaction,
            chapter_candidates=args.chapter_candidates,
            passages_per_chapter=args.passages_per_chapter,
            passage_candidate_pool=args.passage_candidate_pool,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "output": str(args.output.resolve()),
            "examples": len(payload["examples"]),
            "negatives": sum(
                len(example["negative_chunk_ids"]) for example in payload["examples"]
            ),
        }
        if args.json:
            _print_json(summary)
        else:
            print(f"Examples: {summary['examples']}")
            print(f"Negatives: {summary['negatives']}")
            print(f"Training data: {summary['output']}")
        return 0
    if args.group in {"search", "similar"}:
        if not 1 <= args.top <= 100:
            raise WeirwoodError("--top must be between 1 and 100")
        if args.pov is not None:
            available = {chunk.pov for chunk in loaded.chunks}
            if args.pov.upper() not in available:
                raise WeirwoodError(
                    f"unknown POV {args.pov.upper()!r}; choose one of: "
                    f"{', '.join(sorted(available))}"
                )
        if args.book is not None:
            available_books = {chunk.book_id for chunk in loaded.chunks}
            if args.book.casefold() not in available_books:
                raise WeirwoodError(
                    f"unknown book {args.book.casefold()!r}; choose one of: "
                    f"{', '.join(sorted(available_books))}"
                )
    if args.group == "similar":
        results = similar_chunks(
            loaded, args.chunk_id, top=args.top, pov=args.pov, book=args.book
        )
    else:
        if args.batch_size < 1:
            raise WeirwoodError("--batch-size must be positive")
        if args.group == "search" and not args.query.strip():
            raise WeirwoodError("query must not be empty")
        if args.rerank and args.mode != "semantic" and not args.hierarchical:
            raise WeirwoodError(
                "--rerank supports semantic mode or hierarchical hybrid mode"
            )
        if args.reranker_batch_size < 1:
            raise WeirwoodError("--reranker-batch-size must be positive")
        cases = load_benchmarks(args.queries) if args.group == "evaluate" else None
        encoder = None
        if args.mode != "lexical":
            encoder = _encoder_for_manifest(
                loaded.manifest, encoder_factory, batch_size=args.batch_size
            )
        reranker = None
        if args.rerank:
            reranker = reranker_factory(
                kind=args.reranker_kind,
                model_id=args.reranker_model,
                revision=args.reranker_revision,
                batch_size=args.reranker_batch_size,
                show_progress=not args.json,
            )
        if args.group == "search":
            results = search_index(
                loaded,
                args.query,
                encoder,
                mode=args.mode,
                top=args.top,
                pov=args.pov,
                book=args.book,
                semantic_weight=args.semantic_weight,
                context_vector_weight=args.context_vector_weight,
                scene_window_weight=args.scene_window_weight,
                scene_lexical_weight=args.scene_lexical_weight,
                lexical_evidence_weight=args.lexical_evidence_weight,
                event_weight=args.event_weight,
                candidate_pool=args.candidate_pool,
                deduplicate_chapters=args.deduplicate_chapters,
                reranker=reranker,
                rerank_candidates=args.rerank_candidates,
                rerank_context_words=args.rerank_context_words,
                rerank_fusion_weight=args.rerank_fusion_weight,
                hierarchical=args.hierarchical,
                chapter_candidates=args.chapter_candidates,
                passages_per_chapter=args.passages_per_chapter,
                passage_candidate_pool=args.passage_candidate_pool,
                chapter_evidence_passages=args.chapter_evidence_passages,
                chapter_weight=args.chapter_weight,
                neighbor_weight=args.neighbor_weight,
                retention_mode=args.retention_mode,
                narrative=args.narrative,
                late_interaction=args.late_interaction,
            )
        else:
            assert cases is not None
            report = evaluate_benchmark(
                loaded,
                cases,
                encoder,
                mode=args.mode,
                semantic_weight=args.semantic_weight,
                context_vector_weight=args.context_vector_weight,
                scene_window_weight=args.scene_window_weight,
                scene_lexical_weight=args.scene_lexical_weight,
                lexical_evidence_weight=args.lexical_evidence_weight,
                event_weight=args.event_weight,
                candidate_pool=args.candidate_pool,
                deduplicate_chapters=args.deduplicate_chapters,
                reranker=reranker,
                rerank_candidates=args.rerank_candidates,
                rerank_context_words=args.rerank_context_words,
                rerank_fusion_weight=args.rerank_fusion_weight,
                hierarchical=args.hierarchical,
                chapter_candidates=args.chapter_candidates,
                passages_per_chapter=args.passages_per_chapter,
                passage_candidate_pool=args.passage_candidate_pool,
                chapter_evidence_passages=args.chapter_evidence_passages,
                chapter_weight=args.chapter_weight,
                neighbor_weight=args.neighbor_weight,
                retention_mode=args.retention_mode,
                narrative=args.narrative,
                late_interaction=args.late_interaction,
            )
            report["threshold_status"] = decision_gate(report)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            if args.json:
                _print_json(report)
            else:
                _print_evaluation(report)
                if args.output is not None:
                    print(f"JSON report: {args.output.resolve()}")
            return 0

    if args.json:
        _print_json([result.to_dict() for result in results])
    else:
        _print_results(results)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    encoder_factory: EncoderFactory = create_encoder,
    reranker_factory: RerankerFactory = create_reranker,
) -> int:
    try:
        return run(
            argv,
            encoder_factory=encoder_factory,
            reranker_factory=reranker_factory,
        )
    except (WeirwoodError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
