from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weirwood_index.chunking import PROFILES
from weirwood_index.events import load_event_parser
from weirwood_index.indexing import (
    build_context_embeddings,
    build_index,
    enrich_event_records,
    enrich_scene_windows,
    load_index,
)
from weirwood_index.models import Chunk, IndexValidationError, WeirwoodError

from .helpers import FakeEncoder, write_valid_acok_source, write_valid_source


def test_build_writes_ordered_normalized_artifacts(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=220)

    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    loaded = load_index(built.path)

    assert built.chunk_count == len(loaded.chunks) == loaded.embeddings.shape[0]
    assert loaded.embeddings.dtype == np.float32
    assert loaded.embeddings.shape[1] == 16
    assert np.allclose(np.linalg.norm(loaded.embeddings, axis=1), 1.0)
    assert loaded.context_embeddings is not None
    assert np.allclose(np.linalg.norm(loaded.context_embeddings, axis=1), 1.0)
    assert loaded.manifest["source"]["sha256"]
    assert loaded.paragraphs
    assert loaded.manifest["paragraphs"]["count"] == len(loaded.paragraphs)
    assert loaded.paragraphs[0].word_start == 0
    assert (built.path / "paragraphs.jsonl").is_file()
    assert not Path(loaded.manifest["source"]["path"]).is_absolute()
    assert loaded.manifest["chunk_profile"] == {
        "name": "short",
        "overlap": 45,
        "words": 180,
        "strategy": "fixed",
        "min_words": None,
    }
    assert [chunk.id for chunk in loaded.chunks] == [
        json.loads(line)["id"]
        for line in (built.path / "chunks.jsonl").read_text().splitlines()
    ]


def test_context_embeddings_pool_neighbors_without_crossing_chapters() -> None:
    chunks = tuple(
        Chunk(
            id=f"chunk-{position}",
            chapter_id=chapter_id,
            chapter_title=chapter_id,
            chapter_sequence=position,
            pov="TEST",
            pov_ordinal=position,
            chunk_ordinal=ordinal,
            word_start=0,
            word_end=10,
            text="test passage",
        )
        for position, (chapter_id, ordinal) in enumerate(
            (("chapter-a", 1), ("chapter-a", 2), ("chapter-b", 1)), start=1
        )
    )
    vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )

    contexts = build_context_embeddings(chunks, vectors)

    expected_shared = np.asarray([2**-0.5, 2**-0.5], dtype=np.float32)
    assert np.allclose(contexts[0], expected_shared)
    assert np.allclose(contexts[1], expected_shared)
    assert np.allclose(contexts[2], vectors[2])


def test_load_rejects_changed_source(tmp_path) -> None:
    source = write_valid_source(tmp_path)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(IndexValidationError, match="source hash"):
        load_index(built.path)


def test_load_rejects_manifest_dimension_mismatch(tmp_path) -> None:
    source = write_valid_source(tmp_path)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    manifest_path = built.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["vector_dimensions"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(IndexValidationError, match="dimensions"):
        load_index(built.path)


def test_build_rejects_chunk_that_tokenizer_would_truncate(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=220)
    encoder = FakeEncoder()
    encoder.max_tokens = 100

    with pytest.raises(WeirwoodError, match="would be truncated"):
        build_index(
            source=source,
            profile=PROFILES["short"],
            encoder=encoder,
            output_root=tmp_path / "indexes",
        )


def test_build_and_load_multi_book_index(tmp_path) -> None:
    agot = write_valid_source(tmp_path, words_per_chapter=40)
    acok = write_valid_acok_source(tmp_path, words_per_chapter=40)

    built = build_index(
        source=[agot, acok],
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    loaded = load_index(built.path)

    assert {chunk.book_id for chunk in loaded.chunks} == {"agot", "acok"}
    assert [record["book_id"] for record in loaded.manifest["source"]["sources"]] == [
        "agot",
        "acok",
    ]


def test_build_and_load_narrative_view_index(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=220)

    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
        narrative_views=True,
    )
    loaded = load_index(built.path)

    assert "narrative_views" in loaded.manifest
    assert len(loaded.narrative_views) == len(loaded.chunks)
    assert loaded.narrative_embeddings is not None
    assert loaded.narrative_masks is not None
    assert loaded.sentence_embeddings is not None
    assert loaded.sentence_mask is not None
    assert loaded.sentence_embeddings.shape[:2] == loaded.sentence_mask.shape
    assert loaded.sentence_embeddings.shape[0] == len(loaded.chunks)


def test_enrich_and_load_independently_embedded_scene_windows(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=220)
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )

    enriched = enrich_scene_windows(
        load_index(built.path),
        FakeEncoder(),
        window_words=100,
        overlap_words=20,
        entity_scope_words=180,
    )
    loaded = load_index(built.path)

    assert enriched.window_count == len(loaded.scene_windows)
    assert loaded.scene_embeddings is not None
    assert loaded.scene_embeddings.shape[0] == len(loaded.scene_windows)
    assert len(loaded.chunk_scene_positions) == len(loaded.chunks)
    assert loaded.manifest["scene_windows"]["window_words"] == 100


def test_enrich_and_load_structured_event_records(tmp_path) -> None:
    source = write_valid_source(tmp_path, words_per_chapter=40)
    source.write_text(source.read_text().replace("word0", "Ned warned Cersei."))
    built = build_index(
        source=source,
        profile=PROFILES["short"],
        encoder=FakeEncoder(),
        output_root=tmp_path / "indexes",
    )
    enrich_scene_windows(
        load_index(built.path),
        FakeEncoder(),
        window_words=100,
        overlap_words=20,
        entity_scope_words=100,
    )

    enriched = enrich_event_records(load_index(built.path), load_event_parser())
    loaded = load_index(built.path)

    assert enriched.event_count == len(loaded.event_records)
    assert len(loaded.chunk_event_positions) == len(loaded.chunks)
    assert loaded.event_records[0].action == "command"
