from __future__ import annotations

from weirwood_index.models import Chunk
from weirwood_index.scenes import (
    build_scene_windows,
    expand_scene_query,
    map_chunks_to_scene_windows,
)


def _chunk(
    chunk_id: str,
    chapter_id: str,
    ordinal: int,
    start: int,
    words: list[str],
) -> Chunk:
    return Chunk(
        id=chunk_id,
        chapter_id=chapter_id,
        chapter_title=chapter_id,
        chapter_sequence=1,
        pov="EDDARD",
        pov_ordinal=1,
        chunk_ordinal=ordinal,
        word_start=start,
        word_end=start + len(words),
        text=" ".join(words),
    )


def test_scene_windows_propagate_local_entities_and_map_without_crossing_chapters() -> None:
    first_words = ["Cersei", "Lannister", *[f"first{number}" for number in range(118)]]
    second_words = ["she", "warned", *[f"second{number}" for number in range(118)]]
    other_words = [f"other{number}" for number in range(120)]
    chunks = (
        _chunk("a-1", "chapter-a", 1, 0, first_words),
        _chunk("a-2", "chapter-a", 2, 120, second_words),
        _chunk("b-1", "chapter-b", 1, 0, other_words),
    )

    windows = build_scene_windows(
        chunks,
        window_words=100,
        overlap_words=20,
        entity_scope_words=240,
    )
    mappings = map_chunks_to_scene_windows(chunks, windows)

    late_window = next(
        window
        for window in windows
        if window.chapter_id == "chapter-a" and window.word_start > 0
    )
    assert "Cersei Lannister" in late_window.entities
    assert any(alias.startswith("Cersei Lannister") for alias in late_window.aliases)
    assert all(
        windows[position].chapter_id == chunk.chapter_id
        for chunk, positions in zip(chunks, mappings, strict=True)
        for position in positions
    )


def test_scene_query_expands_known_aliases_but_not_ambiguous_roles() -> None:
    expanded = expand_scene_query("Ned warns Cersei")

    assert "Eddard Stark" in expanded
    assert "Cersei Lannister" in expanded
    assert expand_scene_query("the queen gives an order") == "the queen gives an order"
