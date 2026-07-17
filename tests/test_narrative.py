from __future__ import annotations

from weirwood_index.models import Chunk
from weirwood_index.narrative import (
    build_narrative_view,
    expand_narrative_query,
    split_sentences,
)


def test_narrative_view_extracts_metadata_dialogue_events_entities_and_sentences() -> None:
    chunk = Chunk(
        id="agot-001-prologue-c001",
        chapter_id="agot-001-prologue",
        chapter_title="PROLOGUE",
        chapter_sequence=1,
        pov="PROLOGUE",
        pov_ordinal=1,
        chunk_ordinal=1,
        word_start=0,
        word_end=20,
        text=(
            'Ser Waymar Royce raised his sword. "Come no farther," he said. '
            "The ranger fought the Other and his blade shattered."
        ),
    )

    view = build_narrative_view(chunk)

    assert "A Game of Thrones" in view.context
    assert "Ser Waymar Royce" in view.entities
    assert "Come no farther" in view.dialogue
    assert "fought" in view.events
    assert len(view.sentences) == 3
    assert view.summary
    assert view.lexical_text


def test_sentence_split_keeps_compact_narrative_units() -> None:
    assert split_sentences('One stopped. "Who goes there?" Two answered.') == (
        "One stopped.",
        '"Who goes there?"',
        "Two answered.",
    )


def test_query_expansion_adds_distinctive_saga_aliases_only_when_present() -> None:
    assert "Sandor Clegane" in expand_narrative_query(
        "a scarred guard rescues a girl"
    )
    assert expand_narrative_query("a guard rescues a girl") == "a guard rescues a girl"
