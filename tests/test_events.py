from __future__ import annotations

from weirwood_index.events import (
    build_event_records,
    load_event_parser,
    query_event_signature,
    structured_event_scores,
)
from weirwood_index.scenes import SceneWindow


def _window(window_id: str, text: str, aliases: tuple[str, ...]) -> SceneWindow:
    return SceneWindow(
        id=window_id,
        chapter_id="agot-001-test",
        chapter_title="EDDARD I",
        book_id="agot",
        book_title="A Game of Thrones",
        pov="EDDARD",
        word_start=0,
        word_end=len(text.split()),
        text=text,
        aliases=aliases,
    )


def test_dependency_events_capture_direction_action_and_modality() -> None:
    nlp = load_event_parser()
    windows = (
        _window(
            "scene-1",
            "Ned warned Cersei to leave before he told Robert. He did not kill her.",
            ("Eddard Stark", "Cersei Lannister", "Robert Baratheon"),
        ),
    )

    records = build_event_records(windows, nlp)
    warning = records[0]

    assert warning.subject == "eddard stark"
    assert warning.action == "command"
    assert warning.object == "cersei lannister"
    assert next(record for record in records if record.action == "kill").negated is True


def test_structured_event_score_prefers_matching_actor_action_and_object() -> None:
    nlp = load_event_parser()
    records = build_event_records(
        (
            _window(
                "match",
                "Ned warned Cersei to leave.",
                ("Eddard Stark", "Cersei Lannister"),
            ),
            _window(
                "wrong",
                "Jaime attacked Ned in the street.",
                ("Jaime Lannister", "Eddard Stark"),
            ),
        ),
        nlp,
    )

    signature = query_event_signature("Ned warns Cersei", nlp)
    scores = structured_event_scores(records, "Ned warns Cersei", nlp)

    assert signature.entities == frozenset({"eddard stark", "cersei lannister"})
    assert signature.subject == "eddard stark"
    assert signature.object == "cersei lannister"
    assert scores[0] > scores[1]
