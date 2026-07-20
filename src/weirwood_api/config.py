from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _boolean(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class ApiSettings:
    index_path: Path | None
    origin_token: str | None
    verify_source: bool = False
    encoder_batch_size: int = 1
    cache_size: int = 128
    excerpt_chars: int = 360
    max_queued_searches: int = 1

    @classmethod
    def from_environment(cls) -> ApiSettings:
        index_value = os.environ.get("WEIRWOOD_INDEX_PATH")
        return cls(
            index_path=Path(index_value).expanduser() if index_value else None,
            origin_token=os.environ.get("WEIRWOOD_ORIGIN_TOKEN"),
            verify_source=_boolean("WEIRWOOD_VERIFY_SOURCE", False),
            encoder_batch_size=_integer("WEIRWOOD_ENCODER_BATCH_SIZE", 1),
            cache_size=_integer("WEIRWOOD_SEARCH_CACHE_SIZE", 128),
            excerpt_chars=_integer("WEIRWOOD_EXCERPT_CHARS", 360),
            max_queued_searches=_integer("WEIRWOOD_MAX_QUEUED_SEARCHES", 1),
        )
