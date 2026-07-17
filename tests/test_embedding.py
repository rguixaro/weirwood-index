from __future__ import annotations

import pytest

from weirwood_index.embedding import resolve_model_revision
from weirwood_index.models import WeirwoodError


def test_registered_bge_m3_revision_is_pinned() -> None:
    assert resolve_model_revision("BAAI/bge-m3", None) == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )


def test_unregistered_model_requires_explicit_revision() -> None:
    with pytest.raises(WeirwoodError, match="--revision is required"):
        resolve_model_revision("example/unpinned-model", None)

    assert resolve_model_revision("example/unpinned-model", "frozen") == "frozen"
