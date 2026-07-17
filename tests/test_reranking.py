from __future__ import annotations

import numpy as np
import pytest

from weirwood_index.models import WeirwoodError
from weirwood_index.reranking import colbert_maxsim_scores, create_reranker


def test_colbert_maxsim_scores_preserve_per_query_token_matches() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    passages = [
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    ]

    scores = colbert_maxsim_scores(query, passages)

    assert scores.tolist() == pytest.approx([1.0, 0.5])


def test_create_reranker_rejects_unknown_kind() -> None:
    with pytest.raises(WeirwoodError, match="unknown reranker kind"):
        create_reranker(kind="unknown")
