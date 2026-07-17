from __future__ import annotations

import os

import numpy as np
import pytest

from weirwood_index.embedding import SentenceTransformerEncoder


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("WEIRWOOD_RUN_MODEL_TESTS") != "1",
    reason="set WEIRWOOD_RUN_MODEL_TESTS=1 to load the real local model",
)
def test_real_model_smoke() -> None:
    encoder = SentenceTransformerEncoder(show_progress=False)

    passages = encoder.encode_passages(["A short local passage."])
    queries = encoder.encode_queries(["find a brief passage"])

    assert passages.shape == queries.shape == (1, 384)
    assert passages.dtype == queries.dtype == np.float32
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0)
