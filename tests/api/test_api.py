from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

from weirwood_api.config import ApiSettings
from weirwood_api.main import ORIGIN_TOKEN_HEADER, create_app
from weirwood_api.schemas import SearchRequest
from weirwood_api.service import (
    BookResultCount,
    CatalogBook,
    SearchCatalog,
    SearchPagination,
    SearchPayload,
)
from weirwood_index.models import IndexValidationError, SearchValidationError


class FakeRuntime:
    loaded = False
    last_request: SearchRequest | None = None

    def __init__(self) -> None:
        self.catalog_calls = 0

    async def catalog(self) -> SearchCatalog:
        self.catalog_calls += 1
        return SearchCatalog(
            books=(
                CatalogBook(
                    book_id="agot",
                    book_title="A Game of Thrones",
                    book_sequence=1,
                    povs=("ARYA", "EDDARD", "PROLOGUE"),
                ),
            )
        )

    async def search(self, request: SearchRequest) -> SearchPayload:
        self.loaded = True
        self.last_request = request
        return SearchPayload(
            duration_ms=12.3456,
            cached=False,
            pagination=SearchPagination(
                page=1,
                page_size=20,
                total_results=21,
                total_pages=2,
                has_next=True,
            ),
            book_counts=(
                BookResultCount(
                    book_id="agot",
                    book_title="A Game of Thrones",
                    result_count=21,
                ),
            ),
            results=[
                {
                    "rank": 1,
                    "score": 0.75,
                    "chunk": {
                        "id": "agot-001-prologue-c001",
                        "chapter_id": "agot-001-prologue",
                        "chapter_title": "PROLOGUE",
                        "chapter_sequence": 1,
                        "pov": "PROLOGUE",
                        "pov_ordinal": 1,
                        "chunk_ordinal": 1,
                        "word_start": 0,
                        "word_end": 180,
                        "book_id": "agot",
                        "book_title": "A Game of Thrones",
                        "book_sequence": 1,
                    },
                    "context_before": "The preceding passage fragment.",
                    "excerpt": "A short test excerpt.",
                    "context_after": "The following passage fragment.",
                    "retrieval": {"mode": "hierarchical-hybrid"},
                }
            ],
        )


class InvalidFilterRuntime(FakeRuntime):
    async def search(self, request: SearchRequest) -> SearchPayload:
        del request
        raise SearchValidationError("unknown book 'missing'")


class BrokenIndexRuntime(FakeRuntime):
    async def catalog(self) -> SearchCatalog:
        self.catalog_calls += 1
        raise IndexValidationError("private index path must not be exposed")


def settings() -> ApiSettings:
    return ApiSettings(
        index_path=Path("/tmp/test-index"),
        origin_token="test-origin-token",
    )


def test_health_does_not_load_the_model() -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(settings=settings(), runtime=runtime))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": False,
        "index_configured": True,
    }


def test_readiness_loads_and_validates_only_the_index() -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(settings=settings(), runtime=runtime))

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_loaded": False,
        "index_configured": True,
    }
    assert runtime.catalog_calls == 1


def test_readiness_rejects_an_unavailable_index_without_exposing_details() -> None:
    runtime = BrokenIndexRuntime()
    client = TestClient(create_app(settings=settings(), runtime=runtime))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Search index is unavailable."}
    assert "private index path" not in response.text


def test_search_requires_the_origin_credential() -> None:
    client = TestClient(create_app(settings=settings(), runtime=FakeRuntime()))

    response = client.post(
        "/v1/search/",
        content="this is deliberately not valid JSON",
        follow_redirects=False,
    )

    assert response.status_code == 401


def test_catalog_returns_indexed_books_and_povs() -> None:
    client = TestClient(create_app(settings=settings(), runtime=FakeRuntime()))

    response = client.get(
        "/v1/catalog",
        headers={ORIGIN_TOKEN_HEADER: "test-origin-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "books": [
            {
                "book_id": "agot",
                "book_title": "A Game of Thrones",
                "book_sequence": 1,
                "povs": ["ARYA", "EDDARD", "PROLOGUE"],
            }
        ]
    }


def test_search_returns_typed_results() -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(settings=settings(), runtime=runtime))

    response = client.post(
        "/v1/search",
        headers={ORIGIN_TOKEN_HEADER: "test-origin-token"},
        json={
            "query": "  blue   flower at the Wall  ",
            "mode": "lexical",
            "page": 1,
            "page_size": 20,
            "book": "AGOT",
            "povs": ["arya", "BRAN", "arya"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "blue flower at the Wall"
    assert payload["duration_ms"] == 12.346
    assert payload["results"][0]["chunk"]["book_id"] == "agot"
    assert payload["results"][0]["context_before"] == "The preceding passage fragment."
    assert payload["results"][0]["context_after"] == "The following passage fragment."
    assert payload["pagination"] == {
        "page": 1,
        "page_size": 20,
        "total_results": 21,
        "total_pages": 2,
        "has_next": True,
    }
    assert payload["book_counts"] == [
        {
            "book_id": "agot",
            "book_title": "A Game of Thrones",
            "result_count": 21,
        }
    ]
    assert runtime.last_request is not None
    assert runtime.last_request.mode == "lexical"
    assert runtime.last_request.povs == ["ARYA", "BRAN"]


def test_search_rejects_expansive_requests() -> None:
    client = TestClient(create_app(settings=settings(), runtime=FakeRuntime()))

    response = client.post(
        "/v1/search",
        headers={ORIGIN_TOKEN_HEADER: "test-origin-token"},
        json={"query": "blue flower", "top": 50},
    )

    assert response.status_code == 422

    response = client.post(
        "/v1/search",
        headers={ORIGIN_TOKEN_HEADER: "test-origin-token"},
        json={
            "query": "blue flower",
            "mode": "lexical",
            "page": 1,
            "page_size": 51,
        },
    )

    assert response.status_code == 422


def test_search_returns_422_for_filters_unknown_to_the_active_index() -> None:
    client = TestClient(
        create_app(settings=settings(), runtime=InvalidFilterRuntime())
    )

    response = client.post(
        "/v1/search",
        headers={ORIGIN_TOKEN_HEADER: "test-origin-token"},
        json={"query": "blue flower", "book": "missing"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "unknown book 'missing'"}


def test_search_logs_operational_fields_without_the_query(caplog) -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(settings=settings(), runtime=runtime))
    caplog.set_level(logging.INFO, logger="uvicorn.error.weirwood")

    response = client.post(
        "/v1/search",
        headers={ORIGIN_TOKEN_HEADER: "test-origin-token"},
        json={"query": "a private remembered phrase", "mode": "hybrid"},
    )

    assert response.status_code == 200
    assert "search_started mode=hybrid" in caplog.text
    assert "search_completed mode=hybrid" in caplog.text
    assert "result_count=1" in caplog.text
    assert "a private remembered phrase" not in caplog.text
