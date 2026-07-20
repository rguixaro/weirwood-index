from __future__ import annotations

import asyncio
import hmac
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from weirwood_api.config import ApiSettings
from weirwood_api.schemas import (
    BookResultCountResponse,
    CatalogBookResponse,
    HealthResponse,
    SearchCatalogResponse,
    SearchPaginationResponse,
    SearchRequest,
    SearchResponse,
)
from weirwood_api.service import SearchRuntime, WeirwoodSearchRuntime
from weirwood_index.models import SearchValidationError, WeirwoodError

ORIGIN_TOKEN_HEADER = "X-Weirwood-Origin-Token"
logger = logging.getLogger("uvicorn.error.weirwood")


class SearchCapacity:
    def __init__(self, *, max_queued: int) -> None:
        self._semaphore = asyncio.Semaphore(1)
        self._state_lock = asyncio.Lock()
        self._accepted = 0
        self._capacity = 1 + max(0, max_queued)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        async with self._state_lock:
            if self._accepted >= self._capacity:
                logger.warning("search_busy capacity=%d", self._capacity)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="The search engine is busy. Try again shortly.",
                    headers={"Retry-After": "5"},
                )
            self._accepted += 1
        try:
            async with self._semaphore:
                yield
        finally:
            async with self._state_lock:
                self._accepted -= 1


def create_app(
    *,
    settings: ApiSettings | None = None,
    runtime: SearchRuntime | None = None,
) -> FastAPI:
    resolved_settings = settings or ApiSettings.from_environment()
    search_runtime = runtime or WeirwoodSearchRuntime(resolved_settings)
    capacity = SearchCapacity(max_queued=resolved_settings.max_queued_searches)

    application = FastAPI(
        title="Weirwood Index API",
        version="0.1.0",
        description="Passage search over a privately supplied narrative corpus.",
    )
    application.state.settings = resolved_settings
    application.state.search_runtime = search_runtime

    @application.middleware("http")
    async def authenticate_search_origin(request: Request, call_next):
        if request.url.path.rstrip("/") not in {"/v1/search", "/v1/catalog"}:
            return await call_next(request)
        expected = resolved_settings.origin_token
        if not expected:
            logger.error("origin_auth_unavailable path=%s", request.url.path)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Origin authentication is not configured."},
            )
        supplied = request.headers.get(ORIGIN_TOKEN_HEADER)
        if supplied is None or not hmac.compare_digest(supplied, expected):
            logger.warning("origin_auth_rejected path=%s", request.url.path)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid origin credential."},
            )
        return await call_next(request)

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=search_runtime.loaded,
            index_configured=resolved_settings.index_path is not None,
        )

    @application.get("/ready", response_model=HealthResponse, tags=["operations"])
    async def readiness() -> HealthResponse:
        configured = resolved_settings.index_path is not None
        if not configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search index is not configured.",
            )
        try:
            await search_runtime.catalog()
        except WeirwoodError as exc:
            logger.error(
                "readiness_failed error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search index is unavailable.",
            ) from exc
        return HealthResponse(
            status="ready",
            model_loaded=search_runtime.loaded,
            index_configured=True,
        )

    @application.post(
        "/v1/search",
        response_model=SearchResponse,
        tags=["search"],
        dependencies=[],
    )
    async def search(
        payload: SearchRequest,
    ) -> SearchResponse:
        started = time.perf_counter()
        pov_count = len(payload.povs) if payload.povs is not None else int(payload.pov is not None)
        logger.info(
            "search_started mode=%s book_filter=%s pov_filter_count=%d",
            payload.mode,
            payload.book is not None,
            pov_count,
        )
        try:
            async with capacity.acquire():
                result = await search_runtime.search(payload)
        except SearchValidationError as exc:
            logger.info(
                "search_rejected mode=%s reason=invalid_filter",
                payload.mode,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except WeirwoodError as exc:
            logger.error(
                "search_failed mode=%s error_type=%s",
                payload.mode,
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        request_duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "search_completed mode=%s cached=%s result_count=%d "
            "engine_duration_ms=%.3f request_duration_ms=%.3f",
            payload.mode,
            result.cached,
            len(result.results),
            result.duration_ms,
            request_duration_ms,
        )
        return SearchResponse(
            query=payload.query,
            result_count=len(result.results),
            duration_ms=round(result.duration_ms, 3),
            cached=result.cached,
            pagination=(
                SearchPaginationResponse(
                    page=result.pagination.page,
                    page_size=result.pagination.page_size,
                    total_results=result.pagination.total_results,
                    total_pages=result.pagination.total_pages,
                    has_next=result.pagination.has_next,
                )
                if result.pagination is not None
                else None
            ),
            book_counts=[
                BookResultCountResponse(
                    book_id=item.book_id,
                    book_title=item.book_title,
                    result_count=item.result_count,
                )
                for item in result.book_counts
            ],
            results=result.results,
        )

    @application.get(
        "/v1/catalog",
        response_model=SearchCatalogResponse,
        tags=["search"],
    )
    async def catalog() -> SearchCatalogResponse:
        try:
            result = await search_runtime.catalog()
        except WeirwoodError as exc:
            logger.error(
                "catalog_failed error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return SearchCatalogResponse(
            books=[
                CatalogBookResponse(
                    book_id=book.book_id,
                    book_title=book.book_title,
                    book_sequence=book.book_sequence,
                    povs=list(book.povs),
                )
                for book in result.books
            ]
        )

    return application


app = create_app()
