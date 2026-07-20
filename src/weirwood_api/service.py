from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from weirwood_api.config import ApiSettings
from weirwood_api.schemas import SearchRequest
from weirwood_index.embedding import Encoder, create_encoder
from weirwood_index.indexing import LoadedIndex, load_index
from weirwood_index.lexical import BM25Index, best_query_span
from weirwood_index.models import Chunk, Paragraph, SearchValidationError, WeirwoodError
from weirwood_index.retrieval import lexical_search_page, search_index

logger = logging.getLogger("uvicorn.error.weirwood.runtime")


@dataclass(frozen=True)
class SearchPagination:
    page: int
    page_size: int
    total_results: int
    total_pages: int
    has_next: bool


@dataclass(frozen=True)
class BookResultCount:
    book_id: str
    book_title: str
    result_count: int


@dataclass(frozen=True)
class CatalogBook:
    book_id: str
    book_title: str
    book_sequence: int
    povs: tuple[str, ...]


@dataclass(frozen=True)
class SearchCatalog:
    books: tuple[CatalogBook, ...]


def catalog_from_chunks(chunks: Sequence[Chunk]) -> SearchCatalog:
    books: dict[str, tuple[int, str, set[str]]] = {}
    for chunk in chunks:
        if chunk.book_id not in books:
            books[chunk.book_id] = (
                chunk.book_sequence,
                chunk.book_title,
                set(),
            )
        books[chunk.book_id][2].add(chunk.pov)
    return SearchCatalog(
        books=tuple(
            CatalogBook(
                book_id=book_id,
                book_title=book_title,
                book_sequence=book_sequence,
                povs=tuple(sorted(povs)),
            )
            for book_id, (book_sequence, book_title, povs) in sorted(
                books.items(), key=lambda item: item[1][0]
            )
        )
    )


@dataclass(frozen=True)
class SearchPayload:
    results: list[dict[str, Any]]
    duration_ms: float
    cached: bool
    pagination: SearchPagination | None = None
    book_counts: tuple[BookResultCount, ...] = ()


@dataclass(frozen=True)
class PassagePreview:
    text: str
    word_start: int
    word_end: int
    before: str | None
    after: str | None


def validate_search_filters(catalog: SearchCatalog, request: SearchRequest) -> None:
    books = {book.book_id: book for book in catalog.books}
    if request.book is not None and request.book not in books:
        raise SearchValidationError(
            f"unknown book {request.book!r}; choose one of: {', '.join(books)}"
        )

    selected_books = (
        (books[request.book],) if request.book is not None else catalog.books
    )
    available_povs = {
        pov for book in selected_books for pov in book.povs
    }
    requested_povs = set(request.povs or ([request.pov] if request.pov else []))
    unknown_povs = requested_povs - available_povs
    if unknown_povs:
        raise SearchValidationError(
            f"unknown POV {', '.join(sorted(unknown_povs))!r}; choose one of: "
            f"{', '.join(sorted(available_povs))}"
        )


class SearchRuntime(Protocol):
    @property
    def loaded(self) -> bool: ...

    async def catalog(self) -> SearchCatalog: ...

    async def search(self, request: SearchRequest) -> SearchPayload: ...


CONTEXT_WORDS = 28


def _neighbor_fragment(
    neighbor: Chunk | None,
    *,
    global_start: int,
    global_end: int,
    take_from_end: bool,
) -> str | None:
    if neighbor is None or global_start >= global_end:
        return None
    words = neighbor.text.split()
    local_start = max(global_start, neighbor.word_start) - neighbor.word_start
    local_end = min(global_end, neighbor.word_end) - neighbor.word_start
    available = words[max(0, local_start) : max(0, local_end)]
    if not available:
        return None
    selected = available[-CONTEXT_WORDS:] if take_from_end else available[:CONTEXT_WORDS]
    return " ".join(selected)


def neighboring_context(
    chunk: Chunk,
    chunks_by_location: dict[tuple[str, int], Chunk],
) -> tuple[str | None, str | None]:
    previous = chunks_by_location.get((chunk.chapter_id, chunk.chunk_ordinal - 1))
    following = chunks_by_location.get((chunk.chapter_id, chunk.chunk_ordinal + 1))
    before = _neighbor_fragment(
        previous,
        global_start=previous.word_start if previous else chunk.word_start,
        global_end=chunk.word_start,
        take_from_end=True,
    )
    after = _neighbor_fragment(
        following,
        global_start=chunk.word_end,
        global_end=following.word_end if following else chunk.word_end,
        take_from_end=False,
    )
    return before, after


def _focus_word_end(chunk: Chunk, excerpt_chars: int) -> int:
    words = chunk.text.split()
    selected = 0
    characters = 0
    for word in words:
        additional = len(word) + (1 if selected else 0)
        if selected and characters + additional > excerpt_chars:
            break
        characters += additional
        selected += 1
    return chunk.word_start + max(1, selected)


def passage_preview(
    chunk: Chunk,
    query: str,
    *,
    excerpt_chars: int,
) -> PassagePreview | None:
    match = best_query_span(chunk.text, query)
    word_spans = list(re.finditer(r"\S+", chunk.text))
    if not word_spans:
        return None

    if match is None:
        focus_start = len(chunk.text) // 2
        focus_end = focus_start
        center = min(
            range(len(word_spans)),
            key=lambda position: abs(
                (word_spans[position].start() + word_spans[position].end()) // 2
                - focus_start
            ),
        )
        start = center
        end = center + 1
    else:
        focus_start = match.start
        focus_end = match.end
        start = next(
            position for position, span in enumerate(word_spans) if span.end() > match.start
        )
        end = 1 + max(
            position for position, span in enumerate(word_spans) if span.start() < match.end
        )

    if word_spans[end - 1].end() - word_spans[start].start() > excerpt_chars:
        focus_midpoint = (focus_start + focus_end) // 2
        center = min(
            range(start, end),
            key=lambda position: abs(
                (word_spans[position].start() + word_spans[position].end()) // 2
                - focus_midpoint
            ),
        )
        start = center
        end = center + 1

    while True:
        current_start = word_spans[start].start()
        current_end = word_spans[end - 1].end()
        left_padding = max(0, focus_start - current_start)
        right_padding = max(0, current_end - focus_end)
        candidates: list[tuple[int, str]] = []
        if start > 0:
            left_length = current_end - word_spans[start - 1].start()
            if left_length <= excerpt_chars:
                candidates.append((left_padding, "left"))
        if end < len(word_spans):
            right_length = word_spans[end].end() - current_start
            if right_length <= excerpt_chars:
                candidates.append((right_padding, "right"))
        if not candidates:
            break
        side = min(candidates)[1]
        if side == "left":
            start -= 1
        else:
            end += 1

    words = chunk.text.split()
    return PassagePreview(
        text=chunk.text[word_spans[start].start() : word_spans[end - 1].end()],
        word_start=chunk.word_start + start,
        word_end=chunk.word_start + end,
        before=" ".join(words[max(0, start - CONTEXT_WORDS) : start]) or None,
        after=" ".join(words[end : end + CONTEXT_WORDS]) or None,
    )


def paragraph_context(
    chunk: Chunk,
    paragraphs: tuple[Paragraph, ...],
    *,
    excerpt_chars: int,
    focus_word_start: int | None = None,
    focus_word_end: int | None = None,
) -> list[dict[str, Any]]:
    if not paragraphs:
        return []
    focus_start = chunk.word_start if focus_word_start is None else focus_word_start
    focus_end = (
        min(chunk.word_end, _focus_word_end(chunk, excerpt_chars))
        if focus_word_end is None
        else focus_word_end
    )
    window_start = max(0, focus_start - CONTEXT_WORDS)
    window_end = min(paragraphs[-1].word_end, focus_end + CONTEXT_WORDS)
    rendered: list[dict[str, Any]] = []
    for paragraph in paragraphs:
        start = max(window_start, paragraph.word_start)
        end = min(window_end, paragraph.word_end)
        if start >= end:
            continue
        boundaries = sorted(
            {start, end, max(start, min(end, focus_start)), max(start, min(end, focus_end))}
        )
        words = paragraph.text.split()
        fragments: list[dict[str, str]] = []
        for fragment_start, fragment_end in zip(
            boundaries, boundaries[1:], strict=False
        ):
            if fragment_start >= fragment_end:
                continue
            if fragment_end <= focus_start:
                region = "before"
            elif fragment_start >= focus_end:
                region = "after"
            else:
                region = "focus"
            local_start = fragment_start - paragraph.word_start
            local_end = fragment_end - paragraph.word_start
            fragments.append(
                {"region": region, "text": " ".join(words[local_start:local_end])}
            )
        rendered.append(
            {
                "id": paragraph.id,
                "ordinal": paragraph.ordinal,
                "partial_start": start > paragraph.word_start,
                "partial_end": end < paragraph.word_end,
                "fragments": fragments,
            }
        )
    return rendered


class WeirwoodSearchRuntime:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._index: LoadedIndex | None = None
        self._encoder: Encoder | None = None
        self._lexical_index: BM25Index | None = None
        self._chunks_by_location: dict[tuple[str, int], Chunk] = {}
        self._paragraphs_by_chapter: dict[str, tuple[Paragraph, ...]] = {}
        self._catalog: SearchCatalog | None = None
        self._index_load_lock = asyncio.Lock()
        self._encoder_load_lock = asyncio.Lock()
        self._cache: OrderedDict[
            tuple[
                str,
                str,
                int | None,
                int | None,
                int | None,
                str | None,
                str | None,
                tuple[str, ...] | None,
            ],
            tuple[
                list[dict[str, Any]],
                SearchPagination | None,
                tuple[BookResultCount, ...],
            ],
        ] = OrderedDict()

    @property
    def loaded(self) -> bool:
        return self._index is not None and self._encoder is not None

    def _load_index(self) -> None:
        started = time.perf_counter()
        index_path = self.settings.index_path
        if index_path is None:
            raise WeirwoodError("WEIRWOOD_INDEX_PATH is not configured")
        resolved = Path(index_path).expanduser().resolve()
        index = load_index(resolved, verify_source=self.settings.verify_source)
        self._index = index
        self._lexical_index = BM25Index.from_chunks(index.chunks)
        self._chunks_by_location = {
            (chunk.chapter_id, chunk.chunk_ordinal): chunk for chunk in index.chunks
        }
        paragraphs_by_chapter: dict[str, list[Paragraph]] = {}
        for paragraph in index.paragraphs:
            paragraphs_by_chapter.setdefault(paragraph.chapter_id, []).append(paragraph)
        self._paragraphs_by_chapter = {
            chapter_id: tuple(paragraphs)
            for chapter_id, paragraphs in paragraphs_by_chapter.items()
        }
        self._catalog = catalog_from_chunks(index.chunks)
        logger.info(
            "index_loaded book_count=%d chunk_count=%d paragraph_count=%d duration_ms=%.3f",
            len(self._catalog.books),
            len(index.chunks),
            len(index.paragraphs),
            (time.perf_counter() - started) * 1000,
        )

    def _load_encoder(self) -> None:
        started = time.perf_counter()
        assert self._index is not None
        model = self._index.manifest["model"]
        encoder = create_encoder(
            model_id=model["id"],
            revision=model["revision"],
            batch_size=self.settings.encoder_batch_size,
            show_progress=False,
        )
        self._encoder = encoder
        logger.info(
            "model_loaded model_id=%s duration_ms=%.3f",
            model["id"],
            (time.perf_counter() - started) * 1000,
        )

    async def _ensure_index_loaded(self) -> None:
        if self._index is not None and self._lexical_index is not None:
            return
        async with self._index_load_lock:
            if self._index is None or self._lexical_index is None:
                await asyncio.to_thread(self._load_index)

    async def _ensure_encoder_loaded(self) -> None:
        await self._ensure_index_loaded()
        if self._encoder is not None:
            return
        async with self._encoder_load_lock:
            if self._encoder is None:
                await asyncio.to_thread(self._load_encoder)

    async def catalog(self) -> SearchCatalog:
        await self._ensure_index_loaded()
        assert self._catalog is not None
        return self._catalog

    def _search_sync(
        self, request: SearchRequest
    ) -> tuple[
        list[dict[str, Any]],
        SearchPagination | None,
        tuple[BookResultCount, ...],
    ]:
        assert self._index is not None
        assert self._lexical_index is not None
        if request.mode == "lexical":
            assert request.page is not None
            assert request.page_size is not None
            page = lexical_search_page(
                self._index,
                request.query,
                page=request.page,
                page_size=request.page_size,
                pov=request.povs or request.pov,
                book=request.book,
                lexical_index=self._lexical_index,
            )
            results = page.results
            pagination = SearchPagination(
                page=request.page,
                page_size=request.page_size,
                total_results=page.total_results,
                total_pages=math.ceil(page.total_results / request.page_size),
                has_next=request.page * request.page_size < page.total_results,
            )
            book_counts = page.book_counts
        else:
            assert self._encoder is not None
            assert request.top is not None
            results = search_index(
                self._index,
                request.query,
                self._encoder,
                mode="hybrid",
                top=request.top,
                pov=request.povs or request.pov,
                book=request.book,
                semantic_weight=0.5,
                hierarchical=True,
                passages_per_chapter=8,
                lexical_index=self._lexical_index,
            )
            pagination = None
            book_counts = dict(Counter(result.chunk.book_id for result in results))
        payloads: list[dict[str, Any]] = []
        for result in results:
            payload = result.to_dict(excerpt_chars=self.settings.excerpt_chars)
            preview = passage_preview(
                result.chunk,
                request.query,
                excerpt_chars=self.settings.excerpt_chars,
            )
            if preview is None:
                before, after = neighboring_context(
                    result.chunk, self._chunks_by_location
                )
            else:
                payload["excerpt"] = preview.text
                before, after = preview.before, preview.after
            payload["context_before"] = before
            payload["context_after"] = after
            paragraphs = self._paragraphs_by_chapter.get(result.chunk.chapter_id, ())
            payload["paragraphs"] = paragraph_context(
                result.chunk,
                paragraphs,
                excerpt_chars=self.settings.excerpt_chars,
                focus_word_start=preview.word_start if preview else None,
                focus_word_end=preview.word_end if preview else None,
            )
            payloads.append(payload)
        books: dict[str, tuple[int, str]] = {}
        for chunk in self._index.chunks:
            books.setdefault(
                chunk.book_id,
                (chunk.book_sequence, chunk.book_title),
            )
        summary = tuple(
            BookResultCount(
                book_id=book_id,
                book_title=book_title,
                result_count=book_counts.get(book_id, 0),
            )
            for book_id, (_book_sequence, book_title) in sorted(
                books.items(), key=lambda item: item[1][0]
            )
            if request.book is None or request.book == book_id
        )
        return payloads, pagination, summary

    async def search(self, request: SearchRequest) -> SearchPayload:
        cache_key = (
            request.query.casefold(),
            request.mode,
            request.top,
            request.page,
            request.page_size,
            request.book,
            request.pov,
            tuple(request.povs) if request.povs is not None else None,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            results, pagination, book_counts = cached
            return SearchPayload(
                results=results,
                duration_ms=0.0,
                cached=True,
                pagination=pagination,
                book_counts=book_counts,
            )

        await self._ensure_index_loaded()
        assert self._catalog is not None
        validate_search_filters(self._catalog, request)
        if request.mode == "hybrid":
            await self._ensure_encoder_loaded()
        started = time.perf_counter()
        results, pagination, book_counts = await asyncio.to_thread(
            self._search_sync, request
        )
        duration_ms = (time.perf_counter() - started) * 1000

        if self.settings.cache_size > 0:
            self._cache[cache_key] = (results, pagination, book_counts)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.settings.cache_size:
                self._cache.popitem(last=False)

        return SearchPayload(
            results=results,
            duration_ms=duration_ms,
            cached=False,
            pagination=pagination,
            book_counts=book_counts,
        )
