from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PovName = Annotated[str, Field(max_length=32, pattern=r"^[a-zA-Z -]+$")]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    mode: Literal["hybrid", "lexical"] = "hybrid"
    top: int | None = Field(default=None, ge=1, le=10)
    page: int | None = Field(default=None, ge=1, le=1000)
    page_size: int | None = Field(default=None, ge=1, le=50)
    book: str | None = Field(default=None, max_length=16, pattern=r"^[a-zA-Z0-9_-]+$")
    pov: PovName | None = None
    povs: list[PovName] | None = Field(default=None, min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_mode_options(self) -> SearchRequest:
        if self.pov is not None and self.povs is not None:
            raise ValueError("pov and povs cannot be used together")
        if self.mode == "hybrid":
            if self.page is not None or self.page_size is not None:
                raise ValueError("page and page_size are only available in lexical mode")
            self.top = self.top or 5
        else:
            if self.top is not None:
                raise ValueError("top is only available in hybrid mode")
            self.page = self.page or 1
            self.page_size = self.page_size or 20
        return self

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized

    @field_validator("book")
    @classmethod
    def normalize_book(cls, value: str | None) -> str | None:
        return value.casefold() if value else None

    @field_validator("pov")
    @classmethod
    def normalize_pov(cls, value: str | None) -> str | None:
        return value.upper() if value else None

    @field_validator("povs")
    @classmethod
    def normalize_povs(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return sorted({pov.upper() for pov in value})


class ChunkMetadata(BaseModel):
    id: str
    chapter_id: str
    chapter_title: str
    chapter_sequence: int
    pov: str
    pov_ordinal: int
    chunk_ordinal: int
    word_start: int
    word_end: int
    book_id: str
    book_title: str
    book_sequence: int


class PassageFragmentResponse(BaseModel):
    region: Literal["before", "focus", "after"]
    text: str


class PassageParagraphResponse(BaseModel):
    id: str
    ordinal: int
    partial_start: bool
    partial_end: bool
    fragments: list[PassageFragmentResponse]


class SearchResultResponse(BaseModel):
    rank: int
    score: float
    chunk: ChunkMetadata
    context_before: str | None = None
    excerpt: str
    context_after: str | None = None
    retrieval: dict[str, Any] | None = None
    context_word_start: int | None = None
    context_word_end: int | None = None
    paragraphs: list[PassageParagraphResponse] = Field(default_factory=list)


class SearchPaginationResponse(BaseModel):
    page: int
    page_size: int
    total_results: int
    total_pages: int
    has_next: bool


class BookResultCountResponse(BaseModel):
    book_id: str
    book_title: str
    result_count: int


class CatalogBookResponse(BaseModel):
    book_id: str
    book_title: str
    book_sequence: int
    povs: list[str]


class SearchCatalogResponse(BaseModel):
    books: list[CatalogBookResponse]


class SearchResponse(BaseModel):
    query: str
    result_count: int
    duration_ms: float
    cached: bool
    pagination: SearchPaginationResponse | None = None
    book_counts: list[BookResultCountResponse] = Field(default_factory=list)
    results: list[SearchResultResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_configured: bool
