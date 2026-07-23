import { z } from "zod";

import type { components } from "./schema";

type SearchCatalogResponse = components["schemas"]["SearchCatalogResponse"];
type SearchResponse = components["schemas"]["SearchResponse"];

const catalogBookSchema = z
  .object({
    book_id: z.string(),
    book_sequence: z.number().int(),
    book_title: z.string(),
    povs: z.array(z.string())
  })
  .passthrough();

export const searchCatalogResponseSchema: z.ZodType<SearchCatalogResponse> = z
  .object({
    books: z.array(catalogBookSchema)
  })
  .passthrough();

const passageFragmentSchema = z
  .object({
    region: z.enum(["before", "focus", "after"]),
    text: z.string()
  })
  .passthrough();

const passageParagraphSchema = z
  .object({
    fragments: z.array(passageFragmentSchema),
    id: z.string(),
    ordinal: z.number().int(),
    partial_end: z.boolean(),
    partial_start: z.boolean()
  })
  .passthrough();

const chunkMetadataSchema = z
  .object({
    book_id: z.string(),
    book_sequence: z.number().int(),
    book_title: z.string(),
    chapter_id: z.string(),
    chapter_sequence: z.number().int(),
    chapter_title: z.string(),
    chunk_ordinal: z.number().int(),
    id: z.string(),
    pov: z.string(),
    pov_ordinal: z.number().int(),
    word_end: z.number().int(),
    word_start: z.number().int()
  })
  .passthrough();

const searchResultSchema = z
  .object({
    chunk: chunkMetadataSchema,
    context_after: z.string().nullable().optional(),
    context_before: z.string().nullable().optional(),
    context_word_end: z.number().int().nullable().optional(),
    context_word_start: z.number().int().nullable().optional(),
    excerpt: z.string(),
    paragraphs: z.array(passageParagraphSchema).optional(),
    rank: z.number().int(),
    retrieval: z.record(z.string(), z.unknown()).nullable().optional(),
    score: z.number()
  })
  .passthrough();

const bookResultCountSchema = z
  .object({
    book_id: z.string(),
    book_title: z.string(),
    result_count: z.number().int()
  })
  .passthrough();

const searchPaginationSchema = z
  .object({
    has_next: z.boolean(),
    page: z.number().int(),
    page_size: z.number().int(),
    total_pages: z.number().int(),
    total_results: z.number().int()
  })
  .passthrough();

export const searchResponseSchema: z.ZodType<SearchResponse> = z
  .object({
    book_counts: z.array(bookResultCountSchema).optional(),
    cached: z.boolean(),
    duration_ms: z.number(),
    pagination: searchPaginationSchema.nullable().optional(),
    query: z.string(),
    result_count: z.number().int(),
    results: z.array(searchResultSchema)
  })
  .passthrough();

export const apiErrorResponseSchema = z
  .object({
    detail: z.string().optional(),
    message: z.string().optional()
  })
  .passthrough();
