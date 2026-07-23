import type {
  SearchCatalogResponse,
  SearchRequest,
  SearchResponse
} from "../../api/client";

export type SearchMode = "hybrid" | "lexical";

export type SearchValues = {
  query: string;
  mode: SearchMode;
  book: string;
  povs: string[];
};

export type SearchResult = SearchResponse["results"][number];

export type VolumeOption = {
  value: string;
  label: string;
  indexed: boolean;
};

export type HighlightSegment = {
  text: string;
  highlighted: boolean;
};

export const RESULTS_PER_PAGE = 5;
export const MIN_QUERY_LENGTH = 3;
export const MAX_QUERY_LENGTH = 300;

const SEARCH_RESULT_LIMIT = 10;
const LEXICAL_PAGE_SIZE = 10;

export const modeOptions: ReadonlyArray<{
  value: SearchMode;
  label: string;
}> = [
  { value: "hybrid", label: "Semantic" },
  { value: "lexical", label: "Lexical" }
];

const plannedVolumeOptions: ReadonlyArray<{
  value: string;
  label: string;
}> = [
  { value: "all", label: "All volumes" },
  { value: "agot", label: "A Game of Thrones" },
  { value: "acok", label: "A Clash of Kings" },
  { value: "asos", label: "A Storm of Swords" },
  { value: "affc", label: "A Feast for Crows" },
  { value: "adwd", label: "A Dance with Dragons" },
  { value: "thk", label: "The Hedge Knight" },
  { value: "tss", label: "The Sworn Sword" },
  { value: "tmk", label: "The Mystery Knight" },
  { value: "fab", label: "Fire & Blood" }
];

const ignoredQueryTerms = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "had",
  "has",
  "have",
  "he",
  "her",
  "him",
  "his",
  "i",
  "in",
  "is",
  "it",
  "its",
  "of",
  "on",
  "or",
  "she",
  "that",
  "the",
  "their",
  "them",
  "they",
  "this",
  "to",
  "was",
  "were",
  "with",
  "you"
]);

export function readQueryFromUrl(search: string): string {
  return new URLSearchParams(search).get("q") ?? "";
}

export function normalizeQuery(query: string): string {
  return query.trim().replace(/\s+/g, " ");
}

export function isValidQuery(query: string): boolean {
  const length = query.trim().length;
  return length >= MIN_QUERY_LENGTH && length <= MAX_QUERY_LENGTH;
}

export function buildSearchRequest(values: SearchValues): SearchRequest {
  const query = normalizeQuery(values.query);
  const filters = {
    book: values.book === "all" ? null : values.book,
    povs: values.povs.length > 0 ? values.povs : null
  };

  return values.mode === "lexical"
    ? {
        query,
        mode: "lexical",
        page: 1,
        page_size: LEXICAL_PAGE_SIZE,
        ...filters
      }
    : {
        query,
        mode: "hybrid",
        top: SEARCH_RESULT_LIMIT,
        ...filters
      };
}

export function buildVolumeOptions(
  catalog: SearchCatalogResponse | undefined
): VolumeOption[] {
  const catalogBooksById = new Map(
    (catalog?.books ?? []).map((book) => [book.book_id, book])
  );
  const configuredBookIds = new Set(
    plannedVolumeOptions.map((option) => option.value)
  );

  return [
    ...plannedVolumeOptions.map((option) => ({
      ...option,
      label: catalogBooksById.get(option.value)?.book_title ?? option.label,
      indexed: option.value === "all" || catalogBooksById.has(option.value)
    })),
    ...(catalog?.books ?? [])
      .filter((book) => !configuredBookIds.has(book.book_id))
      .map((book) => ({
        value: book.book_id,
        label: book.book_title,
        indexed: true
      }))
  ];
}

export function availablePovs(
  catalog: SearchCatalogResponse | undefined,
  book: string
): string[] {
  if (!catalog?.books) return [];
  if (book !== "all") {
    return catalog.books.find((item) => item.book_id === book)?.povs ?? [];
  }
  return Array.from(new Set(catalog.books.flatMap((item) => item.povs))).sort();
}

export function formatPovLabel(pov: string): string {
  return pov
    .toLocaleLowerCase()
    .replace(/(^|[ -])\p{L}/gu, (letter) => letter.toLocaleUpperCase());
}

export function filterSummary(
  book: string,
  povs: string[],
  volumes: ReadonlyArray<VolumeOption>
): string {
  const volumeLabel =
    volumes.find((volume) => volume.value === book)?.label ?? "All volumes";
  const povSummary =
    povs.length === 0
      ? "All POVs"
      : povs.length === 1
        ? formatPovLabel(povs[0])
        : `${formatPovLabel(povs[0])} +${povs.length - 1}`;
  return `${volumeLabel} · ${povSummary}`;
}

function escapeRegularExpression(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function meaningfulQueryTerms(query: string): string[] {
  const terms = query.match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) ?? [];
  return Array.from(
    new Map(
      terms
        .filter((term) => term.length >= 3)
        .filter((term) => !ignoredQueryTerms.has(term.toLocaleLowerCase()))
        .map((term) => [term.toLocaleLowerCase(), term])
    ).values()
  ).sort((left, right) => right.length - left.length);
}

export function createHighlightMatcher(query: string): RegExp | null {
  const terms = meaningfulQueryTerms(query);
  if (terms.length === 0) return null;

  const alternatives = terms.map(escapeRegularExpression).join("|");
  return new RegExp(
    `(^|[^\\p{L}\\p{N}'’])(${alternatives})(?![\\p{L}\\p{N}'’])`,
    "giu"
  );
}

export function highlightSegments(
  text: string,
  matcher: RegExp | null
): HighlightSegment[] {
  if (!matcher) return [{ text, highlighted: false }];

  const segments: HighlightSegment[] = [];
  let cursor = 0;

  for (const match of text.matchAll(matcher)) {
    const index = match.index ?? 0;
    const prefix = match[1] ?? "";
    const matchedTerm = match[2] ?? "";
    if (index > cursor) {
      segments.push({
        text: text.slice(cursor, index),
        highlighted: false
      });
    }
    if (prefix) {
      segments.push({ text: prefix, highlighted: false });
    }
    segments.push({ text: matchedTerm, highlighted: true });
    cursor = index + match[0].length;
  }

  if (segments.length === 0) return [{ text, highlighted: false }];
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), highlighted: false });
  }
  return segments;
}
