import type { ZodType } from "zod";

import type { components } from "./schema";
import {
  apiErrorResponseSchema,
  searchCatalogResponseSchema,
  searchResponseSchema
} from "./validation";

export type SearchRequest = components["schemas"]["SearchRequest"];
export type SearchResponse = components["schemas"]["SearchResponse"];
export type SearchCatalogResponse = components["schemas"]["SearchCatalogResponse"];

export type BrowserSearchRequest = SearchRequest & {
  turnstileToken: string;
};

export class ApiError extends Error {
  readonly status: number;
  readonly retryAfter: number | null;

  constructor(message: string, status: number, retryAfter: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

function parseRetryAfter(value: string | null): number | null {
  if (value === null) return null;

  const normalized = value.trim();
  if (/^\d+$/.test(normalized)) {
    const seconds = Number(normalized);
    return Number.isSafeInteger(seconds) ? seconds : null;
  }
  if (/^[+-]?\d/.test(normalized)) return null;

  const retryAt = Date.parse(normalized);
  if (!Number.isFinite(retryAt)) return null;

  return Math.max(0, Math.ceil((retryAt - Date.now()) / 1_000));
}

async function responseError(response: Response, fallback: string): Promise<ApiError> {
  let message = fallback;
  try {
    const payload = apiErrorResponseSchema.safeParse(await response.json());
    if (payload.success) {
      message = payload.data.detail ?? payload.data.message ?? message;
    }
  } catch {
    // Keep the fallback when the gateway returns a non-JSON response.
  }
  return new ApiError(
    message,
    response.status,
    parseRetryAfter(response.headers.get("Retry-After"))
  );
}

async function validatedJson<T>(
  response: Response,
  schema: ZodType<T>,
  fallback: string
): Promise<T> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new ApiError(fallback, 502, null);
  }

  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiError(fallback, 502, null);
  }
  return parsed.data;
}

export async function fetchSearchCatalog(
  signal: AbortSignal
): Promise<SearchCatalogResponse> {
  const response = await fetch("/api/catalog", { signal });
  if (!response.ok) {
    throw await responseError(response, "The search filters could not be loaded.");
  }
  return validatedJson(
    response,
    searchCatalogResponseSchema,
    "The search service returned an invalid catalog response."
  );
}

export async function searchBooks(
  request: BrowserSearchRequest,
  signal: AbortSignal
): Promise<SearchResponse> {
  const response = await fetch("/api/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request),
    signal
  });

  if (!response.ok) {
    throw await responseError(response, "The search could not be completed.");
  }

  return validatedJson(
    response,
    searchResponseSchema,
    "The search service returned an invalid search response."
  );
}
