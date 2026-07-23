import {
  ApiError,
  fetchSearchCatalog,
  searchBooks,
  type BrowserSearchRequest
} from "./client";

const request: BrowserSearchRequest = {
  query: "blue flower",
  mode: "hybrid",
  top: 5,
  book: null,
  pov: null,
  turnstileToken: "test-token"
};

function jsonResponse(
  payload: unknown,
  status = 200,
  headers: Record<string, string> = {}
): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json", ...headers }
  });
}

function validSearchResponse(overrides: Record<string, unknown> = {}) {
  return {
    query: "blue flower",
    result_count: 0,
    duration_ms: 4,
    cached: false,
    results: [],
    ...overrides
  };
}

describe("searchBooks", () => {
  it("posts and validates a search response while preserving unknown fields", async () => {
    const fetchMock = vi.spyOn(window, "fetch").mockResolvedValue(
      jsonResponse(validSearchResponse({ future_field: "preserved" }))
    );

    const response = await searchBooks(
      request,
      new AbortController().signal
    );

    expect(response.query).toBe("blue flower");
    expect(response).toHaveProperty("future_field", "preserved");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/search",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("rejects an invalid successful search response with a generic error", async () => {
    vi.spyOn(window, "fetch").mockResolvedValue(
      jsonResponse({ query: "blue flower", cached: false })
    );

    await expect(
      searchBooks(request, new AbortController().signal)
    ).rejects.toEqual(
      expect.objectContaining({
        name: "ApiError",
        message: "The search service returned an invalid search response.",
        status: 502
      })
    );
  });

  it("uses the fallback when an error payload is malformed", async () => {
    vi.spyOn(window, "fetch").mockResolvedValue(
      jsonResponse({ detail: [{ message: "internal validation detail" }] }, 422)
    );

    const error = await searchBooks(
      request,
      new AbortController().signal
    ).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toEqual(
      expect.objectContaining({
        message: "The search could not be completed.",
        status: 422
      })
    );
  });

  it("parses Retry-After delay seconds", async () => {
    vi.spyOn(window, "fetch").mockResolvedValue(
      jsonResponse(
        { message: "Too many requests." },
        429,
        { "Retry-After": "60" }
      )
    );

    const error = await searchBooks(
      request,
      new AbortController().signal
    ).catch((caught: unknown) => caught);

    expect(error).toEqual(
      expect.objectContaining({
        status: 429,
        retryAfter: 60
      })
    );
  });

  it("parses a future Retry-After HTTP date", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-23T12:00:00.000Z"));
    vi.spyOn(window, "fetch").mockResolvedValue(
      jsonResponse(
        { message: "Service unavailable." },
        503,
        { "Retry-After": "Thu, 23 Jul 2026 12:00:30 GMT" }
      )
    );

    try {
      const error = await searchBooks(
        request,
        new AbortController().signal
      ).catch((caught: unknown) => caught);

      expect(error).toEqual(
        expect.objectContaining({
          status: 503,
          retryAfter: 30
        })
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it.each(["-1", "1.5", "not-a-date", "9007199254740992"])(
    "ignores an invalid Retry-After value: %s",
    async (retryAfter) => {
      vi.spyOn(window, "fetch").mockResolvedValue(
        jsonResponse(
          { message: "Too many requests." },
          429,
          { "Retry-After": retryAfter }
        )
      );

      const error = await searchBooks(
        request,
        new AbortController().signal
      ).catch((caught: unknown) => caught);

      expect(error).toEqual(
        expect.objectContaining({
          retryAfter: null
        })
      );
    }
  );

  it("clamps an expired Retry-After HTTP date to zero", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-23T12:00:00.000Z"));
    vi.spyOn(window, "fetch").mockResolvedValue(
      jsonResponse(
        { message: "Service unavailable." },
        503,
        { "Retry-After": "Thu, 23 Jul 2026 11:59:30 GMT" }
      )
    );

    try {
      const error = await searchBooks(
        request,
        new AbortController().signal
      ).catch((caught: unknown) => caught);

      expect(error).toEqual(
        expect.objectContaining({
          retryAfter: 0
        })
      );
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("fetchSearchCatalog", () => {
  it("validates books and POVs while preserving unknown fields", async () => {
    const fetchMock = vi.spyOn(window, "fetch").mockResolvedValue(
      jsonResponse({
        books: [
          {
            book_id: "agot",
            book_title: "A Game of Thrones",
            book_sequence: 1,
            povs: ["ARYA", "EDDARD"],
            future_field: true
          }
        ],
        future_catalog_field: "preserved"
      })
    );

    const catalog = await fetchSearchCatalog(new AbortController().signal);

    expect(catalog.books[0]?.povs).toEqual(["ARYA", "EDDARD"]);
    expect(catalog.books[0]).toHaveProperty("future_field", true);
    expect(catalog).toHaveProperty("future_catalog_field", "preserved");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/catalog",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
  });

  it("rejects an invalid successful catalog response with a generic error", async () => {
    vi.spyOn(window, "fetch").mockResolvedValue(
      jsonResponse({
        books: [
          {
            book_id: "agot",
            book_title: "A Game of Thrones",
            book_sequence: 1
          }
        ]
      })
    );

    await expect(
      fetchSearchCatalog(new AbortController().signal)
    ).rejects.toEqual(
      expect.objectContaining({
        name: "ApiError",
        message: "The search service returned an invalid catalog response.",
        status: 502
      })
    );
  });
});
