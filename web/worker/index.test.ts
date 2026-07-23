import worker, { type Env } from "./index";

type MockLimiter = Env["HYBRID_RATE_LIMITER"] & {
  limit: ReturnType<typeof vi.fn>;
};

function limiter(success = true): MockLimiter {
  return {
    limit: vi.fn().mockResolvedValue({ success })
  } as unknown as MockLimiter;
}

function environment(overrides: Partial<Env> = {}): Env {
  return {
    ASSETS: {
      fetch: vi.fn().mockResolvedValue(new Response("asset"))
    } as unknown as Env["ASSETS"],
    HYBRID_RATE_LIMITER: limiter(),
    LEXICAL_RATE_LIMITER: limiter(),
    LOCATION_RATE_LIMITER: limiter(),
    API_ORIGIN: "https://api.weirwoodindex.com",
    ORIGIN_TOKEN: "origin-secret",
    TURNSTILE_SECRET_KEY: "turnstile-secret",
    TURNSTILE_EXPECTED_HOSTNAME: "weirwoodindex.com",
    ORIGIN_TIMEOUT_MS: "1000",
    ...overrides
  };
}

function searchRequest(payload: Record<string, unknown>): Request {
  return new Request("https://weirwoodindex.com/api/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "CF-Connecting-IP": "192.0.2.10"
    },
    body: JSON.stringify(payload)
  });
}

describe("search gateway", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("verifies the client and forwards a normalized search without its token", async () => {
    const env = environment();
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        Response.json({
          success: true,
          action: "search",
          hostname: "weirwoodindex.com"
        })
      )
      .mockResolvedValueOnce(
        Response.json({
          query: "water dance",
          result_count: 0,
          duration_ms: 1,
          cached: false,
          results: []
        })
      );

    const response = await worker.fetch(
      searchRequest({
        query: "  water   dance  ",
        mode: "hybrid",
        top: 3,
        book: "agot",
        turnstileToken: "client-token"
      }),
      env
    );

    expect(response.status).toBe(200);
    expect(env.HYBRID_RATE_LIMITER.limit).toHaveBeenCalledWith({
      key: "192.0.2.10"
    });
    expect(env.LEXICAL_RATE_LIMITER.limit).not.toHaveBeenCalled();
    expect(env.LOCATION_RATE_LIMITER.limit).toHaveBeenCalledWith({ key: "search" });
    expect(
      (env.HYBRID_RATE_LIMITER as MockLimiter).limit.mock.invocationCallOrder[0] ?? 0
    ).toBeLessThan(
      (env.LOCATION_RATE_LIMITER as MockLimiter).limit.mock.invocationCallOrder[0] ??
        0
    );

    const [originUrl, originInit] = fetchMock.mock.calls[1] ?? [];
    expect(String(originUrl)).toBe("https://api.weirwoodindex.com/v1/search");
    const headers = new Headers(originInit?.headers);
    expect(headers.get("X-Weirwood-Origin-Token")).toBe("origin-secret");
    expect(JSON.parse(String(originInit?.body))).toEqual({
      query: "water dance",
      mode: "hybrid",
      top: 3,
      book: "agot",
      pov: null,
      povs: null
    });
  });

  it("rejects invalid requests before external calls", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const response = await worker.fetch(
      searchRequest({ query: "", turnstileToken: "client-token" }),
      environment()
    );

    expect(response.status).toBe(422);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects failed bot verification before checking capacity", async () => {
    const env = environment();
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({ success: false })
    );

    const response = await worker.fetch(
      searchRequest({ query: "water dance", turnstileToken: "bad-token" }),
      env
    );

    expect(response.status).toBe(403);
    expect(env.HYBRID_RATE_LIMITER.limit).not.toHaveBeenCalled();
    expect(env.LOCATION_RATE_LIMITER.limit).not.toHaveBeenCalled();
  });

  it("returns a retry interval when the client limit is reached", async () => {
    const env = environment({ HYBRID_RATE_LIMITER: limiter(false) });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({
        success: true,
        action: "search",
        hostname: "weirwoodindex.com"
      })
    );

    const response = await worker.fetch(
      searchRequest({ query: "water dance", turnstileToken: "client-token" }),
      env
    );

    expect(response.status).toBe(429);
    expect(response.headers.get("Retry-After")).toBe("60");
    expect(env.LOCATION_RATE_LIMITER.limit).not.toHaveBeenCalled();
  });

  it("checks the lexical client limit before location capacity", async () => {
    const env = environment();
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        Response.json({
          success: true,
          action: "search",
          hostname: "weirwoodindex.com"
        })
      )
      .mockResolvedValueOnce(
        Response.json({
          query: "water dance",
          result_count: 0,
          duration_ms: 1,
          cached: false,
          results: []
        })
      );

    const response = await worker.fetch(
      searchRequest({
        query: "water dance",
        mode: "lexical",
        page: 1,
        page_size: 20,
        turnstileToken: "client-token"
      }),
      env
    );

    expect(response.status).toBe(200);
    expect(env.LEXICAL_RATE_LIMITER.limit).toHaveBeenCalledWith({
      key: "192.0.2.10"
    });
    expect(env.HYBRID_RATE_LIMITER.limit).not.toHaveBeenCalled();
    expect(env.LOCATION_RATE_LIMITER.limit).toHaveBeenCalledWith({ key: "search" });
    expect(
      (env.LEXICAL_RATE_LIMITER as MockLimiter).limit.mock.invocationCallOrder[0] ?? 0
    ).toBeLessThan(
      (env.LOCATION_RATE_LIMITER as MockLimiter).limit.mock.invocationCallOrder[0] ??
        0
    );
  });

  it("returns a retry interval when location capacity is reached", async () => {
    const env = environment({ LOCATION_RATE_LIMITER: limiter(false) });
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({
        success: true,
        action: "search",
        hostname: "weirwoodindex.com"
      })
    );

    const response = await worker.fetch(
      searchRequest({ query: "water dance", turnstileToken: "client-token" }),
      env
    );

    expect(response.status).toBe(429);
    expect(response.headers.get("Retry-After")).toBe("60");
    expect(env.HYBRID_RATE_LIMITER.limit).toHaveBeenCalled();
    expect(env.LOCATION_RATE_LIMITER.limit).toHaveBeenCalledWith({ key: "search" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("forwards catalog requests with the private origin credential", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(Response.json({ books: [] }));
    const response = await worker.fetch(
      new Request("https://weirwoodindex.com/api/catalog"),
      environment()
    );

    expect(response.status).toBe(200);
    const [originUrl, originInit] = fetchMock.mock.calls[0] ?? [];
    expect(String(originUrl)).toBe("https://api.weirwoodindex.com/v1/catalog");
    expect(new Headers(originInit?.headers).get("X-Weirwood-Origin-Token")).toBe(
      "origin-secret"
    );
  });

  it("does not serve the SPA shell for unknown API routes", async () => {
    const env = environment();
    const response = await worker.fetch(
      new Request("https://weirwoodindex.com/api/missing"),
      env
    );

    expect(response.status).toBe(404);
    expect(env.ASSETS.fetch).not.toHaveBeenCalled();
  });
});
