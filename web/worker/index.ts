export interface Env {
  ASSETS: Fetcher;
  HYBRID_RATE_LIMITER: RateLimit;
  LEXICAL_RATE_LIMITER: RateLimit;
  LOCATION_RATE_LIMITER: RateLimit;
  API_ORIGIN: string;
  ORIGIN_TOKEN: string;
  TURNSTILE_SECRET_KEY: string;
  TURNSTILE_EXPECTED_HOSTNAME?: string;
  ORIGIN_TIMEOUT_MS?: string;
}

type SearchMode = "hybrid" | "lexical";

type BrowserSearchRequest = {
  query: string;
  mode: SearchMode;
  top?: number;
  page?: number;
  page_size?: number;
  book: string | null;
  pov: string | null;
  povs: string[] | null;
  turnstileToken: string;
};

type TurnstileResult = {
  success: boolean;
  hostname?: string;
  action?: string;
};

const MAX_BODY_BYTES = 4096;
const JSON_HEADERS = {
  "Content-Type": "application/json",
  "Cache-Control": "no-store",
  "X-Content-Type-Options": "nosniff"
};

function json(
  payload: Record<string, unknown>,
  status = 200,
  headers: Record<string, string> = {}
): Response {
  return Response.json(payload, {
    status,
    headers: { ...JSON_HEADERS, ...headers }
  });
}

function positiveInteger(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function validPov(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= 32 &&
    /^[a-zA-Z -]+$/.test(value)
  );
}

function validatePayload(value: unknown): BrowserSearchRequest | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }

  const payload = value as Record<string, unknown>;
  const allowed = new Set([
    "query",
    "mode",
    "top",
    "page",
    "page_size",
    "book",
    "pov",
    "povs",
    "turnstileToken"
  ]);
  if (Object.keys(payload).some((key) => !allowed.has(key))) return null;

  if (
    typeof payload.query !== "string" ||
    payload.query.trim().length === 0 ||
    payload.query.length > 300 ||
    typeof payload.turnstileToken !== "string" ||
    payload.turnstileToken.length === 0 ||
    payload.turnstileToken.length > 2048
  ) {
    return null;
  }

  if (
    payload.mode !== undefined &&
    payload.mode !== "hybrid" &&
    payload.mode !== "lexical"
  ) {
    return null;
  }

  const mode = (payload.mode as SearchMode | undefined) ?? "hybrid";
  if (
    mode === "hybrid" &&
    ((payload.top !== undefined &&
      (!Number.isInteger(payload.top) ||
        Number(payload.top) < 1 ||
        Number(payload.top) > 10)) ||
      payload.page !== undefined ||
      payload.page_size !== undefined)
  ) {
    return null;
  }
  if (
    mode === "lexical" &&
    (payload.top !== undefined ||
      (payload.page !== undefined &&
        (!Number.isInteger(payload.page) ||
          Number(payload.page) < 1 ||
          Number(payload.page) > 1000)) ||
      (payload.page_size !== undefined &&
        (!Number.isInteger(payload.page_size) ||
          Number(payload.page_size) < 1 ||
          Number(payload.page_size) > 50)))
  ) {
    return null;
  }

  const book = payload.book;
  if (
    book !== undefined &&
    book !== null &&
    (typeof book !== "string" ||
      book.length > 16 ||
      !/^[a-zA-Z0-9_-]+$/.test(book))
  ) {
    return null;
  }

  const pov = payload.pov;
  if (pov !== undefined && pov !== null && !validPov(pov)) return null;

  const povs = payload.povs;
  if (
    povs !== undefined &&
    povs !== null &&
    (!Array.isArray(povs) ||
      povs.length === 0 ||
      povs.length > 50 ||
      povs.some((item) => !validPov(item)))
  ) {
    return null;
  }
  if (pov != null && povs != null) return null;

  return {
    query: payload.query.trim().replace(/\s+/g, " "),
    mode,
    top:
      mode === "hybrid"
        ? payload.top === undefined
          ? 5
          : Number(payload.top)
        : undefined,
    page:
      mode === "lexical"
        ? payload.page === undefined
          ? 1
          : Number(payload.page)
        : undefined,
    page_size:
      mode === "lexical"
        ? payload.page_size === undefined
          ? 20
          : Number(payload.page_size)
        : undefined,
    book: (book as string | null | undefined) ?? null,
    pov: (pov as string | null | undefined) ?? null,
    povs: (povs as string[] | null | undefined) ?? null,
    turnstileToken: payload.turnstileToken
  };
}

async function verifyTurnstile(
  request: Request,
  env: Env,
  token: string
): Promise<boolean> {
  const form = new FormData();
  form.set("secret", env.TURNSTILE_SECRET_KEY);
  form.set("response", token);
  const clientIp = request.headers.get("CF-Connecting-IP");
  if (clientIp) form.set("remoteip", clientIp);

  const response = await fetch(
    "https://challenges.cloudflare.com/turnstile/v0/siteverify",
    { method: "POST", body: form }
  );
  if (!response.ok) return false;

  const result = (await response.json()) as TurnstileResult;
  if (!result.success || result.action !== "search") return false;
  return (
    !env.TURNSTILE_EXPECTED_HOSTNAME ||
    result.hostname === env.TURNSTILE_EXPECTED_HOSTNAME
  );
}

async function enforceRateLimits(
  request: Request,
  env: Env,
  mode: SearchMode
): Promise<boolean> {
  const clientKey = request.headers.get("CF-Connecting-IP") ?? "local-client";
  const clientLimiter =
    mode === "hybrid" ? env.HYBRID_RATE_LIMITER : env.LEXICAL_RATE_LIMITER;
  const client = await clientLimiter.limit({ key: clientKey });
  if (!client.success) return false;

  const location = await env.LOCATION_RATE_LIMITER.limit({ key: "search" });
  return location.success;
}

function originResponse(response: Response): Response {
  const contentType = response.headers.get("Content-Type") ?? "";
  if (!contentType.toLowerCase().includes("application/json")) {
    return json({ detail: "The search service returned an invalid response." }, 502);
  }
  if (response.status === 401 || response.status === 403) {
    return json({ detail: "The search service is temporarily unavailable." }, 503);
  }

  const headers = new Headers(JSON_HEADERS);
  const retryAfter = response.headers.get("Retry-After");
  if (retryAfter) headers.set("Retry-After", retryAfter);
  return new Response(response.body, { status: response.status, headers });
}

async function fetchOrigin(
  env: Env,
  path: string,
  init: RequestInit = {}
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    positiveInteger(env.ORIGIN_TIMEOUT_MS, 90_000)
  );
  const headers = new Headers(init.headers);
  headers.set("X-Weirwood-Origin-Token", env.ORIGIN_TOKEN);

  try {
    const response = await fetch(new URL(path, env.API_ORIGIN), {
      ...init,
      headers,
      signal: controller.signal
    });
    return originResponse(response);
  } catch {
    return json({ detail: "The search service is temporarily unavailable." }, 503);
  } finally {
    clearTimeout(timeout);
  }
}

async function handleCatalog(request: Request, env: Env): Promise<Response> {
  if (request.method !== "GET") {
    return json({ detail: "Method not allowed." }, 405, { Allow: "GET" });
  }
  return fetchOrigin(env, "/v1/catalog");
}

async function handleSearch(request: Request, env: Env): Promise<Response> {
  if (request.method !== "POST") {
    return json({ detail: "Method not allowed." }, 405, { Allow: "POST" });
  }

  const contentType = request.headers.get("Content-Type") ?? "";
  if (!contentType.toLowerCase().startsWith("application/json")) {
    return json({ detail: "Content-Type must be application/json." }, 415);
  }

  const contentLength = Number(request.headers.get("Content-Length") ?? "0");
  if (Number.isFinite(contentLength) && contentLength > MAX_BODY_BYTES) {
    return json({ detail: "Request body is too large." }, 413);
  }

  let rawBody: string;
  try {
    rawBody = await request.text();
  } catch {
    return json({ detail: "Could not read the request body." }, 400);
  }
  if (new TextEncoder().encode(rawBody).byteLength > MAX_BODY_BYTES) {
    return json({ detail: "Request body is too large." }, 413);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawBody);
  } catch {
    return json({ detail: "Request body must be valid JSON." }, 400);
  }

  const payload = validatePayload(parsed);
  if (!payload) return json({ detail: "Invalid search request." }, 422);

  try {
    if (!(await verifyTurnstile(request, env, payload.turnstileToken))) {
      return json({ detail: "Bot verification failed." }, 403);
    }
  } catch {
    return json({ detail: "Bot verification is temporarily unavailable." }, 503);
  }

  try {
    if (!(await enforceRateLimits(request, env, payload.mode))) {
      return json(
        { detail: "Search rate limit exceeded." },
        429,
        { "Retry-After": "60" }
      );
    }
  } catch {
    return json({ detail: "Search capacity checks are unavailable." }, 503);
  }

  return fetchOrigin(env, "/v1/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: payload.query,
      mode: payload.mode,
      top: payload.top,
      page: payload.page,
      page_size: payload.page_size,
      book: payload.book,
      pov: payload.pov,
      povs: payload.povs
    })
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const path = new URL(request.url).pathname.replace(/\/$/, "");
    if (path === "/api/search") return handleSearch(request, env);
    if (path === "/api/catalog") return handleCatalog(request, env);
    if (path.startsWith("/api/")) {
      return json({ detail: "Not found." }, 404);
    }
    return env.ASSETS.fetch(request);
  }
} satisfies ExportedHandler<Env>;
