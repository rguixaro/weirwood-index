import { useEffect, useRef, useState } from "react";

type TurnstileApi = {
  render: (
    element: HTMLElement,
    options: {
      sitekey: string;
      action: string;
      appearance: "interaction-only";
      retry: "auto";
      "retry-interval": number;
      callback: (token: string) => void;
      "error-callback": (errorCode: string) => boolean;
      "expired-callback": () => void;
      "timeout-callback": () => void;
    }
  ) => string;
  reset: (widgetId: string) => void;
  remove: (widgetId: string) => void;
};

declare global {
  interface Window {
    turnstile?: TurnstileApi;
  }
}

type VerificationStatus = "loading" | "retrying" | "ready" | "unavailable";

const TURNSTILE_SCRIPT_ID = "weirwood-turnstile-script";
const TURNSTILE_SCRIPT_URL =
  "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
const LOAD_TIMEOUT_MS = 10_000;
const LOAD_RETRY_DELAY_MS = 1_000;
const WIDGET_RETRY_INTERVAL_MS = 8_000;

let turnstileLoader: Promise<TurnstileApi> | null = null;

function loadTurnstile(): Promise<TurnstileApi> {
  if (window.turnstile) {
    return Promise.resolve(window.turnstile);
  }
  if (turnstileLoader) {
    return turnstileLoader;
  }

  const loader = new Promise<TurnstileApi>((resolve, reject) => {
    const existing = document.getElementById(
      TURNSTILE_SCRIPT_ID
    ) as HTMLScriptElement | null;
    const script = existing ?? document.createElement("script");

    const cleanUp = () => {
      window.clearTimeout(timeout);
      script.removeEventListener("load", handleLoad);
      script.removeEventListener("error", handleError);
    };
    const fail = (error: Error) => {
      cleanUp();
      script.remove();
      reject(error);
    };
    const handleLoad = () => {
      if (window.turnstile) {
        cleanUp();
        resolve(window.turnstile);
      } else {
        fail(new Error("Turnstile loaded without an API."));
      }
    };
    const handleError = () => fail(new Error("Turnstile failed to load."));

    script.addEventListener("load", handleLoad);
    script.addEventListener("error", handleError);
    const timeout = window.setTimeout(
      () => fail(new Error("Turnstile did not load.")),
      LOAD_TIMEOUT_MS
    );

    if (!existing) {
      script.id = TURNSTILE_SCRIPT_ID;
      script.src = TURNSTILE_SCRIPT_URL;
      script.async = true;
      script.defer = true;
      document.head.append(script);
    }
  });

  turnstileLoader = loader.then(
    (turnstile) => {
      turnstileLoader = null;
      return turnstile;
    },
    (error: unknown) => {
      turnstileLoader = null;
      throw error;
    }
  );
  return turnstileLoader;
}

function waitForRetry(): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, LOAD_RETRY_DELAY_MS);
  });
}

type TurnstileWidgetProps = {
  onToken: (token: string | null) => void;
  resetKey: number;
};

export function TurnstileWidget({
  onToken,
  resetKey
}: TurnstileWidgetProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const widgetIdRef = useRef<string | null>(null);
  const [loadCycle, setLoadCycle] = useState(0);
  const siteKey = import.meta.env.VITE_TURNSTILE_SITE_KEY;
  const [status, setStatus] = useState<VerificationStatus>(() =>
    siteKey
      ? "loading"
      : import.meta.env.DEV
        ? "ready"
        : "unavailable"
  );

  useEffect(() => {
    if (!siteKey) {
      onToken(import.meta.env.DEV ? "local-development" : null);
      return;
    }

    let cancelled = false;

    const renderWidget = async () => {
      setStatus("loading");

      for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
          const turnstile = await loadTurnstile();
          if (cancelled || !containerRef.current || widgetIdRef.current) {
            return;
          }
          widgetIdRef.current = turnstile.render(containerRef.current, {
            sitekey: siteKey,
            action: "search",
            appearance: "interaction-only",
            retry: "auto",
            "retry-interval": WIDGET_RETRY_INTERVAL_MS,
            callback: (token) => {
              if (cancelled) return;
              setStatus("ready");
              onToken(token);
            },
            "error-callback": () => {
              if (!cancelled) {
                setStatus("retrying");
                onToken(null);
              }
              return false;
            },
            "expired-callback": () => {
              if (cancelled) return;
              setStatus("loading");
              onToken(null);
            },
            "timeout-callback": () => {
              if (cancelled) return;
              setStatus("retrying");
              onToken(null);
            }
          });
          return;
        } catch {
          if (cancelled) return;
          if (attempt === 0) {
            setStatus("retrying");
            await waitForRetry();
            if (cancelled) return;
          } else {
            setStatus("unavailable");
            onToken(null);
          }
        }
      }
    };

    void renderWidget();

    return () => {
      cancelled = true;
      if (widgetIdRef.current && window.turnstile) {
        window.turnstile.remove(widgetIdRef.current);
        widgetIdRef.current = null;
      }
    };
  }, [loadCycle, onToken, siteKey]);

  useEffect(() => {
    if (!siteKey) {
      onToken(import.meta.env.DEV ? "local-development" : null);
      return;
    }
    if (widgetIdRef.current && window.turnstile) {
      onToken(null);
      window.turnstile.reset(widgetIdRef.current);
    }
  }, [onToken, resetKey, siteKey]);

  return (
    <>
      <div
        ref={containerRef}
        className="min-h-8"
        aria-label="Bot verification"
      />
      {status === "retrying" ? (
        <p
          className="turnstile-feedback -mt-[0.35rem] mb-0 text-[0.86rem] font-semibold text-muted-ink"
          role="status"
        >
          Verification is retrying…
        </p>
      ) : null}
      {status === "unavailable" ? (
        <p
          className="turnstile-feedback is-error -mt-[0.35rem] mb-0 text-[0.86rem] font-semibold text-oxblood-dark"
          role="alert"
        >
          Verification could not load. Check your connection and retry.
          {siteKey ? (
            <>
              {" "}
              <button
                type="button"
                className="turnstile-retry-button cursor-pointer border-0 bg-transparent p-0 font-bold text-inherit underline underline-offset-[0.15em] focus-visible:outline-[3px] focus-visible:outline-offset-[3px] focus-visible:outline-oxblood"
                onClick={() => {
                  onToken(null);
                  setStatus("loading");
                  setLoadCycle((cycle) => cycle + 1);
                }}
              >
                Retry verification
              </button>
            </>
          ) : null}
        </p>
      ) : null}
    </>
  );
}
