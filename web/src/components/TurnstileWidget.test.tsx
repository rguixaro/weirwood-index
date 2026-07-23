import { act, fireEvent, render, screen } from "@testing-library/react";

import { TurnstileWidget } from "./TurnstileWidget";

const SCRIPT_ID = "weirwood-turnstile-script";
const SITE_KEY = "1x00000000000000000000AA";

type RenderOptions = Parameters<
  NonNullable<typeof window.turnstile>["render"]
>[1];

function injectedScript(): HTMLScriptElement {
  const script = document.getElementById(SCRIPT_ID);
  if (!(script instanceof HTMLScriptElement)) {
    throw new Error("Turnstile script was not injected.");
  }
  return script;
}

function turnstileApi() {
  let options: RenderOptions | null = null;
  const api: NonNullable<typeof window.turnstile> = {
    render: vi.fn((_element, renderOptions) => {
      options = renderOptions;
      return "widget-1";
    }),
    reset: vi.fn(),
    remove: vi.fn()
  };

  return {
    api,
    options: () => {
      if (!options) throw new Error("Turnstile was not rendered.");
      return options;
    }
  };
}

async function failScript(script: HTMLScriptElement) {
  await act(async () => {
    fireEvent.error(script);
    await Promise.resolve();
  });
}

async function loadScript(script: HTMLScriptElement) {
  await act(async () => {
    fireEvent.load(script);
    await Promise.resolve();
  });
}

beforeEach(() => {
  vi.stubEnv("VITE_TURNSTILE_SITE_KEY", SITE_KEY);
  delete window.turnstile;
  document.getElementById(SCRIPT_ID)?.remove();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  delete window.turnstile;
  document.getElementById(SCRIPT_ID)?.remove();
});

describe("TurnstileWidget", () => {
  it("loads the script and renders one widget", async () => {
    const onToken = vi.fn();
    const turnstile = turnstileApi();

    render(<TurnstileWidget onToken={onToken} resetKey={0} />);
    const script = injectedScript();
    window.turnstile = turnstile.api;
    await loadScript(script);

    expect(turnstile.api.render).toHaveBeenCalledTimes(1);
    expect(turnstile.options().retry).toBe("auto");
    expect(turnstile.options()["retry-interval"]).toBe(8_000);

    act(() => turnstile.options().callback("verified-token"));

    expect(onToken).toHaveBeenLastCalledWith("verified-token");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("replaces a failed script and retries once", async () => {
    vi.useFakeTimers();
    const onToken = vi.fn();
    const turnstile = turnstileApi();

    render(<TurnstileWidget onToken={onToken} resetKey={0} />);
    const firstScript = injectedScript();
    await failScript(firstScript);

    expect(firstScript.isConnected).toBe(false);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Verification is retrying"
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });

    const secondScript = injectedScript();
    expect(secondScript).not.toBe(firstScript);

    window.turnstile = turnstile.api;
    await loadScript(secondScript);
    act(() => turnstile.options().callback("verified-token"));

    expect(turnstile.api.render).toHaveBeenCalledTimes(1);
    expect(onToken).toHaveBeenLastCalledWith("verified-token");
  });

  it("replaces a script that reaches the load timeout", async () => {
    vi.useFakeTimers();

    render(<TurnstileWidget onToken={vi.fn()} resetKey={0} />);
    const firstScript = injectedScript();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    expect(firstScript.isConnected).toBe(false);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Verification is retrying"
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });

    const replacementScript = injectedScript();
    expect(replacementScript).not.toBe(firstScript);

    const turnstile = turnstileApi();
    window.turnstile = turnstile.api;
    await loadScript(replacementScript);
  });

  it("offers a manual retry after both script attempts fail", async () => {
    vi.useFakeTimers();
    const onToken = vi.fn();

    render(<TurnstileWidget onToken={onToken} resetKey={0} />);
    await failScript(injectedScript());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    await failScript(injectedScript());

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Verification could not load. Check your connection and retry."
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Retry verification" })
    );

    const retryScript = injectedScript();
    const turnstile = turnstileApi();
    window.turnstile = turnstile.api;
    await loadScript(retryScript);
    act(() => turnstile.options().callback("verified-token"));

    expect(onToken).toHaveBeenLastCalledWith("verified-token");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps widget retry enabled and clears its error after success", async () => {
    const onToken = vi.fn();
    const turnstile = turnstileApi();

    render(<TurnstileWidget onToken={onToken} resetKey={0} />);
    window.turnstile = turnstile.api;
    await loadScript(injectedScript());

    let handled = true;
    act(() => {
      handled = turnstile.options()["error-callback"]("300030");
    });

    expect(handled).toBe(false);
    expect(onToken).toHaveBeenLastCalledWith(null);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Verification is retrying"
    );

    act(() => turnstile.options().callback("recovered-token"));

    expect(onToken).toHaveBeenLastCalledWith("recovered-token");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("does not render after it is unmounted during loading", async () => {
    const onToken = vi.fn();
    const turnstile = turnstileApi();
    const view = render(<TurnstileWidget onToken={onToken} resetKey={0} />);
    const script = injectedScript();

    view.unmount();
    window.turnstile = turnstile.api;
    await loadScript(script);

    expect(turnstile.api.render).not.toHaveBeenCalled();
    expect(onToken).not.toHaveBeenCalled();
  });

  it("preserves resetKey widget resets", async () => {
    const onToken = vi.fn();
    const turnstile = turnstileApi();
    const view = render(<TurnstileWidget onToken={onToken} resetKey={0} />);

    window.turnstile = turnstile.api;
    await loadScript(injectedScript());
    view.rerender(<TurnstileWidget onToken={onToken} resetKey={1} />);

    expect(turnstile.api.reset).toHaveBeenCalledWith("widget-1");
    expect(onToken).toHaveBeenLastCalledWith(null);
  });
});
