import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { PropsWithChildren } from "react";

import {
  searchBooks,
  type SearchResponse
} from "../../api/client";
import type { SearchValues } from "./search-model";
import { usePassageSearch } from "./usePassageSearch";

vi.mock("../../api/client", async (importOriginal) => {
  const client = await importOriginal<typeof import("../../api/client")>();
  return {
    ...client,
    searchBooks: vi.fn()
  };
});

const values: SearchValues = {
  query: "water dance",
  mode: "hybrid",
  book: "all",
  povs: []
};

const response: SearchResponse = {
  query: "water dance",
  result_count: 0,
  duration_ms: 1,
  cached: false,
  results: []
};

function wrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  });

  return function QueryWrapper({ children }: PropsWithChildren) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

describe("usePassageSearch", () => {
  beforeEach(() => {
    vi.mocked(searchBooks).mockReset();
  });

  it("claims each Turnstile token before starting its request", async () => {
    let resolveFirstSearch!: (value: SearchResponse) => void;
    const firstSearch = new Promise<SearchResponse>((resolve) => {
      resolveFirstSearch = resolve;
    });
    const searchBooksMock = vi.mocked(searchBooks);
    searchBooksMock
      .mockReturnValueOnce(firstSearch)
      .mockResolvedValueOnce(response);

    const view = renderHook(() => usePassageSearch(), {
      wrapper: wrapper()
    });

    act(() => {
      view.result.current.verification.onToken("single-use-token");
      view.result.current.submit(values);
    });

    await waitFor(() => expect(searchBooksMock).toHaveBeenCalledTimes(1));
    expect(searchBooksMock.mock.calls[0]?.[0].turnstileToken).toBe(
      "single-use-token"
    );
    expect(view.result.current.verification.resetKey).toBe(1);

    act(() => view.result.current.submit(values));

    expect(searchBooksMock).toHaveBeenCalledTimes(1);
    expect(view.result.current.verification.error).toBe(
      "Verification is not ready. Check its status and try again."
    );

    act(() => {
      view.result.current.verification.onToken("fresh-token");
      resolveFirstSearch(response);
    });
    await waitFor(() =>
      expect(view.result.current.results.responseReceived).toBe(true)
    );

    act(() => view.result.current.submit(values));
    await waitFor(() => expect(searchBooksMock).toHaveBeenCalledTimes(2));

    expect(searchBooksMock.mock.calls[1]?.[0].turnstileToken).toBe(
      "fresh-token"
    );
    await waitFor(() => expect(view.result.current.isFetching).toBe(false));
  });
});
