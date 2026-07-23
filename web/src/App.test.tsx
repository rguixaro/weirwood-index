import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { App } from "./App";

const catalogPayload = {
  books: [
    {
      book_id: "agot",
      book_title: "A Game of Thrones",
      book_sequence: 1,
      povs: ["ARYA", "EDDARD", "PROLOGUE", "TYRION"]
    },
    {
      book_id: "acok",
      book_title: "A Clash of Kings",
      book_sequence: 2,
      povs: ["ARYA", "DAENERYS", "DAVOS", "PROLOGUE", "TYRION"]
    }
  ]
};

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

function mockCatalog() {
  return vi.spyOn(window, "fetch").mockResolvedValue(jsonResponse(catalogPayload));
}

function result(rank: number) {
  return {
    rank,
    score: 0.81 - rank / 100,
    chunk: {
      id: `acok-048-daenerys-4-c${String(rank).padStart(3, "0")}`,
      chapter_id: "acok-048-daenerys-4",
      chapter_title: rank === 1 ? "DAENERYS IV" : `CHAPTER ${rank}`,
      chapter_sequence: 48,
      pov: "DAENERYS",
      pov_ordinal: 4,
      chunk_ordinal: rank,
      word_start: (rank - 1) * 135,
      word_end: (rank - 1) * 135 + 180,
      book_id: "acok",
      book_title: "A Clash of Kings",
      book_sequence: 2
    },
    context_before: rank === 1 ? "She remembered the dream." : null,
    excerpt:
      rank === 1
        ? "A blue flower grew from a chink in a wall of ice."
        : `Passage number ${rank}.`,
    context_after: rank === 1 ? "Ice rose around them." : null,
    retrieval: { mode: "hierarchical-hybrid" }
  };
}

function mockSearch(resultCount: number) {
  return vi.spyOn(window, "fetch").mockImplementation(async (input) => {
    if (String(input) === "/api/catalog") return jsonResponse(catalogPayload);
    return jsonResponse({
        query: "blue flower at the Wall",
        result_count: resultCount,
        duration_ms: 82,
        cached: false,
        book_counts: [
          {
            book_id: "acok",
            book_title: "A Clash of Kings",
            result_count: resultCount
          }
        ],
        results: Array.from({ length: resultCount }, (_, index) => result(index + 1))
      });
  });
}

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  );
}

function openFilters(): HTMLElement {
  const trigger = screen.getByRole("button", { name: /Books & POVs/ });
  const panel = document.getElementById("filter-disclosure-panel");
  const collapse = panel?.closest(".filter-collapse");
  if (!panel || !collapse) throw new Error("Filter disclosure panel is missing.");

  expect(trigger).toHaveAttribute("aria-expanded", "false");
  expect(collapse).toHaveAttribute("aria-hidden", "true");
  expect(collapse).toHaveAttribute("inert");
  fireEvent.click(trigger);
  expect(trigger).toHaveAttribute("aria-expanded", "true");
  expect(collapse).toHaveAttribute("aria-hidden", "false");
  expect(collapse).not.toHaveAttribute("inert");
  return panel;
}

function installSingleUseTurnstile() {
  vi.stubEnv("VITE_TURNSTILE_SITE_KEY", "test-site-key");
  window.turnstile = {
    render: vi.fn((_element, options) => {
      options.callback("initial-token");
      return "test-widget";
    }),
    reset: vi.fn(),
    remove: vi.fn()
  };
}

async function enterQueryAndSubmit() {
  fireEvent.change(
    screen.getByPlaceholderText(
      "Daenerys dreams of a blue flower growing from the Wall"
    ),
    { target: { value: "blue flower at the Wall" } }
  );
  const submit = screen.getByRole("button", { name: "Search" });
  await waitFor(() => expect(submit).toBeEnabled());
  fireEvent.click(submit);
}

describe("App", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    delete window.turnstile;
  });

  it("keeps search disabled until the query has three characters", () => {
    mockCatalog();
    renderApp();

    const input = screen.getByRole("searchbox", { name: "Search the books" });
    const submit = screen.getByRole("button", { name: "Search" });
    expect(submit).toBeDisabled();

    fireEvent.change(input, { target: { value: "ty" } });
    expect(submit).toBeDisabled();

    fireEvent.change(input, { target: { value: "tyr" } });
    expect(submit).toBeEnabled();
  });

  it("opens and closes the filter disclosure accessibly", () => {
    mockCatalog();
    renderApp();

    const trigger = screen.getByRole("button", { name: /Books & POVs/ });
    expect(trigger.tagName).toBe("BUTTON");

    const panel = openFilters();
    expect(trigger).toHaveAttribute("aria-controls", panel.id);

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(panel.closest(".filter-collapse")).toHaveAttribute("inert");
  });

  it("shows unindexed volumes as disabled options", async () => {
    mockCatalog();
    renderApp();
    expect(screen.getByText("All volumes · All POVs")).toBeInTheDocument();
    openFilters();

    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: "A Game of Thrones" })
      ).toBeEnabled()
    );

    for (const title of [
      "A Storm of Swords",
      "A Feast for Crows",
      "A Dance with Dragons",
      "The Hedge Knight",
      "The Sworn Sword",
      "The Mystery Knight",
      "Fire & Blood"
    ]) {
      expect(
        screen.getByRole("radio", { name: title })
      ).toBeDisabled();
    }
    expect(screen.queryByText("Not yet indexed")).not.toBeInTheDocument();
    expect(screen.queryByText("Soon")).not.toBeInTheDocument();
  });

  it("submits ten results with the selected controls and renders highlighted context", async () => {
    const fetchMock = mockSearch(1);
    const { container } = renderApp();
    expect(screen.getByRole("radio", { name: "Semantic" })).toBeChecked();
    expect(screen.getByText("Semantic search")).toBeInTheDocument();
    expect(
      screen.getByText("Need exact words? Try lexical search")
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Lexical" }));
    expect(screen.getByText("Lexical search on")).toBeInTheDocument();
    expect(screen.getByText("Return to semantic search")).toBeInTheDocument();
    openFilters();
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: "A Clash of Kings" })
      ).toBeEnabled()
    );
    fireEvent.click(
      screen.getByRole("radio", { name: "A Clash of Kings" })
    );
    expect(screen.queryByRole("checkbox", { name: "Eddard" })).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Davos" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Daenerys" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Tyrion" }));
    expect(
      screen.getByText("A Clash of Kings · Daenerys +1")
    ).toBeInTheDocument();

    await enterQueryAndSubmit();

    expect(await screen.findByText("DAENERYS IV")).toBeInTheDocument();
    expect(
      screen.getByRole("searchbox", { name: "Search the books" })
    ).toHaveValue("blue flower at the Wall");
    const contextBefore = screen.getByText("She remembered the dream.");
    const contextAfter = screen.getByText("Ice rose around them.");
    expect(contextBefore).toHaveClass("passage-context");
    expect(contextAfter).toHaveClass("passage-context");
    expect(contextBefore.parentElement).toBe(contextAfter.parentElement);
    expect(container.querySelector(".passage-prose")).toBeInTheDocument();
    expect(screen.queryByText("POV DAENERYS")).not.toBeInTheDocument();
    expect(screen.queryByText(/Score 0\./)).not.toBeInTheDocument();
    expect(screen.queryByText(/ms search time/)).not.toBeInTheDocument();
    expect(screen.getByText("DAENERYS IV").closest("li")).toHaveClass(
      "search-result-card"
    );
    expect(window.location.search).toContain("q=blue+flower+at+the+Wall");

    const highlightedTerms = Array.from(
      container.querySelectorAll("li strong")
    ).map((element) => element.textContent);
    expect(highlightedTerms).toEqual(["blue", "flower", "wall"]);

    const searchCall = fetchMock.mock.calls.find(
      ([input]) => String(input) === "/api/search"
    );
    const requestBody = JSON.parse(String(searchCall?.[1]?.body)) as {
      mode: string;
      page: number;
      page_size: number;
      book: string;
      povs: string[];
      top?: number;
    };
    expect(requestBody).toMatchObject({
      mode: "lexical",
      page: 1,
      page_size: 10,
      book: "acok",
      povs: ["DAENERYS", "TYRION"]
    });
    expect(requestBody.top).toBeUndefined();
  });

  it("shows five results initially and reveals the next five on request", async () => {
    mockSearch(10);
    renderApp();

    await enterQueryAndSubmit();

    expect(await screen.findByText("CHAPTER 5")).toBeInTheDocument();
    expect(screen.queryByText("CHAPTER 6")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Show 5 more results" })
    );

    expect(await screen.findByText("CHAPTER 10")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Show .* more/ })
    ).not.toBeInTheDocument();
  });

  it("reveals loaded results without a token and waits before fetching another page", async () => {
    installSingleUseTurnstile();
    const fetchMock = vi.spyOn(window, "fetch").mockImplementation(async (input) => {
      if (String(input) === "/api/catalog") return jsonResponse(catalogPayload);
      return jsonResponse({
        query: "blue flower at the Wall",
        result_count: 10,
        duration_ms: 12,
        cached: false,
        pagination: {
          page: 1,
          page_size: 10,
          total_results: 12,
          total_pages: 2,
          has_next: true
        },
        results: Array.from({ length: 10 }, (_, index) => result(index + 1))
      });
    });
    renderApp();
    fireEvent.click(screen.getByRole("radio", { name: "Lexical" }));

    await waitFor(() =>
      expect(window.turnstile?.render).toHaveBeenCalled()
    );
    await enterQueryAndSubmit();
    expect(await screen.findByText("CHAPTER 5")).toBeInTheDocument();
    await waitFor(() =>
      expect(window.turnstile?.reset).toHaveBeenCalled()
    );

    const localResultsButton = screen.getByRole("button", {
      name: "Show 5 more results"
    });
    expect(localResultsButton).toBeEnabled();
    fireEvent.click(localResultsButton);
    expect(await screen.findByText("CHAPTER 10")).toBeInTheDocument();

    expect(
      screen.getByRole("button", { name: "Preparing more…" })
    ).toBeDisabled();
    expect(
      fetchMock.mock.calls.filter(([input]) => String(input) === "/api/search")
    ).toHaveLength(1);
  });

  it("shows a spinner in the search button while the request is running", async () => {
    let resolveSearch!: (response: Response) => void;
    vi.spyOn(window, "fetch").mockImplementation(
      (input) =>
        String(input) === "/api/catalog"
          ? Promise.resolve(jsonResponse(catalogPayload))
          :
        new Promise<Response>((resolve) => {
          resolveSearch = resolve;
        })
    );
    renderApp();

    await enterQueryAndSubmit();

    const searchingButton = await screen.findByRole("button", {
      name: "Searching"
    });
    expect(searchingButton).toBeDisabled();
    expect(searchingButton).toHaveAttribute("aria-busy", "true");
    expect(
      searchingButton.querySelector('[aria-hidden="true"]')
    ).toBeInTheDocument();
    expect(screen.queryByText("Searching…")).not.toBeInTheDocument();

    resolveSearch(
      new Response(
        JSON.stringify({
          query: "blue flower at the Wall",
          result_count: 0,
          duration_ms: 12,
          cached: false,
          results: []
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Search" })).toBeEnabled()
    );
  });

  it("renders preserved EPUB paragraphs as separate blocks", async () => {
    const structured = {
      ...result(1),
      paragraphs: [
        {
          id: "acok-048-daenerys-4-p0041",
          ordinal: 41,
          partial_start: true,
          partial_end: false,
          fragments: [
            { region: "before", text: "She remembered the dream." },
            { region: "focus", text: "A blue flower grew." }
          ]
        },
        {
          id: "acok-048-daenerys-4-p0042",
          ordinal: 42,
          partial_start: false,
          partial_end: true,
          fragments: [{ region: "after", text: "Ice rose around them." }]
        }
      ]
    };
    vi.spyOn(window, "fetch").mockImplementation(async (input) => {
      if (String(input) === "/api/catalog") return jsonResponse(catalogPayload);
      return jsonResponse({
          query: "blue flower at the Wall",
          result_count: 1,
          duration_ms: 12,
          cached: false,
          results: [structured]
        });
    });
    const { container } = renderApp();

    await enterQueryAndSubmit();
    expect(await screen.findByText("DAENERYS IV")).toBeInTheDocument();

    const paragraphs = container.querySelectorAll(".passage-paragraph");
    expect(paragraphs).toHaveLength(2);
    expect(
      (paragraphs[1] as HTMLElement).style.getPropertyValue(
        "--passage-position"
      )
    ).toBe("1");
    const focus = container.querySelector(".passage-focus");
    expect(focus).toHaveTextContent("A blue flower grew.");
    expect(screen.getByText("Ice rose around them.")).toHaveClass(
      "passage-context"
    );
  });

  it("fetches subsequent lexical pages when loaded results are exhausted", async () => {
    const fetchMock = vi.spyOn(window, "fetch").mockImplementation(async (input, init) => {
      if (String(input) === "/api/catalog") return jsonResponse(catalogPayload);
      const request = JSON.parse(String(init?.body)) as { page: number };
      const start = request.page === 1 ? 1 : 11;
      const end = request.page === 1 ? 10 : 12;
      return jsonResponse({
          query: "blue flower at the Wall",
          result_count: end - start + 1,
          duration_ms: 12,
          cached: false,
          book_counts: [
            {
              book_id: "agot",
              book_title: "A Game of Thrones",
              result_count: 7
            },
            {
              book_id: "acok",
              book_title: "A Clash of Kings",
              result_count: 5
            }
          ],
          pagination: {
            page: request.page,
            page_size: 10,
            total_results: 12,
            total_pages: 2,
            has_next: request.page === 1
          },
          results: Array.from(
            { length: end - start + 1 },
            (_, index) => result(start + index)
          )
        });
    });
    renderApp();
    fireEvent.click(screen.getByRole("radio", { name: "Lexical" }));

    await enterQueryAndSubmit();
    expect(await screen.findByText("12 results")).toBeInTheDocument();
    const bookSummary = screen.getByLabelText("Results by volume");
    expect(within(bookSummary).getByText("A Game of Thrones")).toBeInTheDocument();
    expect(within(bookSummary).getByText("7")).toBeInTheDocument();
    expect(within(bookSummary).getByText("A Clash of Kings")).toBeInTheDocument();
    expect(within(bookSummary).getByText("5")).toBeInTheDocument();
    expect(await screen.findByText("CHAPTER 5")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show 5 more results" }));
    expect(await screen.findByText("CHAPTER 10")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show 2 more results" }));
    expect(await screen.findByText("CHAPTER 12")).toBeInTheDocument();
    const searchCalls = fetchMock.mock.calls.filter(
      ([input]) => String(input) === "/api/search"
    );
    expect(searchCalls).toHaveLength(2);
    expect(
      JSON.parse(String(searchCalls[1]?.[1]?.body))
    ).toMatchObject({
      mode: "lexical",
      page: 2,
      page_size: 10
    });
  });

  it("reveals only five new results after a failed page fetch is retried", async () => {
    let pageTwoAttempts = 0;
    const fetchMock = vi.spyOn(window, "fetch").mockImplementation(
      async (input, init) => {
        if (String(input) === "/api/catalog") {
          return jsonResponse(catalogPayload);
        }

        const request = JSON.parse(String(init?.body)) as { page: number };
        if (request.page === 2) {
          pageTwoAttempts += 1;
          if (pageTwoAttempts === 1) {
            return new Response(
              JSON.stringify({ message: "Search temporarily unavailable." }),
              {
                status: 503,
                headers: { "Content-Type": "application/json" }
              }
            );
          }
        }

        const start = request.page === 1 ? 1 : 11;
        const end = request.page === 1 ? 10 : 20;
        return jsonResponse({
          query: "blue flower at the Wall",
          result_count: end - start + 1,
          duration_ms: 12,
          cached: false,
          pagination: {
            page: request.page,
            page_size: 10,
            total_results: 20,
            total_pages: 2,
            has_next: request.page === 1
          },
          results: Array.from(
            { length: end - start + 1 },
            (_, index) => result(start + index)
          )
        });
      }
    );
    renderApp();
    fireEvent.click(screen.getByRole("radio", { name: "Lexical" }));

    await enterQueryAndSubmit();
    expect(await screen.findByText("CHAPTER 5")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Show 5 more results" })
    );
    expect(await screen.findByText("CHAPTER 10")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Show 5 more results" })
    );
    expect(
      await screen.findByText("Search temporarily unavailable.")
    ).toBeInTheDocument();

    fireEvent.click(
      await screen.findByRole("button", { name: "Show 5 more results" })
    );
    expect(await screen.findByText("CHAPTER 15")).toBeInTheDocument();
    expect(screen.queryByText("CHAPTER 16")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Show 5 more results" })
    ).toBeInTheDocument();

    expect(
      fetchMock.mock.calls.filter(([input]) => String(input) === "/api/search")
    ).toHaveLength(3);
  });
});
