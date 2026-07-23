import { useInfiniteQuery } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import {
  ApiError,
  searchBooks,
  type SearchResponse
} from "../../api/client";
import {
  buildSearchRequest,
  isValidQuery,
  RESULTS_PER_PAGE,
  type SearchResult,
  type SearchValues
} from "./search-model";

type Submission = {
  id: number;
  request: ReturnType<typeof buildSearchRequest>;
};

type SearchErrorView = {
  message: string;
  retryAfter: number | null;
};

type SearchDiagnosticsView = {
  cached: boolean;
  durationMs: number;
};

export type SearchResultsView = {
  submissionId: number;
  responseReceived: boolean;
  error: SearchErrorView | null;
  query: string;
  totalResultCount: number;
  bookCounts: NonNullable<SearchResponse["book_counts"]>;
  visibleResults: SearchResult[];
  diagnostics: SearchDiagnosticsView | null;
  showMore: {
    label: string;
    disabled: boolean;
  } | null;
};

export type SearchVerification = {
  error: string | null;
  resetKey: number;
  onToken: (token: string | null) => void;
};

const showSearchDiagnostics =
  import.meta.env.VITE_SHOW_SEARCH_DIAGNOSTICS === "true";

export function usePassageSearch(): {
  verification: SearchVerification;
  results: SearchResultsView;
  isFetching: boolean;
  isSearching: boolean;
  submit: (values: SearchValues) => void;
  showMore: () => void;
} {
  const [turnstileToken, setTurnstileToken] = useState<string | null>(null);
  const [verificationError, setVerificationError] = useState<string | null>(null);
  const [submission, setSubmission] = useState<Submission | null>(null);
  const [turnstileResetKey, setTurnstileResetKey] = useState(0);
  const [visibleResultCount, setVisibleResultCount] = useState(RESULTS_PER_PAGE);
  const sequence = useRef(0);
  const turnstileTokenRef = useRef<string | null>(null);

  const onTurnstileToken = useCallback((token: string | null) => {
    turnstileTokenRef.current = token;
    setTurnstileToken(token);
    if (token) {
      setVerificationError(null);
    }
  }, []);

  const claimTurnstileToken = useCallback(() => {
    const token = turnstileTokenRef.current;
    if (!token) return null;

    turnstileTokenRef.current = null;
    setTurnstileToken(null);
    setTurnstileResetKey((value) => value + 1);
    return token;
  }, []);

  const search = useInfiniteQuery({
    queryKey: submission
      ? [
          "search",
          submission.id,
          submission.request.query,
          submission.request.mode,
          submission.request.book ?? null,
          submission.request.pov ?? null,
          submission.request.povs?.join(",") ?? null,
          submission.request.top ?? null,
          submission.request.page_size ?? null
        ]
      : ["search", "idle"],
    queryFn: ({ signal, pageParam }) => {
      if (!submission) {
        throw new Error("Search submission is missing.");
      }
      const token = claimTurnstileToken();
      if (!token) {
        throw new Error("Bot verification is not ready.");
      }
      const request =
        submission.request.mode === "lexical"
          ? { ...submission.request, page: pageParam }
          : submission.request;
      return searchBooks({ ...request, turnstileToken: token }, signal);
    },
    enabled: submission !== null,
    retry: false,
    staleTime: Infinity,
    initialPageParam: 1,
    getNextPageParam: (lastPage) =>
      lastPage.pagination?.has_next
        ? lastPage.pagination.page + 1
        : undefined
  });

  const submit = useCallback(
    (values: SearchValues) => {
      if (!isValidQuery(values.query)) return;
      if (!turnstileTokenRef.current) {
        setVerificationError(
          "Verification is not ready. Check its status and try again."
        );
        return;
      }

      const request = buildSearchRequest(values);
      const url = new URL(window.location.href);
      url.searchParams.set("q", request.query);
      window.history.replaceState({}, "", url);

      sequence.current += 1;
      setVisibleResultCount(RESULTS_PER_PAGE);
      setSubmission({
        id: sequence.current,
        request
      });
    },
    []
  );

  const apiError = search.error instanceof ApiError ? search.error : null;
  const searchPages = search.data?.pages ?? [];
  const loadedPageCount = searchPages.length;
  const firstSearchPage = searchPages[0];
  const latestSearchPage = searchPages.at(-1);
  const loadedResults = searchPages.flatMap((page) => page.results);
  const visibleResults = loadedResults.slice(0, visibleResultCount);
  const loadedRemaining = Math.max(
    0,
    loadedResults.length - visibleResults.length
  );
  const totalResultCount =
    firstSearchPage?.pagination?.total_results ??
    firstSearchPage?.result_count ??
    0;
  const remainingResultCount = Math.max(
    0,
    totalResultCount - visibleResults.length
  );
  const nextResultCount = Math.min(RESULTS_PER_PAGE, remainingResultCount);
  const requiresNextPageToken =
    loadedRemaining === 0 && Boolean(search.hasNextPage);
  const fetchNextPage = search.fetchNextPage;
  const hasNextPage = search.hasNextPage;
  const isFetchingNextPage = search.isFetchingNextPage;

  const showMore = useCallback(() => {
    if (remainingResultCount === 0 || isFetchingNextPage) return;
    if (loadedRemaining > 0) {
      setVisibleResultCount((count) => count + RESULTS_PER_PAGE);
      return;
    }
    if (!hasNextPage || !turnstileTokenRef.current) return;

    void fetchNextPage().then((result) => {
      if ((result.data?.pages.length ?? 0) > loadedPageCount) {
        setVisibleResultCount((count) => count + RESULTS_PER_PAGE);
      }
    });
  }, [
    loadedRemaining,
    loadedPageCount,
    remainingResultCount,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage
  ]);

  return {
    verification: {
      error: verificationError,
      resetKey: turnstileResetKey,
      onToken: onTurnstileToken
    },
    results: {
      submissionId: submission?.id ?? 0,
      responseReceived: Boolean(firstSearchPage),
      error: search.isError
        ? {
            message: apiError?.message ?? "An unexpected error occurred.",
            retryAfter: apiError?.retryAfter ?? null
          }
        : null,
      query: firstSearchPage?.query ?? "",
      totalResultCount,
      bookCounts: firstSearchPage?.book_counts ?? [],
      visibleResults,
      diagnostics:
        showSearchDiagnostics && latestSearchPage
          ? {
              cached: latestSearchPage.cached,
              durationMs: latestSearchPage.duration_ms
            }
          : null,
      showMore:
        remainingResultCount > 0
          ? {
              label: search.isFetchingNextPage
                ? "Loading more…"
                : requiresNextPageToken && !turnstileToken
                  ? "Preparing more…"
                  : `Show ${nextResultCount} more ${
                      nextResultCount === 1 ? "result" : "results"
                    }`,
              disabled:
                search.isFetchingNextPage ||
                (requiresNextPageToken && !turnstileToken)
            }
          : null
    },
    isFetching: search.isFetching,
    isSearching: search.isFetching && !search.isFetchingNextPage,
    submit,
    showMore
  };
}
