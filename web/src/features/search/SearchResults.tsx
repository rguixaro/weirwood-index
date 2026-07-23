import { useMemo } from "react";
import type { CSSProperties } from "react";

import { ResultPassage } from "./ResultPassage";
import {
  createHighlightMatcher,
  RESULTS_PER_PAGE
} from "./search-model";
import type { SearchResultsView } from "./usePassageSearch";

type SearchResultsProps = {
  results: SearchResultsView;
  onShowMore: () => void;
};

export function SearchResults({
  results,
  onShowMore
}: SearchResultsProps) {
  const matcher = useMemo(
    () => createHighlightMatcher(results.query),
    [results.query]
  );

  return (
    <section
      aria-live="polite"
      className="results-region mt-[clamp(3rem,7vw,5rem)]"
    >
      {results.error ? (
        <div className="search-notice is-error border border-[rgba(122,38,50,0.36)] border-l-[3px] border-l-oxblood bg-[rgba(122,38,50,0.06)] px-[1.4rem] py-5 text-oxblood-dark">
          <p className="notice-title m-0 mb-[0.2rem] font-display text-[1.2rem] leading-[1.55] font-semibold text-oxblood-dark">
            Search unavailable
          </p>
          <p className="m-0 text-[0.9rem] leading-[1.55]">
            {results.error.message}
            {results.error.retryAfter
              ? ` Try again in ${results.error.retryAfter} seconds.`
              : ""}
          </p>
        </div>
      ) : null}

      {results.responseReceived ? (
        <div className="concordance">
          <header className="results-heading mb-4 flex flex-wrap items-baseline justify-between gap-3">
            <h2 className="m-0 font-display text-[clamp(2rem,5vw,2.7rem)] font-medium text-ink [font-kerning:normal]">
              {results.totalResultCount} results
            </h2>
            {results.diagnostics ? (
              <p className="search-diagnostics m-0 text-[0.72rem] text-muted-ink">
                {results.diagnostics.cached
                  ? "Cached result"
                  : `${Math.round(results.diagnostics.durationMs)} ms search time`}
              </p>
            ) : null}
          </header>

          {results.bookCounts.length > 0 ? (
            <dl
              aria-label="Results by volume"
              className="volume-counts mt-0 mb-[1.6rem] flex flex-wrap gap-x-[1.4rem] gap-y-[0.65rem]"
            >
              {results.bookCounts.map((book) => (
                <div
                  key={book.book_id}
                  className="volume-count inline-flex items-baseline gap-2 border-l-2 border-rule pl-[0.7rem] shadow-[inset_4px_0_0_rgba(165,131,85,0.09)]"
                >
                  <dt className="text-[0.8rem] text-muted-ink">
                    {book.book_title}
                  </dt>
                  <dd className="m-0 font-display text-[1.05rem] font-bold text-ink [font-variant-numeric:tabular-nums]">
                    {book.result_count.toLocaleString()}
                  </dd>
                </div>
              ))}
            </dl>
          ) : null}

          <ol className="results-list relative m-0 list-none border-t border-rule p-0">
            {results.visibleResults.map((result) => (
              <li
                key={`${results.submissionId}-${result.chunk.id}`}
                className="search-result-card result-entry grid grid-cols-[4rem_minmax(0,1fr)] gap-[clamp(1rem,4vw,2rem)] border-b border-rule-soft py-[clamp(2rem,5vw,3.5rem)] max-[620px]:block"
                style={
                  {
                    "--result-position":
                      (result.rank - 1) % RESULTS_PER_PAGE
                  } as CSSProperties
                }
              >
                <div className="result-rank pt-[0.12rem] text-right font-display text-[3.15rem] leading-[0.85] font-normal text-muted-ink [font-variant-numeric:oldstyle-nums_tabular-nums] max-[620px]:mb-[0.85rem] max-[620px]:text-left max-[620px]:text-[2.4rem]">
                  <span aria-hidden="true">{result.rank}</span>
                  <span className="sr-only">Result {result.rank}</span>
                </div>
                <div className="result-content min-w-0">
                  <header className="result-heading">
                    <p className="m-0 text-[0.71rem] font-bold text-oxblood uppercase">
                      {result.chunk.book_title}
                    </p>
                    <h3 className="mt-[0.18rem] mb-0 font-display text-[clamp(1.7rem,4vw,2.2rem)] leading-[1.05] font-medium text-ink [font-kerning:normal]">
                      {result.chunk.chapter_title}
                    </h3>
                  </header>

                  <ResultPassage result={result} matcher={matcher} />

                  {results.diagnostics ? (
                    <p className="result-diagnostics mt-4 mb-0 text-right text-[0.72rem] text-muted-ink">
                      Score {result.score.toFixed(4)}
                    </p>
                  ) : null}
                </div>
              </li>
            ))}
          </ol>

          {results.showMore ? (
            <div className="show-more-row mt-[2.2rem] flex justify-center">
              <button
                type="button"
                disabled={results.showMore.disabled}
                onClick={onShowMore}
                className="show-more-button min-h-12 cursor-pointer rounded-[2px] border border-oxblood bg-transparent px-5 py-[0.65rem] font-bold text-oxblood transition-colors duration-150 hover:not-disabled:bg-oxblood hover:not-disabled:text-parchment-light disabled:cursor-wait disabled:border-[#a99a87] disabled:text-muted-ink focus-visible:outline-[3px] focus-visible:outline-offset-[3px] focus-visible:outline-oxblood"
              >
                {results.showMore.label}
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
