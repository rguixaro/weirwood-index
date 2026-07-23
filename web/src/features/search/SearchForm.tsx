import { useForm } from "@tanstack/react-form";
import { useState } from "react";

import type { SearchCatalogResponse } from "../../api/client";
import { TurnstileWidget } from "../../components/TurnstileWidget";
import { SearchFilters } from "./SearchFilters";
import {
  availablePovs,
  filterSummary,
  isValidQuery,
  MAX_QUERY_LENGTH,
  MIN_QUERY_LENGTH,
  modeOptions,
  readQueryFromUrl,
  type SearchValues,
  type VolumeOption
} from "./search-model";

type SearchFormProps = {
  catalog: SearchCatalogResponse | undefined;
  volumeOptions: ReadonlyArray<VolumeOption>;
  isFetching: boolean;
  isSearching: boolean;
  verificationError: string | null;
  turnstileResetKey: number;
  onTurnstileToken: (token: string | null) => void;
  onSubmit: (values: SearchValues) => void;
};

export function SearchForm({
  catalog,
  volumeOptions,
  isFetching,
  isSearching,
  verificationError,
  turnstileResetKey,
  onTurnstileToken,
  onSubmit
}: SearchFormProps) {
  const [initialQuery] = useState(() =>
    readQueryFromUrl(window.location.search)
  );
  const [filtersExpanded, setFiltersExpanded] = useState(false);
  const form = useForm({
    defaultValues: {
      query: initialQuery,
      mode: "hybrid" as SearchValues["mode"],
      book: "all",
      povs: [] as string[]
    } satisfies SearchValues,
    onSubmit: ({ value }) => {
      onSubmit(value);
    }
  });

  return (
    <section
      className="search-register relative border border-rule bg-[rgba(251,244,228,0.52)] p-[clamp(1.35rem,4vw,2.2rem)] shadow-[inset_0_0_0_4px_rgba(165,131,85,0.09),0_0.8rem_2rem_rgba(74,45,22,0.07)] max-[620px]:p-[1.15rem]"
      aria-label="Passage search"
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          event.stopPropagation();
          void form.handleSubmit();
        }}
        className="search-form grid gap-[1.15rem]"
      >
        <div className="search-primary grid grid-cols-[minmax(0,1fr)_auto] items-end gap-3 max-[620px]:grid-cols-1">
          <form.Field name="query">
            {(field) => (
              <label className="query-field block min-w-0">
                <span className="mb-[0.55rem] block font-display text-xl font-semibold text-ink">
                  Search the books
                </span>
                <input
                  type="search"
                  name={field.name}
                  minLength={MIN_QUERY_LENGTH}
                  maxLength={MAX_QUERY_LENGTH}
                  value={field.state.value}
                  onBlur={field.handleBlur}
                  onChange={(event) => field.handleChange(event.target.value)}
                  placeholder="Daenerys dreams of a blue flower growing from the Wall"
                  className="h-[4.1rem] w-full rounded-[2px] border border-[#8d744f] bg-[rgba(255,251,241,0.6)] px-[1.15rem] text-ink shadow-[inset_0_1px_4px_rgba(86,54,26,0.07)] outline-none transition-[border-color,box-shadow,background] duration-[160ms] placeholder:text-muted-ink focus:border-oxblood focus:bg-parchment-light focus:shadow-[inset_0_1px_4px_rgba(86,54,26,0.07),0_0_0_3px_rgba(54,83,71,0.14)] max-[620px]:h-[3.8rem]"
                />
              </label>
            )}
          </form.Field>

          <form.Subscribe selector={(state) => state.values.query}>
            {(query) => (
              <button
                type="submit"
                disabled={!isValidQuery(query) || isFetching}
                aria-busy={isSearching}
                aria-label={isSearching ? "Searching" : undefined}
                className={`search-submit inline-flex h-[4.1rem] min-w-[7.5rem] cursor-pointer items-center justify-center rounded-[2px] border border-oxblood-dark bg-oxblood px-[1.55rem] font-bold text-parchment-light shadow-[inset_0_0_0_1px_rgba(255,255,255,0.13),0_0.3rem_0.7rem_rgba(87,18,27,0.13)] transition-[background,transform,box-shadow] duration-[160ms] hover:not-disabled:-translate-y-px hover:not-disabled:bg-oxblood-dark hover:not-disabled:shadow-[0_0.45rem_0.9rem_rgba(87,18,27,0.18)] disabled:cursor-not-allowed disabled:border-[#b9a98c] disabled:bg-[#d2c5ad] disabled:text-muted-ink disabled:shadow-none max-[620px]:h-[3.6rem] max-[620px]:w-full${isSearching ? " is-searching cursor-wait! bg-oxblood! text-parchment-light!" : ""}`}
              >
                {isSearching ? (
                  <span
                    aria-hidden="true"
                    className="search-spinner size-5 animate-[search-spinner_700ms_linear_infinite] rounded-full border-2 border-[rgba(241,230,207,0.42)] border-t-parchment-light motion-reduce:animate-none"
                  />
                ) : (
                  "Search"
                )}
              </button>
            )}
          </form.Subscribe>
        </div>

        <form.Field name="mode">
          {(field) => (
            <fieldset className="search-mode -mt-[0.15rem] min-w-0 border-0 p-0">
              <legend className="sr-only">Search mode</legend>
              <div className="mode-options flex min-h-[1.6rem] flex-wrap items-center gap-x-4 gap-y-1 max-[620px]:grid max-[620px]:gap-[0.4rem]">
                {modeOptions.map((option) => {
                  const selected = field.state.value === option.value;
                  const text =
                    option.value === "hybrid"
                      ? selected
                        ? "Semantic search"
                        : "Return to semantic search"
                      : selected
                        ? "Lexical search on"
                        : "Need exact words? Try lexical search";
                  return (
                    <label
                      key={option.value}
                      className={`mode-option text-[0.86rem] ${
                        selected
                          ? "is-selected cursor-default font-semibold text-muted-ink"
                          : "cursor-pointer text-oxblood"
                      }`}
                    >
                      <input
                        type="radio"
                        name={field.name}
                        value={option.value}
                        checked={selected}
                        aria-label={option.label}
                        onBlur={field.handleBlur}
                        onChange={() => field.handleChange(option.value)}
                        className="peer sr-only"
                      />
                      <span
                        className={`mode-option-copy peer-focus-visible:outline-[3px] peer-focus-visible:outline-offset-[3px] peer-focus-visible:outline-oxblood ${
                          selected
                            ? "before:mr-[0.45rem] before:text-[0.54rem] before:text-muted-ink before:align-[0.12em] before:content-['◆']"
                            : "underline decoration-1 underline-offset-[0.22em]"
                        }`}
                      >
                        {text}
                      </span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
          )}
        </form.Field>

        <form.Field name="book">
          {(bookField) => (
            <form.Field name="povs">
              {(povsField) => {
                const selectedBook = bookField.state.value;
                const selectedPovs = povsField.state.value;
                return (
                  <SearchFilters
                    expanded={filtersExpanded}
                    summary={filterSummary(
                      selectedBook,
                      selectedPovs,
                      volumeOptions
                    )}
                    volumeOptions={volumeOptions}
                    selectedBook={selectedBook}
                    selectedPovs={selectedPovs}
                    availablePovs={availablePovs(catalog, selectedBook)}
                    onToggle={() =>
                      setFiltersExpanded((expanded) => !expanded)
                    }
                    onBookChange={(book) => {
                      bookField.handleChange(book);
                      const permitted = new Set(availablePovs(catalog, book));
                      povsField.handleChange(
                        selectedPovs.filter((pov) => permitted.has(pov))
                      );
                    }}
                    onBookBlur={bookField.handleBlur}
                    onPovsChange={povsField.handleChange}
                    onPovsBlur={povsField.handleBlur}
                  />
                );
              }}
            </form.Field>
          )}
        </form.Field>

        <div className="turnstile-slot min-h-px [&>div]:min-h-0">
          <TurnstileWidget
            onToken={onTurnstileToken}
            resetKey={turnstileResetKey}
          />
        </div>
        {verificationError ? (
          <p className="form-error -mt-[0.35rem] mb-0 text-[0.86rem] font-semibold text-oxblood-dark">
            {verificationError}
          </p>
        ) : null}
      </form>
    </section>
  );
}
