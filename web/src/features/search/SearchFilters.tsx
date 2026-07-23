import {
  formatPovLabel,
  type VolumeOption
} from "./search-model";

type SearchFiltersProps = {
  expanded: boolean;
  summary: string;
  volumeOptions: ReadonlyArray<VolumeOption>;
  selectedBook: string;
  selectedPovs: string[];
  availablePovs: string[];
  onToggle: () => void;
  onBookChange: (book: string) => void;
  onBookBlur: () => void;
  onPovsChange: (povs: string[]) => void;
  onPovsBlur: () => void;
};

export function SearchFilters({
  expanded,
  summary,
  volumeOptions,
  selectedBook,
  selectedPovs,
  availablePovs,
  onToggle,
  onBookChange,
  onBookBlur,
  onPovsChange,
  onPovsBlur
}: SearchFiltersProps) {
  return (
    <div
      className="filter-disclosure border-y border-rule-soft"
      data-expanded={expanded}
    >
      <button
        id="filter-disclosure-trigger"
        type="button"
        className="filter-summary grid min-h-[3.3rem] w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)_1rem] items-center gap-3 border-0 bg-transparent px-[0.15rem] py-[0.7rem] text-left text-ink focus-visible:outline-[3px] focus-visible:outline-offset-[3px] focus-visible:outline-oxblood max-[620px]:grid-cols-[minmax(0,1fr)_1rem] max-[620px]:gap-x-3 max-[620px]:gap-y-1"
        aria-expanded={expanded}
        aria-controls="filter-disclosure-panel"
        onClick={onToggle}
      >
        <span className="filter-summary-title font-display text-xl font-semibold text-ink">
          Books &amp; POVs
        </span>
        <span className="filter-summary-value min-w-0 overflow-hidden text-right text-[0.86rem] text-ellipsis whitespace-nowrap text-muted-ink max-[620px]:col-start-1 max-[620px]:row-start-2 max-[620px]:text-left">
          {summary}
        </span>
        <span
          className="filter-summary-mark relative size-[0.8rem] max-[620px]:col-start-2 max-[620px]:row-span-2 max-[620px]:row-start-1"
          aria-hidden="true"
        />
      </button>

      <div
        className="filter-collapse"
        aria-hidden={!expanded}
        inert={!expanded}
      >
        <div className="filter-collapse-inner min-h-0 overflow-hidden">
          <div
            id="filter-disclosure-panel"
            className="filter-panel grid grid-cols-[minmax(0,1.25fr)_minmax(15rem,0.75fr)] gap-8 border-t border-dotted border-rule-soft px-[0.15rem] pt-[1.4rem] pb-[1.7rem] max-[820px]:grid-cols-1 max-[620px]:gap-6"
            role="region"
            aria-labelledby="filter-disclosure-trigger"
          >
            <fieldset className="filter-section volume-filter min-w-0 border-0 p-0 [&_legend]:mb-3 [&_legend]:text-[0.875rem] [&_legend]:font-bold [&_legend]:text-ink">
              <legend>Volume</legend>
              <div className="volume-grid grid grid-cols-2 gap-2 max-[620px]:grid-cols-1">
                {volumeOptions.map((option) => (
                  <label
                    key={option.value}
                    className={`filter-option group/volume min-w-0 ${
                      option.indexed
                        ? "cursor-pointer"
                        : "is-disabled cursor-not-allowed opacity-[0.48]"
                    }`}
                  >
                    <input
                      type="radio"
                      name="book"
                      value={option.value}
                      checked={selectedBook === option.value}
                      disabled={!option.indexed}
                      aria-label={option.label}
                      onBlur={onBookBlur}
                      onChange={() => {
                        if (option.indexed) onBookChange(option.value);
                      }}
                      className="peer sr-only"
                    />
                    <span
                      className={`filter-option-surface flex min-h-[2.8rem] items-center justify-center rounded-[2px] border border-[#b9a27d] bg-[rgba(255,251,241,0.45)] px-3 py-[0.55rem] text-center text-[0.82rem] leading-[1.25] font-semibold transition-colors duration-150 peer-checked:border-oxblood-dark peer-checked:bg-oxblood peer-checked:text-parchment-light peer-focus-visible:outline-[3px] peer-focus-visible:outline-offset-2 peer-focus-visible:outline-oxblood ${
                        option.indexed
                          ? "text-ink group-hover/volume:border-oxblood"
                          : "text-muted-ink"
                      }`}
                    >
                      {option.label}
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>

            <fieldset className="filter-section pov-filter min-w-0 border-0 p-0 [&_legend]:mb-3 [&_legend]:text-[0.875rem] [&_legend]:font-bold [&_legend]:text-ink">
              <legend>Point of view</legend>
              <div className="pov-options flex flex-wrap gap-2">
                <button
                  type="button"
                  aria-pressed={selectedPovs.length === 0}
                  disabled={availablePovs.length === 0}
                  onClick={() => onPovsChange([])}
                  className={`pov-option inline-flex min-h-[2.35rem] cursor-pointer items-center rounded-full border px-3 py-[0.4rem] text-[0.82rem] font-semibold transition-colors duration-150 focus-visible:outline-[3px] focus-visible:outline-offset-2 focus-visible:outline-oxblood disabled:cursor-not-allowed disabled:border-[rgba(109,94,85,0.2)] disabled:text-muted-ink ${
                    selectedPovs.length === 0 && availablePovs.length > 0
                      ? "is-selected border-oxblood-dark bg-oxblood text-parchment-light"
                      : "border-[#aa9472] bg-transparent text-muted-ink hover:not-disabled:border-oxblood hover:not-disabled:text-oxblood"
                  }`}
                >
                  All POVs
                </button>
                {availablePovs.map((pov) => (
                  <label
                    key={pov}
                    className="pov-option-label group/pov cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      name="povs"
                      value={pov}
                      checked={selectedPovs.includes(pov)}
                      onBlur={onPovsBlur}
                      onChange={(event) =>
                        onPovsChange(
                          event.target.checked
                            ? [...selectedPovs, pov]
                            : selectedPovs.filter((selected) => selected !== pov)
                        )
                      }
                      className="peer sr-only"
                    />
                    <span className="pov-option inline-flex min-h-[2.35rem] items-center rounded-full border border-[#aa9472] bg-transparent px-3 py-[0.4rem] text-[0.82rem] font-semibold text-muted-ink transition-colors duration-150 group-hover/pov:border-oxblood group-hover/pov:text-oxblood peer-checked:border-oxblood-dark peer-checked:bg-oxblood peer-checked:text-parchment-light peer-focus-visible:outline-[3px] peer-focus-visible:outline-offset-2 peer-focus-visible:outline-oxblood">
                      {formatPovLabel(pov)}
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
          </div>
        </div>
      </div>
    </div>
  );
}
