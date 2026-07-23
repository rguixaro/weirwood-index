import { SearchPage } from "./features/search/SearchPage";

export function App() {
  return (
    <main className="folio-backdrop relative z-[1] min-h-screen px-6 pb-10 max-[820px]:px-5 max-[820px]:pb-5 max-[620px]:p-0">
      <article className="folio relative mx-auto min-h-[calc(100vh-2.5rem)] w-full max-w-[70rem] overflow-hidden border border-[#c2a77a] px-[clamp(2rem,7vw,5.5rem)] pt-[clamp(3.5rem,7vw,6.5rem)] pb-20 shadow-[0_2rem_5rem_rgba(9,4,3,0.55),inset_0_0_0_4px_rgba(165,131,85,0.09),inset_0_0_5rem_rgba(123,80,35,0.08)] max-[820px]:min-h-[calc(100vh-1.25rem)] max-[820px]:px-8 max-[620px]:min-h-screen max-[620px]:border-0 max-[620px]:px-5 max-[620px]:pt-[2.6rem] max-[620px]:pb-16 max-[620px]:shadow-none">
        <header className="folio-header relative flex flex-col items-center text-center">
          <h1 className="m-0 flex items-baseline justify-center gap-[0.18em] font-display text-[clamp(4.25rem,8.5vw,6.8rem)] leading-none font-semibold text-ink [font-kerning:normal] max-[620px]:gap-[0.16em] max-[620px]:text-[clamp(2.8rem,13vw,3.6rem)]">
            <span className="folio-title-name block whitespace-nowrap font-display text-inherit [font-size:inherit] leading-[0.78] [font-weight:inherit]">
              Weirwood
            </span>{" "}
            <span className="folio-title-mark block whitespace-nowrap font-display text-inherit [font-size:inherit] leading-[0.78] [font-weight:inherit]">
              Index
            </span>
          </h1>
          <p className="folio-introduction mt-[1.35rem] mb-0 font-display text-[clamp(1.12rem,2vw,1.35rem)] leading-[1.45] text-muted-ink max-[620px]:mt-[1.15rem]">
            Find the passage you half remember
          </p>
        </header>

        <div
          className="folio-rule relative mt-[clamp(2.5rem,6vw,4.5rem)] mb-8 h-px bg-rule max-[620px]:mt-[2.4rem] max-[620px]:mb-6"
          aria-hidden="true"
        />

        <SearchPage />
      </article>
    </main>
  );
}
