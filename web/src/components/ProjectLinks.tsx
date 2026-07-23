const bookmarkClassName =
  "flex h-[4.25rem] w-[3.2rem] items-start justify-center bg-oxblood-dark pt-[1.05rem] text-parchment-light shadow-[inset_0_0_0_1px_rgba(240,232,216,0.16),0_0.45rem_1rem_rgba(22,15,12,0.22)] transition-[height,background-color,box-shadow] duration-[350ms] ease-[cubic-bezier(0.22,1,0.36,1)] [clip-path:polygon(0_0,100%_0,100%_calc(100%-0.65rem),50%_100%,0_calc(100%-0.65rem))] hover:h-[4.85rem] hover:bg-oxblood hover:shadow-[inset_0_0_0_1px_rgba(240,232,216,0.2),0_0.65rem_1.25rem_rgba(22,15,12,0.2)] focus-visible:h-[4.85rem] focus-visible:bg-oxblood focus-visible:outline-[3px] focus-visible:outline-offset-[3px] focus-visible:outline-oxblood max-[620px]:h-[3.25rem] max-[620px]:w-[2.75rem] max-[620px]:pt-[0.75rem] max-[620px]:hover:h-[3.65rem] max-[620px]:focus-visible:h-[3.65rem]";

const tooltipClassName =
  "pointer-events-none absolute top-[calc(100%+0.55rem)] right-0 z-20 -translate-y-[0.35rem] whitespace-nowrap border border-rule-soft bg-ink px-3 py-[0.4rem] text-[0.72rem] font-semibold text-parchment-light opacity-0 shadow-[0_0.45rem_1rem_rgba(22,15,12,0.2)] transition-[opacity,transform] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:translate-y-0 group-focus-within:opacity-100 max-[620px]:hidden";

export function ProjectLinks() {
  return (
    <nav
      aria-label="Project links"
      className="absolute top-0 right-[clamp(1.25rem,4vw,3.5rem)] z-10 flex gap-[0.45rem]"
    >
      <div className="group relative">
        <a
          href="mailto:info@weirwoodindex.com"
          aria-label="Email Weirwood Index"
          className={bookmarkClassName}
        >
          <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            className="h-5 w-5 fill-none stroke-current [stroke-linecap:round] [stroke-linejoin:round] [stroke-width:1.8]"
          >
            <path d="M3.5 6.5h17v11h-17z" />
            <path d="m4.25 7.25 7.75 6 7.75-6" />
          </svg>
        </a>
        <span aria-hidden="true" className={tooltipClassName}>
          Contact
        </span>
      </div>

      <div className="group relative">
        <a
          href="https://github.com/rguixaro/weirwood-index"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="View Weirwood Index source on GitHub"
          className={bookmarkClassName}
        >
          <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            className="h-5 w-5 fill-current"
          >
            <path d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.02c-3.22.7-3.9-1.37-3.9-1.37-.52-1.34-1.28-1.7-1.28-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.28-5.27-5.68 0-1.26.45-2.28 1.18-3.09-.12-.29-.51-1.47.11-3.05 0 0 .97-.31 3.16 1.18a10.93 10.93 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.58.23 2.76.11 3.05.74.81 1.18 1.83 1.18 3.09 0 4.41-2.71 5.38-5.29 5.67.42.36.79 1.07.79 2.16v3.02c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z" />
          </svg>
        </a>
        <span aria-hidden="true" className={tooltipClassName}>
          Source
        </span>
      </div>
    </nav>
  );
}
