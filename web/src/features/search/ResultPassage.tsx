import type { CSSProperties, ReactNode } from "react";

import {
  highlightSegments,
  type SearchResult
} from "./search-model";

function HighlightedText({
  text,
  matcher
}: {
  text: string;
  matcher: RegExp | null;
}): ReactNode {
  return highlightSegments(text, matcher).map((segment, index) =>
    segment.highlighted ? (
      <strong
        key={`${index}-${segment.text}`}
        className="passage-match font-bold text-oxblood no-underline"
      >
        {segment.text}
      </strong>
    ) : (
      segment.text
    )
  );
}

export function ResultPassage({
  result,
  matcher
}: {
  result: SearchResult;
  matcher: RegExp | null;
}) {
  if (result.paragraphs && result.paragraphs.length > 0) {
    return (
      <div className="passage-prose mt-[1.4rem] max-w-[74ch] font-display [font-kerning:normal] [font-variant-ligatures:common-ligatures_contextual] [text-wrap:pretty] max-[620px]:mt-[1.2rem]">
        {result.paragraphs.map((paragraph, paragraphIndex) => (
          <p
            key={paragraph.id}
            className="passage-paragraph m-0 break-words whitespace-pre-line text-[1.22rem] leading-[1.78] min-[760px]:text-justify max-[620px]:text-[1.14rem] max-[620px]:leading-[1.72] [&+.passage-paragraph]:mt-[0.95rem]"
            style={
              { "--passage-position": paragraphIndex } as CSSProperties
            }
          >
            {paragraph.partial_start ? (
              <span
                className="passage-context text-muted-ink"
                aria-hidden="true"
              >
                …{" "}
              </span>
            ) : null}
            {paragraph.fragments.map((fragment, index) => (
              <span key={`${paragraph.id}-${index}`}>
                {index > 0 ? " " : null}
                <span
                  className={
                    fragment.region === "focus"
                      ? "passage-focus text-ink"
                      : "passage-context text-muted-ink"
                  }
                >
                  <HighlightedText text={fragment.text} matcher={matcher} />
                </span>
              </span>
            ))}
            {paragraph.partial_end ? (
              <span
                className="passage-context text-muted-ink"
                aria-hidden="true"
              >
                {" "}…
              </span>
            ) : null}
          </p>
        ))}
      </div>
    );
  }

  return (
    <div className="passage-prose mt-[1.4rem] max-w-[74ch] font-display [font-kerning:normal] [font-variant-ligatures:common-ligatures_contextual] [text-wrap:pretty] max-[620px]:mt-[1.2rem]">
      <p
        className="passage-paragraph m-0 break-words whitespace-pre-line text-[1.22rem] leading-[1.78] min-[760px]:text-justify max-[620px]:text-[1.14rem] max-[620px]:leading-[1.72] [&+.passage-paragraph]:mt-[0.95rem]"
        style={{ "--passage-position": 0 } as CSSProperties}
      >
        {result.context_before ? (
          <>
            <span className="passage-context text-muted-ink">
              <HighlightedText
                text={result.context_before}
                matcher={matcher}
              />
            </span>{" "}
          </>
        ) : null}
        <span className="passage-focus text-ink">
          <HighlightedText text={result.excerpt} matcher={matcher} />
        </span>
        {result.context_after ? (
          <>
            {" "}
            <span className="passage-context text-muted-ink">
              <HighlightedText text={result.context_after} matcher={matcher} />
            </span>
          </>
        ) : null}
      </p>
    </div>
  );
}
