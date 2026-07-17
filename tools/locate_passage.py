from __future__ import annotations

import argparse
import re
from pathlib import Path

from weirwood_index.corpus import parse_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locate a phrase and print stable chapter-relative word offsets."
    )
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--phrase", action="append", required=True)
    parser.add_argument("--context-words", type=int, default=35)
    args = parser.parse_args()

    corpus = parse_corpus(args.source)
    for phrase in args.phrase:
        pattern = re.compile(re.escape(" ".join(phrase.split())), re.IGNORECASE)
        matches = 0
        for chapter in corpus.chapters:
            normalized = " ".join(chapter.text.split())
            word_matches = list(re.finditer(r"\S+", normalized))
            for match in pattern.finditer(normalized):
                matching_positions = [
                    position
                    for position, word_match in enumerate(word_matches)
                    if word_match.start() <= match.start() < word_match.end()
                ]
                if not matching_positions:
                    continue
                word_start = matching_positions[0]
                word_end = next(
                    (
                        position + 1
                        for position, word_match in enumerate(word_matches[word_start:], word_start)
                        if word_match.start() < match.end() <= word_match.end()
                    ),
                    word_start + 1,
                )
                matched_words = word_end - word_start
                words = normalized.split()
                context_start = max(0, word_start - args.context_words)
                context_end = min(
                    len(words), word_start + matched_words + args.context_words
                )
                print(
                    f"{phrase!r}: {chapter.id} word={word_start} "
                    f"context={context_start}:{context_end}"
                )
                print(" ".join(words[context_start:context_end]))
                matches += 1
        if not matches:
            print(f"{phrase!r}: no matches")


if __name__ == "__main__":
    main()
