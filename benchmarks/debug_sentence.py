#!/usr/bin/env python
"""Debug sentence segmentation for specific sentences from a dataset.

Extract a specific sentence (by number) from a gold standard file and
run phrasplit segmentation on it to debug segmentation issues.

Usage:
    python debug_sentence.py testsets/UD_English.dataset.gold 1
    python debug_sentence.py testsets/UD_English.dataset.gold 1 --context 2
"""

import argparse
import sys
from pathlib import Path

import spacy

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_sentences_from_gold(gold_file: Path) -> list[str]:
    """Read sentences from a gold standard file.

    Args:
        gold_file: Path to gold file (one sentence per line)

    Returns:
        List of sentences
    """
    with open(gold_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def debug_segmentation(
    gold_sentences: list[str],
    start_idx: int,
    model_name: str = "en_core_web_sm",
) -> None:
    """Run segmentation and show detailed debug info with comparison.

    Args:
        gold_sentences: List of expected sentences (gold standard)
        start_idx: Starting sentence number (0-based)
        model_name: spaCy model name
    """
    nlp = spacy.load(model_name)

    # Join sentences and run spaCy
    text = " ".join(gold_sentences)
    doc = nlp(text)
    spacy_sentences = [sent.text.strip() for sent in doc.sents]

    # Determine if there's a problem
    expected_count = len(gold_sentences)
    actual_count = len(spacy_sentences)

    print("=" * 80)
    print(f"{BOLD}DIAGNOSIS:{RESET}")
    print("=" * 80)

    if actual_count == expected_count:
        # Check if they match exactly
        matches = all(
            g == s for g, s in zip(gold_sentences, spacy_sentences, strict=False)
        )
        if matches:
            print(
                f"{GREEN}OK: spaCy correctly identified "
                f"{expected_count} sentences{RESET}"
            )
        else:
            print(
                f"{YELLOW}WARNING: spaCy found {actual_count} sentences "
                f"but boundaries differ{RESET}"
            )
    elif actual_count < expected_count:
        print(f"{RED}PROBLEM: spaCy merged sentences!{RESET}")
        print(f"  Expected: {expected_count} sentences")
        print(f"  Got:      {actual_count} sentences")
        print(f"  Missing:  {expected_count - actual_count} sentence breaks")
    else:
        print(f"{YELLOW}PROBLEM: spaCy over-split!{RESET}")
        print(f"  Expected: {expected_count} sentences")
        print(f"  Got:      {actual_count} sentences")
        print(f"  Extra:    {actual_count - expected_count} sentence breaks")
    print()

    # Show side-by-side comparison
    print("=" * 80)
    print(f"{BOLD}COMPARISON (expected vs actual):{RESET}")
    print("=" * 80)

    # Find where gold sentence boundaries should be in the joined text
    gold_boundaries = []
    pos = 0
    for sent in gold_sentences[:-1]:  # Don't need boundary after last sentence
        pos += len(sent)
        gold_boundaries.append(pos)
        pos += 1  # Space between sentences

    # Find where spaCy put boundaries
    spacy_boundaries = []
    for sent in doc.sents:
        if sent.end_char < len(text):
            spacy_boundaries.append(sent.end_char)

    print(f"\n{BLUE}Expected boundaries at positions:{RESET} {gold_boundaries}")
    print(f"{BLUE}Actual boundaries at positions:{RESET}   {spacy_boundaries}")

    # Show missed and extra boundaries
    missed = set(gold_boundaries) - set(spacy_boundaries)
    extra = set(spacy_boundaries) - set(gold_boundaries)

    if missed:
        print(f"\n{RED}Missed boundaries (false negatives):{RESET}")
        for pos in sorted(missed):
            before = text[max(0, pos - 30) : pos]
            after = text[pos : min(len(text), pos + 30)]
            print(f"  Position {pos}: ...{before}{RED}|MISSING BREAK|{RESET}{after}...")

    if extra:
        print(f"\n{YELLOW}Extra boundaries (false positives):{RESET}")
        for pos in sorted(extra):
            before = text[max(0, pos - 30) : pos]
            after = text[pos : min(len(text), pos + 30)]
            print(
                f"  Position {pos}: ...{before}{YELLOW}|EXTRA BREAK|{RESET}{after}..."
            )

    print()

    # Show gold sentences
    print("=" * 80)
    print(f"{BOLD}EXPECTED SENTENCES (gold standard):{RESET}")
    print("=" * 80)
    for i, sent in enumerate(gold_sentences, start_idx + 1):
        print(f"{GREEN}[{i}]{RESET} {sent}")
    print()

    # Show spaCy sentences
    print("=" * 80)
    print(f"{BOLD}ACTUAL SENTENCES (spaCy output):{RESET}")
    print("=" * 80)
    for i, sent in enumerate(spacy_sentences, 1):
        # Check if this matches any gold sentence
        if sent in gold_sentences:
            print(f"{GREEN}[{i}]{RESET} {sent}")
        else:
            print(f"{RED}[{i}]{RESET} {sent}")
    print()

    # Token analysis around problem areas
    if missed or extra:
        print("=" * 80)
        print(f"{BOLD}TOKEN ANALYSIS (around problem areas):{RESET}")
        print("=" * 80)

        problem_positions = missed | extra
        for prob_pos in sorted(problem_positions):
            print(f"\n{BOLD}Around position {prob_pos}:{RESET}")
            # Find tokens near this position
            for token in doc:
                if (
                    abs(token.idx - prob_pos) < 50
                    or abs(token.idx + len(token) - prob_pos) < 50
                ):
                    marker = ""
                    if token.is_sent_start:
                        marker = f" {GREEN}<-- SENT_START{RESET}"
                    elif token.text in ".!?:":
                        marker = f" {BLUE}<-- punctuation{RESET}"

                    print(
                        f"  Token {token.i:3d}: {token.text!r:15s} "
                        f"pos={token.idx:4d}-{token.idx + len(token):<4d}{marker}"
                    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "gold_file",
        type=Path,
        help="Gold standard file (one sentence per line)",
    )
    parser.add_argument(
        "sentence_num",
        type=int,
        help="Sentence number (1-based)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Number of surrounding sentences to include (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        global RED, GREEN, YELLOW, BLUE, BOLD, RESET
        RED = GREEN = YELLOW = BLUE = BOLD = RESET = ""

    if not args.gold_file.exists():
        print(f"Error: File not found: {args.gold_file}", file=sys.stderr)
        sys.exit(1)

    sentences = get_sentences_from_gold(args.gold_file)

    if args.sentence_num < 1 or args.sentence_num > len(sentences):
        print(
            f"Error: Sentence number must be between 1 and {len(sentences)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get sentence(s) with optional context
    idx = args.sentence_num - 1  # Convert to 0-based
    start_idx = max(0, idx - args.context)
    end_idx = min(len(sentences), idx + args.context + 1)

    selected = sentences[start_idx:end_idx]

    print(f"Gold file: {args.gold_file}")
    print(f"Sentence number: {args.sentence_num} (of {len(sentences)})")
    if args.context > 0:
        print(f"Context: {args.context} sentences before/after")
        print(f"Showing sentences {start_idx + 1} to {end_idx}")
    print()

    debug_segmentation(selected, start_idx, args.model)


if __name__ == "__main__":
    main()
