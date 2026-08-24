#!/usr/bin/env python
"""Evaluate sentence segmentation against a gold standard.

Compares a test file against a gold standard file and calculates
precision, recall, and F-measure for sentence boundary detection.

By default, saves failed cases (false positives and false negatives)
to separate files for analysis. Use --no-errors to disable.

Usage:
    python segmenteval.py gold.txt test.txt
    python segmenteval.py gold.txt test.txt --no-errors
    python segmenteval.py gold.txt test.txt --errors-prefix custom_prefix
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalMetrics:
    """Evaluation metrics and error details."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f_measure: float
    # Contexts around errors for debugging: list of (sentence_num, context)
    false_positive_contexts: list[tuple[int, str]]
    false_negative_contexts: list[tuple[int, str]]


def get_context(text: str, pos: int, context_size: int = 50) -> str:
    """Extract context around a position in text.

    The position marks a sentence boundary (newline). We show context
    before and after the break point, skipping the newline itself.

    Args:
        text: The text to extract context from
        pos: Position in text (points to newline character)
        context_size: Number of characters before and after

    Returns:
        Context string with break point marked as |BREAK|
    """
    start = max(0, pos - context_size)
    # Skip the newline at pos to show what comes after the break
    after_start = pos + 1 if pos < len(text) and text[pos] == "\n" else pos
    end = min(len(text), after_start + context_size)

    before = text[start:pos].replace("\n", " / ")
    after = text[after_start:end].replace("\n", " / ")

    return f"...{before} |BREAK| {after}..."


def evaluate(
    gold_text: str, test_text: str, collect_errors: bool = False
) -> EvalMetrics:
    """Evaluate test segmentation against gold standard.

    Args:
        gold_text: Gold standard text (one sentence per line)
        test_text: Test output text (one sentence per line)
        collect_errors: Whether to collect error contexts

    Returns:
        EvalMetrics with scores and optional error contexts
    """
    # Remove spaces for comparison (only newlines matter for boundaries)
    g = gold_text.replace(" ", "")
    t = test_text.replace(" ", "")

    # Verify content matches (ignoring segmentation)
    g_content = g.replace("\n", "")
    t_content = t.replace("\n", "")

    if g_content != t_content:
        # Find first difference
        for i, (gc, tc) in enumerate(zip(g_content, t_content, strict=False)):
            if gc != tc:
                raise ValueError(
                    f"Content mismatch at position {i}:\n"
                    f"  Gold: {g_content[max(0, i - 20) : i + 20]!r}\n"
                    f"  Test: {t_content[max(0, i - 20) : i + 20]!r}"
                )
        if len(g_content) != len(t_content):
            raise ValueError(
                f"Content length mismatch: gold={len(g_content)}, test={len(t_content)}"
            )

    tp = 0
    fp = 0
    fn = 0
    fp_contexts: list[tuple[int, str]] = []
    fn_contexts: list[tuple[int, str]] = []

    # Build mappings from stripped positions to original positions
    # This allows us to find the correct position in the original text
    # when we detect an error in the stripped text
    g_pos_map: list[int] = []  # stripped pos -> original pos
    for i, c in enumerate(gold_text):
        if c != " ":
            g_pos_map.append(i)

    t_pos_map: list[int] = []  # stripped pos -> original pos
    for i, c in enumerate(test_text):
        if c != " ":
            t_pos_map.append(i)

    gc = 0  # position in stripped gold
    tc = 0  # position in stripped test
    gold_sent_num = 1  # current sentence number in gold (1-based)
    test_sent_num = 1  # current sentence number in test (1-based)

    while gc < len(g) and tc < len(t):
        if g[gc] == "\n":
            if t[tc] == "\n":
                tp += 1
                gc += 1
                tc += 1
                gold_sent_num += 1
                test_sent_num += 1
            else:
                fn += 1
                if collect_errors:
                    # Map stripped position to original position
                    orig_pos = g_pos_map[gc] if gc < len(g_pos_map) else len(gold_text)
                    fn_contexts.append(
                        (gold_sent_num, get_context(gold_text, orig_pos))
                    )
                gc += 1
                gold_sent_num += 1
        elif t[tc] == "\n":
            fp += 1
            if collect_errors:
                # Map stripped position to original position
                orig_pos = t_pos_map[tc] if tc < len(t_pos_map) else len(test_text)
                fp_contexts.append((test_sent_num, get_context(test_text, orig_pos)))
            tc += 1
            test_sent_num += 1
        else:
            gc += 1
            tc += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_measure = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return EvalMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f_measure=f_measure,
        false_positive_contexts=fp_contexts,
        false_negative_contexts=fn_contexts,
    )


def save_errors(
    metrics: EvalMetrics, prefix: str, segmenter: str | None = None
) -> None:
    """Save error contexts to files.

    Args:
        metrics: Evaluation metrics with error contexts
        prefix: Prefix for output files
        segmenter: Optional segmenter name to include in output
    """
    segmenter_info = f" (segmenter: {segmenter})" if segmenter else ""

    # Save false positives (extra breaks inserted by test)
    fp_file = Path(f"{prefix}_false_positives.txt")
    with open(fp_file, "w", encoding="utf-8") as f:
        f.write(f"# False Positives: {metrics.false_positives}{segmenter_info}\n")
        f.write(
            "# These are places where the segmenter "
            "added a break that shouldn't exist\n"
        )
        f.write("# Format: [error_num] (test_sentence_num) context\n")
        f.write("#" + "=" * 78 + "\n\n")
        f.writelines(
            f"[{i}] (sentence {sent_num}) {ctx}\n\n"
            for i, (sent_num, ctx) in enumerate(metrics.false_positive_contexts, 1)
        )
    print(f"Saved false positives to: {fp_file}")

    # Save false negatives (missed breaks)
    fn_file = Path(f"{prefix}_false_negatives.txt")
    with open(fn_file, "w", encoding="utf-8") as f:
        f.write(f"# False Negatives: {metrics.false_negatives}{segmenter_info}\n")
        f.write(
            "# These are places where the segmenter missed a break that should exist\n"
        )
        f.write("# Format: [error_num] (gold_sentence_num) context\n")
        f.write("#" + "=" * 78 + "\n\n")
        f.writelines(
            f"[{i}] (sentence {sent_num}) {ctx}\n\n"
            for i, (sent_num, ctx) in enumerate(metrics.false_negative_contexts, 1)
        )
    print(f"Saved false negatives to: {fn_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    parser.add_argument(
        "goldstandard",
        type=argparse.FileType("rt", encoding="utf-8"),
        help="Gold-standard file (one sentence per line)",
    )
    parser.add_argument(
        "test",
        type=argparse.FileType("rt", encoding="utf-8"),
        help="Test file to compare with gold-standard",
    )
    parser.add_argument(
        "--no-errors",
        action="store_true",
        help="Disable saving error context files (enabled by default)",
    )
    parser.add_argument(
        "--errors-prefix",
        type=str,
        metavar="PREFIX",
        help="Custom prefix for error files (default: derived from test filename)",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        metavar="NAME",
        help="Segmenter name to include in error file headers (e.g., spacy, phrasplit)",
    )

    args = parser.parse_args()

    # Read files
    gold_text = args.goldstandard.read()
    test_text = args.test.read()

    # Determine whether to collect and save errors
    save_errors_enabled = not args.no_errors

    # Determine error file prefix
    if args.errors_prefix:
        error_prefix = args.errors_prefix
    else:
        # Derive from test filename: remove extension and add _errors
        test_path = Path(args.test.name)
        error_prefix = str(test_path.parent / test_path.stem) + "_errors"

    # Evaluate
    try:
        metrics = evaluate(gold_text, test_text, collect_errors=save_errors_enabled)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print(f"True positives: {metrics.true_positives}")
    print(f"False positives: {metrics.false_positives}")
    print(f"False negatives: {metrics.false_negatives}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"F-measure: {metrics.f_measure:.3f}")

    # Save errors by default (unless --no-errors)
    if save_errors_enabled:
        save_errors(metrics, error_prefix, args.segmenter)


if __name__ == "__main__":
    main()
