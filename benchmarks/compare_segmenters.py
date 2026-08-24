#!/usr/bin/env python
"""Compare sentence segmentation between two segmenters.

Finds sentences that:
1. Segmenter A gets right but Segmenter B gets wrong (B regressions)
2. Segmenter B gets right but Segmenter A gets wrong (B improvements)

This helps identify where corrections or changes between segmenters
help or hurt sentence boundary detection.

Usage:
    python compare_segmenters.py en spacy phrasplit -m lg -v all
    python compare_segmenters.py de spacy phrasplit -m sm -v none -o result.txt
    python compare_segmenters.py --list

Examples:
    # Compare spaCy vs phrasplit on English with lg model
    python compare_segmenters.py en spacy phrasplit -m lg -v all

    # Compare on German with md model, none variant (hardest test)
    python compare_segmenters.py de spacy phrasplit -m md -v none

    # Compare sentencizer vs phrasplit
    python compare_segmenters.py en sentencizer phrasplit -m sm -v all

    # Save output to file
    python compare_segmenters.py en spacy phrasplit -m lg -v all -o comparison.txt

    # Use custom files (override auto-detection)
    python compare_segmenters.py en spacy phrasplit -m lg -v all \\
        --file-a custom_a.out --file-b custom_b.out --gold custom.gold
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Valid segmenter names
VALID_SEGMENTERS = ["spacy", "phrasplit", "sentencizer"]

# Valid model sizes
VALID_MODEL_SIZES = ["sm", "md", "lg"]

# Valid test variants
VALID_VARIANTS = ["all", "none", "mixed"]

# Language code to language name mapping
LANG_CODE_TO_NAME = {
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cnr": "Montenegrin",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "hr": "Croatian",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "mt": "Maltese",
    "nb": "Norwegian-Bokmaal",
    "nl": "Dutch",
    "nn": "Norwegian-Nynorsk",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
}


@dataclass
class BoundaryComparison:
    """Comparison of a single boundary between two segmenters."""

    gold_sentence_num: int  # Sentence number in gold (1-based)
    position: int  # Position in original gold text
    context: str  # Context around the boundary
    segmenter_a_found: bool
    segmenter_b_found: bool

    @property
    def a_only(self) -> bool:
        """True if only segmenter A found this boundary correctly."""
        return self.segmenter_a_found and not self.segmenter_b_found

    @property
    def b_only(self) -> bool:
        """True if only segmenter B found this boundary correctly."""
        return self.segmenter_b_found and not self.segmenter_a_found

    @property
    def both(self) -> bool:
        """True if both segmenters found this boundary correctly."""
        return self.segmenter_a_found and self.segmenter_b_found

    @property
    def neither(self) -> bool:
        """True if neither segmenter found this boundary."""
        return not self.segmenter_a_found and not self.segmenter_b_found


@dataclass
class ComparisonResult:
    """Full comparison results between two segmenters."""

    segmenter_a_name: str
    segmenter_b_name: str
    total_boundaries: int  # Total gold boundaries

    # Counts
    both_correct: int = 0
    neither_correct: int = 0
    a_only_correct: int = 0  # A got it, B missed it (B regression)
    b_only_correct: int = 0  # B got it, A missed it (B improvement)

    # Detailed comparisons for differences
    a_only_cases: list[BoundaryComparison] = field(default_factory=list)
    b_only_cases: list[BoundaryComparison] = field(default_factory=list)

    # False positive analysis (extra boundaries not in gold)
    a_false_positives: list[tuple[int, str]] = field(
        default_factory=list
    )  # (position, context)
    b_false_positives: list[tuple[int, str]] = field(default_factory=list)
    a_only_false_positives: list[tuple[int, str]] = field(default_factory=list)
    b_only_false_positives: list[tuple[int, str]] = field(default_factory=list)


def get_context(text: str, pos: int, context_size: int = 50) -> str:
    """Extract context around a position in text.

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


def strip_and_map(text: str) -> tuple[str, list[int]]:
    """Strip spaces from text and create position mapping.

    Args:
        text: Original text

    Returns:
        Tuple of (stripped text, mapping from stripped pos to original pos)
    """
    stripped = text.replace(" ", "")
    pos_map: list[int] = []
    for i, c in enumerate(text):
        if c != " ":
            pos_map.append(i)
    return stripped, pos_map


def get_boundary_positions(stripped_text: str) -> set[int]:
    """Get positions of all newlines (sentence boundaries) in stripped text.

    Args:
        stripped_text: Text with spaces removed

    Returns:
        Set of positions where newlines occur
    """
    return {i for i, c in enumerate(stripped_text) if c == "\n"}


def verify_content_match(gold_content: str, test_content: str, test_name: str) -> None:
    """Verify that content (ignoring newlines) matches between gold and test.

    Args:
        gold_content: Gold text with newlines removed
        test_content: Test text with newlines removed
        test_name: Name of test file for error messages

    Raises:
        ValueError: If content doesn't match
    """
    if gold_content != test_content:
        # Find first difference
        for i, (gc, tc) in enumerate(zip(gold_content, test_content, strict=False)):
            if gc != tc:
                raise ValueError(
                    f"Content mismatch in {test_name} at position {i}:\n"
                    f"  Gold: {gold_content[max(0, i - 20) : i + 20]!r}\n"
                    f"  Test: {test_content[max(0, i - 20) : i + 20]!r}"
                )
        if len(gold_content) != len(test_content):
            raise ValueError(
                f"Content length mismatch in {test_name}: "
                f"gold={len(gold_content)}, test={len(test_content)}"
            )


def compare_segmenters(
    gold_text: str,
    segmenter_a_text: str,
    segmenter_b_text: str,
    segmenter_a_name: str = "Segmenter A",
    segmenter_b_name: str = "Segmenter B",
) -> ComparisonResult:
    """Compare two segmenters against a gold standard.

    Args:
        gold_text: Gold standard text (one sentence per line)
        segmenter_a_text: Output from segmenter A
        segmenter_b_text: Output from segmenter B
        segmenter_a_name: Display name for segmenter A
        segmenter_b_name: Display name for segmenter B

    Returns:
        ComparisonResult with detailed analysis
    """
    # Strip spaces and create position mappings
    gold_stripped, gold_pos_map = strip_and_map(gold_text)
    a_stripped, a_pos_map = strip_and_map(segmenter_a_text)
    b_stripped, b_pos_map = strip_and_map(segmenter_b_text)

    # Verify content matches (ignoring boundaries)
    gold_content = gold_stripped.replace("\n", "")
    a_content = a_stripped.replace("\n", "")
    b_content = b_stripped.replace("\n", "")

    verify_content_match(gold_content, a_content, segmenter_a_name)
    verify_content_match(gold_content, b_content, segmenter_b_name)

    # Get boundary positions in stripped text
    gold_boundaries = get_boundary_positions(gold_stripped)
    a_boundaries = get_boundary_positions(a_stripped)
    b_boundaries = get_boundary_positions(b_stripped)

    result = ComparisonResult(
        segmenter_a_name=segmenter_a_name,
        segmenter_b_name=segmenter_b_name,
        total_boundaries=len(gold_boundaries),
    )

    # Build mapping: for each position in stripped text (no spaces),
    # what position is it in content (no spaces, no newlines)?
    # This allows us to compare boundaries across texts with different newline counts.
    def build_content_pos_map(stripped: str) -> list[int]:
        """Map stripped positions to content positions (excluding newlines)."""
        content_pos = 0
        pos_map = []
        for c in stripped:
            if c == "\n":
                pos_map.append(content_pos)  # newline maps to position after
            else:
                pos_map.append(content_pos)
                content_pos += 1
        return pos_map

    gold_to_content = build_content_pos_map(gold_stripped)
    a_to_content = build_content_pos_map(a_stripped)
    b_to_content = build_content_pos_map(b_stripped)

    # Convert segmenter boundaries to content positions for comparison
    a_boundary_content_pos = {a_to_content[p] for p in a_boundaries}
    b_boundary_content_pos = {b_to_content[p] for p in b_boundaries}

    # Compare each gold boundary using content positions
    gold_sent_num = 1
    for pos in sorted(gold_boundaries):
        # Convert gold position to content position for comparison
        gold_content_pos = gold_to_content[pos]
        a_found = gold_content_pos in a_boundary_content_pos
        b_found = gold_content_pos in b_boundary_content_pos

        # Get context from original gold text
        orig_pos = gold_pos_map[pos] if pos < len(gold_pos_map) else len(gold_text)
        context = get_context(gold_text, orig_pos)

        comparison = BoundaryComparison(
            gold_sentence_num=gold_sent_num,
            position=orig_pos,
            context=context,
            segmenter_a_found=a_found,
            segmenter_b_found=b_found,
        )

        if comparison.both:
            result.both_correct += 1
        elif comparison.neither:
            result.neither_correct += 1
        elif comparison.a_only:
            result.a_only_correct += 1
            result.a_only_cases.append(comparison)
        elif comparison.b_only:
            result.b_only_correct += 1
            result.b_only_cases.append(comparison)

        gold_sent_num += 1

    # Analyze false positives (boundaries in test but not in gold)
    # Convert gold boundaries to content positions for comparison
    gold_boundary_content_pos = {gold_to_content[p] for p in gold_boundaries}

    # Find false positives for each segmenter
    a_fp_content_pos = set()
    for pos in a_boundaries:
        content_pos = a_to_content[pos]
        if content_pos not in gold_boundary_content_pos:
            a_fp_content_pos.add(content_pos)
            orig_pos = a_pos_map[pos] if pos < len(a_pos_map) else len(segmenter_a_text)
            result.a_false_positives.append(
                (orig_pos, get_context(segmenter_a_text, orig_pos))
            )

    b_fp_content_pos = set()
    for pos in b_boundaries:
        content_pos = b_to_content[pos]
        if content_pos not in gold_boundary_content_pos:
            b_fp_content_pos.add(content_pos)
            orig_pos = b_pos_map[pos] if pos < len(b_pos_map) else len(segmenter_b_text)
            result.b_false_positives.append(
                (orig_pos, get_context(segmenter_b_text, orig_pos))
            )

    # Find false positives unique to each segmenter
    a_only_fp = a_fp_content_pos - b_fp_content_pos
    b_only_fp = b_fp_content_pos - a_fp_content_pos

    # Get contexts for unique false positives
    for pos in a_boundaries:
        content_pos = a_to_content[pos]
        if content_pos in a_only_fp:
            orig_pos = a_pos_map[pos] if pos < len(a_pos_map) else len(segmenter_a_text)
            result.a_only_false_positives.append(
                (orig_pos, get_context(segmenter_a_text, orig_pos))
            )

    for pos in b_boundaries:
        content_pos = b_to_content[pos]
        if content_pos in b_only_fp:
            orig_pos = b_pos_map[pos] if pos < len(b_pos_map) else len(segmenter_b_text)
            result.b_only_false_positives.append(
                (orig_pos, get_context(segmenter_b_text, orig_pos))
            )

    return result


def format_report(result: ComparisonResult) -> str:
    """Format a comparison result as a human-readable report.

    Args:
        result: ComparisonResult to format

    Returns:
        Formatted report string
    """
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("SEGMENTER COMPARISON REPORT")
    lines.append(sep)
    lines.append(f"Segmenter A: {result.segmenter_a_name}")
    lines.append(f"Segmenter B: {result.segmenter_b_name}")
    lines.append("")

    # Summary statistics
    lines.append(sep)
    lines.append("SUMMARY: Sentence Boundary Detection (True Positives)")
    lines.append(sep)
    lines.append(f"Total gold boundaries:     {result.total_boundaries}")
    lines.append(f"Both correct:              {result.both_correct}")
    lines.append(f"Neither correct:           {result.neither_correct}")
    lines.append(
        f"{result.segmenter_a_name} only (B regression): {result.a_only_correct}"
    )
    lines.append(
        f"{result.segmenter_b_name} only (B improvement): {result.b_only_correct}"
    )
    lines.append("")

    # Calculate metrics for each
    a_tp = result.both_correct + result.a_only_correct
    a_fn = result.neither_correct + result.b_only_correct
    b_tp = result.both_correct + result.b_only_correct
    b_fn = result.neither_correct + result.a_only_correct

    a_fp = len(result.a_false_positives)
    b_fp = len(result.b_false_positives)

    a_precision = a_tp / (a_tp + a_fp) if (a_tp + a_fp) > 0 else 0.0
    a_recall = a_tp / (a_tp + a_fn) if (a_tp + a_fn) > 0 else 0.0
    a_f1 = (
        2 * a_precision * a_recall / (a_precision + a_recall)
        if (a_precision + a_recall) > 0
        else 0.0
    )

    b_precision = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
    b_recall = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
    b_f1 = (
        2 * b_precision * b_recall / (b_precision + b_recall)
        if (b_precision + b_recall) > 0
        else 0.0
    )

    lines.append(sep)
    lines.append("METRICS COMPARISON")
    lines.append(sep)
    lines.append(
        f"{'Metric':<20} {result.segmenter_a_name:<15} "
        f"{result.segmenter_b_name:<15} {'Difference':<15}"
    )
    lines.append("-" * 65)
    lines.append(f"{'True Positives':<20} {a_tp:<15} {b_tp:<15} {b_tp - a_tp:+d}")
    lines.append(f"{'False Negatives':<20} {a_fn:<15} {b_fn:<15} {b_fn - a_fn:+d}")
    lines.append(f"{'False Positives':<20} {a_fp:<15} {b_fp:<15} {b_fp - a_fp:+d}")
    lines.append(
        f"{'Precision':<20} {a_precision:<15.4f} "
        f"{b_precision:<15.4f} {b_precision - a_precision:+.4f}"
    )
    lines.append(
        f"{'Recall':<20} {a_recall:<15.4f} {b_recall:<15.4f} {b_recall - a_recall:+.4f}"
    )
    lines.append(f"{'F1-Score':<20} {a_f1:<15.4f} {b_f1:<15.4f} {b_f1 - a_f1:+.4f}")
    lines.append("")

    # False positive summary
    lines.append(sep)
    lines.append("FALSE POSITIVES SUMMARY")
    lines.append(sep)
    lines.append(f"{result.segmenter_a_name} total FP: {len(result.a_false_positives)}")
    lines.append(f"{result.segmenter_b_name} total FP: {len(result.b_false_positives)}")
    lines.append(
        f"{result.segmenter_a_name} only FP (B fixed): "
        f"{len(result.a_only_false_positives)}"
    )
    lines.append(
        f"{result.segmenter_b_name} only FP (B introduced): "
        f"{len(result.b_only_false_positives)}"
    )
    lines.append("")

    # Detailed cases: B regressions (A got it right, B missed it)
    if result.a_only_cases:
        lines.append(sep)
        lines.append(
            f"B REGRESSIONS: {result.segmenter_a_name} correct, "
            f"{result.segmenter_b_name} missed ({len(result.a_only_cases)} cases)"
        )
        lines.append(sep)
        lines.append(
            "These are boundaries that B failed to detect but A detected correctly."
        )
        lines.append("")
        for i, case in enumerate(result.a_only_cases, 1):
            lines.append(f"[{i}] Gold sentence {case.gold_sentence_num}")
            lines.append(f"    {case.context}")
            lines.append("")

    # Detailed cases: B improvements (B got it right, A missed it)
    if result.b_only_cases:
        lines.append(sep)
        lines.append(
            f"B IMPROVEMENTS: {result.segmenter_b_name} correct, "
            f"{result.segmenter_a_name} missed ({len(result.b_only_cases)} cases)"
        )
        lines.append(sep)
        lines.append(
            "These are boundaries that B detected correctly but A failed to detect."
        )
        lines.append("")
        for i, case in enumerate(result.b_only_cases, 1):
            lines.append(f"[{i}] Gold sentence {case.gold_sentence_num}")
            lines.append(f"    {case.context}")
            lines.append("")

    # Detailed cases: B-only false positives (B introduced errors)
    if result.b_only_false_positives:
        lines.append(sep)
        count = len(result.b_only_false_positives)
        lines.append(
            f"B INTRODUCED FALSE POSITIVES: "
            f"{result.segmenter_b_name} only ({count} cases)"
        )
        lines.append(sep)
        lines.append(
            f"These are incorrect boundaries that {result.segmenter_b_name} added "
            f"but {result.segmenter_a_name} did not."
        )
        lines.append("")
        for i, (pos, context) in enumerate(result.b_only_false_positives, 1):
            lines.append(f"[{i}] Position {pos}")
            lines.append(f"    {context}")
            lines.append("")

    # Detailed cases: A-only false positives (B fixed errors)
    if result.a_only_false_positives:
        lines.append(sep)
        count = len(result.a_only_false_positives)
        lines.append(
            f"B FIXED FALSE POSITIVES: {result.segmenter_a_name} only ({count} cases)"
        )
        lines.append(sep)
        lines.append(
            f"These are incorrect boundaries that {result.segmenter_a_name} added "
            f"but {result.segmenter_b_name} correctly avoided."
        )
        lines.append("")
        for i, (pos, context) in enumerate(result.a_only_false_positives, 1):
            lines.append(f"[{i}] Position {pos}")
            lines.append(f"    {context}")
            lines.append("")

    return "\n".join(lines)


def resolve_file_path(
    segmenter: str,
    lang: str,
    model_size: str,
    variant: str,
    script_dir: Path,
) -> Path:
    """Resolve output file path for a segmenter.

    Args:
        segmenter: Segmenter name (spacy, phrasplit, sentencizer)
        lang: Language code (en, de, etc.)
        model_size: Model size (sm, md, lg)
        variant: Test variant (all, none, mixed)
        script_dir: Directory containing the script

    Returns:
        Path to the output file
    """
    # Sentencizer doesn't use model size
    if segmenter == "sentencizer":
        filename = f"UD_{lang}_{segmenter}.{variant}.out"
    else:
        filename = f"UD_{lang}_{segmenter}_{model_size}.{variant}.out"

    return script_dir / "outfiles" / filename


def resolve_gold_path(lang: str, script_dir: Path) -> Path:
    """Resolve gold standard file path for a language.

    Args:
        lang: Language code (en, de, etc.)
        script_dir: Directory containing the script

    Returns:
        Path to the gold file
    """
    lang_name = LANG_CODE_TO_NAME.get(lang)
    if not lang_name:
        raise ValueError(f"Unknown language code '{lang}'")

    return script_dir / "testsets" / f"UD_{lang_name}.dataset.gold"


def print_list_info() -> None:
    """Print available options for --list flag."""
    print("Available options:")
    print()
    print("Languages:")
    langs = sorted(LANG_CODE_TO_NAME.keys())
    print(f"  {', '.join(langs)}")
    print()
    print("Segmenters:")
    print(f"  {', '.join(VALID_SEGMENTERS)}")
    print()
    print("Model sizes:")
    print(f"  {', '.join(VALID_MODEL_SIZES)}")
    print("  Note: 'sentencizer' does not use model size (rule-based)")
    print()
    print("Variants:")
    print(f"  {', '.join(VALID_VARIANTS)}")
    print("  - all: Same as gold (trivial test)")
    print("  - none: All sentences on one line (hardest test)")
    print("  - mixed: Paragraph-like (3-8 sentences per line)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available languages, segmenters, model sizes, and variants",
    )
    parser.add_argument(
        "lang",
        type=str,
        nargs="?",
        help="Language code (e.g., en, de, fr)",
    )
    parser.add_argument(
        "segmenter_a",
        type=str,
        nargs="?",
        help=f"First segmenter: {', '.join(VALID_SEGMENTERS)}",
    )
    parser.add_argument(
        "segmenter_b",
        type=str,
        nargs="?",
        help=f"Second segmenter: {', '.join(VALID_SEGMENTERS)}",
    )
    parser.add_argument(
        "--model-size",
        "-m",
        type=str,
        choices=VALID_MODEL_SIZES,
        help=f"Model size: {', '.join(VALID_MODEL_SIZES)} (required)",
    )
    parser.add_argument(
        "--variant",
        "-v",
        type=str,
        choices=VALID_VARIANTS,
        help=f"Test variant: {', '.join(VALID_VARIANTS)} (required)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="FILE",
        help="Output file for the comparison report (default: print to stdout)",
    )
    parser.add_argument(
        "--gold",
        type=str,
        metavar="FILE",
        help="Override gold standard file path (auto-detected by default)",
    )
    parser.add_argument(
        "--file-a",
        type=str,
        metavar="FILE",
        help="Override segmenter A output file path (auto-detected by default)",
    )
    parser.add_argument(
        "--file-b",
        type=str,
        metavar="FILE",
        help="Override segmenter B output file path (auto-detected by default)",
    )

    args = parser.parse_args()

    # Handle --list flag
    if args.list:
        print_list_info()
        sys.exit(0)

    # Validate required positional arguments
    if not args.lang:
        parser.error("the following arguments are required: lang")
    if not args.segmenter_a:
        parser.error("the following arguments are required: segmenter_a")
    if not args.segmenter_b:
        parser.error("the following arguments are required: segmenter_b")

    # Validate required options
    if not args.model_size:
        parser.error("the following arguments are required: --model-size/-m")
    if not args.variant:
        parser.error("the following arguments are required: --variant/-v")

    # Validate language
    if args.lang not in LANG_CODE_TO_NAME:
        print(
            f"Error: Unknown language code '{args.lang}'. "
            f"Use --list to see available languages.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate segmenter names
    for _name, seg in [
        ("segmenter_a", args.segmenter_a),
        ("segmenter_b", args.segmenter_b),
    ]:
        if seg not in VALID_SEGMENTERS:
            print(
                f"Error: Unknown segmenter '{seg}'. "
                f"Valid options: {', '.join(VALID_SEGMENTERS)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Get script directory for resolving paths
    script_dir = Path(__file__).parent

    # Resolve file paths
    if args.gold:
        gold_path = Path(args.gold)
    else:
        gold_path = resolve_gold_path(args.lang, script_dir)

    if args.file_a:
        file_a_path = Path(args.file_a)
    else:
        file_a_path = resolve_file_path(
            args.segmenter_a, args.lang, args.model_size, args.variant, script_dir
        )

    if args.file_b:
        file_b_path = Path(args.file_b)
    else:
        file_b_path = resolve_file_path(
            args.segmenter_b, args.lang, args.model_size, args.variant, script_dir
        )

    # Read files with helpful error messages
    try:
        with open(gold_path, encoding="utf-8") as f:
            gold_text = f.read()
    except FileNotFoundError:
        print(f"Error: Gold file not found: {gold_path}", file=sys.stderr)
        print(
            f"Hint: Run 'python build-testset.py --download {args.lang}' "
            "first to download the dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(file_a_path, encoding="utf-8") as f:
            segmenter_a_text = f.read()
    except FileNotFoundError:
        print(f"Error: Output file not found: {file_a_path}", file=sys.stderr)
        seg_flag = (
            f"--{args.segmenter_a}"
            if args.segmenter_a != "sentencizer"
            else "--sentencizer"
        )
        print(
            f"Hint: Run 'python runbatcheval.py {args.lang} {seg_flag}' "
            "first to generate this file.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(file_b_path, encoding="utf-8") as f:
            segmenter_b_text = f.read()
    except FileNotFoundError:
        print(f"Error: Output file not found: {file_b_path}", file=sys.stderr)
        seg_flag = (
            f"--{args.segmenter_b}"
            if args.segmenter_b != "sentencizer"
            else "--sentencizer"
        )
        print(
            f"Hint: Run 'python runbatcheval.py {args.lang} {seg_flag}' "
            "first to generate this file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compare
    try:
        result = compare_segmenters(
            gold_text,
            segmenter_a_text,
            segmenter_b_text,
            segmenter_a_name=args.segmenter_a,
            segmenter_b_name=args.segmenter_b,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Format report
    report = format_report(result)

    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Comparison report saved to: {output_path}")

        # Also print summary to stdout
        print()
        print("=== SUMMARY ===")
        print(f"Total gold boundaries: {result.total_boundaries}")
        print(f"Both correct: {result.both_correct}")
        print(f"Neither correct: {result.neither_correct}")
        print(f"{args.segmenter_a} only (B regression): {result.a_only_correct}")
        print(f"{args.segmenter_b} only (B improvement): {result.b_only_correct}")
        print(f"{args.segmenter_a} false positives: {len(result.a_false_positives)}")
        print(f"{args.segmenter_b} false positives: {len(result.b_false_positives)}")
    else:
        print(report)


if __name__ == "__main__":
    main()
