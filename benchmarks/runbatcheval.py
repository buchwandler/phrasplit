#!/usr/bin/env python
"""Batch evaluation script for sentence segmentation benchmarks.

This script compares phrasplit (spaCy + corrections) against raw spaCy
on test datasets and evaluates performance against gold standard files.

Usage:
    python runbatcheval.py <language_code>
    python runbatcheval.py en
    python runbatcheval.py --all

Example:
    python runbatcheval.py en           # Evaluate English (spaCy + phrasplit)
    python runbatcheval.py en --spacy   # Evaluate raw spaCy only
    python runbatcheval.py en --phrasplit  # Evaluate phrasplit (spaCy mode) only
    python runbatcheval.py en --phrasplit-simple  # Evaluate phrasplit (simple mode)
    python runbatcheval.py --all        # Evaluate all languages
    python runbatcheval.py en -o results.txt  # Save output to file
    python runbatcheval.py en --model-size sm  # Only test small models
    python runbatcheval.py en --model-size sm md  # Test small and medium models
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Segmenter types
SEGMENTER_SPACY = "spacy"
SEGMENTER_PHRASPLIT = "phrasplit"
SEGMENTER_PHRASPLIT_SIMPLE = "phrasplit_simple"
SEGMENTER_SENTENCIZER = "sentencizer"


class Output:
    """Output handler that writes to stdout and optionally to a file."""

    def __init__(self, file: TextIO | None = None):
        """Initialize output handler.

        Args:
            file: Optional file to write output to (in addition to stdout)
        """
        self.file = file

    def print(self, *args, **kwargs) -> None:
        """Print to stdout and optionally to file."""
        print(*args, **kwargs)
        if self.file:
            # Remove 'file' from kwargs if present, we handle it ourselves
            kwargs.pop("file", None)
            print(*args, **kwargs, file=self.file)


# Global output handler (set in main)
output = Output()


@dataclass
class LanguageConfig:
    """Configuration for a language."""

    name: str
    prefix: str = "UD"  # Default: Universal Dependencies
    models: list[str] = field(default_factory=list)  # spaCy models to test


# Language code to configuration mapping
# Models based on https://spacy.io/models
LANGUAGES: dict[str, LanguageConfig] = {
    # Languages WITH official spaCy models
    "ca": LanguageConfig(
        "Catalan",
        models=["ca_core_news_sm", "ca_core_news_md", "ca_core_news_lg"],
    ),
    "da": LanguageConfig(
        "Danish",
        models=["da_core_news_sm", "da_core_news_md", "da_core_news_lg"],
    ),
    "de": LanguageConfig(
        "German",
        models=["de_core_news_sm", "de_core_news_md", "de_core_news_lg"],
    ),
    "el": LanguageConfig(
        "Greek",
        models=["el_core_news_sm", "el_core_news_md", "el_core_news_lg"],
    ),
    "en": LanguageConfig(
        "English",
        models=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
    ),
    "es": LanguageConfig(
        "Spanish",
        models=["es_core_news_sm", "es_core_news_md", "es_core_news_lg"],
    ),
    "fi": LanguageConfig(
        "Finnish",
        models=["fi_core_news_sm", "fi_core_news_md", "fi_core_news_lg"],
    ),
    "fr": LanguageConfig(
        "French",
        models=["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"],
    ),
    "hr": LanguageConfig(
        "Croatian",
        models=["hr_core_news_sm", "hr_core_news_md", "hr_core_news_lg"],
    ),
    "it": LanguageConfig(
        "Italian",
        models=["it_core_news_sm", "it_core_news_md", "it_core_news_lg"],
    ),
    "ja": LanguageConfig(
        "Japanese",
        models=["ja_core_news_sm", "ja_core_news_md", "ja_core_news_lg"],
    ),
    "ko": LanguageConfig(
        "Korean",
        models=["ko_core_news_sm", "ko_core_news_md", "ko_core_news_lg"],
    ),
    "lt": LanguageConfig(
        "Lithuanian",
        models=["lt_core_news_sm", "lt_core_news_md", "lt_core_news_lg"],
    ),
    "mk": LanguageConfig(
        "Macedonian",
        prefix="SETIMES",
        models=["mk_core_news_sm", "mk_core_news_md", "mk_core_news_lg"],
    ),
    "nb": LanguageConfig(
        "Norwegian-Bokmaal",
        models=["nb_core_news_sm", "nb_core_news_md", "nb_core_news_lg"],
    ),
    "nl": LanguageConfig(
        "Dutch",
        models=["nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg"],
    ),
    "pl": LanguageConfig(
        "Polish",
        models=["pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg"],
    ),
    "pt": LanguageConfig(
        "Portuguese",
        models=["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"],
    ),
    "ro": LanguageConfig(
        "Romanian",
        models=["ro_core_news_sm", "ro_core_news_md", "ro_core_news_lg"],
    ),
    "ru": LanguageConfig(
        "Russian",
        models=["ru_core_news_sm", "ru_core_news_md", "ru_core_news_lg"],
    ),
    "sl": LanguageConfig(
        "Slovenian",
        models=["sl_core_news_sm", "sl_core_news_md", "sl_core_news_lg"],
    ),
    "sv": LanguageConfig(
        "Swedish",
        models=["sv_core_news_sm", "sv_core_news_md", "sv_core_news_lg"],
    ),
    "uk": LanguageConfig(
        "Ukrainian",
        models=["uk_core_news_sm", "uk_core_news_md", "uk_core_news_lg"],
    ),
    "zh": LanguageConfig(
        "Chinese",
        models=["zh_core_web_sm", "zh_core_web_md", "zh_core_web_lg"],
    ),
    # Languages WITHOUT official spaCy models (use multilingual or fallback)
    # These use the multi-language model xx_* or fall back to a similar language
    "bg": LanguageConfig(
        "Bulgarian",
        models=[],  # No official model, will use default
    ),
    "cnr": LanguageConfig(
        "Montenegrin",
        prefix="MESUBS",
        models=[],  # No official model
    ),
    "cs": LanguageConfig(
        "Czech",
        models=[],  # No official model
    ),
    "et": LanguageConfig(
        "Estonian",
        models=[],  # No official model
    ),
    "hu": LanguageConfig(
        "Hungarian",
        models=[],  # No official model
    ),
    "is": LanguageConfig(
        "Icelandic",
        models=[],  # No official model
    ),
    "lv": LanguageConfig(
        "Latvian",
        models=[],  # No official model
    ),
    "mt": LanguageConfig(
        "Maltese",
        models=[],  # No official model
    ),
    "nn": LanguageConfig(
        "Norwegian-Nynorsk",
        models=["nb_core_news_sm"],  # Use Bokmål model
    ),
    "sk": LanguageConfig(
        "Slovak",
        models=[],  # No official model
    ),
    "sq": LanguageConfig(
        "Albanian",
        prefix="SETIMES",
        models=[],  # No official model
    ),
    "sr": LanguageConfig(
        "Serbian",
        models=[],  # No official model
    ),
    "tr": LanguageConfig(
        "Turkish",
        models=[],  # No official model
    ),
}


@dataclass
class EvalResult:
    """Evaluation result for a single test."""

    segmenter: str  # "spacy" or "phrasplit"
    flavour: str
    model: str
    time: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f_measure: float


def run_segmenter(
    segmenter: str,
    lang_name: str,
    test_file: Path,
    out_file: Path,
    model: str | None = None,
    split_on_colon: bool = True,
) -> tuple[float, bool]:
    """Run a sentence segmenter.

    Args:
        segmenter: "spacy", "phrasplit", "phrasplit_simple", or "sentencizer"
        lang_name: Language name (e.g., "English")
        test_file: Input test file
        out_file: Output file
        model: Optional spaCy model override
        split_on_colon: Whether to split on colons (only affects phrasplit)

    Returns:
        Tuple of (elapsed_time, success)
    """
    if segmenter == SEGMENTER_SPACY:
        script = SCRIPT_DIR / "spacy_segmenter.py"
    elif segmenter == SEGMENTER_SENTENCIZER:
        script = SCRIPT_DIR / "sentencizer_segmenter.py"
    elif segmenter == SEGMENTER_PHRASPLIT_SIMPLE:
        script = SCRIPT_DIR / "phrasplit_segmenter.py"
    else:
        script = SCRIPT_DIR / "phrasplit_segmenter.py"

    cmd = [
        sys.executable,
        str(script),
        lang_name,
        str(test_file),
        str(out_file),
    ]

    if model:
        cmd.extend(["--model", model])

    # Add --no-split-on-colon flag for phrasplit if needed
    if segmenter in (SEGMENTER_PHRASPLIT, SEGMENTER_PHRASPLIT_SIMPLE):
        if not split_on_colon:
            cmd.append("--no-split-on-colon")
        # Add --simple flag for simple mode
        if segmenter == SEGMENTER_PHRASPLIT_SIMPLE:
            cmd.append("--simple")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"  Error: {result.stderr}", file=sys.stderr)
        return elapsed, False

    return elapsed, True


def run_segmenteval(
    gold_file: Path,
    test_file: Path,
    errors_prefix: str | None = None,
    segmenter: str | None = None,
) -> dict[str, float] | None:
    """Run segmenteval.py and parse results.

    Args:
        gold_file: Path to gold standard file
        test_file: Path to test output file
        errors_prefix: Optional prefix for error files
        segmenter: Optional segmenter name to include in error files

    Returns:
        Dictionary with evaluation metrics or None on error
    """
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "segmenteval.py"),
        str(gold_file),
        str(test_file),
    ]

    if errors_prefix:
        cmd.extend(["--errors-prefix", errors_prefix])
        if segmenter:
            cmd.extend(["--segmenter", segmenter])
    else:
        cmd.append("--no-errors")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"  Evaluation failed: {result.stderr}", file=sys.stderr)
        return None

    # Parse output
    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.lower().replace(" ", "_").replace("-", "_")
            try:
                metrics[key] = float(value)
            except ValueError:
                pass

    return metrics


def get_model_suffix(model: str) -> str:
    """Get a short suffix for the model name.

    Args:
        model: Full model name

    Returns:
        Short suffix (e.g., 'sm', 'lg')
    """
    if model.endswith("_sm"):
        return "sm"
    elif model.endswith("_lg"):
        return "lg"
    elif model.endswith("_md"):
        return "md"
    elif model.endswith("_trf"):
        return "trf"
    return model.split("_")[-1]


def _get_models_to_test(
    lang_config: LanguageConfig, model_sizes: list[str] | None
) -> list[str | None]:
    """Determine which models to test based on language config and size filter."""
    if not lang_config.models:
        return [None]  # Use default model for the language

    if model_sizes is None or "all" in model_sizes:
        models: list[str | None] = list(lang_config.models)
        return models

    # Filter models by size suffix
    filtered: list[str | None] = [
        m
        for m in lang_config.models
        if any(m.endswith(f"_{size}") for size in model_sizes)
    ]
    return filtered if filtered else [None]  # Fallback to default if no matching models


def _get_output_filename(
    lang_config: LanguageConfig,
    lang_code: str,
    segmenter: str,
    model_suffix: str,
    model: str | None,
    flavour: str,
) -> str:
    """Generate output filename for segmenter results."""
    if model:
        return (
            f"{lang_config.prefix}_{lang_code}_{segmenter}_{model_suffix}.{flavour}.out"
        )
    return f"{lang_config.prefix}_{lang_code}_{segmenter}.{flavour}.out"


def _get_errors_prefix(
    outfiles_dir: Path,
    lang_config: LanguageConfig,
    lang_code: str,
    segmenter: str,
    model_suffix: str,
    model: str | None,
    flavour: str,
    save_errors: bool,
) -> str | None:
    """Generate error file prefix if saving errors."""
    if not save_errors:
        return None
    if model:
        err_name = (
            f"{lang_config.prefix}_{lang_code}_{segmenter}_"
            f"{model_suffix}_{flavour}_errors"
        )
    else:
        err_name = f"{lang_config.prefix}_{lang_code}_{segmenter}_{flavour}_errors"
    return str(outfiles_dir / err_name)


def _print_result_verbose(result: EvalResult) -> None:
    """Print detailed result metrics."""
    output.print(f"  True positives: {result.true_positives}")
    output.print(f"  False positives: {result.false_positives}")
    output.print(f"  False negatives: {result.false_negatives}")
    output.print(f"  Precision: {result.precision:.3f}")
    output.print(f"  Recall: {result.recall:.3f}")
    output.print(f"  F-measure: {result.f_measure:.3f}")


def evaluate_language(
    lang_code: str,
    segmenters: list[str],
    verbose: bool = True,
    save_errors: bool = True,
    split_on_colon: bool = True,
    model_sizes: list[str] | None = None,
) -> list[EvalResult]:
    """Evaluate segmenters on all flavours for a language.

    Args:
        lang_code: Language code
        segmenters: List of segmenters to test
            ("spacy", "phrasplit", "phrasplit_simple", "sentencizer")
        verbose: Print detailed output
        save_errors: Save error files for analysis
        split_on_colon: Whether to split on colons (only affects phrasplit)
        model_sizes: Filter models by size (sm, md, lg, trf). None or ["all"] for all.

    Returns:
        List of evaluation results
    """
    if lang_code not in LANGUAGES:
        print(f"Unknown language code: {lang_code}", file=sys.stderr)
        return []

    lang_config = LANGUAGES[lang_code]

    # Create output directory
    outfiles_dir = SCRIPT_DIR / "outfiles"
    outfiles_dir.mkdir(exist_ok=True)

    # Gold standard file
    gold_file = (
        SCRIPT_DIR
        / "testsets"
        / f"{lang_config.prefix}_{lang_config.name}.dataset.gold"
    )

    if not gold_file.exists():
        print(f"Gold file not found: {gold_file}", file=sys.stderr)
        return []

    models = _get_models_to_test(lang_config, model_sizes)

    if verbose:
        output.print("#" * 50)
        output.print(f"{lang_code} ({lang_config.name})")
        output.print(f"Gold standard: {lang_config.prefix}")
        if lang_config.models:
            output.print(f"Models: {', '.join(m for m in models if m)}")
        output.print(f"Segmenters: {', '.join(segmenters)}")
        output.print("#" * 50)

    results = []

    for model in models:
        model_suffix = get_model_suffix(model) if model else "default"

        if verbose and model:
            output.print(f"\n>>> Model: {model}")

        for segmenter in segmenters:
            if verbose:
                output.print(f"\n--- Segmenter: {segmenter} ---")

            for flavour in ["none", "all", "mixed"]:
                if verbose:
                    label = f"{segmenter}/{model_suffix}" if model else segmenter
                    output.print(f"\nTestset: {flavour} ({label})")

                test_file = (
                    SCRIPT_DIR
                    / "testsets"
                    / f"{lang_config.prefix}_{lang_config.name}.dataset.{flavour}"
                )

                if not test_file.exists():
                    if verbose:
                        output.print(f"  Test file not found: {test_file}")
                    continue

                out_name = _get_output_filename(
                    lang_config, lang_code, segmenter, model_suffix, model, flavour
                )
                out_file = outfiles_dir / out_name

                # Run segmenter
                elapsed, success = run_segmenter(
                    segmenter,
                    lang_config.name,
                    test_file,
                    out_file,
                    model,
                    split_on_colon=split_on_colon,
                )

                if not success:
                    continue

                if verbose:
                    output.print(f"  Time: {elapsed:.2f}s")

                errors_prefix = _get_errors_prefix(
                    outfiles_dir,
                    lang_config,
                    lang_code,
                    segmenter,
                    model_suffix,
                    model,
                    flavour,
                    save_errors,
                )

                # Evaluate
                metrics = run_segmenteval(gold_file, out_file, errors_prefix, segmenter)

                if metrics:
                    result = EvalResult(
                        segmenter=segmenter,
                        flavour=flavour,
                        model=model_suffix,
                        time=elapsed,
                        true_positives=int(metrics.get("true_positives", 0)),
                        false_positives=int(metrics.get("false_positives", 0)),
                        false_negatives=int(metrics.get("false_negatives", 0)),
                        precision=metrics.get("precision", 0.0),
                        recall=metrics.get("recall", 0.0),
                        f_measure=metrics.get("f_measure", 0.0),
                    )
                    results.append(result)

                    if verbose:
                        _print_result_verbose(result)

                if verbose:
                    output.print("=" * 30)

    return results


def print_summary(all_results: dict[str, list[EvalResult]]) -> None:
    """Print a summary table of all results.

    Args:
        all_results: Dictionary mapping language code to results
    """
    output.print("\n" + "=" * 100)
    output.print("SUMMARY")
    output.print("=" * 100)
    header = (
        f"{'Lang':<6} {'Segmenter':<10} {'Model':<8} {'Flavour':<8} "
        f"{'Precision':>10} {'Recall':>10} {'F-measure':>10} {'Time':>8}"
    )
    output.print(header)
    output.print("-" * 100)

    for lang_code in sorted(all_results.keys()):
        results = all_results[lang_code]
        for result in results:
            output.print(
                f"{lang_code:<6} {result.segmenter:<10} {result.model:<8} "
                f"{result.flavour:<8} {result.precision:>10.3f} "
                f"{result.recall:>10.3f} {result.f_measure:>10.3f} "
                f"{result.time:>7.2f}s"
            )


def print_comparison(all_results: dict[str, list[EvalResult]]) -> None:
    """Print a comparison between spaCy and phrasplit.

    Args:
        all_results: Dictionary mapping language code to results
    """
    output.print("\n" + "=" * 80)
    output.print("COMPARISON: phrasplit vs spaCy (F-measure difference)")
    output.print("=" * 80)
    header = (
        f"{'Lang':<6} {'Model':<8} {'Flavour':<8} "
        f"{'spaCy':>10} {'phrasplit':>10} {'Diff':>10}"
    )
    output.print(header)
    output.print("-" * 80)

    for lang_code in sorted(all_results.keys()):
        results = all_results[lang_code]

        # Group by model and flavour
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for result in results:
            key = (result.model, result.flavour)
            if key not in grouped:
                grouped[key] = {}
            grouped[key][result.segmenter] = result.f_measure

        for (model, flavour), scores in sorted(grouped.items()):
            if SEGMENTER_SPACY in scores and SEGMENTER_PHRASPLIT in scores:
                spacy_f = scores[SEGMENTER_SPACY]
                phrasplit_f = scores[SEGMENTER_PHRASPLIT]
                diff = phrasplit_f - spacy_f
                diff_str = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
                output.print(
                    f"{lang_code:<6} {model:<8} {flavour:<8} "
                    f"{spacy_f:>10.3f} {phrasplit_f:>10.3f} {diff_str:>10}"
                )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "lang_code",
        nargs="?",
        choices=list(LANGUAGES.keys()),
        help="Language code (e.g., en, de, fr)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all languages",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported languages",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode (only print summary)",
    )
    parser.add_argument(
        "--no-errors",
        action="store_true",
        help="Don't save error files",
    )
    parser.add_argument(
        "--spacy",
        action="store_true",
        help="Only test raw spaCy segmenter",
    )
    parser.add_argument(
        "--phrasplit",
        action="store_true",
        help="Only test phrasplit segmenter (spaCy mode)",
    )
    parser.add_argument(
        "--phrasplit-simple",
        action="store_true",
        help="Only test phrasplit segmenter (simple/regex mode)",
    )
    parser.add_argument(
        "--sentencizer",
        action="store_true",
        help="Only test rule-based sentencizer (no model required)",
    )
    parser.add_argument(
        "--no-split-on-colon",
        action="store_true",
        help="Disable splitting on colons (only affects phrasplit)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        nargs="+",
        choices=["sm", "md", "lg", "trf", "all"],
        default=["all"],
        help="Filter models by size (sm, md, lg, trf, or all). Can specify multiple.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="FILE",
        help="Write output to file (in addition to stdout)",
    )

    args = parser.parse_args()

    # Initialize output handler with optional file
    global output
    output_file = open(args.output, "w", encoding="utf-8") if args.output else None  # noqa: SIM115
    output = Output(output_file)

    try:
        if args.list:
            print("Supported languages:")
            for code, config in sorted(LANGUAGES.items()):
                model_info = f", models: {len(config.models)}" if config.models else ""
                print(f"  {code}: {config.name} (prefix: {config.prefix}{model_info})")
            print("\nNote: Languages with models test sm, md, lg variants by default.")
            return

        # Determine which segmenters to test
        selected = []
        if args.spacy:
            selected.append(SEGMENTER_SPACY)
        if args.phrasplit:
            selected.append(SEGMENTER_PHRASPLIT)
        if args.phrasplit_simple:
            selected.append(SEGMENTER_PHRASPLIT_SIMPLE)
        if args.sentencizer:
            selected.append(SEGMENTER_SENTENCIZER)

        # Default: test spacy and phrasplit (original behavior)
        if not selected:
            segmenters = [SEGMENTER_SPACY, SEGMENTER_PHRASPLIT]
        else:
            segmenters = selected

        save_errors = not args.no_errors
        split_on_colon = not args.no_split_on_colon
        model_sizes = args.model_size

        if args.all:
            all_results = {}
            for lang_code in LANGUAGES:
                output.print(f"\nEvaluating {lang_code}...")
                results = evaluate_language(
                    lang_code,
                    segmenters,
                    verbose=not args.quiet,
                    save_errors=save_errors,
                    split_on_colon=split_on_colon,
                    model_sizes=model_sizes,
                )
                if results:
                    all_results[lang_code] = results
            print_summary(all_results)
            if len(segmenters) == 2:
                print_comparison(all_results)

        elif args.lang_code:
            results = evaluate_language(
                args.lang_code,
                segmenters,
                verbose=True,
                save_errors=save_errors,
                split_on_colon=split_on_colon,
                model_sizes=model_sizes,
            )
            if results:
                print_summary({args.lang_code: results})
                if len(segmenters) == 2:
                    print_comparison({args.lang_code: results})

        else:
            parser.print_help()
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
