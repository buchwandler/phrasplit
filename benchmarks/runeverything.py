#!/usr/bin/env python
"""Run phrasplit evaluation on all languages.

This script evaluates phrasplit sentence segmentation across all supported
languages and saves results to a file.

Usage:
    python runeverything.py
    python runeverything.py --output results.txt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from runbatcheval import LANGUAGES, evaluate_language, print_summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file for results (default: stdout)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode (only print summary)",
    )

    args = parser.parse_args()

    # Collect results for all languages
    all_results = {}
    failed_languages = []

    print(f"Phrasplit Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluating {len(LANGUAGES)} languages...")
    print()

    for lang_code in sorted(LANGUAGES.keys()):
        lang_name = LANGUAGES[lang_code].name
        print(f"[{lang_code}] {lang_name}...", end=" ", flush=True)

        try:
            results = evaluate_language(lang_code, verbose=not args.quiet)
            if results:
                all_results[lang_code] = results
                avg_f = sum(r.f_measure for r in results) / len(results)
                print(f"F={avg_f:.3f}")
            else:
                print("SKIPPED (no test files)")
                failed_languages.append(lang_code)
        except Exception as e:
            print(f"FAILED ({e})")
            failed_languages.append(lang_code)

    # Print summary
    print_summary(all_results)

    # Print failed languages
    if failed_languages:
        print(f"\nFailed/Skipped languages: {', '.join(failed_languages)}")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Phrasplit Benchmark Results\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Languages evaluated: {len(all_results)}\n")
            f.write("\n")
            header = (
                f"{'Lang':<6} {'Flavour':<8} {'Precision':>10} "
                f"{'Recall':>10} {'F-measure':>10}\n"
            )
            f.write(header)
            f.write("-" * 50 + "\n")

            for lang_code in sorted(all_results.keys()):
                f.writelines(
                    f"{lang_code:<6} {result.flavour:<8} "
                    f"{result.precision:>10.3f} {result.recall:>10.3f} "
                    f"{result.f_measure:>10.3f}\n"
                    for result in all_results[lang_code]
                )

        print(f"\nResults saved to: {output_path}")

    # Return exit code based on success
    return 0 if not failed_languages else 1


if __name__ == "__main__":
    sys.exit(main())
