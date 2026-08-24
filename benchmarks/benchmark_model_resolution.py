"""Measure phrasplit model-resolution hot paths without pass/fail thresholds.

Run this script manually when comparing environments, for example:

    python benchmarks/benchmark_model_resolution.py --iterations 1000
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import phrasplit
from phrasplit.spacy_models import clear_spacy_model_cache, resolve_spacy_model

_TEXT = "Dr. Smith arrived. He sat down."


def _measure(callback: Callable[[], object], iterations: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        callback()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    clear_spacy_model_cache()
    cold = _measure(
        lambda: resolve_spacy_model(language=args.language, require=False),
    )
    warm = _measure(
        lambda: resolve_spacy_model(language=args.language, require=False),
        args.iterations,
    )
    automatic_split = _measure(
        lambda: phrasplit.split_text(_TEXT, language=args.language),
        args.iterations,
    )
    forced_regex = _measure(
        lambda: phrasplit.split_text(_TEXT, use_spacy=False),
        args.iterations,
    )

    print(f"cold automatic resolution: {cold:.6f}s (1 call)")
    print(f"warm automatic resolution: {warm:.6f}s ({args.iterations} calls)")
    print(
        f"automatic split_text: {automatic_split:.6f}s "
        f"({args.iterations} calls)"
    )
    print(
        f"forced-regex split_text: {forced_regex:.6f}s "
        f"({args.iterations} calls)"
    )


if __name__ == "__main__":
    main()
