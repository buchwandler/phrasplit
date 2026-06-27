"""Comparison tests between spaCy and simple splitters.

This module tests both implementations against the same test cases
to measure accuracy differences.
"""

import pytest

from phrasplit import split_sentences

# Test cases (text, expected_sentence_count_range)
COMPARISON_TEST_CASES = [
    ("Dr. Smith is here. She has a Ph.D. in Chemistry.", (2, 2)),
    ("Mr. Brown met Prof. Green. They discussed the U.S.A. case.", (2, 2)),
    ("U.S.A. is big. It has many states.", (2, 2)),
    ("Visit www.example.com. Then send feedback.", (2, 2)),
    ("Mr. J.R.R. Tolkien wrote many books. They were popular.", (2, 2)),
    ("E. coli is a bacteria. Dr. E. Stone confirmed it.", (2, 2)),
    ('She said, "It works!" Then she smiled.', (2, 2)),
    ("Smith & Co. Ltd. is closed. We're switching vendors.", (2, 3)),  # May vary
    ("This is a sentence without trailing punctuation", (1, 1)),
    ("First sentence. Second sentence! Third sentence?", (3, 3)),
    ("What is this? It's amazing! Really incredible.", (3, 3)),
]


class TestSpacyVsSimpleComparison:
    """Compare spaCy and simple implementations on same tests."""

    @pytest.mark.parametrize("text,expected_range", COMPARISON_TEST_CASES)
    def test_sentence_count_spacy(
        self, text: str, expected_range: tuple[int, int]
    ) -> None:
        """Test spaCy implementation."""
        result = split_sentences(text, use_spacy=True)
        min_expected, max_expected = expected_range
        msg = (
            f"spaCy: Expected {min_expected}-{max_expected} sentences, "
            f"got {len(result)}: {result}"
        )
        assert min_expected <= len(result) <= max_expected, msg

    @pytest.mark.parametrize("text,expected_range", COMPARISON_TEST_CASES)
    def test_sentence_count_simple(
        self, text: str, expected_range: tuple[int, int]
    ) -> None:
        """Test simple implementation."""
        result = split_sentences(text, use_spacy=False)
        min_expected, max_expected = expected_range
        # Allow slightly more leniency for simple mode
        msg = (
            f"Simple: Expected {min_expected}-{max_expected} sentences, "
            f"got {len(result)}: {result}"
        )
        assert min_expected <= len(result) <= max_expected + 1, msg

    def test_direct_comparison(self) -> None:
        """Direct comparison of outputs."""
        test_text = (
            "Dr. Smith arrived. She brought her Ph.D. thesis. It was impressive."
        )

        spacy_result = split_sentences(test_text, use_spacy=True)
        simple_result = split_sentences(test_text, use_spacy=False)

        # Both should get 3 sentences
        assert len(spacy_result) == 3
        assert len(simple_result) == 3

        # Content should be very similar (may differ in whitespace/normalization)
        for spacy_sent, simple_sent in zip(spacy_result, simple_result, strict=False):
            # Check that key words are preserved
            spacy_words = set(spacy_sent.lower().split())
            simple_words = set(simple_sent.lower().split())
            # At least 80% word overlap
            overlap = len(spacy_words & simple_words)
            total = len(spacy_words | simple_words)
            assert overlap / total >= 0.8, f"Low overlap: {spacy_sent} vs {simple_sent}"


class TestAccuracyMetrics:
    """Measure accuracy metrics for simple splitter."""

    def test_accuracy_on_common_cases(self) -> None:
        """Test accuracy on common sentence patterns."""
        test_cases = [
            ("Hello world. How are you?", 2),
            ("I like pizza. She likes pasta.", 2),
            ("The cat sat. The dog ran. The bird flew.", 3),
            ("It's sunny! Let's go outside.", 2),
            ("Where is it? I can't find it.", 2),
        ]

        matches = 0
        total = len(test_cases)

        for text, expected_count in test_cases:
            result = split_sentences(text, use_spacy=False)
            if len(result) == expected_count:
                matches += 1

        accuracy = matches / total
        # Simple mode should get at least 80% accuracy on these easy cases
        assert accuracy >= 0.8, (
            f"Simple mode accuracy: {accuracy:.1%} (expected >= 80%)"
        )


def test_performance_indicator() -> None:
    """Quick performance comparison (not a rigorous benchmark)."""
    import time

    # Test text
    text = ". ".join([f"Sentence number {i}" for i in range(100)])

    # spaCy timing
    start = time.time()
    spacy_result = split_sentences(text, use_spacy=True)
    spacy_time = time.time() - start

    # Simple timing
    start = time.time()
    simple_result = split_sentences(text, use_spacy=False)
    simple_time = time.time() - start

    # Simple should be faster
    print("\nPerformance indicator:")
    print(f"  spaCy:  {spacy_time:.4f}s  ({len(spacy_result)} sentences)")
    print(f"  Simple: {simple_time:.4f}s  ({len(simple_result)} sentences)")

    # Calculate speedup, handling case where simple_time is very small or zero
    if simple_time > 0:
        speedup = spacy_time / simple_time
        print(f"  Speedup: {speedup:.2f}x")
    else:
        print("  Speedup: >1000x (simple_time too small to measure)")

    # Both should get same sentence count on this simple text
    assert abs(len(spacy_result) - len(simple_result)) <= 1
