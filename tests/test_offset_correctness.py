"""Tests for offset correctness guarantees."""

from __future__ import annotations

import re

import pytest

from phrasplit import split_with_offsets


def _assert_offsets_valid(text: str, segments) -> None:
    last_end = 0
    seen_ranges: set[tuple[int, int]] = set()

    for seg in segments:
        assert seg.char_start is not None
        assert seg.char_end is not None
        assert 0 <= seg.char_start <= seg.char_end <= len(text)
        assert text[seg.char_start : seg.char_end] == seg.text
        assert seg.char_start >= last_end
        assert (seg.char_start, seg.char_end) not in seen_ranges
        seen_ranges.add((seg.char_start, seg.char_end))
        last_end = seg.char_end


@pytest.mark.parametrize(
    "text",
    [
        "Hello world. Hello world.",
        "I like coffee, and I like tea.",
        "Wait... what? Wait... what?",
        "A: one. B: two. C: three.",
    ],
)
@pytest.mark.parametrize(
    "use_spacy", [False, pytest.param(True, marks=pytest.mark.spacy_model)]
)
def test_regression_strings_offsets(text: str, use_spacy: bool) -> None:
    if use_spacy:
        pytest.importorskip("spacy")

    segments = split_with_offsets(text, mode="sentence", use_spacy=use_spacy)
    assert segments
    _assert_offsets_valid(text, segments)


@pytest.mark.parametrize(
    "use_spacy", [False, pytest.param(True, marks=pytest.mark.spacy_model)]
)
def test_clause_offsets_for_repeated_phrases(use_spacy: bool) -> None:
    if use_spacy:
        pytest.importorskip("spacy")

    text = "I like coffee, and I like tea."
    segments = split_with_offsets(text, mode="clause", use_spacy=use_spacy)
    assert len(segments) >= 2
    _assert_offsets_valid(text, segments)


@pytest.mark.parametrize(
    "use_spacy", [False, pytest.param(True, marks=pytest.mark.spacy_model)]
)
def test_punctuation_and_correction_edges(use_spacy: bool) -> None:
    if use_spacy:
        pytest.importorskip("spacy")

    text = 'She said, "Wait... what?"\u2014then paused; finally: "Go . . . now\u2026"'
    segments = split_with_offsets(text, mode="sentence", use_spacy=use_spacy)
    assert segments
    _assert_offsets_valid(text, segments)


@pytest.mark.parametrize(
    "use_spacy", [False, pytest.param(True, marks=pytest.mark.spacy_model)]
)
def test_paragraph_boundaries_preserved(use_spacy: bool) -> None:
    if use_spacy:
        pytest.importorskip("spacy")

    text = "Para one. Still here.\n\nPara two. Next.\n\nPara three."
    segments = split_with_offsets(text, mode="sentence", use_spacy=use_spacy)
    assert segments
    _assert_offsets_valid(text, segments)

    boundaries = [match.start() for match in re.finditer(r"\n\s*\n", text)]
    for boundary in boundaries:
        for seg in segments:
            assert not (seg.char_start < boundary < seg.char_end)
