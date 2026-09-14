"""Tests for source-level parenthetical boundary detection."""

from __future__ import annotations

import pytest

from phrasplit import (
    ClauseBoundary,
    DetectedBoundary,
    detect_parenthetical_boundaries,
)


def _kinds(text: str) -> list[str]:
    return [boundary.kind for boundary in detect_parenthetical_boundaries(text)]


def test_reported_sentence_returns_only_exact_open_boundary() -> None:
    text = "They changed out their clothes (stained with blood)."

    boundaries = detect_parenthetical_boundaries(text)

    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert isinstance(boundary, DetectedBoundary)
    assert boundary == ClauseBoundary(
        text="(",
        char_start=31,
        char_end=32,
        kind="parenthetical_open",
        meta={"confidence": 1.0, "reason": "balanced_round_parentheses"},
    )
    assert text[boundary.char_start : boundary.char_end] == boundary.text


def test_medial_aside_returns_open_and_close() -> None:
    text = "They changed clothes (stained with blood) before leaving."

    boundaries = detect_parenthetical_boundaries(text)

    assert [boundary.kind for boundary in boundaries] == [
        "parenthetical_open",
        "parenthetical_close",
    ]
    assert [boundary.text for boundary in boundaries] == ["(", ")"]
    assert [(boundary.char_start, boundary.char_end) for boundary in boundaries] == [
        (text.index("("), text.index("(") + 1),
        (text.index(")"), text.index(")") + 1),
    ]
    assert all(text[b.char_start : b.char_end] == b.text for b in boundaries)


@pytest.mark.parametrize(
    "text",
    [
        "Host (aside).",
        "Host (aside),",
        "Host (aside);",
        "Host (aside):",
        "Host (aside)!",
        "Host (aside)?",
        "Host (aside)…",
        "Host (aside)—",
    ],
)
def test_stronger_punctuation_suppresses_close(text: str) -> None:
    boundaries = detect_parenthetical_boundaries(text)

    assert [boundary.kind for boundary in boundaries] == ["parenthetical_open"]


def test_leading_aside_has_only_resumed_host_close() -> None:
    text = "(Aside) Main text."

    boundaries = detect_parenthetical_boundaries(text)

    assert [(boundary.text, boundary.kind) for boundary in boundaries] == [
        (")", "parenthetical_close"),
    ]
    assert boundaries[0].char_start == text.index(")")


@pytest.mark.parametrize(
    "text",
    [
        "The optional word(s) are shown.",
        "Call function(x) before returning.",
        "Use array(2) for this case.",
    ],
)
def test_attached_lexical_forms_are_ignored(text: str) -> None:
    assert detect_parenthetical_boundaries(text) == []


def test_empty_and_unmatched_pairs_are_ignored() -> None:
    assert detect_parenthetical_boundaries("Something () happened.") == []
    assert detect_parenthetical_boundaries("Something (   ) happened.") == []
    assert detect_parenthetical_boundaries("Something (unfinished.") == []
    assert detect_parenthetical_boundaries("Something) happened.") == []


def test_nested_parentheses_emit_only_outer_pair() -> None:
    text = "Host (outer (inner) material) resumes."

    boundaries = detect_parenthetical_boundaries(text)

    assert _kinds(text) == ["parenthetical_open", "parenthetical_close"]
    assert [boundary.char_start for boundary in boundaries] == [
        text.index("("),
        text.rindex(")"),
    ]


def test_enumeration_markers_remain_detectable() -> None:
    text = "I would (a) fail, and (b) get dismantled."

    boundaries = detect_parenthetical_boundaries(text)

    assert [(boundary.text, boundary.kind) for boundary in boundaries] == [
        ("(", "parenthetical_open"),
        (")", "parenthetical_close"),
        ("(", "parenthetical_open"),
        (")", "parenthetical_close"),
    ]


def test_multiple_asides_and_unicode_preserve_source_offsets() -> None:
    text = "Das war falsch (völlig unerwartet) und wurde korrigiert (später) und endet."

    boundaries = detect_parenthetical_boundaries(text, language="de")

    assert _kinds(text) == [
        "parenthetical_open",
        "parenthetical_close",
        "parenthetical_open",
        "parenthetical_close",
    ]
    assert all(text[b.char_start : b.char_end] == b.text for b in boundaries)
    assert [b.char_start for b in boundaries] == sorted(
        b.char_start for b in boundaries
    )


def test_language_is_optional_and_no_nlp_is_needed() -> None:
    text = "Host (aside) resumes."

    assert detect_parenthetical_boundaries(text) == detect_parenthetical_boundaries(
        text, language=None
    )


def test_boundary_invariants_hold_for_all_results() -> None:
    text = "Host (first) resumes (second) and ends."

    boundaries = detect_parenthetical_boundaries(text)

    assert all(0 <= b.char_start < b.char_end <= len(text) for b in boundaries)
    assert all(b.text in {"(", ")"} for b in boundaries)
    assert all(
        b.kind in {"parenthetical_open", "parenthetical_close"} for b in boundaries
    )
    assert len({(b.char_start, b.kind) for b in boundaries}) == len(boundaries)
