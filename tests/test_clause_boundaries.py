"""Controlled tests for syntactic clausal-comma boundary detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from phrasplit import ClauseBoundary, detect_clause_boundaries, splitter
from phrasplit.spacy_models import SpacyModelResolution


@dataclass
class FakeMorph:
    values: dict[str, list[str]]

    def get(self, key: str) -> list[str]:
        return self.values.get(key, [])


class FakeToken:
    def __init__(self, text: str, idx: int, i: int) -> None:
        self.text = text
        self.idx = idx
        self.i = i
        self.dep_ = ""
        self.pos_ = ""
        self.tag_ = ""
        self.morph = FakeMorph({})
        self.head: FakeToken = self


@dataclass
class FakeSpan:
    text: str
    start_char: int
    end_char: int


class FakeDoc:
    def __init__(self, text: str, tokens: list[FakeToken]) -> None:
        self.text = text
        self.tokens = tokens
        self.sents = [FakeSpan(text, 0, len(text))]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.tokens)


class SentenceOnlyDoc:
    def __init__(self, text: str, tokens: list[Any]) -> None:
        self.text = text
        self.sents = [FakeSpan(text, 0, len(text))]
        self.tokens = tokens

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.tokens)


def _tokenize(text: str) -> list[FakeToken]:
    return [
        FakeToken(match.group(), match.start(), index)
        for index, match in enumerate(re.finditer(r"\w+|[^\w\s]", text))
    ]


def _find_token(tokens: list[FakeToken], value: str, start: int = 0) -> FakeToken:
    value = value.lower()
    for token in tokens[start:]:
        if token.text.lower() == value:
            return token
    raise AssertionError(f"token {value!r} not found")


def fake_doc(
    text: str,
    clauses: list[tuple[str, str, str, str | None]],
) -> FakeDoc:
    """Build a document from (subject, predicate, form, finite auxiliary) arcs."""
    tokens = _tokenize(text)
    for subject_text, predicate_text, predicate_form, auxiliary_text in clauses:
        subject = _find_token(tokens, subject_text)
        predicate = _find_token(tokens, predicate_text, subject.i)
        subject.dep_ = "nsubj"
        subject.head = predicate
        predicate.pos_ = "VERB" if predicate_form != "Adj" else "ADJ"
        predicate.morph = FakeMorph({"VerbForm": [predicate_form]})
        if auxiliary_text is not None:
            auxiliary = _find_token(tokens, auxiliary_text, subject.i)
            auxiliary.dep_ = "aux" if predicate_form != "Adj" else "cop"
            auxiliary.head = predicate
            auxiliary.pos_ = "AUX"
            auxiliary.tag_ = "VBD"
            auxiliary.morph = FakeMorph({"VerbForm": ["Fin"]})
    return FakeDoc(text, tokens)


def test_clause_boundary_dataclass_serializes_and_validates() -> None:
    boundary = ClauseBoundary(",", 3, 4, "clausal_comma", {"method": "spacy"})
    assert ClauseBoundary.from_dict(boundary.to_dict()) == boundary
    with pytest.raises(ValueError):
        ClauseBoundary(",", -1, 0, "clausal_comma")
    with pytest.raises(ValueError):
        ClauseBoundary("", 0, 1, "clausal_comma")
    with pytest.raises(ValueError):
        ClauseBoundary(",", 0, 1, "other")  # type: ignore[arg-type]


def test_motivating_regression_has_one_exact_boundary() -> None:
    text = (
        "It had picked up the sound of a explosion, direction suggested it was behind."
    )
    doc = fake_doc(
        text, [("It", "picked", "Part", "had"), ("direction", "suggested", "Fin", None)]
    )
    boundaries = detect_clause_boundaries(text, doc=doc)
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary.text == ","
    assert boundary.char_start == text.index(",")
    assert boundary.text == text[boundary.char_start : boundary.char_end]
    assert boundary.kind == "clausal_comma"
    assert boundary.meta == {
        "method": "spacy",
        "rule": "explicit-subject-finite-clause-v1",
    }


@pytest.mark.parametrize(
    ("text", "clauses"),
    [
        (
            "I opened the door, the room was empty.",
            [("I", "opened", "Fin", None), ("room", "empty", "Part", "was")],
        ),
        (
            "I left, but she stayed.",
            [("I", "left", "Fin", None), ("she", "stayed", "Fin", None)],
        ),
        (
            "When I arrived, the room was empty.",
            [("I", "arrived", "Fin", None), ("room", "empty", "Part", "was")],
        ),
        (
            "It had stopped, the alarm started.",
            [("It", "stopped", "Part", "had"), ("alarm", "started", "Fin", None)],
        ),
        (
            "The room was empty, the hallway was dark.",
            [("room", "empty", "Part", "was"), ("hallway", "dark", "Part", "was")],
        ),
    ],
)
def test_positive_clause_shapes(
    text: str, clauses: list[tuple[str, str, str, str | None]]
) -> None:
    doc = fake_doc(text, clauses)
    boundaries = detect_clause_boundaries(text, doc=doc)
    assert [boundary.text for boundary in boundaries] == [","]
    assert [text[b.char_start : b.char_end] for b in boundaries] == [","]


def test_quoted_clause_is_detected() -> None:
    text = 'He said, "I am here."'
    doc = fake_doc(text, [("He", "said", "Fin", None), ("I", "here", "Part", "am")])
    assert len(detect_clause_boundaries(text, doc=doc)) == 1


@pytest.mark.parametrize(
    ("text", "clauses"),
    [
        (
            "It picked up sound, light, smoke, and debris.",
            [("It", "picked", "Fin", None)],
        ),
        (
            "I opened the door, looked inside, and walked away.",
            [("I", "opened", "Fin", None)],
        ),
        ("In 2026, we started.", [("we", "started", "Fin", None)]),
        ("Hello, John.", []),
        ("The result is, in fact, correct.", [("result", "is", "Fin", None)]),
        ("The man, who was tired, left.", [("who", "tired", "Part", "was")]),
        ("I bought apples, oranges grown locally, and pears.", []),
    ],
)
def test_non_clause_commas_are_not_reported(
    text: str, clauses: list[tuple[str, str, str, str | None]]
) -> None:
    assert detect_clause_boundaries(text, doc=fake_doc(text, clauses)) == []


def test_source_order_and_uniqueness() -> None:
    text = "I left, she stayed, they waited."
    doc = fake_doc(
        text,
        [
            ("I", "left", "Fin", None),
            ("she", "stayed", "Fin", None),
            ("they", "waited", "Fin", None),
        ],
    )
    doc.sents = [FakeSpan(text, 0, len(text)), FakeSpan(text, 0, len(text))]
    boundaries = detect_clause_boundaries(text, doc=doc)
    assert [boundary.char_start for boundary in boundaries] == [
        text.index(","),
        text.index(",") + 12,
    ]
    assert len({boundary.char_start for boundary in boundaries}) == 2


def test_mismatched_doc_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        detect_clause_boundaries("other", doc=fake_doc("source", []))


def test_supplied_doc_bypasses_resolution_and_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "I left, she stayed."
    doc = fake_doc(text, [("I", "left", "Fin", None), ("she", "stayed", "Fin", None)])
    monkeypatch.setattr(splitter, "resolve_spacy_model", pytest.fail)
    monkeypatch.setattr(splitter, "_get_nlp", pytest.fail)
    assert len(detect_clause_boundaries(text, doc=doc)) == 1


def test_supplied_nlp_is_called_once() -> None:
    text = "I left, she stayed."
    doc = fake_doc(text, [("I", "left", "Fin", None), ("she", "stayed", "Fin", None)])
    calls: list[str] = []

    def nlp(value: str) -> FakeDoc:
        calls.append(value)
        return doc

    assert len(detect_clause_boundaries(text, nlp=nlp)) == 1
    assert calls == [text]


def test_regex_and_auto_without_model_do_not_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "I left, she stayed."
    assert detect_clause_boundaries(text, use_spacy=False) == []
    resolution = SpacyModelResolution(
        language="en",
        model=None,
        model_size=None,
        requested_model=None,
        requested_size=None,
        candidates=(),
        attempts=(),
        available=False,
        loadable=False,
        diagnostics=(),
    )
    monkeypatch.setattr(splitter, "resolve_spacy_model", lambda **_: resolution)
    assert detect_clause_boundaries(text) == []


def test_forced_spacy_preserves_resolver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RuntimeError("model unavailable")
    monkeypatch.setattr(
        splitter, "resolve_spacy_model", lambda **_: (_ for _ in ()).throw(error)
    )
    with pytest.raises(RuntimeError, match="model unavailable"):
        detect_clause_boundaries("I left, she stayed.", use_spacy=True)


def test_sentence_only_document_returns_empty() -> None:
    text = "I left, she stayed."
    tokens = [
        SimpleNamespace(text=value, idx=index)
        for index, value in enumerate(text.split())
    ]
    doc = SentenceOnlyDoc(text, tokens)
    assert detect_clause_boundaries(text, doc=doc) == []
