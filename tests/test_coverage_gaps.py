"""Tests for previously uncovered branches across the phrasplit codebase.

This module deliberately exercises:

* validation/error branches of small helpers (``types``, ``utils``, CLI);
* spaCy-dependent code paths through a deterministic *fake* spaCy backend so
  they are covered even in CI environments without spaCy installed;
* fallback/except paths of the regex splitter.

Everything here is deterministic: no real spaCy models are required.
"""

from __future__ import annotations

import importlib
import re
import runpy
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import phrasplit
import phrasplit.cli as cli
import phrasplit.spacy_models as models
import phrasplit.splitter as splitter
import phrasplit.splitter_without_spacy as simple
from phrasplit.abbreviations import get_abbreviations
from phrasplit.spacy_models import (
    _distribution_model_names as _real_distribution_model_names,
)
from phrasplit.types import SplitSegment
from phrasplit.utils import suggest_splitting_mode, validate_no_placeholder_breaks

# ---------------------------------------------------------------------------
# Fake spaCy backend helpers
# ---------------------------------------------------------------------------


class FakeSent:
    """Fake spaCy sentence span: only the attributes splitter uses."""

    def __init__(self, text: str, start_char: int, end_char: int) -> None:
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class FakeDoc:
    """Fake spaCy Doc exposing ``sents``."""

    def __init__(self, sents: list[FakeSent]) -> None:
        self.sents = sents


class SpaceSplitNlp:
    """Fake spaCy pipeline: treats every space-delimited word as a sentence.

    Enough structure for ``_process_long_text``,
    ``_process_long_text_with_offsets`` and friends, without any real NLP.
    """

    def __init__(self, max_length: int = 1_000_000) -> None:
        self.max_length = max_length

    def __call__(self, text: str) -> FakeDoc:
        sents: list[FakeSent] = []
        start = 0
        for i, char in enumerate(text):
            if char == " ":
                sents.append(FakeSent(text[start:i], start, i))
                start = i + 1
        sents.append(FakeSent(text[start:], start, len(text)))
        return FakeDoc(sents)


class FakeSpacy:
    """Fake ``spacy`` package for resolver + loader integration tests."""

    def __init__(self, installed: list[str] | None = None) -> None:
        self.installed = installed if installed is not None else ["en_core_web_sm"]
        self.util = SimpleNamespace(get_installed_models=lambda: self.installed)
        self.nlp = SpaceSplitNlp()
        self.loaded: list[str] = []

    def load(self, name: str) -> SpaceSplitNlp:
        self.loaded.append(name)
        return self.nlp


def install_fake_spacy(monkeypatch: pytest.MonkeyPatch, fake: FakeSpacy) -> None:
    """Wire the fake into both the resolver and splitter._get_nlp."""
    monkeypatch.setattr(models, "_get_spacy", lambda: fake)
    real_import = importlib.import_module

    def fake_import(name: str) -> object:
        if name == "spacy":
            return fake
        return real_import(name)

    monkeypatch.setattr(
        splitter, "importlib", SimpleNamespace(import_module=fake_import)
    )


@pytest.fixture(autouse=True)
def _clean_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate model caches and distribution discovery for every test."""
    models.clear_spacy_model_cache()
    monkeypatch.setattr(models, "_distribution_model_names", lambda: set())
    monkeypatch.setattr(splitter, "_nlp_cache", {})
    yield
    models.clear_spacy_model_cache()


def _seg(
    text: str,
    char_start: int,
    char_end: int,
    *,
    sentence_idx: int = 0,
    clause_idx: int | None = None,
) -> SplitSegment:
    return SplitSegment(
        id=f"p0s{sentence_idx}",
        text=text,
        char_start=char_start,
        char_end=char_end,
        paragraph_idx=0,
        sentence_idx=sentence_idx,
        clause_idx=clause_idx,
    )


# ---------------------------------------------------------------------------
# phrasplit/types.py
# ---------------------------------------------------------------------------


class TestSplitSegmentValidation:
    def test_negative_paragraph_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="paragraph_idx"):
            SplitSegment(
                id="p0s0",
                text="x",
                char_start=0,
                char_end=1,
                paragraph_idx=-1,
                sentence_idx=0,
            )

    def test_negative_sentence_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="sentence_idx"):
            SplitSegment(
                id="p0s0",
                text="x",
                char_start=0,
                char_end=1,
                paragraph_idx=0,
                sentence_idx=-1,
            )

    def test_negative_clause_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="clause_idx"):
            SplitSegment(
                id="p0s0",
                text="x",
                char_start=0,
                char_end=1,
                paragraph_idx=0,
                sentence_idx=0,
                clause_idx=-1,
            )


# ---------------------------------------------------------------------------
# phrasplit/utils.py
# ---------------------------------------------------------------------------


class TestValidateNoPlaceholderBreaks:
    def test_no_placeholders_returns_empty(self) -> None:
        assert (
            validate_no_placeholder_breaks(
                "Plain text without markup.", [], placeholder_pattern=r"\{\{[^}]+\}\}"
            )
            == []
        )

    def test_placeholder_in_gap_between_segments_warns(self) -> None:
        text = "Hello  {{name}}  world"
        segments = [
            _seg("Hello  ", 0, 7),
            _seg("world", 16, 21, sentence_idx=1),
        ]
        warnings_list = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=r"\{\{[^}]+\}\}"
        )
        assert len(warnings_list) == 1
        assert "not contained in any segment" in warnings_list[0]


class TestSuggestSplittingMode:
    def test_no_placeholders_suggests_sentence(self) -> None:
        assert (
            suggest_splitting_mode(
                "Just plain text.", placeholder_pattern=r"\{\{[^}]+\}\}"
            )
            == "sentence"
        )

    def test_many_placeholders_suggest_paragraph(self) -> None:
        text = " ".join(f"{{{{v{i}}}}}" for i in range(7))
        assert (
            suggest_splitting_mode(text, placeholder_pattern=r"\{\{[^}]+\}\}")
            == "paragraph"
        )

    def test_moderate_placeholders_suggest_sentence(self) -> None:
        assert (
            suggest_splitting_mode(
                "{{a}} {{b}} {{c}}", placeholder_pattern=r"\{\{[^}]+\}\}"
            )
            == "sentence"
        )


# ---------------------------------------------------------------------------
# phrasplit/__init__.py
# ---------------------------------------------------------------------------


class TestVersionFallback:
    def test_version_fallback_when_version_module_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "phrasplit._version", None)
        importlib.reload(phrasplit)
        assert phrasplit.__version__ == "0.0.0"
        # Restore the real version module and reload to reset package state.
        sys.modules.pop("phrasplit._version", None)
        importlib.reload(phrasplit)
        assert phrasplit.__version__ != "0.0.0"


# ---------------------------------------------------------------------------
# phrasplit/abbreviations.py
# ---------------------------------------------------------------------------


class TestAbbreviations:
    def test_get_abbreviations_without_language_returns_empty(self) -> None:
        assert get_abbreviations(language=None) == set()


# ---------------------------------------------------------------------------
# phrasplit/cli.py
# ---------------------------------------------------------------------------


class TestModelDiagnostic:
    def test_verbose_simple_backend_message(self, capsys) -> None:
        cli._print_model_diagnostic(language="en", simple=True, verbose=True)
        assert "regex backend" in capsys.readouterr().err

    def test_verbose_spacy_backend_message(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(splitter, "LAST_SPACY_MODEL", "en_core_web_sm")
        cli._print_model_diagnostic(language="en", simple=False, verbose=True)
        assert "spaCy model" in capsys.readouterr().err

    def test_not_verbose_prints_nothing(self, capsys) -> None:
        cli._print_model_diagnostic(language="en", simple=False, verbose=False)
        assert capsys.readouterr().err == ""


class TestWriteOutputPlain:
    def test_print_plain_output(self, capsys) -> None:
        cli.write_output("hello", None, use_rich=False)
        assert capsys.readouterr().out == "hello\n"


class TestCliErrorPaths:
    @pytest.mark.parametrize("command", ["clauses", "longlines"])
    def test_value_error_handled(self, monkeypatch, command: str) -> None:
        def boom(*args, **kwargs) -> list[str]:
            raise ValueError("boom")

        target = "split_clauses" if command == "clauses" else "split_long_lines"
        monkeypatch.setattr(cli, target, boom)
        result = CliRunner().invoke(cli.main, [command], input="Some text.")
        assert result.exit_code == 1
        assert "boom" in result.output

    @pytest.mark.parametrize("command", ["clauses", "longlines"])
    def test_import_error_tips_printed(self, monkeypatch, command: str) -> None:
        def boom(*args, **kwargs) -> list[str]:
            raise ImportError("spacy missing")

        target = "split_clauses" if command == "clauses" else "split_long_lines"
        monkeypatch.setattr(cli, target, boom)
        result = CliRunner().invoke(cli.main, [command], input="Some text.")
        assert result.exit_code == 1
        assert "--simple" in result.output
        assert "pip install phrasplit" in result.output


class TestCliEntrypoint:
    def test_main_entrypoint_guard(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit) as exc:
                runpy.run_module("phrasplit.cli", run_name="__main__")
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# phrasplit/spacy_models.py
# ---------------------------------------------------------------------------


class TestResolutionProperties:
    def test_property_aliases(self) -> None:
        resolution = models.SpacyModelResolution(
            language="en",
            model="en_core_web_sm",
            model_size="sm",
            requested_model=None,
            requested_size=None,
            candidates=("en_core_web_sm",),
            attempts=(models.SpacyModelAttempt("en_core_web_sm", True),),
            available=True,
            loadable=True,
            diagnostics=(),
        )
        assert resolution.selected_model == "en_core_web_sm"
        assert resolution.selected_package == "en_core_web_sm"
        assert resolution.size == "sm"


class TestNormalizationAndInference:
    def test_invalid_language_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid spaCy language code"):
            models.normalize_spacy_language("english")

    def test_infer_spacy_language_edge_cases(self) -> None:
        assert models.infer_spacy_language(None) is None
        assert models.infer_spacy_language("not-a-model") is None
        assert models.infer_spacy_language("en_core_web_sm") == "en"

    def test_infer_spacy_model_size_none(self) -> None:
        assert models.infer_spacy_model_size(None) is None


class TestDistributionDiscovery:
    def test_distribution_model_names_collects_metadata(self, monkeypatch) -> None:
        monkeypatch.setattr(
            models,
            "_distribution_model_names",
            _real_distribution_model_names,
        )
        fake_distributions = [SimpleNamespace(metadata={"Name": "En-Core-Web-Sm"})]
        monkeypatch.setattr(
            models.importlib.metadata,
            "distributions",
            lambda: fake_distributions,
        )
        assert models._distribution_model_names() == {"en_core_web_sm"}

    def test_distribution_model_names_handles_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            models,
            "_distribution_model_names",
            _real_distribution_model_names,
        )

        def boom() -> list[object]:
            raise RuntimeError("metadata unavailable")

        monkeypatch.setattr(models.importlib.metadata, "distributions", boom)
        assert models._distribution_model_names() == set()

    def test_installed_models_ignores_utility_failure(self, monkeypatch) -> None:
        def boom() -> list[str]:
            raise RuntimeError("utility failed")

        fake = SimpleNamespace(
            util=SimpleNamespace(get_installed_models=boom),
            load=lambda name: SimpleNamespace(name=name),
        )
        monkeypatch.setattr(models, "_get_spacy", lambda: fake)
        monkeypatch.setattr(
            models, "_distribution_model_names", lambda: {"en_core_web_sm"}
        )
        assert models.installed_spacy_models() == ("en_core_web_sm",)


class TestResolverBranches:
    def test_resolve_auto_without_spacy_optional(self, monkeypatch) -> None:
        def missing_spacy() -> object:
            raise models.SpacyNotInstalledError("missing")

        monkeypatch.setattr(models, "_get_spacy", missing_spacy)
        resolution = models.resolve_spacy_model(
            language="en", model="auto", require=False
        )
        assert resolution.model is None
        assert resolution.requested_model is None

    def test_no_compatible_model_error_includes_attempt_details(
        self, monkeypatch
    ) -> None:
        fake = FakeSpacy(installed=["en_core_web_sm"])
        monkeypatch.setattr(
            fake, "load", lambda name: (_ for _ in ()).throw(OSError("cannot load"))
        )
        install_fake_spacy(monkeypatch, fake)
        with pytest.raises(models.NoCompatibleSpacyModelError, match="Tried"):
            models.resolve_spacy_model(language="en", require=True)

    def test_no_compatible_model_error_without_candidates(self, monkeypatch) -> None:
        fake = FakeSpacy(installed=[])
        install_fake_spacy(monkeypatch, fake)
        with pytest.raises(models.NoCompatibleSpacyModelError, match="No compatible"):
            models.resolve_spacy_model(language="en", require=True)


# ---------------------------------------------------------------------------
# phrasplit/splitter_without_spacy.py
# ---------------------------------------------------------------------------


class TestLanguagePatternBuilderBranches:
    def test_prefixes_empty_when_only_lowercase_abbreviations(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(simple, "get_abbreviations", lambda *a, **k: {"etc", "inc"})
        patterns = simple._build_language_patterns()
        assert not patterns["prefixes"].search("Dr. Smith")
        assert not patterns["suffixes"].search("Apple Inc.")

    def test_suffixes_empty_when_only_prefixes(self, monkeypatch) -> None:
        monkeypatch.setattr(simple, "get_abbreviations", lambda *a, **k: {"Mr", "Dr"})
        patterns = simple._build_language_patterns()
        assert patterns["prefixes"].search("Dr. Smith")
        assert not patterns["suffixes"].search("Apple Inc.")


class TestSimpleSentenceSplitterErrors:
    def test_non_string_input_raises(self) -> None:
        with pytest.raises(ValueError, match="Input must be a string"):
            simple.split_sentences_simple(123)

    def test_regex_error_falls_back_to_whole_text(self, monkeypatch) -> None:
        def bad_corrections(*args, **kwargs) -> list[str]:
            raise re.error("bad pattern")

        monkeypatch.setattr(splitter, "_apply_corrections", bad_corrections)
        with pytest.warns(UserWarning, match="Regex error"):
            result = simple.split_sentences_simple("Hello. World.")
        assert len(result) == 1

    def test_unexpected_error_returns_empty(self, monkeypatch) -> None:
        def bad_corrections(*args, **kwargs) -> list[str]:
            raise RuntimeError("boom")

        monkeypatch.setattr(splitter, "_apply_corrections", bad_corrections)
        with pytest.warns(UserWarning, match="Unexpected error"):
            result = simple.split_sentences_simple("Hello. World.")
        assert result == []


class TestSplitAtClausesSimple:
    def test_trailing_empty_part_skipped(self) -> None:
        assert simple._split_at_clauses_simple("Hello, world, ", max_length=80) == [
            "Hello, world,"
        ]


# ---------------------------------------------------------------------------
# phrasplit/splitter.py — small helper branches
# ---------------------------------------------------------------------------


class TestHelperBranches:
    def test_first_cased_char_returns_none(self) -> None:
        assert splitter._first_cased_char("123!?") is None

    def test_extract_leading_word_returns_none(self) -> None:
        assert splitter._extract_leading_word("   ") is None
        assert splitter._extract_leading_word("(((") is None

    def test_is_sentence_start_mixed_punctuation(self) -> None:
        assert splitter._is_sentence_start("«»(Hello", 0) is True
        assert splitter._is_sentence_start("»„x", 0) is False


class TestGetNlp:
    def test_import_error_raised_when_spacy_missing(self, monkeypatch) -> None:
        def bad_import(name: str) -> object:
            raise ImportError("no spacy")

        monkeypatch.setattr(
            splitter, "importlib", SimpleNamespace(import_module=bad_import)
        )
        with pytest.raises(ImportError, match="spaCy is required"):
            splitter._get_nlp("en_core_web_sm")

    def test_oserror_when_model_load_fails(self, monkeypatch) -> None:
        fake = FakeSpacy(installed=[])
        monkeypatch.setattr(
            fake, "load", lambda name: (_ for _ in ()).throw(OSError("not found"))
        )
        install_fake_spacy(monkeypatch, fake)
        with pytest.raises(OSError, match="not found"):
            splitter._get_nlp("en_core_web_sm")

    def test_loads_and_caches_model(self, monkeypatch) -> None:
        fake = FakeSpacy(installed=["en_core_web_sm"])
        install_fake_spacy(monkeypatch, fake)
        nlp1 = splitter._get_nlp("en_core_web_sm")
        nlp2 = splitter._get_nlp("en_core_web_sm")
        assert nlp1 is nlp2
        assert fake.loaded == ["en_core_web_sm"]


class TestProcessLongTextWithOffsets:
    def test_short_text_single_doc(self) -> None:
        nlp = SpaceSplitNlp()
        assert splitter._process_long_text_with_offsets("Hello world.", nlp) == [
            (0, 5),
            (6, 12),
        ]

    def test_long_text_chunked_with_complete_sentences(self) -> None:
        nlp = SpaceSplitNlp()
        text = "one two three four five six seven eight nine ten"
        offsets = splitter._process_long_text_with_offsets(
            text, nlp, max_chunk=8, safety_margin=2
        )
        assert offsets == [
            (0, 3),
            (4, 7),
            (8, 13),
            (14, 18),
            (19, 23),
            (24, 27),
            (28, 33),
            (34, 39),
            (40, 44),
            (45, 48),
        ]
        for start, end in offsets:
            assert text[start:end] and not text[start:end].isspace()

    def test_long_text_no_complete_sentence_falls_back(self) -> None:
        nlp = SpaceSplitNlp()
        assert splitter._process_long_text_with_offsets(
            "abcdefghij", nlp, max_chunk=5, safety_margin=2
        ) == [(0, 5), (5, 10)]


class TestSplitSentencesSpacy:
    def test_with_fake_nlp(self, monkeypatch) -> None:
        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())
        assert splitter._split_sentences_spacy(
            "Hello world. Goodbye world.", "en_core_web_sm"
        ) == ["Hello", "world.", "Goodbye", "world."]

    def test_empty_text_returns_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())
        assert splitter._split_sentences_spacy("", "en_core_web_sm") == []


class TestSplitClausesSpacy:
    def test_with_fake_nlp(self, monkeypatch) -> None:
        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())
        assert splitter._split_clauses_spacy("One, two. Three.", "en_core_web_sm") == [
            "One,",
            "two.",
            "Three.",
        ]


class TestSplitAtBoundaries:
    def test_accumulates_and_flushes(self) -> None:
        nlp = SpaceSplitNlp()
        text = (
            "A short one. A much longer sentence, with clauses, that exceeds the "
            "limit. Tiny. Also short."
        )
        result = splitter._split_at_boundaries(text, max_length=10, nlp=nlp)
        assert " ".join(result) == text

    def test_long_sentence_splits_into_clauses(self) -> None:
        nlp = SpaceSplitNlp()
        result = splitter._split_at_boundaries(
            "A. Supercalifragilistic. B.", max_length=5, nlp=nlp
        )
        assert result == ["A.", "Supercalifragilistic.", "B."]

    def test_empty_result_returns_original_text(self) -> None:
        nlp = SpaceSplitNlp()
        assert splitter._split_at_boundaries("", max_length=10, nlp=nlp) == [""]


class TestSplitLongLinesSpacy:
    def test_invalid_max_length_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())
        with pytest.raises(ValueError, match="max_length"):
            splitter._split_long_lines_spacy("x", 0, "en_core_web_sm")

    def test_splits_long_lines(self, monkeypatch) -> None:
        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())
        result = splitter._split_long_lines_spacy(
            "short line\nA much longer line that needs splitting.", 10, "en_core_web_sm"
        )
        assert result == [
            "short line",
            "A much",
            "longer",
            "line that",
            "needs",
            "splitting.",
        ]


class TestValidationAndTrimming:
    def test_validate_offset_segments_out_of_bounds(self) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            splitter._validate_offset_segments("short", [_seg("x", 0, 100)])

    def test_validate_offset_segments_text_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            splitter._validate_offset_segments("hello", [_seg("WRONG", 0, 5)])

    def test_validate_offset_segments_overlap(self) -> None:
        segments = [
            _seg("he", 0, 2),
            _seg("ell", 1, 4, sentence_idx=1),
        ]
        with pytest.raises(ValueError, match="overlap"):
            splitter._validate_offset_segments("hello", segments)

    def test_trim_segment_bounds_all_whitespace(self) -> None:
        assert splitter._trim_segment_bounds("   ", 0, 3) is None
        assert splitter._trim_segment_bounds("  a  ", 0, 5) == (2, 3)


class TestMergeAbbreviationSplitsWithOffsets:
    def test_single_segment_returned_unchanged(self) -> None:
        segments = [("Hi.", 0, 3)]
        assert (
            splitter._merge_abbreviation_splits_with_offsets(
                "Hi.", segments, language="en"
            )
            == segments
        )

    def test_no_abbreviations_returned_unchanged(self, monkeypatch) -> None:
        monkeypatch.setattr(splitter, "get_abbreviations", lambda *a, **k: set())
        segments = [("A.", 0, 2), ("B.", 3, 5)]
        assert (
            splitter._merge_abbreviation_splits_with_offsets(
                "A. B.", segments, language="en"
            )
            == segments
        )

    def test_no_merge_tail_path(self) -> None:
        segments = [("Hello world.", 0, 12), ("Next.", 13, 18)]
        assert (
            splitter._merge_abbreviation_splits_with_offsets(
                "Hello world. Next.", segments, language="en"
            )
            == segments
        )


class TestOffsetSplitHelpers:
    def test_split_after_ellipsis_with_offsets_empty(self) -> None:
        assert splitter._split_after_ellipsis_with_offsets("x", []) == []

    def test_split_urls_with_offsets_single_url_kept(self) -> None:
        text = "Visit https://example.com now."
        segment = (text, 0, len(text))
        assert splitter._split_urls_with_offsets(text, [segment]) == [segment]

    def test_split_after_url_boundaries_with_offsets(self) -> None:
        text = "See https://a.b.Next. Then."
        result = splitter._split_after_url_boundaries_with_offsets(
            text, [(text, 0, len(text))]
        )
        assert result == [("See https://a.b.Next.", 0, 21), ("Then.", 22, 27)]

    def test_dotted_abbreviation_boundaries_with_offsets(self) -> None:
        text = "U.S. However we left."
        result = splitter._split_after_dotted_abbreviation_boundaries_with_offsets(
            text, [(text, 0, len(text))]
        )
        assert result == [("U.S.", 0, 4), ("However we left.", 5, len(text))]


class TestSimpleSentenceSplitPreservingOffsets:
    def test_empty_text(self) -> None:
        assert splitter._simple_sentence_split_preserving_offsets("") == []

    def test_trailing_whitespace_adjusted(self) -> None:
        assert splitter._simple_sentence_split_preserving_offsets(
            "Hello world. X  "
        ) == [("Hello world.", 0, 12), ("X", 13, 14)]


class TestSimpleSentenceSplitWithMarkupOffsets:
    def test_empty_text(self) -> None:
        assert splitter._simple_sentence_split_with_markup_offsets("") == []

    def test_text_ending_in_opening_tag(self) -> None:
        assert splitter._simple_sentence_split_with_markup_offsets("Hello. <em>") == [
            ("Hello. <em>", 0, 11)
        ]

    def test_next_visible_not_sentence_start(self) -> None:
        assert splitter._simple_sentence_split_with_markup_offsets(
            "Hello. <em>world</em>"
        ) == [("Hello. <em>world</em>", 0, 21)]

    def test_sentence_ending_abbreviation_not_blocked(self) -> None:
        assert splitter._simple_sentence_split_with_markup_offsets(
            "Apple Inc. <em>Grows</em>"
        ) == [("Apple Inc.", 0, 10), ("<em>Grows</em>", 11, 25)]

    def test_abbreviation_blocks_boundary(self) -> None:
        assert splitter._simple_sentence_split_with_markup_offsets(
            "Dr. <em>Smith</em> left."
        ) == [("Dr. <em>Smith</em> left.", 0, 24)]

    def test_dotted_acronym_with_sentence_starter_splits(self) -> None:
        assert splitter._simple_sentence_split_with_markup_offsets(
            "U.S. However <em>we</em> left."
        ) == [("U.S.", 0, 4), ("However <em>we</em> left.", 5, 30)]

    def test_final_segment_trailing_whitespace_adjusted(self) -> None:
        assert splitter._simple_sentence_split_with_markup_offsets(
            "Hello. World  "
        ) == [("Hello.", 0, 6), ("World", 7, 12)]


class TestSplitWithOffsetsRegexBranches:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            splitter._split_with_offsets_regex("x", mode="invalid")

    def test_clause_find_fallback_uses_search_start(self, monkeypatch) -> None:
        monkeypatch.setattr(
            splitter, "_split_sentence_into_clauses", lambda s: ["not-a-substring"]
        )
        result = splitter._split_with_offsets_regex("Hello. World.", mode="clause")
        assert result[0].text == "not-a-substring"
        assert result[0].char_start == 0


class TestApplyMaxCharsSplit:
    def test_short_segments_passthrough(self) -> None:
        segments = [_seg("Hi.", 0, 3)]
        assert (
            splitter._apply_max_chars_split("Hi.", segments, max_chars=10) == segments
        )

    def test_leading_whitespace_skipped_and_whitespace_chunk_skipped(self) -> None:
        text = "  aaaa bbbb  "
        segments = [_seg(text, 0, len(text))]
        result = splitter._apply_max_chars_split(text, segments, max_chars=4)
        assert [s.text for s in result] == ["aaaa", "bbbb"]
        assert all(text[s.char_start : s.char_end] == s.text for s in result)

    def test_whitespace_only_segment_breaks(self) -> None:
        segments = [_seg("      ", 0, 6)]
        assert splitter._apply_max_chars_split("      ", segments, max_chars=2) == []


class TestSplitWithOffsetsValidation:
    def test_invalid_max_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            splitter.split_with_offsets("x", max_chars=0)


# ---------------------------------------------------------------------------
# phrasplit/splitter.py — spaCy backend through the public API
# ---------------------------------------------------------------------------


class TestSpacyBackendPublicApi:
    def test_split_clauses_use_spacy_true(self, monkeypatch) -> None:
        fake = FakeSpacy(installed=["en_core_web_sm"])
        install_fake_spacy(monkeypatch, fake)
        assert splitter.split_clauses("One, two. Three.", use_spacy=True) == [
            "One,",
            "two.",
            "Three.",
        ]

    def test_split_long_lines_use_spacy_true(self, monkeypatch) -> None:
        fake = FakeSpacy(installed=["en_core_web_sm"])
        install_fake_spacy(monkeypatch, fake)
        result = splitter.split_long_lines(
            "This is a fairly long line that should be split into pieces.",
            10,
            use_spacy=True,
        )
        assert result
        assert all(isinstance(line, str) and line for line in result)

    def test_split_text_spacy_sentence_and_clause_modes(self, monkeypatch) -> None:
        fake = FakeSpacy(installed=["en_core_web_sm"])
        install_fake_spacy(monkeypatch, fake)
        text = "Hello there. How are you?\n\nNew paragraph."

        sentences = splitter.split_text(text, mode="sentence", use_spacy=True)
        assert [s.text for s in sentences] == [
            "Hello",
            "there.",
            "How",
            "are",
            "you?",
            "New",
            "paragraph.",
        ]
        assert [s.paragraph for s in sentences] == [0, 0, 0, 0, 0, 1, 1]

        clauses = splitter.split_text(text, mode="clause", use_spacy=True)
        assert clauses
        assert all(c.paragraph is not None for c in clauses)

    def test_split_with_offsets_spacy_modes(self, monkeypatch) -> None:
        fake = FakeSpacy(installed=["en_core_web_sm"])
        install_fake_spacy(monkeypatch, fake)
        text = "Hello there. How are you?\n\nNew paragraph."

        # Paragraph mode
        paragraphs = splitter.split_with_offsets(text, mode="paragraph", use_spacy=True)
        assert [p.id for p in paragraphs] == ["p0s0", "p1s0"]
        assert all(text[p.char_start : p.char_end] == p.text for p in paragraphs)

        # Sentence mode with max_chars splitting
        sentences = splitter.split_with_offsets(
            text, mode="sentence", use_spacy=True, max_chars=4
        )
        assert sentences
        assert all(text[s.char_start : s.char_end] == s.text for s in sentences)

        # Clause mode
        clauses = splitter.split_with_offsets(text, mode="clause", use_spacy=True)
        assert clauses
        assert all(c.clause_idx is not None for c in clauses)
        assert all(text[c.char_start : c.char_end] == c.text for c in clauses)
        assert all(text[c.char_start : c.char_end] == c.text for c in clauses)


class TestSplitClausesSpacyEmpty:
    def test_empty_text_returns_empty(self, monkeypatch) -> None:

        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())

        assert splitter._split_clauses_spacy("", "en_core_web_sm") == []


class TestSplitAtClausesMain:
    def test_trailing_empty_part_skipped(self) -> None:

        assert splitter._split_at_clauses("Hello, world, ", max_length=80) == [
            "Hello, world,"
        ]


class TestSplitWithOffsetsSpacyDirect:
    def test_invalid_mode_raises(self, monkeypatch) -> None:

        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())

        with pytest.raises(ValueError, match="mode must be"):
            splitter._split_with_offsets_spacy("x", "en_core_web_sm", mode="invalid")

    def test_paragraph_mode_whitespace_and_empty_paragraphs(self, monkeypatch) -> None:

        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())

        text = "\n\n  A.  \n\n\n\n  B.  "

        result = splitter._split_with_offsets_spacy(
            text, "en_core_web_sm", mode="paragraph"
        )

        assert [s.text for s in result] == ["A.", "B."]

        assert [s.id for s in result] == ["p0s0", "p1s0"]

        assert all(text[s.char_start : s.char_end] == s.text for s in result)

    def test_clause_find_fallback_uses_search_start(self, monkeypatch) -> None:

        monkeypatch.setattr(splitter, "_get_nlp", lambda model: SpaceSplitNlp())

        monkeypatch.setattr(
            splitter, "_split_sentence_into_clauses", lambda s: ["missing-clause"]
        )

        result = splitter._split_with_offsets_spacy(
            "Hello there.", "en_core_web_sm", mode="clause"
        )

        assert result[0].text == "missing-clause"

        assert result[0].char_start == 0


class TestNormalizationDefaults:
    def test_empty_or_none_language_defaults_to_en(self) -> None:

        assert models.normalize_spacy_language(None) == "en"

        assert models.normalize_spacy_language("") == "en"

        assert models.normalize_spacy_language("   ") == "en"


class TestOptionalResolutionNoCandidates:
    def test_optional_resolution_without_candidates_returns_empty(
        self, monkeypatch
    ) -> None:

        fake = FakeSpacy(installed=[])

        install_fake_spacy(monkeypatch, fake)

        resolution = models.resolve_spacy_model(language="en", require=False)

        assert resolution.model is None

        assert resolution.loadable is False
