"""Mocked tests for local spaCy model discovery and backend selection."""

from types import SimpleNamespace

import pytest

import phrasplit.spacy_models as models
import phrasplit.splitter as splitter
from phrasplit import split_sentences


class FakeSpacy:
    def __init__(
        self, installed: list[str], unloadable: set[str] | None = None
    ) -> None:
        self.util = SimpleNamespace(get_installed_models=lambda: installed)
        self.unloadable = unloadable or set()
        self.loaded: list[str] = []

    def load(self, name: str) -> object:
        self.loaded.append(name)
        if name in self.unloadable:
            raise OSError(f"cannot load {name}")
        return SimpleNamespace(name=name)


@pytest.fixture(autouse=True)
def clear_model_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    models.clear_spacy_model_cache()
    monkeypatch.setattr(models, "_distribution_model_names", lambda: set())
    yield
    models.clear_spacy_model_cache()


def install_fake_spacy(monkeypatch: pytest.MonkeyPatch, fake: FakeSpacy) -> None:
    monkeypatch.setattr(models, "_get_spacy", lambda: fake)


def test_highest_available_model_is_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSpacy(["en_core_web_sm", "en_core_web_lg"])
    install_fake_spacy(monkeypatch, fake)

    resolution = models.resolve_spacy_model(language="en", require=True)

    assert resolution.model == "en_core_web_lg"
    assert resolution.model_size == "lg"
    assert resolution.candidates == ("en_core_web_lg", "en_core_web_sm")
    assert fake.loaded == ["en_core_web_lg"]


def test_all_tiers_rank_trf_first_and_partial_families_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSpacy(
        ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]
    )
    install_fake_spacy(monkeypatch, fake)

    assert (
        models.resolve_spacy_model(language="en", require=True).model
        == "en_core_web_trf"
    )


def test_exact_model_and_exact_size_are_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSpacy(["en_core_web_sm", "en_core_web_lg"])
    install_fake_spacy(monkeypatch, fake)

    assert (
        models.resolve_spacy_model(
            language="en", model="en_core_web_sm", require=True
        ).model
        == "en_core_web_sm"
    )
    with pytest.raises(models.NoCompatibleSpacyModelError):
        models.resolve_spacy_model(language="en", size="md", require=True)


def test_language_normalization_and_language_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSpacy(["en_core_web_sm", "de_core_news_md", "zh_core_web_sm"])
    install_fake_spacy(monkeypatch, fake)

    assert models.normalize_spacy_language("en-US") == "en"
    assert models.normalize_spacy_language("en_US") == "en"
    assert models.normalize_spacy_language("cmn") == "zh"
    assert (
        models.resolve_spacy_model(language="de", require=True).model
        == "de_core_news_md"
    )
    assert (
        models.resolve_spacy_model(language="en", require=True).model
        == "en_core_web_sm"
    )


def test_unloadable_highest_candidate_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSpacy(
        ["en_core_web_sm", "en_core_web_lg"], unloadable={"en_core_web_lg"}
    )
    install_fake_spacy(monkeypatch, fake)

    resolution = models.resolve_spacy_model(language="en", require=True)

    assert resolution.model == "en_core_web_sm"
    assert [attempt.model for attempt in resolution.attempts] == [
        "en_core_web_lg",
        "en_core_web_sm",
    ]
    assert resolution.attempts[0].loadable is False


def test_explicit_unloadable_model_fails_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSpacy(["en_core_web_sm"], unloadable={"en_core_web_sm"})
    install_fake_spacy(monkeypatch, fake)

    with pytest.raises(models.ExplicitSpacyModelError, match="failed to load|missing"):
        models.resolve_spacy_model(language="en", model="en_core_web_sm", require=True)
    assert fake.loaded == ["en_core_web_sm"]


def test_optional_resolution_without_spacy_returns_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_spacy() -> object:
        raise models.SpacyNotInstalledError("missing")

    monkeypatch.setattr(models, "_get_spacy", missing_spacy)
    resolution = models.resolve_spacy_model(language="en", require=False)
    assert resolution.model is None
    assert resolution.loadable is False


def test_use_spacy_false_does_not_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_resolution(**_: object) -> object:
        raise AssertionError("regex mode must not invoke model resolution")

    monkeypatch.setattr(splitter, "resolve_spacy_model", fail_resolution)
    assert split_sentences("Dr. Smith arrived.", use_spacy=False) == [
        "Dr. Smith arrived."
    ]


def test_auto_resolution_runs_once_and_passes_concrete_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    resolution = models.SpacyModelResolution(
        language="en",
        model="en_core_web_lg",
        model_size="lg",
        requested_model=None,
        requested_size=None,
        candidates=("en_core_web_lg",),
        attempts=(models.SpacyModelAttempt("en_core_web_lg", True),),
        available=True,
        loadable=True,
        diagnostics=(),
    )

    def fake_resolve(**kwargs: object) -> models.SpacyModelResolution:
        calls.append(kwargs)
        return resolution

    monkeypatch.setattr(splitter, "resolve_spacy_model", fake_resolve)
    monkeypatch.setattr(
        splitter,
        "_split_sentences_spacy",
        lambda text, model, *args, **kwargs: [model],
    )

    assert split_sentences("ignored", language="en") == ["en_core_web_lg"]
    assert len(calls) == 1
    assert calls[0]["require"] is False


def test_abbreviations_fall_back_to_language() -> None:
    from phrasplit.abbreviations import get_abbreviations

    assert "Dr" in get_abbreviations(language="en-US")
    assert "Dr" in get_abbreviations("en_core_web_future")
    assert get_abbreviations("auto", language="de")
