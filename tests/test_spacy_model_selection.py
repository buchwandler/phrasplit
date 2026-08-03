"""Mocked tests for local spaCy model discovery and backend selection."""

from types import SimpleNamespace

import pytest

import phrasplit.spacy_models as models
import phrasplit.splitter as splitter
from phrasplit import split_sentences


class FakeSpacy:
    def __init__(
        self,
        installed: list[str],
        unloadable: set[str] | None = None,
        import_fail: set[str] | None = None,
    ) -> None:
        self.util = SimpleNamespace(get_installed_models=lambda: installed)
        self.unloadable = unloadable or set()
        self.import_fail = import_fail or set()
        self.loaded: list[str] = []

    def load(self, name: str) -> object:
        self.loaded.append(name)
        if name in self.unloadable:
            raise OSError(f"cannot load {name}")
        if name in self.import_fail:
            raise ImportError(f"dependency failed while loading {name}")
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
    with pytest.raises(models.ExplicitSpacyModelError) as caught:
        models.resolve_spacy_model(language="en", model="en_core_web_sm", require=True)
    assert caught.value.resolution is not None
    assert caught.value.resolution.available is True
    assert caught.value.resolution.loadable is False


def test_explicit_missing_model_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSpacy([])
    install_fake_spacy(monkeypatch, fake)
    monkeypatch.setattr(
        models,
        "_load_model",
        lambda _spacy, _model: (_ for _ in ()).throw(OSError("missing")),
    )

    with pytest.raises(models.ExplicitSpacyModelError) as caught:
        models.resolve_spacy_model(
            language="en", model="definitely_missing_model", require=True
        )
    assert caught.value.resolution is not None
    assert caught.value.resolution.available is False
    assert caught.value.resolution.loadable is False


def test_loader_import_error_is_explicit_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSpacy(["en_core_web_sm"], import_fail={"en_core_web_sm"})
    install_fake_spacy(monkeypatch, fake)

    with pytest.raises(
        models.ExplicitSpacyModelError, match="dependency failed"
    ) as caught:
        models.resolve_spacy_model(language="en", model="en_core_web_sm", require=True)
    assert isinstance(caught.value.__cause__, ImportError)


def test_distribution_metadata_is_used_without_spacy_utility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(
        util=SimpleNamespace(), load=lambda name: SimpleNamespace(name=name)
    )
    monkeypatch.setattr(models, "_get_spacy", lambda: fake)
    monkeypatch.setattr(models, "_distribution_model_names", lambda: {"en_core_web_sm"})
    assert (
        models.resolve_spacy_model(language="en", require=True).model
        == "en_core_web_sm"
    )


def test_invalid_language_and_size_are_rejected() -> None:
    with pytest.raises(ValueError, match="language"):
        models.resolve_spacy_model(language="english!!")
    with pytest.raises(ValueError, match="size"):
        models.resolve_spacy_model(language="en", size="xl")  # type: ignore[arg-type]


def test_model_size_warning_for_explicit_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSpacy(["en_core_web_sm"])
    install_fake_spacy(monkeypatch, fake)
    with pytest.warns(UserWarning, match="ignored"):
        resolution = models.resolve_spacy_model(
            language="en", model="en_core_web_sm", size="lg", require=True
        )
    assert resolution.model == "en_core_web_sm"


def test_model_cache_reuse_and_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSpacy(["en_core_web_sm"])
    install_fake_spacy(monkeypatch, fake)
    models.resolve_spacy_model(language="en", model="en_core_web_sm", require=True)
    models.resolve_spacy_model(language="en", model="en_core_web_sm", require=True)
    assert fake.loaded == ["en_core_web_sm"]
    assert models.get_cached_spacy_model("en_core_web_sm") is not None
    models.clear_spacy_model_cache()
    assert models.get_cached_spacy_model("en_core_web_sm") is None


def test_public_resolver_types_are_exported() -> None:
    import phrasplit

    assert phrasplit.SpacyModelAttempt is models.SpacyModelAttempt
    assert phrasplit.ExplicitSpacyModelError is models.ExplicitSpacyModelError


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
