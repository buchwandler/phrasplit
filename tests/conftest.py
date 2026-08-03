"""Shared backend fixtures and markers for deterministic test environments."""

from __future__ import annotations

import pytest


def _find_english_model() -> str | None:
    """Return an installed official English model, if one is loadable by name."""

    try:
        from phrasplit.spacy_models import installed_spacy_models

        candidates = set(installed_spacy_models())
    except (ImportError, RuntimeError):
        return None

    for model in (
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
        "en_core_web_trf",
    ):
        if model in candidates:
            return model
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "spacy_model: requires an installed compatible spaCy language model",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    model_available = _find_english_model() is not None
    reason = "requires an installed official English spaCy model"
    for item in items:
        if item.get_closest_marker("spacy_model") and not model_available:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def spacy_model_name() -> str:
    """Provide a stable official English model name for integration tests."""

    model = _find_english_model()
    if model is None:
        pytest.skip("requires an installed official English spaCy model")
    return model


@pytest.fixture(scope="session")
def require_spacy_model(spacy_model_name: str) -> str:
    """Alias fixture whose name makes a test's model dependency explicit."""

    return spacy_model_name
