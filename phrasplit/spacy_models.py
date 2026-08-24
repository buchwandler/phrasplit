"""Local discovery and selection of official spaCy language models.

The resolver deliberately performs inspection only.  It never downloads a model
or contacts a package index.  This keeps automatic model selection safe for
library and command-line callers alike.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

SpacyModelSize = Literal["sm", "md", "lg", "trf"]

_MODEL_SIZES: tuple[SpacyModelSize, ...] = ("trf", "lg", "md", "sm")
_MODEL_PATTERN = re.compile(
    r"^(?P<language>[a-z]{2,3})_core_(?P<family>web|news)_(?P<size>sm|md|lg|trf)$"
)
_LANGUAGE_ALIASES = {
    "cmn": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "zh-tw": "zh",
    "zh_tw": "zh",
}
_LOADED_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class SpacyModelAttempt:
    """Outcome of trying to inspect one candidate model."""

    model: str
    loadable: bool
    error: str | None = None


@dataclass(frozen=True)
class SpacyModelResolution:
    """Structured result returned by :func:`resolve_spacy_model`.

    ``model`` is the concrete package selected for loading, or ``None`` when
    optional resolution could not find a usable model.  ``candidates`` and
    ``attempts`` make the local decision auditable without exposing spaCy
    internals to callers.
    """

    language: str
    model: str | None
    model_size: SpacyModelSize | None
    requested_model: str | None
    requested_size: SpacyModelSize | None
    candidates: tuple[str, ...]
    attempts: tuple[SpacyModelAttempt, ...]
    available: bool
    loadable: bool
    diagnostics: tuple[str, ...]

    @property
    def selected_model(self) -> str | None:
        """Alias used by callers that prefer an explicit result name."""

        return self.model

    @property
    def selected_package(self) -> str | None:
        """Alias for the concrete selected distribution/package name."""

        return self.model

    @property
    def size(self) -> SpacyModelSize | None:
        """Tier of the selected model, if one was selected."""

        return self.model_size


class SpacyModelResolutionError(RuntimeError):
    """Base error for required model resolution failures."""

    def __init__(self, message: str, *, resolution: SpacyModelResolution | None = None):
        super().__init__(message)
        self.resolution = resolution


class SpacyNotInstalledError(ImportError, SpacyModelResolutionError):
    """Raised when required resolution is requested without spaCy."""


class NoCompatibleSpacyModelError(SpacyModelResolutionError):
    """Raised when no installed compatible model can be loaded."""


class ExplicitSpacyModelError(SpacyModelResolutionError):
    """Raised when an explicitly requested package cannot be loaded."""


def normalize_spacy_language(language: str | None) -> str:
    """Return a normalized spaCy language code.

    Locale separators are normalized and common aliases are collapsed so that
    ``en-US``, ``en_US`` and ``en`` all select the English family, while spaCy's
    ``cmn`` code selects the official Chinese ``zh`` model family.
    """

    if language is None or not str(language).strip():
        return "en"
    normalized = str(language).strip().lower().replace("_", "-")
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized)
    normalized = normalized.split("-", 1)[0]
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized)
    if not re.fullmatch(r"[a-z]{2,3}", normalized):
        raise ValueError(f"Invalid spaCy language code: {language!r}")
    return normalized


def infer_spacy_language(model: str | None) -> str | None:
    """Infer a normalized language from an official model package name."""

    if not model:
        return None
    match = _MODEL_PATTERN.fullmatch(model.strip().lower().replace("-", "_"))
    if not match:
        return None
    return normalize_spacy_language(match.group("language"))


def infer_spacy_model_size(model: str | None) -> SpacyModelSize | None:
    """Infer a model tier from an official package name."""

    if not model:
        return None
    match = _MODEL_PATTERN.fullmatch(model.strip().lower().replace("-", "_"))
    return match.group("size") if match else None  # type: ignore[return-value]


def official_model_prefixes(language: str | None) -> tuple[str, ...]:
    """Return official package-family prefixes for a normalized language."""

    normalized = normalize_spacy_language(language)
    # English uses web; the other official core pipelines generally use news.
    # Both are accepted during discovery because spaCy has shipped both family
    # forms over time and the installed package is the source of truth.
    preferred = "web" if normalized == "en" else "news"
    alternate = "news" if preferred == "web" else "web"
    return (f"{normalized}_core_{preferred}", f"{normalized}_core_{alternate}")


def _get_spacy() -> Any:
    try:
        return importlib.import_module("spacy")
    except ImportError as exc:
        raise SpacyNotInstalledError(
            "spaCy is not installed. Install with: pip install phrasplit[nlp]."
        ) from exc


def get_cached_spacy_model(model: str) -> Any | None:
    """Return a model loaded during resolution, if available."""

    return _LOADED_MODEL_CACHE.get(model)


def clear_spacy_model_cache() -> None:
    """Clear loaded-model and installed-model discovery caches."""
    _LOADED_MODEL_CACHE.clear()
    installed_spacy_models.cache_clear()

def _load_model(spacy: Any, model: str) -> Any:
    cached = _LOADED_MODEL_CACHE.get(model)
    if cached is not None:
        return cached
    loaded = spacy.load(model)
    _LOADED_MODEL_CACHE[model] = loaded
    return loaded


def _distribution_model_names() -> set[str]:
    names: set[str] = set()
    try:
        distributions = importlib.metadata.distributions()
    except Exception:  # noqa: BLE001 - discovery must degrade gracefully
        distributions = ()
    for distribution in distributions:
        name = distribution.metadata.get("Name")
        if name:
            names.add(name.lower().replace("-", "_"))
    return names


@lru_cache(maxsize=1)
def installed_spacy_models() -> tuple[str, ...]:
    """Enumerate locally installed spaCy model package names.

    spaCy's utility is preferred when present, with Python distribution metadata
    as a compatibility fallback for spaCy versions that do not expose it.
    """

    spacy = _get_spacy()
    names: set[str] = set()
    util = getattr(spacy, "util", None)
    get_installed = getattr(util, "get_installed_models", None)
    if callable(get_installed):
        try:
            names.update(
                str(name).lower().replace("-", "_") for name in get_installed()
            )
        except Exception:  # noqa: BLE001, S110 - utility failures are optional
            pass
    names.update(_distribution_model_names())
    return tuple(sorted(names))


def _compatible_candidates(language: str, installed: tuple[str, ...]) -> list[str]:
    prefixes = official_model_prefixes(language)
    prefix_set = set(prefixes)
    candidates: list[str] = []
    for name in installed:
        normalized = name.strip().lower().replace("-", "_")
        match = _MODEL_PATTERN.fullmatch(normalized)
        if (
            match
            and f"{match.group('language')}_core_{match.group('family')}" in prefix_set
        ):
            candidates.append(normalized)
    return candidates


def _rank_candidates(candidates: list[str], size: SpacyModelSize | None) -> list[str]:
    allowed = {size} if size else set(_MODEL_SIZES)
    return sorted(
        {
            candidate
            for candidate in candidates
            if infer_spacy_model_size(candidate) in allowed
        },
        key=lambda candidate: (
            _MODEL_SIZES.index(infer_spacy_model_size(candidate)),  # type: ignore[arg-type]
            candidate,
        ),
    )


def _empty_resolution(
    *,
    language: str,
    model: str | None,
    size: SpacyModelSize | None,
    candidates: tuple[str, ...] = (),
    attempts: tuple[SpacyModelAttempt, ...] = (),
    diagnostics: tuple[str, ...] = (),
    available: bool | None = None,
) -> SpacyModelResolution:
    return SpacyModelResolution(
        language=language,
        model=None,
        model_size=None,
        requested_model=model,
        requested_size=size,
        candidates=candidates,
        attempts=attempts,
        available=bool(candidates) if available is None else available,
        loadable=False,
        diagnostics=diagnostics,
    )


def resolve_spacy_model(
    *,
    language: str | None,
    model: str | None = None,
    size: SpacyModelSize | None = None,
    require: bool = True,
) -> SpacyModelResolution:
    """Resolve an installed and loadable official spaCy model.

    ``model`` is an exact override.  ``None``, an empty string, and ``"auto"``
    activate automatic discovery.  ``size`` is an exact tier filter and never
    falls back to a different tier.
    """

    normalized_language = normalize_spacy_language(language)
    requested_model = model.strip() if isinstance(model, str) else model
    if requested_model and requested_model.lower() == "auto":
        requested_model = None
    if size is not None and size not in _MODEL_SIZES:
        raise ValueError(f"Invalid spaCy model size: {size!r}")
    if requested_model and size is not None:
        warnings.warn(
            "model_size is ignored when an explicit model is supplied.",
            UserWarning,
            stacklevel=2,
        )

    try:
        spacy = _get_spacy()
    except SpacyNotInstalledError:
        if require or requested_model:
            raise
        return _empty_resolution(
            language=normalized_language,
            model=None,
            size=size,
            diagnostics=("spaCy is not installed",),
        )

    if requested_model:
        candidate = requested_model
        installed = set(installed_spacy_models())
        candidate_available = candidate.lower().replace("-", "_") in installed
        failure = "is missing" if not candidate_available else "failed to load"
        try:
            _load_model(spacy, candidate)
        except Exception as exc:
            resolution = _empty_resolution(
                language=normalized_language,
                model=candidate,
                size=size,
                candidates=(candidate,),
                attempts=(SpacyModelAttempt(candidate, False, str(exc)),),
                diagnostics=(
                    (
                        f"explicit model '{candidate}' "
                        f"{failure}"
                    ),
                ),
                available=candidate_available,
            )
            status = (
                "is missing or not installed"
                if not candidate_available
                else "is installed but failed to load"
            )
            raise ExplicitSpacyModelError(
                f"Explicit spaCy model '{candidate}' {status}: {exc}",
                resolution=resolution,
            ) from exc
        return SpacyModelResolution(
            language=normalized_language,
            model=candidate,
            model_size=infer_spacy_model_size(candidate),
            requested_model=candidate,
            requested_size=size,
            candidates=(candidate,),
            attempts=(SpacyModelAttempt(candidate, True),),
            available=True,
            loadable=True,
            diagnostics=(f"selected explicit model '{candidate}'",),
        )

    installed = installed_spacy_models()
    candidates = tuple(
        _rank_candidates(_compatible_candidates(normalized_language, installed), size)
    )
    attempts: list[SpacyModelAttempt] = []
    for candidate in candidates:
        try:
            _load_model(spacy, candidate)
        except Exception as exc:  # noqa: BLE001 - model loaders expose varied errors
            attempts.append(SpacyModelAttempt(candidate, False, str(exc)))
            continue
        return SpacyModelResolution(
            language=normalized_language,
            model=candidate,
            model_size=infer_spacy_model_size(candidate),
            requested_model=None,
            requested_size=size,
            candidates=candidates,
            attempts=tuple(attempts + [SpacyModelAttempt(candidate, True)]),
            available=True,
            loadable=True,
            diagnostics=(f"selected highest available model '{candidate}'",),
        )

    resolution = _empty_resolution(
        language=normalized_language,
        model=None,
        size=size,
        candidates=candidates,
        attempts=tuple(attempts),
        diagnostics=(
            (
                "no loadable official spaCy model found for language "
                f"'{normalized_language}'"
            ),
        ),
    )
    if require:
        if candidates:
            detail = "; ".join(
                f"{attempt.model}: {attempt.error}"
                for attempt in attempts
                if attempt.error
            )
            raise NoCompatibleSpacyModelError(
                f"No compatible loadable spaCy model is installed for language "
                f"'{normalized_language}'. Tried: {detail or ', '.join(candidates)}",
                resolution=resolution,
            )
        raise NoCompatibleSpacyModelError(
            "No compatible spaCy model is installed for language "
            f"'{normalized_language}'.",
            resolution=resolution,
        )
    return resolution


__all__ = [
    "ExplicitSpacyModelError",
    "NoCompatibleSpacyModelError",
    "SpacyModelAttempt",
    "SpacyModelResolution",
    "SpacyModelResolutionError",
    "SpacyModelSize",
    "SpacyNotInstalledError",
    "clear_spacy_model_cache",
    "get_cached_spacy_model",
    "infer_spacy_language",
    "infer_spacy_model_size",
    "installed_spacy_models",
    "normalize_spacy_language",
    "official_model_prefixes",
    "resolve_spacy_model",
]
