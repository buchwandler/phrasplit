"""Phrasplit - Split text into sentences, clauses, or paragraphs."""

from .abbreviations import get_abbreviations
from .spacy_models import (
    ExplicitSpacyModelError,
    NoCompatibleSpacyModelError,
    SpacyModelAttempt,
    SpacyModelResolutionError,
    SpacyModelResolution,
    SpacyModelSize,
    SpacyNotInstalledError,
    normalize_spacy_language,
    resolve_spacy_model,
)
from .splitter import (
    AnalyzedDocument,
    AnalyzedSpan,
    AnalyzedToken,
    Segment,
    SplitDiagnostics,
    SplitTextResult,
    SplitWithOffsetsResult,
    split_clauses,
    split_long_lines,
    split_paragraphs,
    split_sentences,
    split_text,
    split_text_with_diagnostics,
    split_with_offsets_with_diagnostics,
    split_with_offsets,
    iter_split_with_offsets,
)
from .types import SplitSegment
from .utils import (
    COMMON_PATTERNS,
    validate_no_placeholder_breaks,
    suggest_splitting_mode,
)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "COMMON_PATTERNS",
    "AnalyzedDocument",
    "AnalyzedSpan",
    "AnalyzedToken",
    "ExplicitSpacyModelError",
    "NoCompatibleSpacyModelError",
    "Segment",
    "SpacyModelAttempt",
    "SpacyModelResolution",
    "SpacyModelResolutionError",
    "SpacyModelSize",
    "SpacyNotInstalledError",
    "SplitDiagnostics",
    "SplitSegment",
    "SplitTextResult",
    "SplitWithOffsetsResult",
    "__version__",
    "get_abbreviations",
    "iter_split_with_offsets",
    "normalize_spacy_language",
    "resolve_spacy_model",
    "split_clauses",
    "split_long_lines",
    "split_paragraphs",
    "split_sentences",
    "split_text",
    "split_text_with_diagnostics",
    "split_with_offsets",
    "split_with_offsets_with_diagnostics",
    "suggest_splitting_mode",
    "validate_no_placeholder_breaks",
]
