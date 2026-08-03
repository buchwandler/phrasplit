"""Phrasplit - Split text into sentences, clauses, or paragraphs."""

from .abbreviations import get_abbreviations
from .spacy_models import (
    SpacyModelResolution,
    SpacyModelSize,
    normalize_spacy_language,
    resolve_spacy_model,
)
from .splitter import (
    Segment,
    split_clauses,
    split_long_lines,
    split_paragraphs,
    split_sentences,
    split_text,
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
    "__version__",
    "COMMON_PATTERNS",
    "Segment",
    "SplitSegment",
    "SpacyModelResolution",
    "SpacyModelSize",
    "get_abbreviations",
    "split_clauses",
    "split_long_lines",
    "split_paragraphs",
    "split_sentences",
    "split_text",
    "split_with_offsets",
    "iter_split_with_offsets",
    "normalize_spacy_language",
    "resolve_spacy_model",
    "validate_no_placeholder_breaks",
    "suggest_splitting_mode",
]
