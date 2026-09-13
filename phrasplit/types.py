"""Data types for phrasplit segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ClauseBoundary:
    """A high-confidence syntactic clause boundary in the original text."""

    text: str
    char_start: int
    char_end: int
    kind: Literal["clausal_comma"]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the boundary invariants."""
        if self.char_start < 0:
            raise ValueError(f"char_start must be >= 0, got {self.char_start}")
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be greater than "
                f"char_start ({self.char_start})"
            )
        if not self.text:
            raise ValueError("text must not be empty")
        if self.kind != "clausal_comma":
            raise ValueError(f"unsupported boundary kind: {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        """Convert the boundary to a JSON-serializable dictionary."""
        return {
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "kind": self.kind,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClauseBoundary:
        """Create a boundary from a dictionary."""
        return cls(
            text=data["text"],
            char_start=data["char_start"],
            char_end=data["char_end"],
            kind=data["kind"],
            meta=data.get("meta", {}),
        )


@dataclass
class SplitSegment:
    """A text segment with character offsets and hierarchical indices.

    This dataclass represents a segment of text with precise character offsets
    in the original input string, along with hierarchical paragraph/sentence/clause
    indices and a stable identifier.

    Attributes:
        id: Stable identifier in format "p{paragraph}s{sentence}c{clause}"
            Examples: "p0s0", "p0s1c2", "p2s3"
        text: The text content of the segment (exact slice from input)
        char_start: Character offset where segment starts in original text (0-based)
        char_end: Character offset where segment ends in original text (exclusive)
        paragraph_idx: Paragraph index (0-based)
        sentence_idx: Sentence index within paragraph (0-based)
        clause_idx: Clause index within sentence (0-based), or None if not
            splitting clauses
        meta: Additional metadata (e.g., {"method": "spacy", "confidence": 0.95})

    Example:
        >>> segment = SplitSegment(
        ...     id="p0s1",
        ...     text="This is a sentence.",
        ...     char_start=20,
        ...     char_end=39,
        ...     paragraph_idx=0,
        ...     sentence_idx=1,
        ...     clause_idx=None,
        ...     meta={"method": "spacy"}
        ... )
        >>> # Verify offset mapping
        >>> original_text[segment.char_start:segment.char_end] == segment.text
        True

    Note:
        - Offsets (char_start, char_end) are guaranteed to map exactly to the original
          input string, preserving whitespace and formatting
        - IDs are stable: same input + settings = same IDs
        - All fields are JSON-serializable for easy storage/transmission
    """

    id: str
    text: str
    char_start: int
    char_end: int
    paragraph_idx: int
    sentence_idx: int
    clause_idx: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate segment data after initialization."""
        if self.char_start < 0:
            raise ValueError(f"char_start must be >= 0, got {self.char_start}")
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be >= char_start ({self.char_start})"
            )
        if self.paragraph_idx < 0:
            raise ValueError(f"paragraph_idx must be >= 0, got {self.paragraph_idx}")
        if self.sentence_idx < 0:
            raise ValueError(f"sentence_idx must be >= 0, got {self.sentence_idx}")
        if self.clause_idx is not None and self.clause_idx < 0:
            raise ValueError(f"clause_idx must be >= 0 or None, got {self.clause_idx}")

    def to_dict(self) -> dict[str, Any]:
        """Convert segment to a JSON-serializable dictionary.

        Returns:
            Dictionary with all segment fields
        """
        return {
            "id": self.id,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "paragraph_idx": self.paragraph_idx,
            "sentence_idx": self.sentence_idx,
            "clause_idx": self.clause_idx,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitSegment:
        """Create a segment from a dictionary.

        Args:
            data: Dictionary with segment fields

        Returns:
            New SplitSegment instance
        """
        return cls(
            id=data["id"],
            text=data["text"],
            char_start=data["char_start"],
            char_end=data["char_end"],
            paragraph_idx=data["paragraph_idx"],
            sentence_idx=data["sentence_idx"],
            clause_idx=data.get("clause_idx"),
            meta=data.get("meta", {}),
        )
