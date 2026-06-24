"""Text splitting utilities using spaCy for NLP-based sentence and clause detection.

This module provides two implementations:
1. spaCy-based (default when available): High accuracy, handles complex cases
2. Regex-based (fallback): Faster, simpler, good for common cases

The implementation is selected automatically based on spaCy availability,
or can be controlled via the use_spacy parameter.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

from phrasplit.abbreviations import (
    get_abbreviations,
    get_sentence_ending_abbreviations,
    get_sentence_starters,
)
from phrasplit.types import SplitSegment


class Segment(NamedTuple):
    """A text segment with position information.

    Attributes:
        text: The text content of the segment
        paragraph: Paragraph index (0-based) within the document
        sentence: Sentence index (0-based) within the paragraph.
            None for paragraph mode.
    """

    text: str
    paragraph: int
    sentence: int | None = None


if TYPE_CHECKING:
    from spacy.language import Language  # type: ignore[import-not-found]

try:
    import spacy  # type: ignore[import-not-found]

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


# Cache for loaded spaCy model
_nlp_cache: dict[str, Language] = {}

# Placeholders for ellipsis during spaCy processing
# We use Unicode private use area characters to avoid collision with real text
_ELLIPSIS_3_PLACEHOLDER = "\ue000"  # 3 dots: ...
_ELLIPSIS_4_PLACEHOLDER = "\ue001"  # 4 dots: ....
_ELLIPSIS_SPACED_PLACEHOLDER = "\ue002"  # Spaced: . . .
_ELLIPSIS_UNICODE_PLACEHOLDER = "\ue003"  # Unicode ellipsis: …
_ELLIPSIS_LONG_PREFIX = "\ue004"  # Placeholder for 5+ dots (repeat per dot)

# Regex for hyphenated line breaks (e.g., "recom-\nmendation" -> "recommendation")
_HYPHENATED_LINEBREAK = re.compile(r"(\w+)-\s*\n\s*(\w+)")

# URL pattern for splitting
_URL_PATTERN = re.compile(
    r"(?:https?://[^\s\)\]\}\>\"',]+|www\.[^\s\)\]\}\>\"',]+|"
    r"\b[\w.-]+\.[A-Za-z]{2,}(?:/[^\s\)\]\}\>\"',]*)?)"
)

# Pattern to detect abbreviation at end of sentence
_LETTER_PATTERN = r"[^\W\d_]"
_ABBREV_END_PATTERN = re.compile(
    rf"(\b{_LETTER_PATTERN}+(?:\.{_LETTER_PATTERN}+)*)\.\s*$"
)

_URL_TRAILING_PUNCT = ".,)]}>\"'"
_OPENING_PUNCTUATION = "\"'([{«„“”‘’"
_CLOSING_PUNCTUATION = "\"')]}»”’"

_ELLIPSIS_PATTERN = re.compile(r"(?:\.{3,}|(?:\.\s){2,}\.|…)")

# Inline markup (XHTML) tag patterns for the opt-in inline_markup split mode.
# Conservative: an ordinary '<' comparison (e.g. "a < b") does not match because
# an opening tag must start with a letter, and a closing tag must start with '</'.
_OPEN_TAG_PATTERN = re.compile(r"<[A-Za-z][^>]*>")
_CLOSE_TAG_PATTERN = re.compile(r"</[A-Za-z][^>]*>")

# Default maximum chunk size for spaCy processing (will be capped by nlp.max_length)
_DEFAULT_MAX_CHUNK_SIZE = 500000

# Safety margin at chunk boundaries to avoid cutting sentences
_DEFAULT_SAFETY_MARGIN = 100


def _fix_hyphenated_linebreaks(text: str) -> str:
    """
    Fix hyphenated line breaks commonly found in PDFs and OCR text.

    Joins words that were split across lines with a hyphen.
    Example: "recom-\\nmendation" -> "recommendation"

    Args:
        text: Input text

    Returns:
        Text with hyphenated line breaks fixed
    """
    return _HYPHENATED_LINEBREAK.sub(r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    """
    Normalize multiple whitespace characters to single spaces.

    Preserves paragraph breaks (double newlines) but normalizes
    other whitespace sequences.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # First preserve paragraph breaks by using a placeholder
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Normalize other whitespace (but not newlines in paragraph breaks)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text


def _preprocess_text(text: str) -> str:
    """
    Apply preprocessing steps to clean up text before NLP processing.

    Steps:
    1. Fix hyphenated line breaks (common in PDFs)
    2. Normalize whitespace

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    text = _fix_hyphenated_linebreaks(text)
    text = _normalize_whitespace(text)
    return text


def _protect_ellipsis(text: str) -> str:
    """
    Replace ellipsis patterns with placeholders to prevent sentence splitting.

    Handles:
    - Spaced ellipsis: ". . ." (dot-space-dot-space-dot)
    - Regular ellipsis: "..." (three consecutive dots)
    - Four dots: "...." (often used for ellipsis + period)
    - Five or more dots: "....." etc.
    - Unicode ellipsis: U+2026 (single ellipsis character)

    Each pattern is replaced with a unique placeholder that preserves information
    about the original format, allowing exact restoration later. Placeholders
    preserve length to keep offsets aligned with the original input.
    """

    # Replace spaced ellipsis first (. . .) - must come before regular dots
    spaced_placeholder = (
        f"{_ELLIPSIS_SPACED_PLACEHOLDER} "
        f"{_ELLIPSIS_SPACED_PLACEHOLDER} "
        f"{_ELLIPSIS_SPACED_PLACEHOLDER}"
    )
    text = text.replace(". . .", spaced_placeholder)

    # Replace unicode ellipsis
    text = text.replace("\u2026", _ELLIPSIS_UNICODE_PLACEHOLDER)

    # Replace longer dot sequences first (5+ dots), preserving length
    def replace_long_dots(match: re.Match[str]) -> str:
        count = len(match.group(0))
        return _ELLIPSIS_LONG_PREFIX * count

    text = re.sub(r"\.{5,}", replace_long_dots, text)

    # Replace 4 dots
    text = text.replace("....", _ELLIPSIS_4_PLACEHOLDER * 4)

    # Replace 3 dots (must come after 4+ to avoid partial matches)
    text = text.replace("...", _ELLIPSIS_3_PLACEHOLDER * 3)

    return text


def _restore_ellipsis(text: str) -> str:
    """Restore ellipsis placeholders back to their original format."""
    # Restore in reverse order of protection

    # Restore 3 dots
    text = text.replace(_ELLIPSIS_3_PLACEHOLDER * 3, "...")

    # Restore 4 dots
    text = text.replace(_ELLIPSIS_4_PLACEHOLDER * 4, "....")

    # Restore long dot sequences (5+)
    def restore_long_dots(match: re.Match[str]) -> str:
        return "." * len(match.group(0))

    long_pattern = re.compile(re.escape(_ELLIPSIS_LONG_PREFIX) + r"{5,}")
    text = long_pattern.sub(restore_long_dots, text)

    # Restore unicode ellipsis
    text = text.replace(_ELLIPSIS_UNICODE_PLACEHOLDER, "\u2026")

    # Restore spaced ellipsis
    text = text.replace(_ELLIPSIS_SPACED_PLACEHOLDER, ".")

    return text


def _first_cased_char(text: str) -> str | None:
    for char in text:
        if char.isalpha():
            return char
    return None


def _extract_leading_word(text: str) -> str | None:
    stripped = text.lstrip().lstrip(_OPENING_PUNCTUATION)
    if not stripped:
        return None
    first_word = stripped.split()[0]
    return first_word.strip(_CLOSING_PUNCTUATION)


def _skip_opening_tags(text: str, pos: int) -> int:
    """Return the index after a run of opening/empty tags starting at ``pos``."""
    while True:
        match = _OPEN_TAG_PATTERN.match(text, pos)
        if match is None:
            return pos
        pos = match.end()


def _skip_closing_tags(text: str, pos: int) -> int:
    """Return the index after a run of closing tags starting at ``pos``."""
    while True:
        match = _CLOSE_TAG_PATTERN.match(text, pos)
        if match is None:
            return pos
        pos = match.end()


def _is_sentence_start(text: str, start: int, *, allow_lowercase: bool = False) -> bool:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    while index < len(text) and text[index] in _OPENING_PUNCTUATION:
        index += 1
    while index < len(text):
        char = text[index]
        if char.isalpha():
            if char.isupper():
                return True
            return allow_lowercase and char.islower()
        if char.isdigit():
            return True
        if char.isspace() or char in _OPENING_PUNCTUATION:
            index += 1
            continue
        if char in _CLOSING_PUNCTUATION:
            index += 1
            continue
        break
    return False


def _find_ellipsis_split_positions(
    text: str, *, allow_lowercase: bool = False
) -> list[int]:
    positions: list[int] = []
    for match in _ELLIPSIS_PATTERN.finditer(text):
        boundary = match.end()
        while boundary < len(text) and text[boundary] in _CLOSING_PUNCTUATION:
            boundary += 1
        if _is_sentence_start(text, boundary, allow_lowercase=allow_lowercase):
            positions.append(boundary)
    return positions


def _trim_url_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while end > start and text[end - 1] in _URL_TRAILING_PUNCT:
        end -= 1
    return start, end


def _find_url_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _URL_PATTERN.finditer(text):
        start, end = _trim_url_bounds(text, match.start(), match.end())
        if end > start:
            spans.append((start, end))
    return spans


def _split_urls(sentences: list[str]) -> list[str]:
    """
    Split sentences that contain multiple URLs.

    URLs are often listed one per line in source text, but spaCy may merge them.
    This function splits sentences only when there are 2+ URLs present.

    Args:
        sentences: List of sentences from spaCy

    Returns:
        List of sentences with multiple URLs properly separated
    """
    result: list[str] = []

    for sent in sentences:
        url_spans = _find_url_spans(sent)

        if len(url_spans) < 2:
            result.append(sent)
            continue

        # Split at URL boundaries - each URL becomes its own "sentence"
        # along with any text that follows it until the next URL
        last_end = 0
        for i, (url_start, _) in enumerate(url_spans):
            # Text before this URL (only for first URL)
            if i == 0 and url_start > 0:
                prefix = sent[:url_start].strip()
                if prefix:
                    # Include prefix with first URL
                    next_url_start = (
                        url_spans[i + 1][0] if i + 1 < len(url_spans) else len(sent)
                    )
                    part = sent[:next_url_start].strip()
                    result.append(part)
                    last_end = next_url_start
                    continue

            # For subsequent URLs or if no prefix
            if url_start >= last_end:
                next_url_start = (
                    url_spans[i + 1][0] if i + 1 < len(url_spans) else len(sent)
                )
                part = sent[url_start:next_url_start].strip()
                if part:
                    result.append(part)
                last_end = next_url_start

    return result


def _find_url_sentence_split_positions(text: str) -> list[int]:
    positions: list[int] = []
    for _, url_end in _find_url_spans(text):
        boundary = url_end
        while boundary < len(text) and text[boundary] in _CLOSING_PUNCTUATION:
            boundary += 1
        if boundary < len(text) and text[boundary] in ".!?":
            boundary += 1
            if _is_sentence_start(text, boundary):
                positions.append(boundary)
    return sorted(set(positions))


def _split_after_url_boundaries(sentences: list[str]) -> list[str]:
    """Split sentences after URL boundaries when spaCy misses them."""
    result: list[str] = []
    for sent in sentences:
        split_positions = _find_url_sentence_split_positions(sent)
        if not split_positions:
            result.append(sent)
            continue

        start = 0
        for split_pos in split_positions:
            part = sent[start:split_pos].strip()
            if part:
                result.append(part)
            start = split_pos

        remaining = sent[start:].strip()
        if remaining:
            result.append(remaining)

    return result


def _merge_abbreviation_splits(
    sentences: list[str],
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Merge sentences that were incorrectly split after abbreviations.

    spaCy sometimes splits after abbreviations like "M.D." or "U.S." when
    followed by a name or continuation. This function merges such cases.

    Conservative approach: only merge if:
    1. Previous sentence ends with a known abbreviation + period
    2. The abbreviation is NOT one that commonly ends sentences (etc., Inc., etc.)
    3. Next sentence starts with a capital letter (likely a name/continuation)
    4. Next sentence does NOT start with a common sentence starter

    Args:
        sentences: List of sentences from spaCy
        language_model: spaCy language model name (for language-specific abbreviations)

    Returns:
        List of sentences with abbreviation splits merged
    """
    # Get language-specific abbreviations
    abbreviations = get_abbreviations(language_model)

    # If no abbreviations for this language, return unchanged
    if not abbreviations:
        return sentences

    if len(sentences) <= 1:
        return sentences

    # Get common sentence starters and sentence-ending abbreviations
    sentence_starters = get_sentence_starters()
    sentence_ending_abbrevs = get_sentence_ending_abbreviations()

    result: list[str] = []
    i = 0

    while i < len(sentences):
        current = sentences[i]

        # Check if we should merge with the next sentence
        if i + 1 < len(sentences):
            next_sent = sentences[i + 1]

            # Check if current sentence ends with an abbreviation
            match = _ABBREV_END_PATTERN.search(current)
            if match:
                abbrev = match.group(1)
                abbrev_key = abbrev.replace(".", "")
                is_dotted = "." in abbrev
                if abbrev_key and abbrev_key not in sentence_ending_abbrevs:
                    if abbrev_key in abbreviations or is_dotted:
                        first_word = _extract_leading_word(next_sent)
                        if first_word:
                            first_cased = _first_cased_char(first_word)
                            if (
                                first_cased
                                and first_cased.isupper()
                                and first_word not in sentence_starters
                                and not first_word.isupper()
                            ):
                                merged = current + " " + next_sent
                                result.append(merged)
                                i += 2
                                continue

        result.append(current)
        i += 1

    return result


# Pattern to detect ellipsis followed by a new sentence.
# Allows closing quotes/brackets immediately after the ellipsis, e.g. "...' Next"
# so that "'Hello...' 'Is it working?'" can be split after the ellipsis.
_ELLIPSIS_SENTENCE_BREAK = re.compile(
    r'((?:\.{3,}|\. \. \.|…)(?:["\'\)\]\}\u201d\u2019]+)?)\s+'
    r'(["\'\u201c\u201d\u2018\u2019]*[A-Z])',
)


def _split_after_ellipsis(sentences: list[str]) -> list[str]:
    """
    Split sentences that contain ellipsis followed by a new sentence.

    When text like "He was tired.... The next day" is processed, spaCy may not
    recognize the sentence boundary after the ellipsis. This function splits
    such cases by detecting ellipsis (3+ dots or ". . .") followed by whitespace
    and a capital letter (optionally preceded by quotes).

    Args:
        sentences: List of sentences from spaCy

    Returns:
        List of sentences with ellipsis boundaries properly handled
    """
    if not sentences:
        return sentences

    result: list[str] = []
    for sent in sentences:
        split_positions = _find_ellipsis_split_positions(sent)
        if not split_positions:
            result.append(sent)
            continue

        start = 0
        for split_pos in split_positions:
            first_part = sent[start:split_pos].strip()
            if first_part:
                result.append(first_part)
            start = split_pos

        remaining = sent[start:].strip()
        if remaining:
            result.append(remaining)

    return result


def _apply_corrections(
    sentences: list[str],
    language_model: str = "en_core_web_sm",
    split_on_colon: bool = True,
    nlp: Language | None = None,
) -> list[str]:
    """
    Apply post-processing corrections to fix common spaCy segmentation errors.

    Corrections applied (in order):
    1. Merge sentences incorrectly split after abbreviations (reduces count)
    2. Split sentences after ellipsis followed by capital letter (increases count)
    3. Split sentences after URL boundaries missed by spaCy (increases count)
    4. Split sentences containing multiple URLs (increases count)

    Note: Colon handling is minimal - we let spaCy handle colons naturally.
    The split_on_colon parameter is kept for API compatibility but currently
    has no effect (spaCy's default colon behavior is used).

    Args:
        sentences: List of sentences from spaCy
        language_model: spaCy language model name (for language-specific corrections)
        split_on_colon: Kept for API compatibility (currently unused)
        nlp: Optional spaCy language model (currently unused)

    Returns:
        Corrected list of sentences
    """
    # First merge abbreviation splits (need to combine before other splits)
    sentences = _merge_abbreviation_splits(sentences, language_model)

    # Split after ellipsis followed by new sentence
    sentences = _split_after_ellipsis(sentences)

    # Split after URL boundaries missed by spaCy
    sentences = _split_after_url_boundaries(sentences)

    # Split URLs (increases sentence count)
    sentences = _split_urls(sentences)

    return sentences


def _get_nlp(language_model: str = "en_core_web_sm") -> Language:
    """Get or load a spaCy model (cached).

    Args:
        language_model: Name of the spaCy language model to load

    Returns:
        Loaded spaCy Language model

    Raises:
        ImportError: If spaCy is not installed
        OSError: If the specified language model is not found
    """
    if not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is required for this feature. "
            "Install with: pip install phrasplit[nlp]\n"
            "Then download a language model: python -m spacy download en_core_web_sm"
        )

    if language_model not in _nlp_cache:
        try:
            # spacy is guaranteed to be not None here due to SPACY_AVAILABLE check above
            assert spacy is not None
            _nlp_cache[language_model] = spacy.load(language_model)
        except OSError:
            raise OSError(
                f"spaCy language model '{language_model}' not found. "
                f"Download with: python -m spacy download {language_model}"
            ) from None

    return _nlp_cache[language_model]


def _extract_sentences(doc) -> list[str]:
    """Extract sentences from a spaCy Doc object.

    Args:
        doc: A spaCy Doc object

    Returns:
        List of sentence strings (stripped, non-empty)
    """
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _process_long_text(
    text: str,
    nlp: Language,
    max_chunk: int = _DEFAULT_MAX_CHUNK_SIZE,
    safety_margin: int = _DEFAULT_SAFETY_MARGIN,
) -> list[str]:
    """Process text that may exceed spaCy's max_length incrementally.

    Uses index-based tracking to extract sentences from long text without
    cutting sentences at chunk boundaries.

    Args:
        text: Input text (should be preprocessed, ellipsis protected)
        nlp: spaCy Language model
        max_chunk: Maximum characters to process at once
        safety_margin: Buffer at chunk end to avoid cutting sentences

    Returns:
        List of sentence strings (stripped, non-empty)
    """
    # Cap max_chunk to spaCy's limit minus safety margin
    effective_max = min(max_chunk, nlp.max_length - safety_margin)

    if len(text) <= effective_max:
        doc = nlp(text)
        return _extract_sentences(doc)

    sentences: list[str] = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + effective_max, len(text))
        chunk = text[start_idx:end_idx]
        doc = nlp(chunk)

        if end_idx >= len(text):
            # Last chunk - take all sentences
            sentences.extend(_extract_sentences(doc))
            break

        # Not the last chunk - keep only complete sentences
        last_complete_end = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text and sent.end_char < len(chunk) - safety_margin:
                sentences.append(sent_text)
                last_complete_end = sent.end_char

        # Move start index forward
        if last_complete_end > 0:
            start_idx += last_complete_end
        else:
            # No sentence boundary found - take all and move on
            sentences.extend(_extract_sentences(doc))
            start_idx = end_idx

        # Skip leading whitespace for next iteration
        while start_idx < len(text) and text[start_idx] in " \t\n\r":
            start_idx += 1

    return sentences


def _process_long_text_with_offsets(
    text: str,
    nlp: Language,
    max_chunk: int = _DEFAULT_MAX_CHUNK_SIZE,
    safety_margin: int = _DEFAULT_SAFETY_MARGIN,
) -> list[tuple[int, int]]:
    """Process text into sentence offsets without exceeding spaCy limits."""
    effective_max = min(max_chunk, nlp.max_length - safety_margin)

    if len(text) <= effective_max:
        doc = nlp(text)
        return [
            (sent.start_char, sent.end_char)
            for sent in doc.sents
            if sent.text and not sent.text.isspace()
        ]

    offsets: list[tuple[int, int]] = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + effective_max, len(text))
        chunk = text[start_idx:end_idx]
        doc = nlp(chunk)

        if end_idx >= len(text):
            offsets.extend(
                (start_idx + sent.start_char, start_idx + sent.end_char)
                for sent in doc.sents
                if sent.text and not sent.text.isspace()
            )
            break

        last_complete_end = 0
        for sent in doc.sents:
            if sent.text and not sent.text.isspace():
                if sent.end_char < len(chunk) - safety_margin:
                    offsets.append(
                        (start_idx + sent.start_char, start_idx + sent.end_char)
                    )
                    last_complete_end = sent.end_char

        if last_complete_end > 0:
            start_idx += last_complete_end
        else:
            offsets.extend(
                (start_idx + sent.start_char, start_idx + sent.end_char)
                for sent in doc.sents
                if sent.text and not sent.text.isspace()
            )
            start_idx = end_idx

        while start_idx < len(text) and text[start_idx] in " \t\n\r":
            start_idx += 1

    return offsets


def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs (separated by double newlines).

    Applies preprocessing to fix hyphenated line breaks and normalize whitespace.

    Args:
        text: Input text

    Returns:
        List of paragraphs (non-empty, stripped)
    """
    text = _preprocess_text(text)
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences_spacy(
    text: str,
    language_model: str = "en_core_web_sm",
    apply_corrections: bool = True,
    split_on_colon: bool = True,
) -> list[str]:
    """
    Split text into sentences using spaCy (internal implementation).

    Args:
        text: Input text
        language_model: spaCy language model to use
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling).
            Default is True.
        split_on_colon: Kept for API compatibility (currently unused).
            spaCy's default colon behavior is used. Default is True.

    Returns:
        List of sentences
    """
    nlp = _get_nlp(language_model)
    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return []

    result: list[str] = []
    for para in paragraphs:
        # Protect ellipsis from being treated as sentence boundaries
        para = _protect_ellipsis(para)

        # Process paragraph into sentences (handles long text)
        sentences = _process_long_text(para, nlp)

        for sent in sentences:
            # Restore ellipsis in the sentence
            sent = _restore_ellipsis(sent)
            result.append(sent)

    # Apply post-processing corrections if enabled
    if apply_corrections:
        result = _apply_corrections(result, language_model, split_on_colon, nlp)

    return result


def split_sentences(
    text: str,
    language_model: str = "en_core_web_sm",
    apply_corrections: bool = True,
    split_on_colon: bool = True,
    use_spacy: bool | None = None,
) -> list[str]:
    """
    Split text into sentences.

    By default, uses spaCy if available for best accuracy, otherwise falls back
    to regex-based splitting. You can force a specific implementation with use_spacy.

    Args:
        text: Input text
        language_model: Language model name (e.g., "en_core_web_sm", "de_core_news_sm")
            For spaCy mode: Name of the spaCy model to use
            For simple mode: Used to determine language for abbreviation handling
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling).
            Default is True. Only applies to spaCy mode.
        split_on_colon: Deprecated. Kept for API compatibility (currently unused).
            spaCy's default colon behavior is used. Default is True.
        use_spacy: Choose implementation:
            - None (default): Auto-detect spaCy and use if available
            - True: Force spaCy (raise ImportError if not installed)
            - False: Force simple regex-based splitting (no spaCy)

    Returns:
        List of sentences

    Raises:
        ImportError: If use_spacy=True but spaCy is not installed

    Example:
        >>> # Auto-detect (uses spaCy if available)
        >>> sentences = split_sentences(text)
        >>>
        >>> # Force simple mode (even if spaCy is installed)
        >>> sentences = split_sentences(text, use_spacy=False)
        >>>
        >>> # Force spaCy mode (error if not installed)
        >>> sentences = split_sentences(text, use_spacy=True)

    Note:
        The simple mode (regex-based) is faster and has no ML dependencies,
        but is less accurate (~85-90% vs ~95%+ for spaCy) on complex text.
        For best results with complex text, install spaCy:
        pip install phrasplit[nlp]
    """
    # Deprecation warning for split_on_colon
    if not split_on_colon:
        warnings.warn(
            "The split_on_colon parameter is deprecated and has no effect. "
            "It will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Determine which implementation to use
    if use_spacy is None:
        # Auto-detect: use spaCy if available
        use_spacy = SPACY_AVAILABLE
    elif use_spacy and not SPACY_AVAILABLE:
        # User explicitly requested spaCy but it's not available
        raise ImportError(
            "spaCy is not installed. Install with: pip install phrasplit[nlp]\n"
            "Then download a language model: python -m spacy download en_core_web_sm\n"
            "Or use use_spacy=False to use the simple regex-based splitter."
        )

    if use_spacy:
        # Use spaCy-based implementation
        return _split_sentences_spacy(
            text, language_model, apply_corrections, split_on_colon
        )
    else:
        # Use simple regex-based implementation
        # Import here to avoid circular dependency issues
        from phrasplit.splitter_without_spacy import split_sentences_simple

        return split_sentences_simple(text, language_model)


def _split_sentence_into_clauses(sentence: str) -> list[str]:
    """
    Split a sentence into comma-separated parts for audiobook creation.

    Splits only at commas, keeping the comma at the end of each part.
    This creates natural pause points for text-to-speech processing.

    Args:
        sentence: A single sentence

    Returns:
        List of comma-separated parts
    """
    # Pattern to split after comma followed by space
    # Using positive lookbehind to keep comma at end of clause
    parts = re.split(r"(?<=,)\s+", sentence)

    # Filter empty parts and strip whitespace
    clauses = [p.strip() for p in parts if p.strip()]

    return clauses if clauses else [sentence]


def _split_clauses_spacy(
    text: str,
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Split text into comma-separated parts using spaCy (internal implementation).

    Args:
        text: Input text
        language_model: spaCy language model to use

    Returns:
        List of comma-separated parts
    """
    nlp = _get_nlp(language_model)
    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return []

    result: list[str] = []
    for para in paragraphs:
        # Protect ellipsis from being treated as sentence boundaries
        para = _protect_ellipsis(para)

        # Process paragraph into sentences (handles long text)
        sentences = _process_long_text(para, nlp)

        # Process each sentence into clauses
        for sent in sentences:
            # Restore ellipsis in the sentence
            sent = _restore_ellipsis(sent)

            # Split sentence at clause boundaries
            clauses = _split_sentence_into_clauses(sent)
            result.extend(clauses)

    return result


def split_clauses(
    text: str,
    language_model: str = "en_core_web_sm",
    use_spacy: bool | None = None,
) -> list[str]:
    """
    Split text into comma-separated parts for audiobook creation.

    Uses sentence detection, then splits each sentence at commas.
    The comma stays at the end of each part, creating natural pause points
    for text-to-speech processing.

    Args:
        text: Input text
        language_model: Language model name (e.g., "en_core_web_sm")
        use_spacy: Choose implementation:
            - None (default): Auto-detect spaCy and use if available
            - True: Force spaCy (raise ImportError if not installed)
            - False: Force simple regex-based splitting

    Returns:
        List of comma-separated parts

    Raises:
        ImportError: If use_spacy=True but spaCy is not installed

    Example:
        Input: "I do like coffee, and I like wine."
        Output: ["I do like coffee,", "and I like wine."]
    """
    # Determine which implementation to use
    if use_spacy is None:
        use_spacy = SPACY_AVAILABLE
    elif use_spacy and not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is not installed. Install with: pip install phrasplit[nlp]\n"
            "Or use use_spacy=False to use the simple regex-based splitter."
        )

    if use_spacy:
        return _split_clauses_spacy(text, language_model)
    else:
        from phrasplit.splitter_without_spacy import split_clauses_simple

        return split_clauses_simple(text, language_model)


def _split_at_clauses(text: str, max_length: int) -> list[str]:
    """
    Split text at comma boundaries for audiobook creation.

    Args:
        text: Text to split
        max_length: Maximum line length

    Returns:
        List of lines
    """
    # Split at commas, keeping the comma with the preceding text
    parts = re.split(r"(?<=,)\s+", text)

    result: list[str] = []
    current_line = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if not current_line:
            current_line = part
        elif len(current_line) + 1 + len(part) <= max_length:
            current_line += " " + part
        else:
            if current_line:
                result.append(current_line)
            current_line = part

    if current_line:
        result.append(current_line)

    # If still too long, do hard split at word boundaries
    final_result: list[str] = []
    for line in result:
        if len(line) > max_length:
            final_result.extend(_hard_split(line, max_length))
        else:
            final_result.append(line)

    return final_result if final_result else [text]


def _hard_split(text: str, max_length: int) -> list[str]:
    """
    Hard split text at word boundaries when clause splitting isn't enough.

    Args:
        text: Text to split
        max_length: Maximum line length

    Returns:
        List of lines
    """
    words = text.split()
    result: list[str] = []
    current_line = ""

    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + 1 + len(word) <= max_length:
            current_line += " " + word
        else:
            result.append(current_line)
            current_line = word

    if current_line:
        result.append(current_line)

    return result if result else [text]


def _split_at_boundaries(text: str, max_length: int, nlp: Language) -> list[str]:
    """
    Split text at sentence/clause boundaries to fit within max_length.

    Args:
        text: Text to split
        max_length: Maximum line length
        nlp: spaCy language model

    Returns:
        List of lines
    """
    # Protect ellipsis before spaCy processing
    protected_text = _protect_ellipsis(text)

    # Split into sentences (handles long text)
    sentences = _process_long_text(protected_text, nlp)

    result: list[str] = []
    current_line = ""

    for sent in sentences:
        # Restore ellipsis in the sentence
        sent = _restore_ellipsis(sent)
        # If sentence itself exceeds max_length, split at clauses
        if len(sent) > max_length:
            # Flush current line first
            if current_line:
                result.append(current_line)
                current_line = ""
            # Split sentence at clause boundaries
            clause_lines = _split_at_clauses(sent, max_length)
            result.extend(clause_lines)
        elif not current_line:
            current_line = sent
        elif len(current_line) + 1 + len(sent) <= max_length:
            current_line += " " + sent
        else:
            result.append(current_line)
            current_line = sent

    if current_line:
        result.append(current_line)

    return result if result else [text]


def _split_long_lines_spacy(
    text: str,
    max_length: int,
    language_model: str = "en_core_web_sm",
) -> list[str]:
    """
    Split lines exceeding max_length at clause/sentence boundaries using spaCy.

    Args:
        text: Input text
        max_length: Maximum line length in characters (must be positive)
        language_model: spaCy language model to use

    Returns:
        List of lines, each within max_length (except single words exceeding limit)

    Raises:
        ValueError: If max_length is less than 1
    """
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")

    nlp = _get_nlp(language_model)

    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        # Check if line is within limit
        if len(line) <= max_length:
            result.append(line)
            continue

        # Split the long line
        split_lines = _split_at_boundaries(line, max_length, nlp)
        result.extend(split_lines)

    return result


def split_long_lines(
    text: str,
    max_length: int,
    language_model: str = "en_core_web_sm",
    use_spacy: bool | None = None,
) -> list[str]:
    """
    Split lines exceeding max_length at clause/sentence boundaries.

    Strategy:
    1. First try to split at sentence boundaries
    2. If still too long, split at clause boundaries (commas, semicolons, etc.)
    3. If still too long, split at word boundaries

    Args:
        text: Input text
        max_length: Maximum line length in characters (must be positive)
        language_model: Language model name (e.g., "en_core_web_sm")
        use_spacy: Choose implementation:
            - None (default): Auto-detect spaCy and use if available
            - True: Force spaCy (raise ImportError if not installed)
            - False: Force simple regex-based splitting

    Returns:
        List of lines, each within max_length (except single words exceeding limit)

    Raises:
        ValueError: If max_length is less than 1
        ImportError: If use_spacy=True but spaCy is not installed
    """
    # Determine which implementation to use
    if use_spacy is None:
        use_spacy = SPACY_AVAILABLE
    elif use_spacy and not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is not installed. Install with: pip install phrasplit[nlp]\n"
            "Or use use_spacy=False to use the simple regex-based splitter."
        )

    if use_spacy:
        return _split_long_lines_spacy(text, max_length, language_model)
    else:
        from phrasplit.splitter_without_spacy import split_long_lines_simple

        return split_long_lines_simple(text, max_length, language_model)


def split_text(
    text: str,
    mode: str = "sentence",
    language_model: str = "en_core_web_sm",
    apply_corrections: bool = True,
    split_on_colon: bool = True,
    use_spacy: bool | None = None,
) -> list[Segment]:
    """
    Split text into segments with hierarchical position information.

    This function provides a unified interface for text splitting with different
    granularity levels, while preserving paragraph and sentence structure information.
    Useful for audiobook generation where different pause lengths are needed
    between paragraphs vs. sentences vs. clauses.

    Args:
        text: Input text to split
        mode: Splitting mode - one of:
            - "paragraph": Split into paragraphs only
            - "sentence": Split into sentences, grouped by paragraph
            - "clause": Split into clauses (comma-separated), with paragraph
              and sentence info
        language_model: Language model name (e.g., "en_core_web_sm")
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling).
            Default is True. Only applies to spaCy mode and sentence/clause modes.
        split_on_colon: Deprecated. Kept for API compatibility (currently unused).
            spaCy's default colon behavior is used. Default is True.
        use_spacy: Choose implementation:
            - None (default): Auto-detect spaCy and use if available
            - True: Force spaCy (raise ImportError if not installed)
            - False: Force simple regex-based splitting

    Returns:
        List of Segment namedtuples, each containing:
            - text: The segment text
            - paragraph: Paragraph index (0-based)
            - sentence: Sentence index within paragraph (0-based).
              None for paragraph mode.

    Raises:
        ValueError: If mode is not one of "paragraph", "sentence", "clause"
        ImportError: If use_spacy=True but spaCy is not installed

    Example:
        >>> segments = split_text("Hello world. How are you?\\n\\nNew paragraph.")
        >>> for seg in segments:
        ...     print(f"P{seg.paragraph} S{seg.sentence}: {seg.text}")
        P0 S0: Hello world.
        P0 S1: How are you?
        P1 S0: New paragraph.

        >>> # Detect paragraph changes for longer pauses
        >>> for i, seg in enumerate(segments):
        ...     if i > 0 and seg.paragraph != segments[i-1].paragraph:
        ...         print("--- paragraph break ---")
        ...     print(seg.text)
    """
    # Deprecation warning
    if not split_on_colon:
        warnings.warn(
            "The split_on_colon parameter is deprecated and has no effect. "
            "It will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    valid_modes = ("paragraph", "sentence", "clause")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return []

    result: list[Segment] = []

    if mode == "paragraph":
        for para_idx, para in enumerate(paragraphs):
            result.append(Segment(text=para, paragraph=para_idx, sentence=None))
        return result

    # Determine which implementation to use for sentence/clause modes
    if use_spacy is None:
        use_spacy = SPACY_AVAILABLE
    elif use_spacy and not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is not installed. Install with: pip install phrasplit[nlp]\n"
            "Or use use_spacy=False to use the simple regex-based splitter."
        )

    if use_spacy:
        # Use spaCy implementation
        nlp = _get_nlp(language_model)

        for para_idx, para in enumerate(paragraphs):
            # Protect ellipsis from being treated as sentence boundaries
            protected_para = _protect_ellipsis(para)

            # Process paragraph into sentences (handles long text)
            sentences = _process_long_text(protected_para, nlp)

            # Restore ellipsis in sentences
            sentences = [_restore_ellipsis(sent) for sent in sentences]

            # Apply post-processing corrections if enabled
            if apply_corrections:
                sentences = _apply_corrections(
                    sentences, language_model, split_on_colon, nlp
                )

            if mode == "sentence":
                for sent_idx, sent in enumerate(sentences):
                    result.append(
                        Segment(text=sent, paragraph=para_idx, sentence=sent_idx)
                    )

            elif mode == "clause":
                for sent_idx, sent in enumerate(sentences):
                    clauses = _split_sentence_into_clauses(sent)
                    for clause in clauses:
                        result.append(
                            Segment(text=clause, paragraph=para_idx, sentence=sent_idx)
                        )
    else:
        # Use simple implementation
        from phrasplit.splitter_without_spacy import split_sentences_simple

        for para_idx, para in enumerate(paragraphs):
            # Get sentences for this paragraph
            sentences = split_sentences_simple(para, language_model)

            if mode == "sentence":
                for sent_idx, sent in enumerate(sentences):
                    result.append(
                        Segment(text=sent, paragraph=para_idx, sentence=sent_idx)
                    )

            elif mode == "clause":
                for sent_idx, sent in enumerate(sentences):
                    clauses = _split_sentence_into_clauses(sent)
                    for clause in clauses:
                        result.append(
                            Segment(text=clause, paragraph=para_idx, sentence=sent_idx)
                        )

    return result


# =============================================================================
# Offset-preserving segmentation
# =============================================================================


def _make_segment_id(
    paragraph_idx: int, sentence_idx: int, clause_idx: int | None = None
) -> str:
    """Generate a stable segment ID.

    Args:
        paragraph_idx: Paragraph index (0-based)
        sentence_idx: Sentence index (0-based)
        clause_idx: Clause index (0-based), or None

    Returns:
        Stable ID string in format "p{para}s{sent}" or "p{para}s{sent}c{clause}"
    """
    if clause_idx is None:
        return f"p{paragraph_idx}s{sentence_idx}"
    return f"p{paragraph_idx}s{sentence_idx}c{clause_idx}"


def _validate_offset_segments(text: str, segments: list[SplitSegment]) -> None:
    """Validate that segments align to the original text offsets."""
    last_end = 0
    text_length = len(text)

    for seg in segments:
        if not (0 <= seg.char_start <= seg.char_end <= text_length):
            raise ValueError(
                "Segment offsets out of bounds for "
                f"{seg.id}: {seg.char_start}-{seg.char_end}"
            )
        if text[seg.char_start : seg.char_end] != seg.text:
            raise ValueError(
                "Segment text does not match slice for "
                f"{seg.id}: {seg.char_start}-{seg.char_end}"
            )
        if seg.char_start < last_end:
            raise ValueError(
                "Segments overlap or are out of order at "
                f"{seg.id}: {seg.char_start} < {last_end}"
            )
        last_end = seg.char_end


def _trim_segment_bounds(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim whitespace from the segment boundaries without altering offsets."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _merge_abbreviation_splits_with_offsets(
    text: str,
    segments: list[tuple[str, int, int]],
    language_model: str = "en_core_web_sm",
) -> list[tuple[str, int, int]]:
    """Merge abbreviation splits while preserving exact offsets."""
    if len(segments) <= 1:
        return segments

    abbreviations = get_abbreviations(language_model)
    if not abbreviations:
        return segments

    sentence_starters = get_sentence_starters()
    sentence_ending_abbrevs = get_sentence_ending_abbreviations()

    result: list[tuple[str, int, int]] = []
    i = 0

    while i < len(segments):
        current_text, current_start, current_end = segments[i]

        if i + 1 < len(segments):
            next_text, next_start, next_end = segments[i + 1]

            match = _ABBREV_END_PATTERN.search(current_text.rstrip())
            if match:
                abbrev = match.group(1)
                abbrev_key = abbrev.replace(".", "")
                is_dotted = "." in abbrev
                if abbrev_key and abbrev_key not in sentence_ending_abbrevs:
                    if abbrev_key in abbreviations or is_dotted:
                        first_word = _extract_leading_word(next_text)
                        if first_word:
                            first_cased = _first_cased_char(first_word)
                            if (
                                first_cased
                                and first_cased.isupper()
                                and first_word not in sentence_starters
                                and not first_word.isupper()
                            ):
                                merged_text = text[current_start:next_end]
                                result.append((merged_text, current_start, next_end))
                                i += 2
                                continue

        result.append((current_text, current_start, current_end))
        i += 1

    return result


def _split_after_ellipsis_with_offsets(
    text: str, segments: list[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """Split sentences after ellipsis boundaries while preserving offsets."""
    if not segments:
        return segments

    result: list[tuple[str, int, int]] = []

    for seg_text, seg_start, seg_end in segments:
        split_positions = _find_ellipsis_split_positions(seg_text)
        if not split_positions:
            trimmed = _trim_segment_bounds(text, seg_start, seg_end)
            if trimmed:
                start, end = trimmed
                result.append((text[start:end], start, end))
            continue

        current_start = seg_start
        for split_pos in split_positions:
            absolute_end = seg_start + split_pos
            trimmed = _trim_segment_bounds(text, current_start, absolute_end)
            if trimmed:
                start, end = trimmed
                result.append((text[start:end], start, end))
            current_start = absolute_end

        trimmed = _trim_segment_bounds(text, current_start, seg_end)
        if trimmed:
            start, end = trimmed
            result.append((text[start:end], start, end))

    return result


def _split_urls_with_offsets(
    text: str, segments: list[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """Split sentences containing multiple URLs while preserving offsets."""
    result: list[tuple[str, int, int]] = []

    for seg_text, seg_start, seg_end in segments:
        url_spans = _find_url_spans(seg_text)

        if len(url_spans) < 2:
            result.append((seg_text, seg_start, seg_end))
            continue

        last_end = 0
        for i, (url_start, _) in enumerate(url_spans):
            if i == 0 and url_start > 0:
                prefix = seg_text[:url_start].strip()
                if prefix:
                    next_url_start = (
                        url_spans[i + 1][0] if i + 1 < len(url_spans) else len(seg_text)
                    )
                    part_start = seg_start
                    part_end = seg_start + next_url_start
                    trimmed = _trim_segment_bounds(text, part_start, part_end)
                    if trimmed:
                        start, end = trimmed
                        result.append((text[start:end], start, end))
                    last_end = next_url_start
                    continue

            if url_start >= last_end:
                next_url_start = (
                    url_spans[i + 1][0] if i + 1 < len(url_spans) else len(seg_text)
                )
                part_start = seg_start + url_start
                part_end = seg_start + next_url_start
                trimmed = _trim_segment_bounds(text, part_start, part_end)
                if trimmed:
                    start, end = trimmed
                    result.append((text[start:end], start, end))
                last_end = next_url_start

    return result


def _split_after_url_boundaries_with_offsets(
    text: str, segments: list[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """Split sentences after URL boundaries while preserving offsets."""
    result: list[tuple[str, int, int]] = []

    for seg_text, seg_start, seg_end in segments:
        split_positions = _find_url_sentence_split_positions(seg_text)
        if not split_positions:
            result.append((seg_text, seg_start, seg_end))
            continue

        current_start = seg_start
        for split_pos in split_positions:
            absolute_end = seg_start + split_pos
            trimmed = _trim_segment_bounds(text, current_start, absolute_end)
            if trimmed:
                start, end = trimmed
                result.append((text[start:end], start, end))
            current_start = absolute_end

        trimmed = _trim_segment_bounds(text, current_start, seg_end)
        if trimmed:
            start, end = trimmed
            result.append((text[start:end], start, end))

    return result


def _apply_corrections_with_offsets(
    text: str,
    segments: list[tuple[str, int, int]],
    language_model: str = "en_core_web_sm",
) -> list[tuple[str, int, int]]:
    """Apply sentence corrections while preserving exact offsets."""
    segments = _merge_abbreviation_splits_with_offsets(text, segments, language_model)
    segments = _split_after_ellipsis_with_offsets(text, segments)
    segments = _split_after_url_boundaries_with_offsets(text, segments)
    segments = _split_urls_with_offsets(text, segments)
    return segments


def _simple_sentence_split_preserving_offsets(  # noqa: C901
    text: str, language_model: str = "en"
) -> list[tuple[str, int, int]]:
    """Split text into sentences using simple regex, preserving exact character offsets.

    This is a minimal sentence splitter for offset-preserving segmentation.
    It does NO preprocessing or normalization - works on raw text only.

    Args:
        text: Raw text to split (no preprocessing!)
        language_model: Language model for abbreviation handling

    Returns:
        List of (sentence_text, start_offset, end_offset) tuples
        where text[start:end] == sentence_text (exact-slice invariant)
    """
    if not text:
        return []

    # Get abbreviations for this language to avoid splitting on "Dr.", etc.
    # But DO split on sentence-ending abbreviations like "Inc.", "Ltd."
    abbreviations = get_abbreviations(language_model)
    sentence_ending_abbrevs = get_sentence_ending_abbreviations()
    sentence_starters = get_sentence_starters()

    # Split on sentence-ending punctuation followed by a new sentence start
    sentence_pattern = re.compile(r"[.!?]+")

    sentences = []
    last_end = 0

    for match in sentence_pattern.finditer(text):
        match_start = match.start()
        match_end = match.end()

        if match_end >= len(text) or not text[match_end].isspace():
            continue

        if not _is_sentence_start(text, match_end):
            continue

        is_abbreviation = False
        token_start = match_start - 1
        while token_start >= 0 and not text[token_start].isspace():
            token_start -= 1
        token = text[token_start + 1 : match_start]
        token = token.strip(_CLOSING_PUNCTUATION)
        abbrev_token = token.rstrip(".")
        abbrev_key = abbrev_token.replace(".", "")
        dotted_acronym = "." in abbrev_token and all(
            char.isalpha() or char == "." for char in abbrev_token
        )

        if abbrev_key:
            if abbrev_key in sentence_ending_abbrevs:
                is_abbreviation = False
            elif abbrev_key in abbreviations or dotted_acronym:
                next_word = _extract_leading_word(text[match_end:])
                if dotted_acronym and next_word and next_word in sentence_starters:
                    is_abbreviation = False
                else:
                    is_abbreviation = True

        if is_abbreviation:
            continue

        sent_end = match.end()
        sent_text = text[last_end:sent_end].strip()

        if sent_text:
            # Find the exact slice in original text
            # We need to adjust for stripping
            sent_start = last_end
            # Skip leading whitespace
            while sent_start < sent_end and text[sent_start].isspace():
                sent_start += 1
            # Adjust end to exclude trailing whitespace
            sent_end_adjusted = sent_end
            while (
                sent_end_adjusted > sent_start and text[sent_end_adjusted - 1].isspace()
            ):
                sent_end_adjusted -= 1

            if sent_start < sent_end_adjusted:
                sentences.append(
                    (text[sent_start:sent_end_adjusted], sent_start, sent_end_adjusted)
                )

        last_end = sent_end

    # Add final sentence
    if last_end < len(text):
        sent_text = text[last_end:].strip()
        if sent_text:
            # Find exact slice
            sent_start = last_end
            while sent_start < len(text) and text[sent_start].isspace():
                sent_start += 1
            sent_end = len(text)
            while sent_end > sent_start and text[sent_end - 1].isspace():
                sent_end -= 1

            if sent_start < sent_end:
                sentences.append((text[sent_start:sent_end], sent_start, sent_end))

    # Fallback: if no sentences found, return the whole text as one sentence
    if not sentences and text.strip():
        sent_start = 0
        while sent_start < len(text) and text[sent_start].isspace():
            sent_start += 1
        sent_end = len(text)
        while sent_end > sent_start and text[sent_end - 1].isspace():
            sent_end -= 1
        if sent_start < sent_end:
            sentences.append((text[sent_start:sent_end], sent_start, sent_end))

    return sentences


def _simple_sentence_split_with_markup_offsets(  # noqa: C901
    text: str, language_model: str = "en"
) -> list[tuple[str, int, int]]:
    """Markup-aware sentence split preserving exact character offsets.

    Like :func:`_simple_sentence_split_preserving_offsets`, but understands inline
    XHTML tags so sentence boundaries are not blocked by inline markup:

    * Closing tags immediately after sentence punctuation belong to the previous
      segment, keeping fragments balanced (``her.</em>`` keeps ``</em>``).
    * Opening/empty tags after the following whitespace belong to the next
      segment (``slowly. <em>I`` starts the next segment at ``<em>``).

    The exact-slice invariant ``text[start:end] == sentence_text`` is preserved.
    Tags are matched with a conservative regex; ordinary ``<`` comparisons are
    not treated as tags.
    """
    if not text:
        return []

    abbreviations = get_abbreviations(language_model)
    sentence_ending_abbrevs = get_sentence_ending_abbreviations()
    sentence_starters = get_sentence_starters()

    sentence_pattern = re.compile(r"[.!?]+")

    sentences: list[tuple[str, int, int]] = []
    last_end = 0

    for match in sentence_pattern.finditer(text):
        match_start = match.start()
        match_end = match.end()

        # Closing tags after punctuation belong to the previous segment so
        # fragments stay balanced; the boundary sits after them.
        seg_end = _skip_closing_tags(text, match_end)
        if seg_end >= len(text) or not text[seg_end].isspace():
            continue

        # Skip whitespace, then opening/empty tags, to reach the real next start.
        ws_end = seg_end
        while ws_end < len(text) and text[ws_end].isspace():
            ws_end += 1
        next_visible = _skip_opening_tags(text, ws_end)
        if next_visible >= len(text):
            continue
        if not _is_sentence_start(text, next_visible):
            continue

        # Abbreviation handling (same token-before logic as the plain splitter).
        is_abbreviation = False
        token_start = match_start - 1
        while token_start >= 0 and not text[token_start].isspace():
            token_start -= 1
        token = text[token_start + 1 : match_start]
        token = token.strip(_CLOSING_PUNCTUATION)
        abbrev_token = token.rstrip(".")
        abbrev_key = abbrev_token.replace(".", "")
        dotted_acronym = "." in abbrev_token and all(
            char.isalpha() or char == "." for char in abbrev_token
        )

        if abbrev_key:
            if abbrev_key in sentence_ending_abbrevs:
                is_abbreviation = False
            elif abbrev_key in abbreviations or dotted_acronym:
                next_word = _extract_leading_word(text[next_visible:])
                if dotted_acronym and next_word and next_word in sentence_starters:
                    is_abbreviation = False
                else:
                    is_abbreviation = True

        if is_abbreviation:
            continue

        sent_text = text[last_end:seg_end].strip()
        if sent_text:
            sent_start = last_end
            while sent_start < seg_end and text[sent_start].isspace():
                sent_start += 1
            sent_end_adjusted = seg_end
            while (
                sent_end_adjusted > sent_start and text[sent_end_adjusted - 1].isspace()
            ):
                sent_end_adjusted -= 1
            if sent_start < sent_end_adjusted:
                sentences.append(
                    (text[sent_start:sent_end_adjusted], sent_start, sent_end_adjusted)
                )

        last_end = seg_end

    # Add final sentence
    if last_end < len(text):
        sent_text = text[last_end:].strip()
        if sent_text:
            sent_start = last_end
            while sent_start < len(text) and text[sent_start].isspace():
                sent_start += 1
            sent_end = len(text)
            while sent_end > sent_start and text[sent_end - 1].isspace():
                sent_end -= 1
            if sent_start < sent_end:
                sentences.append((text[sent_start:sent_end], sent_start, sent_end))

    # Fallback: if no sentences found, return the whole text as one sentence
    if not sentences and text.strip():
        sent_start = 0
        while sent_start < len(text) and text[sent_start].isspace():
            sent_start += 1
        sent_end = len(text)
        while sent_end > sent_start and text[sent_end - 1].isspace():
            sent_end -= 1
        if sent_start < sent_end:
            sentences.append((text[sent_start:sent_end], sent_start, sent_end))

    return sentences


def _split_with_offsets_regex(
    text: str,
    language_model: str = "en_core_web_sm",
    mode: str = "sentence",
    max_chars: int | None = None,
    inline_markup: bool = False,
) -> list[SplitSegment]:
    """Split text using regex-based approach with character offsets.

    Args:
        text: Input text to split
        language_model: Language model name for abbreviation handling
        mode: Splitting mode ("paragraph", "sentence", or "clause")
        max_chars: Optional maximum segment length

    Returns:
        List of SplitSegment objects with character offsets
    """

    if mode not in ("paragraph", "sentence", "clause"):
        raise ValueError(
            f"mode must be 'paragraph', 'sentence', or 'clause', got {mode!r}"
        )

    # Import regex-based functions

    result: list[SplitSegment] = []

    # Find paragraph boundaries using regex
    para_pattern = re.compile(r"\n\s*\n")
    para_starts = [0]
    for match in para_pattern.finditer(text):
        para_starts.append(match.end())

    para_idx = 0
    for i, para_start in enumerate(para_starts):
        para_end = para_starts[i + 1] if i + 1 < len(para_starts) else len(text)

        # Skip leading whitespace
        while para_start < para_end and text[para_start].isspace():
            para_start += 1

        # Skip trailing whitespace
        while para_end > para_start and text[para_end - 1].isspace():
            para_end -= 1

        # Extract exact slice (no .strip() needed since we adjusted offsets)
        para_text = text[para_start:para_end]

        if not para_text:
            continue

        if mode == "paragraph":
            seg_id = _make_segment_id(para_idx, 0)
            segment = SplitSegment(
                id=seg_id,
                text=para_text,
                char_start=para_start,
                char_end=para_end,
                paragraph_idx=para_idx,
                sentence_idx=0,
                clause_idx=None,
                meta={"method": "regex", "mode": "paragraph"},
            )
            result.append(segment)
            para_idx += 1
            continue

        # For sentence and clause modes, use offset-preserving sentence split
        # This avoids preprocessing that would break the exact-slice invariant
        if inline_markup:
            sentences_with_offsets = _simple_sentence_split_with_markup_offsets(
                para_text, language_model
            )
        else:
            sentences_with_offsets = _simple_sentence_split_preserving_offsets(
                para_text, language_model
            )

        sent_idx = 0
        for sent_text, sent_offset_in_para, sent_end_in_para in sentences_with_offsets:
            # Calculate absolute offsets
            sent_start = para_start + sent_offset_in_para
            sent_end = para_start + sent_end_in_para

            if mode == "sentence":
                seg_id = _make_segment_id(para_idx, sent_idx)
                segment = SplitSegment(
                    id=seg_id,
                    text=sent_text,
                    char_start=sent_start,
                    char_end=sent_end,
                    paragraph_idx=para_idx,
                    sentence_idx=sent_idx,
                    clause_idx=None,
                    meta={"method": "regex", "mode": "sentence"},
                )
                result.append(segment)
            else:  # clause mode
                clauses = _split_sentence_into_clauses(sent_text)
                clause_search_start = 0

                for clause_idx, clause_text in enumerate(clauses):
                    # Find clause within sentence
                    clause_offset = sent_text.find(clause_text, clause_search_start)
                    if clause_offset == -1:
                        clause_offset = clause_search_start

                    clause_start = sent_start + clause_offset
                    clause_end = clause_start + len(clause_text)
                    clause_search_start = clause_offset + len(clause_text)

                    seg_id = _make_segment_id(para_idx, sent_idx, clause_idx)
                    segment = SplitSegment(
                        id=seg_id,
                        text=clause_text,
                        char_start=clause_start,
                        char_end=clause_end,
                        paragraph_idx=para_idx,
                        sentence_idx=sent_idx,
                        clause_idx=clause_idx,
                        meta={"method": "regex", "mode": "clause"},
                    )
                    result.append(segment)

            sent_idx += 1

        para_idx += 1

    # Apply max_chars splitting if needed
    if max_chars is not None:
        result = _apply_max_chars_split(text, result, max_chars)

    return result


def _split_with_offsets_spacy(
    text: str,
    language_model: str = "en_core_web_sm",
    mode: str = "sentence",
    max_chars: int | None = None,
    apply_corrections: bool = True,
) -> list[SplitSegment]:
    """Split text using spaCy-based approach with character offsets.

    Args:
        text: Input text to split
        language_model: spaCy language model name
        mode: Splitting mode ("paragraph", "sentence", or "clause")
        max_chars: Optional maximum segment length
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling, ellipsis).
            Default is True.

    Returns:
        List of SplitSegment objects with character offsets
    """
    if mode not in ("paragraph", "sentence", "clause"):
        raise ValueError(
            f"mode must be 'paragraph', 'sentence', or 'clause', got {mode!r}"
        )

    nlp = _get_nlp(language_model)
    result: list[SplitSegment] = []

    # Find paragraph boundaries
    para_pattern = re.compile(r"\n\s*\n")
    para_starts = [0]
    for match in para_pattern.finditer(text):
        para_starts.append(match.end())

    para_idx = 0
    for i, para_start in enumerate(para_starts):
        para_end = para_starts[i + 1] if i + 1 < len(para_starts) else len(text)

        # Skip leading whitespace
        while para_start < para_end and text[para_start].isspace():
            para_start += 1

        # Skip trailing whitespace
        while para_end > para_start and text[para_end - 1].isspace():
            para_end -= 1

        # Extract exact slice (no .strip() needed since we adjusted offsets)
        para_text = text[para_start:para_end]

        if not para_text:
            continue

        if mode == "paragraph":
            seg_id = _make_segment_id(para_idx, 0)
            segment = SplitSegment(
                id=seg_id,
                text=para_text,
                char_start=para_start,
                char_end=para_end,
                paragraph_idx=para_idx,
                sentence_idx=0,
                clause_idx=None,
                meta={"method": "spacy", "mode": "paragraph"},
            )
            result.append(segment)
            para_idx += 1
            continue

        protected_para = _protect_ellipsis(para_text)
        sent_offsets = _process_long_text_with_offsets(protected_para, nlp)

        sentences: list[tuple[str, int, int]] = []
        for sent_start_in_para, sent_end_in_para in sent_offsets:
            sent_text = para_text[sent_start_in_para:sent_end_in_para]
            if not sent_text or sent_text.isspace():
                continue
            sent_start = para_start + sent_start_in_para
            sent_end = para_start + sent_end_in_para
            sentences.append((sent_text, sent_start, sent_end))

        if apply_corrections:
            sentences = _apply_corrections_with_offsets(text, sentences, language_model)

        for sent_idx, (sent_text, sent_start, sent_end) in enumerate(sentences):
            if mode == "sentence":
                seg_id = _make_segment_id(para_idx, sent_idx)
                segment = SplitSegment(
                    id=seg_id,
                    text=sent_text,
                    char_start=sent_start,
                    char_end=sent_end,
                    paragraph_idx=para_idx,
                    sentence_idx=sent_idx,
                    clause_idx=None,
                    meta={"method": "spacy", "mode": "sentence"},
                )
                result.append(segment)
            else:  # clause mode
                clauses = _split_sentence_into_clauses(sent_text)
                clause_search_start = 0

                for clause_idx, clause_text in enumerate(clauses):
                    clause_offset = sent_text.find(clause_text, clause_search_start)
                    if clause_offset == -1:
                        clause_offset = clause_search_start

                    clause_start = sent_start + clause_offset
                    clause_end = clause_start + len(clause_text)
                    clause_search_start = clause_offset + len(clause_text)

                    seg_id = _make_segment_id(para_idx, sent_idx, clause_idx)
                    segment = SplitSegment(
                        id=seg_id,
                        text=clause_text,
                        char_start=clause_start,
                        char_end=clause_end,
                        paragraph_idx=para_idx,
                        sentence_idx=sent_idx,
                        clause_idx=clause_idx,
                        meta={"method": "spacy", "mode": "clause"},
                    )
                    result.append(segment)

        para_idx += 1

    # Apply max_chars splitting if needed
    if max_chars is not None:
        result = _apply_max_chars_split(text, result, max_chars)

    return result


def _apply_max_chars_split(
    original_text: str, segments: list[SplitSegment], max_chars: int
) -> list[SplitSegment]:
    """Split segments that exceed max_chars while preserving exact offsets.

    Implements the exact-slice policy:
        segment.text == original_text[char_start:char_end]

    Strategy for splitting:
    1. Try to split at whitespace or punctuation boundaries near max_chars
    2. Never use .strip() - preserve exact slices
    3. Offsets always match the exact positions in original text
    4. Skip whitespace-only chunks by advancing boundaries

    Args:
        original_text: The original input text
        segments: List of segments to potentially split
        max_chars: Maximum character length for any segment

    Returns:
        New list of segments with long segments split, maintaining exact-slice invariant
    """
    result: list[SplitSegment] = []

    for seg in segments:
        if len(seg.text) <= max_chars:
            result.append(seg)
            continue

        # Need to split this segment
        # Strategy: split on whitespace or punctuation, never strip
        pos = 0
        sub_idx = 0

        while pos < len(seg.text):
            # Skip leading whitespace to avoid whitespace-only chunks
            while pos < len(seg.text) and seg.text[pos].isspace():
                pos += 1

            if pos >= len(seg.text):
                break

            # Find a good split point within max_chars
            end_pos = min(pos + max_chars, len(seg.text))

            if end_pos < len(seg.text):
                # Try to find whitespace or punctuation boundary
                # Look backwards from end_pos
                for i in range(end_pos - 1, pos, -1):
                    if seg.text[i] in " \t\n,;.!?":
                        end_pos = i + 1
                        break

            # Extract chunk WITHOUT stripping - exact slice
            chunk = seg.text[pos:end_pos]

            # Skip if chunk is only whitespace (advance pos and continue)
            if not chunk or chunk.isspace():
                pos = end_pos
                continue

            # Calculate absolute offsets in original text
            chunk_start = seg.char_start + pos
            chunk_end = seg.char_start + end_pos

            # Generate stable ID with sub-index
            # Use :m{index} suffix for max-chars splits (m = "max-chars")
            if sub_idx == 0:
                chunk_id = f"{seg.id}:m0"
            else:
                chunk_id = f"{seg.id}:m{sub_idx}"

            chunk_seg = SplitSegment(
                id=chunk_id,
                text=chunk,
                char_start=chunk_start,
                char_end=chunk_end,
                paragraph_idx=seg.paragraph_idx,
                sentence_idx=seg.sentence_idx,
                clause_idx=seg.clause_idx,
                meta={
                    **seg.meta,
                    "split_by_max_chars": True,
                    "max_chars_index": sub_idx,
                },
            )
            result.append(chunk_seg)
            sub_idx += 1
            pos = end_pos

    return result


def split_with_offsets(
    text: str,
    *,
    mode: str = "sentence",
    use_spacy: bool | None = None,
    language_model: str = "en_core_web_sm",
    apply_corrections: bool = True,
    max_chars: int | None = None,
    inline_markup: bool = False,
) -> list[SplitSegment]:
    """Split text into segments with character offsets and stable IDs.

    This is the main API for offset-preserving segmentation, designed for
    downstream processing where exact character positions are critical.

    **Exact-Slice Policy**

    This function implements the exact-slice policy: for every returned segment,
    the following invariant ALWAYS holds:

        segment.text == text[segment.char_start:segment.char_end]

    This guarantee means:
    - Offsets map precisely to the original input text
    - No whitespace normalization or stripping breaks the mapping
    - Downstream code can reliably use offsets for span slicing
    - Integration with token alignment and markup slicing is safe

    Key features:
    - Returns segments with precise character offsets (char_start, char_end)
    - Generates stable, hierarchical IDs (e.g., "p0s1", "p0s2c3")
    - Maintains exact-slice invariant in all modes
    - Supports both spaCy (accurate) and regex (fast) backends
    - Optional max_chars safety splitting with deterministic boundaries

    Args:
        text: Input text to split
        mode: Splitting granularity:
            - "paragraph": Split into paragraphs only
            - "sentence": Split into sentences (default)
            - "clause": Split into comma-separated clauses
        use_spacy: Backend selection:
            - None (default): Auto-detect, use spaCy if available
            - True: Force spaCy (raises ImportError if unavailable)
            - False: Force regex-based splitting
        language_model: Language model name (e.g., "en_core_web_sm", "de_core_news_sm")
            Used for both spaCy model selection and abbreviation handling
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling, ellipsis).
            Default is True. Only applies to spaCy mode.
        max_chars: Optional maximum segment length. Segments exceeding this
            will be split further at whitespace/punctuation boundaries while
            maintaining the exact-slice invariant. Split segments get IDs
            like "p0s1:m0", "p0s1:m1", etc.
        inline_markup: Opt-in inline XHTML markup handling (default False). When
            True, the regex backend keeps inline tags balanced across sentence
            boundaries: closing tags after punctuation stay in the previous
            segment and opening tags after whitespace start the next segment.
            Requires the regex backend; passing ``use_spacy=True`` with
            ``inline_markup=True`` raises ValueError. Default behavior (plain
            text) is unchanged when False.

    Returns:
        List of SplitSegment objects, each containing:
            - id: Stable identifier (e.g., "p0s1c2" or "p0s1:m0")
            - text: Segment text content (exact slice of input)
            - char_start, char_end: Character offsets in original text
            - paragraph_idx, sentence_idx, clause_idx: Hierarchical indices
            - meta: Additional metadata (method, mode, etc.)

    Raises:
        ValueError: If mode is invalid or max_chars < 1
        ImportError: If use_spacy=True but spaCy is not installed

    Example:
        >>> text = "Hello world. How are you?\\n\\nNew paragraph."
        >>> segments = split_with_offsets(text, mode="sentence")
        >>> for seg in segments:
        ...     # Verify exact-slice invariant
        ...     assert text[seg.char_start:seg.char_end] == seg.text
        ...     print(f"{seg.id}: {seg.text!r}")
        p0s0: 'Hello world.'
        p0s1: 'How are you?'
        p1s0: 'New paragraph.'

        >>> # With max_chars safety splitting
        >>> long_text = "word " * 100
        >>> segments = split_with_offsets(long_text, max_chars=50)
        >>> all(len(seg.text) <= 50 for seg in segments)
        True
        >>> # Exact-slice invariant still holds
        >>> all(long_text[s.char_start:s.char_end] == s.text for s in segments)
        True

    Note:
        - Segments may include leading/trailing whitespace from the original text
        - IDs are stable and deterministic across runs with same input and settings
        - For SSMD/markup integration, offsets are in the coordinate space of the
          input text (before or after escaping, depending on your workflow)
    """
    if max_chars is not None and max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars}")

    # inline_markup is only supported by the regex backend. Force it and reject
    # an explicit spaCy request so callers do not silently get plain-text splits.
    if inline_markup:
        if use_spacy is True:
            raise ValueError(
                "inline_markup=True is only supported with the regex backend "
                "(use_spacy=False)."
            )
        use_spacy = False

    # Determine which implementation to use
    if use_spacy is None:
        use_spacy = SPACY_AVAILABLE
    elif use_spacy and not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is not installed. Install with: pip install phrasplit[nlp]\n"
            "Then download a language model: python -m spacy download en_core_web_sm\n"
            "Or use use_spacy=False to use the simple regex-based splitter."
        )

    if use_spacy:
        segments = _split_with_offsets_spacy(
            text,
            language_model,
            mode,
            max_chars,
            apply_corrections,
        )
    else:
        segments = _split_with_offsets_regex(
            text, language_model, mode, max_chars, inline_markup=inline_markup
        )

    _validate_offset_segments(text, segments)
    return segments


def iter_split_with_offsets(
    text: str,
    *,
    mode: str = "sentence",
    use_spacy: bool | None = None,
    language_model: str = "en_core_web_sm",
    apply_corrections: bool = True,
    max_chars: int | None = None,
    inline_markup: bool = False,
) -> Iterator[SplitSegment]:
    """Streaming iterator variant of split_with_offsets().

    Yields segments one by one in document order, enabling memory-efficient
    processing of large texts and streaming TTS synthesis.

    Args:
        text: Input text to split
        mode: Splitting granularity ("paragraph", "sentence", or "clause")
        use_spacy: Backend selection (None=auto, True=spaCy, False=regex)
        language_model: Language model name for NLP/abbreviations
        apply_corrections: Whether to apply post-processing corrections for
            common spaCy errors (URL splitting, abbreviation handling, ellipsis).
            Default is True. Only applies to spaCy mode.
        max_chars: Optional maximum segment length

    Yields:
        SplitSegment objects in document order

    Example:
        >>> text = "First sentence. Second sentence.\\n\\nNew paragraph."
        >>> for segment in iter_split_with_offsets(text, mode="sentence"):
        ...     print(f"{segment.id}: {segment.text}")
        p0s0: First sentence.
        p0s1: Second sentence.
        p1s0: New paragraph.

    Note:
        - Segments are yielded in document order
        - No global state or caching
        - Same offset guarantees as split_with_offsets()
    """
    # For now, use the non-streaming implementation and yield
    # In the future, this could be optimized for true streaming
    segments = split_with_offsets(
        text,
        mode=mode,
        use_spacy=use_spacy,
        language_model=language_model,
        apply_corrections=apply_corrections,
        max_chars=max_chars,
        inline_markup=inline_markup,
    )
    yield from segments
