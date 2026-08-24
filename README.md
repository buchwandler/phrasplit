[![PyPI - Version](https://img.shields.io/pypi/v/phrasplit)](https://pypi.org/project/phrasplit/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/phrasplit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/phrasplit)
[![codecov](https://codecov.io/gh/buchwandler/phrasplit/graph/badge.svg?token=iCHXwbjAXG)](https://codecov.io/gh/buchwandler/phrasplit)

# phrasplit

A Python library for splitting text into sentences, clauses, or paragraphs. Choose
between spaCy NLP for best accuracy or fast regex-based splitting for simple use cases.

## Features

- **Two modes**: spaCy (accurate) or simple regex (fast, no ML dependencies)
- **Sentence splitting**: Intelligent sentence boundary detection
- **Clause splitting**: Split sentences at commas for natural pause points
- **Paragraph splitting**: Split text at double newlines (no spaCy needed)
- **Hierarchical splitting**: Split text with paragraph/sentence position tracking
- **Long line splitting**: Break long lines at sentence/clause boundaries
- **Abbreviation handling**: Correctly handles Mr., Dr., U.S.A., etc.
- **Ellipsis support**: Preserves ellipses without incorrect splitting
- **Offset-preserving segmentation**: Exact-slice offsets with stable IDs
- **25+ languages**: Multi-language abbreviation support

## Installation

### Basic Installation (Simple Mode)

For fast, regex-based splitting without ML dependencies:

```bash
pip install phrasplit
```

### Full Installation (spaCy Mode)

For best accuracy with complex text, install with spaCy:

```bash
pip install phrasplit[nlp]
# Install one or more local spaCy models (optional; automatic mode never downloads)
python -m spacy download en_core_web_sm
```

## Performance Comparison

| Mode       | Speed       | Accuracy | Dependencies            | When to Use                      |
| ---------- | ----------- | -------- | ----------------------- | -------------------------------- |
| **Simple** | ~60x faster | ~85-90%  | None (regex only)       | Simple text, speed-critical apps |
| **spaCy**  | Baseline    | ~95%+    | spaCy + models (~500MB) | Complex text, best accuracy      |

Illustrative benchmark results (1000 sentences; recorded before 0.3.4 without a retained
hardware, Python, or model-version manifest):

- spaCy: 1091ms
- Simple: 17ms (63x faster)

Both modes produce nearly identical results for well-formatted text.

## Quick Start

### Auto-Detection (Recommended)

Phrasplit automatically uses the highest-quality installed and loadable spaCy model for
the requested language (`trf > lg > md > sm`), otherwise it falls back to simple mode.
Automatic selection never downloads models.

```python
from phrasplit import split_sentences, split_clauses, split_paragraphs

# Uses the highest installed English model, otherwise simple mode
text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
sentences = split_sentences(text)
# ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']

# Force simple mode (even if spaCy is installed)
sentences = split_sentences(text, use_spacy=False)

# Force spaCy mode (requires spaCy and a compatible local model)
sentences = split_sentences(text, use_spacy=True)
```

Model selection is language-based and accepts exact overrides:

```python
from phrasplit import split_text

split_text(text, language="en")                         # highest installed English model
split_text(text, language="de")                         # highest installed German model
split_text(text, language_model="en_core_web_sm")       # exact package
split_text(text, language="en", model_size="lg")        # exact tier, no fallback
split_text(text, language="en", use_spacy=False)        # force regex
```

For integrations that need to report which backend and model were actually used, use the
additive detailed API. It performs backend/model resolution once and leaves
`split_text()`'s list return contract unchanged:

```python
from phrasplit import split_text_with_diagnostics

result = split_text_with_diagnostics(
    "Hello. World.", language="en", use_spacy=None
)
print(result.diagnostics.backend)
print(result.diagnostics.selected_model)
for segment in result.segments:
    print(segment.text)
```

### Python API

```python
from phrasplit import split_sentences, split_clauses, split_paragraphs, split_long_lines

# Split text into sentences
text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
sentences = split_sentences(text)
# ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']

# Split sentences into comma-separated parts (for audiobook pauses)
text = "I like coffee, and I like tea."
clauses = split_clauses(text)
# ['I like coffee,', 'and I like tea.']

# Split text into paragraphs (no spaCy needed)
text = "First paragraph.\n\nSecond paragraph."
paragraphs = split_paragraphs(text)
# ['First paragraph.', 'Second paragraph.']

# Split long lines at natural boundaries
text = "This is a very long sentence that needs to be split."
lines = split_long_lines(text, max_length=30)
```

### Hierarchical Splitting with Position Tracking

For audiobook generation where you need different pause lengths between paragraphs,
sentences, and clauses, use `split_text()`:

```python
from phrasplit import split_text, Segment

# Split into sentences with paragraph tracking
text = "First sentence. Second sentence.\n\nNew paragraph here."
segments = split_text(text, mode="sentence")

for seg in segments:
    print(f"P{seg.paragraph} S{seg.sentence}: {seg.text}")
# P0 S0: First sentence.
# P0 S1: Second sentence.
# P1 S0: New paragraph here.

# Detect paragraph changes for longer pauses
for i, seg in enumerate(segments):
    if i > 0 and seg.paragraph != segments[i-1].paragraph:
        print("--- paragraph break (add longer pause) ---")
    print(seg.text)
```

Available modes:

- `"paragraph"`: Returns paragraphs (sentence=None)
- `"sentence"`: Returns sentences with paragraph index
- `"clause"`: Returns clauses with paragraph and sentence indices

### Offset-Preserving Segmentation

For TTS pipelines, markup processing, and token alignment where exact character
positions are critical, use `split_with_offsets()`:

```python
from phrasplit import split_with_offsets

text = "Hello world. How are you?\n\nNew paragraph."
segments = split_with_offsets(text, mode="sentence")

for seg in segments:
    # Exact-slice invariant: text[char_start:char_end] == seg.text
    assert text[seg.char_start:seg.char_end] == seg.text
    print(f"{seg.id}: {seg.text!r} [{seg.char_start}:{seg.char_end}]")

# Output:
# p0s0: 'Hello world.' [0:12]
# p0s1: 'How are you?' [13:25]
# p1s0: 'New paragraph.' [27:41]
```

**Exact-Slice Policy**

`split_with_offsets()` implements an exact-slice policy that guarantees:

- `segment.text == text[segment.char_start:segment.char_end]` **always** holds
- No whitespace stripping or normalization breaks this mapping
- Offsets are safe for span slicing, token alignment, and markup integration
- Deterministic and stable across runs
- Offsets are computed against the original input text
- Offsets are monotonic and non-overlapping

**Safety Splitting with `max_chars`**

```python
long_text = "word " * 100
segments = split_with_offsets(long_text, max_chars=50)

# All segments respect max length
assert all(len(seg.text) <= 50 for seg in segments)

# Exact-slice invariant still holds
assert all(long_text[s.char_start:s.char_end] == s.text for s in segments)

# IDs are stable: "p0s0:m0", "p0s0:m1", etc.
print([s.id for s in segments])
```

**Integration with SSMD and Markup**

```python
from phrasplit import split_with_offsets, validate_no_placeholder_breaks, COMMON_PATTERNS

# Example: segment text with SSMD markup
text_with_markup = "Hello [world]{lang='de'}. How are you?"

# Option 1: Segment first, then validate
segments = split_with_offsets(text_with_markup, mode="sentence")
warnings = validate_no_placeholder_breaks(
    text_with_markup,
    segments,
    placeholder_pattern=COMMON_PATTERNS["ssmd"]
)

# Option 2: Escape markup, segment, then unescape each segment
# (See docs/offsets.md for detailed workflow)
```

### Command Line Interface

```bash
# Split into sentences (selects a compatible local model or uses regex)
phrasplit sentences input.txt -o output.txt

# Force simple mode (60x faster, no spaCy required)
phrasplit sentences input.txt --simple

# Split into clauses
phrasplit clauses input.txt -o output.txt

# Use simple mode for clauses (faster)
phrasplit clauses input.txt --simple -o output.txt

# Split into paragraphs (no spaCy needed)
phrasplit paragraphs input.txt -o output.txt

# Split long lines (default max 80 characters)
phrasplit longlines input.txt -o output.txt --max-length 60

# Long lines with simple mode
phrasplit longlines input.txt --simple --max-length 60

# Use a different spaCy model (only for spaCy mode)
phrasplit sentences input.txt --model en_core_web_lg

# Read from stdin (pipe or redirect)
echo "Hello world. This is a test." | phrasplit sentences
cat input.txt | phrasplit clauses --simple -o output.txt

# Explicit stdin with dash
phrasplit sentences - < input.txt
```

## API Reference

### `split_sentences(text, language_model=None, apply_corrections=True, use_spacy=None, language="en", model_size=None)`

Split text into sentences.

**Parameters:**

- `text`: Input text string
- `language`: Independent language hint for automatic selection and abbreviations
- `language_model`: Optional exact model package; `None`, `""`, and `"auto"` select
  automatically
- `model_size`: Optional exact tier (`sm`, `md`, `lg`, or `trf`), with no fallback
- `apply_corrections`: Apply post-processing corrections for URLs and abbreviations
  (default: True, only applies to spaCy mode)
- `use_spacy`: Choose implementation:
  - `None` (default): Use the best installed compatible model, or regex if none is
    loadable
  - `True`: Require spaCy and a compatible loadable model
  - `False`: Force simple regex mode

**Returns:** List of sentences

**Raises:** `SpacyNotInstalledError` when forced spaCy is not installed,
`NoCompatibleSpacyModelError` when no compatible model loads, or
`ExplicitSpacyModelError` when an explicit model fails.

### `split_clauses(text, language_model=None, use_spacy=None, language="en", model_size=None)`

Split text into comma-separated parts. Useful for creating natural pause points in
audiobook/TTS applications.

**Parameters:**

- `text`: Input text string
- `language`: Independent language hint for model discovery and abbreviations
- `language_model`: Optional exact model package
- `model_size`: Optional exact model tier
- `use_spacy`: Choose implementation (default: None for auto-detect)

**Returns:** List of clauses (comma stays at end of each part)

### `split_paragraphs(text)`

Split text into paragraphs at double newlines. Works without spaCy.

**Parameters:**

- `text`: Input text string

**Returns:** List of paragraphs

### `split_text(text, mode="sentence", language_model=None, apply_corrections=True, use_spacy=None, language="en", model_size=None)`

Split text into segments with hierarchical position information.

**Parameters:**

- `text`: Input text string
- `mode`: Splitting mode - "paragraph", "sentence", or "clause"
- `language`: Independent language hint for automatic model selection
- `language_model`: Optional exact model package
- `model_size`: Optional exact model tier
- `apply_corrections`: Apply post-processing corrections (default: True)
- `use_spacy`: Choose implementation (default: None for auto-detect)

**Returns:** List of `Segment` namedtuples with fields:

- `text`: The segment text
- `paragraph`: Paragraph index (0-based)
- `sentence`: Sentence index within paragraph (0-based), None for paragraph mode

### `split_long_lines(text, max_length, language_model=None, use_spacy=None, language="en", model_size=None)`

Split lines exceeding max_length at sentence/clause boundaries.

**Parameters:**

- `text`: Input text string
- `max_length`: Maximum line length in characters (must be >= 1)
- `language`: Independent language hint for automatic model selection
- `language_model`: Optional exact model package
- `model_size`: Optional exact model tier
- `use_spacy`: Choose implementation (default: None for auto-detect)

**Returns:** List of lines, each within max_length (except single words exceeding limit)

**Raises:** `ValueError` if max_length is less than 1

### `split_with_offsets(text, mode="sentence", use_spacy=None, language_model=None, language="en", model_size=None, inline_markup=False)`

Returns exact-slice segments with stable IDs and character offsets. `inline_markup=True`
is an opt-in regex-only mode for balanced inline XHTML tags; passing it with
`use_spacy=True` raises `ValueError`.

```python
from phrasplit import split_with_offsets

text = "One. <em>Two.</em> Three."
segments = split_with_offsets(text, use_spacy=False, inline_markup=True)
assert all(text[s.char_start:s.char_end] == s.text for s in segments)
```

### `iter_split_with_offsets(...)`

This iterator currently delegates to `split_with_offsets()` and materializes the full
list before yielding. It preserves exact offsets but does not yet provide bounded-memory
or incremental streaming behavior.

## Use Cases

### Audiobook Creation

Split text with paragraph awareness for different pause lengths:

```python
from phrasplit import split_text

text = """When the sun rose, the birds began to sing.

A new day had started. The adventure continues."""

segments = split_text(text, mode="clause")

for i, seg in enumerate(segments):
    # Add longer pause between paragraphs
    if i > 0 and seg.paragraph != segments[i-1].paragraph:
        add_pause(duration=1.0)  # Long pause for paragraph
    # Add medium pause between sentences
    elif i > 0 and seg.sentence != segments[i-1].sentence:
        add_pause(duration=0.5)  # Medium pause for sentence
    else:
        add_pause(duration=0.2)  # Short pause for clause

    synthesize_speech(seg.text)
```

### Subtitle Generation

Split long lines to fit subtitle constraints:

```python
from phrasplit import split_long_lines

text = "This is a very long sentence that would not fit on a single subtitle line."
lines = split_long_lines(text, max_length=42)
```

### Text Processing Pipelines

```python
from phrasplit import split_paragraphs, split_sentences

text = open("book.txt").read()

for paragraph in split_paragraphs(text):
    for sentence in split_sentences(paragraph):
        process(sentence)
```

## Requirements

- Python 3.10+
- click 8.0+
- rich 13.0+
- spaCy 3.5+ (optional, for best accuracy)

## Choosing Between Modes

### Use Simple Mode When:

- Processing simple, well-formatted text
- Speed is critical (60-100x faster)
- Deploying in constrained environments (no ML dependencies)
- Installing spaCy models is not feasible (~500MB per language)

### Use spaCy Mode When:

- Processing complex, informal, or poorly formatted text
- Accuracy is paramount (5-10% better)
- Already using spaCy in your pipeline
- Working with academic or literary texts

## Migration Guide

### Upgrading from Previous Versions

Version 0.3.4 keeps spaCy optional while making automatic model selection local-only.
Existing code continues to work:

```python
# Existing code (still works, auto-selects a compatible installed model or regex)
from phrasplit import split_sentences
sentences = split_sentences(text)

# New: Explicit control
sentences = split_sentences(text, use_spacy=False)  # Force simple
sentences = split_sentences(text, use_spacy=True)   # Force spaCy
```

The `split_on_colon` parameter is deprecated and will be removed in a future version.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
