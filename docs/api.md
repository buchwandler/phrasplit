# API Reference

This page contains the complete API reference for phrasplit.

## spaCy Model Resolver

The stable resolver API is available directly from `phrasplit`:

```python
from phrasplit import (
    SpacyModelResolution,
    SpacyModelSize,
    SpacyModelAttempt,
    SpacyModelResolutionError,
    SpacyNotInstalledError,
    NoCompatibleSpacyModelError,
    ExplicitSpacyModelError,
    normalize_spacy_language,
    resolve_spacy_model,
)

resolution = resolve_spacy_model(language="en", require=False)
print(resolution.model)       # concrete selected package or None
print(resolution.model_size)  # sm, md, lg, or trf
print(resolution.attempts)    # local loadability diagnostics
```

Resolution is local-only and never downloads a model. `use_spacy=False` skips resolution
entirely; `use_spacy=True` requires a loadable result; `use_spacy=None` uses spaCy only
when a compatible result exists and otherwise uses regex splitting. Downstream consumers
can use `language`, `language_model`, and `model_size` with the same semantics as the
splitting APIs.

`SpacyModelResolutionError` is the base class for resolver failures. Forced resolution
raises `SpacyNotInstalledError` when spaCy is absent or `NoCompatibleSpacyModelError`
when no compatible installed model loads. An explicit package failure raises
`ExplicitSpacyModelError`; its `resolution` field contains attempts and accurate
`available`/`loadable` flags. `SpacyModelAttempt` records each candidate load outcome.

## Main Functions

```{eval-rst}
.. module:: phrasplit
```

### split_sentences

```{eval-rst}
.. autofunction:: phrasplit.split_sentences
```

**Example:**

```python
from phrasplit import split_sentences

text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
sentences = split_sentences(text)
# ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']

# Use simple mode (no spaCy required)
sentences = split_sentences(text, use_spacy=False)

# Automatic highest-available selection, exact overrides, and regex forcing
sentences = split_sentences(text, language="en")
sentences = split_sentences(text, language="de")
sentences = split_sentences(text, language_model="en_core_web_sm")
sentences = split_sentences(text, language="en", model_size="lg")

# split_on_colon is deprecated (kept for compatibility only)
text = "Note: This is important."
sentences = split_sentences(text, split_on_colon=False)
```

### split_clauses

```{eval-rst}
.. autofunction:: phrasplit.split_clauses
```

**Example:**

```python
from phrasplit import split_clauses

text = "I like coffee, and I like tea."
clauses = split_clauses(text)
# ['I like coffee,', 'and I like tea.']

# Use simple mode for faster processing
clauses = split_clauses(text, use_spacy=False)
```

### split_paragraphs

```{eval-rst}
.. autofunction:: phrasplit.split_paragraphs
```

**Example:**

```python
from phrasplit import split_paragraphs

text = "First paragraph.\n\nSecond paragraph."
paragraphs = split_paragraphs(text)
# ['First paragraph.', 'Second paragraph.']
```

### split_text

```{eval-rst}
.. autofunction:: phrasplit.split_text
```

**Example:**

```python
from phrasplit import split_text, Segment

text = "First sentence. Second sentence.\n\nNew paragraph."
segments = split_text(text, mode="sentence")

for seg in segments:
    print(f"P{seg.paragraph} S{seg.sentence}: {seg.text}")
# P0 S0: First sentence.
# P0 S1: Second sentence.
# P1 S0: New paragraph.

# Clause mode for finer granularity
text = "Hello, world.\n\nGoodbye, friend."
segments = split_text(text, mode="clause")
# Returns clauses with paragraph and sentence indices

# Use simple mode (no spaCy)
segments = split_text(text, mode="sentence", use_spacy=False)
```

### split_text_with_diagnostics

```{eval-rst}
.. autofunction:: phrasplit.split_text_with_diagnostics
```

Use this additive API when a caller needs the backend/model chosen by the actual split
operation. It returns a `SplitTextResult` with `segments` and frozen `SplitDiagnostics`:

```python
from phrasplit import split_text_with_diagnostics

result = split_text_with_diagnostics("Hello. World.", language="en")
print(result.diagnostics.backend)
print(result.diagnostics.selected_model)
print(result.diagnostics.selected_model_size)
```

`SplitDiagnostics.backend` is `"none"` for paragraph mode, `"regex"` for the regex
backend, and `"spacy"` for spaCy. `resolution` is `None` for forced regex and paragraph
mode; automatic fallback retains a resolution whose selected model is `None`. The
existing `split_text()` function remains a compatibility wrapper returning
`list[Segment]`.

### split_long_lines

```{eval-rst}
.. autofunction:: phrasplit.split_long_lines
```

**Example:**

```python
from phrasplit import split_long_lines

text = "This is a very long sentence that needs to be split into smaller parts."
lines = split_long_lines(text, max_length=40)

# Use simple mode
lines = split_long_lines(text, max_length=40, use_spacy=False)
```

### Offset APIs

```{eval-rst}
.. autofunction:: phrasplit.split_with_offsets_with_diagnostics
.. autofunction:: phrasplit.split_with_offsets
.. autofunction:: phrasplit.iter_split_with_offsets
```

Use `split_with_offsets_with_diagnostics()` when an integration needs both exact source
offsets and the backend/model chosen by the operation that produced them. It returns a
`SplitWithOffsetsResult` containing `segments` and `SplitDiagnostics`:

```python
from phrasplit import split_with_offsets_with_diagnostics

text = "Hello. World."
result = split_with_offsets_with_diagnostics(text, language="en")

print(result.diagnostics.backend)
print(result.diagnostics.selected_model)

for segment in result.segments:
    assert segment.text == text[segment.char_start:segment.char_end]
```

`split_with_offsets()` remains a compatibility wrapper returning `list[SplitSegment]`.
Both APIs preserve the exact-slice invariant. `inline_markup=True` is regex-only and
raises `ValueError` when explicitly combined with spaCy. The iterator is currently a
facade over the list API and does not claim incremental or bounded-memory processing.

## Data Types

### Segment

```{eval-rst}
.. autoclass:: phrasplit.Segment
   :members:
   :undoc-members:
```

A named tuple representing a text segment with position information.

**Fields:**

- `text` (str): The text content of the segment
- `paragraph` (int): Paragraph index (0-based) within the document
- `sentence` (int | None): Sentence index (0-based) within the paragraph. None for
  paragraph mode.

**Example:**

```python
from phrasplit import split_text, Segment

segments = split_text("Hello world.", mode="sentence")
seg = segments[0]

# Access by name
print(seg.text)       # "Hello world."
print(seg.paragraph)  # 0
print(seg.sentence)   # 0

# Access by index
print(seg[0])  # "Hello world."
print(seg[1])  # 0
print(seg[2])  # 0

# Unpack
text, para, sent = seg
```

## Module Contents

### splitter module

```{eval-rst}
.. automodule:: phrasplit.splitter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _get_nlp, _protect_ellipsis, _restore_ellipsis, _split_sentence_into_clauses, _split_at_clauses, _hard_split, _split_at_boundaries
```

## Type Information

phrasplit is fully typed and includes a `py.typed` marker file for PEP 561 compliance.
You can use it with mypy and other type checkers.

Function signatures:

```python
from typing import NamedTuple

class Segment(NamedTuple):
    text: str
    paragraph: int
    sentence: int | None = None

def split_sentences(
    text: str,
    language_model: str | None = None,
    apply_corrections: bool = True,
    split_on_colon: bool = True,
    use_spacy: bool | None = None,
    *,
    language: str = "en",
    model_size: str | None = None,
) -> list[str]: ...

def split_clauses(
    text: str,
    language_model: str | None = None,
    use_spacy: bool | None = None,
    *,
    language: str = "en",
    model_size: str | None = None,
) -> list[str]: ...

def split_paragraphs(text: str) -> list[str]: ...

def split_text(
    text: str,
    mode: str = "sentence",
    language_model: str | None = None,
    apply_corrections: bool = True,
    split_on_colon: bool = True,
    use_spacy: bool | None = None,
    *,
    language: str = "en",
    model_size: str | None = None,
) -> list[Segment]: ...

def split_long_lines(
    text: str,
    max_length: int,
    language_model: str | None = None,
    use_spacy: bool | None = None,
    *,
    language: str = "en",
    model_size: str | None = None,
) -> list[str]: ...
```

## Injected spaCy analysis

The splitter can consume analysis prepared by an orchestrator. `doc` must expose `text`,
`sents`, and sentence spans with `start_char` and `end_char`; it is checked against the
exact input string. `nlp` is a caller-owned callable pipeline used when `doc` is not
supplied.

```python
prepared = spokenform.prepare(raw_text)
doc = nlp(prepared.spoken_text)
result = split_with_offsets_with_diagnostics(
    prepared.spoken_text, language="de", doc=doc
 )
```

A supplied `doc` is preferred over `nlp`, bypasses model resolution and inference, and
cannot be combined with `use_spacy=False`. Diagnostics report `analysis_source` as
`provided-document` or `provided-pipeline`, set `model_owned_by_caller` to `True`, and
leave `selected_model` unset when no model identity is available. Phrasplit still
applies its corrections and exact-slice offset projection.

Phrasplit only reads injected resources during the call. It does not mutate, cache,
close, unload, or take ownership of a supplied document or pipeline.
