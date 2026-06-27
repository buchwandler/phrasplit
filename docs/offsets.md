# Offset-Preserving Segmentation

## Overview

The offset-preserving segmentation API provides precise character-level mapping between
segments and the original input text. This is essential for downstream processing tasks
like text-to-speech synthesis, where you need to maintain exact correspondence between
processed segments and source text.

## Key Features

1. **Character Offsets**: Each segment includes `char_start` and `char_end` that map
   exactly to the original input
2. **Stable IDs**: Hierarchical identifiers (e.g., `p0s1c2`) that are stable across runs
3. **Exact Slices**: Segment text always equals the exact input slice (no normalization)
4. **Monotonic Offsets**: Segments are ordered with non-overlapping ranges
5. **Backend Agnostic**: Works with both spaCy (accurate) and regex (fast) backends

## Coordinate System

All character offsets use Python's standard 0-based indexing with exclusive end
positions:

```python
text = "Hello world. How are you?"
segment = SplitSegment(
    id="p0s0",
    text="Hello world.",
    char_start=0,
    char_end=12,  # exclusive
    ...
)

# Verify offset mapping
assert text[segment.char_start:segment.char_end] == segment.text
```

## Offset Guarantees

**Exact Mapping**

The fundamental guarantee is that for any segment:

```python
original_text[segment.char_start:segment.char_end] == segment.text
```

This holds true even with:

- Multiple paragraphs
- Complex punctuation
- Unicode characters
- Whitespace variations

Offsets are always computed against the original input text, even when the splitter
applies internal corrections (ellipsis protection, abbreviation fixes).

**Monotonic, Non-Overlapping Offsets**

Segments are emitted in document order and do not overlap:

```python
last_end = 0
for seg in segments:
    assert seg.char_start >= last_end
    last_end = seg.char_end
```

**Runtime Validation**

`split_with_offsets()` validates its output (bounds, exact-slice, and ordering). If any
invariant is violated, it raises `ValueError` with details about the invalid segment.

**Whitespace Handling**

Segment text is always an exact slice of the original input. The splitter skips
leading/trailing whitespace at paragraph and sentence boundaries, so whitespace between
segments may be excluded. Compare offsets using direct equality:

```python
extracted = text[segment.char_start:segment.char_end]
assert segment.text == extracted
```

## Usage Examples

### Basic Usage

```python
from phrasplit import split_with_offsets

text = "Hello world. How are you?\\n\\nNew paragraph."
segments = split_with_offsets(text, mode="sentence")

for seg in segments:
    print(f"{seg.id}: {seg.text}")
    # Verify offset
    assert text[seg.char_start:seg.char_end] == seg.text
```

### Modes

**Paragraph Mode**

```python
segments = split_with_offsets(text, mode="paragraph")
# Returns one segment per paragraph
```

**Sentence Mode** (default)

```python
segments = split_with_offsets(text, mode="sentence")
# Returns one segment per sentence
```

**Clause Mode**

```python
segments = split_with_offsets(text, mode="clause")
# Returns one segment per comma-separated clause
```

### Backend Selection

```python
# Auto-detect (uses spaCy if available)
segments = split_with_offsets(text)

# Force spaCy (more accurate)
segments = split_with_offsets(text, use_spacy=True)

# Force regex (faster, no dependencies)
segments = split_with_offsets(text, use_spacy=False)
```

### Max Length Splitting

```python
# Split long segments at word boundaries
segments = split_with_offsets(text, max_chars=100)

# All segments will be ≤ 100 characters
assert all(len(s.text) <= 100 for s in segments)

# Offsets still work correctly
for seg in segments:
    assert text[seg.char_start:seg.char_end] == seg.text
```

## Working with Offsets

### Extract Surrounding Context

```python
segment = segments[0]

# Get 20 characters before segment
context_start = max(0, segment.char_start - 20)
before = text[context_start:segment.char_start]

# Get 20 characters after segment
context_end = min(len(text), segment.char_end + 20)
after = text[segment.char_end:context_end]
```

### Reconstruct Original Text

```python
# Segments don't overlap, but may not cover all text
# (whitespace between paragraphs may be skipped)

reconstructed_parts = []
last_end = 0

for seg in segments:
    # Add any gap
    if seg.char_start > last_end:
        reconstructed_parts.append(text[last_end:seg.char_start])

    # Add segment
    reconstructed_parts.append(seg.text)
    last_end = seg.char_end

# Add any trailing text
if last_end < len(text):
    reconstructed_parts.append(text[last_end:])

reconstructed = "".join(reconstructed_parts)
```

### Map Edits Back to Original

```python
# Process segments (e.g., TTS synthesis)
processed = []
for seg in segments:
    result = some_processing(seg.text)
    processed.append({
        "original_position": (seg.char_start, seg.char_end),
        "original_text": seg.text,
        "result": result
    })
```

## Edge Cases

### Empty Text

```python
segments = split_with_offsets("")
assert segments == []
```

### Whitespace-Only Text

```python
segments = split_with_offsets("   \\n\\n   ")
assert segments == []
```

### No Sentence Breaks

```python
text = "Just one sentence"
segments = split_with_offsets(text, mode="sentence")
assert len(segments) == 1
assert segments[0].char_start == 0
```

### Unicode and Special Characters

```python
text = "Café résumé. Über große Möbel."
segments = split_with_offsets(text)

# Offsets work correctly with multibyte characters
for seg in segments:
    assert text[seg.char_start:seg.char_end] == seg.text
```

## Boundary Whitespace

Segments are anchored to the first and last non-whitespace characters inside each
paragraph. Whitespace between paragraphs or sentences may not be included in any
segment, but the exact-slice invariant always holds.

```python
text = "  Hello.  \\n\\n  World.  "
segments = split_with_offsets(text, mode="sentence")

assert segments[0].text == "Hello."
assert segments[1].text == "World."

for seg in segments:
    extracted = text[seg.char_start:seg.char_end]
    assert seg.text == extracted
```

## Integration with Markup Languages

When working with text containing markup (SSMD, Speech Markdown, Mustache templates,
etc.), you need to ensure segmentation doesn't break markup tags or placeholders.

### Common Pattern Library

The `COMMON_PATTERNS` dictionary provides regex patterns for common markup formats:

```python
from phrasplit import COMMON_PATTERNS

# Available patterns:
# - "ssmd": SSMD format [text]{lang="de"}
# - "speechmarkdown": Speech Markdown ((text)[key:"value";...])
# - "mustache": Mustache templates {{variable}}
# - "html_tag": HTML/XML tags <tag>...</tag>
# - "markdown_link": Markdown links [text](url)
```

### SSMD Integration Example

For SSMD (Structured Synthesis Markup Document) escaped text:

```python
from phrasplit import (
    split_with_offsets,
    validate_no_placeholder_breaks,
    COMMON_PATTERNS
)
import ssmd  # hypothetical SSMD library

# 1. Escape markup before splitting
original = "Hello [world]{lang='de'}. How are you?"
escaped = ssmd.escape_ssmd_syntax(original)

# 2. Split with offsets
segments = split_with_offsets(escaped, mode="sentence")

# 3. Validate that placeholders weren't broken
warnings = validate_no_placeholder_breaks(
    escaped,
    segments,
    placeholder_pattern=COMMON_PATTERNS["ssmd"]
)
if warnings:
    for w in warnings:
        print(f"Warning: {w}")

# 4. Unescape each segment
for seg in segments:
    unescaped = ssmd.unescape_ssmd_syntax(seg.text)
    # Process unescaped text...
```

### Custom Markup Patterns

For custom markup formats, provide your own regex pattern:

```python
from phrasplit import split_with_offsets, validate_no_placeholder_breaks

# Custom pattern for your markup format
my_pattern = r"\[\[([^\]]+)\]\]"  # Matches [[placeholder]]

text = "Hello [[name]]. Welcome to [[location]]."
segments = split_with_offsets(text, mode="sentence")

# Validate with custom pattern
warnings = validate_no_placeholder_breaks(
    text,
    segments,
    placeholder_pattern=my_pattern
)
```

### Choosing Safe Splitting Mode

Use `suggest_splitting_mode()` to find a safe mode that avoids breaking markup:

```python
from phrasplit import suggest_splitting_mode, COMMON_PATTERNS

text = "First sentence with [markup]{lang='de'}. Second sentence."

# Get suggested mode that won't break SSMD placeholders
mode = suggest_splitting_mode(
    text,
    placeholder_pattern=COMMON_PATTERNS["ssmd"]
)
print(f"Recommended mode: {mode}")

# Use the suggested mode
segments = split_with_offsets(text, mode=mode)
```

## See Also

- {doc}`streaming` - Streaming iterator API
- {doc}`api` - Complete API reference
- {doc}`examples` - More examples
