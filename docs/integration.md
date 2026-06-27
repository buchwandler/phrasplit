# Integration Contract for Pipelines

## Overview

This document specifies the contract and guarantees provided by `split_with_offsets()`
for integration into text processing pipelines, particularly for TTS synthesis, token
alignment, and markup processing.

## Core Guarantee: Exact-Slice Policy

### The Invariant

For every segment returned by `split_with_offsets()`, the following invariant **ALWAYS**
holds:

```python
segment.text == input_text[segment.char_start:segment.char_end]
```

This is called the **exact-slice policy**.

### What This Means

1. **No Stripping**: Segment text is never stripped, trimmed, or normalized
2. **Precise Mapping**: Offsets point to the exact substring in the original input
3. **Whitespace Preserved**: Leading/trailing whitespace from the original text is
   included
4. **Safe for Slicing**: You can reliably extract segments using Python slicing
5. **Monotonic Order**: Segments are emitted in order with non-overlapping ranges

### Example

```python
from phrasplit import split_with_offsets

text = "  Hello world.  \n\n  New paragraph.  "
segments = split_with_offsets(text, mode="sentence")

for seg in segments:
    # This will ALWAYS pass
    assert text[seg.char_start:seg.char_end] == seg.text

    # Segments may include whitespace from original text
    print(repr(seg.text))
# Output might be: 'Hello world.' or '  Hello world.' depending on boundaries
```

## Coordinate Space

### Character Offsets

- All `char_start` and `char_end` values are in the **coordinate space of the input
  text**
- Offsets are 0-based with exclusive end positions (Python slice convention)
- Offsets are measured in Unicode code points (characters), not bytes

### SSMD/Markup Workflows

When integrating with SSMD or other markup systems, offsets refer to whichever version
of the text you pass in:

**Option 1: Segment raw markup**

```python
text_with_markup = "Hello [world]{lang='de'}. How are you?"
segments = split_with_offsets(text_with_markup)
# Offsets are in the coordinate space of text_with_markup
```

**Option 2: Segment after escaping**

```python
import ssmd  # hypothetical
escaped = ssmd.escape(text_with_markup)
segments = split_with_offsets(escaped)
# Offsets are in the coordinate space of escaped text
# Use these offsets to slice escaped text, then unescape each segment
```

## Determinism and Stability

### Stable IDs

Segment IDs are **deterministic** and **stable** across runs:

- Same input text + same parameters → same IDs
- ID format: `p{para}s{sent}c{clause}` or `p{para}s{sent}c{clause}:m{index}`
- IDs only change if segmentation rules change

### Deterministic Splitting

`split_with_offsets()` is fully deterministic:

- Same input always produces same segments
- Same offsets, same text, same IDs
- No randomness or runtime variation

## Max-Chars Safety Splitting

### Behavior

When `max_chars` is specified:

1. Segments exceeding `max_chars` are split further
2. Splitting happens at whitespace or punctuation boundaries
3. **Exact-slice policy is maintained** even after splitting
4. Sub-segments get IDs like `p0s1:m0`, `p0s1:m1`, etc.

### Example

```python
long_text = "This is a very long sentence that needs splitting."
segments = split_with_offsets(long_text, max_chars=20)

for seg in segments:
    assert len(seg.text) <= 20  # Respects max_chars
    assert long_text[seg.char_start:seg.char_end] == seg.text  # Invariant holds
    print(f"{seg.id}: {seg.text!r}")
# Output:
# p0s0:m0: 'This is a very long '
# p0s0:m1: 'sentence that needs '
# p0s0:m2: 'splitting.'
```

### Guarantee

- All resulting segments have `len(seg.text) <= max_chars` (except single words/tokens
  that exceed the limit)
- The exact-slice invariant holds for all sub-segments
- IDs are stable and deterministic

## Integration Examples

### TTS Pipeline

```python
from phrasplit import split_with_offsets

# Segment text
text = "Long document text..."
segments = split_with_offsets(text, mode="sentence", max_chars=500)

# Process each segment
for seg in segments:
    # Offsets allow you to track position in original
    audio = tts_engine.synthesize(seg.text)

    # Store mapping from audio to original text position
    audio_segments.append({
        "audio": audio,
        "text_start": seg.char_start,
        "text_end": seg.char_end,
        "text": seg.text,
        "id": seg.id
    })
```

### Token Alignment

```python
from phrasplit import split_with_offsets

text = "Hello world. How are you?"
segments = split_with_offsets(text, mode="sentence")

for seg in segments:
    # Tokenize the segment
    tokens = tokenizer.tokenize(seg.text)

    # Token offsets are relative to seg.text
    for token in tokens:
        # Map to absolute position in original text
        abs_start = seg.char_start + token.char_start
        abs_end = seg.char_start + token.char_end

        # Verify
        assert text[abs_start:abs_end] == token.text
```

### SSMD Span Slicing

```python
from phrasplit import split_with_offsets, COMMON_PATTERNS, validate_no_placeholder_breaks

ssmd_text = "Hello [world]{lang='de'}. How are [you]{emphasis}?"
segments = split_with_offsets(ssmd_text, mode="sentence")

# Validate placeholders weren't broken
warnings = validate_no_placeholder_breaks(
    ssmd_text,
    segments,
    placeholder_pattern=COMMON_PATTERNS["ssmd"]
)

if not warnings:
    # Safe to slice - placeholders are intact
    for seg in segments:
        # Extract SSMD markup for this segment
        ssmd_slice = ssmd_text[seg.char_start:seg.char_end]
        # Process with SSMD parser...
```

## Error Handling

### Invalid Inputs

The function validates inputs and raises exceptions for invalid parameters:

```python
# ValueError if max_chars < 1
split_with_offsets(text, max_chars=0)  # raises ValueError

# ValueError if mode is invalid
split_with_offsets(text, mode="invalid")  # raises ValueError

# ImportError if spaCy requested but not available
split_with_offsets(text, use_spacy=True)  # raises ImportError if no spaCy
```

### Empty or Whitespace-Only Text

```python
segments = split_with_offsets("")
assert segments == []

segments = split_with_offsets("   \n\n   ")
assert segments == []  # No non-whitespace content
```

## Performance Considerations

### Backend Selection

- **Regex backend** (`use_spacy=False`): 60x faster, good for simple text
- **spaCy backend** (`use_spacy=True`): More accurate, better for complex text
- **Auto-detect** (`use_spacy=None`, default): Uses spaCy if installed

### Memory Usage

- `split_with_offsets()`: Returns all segments at once (O(n) memory)
- `iter_split_with_offsets()`: Streaming iterator (O(1) memory per segment)

For large documents (> 1 MB), consider using the iterator:

```python
from phrasplit import iter_split_with_offsets

for segment in iter_split_with_offsets(large_text, max_chars=500):
    process(segment)  # Stream processing
```

## Versioning and Compatibility

### Semantic Versioning

- The exact-slice invariant is part of the API contract
- Breaking this invariant would be a major version change
- IDs, offsets, and segmentation behavior follow semver

### Backward Compatibility

New code using `split_with_offsets()` should:

- Always verify the invariant in tests:
  `assert text[seg.char_start:seg.char_end] == seg.text`
- Not assume segments are stripped or normalized
- Handle segments that may contain leading/trailing whitespace

## Summary

### Key Guarantees

1. ✅ **Exact-slice invariant**: `text[seg.char_start:seg.char_end] == seg.text`
2. ✅ **Deterministic**: Same input → same output
3. ✅ **Stable IDs**: IDs don't change across runs
4. ✅ **Coordinate space**: Offsets in original input text
5. ✅ **Max-chars safety**: Invariant holds even with splitting

### Best Practices

- Always test the exact-slice invariant in your integration tests
- Don't strip or normalize segment text if you need to use offsets later
- Use `max_chars` to ensure segments fit within processing constraints
- Validate markup integrity with `validate_no_placeholder_breaks()`

### References

- [Offset Coordinate System](offsets.md)
- [Streaming API](streaming.md)
- [SSMD Integration Examples](offsets.md#integration-with-markup-languages)
