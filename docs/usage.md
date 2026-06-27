# Usage Guide

This guide covers how to use phrasplit's Python API for text splitting.

## Splitting Sentences

The {func}`~phrasplit.split_sentences` function intelligently detects sentence
boundaries. It can use either spaCy's NLP pipeline (high accuracy) or regex-based
splitting (lightweight, no dependencies):

```python
from phrasplit import split_sentences

text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
sentences = split_sentences(text)
print(sentences)
# ['Dr. Smith is here.', 'She has a Ph.D. in Chemistry.']
```

The function correctly handles:

- **Abbreviations**: Mr., Mrs., Dr., Prof., etc.
- **Acronyms**: U.S.A., U.K., etc.
- **Titles**: Ph.D., M.D., etc.
- **URLs**: www.example.com
- **Ellipses**: Text with... ellipses

Example with abbreviations:

```python
text = "Mr. Brown met Prof. Green. They discussed the U.S.A. case."
sentences = split_sentences(text)
# ['Mr. Brown met Prof. Green.', 'They discussed the U.S.A. case.']
```

### Choosing Processing Mode

phrasplit automatically detects if spaCy is available and uses the appropriate mode. You
can explicitly control this with the `use_spacy` parameter:

```python
# Automatic detection (default)
sentences = split_sentences(text)
# Uses spaCy if available, otherwise falls back to regex

# Force simple mode (no spaCy required)
sentences = split_sentences(text, use_spacy=False)
# Uses regex-based splitting, faster and lightweight

# Force spaCy mode (will error if spaCy not installed)
sentences = split_sentences(text, use_spacy=True)
# Uses spaCy NLP for higher accuracy
```

**When to use simple mode (use_spacy=False):**

- You don't have spaCy installed
- You need faster processing for simple text
- You want to minimize dependencies and memory usage
- Your text has clear sentence boundaries

**When to use spaCy mode (use_spacy=True):**

- You need the highest accuracy
- Your text has complex abbreviations or edge cases
- You're processing professional/academic content
- Quality is more important than speed

### Colon Splitting (Deprecated)

The `split_on_colon` parameter is deprecated and no longer affects output. It is
accepted for API compatibility and raises a `DeprecationWarning` when set to `False`.
Colon handling is delegated to the active backend (spaCy or the regex fallback).

## Splitting Clauses

The {func}`~phrasplit.split_clauses` function splits text at commas, creating natural
pause points ideal for audiobook and text-to-speech applications. Like
{func}`~phrasplit.split_sentences`, it supports both spaCy and simple modes:

```python
from phrasplit import split_clauses

text = "I like coffee, and I like tea."

# Automatic mode detection
clauses = split_clauses(text)
print(clauses)
# ['I like coffee,', 'and I like tea.']

# Force simple mode
clauses = split_clauses(text, use_spacy=False)
# Uses regex-based clause splitting
```

The comma is kept at the end of each clause, preserving the original punctuation.

More complex example:

```python
text = "When the sun rose, the birds began to sing, and the day started."
clauses = split_clauses(text)
# ['When the sun rose,', 'the birds began to sing,', 'and the day started.']
```

## Splitting Paragraphs

The {func}`~phrasplit.split_paragraphs` function splits text at double newlines:

```python
from phrasplit import split_paragraphs

text = """First paragraph with some text.

Second paragraph with more text.

Third paragraph."""

paragraphs = split_paragraphs(text)
# ['First paragraph with some text.',
#  'Second paragraph with more text.',
#  'Third paragraph.']
```

The function handles multiple blank lines and whitespace-only lines:

```python
text = "First.\n\n\n\nSecond."  # Multiple blank lines
paragraphs = split_paragraphs(text)
# ['First.', 'Second.']
```

## Hierarchical Splitting with split_text

The {func}`~phrasplit.split_text` function provides a unified interface for splitting
text while preserving paragraph and sentence structure. This is particularly useful for
audiobook generation where you need different pause lengths between paragraphs,
sentences, and clauses.

```python
from phrasplit import split_text, Segment

text = "First sentence. Second sentence.\n\nNew paragraph here."
segments = split_text(text, mode="sentence")

for seg in segments:
    print(f"Paragraph {seg.paragraph}, Sentence {seg.sentence}: {seg.text}")
# Paragraph 0, Sentence 0: First sentence.
# Paragraph 0, Sentence 1: Second sentence.
# Paragraph 1, Sentence 0: New paragraph here.
```

### Available Modes

- `"paragraph"`: Returns paragraphs only (`sentence` is None)
- `"sentence"`: Returns sentences with paragraph tracking
- `"clause"`: Returns clauses with paragraph and sentence tracking

```python
# Paragraph mode
segments = split_text(text, mode="paragraph")
# Each segment has sentence=None

# Sentence mode (default)
segments = split_text(text, mode="sentence")
# Each segment has paragraph and sentence indices

# Clause mode - finest granularity
text = "Hello, world. Goodbye, friend."
segments = split_text(text, mode="clause")
# Returns: Hello, | world. | Goodbye, | friend.
# All with paragraph and sentence tracking
```

### Detecting Structure Changes

Use the `Segment` fields to detect when paragraphs or sentences change:

```python
from phrasplit import split_text

text = "Sent 1. Sent 2.\n\nSent 3."
segments = split_text(text, mode="sentence")

for i, seg in enumerate(segments):
    if i > 0 and seg.paragraph != segments[i-1].paragraph:
        print("--- New Paragraph ---")
    print(seg.text)
```

## Splitting Long Lines

The {func}`~phrasplit.split_long_lines` function breaks long lines at natural boundaries
(sentences and clauses) to fit within a maximum length:

```python
from phrasplit import split_long_lines

text = "This is a very long sentence. This is another sentence that makes it even longer."
lines = split_long_lines(text, max_length=40)
# Each line will be <= 40 characters when possible
```

The splitting strategy:

1. First, try to split at sentence boundaries
2. If still too long, split at clause boundaries (commas)
3. If still too long, split at word boundaries

## Using Different Language Models

All functions that use spaCy accept a `language_model` parameter:

```python
from phrasplit import split_sentences

# Use a larger, more accurate model
sentences = split_sentences(text, language_model="en_core_web_lg")

# Use a model for another language
sentences = split_sentences(german_text, language_model="de_core_news_sm")
```

Make sure to download the model first:

```bash
python -m spacy download de_core_news_sm
```

## Processing Pipeline Example

Here's a complete example of processing a document:

```python
from phrasplit import split_paragraphs, split_sentences, split_clauses

def process_document(text):
    """Process a document into structured parts."""
    result = []

    for para_idx, paragraph in enumerate(split_paragraphs(text)):
        para_data = {"paragraph": para_idx + 1, "sentences": []}

        for sent_idx, sentence in enumerate(split_sentences(paragraph)):
            sent_data = {
                "sentence": sent_idx + 1,
                "text": sentence,
                "clauses": split_clauses(sentence)
            }
            para_data["sentences"].append(sent_data)

        result.append(para_data)

    return result

# Example usage
text = """Hello world, this is a test. Another sentence here.

Second paragraph with more content, and some clauses."""

structure = process_document(text)
```

### Simplified Pipeline with split_text

The same can be achieved more simply with {func}`~phrasplit.split_text`:

```python
from phrasplit import split_text

text = """Hello world, this is a test. Another sentence here.

Second paragraph with more content, and some clauses."""

# Get all clauses with full structure information
segments = split_text(text, mode="clause")

for seg in segments:
    print(f"P{seg.paragraph} S{seg.sentence}: {seg.text}")
```

The `use_spacy` parameter is also available for {func}`~phrasplit.split_text`:

```python
# Use simple mode for faster processing
segments = split_text(text, mode="sentence", use_spacy=False)

# Explicitly use spaCy mode
segments = split_text(text, mode="sentence", use_spacy=True)
```
