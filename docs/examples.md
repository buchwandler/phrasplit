# Examples

This page provides practical examples of using phrasplit for various use cases.

## Audiobook Creation

Split text at natural pause points for text-to-speech processing:

```python
from phrasplit import split_sentences, split_clauses

def prepare_for_tts(text):
    """Prepare text for text-to-speech with natural pauses."""
    parts = []

    for sentence in split_sentences(text):
        # Split long sentences at commas for natural pauses
        clauses = split_clauses(sentence)
        parts.extend(clauses)

    return parts

text = """
When the sun rose over the mountains, the valley was filled with golden light.
Birds began to sing their morning songs, and the world slowly awakened.
"""

parts = prepare_for_tts(text)
for part in parts:
    print(part)
    # Each part can be sent to TTS with appropriate pauses between them
```

### Audiobook with Paragraph Awareness

For more control over pause lengths, use {func}`~phrasplit.split_text` to track
paragraph and sentence boundaries:

```python
from phrasplit import split_text

def create_audiobook_segments(text, mode="sentence"):
    """
    Create audiobook segments with pause markers.

    Args:
        text: The text to process
        mode: "sentence" or "clause" for granularity

    Returns:
        List of (text, pause_type) tuples
    """
    segments = split_text(text, mode=mode)
    result = []

    for i, seg in enumerate(segments):
        if not seg.text.strip():
            continue

        # Determine pause type based on structure change
        if i == 0:
            pause_type = "none"
        elif seg.paragraph != segments[i-1].paragraph:
            pause_type = "paragraph"  # Long pause (e.g., 1.0s)
        elif seg.sentence != segments[i-1].sentence:
            pause_type = "sentence"   # Medium pause (e.g., 0.5s)
        else:
            pause_type = "clause"     # Short pause (e.g., 0.2s)

        result.append((seg.text, pause_type))

    return result

text = """
The adventure begins here. Our hero sets out on a journey.

Many challenges lay ahead. But courage would see them through.
"""

segments = create_audiobook_segments(text, mode="clause")
for text, pause in segments:
    print(f"[{pause:>10}] {text}")

# Output:
# [      none] The adventure begins here.
# [  sentence] Our hero sets out on a journey.
# [ paragraph] Many challenges lay ahead.
# [  sentence] But courage would see them through.
```

### Complete Audiobook Processor

A full example integrating with a TTS system:

```python
from phrasplit import split_text, Segment

class AudiobookProcessor:
    """Process text for audiobook generation."""

    PAUSE_DURATIONS = {
        "paragraph": 1.0,
        "sentence": 0.5,
        "clause": 0.2,
    }

    def __init__(self, tts_engine):
        self.tts = tts_engine

    def process_chapter(self, text, mode="sentence"):
        """Process a chapter into audio segments."""
        segments = split_text(text, mode=mode)
        segments = [s for s in segments if s.text.strip()]

        audio_segments = []

        for i, seg in enumerate(segments):
            # Generate audio for text
            audio = self.tts.synthesize(seg.text)
            audio_segments.append(audio)

            # Add appropriate pause
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                if next_seg.paragraph != seg.paragraph:
                    pause = self.PAUSE_DURATIONS["paragraph"]
                elif next_seg.sentence != seg.sentence:
                    pause = self.PAUSE_DURATIONS["sentence"]
                else:
                    pause = self.PAUSE_DURATIONS["clause"]

                audio_segments.append(self.tts.silence(pause))

        return self.tts.concatenate(audio_segments)
```

## Subtitle Generation

Create subtitles that fit within character limits:

```python
from phrasplit import split_long_lines

def create_subtitles(transcript, max_chars=42):
    """Create subtitles from transcript with length limits."""
    lines = split_long_lines(transcript, max_length=max_chars)

    subtitles = []
    for i, line in enumerate(lines, 1):
        subtitle = {
            "index": i,
            "text": line,
            "chars": len(line)
        }
        subtitles.append(subtitle)

    return subtitles

transcript = """
This is a very long sentence that would not fit on a single subtitle line
and needs to be broken up into smaller, more readable chunks for the viewer.
"""

subtitles = create_subtitles(transcript)
for sub in subtitles:
    print(f"{sub['index']}: {sub['text']} ({sub['chars']} chars)")
```

## E-book Processing

Process an e-book into structured data:

```python
from phrasplit import split_paragraphs, split_sentences
import json

def process_ebook(text):
    """Convert e-book text to structured JSON."""
    chapters = []
    current_chapter = {"paragraphs": []}

    for para in split_paragraphs(text):
        # Detect chapter headers (simple example)
        if para.startswith("Chapter"):
            if current_chapter["paragraphs"]:
                chapters.append(current_chapter)
            current_chapter = {
                "title": para,
                "paragraphs": []
            }
        else:
            sentences = split_sentences(para)
            current_chapter["paragraphs"].append({
                "text": para,
                "sentences": sentences,
                "sentence_count": len(sentences)
            })

    if current_chapter["paragraphs"]:
        chapters.append(current_chapter)

    return chapters

# Example usage
book_text = """
Chapter 1

It was the best of times. It was the worst of times.

The city was alive with activity. People rushed through the streets.

Chapter 2

A new day dawned. The adventure continued.
"""

structure = process_ebook(book_text)
print(json.dumps(structure, indent=2))
```

## Text Analysis

Analyze text statistics:

```python
from phrasplit import split_paragraphs, split_sentences, split_clauses

def analyze_text(text):
    """Generate text statistics."""
    paragraphs = split_paragraphs(text)

    total_sentences = 0
    total_clauses = 0
    sentence_lengths = []

    for para in paragraphs:
        sentences = split_sentences(para)
        total_sentences += len(sentences)

        for sent in sentences:
            sentence_lengths.append(len(sent))
            clauses = split_clauses(sent)
            total_clauses += len(clauses)

    stats = {
        "paragraphs": len(paragraphs),
        "sentences": total_sentences,
        "clauses": total_clauses,
        "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths),
        "avg_sentences_per_paragraph": total_sentences / len(paragraphs),
        "avg_clauses_per_sentence": total_clauses / total_sentences,
    }

    return stats

text = """
The quick brown fox jumps over the lazy dog. This sentence is shorter.

Another paragraph here, with some clauses, and more content.
Final sentence of the document.
"""

stats = analyze_text(text)
for key, value in stats.items():
    print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
```

## Batch Processing

Process multiple files:

```python
from pathlib import Path
from phrasplit import split_sentences

def process_directory(input_dir, output_dir):
    """Process all text files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for txt_file in input_path.glob("*.txt"):
        print(f"Processing {txt_file.name}...")

        text = txt_file.read_text(encoding="utf-8")
        sentences = split_sentences(text)

        output_file = output_path / txt_file.name
        output_file.write_text("\n".join(sentences), encoding="utf-8")

        print(f"  -> {len(sentences)} sentences written to {output_file}")

# Example usage
# process_directory("./books", "./processed")
```

## Working with Different Languages

Use language-specific models:

```python
from phrasplit import split_sentences

# German text
german_text = "Guten Tag. Wie geht es Ihnen? Das Wetter ist schön."
# First: python -m spacy download de_core_news_sm
german_sentences = split_sentences(german_text, language_model="de_core_news_sm")

# French text
french_text = "Bonjour. Comment allez-vous? Il fait beau aujourd'hui."
# First: python -m spacy download fr_core_news_sm
french_sentences = split_sentences(french_text, language_model="fr_core_news_sm")

# Spanish text
spanish_text = "Hola. ¿Cómo estás? El tiempo es bueno."
# First: python -m spacy download es_core_news_sm
spanish_sentences = split_sentences(spanish_text, language_model="es_core_news_sm")
```

## Integration with pandas

Process text data in DataFrames:

```python
import pandas as pd
from phrasplit import split_sentences, split_clauses, split_text

# Sample data
data = {
    "id": [1, 2, 3],
    "text": [
        "Hello world. How are you?",
        "The cat sat on the mat, and the dog barked.",
        "Dr. Smith arrived. He was late, unfortunately."
    ]
}
df = pd.DataFrame(data)

# Add sentence count
df["sentence_count"] = df["text"].apply(lambda x: len(split_sentences(x)))

# Add clause count
df["clause_count"] = df["text"].apply(lambda x: len(split_clauses(x)))

# Explode into one row per sentence
df_sentences = df.assign(
    sentence=df["text"].apply(split_sentences)
).explode("sentence")

print(df_sentences)
```

### Using split_text with pandas

For more detailed analysis with structure information:

```python
import pandas as pd
from phrasplit import split_text

text = """First paragraph sentence one. Sentence two.

Second paragraph here. Another sentence."""

# Convert segments to DataFrame
segments = split_text(text, mode="sentence")
df = pd.DataFrame([
    {"text": s.text, "paragraph": s.paragraph, "sentence": s.sentence}
    for s in segments
])

print(df)
#                        text  paragraph  sentence
# 0  First paragraph sentence one.          0         0
# 1               Sentence two.          0         1
# 2      Second paragraph here.          1         0
# 3          Another sentence.          1         1

# Group by paragraph
for para_id, group in df.groupby("paragraph"):
    print(f"\nParagraph {para_id}:")
    for _, row in group.iterrows():
        print(f"  S{row['sentence']}: {row['text']}")
```
