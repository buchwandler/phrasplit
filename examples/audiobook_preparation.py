#!/usr/bin/env python3
"""Audiobook preparation example using phrasplit.

This script demonstrates how to prepare text for text-to-speech (TTS)
processing by splitting text into natural chunks with appropriate pause points.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from phrasplit import split_clauses, split_paragraphs, split_sentences


@dataclass
class AudioChunk:
    """Represents a chunk of text for TTS processing."""

    text: str
    paragraph_index: int
    sentence_index: int
    clause_index: int
    pause_after: str  # "short", "medium", "long"

    def __str__(self) -> str:
        p, s, c = self.paragraph_index, self.sentence_index, self.clause_index
        return f"[P{p}:S{s}:C{c}] {self.text}"


def prepare_for_tts(text: str) -> Iterator[AudioChunk]:
    """
    Prepare text for text-to-speech processing.

    Yields AudioChunk objects with text and pause information:
    - Clauses get short pauses (comma pauses)
    - Sentences get medium pauses
    - Paragraphs get long pauses

    Args:
        text: Input text to prepare

    Yields:
        AudioChunk objects ready for TTS processing
    """
    paragraphs = split_paragraphs(text)

    for para_idx, paragraph in enumerate(paragraphs, 1):
        sentences = split_sentences(paragraph)

        for sent_idx, sentence in enumerate(sentences, 1):
            clauses = split_clauses(sentence)
            is_last_sentence = sent_idx == len(sentences)

            for clause_idx, clause in enumerate(clauses, 1):
                is_last_clause = clause_idx == len(clauses)

                # Determine pause type
                if is_last_clause and is_last_sentence:
                    pause = "long"  # End of paragraph
                elif is_last_clause:
                    pause = "medium"  # End of sentence
                else:
                    pause = "short"  # End of clause (comma)

                yield AudioChunk(
                    text=clause,
                    paragraph_index=para_idx,
                    sentence_index=sent_idx,
                    clause_index=clause_idx,
                    pause_after=pause,
                )


def generate_ssml(chunks: list[AudioChunk]) -> str:
    """
    Generate SSML (Speech Synthesis Markup Language) from audio chunks.

    SSML is supported by many TTS engines like Google Cloud TTS,
    Amazon Polly, and Microsoft Azure TTS.

    Args:
        chunks: List of AudioChunk objects

    Returns:
        SSML formatted string
    """
    # Pause durations in milliseconds
    pause_durations = {
        "short": 300,  # Comma pause
        "medium": 600,  # Sentence pause
        "long": 1000,  # Paragraph pause
    }

    ssml_parts = ['<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">']

    for chunk in chunks:
        # Add the text
        ssml_parts.append(f"  {chunk.text}")
        # Add the pause
        duration = pause_durations[chunk.pause_after]
        ssml_parts.append(f'  <break time="{duration}ms"/>')

    ssml_parts.append("</speak>")
    return "\n".join(ssml_parts)


def generate_timing_script(
    chunks: list[AudioChunk], words_per_minute: int = 150
) -> str:
    """
    Generate a timing script for audiobook recording.

    Estimates timing based on word count and pause durations.

    Args:
        chunks: List of AudioChunk objects
        words_per_minute: Average speaking rate

    Returns:
        Formatted timing script
    """
    # Pause durations in seconds
    pause_durations = {
        "short": 0.3,
        "medium": 0.6,
        "long": 1.0,
    }

    lines = ["AUDIOBOOK TIMING SCRIPT", "=" * 50, ""]
    current_time = 0.0
    current_paragraph = 0

    for chunk in chunks:
        # Add paragraph header
        if chunk.paragraph_index != current_paragraph:
            current_paragraph = chunk.paragraph_index
            lines.append(f"\n--- Paragraph {current_paragraph} ---\n")

        # Calculate duration for this chunk
        word_count = len(chunk.text.split())
        speech_duration = (word_count / words_per_minute) * 60
        pause_duration = pause_durations[chunk.pause_after]

        # Format timestamp
        minutes = int(current_time // 60)
        seconds = current_time % 60
        timestamp = f"{minutes:02d}:{seconds:05.2f}"

        lines.append(f"[{timestamp}] {chunk.text}")
        lines.append(f"           ^ {chunk.pause_after} pause")

        current_time += speech_duration + pause_duration

    # Add total duration
    total_minutes = int(current_time // 60)
    total_seconds = current_time % 60
    lines.append(f"\n{'=' * 50}")
    lines.append(f"Estimated total duration: {total_minutes:02d}:{total_seconds:05.2f}")

    return "\n".join(lines)


def simple_tts_preparation(text: str) -> list[str]:
    """
    Simple approach: just split into clauses for TTS.

    This is the simplest way to prepare text for TTS -
    just split at natural pause points (commas).

    Args:
        text: Input text

    Returns:
        List of text chunks ready for TTS
    """
    return split_clauses(text)


def main() -> None:
    """Demonstrate audiobook preparation features."""
    sample_text = """
The old lighthouse stood on the rocky cliff, its beacon sweeping across the dark waters.
For over a hundred years, it had guided ships safely to harbor.

Captain Sarah Mitchell watched from her vessel, the Northern Star.
The storm was approaching fast, bringing with it fierce winds and towering waves.
She knew they had to reach the harbor before it hit, or face the wrath of the sea.

"All hands on deck!" she commanded, her voice cutting through the howling wind.
The crew responded instantly, years of training taking over.
They would make it, she was certain of that.
    """.strip()

    print("AUDIOBOOK PREPARATION EXAMPLE")
    print("=" * 60)

    # Simple approach
    print("\n1. SIMPLE CLAUSE SPLITTING")
    print("-" * 40)
    simple_chunks = simple_tts_preparation(sample_text)
    print(f"Total chunks: {len(simple_chunks)}")
    print("\nFirst 5 chunks:")
    for i, chunk in enumerate(simple_chunks[:5], 1):
        print(f"  {i}. {chunk}")

    # Advanced approach with AudioChunks
    print("\n\n2. ADVANCED TTS PREPARATION")
    print("-" * 40)
    chunks = list(prepare_for_tts(sample_text))
    print(f"Total chunks: {len(chunks)}")
    print("\nFirst 10 chunks with pause information:")
    for chunk in chunks[:10]:
        print(f"  {chunk} [{chunk.pause_after} pause]")

    # Generate SSML
    print("\n\n3. SSML OUTPUT (first 20 lines)")
    print("-" * 40)
    ssml = generate_ssml(chunks)
    ssml_lines = ssml.split("\n")
    for line in ssml_lines[:20]:
        print(line)
    if len(ssml_lines) > 20:
        print(f"  ... ({len(ssml_lines) - 20} more lines)")

    # Generate timing script
    print("\n\n4. TIMING SCRIPT")
    print("-" * 40)
    timing_script = generate_timing_script(chunks)
    print(timing_script)


if __name__ == "__main__":
    main()
