#!/usr/bin/env python3
"""Subtitle generation example using phrasplit.

This script demonstrates how to create subtitles from text transcripts
using phrasplit's line splitting capabilities.
"""

from dataclasses import dataclass

from phrasplit import split_long_lines, split_sentences


@dataclass
class Subtitle:
    """Represents a single subtitle entry."""

    index: int
    start_time: str
    end_time: str
    text: str

    def to_srt(self) -> str:
        """Convert to SRT format."""
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.text}\n"

    def to_vtt(self) -> str:
        """Convert to WebVTT format."""
        # WebVTT uses . instead of , for milliseconds
        start = self.start_time.replace(",", ".")
        end = self.end_time.replace(",", ".")
        return f"{start} --> {end}\n{self.text}\n"


def format_timestamp(seconds: float, use_comma: bool = True) -> str:
    """
    Format seconds as SRT/VTT timestamp.

    Args:
        seconds: Time in seconds
        use_comma: Use comma for milliseconds (SRT) vs dot (VTT)

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    sep = "," if use_comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def generate_subtitles(
    text: str,
    max_chars: int = 42,
    words_per_minute: int = 150,
    min_duration: float = 1.0,
    max_duration: float = 7.0,
) -> list[Subtitle]:
    """
    Generate subtitles from text.

    Args:
        text: Input text/transcript
        max_chars: Maximum characters per subtitle line
        words_per_minute: Average speaking rate for timing
        min_duration: Minimum subtitle duration in seconds
        max_duration: Maximum subtitle duration in seconds

    Returns:
        List of Subtitle objects
    """
    # Split text into manageable lines
    lines = split_long_lines(text, max_length=max_chars)

    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]

    subtitles = []
    current_time = 0.0

    for i, line in enumerate(lines, 1):
        # Calculate duration based on word count
        word_count = len(line.split())
        duration = (word_count / words_per_minute) * 60

        # Apply min/max constraints
        duration = max(min_duration, min(duration, max_duration))

        start_time = format_timestamp(current_time)
        end_time = format_timestamp(current_time + duration)

        subtitles.append(
            Subtitle(
                index=i,
                start_time=start_time,
                end_time=end_time,
                text=line,
            )
        )

        current_time += duration

    return subtitles


def generate_two_line_subtitles(
    text: str,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
) -> list[str]:
    """
    Generate subtitles with up to 2 lines each.

    Many subtitle standards prefer 2 lines maximum with
    ~42 characters per line.

    Args:
        text: Input text
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum lines per subtitle (usually 2)

    Returns:
        List of subtitle text blocks
    """
    # First split into sentences
    sentences = split_sentences(text)

    subtitles = []
    current_block: list[str] = []
    current_length = 0

    for sentence in sentences:
        # Split sentence if too long
        lines = split_long_lines(sentence, max_length=max_chars_per_line)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if adding this line would exceed limits
            if len(current_block) >= max_lines:
                # Flush current block
                subtitles.append("\n".join(current_block))
                current_block = []
                current_length = 0

            current_block.append(line)
            current_length += len(line)

    # Don't forget the last block
    if current_block:
        subtitles.append("\n".join(current_block))

    return subtitles


def export_srt(subtitles: list[Subtitle]) -> str:
    """Export subtitles in SRT format."""
    return "\n".join(sub.to_srt() for sub in subtitles)


def export_vtt(subtitles: list[Subtitle]) -> str:
    """Export subtitles in WebVTT format."""
    header = "WEBVTT\n\n"
    content = "\n".join(sub.to_vtt() for sub in subtitles)
    return header + content


def analyze_subtitles(subtitles: list[Subtitle]) -> dict[str, float | int]:
    """
    Analyze subtitle statistics.

    Args:
        subtitles: List of subtitles

    Returns:
        Dictionary of statistics
    """
    if not subtitles:
        return {}

    char_counts = [len(sub.text) for sub in subtitles]
    word_counts = [len(sub.text.split()) for sub in subtitles]

    return {
        "total_subtitles": len(subtitles),
        "avg_chars": sum(char_counts) / len(char_counts),
        "max_chars": max(char_counts),
        "min_chars": min(char_counts),
        "avg_words": sum(word_counts) / len(word_counts),
        "total_words": sum(word_counts),
    }


def main() -> None:
    """Demonstrate subtitle generation features."""
    # Sample transcript
    transcript = """
The history of space exploration is a testament to human curiosity and ingenuity.
From the first satellite launch in 1957 to the moon landing in 1969,
humanity has pushed the boundaries of what's possible.

Today, we stand on the brink of a new era. Private companies are joining
government agencies in the race to explore the cosmos. Mars missions are being
planned, and the dream of interstellar travel inches closer to reality.

But with these advances come new challenges. How do we protect astronauts from
radiation? How do we sustain human life for years in space? These questions
drive scientists and engineers around the world.
    """.strip()

    print("SUBTITLE GENERATION EXAMPLE")
    print("=" * 60)

    # Basic subtitle generation
    print("\n1. BASIC SUBTITLE GENERATION (max 42 chars)")
    print("-" * 40)
    subtitles = generate_subtitles(transcript, max_chars=42)
    print(f"Generated {len(subtitles)} subtitles\n")

    print("First 5 subtitles:")
    for sub in subtitles[:5]:
        print(f"  [{sub.index}] {sub.start_time} -> {sub.end_time}")
        print(f"      {sub.text!r}")
        print(f"      ({len(sub.text)} chars)")
        print()

    # Two-line subtitles
    print("\n2. TWO-LINE SUBTITLES")
    print("-" * 40)
    two_line_subs = generate_two_line_subtitles(transcript, max_chars_per_line=42)
    print(f"Generated {len(two_line_subs)} subtitle blocks\n")

    print("First 3 blocks:")
    for i, block in enumerate(two_line_subs[:3], 1):
        print(f"  Block {i}:")
        for line in block.split("\n"):
            print(f"    {line!r} ({len(line)} chars)")
        print()

    # SRT export
    print("\n3. SRT FORMAT (first 3 entries)")
    print("-" * 40)
    srt_output = export_srt(subtitles[:3])
    print(srt_output)

    # WebVTT export
    print("\n4. WEBVTT FORMAT (first 3 entries)")
    print("-" * 40)
    vtt_output = export_vtt(subtitles[:3])
    print(vtt_output)

    # Statistics
    print("\n5. SUBTITLE STATISTICS")
    print("-" * 40)
    stats = analyze_subtitles(subtitles)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Different line lengths
    print("\n6. COMPARISON OF LINE LENGTHS")
    print("-" * 40)
    for max_chars in [32, 42, 56]:
        subs = generate_subtitles(transcript, max_chars=max_chars)
        stats = analyze_subtitles(subs)
        print(
            f"  max_chars={max_chars}: {stats['total_subtitles']} subtitles, "
            f"avg {stats['avg_chars']:.1f} chars"
        )


if __name__ == "__main__":
    main()
