#!/usr/bin/env python3
"""Text analysis example using phrasplit.

This script demonstrates how to analyze text structure and generate
statistics using phrasplit's splitting capabilities.
"""

from collections import Counter
from dataclasses import dataclass

from phrasplit import split_clauses, split_paragraphs, split_sentences


@dataclass
class TextStatistics:
    """Statistics about a text document."""

    paragraph_count: int
    sentence_count: int
    clause_count: int
    word_count: int
    character_count: int
    avg_sentence_length: float
    avg_paragraph_length: float
    avg_clause_length: float
    sentences_per_paragraph: float
    clauses_per_sentence: float

    def __str__(self) -> str:
        return f"""Text Statistics:
  Paragraphs: {self.paragraph_count}
  Sentences: {self.sentence_count}
  Clauses: {self.clause_count}
  Words: {self.word_count}
  Characters: {self.character_count}

Averages:
  Avg sentence length: {self.avg_sentence_length:.1f} words
  Avg paragraph length: {self.avg_paragraph_length:.1f} words
  Avg clause length: {self.avg_clause_length:.1f} words
  Sentences per paragraph: {self.sentences_per_paragraph:.1f}
  Clauses per sentence: {self.clauses_per_sentence:.1f}"""


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def analyze_text(text: str) -> TextStatistics:
    """
    Analyze text and return comprehensive statistics.

    Args:
        text: Input text to analyze

    Returns:
        TextStatistics object with analysis results
    """
    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)
    clauses = split_clauses(text)

    paragraph_count = len(paragraphs)
    sentence_count = len(sentences)
    clause_count = len(clauses)
    word_count = count_words(text)
    character_count = len(text)

    # Calculate averages (handle division by zero)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    avg_paragraph_length = word_count / paragraph_count if paragraph_count > 0 else 0
    avg_clause_length = word_count / clause_count if clause_count > 0 else 0
    sentences_per_paragraph = (
        sentence_count / paragraph_count if paragraph_count > 0 else 0
    )
    clauses_per_sentence = clause_count / sentence_count if sentence_count > 0 else 0

    return TextStatistics(
        paragraph_count=paragraph_count,
        sentence_count=sentence_count,
        clause_count=clause_count,
        word_count=word_count,
        character_count=character_count,
        avg_sentence_length=avg_sentence_length,
        avg_paragraph_length=avg_paragraph_length,
        avg_clause_length=avg_clause_length,
        sentences_per_paragraph=sentences_per_paragraph,
        clauses_per_sentence=clauses_per_sentence,
    )


def analyze_sentence_lengths(text: str) -> dict[str, int | float | list[int]]:
    """
    Analyze sentence length distribution.

    Args:
        text: Input text

    Returns:
        Dictionary with sentence length statistics
    """
    sentences = split_sentences(text)
    lengths = [count_words(s) for s in sentences]

    if not lengths:
        return {"count": 0, "lengths": []}

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "median": sorted(lengths)[len(lengths) // 2],
        "lengths": lengths,
    }


def analyze_paragraph_structure(text: str) -> list[dict[str, int | str]]:
    """
    Analyze the structure of each paragraph.

    Args:
        text: Input text

    Returns:
        List of paragraph analysis dictionaries
    """
    paragraphs = split_paragraphs(text)
    results = []

    for i, para in enumerate(paragraphs, 1):
        sentences = split_sentences(para)
        clauses = split_clauses(para)

        results.append(
            {
                "paragraph": i,
                "sentences": len(sentences),
                "clauses": len(clauses),
                "words": count_words(para),
                "preview": para[:50] + "..." if len(para) > 50 else para,
            }
        )

    return results


def find_longest_sentences(text: str, n: int = 5) -> list[tuple[int, str]]:
    """
    Find the n longest sentences by word count.

    Args:
        text: Input text
        n: Number of sentences to return

    Returns:
        List of (word_count, sentence) tuples
    """
    sentences = split_sentences(text)
    with_lengths = [(count_words(s), s) for s in sentences]
    with_lengths.sort(reverse=True, key=lambda x: x[0])
    return with_lengths[:n]


def find_shortest_sentences(text: str, n: int = 5) -> list[tuple[int, str]]:
    """
    Find the n shortest sentences by word count.

    Args:
        text: Input text
        n: Number of sentences to return

    Returns:
        List of (word_count, sentence) tuples
    """
    sentences = split_sentences(text)
    with_lengths = [(count_words(s), s) for s in sentences]
    with_lengths.sort(key=lambda x: x[0])
    return with_lengths[:n]


def analyze_clause_complexity(text: str) -> dict[str, int | float]:
    """
    Analyze clause complexity in the text.

    Sentences with more clauses are considered more complex.

    Args:
        text: Input text

    Returns:
        Dictionary with complexity statistics
    """
    sentences = split_sentences(text)

    clause_counts = []
    for sentence in sentences:
        clauses = split_clauses(sentence)
        clause_counts.append(len(clauses))

    if not clause_counts:
        return {"total_sentences": 0}

    # Count by complexity level
    simple = sum(1 for c in clause_counts if c == 1)
    moderate = sum(1 for c in clause_counts if c == 2)
    complex_count = sum(1 for c in clause_counts if c >= 3)

    return {
        "total_sentences": len(clause_counts),
        "simple_sentences": simple,  # 1 clause
        "moderate_sentences": moderate,  # 2 clauses
        "complex_sentences": complex_count,  # 3+ clauses
        "avg_clauses_per_sentence": sum(clause_counts) / len(clause_counts),
        "max_clauses": max(clause_counts),
    }


def generate_readability_report(text: str) -> str:
    """
    Generate a comprehensive readability report.

    Args:
        text: Input text

    Returns:
        Formatted report string
    """
    stats = analyze_text(text)
    sentence_stats = analyze_sentence_lengths(text)
    complexity = analyze_clause_complexity(text)

    lines = [
        "=" * 60,
        "READABILITY ANALYSIS REPORT",
        "=" * 60,
        "",
        "OVERVIEW",
        "-" * 40,
        f"  Total paragraphs: {stats.paragraph_count}",
        f"  Total sentences: {stats.sentence_count}",
        f"  Total words: {stats.word_count}",
        f"  Total characters: {stats.character_count}",
        "",
        "SENTENCE LENGTH",
        "-" * 40,
        f"  Average: {stats.avg_sentence_length:.1f} words",
        f"  Shortest: {sentence_stats.get('min', 'N/A')} words",
        f"  Longest: {sentence_stats.get('max', 'N/A')} words",
        "",
        "COMPLEXITY",
        "-" * 40,
        f"  Simple sentences (1 clause): {complexity.get('simple_sentences', 0)}",
        f"  Moderate sentences (2 clauses): {complexity.get('moderate_sentences', 0)}",
        f"  Complex sentences (3+ clauses): {complexity.get('complex_sentences', 0)}",
        f"  Avg clauses/sentence: {complexity.get('avg_clauses_per_sentence', 0):.1f}",
        "",
        "RECOMMENDATIONS",
        "-" * 40,
    ]

    # Add recommendations based on analysis
    if stats.avg_sentence_length > 25:
        lines.append("  - Consider breaking up long sentences for readability")
    elif stats.avg_sentence_length < 10:
        lines.append("  - Sentences may be too short; consider combining some")
    else:
        lines.append("  - Sentence length is appropriate for most audiences")

    if complexity.get("complex_sentences", 0) > stats.sentence_count * 0.3:
        lines.append("  - Many complex sentences; consider simplifying some")

    if stats.sentences_per_paragraph > 8:
        lines.append("  - Paragraphs are long; consider breaking them up")
    elif stats.sentences_per_paragraph < 2:
        lines.append("  - Paragraphs are short; consider combining some")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main() -> None:
    """Demonstrate text analysis features."""
    sample_text = """
The art of writing is a skill that improves with practice. Every writer,
from beginners to seasoned professionals, faces the challenge of choosing
the right words. This process, while sometimes frustrating, is ultimately
rewarding.

Good writing communicates ideas clearly and concisely. It engages the reader,
holds their attention, and leaves a lasting impression. Whether you're writing
a novel, a business report, or a simple email, these principles apply.

One common mistake is using overly complex sentences. Long, winding sentences
with multiple clauses can confuse readers and obscure your meaning. On the
other hand, using only short sentences can make your writing feel choppy and
immature. The key is balance.

Another important aspect is paragraph structure. Each paragraph should focus
on a single main idea. Supporting sentences should elaborate on or provide
evidence for that idea. Transitions between paragraphs help maintain flow.

Practice makes perfect. Write every day, even if only for a few minutes.
Read widely to expose yourself to different styles. Seek feedback from others.
Most importantly, don't be afraid to revise. Great writing is rewriting.
    """.strip()

    print("TEXT ANALYSIS EXAMPLE")
    print("=" * 60)

    # Basic statistics
    print("\n1. BASIC TEXT STATISTICS")
    print("-" * 40)
    stats = analyze_text(sample_text)
    print(stats)

    # Sentence length analysis
    print("\n\n2. SENTENCE LENGTH DISTRIBUTION")
    print("-" * 40)
    sentence_stats = analyze_sentence_lengths(sample_text)
    print(f"  Count: {sentence_stats['count']}")
    print(f"  Min: {sentence_stats['min']} words")
    print(f"  Max: {sentence_stats['max']} words")
    print(f"  Mean: {sentence_stats['mean']:.1f} words")
    print(f"  Median: {sentence_stats['median']} words")

    # Length distribution
    lengths = sentence_stats["lengths"]
    if isinstance(lengths, list):
        length_dist = Counter(lengths)
        print("\n  Distribution:")
        for length in sorted(length_dist.keys()):
            count = length_dist[length]
            bar = "#" * count
            print(f"    {length:2d} words: {bar} ({count})")

    # Paragraph structure
    print("\n\n3. PARAGRAPH STRUCTURE")
    print("-" * 40)
    para_analysis = analyze_paragraph_structure(sample_text)
    for para in para_analysis:
        print(f"  Paragraph {para['paragraph']}:")
        sent, cl, wd = para["sentences"], para["clauses"], para["words"]
        print(f"    Sentences: {sent}, Clauses: {cl}, Words: {wd}")
        print(f"    Preview: {para['preview']!r}")
        print()

    # Longest/shortest sentences
    print("\n4. EXTREME SENTENCES")
    print("-" * 40)
    print("  Longest sentences:")
    for word_count, sentence in find_longest_sentences(sample_text, 3):
        preview = sentence[:60] + "..." if len(sentence) > 60 else sentence
        print(f"    {word_count} words: {preview!r}")

    print("\n  Shortest sentences:")
    for word_count, sentence in find_shortest_sentences(sample_text, 3):
        print(f"    {word_count} words: {sentence!r}")

    # Complexity analysis
    print("\n\n5. COMPLEXITY ANALYSIS")
    print("-" * 40)
    complexity = analyze_clause_complexity(sample_text)
    total = complexity["total_sentences"]
    simple = complexity["simple_sentences"]
    moderate = complexity["moderate_sentences"]
    complex_cnt = complexity["complex_sentences"]
    print(f"  Total sentences: {total}")
    print(f"  Simple (1 clause): {simple} ({simple / total * 100:.1f}%)")
    print(f"  Moderate (2 clauses): {moderate} ({moderate / total * 100:.1f}%)")
    print(f"  Complex (3+ clauses): {complex_cnt} ({complex_cnt / total * 100:.1f}%)")
    print(f"  Max clauses in a sentence: {complexity['max_clauses']}")

    # Full readability report
    print("\n\n6. FULL READABILITY REPORT")
    report = generate_readability_report(sample_text)
    print(report)


if __name__ == "__main__":
    main()
