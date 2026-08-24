#!/usr/bin/env python3
"""Basic usage examples for phrasplit.

This script demonstrates the core functionality of phrasplit:
- Splitting text into sentences
- Splitting text into clauses
- Splitting text into paragraphs
- Splitting long lines
"""

from phrasplit import split_clauses, split_long_lines, split_paragraphs, split_sentences


def demonstrate_sentence_splitting() -> None:
    """Demonstrate sentence splitting with various edge cases."""
    print("=" * 60)
    print("SENTENCE SPLITTING")
    print("=" * 60)

    # Basic sentences
    text = "Hello world. How are you today? I'm doing great!"
    print("\n1. Basic sentences:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_sentences(text)}")

    # Abbreviations
    text = "Dr. Smith met Prof. Green. They discussed the U.S.A. case."
    print("\n2. Handling abbreviations:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_sentences(text)}")

    # Ellipses
    text = "Wait... what happened? I don't know... maybe later."
    print("\n3. Handling ellipses:")
    print(f"   Input: {text!r}")
    result = split_sentences(text)
    print(f"   Output: {result}")
    print("   Note: Ellipses are normalized to '. . .' format")

    # Quotes and dialogue
    text = 'She said, "It works!" Then she smiled.'
    print("\n4. Quotes and dialogue:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_sentences(text)}")

    # URLs and emails
    text = "Visit www.example.com for details. Contact test@email.com for help."
    print("\n5. URLs and special characters:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_sentences(text)}")


def demonstrate_clause_splitting() -> None:
    """Demonstrate clause splitting at commas."""
    print("\n" + "=" * 60)
    print("CLAUSE SPLITTING")
    print("=" * 60)

    # Basic clauses
    text = "I like coffee, and I like tea."
    print("\n1. Basic clause splitting:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_clauses(text)}")

    # Multiple commas
    text = "First, second, third, and fourth."
    print("\n2. Multiple commas (list):")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_clauses(text)}")

    # Complex sentence
    text = "When the sun rose, the birds began to sing, and the day started."
    print("\n3. Complex sentence with multiple clauses:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_clauses(text)}")

    # No splitting at semicolons or colons
    text = "First clause; second clause: with details."
    print("\n4. Semicolons and colons (no split):")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_clauses(text)}")

    # Parenthetical expressions
    text = "The book, which was published last year, became a bestseller."
    print("\n5. Parenthetical expressions:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_clauses(text)}")


def demonstrate_paragraph_splitting() -> None:
    """Demonstrate paragraph splitting."""
    print("\n" + "=" * 60)
    print("PARAGRAPH SPLITTING")
    print("=" * 60)

    # Basic paragraphs
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    print("\n1. Basic paragraph splitting:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_paragraphs(text)}")

    # Multiple blank lines
    text = "First.\n\n\n\nSecond."
    print("\n2. Multiple blank lines:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_paragraphs(text)}")

    # Whitespace handling
    text = "First.\n   \n   \nSecond."
    print("\n3. Whitespace in blank lines:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_paragraphs(text)}")

    # Single newlines preserved within paragraph
    text = "Line one\nLine two\n\nNew paragraph"
    print("\n4. Single newlines (same paragraph):")
    print(f"   Input: {text!r}")
    result = split_paragraphs(text)
    print(f"   Output: {result}")
    print("   Note: Single newlines don't split paragraphs")


def demonstrate_long_line_splitting() -> None:
    """Demonstrate long line splitting."""
    print("\n" + "=" * 60)
    print("LONG LINE SPLITTING")
    print("=" * 60)

    # Basic long line
    text = (
        "This is a very long sentence that needs to be split. This is another sentence."
    )
    print("\n1. Split at sentence boundaries (max_length=40):")
    print(f"   Input: {text!r}")
    result = split_long_lines(text, max_length=40)
    print(f"   Output: {result}")
    for i, line in enumerate(result, 1):
        print(f"   Line {i} ({len(line)} chars): {line!r}")

    # Clause-based splitting
    text = (
        "This is a very long sentence with many clauses, "
        "and it continues here, and goes on."
    )
    print("\n2. Split at clause boundaries when sentence is too long (max_length=35):")
    print(f"   Input: {text!r}")
    result = split_long_lines(text, max_length=35)
    print("   Output:")
    for i, line in enumerate(result, 1):
        print(f"   Line {i} ({len(line)} chars): {line!r}")

    # Word-based splitting (fallback)
    text = "Supercalifragilisticexpialidocious is a very long word"
    print("\n3. Very long words are kept intact (max_length=10):")
    print(f"   Input: {text!r}")
    result = split_long_lines(text, max_length=10)
    print(f"   Output: {result}")
    print("   Note: Words longer than max_length are not broken")

    # Preserving existing line breaks
    text = "Short line.\nAnother short line.\nThird line."
    print("\n4. Preserving existing line breaks (max_length=80):")
    print(f"   Input: {text!r}")
    result = split_long_lines(text, max_length=80)
    print(f"   Output: {result}")


def demonstrate_edge_cases() -> None:
    """Demonstrate edge cases and special handling."""
    print("\n" + "=" * 60)
    print("EDGE CASES")
    print("=" * 60)

    # Empty input
    print("\n1. Empty input:")
    print(f"   split_sentences(''): {split_sentences('')}")
    print(f"   split_clauses(''): {split_clauses('')}")
    print(f"   split_paragraphs(''): {split_paragraphs('')}")

    # Whitespace only
    print("\n2. Whitespace only:")
    print(f"   split_sentences('   '): {split_sentences('   ')}")
    print(f"   split_paragraphs('\\n\\n'): {split_paragraphs(chr(10) + chr(10))}")

    # Single character
    print("\n3. Single character:")
    print(f"   split_sentences('a'): {split_sentences('a')}")

    # Unicode text
    text = "Bonjour le monde. Comment allez-vous?"
    print("\n4. Unicode text:")
    print(f"   Input: {text!r}")
    print(f"   Output: {split_sentences(text)}")

    # Input validation
    print("\n5. Input validation (max_length must be >= 1):")
    try:
        split_long_lines("text", max_length=0)
    except ValueError as e:
        print(f"   split_long_lines('text', max_length=0) raises: {e}")


def main() -> None:
    """Run all demonstrations."""
    print("\nPHRASPLIT - Basic Usage Examples")
    print("================================\n")

    demonstrate_sentence_splitting()
    demonstrate_clause_splitting()
    demonstrate_paragraph_splitting()
    demonstrate_long_line_splitting()
    demonstrate_edge_cases()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
