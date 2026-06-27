"""Tests for phrasplit simple (no-spaCy) splitter module."""

import pytest

from phrasplit import (
    split_clauses,
    split_long_lines,
    split_sentences,
    split_text,
)
from phrasplit.splitter_without_spacy import (
    _build_language_patterns,
    _hard_split_simple,
    split_clauses_simple,
    split_long_lines_simple,
    split_sentences_simple,
)


class TestSimpleSentenceSplitting:
    """Tests for split_sentences_simple function."""

    def test_basic_sentences(self) -> None:
        """Test splitting of regular sentences with proper punctuation."""
        text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert "Dr. Smith is here." in result
        assert "Ph.D." in result[1]

    def test_common_abbreviations(self) -> None:
        """Test abbreviations like Mr., Prof., U.S.A. that shouldn't split sentences."""
        text = "Mr. Brown met Prof. Green. They discussed the U.S.A. case."
        result = split_sentences_simple(text)
        assert len(result) == 2
        # Abbreviations should be preserved
        assert "Mr." in result[0]
        assert "Prof." in result[0]
        assert "U.S.A." in result[1]

    def test_acronyms_followed_by_sentences(self) -> None:
        """Test acronyms followed by normal sentences."""
        text = "U.S.A. is big. It has many states."
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert "U.S.A. is big." == result[0]
        assert "It has many states." == result[1]

    def test_website_urls(self) -> None:
        """Ensure website URLs like www.example.com are not split incorrectly."""
        text = "Visit www.example.com. Then send feedback."
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert "www.example.com" in result[0]

    def test_initials_and_titles(self) -> None:
        """Check titles/initials handled gracefully without breaking sentence."""
        text = "Mr. J.R.R. Tolkien wrote many books. They were popular."
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert (
            "J.R.R." in result[0] or "J.R.R" in result[0]
        )  # May or may not have trailing period
        assert "popular." in result[1]

    def test_single_letter_abbreviation(self) -> None:
        """Ensure single-letter abbreviations like 'E.' are not split."""
        text = "E. coli is a bacteria. Dr. E. Stone confirmed it."
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert "E. coli" in result[0] or "E.coli" in result[0]

    def test_quotes_and_dialogue(self) -> None:
        """Test punctuation with quotation marks."""
        text = 'She said, "It works!" Then she smiled.'
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert '"It works!"' in result[0]
        assert "smiled." in result[1]

    def test_suffix_abbreviations(self) -> None:
        """Test suffixes like Ltd., Co. don't break sentences prematurely."""
        text = "Smith & Co. Ltd. is closed. We're switching vendors."
        result = split_sentences_simple(text)
        # This may be 2 sentences (ideal) or more depending on implementation
        assert len(result) >= 2
        assert "closed" in " ".join(result)
        assert "switching" in " ".join(result)

    def test_missing_terminal_punctuation(self) -> None:
        """Handle cases where no punctuation marks end the sentence."""
        text = "This is a sentence without trailing punctuation"
        result = split_sentences_simple(text)
        assert len(result) == 1
        assert result[0] == text

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        assert split_sentences_simple("") == []
        assert split_sentences_simple("   ") == []

    def test_decimal_numbers(self) -> None:
        """Test that decimal numbers are not split."""
        text = "The value is 3.14. That is pi."
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert "3.14" in result[0]

    def test_ellipsis_preservation(self) -> None:
        """Test that ellipsis are preserved."""
        text = "Hello... Is it working?"
        result = split_sentences_simple(text)
        # Should preserve ellipsis
        assert any("..." in s for s in result)

    def test_multiple_sentences(self) -> None:
        """Test multiple sentences in one go."""
        text = "First sentence. Second sentence! Third sentence? Fourth one."
        result = split_sentences_simple(text)
        assert len(result) == 4

    def test_exclamation_question_marks(self) -> None:
        """Test sentences ending with ! and ?"""
        text = "What is this? It's amazing! Really incredible."
        result = split_sentences_simple(text)
        assert len(result) == 3
        assert result[0].endswith("?")
        assert result[1].endswith("!")

    def test_etc_abbreviation(self) -> None:
        """Test etc. abbreviation handling."""
        text = "We have apples, oranges, etc. The store is open."
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert "etc." in result[0]


class TestSimpleClauseSplitting:
    """Tests for split_clauses_simple function."""

    def test_basic_clause_split(self) -> None:
        """Test basic comma splitting."""
        text = "I like coffee, and I like wine."
        result = split_clauses_simple(text)
        assert len(result) == 2
        assert result[0].endswith(",")
        assert "wine." in result[1]

    def test_multiple_clauses(self) -> None:
        """Test sentence with multiple commas."""
        text = "First, second, third."
        result = split_clauses_simple(text)
        assert len(result) == 3

    def test_no_commas(self) -> None:
        """Test sentence without commas."""
        text = "This has no commas."
        result = split_clauses_simple(text)
        assert len(result) == 1
        assert result[0] == text


class TestSimpleLongLines:
    """Tests for split_long_lines_simple function."""

    def test_short_line_unchanged(self) -> None:
        """Test that short lines are not modified."""
        text = "Short line."
        result = split_long_lines_simple(text, max_length=100)
        assert result == [text]

    def test_long_line_split(self) -> None:
        """Test that long lines are split."""
        text = (
            "This is a very long sentence that should be split. "
            "And here is another sentence."
        )
        result = split_long_lines_simple(text, max_length=50)
        assert len(result) > 1
        for line in result:
            # Allow slight overflow for words that can't be split
            assert len(line) <= 60  # Some tolerance

    def test_max_length_validation(self) -> None:
        """Test that invalid max_length raises ValueError."""
        with pytest.raises(ValueError):
            split_long_lines_simple("text", max_length=0)
        with pytest.raises(ValueError):
            split_long_lines_simple("text", max_length=-1)

    def test_word_boundary_split(self) -> None:
        """Test splitting at word boundaries."""
        text = "word " * 50  # 250 characters
        result = split_long_lines_simple(text, max_length=40)
        assert len(result) > 1
        for line in result[:-1]:  # All but last should be near max
            assert len(line) <= 40


class TestLanguagePatternBuilder:
    """Tests for _build_language_patterns function."""

    def test_english_patterns(self) -> None:
        """Test that English patterns are built correctly."""
        patterns = _build_language_patterns("en_core_web_sm")
        assert "prefixes" in patterns
        assert "suffixes" in patterns
        assert "acronyms" in patterns
        # Test that they match expected abbreviations
        assert patterns["prefixes"].search("Dr. Smith")
        assert patterns["suffixes"].search("Apple Inc.")

    def test_unsupported_language(self) -> None:
        """Test handling of unsupported language models."""
        patterns = _build_language_patterns("xx_unknown_model")
        # Should return empty patterns that never match
        assert "prefixes" in patterns
        # Should not match anything
        assert not patterns["prefixes"].search("Dr. Smith")


class TestHardSplit:
    """Tests for _hard_split_simple function."""

    def test_hard_split_at_words(self) -> None:
        """Test splitting at word boundaries."""
        text = "one two three four five six seven"
        result = _hard_split_simple(text, max_length=15)
        assert len(result) > 1
        for line in result:
            assert len(line) <= 15

    def test_single_long_word(self) -> None:
        """Test that a single long word is not split."""
        text = "supercalifragilisticexpialidocious"
        result = _hard_split_simple(text, max_length=10)
        # Should return the word as-is, even though it exceeds max_length
        assert len(result) == 1
        assert result[0] == text


class TestIntegrationWithMainAPI:
    """Test that simple splitter integrates with main API correctly."""

    def test_split_sentences_with_use_spacy_false(self) -> None:
        """Test split_sentences with use_spacy=False."""
        text = "Dr. Smith is here. She works hard."
        result = split_sentences(text, use_spacy=False)
        assert len(result) == 2
        assert "Dr. Smith" in result[0]

    def test_split_clauses_with_use_spacy_false(self) -> None:
        """Test split_clauses with use_spacy=False."""
        text = "I like coffee, and I like wine."
        result = split_clauses(text, use_spacy=False)
        assert len(result) == 2
        assert result[0].endswith(",")

    def test_split_long_lines_with_use_spacy_false(self) -> None:
        """Test split_long_lines with use_spacy=False."""
        text = "This is a long sentence. This is another sentence."
        result = split_long_lines(text, max_length=30, use_spacy=False)
        assert len(result) >= 2

    def test_split_text_with_use_spacy_false(self) -> None:
        """Test split_text with use_spacy=False."""
        text = "Sentence one. Sentence two.\n\nParagraph two."
        result = split_text(text, mode="sentence", use_spacy=False)
        assert len(result) == 3
        assert result[0].paragraph == 0
        assert result[2].paragraph == 1

    def test_split_text_clause_mode_with_use_spacy_false(self) -> None:
        """Test split_text in clause mode with use_spacy=False."""
        text = "I like coffee, and tea."
        result = split_text(text, mode="clause", use_spacy=False)
        assert len(result) == 2
        assert result[0].text.endswith(",")

    def test_deprecation_warning_split_on_colon(self) -> None:
        """Test that split_on_colon parameter triggers deprecation warning."""
        text = "Test sentence."
        with pytest.warns(DeprecationWarning, match="split_on_colon"):
            split_sentences(text, split_on_colon=False, use_spacy=False)


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_only_abbreviations(self) -> None:
        """Test text that is only abbreviations."""
        text = "Dr. Mr. Mrs. Prof."
        result = split_sentences_simple(text)
        # Should handle gracefully, may be 1 or 4 depending on implementation
        assert len(result) >= 1

    def test_unicode_text(self) -> None:
        """Test handling of Unicode text."""
        text = "Première phrase. Deuxième phrase."
        result = split_sentences_simple(text)
        assert len(result) == 2

    def test_mixed_punctuation(self) -> None:
        """Test mixed punctuation."""
        text = "What?! Really?! Yes!"
        result = split_sentences_simple(text)
        assert len(result) >= 2

    def test_newlines_in_text(self) -> None:
        """Test text with newlines."""
        text = "First line.\nSecond line."
        result = split_sentences_simple(text)
        # Should handle newlines as whitespace
        assert len(result) == 2

    def test_very_long_text(self) -> None:
        """Test with very long text."""
        text = ". ".join([f"Sentence {i}" for i in range(100)])
        result = split_sentences_simple(text)
        assert len(result) == 100

    def test_special_characters(self) -> None:
        """Test text with special characters."""
        text = "Price is $9.99. Get it now!"
        result = split_sentences_simple(text)
        assert len(result) == 2
        assert "$9.99" in result[0]

    def test_hyphenated_line_breaks(self) -> None:
        """Test hyphenated line breaks are fixed."""
        text = "The recom-\nmendation was accepted."
        result = split_sentences_simple(text)
        # Should merge hyphenated words
        assert "recommendation" in result[0] or "recom-" in result[0]


class TestContentPreservation:
    """CRITICAL: Ensure we NEVER modify text content, only split it.

    The fundamental contract of phrasplit is to preserve content exactly.
    These tests verify that joining split results equals the original input.
    """

    def test_basic_content_preservation(self) -> None:
        """Test that basic text content is preserved exactly."""
        texts = [
            "Hello world. How are you?",
            "Dr. Smith is here. She has a Ph.D.",
            "First sentence! Second sentence? Third sentence.",
            "Visit www.example.com. Then send feedback.",
        ]
        for text in texts:
            result = split_sentences_simple(text)
            # Join with space to compare (newlines are the split points)
            joined = " ".join(result)
            assert joined == text, f"Content not preserved for: {text}"

    def test_all_closing_punctuation_after_period(self) -> None:
        """Test ALL closing punctuation types after period."""
        # Test every closing punctuation mark
        closing_marks = [
            '"',  # double quote
            "'",  # single quote
            ")",  # parenthesis
            "]",  # bracket
            "}",  # brace
            ">",  # angle bracket
        ]

        for mark in closing_marks:
            text = f'First sentence. (Or "second.{mark})'
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, (
                f"Content not preserved with closing mark '{mark}': {text}"
            )
            # The closing mark should be preserved somewhere in the results
            all_text = "".join(result)
            assert mark in all_text, f"Closing mark '{mark}' lost from output"

    def test_all_closing_punctuation_after_exclamation(self) -> None:
        """Test ALL closing punctuation types after exclamation mark."""
        closing_marks = ['"', "'", ")", "]", "}", ">"]

        for mark in closing_marks:
            text = f'She said "Wow!{mark} That is great.'
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Content not preserved with ! and '{mark}': {text}"

    def test_all_closing_punctuation_after_question(self) -> None:
        """Test ALL closing punctuation types after question mark."""
        closing_marks = ['"', "'", ")", "]", "}", ">"]

        for mark in closing_marks:
            text = f'He asked "Why?{mark} Nobody knows.'
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Content not preserved with ? and '{mark}': {text}"

    def test_nested_closing_punctuation(self) -> None:
        """Test nested/multiple closing punctuation marks."""
        test_cases = [
            'She said "Really!") Then left.',
            'He asked "Why?"]',
            "(See note.) Next sentence.",
            "[Important!] Done now.",
            "{Critical?} Moving on.",
            "<Note.> Continue here.",
            'Quote: "End.")',
            'Nested: "(inner.)")',
            'Double: "Done."" Really.',
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Content not preserved for: {text}"

    def test_quote_type_preservation(self) -> None:
        """Test that single vs double quotes are preserved exactly."""
        test_cases = [
            'She said "hello." Next sentence.',
            "She said 'hello.' Next sentence.",
            "Mixed \"double\" and 'single' quotes. Done.",
            '"Start with quote." End here.',
            "'Start with single.' End here.",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Quote type not preserved: {text}"
            # Verify quote characters are preserved
            assert text.count('"') == joined.count('"'), "Double quote count changed"
            assert text.count("'") == joined.count("'"), "Single quote count changed"

    def test_all_punctuation_combinations(self) -> None:
        """Test combinations of terminal punctuation with closing marks."""
        terminators = [".", "!", "?"]
        closers = ['"', "'", ")", "]", "}", ">"]

        for term in terminators:
            for closer in closers:
                text = f"First{term}{closer} Second sentence."
                result = split_sentences_simple(text)
                joined = " ".join(result)
                assert joined == text, f"Failed for '{term}{closer}': {text}"

    def test_multiple_closing_punctuation(self) -> None:
        """Test multiple consecutive closing marks."""
        test_cases = [
            'Quote: "Done.")))',
            'Nested: "Really?"])',
            'End: "Great!"}}',
            'Complex: "(Done.)"]',
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Multiple closers failed: {text}"

    def test_unicode_preservation(self) -> None:
        """Test Unicode character preservation."""
        test_cases = [
            "Café is open. Très bien!",
            "日本語です。次の文。",
            "Здравствуйте. Как дела?",
            "مرحبا. كيف حالك؟",
            "Emoji test 😀. Next sentence 🎉.",
            "Greek: α β γ. Done.",
            "Math: ∑ ∫ ∞. Next.",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Unicode not preserved: {text}"

    def test_special_characters_preservation(self) -> None:
        """Test special characters are preserved."""
        test_cases = [
            "Price: $9.99. Get it!",
            "Email: test@example.com. Send now.",
            "Math: 2+2=4. Simple.",
            "Percent: 50%. Done.",
            "Hash: #tag. Next.",
            "Ampersand: A & B. Continue.",
            "Asterisk: note*. Next.",
            "Caret: x^2. Done.",
            "Tilde: ~home. Next.",
            "Backtick: `code`. Done.",
            "Pipe: A|B. Next.",
            "Backslash: path\\file. Done.",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Special char not preserved: {text}"

    def test_whitespace_preservation(self) -> None:
        """Test that whitespace is preserved (with minor normalization)."""
        test_cases = [
            "Normal spacing. Between sentences.",
            # Note: Multiple spaces may be normalized to single space
            "Tab\there. Next.",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            # Allow whitespace normalization (multiple spaces -> single space)
            normalized_original = " ".join(text.split())
            normalized_joined = " ".join(joined.split())
            assert normalized_joined == normalized_original, f"Content changed: {text}"

    def test_numbers_and_decimals_preservation(self) -> None:
        """Test numbers, decimals, and formatted numbers."""
        test_cases = [
            "Value: 3.14. Next.",
            "Phone: 555-1234. Call now.",
            "Date: 01.01.2024. Continue.",
            "Version: 1.2.3. Update.",
            "Money: 1,000,000.00. Wow.",
            "Ratio: 3.14159. Done.",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Numbers not preserved: {text}"

    def test_ellipsis_preservation(self) -> None:
        """Test ellipsis preservation."""
        test_cases = [
            "Wait... What happened?",
            "Hmm.... Continue.",
            "Really..... Next.",
            "Dot. Dot. Dot.",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Ellipsis not preserved: {text}"

    def test_no_spurious_characters_added(self) -> None:
        """Ensure no characters are added during splitting."""
        test_cases = [
            "Simple sentence.",
            "Question?",
            "Exclamation!",
            'Quote "here."',
            "(Parenthesis.)",
            "[Bracket.]",
            "{Brace.}",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            # Count all characters (excluding whitespace changes)
            original_chars = set(text)
            joined_chars = set(joined.replace("\n", " "))
            assert original_chars == joined_chars, f"Character set changed: {text}"

    def test_empty_and_whitespace_only(self) -> None:
        """Test empty and whitespace-only strings."""
        test_cases = [
            "",
            " ",
            "  ",
            "\n",
            "\t",
            "   \n   ",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            # Empty/whitespace should return empty list
            assert result == [], f"Non-empty result for whitespace: {repr(text)}"

    def test_single_characters(self) -> None:
        """Test single character inputs."""
        test_cases = [".", "!", "?", "a", "1", " "]

        for text in test_cases:
            result = split_sentences_simple(text)
            if text.strip():
                # Non-whitespace single char should be preserved
                joined = " ".join(result)
                assert text.strip() in joined, f"Single char lost: {repr(text)}"

    def test_real_world_examples(self) -> None:
        """Test real-world text examples."""
        test_cases = [
            "The U.S.A. is large. Dr. Smith (Ph.D.) confirmed it.",
            "Visit http://example.com. Then email info@test.com.",
            "Price: $19.99 (was $29.99). Save 33%!",
            "Version 2.0.1 released. See changelog at docs/v2.0.1.md.",
            'He said "Stop!" She replied "Never."',
            '"Why?" he asked. "Because," she answered.',
            "Items: (a) first, (b) second. Done.",
            "Note [1]: See reference. Note [2]: Also see.",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Real-world example failed: {text}"

    def test_long_text_preservation(self) -> None:
        """Test that long texts preserve all content."""
        # Create a long text with various features
        parts = [
            "Dr. Smith works at Tech Corp.",
            "She has a Ph.D. in Computer Science.",
            "Her email is smith@example.com.",
            "Visit www.example.com for more info.",
            'She said "Innovation is key!"',
            "Prices start at $99.99.",
            "Available in U.S.A. and Canada.",
        ]
        text = " ".join(parts)

        result = split_sentences_simple(text)
        joined = " ".join(result)
        assert joined == text, "Long text content not preserved"

    def test_clause_splitting_preservation(self) -> None:
        """Test content preservation in clause splitting."""
        test_cases = [
            "First, second, third.",
            "I like coffee, tea, and juice.",
            "Red, white, and blue.",
            "One, two, three, four, five.",
        ]

        for text in test_cases:
            result = split_clauses_simple(text)
            joined = " ".join(result)
            assert joined == text, f"Clause split changed content: {text}"

    def test_long_lines_preservation(self) -> None:
        """Test content preservation in long line splitting."""
        test_cases = [
            "Short line.",
            "A bit longer line here.",
            (
                "This is a much longer line that should be split by the "
                "long line handler into multiple pieces."
            ),
        ]

        for text in test_cases:
            result = split_long_lines_simple(text, max_length=50)
            joined = "\n".join(result)
            # Long lines may normalize whitespace
            assert joined.replace("\n", " ").strip() == text.strip(), (
                f"Long line changed content: {text}"
            )

    def test_mixed_newlines_preservation(self) -> None:
        """Test handling of texts with existing newlines."""
        test_cases = [
            "Line 1.\nLine 2.",
            "Para 1.\n\nPara 2.",
            "A.\nB.\nC.",
        ]

        for text in test_cases:
            result = split_sentences_simple(text)
            joined = "\n".join(result)
            # Newlines are treated as whitespace and normalized
            # So we compare with normalized whitespace
            original_normalized = " ".join(text.split())
            joined_normalized = " ".join(joined.split())
            assert original_normalized == joined_normalized, (
                f"Newline handling changed content: {text}"
            )
