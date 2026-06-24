"""Tests for offset-preserving segmentation."""

import pytest

from phrasplit import (
    COMMON_PATTERNS,
    SplitSegment,
    iter_split_with_offsets,
    split_with_offsets,
    suggest_splitting_mode,
    validate_no_placeholder_breaks,
)
from phrasplit.splitter import (
    _apply_corrections_with_offsets,
    _split_after_ellipsis_with_offsets,
)


class TestSplitWithOffsets:
    """Tests for split_with_offsets function."""

    def test_basic_sentence_offsets(self) -> None:
        """Test that offsets correctly map to original text."""
        text = "Hello world. How are you?"
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        assert len(segments) == 2

        # Verify offsets map correctly
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

        # Check specific values
        assert segments[0].text == "Hello world."
        assert segments[0].char_start == 0
        assert segments[1].text == "How are you?"

    def test_paragraph_mode_offsets(self) -> None:
        """Test paragraph mode with offsets."""
        text = "First paragraph.\n\nSecond paragraph."
        segments = split_with_offsets(text, mode="paragraph", use_spacy=False)

        assert len(segments) == 2

        # Verify offsets
        for seg in segments:
            # Note: paragraph text is stripped, so we check containment
            assert seg.text in text
            assert seg.paragraph_idx in (0, 1)
            assert seg.sentence_idx == 0
            assert seg.clause_idx is None

    def test_clause_mode_offsets(self) -> None:
        """Test clause mode with offsets."""
        text = "I like coffee, and I like tea."
        segments = split_with_offsets(text, mode="clause", use_spacy=False)

        # Should split on comma
        assert len(segments) >= 2

        # Verify offsets
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_multiline_text_offsets(self) -> None:
        """Test offsets with multiple paragraphs."""
        text = "Para 1 sent 1. Para 1 sent 2.\n\nPara 2 sent 1."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Verify all offsets
        for seg in segments:
            extracted = text[seg.char_start : seg.char_end]
            # Handle whitespace differences
            assert seg.text.strip() == extracted.strip()

    def test_whitespace_preservation(self) -> None:
        """Test that offsets preserve exact whitespace."""
        text = "  Hello world.  "
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        assert len(segments) == 1
        # Text should be the trimmed sentence, but offsets should be correct
        seg = segments[0]
        assert text[seg.char_start : seg.char_end].strip() == seg.text.strip()

    def test_punctuation_edge_cases(self) -> None:
        """Test offset accuracy with various punctuation."""
        text = 'She said, "Hello!" Then she left.'
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Verify offsets
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_abbreviations_offsets(self) -> None:
        """Test that abbreviations don't break offset calculation."""
        text = "Dr. Smith works at U.S.A. Inc. She is busy."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Should be 2 sentences
        assert len(segments) == 2

        # Verify offsets
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        segments = split_with_offsets("", mode="sentence", use_spacy=False)
        assert len(segments) == 0

        segments = split_with_offsets("   ", mode="sentence", use_spacy=False)
        assert len(segments) == 0

    def test_max_chars_splitting(self) -> None:
        """Test max_chars parameter splits long segments."""
        # Create a long sentence
        text = "word " * 50  # 250 characters
        segments = split_with_offsets(
            text, mode="sentence", use_spacy=False, max_chars=50
        )

        # Should be split into multiple segments
        assert len(segments) > 1

        # All segments should be <= max_chars
        for seg in segments:
            assert len(seg.text) <= 50

        # Verify offsets still work
        for seg in segments:
            extracted = text[seg.char_start : seg.char_end]
            assert seg.text.strip() == extracted.strip()

    def test_max_chars_respects_boundaries(self) -> None:
        """Test that max_chars splitting prefers word boundaries."""
        text = (
            "This is a very long sentence with many words "
            "that exceeds the maximum character limit."
        )
        segments = split_with_offsets(
            text, mode="sentence", use_spacy=False, max_chars=30
        )

        # Check that we don't have split words (no segment starts/ends mid-word)
        for seg in segments:
            # Segments should start and end with non-space or be at text boundaries
            if seg.char_start > 0:
                assert (
                    text[seg.char_start - 1].isspace()
                    or text[seg.char_start - 1] in ".,;!?"
                )

    def test_offsets_with_spacy(self) -> None:
        """Test offset accuracy with spaCy backend."""
        pytest.importorskip("spacy")

        text = "Hello world. How are you?\n\nNew paragraph."
        segments = split_with_offsets(text, mode="sentence", use_spacy=True)

        # Verify offsets
        for seg in segments:
            # Handle potential whitespace differences
            extracted = text[seg.char_start : seg.char_end]
            assert seg.text.strip() == extracted.strip()

    def test_abbreviation_merge_with_quotes_offsets(self) -> None:
        """Test abbreviation merging with quotes in offset mode."""
        pytest.importorskip("spacy")

        text = 'Dr. "Smith" arrived.'
        segments = split_with_offsets(text, mode="sentence", use_spacy=True)
        assert [seg.text for seg in segments] == ['Dr. "Smith" arrived.']
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_abbreviation_merge_with_brackets_offsets(self) -> None:
        """Test abbreviation merging with brackets in offset mode."""
        pytest.importorskip("spacy")

        text = "Prof. (Müller) sagte nein. Dann ging er."
        segments = split_with_offsets(text, mode="sentence", use_spacy=True)
        assert [seg.text for seg in segments] == [
            "Prof. (Müller) sagte nein.",
            "Dann ging er.",
        ]
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    @pytest.mark.parametrize("use_spacy", [False, True])
    def test_dotted_acronyms_offsets(self, use_spacy: bool) -> None:
        """Test dotted acronyms are preserved in offset mode."""
        if use_spacy:
            pytest.importorskip("spacy")

        text = "He lives in the U.S. He moved last year."
        segments = split_with_offsets(text, mode="sentence", use_spacy=use_spacy)
        assert [seg.text for seg in segments] == [
            "He lives in the U.S.",
            "He moved last year.",
        ]
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

        text = "Das ist z.B. gut. Wirklich."
        segments = split_with_offsets(text, mode="sentence", use_spacy=use_spacy)
        assert [seg.text for seg in segments] == ["Das ist z.B. gut.", "Wirklich."]
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    @pytest.mark.parametrize("use_spacy", [False, True])
    def test_url_offsets_with_www_and_bare_domains(self, use_spacy: bool) -> None:
        """Test offsets with www and bare-domain URLs."""
        if use_spacy:
            pytest.importorskip("spacy")

        text = "See www.example.com. It works."
        segments = split_with_offsets(text, mode="sentence", use_spacy=use_spacy)
        assert [seg.text for seg in segments] == ["See www.example.com.", "It works."]
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

        text = "Visit example.com/path). Then continue."
        segments = split_with_offsets(text, mode="sentence", use_spacy=use_spacy)
        assert [seg.text for seg in segments] == [
            "Visit example.com/path).",
            "Then continue.",
        ]
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    @pytest.mark.parametrize("use_spacy", [False, True])
    def test_ellipsis_offsets_unicode_digits(self, use_spacy: bool) -> None:
        """Test ellipsis boundaries with unicode and digits in offset mode."""
        if use_spacy:
            pytest.importorskip("spacy")

        text = "Wait... 2025 was wild. True."
        segments = split_with_offsets(text, mode="sentence", use_spacy=use_spacy)
        assert segments[0].text == "Wait..."
        assert segments[1].text.startswith("2025")
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text


class TestOffsetCorrections:
    """Tests for offset-preserving corrections."""

    def test_apply_corrections_with_offsets_merges_and_splits_urls(self) -> None:
        """Test abbreviation merging and URL splitting with offsets."""
        text = "Dr. Smith recommends https://a.com https://b.com today."
        segments = [(text[:3], 0, 3), (text[4:], 4, len(text))]

        result = _apply_corrections_with_offsets(text, segments, "en_core_web_sm")

        assert len(result) == 2
        assert result[0][0] == "Dr. Smith recommends https://a.com"
        assert result[1][0] == "https://b.com today."

        for seg_text, start, end in result:
            assert text[start:end] == seg_text

    def test_split_after_ellipsis_with_offsets(self) -> None:
        """Test ellipsis splitting preserves exact offsets."""
        text = "He was tired.... The next day he left."
        segments = [(text, 0, len(text))]

        result = _split_after_ellipsis_with_offsets(text, segments)

        assert [seg[0] for seg in result] == [
            "He was tired....",
            "The next day he left.",
        ]

        for seg_text, start, end in result:
            assert text[start:end] == seg_text

    def test_spacy_offsets_apply_corrections_ellipsis(self) -> None:
        """Test spaCy offsets apply ellipsis corrections."""
        pytest.importorskip("spacy")

        text = "He was tired.... The next day he left."
        segments = split_with_offsets(text, mode="sentence", use_spacy=True)

        assert [seg.text for seg in segments] == [
            "He was tired....",
            "The next day he left.",
        ]


class TestSpacyOffsetChunking:
    """Tests for spaCy offset chunking on long text."""

    def test_spacy_offset_chunking_long_paragraph(self) -> None:
        """Ensure long texts are chunked without max_length errors."""
        pytest.importorskip("spacy")

        from phrasplit.splitter import _get_nlp

        nlp = _get_nlp("en_core_web_sm")
        original_max = nlp.max_length
        try:
            nlp.max_length = 500
            chunk = "Hello world. "
            repeat_count = (nlp.max_length // len(chunk)) + 5
            text = chunk * repeat_count

            segments = split_with_offsets(text, mode="sentence", use_spacy=True)

            assert segments
            assert len(segments) == repeat_count
            assert all(seg.text == "Hello world." for seg in segments)
            for seg in segments:
                assert text[seg.char_start : seg.char_end] == seg.text
        finally:
            nlp.max_length = original_max


class TestStableIDs:
    """Tests for stable ID generation."""

    def test_id_format_sentence(self) -> None:
        """Test ID format for sentence mode."""
        text = "Sentence one. Sentence two.\n\nParagraph two."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        assert segments[0].id == "p0s0"
        assert segments[1].id == "p0s1"
        assert segments[2].id == "p1s0"

    def test_id_format_clause(self) -> None:
        """Test ID format for clause mode."""
        text = "I like coffee, and tea."
        segments = split_with_offsets(text, mode="clause", use_spacy=False)

        # Should have clause indices
        for seg in segments:
            assert "c" in seg.id
            assert seg.clause_idx is not None

    def test_id_format_paragraph(self) -> None:
        """Test ID format for paragraph mode."""
        text = "Para 1.\n\nPara 2."
        segments = split_with_offsets(text, mode="paragraph", use_spacy=False)

        assert segments[0].id == "p0s0"
        assert segments[1].id == "p1s0"

    def test_hierarchical_indices(self) -> None:
        """Test that hierarchical indices are correct."""
        text = "P0 S0. P0 S1.\n\nP1 S0. P1 S1."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Check paragraph indices
        assert segments[0].paragraph_idx == 0
        assert segments[1].paragraph_idx == 0
        assert segments[2].paragraph_idx == 1
        assert segments[3].paragraph_idx == 1

        # Check sentence indices
        assert segments[0].sentence_idx == 0
        assert segments[1].sentence_idx == 1
        assert segments[2].sentence_idx == 0
        assert segments[3].sentence_idx == 1


class TestIterSplitWithOffsets:
    """Tests for streaming iterator API."""

    def test_iterator_yields_segments(self) -> None:
        """Test that iterator yields segments in order."""
        text = "First. Second. Third."
        segments_list = list(
            iter_split_with_offsets(text, mode="sentence", use_spacy=False)
        )

        assert len(segments_list) == 3
        assert all(isinstance(seg, SplitSegment) for seg in segments_list)

    def test_iterator_order(self) -> None:
        """Test that iterator yields in document order."""
        text = "S1. S2. S3."
        segments = list(iter_split_with_offsets(text, mode="sentence", use_spacy=False))

        # Check order by char_start
        for i in range(len(segments) - 1):
            assert segments[i].char_start < segments[i + 1].char_start

    def test_iterator_matches_list(self) -> None:
        """Test that iterator produces same results as list version."""
        text = "Test sentence one. Test sentence two.\n\nNew paragraph."

        list_segments = split_with_offsets(text, mode="sentence", use_spacy=False)
        iter_segments = list(
            iter_split_with_offsets(text, mode="sentence", use_spacy=False)
        )

        assert len(list_segments) == len(iter_segments)
        for ls, its in zip(list_segments, iter_segments, strict=False):
            assert ls.id == its.id
            assert ls.text == its.text
            assert ls.char_start == its.char_start
            assert ls.char_end == its.char_end


class TestSplitSegmentDataclass:
    """Tests for SplitSegment dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        seg = SplitSegment(
            id="p0s1",
            text="Test",
            char_start=5,
            char_end=9,
            paragraph_idx=0,
            sentence_idx=1,
            clause_idx=None,
            meta={"method": "spacy"},
        )

        d = seg.to_dict()
        assert d["id"] == "p0s1"
        assert d["text"] == "Test"
        assert d["char_start"] == 5
        assert d["char_end"] == 9
        assert d["paragraph_idx"] == 0
        assert d["sentence_idx"] == 1
        assert d["clause_idx"] is None
        assert d["meta"] == {"method": "spacy"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        d = {
            "id": "p0s1c2",
            "text": "Test text",
            "char_start": 10,
            "char_end": 19,
            "paragraph_idx": 0,
            "sentence_idx": 1,
            "clause_idx": 2,
            "meta": {"custom": "value"},
        }

        seg = SplitSegment.from_dict(d)
        assert seg.id == "p0s1c2"
        assert seg.text == "Test text"
        assert seg.char_start == 10
        assert seg.char_end == 19
        assert seg.paragraph_idx == 0
        assert seg.sentence_idx == 1
        assert seg.clause_idx == 2
        assert seg.meta == {"custom": "value"}

    def test_validation_negative_start(self) -> None:
        """Test validation of negative char_start."""
        with pytest.raises(ValueError, match="char_start must be >= 0"):
            SplitSegment(
                id="p0s0",
                text="Test",
                char_start=-1,
                char_end=4,
                paragraph_idx=0,
                sentence_idx=0,
            )

    def test_validation_end_before_start(self) -> None:
        """Test validation of char_end < char_start."""
        with pytest.raises(ValueError, match="char_end .* must be >= char_start"):
            SplitSegment(
                id="p0s0",
                text="Test",
                char_start=10,
                char_end=5,
                paragraph_idx=0,
                sentence_idx=0,
            )


class TestPlaceholderValidation:
    """Tests for placeholder/markup validation utilities."""

    def test_no_breaks_detected(self) -> None:
        """Test that intact placeholders produce no warnings."""
        text = "Hello {{tag}}. Another sentence."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=COMMON_PATTERNS["mustache"]
        )
        assert len(warnings) == 0

    def test_placeholder_break_detected(self) -> None:
        """Test detection of broken placeholders."""
        # Manually create segments that split a placeholder
        text = "Hello {{placeholder}} world."
        # Create segments that would split the placeholder
        from phrasplit import SplitSegment

        segments = [
            SplitSegment(
                id="p0s0",
                text="Hello {{place",
                char_start=0,
                char_end=13,
                paragraph_idx=0,
                sentence_idx=0,
            ),
            SplitSegment(
                id="p0s1",
                text="holder}} world.",
                char_start=13,
                char_end=28,
                paragraph_idx=0,
                sentence_idx=1,
            ),
        ]

        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=COMMON_PATTERNS["mustache"]
        )
        assert len(warnings) > 0
        assert "split across segments" in warnings[0]

    def test_custom_placeholder_pattern(self) -> None:
        """Test validation with custom placeholder pattern."""
        text = "Hello <TAG>world</TAG>. Another sentence."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Custom pattern for XML-style tags
        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=r"<[^>]+>[^<]*</[^>]+>"
        )
        assert len(warnings) == 0

    def test_ssmd_pattern(self) -> None:
        """Test validation with SSMD pattern."""
        text = "Hello [world]{lang='de'}. Another [sentence]{lang='en'}."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=COMMON_PATTERNS["ssmd"]
        )
        assert len(warnings) == 0

    def test_speechmarkdown_pattern(self) -> None:
        """Test validation with Speech Markdown pattern."""
        text = "Hello ((world)[rate:'slow';]). How are you?"
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        warnings = validate_no_placeholder_breaks(
            text, segments, placeholder_pattern=COMMON_PATTERNS["speechmarkdown"]
        )
        assert len(warnings) == 0

    def test_invalid_pattern_raises_error(self) -> None:
        """Test that invalid regex pattern raises ValueError."""
        text = "Hello world."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        with pytest.raises(ValueError, match="Invalid regex pattern"):
            validate_no_placeholder_breaks(
                text, segments, placeholder_pattern=r"[invalid(regex"
            )

    def test_suggest_splitting_mode(self) -> None:
        """Test splitting mode suggestion."""
        # Few placeholders - should suggest clause mode
        text = "Short {{tag}}. Another {{tag}}."
        mode = suggest_splitting_mode(
            text, placeholder_pattern=COMMON_PATTERNS["mustache"]
        )
        assert mode in ("paragraph", "sentence", "clause")

    def test_suggest_splitting_mode_invalid_pattern(self) -> None:
        """Test that suggest_splitting_mode validates regex pattern."""
        text = "Hello world."

        with pytest.raises(ValueError, match="Invalid regex pattern"):
            suggest_splitting_mode(text, placeholder_pattern=r"[invalid(regex")


class TestExactSliceInvariant:
    """Tests for exact-slice policy:
    segment.text == text[char_start:char_end]"""

    def test_exact_slice_basic(self) -> None:
        """Test exact-slice invariant with simple text."""
        text = "Hello world. How are you?"

        for mode in ["paragraph", "sentence", "clause"]:
            for use_spacy in [True, False]:
                segments = split_with_offsets(text, mode=mode, use_spacy=use_spacy)

                for seg in segments:
                    # Exact-slice invariant: NO .strip() allowed
                    assert text[seg.char_start : seg.char_end] == seg.text, (
                        f"Invariant broken for mode={mode}, use_spacy={use_spacy}, "
                        f"segment={seg.id}"
                    )

    def test_exact_slice_with_leading_whitespace(self) -> None:
        """Test exact-slice with leading whitespace."""
        text = "   Hello world.   "
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_with_trailing_whitespace(self) -> None:
        """Test exact-slice with trailing whitespace."""
        text = "Hello world.   \n\n   New paragraph.   "
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_with_multiple_spaces(self) -> None:
        """Test exact-slice with multiple spaces between words."""
        text = "Hello    world.    How   are   you?"
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_with_newlines_and_tabs(self) -> None:
        """Test exact-slice with newlines and tabs."""
        text = "Hello\tworld.\n\nHow\tare\tyou?"
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_with_max_chars_small(self) -> None:
        """Test exact-slice invariant when max_chars is smaller than average word."""
        text = "This is a very long sentence that needs splitting."
        segments = split_with_offsets(text, max_chars=10, use_spacy=False)

        # All segments should be <= 10 chars
        assert all(len(seg.text) <= 10 for seg in segments)

        # Exact-slice invariant must hold
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_with_max_chars_exact_boundary(self) -> None:
        """Test exact-slice when max_chars equals a word boundary."""
        text = "word1 word2 word3 word4 word5"
        max_chars = 12  # Exactly fits "word1 word2 "
        segments = split_with_offsets(text, max_chars=max_chars, use_spacy=False)

        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_with_max_chars_multiple_subsegments(self) -> None:
        """Test exact-slice when segment splits into multiple subsegments."""
        # Create a long sentence
        text = "word " * 50  # 250 characters
        segments = split_with_offsets(text.strip(), max_chars=30, use_spacy=False)

        # Should have multiple subsegments
        assert len(segments) > 5

        # All must maintain invariant
        for seg in segments:
            assert text.strip()[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_recomposition(self) -> None:
        """Test that segments can be used to reconstruct original text."""
        text = "First sentence. Second sentence.\n\nNew paragraph here."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Sort by char_start (should already be sorted)
        segments_sorted = sorted(segments, key=lambda s: s.char_start)

        # Reconstruct by extracting each segment using offsets
        reconstructed_parts = []
        last_end = 0

        for seg in segments_sorted:
            # Add any gap between segments
            if seg.char_start > last_end:
                reconstructed_parts.append(text[last_end : seg.char_start])

            # Add segment text via offset
            reconstructed_parts.append(text[seg.char_start : seg.char_end])
            last_end = seg.char_end

        # Add any trailing text
        if last_end < len(text):
            reconstructed_parts.append(text[last_end:])

        reconstructed = "".join(reconstructed_parts)

        # Should match original exactly
        assert reconstructed == text

    def test_exact_slice_with_unicode(self) -> None:
        """Test exact-slice with Unicode characters."""
        text = "Café résumé. Über große Möbel. 你好世界。"
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_determinism(self) -> None:
        """Test that exact same segments are produced across multiple runs."""
        text = "Hello world. How are you?\n\nNew paragraph."

        # Run multiple times
        runs = [
            split_with_offsets(text, mode="sentence", use_spacy=False) for _ in range(3)
        ]

        # All runs should produce identical results
        for i in range(1, len(runs)):
            assert len(runs[0]) == len(runs[i])
            for seg1, seg2 in zip(runs[0], runs[i], strict=False):
                assert seg1.id == seg2.id
                assert seg1.text == seg2.text
                assert seg1.char_start == seg2.char_start
                assert seg1.char_end == seg2.char_end
                # Verify invariant
                assert text[seg1.char_start : seg1.char_end] == seg1.text

    def test_exact_slice_with_max_chars_determinism(self) -> None:
        """Test determinism with max_chars splitting."""
        text = "word " * 100

        # Run multiple times
        runs = [
            split_with_offsets(text.strip(), max_chars=50, use_spacy=False)
            for _ in range(3)
        ]

        # All runs should produce identical results
        for i in range(1, len(runs)):
            assert len(runs[0]) == len(runs[i])
            for seg1, seg2 in zip(runs[0], runs[i], strict=False):
                assert seg1.id == seg2.id
                assert seg1.text == seg2.text
                assert seg1.char_start == seg2.char_start
                assert seg1.char_end == seg2.char_end

    def test_exact_slice_empty_after_trimming(self) -> None:
        """Test that whitespace-only segments are handled correctly."""
        text = "Hello.   \n\n\n   World."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # All segments should have non-whitespace content or be exact slices
        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text
            # Segments shouldn't be purely whitespace
            assert seg.text.strip(), f"Empty segment: {seg.id}"

    def test_exact_slice_with_spacy_backend(self) -> None:
        """Test exact-slice invariant with spaCy backend."""
        pytest.importorskip("spacy")

        text = "Dr. Smith is here. She has a Ph.D. in Chemistry."
        segments = split_with_offsets(text, mode="sentence", use_spacy=True)

        for seg in segments:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_exact_slice_with_max_chars_spacy(self) -> None:
        """Test exact-slice with max_chars using spaCy backend."""
        pytest.importorskip("spacy")

        text = "This is a very long sentence that will be split. " * 5
        segments = split_with_offsets(text.strip(), max_chars=40, use_spacy=True)

        for seg in segments:
            assert text.strip()[seg.char_start : seg.char_end] == seg.text
            assert len(seg.text) <= 40


class TestInlineMarkupMode:
    """Tests for the opt-in inline_markup XHTML sentence split mode."""

    def test_inline_markup_splits_reported_sample_into_three_sentences(self) -> None:
        """Inline XHTML no longer blocks sentence boundaries;"""
        """fragments stay balanced."""
        sample = (
            "He nodded slowly. "
            "<em>I can\u2019t force her, for all that I need her.</em> "
            "Perhaps Tisamon would have more luck in persuading her."
        )

        segments = split_with_offsets(
            sample,
            mode="sentence",
            use_spacy=False,
            language_model="en_core_web_sm",
            inline_markup=True,
        )

        assert [seg.text for seg in segments] == [
            "He nodded slowly.",
            "<em>I can\u2019t force her, for all that I need her.</em>",
            "Perhaps Tisamon would have more luck in persuading her.",
        ]
        # Exact-slice invariant must hold for every markup segment.
        assert all(
            sample[seg.char_start : seg.char_end] == seg.text for seg in segments
        )

    def test_inline_markup_default_is_unchanged_for_plain_text(self) -> None:
        """Default (inline_markup omitted) behavior is unchanged on plain text."""
        text = "Hello world. How are you?"

        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        assert [seg.text for seg in segments] == ["Hello world.", "How are you?"]

    def test_inline_markup_keeps_closing_tag_with_previous_sentence(self) -> None:
        """A closing tag between punctuation and whitespace"""
        """stays in the previous segment."""
        text = "One sentence. <em>Two sentence.</em> Three sentence."

        segments = split_with_offsets(
            text, mode="sentence", use_spacy=False, inline_markup=True
        )

        assert segments[1].text == "<em>Two sentence.</em>"
        assert all(text[seg.char_start : seg.char_end] == seg.text for seg in segments)

    def test_inline_markup_rejects_explicit_spacy_backend(self) -> None:
        """inline_markup=True requires the regex backend."""
        with pytest.raises(ValueError, match="regex backend"):
            split_with_offsets(
                "a. b.", mode="sentence", use_spacy=True, inline_markup=True
            )
