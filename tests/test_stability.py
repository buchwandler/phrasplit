"""Tests for stable IDs and deterministic segmentation."""

import pytest

from phrasplit import split_with_offsets


class TestStability:
    """Tests for stability and determinism of segmentation."""

    def test_same_input_same_output_regex(self) -> None:
        """Test that same input produces same output (regex backend)."""
        text = "Hello world. How are you?\n\nNew paragraph here."

        # Run multiple times
        results = [
            split_with_offsets(text, mode="sentence", use_spacy=False) for _ in range(3)
        ]

        # All runs should produce identical results
        for i in range(1, len(results)):
            assert len(results[0]) == len(results[i])
            for seg0, segi in zip(results[0], results[i], strict=False):
                assert seg0.id == segi.id
                assert seg0.text == segi.text
                assert seg0.char_start == segi.char_start
                assert seg0.char_end == segi.char_end
                assert seg0.paragraph_idx == segi.paragraph_idx
                assert seg0.sentence_idx == segi.sentence_idx
                assert seg0.clause_idx == segi.clause_idx

    def test_same_input_same_output_spacy(self) -> None:
        """Test that same input produces same output (spaCy backend)."""
        pytest.importorskip("spacy")

        text = "First sentence. Second sentence.\n\nNew paragraph."

        # Run multiple times
        results = [
            split_with_offsets(text, mode="sentence", use_spacy=True) for _ in range(3)
        ]

        # All runs should produce identical results
        for i in range(1, len(results)):
            assert len(results[0]) == len(results[i])
            for seg0, segi in zip(results[0], results[i], strict=False):
                assert seg0.id == segi.id
                assert seg0.text == segi.text
                assert seg0.char_start == segi.char_start
                assert seg0.char_end == segi.char_end

    def test_stable_ids_across_modes(self) -> None:
        """Test that IDs follow predictable pattern across modes."""
        text = "Sentence 1. Sentence 2.\n\nParagraph 2, sentence 1."

        # Sentence mode
        sent_segs = split_with_offsets(text, mode="sentence", use_spacy=False)
        assert sent_segs[0].id == "p0s0"
        assert sent_segs[1].id == "p0s1"
        assert sent_segs[2].id == "p1s0"

        # Paragraph mode
        para_segs = split_with_offsets(text, mode="paragraph", use_spacy=False)
        assert para_segs[0].id == "p0s0"
        assert para_segs[1].id == "p1s0"

    def test_id_uniqueness(self) -> None:
        """Test that all segment IDs are unique."""
        text = "S1. S2. S3.\n\nP2 S1. P2 S2."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        ids = [seg.id for seg in segments]
        assert len(ids) == len(set(ids)), "All IDs should be unique"

    def test_hierarchical_consistency(self) -> None:
        """Test that hierarchical indices are consistent with IDs."""
        text = "P0S0. P0S1.\n\nP1S0."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        for seg in segments:
            # Parse ID
            parts = seg.id.split("s")
            para_part = parts[0].replace("p", "")
            sent_part = parts[1].split("c")[0]

            # Verify indices match ID
            assert int(para_part) == seg.paragraph_idx
            assert int(sent_part) == seg.sentence_idx

    def test_ordering_stability(self) -> None:
        """Test that segments are always in document order."""
        text = "S1. S2. S3.\n\nP2S1. P2S2. P2S3."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Check char_start ordering
        for i in range(len(segments) - 1):
            assert segments[i].char_start <= segments[i + 1].char_start, (
                "Segments should be in document order"
            )

    def test_clause_id_stability(self) -> None:
        """Test stable IDs in clause mode."""
        text = "I like coffee, tea, and juice."
        segments = split_with_offsets(text, mode="clause", use_spacy=False)

        # All should have same paragraph and sentence index
        for seg in segments:
            assert seg.paragraph_idx == 0
            assert seg.sentence_idx == 0
            assert seg.clause_idx is not None

        # Clause indices should be sequential
        clause_indices = [seg.clause_idx for seg in segments]
        assert clause_indices == list(range(len(clause_indices)))

    def test_max_chars_id_stability(self) -> None:
        """Test that max_chars splitting produces stable IDs."""
        text = "word " * 50
        segments1 = split_with_offsets(
            text, mode="sentence", use_spacy=False, max_chars=50
        )
        segments2 = split_with_offsets(
            text, mode="sentence", use_spacy=False, max_chars=50
        )

        assert len(segments1) == len(segments2)
        for seg1, seg2 in zip(segments1, segments2, strict=False):
            assert seg1.id == seg2.id

    def test_index_reset_per_paragraph(self) -> None:
        """Test that sentence indices reset for each paragraph."""
        text = "P0S0. P0S1. P0S2.\n\nP1S0. P1S1."
        segments = split_with_offsets(text, mode="sentence", use_spacy=False)

        # Find paragraph boundary
        para0_segs = [s for s in segments if s.paragraph_idx == 0]
        para1_segs = [s for s in segments if s.paragraph_idx == 1]

        # Check sentence indices reset
        assert para0_segs[0].sentence_idx == 0
        assert para1_segs[0].sentence_idx == 0, (
            "Sentence indices should reset per paragraph"
        )

    def test_metadata_consistency(self) -> None:
        """Test that metadata is consistent across runs."""
        text = "Test sentence."

        segments_regex = split_with_offsets(text, mode="sentence", use_spacy=False)
        assert segments_regex[0].meta["method"] == "regex"
        assert segments_regex[0].meta["mode"] == "sentence"

    def test_offset_calculation_consistency(self) -> None:
        """Test that offset calculations are always correct."""
        texts = [
            "Simple sentence.",
            "Dr. Smith works here.",
            "Quote: 'Hello world!'",
            "Multiple. Sentences. Here.",
            "With\n\nParagraphs.",
        ]

        for text in texts:
            segments = split_with_offsets(text, mode="sentence", use_spacy=False)

            for seg in segments:
                # Verify offset consistency
                extracted = text[seg.char_start : seg.char_end]
                # Allow for whitespace normalization
                assert seg.text.strip() == extracted.strip(), (
                    f"Offset mismatch for text: {text!r}"
                )

    def test_stability_with_special_characters(self) -> None:
        """Test stability with special characters and Unicode."""
        text = "Café résumé. Über große Möbel.\n\n日本語テスト。"
        segments1 = split_with_offsets(text, mode="sentence", use_spacy=False)
        segments2 = split_with_offsets(text, mode="sentence", use_spacy=False)

        assert len(segments1) == len(segments2)
        for seg1, seg2 in zip(segments1, segments2, strict=False):
            assert seg1.id == seg2.id
            assert seg1.text == seg2.text
            assert seg1.char_start == seg2.char_start
            assert seg1.char_end == seg2.char_end
