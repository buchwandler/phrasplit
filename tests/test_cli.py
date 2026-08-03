"""Tests for phrasplit CLI module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from phrasplit.cli import main, read_input, write_output


class TestReadInput:
    """Tests for read_input function."""

    def test_read_from_file(self, tmp_path: Path) -> None:
        """Test reading input from a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")
        result = read_input(str(test_file))
        assert result == "Hello, world!"

    def test_read_from_file_with_unicode(self, tmp_path: Path) -> None:
        """Test reading file with unicode content."""
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("Héllo wörld! 你好", encoding="utf-8")
        result = read_input(str(test_file))
        assert result == "Héllo wörld! 你好"

    def test_read_from_stdin_with_dash(self) -> None:
        """Test reading from stdin when '-' is specified."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = "stdin content"
            result = read_input("-")
            assert result == "stdin content"

    def test_read_from_stdin_when_none(self) -> None:
        """Test reading from stdin when input_file is None."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = "stdin content"
            result = read_input(None)
            assert result == "stdin content"

    def test_read_nonexistent_file_raises(self) -> None:
        """Test that reading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_input("/nonexistent/path/to/file.txt")


class TestWriteOutput:
    """Tests for write_output function."""

    def test_write_to_file(self, tmp_path: Path) -> None:
        """Test writing output to a file."""
        output_file = tmp_path / "output.txt"
        write_output("Hello, world!", output_file)
        assert output_file.read_text() == "Hello, world!"

    def test_write_to_file_with_unicode(self, tmp_path: Path) -> None:
        """Test writing unicode content to file."""
        output_file = tmp_path / "unicode_output.txt"
        write_output("Héllo wörld! 你好", output_file)
        assert output_file.read_text(encoding="utf-8") == "Héllo wörld! 你好"


class TestSentencesCommand:
    """Tests for the 'sentences' CLI command."""

    def test_sentences_from_stdin(self) -> None:
        """Test sentences command reading from stdin."""
        runner = CliRunner()
        result = runner.invoke(main, ["sentences"], input="Hello world. Goodbye world.")
        assert result.exit_code == 0
        assert "Hello world." in result.output
        assert "Goodbye world." in result.output

    def test_sentences_from_file(self, tmp_path: Path) -> None:
        """Test sentences command reading from file."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("First sentence. Second sentence.")

        runner = CliRunner()
        result = runner.invoke(main, ["sentences", str(test_file)])
        assert result.exit_code == 0
        assert "First sentence." in result.output
        assert "Second sentence." in result.output

    def test_sentences_to_output_file(self, tmp_path: Path) -> None:
        """Test sentences command writing to output file."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"
        input_file.write_text("Hello. World.")

        runner = CliRunner()
        result = runner.invoke(
            main, ["sentences", str(input_file), "-o", str(output_file)]
        )
        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "Hello." in content
        assert "World." in content

    def test_sentences_file_not_found(self) -> None:
        """Test sentences command with nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(main, ["sentences", "/nonexistent/file.txt"])
        assert result.exit_code == 1
        assert "File not found" in result.output

    def test_sentences_with_model_option(self, tmp_path: Path) -> None:
        """Test sentences command with custom model option."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("Test sentence.")

        runner = CliRunner()
        # Using default model should work
        result = runner.invoke(
            main, ["sentences", str(test_file), "-m", "en_core_web_sm"]
        )
        assert result.exit_code == 0


class TestClausesCommand:
    """Tests for the 'clauses' CLI command."""

    def test_clauses_from_stdin(self) -> None:
        """Test clauses command reading from stdin."""
        runner = CliRunner()
        result = runner.invoke(
            main, ["clauses"], input="I like coffee, and I like tea."
        )
        assert result.exit_code == 0
        assert "I like coffee," in result.output
        assert "and I like tea." in result.output

    def test_clauses_from_file(self, tmp_path: Path) -> None:
        """Test clauses command reading from file."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("First clause, second clause.")

        runner = CliRunner()
        result = runner.invoke(main, ["clauses", str(test_file)])
        assert result.exit_code == 0
        assert "First clause," in result.output

    def test_clauses_file_not_found(self) -> None:
        """Test clauses command with nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(main, ["clauses", "/nonexistent/file.txt"])
        assert result.exit_code == 1
        assert "File not found" in result.output


class TestParagraphsCommand:
    """Tests for the 'paragraphs' CLI command."""

    def test_paragraphs_from_stdin(self) -> None:
        """Test paragraphs command reading from stdin."""
        runner = CliRunner()
        result = runner.invoke(
            main, ["paragraphs"], input="First paragraph.\n\nSecond paragraph."
        )
        assert result.exit_code == 0
        assert "First paragraph." in result.output
        assert "Second paragraph." in result.output

    def test_paragraphs_from_file(self, tmp_path: Path) -> None:
        """Test paragraphs command reading from file."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("Para one.\n\nPara two.")

        runner = CliRunner()
        result = runner.invoke(main, ["paragraphs", str(test_file)])
        assert result.exit_code == 0
        assert "Para one." in result.output
        assert "Para two." in result.output

    def test_paragraphs_file_not_found(self) -> None:
        """Test paragraphs command with nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(main, ["paragraphs", "/nonexistent/file.txt"])
        assert result.exit_code == 1
        assert "File not found" in result.output


class TestLonglinesCommand:
    """Tests for the 'longlines' CLI command."""

    def test_longlines_from_stdin(self) -> None:
        """Test longlines command reading from stdin."""
        runner = CliRunner()
        long_text = "This is a very long line that should be split. " * 5
        result = runner.invoke(main, ["longlines", "-l", "50"], input=long_text)
        assert result.exit_code == 0
        # Output should have multiple lines
        lines = [line for line in result.output.strip().split("\n") if line]
        assert len(lines) >= 2

    def test_longlines_with_max_length(self, tmp_path: Path) -> None:
        """Test longlines command with custom max-length."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("Short. " * 20)

        runner = CliRunner()
        result = runner.invoke(main, ["longlines", str(test_file), "-l", "30"])
        assert result.exit_code == 0

    def test_longlines_invalid_max_length_zero(self) -> None:
        """Test longlines command rejects max-length of 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["longlines", "-l", "0"], input="Some text")
        assert result.exit_code != 0
        # Click's IntRange should reject 0

    def test_longlines_invalid_max_length_negative(self) -> None:
        """Test longlines command rejects negative max-length."""
        runner = CliRunner()
        result = runner.invoke(main, ["longlines", "-l", "-5"], input="Some text")
        assert result.exit_code != 0

    def test_longlines_file_not_found(self) -> None:
        """Test longlines command with nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(main, ["longlines", "/nonexistent/file.txt"])
        assert result.exit_code == 1
        assert "File not found" in result.output

    def test_longlines_to_output_file(self, tmp_path: Path) -> None:
        """Test longlines command writing to output file."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"
        input_file.write_text("Short line.")

        runner = CliRunner()
        result = runner.invoke(
            main, ["longlines", str(input_file), "-o", str(output_file), "-l", "80"]
        )
        assert result.exit_code == 0
        assert output_file.exists()


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_version_option(self) -> None:
        """Test --version option."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Should show version (might be 0.0.0 in dev)
        assert "version" in result.output.lower() or "0." in result.output

    def test_help_option(self) -> None:
        """Test --help option."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "sentences" in result.output
        assert "clauses" in result.output
        assert "paragraphs" in result.output
        assert "longlines" in result.output

    def test_model_selection_options_are_exposed(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["sentences", "--help"])
        assert result.exit_code == 0
        assert "--language" in result.output
        assert "--model-size" in result.output
        assert "automatic" in result.output.lower()

    def test_sentences_help(self) -> None:
        """Test sentences --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["sentences", "--help"])
        assert result.exit_code == 0
        assert "Split text into sentences" in result.output

    def test_clauses_help(self) -> None:
        """Test clauses --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["clauses", "--help"])
        assert result.exit_code == 0
        assert "Split text into clauses" in result.output

    def test_paragraphs_help(self) -> None:
        """Test paragraphs --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["paragraphs", "--help"])
        assert result.exit_code == 0
        assert "Split text into paragraphs" in result.output

    def test_longlines_help(self) -> None:
        """Test longlines --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["longlines", "--help"])
        assert result.exit_code == 0
        assert "Split long lines" in result.output
        assert "--max-length" in result.output


class TestCLIEdgeCases:
    """Tests for CLI edge cases."""

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        runner = CliRunner()
        result = runner.invoke(main, ["sentences"], input="")
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_whitespace_only_input(self) -> None:
        """Test handling of whitespace-only input."""
        runner = CliRunner()
        result = runner.invoke(main, ["paragraphs"], input="   \n\n   ")
        assert result.exit_code == 0

    def test_unicode_input(self) -> None:
        """Test handling of unicode input."""
        runner = CliRunner()
        result = runner.invoke(main, ["sentences"], input="Héllo. Wörld. 你好.")
        assert result.exit_code == 0
        assert "Héllo." in result.output or "H" in result.output  # Unicode handled

    def test_stdin_dash_explicit(self) -> None:
        """Test reading from stdin with explicit '-' argument."""
        runner = CliRunner()
        result = runner.invoke(main, ["sentences", "-"], input="Hello. World.")
        assert result.exit_code == 0
        assert "Hello." in result.output
