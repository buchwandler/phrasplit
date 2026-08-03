"""Command-line interface for phrasplit."""

import sys
from pathlib import Path

import click
from rich.console import Console

from . import splitter as splitter_module
from .splitter import split_clauses, split_long_lines, split_paragraphs, split_sentences

console = Console()
error_console = Console(stderr=True)


def _print_model_diagnostic(*, language: str, simple: bool, verbose: bool) -> None:
    if not verbose:
        return
    if simple or splitter_module.LAST_SPACY_MODEL is None:
        error_console.print(
            f"[dim]Using regex backend for language {language!r}; "
            "no spaCy model selected.[/dim]"
        )
    else:
        error_console.print(
            f"[dim]Using spaCy model {splitter_module.LAST_SPACY_MODEL!r} "
            f"for language {language!r}.[/dim]"
        )


def read_input(input_file: str | None) -> str:
    """Read input from file or stdin.

    Args:
        input_file: Path to input file, '-' for stdin, or None for stdin

    Returns:
        Text content
    """
    if input_file is None or input_file == "-":
        return sys.stdin.read()
    return Path(input_file).read_text(encoding="utf-8")


def write_output(text: str, output: Path | None, use_rich: bool = True) -> None:
    """Write output to file or stdout.

    Args:
        text: Text to write
        output: Output file path or None for stdout
        use_rich: Whether to use rich console for stdout
    """
    if output:
        output.write_text(text, encoding="utf-8")
        error_console.print(f"[green]Output written to {output}[/green]")
    elif use_rich:
        console.print(text)
    else:
        print(text)


@click.group()
@click.version_option()
def main() -> None:
    """Phrasplit - Split text into sentences, clauses, or paragraphs."""
    pass


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
@click.option(
    "-m",
    "--model",
    default=None,
    help="Exact spaCy model package (default: automatic local selection)",
)
@click.option(
    "--language",
    default="en",
    show_default=True,
    help="Language hint used for automatic model selection and abbreviations",
)
@click.option(
    "--model-size",
    type=click.Choice(["sm", "md", "lg", "trf"]),
    default=None,
    help="Exact automatic model tier; never falls back to another tier",
)
@click.option(
    "--simple",
    is_flag=True,
    help="Use simple regex-based splitting (faster, no spaCy required)",
)
@click.option(
    "--verbose", is_flag=True, help="Print backend and selected-model diagnostics"
)
def sentences(
    input_file: str | None,
    output: Path | None,
    model: str | None,
    language: str,
    model_size: str | None,
    simple: bool,
    verbose: bool,
) -> None:
    """Split text into sentences.

    INPUT_FILE: Path to input file, or '-' for stdin. Reads from stdin if omitted.

    By default, selects the highest-quality compatible installed and loadable spaCy
    model for the requested language, or falls back to regex. Use --simple to force
    regex-based splitting without spaCy.
    """
    try:
        text = read_input(input_file)
    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {input_file}")
        sys.exit(1)

    try:
        use_spacy = None if not simple else False
        result = split_sentences(
            text,
            language_model=model,
            language=language,
            model_size=model_size,  # type: ignore[arg-type]
            use_spacy=use_spacy,
        )
        _print_model_diagnostic(language=language, simple=simple, verbose=verbose)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        if isinstance(e, ImportError):
            error_console.print(
                "\n[yellow]Tip:[/yellow] Use --simple flag for regex-based splitting,"
            )
            error_console.print(
                "[yellow]or install spaCy for better accuracy:[/yellow]"
            )
            error_console.print("  pip install phrasplit[nlp]")
            error_console.print(
                "  Install a compatible local spaCy model; automatic mode never "
                "downloads one."
            )
        sys.exit(1)

    output_text = "\n".join(result)
    write_output(output_text, output)


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
@click.option(
    "-m",
    "--model",
    default=None,
    help="Exact spaCy model package (default: automatic local selection)",
)
@click.option("--language", default="en", show_default=True, help="Language hint")
@click.option(
    "--model-size",
    type=click.Choice(["sm", "md", "lg", "trf"]),
    default=None,
    help="Exact automatic model tier",
)
@click.option(
    "--simple",
    is_flag=True,
    help="Use simple regex-based splitting (faster, no spaCy required)",
)
@click.option(
    "--verbose", is_flag=True, help="Print backend and selected-model diagnostics"
)
def clauses(
    input_file: str | None,
    output: Path | None,
    model: str | None,
    language: str,
    model_size: str | None,
    simple: bool,
    verbose: bool,
) -> None:
    """Split text into clauses (at commas).

    INPUT_FILE: Path to input file, or '-' for stdin. Reads from stdin if omitted.

    By default, selects the highest-quality compatible installed and loadable spaCy
    model for the requested language, or falls back to regex. Use --simple to force
    regex-based splitting without spaCy.
    """
    try:
        text = read_input(input_file)
    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {input_file}")
        sys.exit(1)

    try:
        use_spacy = None if not simple else False
        result = split_clauses(
            text,
            language_model=model,
            language=language,
            model_size=model_size,  # type: ignore[arg-type]
            use_spacy=use_spacy,
        )
        _print_model_diagnostic(language=language, simple=simple, verbose=verbose)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        if isinstance(e, ImportError):
            error_console.print(
                "\n[yellow]Tip:[/yellow] Use --simple flag for regex-based splitting,"
            )
            error_console.print(
                "[yellow]or install spaCy for better accuracy:[/yellow]"
            )
            error_console.print("  pip install phrasplit[nlp]")
            error_console.print(
                "  Install a compatible local spaCy model; automatic mode never "
                "downloads one."
            )
        sys.exit(1)

    output_text = "\n".join(result)
    write_output(output_text, output)


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
def paragraphs(
    input_file: str | None,
    output: Path | None,
) -> None:
    """Split text into paragraphs.

    INPUT_FILE: Path to input file, or '-' for stdin. Reads from stdin if omitted.
    """
    try:
        text = read_input(input_file)
    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {input_file}")
        sys.exit(1)

    result = split_paragraphs(text)
    output_text = "\n\n".join(result)
    write_output(output_text, output)


@main.command()
@click.argument("input_file", required=False, default=None)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
@click.option(
    "-l",
    "--max-length",
    default=80,
    type=click.IntRange(min=1),
    help="Maximum line length (default: 80, must be >= 1)",
)
@click.option(
    "-m",
    "--model",
    default=None,
    help="Exact spaCy model package (default: automatic local selection)",
)
@click.option("--language", default="en", show_default=True, help="Language hint")
@click.option(
    "--model-size",
    type=click.Choice(["sm", "md", "lg", "trf"]),
    default=None,
    help="Exact automatic model tier",
)
@click.option(
    "--simple",
    is_flag=True,
    help="Use simple regex-based splitting (faster, no spaCy required)",
)
@click.option(
    "--verbose", is_flag=True, help="Print backend and selected-model diagnostics"
)
def longlines(
    input_file: str | None,
    output: Path | None,
    max_length: int,
    model: str | None,
    language: str,
    model_size: str | None,
    simple: bool,
    verbose: bool,
) -> None:
    """Split long lines at sentence/clause boundaries.

    INPUT_FILE: Path to input file, or '-' for stdin. Reads from stdin if omitted.

    By default, selects the highest-quality compatible installed and loadable spaCy
    model for the requested language, or falls back to regex. Use --simple to force
    regex-based splitting without spaCy.
    """
    try:
        text = read_input(input_file)
    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {input_file}")
        sys.exit(1)

    try:
        use_spacy = None if not simple else False
        result = split_long_lines(
            text,
            max_length=max_length,
            language_model=model,
            language=language,
            model_size=model_size,  # type: ignore[arg-type]
            use_spacy=use_spacy,
        )
        _print_model_diagnostic(language=language, simple=simple, verbose=verbose)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        error_console.print(f"[red]Error:[/red] {e}")
        if isinstance(e, ImportError):
            error_console.print(
                "\n[yellow]Tip:[/yellow] Use --simple flag for regex-based splitting,"
            )
            error_console.print(
                "[yellow]or install spaCy for better accuracy:[/yellow]"
            )
            error_console.print("  pip install phrasplit[nlp]")
            error_console.print(
                "  Install a compatible local spaCy model; automatic mode never "
                "downloads one."
            )
        sys.exit(1)

    output_text = "\n".join(result)
    write_output(output_text, output)


if __name__ == "__main__":
    main()
