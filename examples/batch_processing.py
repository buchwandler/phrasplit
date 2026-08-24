#!/usr/bin/env python3
"""Batch processing example using phrasplit.

This script demonstrates how to process multiple files efficiently
using phrasplit, with progress tracking and error handling.
"""

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from phrasplit import split_clauses, split_paragraphs, split_sentences


@dataclass
class ProcessingResult:
    """Result of processing a single file."""

    filepath: str
    success: bool
    output_count: int
    error_message: str | None = None

    def __str__(self) -> str:
        if self.success:
            return f"{self.filepath}: {self.output_count} items"
        return f"{self.filepath}: ERROR - {self.error_message}"


def process_file(
    filepath: str | Path,
    split_func: Callable[[str], list[str]],
    encoding: str = "utf-8",
) -> ProcessingResult:
    """
    Process a single file with the given splitting function.

    Args:
        filepath: Path to the input file
        split_func: Splitting function to apply (split_sentences, etc.)
        encoding: File encoding

    Returns:
        ProcessingResult with success status and output count
    """
    filepath = Path(filepath)

    try:
        text = filepath.read_text(encoding=encoding)
        result = split_func(text)

        return ProcessingResult(
            filepath=str(filepath),
            success=True,
            output_count=len(result),
        )
    except FileNotFoundError:
        return ProcessingResult(
            filepath=str(filepath),
            success=False,
            output_count=0,
            error_message="File not found",
        )
    except UnicodeDecodeError as e:
        return ProcessingResult(
            filepath=str(filepath),
            success=False,
            output_count=0,
            error_message=f"Encoding error: {e}",
        )
    except Exception as e:  # noqa: BLE001
        return ProcessingResult(
            filepath=str(filepath),
            success=False,
            output_count=0,
            error_message=str(e),
        )


def process_files_sequential(
    filepaths: Sequence[str | Path],
    split_func: Callable[[str], list[str]],
    callback: Callable[[ProcessingResult], None] | None = None,
) -> list[ProcessingResult]:
    """
    Process multiple files sequentially.

    Args:
        filepaths: List of file paths to process
        split_func: Splitting function to apply
        callback: Optional callback called after each file is processed

    Returns:
        List of ProcessingResult objects
    """
    results = []

    for filepath in filepaths:
        result = process_file(filepath, split_func)
        results.append(result)

        if callback:
            callback(result)

    return results


def process_files_parallel(
    filepaths: Sequence[str | Path],
    split_func: Callable[[str], list[str]],
    max_workers: int = 4,
    callback: Callable[[ProcessingResult], None] | None = None,
) -> list[ProcessingResult]:
    """
    Process multiple files in parallel using threads.

    Args:
        filepaths: List of file paths to process
        split_func: Splitting function to apply
        max_workers: Maximum number of parallel workers
        callback: Optional callback called after each file completes

    Returns:
        List of ProcessingResult objects
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_path = {
            executor.submit(process_file, fp, split_func): fp for fp in filepaths
        }

        # Collect results as they complete
        for future in as_completed(future_to_path):
            result = future.result()
            results.append(result)

            if callback:
                callback(result)

    return results


def process_directory(
    directory: str | Path,
    pattern: str = "*.txt",
    split_func: Callable[[str], list[str]] = split_sentences,
    recursive: bool = False,
) -> list[ProcessingResult]:
    """
    Process all matching files in a directory.

    Args:
        directory: Directory path
        pattern: Glob pattern for matching files
        split_func: Splitting function to apply
        recursive: Whether to search recursively

    Returns:
        List of ProcessingResult objects
    """
    directory = Path(directory)

    if recursive:
        filepaths = list(directory.rglob(pattern))
    else:
        filepaths = list(directory.glob(pattern))

    return process_files_sequential(filepaths, split_func)


def batch_split_to_files(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.txt",
    split_func: Callable[[str], list[str]] = split_sentences,
    separator: str = "\n",
) -> list[ProcessingResult]:
    """
    Split files and write results to output directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory (will be created if needed)
        pattern: Glob pattern for matching input files
        split_func: Splitting function to apply
        separator: Separator between split items in output

    Returns:
        List of ProcessingResult objects
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for input_path in input_dir.glob(pattern):
        try:
            text = input_path.read_text()
            split_result = split_func(text)

            # Create output filename
            output_path = output_dir / f"{input_path.stem}_split{input_path.suffix}"
            output_path.write_text(separator.join(split_result))

            results.append(
                ProcessingResult(
                    filepath=str(input_path),
                    success=True,
                    output_count=len(split_result),
                )
            )
        except Exception as e:  # noqa: BLE001
            results.append(
                ProcessingResult(
                    filepath=str(input_path),
                    success=False,
                    output_count=0,
                    error_message=str(e),
                )
            )

    return results


def generate_batch_report(results: list[ProcessingResult]) -> str:
    """
    Generate a summary report for batch processing results.

    Args:
        results: List of ProcessingResult objects

    Returns:
        Formatted report string
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    total_items = sum(r.output_count for r in results if r.success)

    lines = [
        "=" * 60,
        "BATCH PROCESSING REPORT",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 40,
        f"  Total files: {total}",
        f"  Successful: {successful}",
        f"  Failed: {failed}",
        f"  Total items processed: {total_items}",
        "",
    ]

    if successful > 0:
        lines.extend(
            [
                "SUCCESSFUL FILES",
                "-" * 40,
            ]
        )
        for r in results:
            if r.success:
                lines.append(f"  {r.filepath}: {r.output_count} items")
        lines.append("")

    if failed > 0:
        lines.extend(
            [
                "FAILED FILES",
                "-" * 40,
            ]
        )
        for r in results:
            if not r.success:
                lines.append(f"  {r.filepath}: {r.error_message}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def create_sample_files(directory: Path) -> list[Path]:
    """Create sample text files for demonstration."""
    directory.mkdir(parents=True, exist_ok=True)

    samples = {
        "sample1.txt": """The quick brown fox jumps over the lazy dog.
This is a simple sentence. And here is another one!

A second paragraph begins here. It contains multiple sentences.
Each sentence will be processed separately.""",
        "sample2.txt": """Technology has transformed our daily lives.
From smartphones to smart homes, we are surrounded by innovation.

Artificial intelligence is becoming more prevalent.
Machine learning algorithms power many of the services we use daily.
The future promises even more exciting developments.""",
        "sample3.txt": """Writing good code is an art. It requires practice.
Clean code is easy to read and maintain. Comments help explain complex logic.

Testing is crucial for software quality.
Unit tests catch bugs early. Integration tests ensure components work together.
Never skip the testing phase!""",
    }

    created_files = []
    for filename, content in samples.items():
        filepath = directory / filename
        filepath.write_text(content)
        created_files.append(filepath)

    return created_files


def main() -> None:
    """Demonstrate batch processing features."""
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)

    # Create temporary sample files
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = Path(tmpdir) / "samples"
        output_dir = Path(tmpdir) / "output"

        print("\n1. CREATING SAMPLE FILES")
        print("-" * 40)
        sample_files = create_sample_files(sample_dir)
        print(f"Created {len(sample_files)} sample files in {sample_dir}")
        for f in sample_files:
            print(f"  - {f.name}")

        # Sequential processing
        print("\n\n2. SEQUENTIAL PROCESSING")
        print("-" * 40)

        def progress_callback(result: ProcessingResult) -> None:
            status = "OK" if result.success else "FAIL"
            print(f"  [{status}] {os.path.basename(result.filepath)}")

        results = process_files_sequential(
            sample_files,
            split_sentences,
            callback=progress_callback,
        )
        print(f"\nProcessed {len(results)} files")

        # Parallel processing
        print("\n\n3. PARALLEL PROCESSING")
        print("-" * 40)
        results = process_files_parallel(
            sample_files,
            split_sentences,
            max_workers=2,
            callback=progress_callback,
        )
        print(f"\nProcessed {len(results)} files in parallel")

        # Directory processing
        print("\n\n4. DIRECTORY PROCESSING")
        print("-" * 40)
        results = process_directory(
            sample_dir,
            pattern="*.txt",
            split_func=split_paragraphs,
        )
        for r in results:
            print(f"  {os.path.basename(r.filepath)}: {r.output_count} paragraphs")

        # Different splitting functions
        print("\n\n5. COMPARING SPLIT FUNCTIONS")
        print("-" * 40)
        test_file = sample_files[0]
        print(f"File: {test_file.name}")

        for name, func in [
            ("Sentences", split_sentences),
            ("Clauses", split_clauses),
            ("Paragraphs", split_paragraphs),
        ]:
            result = process_file(test_file, func)
            print(f"  {name}: {result.output_count} items")

        # Batch split to files
        print("\n\n6. BATCH SPLIT TO OUTPUT FILES")
        print("-" * 40)
        results = batch_split_to_files(
            sample_dir,
            output_dir,
            pattern="*.txt",
            split_func=split_sentences,
            separator="\n---\n",
        )
        print(f"Output directory: {output_dir}")
        for output_file in output_dir.glob("*.txt"):
            print(f"  Created: {output_file.name}")

        # Generate report
        print("\n\n7. BATCH PROCESSING REPORT")
        report = generate_batch_report(results)
        print(report)

        # Error handling demonstration
        print("\n8. ERROR HANDLING")
        print("-" * 40)
        # Include a non-existent file
        test_paths = [*sample_files, Path("nonexistent.txt")]
        results = process_files_sequential(test_paths, split_sentences)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        for r in failed:
            print(f"    - {r.filepath}: {r.error_message}")


if __name__ == "__main__":
    main()
