#!/usr/bin/env python
"""Download Universal Dependencies datasets and build test sets.

This script downloads CoNLL-U files from Universal Dependencies repositories
and extracts sentences to create gold standard files for benchmarking.

Usage:
    python build-testset.py                    # Build from existing .gold files
    python build-testset.py --download         # Download and build all datasets
    python build-testset.py --download en      # Download and build English only
    python build-testset.py --list             # List available datasets

The script creates three test variants for each gold standard:
    - .all:   Same as gold (one sentence per line)
    - .none:  All text on one line (no breaks)
    - .mixed: Paragraph-like (3-8 sentences per line)
"""

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from random import randint, seed

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
TESTSETS_DIR = SCRIPT_DIR / "testsets"

# Base URL for Universal Dependencies raw files
UD_BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies"


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str  # Language name for output file
    repo: str  # GitHub repo name
    file: str  # CoNLL-U filename
    prefix: str = "UD"  # Output file prefix


# Dataset configurations
# Format: lang_code -> DatasetConfig
DATASETS: dict[str, DatasetConfig] = {
    "bg": DatasetConfig(
        "Bulgarian",
        "UD_Bulgarian-BTB",
        "bg_btb-ud-train.conllu",
    ),
    "cnr": DatasetConfig(
        "Montenegrin",
        "UD_Montenegrin-MESUBS",
        "cnr_mesubs-ud-train.conllu",
        prefix="MESUBS",
    ),
    "cs": DatasetConfig(
        "Czech",
        "UD_Czech-PDT",
        "cs_pdt-ud-train.conllu",
    ),
    "da": DatasetConfig(
        "Danish",
        "UD_Danish-DDT",
        "da_ddt-ud-train.conllu",
    ),
    "de": DatasetConfig(
        "German",
        "UD_German-GSD",
        "de_gsd-ud-train.conllu",
    ),
    "el": DatasetConfig(
        "Greek",
        "UD_Greek-GDT",
        "el_gdt-ud-train.conllu",
    ),
    "en": DatasetConfig(
        "English",
        "UD_English-EWT",
        "en_ewt-ud-train.conllu",
    ),
    "es": DatasetConfig(
        "Spanish",
        "UD_Spanish-GSD",
        "es_gsd-ud-train.conllu",
    ),
    "et": DatasetConfig(
        "Estonian",
        "UD_Estonian-EDT",
        "et_edt-ud-train.conllu",
    ),
    "fi": DatasetConfig(
        "Finnish",
        "UD_Finnish-TDT",
        "fi_tdt-ud-train.conllu",
    ),
    "fr": DatasetConfig(
        "French",
        "UD_French-GSD",
        "fr_gsd-ud-train.conllu",
    ),
    "hr": DatasetConfig(
        "Croatian",
        "UD_Croatian-SET",
        "hr_set-ud-train.conllu",
    ),
    "hu": DatasetConfig(
        "Hungarian",
        "UD_Hungarian-Szeged",
        "hu_szeged-ud-train.conllu",
    ),
    "is": DatasetConfig(
        "Icelandic",
        "UD_Icelandic-IcePaHC",
        "is_icepahc-ud-train.conllu",
    ),
    "it": DatasetConfig(
        "Italian",
        "UD_Italian-ISDT",
        "it_isdt-ud-train.conllu",
    ),
    "lt": DatasetConfig(
        "Lithuanian",
        "UD_Lithuanian-ALKSNIS",
        "lt_alksnis-ud-train.conllu",
    ),
    "lv": DatasetConfig(
        "Latvian",
        "UD_Latvian-LVTB",
        "lv_lvtb-ud-train.conllu",
    ),
    "mk": DatasetConfig(
        "Macedonian",
        "UD_Macedonian-MTB",
        "mk_mtb-ud-train.conllu",
        prefix="SETIMES",
    ),
    "mt": DatasetConfig(
        "Maltese",
        "UD_Maltese-MUDT",
        "mt_mudt-ud-train.conllu",
    ),
    "nb": DatasetConfig(
        "Norwegian-Bokmaal",
        "UD_Norwegian-Bokmaal",
        "no_bokmaal-ud-train.conllu",
    ),
    "nl": DatasetConfig(
        "Dutch",
        "UD_Dutch-Alpino",
        "nl_alpino-ud-train.conllu",
    ),
    "nn": DatasetConfig(
        "Norwegian-Nynorsk",
        "UD_Norwegian-Nynorsk",
        "no_nynorsk-ud-train.conllu",
    ),
    "pl": DatasetConfig(
        "Polish",
        "UD_Polish-PDB",
        "pl_pdb-ud-train.conllu",
    ),
    "pt": DatasetConfig(
        "Portuguese",
        "UD_Portuguese-Bosque",
        "pt_bosque-ud-train.conllu",
    ),
    "ro": DatasetConfig(
        "Romanian",
        "UD_Romanian-RRT",
        "ro_rrt-ud-train.conllu",
    ),
    "sk": DatasetConfig(
        "Slovak",
        "UD_Slovak-SNK",
        "sk_snk-ud-train.conllu",
    ),
    "sl": DatasetConfig(
        "Slovenian",
        "UD_Slovenian-SSJ",
        "sl_ssj-ud-train.conllu",
    ),
    "sq": DatasetConfig(
        "Albanian",
        "UD_Albanian-TSA",
        "sq_tsa-ud-train.conllu",
        prefix="SETIMES",
    ),
    "sr": DatasetConfig(
        "Serbian",
        "UD_Serbian-SET",
        "sr_set-ud-train.conllu",
    ),
    "sv": DatasetConfig(
        "Swedish",
        "UD_Swedish-Talbanken",
        "sv_talbanken-ud-train.conllu",
    ),
    "tr": DatasetConfig(
        "Turkish",
        "UD_Turkish-IMST",
        "tr_imst-ud-train.conllu",
    ),
    "uk": DatasetConfig(
        "Ukrainian",
        "UD_Ukrainian-IU",
        "uk_iu-ud-train.conllu",
    ),
}


def extract_sentences_from_conllu(content: str) -> list[str]:
    """Extract sentence texts from CoNLL-U format.

    Args:
        content: CoNLL-U file content

    Returns:
        List of sentence strings
    """
    sentences = []

    for line in content.split("\n"):
        # Look for # text = ... lines
        if line.startswith("# text = "):
            sentence = line[9:].strip()  # Remove "# text = " prefix
            if sentence:
                sentences.append(sentence)

    return sentences


def download_dataset(lang_code: str, config: DatasetConfig) -> list[str] | None:
    """Download a CoNLL-U dataset and extract sentences.

    Args:
        lang_code: Language code
        config: Dataset configuration

    Returns:
        List of sentences or None on error
    """
    url = f"{UD_BASE_URL}/{config.repo}/refs/heads/master/{config.file}"

    print(f"  Downloading from {config.repo}...")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8")
    except Exception as e:
        print(f"  Error downloading {lang_code}: {e}")
        return None

    sentences = extract_sentences_from_conllu(content)
    print(f"  Extracted {len(sentences)} sentences")

    return sentences


def save_gold_file(sentences: list[str], output_path: Path) -> None:
    """Save sentences to a gold standard file.

    Args:
        sentences: List of sentences
        output_path: Output file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(sentence + "\n" for sentence in sentences)


def build_test_variants(gold_path: Path) -> None:
    """Build test variants from a gold standard file.

    Creates three variants:
    - .all:   Same as gold (one sentence per line)
    - .none:  All text on one line
    - .mixed: Paragraph-like (3-8 sentences per line)

    Args:
        gold_path: Path to gold standard file
    """
    # Read sentences from gold file
    with open(gold_path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if not sentences:
        print(f"  Warning: No sentences in {gold_path}")
        return

    base_name = str(gold_path).replace(".gold", "")

    # All: Same as gold
    all_path = Path(base_name + ".all")
    with open(all_path, "w", encoding="utf-8") as f:
        f.writelines(sent + "\n" for sent in sentences)

    # None: All on one line
    none_path = Path(base_name + ".none")
    with open(none_path, "w", encoding="utf-8") as f:
        f.write(" ".join(sentences) + "\n")

    # Mixed: Paragraph-like (3-8 sentences per paragraph)
    # Use fixed seed for reproducibility
    seed(42)
    mixed_path = Path(base_name + ".mixed")
    with open(mixed_path, "w", encoding="utf-8") as f:
        paragraph = []
        paragraph_length = 0

        for sent in sentences:
            if paragraph_length == 0:
                paragraph_length = randint(3, 8)

            paragraph.append(sent)
            paragraph_length -= 1

            if paragraph_length == 0:
                f.write(" ".join(paragraph) + "\n")
                paragraph = []

        # Write remaining sentences
        if paragraph:
            f.write(" ".join(paragraph) + "\n")

    print("  Created: .all, .none, .mixed variants")


def download_and_build(lang_code: str) -> bool:
    """Download dataset and build all test files for a language.

    Args:
        lang_code: Language code

    Returns:
        True on success
    """
    if lang_code not in DATASETS:
        print(f"Unknown language code: {lang_code}")
        return False

    config = DATASETS[lang_code]
    print(f"\n[{lang_code}] {config.name}")

    # Download and extract sentences
    sentences = download_dataset(lang_code, config)
    if not sentences:
        return False

    # Ensure testsets directory exists
    TESTSETS_DIR.mkdir(exist_ok=True)

    # Save gold file
    gold_path = TESTSETS_DIR / f"{config.prefix}_{config.name}.dataset.gold"
    save_gold_file(sentences, gold_path)
    print(f"  Saved: {gold_path.name}")

    # Build test variants
    build_test_variants(gold_path)

    return True


def build_from_existing() -> None:
    """Build test variants from existing gold files."""
    if not TESTSETS_DIR.exists():
        print(f"Testsets directory not found: {TESTSETS_DIR}")
        return

    gold_files = list(TESTSETS_DIR.glob("*.gold"))
    if not gold_files:
        print("No .gold files found in testsets directory")
        return

    print(f"Building test variants from {len(gold_files)} gold files...")

    for gold_path in sorted(gold_files):
        print(f"\n{gold_path.name}")
        build_test_variants(gold_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "lang_codes",
        nargs="*",
        help="Language codes to download (default: all)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets from Universal Dependencies",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for code, config in sorted(DATASETS.items()):
            print(f"  {code}: {config.name} ({config.repo})")
        return

    if args.download:
        # Download mode
        if args.lang_codes:
            # Download specific languages
            for lang_code in args.lang_codes:
                download_and_build(lang_code)
        else:
            # Download all
            print(f"Downloading {len(DATASETS)} datasets...")
            success = 0
            failed = []

            for lang_code in sorted(DATASETS.keys()):
                if download_and_build(lang_code):
                    success += 1
                else:
                    failed.append(lang_code)

            print(f"\n{'=' * 40}")
            print(f"Downloaded: {success}/{len(DATASETS)}")
            if failed:
                print(f"Failed: {', '.join(failed)}")
    else:
        # Build from existing gold files
        build_from_existing()


if __name__ == "__main__":
    main()
