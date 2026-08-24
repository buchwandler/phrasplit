#!/usr/bin/env python
"""spaCy Sentencizer-based sentence segmenter for benchmarking.

This script segments text into sentences using spaCy's rule-based Sentencizer
component (not the dependency parser) and outputs one sentence per line
for comparison with gold standard files.

The Sentencizer is a fast, rule-based alternative to parser-based sentence
segmentation. It uses punctuation characters to determine sentence boundaries.

This provides a baseline for rule-based segmentation to compare against
parser-based spaCy and phrasplit.
"""

import argparse
import os
import sys

import spacy

# Maximum characters to process at once (spaCy default limit is 1,000,000)
MAX_CHUNK_SIZE = 500000


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split text into chunks that won't exceed spaCy's max_length.

    Splits at sentence-ending punctuation followed by whitespace to minimize
    breaking sentences across chunks.

    Args:
        text: The text to chunk
        max_size: Maximum size of each chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_size:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_size:
            chunks.append(remaining)
            break

        # Find a good split point (sentence boundary) before max_size
        chunk = remaining[:max_size]

        # Look for last sentence-ending punctuation followed by space
        # Search backwards from the end of the chunk
        split_pos = -1
        for i in range(len(chunk) - 1, max(0, len(chunk) - 10000), -1):
            if chunk[i] in ".!?" and i + 1 < len(chunk) and chunk[i + 1] in " \n\t":
                split_pos = i + 1
                break

        if split_pos == -1:
            # No good split point found, just split at max_size
            split_pos = max_size

        chunks.append(remaining[:split_pos])
        remaining = remaining[split_pos:].lstrip(" \t\n\r")

    return chunks


def split_sentences_sentencizer(text: str, nlp) -> list[str]:
    """Split text into sentences using spaCy's Sentencizer.

    Uses rule-based sentence boundaries (punctuation-based).

    Args:
        text: Input text
        nlp: Loaded spaCy blank model with sentencizer

    Returns:
        List of sentences with original characters preserved
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        # Use character spans to preserve original text exactly
        # Only strip regular whitespace (space, tab, newline), not nbsp
        sent_text = text[sent.start_char : sent.end_char].strip(" \t\n\r")
        if sent_text:
            sentences.append(sent_text)

    return sentences


# Language code to spaCy language code mapping
# Some languages use different codes in spaCy
LANG_CODE_TO_SPACY = {
    "bg": "bg",  # Bulgarian
    "ca": "ca",  # Catalan
    "cnr": "sr",  # Montenegrin -> Serbian (closest)
    "cs": "cs",  # Czech
    "da": "da",  # Danish
    "de": "de",  # German
    "el": "el",  # Greek
    "en": "en",  # English
    "es": "es",  # Spanish
    "et": "et",  # Estonian
    "fi": "fi",  # Finnish
    "fr": "fr",  # French
    "hr": "hr",  # Croatian
    "hu": "hu",  # Hungarian
    "is": "is",  # Icelandic
    "it": "it",  # Italian
    "ja": "ja",  # Japanese
    "ko": "ko",  # Korean
    "lt": "lt",  # Lithuanian
    "lv": "lv",  # Latvian
    "mk": "mk",  # Macedonian
    "mt": "mt",  # Maltese
    "nb": "nb",  # Norwegian Bokmål
    "nl": "nl",  # Dutch
    "nn": "nn",  # Norwegian Nynorsk
    "pl": "pl",  # Polish
    "pt": "pt",  # Portuguese
    "ro": "ro",  # Romanian
    "ru": "ru",  # Russian
    "sk": "sk",  # Slovak
    "sl": "sl",  # Slovenian
    "sq": "sq",  # Albanian
    "sr": "sr",  # Serbian
    "sv": "sv",  # Swedish
    "tr": "tr",  # Turkish
    "uk": "uk",  # Ukrainian
    "zh": "zh",  # Chinese
}

# Language name to code mapping
LANG_NAME_TO_CODE = {
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Montenegrin": "cnr",
    "Czech": "cs",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Macedonian": "mk",
    "Maltese": "mt",
    "Norwegian-Bokmaal": "nb",
    "Dutch": "nl",
    "Norwegian-Nynorsk": "nn",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Albanian": "sq",
    "Serbian": "sr",
    "Swedish": "sv",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Chinese": "zh",
}


def get_spacy_lang_code(lang: str) -> str:
    """Get the spaCy language code for a language.

    Args:
        lang: Language name or code

    Returns:
        spaCy language code
    """
    # If it's a language name, convert to code first
    if lang in LANG_NAME_TO_CODE:
        lang = LANG_NAME_TO_CODE[lang]

    # Get spaCy language code
    return LANG_CODE_TO_SPACY.get(lang, "en")


def create_sentencizer_nlp(lang_code: str):
    """Create a spaCy blank model with sentencizer.

    Args:
        lang_code: spaCy language code (e.g., 'en', 'de')

    Returns:
        spaCy Language object with sentencizer
    """
    try:
        nlp = spacy.blank(lang_code)
    except Exception:
        # Fall back to English if language not supported
        print(
            f"Warning: Language '{lang_code}' not supported, falling back to English",
            file=sys.stderr,
        )
        nlp = spacy.blank("en")

    # Add the sentencizer component
    nlp.add_pipe("sentencizer")

    return nlp


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    parser.add_argument("lang", type=str, help="Language name or code")
    parser.add_argument("inputfile", type=str, help="Input file")
    parser.add_argument("outputfile", type=str, help="Output file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ignored (for compatibility with other segmenters)",
    )

    args = parser.parse_args()

    # Get the spaCy language code
    spacy_lang = get_spacy_lang_code(args.lang)

    # Create blank model with sentencizer
    nlp = create_sentencizer_nlp(spacy_lang)

    with open(args.inputfile, encoding="utf-8") as input_f:
        with open(args.outputfile, "w", encoding="utf-8") as output_f:
            for line in input_f:
                # Only strip regular whitespace, preserve nbsp
                line = line.strip(" \t\n\r")
                if not line:
                    continue

                # Chunk the line if it's too long for spaCy
                chunks = chunk_text(line)

                for chunk in chunks:
                    # Split the chunk into sentences using sentencizer
                    sentences = split_sentences_sentencizer(chunk, nlp)

                    # Write each sentence on its own line
                    output_f.writelines(sentence + "\n" for sentence in sentences)


if __name__ == "__main__":
    main()
