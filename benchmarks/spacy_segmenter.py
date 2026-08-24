#!/usr/bin/env python
"""Raw spaCy sentence segmenter for benchmarking.

This script segments text into sentences using raw spaCy output
(no post-processing corrections) and outputs one sentence per line
for comparison with gold standard files.

This provides baseline spaCy performance to compare against phrasplit.
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


def split_sentences_raw(text: str, nlp) -> list[str]:
    """Split text into sentences using raw spaCy (no corrections).

    Uses raw spaCy sentence boundaries to preserve exact character positions.

    Args:
        text: Input text
        nlp: Loaded spaCy model

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


# Language to spaCy model mapping
LANG_TO_MODEL = {
    # Languages with dedicated spaCy models
    "Bulgarian": "en_core_web_sm",  # No Bulgarian model, fallback to English
    "Catalan": "ca_core_news_sm",
    "Chinese": "zh_core_web_sm",
    "Montenegrin": "en_core_web_sm",  # No Montenegrin model
    "Czech": "en_core_web_sm",  # No Czech model
    "Danish": "da_core_news_sm",
    "German": "de_core_news_sm",
    "Greek": "el_core_news_sm",
    "English": "en_core_web_sm",
    "Spanish": "es_core_news_sm",
    "Estonian": "en_core_web_sm",  # No Estonian model
    "Finnish": "fi_core_news_sm",
    "French": "fr_core_news_sm",
    "Croatian": "hr_core_news_sm",
    "Hungarian": "en_core_web_sm",  # No Hungarian model
    "Icelandic": "en_core_web_sm",  # No Icelandic model
    "Italian": "it_core_news_sm",
    "Japanese": "ja_core_news_sm",
    "Korean": "ko_core_news_sm",
    "Lithuanian": "lt_core_news_sm",
    "Latvian": "en_core_web_sm",  # No Latvian model
    "Macedonian": "mk_core_news_sm",
    "Maltese": "en_core_web_sm",  # No Maltese model
    "Norwegian-Bokmaal": "nb_core_news_sm",
    "Norwegian-Nynorsk": "nb_core_news_sm",  # Use Bokmaal for Nynorsk
    "Dutch": "nl_core_news_sm",
    "Polish": "pl_core_news_sm",
    "Portuguese": "pt_core_news_sm",
    "Romanian": "ro_core_news_sm",
    "Russian": "ru_core_news_sm",
    "Slovak": "en_core_web_sm",  # No Slovak model
    "Slovenian": "sl_core_news_sm",
    "Albanian": "en_core_web_sm",  # No Albanian model
    "Serbian": "en_core_web_sm",  # No Serbian model
    "Swedish": "sv_core_news_sm",
    "Turkish": "en_core_web_sm",  # No Turkish model
    "Ukrainian": "uk_core_news_sm",
}

# Language code to language name mapping
LANG_CODE_TO_NAME = {
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cnr": "Montenegrin",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "hr": "Croatian",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "mt": "Maltese",
    "nb": "Norwegian-Bokmaal",
    "nl": "Dutch",
    "nn": "Norwegian-Nynorsk",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
}


def get_spacy_model(lang: str) -> str:
    """Get the appropriate spaCy model for a language.

    Args:
        lang: Language name or code

    Returns:
        spaCy model name
    """
    # Convert language code to name if needed
    if lang in LANG_CODE_TO_NAME:
        lang = LANG_CODE_TO_NAME[lang]

    return LANG_TO_MODEL.get(lang, "en_core_web_sm")


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
        help="Override spaCy model (e.g., en_core_web_lg)",
    )

    args = parser.parse_args()

    # Get the appropriate spaCy model
    if args.model:
        model_name = args.model
    else:
        model_name = get_spacy_model(args.lang)

    # Try to load the model, fall back to English if not available
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(
            f"Warning: Model {model_name} not found, falling back to en_core_web_sm",
            file=sys.stderr,
        )
        nlp = spacy.load("en_core_web_sm")

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
                    # Split the chunk into sentences (raw spaCy, no corrections)
                    sentences = split_sentences_raw(chunk, nlp)

                    # Write each sentence on its own line
                    output_f.writelines(sentence + "\n" for sentence in sentences)


if __name__ == "__main__":
    main()
