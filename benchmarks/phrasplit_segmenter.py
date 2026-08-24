#!/usr/bin/env python
"""Phrasplit sentence segmenter for benchmarking.

This script segments text into sentences using the phrasplit library
and outputs one sentence per line for comparison with gold standard files.

This tests the actual phrasplit.split_sentences() function including:
- Whitespace normalization
- Ellipsis protection
- Post-processing corrections (abbreviation merge, URL split)
- Long text handling (automatic chunking for text exceeding spaCy's max_length)
"""

import argparse
import os
import sys

# Add parent directory to path so we can import phrasplit
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from phrasplit import split_sentences

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
    parser.add_argument(
        "--no-split-on-colon",
        action="store_true",
        default=False,
        help="Disable splitting on colons",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        default=False,
        help="Use simple regex-based splitting (no spaCy required)",
    )

    args = parser.parse_args()

    # Get the appropriate spaCy model
    if args.model:
        model_name = args.model
    else:
        model_name = get_spacy_model(args.lang)

    with open(args.inputfile, encoding="utf-8") as input_f:
        with open(args.outputfile, "w", encoding="utf-8") as output_f:
            for line in input_f:
                line = line.strip()
                if not line:
                    continue

                # Use the actual phrasplit.split_sentences() function
                # Long text handling is built-in (automatic chunking)
                split_on_colon = not args.no_split_on_colon
                use_spacy = None if not args.simple else False
                sentences = split_sentences(
                    line,
                    language_model=model_name,
                    split_on_colon=split_on_colon,
                    use_spacy=use_spacy,
                )

                # Write each sentence on its own line
                output_f.writelines(sentence + "\n" for sentence in sentences)


if __name__ == "__main__":
    main()
