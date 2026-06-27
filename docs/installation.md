# Installation

## Requirements

- Python 3.9 or higher
- click 8.0 or higher
- rich 13.0 or higher
- spaCy 3.5 or higher (optional, for high-accuracy mode)

## Installation Options

phrasplit can be installed with or without spaCy support, depending on your needs.

### Basic Installation (Lightweight)

For a lightweight installation without spaCy (uses regex-based splitting):

```bash
pip install phrasplit
```

This installation is sufficient for basic text splitting and has no heavy dependencies.

### Full Installation with spaCy (Recommended)

For higher accuracy sentence splitting with spaCy NLP:

```bash
pip install phrasplit[nlp]
```

Or install spaCy separately:

```bash
pip install phrasplit
pip install spacy>=3.5.0
```

## Installing from Source

To install from source, clone the repository and install:

```bash
git clone https://github.com/buchwandler/phrasplit.git
cd phrasplit
pip install -e .
```

For development, install with dev dependencies:

```bash
pip install -e ".[dev]"
```

For spaCy support when installing from source:

```bash
pip install -e ".[nlp]"
```

## Installing spaCy Language Models

If you installed with spaCy support, you'll need a language model. The default model is
`en_core_web_sm` (English). Install it with:

```bash
python -m spacy download en_core_web_sm
```

For better accuracy, you can use larger models:

```bash
# Medium model (more accurate)
python -m spacy download en_core_web_md

# Large model (most accurate)
python -m spacy download en_core_web_lg
```

For other languages, see the [spaCy models documentation](https://spacy.io/models).

## Verifying Installation

You can verify your installation by running:

```python
import phrasplit
print(phrasplit.__version__)

from phrasplit import split_sentences

# Works with or without spaCy
print(split_sentences("Hello world. How are you?"))
# ['Hello world.', 'How are you?']

# Explicitly use simple mode (no spaCy required)
print(split_sentences("Hello world. How are you?", use_spacy=False))
# ['Hello world.', 'How are you?']
```

## Choosing Between Modes

**Simple Mode (use_spacy=False)**

- No spaCy installation required
- Faster processing
- Lower memory usage
- Good for straightforward text
- Uses regex-based splitting

**spaCy Mode (use_spacy=True, default if available)**

- Requires spaCy and language models
- Higher accuracy
- Better handling of complex cases
- Uses NLP-based analysis
- Recommended for production use
