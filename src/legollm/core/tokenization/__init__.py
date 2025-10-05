"""Tokenization package.

This package provides tokenization utilities for text processing.

Components:
    - SimpleTokenizer: Word-level tokenizer with regex-based splitting
    - Vocabulary utilities: Functions for building, saving, and loading vocabularies
    - Special tokens: Standard tokens used across tokenizers

Example:
    >>> from legollm.core.tokenization import SimpleTokenizer, load_vocab
    >>> vocab = load_vocab("models/vocab.json")
    >>> tokenizer = SimpleTokenizer(vocab)
    >>> ids = tokenizer.encode("Hello world!")
"""

from legollm.core.tokenization.tokenizer import PUNCTUATION, SimpleTokenizer
from legollm.core.tokenization.vocabulary import (
    END_OF_TEXT_TOKEN,
    UNK_TOKEN,
    build_vocab_from_tokens,
    load_vocab,
    save_vocab,
)

__all__ = [
    "END_OF_TEXT_TOKEN",
    "PUNCTUATION",
    "UNK_TOKEN",
    "SimpleTokenizer",
    "build_vocab_from_tokens",
    "load_vocab",
    "save_vocab",
]
