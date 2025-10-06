"""Tokenization package.

This package provides tokenization utilities for text processing.

Example:
    >>> from legollm.core.tokenization import SimpleTokenizer, load_vocab
    >>> vocab = load_vocab("models/vocab.json")
    >>> tokenizer = SimpleTokenizer(vocab)
    >>> ids = tokenizer.encode("Hello world!")
"""

from legollm.core.tokenization.naive_bpe_tokenizer import NaiveBPETokenizer
from legollm.core.tokenization.simple_tokenizer import PUNCTUATION, SimpleTokenizer
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
    "NaiveBPETokenizer",
    "SimpleTokenizer",
    "build_vocab_from_tokens",
    "load_vocab",
    "save_vocab",
]
