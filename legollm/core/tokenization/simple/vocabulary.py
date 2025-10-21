"""Vocabulary utilities for tokenization.

This module provides utility functions for building, saving, and loading
vocabularies. These are pure functions with no state, making them easy to
test and compose.

Created by @pytholic on 2025.09.21
"""

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from legollm.core.exceptions import TokenizerError

# Special tokens used across different tokenizers
UNK_TOKEN = "<|UNK|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"
# PAD_TOKEN = "<|pad|>"
# BOS_TOKEN = "<|bos|>"  # Beginning of sequence
# EOS_TOKEN = "<|eos|>"  # End of sequence


def build_vocab_from_tokens(
    tokens: Iterable[str],
    special_tokens: list[str] | None = None,
    min_frequency: int = 1,
) -> dict[str, int]:
    """Build vocabulary from a collection of tokens.

    Utility function for word-level tokenizers that don't learn vocabulary.
    Deduplicates tokens, filters by frequency, and adds special tokens.

    Args:
        tokens: Iterable of token strings.
        special_tokens: Additional special tokens to include.
            Defaults to [UNK_TOKEN, END_OF_TEXT_TOKEN].
        min_frequency: Minimum frequency for a token to be included in vocabulary.
            Tokens appearing fewer times are excluded. Default is 1 (include all).

    Returns:
        Dictionary mapping tokens to integer IDs. Tokens are sorted alphabetically,
        with special tokens added at the end.

    Raises:
        TokenizerError: If the tokens iterable is empty.

    Example:
        >>> tokens = ["hello", "world", "hello", "!"]
        >>> vocab = build_vocab_from_tokens(tokens)
        >>> vocab
        {'!': 0, 'hello': 1, 'world': 2, '<|UNK|>': 3, '<|endoftext|>': 4}

        >>> # With minimum frequency filter
        >>> vocab = build_vocab_from_tokens(tokens, min_frequency=2)
        >>> vocab
        {'hello': 0, '<|UNK|>': 1, '<|endoftext|>': 2}
    """
    if special_tokens is None:
        special_tokens = [UNK_TOKEN, END_OF_TEXT_TOKEN]

    token_list = list(tokens)
    if not token_list:
        raise TokenizerError("Cannot build vocabulary from empty tokens list")

    # Fast path: when min_frequency=1, just deduplicate
    if min_frequency == 1:
        unique_tokens = sorted(set(token_list))
    else:
        # Slow path: count frequencies and filter
        token_counts = Counter(token_list)
        unique_tokens = sorted(
            [token for token, count in token_counts.items() if count >= min_frequency]
        )

    unique_tokens.extend(special_tokens)
    return {token: idx for idx, token in enumerate(unique_tokens)}


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    """Save vocabulary to JSON file.

    Args:
        vocab: Vocabulary dictionary mapping tokens to IDs.
        path: Output file path. Parent directory must exist.

    Raises:
        FileNotFoundError: If the parent directory does not exist.

    Example:
        >>> vocab = {"hello": 0, "world": 1}
        >>> save_vocab(vocab, "models/vocab.json")
    """
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory {path.parent} does not exist")

    with path.open("w") as f:
        json.dump(vocab, f, indent=2)


def load_vocab(path: str | Path) -> dict[str, int]:
    """Load vocabulary from JSON file.

    Args:
        path: Input file path.

    Returns:
        Vocabulary dictionary mapping tokens to IDs.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        >>> vocab = load_vocab("models/vocab.json")
        >>> vocab
        {"hello": 0, "world": 1}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")

    with path.open() as f:
        return json.load(f)
