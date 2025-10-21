"""Interface contracts for the project.

Created by @pytholic on 2025.09.14
"""

from typing import Protocol


class Tokenizer(Protocol):
    """Core tokenizer interface - what all tokenizers MUST support.

    This is the minimal contract that every tokenizer must implement,
    regardless of whether it's word-level, subword-level, or character-level.
    """

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs.

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.

        Raises:
            TokenizerError: If tokenizer is not properly initialized.
        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text.

        Args:
            token_ids: A list of integer token IDs.

        Returns:
            The decoded text string.
        """
        ...


class TrainableTokenizer(Tokenizer, Protocol):
    """Tokenizers that require training (BPE, WordPiece, SentencePiece, etc.).

    These tokenizers learn their vocabulary and/or merge rules from a corpus
    during a training phase. They extend the base Tokenizer interface with
    training and persistence capabilities.
    """

    def train(
        self,
        text: str,
        vocab_size: int,
        *,  # Force keyword-only
        special_tokens: dict[str, int] | None = None,
        verbose: bool = False,
    ) -> None:
        """Train the tokenizer on corpus to learn merges/vocabulary.

        Args:
            text: Training corpus.
            vocab_size: Target vocabulary size after training.
            special_tokens: Additional special tokens to include.
            verbose: Whether to print verbose output.

        Raises:
            TokenizerError: If training fails or parameters are invalid.
        """
        ...

    def save(self, path: str) -> None:
        """Save trained tokenizer state (vocabulary, merges, config, etc.).

        Args:
            path: File path to save the tokenizer state.

        Raises:
            TokenizerError: If tokenizer is not trained or save fails.
        """
        ...

    def save_readable(self, path: str) -> None:
        """Save trained tokenizer state (vocabulary, merges, config, etc.) in a human readable format.

        Args:
            path: File path to save the tokenizer state in a human readable format.

        Raises:
            TokenizerError: If tokenizer is not trained or save fails.
        """
        ...

    def load(self, path: str) -> None:
        """Load trained tokenizer state from file.

        Args:
            path: File path to load the tokenizer state from.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            TokenizerError: If the file is corrupted or incompatible.
        """

    ...


class PreTokenizable(Protocol):
    """Tokenizers that support pre-tokenization (word-level, whitespace, etc.).

    Pre-tokenization is the step of splitting text into discrete tokens before
    any vocabulary mapping or subword splitting. This is useful for:
    - Building vocabularies from raw text
    - Word-level tokenization
    - Custom tokenization schemes
    """

    def tokenize(self, text: str) -> list[str]:
        """Split text into discrete tokens (pre-tokenization step).

        This method performs pure tokenization without any vocabulary lookup.
        It's typically used to prepare text for vocabulary building.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of string tokens.

        Example:
            >>> tokenizer.tokenize("Hello, world!")
            ["Hello", ",", "world", "!"]
        """
        ...
