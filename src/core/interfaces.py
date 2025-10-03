"""Interface contracts for the project.

Created by @pytholic on 2025.09.14
"""

from typing import Protocol


class Tokenizer(Protocol):
    """Tokenizer interface.

    All concrete implementations must:
    1. Accept a vocabulary in __init__ and create str_to_int and int_to_str mappings
    2. Expose str_to_int and int_to_str as instance attributes
    """

    # Required instance attributes that implementations must provide
    str_to_int: dict[str, int]
    int_to_str: dict[int, str]

    def __init__(self, vocab: dict[str, int]) -> None:
        """Initialize the tokenizer with vocabulary.

        Implementations must create str_to_int and int_to_str mappings from vocab.

        Args:
            vocab: Dictionary mapping tokens (strings) to their integer IDs.
        """
        ...

    def tokenize(self, text: str) -> list[str]:
        """Tokenize the text."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode the text into a list of token IDs.

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.
        """
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into text.

        Args:
            tokens: A list of integer token IDs.

        Returns:
            The decoded text string.
        """
        ...
