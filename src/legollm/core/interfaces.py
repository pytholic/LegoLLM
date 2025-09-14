"""Interface contracts for the project.

Created by @pytholic on 2025.09.14
"""

from typing import Protocol


class Tokenizer(Protocol):
    """Tokenizer interface."""

    def tokenize(
        self, text: str, sep_punc: bool = True, strip_empty: bool = True
    ) -> list[str]:
        """Tokenize the text.

        Args:
            text: The text to tokenize.
            sep_punc: Whether to separate punctuation.
            strip_empty: Whether to strip empty or whitespace tokens.

        Returns:
            A list of tokens.
        """
        ...
