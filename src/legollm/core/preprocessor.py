"""Text preprocessing pipeline.

Created by @pytholic on 2025.09.14
"""

from .interfaces import Tokenizer


class SimpleTextPreprocessor:
    """Simple text preprocessing pipeline for preparing data for training.

    Implementation details:
        - Convert raw text to tokens
        - Clean tokens (e.g. remove duplicates)
        - Create vocabulary
        - Attach IDs to tokens
    """

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def process_text(self, text: str, tokenizer: Tokenizer) -> dict[str, int]:
        """Tokenize and clean text.

        Args:
            text: The text to process.
            tokenizer: The tokenizer to use.

        Returns:
            A dictionary of tokens and their IDs.
        """
        tokens = tokenizer.tokenize(text)
        unique_tokens = self._clean_text(tokens)
        vocabulary = self._build_vocabulary(unique_tokens)
        self.vocabulary.update(vocabulary)
        return vocabulary

    def _clean_text(self, tokens: list[str]) -> list[str]:
        """Clean text.
        Takes a list of tokens, removes duplicates and returns a list of unique tokens.

        Args:
            tokens: The list of tokens to clean.

        Returns:
            A list of unique tokens.
        """
        return list(set(tokens))

    def _build_vocabulary(self, tokens: list[str]) -> dict[str, int]:
        """Build vocabulary.
        Takes a list of tokens and returns a vocabulary object.

        Args:
            tokens: The list of tokens to build a vocabulary from.

        Returns:
            A dictionary of tokens and their IDs.
        """
        return {text: i for i, text in enumerate(tokens)}
