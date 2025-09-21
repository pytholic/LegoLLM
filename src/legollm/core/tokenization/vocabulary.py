"""Vocabulary module.

Created by @pytholic on 2025.09.21
"""

import json


class VocabularyBuilder:
    """Builds vocabulary from tokens using different strategies."""

    def __init__(self) -> None:
        """Initializes the vocabulary builder."""
        self.vocabulary: dict[str, int] = {}

    def build_from_tokens(self, tokens: list[str]) -> dict[str, int]:
        """Builds vocabulary from tokens.

        Args:
            tokens: List of tokens to build vocabulary from.
        """
        return {text: i for i, text in enumerate(tokens)}


class VocabularyManager:
    """Handles saving/loading vocabularies."""

    def save(self, vocab: dict[str, int], path: str) -> None:
        """Saves vocabulary to a file."""
        with open(path, "w") as f:
            json.dump(vocab, f)

    def load(self, path: str) -> dict[str, int]:
        """Loads vocabulary from a file."""
        with open(path) as f:
            return json.load(f)
