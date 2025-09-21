"""Vocabulary module.

Created by @pytholic on 2025.09.21
"""

import json
from pathlib import Path


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
        tokens = self._remove_duplicates(tokens)
        if not tokens:
            raise ValueError("Cannot build vocabulary from empty tokens list")
        return {text: i for i, text in enumerate(tokens)}

    def _remove_duplicates(self, tokens: list[str]) -> list[str]:
        """Remove duplicates from a list of tokens."""
        return sorted(set(tokens))


class VocabularyManager:
    """Handles saving/loading vocabularies."""

    def save(self, vocab: dict[str, int], path: Path) -> None:
        """Saves vocabulary to a file."""
        if not path.parent.exists():
            raise FileNotFoundError(f"Parent directory {path.parent} does not exist")
        with open(path, "w") as f:
            json.dump(vocab, f)

    def load(self, path: Path) -> dict[str, int]:
        """Loads vocabulary from a file."""
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        with open(path) as f:
            return json.load(f)
