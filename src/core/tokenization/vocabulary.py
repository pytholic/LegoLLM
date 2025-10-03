"""Vocabulary module.

Created by @pytholic on 2025.09.21
"""

import json
from pathlib import Path

UNK_TOKEN = "<|UNK|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"


class VocabularyBuilder:
    """Builds vocabulary from tokens using different strategies."""

    def build_from_tokens(self, tokens: list[str]) -> dict[str, int]:
        """Builds vocabulary from tokens.

        Args:
            tokens: List of tokens to build vocabulary from.

        Returns:
            Dictionary mapping tokens (strings) to their integer IDs.

        Raises:
            ValueError: If the tokens list is empty.
        """
        if not tokens:
            raise ValueError("Cannot build vocabulary from empty tokens list")

        tokens = self._remove_duplicates(tokens)
        tokens.extend([UNK_TOKEN, END_OF_TEXT_TOKEN])
        return {text: i for i, text in enumerate(tokens)}

    def _remove_duplicates(self, tokens: list[str]) -> list[str]:
        """Remove duplicates from a list of tokens.

        Args:
            tokens: List of tokens to remove duplicates from.

        Returns:
            List of tokens with duplicates removed.
        """
        return sorted(set(tokens))


class VocabularyManager:
    """Handles saving/loading vocabularies."""

    def save(self, vocab: dict[str, int], path: str | Path) -> None:
        """Saves vocabulary to a file.

        Args:
            vocab: Dictionary mapping tokens (strings) to their integer IDs.
            path: Path to the file to save the vocabulary to.

        Raises:
            FileNotFoundError: If the parent directory does not exist.
        """
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"Parent directory {path.parent} does not exist")
        with open(path, "w") as f:
            json.dump(vocab, f)

    def load(self, path: str | Path) -> dict[str, int]:
        """Loads vocabulary from a file.

        Args:
            path: Path to the file to load the vocabulary from.

        Returns:
            Dictionary mapping tokens (strings) to their integer IDs.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        with open(path) as f:
            return json.load(f)
