"""Tokenization module.

Created by @pytholic on 2025.09.14
"""

import re

PUNCTUATION = frozenset(r".,!?;:-()[]{}\"")


class SimpleTokenizer:
    """Simple tokenizer that tokenizes text at whitespace."""

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        """Initialize the tokenizer with vocabulary mappings.

        Args:
            vocab: Dictionary mapping tokens (strings) to their integer IDs.
        """
        self.vocab = vocab
        self.str_to_int = vocab if vocab else {}
        self.int_to_str = {v: k for k, v in vocab.items()} if vocab else {}

    def tokenize(self, text: str) -> list[str]:
        """Pure tokenization logic - no vocab needed."""
        pattern = r"\w+(?:'\w+)*|[^\w\s]"
        tokens = re.findall(pattern, text)
        return [token for token in tokens if token.strip()]

    def encode(self, text: str) -> list[int]:
        """Tokenize + convert to IDs."""
        if not self.vocab:
            raise ValueError("Cannot encode without vocabulary. Train tokenizer first.")

        tokens = self.tokenize(text)
        unk_id = self.str_to_int.get("<UNK>", 0)
        return [self.str_to_int.get(token, unk_id) for token in tokens]

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into text."""
        if not tokens:
            return ""

        text_tokens = [self.int_to_str[token] for token in tokens]

        text = ""
        for token in text_tokens:
            if token in PUNCTUATION:
                text += token
            else:
                text += " " + token
        return text.strip()
