"""Naive BPE tokenizer for educational purposes.

This is a simple, straightforward implementation of Byte-Pair Encoding
to demonstrate the core concepts. For production use, see regex_bpe_tokenizer.py.

Created by @pytholic on 2025.10.06
"""

import logging

from tqdm import tqdm

from legollm.core.exceptions import TokenizerError
from legollm.core.logging import logger
from legollm.core.tokenization.bpe.base_bpe import BaseBPETokenizer

logger.setLevel(logging.DEBUG)


class NaiveBPETokenizer(BaseBPETokenizer):
    """Byte-Pair Encoding tokenizer - requires training.

    This is a naive implementation for educational purposes.
    It demonstrates the core BPE algorithm without optimizations.

    Inherits all common functionality from BaseBPETokenizer:
    - vocab, merges, _is_trained state
    - save(), load() methods
    - Helper methods: _compute_pair_freq(), _find_most_freq_pair(), _merge_pair()
    """

    def train(
        self,
        text: str,
        vocab_size: int,
        *,
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

        Example:
            >>> tokenizer = NaiveBPETokenizer()
            >>> tokenizer.train(text="Hello world!", vocab_size=300)
            >>> tokenizer.encode("Hello world!")
            [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]
        """
        num_merges = vocab_size - self.INITIAL_VOCAB_SIZE
        token_ids = list(text.encode("utf-8"))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(self.INITIAL_VOCAB_SIZE)}

        for i in tqdm(range(num_merges), desc="Training BPE tokenizer"):
            pair_freq = self._compute_pair_freq(token_ids)
            pair_id, occurrences = self._find_most_freq_pair(pair_freq)

            if pair_id is None:
                break

            idx = self.INITIAL_VOCAB_SIZE + i
            merges[pair_id] = idx
            token_ids = self._merge_pair(token_ids, pair_id, idx)
            vocab[idx] = vocab[pair_id[0]] + vocab[pair_id[1]]

            if verbose:
                logger.info(
                    f"merge {i + 1}/{num_merges}: {pair_id} -> {idx} ({vocab[idx]} had {occurrences} occurrences)"
                )

        self.merges = merges
        self.vocab = vocab
        self._is_trained = True

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using learned merges.

        Apply merges in the order they were learned (by merge index).

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.

        Raises:
            TokenizerError: If tokenizer is not trained.

        Example:
            >>> tokenizer = NaiveBPETokenizer()
            >>> tokenizer.train(text="Hello world!", vocab_size=300)
            >>> tokenizer.encode("Hello world!")
            [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]
        """
        if not self._is_trained:
            raise TokenizerError("Tokenizer must be trained before encoding. Call train() first.")

        if not text:
            return []

        token_ids = list(text.encode("utf-8"))

        while len(token_ids) >= 2:
            pair_freq = self._compute_pair_freq(token_ids)

            pair_id = min(pair_freq, key=lambda p: self.merges.get(p, float("inf")))  # pyright: ignore

            if pair_id not in self.merges:
                break

            idx = self.merges[pair_id]
            token_ids = self._merge_pair(token_ids, pair_id, idx)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: A list of integer token IDs.

        Returns:
            The decoded text string.

        Raises:
            TokenizerError: If tokenizer is not trained.

        Example:
            >>> tokenizer = NaiveBPETokenizer()
            >>> tokenizer.train(text="Hello world!", vocab_size=300)
            >>> ids = tokenizer.encode("Hello")
            >>> tokenizer.decode(ids)
            "Hello"
        """
        if not self._is_trained:
            raise TokenizerError(
                "Tokenizer must be trained before decoding. Call train() or load() first."
            )

        if not token_ids:
            return ""

        token_bytes = b"".join(self.vocab[idx] for idx in token_ids)
        text = token_bytes.decode("utf-8", errors="replace")
        return text
