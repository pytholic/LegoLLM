"""Naive BPE tokenizer for educational purposes.

This is a simple, straightforward implementation of Byte-Pair Encoding
to demonstrate the core concepts. For production use, see bpe.py.

Created by @pytholic on 2025.10.06
"""

import base64
import json
import logging
from collections import Counter, deque
from itertools import pairwise

import regex as rex
from tqdm import tqdm

from legollm.core.exceptions import TokenizerError
from legollm.core.logging import logger

logger.setLevel(logging.DEBUG)


class NaiveBPETokenizer:
    """Byte-Pair Encoding tokenizer - requires training.

    This is a naive implementation for educational purposes.
    It demonstrates the core BPE algorithm without optimizations.
    """

    INITIAL_VOCAB_SIZE = 256

    def __init__(self) -> None:
        """Initialize the tokenizer."""
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self._is_trained = False

    def train(
        self,
        text: str,
        vocab_size: int,
        *,
        min_frequency: int = 1,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        """Train the tokenizer on corpus to learn merges/vocabulary."""
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

    def save(self, path: str) -> None:
        """Save vocab and merges to file.

        Args:
            path: File path to save the tokenizer state to.

        Raises:
            TokenizerError: If tokenizer is not trained.
        """
        if not self._is_trained:
            raise TokenizerError("Tokenizer must be trained before saving. Call train() first.")

        save_data = {
            "vocab": {
                str(idx): base64.b64encode(token_bytes).decode("ascii")
                for idx, token_bytes in self.vocab.items()
            },
            "merges": {f"{p0},{p1}": v for (p0, p1), v in self.merges.items()},
        }
        with open(path, "w") as f:
            json.dump(save_data, f)

    def save_readable(self, path: str) -> None:
        """Save vocab and merges to file in a human readable format."""
        if not self._is_trained:
            raise TokenizerError("Tokenizer must be trained before saving. Call train() first.")

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"BPE Tokenizer - {len(self.vocab)} tokens\n")
            f.write("=" * 50 + "\n\n")

            f.write("Base vocabulary (bytes 0-255):\n")
            f.write("-" * 50 + "\n")
            for idx in range(self.INITIAL_VOCAB_SIZE):
                if idx in self.vocab:
                    rendered = self._render_token(self.vocab[idx])
                    f.write(f"{idx:3d}: {rendered}\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Learned merges ({len(self.merges)} tokens):\n")
            f.write("-" * 50 + "\n")
            for (p0, p1), idx in self.merges.items():
                merged = self._render_token(self.vocab[idx])
                f.write(f"[{p0}] + [{p1}] -> [{merged}] (token {idx})\n")

    def _render_token(self, token_bytes: bytes) -> str:
        """Render token, escaping control chars."""
        s = token_bytes.decode("utf-8", errors="replace")

        if "�" in s:
            return " ".join(f"\\x{b:02x}" for b in token_bytes)

        s = rex.sub(r"\p{C}", lambda m: f"\\u{ord(m.group(0)):04x}", s)

        return s

    def load(self, path: str) -> None:
        """Load vocab and merges from file.

        Args:
            path: File path to load the tokenizer state from.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        with open(path) as f:
            data = json.load(f)

        self.vocab = {
            int(idx): base64.b64decode(token_b64) for idx, token_b64 in data["vocab"].items()
        }
        self.merges = {tuple(map(int, k.split(","))): int(idx) for k, idx in data["merges"].items()}
        self._is_trained = True

    def _compute_pair_freq(self, token_ids: list[int]) -> list[tuple[tuple[int, int], int]] | None:
        """Compute frequency of each consecutive pair in the text."""
        pair_counts = Counter(pairwise(token_ids))
        if not pair_counts:
            return None

        return pair_counts.most_common()

    def _find_most_freq_pair(
        self, pair_freq: list[tuple[tuple[int, int], int]] | None
    ) -> tuple[tuple[int, int] | None, int | None]:
        """Find the pair ID from the list of pair frequencies."""
        if not pair_freq:
            return None, None
        pair_id, occurrences = pair_freq[0]
        return pair_id, occurrences

    def _merge_pair(self, token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Merge the given pair ID with a new token ID."""
        dq = deque(token_ids)
        replaced_token_ids: list[int] = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair:
                replaced_token_ids.append(new_id)
                dq.popleft()
            else:
                replaced_token_ids.append(current)

        return replaced_token_ids
