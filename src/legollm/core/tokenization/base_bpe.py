"""Base class for BPE tokenizers.

This base class provides common functionality for all BPE tokenizer implementations:
- Shared state management (vocab, merges, training status)
- Common helper methods (pair counting, merging, rendering)
- Serialization (save/load)

Subclasses only need to implement: train(), encode(), decode()

Created by @pytholic on 2025.10.06
"""

import base64
import json
from abc import ABC, abstractmethod
from collections import Counter, deque
from itertools import pairwise

import regex as rex

from legollm.core.exceptions import TokenizerError


class BaseBPETokenizer(ABC):
    """Base class for BPE tokenizers with shared functionality."""

    INITIAL_VOCAB_SIZE = 256

    def __init__(self) -> None:
        """Initialize the tokenizer with empty state."""
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if the tokenizer is trained."""
        return self._is_trained

    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using learned merges."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def _compute_pair_freq(token_ids: list[int]) -> list[tuple[tuple[int, int], int]] | None:
        """Compute frequency of each consecutive pair in the text.

        Args:
            token_ids: List of token IDs.

        Returns:
            List of (pair, frequency) tuples sorted by frequency, or None if no pairs.
        """
        pair_counts = Counter(pairwise(token_ids))
        if not pair_counts:
            return None
        return pair_counts.most_common()

    @staticmethod
    def _find_most_freq_pair(
        pair_freq: list[tuple[tuple[int, int], int]] | None,
    ) -> tuple[tuple[int, int] | None, int | None]:
        """Find the most frequent pair from pair frequencies.

        Args:
            pair_freq: List of (pair, frequency) tuples.

        Returns:
            Tuple of (most_frequent_pair, occurrences) or (None, None) if empty.
        """
        if not pair_freq:
            return None, None
        pair_id, occurrences = pair_freq[0]
        return pair_id, occurrences

    @staticmethod
    def _merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Merge all occurrences of a pair with a new token ID.

        Args:
            token_ids: List of token IDs.
            pair: Pair of token IDs to merge.
            new_id: New token ID to replace the merged pair.

        Returns:
            List of token IDs with the pair replaced by new_id.
        """
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

    def save(self, path: str) -> None:
        """Save vocab and merges to file.

        Args:
            path: Path to save the tokenizer state.

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

    def load(self, path: str) -> None:
        """Load vocab and merges from file.

        Args:
            path: Path to load the tokenizer state from.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        with open(path) as f:
            save_data = json.load(f)

        self.vocab = {
            int(idx): base64.b64decode(token_b64) for idx, token_b64 in save_data["vocab"].items()
        }
        self.merges = {tuple(map(int, p.split(","))): v for p, v in save_data["merges"].items()}  # pyright: ignore
        self._is_trained = True

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

    @staticmethod
    def _render_token(token_bytes: bytes) -> str:
        """Render token with escape sequences for control characters.

        Args:
            token_bytes: Token as bytes.

        Returns:
            Human-readable string representation.
        """
        s = token_bytes.decode("utf-8", errors="replace")
        if "�" in s:
            return " ".join(f"\\x{b:02x}" for b in token_bytes)
        s = rex.sub(r"\p{C}", lambda m: f"\\u{ord(m.group(0)):04x}", s)
        return s
