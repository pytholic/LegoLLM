"""Tokenization module.

Created by @pytholic on 2025.09.14
"""

import base64
import json
import logging

#   import unicodedata
from collections import Counter, deque
from itertools import pairwise

import regex as rex
from tqdm import tqdm

from legollm.core.exceptions import TokenizerError
from legollm.core.logging import logger
from legollm.core.tokenization.vocabulary import UNK_TOKEN, build_vocab_from_tokens

logger.setLevel(logging.DEBUG)

PUNCTUATION = frozenset(r".,!?;:-()[]{}\"")


class SimpleTokenizer:
    """Word-level tokenizer with regex-based splitting.

    This tokenizer splits text into words and punctuation using regex patterns.
    It requires a pre-built vocabulary for encoding/decoding operations.

    The tokenizer implements both the Tokenizer and PreTokenizable protocols,
    making it suitable for word-level tokenization tasks.
    """

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        """Initialize the tokenizer with vocabulary mappings.

        Args:
            vocab: Dictionary mapping tokens (strings) to their integer IDs.
                If None, the tokenizer can only be used for tokenization,
                not encoding/decoding.
        """
        self.vocab = vocab
        self.str_to_int = vocab if vocab else {}
        self.int_to_str = {v: k for k, v in vocab.items()} if vocab else {}

    def tokenize(self, text: str) -> list[str]:
        """Split text into tokens (pre-tokenization step).

        This is pure tokenization logic that doesn't require vocabulary.
        It's used to prepare text for vocabulary building or analysis.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of string tokens.

        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokenizer.tokenize("Hello, world!")
            ["Hello", ",", "world", "!"]
        """
        pattern = r"\w+(?:'\w+)*|[^\w\s]"
        tokens = rex.findall(pattern, text)  # pyright: ignore
        return [token for token in tokens if token.strip()]

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs.

        Tokenizes the text and maps each token to its vocabulary ID.
        Unknown tokens are mapped to the UNK token ID.

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.

        Raises:
            TokenizerError: If vocabulary is not provided or missing UNK token.

        Example:
            >>> vocab = {"Hello": 0, "world": 1, "<|UNK|>": 2}
            >>> tokenizer = SimpleTokenizer(vocab)
            >>> tokenizer.encode("Hello world")
            [0, 1]
        """
        if not self.vocab:
            raise TokenizerError(
                "Cannot encode without vocabulary. "
                "Initialize with vocab or use from_corpus() classmethod."
            )

        tokens = self.tokenize(text)
        unk_id = self.str_to_int.get(UNK_TOKEN)
        if unk_id is None:
            raise TokenizerError(
                f"Vocabulary must contain {UNK_TOKEN} for unknown token handling. "
                f"Use build_vocab_from_tokens() to create a proper vocabulary."
            )
        return [self.str_to_int.get(token, unk_id) for token in tokens]

    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back to text.

        Args:
            tokens: A list of integer token IDs.

        Returns:
            The decoded text string with proper spacing around punctuation.

        Example:
            >>> vocab = {"Hello": 0, "world": 1, "!": 2}
            >>> tokenizer = SimpleTokenizer(vocab)
            >>> tokenizer.decode([0, 1, 2])
            "Hello world!"
        """
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

    @classmethod
    def from_corpus(
        cls,
        corpus: str | list[str],
        min_frequency: int = 1,
        special_tokens: list[str] | None = None,
    ) -> "SimpleTokenizer":
        """Create tokenizer with vocabulary built from corpus.

        Convenience method that combines tokenization and vocabulary building.
        This is the recommended way to create a SimpleTokenizer from scratch.

        Args:
            corpus: Text or list of texts to build vocabulary from.
            min_frequency: Minimum token frequency to include in vocabulary.
                Tokens appearing fewer times are excluded. Default is 1.
            special_tokens: Additional special tokens to include.
                Defaults to [UNK_TOKEN, END_OF_TEXT_TOKEN].

        Returns:
            Initialized tokenizer with vocabulary built from corpus.

        Example:
            >>> corpus = ["Hello world!", "Hello there!"]
            >>> tokenizer = SimpleTokenizer.from_corpus(corpus)
            >>> tokenizer.encode("Hello world!")
            [2, 4, 0]

            >>> # With frequency filtering
            >>> tokenizer = SimpleTokenizer.from_corpus(corpus, min_frequency=2)
            >>> # Only "Hello" appears twice, other tokens filtered out
        """
        # Tokenize corpus
        temp_tokenizer = cls()
        if isinstance(corpus, str):
            corpus = [corpus]

        all_tokens: list[str] = []
        for text in corpus:
            all_tokens.extend(temp_tokenizer.tokenize(text))

        # Build vocabulary
        vocab = build_vocab_from_tokens(
            all_tokens, special_tokens=special_tokens, min_frequency=min_frequency
        )

        return cls(vocab)


class NaiveBPETokenizer:
    """Byte-Pair Encoding tokenizer - requires training."""

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
        # Let's write our training pipeline for bpe
        num_merges = vocab_size - self.INITIAL_VOCAB_SIZE
        token_ids = list(text.encode("utf-8"))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(self.INITIAL_VOCAB_SIZE)}

        # Iteratively merge the most frequent pairs
        for i in tqdm(range(num_merges), desc="Training BPE tokenizer"):
            pair_freq = self._compute_pair_freq(token_ids)
            pair_id, occurrences = self._find_most_freq_pair(pair_freq)

            if pair_id is None:
                break

            idx = self.INITIAL_VOCAB_SIZE + i
            merges[pair_id] = idx
            token_ids = self._merge_pair(token_ids, pair_id, idx)
            vocab[idx] = vocab[pair_id[0]] + vocab[pair_id[1]]

            # let's verbose and also use tqdm progressbar instead of print
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
            >>> tokenizer = BPETokenizer()
            >>> tokenizer.train(texts="Hello world!", vocab_size=100)
            >>> tokenizer.encode("Hello world!")
            [0, 1]

            >>> tokenizer.load("data/bpe_tokenizer.json")
            >>> tokenizer.encode("Hello world!")
            [0, 1]
        """
        if not self._is_trained:
            raise TokenizerError("Tokenizer must be trained before encoding. Call train() first.")

        if not text:
            return []

        token_ids = list(text.encode("utf-8"))

        while len(token_ids) >= 2:  # we need at least two tokens to merge
            pair_freq = self._compute_pair_freq(token_ids)

            # Find the pair with the lowest merge index (earliest learned)
            pair_id = min(pair_freq, key=lambda p: self.merges.get(p, float("inf")))  # pyright: ignore

            # If there are no merges available, this key will just be inf for each pair
            # and the min will be the first pair in the list.
            if pair_id not in self.merges:
                break

            # Replace the pair with the new token ID
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
            >>> tokenizer = BPETokenizer()
            >>> tokenizer.train(texts="Hello world!", vocab_size=100)
            >>> tokenizer.decode([0, 1])
            "Hello world!"

            >>> tokenizer.load("data/bpe_tokenizer.json")
            >>> tokenizer.decode([0, 1])
            "Hello world!"
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
        """Save vocab and merges to file in a human readable format.

        Args:
            path: File path to save the tokenizer state to.

        Raises:
            TokenizerError: If tokenizer is not trained.
        """
        if not self._is_trained:
            raise TokenizerError("Tokenizer must be trained before saving. Call train() first.")

        # vocab should be in integer keys: string characters format
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

            # Base tokens
            f.write("Base vocabulary (bytes 0-255):\n")
            f.write("-" * 50 + "\n")
            for idx in range(self.INITIAL_VOCAB_SIZE):
                if idx in self.vocab:
                    rendered = self._render_token(self.vocab[idx])
                    f.write(f"{idx:3d}: {rendered}\n")

            # Merges
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Learned merges ({len(self.merges)} tokens):\n")
            f.write("-" * 50 + "\n")
            for (p0, p1), idx in self.merges.items():
                merged = self._render_token(self.vocab[idx])
                f.write(f"[{p0}] + [{p1}] -> [{merged}] (token {idx})\n")

    def _render_token(self, token_bytes: bytes) -> str:
        """Render token, escaping control chars (minbpe-style)."""
        # Decode with replacement
        s = token_bytes.decode("utf-8", errors="replace")

        # If decode failed (contains �), show as hex
        if "�" in s:
            return " ".join(f"\\x{b:02x}" for b in token_bytes)

        s = rex.sub(r"\p{C}", lambda m: f"\\u{ord(m.group(0)):04x}", s)

        return s

    def load(self, path: str) -> None:
        """Load vocab and merges from file.

        Args:
            path: File path to load the tokenizer state from.

        Raises:
            TokenizerError: If tokenizer is not trained.
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
            current = dq.popleft()  # remove from front
            # Check if current + next item form the pair
            if dq and (current, dq[0]) == pair:
                replaced_token_ids.append(new_id)
                dq.popleft()  # remove second element of the pair
            else:
                replaced_token_ids.append(current)

        return replaced_token_ids
