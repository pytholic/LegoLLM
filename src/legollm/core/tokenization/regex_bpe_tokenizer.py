"""Regex-based BPE tokenizer.

Created by @pytholic on 2025.10.06
"""

from collections import Counter

import regex as rex
from tqdm import tqdm

from legollm.core.exceptions import TokenizerError
from legollm.core.logging import logger
from legollm.core.tokenization.base_bpe import BaseBPETokenizer


class RegexBPETokenizer(BaseBPETokenizer):
    """Regex-based BPE tokenizer matching GPT-2/GPT-4 behavior.

    Key differences from NaiveBPETokenizer:
    1. Splits text into chunks using regex before applying BPE
    2. Supports special tokens with configurable handling
    3. Separate encode() and encode_ordinary() methods
    """

    INITIAL_VOCAB_SIZE = 256
    # the main GPT text split patterns, see:
    # ref: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    GPT2_SPLIT_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    GPT4_SPLIT_PATTERN = (
        r"'(?i:[sdmt]|ll|ve|re)|"  # Contractions
        r"[^\r\n\p{L}\p{N}]?+\p{L}+|"  # Letters with optional prefix
        r"\p{N}{1,3}|"  # Numbers (1-3 digits)
        r" ?[^\s\p{L}\p{N}]++[\r\n]*|"  # Non-alphanumeric with optional space
        r"\s*[\r\n]|"  # Newlines with optional whitespace
        r"\s+(?!\S)|"  # Whitespace not followed by non-whitespace
        r"\s+"  # Remaining whitespace
    )

    def __init__(self, pattern: str | None = None) -> None:
        """Initialize with optional regex pattern (defaults to GPT-4)."""
        self.pattern: str = pattern or self.GPT4_SPLIT_PATTERN
        self.compiled_pattern: rex.Pattern = rex.compile(self.pattern)
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self._is_trained: bool = False

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
            ```python
                text = read_file("data/blog.txt")
                tokenizer = RegexBPETokenizer()

                tokenizer.train(text, vocab_size=276, verbose=True)
                tokenizer.save("data/bpe_tokenizer.json")
                tokenizer.save_readable("data/bpe_tokenizer_readable.txt")
                tokenizer.load("data/bpe_tokenizer.json")
            ```
        """
        if vocab_size <= self.INITIAL_VOCAB_SIZE:
            raise TokenizerError(
                f"vocab_size must be at least {self.INITIAL_VOCAB_SIZE}, got {vocab_size}"
            )

        num_merges = vocab_size - self.INITIAL_VOCAB_SIZE

        # Step 1: Split text into chunks using regex
        text_chunks = self.compiled_pattern.findall(text)

        # Step 2: Convert each chunk to token IDs
        token_ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # Step 3: Initialize vocab with base bytes (0-255)
        vocab = {idx: bytes([idx]) for idx in range(self.INITIAL_VOCAB_SIZE)}
        merges = {}

        # Step 4: Iteratively merge the most common pairs
        for i in tqdm(range(num_merges), desc="Training BPE tokenizer", disable=not verbose):
            pair_freq = self._compute_pair_freq_chunks(token_ids)
            pair_id, occurrences = self._find_most_freq_pair(pair_freq)

            if pair_id is None:
                break

            idx = self.INITIAL_VOCAB_SIZE + i

            # merge this pair in ALL chunks
            token_ids = [self._merge_pair(chunk_ids, pair_id, idx) for chunk_ids in token_ids]

            # record merge and update vocab
            merges[pair_id] = idx
            vocab[idx] = vocab[pair_id[0]] + vocab[pair_id[1]]

            if verbose:
                logger.info(
                    f"merge {i + 1}/{num_merges}: {pair_id} -> {idx} ({vocab[idx]} had {occurrences} occurrences)"
                )

        self.vocab = vocab
        self.merges = merges

        # register special tokens if provided
        if special_tokens:
            self.register_special_tokens(special_tokens)

        self._is_trained = True

    def _compute_pair_freq_chunks(
        self, token_ids: list[list[int]]
    ) -> list[tuple[tuple[int, int], int]] | None:
        """Compute pair frequencies across all chunks.

        Args:
            token_ids: List of token ID lists (one per chunk)

        Returns:
            List of (pair, frequency) tuples sorted by frequency, or None if no pairs.
        """
        all_stats = Counter()

        # Accumulate pair frequencies from all chunks
        for chunk_ids in token_ids:
            pair_freq = self._compute_pair_freq(chunk_ids)
            if pair_freq:
                all_stats.update(dict(pair_freq))

        if not all_stats:
            return None

        return all_stats.most_common()

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """Register special tokens (e.g., <|endoftext|>).

        Args:
            special_tokens: Dictionary of special tokens and their indices.

        Example:
            >>> tokenizer.register_special_tokens({"<|endoftext|>": 100257})
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using learned merges."""
        pass

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        pass
