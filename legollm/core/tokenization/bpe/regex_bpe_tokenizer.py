"""Regex-based Byte Pair Encoding (BPE) tokenizer implementation.

This implements the BPE algorithm as described in Sennrich et al. (2016).
BPE iteratively merges the most frequent pair of bytes/characters to build
a vocabulary of subword units.

Key Design Decisions:
    - Uses regex pre-tokenization pattern (GPT-4 style) to prevent merging
      across certain boundaries (e.g., don't merge across spaces in contractions)
    - Special tokens handled separately from merge vocabulary
    - Supports both training from scratch and loading pretrained vocab

References:
    - Sennrich et al. (2016): https://arxiv.org/abs/1508.07909
    - OpenAI tiktoken: https://github.com/openai/tiktoken

Example:
    >>> tokenizer = RegexBPETokenizer()
    >>> tokenizer.train(text="Hello world!", vocab_size=300)
    >>> tokenizer.encode("Hello<|endoftext|>world")
    [72, 101, 108, 108, 111, 100257, 119, 111, 114, 108, 100]
    >>> tokenizer.decode([72, 101, 108, 108, 111, 100257, 119, 111, 114, 108, 100])
    "Hello<|endoftext|>world"

Created by @pytholic on 2025.10.06
"""

from collections import Counter
from typing import Final

import regex as rex

from legollm.core.exceptions import TokenizerError
from legollm.core.tokenization.bpe.base_bpe import BaseBPETokenizer
from legollm.logging import logger, progress_bar


class RegexBPETokenizer(BaseBPETokenizer):
    """Regex-based BPE tokenizer matching GPT-2/GPT-4 behavior.

    Key differences from NaiveBPETokenizer:
    1. Splits text into chunks using regex before applying BPE
    2. Supports special tokens with configurable handling
    """

    INITIAL_VOCAB_SIZE = 256

    # Default special tokens (matching GPT-2)
    DEFAULT_SPECIAL_TOKENS: Final[dict[str, int]] = {
        "<|endoftext|>": 100257,
    }

    # GPT-4 uses additional special tokens
    GPT4_SPECIAL_TOKENS: Final[dict[str, int]] = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,  # fill-in-the-middle (code completion)
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    }

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
        if vocab_size < self.INITIAL_VOCAB_SIZE:
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
        with progress_bar("Training BPE tokenizer", total=num_merges) as (progress, train_task):
            for i in range(num_merges):
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

                progress.update(train_task, advance=1)
                if verbose:
                    logger.info(
                        f"merge {i + 1}/{num_merges}: {pair_id} -> {idx} ({vocab[idx]} had {occurrences} occurrences)"
                    )

        self.vocab = vocab
        self.merges = merges

        self.register_special_tokens(special_tokens.copy() if special_tokens else None)

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
        all_freqs = Counter()

        # Accumulate pair frequencies from all chunks
        for chunk_ids in token_ids:
            pair_freq = self._compute_pair_freq(chunk_ids)
            if pair_freq:
                all_freqs.update(dict(pair_freq))

        if not all_freqs:
            return None

        return all_freqs.most_common()

    def register_special_tokens(self, special_tokens: dict[str, int] | None = None) -> None:
        """Register special tokens (e.g., <|endoftext|>).

        Args:
            special_tokens: Dictionary of special tokens and their indices.

        Example:
            >>> tokenizer.register_special_tokens({"<|endoftext|>": 100257})
        """
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS.copy()
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def _encode_chunk(self, text_chunk: str) -> list[int]:
        """Encode a single text chunk with BPE merges.

        Args:
            text_chunk: A single text chunk (from regex splitting) to encode.

        Returns:
            List of token IDs for the chunk.
        """
        # Convert chunk to byte IDs
        token_ids = list(text_chunk.encode("utf-8"))

        # Apply BPE merges in learned order
        while len(token_ids) >= 2:
            pair_freq = self._compute_pair_freq(token_ids)

            if not pair_freq:
                break

            # Find the pair with the lowest merge index (earliest learned)
            # pair_freq is list of ((pair), frequency), so extract pair with p[0]
            pair_id = min(pair_freq, key=lambda p: self.merges.get(p[0], float("inf")))[0]

            # If this pair was not learned during training, stop
            if pair_id not in self.merges:
                break

            # Get the new token ID for this merge and apply it
            idx = self.merges[pair_id]
            token_ids = self._merge_pair(token_ids, pair_id, idx)

        return token_ids

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using learned merges and special tokens.

        This method:
        1. Splits text on special tokens (e.g., <|endoftext|>)
        2. For non-special text, applies regex chunking and BPE merges
        3. Returns combined token IDs including special token IDs

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.

        Raises:
            TokenizerError: If tokenizer is not trained.

        Example:
            >>> tokenizer = RegexBPETokenizer()
            >>> tokenizer.train(text="Hello world!", vocab_size=300)
            >>> tokenizer.encode("Hello<|endoftext|>world")
            [72, 101, 108, 108, 111, 100257, 119, 111, 114, 108, 100]
        """
        if not self._is_trained:
            raise TokenizerError("Tokenizer must be trained before encoding. Call train() first.")

        if not text:
            return []

        token_ids: list[int] = []

        # If no special tokens registered, encode normally
        if not self.special_tokens:
            # Split text into chunks and encode each
            text_chunks = self.compiled_pattern.findall(text)
            for chunk in text_chunks:
                token_ids.extend(self._encode_chunk(chunk))
            return token_ids

        # Build regex pattern to split on special tokens
        # Escape special tokens and join with |
        special_pattern = "|".join(rex.escape(token) for token in self.special_tokens)
        # Create pattern that captures special tokens: (token1|token2|...)
        split_pattern = f"({special_pattern})"

        # Split text on special tokens, keeping the special tokens
        splits = rex.split(split_pattern, text)

        # Process each part
        for split in splits:
            if not split:  # Skip empty strings
                continue

            if split in self.special_tokens:
                # This is a special token, add its ID directly
                token_ids.append(self.special_tokens[split])
            else:
                # This is regular text, apply regex chunking and BPE
                text_chunks = self.compiled_pattern.findall(split)
                for chunk in text_chunks:
                    token_ids.extend(self._encode_chunk(chunk))

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
            >>> tokenizer = RegexBPETokenizer()
            >>> tokenizer.train(text="Hello world!", vocab_size=300)
            >>> tokenizer.decode([72, 101, 108, 108, 111])
            "Hello"

            >>> tokenizer.decode([72, 101, 100257, 111])  # With special token
            "He<|endoftext|>o"
        """
        # Step 1: Validation
        if not self._is_trained:
            raise TokenizerError(
                "Tokenizer must be trained before decoding. Call train() or load() first."
            )

        if not token_ids:
            return ""

        # Step 2: Conversion - token IDs -> bytes -> text
        # Option 1: String concatenation - O(n²), creates new string each time
        # Option 2: List and join at the end - O(n), faster than string concatenation
        split_bytes: list[bytes] = []
        for token_id in token_ids:
            split_bytes.append(self._token_id_to_bytes(token_id))

        text = b"".join(split_bytes).decode("utf-8", errors="replace")
        return text

    def _token_id_to_bytes(self, token_id: int) -> bytes:
        """Convert a single token ID to its byte representation.
        It can handle both normal tokens and special tokens.

        Args:
            token_id: Token ID to convert.

        Returns:
            Byte representation of the token.

        Raises:
            TokenizerError: If token ID is unknown.
        """
        if token_id in self.vocab:
            return self.vocab[token_id]
        elif token_id in self.inverse_special_tokens:
            return self.inverse_special_tokens[token_id].encode("utf-8")
        else:
            raise TokenizerError(f"Unknown token ID: {token_id}")
