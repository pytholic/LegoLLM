"""Test BaseBPETokenizer common functionality.

Tests the shared methods that all BPE tokenizers inherit.

Created by @pytholic on 2025.10.06
"""

from pathlib import Path

import pytest

from legollm.core.exceptions import TokenizerError
from legollm.core.tokenization.bpe.base_bpe import BaseBPETokenizer


class ConcreteBPETokenizer(BaseBPETokenizer):
    """Minimal concrete implementation for testing base class."""

    def train(
        self,
        text: str,
        vocab_size: int,
        *,
        min_frequency: int = 1,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        """Minimal train implementation."""
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {(72, 101): 256}  # 'H' + 'e' -> 256
        self.vocab[256] = b"He"  # Add the merged token to vocab
        self._is_trained = True

    def encode(self, text: str) -> list[int]:
        """Minimal encode implementation."""
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        """Minimal decode implementation."""
        return bytes(token_ids).decode("utf-8", errors="replace")


class TestBaseBPETokenizer:
    """Test BaseBPETokenizer shared functionality."""

    def test_initialization(self):
        """Test that tokenizer initializes with empty state."""
        tokenizer = ConcreteBPETokenizer()
        assert tokenizer.vocab == {}
        assert tokenizer.merges == {}
        assert not tokenizer.is_trained
        assert tokenizer.INITIAL_VOCAB_SIZE == 256

    def test_is_trained_property(self):
        """Test is_trained property."""
        tokenizer = ConcreteBPETokenizer()
        assert not tokenizer.is_trained

        tokenizer._is_trained = True
        assert tokenizer.is_trained

    def test_compute_pair_freq(self):
        """Test _compute_pair_freq method."""
        tokenizer = ConcreteBPETokenizer()
        token_ids = [1, 2, 2, 3, 2, 2]

        pair_freq = tokenizer._compute_pair_freq(token_ids)

        assert pair_freq is not None
        assert len(pair_freq) == 4  # (1,2), (2,2), (2,3), (3,2)
        assert pair_freq[0] == ((2, 2), 2)  # Most frequent
        assert ((1, 2), 1) in pair_freq
        assert ((2, 3), 1) in pair_freq
        assert ((3, 2), 1) in pair_freq

    def test_compute_pair_freq_empty(self):
        """Test _compute_pair_freq with single token."""
        tokenizer = ConcreteBPETokenizer()
        token_ids = [1]

        pair_freq = tokenizer._compute_pair_freq(token_ids)

        assert pair_freq is None

    def test_find_most_freq_pair(self):
        """Test _find_most_freq_pair method."""
        tokenizer = ConcreteBPETokenizer()
        pair_freq = [((2, 2), 5), ((1, 2), 3), ((3, 4), 1)]

        pair_id, occurrences = tokenizer._find_most_freq_pair(pair_freq)

        assert pair_id == (2, 2)
        assert occurrences == 5

    def test_find_most_freq_pair_empty(self):
        """Test _find_most_freq_pair with None."""
        tokenizer = ConcreteBPETokenizer()

        pair_id, occurrences = tokenizer._find_most_freq_pair(None)

        assert pair_id is None
        assert occurrences is None

    def test_merge_pair(self):
        """Test _merge_pair method."""
        tokenizer = ConcreteBPETokenizer()
        token_ids = [1, 2, 2, 3, 2, 2, 4]
        pair = (2, 2)
        new_id = 256

        result = tokenizer._merge_pair(token_ids, pair, new_id)

        assert result == [1, 256, 3, 256, 4]

    def test_merge_pair_no_match(self):
        """Test _merge_pair when pair doesn't exist."""
        tokenizer = ConcreteBPETokenizer()
        token_ids = [1, 2, 3, 4]
        pair = (5, 6)
        new_id = 256

        result = tokenizer._merge_pair(token_ids, pair, new_id)

        assert result == token_ids

    def test_save_and_load(self, tmp_path: Path):
        """Test save and load methods."""
        tokenizer = ConcreteBPETokenizer()
        tokenizer.train("test", 300)

        save_path = tmp_path / "test_tokenizer.json"
        tokenizer.save(str(save_path))

        new_tokenizer = ConcreteBPETokenizer()
        new_tokenizer.load(str(save_path))

        assert new_tokenizer.is_trained
        assert new_tokenizer.vocab == tokenizer.vocab
        assert new_tokenizer.merges == tokenizer.merges

    def test_save_untrained_raises_error(self, tmp_path: Path):
        """Test that saving untrained tokenizer raises error."""
        tokenizer = ConcreteBPETokenizer()
        save_path = tmp_path / "test_tokenizer.json"

        with pytest.raises(TokenizerError, match="must be trained before saving"):
            tokenizer.save(str(save_path))

    def test_render_token_ascii(self):
        """Test _render_token with ASCII characters."""
        tokenizer = ConcreteBPETokenizer()
        token_bytes = b"Hello"

        rendered = tokenizer._render_token(token_bytes)

        assert rendered == "Hello"

    def test_render_token_control_chars(self):
        """Test _render_token with control characters."""
        tokenizer = ConcreteBPETokenizer()
        token_bytes = b"Hello\n\t"

        rendered = tokenizer._render_token(token_bytes)

        assert "\\u000a" in rendered  # newline
        assert "\\u0009" in rendered  # tab

    def test_render_token_invalid_utf8(self):
        """Test _render_token with invalid UTF-8."""
        tokenizer = ConcreteBPETokenizer()
        token_bytes = b"\xff\xfe"

        rendered = tokenizer._render_token(token_bytes)

        assert "\\x" in rendered  # Should show hex representation

    def test_save_readable(self, tmp_path: Path):
        """Test save_readable creates correctly formatted output."""
        tokenizer = ConcreteBPETokenizer()
        # Create complete vocabulary (0-255 base + 1 merge)
        tokenizer.vocab = {idx: bytes([idx]) for idx in range(256)}
        tokenizer.vocab[256] = b"He"  # Add merged token
        tokenizer.merges = {(72, 101): 256}
        tokenizer._is_trained = True

        save_path = tmp_path / "test_readable.txt"
        tokenizer.save_readable(str(save_path))

        # Verify file exists and has expected structure
        assert save_path.exists()
        content = save_path.read_text(encoding="utf-8")

        # Check for expected sections
        assert "BPE Tokenizer - 257 tokens" in content
        assert "Vocabulary:" in content
        assert "Learned merges (1 tokens):" in content
        assert "[72] + [101] -> [He] (token 256)" in content
        assert "  0: \\u0000" in content  # First byte
        assert "255: \\xff" in content  # Last byte

    def test_save_readable_untrained_raises_error(self, tmp_path: Path):
        """Test that save_readable raises error when tokenizer is untrained."""
        tokenizer = ConcreteBPETokenizer()
        save_path = tmp_path / "test_readable.txt"

        with pytest.raises(TokenizerError, match="must be trained before saving"):
            tokenizer.save_readable(str(save_path))
