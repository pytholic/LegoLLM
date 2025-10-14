"""Unit tests for RegexBPETokenizer.

Tests the regex-based BPE tokenizer with special token handling.

Created by @pytholic on 2025.10.13
"""

from pathlib import Path

import pytest

from legollm.core.exceptions import TokenizerError
from legollm.core.tokenization.regex_bpe_tokenizer import RegexBPETokenizer


class TestRegexBPETokenizerInitialization:
    """Test RegexBPETokenizer initialization and configuration."""

    def test_default_initialization(self):
        """Test tokenizer initializes with GPT-4 pattern by default."""
        tokenizer = RegexBPETokenizer()
        assert tokenizer.pattern == RegexBPETokenizer.GPT4_SPLIT_PATTERN
        assert tokenizer.vocab == {}
        assert tokenizer.merges == {}
        assert tokenizer.special_tokens == {}
        assert not tokenizer.is_trained

    def test_custom_pattern_initialization(self):
        """Test tokenizer initializes with custom pattern."""
        custom_pattern = r"\w+|\s+"
        tokenizer = RegexBPETokenizer(pattern=custom_pattern)
        assert tokenizer.pattern == custom_pattern

    @pytest.mark.parametrize(
        "pattern",
        [
            pytest.param(RegexBPETokenizer.GPT2_SPLIT_PATTERN, id="gpt2_pattern"),
            pytest.param(RegexBPETokenizer.GPT4_SPLIT_PATTERN, id="gpt4_pattern"),
            pytest.param(r"\w+", id="simple_word_pattern"),
        ],
    )
    def test_pattern_compilation(self, pattern: str):
        """Test that various patterns compile successfully."""
        tokenizer = RegexBPETokenizer(pattern=pattern)
        assert tokenizer.compiled_pattern is not None
        # Test that pattern can be used
        result = tokenizer.compiled_pattern.findall("Hello world!")
        assert len(result) > 0


class TestRegexBPETokenizerTraining:
    """Test RegexBPETokenizer training functionality."""

    def test_train_basic(self, simple_training_text: str):
        """Test basic training workflow."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(simple_training_text, vocab_size=280)

        assert tokenizer.is_trained
        assert len(tokenizer.vocab) <= 280
        assert len(tokenizer.merges) == len(tokenizer.vocab) - 256

    def test_train_with_verbose(self, simple_training_text: str):
        """Test training with verbose output."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(simple_training_text, vocab_size=270, verbose=True)

        # Verbose mode should complete successfully
        assert tokenizer.is_trained

    def test_train_invalid_vocab_size(self, simple_training_text: str):
        """Test training with invalid vocabulary size raises error."""
        tokenizer = RegexBPETokenizer()

        with pytest.raises(TokenizerError, match="vocab_size must be at least 256"):
            tokenizer.train(simple_training_text, vocab_size=200)

    def test_train_creates_valid_vocab(self, simple_training_text: str):
        """Test that training creates valid vocabulary with byte values."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(simple_training_text, vocab_size=270)

        # Check base vocabulary (0-255)
        for i in range(256):
            assert i in tokenizer.vocab
            assert tokenizer.vocab[i] == bytes([i])

        # Check merged tokens are byte sequences
        for idx in range(256, len(tokenizer.vocab)):
            assert isinstance(tokenizer.vocab[idx], bytes)
            assert len(tokenizer.vocab[idx]) >= 2  # Merged tokens should be at least 2 bytes

    def test_train_creates_valid_merges(self, simple_training_text: str):
        """Test that training creates valid merge rules."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(simple_training_text, vocab_size=270)

        # All merge keys should be tuples of ints
        assert all(
            isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(x, int) for x in pair)
            for pair in tokenizer.merges.keys()
        )

        # All merge values should be ints >= 256
        assert all(isinstance(idx, int) and idx >= 256 for idx in tokenizer.merges.values())

    def test_train_with_special_tokens(self, simple_training_text: str):
        """Test training with custom special tokens."""
        special_tokens = {"<|custom|>": 100300, "<|test|>": 100301}
        tokenizer = RegexBPETokenizer()
        tokenizer.train(simple_training_text, vocab_size=270, special_tokens=special_tokens)

        assert tokenizer.special_tokens == special_tokens
        assert tokenizer.inverse_special_tokens == {100300: "<|custom|>", 100301: "<|test|>"}

    def test_train_multilingual(self, multilingual_text: str):
        """Test training on multilingual text."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train(multilingual_text, vocab_size=300)

        assert tokenizer.is_trained
        # Should be able to encode all languages
        encoded = tokenizer.encode(multilingual_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == multilingual_text

    @pytest.mark.parametrize(
        "vocab_size,expected_merges",
        [
            pytest.param(260, 4, id="small_vocab"),
            pytest.param(280, 24, id="medium_vocab"),
            pytest.param(300, 28, id="large_vocab_limited_by_text"),
        ],
    )
    def test_train_different_vocab_sizes(
        self, simple_training_text: str, vocab_size: int, expected_merges: int
    ):
        """Test training with different vocabulary sizes.

        Note: The actual vocab size may be less than requested if the text
        doesn't have enough unique pairs to merge.
        """
        tokenizer = RegexBPETokenizer()
        tokenizer.train(simple_training_text, vocab_size=vocab_size)

        # Check that expected number of merges were created
        assert len(tokenizer.merges) == expected_merges
        # Vocab size = 256 (base bytes) + number of merges
        assert len(tokenizer.vocab) == 256 + expected_merges
        # Should not exceed requested vocab size
        assert len(tokenizer.vocab) <= vocab_size


class TestRegexBPETokenizerEncoding:
    """Test RegexBPETokenizer encoding functionality."""

    @pytest.fixture
    def trained_tokenizer(self) -> RegexBPETokenizer:
        """Fixture providing a trained tokenizer."""
        tokenizer = RegexBPETokenizer()
        training_text = "Hello world! This is a test. Hello again."
        tokenizer.train(training_text, vocab_size=280)
        return tokenizer

    def test_encode_untrained_raises_error(self):
        """Test encoding with untrained tokenizer raises error."""
        tokenizer = RegexBPETokenizer()

        with pytest.raises(TokenizerError, match="must be trained before encoding"):
            tokenizer.encode("test")

    def test_encode_empty_string(self, trained_tokenizer: RegexBPETokenizer):
        """Test encoding empty string returns empty list."""
        result = trained_tokenizer.encode("")
        assert result == []

    def test_encode_returns_integers(self, trained_tokenizer: RegexBPETokenizer):
        """Test that encoding returns list of integers."""
        result = trained_tokenizer.encode("Hello")
        assert isinstance(result, list)
        assert all(isinstance(token_id, int) for token_id in result)

    def test_encode_basic_text(self, trained_tokenizer: RegexBPETokenizer):
        """Test encoding basic text."""
        text = "Hello world"
        encoded = trained_tokenizer.encode(text)
        assert len(encoded) > 0
        assert len(encoded) <= len(text.encode("utf-8"))  # Should compress

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("Hello", id="single_word"),
            pytest.param("Hello world!", id="with_punctuation"),
            pytest.param("Test123", id="with_numbers"),
            pytest.param("café", id="with_accents"),
            pytest.param("안녕하세요", id="korean"),
            pytest.param("🚀 emoji", id="with_emoji"),
        ],
    )
    def test_encode_various_inputs(self, trained_tokenizer: RegexBPETokenizer, text: str):
        """Test encoding various input types."""
        encoded = trained_tokenizer.encode(text)
        assert len(encoded) > 0
        # All token IDs should be valid (in vocab or special tokens)
        assert all(
            token_id in trained_tokenizer.vocab
            or token_id in trained_tokenizer.inverse_special_tokens
            for token_id in encoded
        )

    def test_encode_with_special_tokens(self):
        """Test encoding text containing special tokens."""
        tokenizer = RegexBPETokenizer()
        text = "Hello world"
        special_tokens = {"<|endoftext|>": 100257}
        tokenizer.train(text, vocab_size=280, special_tokens=special_tokens)

        # Encode text with special token
        encoded = tokenizer.encode("Hello<|endoftext|>world")
        assert 100257 in encoded  # Special token ID should be present

    def test_encode_multiple_special_tokens(self):
        """Test encoding with multiple special tokens."""
        tokenizer = RegexBPETokenizer()
        text = "Hello world test"
        special_tokens = {"<|start|>": 100257, "<|end|>": 100258}
        tokenizer.train(text, vocab_size=280, special_tokens=special_tokens)

        encoded = tokenizer.encode("<|start|>Hello<|end|>")
        assert 100257 in encoded
        assert 100258 in encoded

    def test_encode_special_token_at_boundaries(self):
        """Test special tokens at text boundaries."""
        tokenizer = RegexBPETokenizer()
        text = "Hello world"
        special_tokens = {"<|endoftext|>": 100257}
        tokenizer.train(text, vocab_size=280, special_tokens=special_tokens)

        # Test at start, middle, and end
        for test_text in [
            "<|endoftext|>Hello",
            "Hello<|endoftext|>",
            "<|endoftext|>",
        ]:
            encoded = tokenizer.encode(test_text)
            assert 100257 in encoded


class TestRegexBPETokenizerDecoding:
    """Test RegexBPETokenizer decoding functionality."""

    @pytest.fixture
    def trained_tokenizer(self) -> RegexBPETokenizer:
        """Fixture providing a trained tokenizer."""
        tokenizer = RegexBPETokenizer()
        training_text = "Hello world! This is a test. Hello again."
        tokenizer.train(training_text, vocab_size=280)
        return tokenizer

    def test_decode_untrained_raises_error(self):
        """Test decoding with untrained tokenizer raises error."""
        tokenizer = RegexBPETokenizer()

        with pytest.raises(TokenizerError, match="must be trained before decoding"):
            tokenizer.decode([72, 101, 108, 108, 111])

    def test_decode_empty_list(self, trained_tokenizer: RegexBPETokenizer):
        """Test decoding empty list returns empty string."""
        result = trained_tokenizer.decode([])
        assert result == ""

    def test_decode_returns_string(self, trained_tokenizer: RegexBPETokenizer):
        """Test that decoding returns string."""
        result = trained_tokenizer.decode([72, 101, 108, 108, 111])  # "Hello"
        assert isinstance(result, str)

    def test_decode_basic_tokens(self, trained_tokenizer: RegexBPETokenizer):
        """Test decoding basic byte tokens."""
        # ASCII bytes for "Hi"
        token_ids = [72, 105]
        decoded = trained_tokenizer.decode(token_ids)
        assert decoded == "Hi"

    def test_decode_with_special_tokens(self):
        """Test decoding with special tokens."""
        tokenizer = RegexBPETokenizer()
        text = "Hello world"
        special_tokens = {"<|endoftext|>": 100257}
        tokenizer.train(text, vocab_size=280, special_tokens=special_tokens)

        # Decode with special token
        token_ids = [72, 101, 100257, 111]  # "He<|endoftext|>o"
        decoded = tokenizer.decode(token_ids)
        assert "<|endoftext|>" in decoded

    def test_decode_unknown_token_raises_error(self, trained_tokenizer: RegexBPETokenizer):
        """Test decoding unknown token ID raises error."""
        with pytest.raises(TokenizerError, match="Unknown token ID"):
            trained_tokenizer.decode([999999])

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("Hello world!", id="basic_text"),
            pytest.param("Test123", id="with_numbers"),
            pytest.param("café résumé", id="with_accents"),
            pytest.param("안녕하세요", id="korean"),
            pytest.param("Hello\nworld", id="with_newline"),
            pytest.param("Tab\there", id="with_tab"),
        ],
    )
    def test_encode_decode_roundtrip(self, trained_tokenizer: RegexBPETokenizer, text: str):
        """Test that encode->decode roundtrip preserves text."""
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text

    def test_decode_space_after_punctuation(self, trained_tokenizer: RegexBPETokenizer):
        """Test decoding text has proper space after punctuation."""
        text = "Hello, world!"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text
        assert decoded == "Hello, world!"


class TestRegexBPETokenizerSpecialTokens:
    """Test special token handling."""

    def test_register_special_tokens_default(self):
        """Test registering default special tokens."""
        tokenizer = RegexBPETokenizer()
        tokenizer.register_special_tokens()

        assert tokenizer.special_tokens == RegexBPETokenizer.DEFAULT_SPECIAL_TOKENS
        assert tokenizer.inverse_special_tokens == {
            100257: "<|endoftext|>",
        }

    def test_register_special_tokens_custom(self):
        """Test registering custom special tokens."""
        custom_tokens = {"<|custom|>": 100300}
        tokenizer = RegexBPETokenizer()
        tokenizer.register_special_tokens(custom_tokens)

        assert tokenizer.special_tokens == custom_tokens
        assert tokenizer.inverse_special_tokens == {100300: "<|custom|>"}

    def test_register_special_tokens_gpt4(self):
        """Test registering GPT-4 special tokens."""
        tokenizer = RegexBPETokenizer()
        tokenizer.register_special_tokens(RegexBPETokenizer.GPT4_SPECIAL_TOKENS.copy())

        assert len(tokenizer.special_tokens) == 5
        assert "<|fim_prefix|>" in tokenizer.special_tokens
        assert "<|fim_middle|>" in tokenizer.special_tokens

    def test_encode_without_special_tokens(self):
        """Test encoding when no special tokens are registered."""
        tokenizer = RegexBPETokenizer()
        text = "Hello world"
        tokenizer.train(text, vocab_size=280)  # No special tokens

        # Should encode normally without special token handling
        encoded = tokenizer.encode("Hello<|test|>world")
        # <|test|> should be encoded as regular bytes, not as special token
        assert all(token_id < 100000 for token_id in encoded)


class TestRegexBPETokenizerPersistence:
    """Test save/load functionality."""

    @pytest.fixture
    def trained_tokenizer(self) -> RegexBPETokenizer:
        """Fixture providing a trained tokenizer."""
        tokenizer = RegexBPETokenizer()
        training_text = "Hello world! This is a test."
        special_tokens = {"<|endoftext|>": 100257}
        tokenizer.train(training_text, vocab_size=280, special_tokens=special_tokens)
        return tokenizer

    def test_save_and_load(self, trained_tokenizer: RegexBPETokenizer, tmp_path: Path):
        """Test save and load preserves tokenizer state."""
        save_path = tmp_path / "tokenizer.json"
        trained_tokenizer.save(str(save_path))

        # Load into new tokenizer
        new_tokenizer = RegexBPETokenizer()
        new_tokenizer.load(str(save_path))

        assert new_tokenizer.is_trained
        assert new_tokenizer.vocab == trained_tokenizer.vocab
        assert new_tokenizer.merges == trained_tokenizer.merges

    def test_save_and_load_preserves_encoding(
        self, trained_tokenizer: RegexBPETokenizer, tmp_path: Path
    ):
        """Test that save/load preserves encoding behavior."""
        test_text = "Hello world test"
        original_encoded = trained_tokenizer.encode(test_text)

        # Save and load
        save_path = tmp_path / "tokenizer.json"
        trained_tokenizer.save(str(save_path))

        new_tokenizer = RegexBPETokenizer()
        new_tokenizer.load(str(save_path))

        # Encoding should be identical
        new_encoded = new_tokenizer.encode(test_text)
        assert new_encoded == original_encoded

    def test_save_readable(self, trained_tokenizer: RegexBPETokenizer, tmp_path: Path):
        """Test save_readable creates human-readable output."""
        save_path = tmp_path / "tokenizer_readable.txt"
        trained_tokenizer.save_readable(str(save_path))

        # Check file exists and has content
        assert save_path.exists()
        content = save_path.read_text()
        assert len(content) > 0
        assert "Learned merges" in content or "Merges:" in content
        assert "Vocabulary:" in content


class TestRegexBPETokenizerPatternBehavior:
    """Test regex pattern splitting behavior."""

    def test_gpt2_pattern_splits_correctly(self):
        """Test GPT-2 pattern splits text as expected."""
        tokenizer = RegexBPETokenizer(pattern=RegexBPETokenizer.GPT2_SPLIT_PATTERN)
        text = "Hello world! How's it going?"

        chunks = tokenizer.compiled_pattern.findall(text)
        # Should split into meaningful chunks
        assert len(chunks) > 0
        # Joining chunks should reconstruct original
        assert "".join(chunks) == text

    def test_gpt4_pattern_splits_correctly(self):
        """Test GPT-4 pattern splits text as expected."""
        tokenizer = RegexBPETokenizer(pattern=RegexBPETokenizer.GPT4_SPLIT_PATTERN)
        text = "Hello world! How's it going?"

        chunks = tokenizer.compiled_pattern.findall(text)
        assert len(chunks) > 0
        assert "".join(chunks) == text

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("Hello world!", id="basic"),
            pytest.param("Test123", id="alphanumeric"),
            pytest.param("Hello\nworld", id="with_newline"),
            pytest.param("  spaces  ", id="with_spaces"),
            pytest.param("can't won't", id="contractions"),
        ],
    )
    def test_pattern_preserves_text(self, text: str):
        """Test that pattern splitting preserves original text."""
        tokenizer = RegexBPETokenizer()
        chunks = tokenizer.compiled_pattern.findall(text)
        reconstructed = "".join(chunks)
        assert reconstructed == text


class TestRegexBPETokenizerEdgeCases:
    """Test edge cases and error conditions."""

    def test_encode_very_long_text(self):
        """Test encoding very long text."""
        tokenizer = RegexBPETokenizer()
        short_text = "Hello world! "
        tokenizer.train(short_text * 10, vocab_size=280)

        # Encode very long text
        long_text = short_text * 1000
        encoded = tokenizer.encode(long_text)
        assert len(encoded) > 0

        # Verify roundtrip
        decoded = tokenizer.decode(encoded)
        assert decoded == long_text

    def test_encode_unicode_edge_cases(self):
        """Test encoding various Unicode edge cases."""
        tokenizer = RegexBPETokenizer()
        text = "Hello 世界 🌍"
        tokenizer.train(text, vocab_size=280)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_train_with_minimal_text(self):
        """Test training with minimal text."""
        tokenizer = RegexBPETokenizer()
        minimal_text = "aa"
        tokenizer.train(minimal_text, vocab_size=257)

        assert tokenizer.is_trained
        # Should create at least one merge
        assert len(tokenizer.merges) >= 1

    def test_encode_chunk_single_byte(self):
        """Test encoding chunk with single byte."""
        tokenizer = RegexBPETokenizer()
        tokenizer.train("Hello world", vocab_size=280)

        # Single character should still encode
        encoded = tokenizer.encode("H")
        assert len(encoded) == 1

    def test_special_token_not_in_training_text(self):
        """Test special token that doesn't appear in training text."""
        tokenizer = RegexBPETokenizer()
        text = "Hello world"
        special_tokens = {"<|rare|>": 100257}
        tokenizer.train(text, vocab_size=280, special_tokens=special_tokens)

        # Should still handle the special token correctly
        encoded = tokenizer.encode("Hello<|rare|>world")
        assert 100257 in encoded
        decoded = tokenizer.decode(encoded)
        assert "<|rare|>" in decoded
