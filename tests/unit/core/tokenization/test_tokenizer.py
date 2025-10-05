"""Test the tokenizer module.

Created by @pytholic on 2025.09.14
"""

import pytest

from legollm.core.exceptions import TokenizerError
from legollm.core.tokenization.tokenizer import SimpleTokenizer


class TestSimpleTokenizer:
    """Test the SimpleTokenizer class."""

    @pytest.mark.parametrize(
        "text, expected_tokens",
        [
            pytest.param(
                "Hello world This, is a test.",
                ["Hello", "world", "This", ",", "is", "a", "test", "."],
                id="simple",
            ),
            pytest.param(
                "Hello world! This, is-- a test.",
                ["Hello", "world", "!", "This", ",", "is", "-", "-", "a", "test", "."],
                id="with_punctuation",
            ),
        ],
    )
    def test_tokenize(self, text: str, expected_tokens: list[str]):
        """Test the tokenize method."""
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        assert tokens == expected_tokens

    def test_encode_and_decode(self):
        """Test the encode and decode methods."""
        text = "Hello world! This, is-- a test."
        vocab = {
            "Hello": 1,
            "world": 2,
            "This": 3,
            "is": 4,
            "a": 5,
            "test": 6,
            "!": 7,
            ",": 8,
            "-": 9,
            ".": 10,
            "<|UNK|>": 0,  # Required for unknown token handling
        }
        tokenizer = SimpleTokenizer(vocab)
        ids = tokenizer.encode(text)
        assert ids == [1, 2, 7, 3, 8, 4, 9, 9, 5, 6, 10]
        assert tokenizer.decode(ids) == text

    def test_encode_and_decode_with_unknown_token(self):
        """Test the encode and decode methods with an unknown token."""
        text = "Hello world! This, is-- a test. sasdads token."
        expected_decoded = "Hello world! This, is-- a test. <|UNK|> token."
        vocab = {
            "Hello": 1,
            "world": 2,
            "This": 3,
            "is": 4,
            "a": 5,
            "test": 6,
            "!": 7,
            ",": 8,
            "-": 9,
            ".": 10,
            "token": 11,
            "<|UNK|>": 0,
        }
        tokenizer = SimpleTokenizer(vocab)
        ids = tokenizer.encode(text)
        assert ids == [1, 2, 7, 3, 8, 4, 9, 9, 5, 6, 10, 0, 11, 10]
        assert tokenizer.decode(ids) == expected_decoded

    def test_encode_raise_error_on_no_vocab(self):
        """Test the encode method raises an error on no vocabulary."""
        tokenizer = SimpleTokenizer()
        with pytest.raises(
            TokenizerError,
            match="Cannot encode without vocabulary. Initialize with vocab or use from_corpus",
        ):
            tokenizer.encode("Hello world! This, is-- a test.")

    def test_encode_raise_error_on_no_unk_token(self):
        """Test the encode method raises an error on no UNK token."""
        vocab = {
            "Hello": 1,
            "world": 2,
            "This": 3,
            "is": 4,
            "a": 5,
            "test": 6,
        }
        tokenizer = SimpleTokenizer(vocab)
        with pytest.raises(
            TokenizerError, match="Vocabulary must contain <|UNK|> for unknown token handling"
        ):
            tokenizer.encode("Hello world! This, is-- a test.")
