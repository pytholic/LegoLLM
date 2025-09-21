"""Test the tokenizer module.

Created by @pytholic on 2025.09.14
"""

import pytest

from legollm.core.tokenization.tokenizer import SimpleTokenizer


@pytest.fixture
def tokenizer() -> SimpleTokenizer:
    """Fixture for the SimpleTokenizer class."""
    return SimpleTokenizer()


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
    def test_tokenize(self, text: str, expected_tokens: list[str], tokenizer: SimpleTokenizer):
        """Test the tokenize method."""
        tokens = tokenizer.tokenize(text)
        assert tokens == expected_tokens

    def test_encode_and_decode(self, tokenizer: SimpleTokenizer):
        """Test the encode and decode methods."""
        text = "Hello world! This, is-- a test."
        vocabulary = {
            "Hello": 0,
            "world": 1,
            "This": 2,
            "is": 3,
            "a": 4,
            "test": 5,
            "!": 6,
            ",": 7,
            "-": 8,
            ".": 9,
        }
        tokenizer = SimpleTokenizer(vocabulary)
        ids = tokenizer.encode(text)
        assert ids == [0, 1, 6, 2, 7, 3, 8, 8, 4, 5, 9]
        assert tokenizer.decode(ids) == text
