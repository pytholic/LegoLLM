"""Test the tokenizer module.

Created by @pytholic on 2025.09.14
"""

import pytest

from legollm.core.tokenizer import SimpleTokenizer


class TestSimpleTokenizer:
    """Test the WhiteSpaceTokenizer class."""

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
    def test_tokenize(self, text: str, expected_tokens: list[str]) -> None:
        """Test the tokenize method."""
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        assert tokens == expected_tokens
