"""Test vocabulary utility functions.

Created by @pytholic on 2025.09.21
"""

from pathlib import Path

import pytest

from legollm.core.exceptions import TokenizerError
from legollm.core.tokenization.simple.vocabulary import (
    END_OF_TEXT_TOKEN,
    UNK_TOKEN,
    build_vocab_from_tokens,
    load_vocab,
    save_vocab,
)


class TestBuildVocabFromTokens:
    """Test the build_vocab_from_tokens utility function."""

    def test_build_from_tokens_success(self):
        """Test building vocabulary from tokens."""
        tokens = ["hello", "world", "!"]
        vocabulary = build_vocab_from_tokens(tokens)
        # Should include original tokens + special tokens
        expected = {"!": 0, "hello": 1, "world": 2, UNK_TOKEN: 3, END_OF_TEXT_TOKEN: 4}
        assert vocabulary == expected

    def test_build_from_tokens_remove_duplicates(self):
        """Test that duplicate tokens are removed."""
        tokens = ["hello", "world", "hello"]
        vocabulary = build_vocab_from_tokens(tokens)
        # Should include deduplicated tokens + special tokens
        expected = {"hello": 0, "world": 1, UNK_TOKEN: 2, END_OF_TEXT_TOKEN: 3}
        assert vocabulary == expected

    def test_build_from_tokens_with_min_frequency(self):
        """Test building vocabulary with minimum frequency filter."""
        tokens = ["hello", "hello", "world", "test"]
        vocabulary = build_vocab_from_tokens(tokens, min_frequency=2)
        # Only "hello" appears twice, others filtered out
        expected = {"hello": 0, UNK_TOKEN: 1, END_OF_TEXT_TOKEN: 2}
        assert vocabulary == expected

    def test_build_from_tokens_with_custom_special_tokens(self):
        """Test building vocabulary with custom special tokens."""
        tokens = ["hello", "world"]
        special_tokens = ["<pad>", "<eos>"]
        vocabulary = build_vocab_from_tokens(tokens, special_tokens=special_tokens)
        # Should include original tokens + custom special tokens
        expected = {"hello": 0, "world": 1, "<pad>": 2, "<eos>": 3}
        assert vocabulary == expected

    def test_build_from_tokens_raise_error_on_empty_list(self):
        """Test that empty token list raises an error."""
        tokens = []
        with pytest.raises(TokenizerError, match="Cannot build vocabulary from empty tokens list"):
            build_vocab_from_tokens(tokens)

    def test_special_tokens_in_vocabulary(self):
        """Test that special tokens are properly added to vocabulary."""
        training_tokens = ["hello", "world", "test"]
        vocab = build_vocab_from_tokens(training_tokens)

        # Verify special tokens are present
        assert UNK_TOKEN in vocab
        assert END_OF_TEXT_TOKEN in vocab

        # Verify they have valid IDs
        assert isinstance(vocab[UNK_TOKEN], int)
        assert isinstance(vocab[END_OF_TEXT_TOKEN], int)

        # Verify all original tokens are also present
        for token in training_tokens:
            assert token in vocab


class TestSaveAndLoadVocab:
    """Test the save_vocab and load_vocab utility functions."""

    def test_save_and_load(self, tmp_path: Path):
        """Test saving and loading vocabulary."""
        vocabulary = {"!": 0, "hello": 1, "world": 2}
        save_vocab(vocabulary, tmp_path / "test_vocabulary.json")
        loaded_vocabulary = load_vocab(tmp_path / "test_vocabulary.json")
        assert loaded_vocabulary == vocabulary

    def test_load_raise_error_on_invalid_path(self):
        """Test that loading from invalid path raises an error."""
        file_name = "invalid_path.json"
        path = Path(file_name)
        expected_match = f"File {file_name} does not exist"
        with pytest.raises(FileNotFoundError, match=expected_match):
            load_vocab(path)

    def test_save_raise_error_on_invalid_path(self, tmp_path: Path):
        """Test that saving to invalid path raises an error."""
        file_name = "vocabulary.json"
        path = tmp_path / "non_existent_dir" / file_name
        vocabulary = {"!": 0, "hello": 1, "world": 2}

        with pytest.raises(
            FileNotFoundError, match=f"Parent directory {path.parent} does not exist"
        ):
            save_vocab(vocabulary, path)
