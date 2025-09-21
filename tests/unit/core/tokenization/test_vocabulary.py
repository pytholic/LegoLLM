from pathlib import Path

import pytest

from legollm.core.tokenization.vocabulary import (
    END_OF_TEXT_TOKEN,
    UNK_TOKEN,
    VocabularyBuilder,
    VocabularyManager,
)


@pytest.fixture
def vocabulary_builder() -> VocabularyBuilder:
    """Fixture for the VocabularyBuilder class."""
    return VocabularyBuilder()


@pytest.fixture
def vocabulary_manager() -> VocabularyManager:
    """Fixture for the VocabularyManager class."""
    return VocabularyManager()


class TestVocabularyBuilder:
    """Test the VocabularyBuilder class."""

    def test_build_from_tokens_success(self, vocabulary_builder: VocabularyBuilder):
        """Test the build_from_tokens method."""
        tokens = ["hello", "world", "!"]
        vocabulary = vocabulary_builder.build_from_tokens(tokens)
        # Should include original tokens + special tokens
        expected = {"!": 0, "hello": 1, "world": 2, UNK_TOKEN: 3, END_OF_TEXT_TOKEN: 4}
        assert vocabulary == expected

    def test_build_from_tokens_remove_duplicates(self, vocabulary_builder: VocabularyBuilder):
        """Test the build_from_tokens method removes duplicates."""
        tokens = ["hello", "world", "hello"]
        vocabulary = vocabulary_builder.build_from_tokens(tokens)
        # Should include deduplicated tokens + special tokens
        expected = {"hello": 0, "world": 1, UNK_TOKEN: 2, END_OF_TEXT_TOKEN: 3}
        assert vocabulary == expected

    def test_build_from_tokens_raise_error_on_empty_list(
        self, vocabulary_builder: VocabularyBuilder
    ):
        """Test the build_from_tokens method with an empty list."""
        tokens = []
        with pytest.raises(ValueError, match="Cannot build vocabulary from empty tokens list"):
            vocabulary_builder.build_from_tokens(tokens)

    def test_special_tokens_in_vocabulary(self):
        """Test that special tokens (UNK, END_OF_TEXT) are properly added to vocabulary."""
        training_tokens = ["hello", "world", "test"]

        vocab_builder = VocabularyBuilder()
        vocab = vocab_builder.build_from_tokens(training_tokens)

        # Verify special tokens are present
        assert UNK_TOKEN in vocab
        assert END_OF_TEXT_TOKEN in vocab

        # Verify they have valid IDs
        assert isinstance(vocab[UNK_TOKEN], int)
        assert isinstance(vocab[END_OF_TEXT_TOKEN], int)

        # Verify all original tokens are also present
        for token in training_tokens:
            assert token in vocab


class TestVocabularyManager:
    """Test the VocabularyManager class."""

    def test_save_and_load(self, vocabulary_manager: VocabularyManager, tmp_path: Path):
        """Test the save and load methods."""
        vocabulary = {"!": 0, "hello": 1, "world": 2}
        vocabulary_manager.save(vocabulary, tmp_path / "test_vocabulary.json")
        loaded_vocabulary = vocabulary_manager.load(tmp_path / "test_vocabulary.json")
        assert loaded_vocabulary == vocabulary

    def test_load_raise_error_on_invalid_path(self, vocabulary_manager: VocabularyManager):
        """Test the load method raises an error on an invalid path."""
        file_name = "invalid_path.json"
        path = Path(file_name)
        expected_match = f"File {file_name} does not exist"
        with pytest.raises(FileNotFoundError, match=expected_match):
            vocabulary_manager.load(path)

    def test_save_raise_error_on_invalid_path(
        self, vocabulary_manager: VocabularyManager, tmp_path: Path
    ):
        """Test the save method raises an error when parent directory doesn't exist."""
        file_name = "vocabulary.json"
        path = tmp_path / "non_existent_dir" / file_name
        vocabulary = {"!": 0, "hello": 1, "world": 2}

        with pytest.raises(
            FileNotFoundError, match=f"Parent directory {path.parent} does not exist"
        ):
            vocabulary_manager.save(vocabulary, path)
