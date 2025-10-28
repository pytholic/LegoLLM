"""Test the prepare script.

Created by @pytholic on 2025.10.20
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pytest
from pytest import TempPathFactory
from pytest_mock import MockerFixture

from legollm.core.interfaces import TrainableTokenizer
from legollm.utils import read_text_file
from scripts.prepare import (
    DatasetConfig,
    DocumentSplitMethod,
    TokenizerType,
    prepare_dataset,
    split_documents,
    train_tokenizer,
)

TOKENIZER_TYPE = TokenizerType.REGEX_BPE


@pytest.fixture(scope="class")
def fixtures_dir() -> Path:
    """Return path to unit test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="class")
def shared_tokenizer_dir(tmp_path_factory: TempPathFactory) -> Path:
    """Create a shared tokenizer directory for the entire test class."""
    return tmp_path_factory.mktemp("tokenizers")


@pytest.fixture(scope="class")
def trained_tokenizer(shared_tokenizer_dir: Path, fixtures_dir: Path) -> TrainableTokenizer:
    """Train a tokenizer and save it to the shared tokenizer directory."""
    tokenizer = train_tokenizer(
        tokenizer_type=TOKENIZER_TYPE,
        text=read_text_file(fixtures_dir / "sample_input.txt"),
        vocab_size=256,
        special_tokens={"<|endoftext|>": 100257},
        verbose=False,
    )

    tokenizer_path = shared_tokenizer_dir / f"sample_input_{TOKENIZER_TYPE}.json"

    tokenizer.save(tokenizer_path.as_posix())
    return shared_tokenizer_dir


@pytest.fixture
def test_config(tmp_path: Path, fixtures_dir: Path) -> DatasetConfig:
    """Test config for testing."""
    return DatasetConfig(
        name="test_dataset",
        raw_file=fixtures_dir / "sample_input.txt",
        processed_dir=tmp_path / "test_processed",
        tokenizer_type=TOKENIZER_TYPE,
        vocab_size=256,
        train_split=0.8,
        block_size=64,
        special_tokens={"<|endoftext|>": 100257},
        min_document_chars=0,  # no filtering
    )


@pytest.fixture
def test_input(fixtures_dir: Path) -> Path:
    """Test input for testing."""
    return fixtures_dir / "sample_input.txt"


class TestSplitDocuments:
    """Test the split_documents function and its various split strategies."""

    def test_splitter_none_return_whole_text(self, test_config: DatasetConfig):
        """Test that the splitter none returns the whole text."""
        test_config.document_splitter = DocumentSplitMethod.NONE
        text = read_text_file(test_config.raw_file)
        docs = split_documents(test_config, text)
        assert len(docs) == 1
        assert docs[0] == text, "Text was not split into a single document"

    def test_splitter_double_newline_return_multiple_basic_documents(
        self, test_config: DatasetConfig
    ):
        """Test that the splitter double newline returns multiple basic documents."""
        test_config.document_splitter = DocumentSplitMethod.DOUBLE_NEWLINE
        text = """
        This is a test document.
        It contains multiple lines.\n\n
        This is another test document.
        It contains multiple lines.\n\n
        This is a third test document.
        It contains multiple lines.\n\n
        This is a fourth test document.
        It contains multiple lines.\n\n
        """
        docs = split_documents(test_config, text)
        assert len(docs) == 4
        assert all(len(doc) > 0 for doc in docs), "Some documents are empty"
        assert all(doc.count("\n") == 1 for doc in docs), (
            "Some documents are not basic documents (i.e. contain multiple newlines)"
        )  # internal new lines are preserved

    def test_splitter_double_newline_handles_excessive_blanks(self, test_config: DatasetConfig):
        """Test that the splitter double newline handles excessive blanks and no empty documents.

        Example:
            ```text
            First sentence.
            <-- Blank Line 1 -->
            <-- Blank Line 2 -->
            <-- Blank Line 3 -->
            Second sentence.
            <-- Blank Line 1 -->
            <-- Blank Line 2 -->
            <-- Blank Line 3 -->
            Third sentence.
            <-- Blank Line 1 -->
            <-- Blank Line 2 -->
            ```
        Expected output:
            Documents:
            - First sentence.
            - Second sentence.
            - Third sentence.
        """
        test_config.document_splitter = DocumentSplitMethod.DOUBLE_NEWLINE
        text = """
        First sentence.\n\n\n
        Second sentence.\n\n\n\n\n
        Third sentence.\n\n
        Fourth sentence.
        """
        docs = split_documents(test_config, text)
        assert len(docs) == 4
        assert docs[0] == "First sentence."
        assert docs[1] == "Second sentence."
        assert docs[2] == "Third sentence."
        assert docs[3] == "Fourth sentence."

    def test_splitter_regex_requires_pattern(self, test_config: DatasetConfig):
        """Test that the splitter regex requires a pattern."""
        test_config.document_splitter = DocumentSplitMethod.REGEX
        with pytest.raises(
            ValueError, match="document_regex must be provided when document_splitter=regex"
        ):
            split_documents(test_config, "Hello, world!")

    @pytest.mark.parametrize(
        "text, pattern, expected_docs",
        [
            pytest.param(
                "First sentence.\n\nSecond sentence.\n\nThird sentence.",
                r"[\r\n]{2,}",
                ["First sentence.", "Second sentence.", "Third sentence."],
                id="basic",
            ),
            pytest.param(
                "Chapter 1###Chapter 2###Chapter 3",
                r"###",
                ["Chapter 1", "Chapter 2", "Chapter 3"],
                id="multiple_hashes",
            ),
            pytest.param(
                "FieldA\tFieldB\tFieldC",
                r"\s+",
                ["FieldA", "FieldB", "FieldC"],
                id="multiple_tabs",
            ),
            pytest.param(
                "Header<doc>Content 1</doc><doc>Content 2</doc>",
                r"(?:<doc>|<\/doc>)",
                ["Header", "Content 1", "Content 2"],
                id="multiple_xml_tags",
            ),
            pytest.param(
                "Part A--SEP--Part B--SEP--Part C",
                r"--SEP--",
                ["Part A", "Part B", "Part C"],
                id="multiple_separators",
            ),
        ],
    )
    def test_splitter_regex_splits_basic_documents(
        self, test_config: DatasetConfig, text: str, pattern: str, expected_docs: list[str]
    ):
        """Test that the splitter regex splits basic documents."""
        test_config.document_splitter = DocumentSplitMethod.REGEX
        test_config.document_regex = pattern
        docs = split_documents(test_config, text)
        assert len(docs) == len(expected_docs)
        assert docs == expected_docs, f"Docs: {docs} != Expected docs: {expected_docs}"

    def test_splitter_regex_handles_empty_documents(self, test_config: DatasetConfig):
        """Test that the splitter regex handles empty documents."""
        test_config.document_splitter = DocumentSplitMethod.REGEX
        test_config.document_regex = r"[\r\n]{2,}"
        docs = split_documents(test_config, "")
        assert len(docs) == 0

    @pytest.mark.parametrize(
        "text, min_document_chars, expected_docs",
        [
            pytest.param(
                "First sentence.\n\n Second sentence.",
                0,
                ["First sentence.", "Second sentence."],
                id="no_filter",
            ),
            pytest.param(
                "First\n\nsentence. Second sentence.",
                10,
                ["sentence. Second sentence."],
                id="filter_short",
            ),
            pytest.param(
                "First sentence. This is a long sentence.",
                50,
                [],
                id="filter_long",
            ),
        ],
    )
    def test_min_document_chars_filters_documents(
        self,
        test_config: DatasetConfig,
        text: str,
        min_document_chars: int,
        expected_docs: list[str],
    ):
        """Test that the min_document_chars filters documents."""
        test_config.min_document_chars = min_document_chars
        test_config.document_splitter = DocumentSplitMethod.DOUBLE_NEWLINE
        docs = split_documents(test_config, text)
        assert docs == expected_docs, f"Docs: {docs} != Expected docs: {expected_docs}"


class TestShuffleDocuments:
    def test_no_shuffle_order_preserved(self, test_config: DatasetConfig):
        """Test that the order of documents is preserved when shuffling is disabled."""
        test_config.shuffle_documents = False
        text = "First sentence. Second sentence. Third sentence."
        expected_docs = ["First sentence. Second sentence. Third sentence."]
        docs = split_documents(test_config, text)
        assert docs == expected_docs, f"Docs: {docs} != Expected docs: {expected_docs}"

    def test_shuffle_deterministic_with_seed(self, test_config: DatasetConfig):
        """Test that the order of documents is deterministic when shuffling is enabled with a seed."""
        test_config.shuffle_documents = True
        test_config.document_splitter = DocumentSplitMethod.DOUBLE_NEWLINE
        test_config.shuffle_seed = 42
        text = "First sentence.\n\n Second sentence.\n\n Third sentence."
        expected_docs = ["Second sentence.", "First sentence.", "Third sentence."]
        docs = split_documents(test_config, text)
        assert docs == expected_docs, f"Docs: {docs} != Expected docs: {expected_docs}"

    def test_shuffle_changes_with_different_seed(self, test_config: DatasetConfig):
        """Test that the order of documents changes with a different seed."""
        test_config.shuffle_documents = True
        test_config.document_splitter = DocumentSplitMethod.DOUBLE_NEWLINE
        test_config.shuffle_seed = 42
        text = "First sentence.\n\n Second sentence.\n\n Third sentence."
        expected_docs1 = ["Second sentence.", "First sentence.", "Third sentence."]
        docs1 = split_documents(test_config, text)
        assert docs1 == expected_docs1, f"Docs: {docs1} != Expected docs: {expected_docs1}"
        test_config.shuffle_seed = 43
        docs2 = split_documents(test_config, text)
        assert docs2 != docs1, "Docs are not different with different seed"

    def test_shuffle_content_preserved(self, test_config: DatasetConfig):
        """Test that the content of documents is preserved when shuffling."""
        test_config.shuffle_documents = True
        test_config.document_splitter = DocumentSplitMethod.DOUBLE_NEWLINE
        test_config.shuffle_seed = 42
        text = "First sentence.\n\n Second sentence.\n\n Third sentence."
        expected_docs = ["First sentence.", "Second sentence.", "Third sentence."]
        docs = sorted(split_documents(test_config, text))
        assert docs == expected_docs, f"Docs: {docs} != Expected docs: {expected_docs}"

    def test_shuffle_skipped_if_less_than_two_documents(
        self, test_config: DatasetConfig, mocker: MockerFixture
    ):
        """Test that the shuffle is skipped if there are less than two documents."""
        test_config.shuffle_documents = True
        test_config.document_splitter = DocumentSplitMethod.DOUBLE_NEWLINE
        test_config.shuffle_seed = 42
        text = "First sentence."
        spy = mocker.spy(random.Random, "shuffle")
        docs = split_documents(test_config, text)
        assert docs == ["First sentence."], f"Docs: {docs} != Expected docs: ['First sentence.']"
        assert spy.call_count == 0, "shuffle was not called"


class TestPrepare:
    """Test the prepare script."""

    def test_prepare_creates_correct_files(
        self, test_config: DatasetConfig, trained_tokenizer: Path
    ):
        """Test that the prepare script creates the correct files."""
        prepare_dataset(test_config, tokenizer_dir=trained_tokenizer)
        assert test_config.processed_dir.exists()
        assert (test_config.processed_dir / "train.bin").exists()
        assert (test_config.processed_dir / "val.bin").exists()
        assert (test_config.processed_dir / "meta.json").exists()

    def test_prepare_exits_early_if_files_exist(
        self, test_config: DatasetConfig, trained_tokenizer: Path
    ):
        """Test that the prepare script does not re-process existing files."""
        # First run
        prepare_dataset(test_config, tokenizer_dir=trained_tokenizer)

        # Get file modification times
        train_path = test_config.processed_dir / "train.bin"
        val_path = test_config.processed_dir / "val.bin"
        train_mtime_1 = train_path.stat().st_mtime
        val_mtime_1 = val_path.stat().st_mtime

        time.sleep(0.1)

        # Second run
        prepare_dataset(test_config, tokenizer_dir=trained_tokenizer)

        # Verify files weren't modified (same modification times)
        train_mtime_2 = train_path.stat().st_mtime
        val_mtime_2 = val_path.stat().st_mtime

        assert train_mtime_1 == train_mtime_2, "train.bin was modified"
        assert val_mtime_1 == val_mtime_2, "val.bin was modified"

    def test_train_val_split(self, test_config: DatasetConfig, trained_tokenizer: Path):
        """Test that the train/val split is correct."""
        prepare_dataset(test_config, tokenizer_dir=trained_tokenizer)
        assert test_config.processed_dir.exists()
        assert (test_config.processed_dir / "train.bin").exists()
        assert (test_config.processed_dir / "val.bin").exists()
        assert (test_config.processed_dir / "meta.json").exists()

        # load the train/val splits
        with open(test_config.processed_dir / "meta.json") as f:
            meta = json.load(f)
        data_dtype = np.dtype(meta["data_dtype"])

        train_tokens = np.fromfile(test_config.processed_dir / "train.bin", dtype=data_dtype)
        val_tokens = np.fromfile(test_config.processed_dir / "val.bin", dtype=data_dtype)

        total_tokens = len(train_tokens) + len(val_tokens)
        # check that the train/val split is correct
        assert abs(len(train_tokens) / total_tokens - test_config.train_split) < 0.01

    def test_no_data_leakage(self, test_config: DatasetConfig, trained_tokenizer: Path):
        """Test that there is no data leakage between train and val splits."""
        prepare_dataset(test_config, tokenizer_dir=trained_tokenizer)
        assert test_config.processed_dir.exists()
        assert (test_config.processed_dir / "train.bin").exists()
        assert (test_config.processed_dir / "val.bin").exists()
        assert (test_config.processed_dir / "meta.json").exists()

        # load the train/val splits
        with open(test_config.processed_dir / "meta.json") as f:
            meta = json.load(f)
        data_dtype = np.dtype(meta["data_dtype"])

        train_tokens = np.fromfile(test_config.processed_dir / "train.bin", dtype=data_dtype)
        val_tokens = np.fromfile(test_config.processed_dir / "val.bin", dtype=data_dtype)

        assert len(train_tokens) + len(val_tokens) == meta["total_tokens"], (
            "Total tokens mismatch => Data leakage detected"
        )

    def test_metadata_accuracy(self, test_config: DatasetConfig, trained_tokenizer: Path):
        """Test that meta.json contains correct information."""
        prepare_dataset(test_config, tokenizer_dir=trained_tokenizer)

        with open(test_config.processed_dir / "meta.json") as f:
            meta = json.load(f)

        # Load actual data
        data_dtype = np.dtype(meta["data_dtype"])
        train_tokens = np.fromfile(test_config.processed_dir / "train.bin", dtype=data_dtype)
        val_tokens = np.fromfile(test_config.processed_dir / "val.bin", dtype=data_dtype)

        # Verify metadata matches reality
        assert meta["train_tokens"] == len(train_tokens)
        assert meta["val_tokens"] == len(val_tokens)
        assert meta["dataset_name"] == test_config.name
        assert meta["train_split"] == test_config.train_split
        assert "vocab_size" in meta
        assert "tokenizer_path" in meta
