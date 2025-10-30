"""Integration tests for prepare.py end-to-end pipeline.

Tests the full prepare_dataset workflow including:
- Document splitting and shuffling
- Tokenization
- Train/val splitting
- Metadata generation

Created by @pytholic on 2025.10.28
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare import DatasetConfig, prepare_dataset


@pytest.fixture(scope="class")
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent.parent.parent / "unit" / "fixtures"


@pytest.fixture(scope="class")
def shared_tokenizer_dir(tmp_path_factory) -> Path:
    """Create a shared tokenizer directory for the entire test class."""
    return tmp_path_factory.mktemp("tokenizers")


@pytest.fixture(scope="class")
def trained_tokenizer(shared_tokenizer_dir: Path, fixtures_dir: Path):
    """Train tokenizer once and reuse across all tests in the class.

    Note:
        - Trained on sample_input.txt, but tests may use different text.
    This is intentional - we're testing the prepare pipeline, not tokenization quality. The small vocab_size (256) ensures byte-level fallback works for any text.
    """
    from legollm.utils import read_text_file
    from scripts.prepare import train_tokenizer

    text = read_text_file(fixtures_dir / "sample_input.txt")

    tokenizer = train_tokenizer(
        tokenizer_type="regex_bpe",
        text=text,
        vocab_size=256,
        special_tokens={"<|endoftext|>": 100257},
        verbose=False,
    )

    tokenizer_path = shared_tokenizer_dir / "test_dataset_regex_bpe.json"
    tokenizer.save(tokenizer_path.as_posix())

    return shared_tokenizer_dir


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file with multiple documents."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "First document with some content.\n"
        "It has multiple lines.\n\n"
        "Second document is here.\n"
        "Also with multiple lines.\n\n"
        "Third document appears.\n"
        "More content here.\n\n"
        "Fourth document last one.\n"
        "Final lines of text."
    )
    return text_file


class TestPrepareDatasetIntegration:
    """Integration tests for the full prepare_dataset pipeline."""

    def test_prepare_writes_document_metadata(
        self, tmp_path: Path, sample_text_file: Path, trained_tokenizer: Path
    ):
        """Verify meta.json contains all document-related fields."""
        config = DatasetConfig(
            name="test_doc_meta",
            raw_file=sample_text_file,
            processed_dir=tmp_path / "processed",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            document_splitter="double_newline",
            shuffle_documents=True,
            shuffle_seed=42,
            min_document_chars=10,
        )

        prepare_dataset(config, tokenizer_dir=trained_tokenizer)

        meta_path = config.processed_dir / "meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        # Check all document-related metadata fields exist
        required_fields = [
            "num_documents",
            "document_splitter",
            "document_regex",
            "shuffle_documents",
            "shuffle_seed",
            "min_document_chars",
            "join_separator",
        ]
        for field in required_fields:
            assert field in meta, f"Missing field: {field}"

        # Verify values match config
        assert meta["document_splitter"] == "double_newline"
        assert meta["shuffle_documents"] is True
        assert meta["shuffle_seed"] == 42
        assert meta["num_documents"] == 4  # Sample has 4 documents

    def test_prepare_respects_splitter_and_join(
        self, tmp_path: Path, sample_text_file: Path, trained_tokenizer: Path
    ):
        """Verify document splitting affects token count and metadata."""
        # Test with no splitting
        config_no_split = DatasetConfig(
            name="test_no_split",
            raw_file=sample_text_file,
            processed_dir=tmp_path / "no_split",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            document_splitter="none",
        )
        prepare_dataset(config_no_split, tokenizer_dir=trained_tokenizer)

        # Test with double newline splitting
        config_with_split = DatasetConfig(
            name="test_with_split",
            raw_file=sample_text_file,
            processed_dir=tmp_path / "with_split",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            document_splitter="double_newline",
            shuffle_documents=False,
        )
        prepare_dataset(config_with_split, tokenizer_dir=trained_tokenizer)

        # Load metadata
        with open(config_no_split.processed_dir / "meta.json") as f:
            meta_no_split = json.load(f)
        with open(config_with_split.processed_dir / "meta.json") as f:
            meta_with_split = json.load(f)

        # No split should have 1 document
        assert meta_no_split["num_documents"] == 1

        # With split should have 2 documents (after min_document_chars=50 filtering)
        assert meta_with_split["num_documents"] == 2

        # Split version should have fewer characters due to filtering short documents
        assert meta_with_split["total_characters"] < meta_no_split["total_characters"]

    def test_prepare_shuffle_deterministic(
        self, tmp_path: Path, sample_text_file: Path, trained_tokenizer: Path
    ):
        """Verify same seed produces identical results."""
        seed = 1337

        # First run
        config1 = DatasetConfig(
            name="test_shuffle_1",
            raw_file=sample_text_file,
            processed_dir=tmp_path / "shuffle_1",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            document_splitter="double_newline",
            shuffle_documents=True,
            shuffle_seed=seed,
        )
        prepare_dataset(config1, tokenizer_dir=trained_tokenizer)

        # Second run with same seed
        config2 = DatasetConfig(
            name="test_shuffle_2",
            raw_file=sample_text_file,
            processed_dir=tmp_path / "shuffle_2",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            document_splitter="double_newline",
            shuffle_documents=True,
            shuffle_seed=seed,
        )
        prepare_dataset(config2, tokenizer_dir=trained_tokenizer)

        # Load both train.bin files
        with open(config1.processed_dir / "meta.json") as f:
            meta1 = json.load(f)
        with open(config2.processed_dir / "meta.json") as f:
            meta2 = json.load(f)

        dtype = np.dtype(meta1["data_dtype"])
        train1 = np.fromfile(config1.processed_dir / "train.bin", dtype=dtype)
        train2 = np.fromfile(config2.processed_dir / "train.bin", dtype=dtype)

        # Should be identical
        assert np.array_equal(train1, train2), "Same seed should produce identical results"
        assert meta1["total_tokens"] == meta2["total_tokens"]

    def test_prepare_different_seeds_change_order(self, tmp_path: Path, trained_tokenizer: Path):
        """Verify different seeds produce different token orders."""
        # Create text with 5 documents to ensure different shuffle orders
        text_file = tmp_path / "multi_docs.txt"
        text_file.write_text(
            "Document one with enough content to pass minimum threshold.\n\n"
            "Document two with enough content to pass minimum threshold.\n\n"
            "Document three with enough content to pass minimum threshold.\n\n"
            "Document four with enough content to pass minimum threshold.\n\n"
            "Document five with enough content to pass minimum threshold."
        )

        # First run with seed 1
        config1 = DatasetConfig(
            name="test_seed_1",
            raw_file=text_file,
            processed_dir=tmp_path / "seed_1",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            document_splitter="double_newline",
            shuffle_documents=True,
            shuffle_seed=1,
            min_document_chars=20,  # Lower threshold to keep all documents
        )
        prepare_dataset(config1, tokenizer_dir=trained_tokenizer)

        # Second run with seed 2
        config2 = DatasetConfig(
            name="test_seed_2",
            raw_file=text_file,
            processed_dir=tmp_path / "seed_2",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            document_splitter="double_newline",
            shuffle_documents=True,
            shuffle_seed=2,
            min_document_chars=20,  # Lower threshold to keep all documents
        )
        prepare_dataset(config2, tokenizer_dir=trained_tokenizer)

        # Load both train.bin files
        with open(config1.processed_dir / "meta.json") as f:
            meta1 = json.load(f)
        with open(config2.processed_dir / "meta.json") as f:
            meta2 = json.load(f)

        dtype = np.dtype(meta1["data_dtype"])
        train1 = np.fromfile(config1.processed_dir / "train.bin", dtype=dtype)
        train2 = np.fromfile(config2.processed_dir / "train.bin", dtype=dtype)

        # Should be different (different order)
        assert not np.array_equal(train1, train2), "Different seeds should produce different orders"

        # But same total tokens
        assert len(train1) == len(train2)
        assert meta1["total_tokens"] == meta2["total_tokens"]

    def test_train_val_split_stable_with_shuffle(
        self, tmp_path: Path, sample_text_file: Path, trained_tokenizer: Path
    ):
        """Verify train/val split ratio holds after document shuffling."""
        config = DatasetConfig(
            name="test_split_stable",
            raw_file=sample_text_file,
            processed_dir=tmp_path / "split_stable",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            train_split=0.8,
            document_splitter="double_newline",
            shuffle_documents=True,
            shuffle_seed=42,
        )
        prepare_dataset(config, tokenizer_dir=trained_tokenizer)

        # Load metadata and data
        with open(config.processed_dir / "meta.json") as f:
            meta = json.load(f)

        dtype = np.dtype(meta["data_dtype"])
        train_tokens = np.fromfile(config.processed_dir / "train.bin", dtype=dtype)
        val_tokens = np.fromfile(config.processed_dir / "val.bin", dtype=dtype)

        # Check split ratio
        total_tokens = len(train_tokens) + len(val_tokens)
        actual_split = len(train_tokens) / total_tokens

        assert abs(actual_split - config.train_split) < 0.01, (
            f"Train split {actual_split:.3f} differs from expected {config.train_split}"
        )

        # Check no data leakage
        assert total_tokens == meta["total_tokens"], "Total tokens mismatch"
        assert len(train_tokens) == meta["train_tokens"]
        assert len(val_tokens) == meta["val_tokens"]

    def test_min_document_chars_filters_correctly(self, tmp_path: Path, trained_tokenizer: Path):
        """Verify min_document_chars filters out short documents."""
        # Create text with documents of varying lengths
        text_file = tmp_path / "varying_lengths.txt"
        text_file.write_text(
            "Short.\n\n"  # 6 chars (will be filtered if min > 6)
            "This is a medium length document with more content.\n\n"  # ~50 chars
            "A.\n\n"  # 2 chars (will be filtered)
            "Another long document with plenty of text to meet the minimum threshold."  # ~70 chars
        )

        # Test with min_document_chars=20
        config = DatasetConfig(
            name="test_min_chars",
            raw_file=text_file,
            processed_dir=tmp_path / "filtered",
            tokenizer_type="regex_bpe",
            vocab_size=256,
            document_splitter="double_newline",
            min_document_chars=20,
        )
        prepare_dataset(config, tokenizer_dir=trained_tokenizer)

        with open(config.processed_dir / "meta.json") as f:
            meta = json.load(f)

        # Should only have 2 documents (the two long ones)
        assert meta["num_documents"] == 2, "Should filter out short documents"
