"""Test the prepare script.

Created by @pytholic on 2025.10.20
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare import DatasetConfig, prepare_dataset


@pytest.fixture
def test_config(tmp_path: Path, fixtures_dir: Path) -> DatasetConfig:
    """Test config for testing."""
    print(fixtures_dir)
    return DatasetConfig(
        name="test_dataset",
        raw_file=fixtures_dir / "test_input.txt",
        processed_dir=tmp_path / "test_processed",
        tokenizer_type="regex_bpe",
        vocab_size=256,
        train_split=0.8,
        block_size=64,
        special_tokens={"<|endoftext|>": 255},
    )


@pytest.fixture
def test_input(fixtures_dir: Path) -> Path:
    """Test input for testing."""
    return fixtures_dir / "test_input.txt"


class TestPrepare:
    """Test the prepare script."""

    def test_prepare_creates_correct_files(self, test_config: DatasetConfig):
        """Test that the prepare script creates the correct files."""
        prepare_dataset(test_config)
        assert test_config.processed_dir.exists()
        assert (test_config.processed_dir / "train.bin").exists()
        assert (test_config.processed_dir / "val.bin").exists()
        assert (test_config.processed_dir / "meta.json").exists()

    def test_prepare_exits_early_if_files_exist(self, test_config: DatasetConfig):
        """Test that the prepare script does not re-process existing files."""
        # First run
        prepare_dataset(test_config)

        # Get file modification times
        train_path = test_config.processed_dir / "train.bin"
        val_path = test_config.processed_dir / "val.bin"
        train_mtime_1 = train_path.stat().st_mtime
        val_mtime_1 = val_path.stat().st_mtime

        time.sleep(0.1)

        # Second run
        prepare_dataset(test_config)

        # Verify files weren't modified (same modification times)
        train_mtime_2 = train_path.stat().st_mtime
        val_mtime_2 = val_path.stat().st_mtime

        assert train_mtime_1 == train_mtime_2, "train.bin was modified"
        assert val_mtime_1 == val_mtime_2, "val.bin was modified"

    def test_train_val_split(self, test_config: DatasetConfig):
        """Test that the train/val split is correct."""
        prepare_dataset(test_config)
        assert test_config.processed_dir.exists()
        assert (test_config.processed_dir / "train.bin").exists()
        assert (test_config.processed_dir / "val.bin").exists()
        assert (test_config.processed_dir / "meta.json").exists()

        # load the train/val splits
        train_tokens = np.fromfile(test_config.processed_dir / "train.bin", dtype=np.uint16)
        val_tokens = np.fromfile(test_config.processed_dir / "val.bin", dtype=np.uint16)

        total_tokens = len(train_tokens) + len(val_tokens)
        # check that the train/val split is correct
        assert abs(len(train_tokens) / total_tokens - test_config.train_split) < 0.01

    def test_no_data_leakage(self, test_config: DatasetConfig):
        """Test that there is no data leakage between train and val splits."""
        prepare_dataset(test_config)
        assert test_config.processed_dir.exists()
        assert (test_config.processed_dir / "train.bin").exists()
        assert (test_config.processed_dir / "val.bin").exists()
        assert (test_config.processed_dir / "meta.json").exists()

        # load the train/val splits
        train_tokens = np.fromfile(test_config.processed_dir / "train.bin", dtype=np.uint16)
        val_tokens = np.fromfile(test_config.processed_dir / "val.bin", dtype=np.uint16)

        with open(test_config.processed_dir / "meta.json") as f:
            meta = json.load(f)

        assert len(train_tokens) + len(val_tokens) == meta["total_tokens"], (
            "Total tokens mismatch => Data leakage detected"
        )

    def test_metadata_accuracy(self, test_config: DatasetConfig):
        """Test that meta.json contains correct information."""
        prepare_dataset(test_config)

        with open(test_config.processed_dir / "meta.json") as f:
            meta = json.load(f)

        # Load actual data
        train_tokens = np.fromfile(test_config.processed_dir / "train.bin", dtype=np.uint16)
        val_tokens = np.fromfile(test_config.processed_dir / "val.bin", dtype=np.uint16)

        # Verify metadata matches reality
        assert meta["train_tokens"] == len(train_tokens)
        assert meta["val_tokens"] == len(val_tokens)
        assert meta["dataset_name"] == test_config.name
        assert meta["train_split"] == test_config.train_split
        assert "vocab_size" in meta
        assert "tokenizer_path" in meta
