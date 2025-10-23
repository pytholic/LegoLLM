"""Test the DataLoader class.

Created by @pytholic on 2025.10.23
"""

from collections import deque
from pathlib import Path

import numpy as np
import pytest
import regex as rex
import torch
from pytest_mock import MockerFixture

from legollm.core.exceptions import DataLoaderError
from legollm.data.dataloader import DataLoader, DataLoaderConfig


def _create_dataset(dir_path: Path, tokens: np.ndarray, vocab_size: int = 2048) -> Path:
    """Create a minimal dataset directory with train.bin and meta.json."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "train.bin").write_bytes(tokens.tobytes())
    (dir_path / "meta.json").write_text(f'{{"vocab_size": {vocab_size}}}')
    return dir_path


class TestDataLoaderConfig:
    """Test DataLoaderConfig validation and initialization."""

    @pytest.mark.parametrize("block_size", [0, -1])
    def test_config_validation_block_size(self, tmp_path: Path, block_size: int):
        """Reject invalid block_size values."""
        ds_dir = _create_dataset(tmp_path / "ds", np.arange(100, dtype=np.uint16))
        with pytest.raises(DataLoaderError):
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=block_size,
                batch_size=2,
                split="train",
            )

    @pytest.mark.parametrize(
        "batch_size",
        [
            pytest.param(0, id="zero_batch_size"),
            pytest.param(-1, id="negative_batch_size"),
        ],
    )
    def test_config_validation_batch_size(self, tmp_path: Path, batch_size: int):
        """Reject invalid batch_size values."""
        ds_dir = _create_dataset(tmp_path / "ds", np.arange(100, dtype=np.uint16))
        with pytest.raises(DataLoaderError, match=f"batch_size must be positive, got {batch_size}"):
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=4,
                batch_size=batch_size,
                split="train",
            )

    def test_config_validation_dataset_path(self):
        """Raise error when dataset path does not exist."""
        with pytest.raises(DataLoaderError):
            DataLoaderConfig(
                dataset_path=Path("nonexistent"),
                block_size=4,
                batch_size=2,
                split="train",
            )

    @pytest.mark.parametrize(
        "batch_size, block_size, expected_buffer_size",
        [
            (2, 4, 10000),
            (3, 5, 10000),
            (4, 6, 14000),
            (5, 7, 20000),
        ],
    )
    def test_config_auto_buffer_calculation(
        self, tmp_path: Path, batch_size: int, block_size: int, expected_buffer_size: int
    ):
        """Auto-calculate buffer_size when None."""
        ds_dir = _create_dataset(tmp_path / "ds", np.arange(1000, dtype=np.uint16))
        config = DataLoaderConfig(
            dataset_path=ds_dir,
            batch_size=batch_size,
            block_size=block_size,
            token_buffer_size=None,
            split="train",
        )
        assert config.token_buffer_size == expected_buffer_size

    def test_config_token_buffer_min_size_validation(self, tmp_path: Path):
        """Reject token_buffer_size less than min required."""
        ds_dir = _create_dataset(tmp_path / "ds", np.arange(50, dtype=np.uint16))
        block_size = 4
        batch_size = 2
        token_buffer_size = 5
        min_required = batch_size * (block_size + 1)
        expected_message = f"token_buffer_size ({token_buffer_size}) must be >= batch_size * (block_size + 1) = {min_required}"

        with pytest.raises(
            DataLoaderError,
            match=rex.escape(expected_message),
        ):
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=block_size,
                batch_size=batch_size,
                token_buffer_size=token_buffer_size,
                split="train",
            )


class TestDataLoader:
    """Test DataLoader initialization and core functionality."""

    def test_dataloader_init(self, tmp_path: Path):
        """Verify correct initialization of DataLoader attributes."""
        tokens = np.arange(1000, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "init_ds", tokens, vocab_size=256)
        config = DataLoaderConfig(
            dataset_path=ds_dir,
            block_size=4,
            batch_size=2,
            device="cpu",
            token_buffer_size=500,
            dtype=np.dtype(np.uint16),
            split="train",
        )
        dl = DataLoader(config)
        assert all(
            [
                dl.config == config,
                dl.vocab_size == 256,
                dl.dtype == np.uint16,
                dl.data.shape == (1000,),
                dl.data.dtype == np.uint16,
                dl.token_buffer == deque(maxlen=500),
                dl.current_pos == 0,
            ]
        )

    @pytest.mark.parametrize(
        "batch_size, block_size",
        [
            pytest.param(1, 4, id="batch_size_1_block_size_4"),
            pytest.param(2, 8, id="batch_size_2_block_size_8"),
        ],
    )
    def test_get_batch_shapes(self, tmp_path: Path, batch_size: int, block_size: int):
        """Verify x and y have expected shapes (batch_size, block_size)."""
        needed = batch_size * (block_size + 1)
        tokens = np.arange(needed * 3, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "shapes_ds", tokens)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=block_size,
                batch_size=batch_size,
                device="cpu",
                token_buffer_size=max(needed, 64),
                dtype=None,
                split="train",
            )
        )
        x, y = dl.get_batch()
        assert x.shape == (batch_size, block_size)
        assert y.shape == (batch_size, block_size)

    def test_target_is_shifted_input(self, tmp_path: Path):
        """Verify y is x shifted by one token (autoregressive target)."""
        batch_size, block_size = 2, 6
        needed = batch_size * (block_size + 1)
        tokens = np.arange(needed * 2, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "shift_ds", tokens)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=block_size,
                batch_size=batch_size,
                device="cpu",
                token_buffer_size=needed * 2,
                dtype=None,
                split="train",
            )
        )
        x, y = dl.get_batch()
        assert (y[:, :-1] == x[:, 1:]).all()

    def test_correct_dtype_loading(self, tmp_path: Path):
        """Derive dtype from metadata vocab_size when not explicitly provided."""
        tokens = np.arange(100, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "dtype_ds", tokens, vocab_size=2048)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=4,
                batch_size=2,
                device="cpu",
                token_buffer_size=64,
                dtype=None,
                split="train",
            )
        )
        assert dl.dtype == np.uint16
        assert dl.data.dtype == np.uint16

    @pytest.mark.parametrize(
        "device_param",
        [
            pytest.param("cpu", id="cpu"),
            pytest.param(
                "mps",
                id="mps",
                marks=pytest.mark.skipif(
                    not torch.backends.mps.is_available(), reason="MPS not available"
                ),
            ),
            pytest.param(
                "cuda",
                id="cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_device_placement(self, tmp_path: Path, device_param: str):
        """Place tensors on the requested device (CPU/MPS/CUDA)."""
        tokens = np.arange(100, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "device_ds", tokens)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=4,
                batch_size=2,
                device=device_param,
                token_buffer_size=64,
                dtype=None,
                split="train",
            )
        )
        x, y = dl.get_batch()
        assert x.device.type == device_param
        assert y.device.type == device_param


class TestDataLoaderBufferManagement:
    """Test buffer refill logic and sequential reading."""

    def test_buffer_refills_when_empty(self, tmp_path: Path, mocker: MockerFixture):
        """Trigger refill when buffer lacks enough tokens for a batch."""
        # Use minimal tokens to speed up test
        tokens = np.arange(20, dtype=np.uint16)  # Reduced from 100 to 20
        ds_dir = _create_dataset(tmp_path / "refill_ds", tokens)

        config = DataLoaderConfig(
            dataset_path=ds_dir,
            block_size=4,
            batch_size=2,
            token_buffer_size=12,  # set to a value > needed_tokens (10) to avoid validation error
            split="train",
        )
        dl = DataLoader(config)

        # Spy setup
        spy = mocker.spy(dl, "_refill_buffer")

        # First batch triggers refill (empty buffer)
        dl.get_batch()
        assert spy.call_count >= 1

        # Drain buffer to force another refill
        spy.reset_mock()
        dl.get_batch()  # Uses remaining buffer
        dl.get_batch()  # Should trigger refill
        assert spy.call_count >= 1

    def test_no_unnecessary_refills(self, tmp_path: Path, mocker: MockerFixture):
        """Skip refill when buffer already has enough tokens."""
        tokens = np.arange(100, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "no_refill_ds", tokens)

        config = DataLoaderConfig(
            dataset_path=ds_dir,
            block_size=4,
            batch_size=2,
            token_buffer_size=64,  # Big enough
            split="train",
        )
        dl = DataLoader(config)

        spy = mocker.spy(dl, "_refill_buffer")

        # Pre-fill buffer
        dl._refill_buffer()
        assert spy.call_count == 1  # Pre-fill counted

        # Get batch - should NOT trigger additional refill
        dl.get_batch()
        assert spy.call_count == 1  # Still 1 (no new calls)

    def test_sequential_reading(self, tmp_path: Path):
        """Read tokens sequentially from memmap (0, 1, 2, ...)."""
        seq = np.arange(100, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "seq_ds", seq)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=4,
                batch_size=1,
                device="cpu",
                token_buffer_size=16,
                dtype=None,
                split="train",
            )
        )
        x, y = dl.get_batch()
        assert x.shape == (1, 4)
        assert y.shape == (1, 4)
        assert x[0].tolist() == [0, 1, 2, 3]
        assert y[0].tolist() == [1, 2, 3, 4]

    def test_loops_back_at_end(self, tmp_path: Path):
        """Wrap current_pos to 0 when reaching EOF during refill."""
        data = np.arange(6, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "loop_ds", data)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=2,
                batch_size=1,
                device="cpu",
                token_buffer_size=10,  # > len(data) ensures wrap
                dtype=None,
                split="train",
            )
        )
        dl._refill_buffer()
        assert dl.current_pos == 0


class TestDataLoaderIterator:
    """Test iterator protocol and infinite yielding."""

    def test_iter_yields_batches(self, tmp_path: Path):
        """Verify __iter__ yields batches with correct shapes."""
        tokens = np.arange(100, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "iter_ds", tokens)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=3,
                batch_size=2,
                device="cpu",
                token_buffer_size=64,
                dtype=None,
                split="train",
            )
        )
        it = iter(dl)
        x, y = next(it)
        assert x.shape == (2, 3)
        assert y.shape == (2, 3)

    def test_iter_infinite(self, tmp_path: Path):
        """Verify iterator never raises StopIteration."""
        tokens = np.arange(100, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "iter_inf_ds", tokens)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=3,
                batch_size=2,
                device="cpu",
                token_buffer_size=64,
                dtype=None,
                split="train",
            )
        )
        it = iter(dl)
        for _ in range(3):
            x, y = next(it)
            assert x.shape == (2, 3)
            assert y.shape == (2, 3)


class TestDataLoaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_batch_dataset(self, tmp_path: Path):
        """Handle dataset with exactly one batch worth of tokens."""
        batch_size, block_size = 1, 4
        needed = batch_size * (block_size + 1)
        data = np.arange(needed, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "single_batch_ds", data)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=block_size,
                batch_size=batch_size,
                device="cpu",
                token_buffer_size=needed,
                dtype=None,
                split="train",
            )
        )
        x, y = dl.get_batch()
        assert x.shape == (batch_size, block_size)
        assert y.shape == (batch_size, block_size)

    def test_buffer_larger_than_data(self, tmp_path: Path):
        """Handle buffer size larger than dataset length."""
        data = np.arange(10, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "large_buffer_ds", data)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=4,
                batch_size=1,
                device="cpu",
                token_buffer_size=100,
                dtype=None,
                split="train",
            )
        )
        x, y = dl.get_batch()
        assert x.shape == (1, 4)
        assert y.shape == (1, 4)

    def test_exact_multiple_of_batch(self, tmp_path: Path):
        """Handle data length that is an exact multiple of batch requirement."""
        batch_size, block_size = 2, 3
        needed = batch_size * (block_size + 1)
        data = np.arange(2 * needed, dtype=np.uint16)
        ds_dir = _create_dataset(tmp_path / "exact_multiple_ds", data)
        dl = DataLoader(
            DataLoaderConfig(
                dataset_path=ds_dir,
                block_size=block_size,
                batch_size=batch_size,
                device="cpu",
                token_buffer_size=needed,
                dtype=None,
                split="train",
            )
        )
        for _ in range(2):
            x, y = dl.get_batch()
            assert x.shape == (batch_size, block_size)
            assert y.shape == (batch_size, block_size)
