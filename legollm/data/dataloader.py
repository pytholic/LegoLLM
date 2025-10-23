"""DataLoader class and config for training.

NOTE: Only sequential reading from .bin text files is supported currently.

Created by @pytholic on 2025.10.23"""

from collections import deque
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from legollm.core.exceptions import DataLoaderError
from legollm.utils import get_dtype_for_vocab, load_dataset_metadata

MIN_BUFFER = 10_000  # 20 KB minimum
MAX_BUFFER = 10_000_000  # 20 MB maximum
DEFAULT_BUFFER_BATCHES = 500


@dataclass
class DataLoaderConfig:
    """Configuration for the DataLoader."""

    dataset_path: Path
    block_size: int
    batch_size: int
    device: str = "cpu"
    token_buffer_size: int | None = None
    dtype: np.dtype | None = None
    split: str = "train"  # "train" or "val" or custom name

    def __post_init__(self) -> None:
        """Post-init validation and calculation."""
        self.token_buffer_size = self.token_buffer_size or self._calculate_buffer_size()

        # Add check for token_buffer_size >= min required
        # else we will get stuff in infinite loop in _refill_buffer()
        min_required = self.batch_size * (self.block_size + 1)
        if self.token_buffer_size < min_required:
            raise DataLoaderError(
                f"token_buffer_size ({self.token_buffer_size}) must be >= "
                f"batch_size * (block_size + 1) = {min_required}"
            )

        # Validate
        if self.block_size <= 0:
            raise DataLoaderError(f"block_size must be positive, got {self.block_size}")
        if self.batch_size <= 0:
            raise DataLoaderError(f"batch_size must be positive, got {self.batch_size}")
        if not self.dataset_path.exists():
            raise DataLoaderError(f"Dataset path not found: {self.dataset_path}")

    def _calculate_buffer_size(self) -> int:
        """Auto-calculate buffer size based on batch size and block size."""
        tokens_per_batch = self.batch_size * (self.block_size + 1)
        buffer_size = DEFAULT_BUFFER_BATCHES * tokens_per_batch

        # clamp to reasonable limits
        return int(max(MIN_BUFFER, min(buffer_size, MAX_BUFFER)))


class DataLoader:
    """DataLoader for training.

    Responsibilities:
        1. Load .bin file via memmap
        2. Manage token buffer
        3. Create and return batches
    """

    def __init__(self, config: DataLoaderConfig) -> None:
        """Initialize the DataLoader."""
        self.config = config
        data_file_path = config.dataset_path / f"{config.split}.bin"
        metadata = load_dataset_metadata(config.dataset_path)
        self.vocab_size = int(metadata["vocab_size"])
        self.dtype = config.dtype or get_dtype_for_vocab(self.vocab_size)

        # Load data
        self.data = np.memmap(data_file_path, dtype=self.dtype, mode="r")
        self.token_buffer: deque[int] = deque(maxlen=config.token_buffer_size)
        self.current_pos = 0

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of training data."""
        needed_tokens = self.config.batch_size * (self.config.block_size + 1)

        # Refill buffer if needed
        while len(self.token_buffer) < needed_tokens:
            self._refill_buffer()

        # Pop tokens from buffer
        tokens: list[int] = [self.token_buffer.popleft() for _ in range(needed_tokens)]
        tokens_tensor = torch.tensor(tokens, dtype=torch.int64)

        # Reshape to (batch_size, block_size + 1)
        tokens_tensor = tokens_tensor.view(self.config.batch_size, self.config.block_size + 1)

        # Split into input and target tensors
        x = tokens_tensor[:, :-1].to(self.config.device)
        y = tokens_tensor[:, 1:].to(self.config.device)
        return x, y

    def _refill_buffer(self) -> None:
        """Refill buffer with sequential reads from memmap."""
        chunk_size = self.token_buffer.maxlen
        end_pos = min(self.current_pos + chunk_size, len(self.data))

        new_tokens = self.data[self.current_pos : end_pos].astype(np.int64)
        self.token_buffer.extend(new_tokens)

        self.current_pos = end_pos

        # Loop back to start if we reach the end
        if self.current_pos >= len(self.data):
            self.current_pos = 0

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Make DataLoader iterable - yields batches infinitely."""
        while True:
            yield self.get_batch()


if __name__ == "__main__":
    num_batches = 3

    config = DataLoaderConfig(
        dataset_path=Path("data/processed/tiny_shakespeare"),
        block_size=4,
        batch_size=num_batches,
        device="cpu",
        token_buffer_size=10000,
        dtype=np.uint16,
    )

    dl = DataLoader(config)

    for i in range(num_batches):
        print("-" * 70)
        print(f"Iterating {i}th batch")
        print("-" * 70)
        batch = next(iter(dl))
        print(f"Input: {batch[0].numpy().tolist()}")
        print(f"Target:     {batch[1].numpy().tolist()}")
        print("-" * 70)
        print()
