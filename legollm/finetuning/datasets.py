"""Fine-tuning dataset utilities - placeholder stub.

TODO: Implement dataset utilities.

Common datasets for instruction tuning:
- Alpaca: 52k instruction-following examples
- Dolly: 15k human-generated instruction examples
- LIMA: 1k high-quality examples
- Custom datasets in various formats
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_alpaca_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load Alpaca-format dataset.

    TODO: Implement dataset loading.

    Alpaca format:
        {
            "instruction": "...",
            "input": "...",  # optional
            "output": "..."
        }

    Returns:
        List of instruction-output pairs
    """
    raise NotImplementedError


def load_jsonl_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load JSONL format dataset.

    TODO: Implement dataset loading.

    Each line is a JSON object with instruction/output.
    """
    raise NotImplementedError


class PreferenceDataset(Dataset):
    """Dataset for preference/RLHF training.

    TODO: Implement preference dataset.

    Format:
        {
            "prompt": "...",
            "chosen": "...",   # preferred response
            "rejected": "..."  # non-preferred response
        }

    Used for DPO, RLHF, and similar preference-based training.
    """

    def __init__(
        self, data: list[dict[str, str]], tokenizer: object, max_length: int = 1024
    ) -> None:
        """Initialize PreferenceDataset."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return dataset length (not implemented)."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get item by index (not implemented)."""
        raise NotImplementedError
