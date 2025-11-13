"""Utility functions for the project.

Created by @pytholic on 2025.10.19
"""

import json
from pathlib import Path

import numpy as np
import torch.nn as nn


def read_text_file(filepath: str | Path) -> str:
    """Read a text file and return the content as a string."""
    filepath = Path(filepath)
    with open(filepath, encoding="utf-8") as file:
        return file.read()


def get_dtype_for_vocab(vocab_size: int) -> type:
    """Return appropriate numpy dtype based on vocabulary size.

    Args:
        vocab_size: Number of tokens in vocabulary

    Returns:
        numpy dtype (uint8, uint16, or uint32)

    Examples:
        >>> get_dtype_for_vocab(256)
        numpy.uint8
        >>> get_dtype_for_vocab(50257)  # GPT-2
        numpy.uint16
        >>> get_dtype_for_vocab(100000)  # Large vocab
        numpy.uint32
    """
    if vocab_size <= 256:
        return np.uint8
    elif vocab_size <= 65535:
        return np.uint16
    elif vocab_size <= 4_294_967_295:
        return np.uint32
    else:
        raise ValueError(f"Vocabulary size {vocab_size} too large for uint32")


def load_dataset_metadata(data_path: Path) -> dict[str, str | int | float]:
    """Load metadata for a tokenized dataset."""
    metadata_path = data_path / "meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path.as_posix()}")
    with open(metadata_path) as f:
        return json.load(f)


def count_model_params(model: nn.Module) -> float:
    """Count the number of parameters in the PyTorch neural network model in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6  # in millions
