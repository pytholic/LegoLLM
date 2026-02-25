"""Utility functions for the project.

Created by @pytholic on 2025.10.19
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from legollm.logging import logger


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


def load_pretrained_state_dict(
    model_path: Path,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> torch.Tensor:
    """Load a pretrained model's state dictionary from a checkpoint.

    Args:
        model_path: Path to the checkpoint file.
        map_location: Device to map the checkpoint to.
        weights_only: Whether to load only the weights of the model.

    Returns:
        The state dictionary of the pretrained model.
    """
    checkpoint = torch.load(model_path, map_location=map_location, weights_only=weights_only)
    return checkpoint["model_state_dict"]


def log_and_raise_exception(
    err_msg: str, exception_type: type[Exception], from_exc: Exception | None = None
) -> None:
    """Utility function to log an error and raise an exception, optionally preserving the chain.

    Args:
        err_msg : str
            Error message for logging and the new exception.
        exception_type : type[Exception]
            The type of exception to raise.
        from_exc : Exception | None
            The original exception to chain from. Defaults to None.

    Raises:
        An exception of the given type.
    """
    logger.error(err_msg)
    raise exception_type(err_msg) from from_exc


def get_device() -> str:
    """Get the device to use for the model.

    Returns:
        The device to use for the model.
    """
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
