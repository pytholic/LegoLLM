"""Instruction tuning utilities - placeholder stub.

TODO: Implement instruction tuning.

Reference: Sebastian's LLMs-from-scratch/ch07/

Instruction tuning fine-tunes a pretrained LLM to follow instructions,
typically using prompt-response pairs.

Key components:
- InstructionDataset: Format and tokenize instruction data
- Collate function for batching variable-length sequences
- Training loop with appropriate loss masking
"""

import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    """Dataset for instruction tuning.

    TODO: Implement instruction dataset.

    Formats data as:
    - Input: instruction/prompt
    - Output: response

    With appropriate special tokens for the model being fine-tuned.

    See Sebastian's ch07 for reference implementation.
    """

    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: object,
        max_length: int = 1024,
    ) -> None:
        """Initialize instruction dataset.

        Args:
            data: List of dicts with "instruction" and "output" keys
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        raise NotImplementedError(
            "InstructionDataset not yet implemented. See tmp/LLMs-from-scratch/ch07/ for reference."
        )

    def __len__(self) -> int:
        """Return dataset length (not implemented)."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get item by index (not implemented)."""
        raise NotImplementedError


def instruction_collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Collate function for instruction tuning batches.

    TODO: Implement collate function.

    Handles:
    - Padding to max length in batch
    - Creating attention masks
    - Setting ignore_index for padding in labels

    See Sebastian's ch07 for reference.
    """
    raise NotImplementedError
