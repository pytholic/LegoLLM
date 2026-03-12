"""Instruction dataset for supervised fine-tuning.

Created by @pytholic on 2026.02.24
"""

import json
import os

import requests
import torch
from torch.utils.data import Dataset

from legollm.core.interfaces import Tokenizer


def download_and_load_instruction_dataset(file_path: str, url: str) -> list[dict[str, str]]:
    """Download and load the instruction dataset from a URL.

    If the file does not exist, it will be downloaded from the URL and
    saved to the file path.

    Returns:
        list[dict[str, str]]: The instruction dataset.

    Note:
        - Handles both `json` and `jsonl` files.
    """
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, encoding="utf-8") as file:
        if file_path.endswith(".jsonl"):
            data = [json.loads(line) for line in file if line.strip()]
        else:
            data = json.load(file)

    return data


class InstructionDataset(Dataset):
    """Instruction dataset.

    Args:
        data: List of dicts with "instruction", "input", and "output" keys.
    """

    def __init__(self, data: list[dict[str, str]], tokenizer: Tokenizer) -> None:
        """Initialize the instruction dataset.

        Args:
            data: List of dicts with "instruction", "input", and "output" keys.
            tokenizer: The tokenizer to use.
        """
        self.data = data

        # Pre-tokenize the data
        self.encoded_data: list[list[int]] = []

        for sample in self.data:
            formatted_input: str = format_input(sample)
            response_text: str = f"\n\n### Response:\n{sample['output']}"
            full_sample = formatted_input + response_text
            self.encoded_data.append(tokenizer.encode(full_sample))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> list[int]:
        """Return the item at the given index."""
        return self.encoded_data[idx]


def format_input(sample: dict[str, str]) -> str:
    """Format an instruction sample into Alpaca-style prompt (without response)."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{sample['instruction']}"
    )

    input_text = f"\n\n### Input:\n{sample['input']}" if sample["input"] else ""

    return instruction_text + input_text


def custom_collate_fn(
    batch: list[list[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int = 1024,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for the instruction dataset.

    Args:
        batch: List of lists of integers.
        pad_token_id: The token ID to use for padding.
        ignore_index: The index to use for ignored tokens.
        allowed_max_length: The maximum length of the input.
        device: The device to use.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The input and target tensors.

    Note:
        - By default, PyTorch has the cross_entropy(..., ignore_index=-100) setting
        to ignore examples corresponding to the label -100
    """
    # Truncate samples that exceed allowed_max_length (reserve 1 token for EOS)
    batch = [sample[: allowed_max_length - 1] for sample in batch]

    # Find the longest sequence in the batch (+1 for the EOS token we append)
    batch_max_length = min(max(len(x) + 1 for x in batch), allowed_max_length)

    inputs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    for sample in batch:
        new_sample = sample.copy()
        # Add an <|endoftext|> token
        new_sample.append(pad_token_id)
        # Pad sequences to batch_max_length
        padded_sample = new_sample + [pad_token_id] * (batch_max_length - len(new_sample))
        inputs = torch.tensor(padded_sample[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded_sample[1:])  # Shift +1 to the right for targets

        # Mask all padding tokens in targets with ignore_index,
        # but keep the first pad token (EOS) so the model learns to predict end of sequence
        mask = targets == pad_token_id
        eos_indices = mask.nonzero(as_tuple=False)
        if eos_indices.numel() > 1:
            targets[eos_indices[1:]] = ignore_index

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs = torch.stack(inputs_list).to(device)
    targets = torch.stack(targets_list).to(device)
    return inputs, targets
