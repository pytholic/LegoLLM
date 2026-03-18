"""Split the instruction dataset into train, test and validation sets.

Created by @pytholic on 2026.03.15
"""

import json

from legollm.post_training.sft.instruction_dataset import download_and_load_instruction_dataset

DATASET_URL = "https://raw.githubusercontent.com/pytholic/LegoLLM/refs/heads/main/data/finetuning/instruction_dataset.json"
DATASET_PATH = "data/finetuning/instruction_dataset.json"


def split_dataset(data: list[dict[str, str]]) -> None:
    """Split the dataset into train, test and validation sets
    and write to files."""
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    with open("data/finetuning/instruction_dataset_train.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open("data/finetuning/instruction_dataset_test.json", "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    with open("data/finetuning/instruction_dataset_val.json", "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print("Split dataset into train, test and validation sets")
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Validation set: {len(val_data)} samples")


if __name__ == "__main__":
    data = download_and_load_instruction_dataset(DATASET_PATH, DATASET_URL)
    split_dataset(data)
