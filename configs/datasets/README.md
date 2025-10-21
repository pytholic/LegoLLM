# Dataset Configurations

This directory contains YAML configuration files for different datasets used in the LegoLLM project.

Each `.yaml` file defines the parameters for processing a specific dataset, including:

- `name`: The name of the dataset.
- `raw_file`: Path to the raw input text file.
- `processed_dir`: Directory where processed data (tokenized, train/val splits) will be saved.
- `tokenizer_type`: The type of tokenizer to use (e.g., `regex_bpe`, `simple`).
- `vocab_size`: The target vocabulary size for the tokenizer.
- `train_split`: The proportion of data to use for the training set (e.g., 0.9 for 90%).
- `block_size`: The sequence length for chunking the data.
- `special_tokens`: (Optional) A dictionary of special tokens and their IDs.

These configurations are used by the `scripts/prepare.py` script to automate the dataset preparation process.
