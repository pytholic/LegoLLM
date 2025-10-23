# Scripts

This directory contains executable Python scripts for data management and preprocessing in the LegoLLM project.

## Available Commands

After installing the package (`uv sync` or `pip install -e .`), the following commands are available globally:

### `data-prepare`

Prepares raw text data for training by tokenizing and splitting into train/validation sets.

**Usage:**

```shell
data-prepare --config configs/datasets/tiny_shakespeare.yaml [--verbose]
```

**Arguments:**

- `--config`: Path to the dataset configuration YAML file (required)
- `--verbose`: Enable verbose output (optional)

**What it does:**

1. Loads or trains a tokenizer based on the configuration
2. Tokenizes the raw text
3. Splits into train/validation sets
4. Saves binary files (`train.bin`, `val.bin`) and metadata (`meta.json`)

### `download-data`

Downloads raw datasets from their sources.

**Usage:**

```shell
data-download <dataset_name> <output_dir>
```

**Arguments:**

- `dataset_name`: Name of the dataset (`tiny_shakespeare`, `the_verdict`)
- `output_dir`: Directory where the downloaded file will be saved

**Example:**

```shell
data-download tiny_shakespeare ./data/raw/tiny_shakespeare/
```

### `data-summary`

Provides a summary of a raw text file (character count, file size).

**Usage:**

```shell
data-summary <input_file>
```

**Arguments:**

- `input_file`: Path to the text file to analyze

**Example:**

```shell
data-summary ./data/raw/tiny_shakespeare/tiny_shakespeare.txt
```

**Output:**

```
[2025-10-19 16:37:10] INFO     {'size_in_mb': '1.06 MB', 'total_characters': 1115390}
```

## Development

These scripts are defined as entry points in `pyproject.toml` under `[project.scripts]`. When the package is installed, Python automatically creates executable wrappers for these commands in the virtual environment's `bin/` directory.
