# Data Management

This directory is responsible for managing all data-related aspects of the LegoLLM project, including raw datasets, processed training/validation splits, and trained tokenizer models.

## Directory Structure

- `raw/`: Contains original, unprocessed dataset files.
- `processed/`: Stores tokenized and split (train/validation) datasets, ready for model training. Each dataset has its own subdirectory with a `meta.json` file summarizing its processing.
- `tokenizers/`: Holds trained tokenizer models (e.g., BPE models) in JSON format.

## Available Commands

After installing the package, the following commands are available:

- `download-data`: Downloads raw datasets from their sources.
- `dataset-summary`: Provides a summary of raw text files (e.g., character count, size).
- `prepare`: The main script for processing raw data. It handles tokenization, splitting into train/validation sets, and saving metadata.

## Usage

### Download Data

To download raw datasets:

```shell
download-data the_verdict ./data/raw/the_verdict/
download-data tiny_shakespeare ./data/raw/tiny_shakespeare/
```

### Prepare Dataset

To tokenize and split a dataset using a configuration (configurations are in `configs/datasets/`), run following in project's root:

```shell
prepare --config configs/datasets/tiny_shakespeare.yaml
```

### Print Raw Data Summary

To print a summary of a raw text file:

```shell
dataset-summary ./data/raw/tiny_shakespeare/tiny_shakespeare.txt
```

The output will look like:

```
[2025-10-19 16:37:10] INFO     {'size_in_mb': '1.06 MB', 'total_characters': 1115390}
```
