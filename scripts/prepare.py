"""Prepare the text data for training.

Created by @pytholic on 2025.10.19
"""

import argparse
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import yaml

from legollm.core.interfaces import TrainableTokenizer
from legollm.core.tokenization.bpe.regex_bpe_tokenizer import RegexBPETokenizer
from legollm.logging import logger
from legollm.utils import get_dtype_for_vocab, read_text_file


class TokenizerType(StrEnum):
    """Type of tokenizer to use."""

    REGEX_BPE = "regex_bpe"
    SIMPLE = "simple_tokenizer"


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    raw_file: Path
    processed_dir: Path
    tokenizer_type: str
    vocab_size: int
    train_split: float = 0.9
    block_size: int = 256  # number of tokens in a block for training
    special_tokens: dict[str, int] | None = None


def load_config(yaml_path: str) -> DatasetConfig:
    """Load a dataset configuration from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    data["raw_file"] = Path(data["raw_file"])
    data["processed_dir"] = Path(data["processed_dir"])
    return DatasetConfig(**data)


def create_tokenizer(tokenizer_type: str) -> RegexBPETokenizer:
    """Factory function to create a tokenizer instance based on type.

    Args:
        tokenizer_type: Type of tokenizer to create.

    Note:
        - Only RegexBPE tokenizer is supported at the moment for simplicity.
    """
    if tokenizer_type == TokenizerType.REGEX_BPE:
        tokenizer = RegexBPETokenizer(pattern=RegexBPETokenizer.GPT4_SPLIT_PATTERN)
        return tokenizer
    raise ValueError(f"Tokenizer type {tokenizer_type} is not supported at the moment.")


def prepare_dataset(config: DatasetConfig, verbose: bool = False) -> None:
    """Prepare a single dataset: tokenize and save as binary files.

    Args:
        config: Configuration for the dataset.
        verbose: Whether to print verbose output.

    Steps:
        1. Read raw text file
        2. Train or load tokenizer
        3. Encode text into token IDs
        4. Split into train/val
        5. Save as .bin files + metadata

    Implementation details:
        - Using Bin files for efficiency as they provide:
            - Fast I/O operations
            - Efficient memory usage
    """
    logger.info(f"📦 Preparing dataset: {config.name}")
    logger.info(f"   Raw file: {config.raw_file}")
    logger.info(f"   Tokenizer: {config.tokenizer_type} (vocab_size={config.vocab_size})")

    # Step 1: Read raw text file
    logger.info("Reading raw text...")
    text = read_text_file(config.raw_file)
    logger.info(f"   Total characters: {len(text):,}")

    # Step 2: Train or load tokenizer
    tokenizer_dir = Path("data/tokenizers")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_dir / f"{config.name}_{config.tokenizer_type}.json"

    # let's add tqdm to the progress bar for the tokenizer training
    # also let's use richbar for tqdm progress bar
    if tokenizer_path.exists():
        logger.info(f"   Loading existing tokenizer from {tokenizer_path}")
        tokenizer = load_tokenizer(config.tokenizer_type, tokenizer_path)
    else:
        logger.info(f"Training new {config.tokenizer_type} tokenizer...")
        tokenizer = train_tokenizer(
            tokenizer_type=config.tokenizer_type,
            text=text,
            vocab_size=config.vocab_size,
            special_tokens=config.special_tokens,
            verbose=verbose,
        )
        tokenizer.save(tokenizer_path.as_posix())
        logger.info(f"   Saved {config.tokenizer_type} tokenizer to {tokenizer_path.as_posix()}")

        # save human readable tokenizer
        readable_tokenizer_path = tokenizer_path.with_suffix(".txt")
        tokenizer.save_readable(readable_tokenizer_path.as_posix())
        logger.info(
            f"   Saved human readable {config.tokenizer_type} tokenizer to {readable_tokenizer_path}"
        )

    # Step 3: Encode text into token IDs
    # if splits are already present, we should avoid further processing
    if not config.processed_dir.exists():
        config.processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = config.processed_dir / "train.bin"
    val_path = config.processed_dir / "val.bin"
    if train_path.exists() and val_path.exists():
        logger.info(
            f"   Train/val splits already exist at {train_path.as_posix()} and {val_path.as_posix()}"
        )
        return

    logger.info("Encoding text into token IDs...")
    token_ids = tokenizer.encode(text)
    logger.info(f"   Total tokens: {len(token_ids):,}")
    logger.info(f"   Characters per token: {len(text) / len(token_ids):.2f} chars/token")

    # Step 4: Split into train/val
    dtype = get_dtype_for_vocab(config.vocab_size)
    split_idx = int(len(token_ids) * config.train_split)
    train_tokens = np.array(token_ids[:split_idx], dtype=dtype)
    val_tokens = np.array(token_ids[split_idx:], dtype=dtype)

    logger.info(
        f"Split: {len(train_tokens):,} train tokens, {len(val_tokens):,} validation tokens (dtype={dtype})"
    )

    # Step 5: Save as .bin files
    logger.info("Saving train/val dataset files...")
    train_path = config.processed_dir / "train.bin"
    val_path = config.processed_dir / "val.bin"
    train_tokens.tofile(train_path.as_posix())
    val_tokens.tofile(val_path.as_posix())
    logger.info(f"Saved files to {train_path.as_posix()} and {val_path.as_posix()}")

    # Step 6: Save metadata
    logger.info("Saving metadata...")
    meta = {
        "dataset_name": config.name,
        "raw_file": config.raw_file.as_posix(),
        "processed_dir": config.processed_dir.as_posix(),
        "total_characters": len(text),
        "vocab_size": len(tokenizer.vocab),
        "tokenizer_type": config.tokenizer_type,
        "tokenizer_path": str(tokenizer_path),
        "total_tokens": len(token_ids),
        "characters_per_token": len(text) / len(token_ids),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_split": config.train_split,
        "block_size": config.block_size,
    }
    meta_path = config.processed_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, indent=2, fp=f)
    logger.info(f"Saved metadata to {meta_path.as_posix()}")

    logger.info("✅ Dataset prepared successfully!")
    logger.info(f"   Processed data saved to: {config.processed_dir}")


def train_tokenizer(
    tokenizer_type: str,
    text: str,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
    verbose: bool = False,
) -> TrainableTokenizer:
    """Train a new tokenizer on the given text.

    Programs to the TrainableTokenizer interface.

    Args:
        tokenizer_type: Type of tokenizer to create and train.
        text: Training corpus.
        vocab_size: Target vocabulary size (may be None for non-BPE tokenizers).
        special_tokens: Special tokens to include.
        verbose: Whether to print verbose output.

    Returns:
        A trained tokenizer implementing TrainableTokenizer protocol.
    """
    try:
        tokenizer = create_tokenizer(tokenizer_type)
        tokenizer.train(text, vocab_size, special_tokens=special_tokens, verbose=verbose)
        return tokenizer
    except Exception as e:
        logger.error(f"Error training {tokenizer_type}: {e}")
        raise


def load_tokenizer(tokenizer_type: str, tokenizer_path: Path) -> TrainableTokenizer:
    """Load a tokenizer from a file.

    Programs to the TrainableTokenizer interface.

    Args:
        tokenizer_type: Type of tokenizer to load.
        tokenizer_path: Path to the saved tokenizer file.

    Returns:
        A loaded tokenizer implementing TrainableTokenizer protocol.
    """
    try:
        tokenizer = create_tokenizer(tokenizer_type)
        tokenizer.load(tokenizer_path.as_posix())
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading {tokenizer_type} from {tokenizer_path.as_posix()}: {e}")
        raise


def main() -> None:
    """Main entry point for the prepare script."""
    parser = argparse.ArgumentParser(description="Prepare the text data for training.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the dataset configuration YAML file."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        prepare_dataset(config, verbose=args.verbose)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
