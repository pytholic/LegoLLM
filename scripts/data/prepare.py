"""Prepare the text data for training.

This involves
- Tokenizing the text
- Splitting the text into train/val splits
- Saving the data as binary files
- Saving the metadata

Created by @pytholic on 2025.10.19
"""

import argparse
import json
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import regex as rex
import yaml

from legollm.core.exceptions import TokenizerError
from legollm.core.interfaces import DocumentSplitter, TrainableTokenizer
from legollm.core.tokenization.bpe.regex_bpe_tokenizer import RegexBPETokenizer
from legollm.logging import logger
from legollm.utils import get_dtype_for_vocab, read_text_file


class TokenizerType(StrEnum):
    """Type of tokenizer to use."""

    REGEX_BPE = "regex_bpe"
    SIMPLE = "simple_tokenizer"


class NoSplitter(DocumentSplitter):
    """No split strategy."""

    def split(self, text: str) -> list[str]:
        """Split text into documents."""
        return [text]


class DoubleNewlineSplitter(DocumentSplitter):
    """Double newline split strategy."""

    def split(self, text: str) -> list[str]:
        """Split text into documents (remove leading and trailing whitespace)."""
        return [doc.strip() for doc in rex.split(r"\n{2,}", text) if doc.strip()]


class RegexSplitter(DocumentSplitter):
    """Split using custom regex pattern."""

    def __init__(self, pattern: str) -> None:
        """Initialize the RegexSplitter."""
        if not pattern:
            raise ValueError("Regex pattern cannot be empty")
        self.pattern = pattern
        self.compiled_pattern = rex.compile(pattern)

    def split(self, text: str) -> list[str]:
        """Split text into documents (remove leading and trailing whitespace)."""
        return [doc.strip() for doc in rex.split(self.pattern, text) if doc.strip()]


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

    # Document handling
    document_splitter: str = "none"  # one of: "none", "double_newline", "regex"
    document_regex: str | None = None  # required when document_splitter == "regex"
    shuffle_documents: bool = False
    shuffle_seed: int | None = 42
    min_document_chars: int = 50  # drop docs shorter than this
    join_separator: str = "\n\n"  # used to join documents after optional shuffling


class DocumentSplitMethod(StrEnum):
    """Supported document split methods."""

    NONE = "none"
    DOUBLE_NEWLINE = "double_newline"
    REGEX = "regex"


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


def get_splitter(config: DatasetConfig) -> DocumentSplitter:
    """Factory function to create appropriate splitter strategy."""
    try:
        method = DocumentSplitMethod(config.document_splitter)
    except ValueError as e:
        raise ValueError(
            f"Unsupported document_splitter: {config.document_splitter}. "
            f"Use one of: {[m.value for m in DocumentSplitMethod]}"
        ) from e

    if method == DocumentSplitMethod.NONE:
        return NoSplitter()
    elif method == DocumentSplitMethod.DOUBLE_NEWLINE:
        return DoubleNewlineSplitter()
    elif method == DocumentSplitMethod.REGEX:
        if not config.document_regex:
            raise ValueError(
                f"document_regex must be provided when document_splitter={method.value}"
            )
        return RegexSplitter(config.document_regex)
    else:
        raise ValueError(f"Unknown split method: {method}")


def split_documents(config: DatasetConfig, text: str) -> list[str]:
    """Split raw text into documents based on configuration.

    Implementation details:
    - When splitting is disabled ("none"), the entire text is treated as a single document.
    - For "double_newline", consecutive blank lines delimit documents.
    - For "regex", the provided pattern is used with re.split() as the delimiter.

    Args:
        config: Dataset configuration.
        text: Raw text to split.

    Returns:
        List of documents.
    """
    # Get splitter and split documents
    splitter = get_splitter(config)
    docs = splitter.split(text)

    # Filter by min_document_chars if specified
    if config.min_document_chars > 0:
        docs = [doc for doc in docs if len(doc) >= config.min_document_chars]

    # Optional deterministic shuffle
    if config.shuffle_documents and len(docs) > 1:
        rng = random.Random(config.shuffle_seed)
        rng.shuffle(docs)

    return docs


def prepare_dataset(
    config: DatasetConfig, verbose: bool = False, tokenizer_dir: Path | None = None
) -> None:
    """Prepare a single dataset: tokenize and save as binary files.

    Args:
        config: Configuration for the dataset.
        verbose: Whether to print verbose output.
        tokenizer_dir: Directory to save the tokenizer.

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
    tokenizer_dir = tokenizer_dir or Path("data/tokenizers")
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
    # If splits are already present, we should avoid further processing
    if not config.processed_dir.exists():
        config.processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = config.processed_dir / "train.bin"
    val_path = config.processed_dir / "val.bin"
    if train_path.exists() and val_path.exists():
        logger.info(
            f"   Train/val splits already exist at {train_path.as_posix()} and {val_path.as_posix()}"
        )
        return

    # Optionally split and shuffle documents
    if config.document_splitter != DocumentSplitMethod.NONE:
        docs = split_documents(config, text)
        processed_text = config.join_separator.join(docs)
        num_documents = len(docs)
        if len(docs) > 1:
            logger.info(
                f"   Documents: {len(docs):,} | splitter={config.document_splitter} "
                f"| shuffled={config.shuffle_documents} seed={config.shuffle_seed}"
            )
    else:
        processed_text = text
        num_documents = 1

    logger.info("Encoding text into token IDs...")
    token_ids = tokenizer.encode(processed_text)
    logger.info(f"   Total tokens: {len(token_ids):,}")
    if len(token_ids) == 0:
        raise TokenizerError(
            f"Tokenization produced zero tokens. Check document_splitter={config.document_splitter}, "
            f"min_document_chars={config.min_document_chars} settings."
        )
    logger.info(f"   Characters per token: {len(processed_text) / len(token_ids):.2f} chars/token")

    # Step 4: Split into train/val
    dtype = get_dtype_for_vocab(config.vocab_size)
    split_idx = int(len(token_ids) * config.train_split)
    train_tokens = np.array(token_ids[:split_idx], dtype=dtype)
    val_tokens = np.array(token_ids[split_idx:], dtype=dtype)

    logger.info(
        f"Split: {len(train_tokens):,} train tokens, {len(val_tokens):,} validation tokens (dtype={dtype.__name__})"
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
        "total_characters": len(processed_text),
        "vocab_size": len(tokenizer.vocab),
        "tokenizer_type": config.tokenizer_type,
        "tokenizer_path": str(tokenizer_path),
        "total_tokens": len(token_ids),
        "characters_per_token": len(processed_text) / len(token_ids),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_split": config.train_split,
        "block_size": config.block_size,
        "data_dtype": dtype.__name__,
        "num_documents": num_documents,
        "document_splitter": config.document_splitter,
        "document_regex": config.document_regex,
        "shuffle_documents": config.shuffle_documents,
        "shuffle_seed": config.shuffle_seed,
        "min_document_chars": config.min_document_chars,
        "join_separator": config.join_separator,
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
