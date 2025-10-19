import argparse
import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legollm.core.logging import progress_bar


def find_text_files(path: str | Path) -> list[Path]:
    """Find all .txt files in a given path (file or directory)."""
    path = Path(path)

    if path.is_file():
        return [path]
    elif path.is_dir():
        return sorted(path.rglob("*.txt"))
    else:
        return []


def stream_documents(text_files: list[Path], min_length: int = 10) -> Iterator[str]:
    """Stream documents from text files without loading everything into memory.
    Yields one document at a time.
    """
    with progress_bar("Processing files", total=len(text_files)) as (progress, task):
        for file_path in text_files:
            content = file_path.read_text(encoding="utf-8", errors="replace")

            # Split on double newlines for document boundaries
            documents = re.split(r"\n\s*\n", content)

            for doc in documents:
                doc = doc.strip()
                if len(doc) >= min_length:
                    yield doc

            progress.update(task, advance=1)


def create_shards(
    input_path: str | Path,
    output_dir: str | Path,
    num_shards: int,
    shuffle: bool = True,
    seed: int = 42,
    min_doc_length: int = 10,
) -> None:
    """Reads text data, splits it into documents, optionally shuffles them,
    and writes them into Parquet shards with each document as a separate row.

    Args:
        input_path: Path to input text file or directory
        output_dir: Directory where Parquet shards will be saved
        num_shards: Number of shards to create
        shuffle: Whether to shuffle documents
        seed: Random seed for reproducibility
        min_doc_length: Minimum document length in characters
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print("Starting sharding process...")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Shards: {num_shards}, Shuffle: {shuffle}, Seed: {seed}")

    # Find text files
    text_files = find_text_files(input_path)
    if not text_files:
        print(f"Error: No .txt files found at '{input_path}'")
        return

    print(f"Found {len(text_files)} text file(s)")

    # Stream and collect documents
    documents = list(stream_documents(text_files, min_doc_length))

    if not documents:
        print("Error: No valid documents found")
        return

    print(f"Collected {len(documents):,} documents")

    # Calculate statistics
    doc_lengths = [len(doc) for doc in documents]
    print(
        f"Doc length - Min: {min(doc_lengths):,}, Max: {max(doc_lengths):,}, "
        f"Mean: {sum(doc_lengths) // len(doc_lengths):,}"
    )

    # Shuffle with seed for reproducibility
    if shuffle:
        print(f"Shuffling with seed {seed}...")
        random.seed(seed)
        random.shuffle(documents)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {output_dir}")

    # Partition documents into shards
    total_docs = len(documents)
    docs_per_shard = (total_docs + num_shards - 1) // num_shards

    print(f"Writing ~{docs_per_shard:,} docs per shard...")

    shard_num = 0
    num_iterations = (total_docs + docs_per_shard - 1) // docs_per_shard

    with progress_bar("Writing shards", total=num_iterations) as (progress, task):
        for i in range(0, total_docs, docs_per_shard):
            shard_docs = documents[i : i + docs_per_shard]

            # Store each document as a separate row (better for DataLoaders)
            df = pl.DataFrame(
                {
                    "text": shard_docs,
                    "length": [len(doc) for doc in shard_docs],
                }
            )

            output_path = output_dir / f"shard_{shard_num:04d}.parquet"
            df.write_parquet(output_path)

            shard_num += 1
            progress.update(task, advance=1)

    print(f"\n{'=' * 50}")
    print(f"✓ Created {shard_num} shards in '{output_dir}'")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert text data into shuffled Parquet shards for LLM training."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input text file or directory containing .txt files",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where Parquet shards will be saved",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=5,
        help="Number of shards to create (default: 100)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable document shuffling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--min-doc-length",
        type=int,
        default=10,
        help="Minimum document length in characters (default: 10)",
    )

    args = parser.parse_args()

    create_shards(
        input_path=args.input_path,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        min_doc_length=args.min_doc_length,
    )
