"""Script to generate a small summary of input data (.txt) file.

Created by @pytholic on 2025.10.19
"""

import argparse
import logging
from pprint import pformat

from legollm.core.logging import logger
from legollm.utils import read_text_file

logger.setLevel(logging.INFO)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate a summary of input data file.")
    parser.add_argument("input_file", type=str, help="Path to input data file.")
    args = parser.parse_args()

    text = read_text_file(args.input_file)
    summary = generate_summary(text)
    logger.info(pformat(summary))


def generate_summary(text: str) -> dict[str, int | str]:
    """Generate a summary of the input text.

    We want mainly two things:
    1. Total number of characters in the input text.
    2. Size of the input text in MB.
    """
    return {
        "total_characters": len(text),
        "size_in_mb": f"{len(text) / 1024 / 1024:.2f} MB",
    }


if __name__ == "__main__":
    main()
