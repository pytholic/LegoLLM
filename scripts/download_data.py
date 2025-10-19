"""Script to download raw data from the web.

Created by @pytholic on 2025.10.19
"""

import argparse
import logging
from pathlib import Path

from legollm.core.logging import logger
from legollm.core.utils import download_file_from_url

logger.setLevel(logging.INFO)

DATA_URLS = {
    "tiny_shakespeare": "https://raw.githubusercontent.com/pytholic/LegoLLM/refs/heads/feat/data/data/raw/tiny_shakespeare/tiny_shakespeare.txt",
    "the-verdict": "https://raw.githubusercontent.com/pytholic/LegoLLM/refs/heads/feat/data/data/raw/the-verdict/the-verdict.txt",
}


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Download raw data from the web.")
    parser.add_argument("data", type=str, help="['tiny_shakespeare', 'the-verdict']")
    parser.add_argument("output_dir", type=str, help="Path to save the downloaded file.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data not in DATA_URLS:
        logger.error(f"Invalid data: {args.data}")
        raise ValueError(f"Invalid data: {args.data}")

    url = DATA_URLS[args.data]
    file_name = Path(url).name

    logger.info(f"Downloading file from {url} to {output_dir / file_name}.")
    download_file_from_url(url, output_dir / file_name)
    logger.info(f"Downloaded file {file_name} to {output_dir}.")


if __name__ == "__main__":
    main()
