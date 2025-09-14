"""Main entry point for the application."""

import urllib.request
from pathlib import Path

from src.logging import logger


def download_file_from_url(url: str, filepath: str) -> None:
    """Download data from a URL.

    Args:
        url: The URL to download data from.
        filepath: The path to save the downloaded file.
    """
    # encoding does not seem to be working
    logger.info(f"Downloading data from {url}...")
    urllib.request.urlretrieve(url, filepath)
    logger.info("Data downloaded successfully.")


def main() -> None:
    """Main function."""
    filepath = Path("data/the-verdict.txt")
    if not Path(filepath).exists():
        download_file_from_url(
            url="https://raw.githubusercontent.com/pytholic/LegoLLM/main/data/the-verdict.txt",
            filepath=str(filepath),
        )
    else:
        logger.info(f"File {filepath} already exists.")

    with open(filepath, encoding="utf-8") as file:
        data = file.read()
        logger.info(f"Length of Data: {len(data)}")


if __name__ == "__main__":
    main()
