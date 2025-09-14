"""Main entry point for the application."""

import urllib.request

from legollm import logger
from legollm.core.preprocessor import SimpleTextPreprocessor
from legollm.core.tokenizer import WhiteSpaceTokenizer


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


def read_file(filepath: str) -> str:
    """Read a file."""
    with open(filepath, encoding="utf-8") as file:
        return file.read()


def main() -> None:
    """Main function."""
    raw_test = read_file("data/the-verdict.txt")
    tokenizer = WhiteSpaceTokenizer()
    preprocessor = SimpleTextPreprocessor()
    vocabulary = preprocessor.process_text(raw_test, tokenizer)
    # print first 10 dataclass items    
    print(vocabulary)


if __name__ == "__main__":
    main()
