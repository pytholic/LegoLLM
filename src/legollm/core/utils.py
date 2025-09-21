import urllib.request

from legollm import logger


def download_file_from_url(url: str, filepath: str) -> None:
    """Download data from a URL.

    Args:
        url: The URL to download data from.
        filepath: The path to save the downloaded file.
    """
    logger.info(f"Downloading data from {url}...")
    urllib.request.urlretrieve(url, filepath)
    logger.info("Data downloaded successfully.")


def read_file(filepath: str) -> str:
    """Read a file."""
    with open(filepath, encoding="utf-8") as file:
        return file.read()


def build_vocabulary(tokens: list[str]) -> dict[str, int]:
    """Build a vocabulary from a list of tokens."""
    return {token: i for i, token in enumerate(tokens)}


def remove_duplicates(tokens: list[str]) -> list[str]:
    """Remove duplicates from a list of tokens."""
    return list(set(tokens))
