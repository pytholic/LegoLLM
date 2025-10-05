"""Main entry point for the application."""

import logging
from pprint import pformat

import tiktoken

from legollm.core.logging import logger
from legollm.core.tokenization.tokenizer import NaiveBPETokenizer
from legollm.core.utils import read_file

logger.setLevel(logging.DEBUG)


def main() -> None:
    """Main function."""
    text = read_file("data/blog.txt")
    tokenizer = NaiveBPETokenizer()

    # let's train the tokenizer
    tokenizer.train(text, vocab_size=276, verbose=True)
    tokenizer.save("data/bpe_tokenizer.json")
    tokenizer.load("data/bpe_tokenizer.json")
    logger.info(pformat(f"Vocab: {tokenizer.vocab}"))
    logger.info(pformat(f"Merges: {tokenizer.merges}"))
    logger.debug(tokenizer.decode(tokenizer.encode(text)) == text)


def tiktoken_encode_decode(text: str) -> None:
    """Encode and decode text using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    logger.debug(enc.decode(enc.encode(text)) == text)


if __name__ == "__main__":
    # tiktoken_encode_decode("안녕하세요 오늘은 날씨가 좋네안 😀 (Hello, today is a good day 😀)")
    main()
