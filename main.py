"""Main entry point for the application."""

import logging

import tiktoken

from data.utils import read_file
from legollm.core.logging import logger
from legollm.core.tokenization import RegexBPETokenizer

logger.setLevel(logging.DEBUG)


def main() -> None:
    """Main function."""
    text = read_file("data/the-verdict/raw/the-verdict.txt")
    tokenizer = RegexBPETokenizer()

    tokenizer.train(text, vocab_size=276, verbose=True, special_tokens={"<|endoftext|>": 100257})
    tokenizer.save("data/bpe_tokenizer.json")
    tokenizer.save_readable("data/bpe_tokenizer_readable.txt")
    tokenizer.load("data/bpe_tokenizer.json")
    text = "안녕하세요 오늘은 날씨가 좋네안 😀 (Hello, today is a good day 😀)"
    logger.debug(tokenizer.decode(tokenizer.encode(text)) == text)
    tiktoken_tokenizer = tiktoken_encode_decode(text)
    logger.debug(tiktoken_tokenizer)
    logger.debug(tokenizer.decode(tokenizer.encode(text)))


def tiktoken_encode_decode(text: str) -> None:
    """Encode and decode text using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    # logger.debug(enc.decode(enc.encode(text)) == text)
    return enc.decode(enc.encode(text, allowed_special={"<|endoftext|>"}))


if __name__ == "__main__":
    main()
