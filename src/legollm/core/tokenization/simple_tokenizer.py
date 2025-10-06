"""Simple word-level tokenizer.

Created by @pytholic on 2025.09.14
"""

import regex as rex

from legollm.core.exceptions import TokenizerError
from legollm.core.tokenization.vocabulary import UNK_TOKEN, build_vocab_from_tokens

PUNCTUATION = frozenset(r".,!?;:-()[]{}\"")


class SimpleTokenizer:
    """Word-level tokenizer with regex-based splitting.

    This tokenizer splits text into words and punctuation using regex patterns.
    It requires a pre-built vocabulary for encoding/decoding operations.

    The tokenizer implements both the Tokenizer and PreTokenizable protocols,
    making it suitable for word-level tokenization tasks.
    """

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        """Initialize the tokenizer with vocabulary mappings.

        Args:
            vocab: Dictionary mapping tokens (strings) to their integer IDs.
                If None, the tokenizer can only be used for tokenization,
                not encoding/decoding.
        """
        self.vocab = vocab
        self.str_to_int = vocab if vocab else {}
        self.int_to_str = {v: k for k, v in vocab.items()} if vocab else {}

    def tokenize(self, text: str) -> list[str]:
        """Split text into tokens (pre-tokenization step).

        This is pure tokenization logic that doesn't require vocabulary.
        It's used to prepare text for vocabulary building or analysis.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of string tokens.

        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokenizer.tokenize("Hello, world!")
            ["Hello", ",", "world", "!"]
        """
        pattern = r"\w+(?:'\w+)*|[^\w\s]"
        tokens = rex.findall(pattern, text)  # pyright: ignore
        return [token for token in tokens if token.strip()]

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs.

        Tokenizes the text and maps each token to its vocabulary ID.
        Unknown tokens are mapped to the UNK token ID.

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.

        Raises:
            TokenizerError: If vocabulary is not provided or missing UNK token.

        Example:
            >>> vocab = {"Hello": 0, "world": 1, "<|UNK|>": 2}
            >>> tokenizer = SimpleTokenizer(vocab)
            >>> tokenizer.encode("Hello world")
            [0, 1]
        """
        if not self.vocab:
            raise TokenizerError(
                "Cannot encode without vocabulary. "
                "Initialize with vocab or use from_corpus() classmethod."
            )

        tokens = self.tokenize(text)
        unk_id = self.str_to_int.get(UNK_TOKEN)
        if unk_id is None:
            raise TokenizerError(
                f"Vocabulary must contain {UNK_TOKEN} for unknown token handling. "
                f"Use build_vocab_from_tokens() to create a proper vocabulary."
            )
        return [self.str_to_int.get(token, unk_id) for token in tokens]

    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back to text.

        Args:
            tokens: A list of integer token IDs.

        Returns:
            The decoded text string with proper spacing around punctuation.

        Example:
            >>> vocab = {"Hello": 0, "world": 1, "!": 2}
            >>> tokenizer = SimpleTokenizer(vocab)
            >>> tokenizer.decode([0, 1, 2])
            "Hello world!"
        """
        if not tokens:
            return ""

        text_tokens = [self.int_to_str[token] for token in tokens]

        text = ""
        for token in text_tokens:
            if token in PUNCTUATION:
                text += token
            else:
                text += " " + token
        return text.strip()

    @classmethod
    def from_corpus(
        cls,
        corpus: str | list[str],
        min_frequency: int = 1,
        special_tokens: list[str] | None = None,
    ) -> "SimpleTokenizer":
        """Create tokenizer with vocabulary built from corpus.

        Convenience method that combines tokenization and vocabulary building.
        This is the recommended way to create a SimpleTokenizer from scratch.

        Args:
            corpus: Text or list of texts to build vocabulary from.
            min_frequency: Minimum token frequency to include in vocabulary.
                Tokens appearing fewer times are excluded. Default is 1.
            special_tokens: Additional special tokens to include.
                Defaults to [UNK_TOKEN, END_OF_TEXT_TOKEN].

        Returns:
            Initialized tokenizer with vocabulary built from corpus.

        Example:
            >>> corpus = ["Hello world!", "Hello there!"]
            >>> tokenizer = SimpleTokenizer.from_corpus(corpus)
            >>> tokenizer.encode("Hello world!")
            [2, 4, 0]

            >>> # With frequency filtering
            >>> tokenizer = SimpleTokenizer.from_corpus(corpus, min_frequency=2)
            >>> # Only "Hello" appears twice, other tokens filtered out
        """
        temp_tokenizer = cls()
        if isinstance(corpus, str):
            corpus = [corpus]

        all_tokens: list[str] = []
        for text in corpus:
            all_tokens.extend(temp_tokenizer.tokenize(text))

        vocab = build_vocab_from_tokens(
            all_tokens, special_tokens=special_tokens, min_frequency=min_frequency
        )

        return cls(vocab)
