"""Tokenization module.

Created by @pytholic on 2025.09.14
"""

import re


class WhiteSpaceTokenizer:
    """White space tokenizer."""

    def tokenize(
        self, text: str, sep_punc: bool = True, strip_empty: bool = True
    ) -> list[str]:
        """Tokenize the text.

        Args:
            text: The text to tokenize.
            sep_punc: Whether to separate punctuation.
            strip_empty: Whether to strip empty tokens.

        Returns:
            A list of tokens.
        """
        tokens = re.findall(r"\w+|\s+|[^\w\s]" if sep_punc else r"\s+", text)
        if strip_empty:
            return [token for token in tokens if token.strip()]
        return tokens
