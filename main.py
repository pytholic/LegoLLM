"""Main entry point for the application."""

import logging
from collections import Counter, deque
from itertools import pairwise

from legollm.core.logging import logger
from legollm.core.utils import read_file

logger.setLevel(logging.DEBUG)


def main() -> None:
    """Main function."""
    text = read_file("data/blog.txt")
    token_ids = list(text.encode("utf-8"))

    curr_vocab_size = 256
    new_vocab_size = 276
    num_merges = new_vocab_size - curr_vocab_size
    # copy token ids so that we do not destroy the original
    ids = token_ids.copy()
    merges = {}
    for i in range(num_merges):
        word_freq = compute_pair_freq(ids)
        pair = find_most_freq_pair(word_freq)
        idx = curr_vocab_size + i
        logger.debug(f"Merging pair {pair} with index {idx}")
        ids = merge_pair(ids, pair, idx)
        merges[pair] = idx

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    # test encode
    text = r"""The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). """
    # print(decode(encode(text, merges), vocab) == text)
    logger.debug(decode(encode(text, merges), vocab) == text)


# Step 1: We need to compute frequency of each consecutive pair in the text
def compute_pair_freq(token_ids: list[int]) -> list[tuple[tuple[int, int], int]] | None:
    """Compute frequency of each consecutive pair in the text."""
    pair_counts = Counter(pairwise(token_ids))
    if not pair_counts:
        return None

    return pair_counts.most_common()


def find_most_freq_pair(pair_freq: list[tuple[tuple[int, int], int]]) -> tuple[int, int] | None:
    """Find the pair ID from the list of pair frequencies."""
    return pair_freq[0][0] if pair_freq else None


def merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Merge the given pair ID with a new token ID."""
    dq = deque(token_ids)
    replaced_token_ids: list[int] = []

    while dq:
        current = dq.popleft()  # remove from front
        # Check if current + next item form the pair
        if dq and (current, dq[0]) == pair:
            replaced_token_ids.append(new_id)
            dq.popleft()  # remove second element of the pair
        else:
            replaced_token_ids.append(current)

    return replaced_token_ids


def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decode token IDs back to text."""
    token_bytes = b"".join(vocab[idx] for idx in token_ids)
    text = token_bytes.decode("utf-8", errors="replace")
    return text


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Encode text into token IDs using learned merges.

    Apply merges in the order they were learned (by merge index).
    """
    token_ids = list(text.encode("utf-8"))

    while len(token_ids) >= 2:  # we need at least two tokens to merge
        pair_freq = compute_pair_freq(token_ids)

        # Find the pair with the lowest merge index (earliest learned)
        pair = min(pair_freq, key=lambda p: merges.get(p, float("inf")))

        # If this pair wasn't learned during training, stop
        if pair not in merges:
            break

        # Replace the pair with the new token ID
        idx = merges[pair]
        token_ids = merge_pair(token_ids, pair, idx)

    return token_ids


if __name__ == "__main__":
    main()
