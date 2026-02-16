"""Text generation utilities for generative models.

Created by @pytholic on 2025.11.14
Moved to generation/ on 2026.01.15
"""

from collections.abc import Generator
from enum import StrEnum

import torch
import torch.nn.functional as F

from legollm.core.interfaces import Tokenizer


class SamplingStrategy(StrEnum):
    """Sampling strategy to use for text generation.

    GREEDY: Pick the most likely token i.e. highest probability.
    STOCHASTIC: Sample from the distribution using temperature and top-k/top-p filtering.
    """

    GREEDY = "greedy"
    STOCHASTIC = "stochastic"


@torch.inference_mode()
def generate_text(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    strategy: SamplingStrategy = SamplingStrategy.GREEDY,
    eos_token_id: int | None = None,
    stream: bool = False,
) -> torch.Tensor | Generator[int]:
    """Generate text tokens autoregressively.

    Uses a pre-allocated buffer to avoid repeated tensor allocations.

    Args:
        model: The model to generate text from.
        token_ids: Starting context tokens (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens for sampling
        top_p: Nucleus sampling threshold (cumulative probability)
        strategy: Sampling strategy to use
        eos_token_id: Stop generation if this token is generated
        stream: If True, yields token IDs one at a time instead of returning full tensor.

    Returns:
        If stream=False: Generated token IDs including context (batch_size, seq_len + new_tokens)
        If stream=True: Generator yielding individual token IDs.
    """
    if stream:
        return _generate_stream(
            model, token_ids, max_new_tokens, temperature, top_k, top_p, strategy, eos_token_id
        )

    model.eval()
    seq_len = token_ids.size(1)
    context_length = _get_context_length(model, seq_len)

    # Pre-allocate buffer for all generated tokens
    buffer = torch.zeros(
        (token_ids.size(0), seq_len + max_new_tokens),
        dtype=torch.long,
        device=token_ids.device,
    )

    buffer[:, :seq_len] = token_ids
    total_len = seq_len

    for _ in range(max_new_tokens):
        start = max(0, total_len - context_length)
        token_ids_context = buffer[:, start:total_len]

        logits = model(token_ids_context)
        logits = logits[:, -1, :]

        next_token = _sample_next_token(logits, strategy, temperature, top_k, top_p)

        buffer[:, total_len] = next_token.squeeze(-1)
        total_len += 1

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return buffer[:, :total_len]


@torch.inference_mode()
def _generate_stream(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    strategy: SamplingStrategy,
    eos_token_id: int | None,
) -> Generator[int]:
    """Internal streaming generator. Called by generate_text(stream=True)."""
    model.eval()
    seq_len = token_ids.size(1)
    context_length = _get_context_length(model, seq_len)

    buffer = torch.zeros(
        (token_ids.size(0), seq_len + max_new_tokens),
        dtype=torch.long,
        device=token_ids.device,
    )
    buffer[:, :seq_len] = token_ids
    total_len = seq_len

    for _ in range(max_new_tokens):
        start = max(0, total_len - context_length)
        token_ids_context = buffer[:, start:total_len]

        logits = model(token_ids_context)
        logits = logits[:, -1, :]

        next_token = _sample_next_token(logits, strategy, temperature, top_k, top_p)
        token_id: int = int(next_token.item())

        buffer[:, total_len] = token_id
        total_len += 1

        yield token_id

        if eos_token_id is not None and token_id == eos_token_id:
            break


def _get_context_length(model: torch.nn.Module, fallback: int) -> int:
    """Get context length from model config."""
    if hasattr(model, "context_length"):
        return model.context_length
    if hasattr(model, "config"):
        cfg = model.config
        return cfg["context_length"] if isinstance(cfg, dict) else cfg.context_length
    return fallback


def _sample_next_token(
    logits: torch.Tensor,
    strategy: SamplingStrategy,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Sample next token from logits using the given strategy."""
    if strategy == SamplingStrategy.STOCHASTIC:
        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None:
            logits = apply_top_k_filtering(logits, top_k)
        if top_p is not None:
            logits = apply_top_p_filtering(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    elif strategy == SamplingStrategy.GREEDY:
        return torch.argmax(logits, dim=-1, keepdim=True)
    else:
        raise ValueError(f"Invalid sampling strategy: {strategy}")


def apply_top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter logits to keep only top-k tokens.

    Args:
        logits: Logits tensor (batch, vocab_size)
        top_k: Number of top tokens to keep

    Returns:
        Filtered logits with non-top-k tokens set to -inf
    """
    # Get top-k values and indices
    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))

    # Get threshold (k-th largest value)
    threshold = top_k_values[:, [-1]]

    # Mask out values below threshold
    logits[logits < threshold] = -float("inf")

    return logits


def apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Filter logits to keep only top-p tokens (nucleus sampling).

    Args:
        logits: Logits tensor (batch, vocab_size)
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with non-top-p tokens set to -inf

    Implementation details:
        We want to keep at least one token that gets us to the threshold.

    Example:
    - Logits: [8.0, 5.0, 3.0, 2.0, 1.0]
    - top_p: 0.7

    Position | Logit | Prob  | CumSum | > 0.7? | After Shift | Action
    ---------|-------|-------|--------|--------|-------------|--------
    0     | 8.0   | 94.3% | 94.3%  |  True  |   False     | KEEP
    1     | 5.0   |  4.7% | 99.0%  |  True  |   True      | REMOVE
    2     | 3.0   |  0.6% | 99.6%  |  True  |   True      | REMOVE
    3     | 2.0   |  0.2% | 99.9%  |  True  |   True      | REMOVE
    4     | 1.0   |  0.1% | 100.0% |  True  |   True      | REMOVE

    References:
    - https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
    """
    filter_value = -float("inf")

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    # Shift right to keep first token above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Scatter the removal mask back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove
    )

    # Apply mask
    logits[indices_to_remove] = filter_value

    return logits


def generate_and_decode(
    prompt: str,
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    strategy: SamplingStrategy = SamplingStrategy.GREEDY,
    stream: bool = False,
) -> str | Generator[str]:
    """High-level function: encode prompt, generate, decode.

    Args:
        prompt: The prompt to generate text from.
        tokenizer: The tokenizer to use.
        model: The model to generate text from.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature to use for stochastic sampling.
        top_k: The number of top tokens to keep for stochastic sampling.
        top_p: The cumulative probability threshold for stochastic sampling.
        strategy: The sampling strategy to use.
        stream: If True, yields decoded tokens one at a time.

    Returns:
        If stream=False: The full generated text as a string.
        If stream=True: Generator yielding decoded token strings.
    """
    # Encode prompt
    token_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
    ).unsqueeze(0)  # add batch dimension

    # Move prompt to model's device
    token_ids = token_ids.to(model.device)

    if stream:
        return _stream_and_decode(
            tokenizer,
            model,
            token_ids,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            strategy,
        )

    # Generate all at once
    generated_token_ids = generate_text(
        model=model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        strategy=strategy,
    )

    # Decode
    return tokenizer.decode(generated_token_ids[0].tolist())


def _stream_and_decode(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    strategy: SamplingStrategy,
) -> Generator[str]:
    """Internal streaming decoder. Called by generate_and_decode(stream=True)."""
    token_stream: Generator[int] = generate_text(  # type: ignore[assignment]
        model=model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        strategy=strategy,
        stream=True,
    )
    for token_id in token_stream:
        yield tokenizer.decode([token_id])


if __name__ == "__main__":
    import torch

    from legollm.architectures import GPT2, GPT2_CONFIG_124M
    from legollm.config import TOKENIZERS_DIR
    from legollm.core.tokenization import RegexBPETokenizer

    torch.manual_seed(42)

    model = GPT2(GPT2_CONFIG_124M)
    tokenizer = RegexBPETokenizer()

    trained_tokenizer = TOKENIZERS_DIR / "tiny_shakespeare_regex_bpe.json"
    tokenizer.load(str(trained_tokenizer))

    generated_text = generate_and_decode(
        prompt="I am a software engineer and I love to code",
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=10,
        temperature=10.0,
        top_k=20,
        top_p=0.99,
        strategy=SamplingStrategy.STOCHASTIC,
    )
    print(generated_text)
