"""Text generation utilities for generative models.

Created by @pytholic on 2025.11.14
"""

# first let's test if typing literal disallows rest of the values

from enum import StrEnum

import torch
import torch.nn.functional as F


class SamplingStrategy(StrEnum):
    """Sampling strategy to use for text generation."""

    GREEDY = "greedy"
    SAMPLE = "sample"


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
) -> torch.Tensor:
    """Generate text tokens autoregressively.

    Args:
        model: The model to generate text from.
        token_ids: Starting context tokens (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens for sampling
        top_p: Nucleus sampling threshold (cumulative probability)
        strategy: Sampling strategy to use
        eos_token_id: Stop generation if this token is generated

    Returns:
        Generated token IDs including context (batch_size, seq_len + new_tokens)
    """
    model.eval()
    rng = None

    for _ in range(max_new_tokens):
        context_length = model.config.context_length
        # Token IDs grow so we need to limit it to the most recent tokens within the context length
        token_ids_context = token_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(token_ids_context)  # (batch, seq, vocab)
            logits = logits[:, -1, :]  # get las position (batch, vocab)

        if strategy == SamplingStrategy.SAMPLE:
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                logits = apply_top_k_filtering(logits, top_k)
            if top_p is not None:
                logits = apply_top_p_filtering(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=rng)

        elif strategy == SamplingStrategy.GREEDY:
            # Greedy: just pick argmax (no filters needed)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        else:
            raise ValueError(f"Invalid sampling strategy: {strategy}")

        # Append next token to context
        token_ids = torch.cat([token_ids, next_token], dim=1)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return token_ids


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
    """Filter logits to keep only top-p tokens.

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
    0     | 8.0   | 94.3% | 94.3%  |  True  |   False     | ✅ KEEP
    1     | 5.0   |  4.7% | 99.0%  |  True  |   True      | ❌ REMOVE (cumsum > 0.7)
    2     | 3.0   |  0.6% | 99.6%  |  True  |   True      | ❌ REMOVE (cumsum > 0.7)
    3     | 2.0   |  0.2% | 99.9%  |  True  |   True      | ❌ REMOVE (cumsum > 0.7)
    4     | 1.0   |  0.1% | 100.0% |  True  |   True      | ❌ REMOVE (cumsum > 0.7)
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


if __name__ == "__main__":
    # let's test our generate_text function
    import torch

    from legollm.models.architectures.gpt import GPT, GPT2Config125M
    from legollm.models.generate import generate_text

    torch.manual_seed(42)

    model = GPT(GPT2Config125M())
    token_ids = torch.randint(0, 10, (1, 5))
    generated_token_ids = generate_text(
        model,
        token_ids,
        max_new_tokens=5,
        temperature=5.0,
        top_k=5,
        top_p=0.9,
        strategy=SamplingStrategy.SAMPLE,
    )
    print(generated_token_ids)
