"""Text generation utilities for generative models.

Created by @pytholic on 2025.11.14
"""

# first let's test if typing literal disallows rest of the values

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

        if strategy == SamplingStrategy.STOCHASTIC:
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
) -> str:
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

    Returns:
        The generated text.
    """
    # Encode prompt
    token_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
    ).unsqueeze(0)  # add batch dimension

    # Move prompt to model's device
    token_ids = token_ids.to(model.device)

    # Generate
    generated_token_ids = generate_text(
        model=model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        strategy=strategy,
        eos_token_id=None,
    )

    # Decode
    generated_text = tokenizer.decode(generated_token_ids[0].tolist())
    return generated_text


if __name__ == "__main__":
    import torch

    from legollm.config import TOKENIZERS_DIR
    from legollm.core.tokenization import RegexBPETokenizer
    from legollm.models.architectures.gpt import GPT, GPT2Config125M
    from legollm.models.generate import generate_text

    torch.manual_seed(42)

    model = GPT(GPT2Config125M())
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
