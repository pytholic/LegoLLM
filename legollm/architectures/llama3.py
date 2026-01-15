"""Llama 3 architecture - placeholder stub.

TODO: Implement Llama 3 architecture.

Reference: Sebastian's LLMs-from-scratch/pkg/llms_from_scratch/llama3.py

This file should contain:
- Configuration dicts (LLAMA32_CONFIG_1B, LLAMA32_CONFIG_3B, etc.)
- Llama3Model class
- TransformerBlock with RMSNorm (pre-norm)
- GroupedQueryAttention (GQA)
- SwiGLU FeedForward
- RoPE (Rotary Position Embedding)
- Llama3Tokenizer
- ChatFormat for instruction tuning
- load_weights_into_llama() for loading pretrained weights

Key differences from GPT-2:
- RMSNorm instead of LayerNorm
- RoPE instead of learned positional embeddings
- Grouped-Query Attention (GQA) instead of full MHA
- SwiGLU activation instead of GELU
- No bias in linear layers
"""

import torch
import torch.nn as nn

# =============================================================================
# Configuration (TODO: Fill in actual values)
# =============================================================================

LLAMA32_CONFIG_1B: dict[str, int | float | torch.dtype] = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "n_kv_groups": 8,  # For GQA
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    # "rope_freq": {...}  # RoPE frequency scaling config
}

LLAMA32_CONFIG_3B: dict[str, int | float | torch.dtype] = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 3072,
    "n_heads": 24,
    "n_layers": 28,
    "hidden_dim": 8192,
    "n_kv_groups": 8,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
}


# =============================================================================
# Model (TODO: Implement)
# =============================================================================


class Llama3Model(nn.Module):
    """Llama 3 language model.

    TODO: Implement the full Llama 3 architecture.

    Key components needed:
    - Token embeddings (no positional - RoPE is applied in attention)
    - TransformerBlock with GQA and SwiGLU
    - RMSNorm for final normalization
    - Output projection head

    See Sebastian's llama3.py for reference implementation.
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize Llama3Model."""
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError(
            "Llama3Model not yet implemented. "
            "See tmp/LLMs-from-scratch/pkg/llms_from_scratch/llama3.py for reference."
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass (not implemented)."""
        raise NotImplementedError


# =============================================================================
# Tokenizer (TODO: Implement)
# =============================================================================


class Llama3Tokenizer:
    """Llama 3 tokenizer wrapper.

    TODO: Implement tokenizer that wraps tiktoken with Llama-3 special tokens.

    Required:
    - Load BPE merges from model file
    - Handle special tokens (<|begin_of_text|>, <|end_of_text|>, etc.)
    - encode() and decode() methods
    """

    def __init__(self, model_path: str) -> None:
        """Initialize Llama3Tokenizer."""
        raise NotImplementedError(
            "Llama3Tokenizer not yet implemented. "
            "See tmp/LLMs-from-scratch/pkg/llms_from_scratch/llama3.py for reference."
        )

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """Encode text to token IDs (not implemented)."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text (not implemented)."""
        raise NotImplementedError


class ChatFormat:
    """Chat format for instruction-tuned Llama 3 models.

    TODO: Implement chat template formatting.

    Formats messages with special tokens:
    - <|start_header_id|>role<|end_header_id|>
    - <|eot_id|> for end of turn
    """

    def __init__(
        self, tokenizer: Llama3Tokenizer, default_system: str = "You are a helpful assistant."
    ) -> None:
        """Initialize ChatFormat."""
        raise NotImplementedError

    def encode(self, user_message: str, system_message: str | None = None) -> list[int]:
        """Encode chat messages (not implemented)."""
        raise NotImplementedError


# =============================================================================
# Weight Loading (TODO: Implement)
# =============================================================================


def load_weights_into_llama(model: Llama3Model, params: dict[str, torch.Tensor]) -> None:
    """Load pretrained weights into Llama 3 model.

    TODO: Implement weight loading from HuggingFace checkpoints.

    See Sebastian's llama3.py for reference implementation.
    """
    raise NotImplementedError
