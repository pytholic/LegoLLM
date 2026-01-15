"""Qwen 3 architecture - placeholder stub.

TODO: Implement Qwen 3 architecture.

Reference: Sebastian's LLMs-from-scratch/pkg/llms_from_scratch/qwen3.py

This file should contain:
- Configuration dicts (QWEN_CONFIG_06_B, QWEN3_CONFIG_1_7B, etc.)
- Qwen3Model class
- TransformerBlock with RMSNorm
- GroupedQueryAttention with optional QK normalization
- SwiGLU FeedForward
- MoE (Mixture of Experts) FeedForward for MoE variants
- RoPE implementation
- Qwen3Tokenizer
- load_weights_into_qwen() for loading pretrained weights

Key features of Qwen 3:
- Similar to Llama but with QK normalization option
- MoE variants available (e.g., QWEN3_CONFIG_30B_A3B)
- Different tokenizer (uses tokenizers library)
"""

import torch
import torch.nn as nn

# =============================================================================
# Configuration (TODO: Fill in actual values)
# =============================================================================

QWEN_CONFIG_06_B: dict[str, int | float | bool | torch.dtype] = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,  # Qwen uses QK normalization
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

QWEN3_CONFIG_1_7B: dict[str, int | float | bool | torch.dtype] = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2048,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 6144,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

QWEN3_CONFIG_4B: dict[str, int | float | bool | torch.dtype] = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2560,
    "n_heads": 32,
    "n_layers": 36,
    "hidden_dim": 9728,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# Mixture of Experts configuration
QWEN3_CONFIG_30B_A3B_MOE: dict[str, int | float | bool | torch.dtype] = {
    "vocab_size": 151_936,
    "context_length": 262_144,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 48,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 10_000_000.0,
    "dtype": torch.bfloat16,
    # MoE specific
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}


# =============================================================================
# Model (TODO: Implement)
# =============================================================================


class Qwen3Model(nn.Module):
    """Qwen 3 language model.

    TODO: Implement the full Qwen 3 architecture.

    Key components needed:
    - Token embeddings
    - TransformerBlock with GQA (with QK norm) and SwiGLU/MoE
    - RMSNorm for normalization
    - Support for both dense and MoE variants

    See Sebastian's qwen3.py for reference implementation.
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize Qwen3Model."""
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError(
            "Qwen3Model not yet implemented. "
            "See tmp/LLMs-from-scratch/pkg/llms_from_scratch/qwen3.py for reference."
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass (not implemented)."""
        raise NotImplementedError


# =============================================================================
# MoE FeedForward (TODO: Implement)
# =============================================================================


class MoEFeedForward(nn.Module):
    """Mixture of Experts Feed-Forward layer.

    TODO: Implement MoE with top-k expert routing.

    Features:
    - Gate network to select experts
    - Multiple expert MLPs
    - Top-k routing with softmax weighting
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize MoEFeedForward."""
        super().__init__()
        raise NotImplementedError(
            "MoEFeedForward not yet implemented. "
            "See tmp/LLMs-from-scratch/pkg/llms_from_scratch/qwen3.py for reference."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (not implemented)."""
        raise NotImplementedError


# =============================================================================
# Tokenizer (TODO: Implement)
# =============================================================================


class Qwen3Tokenizer:
    """Qwen 3 tokenizer.

    TODO: Implement tokenizer using the tokenizers library.

    Features:
    - Special tokens handling (<|endoftext|>, <|im_start|>, <|im_end|>, etc.)
    - Chat template wrapping
    - Support for thinking tokens (<think>, </think>)
    """

    def __init__(
        self,
        tokenizer_file_path: str = "tokenizer.json",
        repo_id: str | None = None,
        apply_chat_template: bool = True,
    ) -> None:
        """Initialize Qwen3Tokenizer."""
        raise NotImplementedError(
            "Qwen3Tokenizer not yet implemented. "
            "See tmp/LLMs-from-scratch/pkg/llms_from_scratch/qwen3.py for reference."
        )

    def encode(self, text: str, chat_wrapped: bool | None = None) -> list[int]:
        """Encode text to token IDs (not implemented)."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text (not implemented)."""
        raise NotImplementedError


# =============================================================================
# Weight Loading (TODO: Implement)
# =============================================================================


def load_weights_into_qwen(
    model: Qwen3Model, param_config: dict, params: dict[str, torch.Tensor]
) -> None:
    """Load pretrained weights into Qwen 3 model.

    TODO: Implement weight loading from HuggingFace checkpoints.

    See Sebastian's qwen3.py for reference implementation.
    """
    raise NotImplementedError
