"""Model definitions for the LegoLLM project.

Created by @pytholic on 2025.10.28

NOTE: This module is maintained for backward compatibility.
New code should use:
- legollm.architectures for self-contained model architectures
- legollm.components for reusable building blocks
- legollm.generation for text generation utilities
"""

# Backward-compatible re-exports
from legollm.models.architectures.gpt import GPT, GPT2Config125M
from legollm.models.attention import (
    CausalAttention,
    MultiHeadCausalAttention,
    MultiHeadCausalAttentionPyTorch,
)
from legollm.models.positional_embedding import PositionalEmbedding
from legollm.models.token_embedding import TokenEmbedding
from legollm.models.transformer_embeddings import TransformerEmbeddings

__all__ = [
    "GPT",
    "CausalAttention",
    "GPT2Config125M",
    "MultiHeadCausalAttention",
    "MultiHeadCausalAttentionPyTorch",
    "PositionalEmbedding",
    "TokenEmbedding",
    "TransformerEmbeddings",
]
