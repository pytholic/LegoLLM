"""Attention mechanism implementations.

Available:
- MultiHeadCausalAttention: Standard multi-head causal attention
- MultiHeadCausalAttentionPyTorch: Using PyTorch's scaled_dot_product_attention

Todo:
- GroupedQueryAttention: Grouped-query attention for Llama/Qwen
- FlashAttention: Flash attention wrapper
"""

from legollm.components.attention.multi_head import (
    CausalAttention,
    MultiHeadCausalAttention,
    MultiHeadCausalAttentionPyTorch,
)

__all__ = [
    "CausalAttention",
    "MultiHeadCausalAttention",
    "MultiHeadCausalAttentionPyTorch",
]
