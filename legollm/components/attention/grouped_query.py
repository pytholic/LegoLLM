"""Grouped-Query Attention (GQA) - placeholder stub.

TODO: Implement GQA.

Reference: Sebastian's LLMs-from-scratch/pkg/llms_from_scratch/llama3.py

GQA uses fewer key-value heads than query heads, reducing memory usage
while maintaining most of the performance of full multi-head attention.

Key features:
- num_kv_groups < num_heads
- K and V are repeated to match Q head count
- Optional QK normalization (used in Qwen)
- Typically combined with RoPE
"""

import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention.

    TODO: Implement GQA.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        num_heads: Number of query heads
        num_kv_groups: Number of key-value groups (< num_heads)
        head_dim: Dimension per head (optional, computed if not provided)
        qk_norm: Whether to apply RMSNorm to Q and K (Qwen style)
        dtype: Data type for weights

    Key implementation details:
    - W_query projects to d_out (num_heads * head_dim)
    - W_key, W_value project to num_kv_groups * head_dim
    - K and V are repeated using repeat_interleave to match num_heads
    - RoPE is applied to Q and K
    - Causal masking for autoregressive generation

    See Sebastian's llama3.py GroupedQueryAttention for reference.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize GroupedQueryAttention."""
        super().__init__()
        raise NotImplementedError(
            "GroupedQueryAttention not yet implemented. "
            "See tmp/LLMs-from-scratch/pkg/llms_from_scratch/llama3.py for reference."
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_in)
            mask: Causal mask
            cos: RoPE cosine values
            sin: RoPE sine values

        Returns:
            Output tensor (batch, seq_len, d_out)
        """
        raise NotImplementedError
