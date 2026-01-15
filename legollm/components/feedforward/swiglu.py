"""SwiGLU Feed-Forward Network - placeholder stub.

TODO: Implement SwiGLU.

Reference: Sebastian's LLMs-from-scratch/pkg/llms_from_scratch/llama3.py

SwiGLU is a gated linear unit with SiLU (Swish) activation,
used in modern LLMs like Llama and Qwen.

Equation: SwiGLU(x) = (SiLU(W1 @ x) * (W2 @ x)) @ W3

Key differences from standard MLP:
- Two parallel projections (gate and up)
- Element-wise multiplication with gated activation
- Typically no bias
"""

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    TODO: Implement SwiGLU.

    Args:
        emb_dim: Input/output embedding dimension
        hidden_dim: Hidden dimension (intermediate size)
        dtype: Data type for weights
        bias: Whether to use bias (typically False for modern LLMs)

    Architecture:
        gate = SiLU(fc1(x))  # Gate projection with SiLU activation
        up = fc2(x)          # Up projection (no activation)
        out = fc3(gate * up) # Down projection

    See Sebastian's llama3.py FeedForward for reference.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        dtype: torch.dtype | None = None,
        bias: bool = False,
    ) -> None:
        """Initialize SwiGLU."""
        super().__init__()
        raise NotImplementedError(
            "SwiGLU not yet implemented. "
            "See tmp/LLMs-from-scratch/pkg/llms_from_scratch/llama3.py FeedForward for reference."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., emb_dim)

        Returns:
            Output tensor of same shape
        """
        raise NotImplementedError
