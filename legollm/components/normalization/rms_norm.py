"""Root Mean Square Layer Normalization (RMSNorm) - placeholder stub.

TODO: Implement RMSNorm.

Reference: Sebastian's LLMs-from-scratch/pkg/llms_from_scratch/qwen3.py

RMSNorm is a simplified version of LayerNorm that only normalizes
by the root mean square, without re-centering. It's faster and works
well for modern LLMs like Llama and Qwen.

Equation: x_norm = x / sqrt(mean(x^2) + eps) * scale
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    TODO: Implement RMSNorm.

    Args:
        emb_dim: Embedding dimension
        eps: Small constant for numerical stability
        bias: Whether to include a learnable bias (usually False)

    See Sebastian's qwen3.py RMSNorm for reference.
    """

    def __init__(
        self,
        emb_dim: int,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        """Initialize RMSNorm."""
        super().__init__()
        raise NotImplementedError(
            "RMSNorm not yet implemented. "
            "See tmp/LLMs-from-scratch/pkg/llms_from_scratch/qwen3.py for reference."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., emb_dim)

        Returns:
            Normalized tensor of same shape
        """
        raise NotImplementedError
