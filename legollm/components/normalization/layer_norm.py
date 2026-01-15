"""Layer normalization implementation.

Moved to components on 2026.01.15
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift.

    This is the standard LayerNorm used in GPT-2.
    For modern architectures like Llama/Qwen, see RMSNorm.

    Args:
        embed_dim: Dimension of input features
        eps: Small constant for numerical stability
    """

    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        """Initialize LayerNorm."""
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., embed_dim)

        Returns:
            Normalized tensor of same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * x + self.shift
