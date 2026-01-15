"""MLP (feed-forward network) implementation.

Moved to components on 2026.01.15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """GELU-activated feed-forward network.

    This is the standard MLP used in GPT-2.
    For modern architectures like Llama/Qwen, see SwiGLU.

    Args:
        embed_dim: Input and output dimension
        hidden_dim: Hidden layer dimension (typically 4x embed_dim)
        dropout: Dropout probability
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        """Initialize MLP."""
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., embed_dim)

        Returns:
            Output tensor of same shape
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)
