"""Transformer block implementation.

Moved to components on 2026.01.15
"""

import torch
import torch.nn as nn

from legollm.components.attention import MultiHeadCausalAttention
from legollm.components.feedforward import MLP
from legollm.components.normalization import LayerNorm


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (GPT-2 style).

    Architecture:
        x -> LayerNorm -> Attention -> + -> LayerNorm -> MLP -> +
        |______________________________|   |___________________|

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        context_length: Maximum sequence length
        hidden_dim: MLP hidden dimension
        dropout: Dropout probability
        bias: Whether to use bias in attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_length: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize TransformerBlock."""
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadCausalAttention(
            d_in=embed_dim,
            d_out=embed_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            causal=True,
        )
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of same shape
        """
        # Attention with residual
        x = x + self.drop_shortcut(self.attn(self.ln1(x)))
        # MLP with residual
        x = x + self.drop_shortcut(self.mlp(self.ln2(x)))
        return x
