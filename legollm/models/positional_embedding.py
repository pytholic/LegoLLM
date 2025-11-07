"""Positional Embedding module.

Created by @pytholic on 2025.10.30
"""

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings (GPT-2 style)."""

    def __init__(self, context_length: int, embed_dim: int) -> None:
        """Initialize the PositionalEmbedding layer.

        Args:
            context_length: Maximum sequence length (context window)
            embed_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Embedding(context_length, embed_dim)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Forward pass.

        Args:
            seq_len: Length of current sequence

        Returns:
            pos_embeddings: (seq_len, embed_dim)
        """
        positions = torch.arange(seq_len, device=self.pos_embedding.weight.device)
        return self.pos_embedding(positions)
