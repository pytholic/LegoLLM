"""Token embeddings model.

Created by @pytholic on 2025.10.28
"""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Token embedding layer.

    Attributes:
        vocab_size: Size of vocabulary
        embed_dim: Dimension of embedding vectors
        embedding: Embedding layer

    Input/Output shapes:
        Input shape: (batch_size, sequence_length)
        Output shape: (batch_size, sequence_length, embed_dim)

    Example:
        >>> token_embedding = TokenEmbedding(vocab_size=100, embed_dim=10)
        >>> input = torch.randint(0, 100, (2, 5))
        >>> output = token_embedding(input)
        >>> print(output.shape)
        torch.Size([2, 5, 10])
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """Initialize the TokenEmbedding layer.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Single lookup table
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (batch_size, seq_len) - integer token IDs
        Returns:
            embeddings: (batch_size, seq_len, embed_dim)
        """
        return self.embedding(token_ids)
