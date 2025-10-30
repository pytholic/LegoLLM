"""Transformer Embeddings module.

Created by @pytholic on 2025.10.30
"""

import torch
import torch.nn as nn

from legollm.core.exceptions import EmbeddingsError
from legollm.models.positional_embedding import PositionalEmbedding
from legollm.models.token_embedding import TokenEmbedding


class TransformerEmbeddings(nn.Module):
    """Combined token + positional embeddings with dropout.

    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self, vocab_size: int, embed_dim: int, max_seq_len: int, dropout: float = 0.1
    ) -> None:
        """Initialize the TransformerEmbeddings layer.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (batch_size, seq_len) - integer token IDs
            seq_len: Length of current sequence

        Returns:
            embeddings: (batch_size, seq_len, embed_dim)

        Raises:
            EmbeddingsError: If the sequence length is greater than max_seq_len
        """
        if seq_len > self.positional_embedding.max_seq_len:
            raise EmbeddingsError(
                f"Sequence length {seq_len} is greater than max_seq_len {self.positional_embedding.max_seq_len}"
            )
        # Token embeddings: (batch_size, seq_len, embed_dim)
        token_embeddings = self.token_embedding(token_ids)
        # Positional embeddings: (seq_len, embed_dim)
        pos_embeddings = self.positional_embedding(seq_len)
        # Combine embeddings: (batch_size, seq_len, embed_dim)
        embeddings = token_embeddings + pos_embeddings  # Broadcasting handles batch dimension
        embeddings = self.dropout(embeddings)
        return embeddings
