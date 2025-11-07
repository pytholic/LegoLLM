"""Transformer Embeddings module.

Created by @pytholic on 2025.10.30
"""

import torch
import torch.nn as nn

from legollm.core.exceptions import EmbeddingsError
from legollm.models.positional_embedding import PositionalEmbedding
from legollm.models.token_embedding import TokenEmbedding


class TransformerEmbeddings(nn.Module):
    """Combined token + positional embeddings with dropout."""

    def __init__(
        self, vocab_size: int, embed_dim: int, context_length: int, dropout: float = 0.1
    ) -> None:
        """Initialize the TransformerEmbeddings layer.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            context_length: Context window length
            dropout: Dropout probability
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (batch_size, seq_len) - integer token IDs
            seq_len: Length of current sequence

        Returns:
            embeddings: (batch_size, seq_len, embed_dim)

        Raises:
            EmbeddingsError: If the sequence length is greater than context length
        """
        if seq_len > self.positional_embedding.context_length:
            raise EmbeddingsError(
                f"Sequence length {seq_len} is greater than context length {self.positional_embedding.context_length}"
            )
        # Token embeddings: (batch_size, seq_len, embed_dim)
        token_embeddings = self.token_embedding(token_ids)
        # Positional embeddings: (seq_len, embed_dim)
        pos_embeddings = self.positional_embedding(seq_len)
        # Combine embeddings: (batch_size, seq_len, embed_dim)
        embeddings = token_embeddings + pos_embeddings  # Broadcasting handles batch dimension
        embeddings = self.dropout(embeddings)
        return embeddings
