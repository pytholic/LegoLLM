"""Transformer embeddings (token + positional).

Created by @pytholic on 2025.10.30
Moved to components on 2026.01.15
"""

import torch
import torch.nn as nn

from legollm.components.embeddings.positional import PositionalEmbedding
from legollm.components.embeddings.token import TokenEmbedding
from legollm.core.exceptions import EmbeddingsError


class TransformerEmbeddings(nn.Module):
    """Combined token + positional embeddings with dropout.

    This is the standard embedding layer used in GPT-2 style models,
    combining learned token embeddings with learned positional embeddings.

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Dimension of embedding vectors
        context_length: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_length: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize TransformerEmbeddings."""
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (batch_size, seq_len) - integer token IDs
            seq_len: Optional length of current sequence

        Returns:
            embeddings: (batch_size, seq_len, embed_dim)

        Raises:
            EmbeddingsError: If sequence length exceeds context length
        """
        if seq_len is None:
            seq_len = token_ids.size(1)

        if seq_len > self.positional_embedding.context_length:
            raise EmbeddingsError(
                f"Sequence length {seq_len} exceeds context length "
                f"{self.positional_embedding.context_length}"
            )

        # Token embeddings: (batch_size, seq_len, embed_dim)
        token_embeds = self.token_embedding(token_ids)

        # Positional embeddings: (seq_len, embed_dim)
        pos_embeds = self.positional_embedding(seq_len)

        # Combine and apply dropout
        embeddings = token_embeds + pos_embeds  # Broadcasting handles batch dim
        return self.dropout(embeddings)
