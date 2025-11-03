"""Attention module for the transformer architecture.

Created by @pytholic on 2025.10.31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    """Causal scaled dot-product attention (single head).

    Notes:
        - This is a "Decoder" attention block. We also have "encoder" attention block for some application like sentiment analysis, summarization, etc. where we want to attend to the entire sequence.
        - Modern LLMs do not use dropout for attention weights. Added here for the sake of completeness.
        - Causal masking prevents attending to future tokens (autoregressive)
    """

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        context_length: int,
        dropout: float = 0.0,  # not used in modern LLMs
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        """Initialize the Attention layer.

        Args:
            embed_dim: Embedding dimension (input/output size)
            head_dim: Dimension of queries, keys, values
            context_length: Context window length
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in linear projections
            causal: Whether to use causal masking
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.causal = causal

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=bias)

        # Linear projection for output
        self.out_proj = nn.Linear(head_dim, embed_dim, bias=bias)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Register buffer for causal mask
        if causal:
            self.register_buffer(
                "causal_mask", torch.tril(torch.ones(context_length, context_length))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size (B), seq_len (L), embed_dim (D))

        Returns:
            Output tensor of shape (batch_size (B), seq_len (L), embed_dim (D))
        """
        B, L, D = x.shape
        # Project to Q, K, and V
        Q = self.q_proj(x)  # (B, L, head_dim (H))
        K = self.k_proj(x)  # (B, L, H)
        V = self.v_proj(x)  # (B, L, H)

        # Scale dot-product attention
        # scores = QK^T / sqrt(d_k)
        # -2,-1 means last two dimensions i.e. seq_len, head_dim
        attn_scores = (
            Q @ K.transpose(-2, -1) / self.head_dim**0.5
        )  # (B, L, head_dim) @ (B, head_dim, L) -> (B, L, L)

        # Apply causal mask if provided
        # NOTE: We need to account for the cases where the sequence length is less than the maximum context length
        if self.causal:
            attn_scores.masked_fill_(
                self.causal_mask[:L, :L] == 0, float("-inf")
            )  # *_ ops are in-place

        # Softmax attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        print(f"DEBUG: attn_weights:\n{attn_weights}")

        # DEBUG: verify that the attention weights sum to 1 for each token
        assert torch.allclose(
            attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-6
        ), "Attention weights should sum to 1 for each token"

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        output = attn_weights @ V

        # Project output
        output = self.out_proj(output)
        return output
