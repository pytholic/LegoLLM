"""Multi-head attention implementations.

Created by @pytholic on 2025.10.31
Moved to components on 2026.01.15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    """Causal scaled dot-product attention (single head).

    Notes:
        - This is a "Decoder" attention block for autoregressive generation
        - Modern LLMs do not use dropout for attention weights
        - Causal masking prevents attending to future tokens
    """

    def __init__(
        self,
        d_in: int,
        head_dim: int,
        context_length: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        """Initialize the Attention layer.

        Args:
            d_in: Input dimension (embedding dimension)
            head_dim: Dimension of each head
            context_length: Context window length
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in linear projections
            causal: Whether to use causal masking
        """
        super().__init__()
        self.d_in = d_in
        self.head_dim = head_dim
        self.causal = causal

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(d_in, head_dim, bias=bias)
        self.k_proj = nn.Linear(d_in, head_dim, bias=bias)
        self.v_proj = nn.Linear(d_in, head_dim, bias=bias)

        # Linear projection for output
        self.out_proj = nn.Linear(head_dim, d_in, bias=bias)

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
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_in)
        """
        _, seq_len, _ = x.shape

        # Project to Q, K, and V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scale dot-product attention
        attn_scores = Q @ K.transpose(-2, -1)

        # Apply causal mask if provided
        if self.causal:
            mask = self.causal_mask[:seq_len, :seq_len]
            attn_scores.masked_fill_(mask == 0, float("-inf"))

        # Softmax attention weights
        attn_weights = F.softmax(attn_scores * self.head_dim**-0.5, dim=-1)

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        output = attn_weights @ V

        # Project output
        return self.out_proj(output)


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal scaled dot-product attention."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        """Initialize the MultiHeadCausalAttention layer.

        Args:
            d_in: Input dimension
            d_out: Output dimension, must be divisible by num_heads
            context_length: Context window length
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in linear projections
            causal: Whether to use causal masking
        """
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.bias = bias
        self.causal = causal

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(d_in, d_out, bias=bias)
        self.k_proj = nn.Linear(d_in, d_out, bias=bias)
        self.v_proj = nn.Linear(d_in, d_out, bias=bias)

        # Linear projection for output
        self.out_proj = nn.Linear(d_out, d_out, bias=bias)

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
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, and V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (b, seq_len, d_out) -> (b, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = Q @ K.transpose(-2, -1) * self.head_dim**-0.5

        # Apply causal mask
        if self.causal:
            mask = self.causal_mask[:seq_len, :seq_len]
            attn_scores.masked_fill_(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = attn_weights @ V

        # Combine heads: (b, num_heads, seq_len, head_dim) -> (b, seq_len, d_out)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        return self.out_proj(context)


class MultiHeadCausalAttentionPyTorch(nn.Module):
    """Multi-head causal attention using PyTorch's scaled_dot_product_attention.

    This version leverages PyTorch's optimized attention implementation,
    which can use Flash Attention on compatible hardware.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        """Initialize the MultiHeadCausalAttentionPyTorch layer."""
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout_p = dropout
        self.bias = bias
        self.causal = causal

        # Linear projections
        self.q_proj = nn.Linear(d_in, d_out, bias=bias)
        self.k_proj = nn.Linear(d_in, d_out, bias=bias)
        self.v_proj = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's optimized attention
        dropout = 0.0 if not self.training else self.dropout_p
        context = F.scaled_dot_product_attention(Q, K, V, dropout_p=dropout, is_causal=self.causal)

        # Combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        return self.out_proj(context)
