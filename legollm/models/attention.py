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
        d_in: int,
        head_dim: int,
        context_length: int,
        dropout: float = 0.0,  # not used in modern LLMs
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
        Q = self.q_proj(x)  # (b, seq_len, head_dim)
        K = self.k_proj(x)  # (b, seq_len, head_dim)
        V = self.v_proj(x)  # (b, seq_len, head_dim)

        # Scale dot-product attention
        # scores = QK^T / sqrt(d_k)
        # -2,-1 means last two dimensions i.e. seq_len, head_dim
        attn_scores = Q @ K.transpose(
            -2, -1
        )  # (b, seq_len, head_dim) @ (b, head_dim, seq_len) -> (b, seq_len, seq_len)

        # Apply causal mask if provided
        # NOTE: We need to account for the cases where the sequence length is less than the maximum context length
        if self.causal:
            attn_scores.masked_fill_(
                self.causal_mask[:seq_len, :seq_len] == 0, float("-inf")
            )  # *_ ops are in-place

        # Softmax attention weights
        attn_weights = F.softmax(
            attn_scores * self.head_dim**-0.5, dim=-1
        )  # scale by sqrt(head_dim)

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        output = attn_weights @ V

        # Project output
        output = self.out_proj(output)
        return output


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
            d_out: Dimension of the concatenated heads output, must be divisible by num_heads
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
        self.head_dim = d_out // num_heads  # Each head processes d_out // num_heads dimension
        self.dropout = dropout
        self.bias = bias
        self.causal = causal

        # Linear projections for queries, keys, values
        # Project from d_in to d_out (num_heads * head_dim)
        self.q_proj = nn.Linear(d_in, d_out, bias=bias)
        self.k_proj = nn.Linear(d_in, d_out, bias=bias)
        self.v_proj = nn.Linear(d_in, d_out, bias=bias)

        # Linear projection for output (combining heads)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

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
        Q = self.q_proj(x)  # (b, seq_len, d_out)
        K = self.k_proj(x)  # (b, seq_len, d_out)
        V = self.v_proj(x)  # (b, seq_len, d_out)

        # Split into heads
        # Unroll last dim: (b, seq_len, d_out) -> (b, seq_len, num_heads, head_dim)
        # Transpose to: (b, num_heads, seq_len, head_dim)
        Q_ = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_ = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_ = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scale dot-product attention
        # scores = Q_K_^T / sqrt(head_dim)
        attn_scores = Q_ @ K_.transpose(-2, -1) * self.head_dim**-0.5

        # Apply causal mask if provided
        # NOTE: We need to account for the cases where the sequence length is less than the maximum context length
        if self.causal:
            attn_scores.masked_fill_(self.causal_mask[:seq_len, :seq_len] == 0, float("-inf"))

        # Softmax attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context_vector = attn_weights @ V_  # (b, num_heads, seq_len, head_dim)

        # Combine heads
        # (b, num_heads, seq_len, head_dim) -> (b, seq_len, num_heads, head_dim) -> (b, seq_len, d_out)
        context_vector = (
            context_vector.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)
        )

        # Project output
        output = self.out_proj(context_vector)
        return output


class MultiHeadCausalAttentionPyTorch(nn.Module):
    """Multi-head causal scaled dot-product attention.

    Using Pytorch's built-in scaled dot-product attention for calculating the attention weights and values.
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
        """Initialize the MultiHeadCausalAttentionPyTorch layer.

        Args:
            d_in: Input dimension
            d_out: Dimension of the concatenated heads output, must be divisible by num_heads
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
        self.head_dim = d_out // num_heads  # Each head processes d_out // num_heads dimension
        self.dropout = dropout
        self.bias = bias
        self.causal = causal

        # Linear projections for queries, keys, values
        # Project from d_in to d_out (num_heads * head_dim)
        self.q_proj = nn.Linear(d_in, d_out, bias=bias)
        self.k_proj = nn.Linear(d_in, d_out, bias=bias)
        self.v_proj = nn.Linear(d_in, d_out, bias=bias)

        # Linear projection for output (combining heads)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

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
        Q = self.q_proj(x)  # (b, seq_len, d_out)
        K = self.k_proj(x)  # (b, seq_len, d_out)
        V = self.v_proj(x)  # (b, seq_len, d_out)

        # Split into heads
        # Unroll last dim: (b, seq_len, d_out) -> (b, seq_len, num_heads, head_dim)
        # Transpose to: (b, num_heads, seq_len, head_dim)
        Q_ = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_ = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_ = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scale dot-product attention
        # use Pytorch's built-in scaled dot-product attention
        use_dropout = 0.0 if not self.training else self.dropout

        context_vector = F.scaled_dot_product_attention(
            Q_, K_, V_, dropout_p=use_dropout, attn_mask=None, is_causal=self.causal
        )

        # Combine heads
        # (b, num_heads, seq_len, head_dim) -> (b, seq_len, num_heads, head_dim) -> (b, seq_len, d_out)
        context_vector = (
            context_vector.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)
        )

        # Project output
        output = self.out_proj(context_vector)
        return output


if __name__ == "__main__":
    import time

    import torch

    from legollm.models.token_embedding import TokenEmbedding

    torch.manual_seed(42)

    context_length = 5
    d_in = 2
    d_out = 4
    num_heads = 2
    dropout = 0.0
    bias = True
    causal = True

    # get a batch of input tokens
    x = torch.randint(0, 5, (3, context_length))
    print(f"Input tokens: {x}")
    print(f"Input tokens shape: {x.shape}")
    print("*" * 100)

    # convert ints to embeddings
    token_embedding = TokenEmbedding(5, d_in)
    x_embed = token_embedding(x)
    print(f"Input embeddings: {x_embed}")
    print(f"Input embeddings shape: {x_embed.shape}")
    print("*" * 100)

    attention_torch = MultiHeadCausalAttentionPyTorch(
        d_in, d_out, context_length, num_heads, dropout, bias, causal
    )
    attention_manual = MultiHeadCausalAttention(
        d_in, d_out, context_length, num_heads, dropout, bias, causal
    )

    # compare time between manual and torch implementation for multiple and average
    num_iterations = 100
    time_manual = 0
    time_torch = 0
    for _ in range(num_iterations):
        start_time = time.time()
        output = attention_torch(x_embed)
        end_time = time.time()
        time_torch += end_time - start_time
        start_time = time.time()
        output_manual = attention_manual(x_embed)
        end_time = time.time()
        time_manual += end_time - start_time
    time_manual /= num_iterations
    time_torch /= num_iterations
    print(f"Time manual: {time_manual}")
    print(f"Time torch: {time_torch}")
    print(f"Speedup: {time_manual / time_torch:.3f}x")
    print("*" * 100)
