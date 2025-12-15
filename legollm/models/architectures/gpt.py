"""GPT architecture implementations.

Created by @pytholic on 2025.11.09
"""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from legollm.models.attention import MultiHeadCausalAttention
from legollm.models.transformer_embeddings import TransformerEmbeddings


class GPTConfig(Protocol):
    """GPT configuration."""

    vocab_size: int
    context_length: int
    num_layers: int
    num_heads: int
    embed_dim: int
    hidden_dim: int
    dropout: float
    bias: bool
    dtype: torch.dtype


@dataclass
class GPT2Config125M:
    """GPT2 configuration for 125M parameters."""

    vocab_size: int = 50257
    context_length: int = 1024
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    hidden_dim: int = 4 * embed_dim
    dropout: float = 0.1
    bias: bool = True
    dtype: torch.dtype = torch.float32


config_mapping = {
    "125M": GPT2Config125M(),
}


class GPT(nn.Module):
    """GPT architecture."""

    def __init__(self, config: GPTConfig = GPT2Config125M()) -> None:
        """Initialize the GPT architecture."""
        super().__init__()
        assert config.vocab_size is not None, "Vocabulary size is required"
        assert config.context_length is not None, "Context length is required"
        self.config = config

        self.transformer_embeddings = TransformerEmbeddings(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            context_length=config.context_length,
            dropout=config.dropout,
        )
        self.trt_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.drop_emb = nn.Dropout(config.dropout)  # dropout after transformer embeddings
        self.ln_f = LayerNorm(config.embed_dim)  # final layer norm
        self.out_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )  # project logits to vocab size
        # Tie the weights of the output head to the input embeddings (GPT-2 weight tying).
        self.out_head.weight = self.transformer_embeddings.token_embedding.embedding.weight

        # Convert entire model to specified dtype
        self.to(config.dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GPT architecture."""
        seq_len = token_ids.size(1)
        x = self.transformer_embeddings(token_ids=token_ids, seq_len=seq_len)
        x = self.drop_emb(x)
        for block in self.trt_blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.out_head(x)
        return logits

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return next(self.parameters()).dtype


class LayerNorm(nn.Module):
    """Layer normalization."""

    def __init__(self, embed_dim: int) -> None:
        """Initialize the LayerNorm layer."""
        super().__init__()
        self.eps = 1e-5
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = self.scale * x + self.shift
        return x


class TransformerBlock(nn.Module):
    """A single transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the TransformerBlock layer."""
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(config.embed_dim)
        self.attn = MultiHeadCausalAttention(
            d_in=config.embed_dim,
            d_out=config.embed_dim,
            context_length=config.context_length,
            num_heads=config.num_heads,
            dropout=config.dropout,
            bias=config.bias,
            causal=True,
        )
        self.ln2 = LayerNorm(config.embed_dim)
        self.mlp = MLP(config.embed_dim, config.hidden_dim, config.dropout)
        self.drop_shortcut = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_shortcut(self.attn(self.ln1(x)))
        x = x + self.drop_shortcut(self.mlp(self.ln2(x)))
        return x


class MLP(nn.Module):
    """Multi-layer perceptron feed-forward network."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        """Initialize the MLP layer.

        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    import torch

    from legollm.utils import count_model_params

    model = GPT(GPT2Config125M())
    print(f"Number of trainable parameters in the model: {count_model_params(model):.2f}M")
