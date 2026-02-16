"""GPT-2 architecture - self-contained implementation.

This file contains everything needed for GPT-2:
- Configuration dictionaries
- All model components (LayerNorm, MLP, Attention, TransformerBlock)
- The main GPT2 model class

Created by @pytholic on 2025.11.09
Restructured to self-contained format on 2026.01.15
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# =============================================================================
# Configuration
# =============================================================================


class GPT2Config(Protocol):
    """GPT-2 configuration protocol."""

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
class GPT2ConfigDataclass:
    """GPT-2 configuration dataclass."""

    vocab_size: int = 50257
    context_length: int = 1024
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    hidden_dim: int = 3072  # 4 * embed_dim
    dropout: float = 0.1
    bias: bool = True
    dtype: torch.dtype = torch.float32


# Standard GPT-2 configurations
GPT2_CONFIG_124M: dict[str, int | float | bool | torch.dtype] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 12,
    "num_heads": 12,
    "embed_dim": 768,
    "hidden_dim": 3072,
    "dropout": 0.1,
    "bias": True,
    "dtype": torch.float32,
}

GPT2_CONFIG_355M: dict[str, int | float | bool | torch.dtype] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "embed_dim": 1024,
    "hidden_dim": 4096,
    "dropout": 0.1,
    "bias": True,
    "dtype": torch.float32,
}

GPT2_CONFIG_774M: dict[str, int | float | bool | torch.dtype] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 36,
    "num_heads": 20,
    "embed_dim": 1280,
    "hidden_dim": 5120,
    "dropout": 0.1,
    "bias": True,
    "dtype": torch.float32,
}

GPT2_CONFIG_1558M: dict[str, int | float | bool | torch.dtype] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 48,
    "num_heads": 25,
    "embed_dim": 1600,
    "hidden_dim": 6400,
    "dropout": 0.1,
    "bias": True,
    "dtype": torch.float32,
}


class GPT2Variant(StrEnum):
    """Available pretrained GPT-2 model variants."""

    GPT2 = "gpt2"  # gpt2-small (124M)
    GPT2_MEDIUM = "gpt2-medium"  # gpt2-medium (355M)
    GPT2_LARGE = "gpt2-large"  # gpt2-large (774M)
    GPT2_XL = "gpt2-xl"  # gpt2-xl (1558M)


# Variant → config lookup
MODEL_CONFIGS: dict[GPT2Variant, dict[str, int | float | bool | torch.dtype]] = {
    GPT2Variant.GPT2: GPT2_CONFIG_124M,
    GPT2Variant.GPT2_MEDIUM: GPT2_CONFIG_355M,
    GPT2Variant.GPT2_LARGE: GPT2_CONFIG_774M,
    GPT2Variant.GPT2_XL: GPT2_CONFIG_1558M,
}


# =============================================================================
# Layer Normalization
# =============================================================================


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift."""

    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        """Initialize LayerNorm."""
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * x + self.shift


# =============================================================================
# Feed-Forward Network (MLP)
# =============================================================================


class MLP(nn.Module):
    """GELU-activated feed-forward network."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        """Initialize MLP."""
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


# =============================================================================
# Multi-Head Causal Attention
# =============================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize MultiHeadAttention."""
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) * (self.head_dim**-0.5)

        # Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = attn_weights @ V

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(context)


# =============================================================================
# Transformer Block
# =============================================================================


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: GPT2Config | dict) -> None:
        """Initialize TransformerBlock."""
        super().__init__()

        # Handle both dict and dataclass configs
        if isinstance(cfg, dict):
            embed_dim = cfg["embed_dim"]
            num_heads = cfg["num_heads"]
            context_length = cfg["context_length"]
            hidden_dim = cfg["hidden_dim"]
            dropout = cfg["dropout"]
            bias = cfg["bias"]
        else:
            embed_dim = cfg.embed_dim
            num_heads = cfg.num_heads
            context_length = cfg.context_length
            hidden_dim = cfg.hidden_dim
            dropout = cfg.dropout
            bias = cfg.bias

        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
            bias=bias,
        )
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Attention with residual
        x = x + self.drop_shortcut(self.attn(self.ln1(x)))
        # MLP with residual
        x = x + self.drop_shortcut(self.mlp(self.ln2(x)))
        return x


# =============================================================================
# GPT-2 Model
# =============================================================================


class GPT2(nn.Module):
    """GPT-2 language model.

    Args:
        cfg: Configuration dict or dataclass with model hyperparameters

    Example:
        >>> model = GPT2(GPT2_CONFIG_124M)
        >>> tokens = torch.randint(0, 50257, (1, 10))
        >>> logits = model(tokens)  # (1, 10, 50257)
    """

    def __init__(self, cfg: GPT2Config | dict = GPT2_CONFIG_124M) -> None:
        """Initialize GPT2."""
        super().__init__()

        # Handle both dict and dataclass configs
        if isinstance(cfg, dict):
            vocab_size = cfg["vocab_size"]
            context_length = cfg["context_length"]
            embed_dim = cfg["embed_dim"]
            num_layers = cfg["num_layers"]
            dropout = cfg["dropout"]
            dtype = cfg.get("dtype", torch.float32)
        else:
            vocab_size = cfg.vocab_size
            context_length = cfg.context_length
            embed_dim = cfg.embed_dim
            num_layers = cfg.num_layers
            dropout = cfg.dropout
            dtype = getattr(cfg, "dtype", torch.float32)

        self.cfg = cfg

        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.drop_emb = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(num_layers)])

        # Final layer norm
        self.ln_f = LayerNorm(embed_dim)

        # Output head (tied with token embeddings)
        self.out_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.out_head.weight = self.tok_emb.weight  # Weight tying

        # Convert to specified dtype
        self.to(dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (batch_size, seq_len) integer token IDs

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        # Get embeddings
        tok_embeds = self.tok_emb(token_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=token_ids.device))
        x = self.drop_emb(tok_embeds + pos_embeds)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and output projection
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

    @property
    def config(self) -> GPT2Config | dict:
        """Get the model configuration (for compatibility with generate.py)."""
        return self.cfg

    @property
    def context_length(self) -> int:
        """Get the context length."""
        if isinstance(self.cfg, dict):
            return self.cfg["context_length"]
        return self.cfg.context_length


# =============================================================================
# Weight Loading (for pretrained models)
# =============================================================================


def load_gpt2_weights(model: GPT2, variant: GPT2Variant = GPT2Variant.GPT2) -> None:
    """Download and load pretrained GPT-2 weights into the model.

    The model must have been created with the matching config (e.g. GPT2_CONFIG_124M
    for GPT2Variant.GPT2). If there's a mismatch, load_state_dict will raise an error.

    Args:
        model: GPT2 model instance.
        variant: Which pretrained GPT-2 to download.
    """
    safetensors_path = _download_gpt2_safetensors(variant)
    hf_weights = load_file(safetensors_path)
    mapped = _map_hf_weights(hf_weights)
    # strict=False because out_head.weight is tied to tok_emb.weight (not in mapped dict)
    model.load_state_dict(mapped, strict=False)


def _download_gpt2_safetensors(
    variant: GPT2Variant = GPT2Variant.GPT2, cache_dir: Path | None = None
) -> Path:
    """Download GPT-2 safetensors file from HuggingFace Hub.

    Args:
        variant: Which pretrained GPT-2 to download.
        cache_dir: Directory to cache downloads (default: ~/.cache/legollm/).

    Returns:
        Path to the downloaded safetensors file.
    """
    url = f"https://huggingface.co/openai-community/{variant.value}/resolve/main/model.safetensors"

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "legollm"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / f"{variant.value}.safetensors"
    if not output_path.exists():
        print(f"Downloading {variant.value} weights from HuggingFace...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {output_path}")

    return output_path


def _map_hf_weights(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HuggingFace GPT-2 state dict keys to our GPT2 model keys.

    Three transforms:
    1. Key renaming (e.g. "h.0.ln_1.weight" → "blocks.0.ln1.scale")
    2. Transposing Conv1D weights → Linear weights (.T on 4 matrices per block)
    3. Splitting fused c_attn (QKV) into separate q_proj, k_proj, v_proj
    """
    mapped: dict[str, torch.Tensor] = {}

    # Embeddings
    mapped["tok_emb.weight"] = hf["wte.weight"]
    mapped["pos_emb.weight"] = hf["wpe.weight"]

    # Final layer norm
    mapped["ln_f.scale"] = hf["ln_f.weight"]
    mapped["ln_f.shift"] = hf["ln_f.bias"]

    # out_head.weight is tied to tok_emb.weight — skip it

    # Transformer blocks
    num_layers = sum(1 for k in hf if k.startswith("h.") and k.endswith(".ln_1.weight"))

    for i in range(num_layers):
        # Layer norms
        mapped[f"blocks.{i}.ln1.scale"] = hf[f"h.{i}.ln_1.weight"]
        mapped[f"blocks.{i}.ln1.shift"] = hf[f"h.{i}.ln_1.bias"]
        mapped[f"blocks.{i}.ln2.scale"] = hf[f"h.{i}.ln_2.weight"]
        mapped[f"blocks.{i}.ln2.shift"] = hf[f"h.{i}.ln_2.bias"]

        # Attention: split fused c_attn into q, k, v and transpose
        q_w, k_w, v_w = hf[f"h.{i}.attn.c_attn.weight"].chunk(3, dim=-1)
        mapped[f"blocks.{i}.attn.q_proj.weight"] = q_w.T
        mapped[f"blocks.{i}.attn.k_proj.weight"] = k_w.T
        mapped[f"blocks.{i}.attn.v_proj.weight"] = v_w.T

        q_b, k_b, v_b = hf[f"h.{i}.attn.c_attn.bias"].chunk(3, dim=0)
        mapped[f"blocks.{i}.attn.q_proj.bias"] = q_b
        mapped[f"blocks.{i}.attn.k_proj.bias"] = k_b
        mapped[f"blocks.{i}.attn.v_proj.bias"] = v_b

        # Attention output projection: transpose
        mapped[f"blocks.{i}.attn.out_proj.weight"] = hf[f"h.{i}.attn.c_proj.weight"].T
        mapped[f"blocks.{i}.attn.out_proj.bias"] = hf[f"h.{i}.attn.c_proj.bias"]

        # MLP: transpose both weight matrices
        mapped[f"blocks.{i}.mlp.fc1.weight"] = hf[f"h.{i}.mlp.c_fc.weight"].T
        mapped[f"blocks.{i}.mlp.fc1.bias"] = hf[f"h.{i}.mlp.c_fc.bias"]
        mapped[f"blocks.{i}.mlp.fc2.weight"] = hf[f"h.{i}.mlp.c_proj.weight"].T
        mapped[f"blocks.{i}.mlp.fc2.bias"] = hf[f"h.{i}.mlp.c_proj.bias"]

    return mapped


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    from legollm.utils import count_model_params

    # Test with dict config
    model = GPT2(GPT2_CONFIG_124M)
    print(f"GPT-2 124M parameters: {count_model_params(model):.2f}M")

    # Test forward pass
    tokens = torch.randint(0, 50257, (2, 10))
    logits = model(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")

    # Test with dataclass config
    config = GPT2ConfigDataclass()
    model2 = GPT2(config)
    print(f"GPT-2 (dataclass) parameters: {count_model_params(model2):.2f}M")
