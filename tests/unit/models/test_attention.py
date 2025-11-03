"""Unit tests for the attention modules.

Created by @pytholic on 2025.11.02
"""
# pyright: reportMissingModuleSource=false

from __future__ import annotations

import math

import torch

from legollm.models.attention import CausalAttention  # type: ignore[import]


def _set_identity_projections(module: CausalAttention) -> None:
    """Configure projections to behave like identity mappings.

    Ensures the attention computations are deterministic and easy to verify.
    """
    with torch.no_grad():
        eye = torch.eye(module.head_dim, module.embed_dim)
        module.q_proj.weight.copy_(eye)
        module.k_proj.weight.copy_(eye)
        module.v_proj.weight.copy_(eye)
        module.out_proj.weight.copy_(torch.eye(module.embed_dim, module.head_dim))


class TestCausalAttention:
    """Test the CausalAttention module."""

    def test_output_shape(self) -> None:
        """Test the output shape of the Attention module."""
        embed_dim = head_dim = 4
        context_length = seq_len = 4

        attention = CausalAttention(
            embed_dim=embed_dim,
            head_dim=head_dim,
            context_length=context_length,
            dropout=0.0,
            bias=False,
            causal=True,
        )
        attention.eval()
        _set_identity_projections(attention)

        x = torch.arange(seq_len * embed_dim, dtype=torch.float32).view(1, seq_len, embed_dim)
        output = attention(x)
        assert output.shape == (1, seq_len, embed_dim)

    def test_attention_causal_mask_blocks_future_positions(self) -> None:
        """Ensure causal masking prevents attending to future tokens."""
        embed_dim = head_dim = 4
        context_length = seq_len = 4

        attention = CausalAttention(
            embed_dim=embed_dim,
            head_dim=head_dim,
            context_length=context_length,
            dropout=0.0,
            bias=False,
            causal=True,
        )
        attention.eval()
        _set_identity_projections(attention)

        x = torch.arange(seq_len * embed_dim, dtype=torch.float32).view(1, seq_len, embed_dim)

        output = attention(x)

        q = attention.q_proj(x)
        k = attention.k_proj(x)
        v = attention.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(attention.head_dim)
        mask = attention.causal_mask[:seq_len, :seq_len]  # type: ignore[attr-defined]
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(masked_scores, dim=-1)
        expected = attention.out_proj(weights @ v)

        assert torch.allclose(output, expected, atol=1e-6)

        future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        masked_weights = weights[0].masked_select(future_mask)
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))

    def test_attention_without_causal_mask_allows_future_positions(self) -> None:
        """Without causal masking, tokens may attend to future positions."""
        embed_dim = head_dim = 4
        context_length = seq_len = 4

        attention = CausalAttention(
            embed_dim=embed_dim,
            head_dim=head_dim,
            context_length=context_length,
            dropout=0.0,
            bias=False,
            causal=False,
        )
        attention.eval()
        _set_identity_projections(attention)

        x = torch.arange(seq_len * embed_dim, dtype=torch.float32).view(1, seq_len, embed_dim)

        q = attention.q_proj(x)
        k = attention.k_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(attention.head_dim)
        weights = torch.softmax(scores, dim=-1)

        future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        masked_weights = weights[0].masked_select(future_mask)
        assert torch.any(masked_weights > 0)
